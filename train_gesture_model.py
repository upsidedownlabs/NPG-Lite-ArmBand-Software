"""
train_gesture_model.py
------------------------
Trains a gesture-classification model on the FILTERED EMG CSVs produced by
record_gesture.py.

record_gesture.py organizes recordings per subject:
    training_data/<subject>/<gesture>/*.csv
    training_data/dataset_index.json   <- ONE central file: every subject,
                                           gesture, and recording session,
                                           each with an include_in_training
                                           true/false flag

This script reads that single dataset_index.json automatically and skips
any session flagged include_in_training: false - no prompts, nothing to
select here. To change which sessions are used, edit the flags with
record_gesture.py's 'sessions' menu (or by hand in the JSON) before
running this script. Pass --subject to train on just one person's data;
by default every subject in the index is combined. Older flat layouts
(CSVs directly under training_data/<gesture>/, no subject folder or index)
still work unfiltered, exactly as before.

Works with a single device's data (any number of dev*.csv files - the
device index in the filename is ignored, only the gesture folder matters).

Pipeline:
    1. Load every CSV, keep the ch1..chN (EMG) columns plus any
       accel_x/accel_y/accel_z (or accel_x_dev1-style, if multiple bands
       are configured) columns written by record_gesture.py's single-band
       + accel setup. If a matching
       "<file>.beep.json" sidecar exists (written by record_gesture.py's
       beep cue), it lists the clean sample-index segments- for active
       gestures, one short segment per rep (right after each beep, skipping
       reaction time); for 'rest', one long segment covering the rest of
       the trial. Files with no sidecar (recorded before the beep cue
       existed) fall back to using the whole trial, and are called out in a
       warning so you know which ones are worth re-recording.
    2. Slice each file into overlapping windows, drawn only from inside its
       clean segments (never crossing a segment boundary, since a window
       straddling a reaction-time gap or inter-rep relax time would mix
       unrelated/mislabeled data - same logic already applied at file
       boundaries so windows never cross between separate trials).
    3. Z-score normalize per channel using TRAIN-set statistics only.
    4. Train the chosen architecture (cnn / lstm / cnn_lstm).
    5. Evaluate on the held-out validation AND test splits, and save a
       per-class report + confusion matrix for the test split (the split
       EarlyStopping/checkpointing never looked at).
    6. Save:
         gesture_model/model.keras              <- trained model
         gesture_model/meta.json                <- window_size, stride,
                                                    channels, sampling_rate,
                                                    class names, scaler,
                                                    train/val/test metrics
         gesture_model/replay_buffer.npz         <- small labeled sample of
                                                    the training windows,
                                                    used later by
                                                    realtime_classify_and_train.py
                                                    to avoid catastrophic
                                                    forgetting during online
                                                    fine-tuning
         gesture_model/classification_report.txt <- per-class precision/
                                                    recall/F1 on the test set
         gesture_model/confusion_matrix.csv       <- raw confusion matrix
         gesture_model/confusion_matrix.png        <- plotted confusion
                                                    matrix (best-effort, only
                                                    if matplotlib is available)

Usage:
    python train_gesture_model.py --data_dir training_data --arch cnn_lstm
    python train_gesture_model.py --data_dir training_data --subject alice
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from model_architectures import build_model

# Must match the SAMPLING_RATE used in record_gesture.py (the Notch/EXG
# filter coefficients only exist for 250 and 500 Hz - see that file's
# docstring for the firmware-rate mismatch note).
SAMPLING_RATE = 500

DATASET_INDEX_FILENAME = "dataset_index.json"


def load_beep_segments(csv_path):
    """Look for the "<csv_path>.beep.json" sidecar written by record_gesture.py
    and return its list of clean (start, end) sample-index segments - the
    parts of the trial that are actually on-gesture (or, for rest, past the
    startup beep) rather than reaction time / inter-rep relax time. Returns
    None if no sidecar exists (older recording made before the beep cue was
    added), so the caller can tell that apart from "sidecar says there's
    nothing usable" (empty list)."""
    beep_path = csv_path[:-4] + ".beep.json" if csv_path.endswith(".csv") else csv_path + ".beep.json"
    if not os.path.exists(beep_path):
        return None
    with open(beep_path) as f:
        meta = json.load(f)
    if "segments" in meta:
        return [tuple(s) for s in meta["segments"]]
    # Backward compat with the older single-beep sidecar format.
    if "valid_start_index" in meta and "total_samples" in meta:
        return [(meta["valid_start_index"], meta["total_samples"])]
    return None


def trial_number_from_filename(csv_path):
    """Parse the trial number out of a "<gesture>_trial<N>_..." filename,
    matching the naming scheme written by record_gesture.py. Returns None
    if the filename doesn't follow that pattern."""
    fname = os.path.basename(csv_path)
    try:
        after_trial = fname.split("_trial", 1)[1]
        num_str = after_trial.split("_", 1)[0]
        return int(num_str)
    except (IndexError, ValueError):
        return None


def load_central_index(data_dir):
    """Return the parsed central dataset_index.json living directly under
    data_dir (the ONE file record_gesture.py maintains), or None if it
    doesn't exist yet (older flat layout, or nothing recorded with this
    version of record_gesture.py) - in which case every CSV found is used
    unfiltered."""
    path = os.path.join(data_dir, DATASET_INDEX_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_session_included(index, subject_name, gesture_name, trial_num):
    """True unless the central index explicitly flags this session as
    include_in_training: false. Sessions with no matching index entry (e.g.
    a CSV copied in by hand, or no index at all) default to included."""
    if index is None or trial_num is None:
        return True
    sessions = (
        index.get("subjects", {})
        .get(subject_name, {})
        .get(gesture_name, {})
        .get("sessions", [])
    )
    for s in sessions:
        if s.get("trial") == trial_num:
            return s.get("include_in_training", True)
    return True


def discover_subject_dirs(data_dir, subject_arg):
    """Figure out which folder(s) under data_dir to treat as "one subject's
    data". If --subject was given, use exactly that folder. Otherwise
    auto-discover: if data_dir's immediate subfolders directly contain CSVs,
    that's the older flat layout (gesture folders right under data_dir) -
    treat data_dir itself as one unfiltered "subject". Otherwise every
    immediate subfolder of data_dir is treated as its own subject (each
    with its own gesture subfolders)."""
    if subject_arg:
        subject_dir = os.path.join(data_dir, subject_arg)
        if not os.path.isdir(subject_dir):
            raise RuntimeError(f"No such subject folder: '{subject_dir}'.")
        return [subject_dir]

    immediate_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)
    )
    if not immediate_dirs:
        raise RuntimeError(f"No subfolders found under '{data_dir}'.")

    has_direct_csvs = any(glob.glob(os.path.join(d, "*.csv")) for d in immediate_dirs)
    if has_direct_csvs:
        return [data_dir]  # old flat layout: immediate_dirs ARE the gesture folders
    return immediate_dirs


def load_gesture_files(data_dir, subject_arg=None):
    """Returns (items, excluded, subject_dirs).
    items is a list of (gesture_label, dataframe, path, segments) for every
    CSV found that's actually usable for training. segments is None when
    the file has no beep sidecar. excluded is a list of (path, reason) for
    CSVs that were found but skipped because the central dataset_index.json
    flagged them include_in_training: false. Reads that ONE central index
    file once and filters automatically - no selection step here."""
    items = []
    excluded = []
    subject_dirs = discover_subject_dirs(data_dir, subject_arg)
    index = load_central_index(data_dir)

    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)
        gesture_dirs = sorted(
            d for d in glob.glob(os.path.join(subject_dir, "*")) if os.path.isdir(d)
        )
        for gdir in gesture_dirs:
            label = os.path.basename(gdir)
            csv_files = sorted(glob.glob(os.path.join(gdir, "*.csv")))
            for f in csv_files:
                trial_num = trial_number_from_filename(f)
                if not is_session_included(index, subject_name, label, trial_num):
                    excluded.append((f, "include_in_training: false"))
                    continue
                df = pd.read_csv(f)
                segments = load_beep_segments(f)
                items.append((label, df, f, segments))

    if not items:
        raise RuntimeError(f"No usable gesture CSVs found under '{data_dir}'.")
    return items, excluded, subject_dirs


def get_channel_columns(df):
    """Every feature column fed to the model: EMG channels (ch1..chN) plus
    any accelerometer columns (accel_x/accel_y/accel_z, or their per-device
    suffixed form accel_x_dev1 etc. if more than one band is configured in
    record_gesture.py). Order matches the CSV's own column order, which is
    always EMG-block-then-accel-block (see merge_and_save() in
    record_gesture.py) so it's consistent across every recorded file and
    matches the feature order gesture_ui_server2.py builds in real time."""
    return [c for c in df.columns if c.startswith("ch") or c.startswith("accel")]


def window_dataframe(df, ch_cols, window_size, stride, segments=None):
    """Slice one file's channel data into overlapping windows, drawn only
    from inside the given (start, end) segments - never crossing a segment
    boundary (a window straddling a beep's reaction-time gap, or the relax
    time between two reps, would mix unrelated/mislabeled data). If
    segments is None, the whole file is treated as one segment.
    Returns array of shape (num_windows, window_size, num_channels)."""
    data = df[ch_cols].to_numpy(dtype=np.float32)
    n = data.shape[0]
    if segments is None:
        segments = [(0, n)]

    windows = []
    for seg_start, seg_end in segments:
        seg_start = max(0, int(seg_start))
        seg_end = min(n, int(seg_end))
        start = seg_start
        while start + window_size <= seg_end:
            windows.append(data[start:start + window_size])
            start += stride

    if not windows:
        return np.empty((0, window_size, len(ch_cols)), dtype=np.float32)
    return np.stack(windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data")
    ap.add_argument("--subject", default=None,
                     help="train on only this subject's folder (training_data/<subject>/). "
                          "If omitted, every subject in training_data/dataset_index.json "
                          "is auto-discovered and combined.")
    ap.add_argument("--out_dir", default="gesture_model")
    ap.add_argument("--arch", choices=["cnn", "lstm", "cnn_lstm"], default="cnn_lstm")
    ap.add_argument("--window_size", type=int, default=250,
                     help="samples per window (250 @ 500Hz = 0.5s)")
    ap.add_argument("--stride", type=int, default=125,
                     help="hop between windows (125 = 50%% overlap)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--replay_samples_per_class", type=int, default=60,
                     help="windows per class kept for later online fine-tuning")
    ap.add_argument("--ignore_beep_marker", action="store_true",
                     help="use the FULL length of every trial instead of trimming to "
                          "the beep-derived clean segments. Only for debugging/"
                          "comparison - normally you want the trimming, since it's "
                          "what keeps reaction-time and inter-rep relax samples out "
                          "of the gesture windows.")
    args = ap.parse_args()

    # window_dataframe() advances with `start += stride` while
    # `start + window_size <= seg_end`. A zero/negative stride never advances
    # (infinite loop appending windows until memory runs out) and a
    # zero/negative window_size makes the guard meaningless, so reject both here
    # rather than hanging halfway through a long training run.
    if args.window_size <= 0:
        raise SystemExit(f"--window_size must be positive, got {args.window_size}.")
    if args.stride <= 0:
        raise SystemExit(f"--stride must be positive, got {args.stride}.")
    if args.stride > args.window_size:
        print(f"  (warning) --stride {args.stride} > --window_size {args.window_size}: "
              f"consecutive windows will skip {args.stride - args.window_size} "
              f"sample(s) of every segment.")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CSVs...")
    items, excluded, subject_dirs = load_gesture_files(args.data_dir, args.subject)
    found_labels = set(label for label, _, _, _ in items)
    subject_names = [os.path.basename(d) for d in subject_dirs]
    print(f"  Subject(s): {', '.join(subject_names)}")
    print(f"  Found {len(items)} usable file(s) across "
          f"{len(found_labels)} gesture(s).")
    if excluded:
        print(f"  Skipped {len(excluded)} session(s) flagged "
              f"include_in_training: false in dataset_index.json:")
        for f, reason in excluded:
            print(f"    {f}  ({reason})")

    if "rest" not in found_labels:
        print(
            "\n  (warning) No 'rest' class found in the training data. Without it, "
            "the model will always force a prediction into one of your active "
            "gestures, even when the hand is idle. Run record_gesture.py again "
            "(it now offers to record 'rest' up front) and re-train before doing "
            "real-time classification with a confidence threshold.\n"
        )

    # Determine feature columns (EMG + accel) from the first file, verify
    # every other file has the exact same set of columns in the same
    # order- not just the same count, since e.g. a file recorded with the
    # IMU characteristic missing would still have the right accel column
    # names (zeros) but a mismatched name would mean a different band
    # config was used for that recording.
    ch_cols = get_channel_columns(items[0][1])
    num_channels = len(ch_cols)
    for label, df, f, _ in items:
        cols = get_channel_columns(df)
        if cols != ch_cols:
            raise RuntimeError(
                f"Feature column mismatch in {f}: expected {ch_cols}, "
                f"got {cols}. All recordings must come from the same "
                f"band configuration (same boards/channel counts) for one model."
            )

    print(f"  Feature columns ({num_channels} total): {ch_cols}")

    all_windows = []
    all_labels = []
    no_beep_files = []
    for label, df, f, segments in items:
        cols = get_channel_columns(df)
        if segments is None:
            no_beep_files.append(f)
            segments = None  # window_dataframe treats None as "whole file"
        if args.ignore_beep_marker:
            segments = None
        w = window_dataframe(df, cols, args.window_size, args.stride, segments=segments)
        if w.shape[0] == 0:
            reason = ("no beep-derived segments long enough for one window"
                      if segments else f"shorter than one window ({len(df)} samples)")
            print(f"  (skip) {f}: {reason}")
            continue
        all_windows.append(w)
        all_labels.extend([label] * w.shape[0])

    if no_beep_files and not args.ignore_beep_marker:
        print(
            f"\n  (warning) {len(no_beep_files)} file(s) have no '.beep.json' marker "
            f"(recorded before the beep cue was added to record_gesture.py) - the "
            f"FULL trial was used for those, reaction-time/relax time included. "
            f"Re-record them for cleaner labels:"
        )
        for f in no_beep_files:
            print(f"    {f}")
        print()

    if not all_windows:
        raise RuntimeError(
            "No usable windows produced from any file - check window_size/stride "
            "against your recording lengths, or re-record with longer/more reps."
        )

    X = np.concatenate(all_windows, axis=0)
    y_raw = np.array(all_labels)
    print(f"  Total windows: {X.shape[0]}  shape={X.shape}")

    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    y_onehot = keras.utils.to_categorical(y_int, num_classes=num_classes)
    print(f"  Classes: {list(le.classes_)}")

    # Train/val/test split, stratified so every gesture is represented in each split.
    X_train, X_temp, y_train, y_temp, yi_train, yi_temp = train_test_split(
        X, y_onehot, y_int, test_size=0.3, random_state=42, stratify=y_int
    )
    X_val, X_test, y_val, y_test, yi_val, yi_test = train_test_split(
        X_temp, y_temp, yi_temp, test_size=0.5, random_state=42, stratify=yi_temp
    )
    print(f"  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Z-score normalization, stats computed on TRAIN split only.
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std[std == 0] = 1.0

    def normalize(arr):
        return (arr - mean) / std

    X_train_n = normalize(X_train)
    X_val_n = normalize(X_val)
    X_test_n = normalize(X_test)

    print(f"Building '{args.arch}' model...")
    model = build_model(args.arch, args.window_size, num_channels, num_classes)
    model.summary()

    ckpt_path = os.path.join(args.out_dir, "model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                         save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # model.keras was already saved as the best checkpoint by ModelCheckpoint,
    # but save again to be safe in case training ended without improvement.
    model.save(ckpt_path)

    # ---- Validate on all three splits ----
    # Train accuracy is reported mainly as a sanity check (a huge gap vs val
    # means overfitting). Val accuracy is what EarlyStopping/ModelCheckpoint/
    # ReduceLROnPlateau watched during training, so it's a bit optimistic.
    # Test accuracy is the split the model never influenced at all - that's
    # the number to trust for "how good is this model really".
    train_loss, train_acc = model.evaluate(X_train_n, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val_n, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"\nTrain accuracy:      {train_acc:.4f}  (train loss: {train_loss:.4f})")
    print(f"Validation accuracy: {val_acc:.4f}  (val loss: {val_loss:.4f})")
    print(f"Test accuracy:       {test_acc:.4f}  (test loss: {test_loss:.4f})")
    if train_acc - val_acc > 0.15:
        print("  (note) train accuracy is notably higher than validation - the model "
              "may be overfitting. More trials per gesture, more 'rest' data, or "
              "stronger dropout can help.")

    # ---- Per-class breakdown + confusion matrix on the TEST split ----
    y_test_pred = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
    y_test_true = yi_test

    report_txt = classification_report(
        y_test_true, y_test_pred, labels=range(num_classes),
        target_names=list(le.classes_), digits=3, zero_division=0,
    )
    report_dict = classification_report(
        y_test_true, y_test_pred, labels=range(num_classes),
        target_names=list(le.classes_), digits=3, zero_division=0,
        output_dict=True,
    )
    print("\nPer-class test performance:\n")
    print(report_txt)

    cm = confusion_matrix(y_test_true, y_test_pred, labels=range(num_classes))
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows = true, cols = predicted):\n")
    print(cm_df.to_string())

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.2 * num_classes + 2, 1.2 * num_classes + 2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(le.classes_, rotation=45, ha="right")
        ax.set_yticklabels(le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test set)")
        thresh = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  (note) could not save confusion_matrix.png ({e}); "
              f"confusion_matrix.csv still has the raw numbers.")

    meta = {
        "arch": args.arch,
        "window_size": args.window_size,
        "stride": args.stride,
        "num_channels": num_channels,
        "channel_columns": ch_cols,
        "sampling_rate": SAMPLING_RATE,
        "classes": list(le.classes_),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "used_beep_marker": not args.ignore_beep_marker,
        "files_without_beep_marker": no_beep_files,
        "subjects": subject_names,
        "excluded_sessions": [f for f, _ in excluded],
        "train_accuracy": float(train_acc),
        "train_loss": float(train_loss),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "test_classification_report": report_dict,
        "test_confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Small labeled replay buffer (RAW, unnormalized windows) for later
    # online fine-tuning, sampled evenly across classes so a live session
    # doesn't overwrite what the model already learned.
    replay_X = []
    replay_y = []
    rng = np.random.default_rng(42)
    for cls_idx in range(num_classes):
        idx = np.where(yi_train == cls_idx)[0]
        take = min(args.replay_samples_per_class, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        replay_X.append(X_train[chosen])
        replay_y.append(y_train[chosen])
    replay_X = np.concatenate(replay_X, axis=0)
    replay_y = np.concatenate(replay_y, axis=0)
    np.savez(os.path.join(args.out_dir, "replay_buffer.npz"),
             X=replay_X, y=replay_y)

    print(
        f"\nSaved model.keras + meta.json + replay_buffer.npz + "
        f"classification_report.txt + confusion_matrix.csv(/.png) to "
        f"'{args.out_dir}/'"
    )


if __name__ == "__main__":
    main()