"""
model_architectures.py
-----------------------
CNN, LSTM, and hybrid CNN-LSTM architectures for EMG gesture classification.

All models take a sliding window of FILTERED sensor samples as input:
    shape = (window_size, num_channels)
and output a softmax distribution over the gesture classes.

num_channels is generic- it doesn't care whether the columns are EMG
channels or accelerometer axes. With the single-band + accel setup in
record_gesture.py, num_channels = (EMG channels) + 3 (accel_x/y/z), and
these architectures don't need any change to consume that: the CNN/LSTM
layers just learn from whatever's in each column.

Usage:
    from model_architectures import build_model
    model = build_model("cnn_lstm", window_size=250, num_channels=3, num_classes=5)
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(window_size, num_channels, num_classes):
    """Pure 1D-CNN. Fast, cheap, good at picking up local muscle-activation
    shapes. Doesn't explicitly model how those shapes evolve over time."""
    inputs = keras.Input(shape=(window_size, num_channels), name="emg_window")

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="emg_cnn")


def build_lstm_model(window_size, num_channels, num_classes):
    """Pure bidirectional LSTM. Models temporal evolution directly off the
    raw filtered samples. Slower to train than the CNN, and on short EMG
    windows it usually performs a bit worse than the hybrid below."""
    inputs = keras.Input(shape=(window_size, num_channels), name="emg_window")

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="emg_lstm")


def build_cnn_lstm_model(window_size, num_channels, num_classes):
    """CNN feature extractor -> LSTM temporal head.
    Usually the best option for EMG gesture windows: the CNN layers pick up
    local activation bursts/shapes per channel, the LSTM layers track how
    those shapes move over the window. This is the recommended default."""
    inputs = keras.Input(shape=(window_size, num_channels), name="emg_window")

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="emg_cnn_lstm")


def build_model(arch, window_size, num_channels, num_classes, learning_rate=1e-3):
    if arch == "cnn":
        model = build_cnn_model(window_size, num_channels, num_classes)
    elif arch == "lstm":
        model = build_lstm_model(window_size, num_channels, num_classes)
    elif arch == "cnn_lstm":
        model = build_cnn_lstm_model(window_size, num_channels, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'cnn', 'lstm', or 'cnn_lstm'.")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
