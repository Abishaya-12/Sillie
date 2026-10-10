"""Train and evaluate a sign classifier from collected landmark samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config import config

RANDOM_SEED = 42
FEATURE_COUNT = 63


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load per-label CSV files and return features, class IDs, and labels."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No label CSV files found in data directory: {data_dir}")

    labels = [csv_file.stem for csv_file in csv_files]
    features: list[list[float]] = []
    class_ids: list[int] = []

    for class_id, csv_file in enumerate(csv_files):
        load_label_rows(csv_file, class_id, features, class_ids)

    if len(labels) < 2:
        raise ValueError("Training requires at least two label CSV files.")
    return np.asarray(features, dtype=np.float32), np.asarray(class_ids, dtype=np.int64), labels


def load_label_rows(
    csv_file: Path,
    class_id: int,
    features: list[list[float]],
    class_ids: list[int],
) -> None:
    """Read and validate all landmark rows for one label CSV."""
    with csv_file.open("r", newline="", encoding="utf-8") as source:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != FEATURE_COUNT:
                raise ValueError(
                    f"{csv_file}:{row_number} must contain exactly {FEATURE_COUNT} values."
                )
            try:
                values = [float(value) for value in row]
            except ValueError as error:
                raise ValueError(
                    f"{csv_file}:{row_number} contains a non-numeric value."
                ) from error
            if not np.isfinite(values).all():
                raise ValueError(f"{csv_file}:{row_number} contains a non-finite value.")
            features.append(values)
            class_ids.append(class_id)


def split_dataset(
    features: np.ndarray, class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make stratified 70/15/15 train, validation, and test splits."""
    try:
        x_train, x_remaining, y_train, y_remaining = train_test_split(
            features,
            class_ids,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=class_ids,
        )
        x_validation, x_test, y_validation, y_test = train_test_split(
            x_remaining,
            y_remaining,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=y_remaining,
        )
    except ValueError as error:
        raise ValueError(
            "Could not make stratified 70/15/15 splits. Collect more samples for "
            "each label, then try again."
        ) from error

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(num_classes: int) -> tf.keras.Model:
    """Build the dense neural network for landmark classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(model: tf.keras.Model, labels: list[str]) -> None:
    """Save the trained Keras model and its class-label ordering."""
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_dir / "sign_model.keras")
    labels_path = config.model_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")


def train() -> None:
    """Train the classifier, report test metrics, and save its artifacts."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    features, class_ids, labels = load_dataset(config.data_dir)
    splits = split_dataset(features, class_ids)
    x_train, x_validation, x_test, y_train, y_validation, y_test = splits

    model = build_model(len(labels))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=200,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(labels)))
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted; label order={labels}):")
    print(matrix)
    save_artifacts(model, labels)
    print(f"Saved model and labels to {config.model_dir}")


if __name__ == "__main__":
    train()
