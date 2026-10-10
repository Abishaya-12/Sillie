"""Train and evaluate a sign classifier from collected landmark samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import config

if TYPE_CHECKING:
    import tensorflow as tf

RANDOM_SEED = 42
FEATURE_COUNT = 63


def load_dataset(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load per-label CSVs and return features, classes, session groups, and labels."""
    label_files = collect_label_files(data_dir)
    if not label_files:
        raise ValueError(f"No label CSV files found in data directory: {data_dir}")

    labels = sorted(label_files)
    features: list[list[float]] = []
    class_ids: list[int] = []
    group_ids: list[str] = []

    for class_id, label in enumerate(labels):
        for csv_file, session_id in label_files[label]:
            load_label_rows(csv_file, class_id, session_id, features, class_ids, group_ids)

    classes = np.asarray(class_ids, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=str)
    if len(labels) < 2:
        raise ValueError("Training requires at least two label directories or legacy CSV files.")
    if len(features) == 0:
        raise ValueError("No sample rows found in the label CSV files.")
    print_label_counts(labels, classes, groups)
    return np.asarray(features, dtype=np.float32), classes, groups, labels


def collect_label_files(data_dir: Path) -> dict[str, list[tuple[Path, str]]]:
    """Find both legacy flat CSVs and per-session CSVs grouped by label."""
    label_files: dict[str, list[tuple[Path, str]]] = {}
    if not data_dir.exists():
        return label_files

    for csv_file in sorted(data_dir.glob("*.csv")):
        label_files.setdefault(csv_file.stem, []).append((csv_file, "legacy"))
    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for csv_file in sorted(label_dir.glob("*.csv")):
            label_files.setdefault(label_dir.name, []).append((csv_file, csv_file.stem))
    return label_files


def print_label_counts(labels: list[str], class_ids: np.ndarray, group_ids: np.ndarray) -> None:
    """Print each label's sample count and number of distinct sessions."""
    print("Dataset counts:")
    for class_id, label in enumerate(labels):
        label_mask = class_ids == class_id
        session_count = len(np.unique(group_ids[label_mask]))
        print(f"  {label}: {int(np.sum(label_mask))} samples, {session_count} sessions")


def load_label_rows(
    csv_file: Path,
    class_id: int,
    session_id: str,
    features: list[list[float]],
    class_ids: list[int],
    group_ids: list[str],
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
            group_ids.append(session_id)


def split_dataset(
    features: np.ndarray, class_ids: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make session-isolated splits, falling back to stratification if needed."""
    session_counts = [
        len(np.unique(group_ids[class_ids == class_id]))
        for class_id in np.unique(class_ids)
    ]
    if any(count < 3 for count in session_counts):
        print(
            "WARNING: At least one label has fewer than 3 sessions. "
            "Falling back to stratified row-level splits; accuracy will be optimistic."
        )
        return stratified_split(features, class_ids)
    return grouped_split(features, class_ids, group_ids)


def stratified_split(
    features: np.ndarray, class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make the seeded stratified row-level fallback split."""
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


def grouped_split(
    features: np.ndarray, class_ids: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find deterministic 70/15/15 splits with no session shared between sets."""
    class_values = np.unique(class_ids)
    outer_splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    for attempt, (train_indices, remainder_indices) in enumerate(
        outer_splitter.split(features, class_ids, group_ids)
    ):
        remainder_groups = group_ids[remainder_indices]
        inner_splitter = GroupShuffleSplit(
            n_splits=20,
            test_size=0.50,
            random_state=RANDOM_SEED + attempt,
        )
        for validation_local, test_local in inner_splitter.split(
            features[remainder_indices],
            class_ids[remainder_indices],
            remainder_groups,
        ):
            validation_indices = remainder_indices[validation_local]
            test_indices = remainder_indices[test_local]
            if not all_classes_present(class_ids, (train_indices,), class_values):
                continue
            return (
                features[train_indices],
                features[validation_indices],
                features[test_indices],
                class_ids[train_indices],
                class_ids[validation_indices],
                class_ids[test_indices],
            )

    raise ValueError(
        "Could not create session-isolated 70/15/15 splits with every label in training. "
        "Collect more sessions per label or make session sample counts more balanced."
    )


def all_classes_present(
    class_ids: np.ndarray, partitions: tuple[np.ndarray, ...], classes: np.ndarray
) -> bool:
    """Check that every split contains samples for every label."""
    return all(np.array_equal(np.unique(class_ids[indices]), classes) for indices in partitions)


def build_model(num_classes: int) -> tf.keras.Model:
    """Build the dense neural network for landmark classification."""
    import tensorflow as tf

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
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    features, class_ids, group_ids, labels = load_dataset(config.data_dir)
    splits = split_dataset(features, class_ids, group_ids)
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
