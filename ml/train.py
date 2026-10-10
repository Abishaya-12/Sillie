"""Train and evaluate a sequence-based Sillie sign classifier."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import (
    FEATURES_PER_FRAME,
    IDLE_LABEL,
    SEQUENCE_LENGTH,
    config,
)

if TYPE_CHECKING:
    import tensorflow as tf

RANDOM_SEED = 42
FEATURE_COUNT = SEQUENCE_LENGTH * FEATURES_PER_FRAME
WRIST_OFFSET_START = FEATURES_PER_FRAME - 3
NOISE_STANDARD_DEVIATION = 0.01
WRIST_SCALE_MIN = 0.9
WRIST_SCALE_MAX = 1.1


def load_dataset(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load valid per-session sequence rows and return features, labels, and groups."""
    label_files = collect_label_files(data_dir)
    features: list[np.ndarray] = []
    label_names: list[str] = []
    group_names: list[str] = []
    labels: list[str] = []

    for label in sorted(label_files):
        label_windows: list[np.ndarray] = []
        label_groups: list[str] = []
        for csv_file, session_id in label_files[label]:
            windows = load_sequence_file(csv_file)
            if windows is None:
                continue
            label_windows.extend(windows)
            label_groups.extend([f"{label}:{session_id}"] * len(windows))
        if label_windows:
            class_id = len(labels)
            labels.append(label)
            features.extend(label_windows)
            label_names.extend([str(class_id)] * len(label_windows))
            group_names.extend(label_groups)

    if len(labels) < 2:
        raise ValueError("At least two labels with valid sequence files are required.")

    class_ids = np.asarray(label_names, dtype=np.int64)
    groups = np.asarray(group_names, dtype=str)
    print_label_counts(labels, class_ids, groups)
    if IDLE_LABEL not in labels:
        print(f"WARNING: No '{IDLE_LABEL}' label was found; the model cannot reject idle signs.")
    return np.stack(features), class_ids, groups, labels


def collect_label_files(data_dir: Path) -> dict[str, list[tuple[Path, str]]]:
    """Find per-session CSV files and warn about unsupported flat legacy files."""
    label_files: dict[str, list[tuple[Path, str]]] = {}
    if not data_dir.exists():
        return label_files

    for legacy_file in sorted(data_dir.glob("*.csv")):
        print(f"WARNING: Skipping legacy single-frame file: {legacy_file}")
    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = sorted(label_dir.glob("*.csv"))
        if files:
            label_files[label_dir.name] = [
                (csv_file, csv_file.stem) for csv_file in files
            ]
    return label_files


def load_sequence_file(csv_file: Path) -> list[np.ndarray] | None:
    """Read a session file, skipping the whole file if any row is invalid."""
    windows: list[np.ndarray] = []
    expected_count = SEQUENCE_LENGTH * FEATURES_PER_FRAME
    try:
        with csv_file.open("r", newline="", encoding="utf-8") as source:
            for row_number, row in enumerate(csv.reader(source), start=1):
                if len(row) != expected_count:
                    print(
                        f"WARNING: Skipping {csv_file}: row {row_number} has "
                        f"{len(row)} values; expected {expected_count}."
                    )
                    return None
                try:
                    values = np.asarray([float(value) for value in row], dtype=np.float32)
                except ValueError:
                    print(f"WARNING: Skipping {csv_file}: row {row_number} is not numeric.")
                    return None
                if not np.isfinite(values).all():
                    print(f"WARNING: Skipping {csv_file}: row {row_number} is not finite.")
                    return None
                windows.append(values.reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME))
    except OSError as error:
        print(f"WARNING: Skipping unreadable sample file {csv_file}: {error}")
        return None

    return windows


def print_label_counts(labels: list[str], class_ids: np.ndarray, group_ids: np.ndarray) -> None:
    """Print the accepted window and session counts for each label."""
    print("Dataset counts:")
    for class_id, label in enumerate(labels):
        label_mask = class_ids == class_id
        session_count = len(np.unique(group_ids[label_mask]))
        print(f"  {label}: {int(np.sum(label_mask))} windows, {session_count} sessions")


def split_dataset(
    features: np.ndarray, class_ids: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make session-isolated splits, falling back to stratification when needed."""
    session_counts = [
        len(np.unique(group_ids[class_ids == class_id]))
        for class_id in np.unique(class_ids)
    ]
    if any(count < 3 for count in session_counts):
        print(
            "WARNING: At least one label has fewer than 3 sessions. "
            "Falling back to stratified window-level splits; accuracy will be optimistic."
        )
        return stratified_split(features, class_ids)
    return grouped_split(features, class_ids, group_ids)


def stratified_split(
    features: np.ndarray, class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make the seeded stratified window-level fallback split."""
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
            "Could not make stratified 70/15/15 splits. Collect more sequence windows "
            "for each label, then try again."
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
        inner_splitter = GroupShuffleSplit(
            n_splits=20,
            test_size=0.50,
            random_state=RANDOM_SEED + attempt,
        )
        for validation_local, test_local in inner_splitter.split(
            features[remainder_indices],
            class_ids[remainder_indices],
            group_ids[remainder_indices],
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
        "Could not create session-isolated splits with every label in training. "
        "Collect more sessions per label or balance session sizes."
    )


def all_classes_present(
    class_ids: np.ndarray, partitions: tuple[np.ndarray, ...], classes: np.ndarray
) -> bool:
    """Check that each requested split contains every class."""
    return all(np.array_equal(np.unique(class_ids[index]), classes) for index in partitions)


def augment_training_data(
    features: np.ndarray, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Add Gaussian noise and small per-window scale variation to wrist offsets."""
    random = np.random.default_rng(seed)
    augmented = features.copy()
    augmented += random.normal(
        0.0, NOISE_STANDARD_DEVIATION, size=augmented.shape
    ).astype(np.float32)
    scales = random.uniform(
        WRIST_SCALE_MIN, WRIST_SCALE_MAX, size=(len(augmented), 1, 1)
    ).astype(np.float32)
    augmented[:, :, WRIST_OFFSET_START:] *= scales
    return np.concatenate((features, augmented), axis=0)


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a GRU classifier for a sequence of hand landmark features."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
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
    """Save the trained sequence model, labels, and input-dimension metadata."""
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_dir / "sign_model.keras")
    (config.model_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
    }
    (config.model_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def train() -> None:
    """Train, evaluate, and save a sequence-based sign classifier."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    features, class_ids, group_ids, labels = load_dataset(config.data_dir)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(
        features, class_ids, group_ids
    )
    x_train = augment_training_data(x_train)

    model = build_model(len(labels))
    model.fit(
        x_train,
        np.concatenate((y_train, y_train)),
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
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, predictions, labels=np.arange(len(labels)), zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(labels)))
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Per-class precision / recall:")
    for label, label_precision, label_recall in zip(labels, precision, recall):
        print(f"  {label}: precision={label_precision:.4f}, recall={label_recall:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted; label order={labels}):")
    print(matrix)
    save_artifacts(model, labels)
    print(f"Saved model and labels to {config.model_dir}")


if __name__ == "__main__":
    train()
