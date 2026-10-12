"""Train, evaluate, and save the body-aware sequence classifier."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (
    FEATURES_PER_FRAME,
    FEATURE_VERSION,
    IDLE_LABEL,
    RAW_FRAME_SIZE,
    SEQUENCE_LENGTH,
    config,
)
from ml.features import window_features

if TYPE_CHECKING:
    import tensorflow as tf

RANDOM_SEED = 42
RAW_WINDOW_SIZE = SEQUENCE_LENGTH * RAW_FRAME_SIZE
AUGMENTATION_NOISE_STD = 0.003
MAX_AUGMENTATION_SHIFT = 3
AUGMENTATION_JITTER = 0.03


def collect_label_files(data_dir: Path) -> dict[str, list[tuple[Path, str]]]:
    """Find per-session sample CSV files and warn about flat legacy files."""
    label_files: dict[str, list[tuple[Path, str]]] = {}
    if not data_dir.exists():
        return label_files

    for legacy_file in sorted(data_dir.glob("*.csv")):
        print(f"WARNING: Skipping legacy or obsolete data file: {legacy_file}")
    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = sorted(label_dir.glob("*.csv"))
        if files:
            label_files[label_dir.name] = [
                (csv_file, csv_file.stem) for csv_file in files
            ]
    return label_files


def load_raw_window_file(csv_file: Path) -> list[np.ndarray] | None:
    """Read raw session windows, skipping files with invalid or old-format rows."""
    windows: list[np.ndarray] = []
    try:
        with csv_file.open("r", newline="", encoding="utf-8") as source:
            for row_number, row in enumerate(csv.reader(source), start=1):
                if len(row) != RAW_WINDOW_SIZE:
                    print(
                        f"WARNING: Skipping {csv_file}: row {row_number} has "
                        f"{len(row)} values; expected {RAW_WINDOW_SIZE} raw values. "
                        "Re-record this session."
                    )
                    return None
                try:
                    values = np.asarray(row, dtype=np.float64)
                except ValueError:
                    print(f"WARNING: Skipping {csv_file}: row {row_number} is not numeric.")
                    return None
                if not np.isfinite(values).all():
                    print(f"WARNING: Skipping {csv_file}: row {row_number} is not finite.")
                    return None
                windows.append(values.reshape(SEQUENCE_LENGTH, RAW_FRAME_SIZE))
    except OSError as error:
        print(f"WARNING: Skipping unreadable sample file {csv_file}: {error}")
        return None
    return windows


def load_dataset(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load valid raw windows with integer labels and session group IDs."""
    raw_windows: list[np.ndarray] = []
    label_indices: list[int] = []
    group_ids: list[str] = []
    labels: list[str] = []

    for label, files in sorted(collect_label_files(data_dir).items()):
        label_windows: list[np.ndarray] = []
        label_groups: list[str] = []
        for csv_file, session_id in files:
            windows = load_raw_window_file(csv_file)
            if windows is None:
                continue
            label_windows.extend(windows)
            label_groups.extend([f"{label}:{session_id}"] * len(windows))
        if not label_windows:
            continue
        class_index = len(labels)
        labels.append(label)
        raw_windows.extend(label_windows)
        label_indices.extend([class_index] * len(label_windows))
        group_ids.extend(label_groups)

    if len(labels) < 2:
        raise ValueError("At least two labels with valid raw sequence windows are required.")

    class_ids = np.asarray(label_indices, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=str)
    raw_data = np.stack(raw_windows).astype(np.float32)
    print_label_counts(labels, class_ids, groups)
    warn_dataset_quality(labels, class_ids, groups)
    return raw_data, class_ids, groups, labels


def print_label_counts(
    labels: list[str], class_ids: np.ndarray, group_ids: np.ndarray
) -> None:
    """Print accepted raw-window and session counts for each label."""
    print("Dataset counts:")
    for class_id, label in enumerate(labels):
        label_mask = class_ids == class_id
        sessions = np.unique(group_ids[label_mask])
        dataset_sessions = [
            session
            for session in sessions
            if session.rsplit(":", 1)[-1].startswith("ds-")
        ]
        own_sessions = len(sessions) - len(dataset_sessions)
        print(
            f"  {label}: {int(np.sum(label_mask))} windows, "
            f"{own_sessions} own sessions, {len(dataset_sessions)} dataset sessions"
        )


def warn_dataset_quality(
    labels: list[str], class_ids: np.ndarray, group_ids: np.ndarray
) -> None:
    """Warn about sparse sessions, missing idle data, and severe class imbalance."""
    counts = np.bincount(class_ids, minlength=len(labels))
    for class_id, label in enumerate(labels):
        sessions = np.unique(group_ids[class_ids == class_id])
        own_sessions = [
            session
            for session in sessions
            if not session.rsplit(":", 1)[-1].startswith("ds-")
        ]
        if len(own_sessions) < 2:
            print(
                f"WARNING: Label '{label}' has {len(own_sessions)} own sessions; "
                "at least 2 are required for validation and test."
            )
    if not any(label.lower() == IDLE_LABEL.lower() for label in labels):
        print(f"WARNING: No '{IDLE_LABEL}' label was found.")
    if counts.max() > 2 * counts.min():
        smallest = labels[int(np.argmin(counts))]
        largest = labels[int(np.argmax(counts))]
        print(
            f"WARNING: Class imbalance: '{largest}' has over twice as many "
            f"windows as '{smallest}'."
        )


def split_dataset(
    raw_windows: np.ndarray,
    class_ids: np.ndarray,
    group_ids: np.ndarray,
    allow_ds_only: bool = False,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep ds- sessions in training and split own sessions across all sets."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_sessions = np.unique(group_ids[class_mask])
        dataset_sessions = [
            session
            for session in class_sessions
            if session.rsplit(":", 1)[-1].startswith("ds-")
        ]
        own_sessions = [
            session
            for session in class_sessions
            if not session.rsplit(":", 1)[-1].startswith("ds-")
        ]
        label = labels[int(class_id)] if labels is not None else str(int(class_id))
        if len(own_sessions) < 2:
            if not allow_ds_only:
                raise ValueError(
                    f"Label class {label} has {len(own_sessions)} own sessions; "
                    "at least 2 are required. Collect more own sessions or use "
                    "--allow-ds-only to continue with a loud warning."
                )
            print(
                f"WARNING: --allow-ds-only enabled for label class {label}; "
                f"only {len(own_sessions)} own sessions are available. "
                "Dataset sessions will remain training-only."
            )

        shuffled_own = list(own_sessions)
        rng.shuffle(shuffled_own)
        if len(shuffled_own) >= 3:
            own_train_count = len(shuffled_own) - 2
            own_train = shuffled_own[:own_train_count]
            own_validation = shuffled_own[own_train_count:own_train_count + 1]
            own_test = shuffled_own[own_train_count + 1:]
        elif len(shuffled_own) == 2:
            own_train = []
            own_validation = shuffled_own[:1]
            own_test = shuffled_own[1:]
        elif len(shuffled_own) == 1:
            own_train = []
            own_validation = shuffled_own
            own_test = []
        else:
            own_train = []
            own_validation = []
            own_test = []

        selected = set(dataset_sessions + own_train)
        validation_selected = set(own_validation)
        test_selected = set(own_test)
        train_indices.extend(
            int(index)
            for index in np.flatnonzero(class_mask)
            if group_ids[index] in selected
        )
        validation_indices.extend(
            int(index)
            for index in np.flatnonzero(class_mask)
            if group_ids[index] in validation_selected
        )
        test_indices.extend(
            int(index)
            for index in np.flatnonzero(class_mask)
            if group_ids[index] in test_selected
        )

    if not train_indices:
        raise ValueError(
            "Training split is empty. Import dataset videos or collect at least "
            "3 own sessions per label."
        )
    if not np.array_equal(
        np.unique(class_ids[train_indices]), np.unique(class_ids)
    ):
        raise ValueError(
            "Every label needs training data. Import dataset videos or collect "
            "at least 3 own sessions for each label."
        )

    indices = (train_indices, validation_indices, test_indices)
    print_split_counts(class_ids, group_ids, indices, labels)
    return (
        raw_windows[train_indices],
        raw_windows[validation_indices],
        raw_windows[test_indices],
        class_ids[train_indices],
        class_ids[validation_indices],
        class_ids[test_indices],
    )


def print_split_counts(
    class_ids: np.ndarray,
    group_ids: np.ndarray,
    partitions: tuple[list[int], list[int], list[int]],
    labels: list[str] | None = None,
) -> None:
    """Print window counts by label, split, and own/dataset source."""
    for split_name, indices in zip(("train", "validation", "test"), partitions):
        index_array = np.asarray(indices, dtype=int)
        sources = np.asarray([
            "dataset"
            if group_ids[index].rsplit(":", 1)[-1].startswith("ds-")
            else "own"
            for index in index_array
        ])
        print(f"{split_name.title()} windows by source:")
        for class_id in np.unique(class_ids):
            class_mask = class_ids[index_array] == class_id
            own_count = int(np.sum(class_mask & (sources == "own")))
            dataset_count = int(np.sum(class_mask & (sources == "dataset")))
            print(
                f"  {labels[int(class_id)] if labels else f'class {int(class_id)}'}: "
                f"own={own_count}, "
                f"dataset={dataset_count}, total={own_count + dataset_count}"
            )


def stratified_split(
    raw_windows: np.ndarray, class_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split windows 70/15/15 with a fixed, stratified random seed."""
    try:
        x_train, x_remaining, y_train, y_remaining = train_test_split(
            raw_windows,
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
            "Could not make stratified 70/15/15 splits. Collect more windows "
            "for each label and try again."
        ) from error
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def grouped_split(
    raw_windows: np.ndarray, class_ids: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sessions deterministically, ensuring no group crosses partitions."""
    all_classes = np.unique(class_ids)
    outer = GroupShuffleSplit(n_splits=100, test_size=0.30, random_state=RANDOM_SEED)
    for attempt, (train_idx, remaining_idx) in enumerate(
        outer.split(raw_windows, class_ids, group_ids)
    ):
        inner = GroupShuffleSplit(
            n_splits=30, test_size=0.50, random_state=RANDOM_SEED + attempt
        )
        for validation_local, test_local in inner.split(
            raw_windows[remaining_idx],
            class_ids[remaining_idx],
            group_ids[remaining_idx],
        ):
            validation_idx = remaining_idx[validation_local]
            test_idx = remaining_idx[test_local]
            if not np.array_equal(np.unique(class_ids[train_idx]), all_classes):
                continue
            partitions = (train_idx, validation_idx, test_idx)
            if any(
                set(group_ids[partitions[first]])
                & set(group_ids[partitions[second]])
                for first, second in ((0, 1), (0, 2), (1, 2))
            ):
                continue
            return (
                raw_windows[train_idx],
                raw_windows[validation_idx],
                raw_windows[test_idx],
                class_ids[train_idx],
                class_ids[validation_idx],
                class_ids[test_idx],
            )
    raise ValueError(
        "Could not create session-isolated splits with every class in training. "
        "Collect more sessions or balance the session sizes."
    )


def augment_raw_training_data(
    raw_windows: np.ndarray, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Augment training-only raw windows with coordinate noise, jitter, and shifts."""
    random = np.random.default_rng(seed)
    augmented = raw_windows.copy().astype(np.float32)
    augmented += random.normal(
        0.0, AUGMENTATION_NOISE_STD, size=augmented.shape
    ).astype(np.float32)

    hand = augmented[:, :, :63].reshape(-1, SEQUENCE_LENGTH, 21, 3)
    pose = augmented[:, :, 63:].reshape(-1, SEQUENCE_LENGTH, 33, 2)
    translations = random.uniform(
        -AUGMENTATION_JITTER, AUGMENTATION_JITTER, size=(len(augmented), 1, 1, 2)
    ).astype(np.float32)
    hand[:, :, :, :2] += translations
    pose += translations
    scales = random.uniform(
        1.0 - AUGMENTATION_JITTER,
        1.0 + AUGMENTATION_JITTER,
        size=(len(augmented), 1, 1, 1),
    ).astype(np.float32)
    hand *= scales
    pose *= scales

    shifts = random.integers(
        -MAX_AUGMENTATION_SHIFT,
        MAX_AUGMENTATION_SHIFT + 1,
        size=len(augmented),
    )
    time_indices = np.clip(
        np.arange(SEQUENCE_LENGTH)[None, :] + shifts[:, None],
        0,
        SEQUENCE_LENGTH - 1,
    )
    augmented = augmented[np.arange(len(augmented))[:, None], time_indices]
    return np.concatenate((raw_windows, augmented), axis=0)


def transform_windows(raw_windows: np.ndarray) -> np.ndarray:
    """Compute shared body-aware features once for each independent raw window."""
    return np.stack([window_features(window) for window in raw_windows])


def build_model(
    num_classes: int, training_features: np.ndarray
) -> tf.keras.Model:
    """Build the normalized Conv1D-GRU sequence classifier."""
    import tensorflow as tf

    normalization = tf.keras.layers.Normalization(axis=-1)
    normalization.adapt(training_features)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=training_features.shape[1:]),
            normalization,
            tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
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


def make_report(
    labels: list[str], truth: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    """Build per-class metrics, a confusion matrix, and plain-language confusions."""
    class_values = np.arange(len(labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=class_values, zero_division=0
    )
    matrix = confusion_matrix(truth, predictions, labels=class_values)
    totals = matrix.sum(axis=1)
    confusions: list[dict[str, object]] = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            count = int(matrix[actual_index, predicted_index])
            if actual_index == predicted_index or count == 0:
                continue
            rate = count / max(int(totals[actual_index]), 1)
            confusions.append({
                "actual": actual_label,
                "predicted": predicted_label,
                "windows": count,
                "rate": rate,
                "description": (
                    f"{actual_label} -> {predicted_label}: {rate:.0%} of test windows"
                ),
            })
    confusions.sort(key=lambda item: (-float(item["rate"]), -int(item["windows"])))
    return {
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "top_confusions": confusions[:5],
    }


def save_artifacts(
    model: tf.keras.Model, labels: list[str], report: dict[str, object]
) -> None:
    """Save the trained model, labels, metadata, and evaluation report."""
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_dir / "sign_model.keras")
    (config.model_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "raw_frame_size": RAW_FRAME_SIZE,
        "features_per_frame": FEATURES_PER_FRAME,
        "feature_version": FEATURE_VERSION,
    }
    (config.model_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (config.model_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def train(allow_ds_only: bool = False) -> None:
    """Train and evaluate the body-aware sign model from raw sequences."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    raw_windows, class_ids, group_ids, labels = load_dataset(config.data_dir)
    x_train_raw, x_validation_raw, x_test_raw, y_train, y_validation, y_test = (
        split_dataset(raw_windows, class_ids, group_ids, allow_ds_only, labels)
    )
    x_train_raw_augmented = augment_raw_training_data(x_train_raw)
    x_train = transform_windows(x_train_raw_augmented)
    x_validation = transform_windows(x_validation_raw) if len(x_validation_raw) else None
    x_test = transform_windows(x_test_raw) if len(x_test_raw) else None

    model = build_model(len(labels), x_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    model.fit(
        x_train,
        np.concatenate((y_train, y_train)),
        validation_data=(
            (x_validation, y_validation) if x_validation is not None else None
        ),
        class_weight={
            int(class_id): float(weight)
            for class_id, weight in zip(np.unique(y_train), class_weights)
        },
        epochs=200,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if x_validation is not None else "loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    if x_test is not None:
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        predicted_ids = np.argmax(model.predict(x_test, verbose=0), axis=1)
        report = make_report(labels, y_test, predicted_ids)
        report["test_accuracy"] = float(test_accuracy)
        print(f"Test accuracy: {test_accuracy:.4f}")
    else:
        report = {
            "per_class": {},
            "confusion_matrix": [],
            "labels": labels,
            "top_confusions": [],
            "test_accuracy": None,
        }
        print("Test accuracy: unavailable (no own test sessions).")
    print("Per-class precision / recall / F1:")
    for label, metrics in report["per_class"].items():
        print(
            f"  {label}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
        )
    print(f"Confusion matrix (rows=true, columns=predicted; order={labels}):")
    print(np.asarray(report["confusion_matrix"]))
    print("Top confusions:")
    for confusion in report["top_confusions"]:
        print(f"  {confusion['description']}")
    save_artifacts(model, labels, report)
    print(f"Saved model, labels, metadata, and report to {config.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-ds-only",
        action="store_true",
        help="Allow labels with fewer than 2 own sessions; evaluation may be incomplete.",
    )
    train(parser.parse_args().allow_ds_only)
