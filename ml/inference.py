"""Lazy inference for the trained body-aware sequence classifier."""

from __future__ import annotations

import importlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    FEATURES_PER_FRAME,
    FEATURE_VERSION,
    RAW_FRAME_SIZE,
    SEQUENCE_LENGTH,
)
from ml.features import window_features

RAW_WINDOW_SIZE = SEQUENCE_LENGTH * RAW_FRAME_SIZE


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained sign model artifacts are unavailable."""


class ModelConfigurationError(ValueError):
    """Raised when saved model metadata differs from this feature pipeline."""


class SignClassifier:
    """Load and cache the Keras model only when a prediction is requested."""

    def __init__(self, model_dir: Path) -> None:
        """Create an unloaded classifier using the supplied model directory."""
        self._model_dir = model_dir
        self._model: Any | None = None
        self._labels: list[str] | None = None
        self._load_lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        """Return the trained model path."""
        return self._model_dir / "sign_model.keras"

    @property
    def labels_path(self) -> Path:
        """Return the class-label list path."""
        return self._model_dir / "labels.json"

    @property
    def metadata_path(self) -> Path:
        """Return the feature metadata path."""
        return self._model_dir / "meta.json"

    def predict(self, window: list[float]) -> list[tuple[str, float]]:
        """Return the top three label scores for one flat raw sequence window."""
        if len(window) != RAW_WINDOW_SIZE or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in window
        ):
            raise ValueError(
                f"window must contain exactly {RAW_WINDOW_SIZE} finite raw coordinates."
            )

        self._ensure_loaded()
        raw_window = np.asarray(window, dtype=np.float32).reshape(
            SEQUENCE_LENGTH, RAW_FRAME_SIZE
        )
        features = window_features(raw_window)[None, ...]
        probabilities = np.asarray(
            self._model.predict(features, verbose=0)[0], dtype=np.float64
        )
        if probabilities.ndim != 1 or len(probabilities) != len(self._labels):
            raise ValueError("The model output count does not match its labels.json file.")
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(self._labels))]
        return [
            (self._labels[int(index)], float(probabilities[index]))
            for index in top_indices
        ]

    def status(self) -> tuple[bool, list[str]]:
        """Report artifact availability without importing TensorFlow."""
        required = (self.model_path, self.labels_path, self.metadata_path)
        if not all(path.is_file() for path in required):
            return False, []
        if self._labels is None:
            self._labels = self._read_labels()
        return self._model is not None, list(self._labels)

    def configuration_error(self) -> str | None:
        """Return an incompatibility reason without loading TensorFlow, if any."""
        if not all(
            path.is_file() for path in (self.model_path, self.labels_path, self.metadata_path)
        ):
            return None
        try:
            self._validate_metadata()
        except ModelConfigurationError as error:
            return str(error)
        return None

    def _ensure_loaded(self) -> None:
        """Validate metadata and load the Keras model once, on first prediction."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            required = (self.model_path, self.labels_path, self.metadata_path)
            if not all(path.is_file() for path in required):
                raise ModelNotFoundError(
                    f"Trained model artifacts are missing from {self._model_dir}."
                )
            self._validate_metadata()
            labels = self._read_labels()
            tensorflow = importlib.import_module("tensorflow")
            model = tensorflow.keras.models.load_model(self.model_path)
            self._labels = labels
            self._model = model

    def _validate_metadata(self) -> None:
        """Reject model files trained against another raw/feature representation."""
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelConfigurationError(
                f"Could not read feature metadata from {self.metadata_path}; retrain the model."
            ) from error
        expected = {
            "sequence_length": SEQUENCE_LENGTH,
            "raw_frame_size": RAW_FRAME_SIZE,
            "features_per_frame": FEATURES_PER_FRAME,
            "feature_version": FEATURE_VERSION,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise ModelConfigurationError(
                "The trained model uses an incompatible feature format. "
                "Collect raw hand-and-pose windows and run `python -m ml.train` again."
            )

    def _read_labels(self) -> list[str]:
        """Read and validate class labels from labels.json."""
        try:
            labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read model labels from {self.labels_path}.") from error
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(f"Labels in {self.labels_path} must be a non-empty string list.")
        return labels
