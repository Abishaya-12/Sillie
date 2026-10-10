"""Lazy inference for trained Sillie sequence classifiers."""

from __future__ import annotations

import importlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from config import FEATURES_PER_FRAME, SEQUENCE_LENGTH

FEATURE_COUNT = SEQUENCE_LENGTH * FEATURES_PER_FRAME


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained sign model artifacts are unavailable."""


class ModelConfigurationError(ValueError):
    """Raised when model metadata is incompatible with current input dimensions."""


class SignClassifier:
    """Load and cache the trained sequence model only when prediction is needed."""

    def __init__(self, model_dir: Path) -> None:
        """Create an unloaded classifier using the configured model directory."""
        self._model_dir = model_dir
        self._model: Any | None = None
        self._labels: list[str] | None = None
        self._load_lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        """Return the trained Keras model path."""
        return self._model_dir / "sign_model.keras"

    @property
    def labels_path(self) -> Path:
        """Return the class-label JSON path."""
        return self._model_dir / "labels.json"

    @property
    def metadata_path(self) -> Path:
        """Return the model input metadata path."""
        return self._model_dir / "meta.json"

    def predict(self, window: list[float]) -> tuple[str, float]:
        """Return the most likely label and probability for one input sequence."""
        if len(window) != FEATURE_COUNT or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in window
        ):
            raise ValueError(
                f"window must contain exactly {FEATURE_COUNT} finite numbers."
            )

        self._ensure_loaded()
        input_window = np.asarray(window, dtype=np.float32).reshape(
            1, SEQUENCE_LENGTH, FEATURES_PER_FRAME
        )
        probabilities = self._model.predict(input_window, verbose=0)[0]
        class_index = int(probabilities.argmax())
        return self._labels[class_index], float(probabilities[class_index])

    def status(self) -> tuple[bool, list[str]]:
        """Return cached model state and labels without importing TensorFlow."""
        if not all(
            path.is_file()
            for path in (self.model_path, self.labels_path, self.metadata_path)
        ):
            return False, []
        if self._labels is None:
            self._labels = self._read_labels()
        return self._model is not None, list(self._labels)

    def _ensure_loaded(self) -> None:
        """Load the model once, validating its saved input metadata first."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            required_files = (self.model_path, self.labels_path, self.metadata_path)
            if not all(path.is_file() for path in required_files):
                raise ModelNotFoundError(
                    f"Trained model, labels, and metadata are required in {self._model_dir}."
                )

            self._validate_metadata()
            labels = self._read_labels()
            tensorflow = importlib.import_module("tensorflow")
            model = tensorflow.keras.models.load_model(self.model_path)
            self._labels = labels
            self._model = model

    def _validate_metadata(self) -> None:
        """Raise a clear error if saved dimensions differ from current config."""
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelConfigurationError(
                f"Could not read sequence metadata from {self.metadata_path}; retrain the model."
            ) from error
        expected = {
            "sequence_length": SEQUENCE_LENGTH,
            "features_per_frame": FEATURES_PER_FRAME,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise ModelConfigurationError(
                "The trained model's sequence dimensions do not match the current "
                "configuration. Run `python -m ml.train` to retrain it."
            )

    def _read_labels(self) -> list[str]:
        """Read and validate class labels from model metadata."""
        try:
            labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read model labels from {self.labels_path}.") from error
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(f"Model labels in {self.labels_path} must be a non-empty string list.")
        return labels
