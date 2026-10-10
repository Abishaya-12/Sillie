"""Lazy inference for trained Sillie sign classifiers."""

from __future__ import annotations

import importlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COUNT = 63


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained sign model artifacts are unavailable."""


class SignClassifier:
    """Load and cache the trained model only when a prediction is requested."""

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

    def predict(self, features: list[float]) -> tuple[str, float]:
        """Return the most likely label and its probability for 63 features."""
        if len(features) != FEATURE_COUNT or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in features
        ):
            raise ValueError("features must contain exactly 63 finite numbers.")

        self._ensure_loaded()
        input_features = np.asarray([features], dtype=np.float32)
        probabilities = self._model.predict(input_features, verbose=0)[0]
        class_index = int(probabilities.argmax())
        return self._labels[class_index], float(probabilities[class_index])

    def status(self) -> tuple[bool, list[str]]:
        """Return model-loaded state and labels without importing TensorFlow."""
        if not self.model_path.is_file() or not self.labels_path.is_file():
            return False, []
        if self._labels is None:
            self._labels = self._read_labels()
        return self._model is not None, list(self._labels)

    def _ensure_loaded(self) -> None:
        """Load both artifacts once, importing TensorFlow only when needed."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_file() or not self.labels_path.is_file():
                raise ModelNotFoundError(
                    f"Trained model files are missing from {self._model_dir}."
                )

            labels = self._read_labels()
            tensorflow = importlib.import_module("tensorflow")
            model = tensorflow.keras.models.load_model(self.model_path)
            self._labels = labels
            self._model = model

    def _read_labels(self) -> list[str]:
        """Read and validate class labels from the model metadata."""
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
