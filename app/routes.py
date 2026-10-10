"""Route definitions for the Sillie web app."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, TypeGuard

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from config import (
    FEATURES_PER_FRAME,
    IDLE_LABEL,
    SAMPLE_RATE_HZ,
    SEQUENCE_LENGTH,
)
from ml.inference import ModelConfigurationError, ModelNotFoundError, SignClassifier

blueprint = Blueprint("app", __name__)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@blueprint.route("/")
def index() -> str:
    """Serve the main HTML page."""
    return render_template("index.html")


@blueprint.route("/collect")
def collect() -> str:
    """Serve the labelled sample collection page."""
    return render_template("collect.html")


@blueprint.route("/api/health")
def health() -> tuple[Response, int]:
    """Return a lightweight health-check payload."""
    return jsonify({"status": "ok", "service": "sillie"}), 200


@blueprint.route("/api/config")
def api_config() -> Response:
    """Expose the shared sequence and feature dimensions to browser clients."""
    return jsonify({
        "sequence_length": SEQUENCE_LENGTH,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "features_per_frame": FEATURES_PER_FRAME,
        "idle_label": IDLE_LABEL,
    })


@blueprint.route("/api/samples", methods=["POST"])
def save_samples() -> tuple[Response, int]:
    """Validate and append normalized hand samples to the label CSV."""
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    label = payload.get("label")
    if not is_valid_label(label):
        return jsonify({
            "error": "label must contain only letters, numbers, dashes, or underscores."
        }), 400

    session = payload.get("session")
    if not is_valid_label(session):
        return jsonify({
            "error": "session must contain only letters, numbers, dashes, or underscores."
        }), 400

    samples = normalize_sample_rows(
        payload.get("samples"), SEQUENCE_LENGTH * FEATURES_PER_FRAME
    )
    if samples is None:
        return jsonify({
            "error": (
                "samples must be a list of rows with exactly "
                f"{SEQUENCE_LENGTH * FEATURES_PER_FRAME} finite numbers. "
                "The format changed; re-record samples using the current sequence collector."
            )
        }), 400

    data_dir: Path = current_app.config["DATA_DIR"]
    label_dir = data_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sample_file = label_dir / f"{session}.csv"
    with sample_file.open("a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerows(samples)

    total_samples, session_count = count_label_samples(
        data_dir, label, SEQUENCE_LENGTH * FEATURES_PER_FRAME
    )
    return jsonify({
        "label": label,
        "samples": total_samples,
        "sessions": session_count,
    }), 200


@blueprint.route("/api/samples/summary")
def sample_summary() -> Response:
    """Return the number of stored samples for each valid label."""
    data_dir: Path = current_app.config["DATA_DIR"]
    labels = {
        sample_file.stem
        for sample_file in data_dir.glob("*.csv")
        if is_valid_label(sample_file.stem)
    } if data_dir.exists() else set()
    if data_dir.exists():
        labels.update(
            label_dir.name
            for label_dir in data_dir.iterdir()
            if label_dir.is_dir()
            and is_valid_label(label_dir.name)
            and any(label_dir.glob("*.csv"))
        )
    counts = {
        label: dict(zip(
            ("samples", "sessions"),
            count_label_samples(data_dir, label, SEQUENCE_LENGTH * FEATURES_PER_FRAME),
        ))
        for label in sorted(labels)
    }
    return jsonify(counts)


@blueprint.route("/api/predict", methods=["POST"])
def predict_sign() -> tuple[Response, int]:
    """Validate landmark features and return the classifier's prediction."""
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    if "features" in payload:
        return jsonify({
            "error": "The prediction format changed: send a sequence in the 'window' field."
        }), 400

    window = normalize_feature_vector(
        payload.get("window"), SEQUENCE_LENGTH * FEATURES_PER_FRAME
    )
    if window is None:
        return jsonify({
            "error": (
                "window must be a list of exactly "
                f"{SEQUENCE_LENGTH * FEATURES_PER_FRAME} finite numbers."
            )
        }), 400

    classifier: SignClassifier = current_app.extensions["sign_classifier"]
    try:
        label, confidence = classifier.predict(window)
    except ModelNotFoundError:
        return jsonify({
            "error": "No trained model found. Run `python -m ml.train` first."
        }), 503
    except ModuleNotFoundError as error:
        if error.name != "tensorflow":
            raise
        return jsonify({
            "error": (
                "TensorFlow is not installed in the Python environment running the "
                "server. Start it with `.venv\\Scripts\\python.exe server.py` after "
                "installing `requirements-ml.txt` in that environment."
            )
        }), 503
    except ModelConfigurationError as error:
        return jsonify({"error": str(error)}), 503
    except ValueError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify({"label": label, "confidence": confidence}), 200


@blueprint.route("/api/model/status")
def model_status() -> Response:
    """Report cached model state and labels without loading model dependencies."""
    classifier: SignClassifier = current_app.extensions["sign_classifier"]
    loaded, labels = classifier.status()
    return jsonify({"loaded": loaded, "labels": labels})


def is_valid_label(label: object) -> TypeGuard[str]:
    """Return whether a label is safe for use as a dataset filename."""
    return isinstance(label, str) and LABEL_PATTERN.fullmatch(label) is not None


def normalize_sample_rows(
    samples: object, expected_length: int
) -> list[list[float]] | None:
    """Validate sample rows and convert their numeric values to floats."""
    if not isinstance(samples, list):
        return None

    rows: list[list[float]] = []
    for sample in samples:
        if not isinstance(sample, list) or len(sample) != expected_length:
            return None
        if any(not is_finite_number(value) for value in sample):
            return None
        rows.append([float(value) for value in sample])
    return rows


def normalize_feature_vector(
    features: object, expected_length: int
) -> list[float] | None:
    """Validate and normalize one configured sequence feature vector."""
    if not isinstance(features, list) or len(features) != expected_length:
        return None
    if any(not is_finite_number(value) for value in features):
        return None
    return [float(value) for value in features]


def is_finite_number(value: object) -> bool:
    """Return whether a JSON numeric value can be stored as a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def count_samples(sample_file: Path, expected_length: int) -> int:
    """Count valid sequence-window CSV rows in a sample file."""
    with sample_file.open("r", newline="", encoding="utf-8") as csv_file:
        return sum(1 for row in csv.reader(csv_file) if len(row) == expected_length)


def count_label_samples(
    data_dir: Path, label: str, expected_length: int
) -> tuple[int, int]:
    """Count valid sequence windows and sessions for a label."""
    label_dir = data_dir / label
    session_files = list(label_dir.glob("*.csv")) if label_dir.exists() else []
    legacy_file = data_dir / f"{label}.csv"
    sample_count = sum(count_samples(path, expected_length) for path in session_files)
    session_count = len(session_files)
    if legacy_file.is_file():
        session_count += 1
    return sample_count, session_count
