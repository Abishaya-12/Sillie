"""Route definitions for the Sillie web app."""

from __future__ import annotations

import csv
import re
import secrets
from pathlib import Path
from typing import Any, TypeGuard

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from config import (
    FEATURES_PER_FRAME,
    FEATURE_VERSION,
    IDLE_LABEL,
    RAW_FRAME_SIZE,
    SAMPLE_RATE_HZ,
    SEQUENCE_LENGTH,
)
from .extensions import csrf
from .storage import is_finite_number, normalize_sample_rows, write_sample_rows
from ml.inference import ModelConfigurationError, ModelNotFoundError, SignClassifier

blueprint = Blueprint("app", __name__)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@blueprint.route("/")
def index() -> Response:
    """Direct visitors to the lobby or the login page."""
    endpoint = "app.lobby" if current_user.is_authenticated else "auth.login"
    return redirect(url_for(endpoint))


@blueprint.get("/translate")
def translate() -> str:
    """Serve the public sign translator."""
    return render_template("index.html")


@blueprint.get("/lobby")
@login_required
def lobby() -> str:
    """Serve the authenticated user lobby."""
    return render_template("lobby.html")


@blueprint.post("/lobby/create-room")
@login_required
def create_room() -> Response:
    """Generate a new six-character room code and open its call page."""
    code = "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(6))
    return redirect(url_for("app.call_room", code=code))


@blueprint.post("/lobby/join-room")
@login_required
def join_room() -> Response:
    """Validate the entered room code and open its call page."""
    code = request.form.get("room_code", "").strip().upper()
    if len(code) != 6 or any(character not in ROOM_CODE_ALPHABET for character in code):
        flash("Enter a valid 6-character room code.", "error")
        return redirect(url_for("app.lobby"))
    return redirect(url_for("app.call_room", code=code))


@blueprint.get("/call/<code>")
@login_required
def call_room(code: str) -> str | tuple[str, int]:
    """Render the authenticated one-to-one video call page."""
    if len(code) != 6 or any(character not in ROOM_CODE_ALPHABET for character in code):
        return render_template("lobby.html"), 404
    return render_template("call.html", room_code=code)


@blueprint.get("/api/ice-config")
@login_required
def ice_config() -> Response:
    """Return public STUN settings and a fully configured optional TURN server."""
    ice_servers: list[dict[str, Any]] = [
        {"urls": "stun:stun.l.google.com:19302"}
    ]
    turn_url = current_app.config.get("SILLIE_TURN_URL")
    turn_username = current_app.config.get("SILLIE_TURN_USERNAME")
    turn_credential = current_app.config.get("SILLIE_TURN_CREDENTIAL")
    if turn_url and turn_username and turn_credential:
        ice_servers.append({
            "urls": turn_url,
            "username": turn_username,
            "credential": turn_credential,
        })
    return jsonify({"iceServers": ice_servers})


@blueprint.route("/collect")
@login_required
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
        "raw_frame_size": RAW_FRAME_SIZE,
        "features_per_frame": FEATURES_PER_FRAME,
        "idle_label": IDLE_LABEL,
        "feature_version": FEATURE_VERSION,
    })


@blueprint.route("/api/samples", methods=["POST"])
@login_required
@csrf.exempt
def save_samples() -> tuple[Response, int]:
    """Validate and append raw hand-and-pose sequence windows to session storage."""
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
        payload.get("samples"), SEQUENCE_LENGTH * RAW_FRAME_SIZE
    )
    if samples is None:
        return jsonify({
            "error": (
                "samples must be a list of rows with exactly "
                f"{SEQUENCE_LENGTH * RAW_FRAME_SIZE} finite numbers. "
                "The format changed to raw hand-and-pose sequences; re-record samples."
            )
        }), 400

    data_dir: Path = current_app.config["DATA_DIR"]
    write_sample_rows(
        data_dir,
        label,
        session,
        samples,
        SEQUENCE_LENGTH * RAW_FRAME_SIZE,
    )

    total_samples, session_count = count_label_samples(
        data_dir, label, SEQUENCE_LENGTH * RAW_FRAME_SIZE
    )
    return jsonify({
        "label": label,
        "samples": total_samples,
        "sessions": session_count,
    }), 200


@blueprint.route("/api/samples/summary")
@login_required
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
            count_label_samples(data_dir, label, SEQUENCE_LENGTH * RAW_FRAME_SIZE),
        ))
        for label in sorted(labels)
    }
    return jsonify(counts)


@blueprint.route("/api/predict", methods=["POST"])
@csrf.exempt
def predict_sign() -> tuple[Response, int]:
    """Validate a raw landmark sequence and return ranked sign predictions."""
    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    if "features" in payload:
        return jsonify({
            "error": (
                "The prediction format changed to raw hand-and-pose sequences; "
                f"send a {SEQUENCE_LENGTH * RAW_FRAME_SIZE}-value window in the 'window' field."
            )
        }), 400

    window = normalize_raw_window(
        payload.get("window"), SEQUENCE_LENGTH * RAW_FRAME_SIZE
    )
    if window is None:
        return jsonify({
            "error": (
                "window must be a list of exactly "
                f"{SEQUENCE_LENGTH * RAW_FRAME_SIZE} finite numbers."
            )
        }), 400

    classifier: SignClassifier = current_app.extensions["sign_classifier"]
    try:
        predictions = classifier.predict(window)
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
    if not predictions:
        return jsonify({"error": "The classifier returned no predictions."}), 503
    top = [
        {"label": label, "confidence": confidence}
        for label, confidence in predictions
    ]
    return jsonify({
        "label": top[0]["label"],
        "confidence": top[0]["confidence"],
        "top": top,
    }), 200


@blueprint.route("/api/model/status")
def model_status() -> Response:
    """Report cached model state and labels without loading model dependencies."""
    classifier: SignClassifier = current_app.extensions["sign_classifier"]
    loaded, labels = classifier.status()
    configuration_error = classifier.configuration_error()
    return jsonify({
        "loaded": loaded,
        "labels": labels,
        "compatible": configuration_error is None,
        "error": configuration_error,
    })


def is_valid_label(label: object) -> TypeGuard[str]:
    """Return whether a label is safe for use as a dataset filename."""
    return isinstance(label, str) and LABEL_PATTERN.fullmatch(label) is not None


def normalize_raw_window(
    features: object, expected_length: int
) -> list[float] | None:
    """Validate and convert one flat raw landmark sequence to Python floats."""
    if not isinstance(features, list) or len(features) != expected_length:
        return None
    if any(not is_finite_number(value) for value in features):
        return None
    return [float(value) for value in features]


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
