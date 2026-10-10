"""Flask application factory for Sillie."""

from __future__ import annotations

from flask import Flask

from config import FEATURES_PER_FRAME, IDLE_LABEL, SAMPLE_RATE_HZ, SEQUENCE_LENGTH, config


def create_app() -> Flask:
    """Create and configure the application instance."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    app.config["DEBUG"] = config.debug
    app.config["HOST"] = config.host
    app.config["PORT"] = config.port
    app.config["DATA_DIR"] = config.data_dir
    app.config["MODEL_DIR"] = config.model_dir
    app.config["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH
    app.config["SAMPLE_RATE_HZ"] = SAMPLE_RATE_HZ
    app.config["FEATURES_PER_FRAME"] = FEATURES_PER_FRAME
    app.config["IDLE_LABEL"] = IDLE_LABEL

    from .routes import blueprint
    from ml.inference import SignClassifier

    app.register_blueprint(blueprint)
    app.extensions["sign_classifier"] = SignClassifier(app.config["MODEL_DIR"])
    return app
