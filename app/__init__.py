"""Flask application factory for Sillie."""

from __future__ import annotations

from flask import Flask

from config import config


def create_app() -> Flask:
    """Create and configure the application instance."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    app.config["DEBUG"] = config.debug
    app.config["HOST"] = config.host
    app.config["PORT"] = config.port
    app.config["DATA_DIR"] = config.data_dir
    app.config["MODEL_DIR"] = config.model_dir

    from .routes import blueprint
    from ml.inference import SignClassifier

    app.register_blueprint(blueprint)
    app.extensions["sign_classifier"] = SignClassifier(app.config["MODEL_DIR"])
    return app
