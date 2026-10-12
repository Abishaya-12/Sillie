"""Flask application factory for Sillie."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask
from flask_login import current_user
from config import (
    FEATURES_PER_FRAME,
    FEATURE_VERSION,
    IDLE_LABEL,
    RAW_FRAME_SIZE,
    SAMPLE_RATE_HZ,
    SEQUENCE_LENGTH,
    config,
)
from .extensions import csrf, db, login_manager


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the application instance."""
    app = Flask(__name__)
    secret_key = os.getenv("SILLIE_SECRET_KEY") or config.secret_key
    app.config["SECRET_KEY"] = secret_key
    app.config["DEBUG"] = config.debug
    app.config["HOST"] = config.host
    app.config["PORT"] = config.port
    app.config["SQLALCHEMY_DATABASE_URI"] = config.database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SILLIE_TURN_URL"] = config.turn_url
    app.config["SILLIE_TURN_USERNAME"] = config.turn_username
    app.config["SILLIE_TURN_CREDENTIAL"] = config.turn_credential
    app.config["DATA_DIR"] = config.data_dir
    app.config["MODEL_DIR"] = config.model_dir
    app.config["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH
    app.config["SAMPLE_RATE_HZ"] = SAMPLE_RATE_HZ
    app.config["FEATURES_PER_FRAME"] = FEATURES_PER_FRAME
    app.config["RAW_FRAME_SIZE"] = RAW_FRAME_SIZE
    app.config["FEATURE_VERSION"] = FEATURE_VERSION
    app.config["IDLE_LABEL"] = IDLE_LABEL
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    if test_config is not None:
        app.config.update(test_config)
    if test_config is None or "SESSION_COOKIE_SECURE" not in test_config:
        app.config["SESSION_COOKIE_SECURE"] = not app.config["DEBUG"]
    if not app.config["DEBUG"] and not app.config.get("SECRET_KEY"):
        raise RuntimeError(
            "SILLIE_SECRET_KEY must be set when debug mode is disabled."
        )
    if not app.config["SECRET_KEY"]:
        app.config["SECRET_KEY"] = "local-development-only-key"

    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:"):
        Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = "auth.login"

    from .auth import auth_blueprint
    from .admin import admin_blueprint, register_admin_cli
    from .models import User
    from .routes import blueprint
    from .signaling import socketio
    from ml.inference import SignClassifier

    app.register_blueprint(blueprint)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(admin_blueprint)
    socketio.init_app(app, async_mode="threading")
    register_admin_cli(app)
    app.extensions["sign_classifier"] = SignClassifier(app.config["MODEL_DIR"])

    @app.before_request
    def update_authenticated_last_seen() -> None:
        """Update activity timestamps no more than once every 60 seconds."""
        if not current_user.is_authenticated:
            return
        now = datetime.now(UTC)
        last_seen = current_user.last_seen
        if last_seen is not None:
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=UTC)
            if (now - last_seen).total_seconds() < 60:
                return
        current_user.last_seen = now
        db.session.commit()

    with app.app_context():
        db.create_all()
    return app
