"""Tests for admin access, last-seen throttling, and socket presence."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db
from app.models import User
from app.signaling import (
    active_rooms,
    connected_user_sockets,
    online_user_ids,
    rooms,
    rooms_lock,
    signal_times,
    socket_rooms,
    socket_users,
    socketio,
)


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Flask]:
    """Create an isolated database and test users for the admin panel."""
    application = create_app({
        "TESTING": True,
        "DEBUG": True,
        "SECRET_KEY": "admin-tests-only-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "DATA_DIR": tmp_path,
    })
    with rooms_lock:
        rooms.clear()
        socket_rooms.clear()
        signal_times.clear()
        socket_users.clear()
        connected_user_sockets.clear()
    with application.app_context():
        db.session.add_all([
            User(
                username="player_admin",
                password_hash=generate_password_hash("test-password"),
                display_name="Admin Player",
                accessibility_profile="hearing",
                is_admin=True,
            ),
            User(
                username="player_normal",
                password_hash=generate_password_hash("test-password"),
                display_name="Normal Player",
                accessibility_profile="deaf",
            ),
        ])
        db.session.commit()
    yield application
    with rooms_lock:
        rooms.clear()
        socket_rooms.clear()
        signal_times.clear()
        socket_users.clear()
        connected_user_sockets.clear()
    with application.app_context():
        db.session.remove()
        db.drop_all()


def logged_in_client(app: Flask, username: str) -> FlaskClient:
    """Return a test client logged in as the named test account."""
    client = app.test_client()
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == username)
        ).scalar_one()
        user_id = str(user.id)
    with client.session_transaction() as session:
        session["_user_id"] = user_id
        session["_fresh"] = True
    return client


def test_anonymous_and_non_admin_receive_not_found(app: Flask) -> None:
    """Both admin UI and status API hide existence from unauthorized users."""
    for path in ("/admin", "/admin/api/status"):
        assert app.test_client().get(path).status_code == 404
        assert logged_in_client(app, "player_normal").get(path).status_code == 404


def test_admin_panel_and_status_exclude_credentials(app: Flask) -> None:
    """Admin pages expose only allowed account fields and disable response caching."""
    client = logged_in_client(app, "player_admin")
    page = client.get("/admin")
    response = client.get("/admin/api/status")

    assert page.status_code == 200
    assert b"ADMIN" in page.data
    assert b"password_hash" not in page.data
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "password_hash" not in response.get_data(as_text=True)
    assert "email" not in response.json


def test_make_admin_cli_command_updates_user(app: Flask) -> None:
    """The make-admin CLI command grants role access to an existing account."""
    result = app.test_cli_runner().invoke(
        args=["make-admin", "PLAYER_NORMAL"]
    )

    assert result.exit_code == 0
    assert "Administrator access granted" in result.output
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_normal")
        ).scalar_one()
        assert user.is_admin


def test_last_seen_is_throttled_for_sixty_seconds(app: Flask) -> None:
    """Recent requests do not rewrite last_seen until its 60-second interval passes."""
    client = logged_in_client(app, "player_normal")
    original = datetime.now(UTC) - timedelta(seconds=30)
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_normal")
        ).scalar_one()
        user.last_seen = original
        db.session.commit()

    assert client.get("/api/health").status_code == 200
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_normal")
        ).scalar_one()
        unchanged = user.last_seen
        if unchanged is not None and unchanged.tzinfo is None:
            unchanged = unchanged.replace(tzinfo=UTC)
        assert unchanged == original
        user.last_seen = datetime.now(UTC) - timedelta(seconds=61)
        db.session.commit()

    assert client.get("/api/health").status_code == 200
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_normal")
        ).scalar_one()
        assert user.last_seen is not None
        updated = user.last_seen
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        assert updated > original


def test_presence_counts_multiple_sockets_for_same_user(app: Flask) -> None:
    """A user remains online until their last Socket.IO session disconnects."""
    user_client = logged_in_client(app, "player_normal")
    first = socketio.test_client(app, flask_test_client=user_client)
    second = socketio.test_client(app, flask_test_client=user_client)
    with app.app_context():
        user_id = db.session.execute(
            db.select(User.id).where(User.username == "player_normal")
        ).scalar_one()

    assert online_user_ids() == {user_id}
    assert connected_user_sockets[user_id] == 2
    first.disconnect()
    assert online_user_ids() == {user_id}
    assert connected_user_sockets[user_id] == 1
    second.disconnect()
    assert online_user_ids() == set()
    assert active_rooms() == []
