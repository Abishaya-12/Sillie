"""Tests for account authentication and the protected lobby."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.auth import LOGIN_FAILURES
from app.extensions import db
from app.models import User

CSRF_PATTERN = re.compile(rb'name="csrf_token" value="([^"]+)"')


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Flask]:
    """Create an isolated test app and database for each test."""
    application = create_app({
        "TESTING": True,
        "DEBUG": True,
        "SECRET_KEY": "auth-tests-only-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "DATA_DIR": tmp_path,
    })
    with application.app_context():
        db.create_all()
    LOGIN_FAILURES.clear()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Return a test client for the isolated application."""
    return app.test_client()


def csrf_token(client: FlaskClient, path: str = "/login") -> str:
    """Fetch a page and extract the CSRF token rendered in its form."""
    response = client.get(path)
    match = CSRF_PATTERN.search(response.data)
    assert match is not None
    return match.group(1).decode()


def register_user(client: FlaskClient, username: str = "player_one") -> None:
    """Register a valid user and assert that registration redirects."""
    token = csrf_token(client, "/register")
    response = client.post(
        "/register",
        data={
            "csrf_token": token,
            "username": username,
            "display_name": "Player One",
            "password": "long-enough-password",
            "confirm_password": "long-enough-password",
            "accessibility_profile": "hearing",
        },
    )
    assert response.status_code == 302
    assert response.location.endswith("/lobby")


def test_register_stores_lowercase_username_and_password_hash(
    app: Flask, client: FlaskClient
) -> None:
    """Registration lowercases usernames and never stores plain-text passwords."""
    token = csrf_token(client, "/register")
    response = client.post(
        "/register",
        data={
            "csrf_token": token,
            "username": "PLAYER_ONE",
            "display_name": "Player One",
            "password": "long-enough-password",
            "confirm_password": "long-enough-password",
            "accessibility_profile": "deaf",
            "is_admin": "true",
        },
    )
    assert response.status_code == 302
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_one")
        ).scalar_one()
        assert user.password_hash != "long-enough-password"
        assert user.accessibility_profile == "deaf"
        assert not user.is_admin


def test_profile_page_updates_accessibility_profile_only(
    app: Flask, client: FlaskClient
) -> None:
    """Users can change their accessibility profile but not admin status."""
    register_user(client)
    page = client.get("/profile")
    match = CSRF_PATTERN.search(page.data)
    assert page.status_code == 200
    assert match is not None

    response = client.post(
        "/profile",
        data={
            "csrf_token": match.group(1).decode(),
            "accessibility_profile": "mute",
            "is_admin": "true",
        },
    )

    assert response.status_code == 302
    assert response.location.endswith("/profile")
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "player_one")
        ).scalar_one()
        assert user.accessibility_profile == "mute"
        assert not user.is_admin


def test_duplicate_username_is_rejected(app: Flask, client: FlaskClient) -> None:
    """A username remains unique regardless of the submitted letter case."""
    register_user(client)
    second_client = app.test_client()
    token = csrf_token(second_client, "/register")
    response = second_client.post(
        "/register",
        data={
            "csrf_token": token,
            "username": "PLAYER_ONE",
            "display_name": "Another Player",
            "password": "long-enough-password",
            "confirm_password": "long-enough-password",
            "accessibility_profile": "hearing",
        },
    )
    assert response.status_code == 200
    assert b"username is already taken" in response.data
    with app.app_context():
        assert db.session.execute(db.select(db.func.count(User.id))).scalar_one() == 1


def test_login_success_failure_and_logout(client: FlaskClient) -> None:
    """Valid credentials log in, invalid ones remain generic, and logout clears auth."""
    register_user(client)
    token = csrf_token(client, "/lobby")
    client.post("/logout", data={"csrf_token": token})

    token = csrf_token(client)
    bad_response = client.post(
        "/login",
        data={
            "csrf_token": token,
            "username": "player_one",
            "password": "wrong-password",
        },
    )
    assert bad_response.status_code == 200
    assert b"Invalid username or password." in bad_response.data

    response = client.post(
        "/login",
        data={
            "csrf_token": token,
            "username": "PLAYER_ONE",
            "password": "long-enough-password",
        },
    )
    assert response.status_code == 302
    assert response.location.endswith("/lobby")
    assert client.get("/lobby").status_code == 200

    response = client.post("/logout", data={"csrf_token": token})
    assert response.status_code == 302
    assert response.location.endswith("/login")
    assert client.get("/lobby").status_code == 302


def test_protected_routes_redirect_to_login(client: FlaskClient) -> None:
    """Lobby, calls, collection pages, and sample endpoints require login."""
    for path in (
        "/lobby",
        "/call/ABC234",
        "/collect",
        "/api/samples/summary",
    ):
        response = client.get(path)
        assert response.status_code == 302
        assert response.location.startswith("/login?next=")

    response = client.post("/api/samples", json={})
    assert response.status_code == 302
    assert response.location.startswith("/login?next=")


def test_login_rejects_external_next_destination(client: FlaskClient) -> None:
    """Successful login falls back to the lobby for external redirect targets."""
    register_user(client)
    token = csrf_token(client, "/lobby")
    client.post("/logout", data={"csrf_token": token})

    token = csrf_token(client)
    response = client.post(
        "/login?next=%2F%2Fevil.example",
        data={
            "csrf_token": token,
            "username": "player_one",
            "password": "long-enough-password",
            "next": "//evil.example",
        },
    )
    assert response.status_code == 302
    assert response.location == "/lobby"


def test_fifth_failed_login_blocks_for_five_minutes(client: FlaskClient) -> None:
    """The fifth failure blocks that username and source IP from logging in."""
    register_user(client)
    token = csrf_token(client, "/lobby")
    client.post("/logout", data={"csrf_token": token})
    token = csrf_token(client)

    for _ in range(5):
        response = client.post(
            "/login",
            data={
                "csrf_token": token,
                "username": "player_one",
                "password": "wrong-password",
            },
        )
        assert response.status_code == 200

    blocked = client.post(
        "/login",
        data={
            "csrf_token": token,
            "username": "player_one",
            "password": "long-enough-password",
        },
    )
    assert blocked.status_code == 200
    assert b"Unable to sign in." in blocked.data
    assert client.get("/lobby").status_code == 302


def test_csrf_is_required_for_html_forms(client: FlaskClient) -> None:
    """Account and lobby POST forms reject requests without a CSRF token."""
    assert client.post(
        "/register",
        data={
            "username": "player_one",
            "display_name": "Player One",
            "password": "long-enough-password",
            "confirm_password": "long-enough-password",
            "accessibility_profile": "hearing",
        },
    ).status_code == 400


def test_create_room_generates_valid_code_and_placeholder(
    client: FlaskClient,
) -> None:
    """The lobby creates a valid code and renders its protected room page."""
    register_user(client)
    token = csrf_token(client, "/lobby")
    response = client.post(
        "/lobby/create-room", data={"csrf_token": token}
    )
    assert response.status_code == 302
    assert re.fullmatch(r"/call/[A-HJ-NP-Z2-9]{6}", response.location)
    assert client.get(response.location).status_code == 200


def test_non_debug_app_requires_a_secret_key() -> None:
    """Production configuration fails fast instead of using a development key."""
    with pytest.raises(RuntimeError, match="SILLIE_SECRET_KEY"):
        create_app({"DEBUG": False, "SECRET_KEY": None})
