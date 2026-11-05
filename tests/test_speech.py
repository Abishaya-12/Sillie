"""Tests for the authenticated Gnani speech API proxy."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.extensions import db
from app.models import User
from app.speech import MAX_AUDIO_SIZE, MAX_TEXT_LENGTH

CSRF_PATTERN = re.compile(rb'name="csrf-token" content="([^"]+)"')


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Flask]:
    """Create an isolated app and authenticated account for speech tests."""
    application = create_app({
        "TESTING": True,
        "DEBUG": True,
        "SECRET_KEY": "speech-tests-only-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "DATA_DIR": tmp_path,
        "GNANI_API_KEY": "server-only-test-key",
    })
    with application.app_context():
        db.session.add(User(
            username="speech_user",
            password_hash="unused-test-hash",
            display_name="Speech User",
            accessibility_profile="hearing",
        ))
        db.session.commit()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Return a test client signed in as the speech test user."""
    test_client = app.test_client()
    with app.app_context():
        user = db.session.execute(
            db.select(User).where(User.username == "speech_user")
        ).scalar_one()
        user_id = str(user.id)
    with test_client.session_transaction() as session:
        session["_user_id"] = user_id
        session["_fresh"] = True
    return test_client


def csrf_headers(client: FlaskClient) -> dict[str, str]:
    """Fetch a CSRF token from the translator page for API POST requests."""
    page = client.get("/translate")
    match = CSRF_PATTERN.search(page.data)
    assert match is not None
    return {"X-CSRFToken": match.group(1).decode()}


def fake_response(
    *, content: bytes = b"", payload: dict[str, str] | None = None,
    content_type: str = "audio/wav", status_code: int = 200,
) -> Mock:
    """Build a small requests.Response-compatible mock."""
    response = Mock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.content = content
    response.headers = {"Content-Type": content_type}
    response.json.return_value = payload or {}
    return response


def test_tts_forwards_documented_request_and_returns_audio(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TTS uses the documented route, header, model, voice, and WAV response."""
    upstream = fake_response(content=b"wav-audio")
    post = Mock(return_value=upstream)
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/tts",
        json={"text": "नमस्ते", "language": "hi-IN", "voice": "Shivani"},
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    assert response.data == b"wav-audio"
    assert response.mimetype == "audio/wav"
    args, kwargs = post.call_args
    assert args == ("https://api.vachana.ai/api/v1/tts/inference",)
    assert kwargs["headers"] == {"X-API-Key-ID": "server-only-test-key"}
    assert kwargs["json"] == {
        "text": "नमस्ते",
        "voice": "Shivani",
        "model": "timbre-v3.0",
        "language": "hi-IN",
        "audio_config": {"container": "wav"},
    }


def test_stt_forwards_multipart_request_and_returns_transcript(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STT sends audio_file and language_code fields and returns transcript text."""
    post = Mock(return_value=fake_response(payload={"transcript": "hello"}))
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/stt",
        data={
            "language": "en-IN",
            "audio": (io.BytesIO(b"audio-data"), "sample.wav", "audio/wav"),
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    assert response.json == {"text": "hello"}
    args, kwargs = post.call_args
    assert args == ("https://api.vachana.ai/stt/v3",)
    assert kwargs["headers"] == {"X-API-Key-ID": "server-only-test-key"}
    assert kwargs["data"] == {
        "model": "prisma-v3.0",
        "language_code": "en-IN",
    }
    assert kwargs["files"]["audio_file"] == (
        "sample.wav", b"audio-data", "audio/wav"
    )


@pytest.mark.parametrize("language", ["indic-auto", "hi-IN,en-IN"])
def test_stt_accepts_prisma_v3_language_options(
    client: FlaskClient,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
) -> None:
    """Prisma v3.0 accepts language detection and distinct language lists."""
    post = Mock(return_value=fake_response(payload={"transcript": "hello"}))
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/stt",
        data={
            "language": language,
            "audio": (io.BytesIO(b"audio-data"), "sample.wav", "audio/wav"),
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    assert post.call_args.kwargs["data"] == {
        "model": "prisma-v3.0",
        "language_code": language,
    }


def test_missing_api_key_returns_503_without_calling_gnani(
    app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing credentials produce a clear response and no outbound request."""
    app.config["GNANI_API_KEY"] = None
    post = Mock()
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/tts",
        json={"text": "hello", "language": "en-IN", "voice": "Shivani"},
        headers=csrf_headers(client),
    )

    assert response.status_code == 503
    assert "GNANI_API_KEY" in response.json["error"]
    post.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"text": "", "language": "hi-IN", "voice": "Shivani"}, 400),
        ({"text": "x" * (MAX_TEXT_LENGTH + 1), "language": "hi-IN", "voice": "Shivani"}, 400),
        ({"text": "hello", "language": "xx-XX", "voice": "Shivani"}, 400),
    ],
)
def test_tts_rejects_invalid_input(
    client: FlaskClient,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, str],
    expected_status: int,
) -> None:
    """TTS validates text bounds and language before contacting Gnani."""
    post = Mock()
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/tts", json=payload, headers=csrf_headers(client)
    )

    assert response.status_code == expected_status
    post.assert_not_called()


def test_stt_rejects_oversized_audio(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STT rejects uploads above the configured 10 MB limit."""
    post = Mock()
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/stt",
        data={
            "language": "hi-IN",
            "audio": (io.BytesIO(b"x" * (MAX_AUDIO_SIZE + 1)), "large.wav"),
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 413
    post.assert_not_called()


def test_stt_rejects_unsupported_audio_format(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STT only accepts the audio filename extensions documented by Gnani."""
    post = Mock()
    monkeypatch.setattr("app.speech.requests.post", post)

    response = client.post(
        "/api/speech/stt",
        data={
            "language": "hi-IN",
            "audio": (io.BytesIO(b"audio-data"), "sample.exe"),
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 400
    post.assert_not_called()


def test_speech_routes_require_login(app: Flask) -> None:
    """Both speech endpoints require an authenticated user session."""
    client = app.test_client()
    headers = csrf_headers(client)

    for path in ("/api/speech/tts", "/api/speech/stt"):
        response = client.post(path, json={}, headers=headers)
        assert response.status_code == 302
        assert response.location.startswith("/login?next=")