"""Authenticated Flask-SocketIO room and ICE configuration tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient
from flask_socketio import SocketIOTestClient
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db
from app.models import User
from app.signaling import (
    MAX_SIGNALS_PER_SECOND,
    rooms,
    rooms_lock,
    signal_times,
    socket_rooms,
    socketio,
)

ROOM_CODE = "ABC234"


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Flask]:
    """Create an isolated app and seed two users for socket tests."""
    application = create_app({
        "TESTING": True,
        "DEBUG": True,
        "SECRET_KEY": "signaling-tests-only-secret",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True},
        "DATA_DIR": tmp_path,
    })
    with rooms_lock:
        rooms.clear()
        socket_rooms.clear()
        signal_times.clear()
    with application.app_context():
        for username in ("caller_one", "caller_two", "caller_three"):
            db.session.add(User(
                username=username,
                password_hash=generate_password_hash("test-password"),
                display_name=username.replace("_", " ").title(),
                accessibility_profile="hearing",
            ))
        db.session.commit()
    yield application
    with rooms_lock:
        rooms.clear()
        socket_rooms.clear()
        signal_times.clear()
    with application.app_context():
        db.session.remove()
        db.drop_all()


def authenticated_client(app: Flask, username: str) -> FlaskClient:
    """Create a Flask test client with the selected user in its session."""
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


def socket_client(app: Flask, username: str) -> SocketIOTestClient:
    """Connect a Socket.IO test client using an authenticated Flask session."""
    return socketio.test_client(
        app,
        flask_test_client=authenticated_client(app, username),
    )


def event_payload(
    client: SocketIOTestClient, event_name: str
) -> list[dict[str, Any]]:
    """Return received event argument payloads for the requested event."""
    return [
        event["args"][0] if event["args"] else {}
        for event in client.get_received()
        if event["name"] == event_name
    ]


def test_join_notifies_both_participants(app: Flask) -> None:
    """The joining peer receives the present user and the existing peer is notified."""
    first = socket_client(app, "caller_one")
    second = socket_client(app, "caller_two")
    first.emit("join", {"room": ROOM_CODE})
    second.emit("join", {"room": ROOM_CODE})

    assert event_payload(second, "peer_present") == [{
        "display_name": "Caller One",
        "profile": "hearing",
    }]
    assert event_payload(first, "peer_joined") == [{
        "display_name": "Caller Two",
        "profile": "hearing",
    }]
    first.disconnect()
    second.disconnect()


def test_third_participant_gets_room_full(app: Flask) -> None:
    """A room with two participants rejects a third socket."""
    first = socket_client(app, "caller_one")
    second = socket_client(app, "caller_two")
    third = socket_client(app, "caller_three")
    first.emit("join", {"room": ROOM_CODE})
    second.emit("join", {"room": ROOM_CODE})
    third.emit("join", {"room": ROOM_CODE})

    assert event_payload(third, "room_full") == [{}]
    assert len(rooms[ROOM_CODE]) == 2
    first.disconnect()
    second.disconnect()
    third.disconnect()


def test_signal_is_relayed_only_to_the_same_room_peer(app: Flask) -> None:
    """Offers are relayed to the room peer and not to unrelated sockets."""
    first = socket_client(app, "caller_one")
    peer = socket_client(app, "caller_two")
    stranger = socket_client(app, "caller_three")
    first.emit("join", {"room": ROOM_CODE})
    peer.emit("join", {"room": ROOM_CODE})
    stranger.emit("join", {"room": "DEF567"})

    first.emit("signal", {"type": "offer", "payload": {"sdp": "test"}})
    assert event_payload(peer, "signal") == [{
        "type": "offer",
        "payload": {"sdp": "test"},
    }]
    assert event_payload(stranger, "signal") == []
    stranger.disconnect()
    first.disconnect()
    peer.disconnect()


def test_unauthenticated_socket_connection_is_rejected(app: Flask) -> None:
    """Sockets without a Flask-Login session cannot connect."""
    client = socketio.test_client(app, flask_test_client=app.test_client())

    assert not client.is_connected()


def test_disconnect_notifies_peer_and_deletes_empty_room(app: Flask) -> None:
    """Disconnecting a member emits peer_left; the final leave deletes the room."""
    first = socket_client(app, "caller_one")
    second = socket_client(app, "caller_two")
    first.emit("join", {"room": ROOM_CODE})
    second.emit("join", {"room": ROOM_CODE})

    second.disconnect()
    assert event_payload(first, "peer_left") == [{}]
    first.disconnect()
    assert ROOM_CODE not in rooms


def test_signals_are_rate_limited_per_socket(
    app: Flask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only 50 signaling events from one socket are relayed in one second."""
    from app import signaling

    monkeypatch.setattr(signaling, "monotonic", lambda: 100.0)
    first = socket_client(app, "caller_one")
    peer = socket_client(app, "caller_two")
    first.emit("join", {"room": ROOM_CODE})
    peer.emit("join", {"room": ROOM_CODE})
    peer.get_received()

    for _ in range(MAX_SIGNALS_PER_SECOND + 1):
        first.emit("signal", {"type": "candidate", "payload": {"candidate": "x"}})

    received_signals = [
        event for event in peer.get_received() if event["name"] == "signal"
    ]
    assert len(received_signals) == MAX_SIGNALS_PER_SECOND
    first.disconnect()
    peer.disconnect()


def test_oversized_signal_is_not_relayed(app: Flask) -> None:
    """Signal envelopes of 20 KiB or more are rejected."""
    first = socket_client(app, "caller_one")
    peer = socket_client(app, "caller_two")
    first.emit("join", {"room": ROOM_CODE})
    peer.emit("join", {"room": ROOM_CODE})
    peer.get_received()

    first.emit("signal", {
        "type": "offer",
        "payload": {"sdp": "x" * (20 * 1024)},
    })

    assert event_payload(peer, "signal") == []
    first.disconnect()
    peer.disconnect()


def test_ice_config_has_stun_without_turn(app: Flask) -> None:
    """ICE config always includes Google STUN and excludes incomplete TURN settings."""
    app.config.update(
        SILLIE_TURN_URL=None,
        SILLIE_TURN_USERNAME=None,
        SILLIE_TURN_CREDENTIAL=None,
    )
    response = authenticated_client(app, "caller_one").get("/api/ice-config")

    assert response.status_code == 200
    assert response.json == {
        "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
    }


def test_call_page_renders_call_controls(app: Flask) -> None:
    """Authenticated room visits receive the video-call markup and room code."""
    response = authenticated_client(app, "caller_one").get(f"/call/{ROOM_CODE}")

    assert response.status_code == 200
    assert b"WAITING FOR PEER" in response.data
    assert b"id=\"remote-video\"" in response.data
    assert b"id=\"local-video\"" in response.data
    assert b"TEST CAPTION" in response.data


def test_ice_config_includes_turn_when_all_settings_exist(app: Flask) -> None:
    """A complete TURN configuration is included without leaking it elsewhere."""
    app.config.update(
        SILLIE_TURN_URL="turn:turn.example.test:3478",
        SILLIE_TURN_USERNAME="turn-user",
        SILLIE_TURN_CREDENTIAL="turn-secret",
    )
    response = authenticated_client(app, "caller_one").get("/api/ice-config")

    assert response.status_code == 200
    assert response.json == {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {
                "urls": "turn:turn.example.test:3478",
                "username": "turn-user",
                "credential": "turn-secret",
            },
        ]
    }
