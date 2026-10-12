"""In-memory, authenticated WebRTC signaling for one-to-one rooms."""

from __future__ import annotations

import json
import re
from collections import deque
from threading import RLock
from time import monotonic
from typing import Any

from flask import request
from flask_login import current_user
from flask_socketio import SocketIO, emit, join_room, leave_room

socketio = SocketIO(cors_allowed_origins=None, async_mode="threading")
ROOM_CODE_PATTERN = re.compile(r"^[A-HJ-NP-Z2-9]{6}$")
MAX_SIGNAL_BYTES = 20 * 1024
MAX_SIGNALS_PER_SECOND = 50
SIGNAL_RATE_WINDOW_SECONDS = 1.0

rooms: dict[str, dict[str, dict[str, Any]]] = {}
socket_rooms: dict[str, str] = {}
signal_times: dict[str, deque[float]] = {}
socket_users: dict[str, int] = {}
connected_user_sockets: dict[int, int] = {}
rooms_lock = RLock()
# Presence and active room data are process-local and assume exactly ONE worker.


@socketio.on("connect")
def handle_connect(auth: dict[str, Any] | None = None) -> bool:
    """Reject socket connections that do not carry an authenticated session."""
    if not current_user.is_authenticated:
        return False
    user_id = int(current_user.get_id())
    with rooms_lock:
        socket_users[request.sid] = user_id
        connected_user_sockets[user_id] = connected_user_sockets.get(user_id, 0) + 1
    return True


@socketio.on("join")
def handle_join(message: dict[str, Any] | None) -> None:
    """Join a valid room if it has capacity and notify both participants."""
    sid = request.sid
    room = message.get("room", "") if isinstance(message, dict) else ""
    if not isinstance(room, str) or not ROOM_CODE_PATTERN.fullmatch(room):
        emit("join_error", {"message": "That room code is not valid."})
        return

    with rooms_lock:
        remove_participant(sid)
        participants = rooms.setdefault(room, {})
        if len(participants) >= 2:
            if not participants:
                rooms.pop(room, None)
            emit("room_full")
            return

        profile = {
            "user_id": int(current_user.get_id()),
            "display_name": current_user.display_name,
            "accessibility_profile": current_user.accessibility_profile,
        }
        existing = next(iter(participants.values()), None)
        participants[sid] = profile
        socket_rooms[sid] = room
        signal_times[sid] = deque()
        join_room(room, sid=sid)
        peer_sid = next(
            (member_sid for member_sid in participants if member_sid != sid),
            None,
        )

    if existing is not None:
        emit("peer_present", public_profile(existing))
        if peer_sid is not None:
            socketio.emit("peer_joined", public_profile(profile), to=peer_sid)


@socketio.on("signal")
def handle_signal(message: dict[str, Any] | None) -> None:
    """Relay a bounded offer, answer, or candidate to only the room peer."""
    sid = request.sid
    now = monotonic()
    with rooms_lock:
        events = signal_times.get(sid)
        if events is None:
            return
        while events and now - events[0] >= SIGNAL_RATE_WINDOW_SECONDS:
            events.popleft()
        if len(events) >= MAX_SIGNALS_PER_SECOND:
            return
        events.append(now)

        room = socket_rooms.get(sid)
        participants = rooms.get(room or "", {})
        peer_sid = next(
            (member_sid for member_sid in participants if member_sid != sid),
            None,
        )
    if peer_sid is None or not isinstance(message, dict):
        return

    signal_type = message.get("type")
    payload = message.get("payload")
    if not isinstance(signal_type, str) or signal_type not in {
        "offer",
        "answer",
        "candidate",
    }:
        return
    try:
        encoded = json.dumps(
            {"type": signal_type, "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return
    if len(encoded) >= MAX_SIGNAL_BYTES:
        return

    socketio.emit(
        "signal",
        {"type": signal_type, "payload": payload},
        to=peer_sid,
    )


@socketio.on("leave")
def handle_leave() -> None:
    """Remove the requesting participant and notify the remaining peer."""
    with rooms_lock:
        remove_participant(request.sid)


@socketio.on("disconnect")
def handle_disconnect(reason: str | None = None) -> None:
    """Remove disconnected participants and clear their rate-limit state."""
    with rooms_lock:
        remove_participant(request.sid)
        signal_times.pop(request.sid, None)
        user_id = socket_users.pop(request.sid, None)
        if user_id is not None:
            remaining = connected_user_sockets.get(user_id, 0) - 1
            if remaining > 0:
                connected_user_sockets[user_id] = remaining
            else:
                connected_user_sockets.pop(user_id, None)


def remove_participant(sid: str) -> None:
    """Remove one socket from its room and notify its peer if present."""
    room = socket_rooms.pop(sid, None)
    if room is None:
        return
    participants = rooms.get(room)
    if participants is None:
        return
    participants.pop(sid, None)
    leave_room(room, sid=sid)
    peer_sid = next(iter(participants), None)
    if peer_sid is not None:
        socketio.emit("peer_left", to=peer_sid)
    else:
        rooms.pop(room, None)
    signal_times.pop(sid, None)


def public_profile(participant: dict[str, Any]) -> dict[str, str]:
    """Return only the public peer identity fields required by the UI."""
    return {
        "display_name": str(participant["display_name"]),
        "profile": str(participant["accessibility_profile"]),
    }


def online_user_ids() -> set[int]:
    """Return user IDs with one or more connected Socket.IO sessions."""
    with rooms_lock:
        return set(connected_user_sockets)


def active_rooms() -> list[dict[str, int | str]]:
    """Return active room codes and participant counts without user identities."""
    with rooms_lock:
        return [
            {"room": room, "participant_count": len(participants)}
            for room, participants in sorted(rooms.items())
            if participants
        ]
