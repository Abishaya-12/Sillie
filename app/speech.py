"""Authenticated server-side integrations with Gnani speech REST APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from flask import Blueprint, Response, current_app, jsonify, request
from flask_login import login_required

speech_blueprint = Blueprint("speech", __name__)
MAX_TEXT_LENGTH = 2_000
MAX_AUDIO_SIZE = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD = 64 * 1024
REQUEST_TIMEOUT_SECONDS = 60
ALLOWED_AUDIO_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"})
STT_LANGUAGES = frozenset({
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN",
    "pa-IN", "ta-IN", "te-IN",
})
TTS_LANGUAGES = STT_LANGUAGES | {"auto", "hi-en"}


def _is_valid_stt_language(language: str) -> bool:
    """Check a Prisma v3.0 language or 1–4 distinct supported languages."""
    if language == "indic-auto":
        return True
    languages = language.split(",")
    return (
        1 <= len(languages) <= 4
        and len(set(languages)) == len(languages)
        and all(code in STT_LANGUAGES for code in languages)
    )


def _api_key() -> str | None:
    """Return the configured Gnani API key, if one is available."""
    return current_app.config.get("GNANI_API_KEY")


def _missing_key_response() -> tuple[Response, int]:
    """Build the response used when Gnani credentials are not configured."""
    return jsonify({"error": "Gnani speech is not configured. Set GNANI_API_KEY."}), 503


def _api_url(path: str) -> str:
    """Build a Gnani API URL from the configured base URL."""
    return f"{current_app.config['GNANI_BASE_URL'].rstrip('/')}/{path.lstrip('/')}"


def _api_headers(api_key: str) -> dict[str, str]:
    """Return the authentication headers required by Gnani REST endpoints."""
    return {"X-API-Key-ID": api_key}


@speech_blueprint.post("/api/speech/tts")
@login_required
def text_to_speech() -> Response | tuple[Response, int]:
    """Synthesize validated text and return the resulting WAV audio."""
    api_key = _api_key()
    if not api_key:
        return _missing_key_response()

    payload: Any = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = payload.get("text")
    language = payload.get("language")
    voice = payload.get("voice")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"text must contain 1 to {MAX_TEXT_LENGTH} characters."
        }), 400
    if not isinstance(language, str) or language not in TTS_LANGUAGES:
        return jsonify({"error": "Unsupported TTS language."}), 400
    if not isinstance(voice, str) or not voice.strip() or len(voice) > 64:
        return jsonify({"error": "voice must contain 1 to 64 characters."}), 400

    return _request_tts(api_key, text.strip(), language, voice.strip())


def _request_tts(
    api_key: str, text: str, language: str, voice: str
) -> Response | tuple[Response, int]:
    """Call Gnani TTS and return its binary audio response."""
    try:
        upstream = requests.post(
            _api_url("/api/v1/tts/inference"),
            headers=_api_headers(api_key),
            json={
                "text": text,
                "voice": voice,
                "model": "timbre-v3.0",
                "language": language,
                "audio_config": {"container": "wav"},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return jsonify({"error": "Gnani TTS is temporarily unavailable."}), 502

    if not upstream.ok or not upstream.content:
        return jsonify({"error": "Gnani TTS could not synthesize this text."}), 502
    content_type = upstream.headers.get("Content-Type", "audio/wav").split(";", 1)[0]
    if not content_type.startswith("audio/"):
        return jsonify({"error": "Gnani TTS returned an invalid audio response."}), 502
    return Response(upstream.content, content_type=content_type)


@speech_blueprint.post("/api/speech/stt")
@login_required
def speech_to_text() -> Response | tuple[Response, int]:
    """Transcribe a bounded audio upload and return its text transcript."""
    api_key = _api_key()
    if not api_key:
        return _missing_key_response()
    if (
        request.content_length is not None
        and request.content_length > MAX_AUDIO_SIZE + MAX_MULTIPART_OVERHEAD
    ):
        return jsonify({"error": "Audio upload exceeds the 10 MB limit."}), 413

    language = request.form.get("language", "")
    if not _is_valid_stt_language(language):
        return jsonify({"error": "Unsupported STT language."}), 400
    audio = request.files.get("audio")
    if audio is None:
        return jsonify({"error": "Audio upload is required in the 'audio' field."}), 400
    filename = audio.filename or ""
    if Path(filename).suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": "Audio must be WAV, MP3, OGG, FLAC, AAC, or M4A."}), 400
    audio_bytes = audio.stream.read(MAX_AUDIO_SIZE + 1)
    if not audio_bytes:
        return jsonify({"error": "Audio upload must not be empty."}), 400
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        return jsonify({"error": "Audio upload exceeds the 10 MB limit."}), 413

    return _request_stt(
        api_key,
        filename,
        audio.mimetype or "application/octet-stream",
        audio_bytes,
        language,
    )


def _request_stt(
    api_key: str, filename: str, mimetype: str, audio: bytes, language: str
) -> tuple[Response, int] | Response:
    """Call Gnani STT and normalize its transcript response."""
    try:
        upstream = requests.post(
            _api_url("/stt/v3"),
            headers=_api_headers(api_key),
            files={"audio_file": (filename, audio, mimetype)},
            data={"model": "prisma-v3.0", "language_code": language},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        payload: Any = upstream.json() if upstream.ok else None
    except (requests.RequestException, ValueError):
        return jsonify({"error": "Gnani STT is temporarily unavailable."}), 502

    transcript = payload.get("transcript") if isinstance(payload, dict) else None
    if not upstream.ok or not isinstance(transcript, str):
        return jsonify({"error": "Gnani STT could not transcribe this audio."}), 502
    return jsonify({"text": transcript})