"""Run the Sillie Flask application."""

from __future__ import annotations

from app import create_app
from app.signaling import socketio

app = create_app()

# Use exactly one worker because signaling rooms and participants live in memory.
# Production: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 100 --timeout 120
if __name__ == "__main__":
    socketio.run(
        app,
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
    )
