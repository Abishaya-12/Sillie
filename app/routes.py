"""Route definitions for the Sillie web app."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template

blueprint = Blueprint("app", __name__)


@blueprint.route("/")
def index() -> str:
    """Serve the main HTML page."""
    return render_template("index.html")


@blueprint.route("/api/health")
def health() -> tuple[str, int]:
    """Return a lightweight health-check payload."""
    return jsonify({"status": "ok", "service": "sillie"}), 200
