"""Admin-only user directory, live status, and account-management commands."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import click
from flask import Blueprint, Flask, abort, jsonify, make_response, render_template
from flask_login import current_user

from .extensions import db
from .models import User
from .signaling import active_rooms, online_user_ids

admin_blueprint = Blueprint("admin", __name__)


def require_admin() -> None:
    """Return a plain not-found response for anonymous and non-admin users."""
    if not current_user.is_authenticated or not current_user.is_admin:
        abort(404)


@admin_blueprint.get("/admin")
def dashboard() -> str:
    """Render the protected administrator dashboard."""
    require_admin()
    users = db.session.execute(
        db.select(
            User.id,
            User.username,
            User.display_name,
            User.accessibility_profile,
            User.created_at,
            User.last_seen,
        ).order_by(User.id)
    ).all()
    return render_template(
        "admin.html",
        users=users,
        online_ids=online_user_ids(),
        active_room_list=active_rooms(),
    )


@admin_blueprint.get("/admin/api/status")
def status() -> Any:
    """Return admin dashboard counters and safe user presence data."""
    require_admin()
    users = db.session.execute(
        db.select(
            User.id,
            User.username,
            User.display_name,
            User.accessibility_profile,
            User.created_at,
            User.last_seen,
        ).order_by(User.id)
    ).all()
    online_ids = online_user_ids()
    payload = {
        "registered_users": len(users),
        "online_now": len(online_ids),
        "active_rooms": len(active_rooms()),
        "rooms": active_rooms(),
        "users": [
            {
                "id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "accessibility_profile": user.accessibility_profile,
                "created_at": isoformat(user.created_at),
                "last_seen": isoformat(user.last_seen),
                "online": user.id in online_ids,
            }
            for user in users
        ],
    }
    response = make_response(jsonify(payload))
    response.headers["Cache-Control"] = "no-store"
    return response


def isoformat(value: datetime | None) -> str | None:
    """Format a database timestamp for the dashboard's JSON response."""
    return value.isoformat() if value is not None else None


def register_admin_cli(app: Flask) -> None:
    """Register server-side-only admin grant and removal commands."""

    @app.cli.command("make-admin")
    @click.argument("username")
    def make_admin(username: str) -> None:
        """Grant administrator access to an existing USERNAME."""
        user = db.session.execute(
            db.select(User).where(User.username == username.strip().lower())
        ).scalar_one_or_none()
        if user is None:
            raise click.ClickException(f"User '{username}' was not found.")
        user.is_admin = True
        db.session.commit()
        click.echo(f"Administrator access granted to {user.username}.")

    @app.cli.command("remove-admin")
    @click.argument("username")
    def remove_admin(username: str) -> None:
        """Remove administrator access from USERNAME."""
        user = db.session.execute(
            db.select(User).where(User.username == username.strip().lower())
        ).scalar_one_or_none()
        if user is None:
            raise click.ClickException(f"User '{username}' was not found.")
        user.is_admin = False
        db.session.commit()
        click.echo(f"Administrator access removed from {user.username}.")
