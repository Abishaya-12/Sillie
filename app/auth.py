"""Account registration, login, and logout routes."""

from __future__ import annotations

import re
import time
import os
from urllib.parse import urlsplit

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db
from .models import User

auth_blueprint = Blueprint("auth", __name__)
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,20}$")
ACCESSIBILITY_PROFILES = {
    "deaf": "Visual-first prompts and sign-focused communication.",
    "blind": "Audio-first prompts and spoken guidance.",
    "mute": "Text and sign input without requiring speech.",
    "hearing": "Standard audio and visual communication.",
}
LOGIN_WINDOW_SECONDS = 5 * 60
LOGIN_BLOCK_SECONDS = 5 * 60
# In-memory rate-limit state intentionally resets when the server restarts.
LOGIN_FAILURES: dict[tuple[str, str], tuple[list[float], float]] = {}


@auth_blueprint.route("/register", methods=["GET", "POST"])
def register() -> Response | str:
    """Create an account after validating all submitted fields."""
    if current_user.is_authenticated:
        return redirect(url_for("app.lobby"))
    if request.method == "GET":
        return render_template(
            "register.html", accessibility_profiles=ACCESSIBILITY_PROFILES
        )

    username = request.form.get("username", "").strip().lower()
    display_name = request.form.get("display_name", "").strip()
    password = request.form.get("password", "")
    confirmation = request.form.get("confirm_password", "")
    profile = request.form.get("accessibility_profile", "")

    if not USERNAME_PATTERN.fullmatch(username):
        flash("Choose a username with 3-20 letters, numbers, or underscores.", "error")
    elif not display_name or len(display_name) > 50:
        flash("Enter a display name of 1-50 characters.", "error")
    elif len(password) < 8:
        flash("Your password must be at least 8 characters.", "error")
    elif password != confirmation:
        flash("Your passwords do not match.", "error")
    elif profile not in ACCESSIBILITY_PROFILES:
        flash("Choose one of the accessibility profiles.", "error")
    elif db.session.execute(
        db.select(User.id).where(User.username == username)
    ).scalar_one_or_none() is not None:
        flash("That username is already taken. Try another.", "error")
    else:
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            display_name=display_name,
            accessibility_profile=profile,
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("That username is already taken. Try another.", "error")
        else:
            login_user(user)
            flash("Account ready. Welcome to Sillie!", "success")
            return redirect(url_for("app.lobby"))

    return render_template(
        "register.html",
        accessibility_profiles=ACCESSIBILITY_PROFILES,
        form_data=request.form,
    )


@auth_blueprint.route("/login", methods=["GET", "POST"])
def login() -> Response | str:
    """Authenticate a user without disclosing which credential was incorrect."""
    if current_user.is_authenticated:
        return redirect(url_for("app.lobby"))
    if request.method == "GET":
        return render_template("login.html", next_url=request.args.get("next", ""))

    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")
    key = (username, request.remote_addr or "")
    now = time.monotonic()
    failures, blocked_until = LOGIN_FAILURES.get(key, ([], 0.0))
    if now < blocked_until:
        flash("Unable to sign in. Check your details and try again shortly.", "error")
        return render_template("login.html", next_url=request.form.get("next", ""))

    recent_failures = [
        failed_at
        for failed_at in failures
        if now - failed_at < LOGIN_WINDOW_SECONDS
    ]
    user = db.session.execute(
        db.select(User).where(User.username == username)
    ).scalar_one_or_none()
    if user is not None and check_password_hash(user.password_hash, password):
        LOGIN_FAILURES.pop(key, None)
        configured_admins = {
            name.strip().lower()
            for name in os.getenv("SILLIE_ADMIN_USERNAMES", "").split(",")
            if name.strip()
        }
        if username in configured_admins and not user.is_admin:
            user.is_admin = True
            db.session.commit()
        login_user(user)
        return redirect(safe_next_url(request.form.get("next", "")))

    recent_failures.append(now)
    blocked_until = (
        now + LOGIN_BLOCK_SECONDS
        if len(recent_failures) >= 5
        else 0.0
    )
    LOGIN_FAILURES[key] = (recent_failures, blocked_until)
    flash("Invalid username or password.", "error")
    return render_template("login.html", next_url=request.form.get("next", ""))


@auth_blueprint.post("/logout")
def logout() -> Response:
    """End the current authenticated session."""
    logout_user()
    flash("You have logged out.", "success")
    return redirect(url_for("auth.login"))


@auth_blueprint.route("/profile", methods=["GET", "POST"])
@login_required
def profile() -> Response | str:
    """Let the signed-in user update their accessibility profile."""
    if request.method == "POST":
        accessibility_profile = request.form.get("accessibility_profile", "")
        if accessibility_profile not in ACCESSIBILITY_PROFILES:
            flash("Choose one of the accessibility profiles.", "error")
        else:
            current_user.accessibility_profile = accessibility_profile
            db.session.commit()
            flash("Your accessibility profile has been updated.", "success")
            return redirect(url_for("auth.profile"))
    return render_template(
        "profile.html",
        accessibility_profiles=ACCESSIBILITY_PROFILES,
        selected_profile=current_user.accessibility_profile,
    )


def safe_next_url(target: str) -> str:
    """Return a same-site relative path or the default lobby path."""
    if (
        not target.startswith("/")
        or target.startswith("//")
        or "\\" in target
        or any(ord(character) < 32 for character in target)
    ):
        return url_for("app.lobby")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return url_for("app.lobby")
    return target
