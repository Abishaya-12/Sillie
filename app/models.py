"""Database models for Sillie accounts."""

from __future__ import annotations

from datetime import UTC, datetime

from flask_login import UserMixin
from sqlalchemy.orm import validates

from .extensions import db


class User(UserMixin, db.Model):
    """A registered Sillie user."""

    __table_args__ = (
        db.CheckConstraint(
            "accessibility_profile IN ('deaf', 'blind', 'mute', 'hearing')",
            name="valid_accessibility_profile",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    display_name = db.Column(db.String(50), nullable=False)
    accessibility_profile = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    @validates("username")
    def normalize_username(self, key: str, username: str) -> str:
        """Store all account usernames in their canonical lowercase form."""
        return username.lower()
