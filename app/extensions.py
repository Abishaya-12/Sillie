"""Shared Flask extensions."""

from typing import TYPE_CHECKING
from __future__ import annotations
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

if TYPE_CHECKING:
    from .models import User

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    """Load a user from the session identifier."""
    from .models import User

    if not user_id.isdecimal():
        return None
    return db.session.get(User, int(user_id))
