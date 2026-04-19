"""FastAPI auth dependencies backed by session_store."""

from __future__ import annotations

from typing import Callable, Dict

from fastapi import Header, HTTPException

from app.backend.session_store import session_store


# Role hierarchy (superior → inferior)
# Admin can do everything; auditor can do analyst + guest actions; analyst can do guest actions.
ROLE_ORDER = ["guest", "analyst", "auditor", "admin"]
_ROLE_RANK = {r: i for i, r in enumerate(ROLE_ORDER)}


def role_rank(role: str) -> int:
    return _ROLE_RANK.get((role or "").strip().lower(), -1)


def has_at_least(role: str, required_role: str) -> bool:
    """Return True if `role` is >= `required_role` in the hierarchy."""
    return role_rank(role) >= role_rank(required_role)


def get_auth_context(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_role: str | None = Header(default=None, alias="X-Role"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> Dict[str, str]:
    user_id = (x_user_id or "").strip().lower()
    role = (x_role or "").strip().lower()
    session_id = (x_session_id or "").strip()

    if not user_id or not role or not session_id:
        raise HTTPException(status_code=401, detail="Missing authentication headers.")

    session = session_store.validate(
        session_id=session_id,
        user_id=user_id,
        role=role,
    )
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    return {
        "user_id": session.user_id,
        "role": session.role,
        "session_id": session.session_id,
    }


def require_roles(*allowed_roles: str) -> Callable[..., Dict[str, str]]:
    allowed = {r.strip().lower() for r in allowed_roles}

    def _dep(
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
        x_role: str | None = Header(default=None, alias="X-Role"),
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> Dict[str, str]:
        auth = get_auth_context(
            x_user_id=x_user_id,
            x_role=x_role,
            x_session_id=x_session_id,
        )
        if auth["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient permissions.")
        return auth

    return _dep


def require_min_role(required_role: str) -> Callable[..., Dict[str, str]]:
    """Require a minimum role in the linear hierarchy.

    Example:
      - require_min_role('analyst') allows analyst/auditor/admin.
    """
    required = (required_role or "").strip().lower()

    def _dep(
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
        x_role: str | None = Header(default=None, alias="X-Role"),
        x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    ) -> Dict[str, str]:
        auth = get_auth_context(
            x_user_id=x_user_id,
            x_role=x_role,
            x_session_id=x_session_id,
        )
        if not has_at_least(auth["role"], required):
            raise HTTPException(status_code=403, detail="Insufficient permissions.")
        return auth

    return _dep
