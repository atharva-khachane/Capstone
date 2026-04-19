"""
Credential store for the SL-RAG demo.

Uses stdlib hashlib (PBKDF2-HMAC-SHA256) and persists runtime-created users in
storage/auth/users.json so admin-created accounts survive restarts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import threading
from typing import Dict, List, Tuple


# Role hierarchy (superior → inferior)
ROLE_ORDER = ["guest", "analyst", "auditor", "admin"]
_ROLE_RANK = {r: i for i, r in enumerate(ROLE_ORDER)}


def _normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def _role_rank(role: str) -> int:
    return _ROLE_RANK.get(_normalize_role(role), -1)


def _pick_highest_role(roles: List[str]) -> str:
    best = "guest"
    best_rank = _role_rank(best)
    for r in roles:
        rr = _role_rank(r)
        if rr > best_rank:
            best = _normalize_role(r)
            best_rank = rr
    return best

# Salt is fixed per deployment; fine for a demo offline system.
_SALT = b"sl_rag_isro_2025"
_AUTH_DIR = Path("storage") / "auth"
_USERS_FILE = _AUTH_DIR / "users.json"
_LOCK = threading.Lock()


def _hash(password: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), _SALT, iterations=100_000
    ).hex()


# fmt: off
_RAW: dict = {
    #  user_id      plain password    primary role
    "admin":   {"password": "admin123",   "role": "admin"},
    "auditor": {"password": "auditor123", "role": "auditor"},
    "analyst": {"password": "analyst123", "role": "analyst"},
    "guest":   {"password": "guest",      "role": "guest"},
}
# fmt: on

# Pre-hash default passwords once at import time
_DEFAULT_USERS: Dict[str, Dict[str, str]] = {
    uid: {"hash": _hash(info["password"]), "role": _normalize_role(info["role"])}
    for uid, info in _RAW.items()
}
USERS: Dict[str, Dict[str, str]] = dict(_DEFAULT_USERS)


def _ensure_storage() -> None:
    _AUTH_DIR.mkdir(parents=True, exist_ok=True)


def _load_users_from_disk() -> None:
    if not _USERS_FILE.exists():
        return

    try:
        payload = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(payload, dict):
        return

    for uid, info in payload.items():
        if not isinstance(uid, str) or not isinstance(info, dict):
            continue
        hash_val = info.get("hash")
        role = info.get("role")
        roles = info.get("roles")
        if not isinstance(hash_val, str):
            continue

        resolved_role: str | None = None
        if isinstance(role, str) and _role_rank(role) >= 0:
            resolved_role = _normalize_role(role)
        elif isinstance(roles, list):
            normalized_roles = [str(r).strip().lower() for r in roles if str(r).strip()]
            normalized_roles = [r for r in normalized_roles if _role_rank(r) >= 0]
            if normalized_roles:
                resolved_role = _pick_highest_role(normalized_roles)

        if not resolved_role:
            continue

        USERS[uid.strip().lower()] = {
            "hash": hash_val,
            "role": resolved_role,
        }


def _save_users_to_disk() -> None:
    _ensure_storage()
    _USERS_FILE.write_text(
        json.dumps(USERS, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def list_users() -> List[Dict[str, object]]:
    with _LOCK:
        return [
            {"user_id": uid, "role": info["role"]}
            for uid, info in sorted(USERS.items())
        ]


def add_user(user_id: str, password: str, role: str) -> Tuple[bool, str]:
    uid = (user_id or "").strip().lower()
    pwd = (password or "").strip()
    r = _normalize_role(role)

    if not uid:
        return False, "User id cannot be empty."
    if len(uid) < 3:
        return False, "User id must be at least 3 characters."
    if len(pwd) < 6:
        return False, "Password must be at least 6 characters."
    if _role_rank(r) < 0:
        return False, f"Invalid role '{r}'."

    with _LOCK:
        if uid in USERS:
            return False, f"User '{uid}' already exists."

        USERS[uid] = {
            "hash": _hash(pwd),
            "role": r,
        }
        _save_users_to_disk()
    return True, ""


class DeleteUserError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def delete_user(user_id: str) -> None:
    """Delete a user account.

    Notes:
        - Default demo accounts are protected from deletion.
        - Removes the user from the in-memory store and persists to disk.

    Raises:
        DeleteUserError: when user does not exist or is protected.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        raise DeleteUserError("User id cannot be empty.")

    if uid == "admin":
        raise DeleteUserError("Cannot delete the default admin user 'admin'.")

    with _LOCK:
        if uid not in USERS:
            raise DeleteUserError(f"User '{uid}' not found.", status_code=404)

        USERS.pop(uid, None)
        _save_users_to_disk()


def get_assigned_role(user_id: str) -> str | None:
    uid = (user_id or "").strip().lower()
    if not uid:
        return None
    user = USERS.get(uid)
    if not user:
        return None
    r = _normalize_role(user.get("role", "guest"))
    return r if _role_rank(r) >= 0 else "guest"


def verify_password(user_id: str, password: str) -> tuple[bool, str]:
    """Verify only the user's password (role is derived from account)."""
    uid = (user_id or "").strip().lower()
    user = USERS.get(uid)
    if user is None:
        return False, f"Unknown user '{user_id}'."

    if not hmac.compare_digest(_hash(password), user["hash"]):
        return False, "Incorrect password."

    return True, ""


def verify(user_id: str, password: str, role: str) -> tuple:
    """Verify credentials and (optional) role selection.

    Notes:
        Users have a single primary role. Role selection is only accepted when
        it matches the assigned role. (Front-end should not allow choosing a
        different role.)

    Returns:
        (ok: bool, error_message: str) — ok=True when credentials are valid.
    """
    user = USERS.get(user_id.lower().strip())
    if user is None:
        return False, f"Unknown user '{user_id}'."

    if not hmac.compare_digest(_hash(password), user["hash"]):
        return False, "Incorrect password."

    requested = _normalize_role(role)
    assigned = _normalize_role(user.get("role", "guest"))
    if _role_rank(requested) < 0:
        return False, f"Invalid role '{requested}'."
    if _role_rank(assigned) < 0:
        assigned = "guest"

    if requested and requested != assigned:
        return False, f"Role '{requested}' is not permitted for '{user_id}'. Assigned: {assigned}."

    return True, ""


_load_users_from_disk()
