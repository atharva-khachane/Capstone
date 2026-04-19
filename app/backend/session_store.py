"""
In-memory session store with TTL for API authentication.

This is intentionally simple for single-process deployments. Session records are
validated against request headers to prevent role/user spoofing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
import uuid
from typing import Dict, Optional


DEFAULT_SESSION_TTL_SECONDS = 8 * 60 * 60  # 8 hours


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    role: str
    created_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class SessionStore:
    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str, role: str) -> SessionRecord:
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            user_id=user_id.strip().lower(),
            role=role.strip().lower(),
            created_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
        )
        with self._lock:
            self._cleanup_locked(now)
            self._sessions[record.session_id] = record
        return record

    def validate(self, session_id: str, user_id: str, role: str) -> Optional[SessionRecord]:
        sid = (session_id or "").strip()
        uid = (user_id or "").strip().lower()
        r = (role or "").strip().lower()
        if not sid or not uid or not r:
            return None

        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup_locked(now)
            record = self._sessions.get(sid)
            if record is None or record.is_expired:
                return None
            if record.user_id != uid or record.role != r:
                return None
            return record

    def revoke_user_sessions(self, user_id: str) -> int:
        """Remove all active sessions for a given user id.

        Returns:
            Number of sessions removed.
        """
        uid = (user_id or "").strip().lower()
        if not uid:
            return 0

        with self._lock:
            to_remove = [sid for sid, rec in self._sessions.items() if rec.user_id == uid]
            for sid in to_remove:
                self._sessions.pop(sid, None)
            return len(to_remove)

    def _cleanup_locked(self, now: datetime) -> None:
        expired = [sid for sid, rec in self._sessions.items() if rec.expires_at <= now]
        for sid in expired:
            self._sessions.pop(sid, None)


session_store = SessionStore()
