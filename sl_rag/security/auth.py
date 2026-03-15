"""
Lightweight auth context and RBAC policy helpers.

This module is intentionally simple and offline-friendly:
- Normalizes caller-provided auth context.
- Enforces role-based query/document access decisions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AuthContext:
    """Identity context associated with a query call."""

    user_id: str = "anonymous"
    role: str = "guest"
    session_id: str = ""


def resolve_auth_context(auth_context: Optional[Dict[str, Any]]) -> AuthContext:
    """Create a safe auth context from a dict-like payload."""
    if not auth_context:
        return AuthContext()

    return AuthContext(
        user_id=str(auth_context.get("user_id", "anonymous")),
        role=str(auth_context.get("role", "guest")).lower(),
        session_id=str(auth_context.get("session_id", "")),
    )


class RBACManager:
    """
    Role-based permissions for SL-RAG query and document access.

    Supported roles:
    - admin: full access
    - analyst: can query and access up to confidential
    - auditor: can query and access up to restricted
    - guest: can query but only public/internal docs
    """

    def __init__(self):
        self.allowed_levels = {
            "admin": {"public", "internal", "confidential", "restricted"},
            "analyst": {"public", "internal", "confidential"},
            "auditor": {"public", "internal", "confidential", "restricted"},
            "guest": {"public", "internal"},
        }
        self.query_roles = {"admin", "analyst", "auditor", "guest"}

    def can_query(self, role: str) -> bool:
        """Return True if a role is allowed to execute queries."""
        return role in self.query_roles

    def can_access_document(self, role: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Return True if role can access a document based on metadata sensitivity."""
        if role not in self.allowed_levels:
            return False
        metadata = metadata or {}
        level = str(metadata.get("sensitivity", "public")).lower()
        return level in self.allowed_levels[role]

    def filter_accessible_results(self, role: str, results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Filter retrieval results to only those accessible for a role."""
        filtered = []
        for chunk, score in results:
            if self.can_access_document(role, getattr(chunk, "metadata", {})):
                filtered.append((chunk, score))
        return filtered
