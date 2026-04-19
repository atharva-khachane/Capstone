"""
Auth route — role-based login with password verification.
Passwords are checked against PBKDF2 hashes in credentials.py.
Returns a session token (UUID) the frontend stores in localStorage.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.backend.pipeline_instance import pipe
from app.backend.auth_dependencies import require_min_role
from app.backend.credentials import (
    add_user,
    delete_user,
    DeleteUserError,
    get_assigned_role,
    list_users,
    verify_password,
    verify as verify_credentials,
)
from app.backend.session_store import session_store

router = APIRouter()

VALID_ROLES = {"admin", "analyst", "auditor", "guest"}
ROLE_RANK = {"guest": 0, "analyst": 1, "auditor": 2, "admin": 3}


class LoginRequest(BaseModel):
    user_id: str
    password: str
    # Back-compat: older clients sent a role. If provided, it must match the
    # account's assigned role.
    role: str | None = None


class LoginResponse(BaseModel):
    user_id: str
    role: str
    session_id: str
    can_query: bool
    pipeline_ready: bool


class CreateUserRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=256)
    # Prefer a single primary role going forward.
    role: str | None = None
    # Back-compat for older clients.
    roles: list[str] | None = None


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    assigned_role = get_assigned_role(req.user_id)
    if not assigned_role or assigned_role not in VALID_ROLES:
        raise HTTPException(status_code=401, detail=f"Unknown user '{req.user_id}'.")

    # If caller provided a role (legacy client), require it to match.
    if isinstance(req.role, str) and req.role.strip():
        provided = req.role.lower().strip()
        ok, error_msg = verify_credentials(req.user_id, req.password, provided)
        if not ok:
            raise HTTPException(status_code=401, detail=error_msg)
    else:
        ok, error_msg = verify_password(req.user_id, req.password)
        if not ok:
            raise HTTPException(status_code=401, detail=error_msg)

    role = assigned_role

    session = session_store.create_session(
        user_id=req.user_id,
        role=role,
    )
    can_query = pipe.rbac.can_query(role) if hasattr(pipe, "rbac") else True

    return LoginResponse(
        user_id=req.user_id.lower().strip(),
        role=role,
        session_id=session.session_id,
        can_query=can_query,
        pipeline_ready=pipe._ready,
    )


@router.get("/roles")
def list_roles():
    """Return available roles and their access levels."""
    return {
        "roles": [
            {"role": "admin",   "access": "Full access (includes all lower roles)"},
            {"role": "auditor", "access": "Audit + restricted docs (includes analyst + guest)"},
            {"role": "analyst", "access": "Query + confidential docs (includes guest)"},
            {"role": "guest",   "access": "Query + public/internal docs"},
        ]
    }


@router.get("/users")
def get_users(_auth: dict = Depends(require_min_role("admin"))):
    """List known users (without password hashes). Admin only."""
    return {"users": list_users()}


@router.post("/users")
def create_user(
    req: CreateUserRequest,
    _auth: dict = Depends(require_min_role("admin")),
):
    """Create a new user account. Admin only."""
    requested_role: str | None = None
    if isinstance(req.role, str) and req.role.strip():
        requested_role = req.role.strip().lower()
    elif isinstance(req.roles, list) and req.roles:
        # If a list is provided (legacy client), pick the most privileged role in it.
        normalized = sorted({(r or "").strip().lower() for r in req.roles if (r or "").strip()})
        normalized = [r for r in normalized if r in VALID_ROLES]
        if normalized:
            requested_role = max(normalized, key=lambda r: ROLE_RANK.get(r, -1))

    if not requested_role:
        raise HTTPException(status_code=400, detail="A role is required.")
    if requested_role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{requested_role}'. Must be one of: {sorted(VALID_ROLES)}",
        )

    ok, err = add_user(user_id=req.user_id, password=req.password, role=requested_role)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    return {
        "status": "created",
        "user_id": req.user_id.strip().lower(),
        "role": requested_role,
    }


@router.delete("/users/{user_id}")
def remove_user(
    user_id: str,
    _auth: dict = Depends(require_min_role("admin")),
):
    """Delete a user account. Admin only."""
    target_uid = (user_id or "").strip().lower()
    caller_uid = (_auth.get("user_id") or "").strip().lower()
    if target_uid == "admin" and caller_uid == "admin":
        raise HTTPException(status_code=400, detail="Default admin cannot delete their own profile.")

    try:
        delete_user(user_id)
    except DeleteUserError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    session_store.revoke_user_sessions(user_id)
    return {"status": "deleted", "user_id": (user_id or "").strip().lower()}
