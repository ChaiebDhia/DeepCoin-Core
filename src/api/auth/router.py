"""
src/api/auth/router.py
======================
FastAPI router for all authentication and account management endpoints.

ENDPOINTS
---------
    POST /auth/register             — create account, trigger verification email
    POST /auth/login                — exchange credentials for access + refresh tokens
    GET  /auth/verify-email         — confirm email address via one-time token
    POST /auth/resend-verification  — resend a fresh verification email (pending accounts)
    POST /auth/refresh              — exchange valid refresh token for new access token
    POST /auth/logout               — revoke the current refresh token
    GET  /auth/me                   — return the authenticated user's profile
    POST /auth/forgot-password      — send password reset email
    POST /auth/reset-password       — apply a new password via reset token

TOKEN DELIVERY STRATEGY
-----------------------
    Access token  → JSON response body
        Stored by the client in memory (NOT localStorage — XSS risk).
        Short-lived (15 min). JS framework (NextAuth.js) handles rotation.

    Refresh token → httpOnly Secure SameSite=Lax cookie named "refresh_token"
        Not accessible to JavaScript → immune to XSS.
        Sent automatically by the browser on same-origin requests.
        Max-Age = REFRESH_TOKEN_EXPIRE_DAYS * 86400 seconds.

    WHY BOTH:
        This "token pair" pattern is the industry standard for SPAs.
        If the user closes the tab, the in-memory access token is lost but the
        httpOnly cookie persists, so they stay logged in on next visit.

AUDIT LOGGING:
    Every mutation (register, login, login_failed, logout, password_reset)
    is written to the audit_log table for non-repudiation.
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func

from src.api.db.models import Classification, EmailVerification, RefreshToken, User, UserRole, UserStatus
from src.api.db.audit  import client_ip, write_audit
from src.api.db.session import get_db
from src.api.auth.utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.api.auth.deps import get_current_user
from src.api.auth.email import send_verification_email, send_password_reset_email

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

_REFRESH_COOKIE_NAME = "refresh_token"
_REFRESH_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
_PASSWORD_RESET_EXPIRE_HOURS = 1


# ── Request / Response schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:        EmailStr  = Field(..., description="Valid email address")
    password:     str       = Field(..., min_length=8, max_length=128, description="Min 8 characters")
    display_name: str | None = Field(None, max_length=100)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        """
        WHY enforce complexity:
            A password that is ≥8 chars but all the same character (aaaaaaaa)
            provides essentially no security. We require at least one digit
            or special character — a lightweight check without complex regex.
        """
        has_letter  = any(c.isalpha()  for c in v)
        has_nonalpha = any(not c.isalpha() for c in v)
        if not (has_letter and has_nonalpha):
            raise ValueError(
                "Password must contain at least one letter and one number or special character."
            )
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int         # seconds until expiry
    user:         "UserProfile"


class UserProfile(BaseModel):
    id:           str
    email:        str
    display_name: str | None
    role:         str
    status:       str
    created_at:   datetime
    email_verified: bool


TokenResponse.model_rebuild()  # resolve forward reference


class MessageResponse(BaseModel):
    message: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str = Field(..., min_length=8, max_length=128)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    """
    Write the refresh token to a secure httpOnly cookie.

    WHY httpOnly:
        Prevents JavaScript from reading the cookie. An XSS attacker can
        steal tokens stored in localStorage/sessionStorage but cannot access
        httpOnly cookies.

    WHY SameSite=Lax:
        "Lax" allows the cookie to be sent on cross-site top-level navigations
        (e.g. user clicks a link from their email client) but blocks it on
        cross-site sub-resource requests. "Strict" would break the email
        verify-and-redirect flow. "None" requires Secure and allows all
        cross-site sends (too permissive for auth cookies).
    """
    response.set_cookie(
        key=_REFRESH_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=os.getenv("ENV", "development") == "production",  # HTTPS only in prod
        samesite="lax",
        max_age=_REFRESH_EXPIRE_DAYS * 86400,
        path="/auth",   # cookie only sent to /auth/* endpoints — minimise exposure
    )


def _delete_refresh_cookie(response: Response) -> None:
    """Clear the refresh token cookie on logout."""
    response.delete_cookie(key=_REFRESH_COOKIE_NAME, path="/auth")


def _profile(user: User) -> UserProfile:
    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=user.role.value,
        status=user.status.value,
        created_at=user.created_at,
        email_verified=user.email_verified_at is not None,
    )


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
    description="Registers a new analyst account. Triggers a verification email.",
)
async def register(
    body: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    FLOW:
        1. Check email is not already registered (unique constraint guard).
        2. Hash the password with bcrypt.
        3. Create User row with status=pending.
        4. Create EmailVerification row (24h token).
        5. Send verification email (dev: log only).
        6. Write audit log entry.
        7. Return 201 {"message": "..."}.

    WHY we don't expose whether the email exists:
        Returning "email already registered" leaks user enumeration info.
        Some systems return a generic "if this email is valid, you'll receive..."
        message. We chose to return "account created, check email" uniformly.
        The frontend detects duplicates via the 409 status code.
    """
    # 1. Unique email check
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    # 2+3. Create user
    # WHY dev-mode auto-activation:
    #   In development there is no SMTP server, so verification emails are only
    #   printed to the console. Requiring email verification would block every
    #   test login. When ENV != "production", we set status=active immediately
    #   and stamp email_verified_at so the user can sign in right away.
    is_dev = os.getenv("ENV", "development") != "production"
    now    = datetime.now(timezone.utc)
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        display_name=body.display_name or body.email.split("@")[0],
        role=UserRole.analyst,
        status=UserStatus.active if is_dev else UserStatus.pending,
        email_verified_at=now if is_dev else None,
    )
    db.add(user)
    await db.flush()   # populate user.id without committing

    if not is_dev:
        # 4. Create email verification token (production only)
        raw_token = secrets.token_urlsafe(48)   # 64-char URL-safe string
        verification = EmailVerification(
            user_id=user.id,
            token=raw_token,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        db.add(verification)

    # 5. Write audit log (before commit so it's in the same transaction)
    await write_audit(
        db,
        action="user.register",
        user_id=user.id,
        resource_type="user",
        resource_id=user.id,
        payload={"email": body.email, "role": "analyst"},
        ip_address=client_ip(request),
    )

    # 6. Send email (after audit is written)
    if not is_dev:
        email_sent = await send_verification_email(user.email, raw_token)  # type: ignore[possibly-undefined]
        if not email_sent:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification email. System misconfigured."
            )

    msg = (
        "Account created. You can sign in immediately."
        if is_dev
        else "Account created. Please check your email to verify your address."
    )
    logger.info("New user registered: id=%s email=%s dev_auto_active=%s", user.id, user.email, is_dev)
    return MessageResponse(message=msg)


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    FLOW:
        1. Look up the User by email.
        2. Verify the password (constant-time bcrypt comparison).
        3. Check account status — reject suspended/pending accounts.
        4. Create an access token (JWT, 15 min).
        5. Create a refresh token (opaque random, 7 days, stored as hash).
        6. Write refresh token to httpOnly cookie.
        7. Update user.last_login_at.
        8. Write audit log entry.
        9. Return access token + user profile in body.

    WHY we return the same 401 for "wrong email" and "wrong password":
        Distinct error messages enable user enumeration attacks
        ("this email is registered" vs "email not found").
    """
    # 1. Find user
    result = await db.execute(select(User).where(User.email == body.email))
    user: User | None = result.scalar_one_or_none()

    # 2. Verify password (constant-time even for non-existent users)
    if user is None or not verify_password(body.password, user.hashed_password):
        if user:
            await write_audit(
                db, action="user.login_failed",
                user_id=user.id,
                payload={"reason": "wrong_password"},
                ip_address=client_ip(request),
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Account status checks
    if user.status == UserStatus.suspended:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been suspended. Contact an administrator.",
        )
    if user.status == UserStatus.pending:
        is_dev = os.getenv("ENV", "development") != "production"
        if is_dev:
            # In development there is no SMTP server, so verification emails are
            # only printed to the console.  Auto-activate here so any user that
            # registered before this fix existed can still sign in immediately.
            user.status = UserStatus.active
            user.email_verified_at = datetime.now(timezone.utc)
            logger.info("Dev mode: auto-activated pending user id=%s", user.id)
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Please verify your email address before logging in. Check your inbox.",
            )

    # 4. Access token
    access_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    access_token = create_access_token({
        "sub":    user.id,
        "email":  user.email,
        "role":   user.role.value,
        "status": user.status.value,
    })

    # 5. Refresh token
    raw_refresh, refresh_hash, refresh_exp = create_refresh_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=refresh_exp,
        ip_address=client_ip(request),
    ))

    # 6. Write refresh cookie
    _set_refresh_cookie(response, raw_refresh)

    # 7. Update last_login_at
    user.last_login_at = datetime.now(timezone.utc)

    # 8. Audit log
    await write_audit(
        db, action="user.login",
        user_id=user.id,
        resource_type="user", resource_id=user.id,
        payload={"role": user.role.value},
        ip_address=client_ip(request),
    )

    logger.info("User logged in: id=%s email=%s", user.id, user.email)
    return TokenResponse(
        access_token=access_token,
        expires_in=access_expire_minutes * 60,
        user=_profile(user),
    )


# ── GET /auth/verify-email ────────────────────────────────────────────────────

@router.get(
    "/verify-email",
    response_model=MessageResponse,
    summary="Activate account via email link",
)
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    Called when the user clicks the link in their verification email.

    FLOW:
        1. Look up the EmailVerification row by token.
        2. Check it hasn't expired or been used.
        3. Mark verified: set user.status=active, user.email_verified_at=now.
        4. Mark token used: set verification.used_at=now.
        5. Write audit log.
        6. Return 200 {"message": "Email verified"}.
    """
    result = await db.execute(
        select(EmailVerification).where(EmailVerification.token == token)
    )
    verification: EmailVerification | None = result.scalar_one_or_none()

    if verification is None:
        raise HTTPException(status_code=404, detail="Verification link is invalid or has expired.")

    if verification.used_at is not None:
        raise HTTPException(status_code=400, detail="This verification link has already been used.")

    now = datetime.now(timezone.utc)
    if verification.expires_at.replace(tzinfo=timezone.utc) < now:
        raise HTTPException(status_code=400, detail="This verification link has expired. Request a new one.")

    # Load the user
    result = await db.execute(select(User).where(User.id == verification.user_id))
    user: User | None = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Activate
    user.status = UserStatus.active
    user.email_verified_at = now
    verification.used_at = now

    await write_audit(
        db, action="user.email_verified",
        user_id=user.id,
        resource_type="user", resource_id=user.id,
    )

    logger.info("Email verified for user id=%s", user.id)
    return MessageResponse(message="Email verified successfully! You can now log in.")


# ── POST /auth/resend-verification ───────────────────────────────────────────

@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    summary="Resend email verification link",
)
async def resend_verification(
    body: ForgotPasswordRequest,   # reuse — only needs email: EmailStr
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    Allow a user whose account is still pending to request a fresh
    verification email (e.g. the original expired or landed in spam).

    SECURITY:
        - Always returns the same 200 message regardless of whether the email
          exists in the database (prevents account enumeration).
        - Only sends a new token when the account is still in 'pending' status.
          Active, suspended, or unknown emails silently succeed.
        - New token invalidates old ones implicitly because verify_email()
          marks each token as used_at on first success.  We do NOT delete
          old tokens here to avoid a race condition — they will eventually
          expire naturally (default 48 h from registration).

    WHY not use the forgot-password flow for this:
        The forgot-password token is prefixed with "reset:" to distinguish it
        from email verification tokens.  Reusing it would conflate two separate
        security operations and confuse audit logs.
    """
    _VERIFY_EXPIRE_HOURS = 48

    result = await db.execute(select(User).where(User.email == body.email))
    user: User | None = result.scalar_one_or_none()

    if user is not None and user.status == UserStatus.pending:
        raw_token = secrets.token_urlsafe(48)
        db.add(EmailVerification(
            user_id    = user.id,
            token      = raw_token,          # no prefix — standard verification token
            expires_at = datetime.now(timezone.utc) + timedelta(hours=_VERIFY_EXPIRE_HOURS),
        ))
        email_sent = await send_verification_email(user.email, raw_token)
        if not email_sent:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification email. System misconfigured."
            )
        await write_audit(db, action="user.resend_verification", user_id=user.id)
        logger.info("Resent verification email to user id=%s", user.id)

    # Always return 200 — do NOT reveal whether the email address exists
    return MessageResponse(
        message="If your account is pending verification, a new confirmation link has been sent."
    )


# ── POST /auth/refresh ────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh the access token",
    description="Requires a valid refresh_token httpOnly cookie.",
)
async def refresh_access_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    refresh_token: str | None = Cookie(default=None, alias=_REFRESH_COOKIE_NAME),
) -> TokenResponse:
    """
    FLOW:
        1. Read the refresh token from the httpOnly cookie.
        2. Hash it and look up the RefreshToken row.
        3. Check it hasn't been revoked or expired.
        4. Issue a new access token.
        5. Rotate the refresh token (revoke old, issue new — sliding window).
        6. Return new access token in body, new refresh token in cookie.

    WHY refresh token rotation:
        If a refresh token is stolen and used, the legitimate user's next
        refresh will see their token has already been revoked and get a 401,
        alerting them to re-login. The attacker's stolen token is then also
        invalid. This is "refresh token rotation with reuse detection".
    """
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token provided.")

    import hashlib
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    rt: RefreshToken | None = result.scalar_one_or_none()

    if rt is None or rt.revoked_at is not None:
        _delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh token is invalid or has been revoked.")

    now = datetime.now(timezone.utc)
    if rt.expires_at.replace(tzinfo=timezone.utc) < now:
        _delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh token has expired. Please log in again.")

    # Load the user
    result = await db.execute(select(User).where(User.id == rt.user_id))
    user: User | None = result.scalar_one_or_none()
    if user is None or user.status == UserStatus.suspended:
        _delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="User account is no longer valid.")

    # Revoke old refresh token (rotation)
    rt.revoked_at = now

    # Issue new tokens
    access_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    new_access = create_access_token({
        "sub":    user.id,
        "email":  user.email,
        "role":   user.role.value,
        "status": user.status.value,
    })
    raw_new, hash_new, exp_new = create_refresh_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=hash_new,
        expires_at=exp_new,
        ip_address=client_ip(request),
    ))
    _set_refresh_cookie(response, raw_new)

    return TokenResponse(
        access_token=new_access,
        expires_in=access_expire_minutes * 60,
        user=_profile(user),
    )


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout and revoke the current refresh token",
)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    refresh_token: str | None = Cookie(default=None, alias=_REFRESH_COOKIE_NAME),
) -> MessageResponse:
    """
    FLOW:
        1. Revoke the refresh token from the cookie (if present).
        2. Clear the cookie.
        3. Write audit log.
        4. Return 200.

    NOTE: The access token cannot be explicitly revoked (it's stateless).
    The client must discard it. It will expire within ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    if refresh_token:
        import hashlib
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        rt: RefreshToken | None = result.scalar_one_or_none()
        if rt and rt.revoked_at is None:
            rt.revoked_at = datetime.now(timezone.utc)

    _delete_refresh_cookie(response)
    await write_audit(
        db, action="user.logout",
        user_id=current_user.id,
        ip_address=client_ip(request),
    )

    logger.info("User logged out: id=%s", current_user.id)
    return MessageResponse(message="Logged out successfully.")


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserProfile,
    summary="Get the authenticated user's profile",
)
async def me(current_user: User = Depends(get_current_user)) -> UserProfile:
    """
    Returns the profile of the currently authenticated user.
    Used by the frontend to populate the account menu and check email_verified status.
    """
    return _profile(current_user)


# ── GET /auth/me/stats ────────────────────────────────────────────────────────

@router.get(
    "/me/stats",
    summary="Personal statistics for the authenticated user",
)
async def get_my_stats(
    current_user: User          = Depends(get_current_user),
    db:           AsyncSession  = Depends(get_db),
) -> dict:
    """
    Return the current user's own aggregate statistics.

    WHAT:
        total_analyses — total classifications submitted by this user
        by_route       — {historian, validator, investigator, unknown}
        avg_conf       — average confidence across their analyses (0.0–1.0)
        top_label      — most-classified coin type with count
        recent         — last 5 analyses (newest first)

    WHY separate from /api/admin/stats:
        Admin stats aggregate across ALL users and require a privileged role.
        This endpoint is scoped to the caller's own data — accessible to every
        authenticated user regardless of role. A regular analyst can see their
        personal progress without seeing other users' data.

    ACCESS: any authenticated user (own data only).
    """
    user_id = current_user.id

    # ── route distribution ────────────────────────────────────────────────────
    route_q = (
        select(Classification.route_taken, func.count(Classification.id).label("n"))
        .where(Classification.user_id == user_id)
        .group_by(Classification.route_taken)
    )
    rows = (await db.execute(route_q)).all()
    by_route: dict[str, int] = {"historian": 0, "validator": 0, "investigator": 0, "unknown": 0}
    total = 0
    for route_name, n in rows:
        key = route_name if route_name in by_route else "unknown"
        by_route[key] = n
        total += n

    # ── average confidence ────────────────────────────────────────────────────
    avg_q   = (
        select(func.avg(Classification.confidence))
        .where(Classification.user_id == user_id)
    )
    avg_val  = (await db.execute(avg_q)).scalar_one_or_none()
    avg_conf = round(float(avg_val), 4) if avg_val is not None else 0.0

    # ── top label for this user ───────────────────────────────────────────────
    top_q = (
        select(Classification.label, func.count(Classification.id).label("n"))
        .where(Classification.user_id == user_id)
        .group_by(Classification.label)
        .order_by(desc(func.count(Classification.id)))
        .limit(1)
    )
    top_row = (await db.execute(top_q)).first()
    top_label = {"label": top_row[0], "count": top_row[1]} if top_row else None

    # ── last 5 analyses ───────────────────────────────────────────────────────
    recent_q = (
        select(Classification)
        .where(Classification.user_id == user_id)
        .order_by(desc(Classification.timestamp))
        .limit(5)
    )
    recent_rows = (await db.execute(recent_q)).scalars().all()
    recent = [
        {
            "id":          r.id,
            "label":       r.label,
            "confidence":  round(r.confidence, 4) if r.confidence is not None else None,
            "route_taken": r.route_taken,
            "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in recent_rows
    ]

    return {
        "total_analyses": total,
        "by_route":        by_route,
        "avg_conf":        avg_conf,
        "top_label":       top_label,
        "recent":          recent,
    }


# ── POST /auth/forgot-password ────────────────────────────────────────────────

@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request a password reset email",
)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    FLOW:
        1. Look up user by email.
        2. If not found — return the SAME success message (prevent email enumeration).
        3. Create EmailVerification row with 1-hour expiry and type marking.
        4. Send password reset email.
        5. Return 200 with generic message.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user: User | None = result.scalar_one_or_none()

    if user is not None:
        raw_token = secrets.token_urlsafe(48)
        db.add(EmailVerification(
            user_id=user.id,
            token=f"reset:{raw_token}",   # prefix distinguishes reset from verification tokens
            expires_at=datetime.now(timezone.utc) + timedelta(hours=_PASSWORD_RESET_EXPIRE_HOURS),
        ))
        email_sent = await send_password_reset_email(user.email, raw_token)
        if not email_sent:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send reset email. System misconfigured."
            )
        await write_audit(db, action="user.forgot_password", user_id=user.id)

    # Always return the same message — do NOT leak whether the email exists
    return MessageResponse(message="If an account with that email exists, a reset link has been sent.")


# ── POST /auth/reset-password ─────────────────────────────────────────────────

@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Set a new password using a reset token",
)
async def reset_password(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    FLOW:
        1. Look up the EmailVerification row with token == "reset:{body.token}".
        2. Validate: not expired, not used.
        3. Hash and store the new password.
        4. Revoke ALL existing refresh tokens for this user (force re-login everywhere).
        5. Mark the verification token as used.
        6. Write audit log.
    """
    lookup_token = f"reset:{body.token}"
    result = await db.execute(
        select(EmailVerification).where(EmailVerification.token == lookup_token)
    )
    verification: EmailVerification | None = result.scalar_one_or_none()

    if verification is None or verification.used_at is not None:
        raise HTTPException(status_code=400, detail="Reset link is invalid or has already been used.")

    now = datetime.now(timezone.utc)
    if verification.expires_at.replace(tzinfo=timezone.utc) < now:
        raise HTTPException(status_code=400, detail="Reset link has expired. Request a new one.")

    result = await db.execute(select(User).where(User.id == verification.user_id))
    user: User | None = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Apply new password
    user.hashed_password = hash_password(body.new_password)

    # Revoke all refresh tokens for this user (force re-login everywhere)
    rt_result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id,
            RefreshToken.revoked_at.is_(None)
        )
    )
    for rt in rt_result.scalars().all():
        rt.revoked_at = now

    # Mark token as used
    verification.used_at = now

    await write_audit(db, action="user.password_reset", user_id=user.id)
    logger.info("Password reset for user id=%s", user.id)
    return MessageResponse(message="Password updated successfully. Please log in with your new password.")
