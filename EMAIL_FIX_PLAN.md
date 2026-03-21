# IMMEDIATE ACTION PLAN — Critical Email/Password Reset Hardening

**Date**: March 20, 2026  
**Priority**: BLOCKING PRODUCTION  
**Owner**: DevOps/Backend Team  
**Target Completion**: Within 3 business days  

---

## Executive Summary

The system has a **critical blocker** preventing production deployment:

**When deployed without `RESEND_API_KEY` set, ALL password reset and registration verification emails are silently not sent. Users can create accounts but cannot verify them, and cannot reset forgotten passwords.**

This document provides step-by-step implementation instructions to harden the email system for enterprise deployment.

---

## Current Broken Flow

```
User registers
    ↓
Server: create User(pending) + EmailVerification token
    ↓
Server: call send_verification_email(email, token)
    ↓
Email system: check if RESEND_API_KEY is set
    ├─ If YES → send via Resend API
    └─ If NO → log to console and return True (LIE!)
    ↓
Server: gets True (success signal) even though email was NOT sent
    ↓
Server: returns HTTP 201 (created successfully)
    ↓
Frontend: shows "Check your inbox!" ✅
    ↓
User: waits forever — email never arrives ❌
```

---

## Fix #1: Add Email Failure Detection (3 hours)

### 1.1 Update `src/api/auth/email.py`

**Current code (BROKEN):**
```python
async def _send(to: str, subject: str, html: str) -> bool:
    if not _resend_available():
        logger.info("[EMAIL DEV-MODE] Would send to=%s ...", to)
        return True  # ← WRONG: Returns success even though email was NOT sent
    
    try:
        # ... send logic
        return True
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False
```

**Fixed code:**
```python
import os

_ENV = os.getenv("ENV", "development")

async def _send(to: str, subject: str, html: str) -> bool:
    """
    Send email via Resend API.
    
    RETURNS:
        True if sent successfully.
        False if send failed.
        
    CRITICAL: In production, ALWAYS requires RESEND_API_KEY.
              Fails fast if key is missing (do not silently return True).
    """
    if not _resend_available():
        if _ENV == "production":
            # PRODUCTION: Missing key is a FATAL error
            logger.critical(
                "EMAIL DISABLED: RESEND_API_KEY is not set. "
                "Email verification and password reset will NOT work. "
                "This is a CRITICAL configuration error."
            )
            raise RuntimeError(
                "RESEND_API_KEY environment variable is required for email delivery."
            )
        else:
            # DEV: Gracefully skip email (print token to console)
            logger.info(
                "[EMAIL DEV-MODE] Would send to=%s subject=%r\n"
                "Token would be: %s",
                to, subject, "{{ TOKEN WOULD BE HERE }}"
            )
            return True  # OK to "succeed" in dev mode
    
    try:
        import resend
        resend.api_key = _RESEND_API_KEY

        def _blocking_send():
            return resend.Emails.send({
                "from": _RESEND_FROM_EMAIL,
                "to": [to],
                "subject": subject,
                "html": html,
            })

        result = await asyncio.to_thread(_blocking_send)
        logger.info("Email sent to=%s id=%s", to, result.get("id", "?"))
        return True

    except Exception as exc:
        logger.error("Failed to send email to=%s: %s", to, exc, exc_info=True)
        return False  # Fail explicitly


def _resend_available() -> bool:
    """Check if Resend is properly configured."""
    return bool(_RESEND_API_KEY)
```

**Changes:**
- Line 1-2: Import ENV setting
- Line 8-10: Check production flag
- Line 11-18: If production AND no key, raise RuntimeError (FAIL FAST)
- Line 19-26: If dev mode AND no key, log to console and return True (graceful)
- Line 46-47: Added `exc_info=True` for full error trace

---

### 1.2 Update `src/api/auth/router.py`

**Current code (BROKEN):**
```python
@router.post("/register")
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)) -> MessageResponse:
    # ... create user, token ...
    
    await send_verification_email(user.email, raw_token)  # ← Ignore return value!
    
    await db.commit()
    return MessageResponse(message="Please verify your email.")  # ← Always success
```

**Fixed code:**
```python
@router.post("/register")
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)) -> MessageResponse:
    # ... create user, token ...
    
    # CRITICAL: Check if email was actually sent
    email_sent_ok = await send_verification_email(user.email, raw_token)
    
    if not email_sent_ok:
        # Email send failed
        await db.rollback()  # undo transaction
        
        logger.error(
            "Registration rejected: email send failed for %s",
            body.email
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not send verification email. Please try again later."
        )
    
    await db.commit()
    return MessageResponse(
        message="Account created. Check your email to verify within 24 hours."
    )
```

**Changes:**
- Line 3: Capture return value (was ignored before)
- Line 5-17: If email failed, rollback and return 500 error
- Line 19-22: On success, confirm to user

---

### 1.3 Update `src/api/auth/router.py` (forgot-password endpoint)

**Current code (BROKEN):**
```python
@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)) -> MessageResponse:
    user = ... # find by email
    
    if user:
        token = secrets.token_urlsafe(48)
        db.add(EmailVerification(...))
        await send_password_reset_email(user.email, token)  # ← Ignore return!
    
    return MessageResponse(message="Check your inbox...")  # ← Always 200
```

**Fixed code:**
```python
@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)) -> MessageResponse:
    user_result = await db.execute(select(User).where(User.email == body.email))
    user = user_result.scalar_one_or_none()
    
    # SECURITY: Always return 200 (prevent email enumeration)
    # BUT: Log and alert on email failures
    
    if user is not None:
        token = secrets.token_urlsafe(48)
        db.add(EmailVerification(
            user_id=user.id,
            token=f"reset:{token}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        ))
        await db.flush()
        
        # CRITICAL: Check if email was sent
        email_sent_ok = await send_password_reset_email(user.email, token)
        
        if not email_sent_ok:
            # Email failed BUT we still return 200 (security)
            # BUT we log and alert the admin
            logger.error(
                "Password reset email failed for user id=%s email=%s",
                user.id, user.email
            )
            await write_audit(
                db,
                action="email.send_failed",
                user_id=user.id,
                payload={"reason": "password_reset_email_failed"}
            )
            # TODO: Alert admin via Slack/PagerDuty
        else:
            logger.info("Password reset email sent to=%s", user.email)
    
    await db.commit()
    
    # ALWAYS return 200 for security (attacker can't tell if email exists)
    return MessageResponse(message="If an account is associated with this email, a reset link has been sent.")
```

**Changes:**
- Line 15-17: Capture email send result
- Line 19-28: Log and alert on email failure (but still return 200)
- Line 30: Write audit log for failure
- Added TODO for Slack alert (implement in Fix #4)

---

## Fix #2: Add Email Log Table (2 hours)

### 2.1 Create migration file

**File**: `migrations/versions/001_email_log.py`

```python
"""
Email audit log table.

Migration: 001_email_log
Date: 2026-03-20
Reason: Track all email sends for debugging and compliance
"""

def upgrade():
    """Create email_log table."""
    op.create_table(
        'email_log',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('user_id', sa.UUID(), nullable=True),
        sa.Column('to_email', sa.String(255), nullable=False),
        sa.Column('template', sa.String(50), nullable=False),
        sa.Column('provider', sa.String(20), nullable=False),
        sa.Column('provider_message_id', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='queued'),
        sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_reason', sa.String(500), nullable=True),
        sa.Column('bounce_type', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "status IN ('queued', 'sent', 'delivered', 'bounced', 'failed')",
            name='email_log_status_valid'
        )
    )
    op.create_index('ix_email_log_user_id', 'email_log', ['user_id'])
    op.create_index('ix_email_log_to_email', 'email_log', ['to_email'])
    op.create_index('ix_email_log_status', 'email_log', ['status'])
    op.create_index('ix_email_log_created_at', 'email_log', ['created_at'], unique=False)


def downgrade():
    """Drop email_log table."""
    op.drop_index('ix_email_log_created_at', table_name='email_log')
    op.drop_index('ix_email_log_status', table_name='email_log')
    op.drop_index('ix_email_log_to_email', table_name='email_log')
    op.drop_index('ix_email_log_user_id', table_name='email_log')
    op.drop_table('email_log')
```

### 2.2 Add ORM model

**File**: `src/api/db/models.py` (add to existing file)

```python
class EmailLog(Base):
    """Audit trail for all email sends."""
    __tablename__ = "email_log"
    
    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=True)
    to_email     = Column(String(255), nullable=False, index=True)
    template     = Column(String(50), nullable=False)  # "verify_email", "reset_password", etc.
    provider     = Column(String(20), nullable=False)  # "resend", "sendgrid", "ses"
    provider_message_id = Column(String(255), nullable=True)
    status       = Column(String(20), default="queued", index=True)  # queued, sent, delivered, bounced, failed
    attempt_count = Column(Integer, default=0)
    error_reason = Column(String(500), nullable=True)
    bounce_type  = Column(String(20), nullable=True)  # permanent, temporary, complaint
    created_at   = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    sent_at      = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    expires_at   = Column(DateTime(timezone=True), nullable=True)
    metadata     = Column(JSON, nullable=True)
```

### 2.3 Run migration

```powershell
# Apply migration to database
& $env:PYTHONPATH\alembic upgrade head
```

---

## Fix #3: Update existing endpoint calls to log emails (1 hour)

### 3.1 Update `resend_verification` endpoint

**File**: `src/api/auth/router.py`

```python
@router.post("/resend-verification")
async def resend_verification(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Resend verification email for pending account."""
    
    user_result = await db.execute(select(User).where(User.email == body.email))
    user = user_result.scalar_one_or_none()
    
    # SECURITY: Always return 200 (prevent enumeration)
    
    if user is not None and user.status == UserStatus.pending:
        raw_token = secrets.token_urlsafe(48)
        db.add(EmailVerification(
            user_id=user.id,
            token=f"verify:{raw_token}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        ))
        await db.flush()
        
        # NEW: Check email send result
        email_sent_ok = await send_verification_email(user.email, raw_token)
        
        if not email_sent_ok:
            logger.error("Resend verification email failed for user %s", user.id)
            # Still return 200 (security), but log failure
        else:
            logger.info("Resend verification email sent to %s", user.email)
    
    await db.commit()
    
    return MessageResponse(
        message="If a pending account exists with this email, a fresh verification link has been sent."
    )
```

---

## Fix #4: Setup Resend Webhook for Delivery Confirmation (2 hours)

### 4.1 Add webhook endpoint

**File**: `src/api/routes/webhooks.py` (create new file)

```python
"""
src/api/routes/webhooks.py
============================
Webhook handlers for email delivery confirmations.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import hmac
import hashlib
import logging

from src.api.db.models import EmailLog
from src.api.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

_RESEND_WEBHOOK_SECRET = os.getenv("RESEND_WEBHOOK_SECRET", "")


def _verify_resend_signature(body: str, signature: str) -> bool:
    """Verify that webhook came from Resend (not attacker)."""
    if not _RESEND_WEBHOOK_SECRET:
        logger.warning("RESEND_WEBHOOK_SECRET not set — webhook verification disabled")
        return True
    
    expected = hmac.new(
        _RESEND_WEBHOOK_SECRET.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


@router.post("/resend")
async def webhook_resend(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Resend.com sends webhook events for email lifecycle.
    
    Triggered by: email.sent, email.delivered, email.bounced, email.complained
    """
    
    # Verify signature
    body_raw = await request.body()
    signature = request.headers.get("X-Resend-Signature", "")
    
    if not _verify_resend_signature(body_raw.decode(), signature):
        logger.warning("Invalid Resend webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    body = await request.json()
    event_type = body.get("type")
    data = body.get("data", {})
    message_id = data.get("email_id")
    
    if not message_id:
        logger.warning("Resend webhook: missing email_id")
        return {"ok": True}  # Resend may retry, so return 200
    
    # Find email log
    result = await db.execute(
        select(EmailLog).where(EmailLog.provider_message_id == message_id)
    )
    email_log = result.scalar_one_or_none()
    
    if not email_log:
        logger.warning("Resend webhook: message_id=%s not found in DB", message_id)
        return {"ok": True}
    
    # Update log based on event
    if event_type == "email.delivered":
        email_log.status = "delivered"
        email_log.delivered_at = datetime.now(timezone.utc)
        logger.info("Email delivered: log_id=%s msg_id=%s", email_log.id, message_id)
    
    elif event_type == "email.bounced":
        email_log.status = "bounced"
        email_log.bounce_type = data.get("bounce_type", "unknown")
        logger.warning("Email bounced: log_id=%s type=%s", email_log.id, email_log.bounce_type)
    
    elif event_type == "email.complained":
        email_log.status = "bounced"
        email_log.bounce_type = "complaint"
        logger.warning("Email complained: log_id=%s", email_log.id)
    
    elif event_type == "email.sent":
        email_log.status = "sent"
        email_log.sent_at = datetime.now(timezone.utc)
        logger.info("Email sent: log_id=%s", email_log.id)
    
    await db.commit()
    return {"ok": True}
```

### 4.2 Register webhook route in main.py

**File**: `src/api/main.py` (add to imports)

```python
from src.api.routes.webhooks import router as webhooks_router

# Add to app.include_router() section:
app.include_router(webhooks_router)
```

### 4.3 Configure webhook in Resend dashboard

1. Go to https://resend.com/webhooks
2. Create new webhook:
   - URL: `https://yourdomain.com/webhooks/resend`
   - Topics: select `email.sent`, `email.delivered`, `email.bounced`, `email.complained`
3. Copy webhook signing secret → add to `.env`:
   ```dotenv
   RESEND_WEBHOOK_SECRET=whk_test_xxxxx
   ```

---

## Fix #5: Add Multiple Email Providers (Fallback) (4 hours)

See **ENTERPRISE_AUDIT.md Section 6.3** for complete implementation.

Quick summary:
1. Create `src/api/email/providers.py` with SendGridProvider + AWSSeProvider classes
2. Update `src/api/auth/email.py` to try Resend first, SendGrid second, SES third
3. Add env vars: `SENDGRID_API_KEY`, `AWS_REGION`, etc.
4. Test failover logic in unit tests

---

## Testing Checklist

After implementing fixes above, verify:

### Pre-deployment smoke test

```bash
# 1. Start API with RESEND_API_KEY NOT set
$env:ENV = "development"
$env:RESEND_API_KEY = ""  # Explicitly clear
uvicorn src.api.main:app --reload

# 2. Try register → should see emails printed to console (not sent to Resend)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Password123!"}'
# Expected: HTTP 201, email token printed to console

# 3. Start API in PRODUCTION mode without RESEND_API_KEY
$env:ENV = "production"
$env:RESEND_API_KEY = ""  # Missing key
uvicorn src.api.main:app

# 4. Try register → should see HTTP 500 (FAIL FAST)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Password123!"}'
# Expected: HTTP 500 "RESEND_API_KEY is required"

# 5. Set RESEND_API_KEY again
$env:RESEND_API_KEY = "re_xxxxx"

# 6. Try register → should return 201 and email should be sent
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"real@example.com","password":"Password123!"}'
# Expected: HTTP 201, real email sent to real@example.com
```

### Verify email_log table

```sql
-- Check PostgreSQL
SELECT * FROM email_log ORDER BY created_at DESC LIMIT 5;

-- Expected output:
-- id | to_email | template | provider | status | created_at | delivered_at
-- ---|----------|----------|----------|--------|------------|---------------
-- UUID | real@example.com | verify_email | resend | sent | 2026-03-20 14:30 | NULL
```

### Verify webhook handling

```bash
# Simulate Resend sending a delivery confirmation webhook
curl -X POST http://localhost:8000/webhooks/resend \
  -H "Content-Type: application/json" \
  -H "X-Resend-Signature: xxxxx" \
  -d '{"type":"email.delivered","data":{"email_id":"UUID_FROM_ABOVE"}}'
# Expected: HTTP 200

# Check log updated
SELECT status, delivered_at FROM email_log WHERE id = 'UUID_FROM_ABOVE';
# Expected: status = "delivered", delivered_at = now()
```

---

## Deployment Checklist

Before going live:

- [ ] RESEND_API_KEY env var is set in all environments (dev, staging, prod)
- [ ] .env example includes RESEND_API_KEY, RESEND_WEBHOOK_SECRET
- [ ] Database migration applied (email_log table created)
- [ ] Webhook endpoint tested (can receive Resend POSTs)
- [ ] Resend webhook URL configured in Resend dashboard
- [ ] Resend webhook secret added to .env
- [ ] All 3 routes tested: register, forgot-password, resend-verification
- [ ] Admin email audit log dashboard accessible at /api/admin/email-logs
- [ ] Slack alert configured for email failures (if applicable)
- [ ] Email logs retention policy set (keep 90 days)
- [ ] Fallback provider (SendGrid) keys added to .env (if using)

---

## Timeline

| Task | Owner | Duration | Dependencies |
|------|-------|----------|---|
| Fix #1: Email failure detection | Backend | 3h | None |
| Fix #2: Add email_log table | Backend | 2h | Fix #1 testing |
| Fix #3: Update endpoints | Backend | 1h | Fix #2 deployed |
| Fix #4: Resend webhooks | Backend | 2h | Fix #2 deployed |
| Fix #5: Fallback providers | Backend | 4h | Fix #4 tested |
| Testing & validation | QA | 2h | All fixes |
| Deployment | DevOps | 1h | All testing |
| **Total** | | **15 hours** | **Can be done in 1-2 days** |

---

## Questions?

See `ENTERPRISE_AUDIT.md` for complete context, business logic, and failure case analysis.
