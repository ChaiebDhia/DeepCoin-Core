# EMAIL/PASSWORD RESET — Current vs Fixed Architecture

## 🔴 CURRENT STATE (BROKEN)

```
┌─────────────────────────────────────────────────────────────────┐
│ USER ACTION: Register with email                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /auth/register                                   │
│ Payload: { email, password }                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: register() endpoint                                    │
│ ✅ Create User(status="pending")                                │
│ ✅ Create EmailVerification token                               │
│ ❌ Call send_verification_email(email, token)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                            │
            ▼ RESEND_API_KEY set         ▼ RESEND_API_KEY empty
            │                            │
     ✅ Send via Resend      ❌ Log "[DEV-MODE]" + return True
            │                            │
            └──────────────┬──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: Gets return value = True (even if skipped!)            │
│ ❌ Does NOT check if email was actually sent                    │
│ ✅ Commits transaction                                          │
│ ✅ Returns HTTP 201 "Check your email"                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND: HTTP 201 received                                     │
│ ✅ Shows message: "Check your inbox! Verification link sent."   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ USER: Checks inbox                                              │
│ ❌ If RESEND_API_KEY was empty: NO EMAIL ARRIVES                │
│ ❌ User is stuck, cannot verify account                         │
└─────────────────────────────────────────────────────────────────┘

RESULT: Silent Failure ❌
───────────────────────────────────────────────────────────
User sees: ✅ "Success!"
User gets: ❌ Nothing

Account Status: PENDING (cannot login until verified)
Email Status: NEVER SENT
Recovery Path: Stuck forever unless admin manually verifies
```

---

## 🟢 FIXED STATE (with all 5 fixes applied)

```
┌─────────────────────────────────────────────────────────────────┐
│ USER ACTION: Register with email                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND STARTUP: Validate configuration                         │
│ IF ENV == "production" AND RESEND_API_KEY empty                 │
│ ❌ RAISE FATAL ERROR immediately                                │
│ ❌ App refuses to start                                         │
│ ℹ️  Admin must fix config before deployment                      │
└─────────────────────────────────────────────────────────────────┘

                    [OR in DEVELOPMENT MODE]

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND STARTUP: In dev mode, allow graceful skip               │
│ ℹ️  Print emails to console instead of sending to Resend        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ USER ACTION: Register with email                                │
│ POST /auth/register { email, password }                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: register() endpoint — FIX #1 applied                   │
│ ✅ Create User(status="pending")                                │
│ ✅ Create EmailVerification token                               │
│ 🆕 Call: email_sent_ok = await send_verification_email(...)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                            │
            ▼ email_sent_ok = True        ▼ email_sent_ok = False
     ✅ Email sent OK           ❌ Email send failed
            │                            │
            │                   ┌────────┴──────────┐
            │                   │                  │
            │          🆕 FIX #2: Log to DB    🆕 FIX #3: Rollback
            │             (email_log table)    transaction
            │                   │
            └───────────┬───────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ email_sent_ok = True          │
        │ ✅ Commit transaction         │
        │ ✅ Return HTTP 201            │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴──────────────────┐
        │                                 │
        ▼ Email was sent                   ▼ Email failed
                                          ❌ Rollback: User never created
        📧 SENT TO USER

        ✅ Frontend shows:               ❌ Frontend shows:
        "Check your inbox"               "Email service unavailable"

                                         ℹ️  User can retry later

        👤 User receives email           👤 User sees error message
        🔓 Clicks verify link            🚫 Cannot proceed
        ✅ Account activated             ↩️  Can try register again
        🔓 Can now login


        [Delivery Confirmation]
        🆕 FIX #4: Resend webhook event
                                          
        Resend confirms: "delivered"
        📨 → 🪝 POST /webhooks/resend
        ✅ email_log.status = "delivered"
        ✅ admin can see delivery success
```

---

## Side-by-side: Forgot Password Flow

### ❌ CURRENT BROKEN

```
User forgets password
           │
           ▼
Frontend: /auth/forgot-password { email }
           │
           ▼
Backend: Look up user by email
         │
         ├─ No user found → return 200 (security: prevent enumeration)
         │
         └─ User found → send_password_reset_email()
                          ├─ If RESEND_API_KEY empty: log + return True
                          └─ Backend: ignores result, returns 200
           │
           ▼
Frontend: HTTP 200 "Check your inbox"
           │
           ▼
User: Waits for email ❌ Never arrives

Result: Cannot reset password (email lost)
```

### ✅ FIXED

```
User forgets password
           │
           ▼
Frontend: /auth/forgot-password { email }
           │
           ▼
Backend: Look up user by email
         │
         ├─ No user found → return 200 (security: prevent enumeration)
         │
         └─ User found → email_sent = await send_password_reset_email()
                          │
                          ├─ Success → log to email_log, return 200
                          │            + 🆕 webhook event monitored
                          │
                          └─ Failure → 🆕 log to email_log (FAILURE)
                             + 🆕 alert admin
                             + 🆕 return 200 anyway (security)
           │
           ▼
Frontend: HTTP 200 "Check your inbox"
           │
           ▼
Admin dashboard: 🆕 New "Failed" entry in email_log
                 Email send to john@example.com failed!
                 
(Meanwhile...)

User: Email arrives (if not failed) ✅
      Clicks link → can reset password

If email failed:
Admin notified immediately
Can investigate + retry manually
```

---

## Failure Scenarios Covered

| Scenario | Before Fix | After Fix |
|----------|-----------|-----------|
| RESEND_API_KEY empty in DEV | Email skipped silently, user sees success | Email printed to console, explicit log ✅ |
| RESEND_API_KEY empty in PROD | Email skipped, user sees success ❌ | App refuses to start ✅ |
| Resend API down (5XX) | Error logged, user sees success ❌ | Logged + admin alerted + fallback tried ✅ |
| User email invalid/bounces | Email bounces, no logging | Webhook received, email_log updated, admin sees ✅ |
| User clicks wrong link | Nothing happens | Error shown to user ✅ |
| 1000 registrations overnight | No tracking | email_log shows all sends, admin can audit ✅ |

---

## Dashboard Visibility (with Fix #3+#4)

### NEW: Admin Email Audit Log

```
GET /api/admin/email-logs?page=1&status=failed

Response:
[
  {
    "id": "uuid-123",
    "to_email": "john@example.com",
    "template": "verify_email",
    "provider": "resend",
    "status": "failed",
    "attempt_count": 2,
    "error_reason": "Resend API returned 400: Invalid email",
    "created_at": "2026-03-20T14:30:00Z",
    "metadata": {
      "user_id": "uuid-456",
      "user_name": "John Doe"
    }
  },
  ...
]

Stats dashboard:
- Total emails sent today: 245
- Delivered: 242 (98.8%) ✅
- Failed: 2 (0.8%) ⚠️
- Bounced: 1 (0.4%) ⚠️
- Provider: Resend (100%)
```

---

## Email Retry Architecture (Fix #2)

```
User registers
     │
     ▼
Call: asyncio.create_task(send_with_retry("verify", email, token))

send_with_retry():
  attempt = 1
  while attempt <= 3:
    try:
      result = await send_via_resend(email, token)
      ✅ Success → log("delivered")
      return True
    except Exception as e:
      ❌ Failure → log("failed", error=e)
      
      if attempt < 3:
        wait_time = 2 ^ attempt  # 2s, 4s, 8s
        log(f"Retry in {wait_time}s...")
        await asyncio.sleep(wait_time)
        attempt += 1
      else:
        log("All 3 retries failed")
        await alert_admin("Email failed after 3 retries")
        return False

Result: Resilient email delivery even if Resend temporarily down
```

---

## Testing Strategy

### Unit Test Example

```python
async def test_register_email_fail_blocks_creation():
    """If email send fails, registration must be rolled back."""
    
    # Mock email provider to fail
    with patch("src.api.auth.email.send_verification_email") as mock_send:
        mock_send.return_value = False  # Email send failed
        
        response = await client.post(
            "/auth/register",
            json={"email": "test@example.com", "password": "Secure123!"}
        )
        
        # Should return error, not success
        assert response.status_code == 500
        assert "Could not send verification email" in response.json()["detail"]
        
        # User should NOT be created in DB
        user = await db.get(User, filters={"email": "test@example.com"})
        assert user is None  # ✅ Transaction rolled back
```

### Integration Test Example

```python
async def test_email_log_created_on_send():
    """Every email send must be logged."""
    
    # Clear logs
    await db.execute(delete(EmailLog))
    
    # Register user
    response = await client.post(
        "/auth/register",
        json={"email": "integration@test.com", "password": "Secure123!"}
    )
    assert response.status_code == 201
    
    # Check email_log entry created
    log_entry = await db.get(
        EmailLog,
        filters={"to_email": "integration@test.com"}
    )
    assert log_entry is not None
    assert log_entry.template == "verify_email"
    assert log_entry.provider == "resend"
    assert log_entry.status == "sent"
    assert log_entry.provider_message_id is not None
```

---

## Deployment Steps

### Day 1: Development & Testing
- [ ] Implement Fix #1 (email failure detection)
- [ ] Implement Fix #2 (email_log table + migration)
- [ ] Run migration on local DB
- [ ] Run smoke tests

### Day 2: Comprehensive Testing
- [ ] Implement Fix #3 (error propagation in endpoints)
- [ ] Implement Fix #4 (webhook handler)
- [ ] Configure Resend webhook
- [ ] Run full integration test suite
- [ ] Test failover (disable Resend, verify error message)

### Day 3: Staging & Production
- [ ] Deploy to staging
- [ ] Run E2E tests against staging
- [ ] Verify email_log table populated
- [ ] Verify webhook receiving confirmations
- [ ] Deploy to production
- [ ] Verify RESEND_API_KEY is set
- [ ] Monitor email_log for 24 hours (watch for failures)

---

## Quick Reference: What Changed

```
BEFORE: if [email fails] → user sees "success" ❌
        email_sent_ok = await send_verification_email(...)
        # ← result ignored, always returns "Check inbox"

AFTER:  email_sent_ok = await send_verification_email(...)
        if not email_sent_ok:
            await db.rollback()
            return HTTP 500 "Email service unavailable"  ✅
        # ← explicit error to user, transaction rolled back
```

```
BEFORE: No visibility into email failures
        Admin has NO audit trail
        
AFTER:  All sends logged to email_log table ✅
        Admin can query failures + debug
        Webhook provides delivery confirmation
        Admin dashboard shows stats
```

```
BEFORE: Single Resend provider
        If Resend down: all emails fail silently
        
AFTER:  Retry logic: 3 attempts with exponential backoff ✅
        Fallback providers: SendGrid, AWS SES ✅
        Webhook confirmation: track delivery status ✅
```
