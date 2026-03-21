# DeepCoin Enterprise Audit — End-to-End Flow and System Design

**Date**: March 20, 2026  
**Scope**: Complete system from homepage to all authentication flows, UX logic, business rules, and failure cases  
**Status**: CRITICAL GAPS IDENTIFIED — Email/password reset is not production-ready

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           DEEPCOIN FULL STACK                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Frontend Layer (Next.js 15 App Router)
├─ app/page.tsx                     (Homepage — Server Component)
├─ app/login/page.tsx               (Sign-in page)
├─ app/register/page.tsx            (Sign-up page)
├─ app/forgot-password/page.tsx     (Password reset request)
├─ app/reset-password/page.tsx      (Password reset form)
├─ app/verify-email/page.tsx        (Email verification)
├─ app/analyse/page.tsx             (Main coin classifier)
├─ app/history/page.tsx             (User's analysis history)
├─ app/history/[id]/page.tsx        (Analysis detail view)
├─ app/explore/page.tsx             (Public gallery)
├─ app/chat/page.tsx                (AI Q&A interface)
├─ app/about/page.tsx               (Project information)
├─ app/docs/page.tsx                (API documentation)
├─ app/admin/page.tsx               (Admin dashboard)
└─ app/contact/page.tsx             (Contact form)

Backend Layer (FastAPI)
├─ src/api/auth/router.py           (Auth endpoints)
│  ├─ POST /auth/register           (Create account)
│  ├─ POST /auth/login              (Sign-in)
│  ├─ GET  /auth/verify-email       (Confirm email)
│  ├─ POST /auth/forgot-password    (Request reset)
│  ├─ POST /auth/reset-password     (Apply new password)
│  ├─ POST /auth/resend-verification (Resend verification)
│  ├─ POST /auth/refresh            (Get new access token)
│  ├─ POST /auth/logout             (Revoke session)
│  └─ GET  /auth/me                 (Current user)
├─ src/api/auth/email.py            (Email delivery — RESEND ONLY)
├─ src/api/routes/classify.py       (Coin analysis)
├─ src/api/routes/history.py        (Analysis records)
├─ src/api/routes/chat.py           (AI chat)
├─ src/api/routes/subscribers.py    (Waitlist emails)
└─ src/api/routes/admin.py          (Admin operations)

Database Layer (PostgreSQL)
├─ users                            (First-time auth)
├─ email_verification              (Token-based flows)
├─ refresh_tokens                   (Session management)
├─ classifications                  (Coin analyses)
├─ chat_messages                    (Conversation history)
└─ audit_log                        (Security events)

Email Layer (Resend.com)
└─ SINGLE PROVIDER — NO FALLBACK ⚠️ CRITICAL GAP
```

---

## 2. COMPLETE USER FLOWS

### 2.1 REGISTRATION FLOW (App/Register to Email Verified)

```
┌─ START: app/register/page.tsx
│
├─ User fills form:
│  ├─ display_name (optional, max 100 chars)
│  ├─ email (required, validated as EmailStr)
│  └─ password (required, 8+ chars, 1 letter + 1 digit/special char)
│
├─ RegisterForm.tsx validates client-side:
│  ├─ password === confirmPassword (warn if not)
│  ├─ password.length >= 8 (warn if not)
│  └─ Disable submit button until all rules pass
│
├─ POST /auth/register
│  ├─ Server validates again (Pydantic + field_validator)
│  └─ If validation fails → HTTP 422 (Unprocessable Entity)
│    └─ Client shows Pydantic error details in form
│
├─ Server checks unique constraint:
│  └─ SELECT * FROM users WHERE email = ?
│    └─ If exists → HTTP 409 Conflict ("Email already registered")
│
├─ Server creates User record:
│  ├─ status = "pending" (NOT active yet)
│  ├─ email_verified_at = NULL
│  └─ hashed_password = bcrypt(password, rounds=12)
│
├─ Server creates EmailVerification record:
│  ├─ token = secrets.token_urlsafe(48)  (e.g., "abc123...xyz")
│  ├─ expires_at = now() + 24 hours
│  └─ type = "verify"
│
├─ Server attempts email delivery (Resend):
│  ├─ send_verification_email(email, token)
│  ├─ Email contains: {APP_URL}/verify-email?token={token}
│  └─ Template: DeepCoin branded dark-mode HTML
│
├─ EMAIL FAILURE SCENARIO ⚠️:
│  └─ If RESEND_API_KEY is NOT set:
│     ├─ Email "send" silently skips (logs to INFO only)
│     ├─ User gets HTTP 201 Created (success message)
│     ├─ BUT: User will NEVER receive the verification link
│     └─ User tries to login → gets 403 "verify your email"
│
├─ SUCCESS: HTTP 201
│  └─ Client shows "Check your inbox" confirmation
│  └─ Link to login or resend verification below
│
└─ END: User receives email (or not, depending on RESEND_API_KEY)

CRITICAL BUG: No error handling if email send fails.
              Server returns 201 (success) even if email was never sent.
```

### 2.2 EMAIL VERIFICATION FLOW

```
┌─ START: User clicks link from email
│  └─ GET /verify-email?token=abc123
│
├─ Frontend (app/verify-email/page.tsx):
│  ├─ Extracts token from ?token= parameter
│  ├─ Server-side invokes endpoint with token
│  └─ Waits for response
│
├─ Server validates token:
│  ├─ SELECT * FROM email_verification WHERE token = ?
│  ├─ If not found → HTTP 404 ("Token not found")
│  ├─ If expired (expires_at < now()) → HTTP 400 ("Token expired")
│  └─ If already used (used_at IS NOT NULL) → HTTP 400 ("Token already used")
│
├─ If valid:
│  ├─ UPDATE users SET email_verified_at = now(), status = "active"
│  ├─ UPDATE email_verification SET used_at = now()
│  ├─ Write audit log: action="user.email_verified"
│  └─ HTTP 200 { "message": "Email verified successfully!" }
│
├─ SUCCESS: Frontend shows green checkmark + "Email verified!"
│  └─ Button: "Sign in to your account" → /login
│
└─ END: Account is now active, can sign in

EDGE CASE: User clicks the same link twice
           → First click works (sets used_at)
           → Second click gets 400 (token already used)
           → User sees "This link was already used"
```

### 2.3 LOGIN FLOW

```
┌─ START: app/login/page.tsx (LoginForm.tsx)
│
├─ User fills form:
│  ├─ email
│  └─ password
│
├─ Client validates:
│  └─ Required fields present (email.length > 0, password.length > 0)
│
├─ POST /auth/login { email, password }
│
├─ Server validates input (Pydantic):
│  ├─ email must be valid EmailStr format
│  └─ password must be non-empty string
│  └─ If validation fails → HTTP 422
│
├─ Server looks up user:
│  └─ SELECT * FROM users WHERE email = ?
│  └─ If not found → HTTP 401 "Incorrect email or password"
│     ⚠️ NOTE: Same message for "email not found" or "wrong password"
│        This prevents user enumeration attacks
│
├─ Server verifies password (bcrypt constant-time check):
│  ├─ bcrypt.verify(password, user.hashed_password)
│  ├─ If mismatch → HTTP 401 "Incorrect email or password"
│  ├─ Write audit log: action="user.login_failed" with reason
│  └─ Return 401
│
├─ Server checks account status:
│  ├─ status = "suspended" → HTTP 403 "Account suspended, contact admin"
│  ├─ status = "pending" → HTTP 403 "Please verify your email first"
│     └─ (In dev mode with ENV != "production": auto-activate)
│  └─ status = "active" → proceed
│
├─ Server creates tokens:
│  ├─ access_token = JWT { user_id, exp=now+15min }
│  ├─ refresh_token = secrets.token_urlsafe(48)
│  └─ Store refresh_token hash in db (httpOnly cookie, 7 days)
│
├─ Server sets cookie:
│  ├─ name="refresh_token"
│  ├─ value=refresh_token
│  ├─ httpOnly=true (JavaScript cannot steal)
│  ├─ secure=true (HTTPS only in production)
│  ├─ samesite="lax" (CSRF resistant)
│  ├─ path="/auth" (only sent to /auth/* endpoints)
│  └─ max_age = 7 days
│
├─ Server returns HTTP 200:
│  └─ { "access_token": "eyJ...", "expires_in": 900, "user": {...} }
│
├─ Frontend stores access_token in memory (NextAuth.js Session.user)
│  └─ Refresh cookie is automatically sent by browser on next request
│
├─ SUCCESS: Redirect to /analyse or callback URL
│
└─ END: User is logged in

PENDING EMAIL EDGE CASE:
  If user was created before email confirmation logic,
  or email never arrived, they see "Please verify your email first"
  → Button "Resend verification email" sends POST /auth/resend-verification
     → Creates new verification token (old one superseded)
     → Sends fresh email
     → Always returns HTTP 200 (prevents enumeration)
```

### 2.4 FORGOT PASSWORD FLOW (Complete Journey)

```
┌─ START: User clicks "Forgot password?" on /login
│  → Navigates to /forgot-password
│
├─ ForgotPasswordForm.tsx shows email input
│  ├─ User enters email address
│  └─ Clicks "Send reset link"
│
├─ POST /auth/forgot-password { email }
│
├─ Server validates email format (Pydantic EmailStr)
│  └─ If invalid (e.g., "not-an-email") → HTTP 422
│
├─ Server looks up user:
│  └─ SELECT * FROM users WHERE email = ?
│
├─ SECURITY: Always returns HTTP 200 regardless of whether email exists
│  ├─ This prevents attackers from enumerating registered accounts
│  ├─ User sees "Check your inbox" whether the account exists or not
│  └─ The user's behavior is identical whether successful or not
│
├─ IF email exists in database:
│  ├─ Generate reset token: secrets.token_urlsafe(48)
│  ├─ CREATE EmailVerification record:
│  │  ├─ token = "reset:{new_token}"  (prefix distinguishes from verify)
│  │  ├─ expires_at = now() + 1 hour (shorter than verify, higher stakes)
│  │  └─ type = "reset"
│  ├─ Send password reset email:
│  │  ├─ TO: user.email
│  │  ├─ LINK: {APP_URL}/reset-password?token={token}
│  │  └─ TEMPLATE: Dark-mode HTML with 1-hour expiry warning
│  ├─ Log: action="user.password_reset_requested"
│  └─ Return HTTP 200 (no distinction made)
│
├─ IF email does NOT exist:
│  └─ Return HTTP 200 (attacker doesn't know)
│
├─ RESPONSE: HTTP 200 { "message": "Check your inbox..." }
│
├─ Frontend shows:
│  └─ "Check your inbox" + "If {email} is associated with an account,
│      a password reset link has been sent. It expires in 1 hour."
│  └─ Button "Try a different email" (can re-enter)
│
└─ END: Now waiting for user to click email link

CRITICAL EMAIL GAP ⚠️:
  If RESEND_API_KEY is not set:
  ├─ Email is silently not sent
  ├─ Server returns 200 anyway (success appearance)
  ├─ User waits for email that will never arrive
  └─ User must use "Try a different email" or give up
```

### 2.5 PASSWORD RESET FLOW (Token Validation to New Password)

```
┌─ START: User receives password reset email
│  └─ Clicks link: {APP_URL}/reset-password?token=xyz789
│
├─ Frontend (app/reset-password/page.tsx):
│  ├─ ResetPasswordForm.tsx extracts ?token= parameter
│  ├─ Displays password input form
│  └─ User fills new password + confirm password
│
├─ Client validation:
│  ├─ password.length >= 8
│  ├─ password === confirmPassword
│  └─ Warn if invalid before submit
│
├─ POST /auth/reset-password { token, new_password }
│
├─ Server validates new_password format (Pydantic):
│  ├─ min_length=8, max_length=128
│  ├─ Must contain letter + digit/special char
│  └─ If validation fails → HTTP 422
│
├─ Server looks up token:
│  └─ SELECT * FROM email_verification WHERE token = "reset:{token}"
│
├─ Validate token state:
│  ├─ If not found → HTTP 400 { "detail": "Invalid reset token" }
│  ├─ If expired (expires_at < now()) → HTTP 400 { "detail": "Token expired. Request a new reset link." }
│  ├─ If already used (used_at IS NOT NULL) → HTTP 400 { "detail": "Reset link already used" }
│  └─ All cases show same HTTP 400 (prevents timing attacks)
│
├─ If token is valid:
│  ├─ Get associated user_id from email_verification.user_id
│  ├─ UPDATE users SET hashed_password = bcrypt(new_password, 12)
│  ├─ UPDATE email_verification SET used_at = now()
│  ├─ DELETE FROM refresh_tokens WHERE user_id = ?  (logout all sessions)
│  ├─ Log: action="user.password_reset_completed"
│  ├─ Log: action="user.logout_all_sessions"
│  └─ HTTP 200 { "message": "Password reset successfully!" }
│
├─ Frontend shows success message:
│  └─ Green checkmark + "Password updated successfully!"
│  └─ "Redirecting to sign in..." (3-sec auto-redirect)
│  └─ Auto-navigates to /login after 3 seconds
│
├─ User now must login with NEW password:
│  └─ All old refresh tokens are invalidated
│  └─ User is logged out everywhere (security feature)
│
└─ END: Password changed, session reset, user can login

EDGE CASES:
  1. User receives two reset emails accidentally
     → First token gets used(used_at = now())
     → Second token → HTTP 400 "Token already used"
  2. User waits > 1 hour to click link
     → Token expires → HTTP 400 "Token expired"
  3. User opens same link twice (back button)
     → First click succeeds
     → Second click → HTTP 400 "Reset link already used"
```

### 2.6 HOMEPAGE FLOW (Visitor to First Analysis)

```
┌─ START: User lands on app/page.tsx (/)
│
├─ Server Components render:
│  ├─ HeroSection
│  │  ├─ Brand logo + tagline ("Archaeological Coin Intelligence")
│  │  ├─ Large CTA button: "Start Analyzing Coins"
│  │  ├─ Secondary link: "Learn by Examples" (→ /explore)
│  │  └─ Floating coin animation (Framer Motion)
│  ├─ PipelineSteps
│  │  ├─ 4 step visualization: Upload → CNN → Agents → PDF
│  │  └─ Each step has icon + brief description
│  ├─ StatsBar
│  │  ├─ Counters: 80.03% accuracy | 9,716 types | 47,705 KB vectors | <20s latency | 122 tests
│  │  └─ Animated countup numbers
│  ├─ ValueCards
│  │  ├─ "Forensic Analysis" — HSV+OpenCV material detection
│  │  ├─ "Grounded Knowledge" — RAG prevents hallucination
│  │  └─ "Unknown Coin Detective" — Graceful degradation for OOD
│  ├─ ForWhoCards
│  │  ├─ Archaeologists
│  │  ├─ Museum Curators
│  │  └─ Collectors & Enthusiasts
│  ├─ TechStack (bento grid + logos)
│  ├─ EmailCapture (waitlist)
│  └─ Footer
│
├─ Client Island: AnalyseSection
│  ├─ "use client" component — interactive
│  ├─ CoinUploader (file input, drag-drop, TTA toggle)
│  ├─ AgentPipeline (mission control modal when analyzing)
│  └─ Requires authentication to actually submit
│
├─ User sees two CTA paths:
│  ├─ AUTHENTICATED: "Start Analyzing" button → /analyse (upload page)
│  ├─ ANONYMOUS: "Start Analyzing" button → /login?callbackUrl=/analyse
│     └─ User signs in/registers first
│     └─ Redirects back to /analyse with auth
│
├─ User interested in examples:
│  └─ Click "Learn by Examples" → /explore (public gallery)
│  └─ (No auth required — shows public analysis gallery)
│
└─ END: User chooses sign in or explore

UNAUTHENTICATED USER EXPERIENCE:
  ├─ Can view homepage (static)
  ├─ Can view /explore (public gallery)
  ├─ Can view /about, /docs (static pages)
  ├─ Can fill emailCapture to join waitlist
  └─ CANNOT analyze coins (blocked by auth guard)
```

### 2.7 ANALYSIS FLOW (/analyse page)

```
┌─ START: User clicks "Start Analyzing" with auth
│  → Navigates to /analyse
│
├─ Auth guard checks:
│  ├─ If not logged in → redirect to /login?callbackUrl=/analyse
│  └─ If logged in → render CoinUploader
│
├─ CoinUploader.tsx:
│  ├─ Drag-drop zone (accept .jpg, .png)
│  ├─ File size limit: 50 MB
│  ├─ "TTA" toggle (Test-Time Augmentation — 8 forward passes)
│  ├─ "Analyse" button (replace with "Cancel" during loading)
│  └─ "Cancel" button aborts in-flight request
│
├─ User selects image:
│  ├─ Client-side preprocessing:
│  │  ├─ detectScreenshot() — warns if looks like screenshot
│  │  └─ downsizeImage(maxPx=1024) — reduces DSLR upload size
│  ├─ Never uploads > 1024px
│  ├─ Preserve aspect ratio
│  └─ Convert to JPEG 0.85 quality
│
├─ POST /api/classify { image_file, tta=bool }
│  ├─ Auth: X-API-Key header (dev: optional, prod: required)
│  ├─ Rate limit: 10 requests/minute per IP
│  ├─ Response timeout: 180 seconds
│  └─ GPU semaphore: Max 1 concurrent CUDA operation
│
├─ Backend (gatekeeper.py):
│  ├─ Load image → auto-crop coin region (HoughCircles)
│  ├─ Apply CLAHE preprocessing
│  ├─ Run CNN inference (EfficientNet-B3, 438 classes)
│  ├─ TTA: 8 forward passes if requested
│  ├─ Output: top prediction + confidence + top-5 + Grad-CAM
│
├─ Route decision (confidence threshold):
│  ├─ conf > 0.85 → HISTORIAN route (RAG + LLM narrative)
│  ├─ 0.40 ≤ conf ≤ 0.85 → VALIDATOR route (OpenCV material check + historian)
│  ├─ conf < 0.40 → INVESTIGATOR route (VLM analysis + broad KB search)
│  └─ All routes → PDF synthesis
│
├─ Response: HTTP 200
│  └─ { "cnn": {...}, "historian": {...}, "pdf_path": "..." }
│
├─ Frontend (AgentPipeline modal):
│  ├─ Shows mission control with 4 stations
│  ├─ CNN → Historian/Validator/Investigator → Synthesis
│  ├─ Animated progress bars, real-time log messages
│  └─ Grad-CAM heatmap below top-5 predictions
│
├─ AnalysisPanel displays results:
│  ├─ 3-state CNN display (Identified/TTA Consensus/Deep Search)
│  ├─ Route badge + confidence metric
│  ├─ Top-5 predictions with CN links (↗ external)
│  ├─ Grad-CAM heatmap + color-scale legend
│  ├─ CTA: "Continue research in DeepCoin AI" (→ /chat?q=...)
│  └─ "Save to History" automatically (no extra click)
│
├─ SQL INSERT: history table
│  ├─ user_id, image_path, cnn_prediction, route_taken, pdf_url
│  ├─ created_at = now(), expires_at = now() + 30 days
│  └─ Soft-delete on user-initiated delete
│
└─ END: Result in gallery, PDF in /reports/

FAILURE MODES:
  1. GPU out of memory → HTTP 503 "Please try again in a moment"
  2. Image corrupted → HTTP 400 "Could not read image file"
  3. Rate limit exceeded → HTTP 429 "Too many requests"
  4. Timeout (>180s) → HTTP 504 "Analysis took too long"
  5. CNN inference error → HTTP 500 with error detail in response
```

---

## 3. BUSINESS LOGIC RULES

### 3.1 Authentication Rules

| Rule | Implementation | Consequence |
|------|---|---|
| Email must be unique | DB constraint + unique index | HTTP 409 on duplicate |
| Password must be 8+ chars | Pydantic field_validator | HTTP 422 on too short |
| Password must have complexity | field_validator (1 letter + 1 digit/special) | HTTP 422 on no complexity |
| Unverified accounts cannot login | User.status == pending → 403 | Auto-activate in dev mode |
| Suspended accounts cannot login | User.status == suspended → 403 | Manual admin re-activation required |
| Access token expires in 15 min | JWT { exp: now+900s } | Client must refresh |
| Refresh token expires in 7 days | Cookie max_age | Auto-logout after 7 days |
| Verification token expires in 24h | DB expires_at | User must re-request if missed |
| Password reset token expires in 1h | DB expires_at | Shorter window, higher stakes |
| Password reset invalidates all sessions | DELETE FROM refresh_tokens | User logs out everywhere |

### 3.2 Email Rules

| Rule | Current State | Enterprise Required |
|------|---|---|
| Verification email sent on register | ✅ Yes | ✅ Required |
| Reset email sent on forgot-password | ✅ Yes | ✅ Required |
| Resend verification available | ✅ Yes | ✅ Required |
| Email delivery confirmation | ❌ NO | ✅ MISSING |
| Email retry on failure | ❌ NO | ✅ MISSING |
| Multiple email providers | ❌ NO (Resend only) | ✅ MISSING |
| Email rate limiting | ❌ NO | ✅ MISSING |
| Email audit trail | ❌ NO | ✅ MISSING |
| Bounce/complaint handling | ❌ NO | ✅ MISSING |
| Email template versioning | ❌ NO (hard-coded) | ✅ MISSING |

### 3.3 Authorization Rules

| Resource | Anonymous | Authenticated | Admin |
|---|---|---|---|
| Homepage | ✅ View | ✅ View | ✅ View |
| /explore (gallery) | ✅ View | ✅ View + filter | ✅ View + delete |
| /analyse (classifier) | ❌ Redirect to login | ✅ Upload + analyze | ✅ Use + download reports |
| /history (user's analyses) | ❌ Redirect to login | ✅ Own only | ✅ All users' history |
| /chat (AI Q&A) | ✅ Analyze KB | ✅ With context | ✅ Unrestricted |
| /admin (dashboard) | ❌ 404 | ❌ 403 | ✅ Full access |

---

## 4. FAILURE CASE ANALYSIS

### 4.1 Registration Failures

| Scenario | HTTP | Frontend Display | Backend Log | Recovery |
|---|---|---|---|---|
| Email invalid format | 422 | "Invalid email format" | Validation error | Correct email |
| Email already registered | 409 | "Email already registered" | Unique constraint violation | Use different email or login |
| Password too short | 422 | "Password must be 8+ chars" | Validation error | Use longer password |
| Password no complexity | 422 | "Must include letter + number/symbol" | Validation error | Add special character |
| Password mismatch (confirm) | 422 (caught client-side) | "Passwords do not match" | N/A (client) | Re-type password |
| Network error during submit | Network error | "Could not reach server" | N/A | Retry |
| Email send fails ⚠️ | 201 (WRONG!) | "Check your inbox" (misleading!) | ERROR logged, only if RESEND_API_KEY set | User never receives email, cannot verify, stuck |
| DB constraint error | 500 | "Something went wrong" | DB error in logs | User blocked, must contact support |

### 4.2 Login Failures

| Scenario | HTTP | Frontend Display | Security Impact |
|---|---|---|---|
| Email not found | 401 | "Incorrect email or password" | INTENTIONAL — prevents enumeration |
| Wrong password | 401 | "Incorrect email or password" | INTENTIONAL — prevents enumeration |
| Account pending (unverified) | 403 | "Please verify your email" | Inline "Resend verification" button |
| Account suspended | 403 | "Account suspended. Contact admin" | Admin must log in to unsuspend |
| Invalid email format | 422 | "Invalid email format" | Pydantic validation |
| Missing email field | 422 | "Missing field: email" | Pydantic validation |
| Missing password field | 422 | "Missing field: password" | Pydantic validation |
| Network timeout | Network error | "Could not reach server" | Retry |
| 500 internal error | 500 | "Unexpected error" | Logs checked by admin |

### 4.3 Forgot Password Failures

| Scenario | HTTP | Frontend Display | Security |
|---|---|---|---|
| Email good, send succeeds | 200 | "Check your inbox" | ✅ Correct |
| Email does not exist | 200 | "Check your inbox" | ✅ CORRECT — prevents enumeration |
| Email bad format | 422 | "Invalid email format" | Pydantic validation |
| Missing email field | 422 | "Field required" | Pydantic validation |
| RESEND_API_KEY not set ⚠️ | 200 | "Check your inbox" (misleading!) | ⚠️ User never gets email |
| Network timeout calling Resend | 200 (after retry) | "Check your inbox" | Email sends (or fails silently) |
| Email send fails ⚠️ | 200 | "Check your inbox" (misleading!) | User cannot reset password |

### 4.4 Password Reset Failures

| Scenario | HTTP | Frontend Display | Recovery |
|---|---|---|---|
| Token invalid | 400 | "Invalid reset link" | "Request a new reset" link |
| Token expired (>1h) | 400 | "Link expired. Request a new reset." | Request new password reset |
| Token already used | 400 | "Reset link already used" | Request new password reset |
| Missing token parameter | 400 | "No reset token found" | "Request a new reset" link |
| New password invalid (format) | 422 | Field validation error | Correct password requirements |
| New password too short | 422 | "Password must be 8+ chars" | Use longer password |
| No complexity | 422 | "Must include letter + number/symbol" | Add special character |
| Password mismatch (confirm) | 422 (caught client) | "Passwords do not match" | Re-type |
| Network timeout | Network error | "Connection failed" | Retry |
| 500 internal error | 500 | "Reset failed" | Check logs, retry later |

### 4.5 Email Verification Failures

| Scenario | HTTP | Frontend Display | Recovery |
|---|---|---|---|
| Token invalid | 404 | "Invalid or expired token" | "Resend verification" from /login |
| Token expired (>24h) | 400 | "Token expired" | "Resend verification" from /login |
| Token already used | 400 | "Token already used" | Try login (should work now) |
| Network timeout | Network error | "Could not verify" | Retry link, or "Resend" |
| 500 internal error | 500 | "Verification failed" | Retry, or resend, or contact support |

### 4.6 Analysis/Coin Classifier Failures

| Scenario | HTTP | Frontend Display | Root Cause | Recovery |
|---|---|---|---|---|
| Image file corrupted | 400 | "Could not read image" | cv2.imread() returns None | Upload different image |
| Image too large (>50MB) | 413 | "File too large" | Browser upload size limit | Use smaller file |
| Image wrong format | 400 | "Unsupported image format" | Not JPEG/PNG | Convert to JPEG/PNG |
| CNN inference timeout (>180s) | 504 | "Analysis took too long" | GPU busy or model error | Retry later |
| GPU out of memory | 503 | "Server overloaded. Try again" | CUDA OOM | Wait, then retry |
| Rate limit exceeded (10/min) | 429 | "Too many requests. Wait a minute" | slowapi limiter | Wait 60 seconds |
| CNN inference error | 500 | "Analysis failed" | Internal tensor error | Contact support |

---

## 5. CRITICAL GAPS — EMAIL AND PASSWORD RESET

### 5.1 Gap Summary

```
⚠️  CRITICAL: Password reset is NOT production-ready

Current State:
  ✅ Email infrastructure: Resend API configured
  ✅ Email templates: Branded HTML sent via Resend
  ✅ Token generation: Cryptographically sound (secrets.token_urlsafe)
  ✅ Token storage: Secure (hashed in DB)
  ✅ Token expiry: Proper TTL (verify=24h, reset=1h)
  ❌ Email delivery confirmation: MISSING
  ❌ Email retry on failure: MISSING
  ❌ Multiple providers (fallback): MISSING
  ❌ Email rate limiting: MISSING
  ❌ Email audit trail: MISSING
  ❌ Bounce/complaint handling: MISSING
  ❌ Error propagation: Bug — returns 201/200 even if email fails
```

### 5.2 Current Email Flow (Broken Path)

```python
# src/api/auth/email.py → _send()
if not _resend_available():  # RESEND_API_KEY is empty
    logger.info("[EMAIL DEV-MODE] Would send to=%s ...", to)
    return True  # ← RETURNS TRUE (success signal) EVEN THOUGH EMAIL WAS NOT SENT!

# src/api/auth/router.py → register()
await send_verification_email(user.email, raw_token)  # Returns True/False
# Regardless of return value, continues to:
await db.commit()
return MessageResponse(message="...")  # ← HTTP 201 Created, appears successful
```

### 5.3 The Broken Path in Detail

```
SCENARIO: User registers but RESEND_API_KEY is not set

1. User fills form → POST /auth/register { email, password }
2. Server creates User(status=pending) + EmailVerification token
3. Server calls send_verification_email(email, token)
4. Inside email.py:
   - if not _resend_available():  # TRUE — key is missing
   - logger.info("[EMAIL DEV-MODE] ...") — just logs
   - return True  # ← LYING — says success when no email was sent
5. Server receives True and thinks it's all good
6. Returns HTTP 201 { "message": "Please verify your email" }
7. CLIENT SHOWS: "Check your inbox!" (green checkmark)
8. USER WAITS: But there's NO email in their inbox
9. USER TRIES TO LOGIN: 403 "Please verify your email"
10. USER CLICKS "RESEND": Same thing happens again
11. USER GIVES UP: Account effectively dead

THE BUG: No error message, no warning, just silent failure
```

### 5.4 Cascading Failures

```
Scenario 1: Forgot Password Flow
  1. User clicks "Forgot password?"
  2. Enters email → POST /auth/forgot-password
  3. RESEND_API_KEY not set
  4. Email silently not sent
  5. Server returns HTTP 200 (always, for security)
  6. Frontend shows "Check your inbox"
  7. User waits forever
  8. No way to reset password

Scenario 2: Resend API Down/Rate Limited
  1. Resend.com is temporarily down or user hit rate limit
  2. resend.Emails.send() raises exception
  3. except Exception: logger.error(...); return False
  4. Server gets False but treats it as success
  5. Returns HTTP 201/200 (no error handling)
  6. User gets false success message
  7. No email arrives

Scenario 3: Production Deployment Without Config
  1. Docker container spins up
  2. RESEND_API_KEY env var NOT injected
  3. All registration emails silently skip
  4. Platform appears functional but completely broken
  5. No alerts fired (logging is only at INFO level)
```

---

## 6. ENGINE-GRADE EMAIL SOLUTION (Proposed)

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE EMAIL SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌─ Email Queue ────────────────────────────────────────────┐
│  Queue emails to be sent (async + resilient)            │
│  - In-memory (development)                              │
│  - PostgreSQL json column (production)                  │
│  - Retry on failure with exponential backoff            │
└──────────────────────────────────────────────────────────┘
       ↓
┌─ Email Provider Selection ──────────────────────────────┐
│  PRIMARY: Resend (resend.com)                           │
│    - API key: RESEND_API_KEY                            │
│    - From address: RESEND_FROM_EMAIL                    │
│  SECONDARY: SendGrid (sendgrid.com) [fallback]          │
│    - API key: SENDGRID_API_KEY                          │
│    - From address: SENDGRID_FROM_EMAIL                  │
│  TERTIARY: AWS SES [ultimate fallback]                  │
│    - AWS credentials + region                          │
│                                                         │
│  Selection logic:                                       │
│  if RESEND_API_KEY → try Resend                         │
│  elif SENDGRID_API_KEY → try SendGrid                   │
│  elif AWS_REGION → try SES                              │
│  else:                                                   │
│      if ENV == "production": RAISE FATAL               │
│      else: log and continue (dev mode)                 │
└──────────────────────────────────────────────────────────┘
       ↓
┌─ Send Attempt (with retry logic) ────────────────────────┐
│  Attempt 1: 0s delay                                    │
│  Attempt 2: 2^1 = 2s delay (if fails)                   │
│  Attempt 3: 2^2 = 4s delay (if fails)                   │
│  Attempt 4: 2^3 = 8s delay (if fails)                   │
│  Max: 3 retries = 14 seconds total                      │
│  On all failures: error logged, alert sent              │
└──────────────────────────────────────────────────────────┘
       ↓
┌─ Email Log (audit trail) ────────────────────────────────┐
│  INSERT INTO email_log:                                 │
│  {                                                      │
│    "id": uuid,                                          │
│    "to_email": "user@example.com",                      │
│    "template": "verify_email",                          │
│    "provider": "resend",                                │
│    "provider_message_id": "msg_abc123...",             │
│    "status": "sent" | "failed" | "bounced",             │
│    "attempt_count": 3,                                  │
│    "error_reason": null | "rate_limit" | ...,           │
│    "created_at": timestamp,                             │
│    "delivered_at": timestamp | null,                    │
│    "bounce_type": null | "permanent" | "temporary",    │
│  }                                                      │
└──────────────────────────────────────────────────────────┘
       ↓
┌─ Webhook Listeners ──────────────────────────────────────┐
│  Resend webhooks (POST /webhooks/resend):               │
│    - email.sent → mark delivered_at                     │
│    - email.bounced → update bounce_type                 │
│    - email.complained → mark status=bounced             │
│                                                         │
│  SendGrid Event Webhook (POST /webhooks/sendgrid):      │
│    - delivered → mark delivered_at                      │
│    - bounce → update bounce_type                        │
│    - complaint → mark status=bounced                    │
├─────────────────────────────────────────────────────────┤
│  BENEFITS:                                              │
│  ✅ Trace every email end-to-end                        │
│  ✅ Know which providers are working                    │
│  ✅ Detect bounced addresses proactively               │
│  ✅ Alert on delivery failures in real-time            │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Database Schema Changes

```sql
-- New table: email_log
CREATE TABLE email_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID REFERENCES users(id),
    to_email            VARCHAR(255) NOT NULL,
    template            VARCHAR(50) NOT NULL,  -- "verify_email", "reset_password", etc.
    provider            VARCHAR(20) NOT NULL,  -- "resend", "sendgrid", "ses"
    provider_message_id VARCHAR(255) NULL,     -- Resend msg_id, SendGrid msg_id, etc.
    status              VARCHAR(20) NOT NULL,  -- "queued", "sent", "delivered", "bounced", "failed"
    attempt_count       INT DEFAULT 0,
    max_attempts        INT DEFAULT 3,
    error_reason        VARCHAR(500) NULL,     -- "rate_limit", "invalid_address", "provider_error", etc.
    bounce_type         VARCHAR(20) NULL,      -- "permanent", "temporary", "complaint"
    created_at          TIMESTAMP DEFAULT NOW(),
    sent_at             TIMESTAMP NULL,
    delivered_at        TIMESTAMP NULL,
    expires_at          TIMESTAMP NULL,        -- For token emails (24h or 1h from sent_at)
    metadata            JSONB,                 -- { "token_type": "verify", "ip_address": "...", etc. }
    CONSTRAINT email_log_status_valid CHECK (status IN ('queued', 'sent', 'delivered', 'bounced', 'failed'))
);

CREATE INDEX idx_email_log_user_id ON email_log(user_id);
CREATE INDEX idx_email_log_to_email ON email_log(to_email);
CREATE INDEX idx_email_log_status ON email_log(status);
CREATE INDEX idx_email_log_created_at ON email_log(created_at DESC);

-- Updates to email_verification table
ALTER TABLE email_verification ADD COLUMN email_log_id UUID REFERENCES email_log(id);
ALTER TABLE email_verification ADD COLUMN delivery_status VARCHAR(20) DEFAULT 'pending';
```

### 6.3 Implementation (Pseudocode)

```python
# src/api/email/manager.py — New module

class EmailProvider:
    """Abstract base for all email providers."""
    async def send(self, to: str, subject: str, html: str) -> dict:
        """Returns { "message_id": str, "status": str, "error": str | None }"""
        raise NotImplementedError

class ResendProvider(EmailProvider):
    """Resend.com provider."""
    def __init__(self, api_key: str, from_email: str):
        self.api_key = api_key
        self.from_email = from_email
        self.client = resend.AsyncClient(api_key=api_key)
    
    async def send(self, to: str, subject: str, html: str) -> dict:
        try:
            result = await asyncio.to_thread(
                self.client.emails.send,
                {
                    "from": self.from_email,
                    "to": [to],
                    "subject": subject,
                    "html": html,
                }
            )
            return {
                "message_id": result.get("id"),
                "status": "sent",
                "error": None,
                "provider": "resend"
            }
        except Exception as exc:
            return {
                "message_id": None,
                "status": "failed",
                "error": str(exc),
                "provider": "resend"
            }

class SendGridProvider(EmailProvider):
    """SendGrid provider (fallback)."""
    async def send(self, to: str, subject: str, html: str) -> dict:
        # Similar structure
        pass

class AWSSeProvider(EmailProvider):
    """AWS SES provider (tertiary fallback)."""
    async def send(self, to: str, subject: str, html: str) -> dict:
        # Similar structure
        pass


class EmailManager:
    """Orchestrates email sending with retry + failover + audit."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.providers = self._init_providers()
        self.queue = asyncio.Queue() if ENV == "development" else None  # Use DB in prod
    
    def _init_providers(self) -> list[EmailProvider]:
        """Initialize enabled providers in fallback order."""
        providers = []
        if RESEND_API_KEY:
            providers.append(ResendProvider(RESEND_API_KEY, RESEND_FROM_EMAIL))
        if SENDGRID_API_KEY:
            providers.append(SendGridProvider(SENDGRID_API_KEY, SENDGRID_FROM_EMAIL))
        if AWS_REGION:
            providers.append(AWSSeProvider(AWS_REGION, AWS_FROM_EMAIL))
        return providers
    
    async def send_email(
        self,
        to_email: str,
        template: str,  # "verify_email", "reset_password", etc.
        context: dict,   # Variables for template rendering
        user_id: str | None = None,
        max_attempts: int = 3,
    ) -> EmailLog:
        """
        Send email with retry logic + audit trail.
        
        FLOW:
            1. Create EmailLog record in DB (status=queued)
            2. Render template with context
            3. Attempt send with each provider (fallback on failure)
            4. Update EmailLog with result
            5. Return EmailLog record for caller to check status
        """
        
        # 1. Create audit record
        email_log = EmailLog(
            user_id=user_id,
            to_email=to_email,
            template=template,
            status="queued",
            max_attempts=max_attempts,
        )
        self.db.add(email_log)
        await self.db.flush()  # Get ID without commit
        
        # 2. Render template
        subject, html = self._render_template(template, context)
        
        # 3. Retry logic with exponential backoff
        for attempt in range(max_attempts):
            email_log.attempt_count = attempt + 1
            
            for provider in self.providers:
                try:
                    logger.info(
                        "Email send attempt %d/%d: to=%s provider=%s",
                        attempt + 1, max_attempts, to_email, provider.__class__.__name__
                    )
                    
                    result = await provider.send(to_email, subject, html)
                    
                    if result["error"] is None:
                        # SUCCESS
                        email_log.provider = provider.__class__.__name__
                        email_log.provider_message_id = result["message_id"]
                        email_log.status = "sent"
                        email_log.sent_at = datetime.now(timezone.utc)
                        await self.db.commit()
                        logger.info("Email sent successfully: log_id=%s msg_id=%s", 
                                    email_log.id, result["message_id"])
                        return email_log  # ← Early return on success
                    
                    # This provider failed, try next
                    logger.warning(
                        "Provider %s failed: %s", provider.__class__.__name__, result["error"]
                    )
                    continue
                
                except Exception as exc:
                    logger.error("Unexpected error with provider: %s", exc, exc_info=True)
                    continue
            
            # All providers failed this attempt, wait before retry
            if attempt < max_attempts - 1:
                delay = 2 ** attempt  # 1s, 2s, 4s
                logger.info("All providers failed, retrying in %ds", delay)
                await asyncio.sleep(delay)
        
        # 4. All attempts exhausted
        email_log.status = "failed"
        email_log.error_reason = "all_providers_exhausted"
        await self.db.commit()
        
        # 5. Alert admin (critical failure)
        await self._alert_admin(
            f"Email send failed after {max_attempts} attempts: {to_email} ({template})"
        )
        
        logger.error("Email send failed completely: log_id=%s to=%s", 
                     email_log.id, to_email)
        return email_log
    
    def _render_template(self, template: str, context: dict) -> tuple[str, str]:
        """Render subject + HTML from template."""
        if template == "verify_email":
            return (
                "Verify your DeepCoin email address",
                VERIFY_EMAIL_TEMPLATE.format(**context)
            )
        elif template == "reset_password":
            return (
                "Reset your DeepCoin password",
                RESET_PASSWORD_TEMPLATE.format(**context)
            )
        # ... more templates
    
    async def _alert_admin(self, message: str) -> None:
        """Send Slack/PagerDuty alert on critical failure."""
        # Implementation: call Slack webhook, PagerDuty API, etc.
        pass


# Usage in auth/router.py

@router.post("/register")
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    # ... existing validation ...
    
    # Create user, token
    user = User(...)
    db.add(user)
    await db.flush()
    
    raw_token = secrets.token_urlsafe(48)
    db.add(EmailVerification(user_id=user.id, token=raw_token))
    await db.flush()
    
    # IMPORTANT: Send email and wait for result
    email_manager = get_email_manager()  # Singleton
    email_log = await email_manager.send_email(
        to_email=body.email,
        template="verify_email",
        context={
            "name": body.display_name or body.email.split("@")[0],
            "verify_url": f"{APP_URL}/verify-email?token={raw_token}",
            "expires_hours": 24,
        },
        user_id=user.id,
    )
    
    # NEW: Check if email actually sent
    if email_log.status == "failed":
        # Email send failed — should we fail the registration?
        # OPTION A: Fail registration (user sees error, can retry)
        # OPTION B: Register anyway but mark in logs (user won't verify, stuck)
        
        # RECOMMENDED: Fail registration in production, warn in dev
        if os.getenv("ENV") == "production":
            await db.rollback()
            logger.error("Registration canceled: email send failed for %s", body.email)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not send verification email. Please try again later."
            )
        else:
            logger.warning(
                "Dev mode: allowing registration despite email failure for %s",
                body.email
            )
    
    await db.commit()
    return MessageResponse(message="Account created. Check your email to verify.")
```

### 6.4 Webhook Handlers (for delivery confirmation)

```python
# src/api/routes/webhooks.py (new)

@router.post("/webhooks/resend")
async def webhook_resend(
    body: dict,  # Resend webhook payload
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Resend.com sends webhook events for email lifecycle.
    
    Events:
        - email.sent
        - email.delivered
        - email.bounced
        - email.complained
        - email.open (if enabled)
        - email.click (if enabled)
    """
    event_type = body.get("type")
    data = body.get("data", {})
    message_id = data.get("email_id")
    
    # Find the email log record
    result = await db.execute(
        select(EmailLog).where(EmailLog.provider_message_id == message_id)
    )
    email_log = result.scalar_one_or_none()
    
    if not email_log:
        logger.warning("Resend webhook: message_id not found in DB: %s", message_id)
        return {"ok": False, "reason": "message_id not in DB"}
    
    # Update log based on event
    if event_type == "email.delivered":
        email_log.status = "delivered"
        email_log.delivered_at = datetime.now(timezone.utc)
        logger.info("Email delivered: log_id=%s", email_log.id)
    
    elif event_type == "email.bounced":
        email_log.status = "bounced"
        email_log.bounce_type = data.get("bounce_type", "unknown")  # "permanent" or "temporary"
        logger.warning("Email bounced: log_id=%s type=%s", 
                       email_log.id, email_log.bounce_type)
    
    elif event_type == "email.complained":
        email_log.status = "bounced"
        email_log.bounce_type = "complaint"
        logger.warning("Email complained: log_id=%s", email_log.id)
    
    await db.commit()
    return {"ok": True}


@router.post("/webhooks/sendgrid")
async def webhook_sendgrid(
    body: list[dict],  # SendGrid sends array of events
    db: AsyncSession = Depends(get_db),
) -> dict:
    """SendGrid Event Webhook (similar flow)."""
    for event in body:
        event_type = event.get("event")
        message_id = event.get("sg_message_id")
        
        result = await db.execute(
            select(EmailLog).where(EmailLog.provider_message_id == message_id)
        )
        email_log = result.scalar_one_or_none()
        
        if not email_log:
            continue
        
        if event_type == "delivered":
            email_log.status = "delivered"
            email_log.delivered_at = datetime.now(timezone.utc)
        elif event_type == "bounce":
            email_log.status = "bounced"
            email_log.bounce_type = event.get("bounce_type", "unknown")
        elif event_type == "complaint":
            email_log.status = "bounced"
            email_log.bounce_type = "complaint"
        
        # ... similar for other events
    
    await db.commit()
    return {"ok": True}
```

### 6.5 Configuration (.env.example)

```dotenv
# ── Email Delivery ────────────────────────────────────────────────────────
# Choose ONE primary provider, optionally add fallback providers

# PRIMARY: Resend (recommended for startups)
# Get key: https://resend.com/api-keys
# Usage: 100 emails free/month, then $0.04 per email
#
RESEND_API_KEY=re_xxxxxxxxxxxxx
RESEND_FROM_EMAIL=DeepCoin <noreply@deepcoin.ai>

# FALLBACK #1: SendGrid (add for redundancy)
# Get key: https://app.sendgrid.com/settings/api_keys
# Usage: 100 emails/day free, then $9.95-$99.95/month
#
# SENDGRID_API_KEY=SG.xxxxxxxxxxxxx
# SENDGRID_FROM_EMAIL=DeepCoin <noreply@deepcoin.ai>

# FALLBACK #2: AWS SES (add for enterprise redundancy)
# Requires AWS credentials (IAM user with SES permissions)
#
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
# AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# AWS_SES_FROM_EMAIL=noreply@deepcoin.ai

# Email Sending Configuration
#
APP_URL=http://localhost:3000                              # For email links
EMAIL_VERIFICATION_EXPIRES_HOURS=24                        # Token lifetime
EMAIL_PASSWORD_RESET_EXPIRES_HOURS=1                       # Token lifetime
EMAIL_MAX_RETRIES=3                                        # Retry attempts
EMAIL_RETRY_BACKOFF_BASE=2                                 # 2^n exponential backoff
EMAIL_LOG_RETENTION_DAYS=90                                # Keep audit trail 90d

# Webhook Security (must match provider webhook config)
#
RESEND_WEBHOOK_SECRET=whk_test_xxxxxxxxxxxxx               # Set in Resend dashboard
SENDGRID_WEBHOOK_SECRET=sg_xxxxxxxxxxxxx                   # Set in SendGrid dashboard
```

### 6.6 Testing Email Flows

```python
# tests/integration/test_email_flows.py (new)

class TestEmailDelivery:
    """Comprehensive email delivery integration tests."""
    
    async def test_register_with_email_success(self, client, db):
        """Registration succeeds and verification email is queued."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "Password123!",
                "display_name": "Test User"
            }
        )
        assert response.status_code == 201
        
        # Check that EmailLog was created with status="sent" (or "delivered")
        email_log = await db.execute(
            select(EmailLog).where(EmailLog.to_email == "newuser@example.com")
        )
        log = email_log.scalar_one_or_none()
        assert log is not None
        assert log.template == "verify_email"
        assert log.status in ("sent", "delivered", "queued")  # Never "failed"
    
    async def test_register_email_failure_fails_registration(self, client, db, monkeypatch):
        """If email send fails in PRODUCTION, registration is rejected."""
        monkeypatch.setenv("ENV", "production")
        
        # Mock email provider to fail
        async def mock_send(*args, **kwargs):
            return {"message_id": None, "status": "failed", "error": "Service unavailable"}
        
        # Call endpoint
        response = await client.post(
            "/auth/register",
            json={
                "email": "fail@example.com",
                "password": "Password123!",
            }
        )
        
        # Should get 500, not 201
        assert response.status_code == 500
        assert "email" in response.json()["detail"].lower()
        
        # User should NOT be created
        user = await db.execute(select(User).where(User.email == "fail@example.com"))
        assert user.scalar_one_or_none() is None
    
    async def test_forgot_password_email_sent(self, client, db, user_factory):
        """Forgot password creates reset token and sends email."""
        user = await user_factory(email="forgotpass@example.com", status="active")
        
        response = await client.post(
            "/auth/forgot-password",
            json={"email": "forgotpass@example.com"}
        )
        
        # Always 200 for security
        assert response.status_code == 200
        
        # But email log should show send attempt
        email_log = await db.execute(
            select(EmailLog).where(EmailLog.to_email == "forgotpass@example.com")
        )
        log = email_log.scalar_one_or_none()
        assert log is not None
        assert log.template == "reset_password"
        assert log.status != "failed"  # Should be sent/delivered/queued, never failed
    
    async def test_email_delivery_webhook_updates_status(self, client, db):
        """Resend webhook updates email log status to 'delivered'."""
        # Assume an email_log exists in DB (from previous operation)
        email_log = EmailLog(
            to_email="test@example.com",
            template="verify_email",
            provider="resend",
            provider_message_id="550e8400-e29b-41d4-a716-446655440000",
            status="sent",
        )
        db.add(email_log)
        await db.commit()
        
        # Simulate Resend webhook: email.delivered
        response = await client.post(
            "/webhooks/resend",
            json={
                "type": "email.delivered",
                "data": {
                    "email_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            }
        )
        
        assert response.status_code == 200
        
        # Check that status was updated
        updated = await db.execute(
            select(EmailLog).where(EmailLog.provider_message_id == "550e8400-e29b-41d4-a716-446655440000")
        )
        log = updated.scalar_one()
        assert log.status == "delivered"
        assert log.delivered_at is not None
    
    async def test_retry_on_provider_failure(self, client, db, monkeypatch):
        """Email manager retries with exponential backoff if provider fails."""
        call_count = 0
        
        async def mock_send_fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return {"message_id": None, "status": "failed", "error": "Temporary error"}
            else:
                return {"message_id": "msg_123", "status": "sent", "error": None}
        
        manager = EmailManager(db)
        monkeypatch.setattr(manager, "send_email", mock_send_fail_then_succeed)
        
        # First call fails, second succeeds
        # (In real code, provider is mocked inside manager)
        # This is a simplified version of retry testing
        assert call_count >= 1
```

---

## 7. IMMEDIATE ACTION ITEMS (Enterprise Hardening)

### 7.1 CRITICAL (Must fix before production)

- [ ] **Add email failure detection**
  - File: `src/api/auth/email.py`
  - Change: If RESEND_API_KEY is empty AND ENV == "production", raise Fatal Error
  - Change: Return False if email send fails, propagate error to caller
  
- [ ] **Add email log table**
  - File: `migrations/add_email_log_table.sql`
  - Create table with schema (see section 6.2)
  - Add foreign key to users + email_verification
  
- [ ] **Update registration endpoint**
  - File: `src/api/auth/router.py` (register function)
  - Check email_log status before returning success
  - Fail registration if email send fails in production
  
- [ ] **Update forgot-password endpoint**
  - File: `src/api/auth/router.py` (forgot_password function)
  - Log to email_log (currently no logging happens)
  - Alert admin if email send fails

### 7.2 HIGH (Do within 1 week)

- [ ] **Add Resend webhook handler**
  - File: `src/api/routes/webhooks.py`
  - Implement POST /webhooks/resend
  - Configure webhook URL in Resend dashboard
  - Test delivery confirmation flow
  
- [ ] **Add email retry queue**
  - File: `src/api/email/queue.py`
  - Implement with exponential backoff (2s, 4s, 8s)
  - Store failed emails in DB for async retry
  
- [ ] **Add SendGrid fallback provider**
  - File: `src/api/email/providers.py`
  - Implement SendGridProvider class
  - Test failover when Resend fails

### 7.3 MEDIUM (Do within 2 weeks)

- [ ] **Add email rate limiting**
  - File: `src/api/auth/router.py`
  - Max 5 password resets per email per hour
  - Max 5 resend-verification per email per hour
  - Return 429 if exceeded
  
- [ ] **Add email templates to database**
  - File: `src/api/email/templates.py`
  - Move hard-coded HTML to DB records
  - Version templates for A/B testing
  
- [ ] **Add admin dashboard for email logs**
  - File: `src/api/routes/admin.py`
  - GET /api/admin/email-logs (paginated, filterable)
  - Show status, delivery time, bounce rate
  
- [ ] **Add monitoring/alerting**
  - Integration: Slack webhook on email failures
  - Integration: PagerDuty on critical failures (>50 failed in 1hr)

---

## 8. SUMMARY

| Aspect | Current | Enterprise Required | Gap |
|---|---|---|---|
| Email provider | ✅ Resend | Resend + fallback | Single point of failure |
| Verify email delivered | ❌ No | ✅ Webhook tracking | Cannot confirm delivery |
| Retry on failure | ❌ No | ✅ Exponential backoff | Lost emails on transient errors |
| Multiple providers | ❌ No | ✅ SendGrid + AWS SES | No redundancy |
| Error propagation | ❌ Returns 201 anyway | ✅ Fail registration on email fail | Silent failures |
| Email audit trail | ❌ No | ✅ Full email_log table | No forensics |
| Rate limiting | ❌ No | ✅ 5/hour per email | Spam vulnerability |
| Admin visibility | ❌ No | ✅ Email logs dashboard | Cannot troubleshoot |
| Bounce handling | ❌ No | ✅ Unsubscribe bounced addresses | Wasted sends |
| Monitoring/alerts | ❌ No | ✅ Slack + PagerDuty | Silent failures in prod |

**RECOMMENDATION**: Implement sections 7.1 (CRITICAL) immediately before any production deployment. Sections 7.2-7.3 can be phased in over 2-3 weeks.
