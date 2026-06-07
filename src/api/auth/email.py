"""
src/api/auth/email.py
=====================
Email delivery for authentication flows (verification + password reset).

PROVIDER: SMTP
    - Simple REST API, generous free tier (3,000 emails/month)
    - Python SDK: `smtp` package
    - Requires SMTP_USER and SMTP_USER environment variables

GRACEFUL DEGRADATION:
    If SMTP_USER is not set (dev/CI environment), email sending is
    SKIPPED and the token is PRINTED to the log at INFO level.
    This means a fresh clone needs zero email configuration to test registration.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import smtplib
from email.message import EmailMessage

from sqlalchemy.ext.asyncio import AsyncSession
from src.api.db.models import EmailLog

logger = logging.getLogger(__name__)

_SENDER_EMAIL = os.getenv("DEEPCOIN_SENDER_EMAIL", "DeepCoin <noreply@deepcoin.ai>")
_APP_URL = os.getenv("APP_URL", "http://localhost:3000")
_PASSWORD_RESET_EXPIRE_HOURS = int(os.getenv("PASSWORD_RESET_EXPIRE_HOURS", "1"))


def _smtp_user() -> str:
  return os.getenv("SMTP_USER", "").strip()


def _smtp_password() -> str:
  return os.getenv("SMTP_PASSWORD", "").strip()


def _send_operator_copy() -> bool:
    return os.getenv("AUTH_SEND_OPERATOR_COPY", "1") == "1"


def _operator_email() -> str:
    return os.getenv("SMTP_OPERATOR_EMAIL", _smtp_user())


def _smtp_available() -> bool:
    """Check whether the SMTP settings are configured."""
    return bool(_smtp_user() and _smtp_password())


async def _send(
    to: str,
    subject: str,
    html: str,
    db: AsyncSession | None = None,
    user_id: str | None = None,
) -> bool:
    """
    Core email dispatch — wraps the synchronous SMTP wrapper in a thread.
    Returns True when the email was delivered or simulated (dev mode),
    False only when in production and sending failed.
    """

    error_msg: str | None = None
    status = "sent"

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = _smtp_user()
    smtp_password = _smtp_password()
    operator_email = _operator_email()
    send_operator = _send_operator_copy()

    env = os.getenv("ENV", "development").lower()

    if not _smtp_available():
        if env == "production":
            error_msg = "SMTP credentials are missing in production."
            logger.error(error_msg)
            status = "failed"
            return False

        # Development: simulate delivery by logging the first href in the HTML
        link_match = re.search(r'href="([^"]+)"', html)
        dev_link = link_match.group(1) if link_match else "No link found"
        logger.warning(
            "[EMAIL DEV-MODE] Simulated email to=%s subject=%s link=%s",
            to,
            subject,
            dev_link,
        )
        # Record audit entry if DB provided and return success for dev mode
        if db is not None:
            try:
                log_entry = EmailLog(
                    user_id=user_id,
                    to_email=to,
                    subject=subject,
                    status=status,
                    error_message=None,
                )
                db.add(log_entry)
                await db.flush()
            except Exception as log_exc:
                logger.error("Failed to write email log (dev-mode): %s", log_exc)
        return True

    def _blocking_send() -> None:
        # Build message
        msg = EmailMessage()
        msg["From"] = _SENDER_EMAIL
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content("Please enable HTML to view this email.")
        msg.add_alternative(html, subtype="html")

        # Deliver via SMTP_SSL
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

            # Optional operator copy
            if send_operator and operator_email and operator_email.lower() != to.lower():
                op = EmailMessage()
                op["From"] = _SENDER_EMAIL
                op["To"] = operator_email
                op["Subject"] = f"[DeepCoin Notice] {subject}"
                op.set_content(
                    "A transactional authentication email was delivered.\n"
                    f"Recipient: {to}\n"
                    f"Subject: {subject}\n"
                    "Security: token/link omitted intentionally.\n"
                )
                server.send_message(op)

    try:
        await asyncio.to_thread(_blocking_send)
        logger.info("Email sent to=%s", to)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        status = "failed"
        logger.error("Failed to send email to=%s: %s", to, exc)

        if env != "production":
            # Provide a dev-mode fallback so local development can proceed
            link_match = re.search(r'href="([^"]+)"', html)
            dev_link = link_match.group(1) if link_match else "No link found"
            logger.warning(
                "[EMAIL DEV-MODE FALLBACK] SMTP failed; simulated delivery to=%s subject=%s link=%s",
                to,
                subject,
                dev_link,
            )
            # Still record the attempt in DB when available
            if db is not None:
                try:
                    log_entry = EmailLog(
                        user_id=user_id,
                        to_email=to,
                        subject=subject,
                        status=status,
                        error_message=error_msg,
                    )
                    db.add(log_entry)
                    await db.flush()
                except Exception as log_exc:
                    logger.error("Failed to write email log (fallback): %s", log_exc)
            return True

        return False

    finally:
        # Audit log for production and dev alike
        if db is not None:
            try:
                # If we already logged above in dev paths, this will add a second record; acceptable for now.
                log_entry = EmailLog(
                    user_id=user_id,
                    to_email=to,
                    subject=subject,
                    status=status,
                    error_message=error_msg,
                )
                db.add(log_entry)
                await db.flush()
            except Exception as log_exc:
                logger.error("Failed to write to email_logs: %s", log_exc)


async def send_verification_email(to_email: str, token: str, db: AsyncSession | None = None, user_id: str | None = None) -> bool:
    verify_url = f"{_APP_URL}/verify-email?token={token}"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>DeepCoin - Verify Your Email</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; background: #f8fafc; margin: 0; padding: 20px;">
      <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        
        <!-- Header Band -->
        <div style="background: linear-gradient(135deg, #020617 0%, #0f172a 100%); padding: 40px 20px; text-align: center;">
          <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: bold;">DeepCoin</h1>
          <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 13px;">Archaeological Coin Intelligence Platform</p>
        </div>
        
        <!-- Content -->
        <div style="padding: 40px 30px;">
          <h2 style="color: #0f172a; font-size: 24px; margin: 0 0 16px 0; font-weight: bold;">Verify Your Email Address</h2>
          
          <p style="color: #334155; line-height: 1.6; margin: 0 0 16px 0; font-size: 15px;">
            Thank you for registering with DeepCoin! 
            Click the button below to activate your account. <strong style="color: #0f172a;">This link expires in 24 hours.</strong>
          </p>
          
          <!-- Action Button -->
          <div style="text-align: center; margin: 32px 0;">
            <a href="{verify_url}"
               style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 14px 48px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block; font-size: 16px; transition: transform 0.2s; border: none; cursor: pointer;">
              Verify Email Address
            </a>
          </div>
          
          <p style="color: #64748b; font-size: 13px; line-height: 1.5; margin: 0 0 16px 0;">
            Or copy and paste this link into your browser:
            <br>
            <code style="background: #f1f5f9; padding: 8px 12px; border-radius: 6px; display: block; margin-top: 8px; word-break: break-all; font-family: 'Monaco', 'Courier New', monospace; font-size: 12px; color: #0f172a;">
              {verify_url}
            </code>
          </p>
          
          <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
          
          <p style="color: #64748b; font-size: 12px; line-height: 1.5; margin: 0;">
            <strong>Did you not create this account?</strong><br>
            If you did not create a DeepCoin account, you can safely ignore this email.
            No further action is needed.
          </p>
        </div>
        
        <!-- Footer -->
        <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 20px 30px; text-align: center;">
          <p style="color: #64748b; font-size: 12px; margin: 0;">
            © 2026 <strong>DeepCoin</strong> • Archaeological Coin Intelligence<br>
            <a href="{_APP_URL}" style="color: #3b82f6; text-decoration: none;">Visit our website</a>
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    return await _send(to_email, "Verify your DeepCoin email address", html, db, user_id)


async def send_password_reset_email(to_email: str, token: str, db: AsyncSession | None = None, user_id: str | None = None) -> bool:
    reset_url = f"{_APP_URL}/reset-password?token={token}"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>DeepCoin - Reset Your Password</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; background: #f8fafc; margin: 0; padding: 20px;">
      <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        
        <!-- Header Band -->
        <div style="background: linear-gradient(135deg, #020617 0%, #0f172a 100%); padding: 40px 20px; text-align: center;">
          <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: bold;">DeepCoin</h1>
          <p style="color: #cbd5e1; margin: 8px 0 0 0; font-size: 13px;">Archaeological Coin Intelligence Platform</p>
        </div>
        
        <!-- Content -->
        <div style="padding: 40px 30px;">
          <h2 style="color: #0f172a; font-size: 24px; margin: 0 0 16px 0; font-weight: bold;">Reset Your Password</h2>
          
          <p style="color: #334155; line-height: 1.6; margin: 0 0 16px 0; font-size: 15px;">
            We received a request to reset your DeepCoin password.
            Click the button below to set a new password. <strong style="color: #0f172a;">This link expires in {_PASSWORD_RESET_EXPIRE_HOURS} hour(s).</strong>
          </p>
          
          <!-- Action Button -->
          <div style="text-align: center; margin: 32px 0;">
            <a href="{reset_url}"
               style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 14px 48px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block; font-size: 16px; transition: transform 0.2s; border: none; cursor: pointer;">
              Reset Password
            </a>
          </div>
          
          <p style="color: #64748b; font-size: 13px; line-height: 1.5; margin: 0 0 16px 0;">
            Or copy and paste this link into your browser:
            <br>
            <code style="background: #f1f5f9; padding: 8px 12px; border-radius: 6px; display: block; margin-top: 8px; word-break: break-all; font-family: 'Monaco', 'Courier New', monospace; font-size: 12px; color: #0f172a;">
              {reset_url}
            </code>
          </p>
          
          <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
          
          <p style="color: #64748b; font-size: 12px; line-height: 1.5; margin: 0;">
            <strong>Did you not request this?</strong><br>
            If you did not request a password reset, you can safely ignore this email.
            Your password will <strong>not</strong> change.<br><br>
            <strong>Security Tip:</strong> Never share this link with anyone, and never give your password to someone who asks for it via email.
          </p>
        </div>
        
        <!-- Footer -->
        <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 20px 30px; text-align: center;">
          <p style="color: #64748b; font-size: 12px; margin: 0;">
            © 2026 <strong>DeepCoin</strong> • Archaeological Coin Intelligence<br>
            <a href="{_APP_URL}" style="color: #3b82f6; text-decoration: none;">Visit our website</a>
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    return await _send(to_email, "Reset your DeepCoin password", html, db, user_id)
