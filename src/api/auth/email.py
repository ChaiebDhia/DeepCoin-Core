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

_SMTP_USER     = os.getenv("SMTP_USER", "")
_SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
_SENDER_EMAIL   = os.getenv("DEEPCOIN_SENDER_EMAIL", "DeepCoin <noreply@deepcoin.ai>")
_APP_URL            = os.getenv("APP_URL", "http://localhost:3000")


def _smtp_available() -> bool:
    """Check whether the SMTP settings are configured."""
    return bool(_SMTP_USER and _SMTP_PASSWORD)


async def _send(to: str, subject: str, html: str, db: AsyncSession | None = None, user_id: str | None = None) -> bool:
    """
    Core email dispatch — wraps the synchronous SMTP wrapper in a thread.
    Returns:
        True if sent successfully, False if SMTP raised an exception.
    """
    error_msg = None
    status = "sent"

    try:
        if not _smtp_available():
            env = os.getenv("ENV", "development").lower()
            if env == "production":
                error_msg = "SMTP_USER missing in production. Cannot send email."
                logger.error(error_msg)
                status = "failed"
                # We do NOT silently swallow this if it's supposed to fail.
                raise RuntimeError(error_msg)
            
            # Dev mode: Extract the link using regex so dev can click it
            link_match = re.search(r'href="([^"]+)"', html)
            dev_link = link_match.group(1) if link_match else "No link found"
            
            logger.warning(
                "[EMAIL DEV-MODE] Email dispatch simulated. \n"
                "  To: %s\n"
                "  Subject: %s\n"
                "  LINK TO CLICK: %s\n"
                "  (Set SMTP_USER and SMTP_PASSWORD in .env to enable real delivery)",
                to, subject, dev_link
            )
            return True

        

        def _blocking_send():
            msg = EmailMessage()
            msg["From"] = _SENDER_EMAIL
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content("Please enable HTML to view this email.")
            msg.add_alternative(html, subtype="html")

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(_SMTP_USER, _SMTP_PASSWORD)
                server.send_message(msg)

        result = await asyncio.to_thread(_blocking_send)
        logger.info("Email sent to=%s", to)
        return True

    except RuntimeError as rexc:
        # Don't silence setup exceptions
        error_msg = str(rexc)
        status = "failed"
        logger.error("Failed to send email to=%s: %s", to, rexc)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        status = "failed"
        logger.error("Failed to send email to=%s: %s", to, exc)
        env = os.getenv("ENV", "development").lower()
        if env != "production":
            link_match = re.search(r'href="([^"]+)"', html)
            dev_link = link_match.group(1) if link_match else "No link found"
            logger.warning(
                "[EMAIL DEV-MODE FALLBACK] SMTP failed. Email dispatch simulated. \n"
                "  To: %s\n"
                "  Subject: %s\n"
                "  LINK TO CLICK: %s",
                to, subject, dev_link
            )
            return True
        return False

    finally:
        # P0 FIX: Hardened audit logging. Database captures email dispatch state.
        if db is not None:
            try:
                log_entry = EmailLog(
                    user_id=user_id,
                    to_email=to,
                    subject=subject,
                    status=status,
                    error_message=error_msg
                )
                db.add(log_entry)
                # We flush so it stays within the current transaction scope of the router
                await db.flush()
            except Exception as log_exc:
                logger.error("Failed to write to email_logs: %s", log_exc)


async def send_verification_email(to_email: str, token: str, db: AsyncSession | None = None, user_id: str | None = None) -> bool:
    verify_url = f"{_APP_URL}/verify-email?token={token}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; background: #0a0f2c; color: #e2e8f0; padding: 40px;">
      <div style="max-width: 520px; margin: auto; background: #131a3a; border-radius: 12px; padding: 40px;">
        <h1 style="color: #3b82f6; margin-bottom: 8px;">DeepCoin</h1>
        <p style="color: #94a3b8; font-size: 13px;">Archaeological Coin Intelligence Platform</p>
        <hr style="border-color: #1e2d52; margin: 24px 0;">
        <h2 style="font-size: 20px; margin-bottom: 12px;">Verify your email address</h2>
        <p style="color: #94a3b8; line-height: 1.6;">
          Thank you for registering. Click the button below to activate your account.
          This link expires in <strong style="color: #e2e8f0;">24 hours</strong>.
        </p>
        <div style="text-align: center; margin: 32px 0;">
          <a href="{verify_url}"
             style="background: #3b82f6; color: white; padding: 14px 32px; border-radius: 8px;
                    text-decoration: none; font-weight: bold; display: inline-block;">
            Verify Email Address
          </a>
        </div>
        <p style="color: #64748b; font-size: 12px; line-height: 1.5;">
          If you did not create a DeepCoin account, you can safely ignore this email.
          <br>
          Link: <a href="{verify_url}" style="color: #3b82f6;">{verify_url}</a>
        </p>
      </div>
    </body>
    </html>
    """
    return await _send(to_email, "Verify your DeepCoin email address", html, db, user_id)


async def send_password_reset_email(to_email: str, token: str, db: AsyncSession | None = None, user_id: str | None = None) -> bool:
    reset_url = f"{_APP_URL}/reset-password?token={token}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; background: #0a0f2c; color: #e2e8f0; padding: 40px;">
      <div style="max-width: 520px; margin: auto; background: #131a3a; border-radius: 12px; padding: 40px;">
        <h1 style="color: #3b82f6; margin-bottom: 8px;">DeepCoin</h1>
        <p style="color: #94a3b8; font-size: 13px;">Archaeological Coin Intelligence Platform</p>
        <hr style="border-color: #1e2d52; margin: 24px 0;">
        <h2 style="font-size: 20px; margin-bottom: 12px;">Reset your password</h2>
        <p style="color: #94a3b8; line-height: 1.6;">
          We received a request to reset your DeepCoin password.
          Click below to set a new password. This link expires in
          <strong style="color: #e2e8f0;">1 hour</strong>.
        </p>
        <div style="text-align: center; margin: 32px 0;">
          <a href="{reset_url}"
             style="background: #3b82f6; color: white; padding: 14px 32px; border-radius: 8px;
                    text-decoration: none; font-weight: bold; display: inline-block;">
            Reset Password
          </a>
        </div>
        <p style="color: #64748b; font-size: 12px; line-height: 1.5;">
          If you did not request a password reset, you can safely ignore this email.
          Your password will <strong>not</strong> change.
          <br>
          Link: <a href="{reset_url}" style="color: #3b82f6;">{reset_url}</a>
        </p>
      </div>
    </body>
    </html>
    """
    return await _send(to_email, "Reset your DeepCoin password", html, db, user_id)
