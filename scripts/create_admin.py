#!/usr/bin/env python3
"""Create an admin user in the Postgres DB.

Usage (host or inside Docker):
  python scripts/create_admin.py --email admin@example.com --password S3curePass! --display "Admin"

When run inside Docker compose, use:
  docker compose run --rm api python scripts/create_admin.py --email admin@example.com --password S3curePass! --display "Admin"
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from src.api.db.models import User, UserRole, UserStatus
from src.api.db.session import AsyncSessionLocal
from src.api.auth.utils import hash_password


async def main(email: str, password: str, display_name: str | None) -> None:
    async with AsyncSessionLocal() as session:
        # check existing
        result = await session.execute(select(User).where(User.email == email))
        existing = result.scalar_one_or_none()
        if existing:
            print(f"User with email {email} already exists: id={existing.id} role={existing.role}")
            return

        hashed = hash_password(password)
        user = User(
            email=email,
            hashed_password=hashed,
            display_name=display_name,
            role=UserRole.admin,
            status=UserStatus.active,
            email_verified_at=datetime.now(timezone.utc),
        )
        session.add(user)
        await session.commit()
        print(f"Created admin user {email}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--display", default="Administrator")
    args = parser.parse_args()
    asyncio.run(main(args.email, args.password, args.display))
