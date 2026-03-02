"""
src/api/auth/__init__.py
========================
Auth package public re-exports.

Consumers should import from the sub-modules directly for specifics,
but common items are re-exported here for convenience:

    from src.api.auth import create_access_token, get_current_user, require_api_key
"""
from src.api.auth.utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from src.api.auth.deps    import get_current_user, require_role, optional_user
from src.api.auth.api_key import require_api_key   # backward-compat: main.py imports this

__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_user",
    "require_role",
    "optional_user",
    "require_api_key",
]
