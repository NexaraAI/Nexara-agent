"""
utils/security.py
Secure master-password handling.
Reads the password ONCE from the named pipe set by start.sh,
then holds it in memory for the process lifetime.
"""

import os
import hashlib
import logging
import functools
from pathlib import Path
from typing import Callable, Awaitable

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger("nexara.security")

_MASTER_HASH: str | None = None   # SHA-256 of the master password
_ADMIN_ID: int | None    = None


def load_password() -> None:
    """
    Called once at startup.
    Reads the password from the named pipe path in NEXARA_PASS_PIPE,
    hashes it, and stores the hash.  The pipe file is deleted by start.sh
    shortly after, so it never persists on disk.
    """
    global _MASTER_HASH, _ADMIN_ID

    _ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

    pipe_path = os.getenv("NEXARA_PASS_PIPE", "")
    if not pipe_path or not Path(pipe_path).exists():
        logger.warning("No NEXARA_PASS_PIPE — password protection disabled.")
        return

    try:
        with open(pipe_path, "r") as f:
            raw = f.read().strip()
        if raw:
            _MASTER_HASH = hashlib.sha256(raw.encode()).hexdigest()
            logger.info("Master password loaded and hashed.")
        else:
            logger.info("Empty password — protection disabled.")
    except Exception as exc:
        logger.error("Failed to read password pipe: %s", exc)


def is_admin(user_id: int) -> bool:
    return _ADMIN_ID is not None and user_id == _ADMIN_ID


def verify_password(candidate: str) -> bool:
    if _MASTER_HASH is None:
        return True  # No password set → always pass
    h = hashlib.sha256(candidate.encode()).hexdigest()
    return h == _MASTER_HASH


def admin_only(func: Callable[..., Awaitable]) -> Callable:
    """Decorator: only the ADMIN_ID can invoke this handler."""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        uid = update.effective_user.id if update.effective_user else 0
        if not is_admin(uid):
            await update.message.reply_text("⛔ Admin only.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper
