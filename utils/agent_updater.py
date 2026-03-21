"""
utils/agent_updater.py — Nexara V1
OTA self-update for core agent files from GitHub.

How it works:
  1. Fetch agent_manifest.json from AGENT_REPO_URL.
  2. Compare SHA256 of each listed file against local copy.
  3. Download changed files to staging dir, verify checksums.
  4. Move staged files into place atomically (backup originals).
  5. Send Telegram alert listing what changed.
  6. Restart via os.execv so new code takes effect immediately.

agent_manifest.json lives in your nexara-agent repo root:
{
  "version": "1.2.0",
  "files": {
    "main.py":               {"checksum": "sha256:...", "url": "main.py"},
    "agent/llm_router.py":   {"checksum": "sha256:...", "url": "agent/llm_router.py"},
    "agent/tool_schema.py":  {"checksum": "sha256:...", "url": "agent/tool_schema.py"},
    "agent/react_loop.py":   {"checksum": "sha256:...", "url": "agent/react_loop.py"},
    "agent/memory.py":       {"checksum": "sha256:...", "url": "agent/memory.py"},
    "agent/planner.py":      {"checksum": "sha256:...", "url": "agent/planner.py"},
    "tasks/scheduler.py":    {"checksum": "sha256:...", "url": "tasks/scheduler.py"},
    "tasks/monitor_task.py": {"checksum": "sha256:...", "url": "tasks/monitor_task.py"},
    "utils/skill_loader.py": {"checksum": "sha256:...", "url": "utils/skill_loader.py"},
    "utils/agent_updater.py":{"checksum": "sha256:...", "url": "utils/agent_updater.py"},
    "config.py":             {"checksum": "sha256:...", "url": "config.py"}
  }
}

Set in .env:
  AGENT_REPO_URL=https://raw.githubusercontent.com/YourOrg/nexara-agent/main
"""

import asyncio
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Awaitable

import httpx

logger = logging.getLogger("nexara.agent_updater")

AGENT_ROOT    = Path(__file__).parent.parent
STAGING_DIR   = Path.home() / ".nexara" / "agent_staging"
MANIFEST_FILE = "agent_manifest.json"


class AgentUpdater:

    def __init__(
        self,
        repo_url: str,
        alert_cb: Callable[[str], Awaitable[None]] | None = None,
    ):
        self._base        = repo_url.rstrip("/")
        self._alert       = alert_cb
        self._last_check  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def check_and_apply(self, force: bool = False) -> str:
        if not self._base:
            return "Agent auto-update disabled (AGENT_REPO_URL not set)."

        now = time.time()
        if not force and now - self._last_check < 300:
            return "Update check skipped — checked less than 5 minutes ago."
        self._last_check = now

        manifest = await self._fetch_manifest()
        if manifest is None:
            return "Could not fetch agent_manifest.json — check AGENT_REPO_URL."

        remote_version = manifest.get("version", "?")
        files          = manifest.get("files", {})
        if not files:
            return "agent_manifest.json has no files listed."

        # Compare checksums
        outdated: dict[str, dict] = {}
        for rel_path, meta in files.items():
            local    = AGENT_ROOT / rel_path
            expected = meta.get("checksum", "")
            if not local.exists() or self._sha256(local) != expected:
                outdated[rel_path] = meta

        if not outdated:
            return f"Agent is already up to date (v{remote_version})."

        logger.info("Agent update v%s — %d changed: %s",
                    remote_version, len(outdated), list(outdated))

        if self._alert:
            await self._alert(
                f"🔄 Agent update v{remote_version} — "
                f"{len(outdated)} file(s) changed. Downloading…"
            )

        # Download to staging
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            for rel_path, meta in outdated.items():
                url         = f"{self._base}/{meta.get('url', rel_path)}"
                expected    = meta.get("checksum", "")
                staged_path = STAGING_DIR / rel_path.replace("/", "_")

                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    code   = resp.text
                    actual = "sha256:" + hashlib.sha256(code.encode()).hexdigest()

                    if expected and actual != expected:
                        logger.error("Checksum mismatch for %s", rel_path)
                        if self._alert:
                            await self._alert(f"⚠️ Checksum mismatch for `{rel_path}` — skipped.")
                        continue

                    staged_path.write_text(code, encoding="utf-8")
                    staged.append((staged_path, AGENT_ROOT / rel_path))

                except Exception as exc:
                    logger.error("Failed to download %s: %s", rel_path, exc)
                    if self._alert:
                        await self._alert(f"❌ Failed to download `{rel_path}`: {exc}")

        if not staged:
            return "No files could be staged safely."

        # Move into place
        changed = []
        for staged_path, final_path in staged:
            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                backup = final_path.with_suffix(final_path.suffix + ".bak")
                if final_path.exists():
                    final_path.rename(backup)
                staged_path.rename(final_path)
                if backup.exists():
                    backup.unlink()
                changed.append(final_path.name)
            except Exception as exc:
                logger.error("Could not apply %s: %s", staged_path, exc)

        if not changed:
            return "Downloaded but could not apply — check permissions."

        msg = (
            f"✅ Agent updated to v{remote_version}\n"
            f"Changed: {', '.join(changed)}\n"
            "Restarting in 3s…"
        )
        if self._alert:
            await self._alert(msg)

        await asyncio.sleep(3)
        self._restart()
        return msg

    async def get_remote_version(self) -> str | None:
        if not self._base:
            return None
        m = await self._fetch_manifest()
        return m.get("version") if m else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_manifest(self) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self._base}/{MANIFEST_FILE}")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("Manifest fetch failed: %s", exc)
            return None

    @staticmethod
    def _sha256(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _restart():
        logger.info("Restarting via os.execv…")
        os.execv(sys.executable, [sys.executable] + sys.argv)
