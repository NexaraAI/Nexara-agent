#!/usr/bin/env python3
"""
utils/generate_manifest.py — Nexara V1
Auto-generates agent_manifest.json with fresh SHA256 checksums.

Run from the nexara-agent repo root after any file change:
    python3 utils/generate_manifest.py

Or add to your push workflow:
    python3 utils/generate_manifest.py && git add agent_manifest.json && git commit -m "chore: update manifest checksums"
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# All core agent files that should be tracked for OTA updates.
# Add new files here when you create them.
TRACKED_FILES = [
    "main.py",
    "config.py",
    "agent/llm_router.py",
    "agent/tool_schema.py",
    "agent/react_loop.py",
    "agent/memory.py",
    "agent/planner.py",
    "tasks/scheduler.py",
    "tasks/monitor_task.py",
    "utils/skill_loader.py",
    "utils/skill_classifier.py",
    "utils/agent_updater.py",
    "utils/token_budget.py",
    "utils/error_formatter.py",
    "utils/platform.py",
    "utils/token_budget.py",
    "utils/skill_router.py",
    "utils/security.py",
    "skills/base.py",
]

MANIFEST_PATH = ROOT / "agent_manifest.json"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_existing() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            pass
    return {"version": "1.0.0", "files": {}}


def bump_version(version: str) -> str:
    """Bump patch version: 1.0.0 → 1.0.1"""
    parts = version.split(".")
    if len(parts) == 3:
        try:
            parts[2] = str(int(parts[2]) + 1)
            return ".".join(parts)
        except ValueError:
            pass
    return version


def main():
    existing = load_existing()
    old_files = existing.get("files", {})
    new_files = {}
    changed   = []
    missing   = []

    for rel in TRACKED_FILES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        checksum = sha256(p)
        new_files[rel] = {"checksum": checksum, "url": rel}

        old_checksum = old_files.get(rel, {}).get("checksum", "")
        if checksum != old_checksum:
            changed.append(rel)

    # Bump version if anything changed
    old_version = existing.get("version", "1.0.0")
    new_version = bump_version(old_version) if changed else old_version

    manifest = {
        "version":     new_version,
        "updated":     __import__("datetime").date.today().isoformat(),
        "description": "Nexara Agent core files — OTA update manifest",
        "files":       new_files,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"✅ agent_manifest.json updated")
    print(f"   Version : {old_version} → {new_version}")
    print(f"   Files   : {len(new_files)} tracked")

    if changed:
        print(f"   Changed : {len(changed)} file(s):")
        for f in changed:
            print(f"     • {f}")
    else:
        print("   Changed : none (checksums unchanged)")

    if missing:
        print(f"   Missing : {len(missing)} file(s) (skipped):")
        for f in missing:
            print(f"     ⚠️  {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
