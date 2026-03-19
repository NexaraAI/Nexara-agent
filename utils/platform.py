"""
utils/platform.py — Nexara V1
Platform detection engine. Runs once at startup, cached forever.
Every other module reads from this — nothing does its own platform checks.

Detects:
  Primary   → android | linux | windows | macos | unknown
  Sub-env   → termux | codespace | wsl | docker | native
  Arch      → arm64 | x86_64
  Tags      → which skill categories to load
"""

import os
import sys
import platform
import socket
import shutil
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


class PlatformType(Enum):
    ANDROID = "android"
    LINUX   = "linux"
    WINDOWS = "windows"
    MACOS   = "macos"
    UNKNOWN = "unknown"


class SubEnv(Enum):
    TERMUX    = "termux"
    CODESPACE = "codespace"
    WSL       = "wsl"
    DOCKER    = "docker"
    NATIVE    = "native"


@dataclass
class PlatformContext:
    platform:   PlatformType
    sub_env:    SubEnv
    arch:       str          # arm64 | x86_64 | unknown
    skill_tags: list[str]    # ["core", "android"] — used by skill loader
    termux_api: bool         # termux-api binaries present
    os_name:    str          # human-readable e.g. "Ubuntu 22.04"
    hostname:   str

    def display(self) -> str:
        icons = {
            PlatformType.ANDROID: "📱",
            PlatformType.LINUX:   "🐧",
            PlatformType.WINDOWS: "🪟",
            PlatformType.MACOS:   "🍎",
            PlatformType.UNKNOWN: "❓",
        }
        sub = f" · {self.sub_env.value}" if self.sub_env != SubEnv.NATIVE else ""
        return f"{icons[self.platform]} {self.os_name}{sub} · {self.arch}"


@lru_cache(maxsize=1)
def detect() -> PlatformContext:
    """
    Detect platform once and cache. Priority order matters — most specific first.
    """
    arch     = _arch()
    hostname = socket.gethostname()

    # ── GitHub Codespace ──────────────────────────────────────────────────────
    if os.getenv("CODESPACES") == "true" or os.getenv("GITHUB_CODESPACE_TOKEN"):
        return PlatformContext(
            platform=PlatformType.LINUX, sub_env=SubEnv.CODESPACE,
            arch=arch, skill_tags=["core", "linux"],
            termux_api=False, os_name="GitHub Codespace (Ubuntu)",
            hostname=hostname,
        )

    # ── WSL ───────────────────────────────────────────────────────────────────
    if _is_wsl():
        return PlatformContext(
            platform=PlatformType.LINUX, sub_env=SubEnv.WSL,
            arch=arch, skill_tags=["core", "linux"],
            termux_api=False, os_name="Windows Subsystem for Linux",
            hostname=hostname,
        )

    # ── Termux / Android ─────────────────────────────────────────────────────
    if Path("/data/data/com.termux").exists():
        api = shutil.which("termux-battery-status") is not None
        return PlatformContext(
            platform=PlatformType.ANDROID, sub_env=SubEnv.TERMUX,
            arch=arch, skill_tags=["core", "android"],
            termux_api=api,
            os_name=f"Android · Termux · API {'on' if api else 'off'}",
            hostname=hostname,
        )

    # ── Docker ────────────────────────────────────────────────────────────────
    if Path("/.dockerenv").exists():
        return PlatformContext(
            platform=PlatformType.LINUX, sub_env=SubEnv.DOCKER,
            arch=arch, skill_tags=["core", "linux"],
            termux_api=False, os_name="Docker (Linux)",
            hostname=hostname,
        )

    # ── macOS ─────────────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        return PlatformContext(
            platform=PlatformType.MACOS, sub_env=SubEnv.NATIVE,
            arch=arch, skill_tags=["core", "macos"],
            termux_api=False, os_name=f"macOS {platform.mac_ver()[0]}",
            hostname=hostname,
        )

    # ── Windows ───────────────────────────────────────────────────────────────
    if sys.platform == "win32" or os.getenv("OS") == "Windows_NT":
        return PlatformContext(
            platform=PlatformType.WINDOWS, sub_env=SubEnv.NATIVE,
            arch=arch, skill_tags=["core", "windows"],
            termux_api=False, os_name=f"Windows {platform.release()}",
            hostname=hostname,
        )

    # ── Linux native ──────────────────────────────────────────────────────────
    if sys.platform.startswith("linux"):
        os_name = "Linux"
        try:
            info = dict(
                line.split("=", 1)
                for line in Path("/etc/os-release").read_text().splitlines()
                if "=" in line
            )
            os_name = info.get("PRETTY_NAME", "Linux").strip('"')
        except Exception:
            pass
        return PlatformContext(
            platform=PlatformType.LINUX, sub_env=SubEnv.NATIVE,
            arch=arch, skill_tags=["core", "linux"],
            termux_api=False, os_name=os_name,
            hostname=hostname,
        )

    return PlatformContext(
        platform=PlatformType.UNKNOWN, sub_env=SubEnv.NATIVE,
        arch=arch, skill_tags=["core"],
        termux_api=False, os_name=sys.platform,
        hostname=hostname,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arch() -> str:
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):  return "arm64"
    if m in ("x86_64",  "amd64"): return "x86_64"
    return m or "unknown"


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False
