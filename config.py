"""config.py — Nexara V1"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]
ADMIN_ID: int       = int(os.environ["ADMIN_ID"])

# ── LLM: Gemini ───────────────────────────────────────────────────────────────
LLM_API_KEY: str  = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ── LLM: Groq ─────────────────────────────────────────────────────────────────
GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── LLM: NVIDIA NIM ───────────────────────────────────────────────────────────
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL: str   = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")

# ── LLM: Ollama (local) ───────────────────────────────────────────────────────
OLLAMA_URL: str   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── LLM: Router config ────────────────────────────────────────────────────────
PRIMARY_PROVIDER: str = os.getenv("PRIMARY_PROVIDER", "gemini")

# ── Agent ─────────────────────────────────────────────────────────────────────
NEXARA_VERSION: str       = os.getenv("NEXARA_VERSION", "1.0.0")
BOT_NAME:       str       = "Nexara"
MAX_HISTORY_TURNS: int    = 40
MAX_REACT_ITERATIONS: int = 14

# ── Android ───────────────────────────────────────────────────────────────────
TERMUX_API_AVAILABLE: bool = os.getenv("TERMUX_API_AVAILABLE", "0") == "1"

# ── Downloads ─────────────────────────────────────────────────────────────────
DOWNLOADS_DIR: Path         = Path.home() / "nexara_downloads"
MAX_DOWNLOAD_SIZE_GB: float = float(os.getenv("MAX_DOWNLOAD_SIZE_GB", "2.0"))

# ── Skills warehouse ──────────────────────────────────────────────────────────
SKILL_CHANNEL: str       = os.getenv("SKILL_CHANNEL", "stable")
SKILL_WAREHOUSE_URL: str = os.getenv("SKILL_WAREHOUSE_URL", "")

# ── Agent auto-updater ────────────────────────────────────────────────────────
# Set to your nexara-agent repo raw GitHub URL, e.g.:
#   https://raw.githubusercontent.com/YourOrg/nexara-agent/main
# Leave empty to disable agent auto-updates.
AGENT_REPO_URL: str = os.getenv("AGENT_REPO_URL", "")

# ── JVM flags (Android/Termux only) ──────────────────────────────────────────
os.environ.setdefault(
    "_JAVA_OPTIONS",
    "-Djdk.attach.allowAttachSelf=true -Djna.nosys=true "
    "-Djna.nounpack=true -Dfile.encoding=UTF-8 "
    "-XX:+UseG1GC -XX:+UnlockExperimentalVMOptions",
)
