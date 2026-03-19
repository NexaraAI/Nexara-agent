#!/usr/bin/env bash
# ============================================================
#  Nexara Agent — OTA Self-Updater
#  Called by the bot as a detached subprocess.
#  Kills the current session, pulls latest code, restarts.
# ============================================================

SESSION="nexara"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$HOME/.nexara/update.log"

mkdir -p "$HOME/.nexara"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OTA update initiated"

  sleep 3  # Let Telegram message send before we kill the process

  # Kill existing tmux session
  tmux kill-session -t "$SESSION" 2>/dev/null
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Session killed"

  cd "$SCRIPT_DIR" || exit 1

  # Attempt git pull; fall back to zip download if no .git dir
  if [[ -d ".git" ]]; then
    git pull --ff-only origin main 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git pull complete"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No .git directory — skipping pull"
  fi

  # Upgrade pip deps silently
  if [[ -f "requirements.txt" ]]; then
    pip install -q -r requirements.txt 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dependencies refreshed"
  fi

  # Persist current version to .env
  NEW_VER=$(grep -oP '(?<=NEXARA_VERSION=).*' .env 2>/dev/null || echo "1.0.0")
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting as v$NEW_VER"

  # Relaunch — export ALL .env vars into the shell, then pass them to the new session
  set -a
  source .env 2>/dev/null
  set +a
  
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  
  # Pass every relevant env var explicitly so the restarted Python process inherits them
  tmux send-keys -t "$SESSION" \
    "cd '$SCRIPT_DIR' && \
     TELEGRAM_TOKEN='$TELEGRAM_TOKEN' \
     ADMIN_ID='$ADMIN_ID' \
     LLM_API_KEY='$LLM_API_KEY' \
     LLM_MODEL='${LLM_MODEL:-gemini-1.5-flash}' \
     OLLAMA_URL='${OLLAMA_URL:-http://localhost:11434}' \
     OLLAMA_MODEL='${OLLAMA_MODEL:-llama3}' \
     GROQ_API_KEY='${GROQ_API_KEY:-}' \
     GROQ_MODEL='${GROQ_MODEL:-llama-3.3-70b-versatile}' \
     NVIDIA_API_KEY='${NVIDIA_API_KEY:-}' \
     NVIDIA_MODEL='${NVIDIA_MODEL:-meta/llama-3.1-70b-instruct}' \
     NEXARA_VERSION='$NEXARA_VERSION' \
     TERMUX_API_AVAILABLE='${TERMUX_API_AVAILABLE:-0}' \
     python3 main.py --post-update" \
    Enter

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OTA complete — session restored"
} >> "$LOG" 2>&1
