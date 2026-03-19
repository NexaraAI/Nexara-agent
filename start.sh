#!/usr/bin/env bash
# ============================================================
#  Nexara Agent V1.0 — Universal Startup Script
#  Works on: Termux (Android) · Linux · GitHub Codespaces
# ============================================================

RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
WHT="\033[38;5;255m"
BLU="\033[38;5;39m"
CYN="\033[38;5;87m"
GRN="\033[38;5;83m"
YLW="\033[38;5;220m"
RED="\033[38;5;203m"
PRP="\033[38;5;141m"
BLK="\033[38;5;235m"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/.nexara"
LOG_FILE="$LOG_DIR/install.log"
SESSION="nexara"
PASS_FILE=""

# ── Create log dir immediately ────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Detect environment ────────────────────────────────────────────────────────
if [[ -d "/data/data/com.termux" ]]; then
  ENV_TYPE="termux"
else
  ENV_TYPE="linux"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
_center() {
  local text="$1" width=56
  local pad=$(( (width - ${#text}) / 2 ))
  printf "%${pad}s%s%${pad}s\n" "" "$text" ""
}
_line() { echo -e "${BLK}  ─────────────────────────────────────────────────────${RESET}"; }
_pad()  { echo ""; }
_info() { echo -e "  ${BLU}◆${RESET}  ${WHT}$1${RESET}"; }
_ok()   { echo -e "  ${GRN}✓${RESET}  ${WHT}$1${RESET}"; }
_warn() { echo -e "  ${YLW}⚠${RESET}  ${YLW}$1${RESET}"; }
_err()  { echo -e "  ${RED}✗${RESET}  ${RED}$1${RESET}"; }
_dim()  { echo -e "  ${DIM}  $1${RESET}"; }

cleanup() {
  tput cnorm 2>/dev/null
  [[ -n "$PASS_FILE" ]] && rm -f "$PASS_FILE" 2>/dev/null
  if [[ "$1" == "err" ]]; then
    _pad; _err "Startup aborted. Check ~/.nexara/install.log for details."; _pad
  fi
  exit "${2:-0}"
}
trap 'cleanup err 1' INT TERM

# ── Banner ────────────────────────────────────────────────────────────────────
clear
_pad
echo -e "${PRP}${BOLD}"
echo    "       ███╗   ██╗███████╗██╗  ██╗ █████╗ ██████╗  █████╗ "
echo    "       ████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗██╔══██╗██╔══██╗"
echo    "       ██╔██╗ ██║█████╗   ╚███╔╝ ███████║██████╔╝███████║"
echo    "       ██║╚██╗██║██╔══╝   ██╔██╗ ██╔══██║██╔══██╗██╔══██║"
echo    "       ██║ ╚████║███████╗██╔╝ ██╗██║  ██║██║  ██║██║  ██║"
echo    "       ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"
echo -e "${RESET}"
if [[ "$ENV_TYPE" == "termux" ]]; then
  echo -e "${DIM}$(_center "Android AI Agent  ·  V1.0 Production Release")${RESET}"
else
  echo -e "${DIM}$(_center "Linux / Codespace  ·  V1.0 Production Release")${RESET}"
fi
_line; _pad

# ── Pre-flight ────────────────────────────────────────────────────────────────
_info "Checking environment..."

if [[ "$ENV_TYPE" == "termux" ]]; then
  _ok "Termux (Android) detected"
else
  _warn "Linux mode — Android hardware skills will be disabled"
fi

if ! command -v python3 &>/dev/null; then
  _err "python3 not found."
  [[ "$ENV_TYPE" == "termux" ]] && _err "Run: pkg install python" || _err "Run: sudo apt install python3"
  exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
_ok "Python $PY_VER detected"

# ── Install tmux ──────────────────────────────────────────────────────────────
_ensure_tmux() {
  # Already installed — done
  if command -v tmux &>/dev/null; then
    _ok "tmux ready"
    return 0
  fi

  _info "Installing tmux..."

  if [[ "$ENV_TYPE" == "termux" ]]; then
    pkg install -y tmux >> "$LOG_FILE" 2>&1
  else
    # Try sudo apt, then apt without sudo, then snap
    sudo apt-get install -y -q tmux >> "$LOG_FILE" 2>&1 \
      || apt-get install -y -q tmux >> "$LOG_FILE" 2>&1 \
      || sudo snap install tmux --classic >> "$LOG_FILE" 2>&1 \
      || true
  fi

  # Final check — if still missing, fall back to nohup mode
  if command -v tmux &>/dev/null; then
    _ok "tmux installed"
  else
    _warn "tmux unavailable — will run agent directly (no detach)"
    USE_NOHUP=1
  fi
}

USE_NOHUP=0
_ensure_tmux

# ── Install git if missing ─────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
  if [[ "$ENV_TYPE" == "termux" ]]; then
    pkg install -y git >> "$LOG_FILE" 2>&1
  else
    sudo apt-get install -y -q git >> "$LOG_FILE" 2>&1 || true
  fi
fi

# ── Config check ──────────────────────────────────────────────────────────────
_pad; _line; _pad

ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  _warn "No .env file found — creating one now."
  _pad
  echo -e "  ${CYN}?${RESET}  ${BOLD}Telegram Bot Token${RESET} ${DIM}(from @BotFather)${RESET}"
  printf  "     ${DIM}→ ${RESET}"; read -r BOT_TOKEN
  _pad
  echo -e "  ${CYN}?${RESET}  ${BOLD}Your Telegram User ID${RESET} ${DIM}(from @userinfobot)${RESET}"
  printf  "     ${DIM}→ ${RESET}"; read -r ADMIN_ID_INPUT
  _pad
  echo -e "  ${CYN}?${RESET}  ${BOLD}Gemini API Key${RESET} ${DIM}(aistudio.google.com — or Enter to skip)${RESET}"
  printf  "     ${DIM}→ ${RESET}"; read -r LLM_KEY
  _pad
  echo -e "  ${CYN}?${RESET}  ${BOLD}Groq API Key${RESET} ${DIM}(console.groq.com — or Enter to skip)${RESET}"
  printf  "     ${DIM}→ ${RESET}"; read -r GROQ_KEY
  cat > "$ENV_FILE" <<EOF
TELEGRAM_TOKEN=$BOT_TOKEN
ADMIN_ID=$ADMIN_ID_INPUT
LLM_API_KEY=$LLM_KEY
GROQ_API_KEY=$GROQ_KEY
NEXARA_VERSION=1.0.0
EOF
  _ok "Config saved to .env"
fi

set -a; source "$ENV_FILE"; set +a

# ── Master password ────────────────────────────────────────────────────────────
_pad; _line; _pad
echo -e "  ${PRP}🔐${RESET}  ${BOLD}Master Password${RESET}"
_dim "Secures admin-only Telegram commands. Leave blank to disable."
_pad
printf "     ${DIM}→ ${RESET}"; read -rs MASTER_PASS; echo ""

# ── Python deps ────────────────────────────────────────────────────────────────
_pad; _line; _pad

REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

if [[ -f "$REQUIREMENTS" ]]; then
  _info "Installing Python dependencies..."

  PIP_CMD="pip3"
  command -v pip3 &>/dev/null || PIP_CMD="python3 -m pip"

  # --break-system-packages only needed on Termux Python >= 3.11
  PIP_EXTRA=""
  if [[ "$ENV_TYPE" == "termux" ]]; then
    python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null \
      && PIP_EXTRA="--break-system-packages"
  fi

  $PIP_CMD install -q -r "$REQUIREMENTS" $PIP_EXTRA >> "$LOG_FILE" 2>&1
  PIP_EXIT=$?

  if [[ $PIP_EXIT -ne 0 ]]; then
    _warn "pip had errors — retrying with --no-cache-dir..."
    $PIP_CMD install -q -r "$REQUIREMENTS" $PIP_EXTRA --no-cache-dir >> "$LOG_FILE" 2>&1
    PIP_EXIT=$?
  fi

  if [[ $PIP_EXIT -eq 0 ]]; then
    _ok "Dependencies up to date"
  else
    _warn "Some deps failed — check ~/.nexara/install.log"
    _dim  "Agent may still work if core packages are already installed."
  fi
else
  _warn "requirements.txt not found — skipping"
fi

# ── Termux:API detection ───────────────────────────────────────────────────────
_pad
if [[ "$ENV_TYPE" == "termux" ]]; then
  if command -v termux-battery-status &>/dev/null; then
    _ok "Termux:API detected — God Mode enabled"
    export TERMUX_API_AVAILABLE=1
  else
    _warn "Termux:API not found — hardware commands disabled"
    _dim  "Install from F-Droid + pkg install termux-api to unlock."
    export TERMUX_API_AVAILABLE=0
  fi
else
  _dim "Android hardware skills disabled (Linux mode)"
  export TERMUX_API_AVAILABLE=0
fi

# ── Password handoff ───────────────────────────────────────────────────────────
PASS_FILE="/tmp/nexara_pass_$$"
echo "$MASTER_PASS" > "$PASS_FILE"
chmod 600 "$PASS_FILE"
export NEXARA_PASS_PIPE="$PASS_FILE"
export NEXARA_VERSION="${NEXARA_VERSION:-1.0.0}"

# ── Build env string for launch ────────────────────────────────────────────────
LAUNCH_ENV="\
  TELEGRAM_TOKEN='${TELEGRAM_TOKEN}' \
  ADMIN_ID='${ADMIN_ID}' \
  LLM_API_KEY='${LLM_API_KEY:-}' \
  LLM_MODEL='${LLM_MODEL:-gemini-1.5-flash}' \
  OLLAMA_URL='${OLLAMA_URL:-http://localhost:11434}' \
  OLLAMA_MODEL='${OLLAMA_MODEL:-llama3}' \
  GROQ_API_KEY='${GROQ_API_KEY:-}' \
  GROQ_MODEL='${GROQ_MODEL:-llama-3.3-70b-versatile}' \
  NEXARA_PASS_PIPE='${PASS_FILE}' \
  NEXARA_VERSION='${NEXARA_VERSION}' \
  TERMUX_API_AVAILABLE='${TERMUX_API_AVAILABLE}'"

LAUNCH_CMD="cd '$SCRIPT_DIR' && $LAUNCH_ENV python3 main.py"

# ── Kill existing session ──────────────────────────────────────────────────────
_pad; _line; _pad

if [[ $USE_NOHUP -eq 0 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
  _warn "Existing session found — restarting..."
  tmux kill-session -t "$SESSION" 2>/dev/null
  sleep 0.5
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
if [[ $USE_NOHUP -eq 0 ]]; then
  # Normal tmux launch
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  tmux send-keys -t "$SESSION" "$LAUNCH_CMD" Enter
  _ok "Agent launched in tmux session '${SESSION}'"

  # Watchdog in second window
  if [[ -f "$SCRIPT_DIR/watchdog.sh" ]]; then
    tmux new-window -t "$SESSION" -n "watchdog" \
      "bash '$SCRIPT_DIR/watchdog.sh'" 2>/dev/null || true
  fi
else
  # Fallback: nohup background launch
  _warn "Launching with nohup (tmux unavailable)..."
  nohup bash -c "$LAUNCH_CMD" >> "$LOG_DIR/nexara.log" 2>&1 &
  echo $! > "$LOG_DIR/nexara.pid"
  _ok "Agent launched (PID $(cat $LOG_DIR/nexara.pid))"
fi

sleep 4
rm -f "$PASS_FILE" 2>/dev/null

# ── Verify ─────────────────────────────────────────────────────────────────────
sleep 1
ALIVE=0
if [[ $USE_NOHUP -eq 0 ]]; then
  tmux has-session -t "$SESSION" 2>/dev/null && ALIVE=1
else
  kill -0 "$(cat $LOG_DIR/nexara.pid 2>/dev/null)" 2>/dev/null && ALIVE=1
fi

if [[ $ALIVE -eq 1 ]]; then
  _ok "Agent is live"
else
  _err "Agent failed to start — check ~/.nexara/nexara.log"
  exit 1
fi

# ── Final card ─────────────────────────────────────────────────────────────────
_pad; _line; _pad
echo -e "  ${GRN}${BOLD}✅  Nexara V${NEXARA_VERSION} is running.${RESET}"
_pad
if [[ $USE_NOHUP -eq 0 ]]; then
  _dim "Attach to session  :  tmux attach -t nexara"
  _dim "Stop               :  tmux kill-session -t nexara"
else
  _dim "View logs          :  tail -f ~/.nexara/nexara.log"
  _dim "Stop               :  kill \$(cat ~/.nexara/nexara.pid)"
fi
_dim "View agent logs    :  tail -f ~/.nexara/nexara.log"
_dim "View install logs  :  tail -f ~/.nexara/install.log"
_pad; _line; _pad
