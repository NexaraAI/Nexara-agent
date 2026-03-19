#!/usr/bin/env bash
# ════════════════════════════════════════
#  Nexara V1.0 — Universal Bootloader
#  Platforms: Termux · Linux · macOS · WSL
# ════════════════════════════════════════
set -euo pipefail

# ── ANSI palette ─────────────────────────────────────────────
RESET="\033[0m";  BOLD="\033[1m";  DIM="\033[2m"
WHT="\033[38;5;255m"
BLU="\033[38;5;39m"
CYN="\033[38;5;87m"
GRN="\033[38;5;83m"
YLW="\033[38;5;220m"
RED="\033[38;5;203m"
PRP="\033[38;5;141m"
BLK="\033[38;5;235m"

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/.nexara"
LOG_FILE="$LOG_DIR/install.log"
ENV_FILE="$SCRIPT_DIR/.env"
SESSION="nexara"
PASS_FILE=""
USE_NOHUP=0
mkdir -p "$LOG_DIR"

# ── Helpers ──────────────────────────────────────────────────
_pad()  { echo ""; }
_line() { echo -e "${BLK}  ────────────────────────────────────${RESET}"; }
_info() { echo -e "  ${BLU}◆${RESET}  ${WHT}$*${RESET}"; }
_ok()   { echo -e "  ${GRN}✓${RESET}  ${WHT}$*${RESET}"; }
_warn() { echo -e "  ${YLW}⚠${RESET}  ${YLW}$*${RESET}"; }
_err()  { echo -e "  ${RED}✗${RESET}  ${RED}$*${RESET}"; }
_dim()  { echo -e "  ${DIM}  $*${RESET}"; }
_head() { echo -e "  ${PRP}${BOLD}$*${RESET}"; }

# ── Cleanup / trap ───────────────────────────────────────────
cleanup() {
  tput cnorm 2>/dev/null || true
  [[ -n "$PASS_FILE" ]] && rm -f "$PASS_FILE" 2>/dev/null || true
  if [[ "${1:-}" == "err" ]]; then
    _pad; _err "Startup aborted."; _dim "See: $LOG_FILE"; _pad
  fi
  exit "${2:-0}"
}
trap 'cleanup err 1' INT TERM ERR

# ════════════════════════════════════════
#  1. OS DETECTION
# ════════════════════════════════════════
detect_os() {
  if [[ -d "/data/data/com.termux" ]]; then
    OS_TYPE="termux"
  elif grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    OS_TYPE="wsl"
  elif [[ "$(uname)" == "Darwin" ]]; then
    OS_TYPE="macos"
  else
    OS_TYPE="linux"
  fi
}

# ════════════════════════════════════════
#  2. MOBILE-SAFE BANNER  (≤ 40 cols)
# ════════════════════════════════════════
show_banner() {
  clear
  _pad
  echo -e "${PRP}${BOLD}"
  echo "  ███╗  ██╗███████╗██╗  ██╗"
  echo "  ████╗ ██║██╔════╝╚██╗██╔╝"
  echo "  ██╔██╗██║█████╗   ╚███╔╝ "
  echo "  ██║╚████║██╔══╝   ██╔██╗ "
  echo "  ██║ ╚███║███████╗██╔╝ ██╗"
  echo "  ╚═╝  ╚══╝╚══════╝╚═╝  ╚═╝"
  echo -e "${DIM}"
  echo    "      █████╗ ██████╗  █████╗ "
  echo    "     ██╔══██╗██╔══██╗██╔══██╗"
  echo    "     ███████║██████╔╝███████║"
  echo    "     ██╔══██║██╔══██╗██╔══██║"
  echo    "     ██║  ██║██║  ██║██║  ██║"
  echo -e "     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝${RESET}"
  _pad
  case "$OS_TYPE" in
    termux) echo -e "  ${DIM}Android Agent  ·  V1.0 Production${RESET}" ;;
    macos)  echo -e "  ${DIM}macOS Agent    ·  V1.0 Production${RESET}" ;;
    wsl)    echo -e "  ${DIM}WSL Agent      ·  V1.0 Production${RESET}" ;;
    *)      echo -e "  ${DIM}Linux Agent    ·  V1.0 Production${RESET}" ;;
  esac
  _pad; _line; _pad
}

# ════════════════════════════════════════
#  3. OS-SPECIFIC SETUP PIPELINES
# ════════════════════════════════════════

# ── Termux (Android) ─────────────────────────────────────────
setup_termux() {
  _head "Termux setup pipeline"
  _pad

  _info "Syncing package index..."
  pkg update -y >> "$LOG_FILE" 2>&1 || true

  local missing=()
  for tool in tmux git python; do
    command -v "$tool" &>/dev/null || missing+=("$tool")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    _info "Installing: ${missing[*]}"
    pkg install -y "${missing[@]}" >> "$LOG_FILE" 2>&1 \
      && _ok "Packages installed" \
      || _warn "Some packages may have failed — check $LOG_FILE"
  else
    _ok "All base packages present"
  fi

  # Termux:API optional enhancement
  if command -v termux-battery-status &>/dev/null; then
    _ok "Termux:API detected — hardware skills enabled"
    export TERMUX_API_AVAILABLE=1
  else
    _warn "Termux:API absent — hardware skills disabled"
    _dim  "Install from F-Droid + pkg install termux-api"
    export TERMUX_API_AVAILABLE=0
  fi
}

# ── Standard Linux / WSL ─────────────────────────────────────
setup_linux() {
  _head "Linux setup pipeline"
  _pad
  export TERMUX_API_AVAILABLE=0

  local PKG_MGR=""
  if   command -v apt-get &>/dev/null; then PKG_MGR="apt"
  elif command -v dnf     &>/dev/null; then PKG_MGR="dnf"
  elif command -v pacman  &>/dev/null; then PKG_MGR="pacman"
  fi

  if [[ -z "$PKG_MGR" ]]; then
    _warn "Unknown package manager — skipping auto-install"
    _dim  "Ensure tmux, git, python3 are installed manually."
    return 0
  fi

  local missing=()
  for tool in tmux git python3; do
    command -v "$tool" &>/dev/null || missing+=("$tool")
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    _ok "All base packages present"; return 0
  fi

  _info "Installing: ${missing[*]} via $PKG_MGR"

  case "$PKG_MGR" in
    apt)
      sudo apt-get update -qq >> "$LOG_FILE" 2>&1 \
        || apt-get update -qq >> "$LOG_FILE" 2>&1 || true
      sudo apt-get install -y -q "${missing[@]}" >> "$LOG_FILE" 2>&1 \
        || apt-get install -y -q "${missing[@]}" >> "$LOG_FILE" 2>&1 \
        || _warn "apt install may have failed"
      ;;
    dnf)
      sudo dnf install -y -q "${missing[@]}" >> "$LOG_FILE" 2>&1 \
        || _warn "dnf install may have failed"
      ;;
    pacman)
      sudo pacman -Sy --noconfirm "${missing[@]}" >> "$LOG_FILE" 2>&1 \
        || _warn "pacman install may have failed"
      ;;
  esac

  for tool in tmux git python3; do
    command -v "$tool" &>/dev/null \
      && _ok "$tool ready" \
      || _warn "$tool still missing after install attempt"
  done
}

# ── macOS ────────────────────────────────────────────────────
setup_macos() {
  _head "macOS setup pipeline"
  _pad
  export TERMUX_API_AVAILABLE=0

  if ! command -v brew &>/dev/null; then
    _warn "Homebrew not found."
    _dim  "Install from https://brew.sh, then re-run this script."
    _pad
    return 0
  fi

  local missing=()
  for tool in tmux git python3; do
    command -v "$tool" &>/dev/null || missing+=("$tool")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    _info "Installing via Homebrew: ${missing[*]}"
    brew install "${missing[@]}" >> "$LOG_FILE" 2>&1 \
      && _ok "Homebrew packages installed" \
      || _warn "brew install may have failed — check $LOG_FILE"
  else
    _ok "All base packages present"
  fi
}

# ── tmux fallback guard ───────────────────────────────────────
_ensure_tmux() {
  if command -v tmux &>/dev/null; then
    _ok "tmux ready"; return 0
  fi
  _warn "tmux unavailable — will use nohup fallback"
  USE_NOHUP=1
}

# ════════════════════════════════════════
#  4. AI CONFIGURATION WIZARD
# ════════════════════════════════════════
_prompt_key() {
  # Usage: _prompt_key "Display Label" VAR_NAME "(hint text)"
  local label="$1" hint="${3:-}"
  echo -e "  ${CYN}?${RESET}  ${BOLD}${label}${RESET} ${DIM}${hint}${RESET}"
  printf "    ${DIM}→ ${RESET}"
  read -r "${2?}"
}

wizard() {
  _pad; _line; _pad
  _head "AI Provider Setup"
  _pad
  _dim  "No .env found — configuring Nexara now."
  _pad

  # ── Provider menu ───────────────────────────────────────────
  echo -e "  ${PRP}${BOLD}Select AI Provider:${RESET}"
  _pad
  echo -e "  ${WHT}[1]${RESET} Gemini Native  ${DIM}(Google AI Studio)${RESET}"
  echo -e "  ${WHT}[2]${RESET} Groq           ${DIM}(Fast inference)${RESET}"
  echo -e "  ${WHT}[3]${RESET} NVIDIA NIM     ${DIM}(NVIDIA cloud)${RESET}"
  echo -e "  ${WHT}[4]${RESET} Ollama         ${DIM}(Local model)${RESET}"
  echo -e "  ${WHT}[5]${RESET} Full Chain     ${DIM}(Custom fallback order)${RESET}"
  _pad

  local CHOICE
  while true; do
    printf "  ${CYN}→${RESET} "; read -r CHOICE
    case "$CHOICE" in [1-5]) break ;; *) _warn "Enter a number 1–5." ;; esac
  done
  _pad

  # ── Declare collector vars ──────────────────────────────────
  local GEMINI_KEY="" GROQ_KEY="" NVIDIA_KEY=""
  local OLLAMA_URL="" OLLAMA_MODEL=""
  local PRIMARY_PROVIDER="" ENABLED_PROVIDERS=""

  case "$CHOICE" in
    1)
      PRIMARY_PROVIDER="gemini"
      ENABLED_PROVIDERS="gemini"
      _prompt_key "Gemini API Key" GEMINI_KEY "(aistudio.google.com)"
      ;;
    2)
      PRIMARY_PROVIDER="groq"
      ENABLED_PROVIDERS="groq"
      _prompt_key "Groq API Key" GROQ_KEY "(console.groq.com)"
      ;;
    3)
      PRIMARY_PROVIDER="nvidia"
      ENABLED_PROVIDERS="nvidia"
      _prompt_key "NVIDIA NIM API Key" NVIDIA_KEY "(build.nvidia.com)"
      ;;
    4)
      PRIMARY_PROVIDER="ollama"
      ENABLED_PROVIDERS="ollama"
      _prompt_key "Ollama Base URL" OLLAMA_URL "(default: http://localhost:11434)"
      [[ -z "$OLLAMA_URL" ]] && OLLAMA_URL="http://localhost:11434"
      _prompt_key "Ollama Model Name" OLLAMA_MODEL "(e.g. llama3, mistral)"
      [[ -z "$OLLAMA_MODEL" ]] && OLLAMA_MODEL="llama3"
      ;;
    5)
      _head "Full Chain Configuration"
      _pad
      echo -e "  ${WHT}Enable which providers? ${DIM}(Y/n for each)${RESET}"
      _pad

      local en_gemini en_groq en_nvidia en_ollama
      printf "  ${CYN}?${RESET}  ${BOLD}Gemini?${RESET} ${DIM}[Y/n]${RESET} "; read -r en_gemini
      printf "  ${CYN}?${RESET}  ${BOLD}Groq?  ${RESET} ${DIM}[Y/n]${RESET} "; read -r en_groq
      printf "  ${CYN}?${RESET}  ${BOLD}NVIDIA?${RESET} ${DIM}[Y/n]${RESET} "; read -r en_nvidia
      printf "  ${CYN}?${RESET}  ${BOLD}Ollama?${RESET} ${DIM}[Y/n]${RESET} "; read -r en_ollama
      _pad

      local ep_list=()
      [[ "$en_gemini" =~ ^[Yy]?$ ]] && {
        ep_list+=("gemini")
        _prompt_key "Gemini API Key" GEMINI_KEY "(aistudio.google.com)"
      }
      [[ "$en_groq" =~ ^[Yy]?$ ]] && {
        ep_list+=("groq")
        _prompt_key "Groq API Key" GROQ_KEY "(console.groq.com)"
      }
      [[ "$en_nvidia" =~ ^[Yy]?$ ]] && {
        ep_list+=("nvidia")
        _prompt_key "NVIDIA NIM API Key" NVIDIA_KEY "(build.nvidia.com)"
      }
      [[ "$en_ollama" =~ ^[Yy]?$ ]] && {
        ep_list+=("ollama")
        _prompt_key "Ollama Base URL" OLLAMA_URL "(default: http://localhost:11434)"
        [[ -z "$OLLAMA_URL" ]] && OLLAMA_URL="http://localhost:11434"
        _prompt_key "Ollama Model Name" OLLAMA_MODEL "(e.g. llama3, mistral)"
        [[ -z "$OLLAMA_MODEL" ]] && OLLAMA_MODEL="llama3"
      }

      if [[ ${#ep_list[@]} -eq 0 ]]; then
        _err "No providers selected — aborting."; cleanup err 1
      fi

      ENABLED_PROVIDERS="${ep_list[*]}"
      _pad

      # ── Primary selection ─────────────────────────────────
      echo -e "  ${WHT}Choose PRIMARY provider:${RESET}"
      local idx=1
      for p in "${ep_list[@]}"; do
        echo -e "  ${WHT}[$idx]${RESET} $p"; (( idx++ ))
      done
      _pad
      local p_choice
      while true; do
        printf "  ${CYN}→${RESET} "; read -r p_choice
        if [[ "$p_choice" =~ ^[0-9]+$ ]] && \
           (( p_choice >= 1 && p_choice <= ${#ep_list[@]} )); then
          PRIMARY_PROVIDER="${ep_list[$((p_choice-1))]}"; break
        fi
        _warn "Enter a number between 1 and ${#ep_list[@]}."
      done
      ;;
  esac

  # ── Telegram (always required) ──────────────────────────────
  _pad; _line; _pad
  _head "Telegram Configuration"
  _pad

  local BOT_TOKEN ADMIN_ID
  _prompt_key "Telegram Bot Token" BOT_TOKEN "(from @BotFather)"
  _pad
  _prompt_key "Your Telegram User ID" ADMIN_ID "(from @userinfobot)"
  _pad

  # ── Write .env ──────────────────────────────────────────────
  cat > "$ENV_FILE" <<EOF
# ── Nexara V1.0 Configuration ────────────────────────
# Generated by start.sh on $(date '+%Y-%m-%d %H:%M:%S')
# ─────────────────────────────────────────────────────

NEXARA_VERSION=1.0.0

# Telegram
TELEGRAM_TOKEN=${BOT_TOKEN}
ADMIN_ID=${ADMIN_ID}

# AI Provider Routing
PRIMARY_PROVIDER=${PRIMARY_PROVIDER}
ENABLED_PROVIDERS="${ENABLED_PROVIDERS}"

# Gemini
GEMINI_API_KEY=${GEMINI_KEY}
LLM_API_KEY=${GEMINI_KEY}

# Groq
GROQ_API_KEY=${GROQ_KEY}
GROQ_MODEL=llama-3.3-70b-versatile

# NVIDIA NIM
NVIDIA_API_KEY=${NVIDIA_KEY}
NVIDIA_MODEL=meta/llama-3.1-70b-instruct

# Ollama
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3}
EOF

  _ok "Config saved → $ENV_FILE"
}

# ════════════════════════════════════════
#  5. DEPENDENCY INSTALLATION
# ════════════════════════════════════════
install_deps() {
  local REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

  if [[ ! -f "$REQUIREMENTS" ]]; then
    _warn "requirements.txt not found — skipping pip install"; return 0
  fi

  _pad; _line; _pad
  _info "Preparing Python dependencies..."
  _pad

  # ── User-facing patience message ────────────────────────────
  echo -e "  ${YLW}${BOLD}⏳  Downloading AI libraries.${RESET}"
  echo -e "  ${YLW}    This may take 1–3 minutes depending on"
  echo -e "      your connection. Please do not close the"
  echo -e "      terminal or press Ctrl+C...${RESET}"
  _pad

  local PIP_CMD="pip3"
  command -v pip3 &>/dev/null || PIP_CMD="python3 -m pip"

  local PIP_EXTRA=""
  if [[ "$OS_TYPE" == "termux" ]]; then
    python3 -c \
      "import sys; exit(0 if sys.version_info>=(3,11) else 1)" 2>/dev/null \
      && PIP_EXTRA="--break-system-packages"
  fi

  local PIP_EXIT=0
  $PIP_CMD install -q -r "$REQUIREMENTS" $PIP_EXTRA \
    >> "$LOG_FILE" 2>&1 || PIP_EXIT=$?

  if [[ $PIP_EXIT -ne 0 ]]; then
    _warn "First pass had errors — retrying with --no-cache-dir..."
    $PIP_CMD install -q -r "$REQUIREMENTS" $PIP_EXTRA \
      --no-cache-dir >> "$LOG_FILE" 2>&1 || PIP_EXIT=$?
  fi

  if [[ $PIP_EXIT -eq 0 ]]; then
    _ok "Python dependencies up to date"
  else
    _warn "Some deps failed — check $LOG_FILE"
    _dim  "Agent may still start if core packages are present."
  fi
}

# ════════════════════════════════════════
#  6. MASTER PASSWORD & SECURE LAUNCH
# ════════════════════════════════════════
get_master_password() {
  _pad; _line; _pad
  _head "Master Password"
  _dim  "Secures admin-only Telegram commands."
  _dim  "Leave blank to disable password protection."
  _pad
  printf "  ${CYN}?${RESET}  ${BOLD}Password${RESET} ${DIM}(hidden):${RESET} "
  read -rs MASTER_PASS; echo ""
  _pad
}

build_launch_env() {
  LAUNCH_ENV="\
TELEGRAM_TOKEN='${TELEGRAM_TOKEN:-}' \
ADMIN_ID='${ADMIN_ID:-}' \
PRIMARY_PROVIDER='${PRIMARY_PROVIDER:-gemini}' \
ENABLED_PROVIDERS='${ENABLED_PROVIDERS:-gemini}' \
LLM_API_KEY='${LLM_API_KEY:-}' \
GEMINI_API_KEY='${GEMINI_API_KEY:-${LLM_API_KEY:-}}' \
GROQ_API_KEY='${GROQ_API_KEY:-}' \
GROQ_MODEL='${GROQ_MODEL:-llama-3.3-70b-versatile}' \
NVIDIA_API_KEY='${NVIDIA_API_KEY:-}' \
NVIDIA_MODEL='${NVIDIA_MODEL:-meta/llama-3.1-70b-instruct}' \
OLLAMA_URL='${OLLAMA_URL:-http://localhost:11434}' \
OLLAMA_MODEL='${OLLAMA_MODEL:-llama3}' \
NEXARA_PASS_PIPE='${PASS_FILE}' \
NEXARA_VERSION='${NEXARA_VERSION:-1.0.0}' \
TERMUX_API_AVAILABLE='${TERMUX_API_AVAILABLE:-0}'"
}

do_launch() {
  _pad; _line; _pad
  _head "Launch Sequence"
  _pad

  # Kill stale session
  if [[ $USE_NOHUP -eq 0 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
    _warn "Stale session found — restarting..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.4
  fi

  local LAUNCH_CMD="cd '${SCRIPT_DIR}' && ${LAUNCH_ENV} python3 main.py"

  # ── tmux launch ───────────────────────────────────────────
  if [[ $USE_NOHUP -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -x 220 -y 50 2>/dev/null \
      || { _warn "tmux new-session failed — falling back to nohup"; USE_NOHUP=1; }
  fi

  if [[ $USE_NOHUP -eq 0 ]]; then
    tmux send-keys -t "$SESSION" "$LAUNCH_CMD" Enter
    _ok "Agent launched in tmux session '${SESSION}'"

    if [[ -f "$SCRIPT_DIR/watchdog.sh" ]]; then
      tmux new-window -t "$SESSION" -n "watchdog" \
        "bash '${SCRIPT_DIR}/watchdog.sh'" 2>/dev/null || true
      _dim "Watchdog active in tmux window 2"
    fi
  else
    # ── nohup fallback ─────────────────────────────────────
    _warn "Launching via nohup (tmux unavailable)..."
    nohup bash -c "$LAUNCH_CMD" >> "$LOG_DIR/nexara.log" 2>&1 &
    echo $! > "$LOG_DIR/nexara.pid"
    _ok "Agent launched (PID $(cat "$LOG_DIR/nexara.pid"))"
  fi

  # ── Deferred secure wipe of password pipe ─────────────────
  # Give main.py 5 s to read the pipe, then shred it silently
  ( sleep 5; rm -f "${PASS_FILE}" 2>/dev/null ) &
  disown $!
}

verify_launch() {
  sleep 2
  local ALIVE=0

  if [[ $USE_NOHUP -eq 0 ]]; then
    tmux has-session -t "$SESSION" 2>/dev/null && ALIVE=1
  else
    local PID_FILE="$LOG_DIR/nexara.pid"
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && ALIVE=1
  fi

  if [[ $ALIVE -eq 1 ]]; then
    _ok "Agent is live"
  else
    _err "Agent failed to start — check $LOG_DIR/nexara.log"
    cleanup err 1
  fi
}

print_footer() {
  _pad; _line; _pad
  echo -e "  ${GRN}${BOLD}✅  Nexara V${NEXARA_VERSION:-1.0.0} is running.${RESET}"
  _pad
  if [[ $USE_NOHUP -eq 0 ]]; then
    _dim "Attach  :  tmux attach -t nexara"
    _dim "Detach  :  Ctrl+b  then  d"
    _dim "Stop    :  tmux kill-session -t nexara"
  else
    _dim "Logs    :  tail -f $LOG_DIR/nexara.log"
    _dim "Stop    :  kill \$(cat $LOG_DIR/nexara.pid)"
  fi
  _dim "Install log : $LOG_FILE"
  _pad; _line; _pad
}

# ════════════════════════════════════════
#  MAIN ENTRYPOINT
# ════════════════════════════════════════
main() {
  detect_os
  show_banner

  # ── Python pre-flight ───────────────────────────────────────
  _info "Checking environment..."
  if ! command -v python3 &>/dev/null; then
    _err "python3 not found."
    case "$OS_TYPE" in
      termux) _err "Run: pkg install python" ;;
      macos)  _err "Run: brew install python" ;;
      *)      _err "Run: sudo apt install python3" ;;
    esac
    exit 1
  fi
  PY_VER=$(python3 -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  _ok "Python $PY_VER detected"

  case "$OS_TYPE" in
    termux) _ok "Termux / Android detected" ;;
    macos)  _ok "macOS detected" ;;
    wsl)    _ok "WSL / Windows detected" ;;
    linux)  _ok "Linux detected" ;;
  esac
  _pad

  # ── OS-specific setup pipeline ──────────────────────────────
  case "$OS_TYPE" in
    termux)    setup_termux ;;
    macos)     setup_macos  ;;
    linux|wsl) setup_linux  ;;
  esac
  _pad

  # ── tmux availability guard ─────────────────────────────────
  _ensure_tmux

  # ── Config wizard or load existing .env ────────────────────
  if [[ ! -f "$ENV_FILE" ]]; then
    wizard
  else
    _pad; _ok ".env found — using existing configuration"
  fi

  # Source .env into the current shell
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a

  # ── Python dependencies ─────────────────────────────────────
  install_deps

  # ── Master password ─────────────────────────────────────────
  get_master_password

  # ── Write password to secure temp pipe ─────────────────────
  PASS_FILE="/tmp/nexara_pass_$$"
  printf '%s' "$MASTER_PASS" > "$PASS_FILE"
  chmod 600 "$PASS_FILE"
  export NEXARA_PASS_PIPE="$PASS_FILE"
  export NEXARA_VERSION="${NEXARA_VERSION:-1.0.0}"

  # ── Build env, launch, verify ───────────────────────────────
  build_launch_env
  do_launch
  verify_launch
  print_footer
}

main "$@"
