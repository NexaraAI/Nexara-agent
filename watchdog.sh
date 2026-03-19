#!/usr/bin/env bash
# ============================================================
#  Nexara V1 Watchdog
#  Monitors the 'nexara' tmux session.
#  Features:
#    - Rapid crash loop detection (5 crashes / 60s)
#    - SQLite DB corruption auto-recovery
#    - Clean multi-platform execution
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="nexara"
RESTART_DELAY=5
LOG="$HOME/.nexara/watchdog.log"
MAX_RESTARTS_PER_HOUR=15
WINDOW_RESTART=0
WINDOW_START=$(date +%s)

# Array to track the last 5 crash timestamps for rapid-loop detection
CRASH_TIMESTAMPS=()

mkdir -p "$HOME/.nexara"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

check_sqlite_health() {
  local db_path="$1"
  # Only check if sqlite3 CLI is installed and DB exists
  if [[ -f "$db_path" ]] && command -v sqlite3 >/dev/null 2>&1; then
    local status
    status=$(sqlite3 "$db_path" "PRAGMA integrity_check;" 2>&1)
    if [[ "$status" != "ok" && "$status" != "" ]]; then
      log "⚠️ CORRUPTION DETECTED in $db_path!"
      log "SQLite returned: $status"
      mv "$db_path" "${db_path}.corrupted.$(date +%s)"
      log "Moved corrupted DB to backup so agent can rebuild and boot cleanly."
      CRASH_REASON="SQLite corruption auto-healed ($db_path)"
    fi
  fi
}

log "Watchdog started. Monitoring session '$SESSION'."

while true; do
  sleep 10

  # 1. Reset hourly counter if needed
  NOW=$(date +%s)
  if (( NOW - WINDOW_START > 3600 )); then
    WINDOW_RESTART=0
    WINDOW_START=$NOW
  fi

  # 2. Check if pane is dead
  PANE_DEAD=$(tmux list-panes -t "${SESSION}:0" -F '#{pane_dead}' 2>/dev/null | head -n 1)
  
  # If command failed, session might not exist. Wait and retry.
  if [[ -z "$PANE_DEAD" ]]; then
    continue
  fi

  if [[ "$PANE_DEAD" == "1" ]]; then
    ((WINDOW_RESTART++))
    log "Agent process died! (Restart $WINDOW_RESTART/$MAX_RESTARTS_PER_HOUR this hour)."
    CRASH_REASON="Process crashed unexpectedly"

    # 3. Rapid crash loop detection
    CRASH_TIMESTAMPS+=($NOW)
    if (( ${#CRASH_TIMESTAMPS[@]} > 5 )); then
      # Keep only the last 5 elements
      CRASH_TIMESTAMPS=("${CRASH_TIMESTAMPS[@]:1:5}")
      
      TIME_SINCE_5TH_CRASH=$(( NOW - CRASH_TIMESTAMPS[0] ))
      if (( TIME_SINCE_5TH_CRASH < 60 )); then
        log "🚨 RAPID CRASH LOOP DETECTED: 5 restarts in ${TIME_SINCE_5TH_CRASH} seconds."
        log "The agent is structurally failing. Halting watchdog to prevent resource exhaustion."
        exit 1
      fi
    fi

    # 4. Hourly hard limit
    if (( WINDOW_RESTART > MAX_RESTARTS_PER_HOUR )); then
      log "🚨 Hourly restart limit reached ($MAX_RESTARTS_PER_HOUR). Halting watchdog."
      exit 1
    fi

    log "Waiting ${RESTART_DELAY}s then checking health..."
    sleep "$RESTART_DELAY"

    # 5. SQLite Health Checks
    check_sqlite_health "$HOME/.nexara/memory.db"
    check_sqlite_health "$HOME/.nexara/tasks.db"

    log "Relaunching agent..."

    # 6. Respawn pane cleanly
    # Sources .env dynamically so we don't have to hardcode 15 export statements
    tmux respawn-pane -t "${SESSION}:0" -k \
      "cd '$SCRIPT_DIR' && \
       if [[ -f .env ]]; then set -a; source .env; set +a; fi && \
       export WATCHDOG_CRASH_REASON='$CRASH_REASON' && \
       python3 main.py" 2>/dev/null \
    || {
      log "respawn-pane failed, trying send-keys..."
      tmux send-keys -t "${SESSION}:0" C-c
      sleep 1
      tmux send-keys -t "${SESSION}:0" "cd '$SCRIPT_DIR' && if [[ -f .env ]]; then set -a; source .env; set +a; fi && export WATCHDOG_CRASH_REASON='$CRASH_REASON' && python3 main.py" Enter
    }
  fi
done
