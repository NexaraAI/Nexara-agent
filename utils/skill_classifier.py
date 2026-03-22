"""
utils/skill_classifier.py — Nexara V1
Lightweight skill selector — runs before the ReAct loop to pick
only the 4-8 most relevant skills for the request.

No LLM call needed. Pure Python keyword + category matching.
Token cost drops from ~2000 (78 skills) to ~150 (6 skills) per call.

Also provides human-readable status messages per skill so the user
sees exactly what the agent is doing instead of "Thinking..."
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger("nexara.skill_classifier")

# ── Always-included skills ────────────────────────────────────────────────────
# These are needed on every agent call regardless of topic.
ALWAYS_INCLUDE: list[str] = [
    "remember",
    "recall",
    "schedule_task",   # needed when any step reveals a scheduling intent
    "file_generate",   # always available — agent may decide to save output
]

# ── Keyword → skill mapping ───────────────────────────────────────────────────
# Most specific matches first. Single keyword hit adds the whole skill list.
KEYWORD_MAP: dict[str, list[str]] = {
    # Web
    "search":       ["web_search", "web_scrape"],
    "google":       ["web_search"],
    "look up":      ["web_search"],
    "find":         ["web_search", "web_scrape"],
    "research":     ["web_search", "web_scrape"],
    "browse":       ["web_scrape"],
    "website":      ["web_scrape"],
    "url":          ["web_scrape", "web_search"],
    "news":         ["news", "web_search"],
    "latest":       ["news", "web_search"],
    "rss":          ["rss_feed"],
    # File generation
    "pdf":          ["file_generate"],
    "docx":         ["file_generate"],
    "word":         ["file_generate"],
    "excel":        ["file_generate"],
    "xlsx":         ["file_generate"],
    "spreadsheet":  ["file_generate"],
    "csv":          ["file_generate"],
    "html":         ["file_generate"],
    "report":       ["file_generate", "web_search"],
    "document":     ["file_generate"],
    "generate":     ["file_generate", "run_code"],
    "create file":  ["file_generate"],
    "save":         ["file_generate", "write_file"],
    "export":       ["file_generate"],
    # File ops
    "read":         ["read_file", "analyze_file"],
    "open file":    ["read_file"],
    "write":        ["write_file"],
    "edit file":    ["read_file", "write_file"],
    "analyze":      ["analyze_file"],
    "analyse":      ["analyze_file"],
    "list files":   ["list_dir"],
    "directory":    ["list_dir"],
    "folder":       ["list_dir"],
    "zip":          ["zip_files"],
    "unzip":        ["command"],
    # Code execution
    "run":          ["run_code", "command"],
    "execute":      ["run_code", "command"],
    "python":       ["run_code"],
    "script":       ["run_code", "command"],
    "code":         ["run_code"],
    # System / install
    "install":      ["apt_install", "command"],
    "apt":          ["apt_install", "apt_search", "apt_update"],
    "java":         ["apt_install", "command"],
    "node":         ["apt_install", "command"],
    "npm":          ["command"],
    "sudo":         ["command"],
    "command":      ["command"],
    "terminal":     ["command"],
    "shell":        ["command"],
    "disk":         ["disk_analyze", "command"],
    "storage":      ["disk_analyze", "system_info"],
    "cpu":          ["system_info", "process_monitor"],
    "memory":       ["system_info"],
    "ram":          ["system_info"],
    "process":      ["process_monitor", "command"],
    "docker":       ["docker_ops"],
    "nginx":        ["nginx_manage"],
    "systemd":      ["systemd"],
    "log":          ["log_tail", "journal_log"],
    "ssh":          ["ssh_exec"],
    "git":          ["git_ops"],
    "cron":         ["cron_manage"],
    # Schedule / remind
    "schedule":     ["schedule_task", "list_schedules"],
    "remind":       ["schedule_task"],
    "reminder":     ["schedule_task"],
    "every day":    ["schedule_task"],
    "every hour":   ["schedule_task"],
    "every week":   ["schedule_task"],
    "every morning":["schedule_task"],
    "at 8am":       ["schedule_task"],
    "at 9am":       ["schedule_task"],
    "daily":        ["schedule_task"],
    "recurring":    ["schedule_task"],
    "in 5 minutes": ["schedule_task"],
    "in 10 minutes":["schedule_task"],
    "in an hour":   ["schedule_task"],
    # Download
    "download":     ["download"],
    "youtube":      ["download"],
    "video":        ["download"],
    "mp3":          ["download"],
    "audio":        ["download"],
    "apk":          ["download"],
    # Android
    "battery":      ["battery", "device_control"],
    "sms":          ["sms"],
    "text message": ["sms"],
    "camera":       ["camera"],
    "screenshot":   ["screen_capture"],
    "flashlight":   ["device_control"],
    "torch":        ["device_control"],
    "volume":       ["device_control"],
    "wifi":         ["wifi_scan"],
    "notification": ["termux_notify"],
    "call log":     ["call_log"],
    "clipboard":    ["clipboard"],
    # Comms
    "email":        ["email_send"],
    "discord":      ["discord_send"],
    "slack":        ["slack_send"],
    "webhook":      ["webhook"],
    "ntfy":         ["ntfy_send"],
    # Data
    "weather":      ["weather"],
    "temperature":  ["weather"],
    "crypto":       ["crypto_price"],
    "bitcoin":      ["crypto_price"],
    "stock":        ["stock_price"],
    "price":        ["crypto_price", "stock_price"],
    "currency":     ["currency", "currency_convert"],
    "convert":      ["currency_convert", "file_generate"],
    "translate":    ["translate"],
    "qr":           ["qr_code"],
    "qr code":      ["qr_code"],
    "hash":         ["hash_tool"],
    "password":     ["password_gen"],
    "uuid":         ["uuid_gen"],
    "base64":       ["base64_tool"],
    "json":         ["json_query", "file_generate"],
    "xml":          ["xml_parse", "file_generate"],
    "yaml":         ["yaml_tool"],
    "csv query":    ["csv_query", "pandas_query"],
    "database":     ["sqlite_manager", "sqlite_query"],
    "sql":          ["sqlite_manager", "sqlite_query"],
    "regex":        ["regex_tool"],
    "diff":         ["text_diff"],
    "markdown":     ["markdown_render", "file_generate"],
    # Speedtest
    "speed test":   ["speedtest"],
    "speedtest":    ["speedtest"],
    "internet speed":["speedtest"],
    "bandwidth":    ["speedtest"],
    "ping":         ["speedtest", "command"],
    # Image
    "image":        ["image_tools"],
    "photo":        ["image_tools", "camera"],
    "resize":       ["image_tools"],
    "convert image":["image_tools"],
    # macOS
    "applescript":  ["applescript"],
    "brew":         ["brew"],
    "homebrew":     ["brew"],
    "launchctl":    ["launchctl"],
    # Windows
    "powershell":   ["powershell"],
    "registry":     ["registry_read"],
    "wsl":          ["wsl_exec"],
    # Memory
    "remember":     ["remember"],
    "recall":       ["recall"],
    "forget":       ["recall"],
    "memory":       ["recall", "remember"],
}

# ── Category → skill mapping ──────────────────────────────────────────────────
# Used when keyword matching gives fewer than MIN_SKILLS results.
# Each category covers a broader intent.

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "web":      ["web", "internet", "online", "browse", "site", "page",
                 "article", "post", "read about", "look", "check"],
    "file":     ["file", "document", "save", "make", "create", "write",
                 "produce", "output", "generate", "export"],
    "system":   ["system", "computer", "machine", "server", "linux",
                 "ubuntu", "install", "package", "software", "tool",
                 "program", "app", "application", "service"],
    "code":     ["code", "script", "program", "function", "class",
                 "debug", "fix", "error", "bug", "test", "build"],
    "data":     ["data", "number", "calculate", "compute", "parse",
                 "query", "database", "table", "row", "column"],
    "comms":    ["send", "message", "notify", "alert", "email",
                 "discord", "slack", "notification"],
    "android":  ["phone", "android", "device", "mobile", "termux"],
    "schedule": ["schedule", "remind", "timer", "alarm", "every",
                 "daily", "weekly", "hourly", "morning", "evening",
                 "night", "at noon", "midnight"],
}

CATEGORY_SKILLS: dict[str, list[str]] = {
    "web":      ["web_search", "web_scrape", "news"],
    "file":     ["file_generate", "read_file", "write_file"],
    "system":   ["command", "apt_install", "system_info", "run_code"],
    "code":     ["run_code", "command", "write_file", "read_file"],
    "data":     ["json_query", "csv_query", "sqlite_query", "pandas_query"],
    "comms":    ["email_send", "discord_send", "slack_send", "ntfy_send"],
    "android":  ["battery", "sms", "device_control", "termux_notify"],
    "schedule": ["schedule_task", "list_schedules", "cancel_schedule"],
}

# Fallback when nothing matches — general-purpose set
DEFAULT_SKILLS: list[str] = [
    "web_search",
    "command",
    "run_code",
    "file_generate",
    "system_info",
]

MAX_SKILLS = 8
MIN_SKILLS = 3


# ── Human-readable status messages per skill ──────────────────────────────────

SKILL_LABELS: dict[str, str] = {
    # Web
    "web_search":        "🔍 Searching the web",
    "web_scrape":        "🌐 Reading webpage",
    "news":              "📰 Fetching news",
    "rss_feed":          "📡 Reading RSS feed",
    # Files
    "file_generate":     "📄 Generating file",
    "read_file":         "📖 Reading file",
    "write_file":        "💾 Writing file",
    "delete_file":       "🗑️ Deleting file",
    "analyze_file":      "🔬 Analysing file",
    "list_dir":          "📁 Listing directory",
    "search_files":      "🔎 Searching files",
    "zip_files":         "🗜️ Compressing files",
    # Code / system
    "run_code":          "⚙️ Executing code",
    "command":           "💻 Running command",
    "apt_install":       "📦 Installing package",
    "apt_remove":        "🗑️ Removing package",
    "apt_search":        "🔍 Searching packages",
    "apt_update":        "🔄 Updating packages",
    "system_info":       "🖥️ Checking system info",
    "process_monitor":   "📊 Monitoring processes",
    "disk_analyze":      "💿 Analysing disk",
    "log_tail":          "📋 Reading logs",
    "docker_ops":        "🐳 Docker operation",
    "nginx_manage":      "🌐 Managing Nginx",
    "systemd":           "⚙️ Managing service",
    "ssh_exec":          "🔐 SSH command",
    "git_ops":           "🔧 Git operation",
    "cron_manage":       "⏱️ Managing cron",
    "python_env":        "🐍 Python environment",
    "ufw_firewall":      "🔥 Firewall operation",
    # Android
    "battery":           "🔋 Checking battery",
    "sms":               "💬 Reading SMS",
    "camera":            "📷 Capturing photo",
    "screen_capture":    "📱 Taking screenshot",
    "device_control":    "📱 Device control",
    "wifi_scan":         "📶 Scanning WiFi",
    "termux_notify":     "🔔 Sending notification",
    "call_log":          "📞 Reading call log",
    "clipboard":         "📋 Clipboard access",
    "media_control":     "🎵 Media control",
    "alarm_set":         "⏰ Setting alarm",
    "tts_speak":         "🔊 Text to speech",
    # Download
    "download":          "⬇️ Downloading file",
    # Schedule
    "schedule_task":     "📅 Scheduling task",
    "list_schedules":    "📅 Listing schedules",
    "cancel_schedule":   "❌ Cancelling schedule",
    # Memory
    "remember":          "🧠 Saving to memory",
    "recall":            "🧠 Searching memory",
    # Comms
    "email_send":        "📧 Sending email",
    "discord_send":      "💬 Sending to Discord",
    "slack_send":        "💬 Sending to Slack",
    "ntfy_send":         "🔔 Sending notification",
    "webhook":           "🔗 Calling webhook",
    # Data
    "weather":           "🌤️ Checking weather",
    "crypto_price":      "₿ Checking crypto price",
    "stock_price":       "📈 Checking stock price",
    "currency":          "💱 Currency lookup",
    "currency_convert":  "💱 Converting currency",
    "translate":         "🌐 Translating text",
    "qr_code":           "📱 Generating QR code",
    "hash_tool":         "🔑 Hashing data",
    "password_gen":      "🔒 Generating password",
    "uuid_gen":          "🆔 Generating UUID",
    "base64_tool":       "🔢 Base64 encode/decode",
    "json_query":        "📊 Querying JSON",
    "xml_parse":         "📊 Parsing XML",
    "yaml_tool":         "📊 Processing YAML",
    "csv_query":         "📊 Querying CSV",
    "pandas_query":      "📊 Analysing data",
    "sqlite_query":      "🗄️ Querying database",
    "sqlite_manager":    "🗄️ Database operation",
    "regex_tool":        "🔍 Regex operation",
    "text_diff":         "📝 Comparing text",
    "text_tools":        "📝 Text processing",
    "markdown_render":   "📝 Rendering markdown",
    "image_tools":       "🖼️ Processing image",
    "speedtest":         "🚀 Running speed test",
    "url_shorten":       "🔗 Shortening URL",
    "timezone":          "🕐 Timezone lookup",
    # macOS
    "applescript":       "🍎 AppleScript",
    "brew":              "🍺 Homebrew",
    "launchctl":         "⚙️ LaunchCtl",
    "pbclipboard":       "📋 macOS clipboard",
    "screen_capture_mac":"📱 Screenshot",
    "disk_util_mac":     "💿 Disk utility",
    "open_url_mac":      "🌐 Opening URL",
    # Windows
    "powershell":        "💠 PowerShell",
    "registry_read":     "🗂️ Registry read",
    "wsl_exec":          "🐧 WSL command",
    "windows_info":      "🪟 Windows info",
    "windows_services":  "⚙️ Windows services",
    "windows_notify":    "🔔 Windows notification",
    "event_log":         "📋 Event log",
    # Self
    "self_heal":         "🔧 Self-healing",
}


def skill_label(name: str, args: dict | None = None) -> str:
    """
    Return a human-readable status string for a skill execution.
    Optionally enriches with arg values (e.g. query text, package name).
    """
    base = SKILL_LABELS.get(name, f"⚡ Running {name}")
    if not args:
        return base

    # Enrich common skills with argument details
    if name == "web_search" and args.get("query"):
        q = args["query"][:50]
        return f"🔍 Searching: {q}"
    if name == "web_scrape" and args.get("url"):
        return f"🌐 Reading: {args['url'][:50]}"
    if name == "file_generate" and args.get("format"):
        fmt = args["format"].upper()
        fn  = args.get("filename", "")
        return f"📄 Generating {fmt}{f': {fn}' if fn else ''}"
    if name in ("apt_install", "apt_remove") and args.get("package"):
        verb = "Installing" if name == "apt_install" else "Removing"
        return f"📦 {verb}: {args['package']}"
    if name == "command" and args.get("command"):
        cmd = args["command"][:40]
        return f"💻 Running: {cmd}"
    if name == "run_code":
        return "⚙️ Executing code..."
    if name == "download" and args.get("url"):
        return f"⬇️ Downloading: {args['url'][:40]}"
    if name == "schedule_task" and args.get("goal"):
        return f"📅 Scheduling: {args['goal'][:40]}"
    if name == "translate" and args.get("target_language"):
        return f"🌐 Translating to {args['target_language']}"
    if name == "weather" and args.get("location"):
        return f"🌤️ Weather: {args['location']}"
    if name == "remember" and args.get("content"):
        return f"🧠 Remembering: {args['content'][:40]}"
    if name == "recall" and args.get("query"):
        return f"🧠 Recalling: {args['query'][:40]}"
    if name == "ssh_exec" and args.get("command"):
        return f"🔐 SSH: {args['command'][:40]}"
    if name in ("discord_send", "slack_send", "email_send"):
        return f"{SKILL_LABELS[name]}..."

    return base


# ── Classifier ────────────────────────────────────────────────────────────────

class SkillClassifier:
    """
    Selects the most relevant skills for a given user request.

    Flow:
      1. Keyword match — most specific, handles ~70% of requests
      2. Category match — broader intent detection for the rest
      3. Always-include set appended
      4. Filter to only active/loaded skills
      5. Cap at MAX_SKILLS
      6. Fallback to DEFAULT_SKILLS if still too few
    """

    def select(self, text: str, active_skills: list[str]) -> list[str]:
        clean    = text.lower()
        selected: set[str] = set()

        # Layer 1 — keyword match
        for keyword, skills in KEYWORD_MAP.items():
            if keyword in clean:
                selected.update(skills)

        # Layer 2 — category match (if not enough from keywords)
        if len(selected) < MIN_SKILLS:
            for category, triggers in CATEGORY_KEYWORDS.items():
                if any(t in clean for t in triggers):
                    selected.update(CATEGORY_SKILLS.get(category, []))
                    if len(selected) >= MIN_SKILLS:
                        break

        # Layer 3 — always include
        selected.update(ALWAYS_INCLUDE)

        # Filter to only loaded skills
        active_set = set(active_skills)
        filtered   = {s for s in selected if s in active_set}

        # If still too few after filtering, add defaults that are loaded
        if len(filtered) < MIN_SKILLS:
            for s in DEFAULT_SKILLS:
                if s in active_set:
                    filtered.add(s)

        result = sorted(filtered)[:MAX_SKILLS]
        logger.debug("Classifier: '%s' → %s", text[:60], result)
        return result

    def explain(self, text: str, active_skills: list[str]) -> str:
        """Debug helper — shows why each skill was selected."""
        clean    = text.lower()
        reasons: dict[str, str] = {}

        for keyword, skills in KEYWORD_MAP.items():
            if keyword in clean:
                for s in skills:
                    reasons[s] = f"keyword:{keyword}"

        for category, triggers in CATEGORY_KEYWORDS.items():
            for t in triggers:
                if t in clean:
                    for s in CATEGORY_SKILLS.get(category, []):
                        if s not in reasons:
                            reasons[s] = f"category:{category}"

        for s in ALWAYS_INCLUDE:
            if s not in reasons:
                reasons[s] = "always"

        active_set = set(active_skills)
        lines = [f"Classifier explanation for: '{text[:60]}'"]
        for skill, reason in sorted(reasons.items()):
            loaded = "✅" if skill in active_set else "❌ not loaded"
            lines.append(f"  {skill} ({reason}) {loaded}")
        return "\n".join(lines)
