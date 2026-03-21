"""
agent/tool_schema.py — Nexara V1
Function-calling schemas for all skills.

V1 change: gemini_tools() and openai_tools() now accept an optional
active_skills list. Only schemas for loaded skills are sent to the LLM.
This reduces token usage by 40-60% on non-Android platforms.
"""

from typing import Any


# ── Schema builder helpers ────────────────────────────────────────────────────

def _str(desc: str, enum: list[str] | None = None) -> dict:
    s: dict = {"type": "string", "description": desc}
    if enum: s["enum"] = enum
    return s

def _int(desc: str, default: int | None = None) -> dict:
    s: dict = {"type": "integer", "description": desc}
    if default is not None: s["default"] = default
    return s

def _bool(desc: str) -> dict:
    return {"type": "boolean", "description": desc}

def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


# ── All tool schemas ──────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [

    # ── Web ───────────────────────────────────────────────────────────────────
    _fn("speedtest", "Run an internet speed test. Returns download/upload Mbit/s and ping.", {}),

    _fn("web_search", "Search the web using DuckDuckGo.",
        {"query": _str("Search query")}, required=["query"]),

    _fn("web_scrape", "Fetch and parse a full web page.",
        {"url":     _str("Full URL to fetch"),
         "extract": _str("What to extract", ["full_text", "summary", "links", "tables"])},
        required=["url"]),

    # ── Downloads ─────────────────────────────────────────────────────────────
    _fn("download", "Download any file: HTTP, YouTube, APK, audio.",
        {"url":         _str("URL to download"),
         "filename":    _str("Optional output filename"),
         "subdir":      _str("Subdirectory in ~/nexara_downloads/"),
         "audio_only":  _bool("Extract audio only (MP3)"),
         "quality":     _str("Video quality", ["best", "worst", "bestvideo+bestaudio"]),
         "install_apk": _bool("Auto-install APK after download")},
        required=["url"]),

    _fn("list_downloads", "List files in the Nexara downloads folder.",
        {"subdir": _str("Optional subdirectory")}),

    _fn("analyze_file", "Analyze a file: PDF text, CSV schema, ZIP contents, image metadata.",
        {"path": _str("Path to file")}, required=["path"]),

    # ── File system ───────────────────────────────────────────────────────────
    _fn("read_file", "Read a text file.",
        {"path":      _str("File path"),
         "max_chars": _int("Max characters to return", 4000)},
        required=["path"]),

    _fn("write_file", "Write or append text to a file.",
        {"path":    _str("File path"),
         "content": _str("Content to write"),
         "mode":    _str("Write mode", ["w", "a"])},
        required=["path", "content"]),

    _fn("list_dir", "List directory contents.",
        {"path":        _str("Directory path (default: ~)"),
         "show_hidden": _bool("Include hidden files")}),

    _fn("search_files", "Search files by name pattern or content.",
        {"root":          _str("Directory to search"),
         "pattern":       _str("Glob pattern e.g. *.pdf"),
         "content_query": _str("Search file contents for this string"),
         "max_results":   _int("Max results", 20)},
        required=["root"]),

    _fn("delete_file", "Delete a file (confirm must be true).",
        {"path":    _str("File to delete"),
         "confirm": _bool("Must be true to proceed")},
        required=["path", "confirm"]),

    _fn("zip_files", "Create a zip archive.",
        {"source": _str("Source path"),
         "output": _str("Output zip path (optional)")},
        required=["source"]),

    # ── Code execution ────────────────────────────────────────────────────────
    _fn("run_code", "Run Python code in an isolated subprocess.",
        {"code":    _str("Python code to execute"),
         "save_as": _str("Optional script filename"),
         "timeout": _int("Timeout in seconds", 30)},
        required=["code"]),

    _fn("list_scripts", "List saved Python scripts.", {}),

    # ── Shell ─────────────────────────────────────────────────────────────────
    _fn("command", "Run a whitelisted shell command.",
        {"command": _str("Shell command")}, required=["command"]),

    # ── Android / Termux:API ──────────────────────────────────────────────────
    _fn("battery",      "Get device battery level and status.", {}),
    _fn("device_stats", "Device snapshot: battery, WiFi, storage, uptime.", {}),

    _fn("app_launcher", "Launch an Android app.",
        {"app": _str("App name or package")}, required=["app"]),

    _fn("camera_capture", "Take a photo with device camera.",
        {"camera_id": _int("Camera (0=rear, 1=front)", 0)}),

    _fn("notification_reader", "Read Android notifications.", {}),

    _fn("location", "Get GPS coordinates.",
        {"provider": _str("Provider", ["gps", "network", "passive"])}),

    _fn("torch", "Toggle device flashlight.",
        {"state": _str("on or off", ["on", "off"])}, required=["state"]),

    _fn("volume", "Get or set device volume.",
        {"stream": _str("Audio stream", ["music", "ring", "alarm", "notification", "call"]),
         "level":  _int("Volume level 0-15 (-1 to query)")}),

    _fn("vibrate", "Vibrate device.",
        {"duration_ms": _int("Duration ms", 500)}),

    # ── SMS / Contacts ────────────────────────────────────────────────────────
    _fn("read_sms", "Read SMS messages.",
        {"limit":      _int("Number of messages", 10),
         "inbox_type": _str("Folder", ["inbox", "sent", "all"])}),

    _fn("send_sms", "Send an SMS.",
        {"number": _str("Phone number"), "message": _str("Message text")},
        required=["number", "message"]),

    _fn("read_contacts", "Search device contacts.",
        {"query": _str("Name to search")}),

    # ── System ────────────────────────────────────────────────────────────────
    _fn("system_info",  "CPU, RAM, disk, network snapshot.", {}),

    _fn("process_list", "List running processes.",
        {"sort_by": _str("Sort by", ["cpu", "mem"]),
         "limit":   _int("Max processes", 15)}),

    _fn("kill_process", "Kill a process.",
        {"pid":    _int("Process ID"),
         "name":   _str("Process name"),
         "signal": _str("Signal", ["TERM", "KILL", "HUP"])}),

    _fn("network_scan", "Scan ports on a host.",
        {"target": _str("Host to scan"),
         "ports":  _str("Comma-separated port list")}),

    _fn("self_heal", "Fix common issues: restart, clear cache, upgrade deps.",
        {"action": _str("Action", ["restart_bot", "clear_cache", "upgrade_deps"])},
        required=["action"]),

    # ── Scheduler ─────────────────────────────────────────────────────────────
    _fn("schedule_task", "Schedule a recurring task.",
        {"goal":     _str("What to do"),
         "schedule": _str("When, e.g. 'every day at 8am'"),
         "task_id":  _str("Optional unique ID")},
        required=["goal", "schedule"]),

    _fn("list_schedules",  "List all scheduled tasks.", {}),
    _fn("cancel_schedule", "Cancel a scheduled task.",
        {"task_id": _str("Task ID")}, required=["task_id"]),

    # ── Memory ────────────────────────────────────────────────────────────────
    _fn("remember", "Store a fact in long-term memory.",
        {"content":    _str("What to remember"),
         "kind":       _str("Memory type", ["fact", "preference", "reflection", "task"]),
         "importance": _int("Importance 1-5", 3)},
        required=["content"]),

    _fn("recall", "Search long-term memory.",
        {"query": _str("What to look for"),
         "kind":  _str("Filter by type", ["fact", "preference", "reflection", "task"])},
        required=["query"]),
]

_SCHEMA_MAP = {s["name"]: s for s in TOOL_SCHEMAS}


# ── Filtered schema builders ──────────────────────────────────────────────────

def gemini_tools(active_skills: list[str] | None = None) -> list[dict]:
    """
    Returns Gemini-format tool declarations.
    If active_skills provided, only includes schemas for those skills.
    """
    schemas = _filter(active_skills)
    return [{"function_declarations": schemas}]


def openai_tools(active_skills: list[str] | None = None) -> list[dict]:
    """Returns OpenAI/Groq/NVIDIA-format tool declarations."""
    schemas = _filter(active_skills)
    return [{"type": "function", "function": s} for s in schemas]


def _filter(active_skills: list[str] | None) -> list[dict]:
    if not active_skills:
        return TOOL_SCHEMAS
    return [s for s in TOOL_SCHEMAS if s["name"] in active_skills]


def get_schema(name: str) -> dict | None:
    return _SCHEMA_MAP.get(name)


def all_skill_names() -> list[str]:
    return list(_SCHEMA_MAP.keys())
