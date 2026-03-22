"""
utils/error_formatter.py — Nexara V1
Translates raw Python exceptions into human-readable messages.
Used by handle_message and react_loop to avoid showing users raw
tracebacks or cryptic exception strings.
"""

import re


# Maps exception type names / message patterns → friendly text
_PATTERNS: list[tuple[str, str]] = [
    # Network
    (r"ConnectTimeout|connect timeout",          "⏱ Network timeout — couldn't reach the server. Try again."),
    (r"ReadTimeout|read timeout",                "⏱ Server took too long to respond. Try again."),
    (r"ConnectionRefused|connection refused",    "🔌 Connection refused — server may be down."),
    (r"DNSLookupError|Name or service not known","🌐 DNS lookup failed — check your network."),
    (r"ConnectError|connection error",           "🌐 Network error — check your connection."),
    (r"SSLError",                                "🔒 SSL/TLS error — certificate issue."),
    # HTTP / API
    (r"HTTP 400|status_code=400",               "❌ Bad request — invalid parameters sent to API."),
    (r"HTTP 401|status_code=401",               "🔑 Unauthorised — check your API key in .env."),
    (r"HTTP 403|status_code=403",               "⛔ Forbidden — API key may lack permissions."),
    (r"HTTP 404|status_code=404",               "🔍 Not found — the resource or model doesn't exist."),
    (r"HTTP 429|status_code=429|rate.?limit",   "🚫 Rate limited — too many requests. Waiting…"),
    (r"HTTP 500|status_code=500",               "💥 Server error — the API is having issues. Try again."),
    (r"HTTP 503|status_code=503",               "🔧 Service unavailable — API is temporarily down."),
    # LLM provider specific
    (r"All LLM providers exhausted",            "😴 All AI providers are unavailable. Check your API keys in .env."),
    (r"GROQ_API_KEY|Groq not configured",       "🔑 Groq API key not set. Add GROQ_API_KEY to .env."),
    (r"LLM_API_KEY|Gemini not configured",      "🔑 Gemini API key not set. Add LLM_API_KEY to .env."),
    (r"NVIDIA_API_KEY|NVIDIA NIM not configured","🔑 NVIDIA API key not set. Add NVIDIA_API_KEY to .env."),
    # File system
    (r"FileNotFoundError|No such file",         "📁 File not found — check the path."),
    (r"PermissionError|Permission denied",      "🔒 Permission denied — can't access that file or directory."),
    (r"IsADirectoryError",                      "📁 That path is a directory, not a file."),
    (r"DiskFull|No space left",                 "💿 Disk is full — free up some space."),
    # Process / command
    (r"CalledProcessError",                     "❌ Command failed — check the command syntax."),
    (r"TimeoutExpired|timed out",               "⏱ Command timed out — it took too long to complete."),
    (r"FileNotFoundError.*command",             "❓ Command not found — may not be installed."),
    # Python / skill
    (r"ModuleNotFoundError|No module named",    "📦 Missing Python package — will be installed on next run."),
    (r"ImportError",                            "📦 Import error — a required library may be missing."),
    (r"JSONDecodeError|json.decoder",           "📋 Invalid JSON — the data couldn't be parsed."),
    (r"KeyError",                               "🔑 Missing key in data — unexpected response format."),
    (r"AttributeError",                         "⚙️ Unexpected response format from the skill."),
    (r"ValueError",                             "⚙️ Invalid value — check the input parameters."),
    (r"TypeError",                              "⚙️ Type mismatch — unexpected data type received."),
    # Memory / DB
    (r"sqlite3\.|OperationalError.*database",   "🗄️ Database error — try /clear if this persists."),
    # Generic
    (r"SIGBUS|Bus error|core dumped",           "💥 Process crashed (SIGBUS) — restarting may help."),
    (r"MemoryError|out of memory",              "💾 Out of memory — the server needs more RAM."),
    (r"RecursionError",                         "🔄 Recursion error — task is too deeply nested."),
]


def friendly(exc: Exception | str) -> str:
    """
    Convert an exception or error string to a user-friendly message.
    Falls back to a cleaned version of the original error.
    """
    raw = str(exc)

    for pattern, message in _PATTERNS:
        if re.search(pattern, raw, re.IGNORECASE):
            return message

    # Generic cleanup — remove Python internals from the message
    cleaned = raw
    # Strip file paths
    cleaned = re.sub(r'File "[^"]+", line \d+, in \S+\s*', '', cleaned)
    # Strip exception class prefix
    cleaned = re.sub(r'^\w+Error:\s*', '', cleaned)
    cleaned = re.sub(r'^\w+Exception:\s*', '', cleaned)
    cleaned = cleaned.strip()

    if not cleaned or cleaned == raw:
        # Last resort — just say something went wrong with a hint
        exc_type = type(exc).__name__ if isinstance(exc, Exception) else "Error"
        return f"⚠️ Something went wrong ({exc_type}). Try again or rephrase your request."

    return f"⚠️ {cleaned[:200]}"
