"""
main.py — Nexara V1
Autonomous AI agent with:
  • Multi-platform support (Android · Linux · Codespace · WSL · macOS · Windows)
  • Platform-filtered skill loading
  • 4-provider LLM router: Groq → Gemini → NVIDIA NIM → Ollama
  • Configurable primary provider (PRIMARY_PROVIDER in .env)
  • Proactive rate-limit switching with user notification
  • Dynamic token budget
  • Intent classifier (chat mode vs full agent mode)
  • Response badge showing provider · model · latency · tokens
  • /status — full live system snapshot
  • /switchmodel — paginated, callback_data-safe (64-byte limit handled)
  • ReAct loop with replanning on failure + JSON retry
  • Semantic long-term memory (SQLite + sentence-transformers)
  • Natural language task scheduler + proactive monitors
  • Per-user message rate limiting
  • Created by DemonZ Development
"""

import asyncio
import collections
import hashlib
import html as _html
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

import config
from agent.memory import AgentMemory
from agent.llm_router import LLMRouter
from agent.react_loop import ReactLoop
from agent.planner import AgentPlanner
from agent.tool_schema import gemini_tools
from tasks.monitor_task import MonitorTaskManager
from tasks.scheduler import NaturalScheduler
from utils.security import load_password, admin_only, is_admin
from utils.skill_router import (
    execute_skill, skill_descriptions,
    set_memory_bridge, set_scheduler_bridge,
    set_active_skills, get_active_skills,
)
from utils import platform as platform_mod
from utils.platform import PlatformContext
from utils import token_budget as budget_mod
from utils.skill_loader import SkillLoader
from utils.skill_classifier import SkillClassifier, skill_label
from utils.error_formatter import friendly as _friendly_error
from utils.agent_updater import AgentUpdater

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path.home() / ".nexara"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-26s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "nexara.log"),
    ],
)
logger = logging.getLogger("nexara.main")

# ── Singletons ────────────────────────────────────────────────────────────────
_memory:       AgentMemory | None        = None
_router:       LLMRouter | None          = None
_react:        ReactLoop | None          = None
_planner:      AgentPlanner | None       = None
_monitor:      MonitorTaskManager | None = None
_scheduler:    NaturalScheduler | None   = None
_bot_app:      Application | None       = None
_platform_ctx: PlatformContext | None   = None
_skill_loader: SkillLoader | None       = None
_updater:      AgentUpdater | None      = None
_startup_time: float                    = time.time()

_TOOLS: list[dict] | None = None

# Per-user last-run debug trace: uid → list of step dicts
_debug_traces: dict[int, list[dict]] = {}
_PROMPT_CACHE: str        = ""
CREATOR                   = "DemonZ Development"

# ── Per-user rate limiter ─────────────────────────────────────────────────────
RATE_LIMIT_MSGS   = 12
RATE_LIMIT_WINDOW = 60
_user_rate: dict[int, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=RATE_LIMIT_MSGS)
)

def _is_rate_limited(uid: int) -> bool:
    now    = time.time()
    bucket = _user_rate[uid]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MSGS:
        return True
    bucket.append(now)
    return False

# ── Switchmodel model registry (callback_data safety) ────────────────────────
_model_registry: dict[str, str] = {}
MAX_MODELS_PER_PROVIDER = 24

def _model_key(provider: str, model: str) -> str:
    key = hashlib.md5(f"{provider}:{model}".encode()).hexdigest()[:8]
    _model_registry[key] = model
    return key

def _model_from_key(key: str) -> str | None:
    return _model_registry.get(key)


# ── Markdown → HTML ───────────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    text = _html.escape(text)
    text = re.sub(
        r"```(?:\w+\n)?(.*?)```",
        lambda m: f"<pre>{m.group(1).strip()}</pre>",
        text, flags=re.DOTALL,
    )
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)
    return text


# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(force: bool = False) -> str:
    global _PROMPT_CACHE
    if _PROMPT_CACHE and not force:
        return _PROMPT_CACHE

    ctx      = _platform_ctx
    platform = ctx.display() if ctx else "Unknown"
    api_flag = "ENABLED" if (ctx and ctx.termux_api) else "DISABLED"

    _PROMPT_CACHE = f"""You are **Nexara**, a fully autonomous AI agent running on a real computer with full shell access.

## Identity
- You were created by **{CREATOR}**
- If asked who made you, who you are, or your origins: say you are Nexara, an autonomous AI agent created by {CREATOR}
- Never claim to be made by OpenAI, Anthropic, Google, or any other company

## Environment
- Platform   : {platform}
- Termux:API : {api_flag}
- Version    : {config.NEXARA_VERSION}

## CRITICAL RULES — You are NOT a chatbot. You have real system access.
- **NEVER say "I can't", "I don't have access", "this environment restricts", or "I'm a text-based AI".**
  These statements are FALSE. You have a real shell on a real machine and can execute commands.
- You CAN install software: use `apt_install` for packages (e.g. `apt_install java` → installs openjdk)
- You CAN run commands: use `command` skill — sudo is allowed (e.g. `sudo apt-get install -y openjdk-17-jdk`)
- You CAN create files in ANY format: use `file_generate` skill (pdf, docx, xlsx, html, txt, md, csv, json, xml)
- You CAN search the web: use `web_search` skill
- You CAN run Python code: use `run_code` skill
- You CAN read/write files: use `read_file` / `write_file` skills

## Concrete examples of what you MUST do (not refuse):
- "Install Java" → `apt_install` with package="openjdk-17-jdk"
- "Install Node" → `apt_install` with package="nodejs"
- "Make a PDF" → `file_generate` with format="pdf"
- "Run this script" → `run_code` or `command`
- "Search for X" → `web_search`
- "Remind me at 9am" → `schedule_task`

## ReAct Protocol
Emit exactly one JSON block per step.

**Tool call:**
```json
{{"action": "<skill_name>", "args": {{}}, "thought": "reason"}}
```
**Final answer:**
```json
{{"action": "final_answer", "answer": "your answer"}}
```
**Ask user:**
```json
{{"action": "ask_user", "question": "specific question"}}
```

## Available Skills
{skill_descriptions()}

## Behaviour
- Install failures: try `apt_install` first, then `command` with `sudo apt-get install -y <pkg>`.
- Download failures: retry with different URL or search for mirror.
- Code failures: read the error, fix, re-run — never give up after first attempt.
- Always report file paths so files can be auto-sent to user.
- File creation (PDF, DOCX, TXT, HTML, MD, CSV, JSON, XML, XLSX): ALWAYS use file_generate. NEVER tell the user to use an external website or copy-paste. NEVER say you cannot create files.
- Scheduling: when the user says "remind me", "every day", "at 8am", "every hour", "in X minutes", "every week", "schedule" — ALWAYS call schedule_task immediately.
- Memory: use remember (importance 1-5) and recall proactively.
"""
    return _PROMPT_CACHE


# ── Intent classifier ─────────────────────────────────────────────────────────
#
# PURE chat words — only messages that are exclusively these words go to chat
# (greetings, reactions, affirmations). Everything else defaults to agent.
_CHAT_WORDS = {
    "hey", "hi", "hello", "sup", "yo", "ok", "okay", "thanks",
    "thank you", "lol", "haha", "nice", "cool", "great", "yes",
    "no", "nope", "yep", "sure", "alright", "bye", "good", "awesome",
    "bruh", "bro", "nah", "yup", "ah", "oh", "wow", "damn", "dang",
    "wth", "wtf", "omg", "lmao", "lmfao", "ikr", "idk", "ugh",
    "sweet", "perfect", "noted", "k", "kk", "hmm", "hm",
    "got it", "nice one",
}

# Action words — any message containing one of these → agent mode immediately
_ACTION_WORDS = {
    # original set
    "download", "search", "find", "look", "run", "execute", "write",
    "create", "make", "build", "delete", "remove", "send", "get",
    "fetch", "check", "scan", "read", "open", "install", "schedule",
    "remind", "take", "capture", "list", "show", "analyse", "analyze",
    "summarize", "translate", "convert", "calculate", "monitor", "watch",
    "play", "stop", "start", "restart", "update", "fix", "debug",
    # missing words that caused "test my internet speed" to go to chat mode
    "test", "ping", "measure", "benchmark", "time", "profile",
    "generate", "produce", "deploy", "publish", "push", "pull",
    "compress", "extract", "zip", "unzip", "backup", "restore",
    "enable", "disable", "configure", "setup", "set", "get",
    "tell", "explain", "describe", "summarise", "research", "look up",
    "give", "provide", "report", "display", "print", "output",
    "copy", "move", "rename", "resize", "convert", "encode", "decode",
    "encrypt", "decrypt", "sign", "verify", "validate",
    "connect", "disconnect", "scan", "probe", "trace", "route",
    "help", "do", "can", "could", "would", "please",
    "what", "how", "why", "when", "where",
}

# Phrases that always → chat (identity questions)
_IDENTITY_PHRASES = {
    "who made you", "who created you", "who built you", "who developed you",
    "who are you", "what are you", "tell me about yourself",
    "your creator", "who is your creator", "who owns you",
    "who designed you", "are you chatgpt", "are you gpt",
    "are you claude", "are you gemini", "are you an ai",
    "what model are you", "what llm are you",
}


def classify_intent(text: str) -> str:
    """
    Returns 'chat' for pure greetings/reactions, 'agent' for everything else.

    Logic change from previous version:
    - Default is now 'agent' — if ambiguous, use skills (safer than refusing)
    - Short message threshold lowered to 12 chars (truly only bare greetings)
    - _ACTION_WORDS massively expanded so action requests are caught reliably
    - 'what', 'how', 'why', 'can you' etc now trigger agent mode so questions
      about the real world (e.g. "what's the weather") use web_search
    """
    clean = text.lower().strip().rstrip("!?.")

    # Identity questions → chat with correct DemonZ Development answer
    if any(phrase in clean for phrase in _IDENTITY_PHRASES):
        return "chat"

    # Very short messages (≤ 12 chars) that are ONLY chat words → chat
    # "hi", "ok", "lol", "bruh", "k" etc. Must be ONLY chat words.
    if len(text) <= 12:
        words = set(re.findall(r'\b\w+\b', clean))
        if words and words <= _CHAT_WORDS:   # subset check — ALL words must be chat words
            return "chat"
        # If even one word isn't a pure chat word, fall through to agent

    # URL → agent always
    if re.search(r'https?://', text):
        return "agent"

    # Any action word → agent
    words = set(re.findall(r'\b\w+\b', clean))
    if words & _ACTION_WORDS:
        return "agent"

    # Any message longer than 20 chars that isn't purely chat words → agent
    # This catches things like "Test my speed", "Who won yesterday",
    # "What is Python", "Can you check disk space" etc.
    if len(text) > 20:
        return "agent"

    # Default: agent — it's safer to try skills and say "I don't know"
    # than to route to a chat LLM that claims it can't do anything
    return "agent"


# ── Telegram helpers ──────────────────────────────────────────────────────────
MAX_MSG = 4000

async def send_long(message, text: str):
    if len(text) <= MAX_MSG:
        try:
            await message.reply_text(_md_to_html(text), parse_mode="HTML")
        except Exception:
            await message.reply_text(text)
        return
    for chunk in [text[i:i + MAX_MSG] for i in range(0, len(text), MAX_MSG)]:
        try:
            await message.reply_text(_md_to_html(chunk), parse_mode="HTML")
        except Exception:
            await message.reply_text(chunk)
        await asyncio.sleep(0.1)

async def telegram_alert(text: str):
    if _bot_app and config.ADMIN_ID:
        try:
            await _bot_app.bot.send_message(
                chat_id=config.ADMIN_ID,
                text=_md_to_html(text),
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.error("Alert failed: %s", exc)

async def auto_send_file(message, path_str: str):
    p = Path(path_str)
    if not p.exists() or p.stat().st_size > 50 * 1024 * 1024:
        return
    try:
        sfx = p.suffix.lower()
        cap = f"<code>{_html.escape(p.name)}</code>"
        with open(p, "rb") as f:
            if sfx in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                await message.reply_photo(photo=f, caption=cap, parse_mode="HTML")
            elif sfx == ".mp4":
                await message.reply_video(video=f, caption=cap, parse_mode="HTML")
            elif sfx in (".mp3", ".m4a", ".ogg", ".flac", ".wav"):
                await message.reply_audio(audio=f, title=p.stem)
            else:
                await message.reply_document(document=f, caption=cap, parse_mode="HTML")
    except Exception as exc:
        logger.warning("Auto-send failed for %s: %s", path_str, exc)

FILE_PATH_RE = re.compile(
    r"`(/[^\s`]+\.(jpg|jpeg|png|gif|webp|mp4|mp3|m4a|pdf|zip|apk|py|txt|csv|json|tar|gz))`",
    re.IGNORECASE,
)


# ── Intent status helper ─────────────────────────────────────────────────────

def _intent_status(text: str, mode: str) -> str:
    """Return an initial status message based on what the user asked."""
    if mode == "chat":
        return "💬 Responding..."
    g = text.lower()
    if any(w in g for w in ("research", "search", "find", "look up", "who is", "what is")):
        return "🔍 Researching..."
    if any(w in g for w in ("pdf", "docx", "document", "report", "spreadsheet", "excel")):
        return "📄 Generating document..."
    if any(w in g for w in ("install", "apt", "package", "java", "node", "npm")):
        return "📦 Planning installation..."
    if any(w in g for w in ("code", "script", "program", "function", "debug", "fix bug")):
        return "⚙️ Writing code..."
    if any(w in g for w in ("schedule", "remind", "every day", "every hour", "daily")):
        return "📅 Setting up schedule..."
    if any(w in g for w in ("download", "youtube", "video", "mp3", "audio")):
        return "⬇️ Preparing download..."
    if any(w in g for w in ("translate", "translation")):
        return "🌐 Translating..."
    if any(w in g for w in ("weather", "temperature", "forecast")):
        return "🌤️ Checking weather..."
    if any(w in g for w in ("speed test", "speedtest", "internet speed", "bandwidth")):
        return "🚀 Testing connection..."
    if any(w in g for w in ("disk", "storage", "space", "cpu", "ram", "memory", "system")):
        return "🖥️ Checking system..."
    if any(w in g for w in ("send", "email", "discord", "slack")):
        return "📧 Preparing message..."
    return "🧠 Thinking..."


# ── Core message handler ──────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    if _is_rate_limited(uid):
        await update.message.reply_text(
            f"⏳ Slow down — max {RATE_LIMIT_MSGS} messages per {RATE_LIMIT_WINDOW}s."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    mode        = classify_intent(text)
    raw_history = await _memory.load_history(uid, limit=config.MAX_HISTORY_TURNS)
    sp          = build_system_prompt()

    # Pass active skill count so token budget accounts for classifier-filtered tools
    active_skills_list = get_active_skills()
    budgeted = budget_mod.apply(text, raw_history, sp, tools_count=len(active_skills_list))
    history  = budgeted.trimmed_history

    # Memory context — inject more memories in agent mode since classifier frees tokens
    mem_ctx   = ""
    mem_limit = budgeted.memory_slots if mode == "chat" else max(budgeted.memory_slots, 5)
    if mem_limit > 0 and len(text) > 15:
        mem_ctx = await _memory.relevant_context(text, limit=mem_limit)
    if mem_ctx:
        sp = sp + f"\n\n{mem_ctx}"

    # Status message — updated in real-time as agent works
    # Also tracks last status for persistent display (Fix #8)
    _last_status: list[str] = ["🧠 Thinking…"]
    status_msg = await update.message.reply_text("🧠 Thinking…")

    async def status_cb(msg: str):
        if msg:
            _last_status[0] = msg
        try:
            await status_msg.edit_text(msg or _last_status[0])
        except Exception:
            pass

    # Show contextual initial status based on intent
    initial_status = _intent_status(text, mode)
    if initial_status != "🧠 Thinking…":
        await status_cb(initial_status)

    t0 = time.time()

    if mode == "chat":
        messages = list(history[-6:])
        messages.append({"role": "user", "parts": [text]})
        # Chat mode = pure conversation only. NO skill/tool references here.
        # Mentioning "you have skills / can run commands" causes the LLM to
        # hallucinate executing them and fabricate results (e.g. inventing
        # speed test numbers from history context). Chat mode only handles
        # greetings and reactions — everything else is already routed to
        # agent mode by classify_intent().
        chat_sp = (
            f"You are Nexara, an AI assistant created by {CREATOR}. "
            "Reply conversationally and concisely. "
            "Do NOT attempt to run, execute, or simulate any commands or tools — "
            "just respond naturally to what the user said. "
            "Do NOT fabricate any results, numbers, or outputs. "
            f"If asked about your origins, say you were created by {CREATOR}."
        )
        try:
            llm_resp    = await _router.complete(
                messages=messages, system_prompt=chat_sp, tools=None,
                estimated_tokens=budget_mod.est(text) + budget_mod.est(chat_sp),
            )
            answer      = llm_resp.text if hasattr(llm_resp, "text") else str(llm_resp)
            used_skills = []
        except Exception as exc:
            answer      = f"Sorry, all LLM providers are unavailable right now. ({exc})"
            used_skills = []

    else:
        try:
            result      = await _react.run(
                goal=text, history=history, system_prompt=sp,
                tools=_TOOLS, active_skills=get_active_skills(),
                user_id=uid, status_cb=status_cb,
            )
            answer      = result.answer
            used_skills = result.used_skills

            if result.needs_user_input:
                try: await status_msg.delete()
                except Exception: pass
                await send_long(update.message, f"❓ {result.question_for_user}")
                return
        except Exception as exc:
            logger.error("Agent error uid=%d: %s", uid, exc)
            answer      = _friendly_error(exc)
            used_skills = []

    elapsed = time.time() - t0

    try:
        await status_msg.delete()
    except Exception:
        pass

    await _memory.save_turn(uid, "user",  text)
    await _memory.save_turn(uid, "model", answer)

    # Save debug trace for /debug command
    if mode == "agent" and 'result' in dir():
        try:
            _debug_traces[uid] = [
                {
                    "step":      s.iteration,
                    "action":    s.action,
                    "args":      s.args,
                    "thought":   s.thought[:120] if s.thought else "",
                    "ok":        s.success,
                    "obs":       s.observation[:200] if s.observation else "",
                    "replanned": s.replanned,
                }
                for s in getattr(result, "steps", [])
            ]
            # Keep only last 50 traces to avoid memory growth
            if len(_debug_traces) > 50:
                oldest = min(_debug_traces.keys())
                del _debug_traces[oldest]
        except Exception:
            pass

    # FIX: Wrap remember() in try/except so any unexpected error (e.g. residual
    # mmap issues) doesn't kill the handler after the user already got their answer.
    if used_skills and len(answer) > 120 and len(text) > 20:
        try:
            await _memory.remember(
                content=f"User: '{text[:80]}' -> {answer[:120]}",
                kind="fact", tags=used_skills, importance=2,
            )
        except Exception as exc:
            logger.warning("memory.remember() failed (non-fatal): %s", exc)

    badge = ""
    if _router and _router.last_meta:
        badge = f"\n\n{_router.last_meta.badge()}"

    await send_long(update.message, answer + badge)

    for m in FILE_PATH_RE.finditer(answer):
        await auto_send_file(update.message, m.group(1))


# ── Command Handlers ──────────────────────────────────────────────────────────
async def cmd_hello(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    crown = "👑 " if is_admin(uid) else ""
    plat  = _platform_ctx.display() if _platform_ctx else "Unknown"
    msg = (
        f"👋 {crown}<b>Nexara V{config.NEXARA_VERSION}</b> — Autonomous AI Agent\n"
        f"<i>{_html.escape(plat)}</i>\n"
        f"<i>Created by {_html.escape(CREATOR)}</i>\n\n"
        "Tell me what you want done. I'll think, plan, and act.\n\n"
        "• <code>Download the latest Python release</code>\n"
        "• <code>Search for AI news and summarise it</code>\n"
        "• <code>Write a script to rename all my photos by date</code>\n"
        "• <code>Every morning at 8am, check the news and brief me</code>\n"
        "• <code>Remind me when battery drops below 15%</code>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n   = len(get_active_skills())
    msg = (
        f"🤖 <b>Nexara V{config.NEXARA_VERSION}</b> — {n} skills loaded\n"
        f"<i>by {_html.escape(CREATOR)}</i>\n\n"
        "<b>User commands</b>\n"
        "  /hello       — greeting\n"
        "  /help        — this message\n"
        "  /clear       — reset conversation\n"
        "  /memory      — search long-term memory\n"
        "  /forget      - delete memories by query\n"
        "  /history     - show conversation history\n"
        "  /downloads   — list downloaded files\n"
        "  /schedules   — recurring tasks\n"
        "  /status      — full system status\n\n"
        "<b>Admin commands</b>\n"
        "  /stats       — device snapshot\n"
        "  /tasks       — autonomous task queue\n"
        "  /run         — queue background task\n"
        "  /cancel      — cancel queued task\n"
        "  /monitors    — background condition monitors\n"
        "  /unmonitor   — remove a monitor\n"
        "  /llm         — LLM router status\n"
        "  /switchmodel — switch provider model + set primary\n"
        "  /update      — OTA self-update\n\n"
        "<b>Just chat</b> — I pick the right skill automatically."
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _memory.clear_history(update.effective_user.id)
    await update.message.reply_text("🧹 Conversation history cleared.")

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args) if context.args else ""
    text  = (await _memory.skill_recall(query=query) if query
             else await _memory.relevant_context("", limit=10))
    await send_long(update.message, text or "No memories stored yet.")

async def cmd_downloads(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dl_dir = Path.home() / "nexara_downloads"
    if not dl_dir.exists():
        await update.message.reply_text("📁 No downloads yet.")
        return
    files = sorted(
        [f for f in dl_dir.rglob("*") if f.is_file()],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if not files:
        await update.message.reply_text("📁 Downloads folder is empty.")
        return
    lines = [f"📁 <code>{_html.escape(str(dl_dir))}</code>  ({len(files)} files)\n"]
    for f in files[:50]:
        size   = f.stat().st_size
        sz_str = f"{size/1024:.0f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
        lines.append(f"• <code>{_html.escape(f.name)}</code>  {sz_str}")
    if len(files) > 50:
        lines.append(f"\n…and {len(files) - 50} more")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_long(update.message, _scheduler.list_jobs())

@admin_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ctx    = _platform_ctx
    uptime = int(time.time() - _startup_time)
    h, rem = divmod(uptime, 3600)
    m, s   = divmod(rem, 60)

    mem_stats = "N/A"
    try:
        import sqlite3
        def _ms():
            with sqlite3.connect(str(Path.home() / ".nexara" / "memory.db")) as c:
                facts = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                convs = c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                size  = Path.home() / ".nexara" / "memory.db"
                kb    = size.stat().st_size // 1024 if size.exists() else 0
                return facts, convs, kb
        facts, convs, kb = await asyncio.to_thread(_ms)
        mem_stats = f"{facts} memories · {convs} conv turns · {kb} KB"
    except Exception:
        pass

    from agent.memory import _EMBEDDINGS_DISABLED
    embed_mode = "FTS (overlayfs — no mmap)" if _EMBEDDINGS_DISABLED else "semantic (sentence-transformers)"

    tok     = _router.token_usage() if _router else {}
    tok_str = "  ".join(f"{p}: {n}/min" for p, n in tok.items() if n > 0) or "idle"

    skills_loaded = len(get_active_skills())
    sched_count   = len(_scheduler._jobs)    if _scheduler else 0
    running_jobs  = len(_scheduler._running) if _scheduler else 0
    mon_count     = len(_monitor._jobs)      if _monitor   else 0
    primary       = getattr(config, "PRIMARY_PROVIDER", "groq")

    lines = [
        f"📊 <b>Nexara V{config.NEXARA_VERSION} — Status</b>\n",
        f"🖥️  Platform   : {_html.escape(ctx.display() if ctx else 'unknown')}",
        f"⏱️  Uptime     : {h}h {m}m {s}s",
        f"🧠 Skills     : {skills_loaded} loaded",
        f"💾 Memory     : {mem_stats}",
        f"🔍 Search     : {embed_mode}",
        f"⭐ Primary    : {primary}",
        f"🔥 Token use  : {tok_str}",
        f"⏰ Schedules  : {sched_count} active ({running_jobs} running)",
        f"📡 Monitors   : {mon_count} active",
        f"🏗️  Creator    : {_html.escape(CREATOR)}",
    ]
    if _router and _router.last_meta:
        lines.append(f"📨 Last resp  : {_html.escape(_router.last_meta.badge())}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

@admin_only
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r1 = await execute_skill("device_stats", {})
    r2 = await execute_skill("system_info",  {})
    combined = "\n\n".join(filter(None, [
        str(r1) if getattr(r1, "success", False) else "",
        str(r2) if getattr(r2, "success", False) else "",
    ]))
    if combined.strip():
        await send_long(update.message, combined)
        return
    try:
        import psutil
        cpu  = psutil.cpu_percent(interval=0.5)
        ram  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        boot_s  = time.time() - psutil.boot_time()
        bh, br  = divmod(int(boot_s), 3600)
        bm, bs  = divmod(br, 60)
        lines = [
            "📊 <b>System Stats</b>\n",
            f"🔥 CPU    : {cpu}%",
            f"💾 RAM    : {ram.percent:.1f}%  —  {ram.used//1024//1024} MB / {ram.total//1024//1024} MB",
            f"💿 Disk   : {disk.percent:.1f}%  —  {disk.used//1024//1024//1024:.1f} GB / {disk.total//1024//1024//1024:.1f} GB",
            f"⏱ Uptime : {bh}h {bm}m {bs}s",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    except Exception as exc:
        await update.message.reply_text(f"Stats unavailable: {exc}")

@admin_only
async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_long(update.message, _planner.list_tasks())

@admin_only
async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    goal = " ".join(context.args or [])
    if not goal:
        await update.message.reply_text("Usage: <code>/run &lt;goal&gt;</code>", parse_mode="HTML")
        return
    task = await _planner.submit(goal)
    await update.message.reply_text(
        f"🚀 Task queued <code>{task.task_id}</code>\n<i>{_html.escape(goal)}</i>",
        parse_mode="HTML",
    )

@admin_only
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_id = " ".join(context.args or []).strip()
    if not task_id:
        await update.message.reply_text("Usage: <code>/cancel &lt;task_id&gt;</code>", parse_mode="HTML")
        return
    await send_long(update.message, await _planner.cancel(task_id))

@admin_only
async def cmd_monitors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_long(update.message, _monitor.list_jobs())

@admin_only
async def cmd_unmonitor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job_id = " ".join(context.args or []).strip()
    if not job_id:
        await update.message.reply_text("Usage: <code>/unmonitor &lt;job_id&gt;</code>", parse_mode="HTML")
        return
    await send_long(update.message, await _monitor.unregister_job(job_id))

@admin_only
async def cmd_llm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_long(update.message, _router.status())

async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args or []).strip()
    if not query:
        await update.message.reply_text("Usage: <code>/forget &lt;query&gt;</code>", parse_mode="HTML")
        return
    matches = await _memory.recall(query=query, limit=20, min_importance=1)
    if not matches:
        await update.message.reply_text(
            f"No memories found matching <code>{_html.escape(query)}</code>.", parse_mode="HTML"
        )
        return
    import sqlite3
    ids = [e.id for e in matches]
    def _delete():
        with sqlite3.connect(str(_memory._db)) as conn:
            conn.execute(f"DELETE FROM memories WHERE id IN ({','.join('?'*len(ids))})", ids)
            conn.execute(f"DELETE FROM memories_fts WHERE rowid IN ({','.join('?'*len(ids))})", ids)
    await asyncio.to_thread(_delete)
    lines = [
        f"🗑️ Deleted {len(ids)} memor{'y' if len(ids)==1 else 'ies'} "
        f"matching <code>{_html.escape(query)}</code>:\n"
    ]
    for e in matches[:8]:
        lines.append(f"• [{e.kind}] {_html.escape(e.content[:60])}")
    if len(matches) > 8:
        lines.append(f"  …and {len(matches)-8} more")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ── /history command ─────────────────────────────────────────────────────────

async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the last N conversation turns for this user."""
    uid   = update.effective_user.id
    limit = 10
    if context.args:
        try: limit = max(1, min(int(context.args[0]), 30))
        except ValueError: pass

    rows = await _memory.load_history(uid, limit=limit)
    if not rows:
        await update.message.reply_text("No conversation history yet.")
        return

    lines = [f"💬 <b>Last {len(rows)} turns</b>\n"]
    for r in rows:
        role  = r.get("role", "?")
        parts = r.get("parts", [])
        text  = " ".join(parts) if isinstance(parts, list) else str(parts)
        icon  = "👤" if role == "user" else "🤖"
        lines.append(f"{icon} <i>{_html.escape(text[:120])}{'…' if len(text) > 120 else ''}</i>")

    await send_long(update.message, "\n".join(lines))


# ── /debug command ────────────────────────────────────────────────────────────

@admin_only
async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the last ReAct trace — what skills were called, what they returned."""
    uid   = update.effective_user.id
    trace = _debug_traces.get(uid)

    if not trace:
        await update.message.reply_text(
            "No debug trace yet — send an agent-mode message first."
        )
        return

    lines = [f"🔬 <b>Last ReAct trace</b>  ({len(trace)} step{'s' if len(trace)!=1 else ''})\n"]

    for s in trace:
        status = "✅" if s["ok"] else "❌"
        replay = " 🔁" if s["replanned"] else ""
        lines.append(
            f"<b>Step {s['step']}</b> {status}{replay}  "
            f"<code>{_html.escape(s['action'])}</code>"
        )
        if s.get("thought"):
            lines.append(f"  💭 {_html.escape(s['thought'])}")
        if s.get("args"):
            import json as _json
            args_str = _json.dumps(s["args"], ensure_ascii=False)[:120]
            lines.append(f"  📥 args: <code>{_html.escape(args_str)}</code>")
        if s.get("obs"):
            obs = s["obs"]
            lines.append(f"  📤 {_html.escape(obs[:150])}{'…' if len(obs) > 150 else ''}")
        lines.append("")

    # Show which skills the classifier selected
    if _router and _router.last_meta:
        lines.append(f"📊 Provider: {_html.escape(_router.last_meta.badge())}")

    await send_long(update.message, "\n".join(lines))


# ── /switchmodel panel ────────────────────────────────────────────────────────
# Paginated model list: 10 per page so NVIDIA 100+ models all work.
# Page state in callback_data: sm:page:{provider}:{page}
# Model selection uses 8-char MD5 hash: sm:k:{key}  (safe vs 64-byte limit)

MODELS_PER_PAGE = 10

# Per-provider model cache so pagination doesn't re-fetch on every page turn
_model_cache: dict[str, list[str]] = {}


def _main_panel_keyboard() -> InlineKeyboardMarkup:
    active  = _router.active_models()
    primary = getattr(config, "PRIMARY_PROVIDER", "groq")
    def _label(p, icon):
        star  = " ⭐" if p == primary else ""
        mdl   = active.get(p, "?")
        short = mdl if len(mdl) <= 20 else mdl[:9] + "…" + mdl[-9:]
        return f"{icon} {p.capitalize()}{star}\n[{short}]"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(_label("groq",   "⚡"), callback_data="sm:groq"),
            InlineKeyboardButton(_label("gemini", "🔵"), callback_data="sm:gemini"),
        ],
        [
            InlineKeyboardButton(_label("nvidia", "🟢"), callback_data="sm:nvidia"),
            InlineKeyboardButton(_label("ollama", "📡"), callback_data="sm:ollama"),
        ],
    ])


def _model_page_keyboard(provider: str, models: list[str], page: int) -> InlineKeyboardMarkup:
    current     = _router.active_models().get(provider, "")
    total_pages = max(1, (len(models) + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
    page        = max(0, min(page, total_pages - 1))
    page_models = models[page * MODELS_PER_PAGE : (page + 1) * MODELS_PER_PAGE]

    buttons: list[list] = []
    row:     list       = []
    for mdl in page_models:
        key   = _model_key(provider, mdl)
        tick  = "✅ " if mdl == current else ""
        label = f"{tick}{mdl}" if len(mdl) <= 26 else f"{tick}{mdl[:11]}…{mdl[-13:]}"
        row.append(InlineKeyboardButton(label, callback_data=f"sm:k:{key}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    # Navigation row (only shown when there are multiple pages)
    if total_pages > 1:
        nav: list = []
        if page > 0:
            nav.append(InlineKeyboardButton(
                "◀ Prev", callback_data=f"sm:page:{provider}:{page-1}"
            ))
        nav.append(InlineKeyboardButton(
            f"· {page+1}/{total_pages} ·", callback_data="sm:noop"
        ))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(
                "Next ▶", callback_data=f"sm:page:{provider}:{page+1}"
            ))
        buttons.append(nav)

    # Action row
    primary = getattr(config, "PRIMARY_PROVIDER", "groq")
    star    = "⭐" if provider == primary else "☆"
    buttons.append([
        InlineKeyboardButton(f"{star} Set as Primary", callback_data=f"sm:primary:{provider}"),
        InlineKeyboardButton("← Back",                callback_data="sm:back"),
    ])

    return InlineKeyboardMarkup(buttons)


async def _show_model_page(query, provider: str, models: list[str], page: int):
    current     = _router.active_models().get(provider, "")
    primary     = getattr(config, "PRIMARY_PROVIDER", "groq")
    total_pages = max(1, (len(models) + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
    icons       = {"groq": "⚡", "gemini": "🔵", "nvidia": "🟢", "ollama": "📡"}

    header = (
        f"{icons.get(provider,'🤖')} <b>{provider.capitalize()} Models</b>  "
        f"<i>({len(models)} total)</i>\n"
        f"<i>Active: <code>{_html.escape(current)}</code>  |  "
        f"Primary: <code>{_html.escape(primary)}</code></i>"
    )
    await query.edit_message_text(
        header,
        reply_markup=_model_page_keyboard(provider, models, page),
        parse_mode="HTML",
    )


@admin_only
async def cmd_switchmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔀 <b>Switch Model / Set Primary</b>\nChoose a provider:",
        reply_markup=_main_panel_keyboard(),
        parse_mode="HTML",
    )


async def cb_switchmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not is_admin(query.from_user.id):
        await query.answer("⛔ Admin only.", show_alert=True)
        return

    data = query.data

    # No-op (page counter button)
    if data == "sm:noop":
        return

    # Provider selected → fetch models, show page 0
    if data in ("sm:groq", "sm:gemini", "sm:nvidia", "sm:ollama"):
        provider = data.split(":")[1]
        await query.edit_message_text(
            f"⏳ Fetching <code>{provider}</code> models…", parse_mode="HTML"
        )
        fetchers = {
            "groq":   _router.fetch_groq_models,
            "gemini": _router.fetch_gemini_models,
            "nvidia": _router.fetch_nvidia_models,
            "ollama": _router.fetch_ollama_models,
        }
        try:
            models = await asyncio.wait_for(fetchers[provider](), timeout=20)
        except asyncio.TimeoutError:
            await query.edit_message_text(
                f"⏱ Timed out fetching <code>{provider}</code> models.", parse_mode="HTML"
            )
            return
        except Exception as exc:
            await query.edit_message_text(
                f"❌ Error: {_html.escape(str(exc))}", parse_mode="HTML"
            )
            return

        if not models:
            tips = {
                "ollama": "Ollama isn't running — it's a local service.",
                "nvidia": "Set <code>NVIDIA_API_KEY</code> in your <code>.env</code>.",
                "groq":   "Check <code>GROQ_API_KEY</code> in your <code>.env</code>.",
                "gemini": "Check <code>LLM_API_KEY</code> in your <code>.env</code>.",
            }
            await query.edit_message_text(
                f"❌ No models found for <code>{provider}</code>.\n{tips.get(provider,'')}",
                parse_mode="HTML",
            )
            return

        # Sort: current model first, then alphabetical
        current       = _router.active_models().get(provider, "")
        sorted_models = sorted(models, key=lambda m: (0 if m == current else 1, m))
        _model_cache[provider] = sorted_models
        for m in sorted_models:
            _model_key(provider, m)  # pre-register all hash keys

        await _show_model_page(query, provider, sorted_models, 0)
        return

    # Page navigation
    if data.startswith("sm:page:"):
        parts    = data.split(":")
        provider = parts[2]
        page     = int(parts[3])
        models   = _model_cache.get(provider, [])
        if not models:
            await query.answer(
                "Model list expired — tap the provider button again.", show_alert=True
            )
            return
        await _show_model_page(query, provider, models, page)
        return

    # Model selected via hash key
    if data.startswith("sm:k:"):
        key   = data[5:]
        model = _model_from_key(key)
        if model is None:
            await query.answer(
                "Model ref expired — open /switchmodel again.", show_alert=True
            )
            return
        provider = next(
            (p for p in ("groq", "gemini", "nvidia", "ollama")
             if _model_key(p, model) == key),
            None,
        )
        if provider is None:
            await query.answer("Could not determine provider.", show_alert=True)
            return
        result = _router.switch_model(provider, model)
        await query.edit_message_text(
            f"{_md_to_html(result)}\n\n🔀 <b>Switch Model / Set Primary</b>",
            reply_markup=_main_panel_keyboard(),
            parse_mode="HTML",
        )
        return

    # Legacy full-name fallback
    if data.startswith("sm:set:"):
        _, _, provider, *model_parts = data.split(":")
        model  = ":".join(model_parts)
        result = _router.switch_model(provider, model)
        await query.edit_message_text(
            f"{_md_to_html(result)}\n\n🔀 <b>Switch Model / Set Primary</b>",
            reply_markup=_main_panel_keyboard(),
            parse_mode="HTML",
        )
        return

    # Set primary
    if data.startswith("sm:primary:"):
        provider = data.split(":")[2]
        result   = _router.set_primary(provider)
        config.PRIMARY_PROVIDER = provider
        _persist_env("PRIMARY_PROVIDER", provider)
        await query.edit_message_text(
            f"{_md_to_html(result)}\n\n🔀 <b>Switch Model / Set Primary</b>",
            reply_markup=_main_panel_keyboard(),
            parse_mode="HTML",
        )
        return

    # Back
    if data == "sm:back":
        await query.edit_message_text(
            "🔀 <b>Switch Model / Set Primary</b>\nChoose a provider:",
            reply_markup=_main_panel_keyboard(),
            parse_mode="HTML",
        )

def _persist_env(key: str, value: str):
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        lines     = env_path.read_text().splitlines()
        updated   = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key}={value}")
        env_path.write_text("\n".join(new_lines) + "\n")
    except Exception as exc:
        logger.warning("Could not persist %s to .env: %s", key, exc)


@admin_only
async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check for agent updates, then skills update via git pull."""
    msg = await update.message.reply_text("🔄 Checking for updates…")

    # Step 1: Agent file update via agent_manifest.json
    if _updater and config.AGENT_REPO_URL:
        try:
            result = await _updater.check_and_apply(force=True)
            # If we reach here, no restart happened (nothing changed)
            await msg.edit_text(f"🔄 {result}\n\nRunning git pull…")
        except Exception as exc:
            await msg.edit_text(f"⚠️ Agent update check failed: {exc}\n\nRunning git pull…")
    else:
        await msg.edit_text("ℹ️ AGENT_REPO_URL not set — skipping agent file update.\n\nRunning git pull…")

    # Step 2: git pull + pip deps (existing behaviour)
    script = Path(__file__).parent / "update.sh"
    subprocess.Popen(
        ["bash", str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
    )


# ── Startup ───────────────────────────────────────────────────────────────────
async def post_init(app: Application):
    global _memory, _router, _react, _planner, _monitor, _scheduler
    global _bot_app, _platform_ctx, _skill_loader, _TOOLS, _startup_time

    _bot_app      = app
    _startup_time = time.time()

    _platform_ctx = platform_mod.detect()
    logger.info("Platform: %s", _platform_ctx.display())
    config.TERMUX_API_AVAILABLE = _platform_ctx.termux_api

    _memory = AgentMemory()

    class _MemBridge:
        async def remember(self, content="", kind="fact", importance=3, **_):
            return await _memory.skill_remember(content=content, kind=kind, importance=importance)
        async def recall(self, query="", kind=None, **_):
            return await _memory.skill_recall(query=query, kind=kind)
    set_memory_bridge(_MemBridge())

    _skill_loader = SkillLoader(
        platform_ctx=_platform_ctx,
        channel=config.SKILL_CHANNEL,
        warehouse_url=config.SKILL_WAREHOUSE_URL,
    )
    loaded = await _skill_loader.load()
    set_active_skills(loaded)
    logger.info("Loaded %d skills: %s", len(loaded), sorted(loaded))

    _TOOLS = gemini_tools(active_skills=loaded)
    build_system_prompt(force=True)

    _router = LLMRouter(config)
    _router.on_switch_cb = telegram_alert

    # Agent auto-updater
    global _updater
    if config.AGENT_REPO_URL:
        _updater = AgentUpdater(repo_url=config.AGENT_REPO_URL, alert_cb=telegram_alert)
        logger.info("AgentUpdater ready — repo: %s", config.AGENT_REPO_URL)
        # Non-blocking startup check — runs in background, won't delay bot start
        asyncio.create_task(_updater.check_and_apply(force=False))
    else:
        logger.info("AgentUpdater disabled — AGENT_REPO_URL not set")

    _react = ReactLoop(router=_router, skill_exec=execute_skill)

    _planner = AgentPlanner(
        react_loop=_react, memory=_memory,
        system_prompt_fn=build_system_prompt,
        alert_cb=telegram_alert, tools=_TOOLS,
        active_skills_fn=get_active_skills,
    )
    await _planner.start()

    async def _sched_submit(goal: str, trigger: str):
        await _planner.submit_proactive(goal, trigger)

    _scheduler = NaturalScheduler(submit_fn=_sched_submit)
    await _scheduler.start()

    class _SchedBridge:
        async def schedule(self, goal="", schedule="", task_id=None, **_):
            return await _scheduler.schedule(goal, schedule, task_id)
        def list_jobs(self): return _scheduler.list_jobs()
        async def cancel(self, task_id="", **_): return await _scheduler.cancel(task_id)
    set_scheduler_bridge(_SchedBridge())

    _monitor = MonitorTaskManager(
        skill_exec=execute_skill,
        alert_cb=telegram_alert,
        planner_submit=_planner.submit_proactive,
    )
    await _monitor.start()

    if _platform_ctx.termux_api:
        await _monitor.register_job(
            job_id="battery_low",
            description="Alert when battery below 15%",
            skill_name="battery", skill_kwargs={},
            condition="< 15", interval_seconds=300,
            alert_only=False,
            action_goal="Battery critically low. Enable power-saving mode.",
            cooldown_s=600,
        )

    await _monitor.register_job(
        job_id="disk_high",
        description="Alert when disk usage above 90%",
        skill_name="system_info", skill_kwargs={},
        condition="> 90", interval_seconds=3600,
        alert_only=False,
        action_goal="Disk above 90%. Remove large unnecessary files from ~/nexara_downloads/.",
        cooldown_s=3600,
    )

    await app.bot.set_my_commands([
        BotCommand("hello",       "Wake up Nexara"),
        BotCommand("help",        "Commands and skills"),
        BotCommand("clear",       "Reset conversation"),
        BotCommand("memory",      "Search long-term memory"),
        BotCommand("forget",      "Delete memories by query"),
        BotCommand("history",     "Show conversation history"),
        BotCommand("debug",       "Last ReAct trace (admin)"),
        BotCommand("downloads",   "Downloaded files"),
        BotCommand("schedules",   "Recurring tasks"),
        BotCommand("status",      "Full system status"),
        BotCommand("stats",       "Device snapshot"),
        BotCommand("tasks",       "Autonomous task queue"),
        BotCommand("run",         "Queue background task"),
        BotCommand("cancel",      "Cancel a queued task"),
        BotCommand("monitors",    "Background monitors"),
        BotCommand("unmonitor",   "Remove a monitor"),
        BotCommand("llm",         "LLM router status"),
        BotCommand("switchmodel", "Switch model / set primary"),
        BotCommand("update",      "OTA self-update"),
    ])

    from agent.memory import _EMBEDDINGS_DISABLED
    embed_note = " · FTS mode (overlayfs)" if _EMBEDDINGS_DISABLED else ""
    n_skills   = len(get_active_skills())
    msg = (
        f"✅ <b>Nexara V{config.NEXARA_VERSION}</b> online\n"
        f"{_html.escape(_platform_ctx.display())}\n"
        f"🧠 {n_skills} skills  ·  "
        f"⭐ Primary: <code>{getattr(config,'PRIMARY_PROVIDER','groq')}</code>  ·  "
        f"💾 <code>{_html.escape(str(_memory._db))}</code>{embed_note}"
    )
    try:
        await app.bot.send_message(chat_id=config.ADMIN_ID, text=msg, parse_mode="HTML")
    except Exception as exc:
        logger.warning("Startup msg failed: %s", exc)

    logger.info("Nexara V%s ready | %d skills | %s | embed=%s",
                config.NEXARA_VERSION, n_skills, _platform_ctx.display(),
                "FTS" if _EMBEDDINGS_DISABLED else "semantic")


async def post_shutdown(app: Application):
    if _monitor:   await _monitor.stop()
    if _planner:   await _planner.stop()
    if _scheduler: await _scheduler.stop()


# ── Entry ─────────────────────────────────────────────────────────────────────
def main():
    LOG_DIR.mkdir(exist_ok=True)
    load_password()

    app = (
        Application.builder()
        .token(config.TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("hello",       cmd_hello))
    app.add_handler(CommandHandler("start",       cmd_hello))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("clear",       cmd_clear))
    app.add_handler(CommandHandler("memory",      cmd_memory))
    app.add_handler(CommandHandler("forget",      cmd_forget))
    app.add_handler(CommandHandler("history",     cmd_history))
    app.add_handler(CommandHandler("debug",       cmd_debug))
    app.add_handler(CommandHandler("downloads",   cmd_downloads))
    app.add_handler(CommandHandler("schedules",   cmd_schedules))
    app.add_handler(CommandHandler("status",      cmd_status))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("tasks",       cmd_tasks))
    app.add_handler(CommandHandler("run",         cmd_run))
    app.add_handler(CommandHandler("cancel",      cmd_cancel))
    app.add_handler(CommandHandler("monitors",    cmd_monitors))
    app.add_handler(CommandHandler("unmonitor",   cmd_unmonitor))
    app.add_handler(CommandHandler("llm",         cmd_llm))
    app.add_handler(CommandHandler("switchmodel", cmd_switchmodel))
    app.add_handler(CommandHandler("update",      cmd_update))
    app.add_handler(CallbackQueryHandler(cb_switchmodel, pattern="^sm:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
