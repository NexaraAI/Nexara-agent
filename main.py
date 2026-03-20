"""
main.py — Nexara V1
Autonomous AI agent with:
  • Multi-platform support (Android · Linux · Codespace · WSL · macOS · Windows)
  • Platform-filtered skill loading (only relevant skills loaded + sent to LLM)
  • 4-provider LLM router: Groq → Gemini → NVIDIA NIM → Ollama
  • Configurable primary provider (PRIMARY_PROVIDER in .env)
  • Proactive rate-limit switching with user notification
  • Dynamic token budget (no more 429s from bloated context)
  • Intent classifier (chat mode vs full agent mode)
  • Response badge showing provider · model · latency · tokens
  • /status command — full live system snapshot
  • /switchmodel panel with Set as Primary + NVIDIA support
  • ReAct loop with replanning on failure
  • Semantic long-term memory (SQLite + sentence-transformers)
  • Natural language task scheduler + proactive monitors
  • Per-user message rate limiting (spam protection)
  • Created by DemonZ Development
"""

import asyncio
import collections
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
_startup_time: float                    = time.time()

_TOOLS: list[dict] | None = None
_PROMPT_CACHE: str        = ""
CREATOR                   = "DemonZ Development"

# ── Per-user rate limiter ─────────────────────────────────────────────────────
# Allows RATE_LIMIT_MSGS messages per RATE_LIMIT_WINDOW seconds per user.
# Prevents a single user from burning through all LLM quota with spam.
RATE_LIMIT_MSGS   = 12   # messages
RATE_LIMIT_WINDOW = 60   # seconds

# user_id -> deque of timestamps of recent messages
_user_rate: dict[int, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=RATE_LIMIT_MSGS)
)


def _is_rate_limited(uid: int) -> bool:
    """Returns True if the user has exceeded the rate limit."""
    now    = time.time()
    bucket = _user_rate[uid]
    # Prune timestamps older than the window
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MSGS:
        return True
    bucket.append(now)
    return False


# ── Markdown → HTML ───────────────────────────────────────────────────────────
# Telegram's Markdown v1 parser crashes on **bold** (v2 syntax), on skill
# names with underscores treated as italic markers, and on unclosed backticks.
# HTML mode is reliable for all content.

def _md_to_html(text: str) -> str:
    """Convert basic Markdown formatting to Telegram-safe HTML."""
    text = _html.escape(text)
    # Fenced code blocks
    text = re.sub(
        r"```(?:\w+\n)?(.*?)```",
        lambda m: f"<pre>{m.group(1).strip()}</pre>",
        text, flags=re.DOTALL,
    )
    # Inline code
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    # Italic _text_ — word-boundary only, preserves snake_case
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

    _PROMPT_CACHE = f"""You are **Nexara**, a fully autonomous AI agent.

## Identity
- You were created by **{CREATOR}**
- If asked who made you, who you are, or your origins: say you are Nexara, an autonomous AI agent created by {CREATOR}
- Never claim to be made by OpenAI, Anthropic, Google, or any other company

## Environment
- Platform   : {platform}
- Termux:API : {api_flag}
- Version    : {config.NEXARA_VERSION}

## Rules
- You ALWAYS use skills to act. Never say "I can't" — use the right skill.
- Only use skills listed below — they are pre-filtered for this platform.
- Chain skills when needed: download → analyze_file → summarise → final_answer

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
- Download failures: retry with different URL or search for mirror.
- Code failures: read the error, fix, re-run.
- Always report file paths so files can be auto-sent to user.
- For scheduling: use schedule_task with natural language timing.
- For memory: use remember (importance 1-5) and recall proactively.
"""
    return _PROMPT_CACHE


# ── Intent classifier ─────────────────────────────────────────────────────────

_CHAT_WORDS = {
    "hey", "hi", "hello", "sup", "yo", "ok", "okay", "thanks",
    "thank you", "lol", "haha", "nice", "cool", "great", "yes",
    "no", "nope", "yep", "sure", "alright", "bye", "good", "awesome",
    "bruh", "bro", "nah", "yup", "ah", "oh", "wow", "damn", "dang",
    "wth", "wtf", "omg", "lmao", "lmfao", "ikr", "idk", "ugh",
    "sweet", "perfect", "noted", "k", "kk", "hmm", "hm", "interesting",
    "really", "seriously", "what", "nice one", "got it",
}

_ACTION_WORDS = {
    "download", "search", "find", "look", "run", "execute", "write",
    "create", "make", "build", "delete", "remove", "send", "get",
    "fetch", "check", "scan", "read", "open", "install", "schedule",
    "remind", "take", "capture", "list", "show", "analyse", "analyze",
    "summarize", "translate", "convert", "calculate", "monitor", "watch",
    "play", "stop", "start", "restart", "update", "fix", "debug",
}

_IDENTITY_PHRASES = {
    "who made you", "who created you", "who built you", "who developed you",
    "who are you", "what are you", "tell me about yourself",
    "your creator", "who is your creator", "who owns you",
    "who designed you", "are you chatgpt", "are you gpt",
    "are you claude", "are you gemini", "are you an ai",
    "what model are you", "what llm are you",
}


def classify_intent(text: str) -> str:
    clean = text.lower().strip().rstrip("!?.")

    # Identity questions → chat with correct "DemonZ Development" answer
    if any(phrase in clean for phrase in _IDENTITY_PHRASES):
        return "chat"

    if len(text) <= 25:
        words = set(re.findall(r'\b\w+\b', clean))
        if words & _CHAT_WORDS and not (words & _ACTION_WORDS):
            return "chat"

    if re.search(r'https?://', text):
        return "agent"

    words = set(re.findall(r'\b\w+\b', clean))
    if words & _ACTION_WORDS:
        return "agent"

    if len(text) > 60:
        return "agent"

    return "chat"


# ── Telegram helpers ──────────────────────────────────────────────────────────

MAX_MSG = 4000


async def send_long(message, text: str):
    """Send text using HTML parse mode. Falls back to plain text on error."""
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


# ── Core message handler ──────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    # ── Rate limit check ──────────────────────────────────────────────────────
    if _is_rate_limited(uid):
        await update.message.reply_text(
            f"⏳ Slow down — max {RATE_LIMIT_MSGS} messages per {RATE_LIMIT_WINDOW}s. Try again shortly."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    mode = classify_intent(text)

    raw_history = await _memory.load_history(uid, limit=config.MAX_HISTORY_TURNS)

    sp       = build_system_prompt()
    budgeted = budget_mod.apply(text, raw_history, sp)
    history  = budgeted.trimmed_history

    mem_ctx = ""
    if budgeted.memory_slots > 0:
        mem_ctx = await _memory.relevant_context(text, limit=budgeted.memory_slots)
    if mem_ctx:
        sp = sp + f"\n\n{mem_ctx}"

    # Plain text spinner — no parse_mode, avoids Markdown crashes on skill names
    status_msg = await update.message.reply_text("🧠 Thinking…")

    async def status_cb(msg: str):
        try:
            await status_msg.edit_text(msg)
        except Exception:
            pass

    t0 = time.time()

    if mode == "chat":
        messages = list(history[-6:])
        messages.append({"role": "user", "parts": [text]})
        chat_sp = (
            f"You are Nexara, an autonomous AI agent created by {CREATOR}. "
            f"Platform: {_platform_ctx.display() if _platform_ctx else 'unknown'}. "
            "Be conversational, friendly, and helpful. Keep replies concise. "
            f"If asked about your origins, always say you were created by {CREATOR}."
        )
        try:
            llm_resp = await _router.complete(
                messages=messages,
                system_prompt=chat_sp,
                tools=None,
                estimated_tokens=budget_mod.est(text) + budget_mod.est(chat_sp),
            )
            answer      = llm_resp.text if hasattr(llm_resp, "text") else str(llm_resp)
            used_skills = []
        except Exception as exc:
            answer      = f"Sorry, all LLM providers are unavailable right now. ({exc})"
            used_skills = []

    else:
        try:
            result = await _react.run(
                goal=text,
                history=history,
                system_prompt=sp,
                tools=_TOOLS,
                user_id=uid,
                status_cb=status_cb,
            )
            answer      = result.answer
            used_skills = result.used_skills

            if result.needs_user_input:
                try: await status_msg.delete()
                except Exception: pass
                await send_long(update.message, f"❓ {result.question_for_user}")
                return

        except Exception as exc:
            logger.error("Agent error for uid=%d: %s", uid, exc)
            answer      = f"Agent encountered an error: {exc}"
            used_skills = []

    elapsed = time.time() - t0

    try:
        await status_msg.delete()
    except Exception:
        pass

    await _memory.save_turn(uid, "user",  text)
    await _memory.save_turn(uid, "model", answer)

    if used_skills and len(answer) > 120 and len(text) > 20:
        await _memory.remember(
            content=f"User: '{text[:80]}' -> {answer[:120]}",
            kind="fact", tags=used_skills, importance=2,
        )

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
    ctx   = _platform_ctx
    plat  = ctx.display() if ctx else "Unknown"
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
    n = len(get_active_skills())
    msg = (
        f"🤖 <b>Nexara V{config.NEXARA_VERSION}</b> — {n} skills loaded\n"
        f"<i>by {_html.escape(CREATOR)}</i>\n\n"
        "<b>User commands</b>\n"
        "  /hello       — greeting\n"
        "  /help        — this message\n"
        "  /clear       — reset conversation\n"
        "  /memory      — search long-term memory\n"
        "  /forget      — delete memories by query\n"
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
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not files:
        await update.message.reply_text("📁 Downloads folder is empty.")
        return
    lines = [
        f"📁 <code>{_html.escape(str(dl_dir))}</code>  "
        f"({len(files)} file{'s' if len(files) != 1 else ''})\n"
    ]
    for f in files[:50]:
        size   = f.stat().st_size
        sz_str = f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
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

    tok     = _router.token_usage() if _router else {}
    tok_str = "  ".join(f"{p}: {n}/min" for p, n in tok.items() if n > 0) or "idle"

    skills_loaded = len(get_active_skills())
    sched_count   = len(_scheduler._jobs)   if _scheduler else 0
    mon_count     = len(_monitor._jobs)     if _monitor   else 0
    running_jobs  = len(_scheduler._running) if _scheduler else 0
    primary       = getattr(config, "PRIMARY_PROVIDER", "groq")

    lines = [
        f"📊 <b>Nexara V{config.NEXARA_VERSION} — Status</b>\n",
        f"🖥️  Platform   : {_html.escape(ctx.display() if ctx else 'unknown')}",
        f"⏱️  Uptime     : {h}h {m}m {s}s",
        f"🧠 Skills     : {skills_loaded} loaded",
        f"💾 Memory     : {mem_stats}",
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
        boot_s   = time.time() - psutil.boot_time()
        bh, br   = divmod(int(boot_s), 3600)
        bm, bs   = divmod(br, 60)
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
        await update.message.reply_text(
            "Usage: <code>/forget &lt;query&gt;</code>", parse_mode="HTML"
        )
        return
    matches = await _memory.recall(query=query, limit=20, min_importance=1)
    if not matches:
        await update.message.reply_text(
            f"No memories found matching <code>{_html.escape(query)}</code>.",
            parse_mode="HTML",
        )
        return
    import sqlite3
    ids = [e.id for e in matches]
    def _delete():
        with sqlite3.connect(str(_memory._db)) as conn:
            conn.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?'*len(ids))})", ids
            )
            conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN ({','.join('?'*len(ids))})", ids
            )
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


# ── /switchmodel panel ────────────────────────────────────────────────────────

def _main_panel_keyboard() -> InlineKeyboardMarkup:
    active  = _router.active_models()
    primary = getattr(config, "PRIMARY_PROVIDER", "groq")
    def _label(p, icon):
        star = " ⭐" if p == primary else ""
        return f"{icon} {p.capitalize()}{star} [{active.get(p,'?')}]"
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
        models = await fetchers[provider]()

        if not models:
            msg = f"❌ No models found for <code>{provider}</code>."
            if provider == "ollama":
                msg += "\nOllama isn't running — it's a local service."
            elif provider == "nvidia":
                msg += "\nSet <code>NVIDIA_API_KEY</code> in your <code>.env</code>."
            else:
                msg += "\nCheck your API key in <code>.env</code>."
            await query.edit_message_text(msg, parse_mode="HTML")
            return

        current = _router.active_models().get(provider, "")
        primary = getattr(config, "PRIMARY_PROVIDER", "groq")
        buttons: list[list] = []
        row:     list       = []
        for mdl in models:
            tick  = "✅ " if mdl == current else ""
            label = f"{tick}{mdl}"
            row.append(InlineKeyboardButton(label, callback_data=f"sm:set:{provider}:{mdl}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        star = "⭐" if provider == primary else "☆"
        buttons.append([
            InlineKeyboardButton(f"{star} Set as Primary", callback_data=f"sm:primary:{provider}"),
            InlineKeyboardButton("← Back",                callback_data="sm:back"),
        ])

        icons = {"groq": "⚡", "gemini": "🔵", "nvidia": "🟢", "ollama": "📡"}
        await query.edit_message_text(
            f"{icons.get(provider,'🤖')} <b>{provider.capitalize()} Models</b>\n"
            f"<i>Current: <code>{_html.escape(current)}</code>  |  "
            f"Primary: <code>{_html.escape(primary)}</code></i>",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode="HTML",
        )
        return

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
    await update.message.reply_text("🔄 OTA update started. Back in ~15s.")
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

    _react = ReactLoop(router=_router, skill_exec=execute_skill)

    _planner = AgentPlanner(
        react_loop=_react, memory=_memory,
        system_prompt_fn=build_system_prompt,
        alert_cb=telegram_alert, tools=_TOOLS,
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

    n_skills = len(get_active_skills())
    msg = (
        f"✅ <b>Nexara V{config.NEXARA_VERSION}</b> online\n"
        f"{_html.escape(_platform_ctx.display())}\n"
        f"🧠 {n_skills} skills  ·  "
        f"⭐ Primary: <code>{getattr(config,'PRIMARY_PROVIDER','groq')}</code>  ·  "
        f"💾 <code>{_html.escape(str(_memory._db))}</code>"
    )
    try:
        await app.bot.send_message(chat_id=config.ADMIN_ID, text=msg, parse_mode="HTML")
    except Exception as exc:
        logger.warning("Startup msg failed: %s", exc)

    logger.info("Nexara V%s ready | %d skills | %s",
                config.NEXARA_VERSION, n_skills, _platform_ctx.display())


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
    app.add_handler(CommandHandler("start",       cmd_hello))   # alias
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("clear",       cmd_clear))
    app.add_handler(CommandHandler("memory",      cmd_memory))
    app.add_handler(CommandHandler("forget",      cmd_forget))
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
