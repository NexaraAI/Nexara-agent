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
"""

import asyncio
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

# Set after skill loading in post_init
_TOOLS: list[dict] | None = None

# ── System prompt (cached) ────────────────────────────────────────────────────
_PROMPT_CACHE: str = ""


def build_system_prompt(force: bool = False) -> str:
    global _PROMPT_CACHE
    if _PROMPT_CACHE and not force:
        return _PROMPT_CACHE

    ctx      = _platform_ctx
    platform = ctx.display() if ctx else "Unknown"
    api_flag = "ENABLED" if (ctx and ctx.termux_api) else "DISABLED"

    _PROMPT_CACHE = f"""You are **Nexara**, a fully autonomous AI agent.

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
}

_ACTION_WORDS = {
    "download", "search", "find", "look", "run", "execute", "write",
    "create", "make", "build", "delete", "remove", "send", "get",
    "fetch", "check", "scan", "read", "open", "install", "schedule",
    "remind", "take", "capture", "list", "show", "analyse", "analyze",
    "summarize", "translate", "convert", "calculate", "monitor", "watch",
    "play", "stop", "start", "restart", "update", "fix", "debug",
}


def classify_intent(text: str) -> str:
    """
    Returns 'chat' for simple conversational messages, 'agent' for action requests.
    Chat mode skips the ReAct loop → faster response, fewer tokens.
    """
    clean = text.lower().strip().rstrip("!?.")

    # Very short — check if it's a pure greeting
    if len(text) <= 20:
        words = set(re.findall(r'\b\w+\b', clean))
        if words & _CHAT_WORDS and not (words & _ACTION_WORDS):
            return "chat"

    # URL present → agent
    if re.search(r'https?://', text):
        return "agent"

    # Action keywords → agent
    words = set(re.findall(r'\b\w+\b', clean))
    if words & _ACTION_WORDS:
        return "agent"

    # Long message → agent
    if len(text) > 60:
        return "agent"

    return "chat"


# ── Telegram helpers ──────────────────────────────────────────────────────────

MAX_MSG = 4000

PROVIDER_ICONS = {
    "groq":   "⚡",
    "gemini": "🔵",
    "nvidia": "🟢",
    "ollama": "📡",
}


async def send_long(message, text: str):
    if len(text) <= MAX_MSG:
        try:
            await message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await message.reply_text(text)
        return
    for chunk in [text[i:i+MAX_MSG] for i in range(0, len(text), MAX_MSG)]:
        try:
            await message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            await message.reply_text(chunk)
        await asyncio.sleep(0.1)


async def telegram_alert(text: str):
    if _bot_app and config.ADMIN_ID:
        try:
            await _bot_app.bot.send_message(
                chat_id=config.ADMIN_ID, text=text, parse_mode="Markdown"
            )
        except Exception as exc:
            logger.error("Alert failed: %s", exc)


async def auto_send_file(message, path_str: str):
    p = Path(path_str)
    if not p.exists() or p.stat().st_size > 50 * 1024 * 1024:
        return
    try:
        sfx = p.suffix.lower()
        with open(p, "rb") as f:
            if sfx in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                await message.reply_photo(photo=f, caption=f"`{p.name}`", parse_mode="Markdown")
            elif sfx == ".mp4":
                await message.reply_video(video=f, caption=f"`{p.name}`", parse_mode="Markdown")
            elif sfx in (".mp3", ".m4a", ".ogg", ".flac", ".wav"):
                await message.reply_audio(audio=f, title=p.stem)
            else:
                await message.reply_document(document=f, caption=f"`{p.name}`", parse_mode="Markdown")
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

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    # ── Intent classification ─────────────────────────────────────────────────
    mode = classify_intent(text)

    # ── Load history ──────────────────────────────────────────────────────────
    raw_history = await _memory.load_history(uid, limit=config.MAX_HISTORY_TURNS)

    # ── Token budget ──────────────────────────────────────────────────────────
    sp     = build_system_prompt()
    budgeted = budget_mod.apply(text, raw_history, sp)
    history  = budgeted.trimmed_history

    # ── Memory context (budget-gated) ─────────────────────────────────────────
    mem_ctx = ""
    if budgeted.memory_slots > 0:
        mem_ctx = await _memory.relevant_context(text, limit=budgeted.memory_slots)
    if mem_ctx:
        sp = sp + f"\n\n{mem_ctx}"

    # ── Status message ────────────────────────────────────────────────────────
    status_msg = await update.message.reply_text("🧠 _Thinking…_", parse_mode="Markdown")

    async def status_cb(msg: str):
        try:
            await status_msg.edit_text(f"_{msg}_", parse_mode="Markdown")
        except Exception:
            pass

    t0 = time.time()

    # ── Run: chat or agent ────────────────────────────────────────────────────
    if mode == "chat":
        # Lightweight single LLM call — no ReAct, no tools
        messages = list(history[-6:])
        messages.append({"role": "user", "parts": [text]})
        chat_sp = (
            f"You are Nexara, an AI assistant. Platform: "
            f"{_platform_ctx.display() if _platform_ctx else 'unknown'}. "
            "Be concise and helpful. For complex tasks tell the user to rephrase as a command."
        )
        try:
            llm_resp = await _router.complete(
                messages=messages,
                system_prompt=chat_sp,
                tools=None,
                estimated_tokens=budget_mod.est(text) + budget_mod.est(chat_sp),
            )
            answer       = llm_resp.text if hasattr(llm_resp, "text") else str(llm_resp)
            used_skills  = []
        except Exception as exc:
            answer      = f"Sorry, all LLM providers are unavailable right now. ({exc})"
            used_skills = []
        needs_input = False

    else:
        # Full ReAct agent loop
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
            needs_input = result.needs_user_input

            if needs_input:
                try: await status_msg.delete()
                except Exception: pass
                await update.message.reply_text(
                    f"❓ {result.question_for_user}", parse_mode="Markdown"
                )
                return

        except Exception as exc:
            answer      = f"Agent encountered an error: {exc}"
            used_skills = []
            needs_input = False

    elapsed = time.time() - t0

    # ── Delete spinner ────────────────────────────────────────────────────────
    try:
        await status_msg.delete()
    except Exception:
        pass

    # ── Persist conversation ──────────────────────────────────────────────────
    await _memory.save_turn(uid, "user",  text)
    await _memory.save_turn(uid, "model", answer)

    # ── Auto-remember meaningful exchanges ───────────────────────────────────
    if used_skills and len(answer) > 120 and len(text) > 20:
        await _memory.remember(
            content=f"User: '{text[:80]}' → {answer[:120]}",
            kind="fact", tags=used_skills, importance=2,
        )

    # ── Build response + badge ────────────────────────────────────────────────
    badge = ""
    if _router and _router.last_meta:
        badge = f"\n\n{_router.last_meta.badge()}"

    await send_long(update.message, answer + badge)

    # ── Auto-send file attachments ────────────────────────────────────────────
    for m in FILE_PATH_RE.finditer(answer):
        await auto_send_file(update.message, m.group(1))


# ── Command Handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    crown = "👑 " if is_admin(uid) else ""
    ctx   = _platform_ctx
    plat  = ctx.display() if ctx else "Unknown"
    await update.message.reply_text(
        f"👋 {crown}**Nexara V{config.NEXARA_VERSION}** — Autonomous AI Agent\n"
        f"_{plat}_\n\n"
        "_Tell me what you want done. I'll think, plan, and act._\n\n"
        "• `Download the latest Python release`\n"
        "• `Search for AI news and summarise it`\n"
        "• `Write a script to rename all my photos by date`\n"
        "• `Every morning at 8am, check the news and brief me`\n"
        "• `Remind me when battery drops below 15%`",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = len(get_active_skills())
    await update.message.reply_text(
        f"🤖 **Nexara V{config.NEXARA_VERSION}** — {n} skills loaded\n\n"
        "**User commands**\n"
        "  /start       — greeting\n"
        "  /help        — this message\n"
        "  /clear       — reset conversation\n"
        "  /memory      — search long-term memory\n"
        "  /forget      — delete memories by query\n"
        "  /downloads   — list downloaded files\n"
        "  /schedules   — recurring tasks\n"
        "  /status      — full system status\n\n"
        "**Admin commands**\n"
        "  /stats       — device snapshot\n"
        "  /tasks       — autonomous task queue\n"
        "  /run         — queue background task\n"
        "  /cancel      — cancel queued task\n"
        "  /monitors    — background condition monitors\n"
        "  /unmonitor   — remove a monitor\n"
        "  /llm         — LLM router status\n"
        "  /switchmodel — switch provider model + set primary\n"
        "  /update      — OTA self-update\n\n"
        "**Just chat** — I pick the right skill automatically.",
        parse_mode="Markdown",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _memory.clear_history(update.effective_user.id)
    await update.message.reply_text("🧹 Conversation history cleared.")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args) if context.args else ""
    text  = (await _memory.skill_recall(query=query) if query
             else await _memory.relevant_context("", limit=10))
    await send_long(update.message, text or "No memories stored yet.")


async def cmd_downloads(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = await execute_skill("list_downloads", {})
    await send_long(update.message, str(r))


async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_scheduler.list_jobs(), parse_mode="Markdown")


@admin_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full live system snapshot."""
    ctx     = _platform_ctx
    uptime  = int(time.time() - _startup_time)
    h, rem  = divmod(uptime, 3600)
    m, s    = divmod(rem, 60)

    # Memory stats
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

    # Token usage
    tok = _router.token_usage() if _router else {}
    tok_str = "  ".join(f"{p}: {n}/min" for p, n in tok.items() if n > 0) or "idle"

    skills_loaded = len(get_active_skills())
    sched_count   = len(_scheduler._jobs) if _scheduler else 0
    mon_count     = len(_monitor._jobs)   if _monitor   else 0

    primary = getattr(config, "PRIMARY_PROVIDER", "groq")

    lines = [
        f"📊 **Nexara V{config.NEXARA_VERSION} — Status**\n",
        f"🖥️  Platform   : {ctx.display() if ctx else 'unknown'}",
        f"⏱️  Uptime     : {h}h {m}m {s}s",
        f"🧠 Skills     : {skills_loaded} loaded · tags: {ctx.skill_tags if ctx else '?'}",
        f"💾 Memory     : {mem_stats}",
        f"⭐ Primary    : {primary}",
        f"🔥 Token use  : {tok_str}",
        f"⏰ Schedules  : {sched_count} active",
        f"📡 Monitors   : {mon_count} active",
    ]
    if _router and _router.last_meta:
        lines.append(f"📨 Last resp  : {_router.last_meta.badge()}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@admin_only
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r1 = await execute_skill("device_stats", {})
    r2 = await execute_skill("system_info",  {})
    await send_long(update.message, f"{r1}\n\n{r2}")


@admin_only
async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_planner.list_tasks(), parse_mode="Markdown")


@admin_only
async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    goal = " ".join(context.args or [])
    if not goal:
        await update.message.reply_text("Usage: `/run <goal>`", parse_mode="Markdown")
        return
    task = await _planner.submit(goal)
    await update.message.reply_text(
        f"🚀 Task queued `{task.task_id}`\n_{goal}_", parse_mode="Markdown"
    )


@admin_only
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_id = " ".join(context.args or []).strip()
    if not task_id:
        await update.message.reply_text("Usage: `/cancel <task_id>`", parse_mode="Markdown")
        return
    await update.message.reply_text(await _planner.cancel(task_id), parse_mode="Markdown")


@admin_only
async def cmd_monitors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_monitor.list_jobs(), parse_mode="Markdown")


@admin_only
async def cmd_unmonitor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job_id = " ".join(context.args or []).strip()
    if not job_id:
        await update.message.reply_text("Usage: `/unmonitor <job_id>`", parse_mode="Markdown")
        return
    await update.message.reply_text(
        await _monitor.unregister_job(job_id), parse_mode="Markdown"
    )


@admin_only
async def cmd_llm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_router.status(), parse_mode="Markdown")


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args or []).strip()
    if not query:
        await update.message.reply_text(
            "Usage: `/forget <query>`", parse_mode="Markdown"
        )
        return
    matches = await _memory.recall(query=query, limit=20, min_importance=1)
    if not matches:
        await update.message.reply_text(
            f"No memories found matching `{query}`.", parse_mode="Markdown"
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
    lines = [f"🗑️ Deleted {len(ids)} memor{'y' if len(ids)==1 else 'ies'} matching `{query}`:\n"]
    for e in matches[:8]:
        lines.append(f"• [{e.kind}] {e.content[:60]}")
    if len(matches) > 8:
        lines.append(f"  …and {len(matches)-8} more")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


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
        "🔀 **Switch Model / Set Primary**\nChoose a provider:",
        reply_markup=_main_panel_keyboard(),
        parse_mode="Markdown",
    )


async def cb_switchmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not is_admin(query.from_user.id):
        await query.answer("⛔ Admin only.", show_alert=True)
        return

    data = query.data

    # ── Provider selected → fetch model list ──────────────────────────────────
    if data in ("sm:groq", "sm:gemini", "sm:nvidia", "sm:ollama"):
        provider = data.split(":")[1]
        await query.edit_message_text(
            f"⏳ Fetching `{provider}` models…", parse_mode="Markdown"
        )
        fetchers = {
            "groq":   _router.fetch_groq_models,
            "gemini": _router.fetch_gemini_models,
            "nvidia": _router.fetch_nvidia_models,
            "ollama": _router.fetch_ollama_models,
        }
        models = await fetchers[provider]()

        if not models:
            msg = f"❌ No models found for `{provider}`."
            if provider == "ollama":
                msg += "\nOllama isn't running — it's a local service."
            elif provider == "nvidia":
                msg += "\nSet `NVIDIA_API_KEY` in your `.env`."
            else:
                msg += "\nCheck your API key in `.env`."
            await query.edit_message_text(msg, parse_mode="Markdown")
            return

        current = _router.active_models().get(provider, "")
        primary = getattr(config, "PRIMARY_PROVIDER", "groq")
        buttons = []
        row     = []
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
            f"{icons.get(provider,'🤖')} **{provider.capitalize()} Models**\n"
            f"_Current: `{current}`  |  Primary: `{primary}`_",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode="Markdown",
        )
        return

    # ── Model selected ────────────────────────────────────────────────────────
    if data.startswith("sm:set:"):
        _, _, provider, *model_parts = data.split(":")
        model  = ":".join(model_parts)
        result = _router.switch_model(provider, model)
        await query.edit_message_text(
            f"{result}\n\n🔀 **Switch Model / Set Primary**",
            reply_markup=_main_panel_keyboard(),
            parse_mode="Markdown",
        )
        return

    # ── Set primary ───────────────────────────────────────────────────────────
    if data.startswith("sm:primary:"):
        provider = data.split(":")[2]
        result   = _router.set_primary(provider)
        config.PRIMARY_PROVIDER = provider  # runtime update
        # Persist to .env
        _persist_env("PRIMARY_PROVIDER", provider)
        await query.edit_message_text(
            f"{result}\n\n🔀 **Switch Model / Set Primary**",
            reply_markup=_main_panel_keyboard(),
            parse_mode="Markdown",
        )
        return

    # ── Back ──────────────────────────────────────────────────────────────────
    if data == "sm:back":
        await query.edit_message_text(
            "🔀 **Switch Model / Set Primary**\nChoose a provider:",
            reply_markup=_main_panel_keyboard(),
            parse_mode="Markdown",
        )


def _persist_env(key: str, value: str):
    """Write/update a key in .env file so it survives restarts."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        lines    = env_path.read_text().splitlines()
        updated  = False
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

    # ── Platform detection ────────────────────────────────────────────────────
    _platform_ctx = platform_mod.detect()
    logger.info("Platform: %s", _platform_ctx.display())

    # Override TERMUX_API_AVAILABLE from detected platform
    config.TERMUX_API_AVAILABLE = _platform_ctx.termux_api

    # ── Memory ────────────────────────────────────────────────────────────────
    _memory = AgentMemory()

    class _MemBridge:
        async def remember(self, content="", kind="fact", importance=3, **_):
            return await _memory.skill_remember(content=content, kind=kind, importance=importance)
        async def recall(self, query="", kind=None, **_):
            return await _memory.skill_recall(query=query, kind=kind)
    set_memory_bridge(_MemBridge())

    # ── Skill loading (platform-filtered) ────────────────────────────────────
    _skill_loader = SkillLoader(
        platform_ctx=_platform_ctx,
        channel=config.SKILL_CHANNEL,
        warehouse_url=config.SKILL_WAREHOUSE_URL,
    )
    loaded = await _skill_loader.load()
    set_active_skills(loaded)
    logger.info("Loaded %d skills: %s", len(loaded), sorted(loaded))

    # Build tool schemas for ONLY loaded skills
    _TOOLS = gemini_tools(active_skills=loaded)

    # Rebuild system prompt with correct platform + skill list
    build_system_prompt(force=True)

    # ── LLM router ───────────────────────────────────────────────────────────
    _router = LLMRouter(config)
    _router.on_switch_cb = telegram_alert

    # ── ReAct loop ────────────────────────────────────────────────────────────
    _react = ReactLoop(router=_router, skill_exec=execute_skill)

    # ── Autonomous planner ────────────────────────────────────────────────────
    _planner = AgentPlanner(
        react_loop=_react, memory=_memory,
        system_prompt_fn=build_system_prompt,
        alert_cb=telegram_alert, tools=_TOOLS,
    )
    await _planner.start()

    # ── Natural language scheduler ────────────────────────────────────────────
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

    # ── Monitor system ────────────────────────────────────────────────────────
    _monitor = MonitorTaskManager(
        skill_exec=execute_skill,
        alert_cb=telegram_alert,
        planner_submit=_planner.submit_proactive,
    )
    await _monitor.start()

    # Default monitors (Android only)
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

    # Disk monitor for all platforms
    await _monitor.register_job(
        job_id="disk_high",
        description="Alert when disk usage above 90%",
        skill_name="system_info", skill_kwargs={},
        condition="> 90", interval_seconds=3600,
        alert_only=False,
        action_goal="Disk above 90%. Remove large unnecessary files from ~/nexara_downloads/.",
        cooldown_s=3600,
    )

    # ── Bot commands menu ─────────────────────────────────────────────────────
    await app.bot.set_my_commands([
        BotCommand("start",       "Wake up Nexara"),
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

    # ── Startup notification ──────────────────────────────────────────────────
    n_skills = len(get_active_skills())
    msg = (
        f"✅ **Nexara V{config.NEXARA_VERSION}** online\n"
        f"{_platform_ctx.display()}\n"
        f"🧠 {n_skills} skills  ·  "
        f"⭐ Primary: `{getattr(config,'PRIMARY_PROVIDER','groq')}`  ·  "
        f"💾 `{_memory._db}`"
    )
    try:
        await app.bot.send_message(chat_id=config.ADMIN_ID, text=msg, parse_mode="Markdown")
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

    app.add_handler(CommandHandler("start",       cmd_start))
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
