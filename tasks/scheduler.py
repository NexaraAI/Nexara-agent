"""
tasks/scheduler.py
Natural language schedule parser → APScheduler jobs.
Now features autonomous SQLite persistence: surviving restarts seamlessly.
"""

import asyncio
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron     import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("nexara.scheduler")

TASKS_DB = Path.home() / ".nexara" / "tasks.db"


@dataclass
class ScheduledJob:
    job_id:       str
    goal:         str
    schedule:     str   # human-readable schedule string
    trigger_desc: str


class NaturalScheduler:
    """
    Parses plain-English schedule descriptions and registers APScheduler jobs.
    Fully persistent: jobs are saved to SQLite and reloaded on startup.
    """

    def __init__(self, submit_fn: Callable[[str, str], Awaitable[None]]):
        self._submit  = submit_fn
        self._sched   = AsyncIOScheduler(timezone="UTC")
        self._jobs:   dict[str, ScheduledJob] = {}
        self._started = False
        self._init_db()

    def _init_db(self):
        TASKS_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(TASKS_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    job_id       TEXT PRIMARY KEY,
                    goal         TEXT NOT NULL,
                    schedule     TEXT NOT NULL,
                    trigger_desc TEXT NOT NULL
                )
            """)

    async def start(self):
        self._sched.start()
        self._started = True
        await self._load_persisted_jobs()
        logger.info("NaturalScheduler started & synced with DB")

    async def stop(self):
        self._sched.shutdown(wait=False)

    # ── Persistence / Loading ─────────────────────────────────────────────────

    async def _load_persisted_jobs(self):
        def _read():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.row_factory = sqlite3.Row
                return conn.execute("SELECT * FROM schedules").fetchall()
        
        rows = await asyncio.to_thread(_read)
        for r in rows:
            trigger, _ = self._parse(r["schedule"])
            if trigger:
                self._sched.add_job(
                    self._fire,
                    trigger=trigger,
                    id=r["job_id"],
                    args=[r["job_id"]],
                    max_instances=1,
                    replace_existing=True,
                    misfire_grace_time=300, # Give it 5 mins grace on restart
                )
                self._jobs[r["job_id"]] = ScheduledJob(
                    job_id=r["job_id"],
                    goal=r["goal"],
                    schedule=r["schedule"],
                    trigger_desc=r["trigger_desc"],
                )
            else:
                logger.warning("Failed to re-parse saved schedule: %s", r["schedule"])

    async def _save_job(self, job: ScheduledJob):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schedules (job_id, goal, schedule, trigger_desc) VALUES (?, ?, ?, ?)",
                    (job.job_id, job.goal, job.schedule, job.trigger_desc)
                )
        await asyncio.to_thread(_write)

    async def _delete_job(self, job_id: str):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute("DELETE FROM schedules WHERE job_id = ?", (job_id,))
        await asyncio.to_thread(_write)

    # ── Register ──────────────────────────────────────────────────────────────

    async def schedule(self, goal: str, schedule_str: str, job_id: str | None = None) -> str:
        jid = job_id or f"sched_{uuid.uuid4().hex[:6]}"

        trigger, desc = self._parse(schedule_str)
        if trigger is None:
            return (
                f"❌ Couldn't parse schedule: `{schedule_str}`\n\n"
                "Try: 'every 30 minutes', 'every day at 9am', "
                "'every monday at noon', 'every hour'"
            )

        self._sched.add_job(
            self._fire,
            trigger=trigger,
            id=jid,
            args=[jid],
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=120,
        )

        job = ScheduledJob(
            job_id=jid,
            goal=goal,
            schedule=schedule_str,
            trigger_desc=desc,
        )
        self._jobs[jid] = job
        await self._save_job(job)
        
        logger.info("Scheduled job '%s': %s → %s", jid, desc, goal[:60])
        return f"⏰ Scheduled `{jid}`: _{goal[:60]}_\n🕐 Runs: **{desc}**"

    async def cancel(self, job_id: str) -> str:
        if job_id not in self._jobs:
            return f"No scheduled job with ID `{job_id}`."
        try:
            self._sched.remove_job(job_id)
        except Exception:
            pass
        
        del self._jobs[job_id]
        await self._delete_job(job_id)
        return f"🗑️ Schedule `{job_id}` cancelled."

    def list_jobs(self) -> str:
        if not self._jobs:
            return "No scheduled tasks."
        lines = ["⏰ **Scheduled Tasks**\n"]
        for j in self._jobs.values():
            lines.append(f"• `{j.job_id}` — {j.goal[:60]}\n  🕐 {j.trigger_desc}")
        return "\n".join(lines)

    # ── Fire ──────────────────────────────────────────────────────────────────

    async def _fire(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        logger.info("Firing scheduled job '%s': %s", job_id, job.goal[:60])
        await self._submit(job.goal, f"Scheduled: {job.trigger_desc}")

    # ── Natural language parser ───────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> tuple:
        """
        Returns (APScheduler trigger, human description) or (None, None).
        Handles: intervals, time-of-day, day-of-week, combined.
        """
        t = text.lower().strip()

        # ── Interval patterns ─────────────────────────────────────────────────
        m = re.search(r"every\s+(\d+)\s+(second|minute|hour|day)s?", t)
        if m:
            qty  = int(m.group(1))
            unit = m.group(2)
            kwargs = {f"{unit}s": qty}
            return IntervalTrigger(**kwargs), f"Every {qty} {unit}(s)"

        m = re.search(r"every\s+(minute|hour|day|week)(?!\s+\d)", t)
        if m:
            unit = m.group(1)
            mapping = {
                "minute": (IntervalTrigger(minutes=1),  "Every minute"),
                "hour":   (IntervalTrigger(hours=1),    "Every hour"),
                "day":    (CronTrigger(hour=0,minute=0), "Every day at midnight"),
                "week":   (CronTrigger(day_of_week="mon", hour=0, minute=0), "Every Monday"),
            }
            return mapping.get(unit, (None, None))

        # ── Time extraction ───────────────────────────────────────────────────
        hour, minute = None, 0
        am_pm_m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
        if am_pm_m:
            hour   = int(am_pm_m.group(1))
            minute = int(am_pm_m.group(2) or 0)
            if am_pm_m.group(3) == "pm" and hour != 12: hour += 12
            elif am_pm_m.group(3) == "am" and hour == 12: hour = 0

        named_times = {
            "midnight": (0, 0), "noon": (12, 0), "midday": (12, 0),
            "morning":  (8, 0), "evening": (18, 0), "night": (21, 0),
        }
        for name, (h, mn) in named_times.items():
            if name in t:
                hour, minute = h, mn
                break

        # ── Day-of-week ───────────────────────────────────────────────────────
        days = {
            "monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
        }
        dow = next((code for name, code in days.items() if name in t), None)

        # ── Build trigger ─────────────────────────────────────────────────────
        if dow and hour is not None:
            return CronTrigger(day_of_week=dow, hour=hour, minute=minute), f"Every {dow.capitalize()} at {hour:02d}:{minute:02d}"
        if dow and hour is None:
            return CronTrigger(day_of_week=dow, hour=0, minute=0), f"Every {dow.capitalize()} at midnight"
        if hour is not None and any(x in t for x in ["every day", "daily", "each day"]):
            return CronTrigger(hour=hour, minute=minute), f"Every day at {hour:02d}:{minute:02d}"
        if hour is not None:
            return CronTrigger(hour=hour, minute=minute), f"Daily at {hour:02d}:{minute:02d}"

        return None, None
