"""
tasks/scheduler.py — Nexara V1
Natural language schedule parser -> APScheduler jobs.
Persistent via SQLite + one-off DateTrigger support.

Fixes applied:
  - Concurrency guard: a running job will not fire again before it finishes.
    Previously, if a proactive job took >interval seconds, APScheduler would
    stack multiple concurrent runs of the same job, spiralling into dozens of
    parallel agent loops that could exhaust all LLM quota.
  - _fire wrapped in try/except/finally — unhandled exceptions inside a job
    were silently swallowed by APScheduler, leaving _running in a dirty state.
  - Cleanup of one-off jobs moved to finally block (was missing on error path).
"""

import asyncio
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron      import CronTrigger
from apscheduler.triggers.interval  import IntervalTrigger
from apscheduler.triggers.date      import DateTrigger

logger = logging.getLogger("nexara.scheduler")

TASKS_DB = Path.home() / ".nexara" / "tasks.db"


@dataclass
class ScheduledJob:
    job_id:       str
    goal:         str
    schedule:     str
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
        self._running: set[str] = set()   # jobs currently executing
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

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _load_persisted_jobs(self):
        def _read():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.row_factory = sqlite3.Row
                return conn.execute("SELECT * FROM schedules").fetchall()

        rows = await asyncio.to_thread(_read)
        for r in rows:
            trigger, _ = self._parse(r["schedule"])

            if isinstance(trigger, DateTrigger) and trigger.run_date < datetime.now(timezone.utc):
                asyncio.create_task(self._delete_job(r["job_id"]))
                continue

            if trigger:
                self._sched.add_job(
                    self._fire,
                    trigger=trigger,
                    id=r["job_id"],
                    args=[r["job_id"]],
                    max_instances=1,
                    replace_existing=True,
                    misfire_grace_time=300,
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
                    "INSERT OR REPLACE INTO schedules (job_id, goal, schedule, trigger_desc) "
                    "VALUES (?, ?, ?, ?)",
                    (job.job_id, job.goal, job.schedule, job.trigger_desc),
                )
        await asyncio.to_thread(_write)

    async def _delete_job(self, job_id: str):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute("DELETE FROM schedules WHERE job_id = ?", (job_id,))
        await asyncio.to_thread(_write)

    # ── Register / Cancel ─────────────────────────────────────────────────────

    async def schedule(self, goal: str, schedule_str: str, job_id: str | None = None) -> str:
        jid = job_id or f"sched_{uuid.uuid4().hex[:6]}"

        if not schedule_str or not schedule_str.strip():
            return (
                "ERROR: Empty schedule string. "
                "Specify a time, e.g. 'every day at 8am' or 'in 5 minutes'."
            )

        trigger, desc = self._parse(schedule_str)
        if trigger is None:
            return (
                f"ERROR: Unrecognised schedule format: '{schedule_str}'\n"
                "Supported formats:\n"
                "- 'in 5 minutes' / 'in 2 hours'\n"
                "- 'every 30 minutes'\n"
                "- 'every day at 9am'\n"
                "- 'every monday at noon'"
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

        job = ScheduledJob(job_id=jid, goal=goal, schedule=schedule_str, trigger_desc=desc)
        self._jobs[jid] = job
        await self._save_job(job)
        logger.info("Scheduled job '%s': %s -> %s", jid, desc, goal[:60])
        return f"Scheduled '{jid}': {goal[:60]}\nRuns: {desc}"

    async def cancel(self, job_id: str) -> str:
        if job_id not in self._jobs:
            return f"No scheduled job with ID '{job_id}'."
        try:
            self._sched.remove_job(job_id)
        except Exception:
            pass
        del self._jobs[job_id]
        self._running.discard(job_id)
        await self._delete_job(job_id)
        return f"Schedule '{job_id}' cancelled."

    def list_jobs(self) -> str:
        if not self._jobs:
            return "No scheduled tasks."
        lines = ["Scheduled Tasks\n"]
        for j in self._jobs.values():
            running = " (running)" if j.job_id in self._running else ""
            lines.append(f"- {j.job_id}: {j.goal[:60]}\n  Runs: {j.trigger_desc}{running}")
        return "\n".join(lines)

    # ── Fire ──────────────────────────────────────────────────────────────────

    async def _fire(self, job_id: str):
        """
        Execute a scheduled job.

        Concurrency guard: if the same job is still running from a previous
        fire (e.g. agent took longer than the interval), skip this trigger
        rather than stacking a second parallel execution.
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        if job_id in self._running:
            logger.warning(
                "Job '%s' already running — skipping this trigger to avoid stacking.", job_id
            )
            return

        self._running.add(job_id)
        logger.info("Firing scheduled job '%s': %s", job_id, job.goal[:60])

        is_oneoff = "Once in" in job.trigger_desc

        try:
            await self._submit(job.goal, f"Scheduled: {job.trigger_desc}")
        except Exception as exc:
            logger.error("Scheduled job '%s' raised an error: %s", job_id, exc)
        finally:
            self._running.discard(job_id)
            if is_oneoff and job_id in self._jobs:
                try:
                    await self.cancel(job_id)
                except Exception:
                    pass

    # ── Natural language parser ───────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> tuple:
        """
        Returns (APScheduler trigger, human description) or (None, None).
        Handles intervals, time-of-day, day-of-week, combined, one-off delays.
        """
        t = text.lower().strip()

        # One-off: "in 5 minutes"
        m = re.search(r"^in\s+(\d+)\s+(second|minute|hour|day)s?", t)
        if m:
            qty  = int(m.group(1))
            unit = m.group(2)
            run_date = datetime.now(timezone.utc) + timedelta(**{f"{unit}s": qty})
            return DateTrigger(run_date=run_date), f"Once in {qty} {unit}(s)"

        # Interval: "every 30 minutes"
        m = re.search(r"every\s+(\d+)\s+(second|minute|hour|day)s?", t)
        if m:
            qty  = int(m.group(1))
            unit = m.group(2)
            return IntervalTrigger(**{f"{unit}s": qty}), f"Every {qty} {unit}(s)"

        # Shorthand: "every minute/hour/day/week"
        m = re.search(r"every\s+(minute|hour|day|week)(?!\s+\d)", t)
        if m:
            mapping = {
                "minute": (IntervalTrigger(minutes=1),            "Every minute"),
                "hour":   (IntervalTrigger(hours=1),              "Every hour"),
                "day":    (CronTrigger(hour=0, minute=0),         "Every day at midnight"),
                "week":   (CronTrigger(day_of_week="mon", hour=0, minute=0), "Every Monday"),
            }
            return mapping.get(m.group(1), (None, None))

        # Time extraction
        hour, minute = None, 0
        am_pm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
        if am_pm:
            hour   = int(am_pm.group(1))
            minute = int(am_pm.group(2) or 0)
            if am_pm.group(3) == "pm" and hour != 12:
                hour += 12
            elif am_pm.group(3) == "am" and hour == 12:
                hour = 0

        for name, (h, mn) in {
            "midnight": (0, 0), "noon": (12, 0), "midday": (12, 0),
            "morning":  (8, 0), "evening": (18, 0), "night": (21, 0),
        }.items():
            if name in t:
                hour, minute = h, mn
                break

        # Day of week
        days = {
            "monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
        }
        dow = next((code for name, code in days.items() if name in t), None)

        if dow and hour is not None:
            return CronTrigger(day_of_week=dow, hour=hour, minute=minute), \
                   f"Every {dow.capitalize()} at {hour:02d}:{minute:02d}"
        if dow:
            return CronTrigger(day_of_week=dow, hour=0, minute=0), \
                   f"Every {dow.capitalize()} at midnight"
        if hour is not None and any(x in t for x in ["every day", "daily", "each day"]):
            return CronTrigger(hour=hour, minute=minute), \
                   f"Every day at {hour:02d}:{minute:02d}"
        if hour is not None:
            return CronTrigger(hour=hour, minute=minute), \
                   f"Daily at {hour:02d}:{minute:02d}"

        return None, None
