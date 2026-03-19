"""
tasks/monitor_task.py  — V1
Condition monitor with proactive action support and SQLite persistence.
Survives OTA updates and unexpected terminations cleanly.
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval  import IntervalTrigger

logger = logging.getLogger("nexara.monitor")

TASKS_DB = Path.home() / ".nexara" / "tasks.db"


@dataclass
class MonitorJob:
    job_id:           str
    description:      str
    skill_name:       str
    skill_kwargs:     dict
    condition:        str
    interval_seconds: int
    alert_only:       bool    = True
    action_goal:      str     = ""
    action_skill:     str     = ""
    action_kwargs:    dict    = field(default_factory=dict)
    cooldown_s:       float   = 300.0
    
    # Runtime states
    triggered_count:  int     = 0
    last_run:         float   = 0.0
    last_triggered:   float   = 0.0
    last_value:       str     = ""


class MonitorTaskManager:

    def __init__(
        self,
        skill_exec: Callable[[str, dict], Awaitable[Any]],
        alert_cb:   Callable[[str], Awaitable[None]],
        planner_submit: Callable[[str, str], Awaitable[Any]] | None = None,
    ):
        self._exec    = skill_exec
        self._alert   = alert_cb
        self._submit  = planner_submit
        self._jobs:   dict[str, MonitorJob] = {}
        self._sched   = AsyncIOScheduler(timezone="UTC")
        self._init_db()

    def _init_db(self):
        TASKS_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(TASKS_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitors (
                    job_id           TEXT PRIMARY KEY,
                    description      TEXT NOT NULL,
                    skill_name       TEXT NOT NULL,
                    skill_kwargs     TEXT,
                    condition        TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL,
                    alert_only       INTEGER NOT NULL,
                    action_goal      TEXT,
                    action_skill     TEXT,
                    action_kwargs    TEXT,
                    cooldown_s       REAL,
                    last_triggered   REAL
                )
            """)

    async def start(self):
        self._sched.start()
        await self._load_persisted_monitors()
        logger.info("MonitorTaskManager started & synced with DB")

    async def stop(self):
        self._sched.shutdown(wait=False)

    # ── Persistence / Loading ─────────────────────────────────────────────────

    async def _load_persisted_monitors(self):
        def _read():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.row_factory = sqlite3.Row
                return conn.execute("SELECT * FROM monitors").fetchall()
                
        rows = await asyncio.to_thread(_read)
        for r in rows:
            job = MonitorJob(
                job_id=r["job_id"],
                description=r["description"],
                skill_name=r["skill_name"],
                skill_kwargs=json.loads(r["skill_kwargs"] or "{}"),
                condition=r["condition"],
                interval_seconds=r["interval_seconds"],
                alert_only=bool(r["alert_only"]),
                action_goal=r["action_goal"] or "",
                action_skill=r["action_skill"] or "",
                action_kwargs=json.loads(r["action_kwargs"] or "{}"),
                cooldown_s=r["cooldown_s"] or 300.0,
                last_triggered=r["last_triggered"] or 0.0,
            )
            self._jobs[job.job_id] = job
            self._sched.add_job(
                self._run_job,
                trigger=IntervalTrigger(seconds=job.interval_seconds),
                id=job.job_id, args=[job.job_id],
                max_instances=1, replace_existing=True,
            )

    async def _save_monitor(self, job: MonitorJob):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO monitors 
                    (job_id, description, skill_name, skill_kwargs, condition, 
                     interval_seconds, alert_only, action_goal, action_skill, 
                     action_kwargs, cooldown_s, last_triggered) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id, job.description, job.skill_name, 
                    json.dumps(job.skill_kwargs), job.condition, 
                    job.interval_seconds, int(job.alert_only), 
                    job.action_goal, job.action_skill, 
                    json.dumps(job.action_kwargs), job.cooldown_s, 
                    job.last_triggered
                ))
        await asyncio.to_thread(_write)

    async def _delete_monitor(self, job_id: str):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute("DELETE FROM monitors WHERE job_id = ?", (job_id,))
        await asyncio.to_thread(_write)

    async def _update_trigger_time(self, job_id: str, ts: float):
        def _write():
            with sqlite3.connect(TASKS_DB) as conn:
                conn.execute("UPDATE monitors SET last_triggered = ? WHERE job_id = ?", (ts, job_id))
        await asyncio.to_thread(_write)

    # ── Register ──────────────────────────────────────────────────────────────

    async def register_job(
        self,
        job_id: str,
        description: str,
        skill_name: str,
        skill_kwargs: dict,
        condition: str,
        interval_seconds: int = 300,
        alert_only: bool = True,
        action_goal: str = "",
        action_skill: str = "",
        action_kwargs: dict | None = None,
        cooldown_s: float = 300.0,
    ) -> str:
        if job_id in self._jobs:
            return f"Job `{job_id}` already registered."

        job = MonitorJob(
            job_id=job_id, description=description,
            skill_name=skill_name, skill_kwargs=skill_kwargs,
            condition=condition, interval_seconds=interval_seconds,
            alert_only=alert_only, action_goal=action_goal,
            action_skill=action_skill, action_kwargs=action_kwargs or {},
            cooldown_s=cooldown_s,
        )
        self._jobs[job_id] = job
        await self._save_monitor(job)

        self._sched.add_job(
            self._run_job,
            trigger=IntervalTrigger(seconds=interval_seconds),
            id=job_id, args=[job_id],
            max_instances=1, replace_existing=True,
        )
        logger.info("Monitor job registered: %s (every %ds)", job_id, interval_seconds)
        return (
            f"✅ Monitor `{job_id}` active — every {interval_seconds}s\n"
            f"Condition: `{condition}` | "
            f"{'Alert only' if alert_only else 'Proactive action'}"
        )

    async def unregister_job(self, job_id: str) -> str:
        if job_id not in self._jobs:
            return f"No job `{job_id}`."
        try:
            self._sched.remove_job(job_id)
        except Exception:
            pass
            
        del self._jobs[job_id]
        await self._delete_monitor(job_id)
        return f"🗑️ Monitor `{job_id}` removed."

    def list_jobs(self) -> str:
        if not self._jobs:
            return "No active monitor jobs."
        lines = ["📡 **Active Monitors**\n"]
        for j in self._jobs.values():
            kind = "🔔 alert" if j.alert_only else "🤖 proactive"
            lines.append(
                f"• `{j.job_id}` [{kind}] — {j.description}\n"
                f"  `{j.condition}` every {j.interval_seconds}s "
                f"| fired {j.triggered_count}x | last: {j.last_value[:40]}"
            )
        return "\n".join(lines)

    # ── Runner ────────────────────────────────────────────────────────────────

    async def _run_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return

        job.last_run = time.time()
        try:
            result    = await self._exec(job.skill_name, job.skill_kwargs)
            output    = str(result)
            job.last_value = output[:100]

            if not self._evaluate(output, job.condition):
                return

            if job.triggered_count > 0 or job.last_triggered > 0:
                elapsed = time.time() - job.last_triggered
                if elapsed < job.cooldown_s:
                    return

            job.triggered_count += 1
            job.last_triggered = time.time()
            
            # Immediately persist the new cooldown stamp
            await self._update_trigger_time(job_id, job.last_triggered)

            alert_msg = (
                f"📡 **Monitor Alert** `{job.job_id}`\n"
                f"_{job.description}_\n\n"
                f"Condition `{job.condition}` triggered\n"
                f"Value: `{output[:200]}`"
            )
            await self._alert(alert_msg)

            # ── Proactive: take action ────────────────────────────────────────
            if not job.alert_only:
                if job.action_skill and job.action_kwargs is not None:
                    await self._exec(job.action_skill, job.action_kwargs)
                    logger.info("Monitor %s triggered direct action: %s", job_id, job.action_skill)

                elif job.action_goal and self._submit:
                    await self._submit(
                        job.action_goal,
                        f"Monitor `{job_id}` fired: {job.condition}",
                    )

        except Exception as exc:
            logger.error("Monitor job %s error: %s", job_id, exc)

    @staticmethod
    def _evaluate(output: str, condition: str) -> bool:
        c = condition.strip()
        num = re.search(r"[-+]?\d+\.?\d*", output)
        if num:
            v = float(num.group())
            ops = {"< ": v.__lt__, "> ": v.__gt__, "==": v.__eq__, "== ": v.__eq__,
                   "<=": v.__le__, "<= ": v.__le__, ">=": v.__ge__, ">= ": v.__ge__}
            for op, fn in ops.items():
                if c.startswith(op):
                    try:
                        return fn(float(c[len(op):].strip()))
                    except ValueError:
                        pass
        if c.startswith("contains "):
            return c[9:].lower() in output.lower()
        if c.startswith("not contains "):
            return c[13:].lower() not in output.lower()
        return False
