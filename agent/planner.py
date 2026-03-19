"""
agent/planner.py  — V1
Autonomous task queue + worker.
New: proactive actions — when a monitor condition fires,
the agent doesn't just ALERT, it ACTS (calls a skill to fix it).
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Any

from agent.memory import AgentMemory
from agent.react_loop import ReactLoop, AgentResult

logger = logging.getLogger("nexara.planner")


class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class AutonomousTask:
    task_id: str
    goal: str
    proactive: bool            = False   # triggered by monitor, not user
    steps: list[str]           = field(default_factory=list)
    status: TaskStatus         = TaskStatus.PENDING
    result: str                = ""
    error: str                 = ""
    created_at: float          = field(default_factory=time.time)
    started_at: float          = 0.0
    completed_at: float        = 0.0


class AgentPlanner:

    def __init__(
        self,
        react_loop: ReactLoop,
        memory: AgentMemory,
        system_prompt_fn: Callable[[], str],
        alert_cb: Callable[[str], Awaitable[None]],
        tools: list[dict] | None = None,
    ):
        self._loop    = react_loop
        self._memory  = memory
        self._sp_fn   = system_prompt_fn
        self._alert   = alert_cb
        self._tools   = tools
        self._tasks:  dict[str, AutonomousTask] = {}
        self._queue:  asyncio.Queue[str]        = asyncio.Queue()
        self._worker: asyncio.Task | None       = None
        self._running = False

    async def start(self):
        self._running = True
        self._worker  = asyncio.create_task(self._run_worker(), name="planner_worker")
        logger.info("AgentPlanner started")

    async def stop(self):
        self._running = False
        if self._worker:
            self._worker.cancel()

    # ── Submit ────────────────────────────────────────────────────────────────

    async def submit(self, goal: str, task_id: str | None = None, proactive: bool = False) -> AutonomousTask:
        tid  = task_id or f"task_{uuid.uuid4().hex[:8]}"
        task = AutonomousTask(task_id=tid, goal=goal, proactive=proactive)
        self._tasks[tid] = task
        await self._queue.put(tid)
        return task

    async def submit_proactive(self, goal: str, trigger: str) -> AutonomousTask:
        """Called by monitor when a condition fires — agent acts autonomously."""
        await self._alert(f"🤖 **Proactive action triggered**\n_{trigger}_\n\nGoal: _{goal}_")
        return await self.submit(goal, proactive=True)

    # ── Query ─────────────────────────────────────────────────────────────────

    def list_tasks(self, limit: int = 10) -> str:
        if not self._tasks:
            return "No tasks queued."
        recent = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]
        icon   = {
            TaskStatus.PENDING:   "🕐",
            TaskStatus.RUNNING:   "⚙️",
            TaskStatus.DONE:      "✅",
            TaskStatus.FAILED:    "❌",
            TaskStatus.CANCELLED: "🚫",
        }
        lines = ["📋 **Autonomous Tasks**\n"]
        for t in recent:
            tag = " _(proactive)_" if t.proactive else ""
            lines.append(f"{icon[t.status]} `{t.task_id}`{tag} — {t.goal[:60]}")
            if t.result:
                lines.append(f"   ↳ {t.result[:80]}")
        return "\n".join(lines)

    async def cancel(self, task_id: str) -> str:
        t = self._tasks.get(task_id)
        if not t:
            return f"Task `{task_id}` not found."
        if t.status == TaskStatus.RUNNING:
            return "Cannot cancel a running task."
        t.status = TaskStatus.CANCELLED
        return f"🚫 `{task_id}` cancelled."

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _run_worker(self):
        while self._running:
            try:
                tid = await asyncio.wait_for(self._queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            task = self._tasks.get(tid)
            if not task or task.status == TaskStatus.CANCELLED:
                self._queue.task_done()
                continue

            task.status     = TaskStatus.RUNNING
            task.started_at = time.time()

            try:
                if not task.proactive:
                    await self._alert(f"🤖 **Task started** `{tid}`\n_{task.goal}_")

                db_id  = await self._memory.log_task(task.goal, [])
                result = await self._loop.run(
                    goal=task.goal,
                    history=[],
                    system_prompt=self._sp_fn(),
                    tools=self._tools,
                    status_cb=lambda m: self._alert(m),
                )

                task.status       = TaskStatus.DONE
                task.result       = result.answer
                task.completed_at = time.time()
                elapsed           = task.completed_at - task.started_at

                await self._memory.complete_task(db_id, result.answer, success=True)
                await self._memory.remember(
                    content=f"Completed: '{task.goal[:100]}' → {result.answer[:150]}",
                    kind="task",
                    tags=result.used_skills,
                    importance=4,
                )
                await self._alert(
                    f"✅ **Task done** `{tid}` _({elapsed:.0f}s)_\n\n{result.answer[:800]}"
                )

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
            except Exception as exc:
                task.status       = TaskStatus.FAILED
                task.error        = str(exc)
                task.completed_at = time.time()
                logger.exception("Task %s failed", tid)
                await self._alert(f"❌ **Task failed** `{tid}`\n`{exc}`")
            finally:
                self._queue.task_done()
