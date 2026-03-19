"""
utils/skill_router.py  — V1
Central skill router.
Handles all registered BaseSkill subclasses PLUS:
  - memory skill bridge (remember / recall)
  - scheduler bridge (schedule_task / list_schedules / cancel_schedule)
"""

import logging
from typing import Callable, Awaitable, Any

from skills.base import SkillMeta, SkillResult

logger = logging.getLogger("nexara.router")

_skill_cache: dict = {}

# Bridges set at boot time by main.py
_memory_bridge:    "MemoryBridge | None"    = None
_scheduler_bridge: "SchedulerBridge | None" = None


def set_memory_bridge(bridge):    global _memory_bridge;    _memory_bridge    = bridge
def set_scheduler_bridge(bridge): global _scheduler_bridge; _scheduler_bridge = bridge


def _get_skill(name: str):
    if name not in _skill_cache:
        cls = SkillMeta.get_registry().get(name)
        if cls is None:
            return None
        _skill_cache[name] = cls()
    return _skill_cache[name]


async def execute_skill(name: str, kwargs: dict) -> SkillResult:
    # ── Memory bridge ─────────────────────────────────────────────────────────
    if name == "remember" and _memory_bridge:
        text = await _memory_bridge.remember(**kwargs)
        return SkillResult(success=True, output=text, data={})

    if name == "recall" and _memory_bridge:
        text = await _memory_bridge.recall(**kwargs)
        return SkillResult(success=True, output=text, data={})

    # ── Scheduler bridge ──────────────────────────────────────────────────────
    if name == "schedule_task" and _scheduler_bridge:
        text = await _scheduler_bridge.schedule(**kwargs)
        return SkillResult(success=True, output=text, data={})

    if name == "list_schedules" and _scheduler_bridge:
        text = _scheduler_bridge.list_jobs()
        return SkillResult(success=True, output=text, data={})

    if name == "cancel_schedule" and _scheduler_bridge:
        text = await _scheduler_bridge.cancel(**kwargs)
        return SkillResult(success=True, output=text, data={})

    # ── Regular skill ─────────────────────────────────────────────────────────
    skill = _get_skill(name)
    if skill is None:
        return SkillResult(
            success=False, output="",
            error=f"Unknown skill `{name}`. Available: {sorted(SkillMeta.get_registry())}",
        )
    logger.info("Executing skill '%s' kwargs=%s", name, kwargs)
    try:
        return await skill.execute(**kwargs)
    except Exception as exc:
        logger.exception("Skill '%s' raised: %s", name, exc)
        return SkillResult(success=False, output="", error=str(exc))


def skill_descriptions() -> str:
    reg = SkillMeta.get_registry()
    lines = []
    for name, cls in sorted(reg.items()):
        lines.append(f"- **{name}**: {getattr(cls, 'description', '')}")
    # Memory + scheduler (virtual skills)
    lines.append("- **remember**: Store a fact in long-term memory")
    lines.append("- **recall**: Search long-term memory semantically")
    lines.append("- **schedule_task**: Schedule a recurring autonomous task")
    lines.append("- **list_schedules**: List all scheduled tasks")
    lines.append("- **cancel_schedule**: Cancel a scheduled task by ID")
    return "\n".join(lines)
