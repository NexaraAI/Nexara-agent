"""
agent/react_loop.py  — V1
ReAct engine using structured function-calling.
New in V1:
  • Uses LLMRouter instead of raw genai calls
  • Structured ToolCall responses (no regex parsing)
  • Replanning: detects skill failures and asks LLM for alternatives
  • Per-user concurrency semaphore (fairness)
  • Step streaming: status_cb fires on THINK too, not just ACT
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any

from agent.llm_router import LLMRouter, LLMResponse, ToolCall, TextResponse, AskUser

logger = logging.getLogger("nexara.react")

MAX_ITERATIONS    = 14
MAX_REPLAN_DEPTH  = 3    # how many times we'll try an alternative on failure
_USER_SEMS: dict[int, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(2))


@dataclass
class Step:
    iteration: int
    thought: str
    action: str
    args: dict
    observation: str
    success: bool
    replanned: bool = False


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]          = field(default_factory=list)
    used_skills: list[str]     = field(default_factory=list)
    iterations: int            = 0
    needs_user_input: bool     = False
    question_for_user: str     = ""


class ReactLoop:

    def __init__(
        self,
        router: LLMRouter,
        skill_exec: Callable[[str, dict], Awaitable[Any]],
    ):
        self._router = router
        self._exec   = skill_exec

    async def run(
        self,
        goal: str,
        history: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        user_id: int = 0,
        status_cb: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        """
        Acquire per-user semaphore → run ReAct loop → release.
        """
        sem = _USER_SEMS[user_id]
        async with sem:
            return await self._run_inner(goal, history, system_prompt, tools, status_cb)

    async def _run_inner(self, goal, history, system_prompt, tools, status_cb) -> AgentResult:
        steps:       list[Step] = []
        used_skills: list[str]  = []
        replan_budget            = MAX_REPLAN_DEPTH

        messages = list(history)
        messages.append({"role": "user", "parts": [goal]})

        for i in range(1, MAX_ITERATIONS + 1):
            # ── THINK ────────────────────────────────────────────────────────
            if status_cb:
                await status_cb(f"🧠 Thinking… _(step {i})_")

            response: LLMResponse = await self._router.complete(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            # ── TERMINAL: text answer ─────────────────────────────────────────
            if isinstance(response, TextResponse):
                steps.append(Step(i, response.text, "final_answer", {}, "", True))
                return AgentResult(
                    answer=response.text,
                    steps=steps,
                    used_skills=used_skills,
                    iterations=i,
                )

            # ── TERMINAL: ask user ────────────────────────────────────────────
            if isinstance(response, AskUser):
                return AgentResult(
                    answer="",
                    steps=steps,
                    used_skills=used_skills,
                    iterations=i,
                    needs_user_input=True,
                    question_for_user=response.question,
                )

            # ── ACT ──────────────────────────────────────────────────────────
            tc: ToolCall = response
            if status_cb:
                thought_preview = tc.thought[:60] + "…" if tc.thought else ""
                await status_cb(
                    f"⚙️ `{tc.name}` — {thought_preview}" if thought_preview else f"⚙️ `{tc.name}`"
                )

            try:
                result  = await self._exec(tc.name, tc.args)
                obs     = str(result)
                ok      = getattr(result, "success", True)
            except Exception as exc:
                obs = f"[Fatal error in {tc.name}: {exc}]"
                ok  = False

            used_skills.append(tc.name)
            step = Step(i, tc.thought, tc.name, tc.args, obs, ok)
            steps.append(step)

            # ── REPLAN on failure ─────────────────────────────────────────────
            if not ok and replan_budget > 0:
                replan_budget -= 1
                replan_msg = (
                    f"The skill `{tc.name}` failed with: {obs}\n"
                    f"Think of an alternative approach. Replanning budget: {replan_budget} attempt(s) left."
                )
                messages.append({"role": "user", "parts": [replan_msg]})
                step.replanned = True
                if status_cb:
                    await status_cb(f"🔁 Replanning after `{tc.name}` failure…")
                continue   # skip normal observe-append, go straight to next THINK

            # ── OBSERVE ───────────────────────────────────────────────────────
            messages.append({
                "role": "user",
                "parts": [
                    f"[Observation from `{tc.name}`]:\n{obs}\n\n"
                    "Continue with the next step or emit a final_answer if the task is complete."
                ],
            })

            logger.info("Step %d/%d: %s → ok=%s | %s", i, MAX_ITERATIONS, tc.name, ok, obs[:80])

        # Max iterations — force final answer
        messages.append({
            "role": "user",
            "parts": ["Maximum steps reached. Summarise everything you've found as a final answer."],
        })
        response = await self._router.complete(messages, system_prompt, tools)
        answer = response.text if isinstance(response, TextResponse) else \
                 (response.answer if hasattr(response, "answer") else str(response))

        return AgentResult(
            answer=answer.strip(),
            steps=steps,
            used_skills=used_skills,
            iterations=MAX_ITERATIONS,
        )
