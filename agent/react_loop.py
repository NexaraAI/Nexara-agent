"""
agent/react_loop.py — Nexara V1
ReAct engine using structured function-calling.
  • Uses LLMRouter instead of raw genai calls
  • Structured ToolCall responses (no regex parsing)
  • Replanning: detects skill failures and asks LLM for alternatives
  • Per-user concurrency semaphore (fairness)
  • Step streaming: status_cb fires on THINK too, not just ACT
  • JSON retry: malformed LLM responses get one correction prompt
  • Per-step LLM timeout (30s) + skill execution timeout (60s)
  • Plain-text status updates — no Markdown in status_cb (prevents Telegram crashes)
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any

from agent.llm_router import LLMRouter, LLMResponse, ToolCall, TextResponse, AskUser

logger = logging.getLogger("nexara.react")

MAX_ITERATIONS   = 14
MAX_REPLAN_DEPTH = 3
STEP_LLM_TIMEOUT = 45   # seconds — per LLM call inside the loop
SKILL_TIMEOUT    = 60   # seconds — per skill execution
JSON_RETRY_LIMIT = 2    # retry malformed JSON this many times before giving up

_USER_SEMS: dict[int, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(2))


@dataclass
class Step:
    iteration:   int
    thought:     str
    action:      str
    args:        dict
    observation: str
    success:     bool
    replanned:   bool = False


@dataclass
class AgentResult:
    answer:            str
    steps:             list[Step] = field(default_factory=list)
    used_skills:       list[str]  = field(default_factory=list)
    iterations:        int        = 0
    needs_user_input:  bool       = False
    question_for_user: str        = ""


class ReactLoop:

    def __init__(
        self,
        router:     LLMRouter,
        skill_exec: Callable[[str, dict], Awaitable[Any]],
    ):
        self._router = router
        self._exec   = skill_exec

    async def run(
        self,
        goal:          str,
        history:       list[dict],
        system_prompt: str,
        tools:         list[dict] | None = None,
        user_id:       int = 0,
        status_cb:     Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        sem = _USER_SEMS[user_id]
        async with sem:
            return await self._run_inner(goal, history, system_prompt, tools, status_cb)

    # ── Safe status helper ────────────────────────────────────────────────────
    # Plain text only — Telegram's Markdown v1 parser crashes on skill names
    # that contain underscores (e.g. web_search, system_info).

    @staticmethod
    async def _status(cb, msg: str):
        if cb:
            try:
                await cb(msg)
            except Exception:
                pass

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def _run_inner(self, goal, history, system_prompt, tools, status_cb) -> AgentResult:
        steps:         list[Step] = []
        used_skills:   list[str]  = []
        replan_budget              = MAX_REPLAN_DEPTH
        json_retries               = 0

        messages = list(history)
        messages.append({"role": "user", "parts": [goal]})

        for i in range(1, MAX_ITERATIONS + 1):

            # ── THINK ────────────────────────────────────────────────────────
            await self._status(status_cb, f"Thinking... (step {i})")

            try:
                response: LLMResponse = await asyncio.wait_for(
                    self._router.complete(
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=tools,
                    ),
                    timeout=STEP_LLM_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timed out on step %d", i)
                # Try to get a final answer from whatever context we have
                await self._status(status_cb, "LLM timed out — wrapping up...")
                break
            except Exception as exc:
                logger.error("LLM error on step %d: %s", i, exc)
                await self._status(status_cb, f"LLM error: {exc}")
                return AgentResult(
                    answer=f"I ran into an error communicating with the AI: {exc}",
                    steps=steps, used_skills=used_skills, iterations=i,
                )

            # ── TERMINAL: plain text answer ───────────────────────────────────
            if isinstance(response, TextResponse):
                # Sanity check: if the text looks like a broken JSON attempt,
                # give the model one correction prompt before accepting it.
                raw = response.text
                if (json_retries < JSON_RETRY_LIMIT
                        and raw.strip().startswith("{")
                        and '"action"' not in raw):
                    json_retries += 1
                    messages.append({
                        "role": "user",
                        "parts": [
                            f"Your response was not valid ReAct JSON: {raw[:200]}\n"
                            "Please emit a single valid JSON block using the format shown in the system prompt. "
                            'Use {"action":"final_answer","answer":"..."} if you are done.'
                        ],
                    })
                    await self._status(status_cb, f"Retrying malformed JSON (attempt {json_retries})...")
                    continue

                steps.append(Step(i, raw, "final_answer", {}, "", True))
                return AgentResult(
                    answer=raw,
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
            json_retries = 0   # reset on successful tool call parse

            preview = (tc.thought[:60] + "…") if tc.thought else ""
            if preview:
                await self._status(status_cb, f"Using {tc.name} — {preview}")
            else:
                await self._status(status_cb, f"Using {tc.name}...")

            try:
                result = await asyncio.wait_for(
                    self._exec(tc.name, tc.args),
                    timeout=SKILL_TIMEOUT,
                )
                obs = str(result)
                ok  = getattr(result, "success", True)
            except asyncio.TimeoutError:
                obs = f"[Skill {tc.name} timed out after {SKILL_TIMEOUT}s]"
                ok  = False
                logger.warning("Skill '%s' timed out", tc.name)
            except Exception as exc:
                obs = f"[Fatal error in {tc.name}: {exc}]"
                ok  = False
                logger.error("Skill '%s' raised: %s", tc.name, exc)

            used_skills.append(tc.name)
            step = Step(i, tc.thought, tc.name, tc.args, obs, ok)
            steps.append(step)

            logger.info("Step %d/%d: %s -> ok=%s | %s",
                        i, MAX_ITERATIONS, tc.name, ok, obs[:80])

            # ── REPLAN on failure ─────────────────────────────────────────────
            if not ok and replan_budget > 0:
                replan_budget -= 1
                step.replanned = True
                messages.append({
                    "role": "user",
                    "parts": [
                        f"The skill '{tc.name}' failed: {obs}\n"
                        f"Think of an alternative approach. "
                        f"Replanning budget remaining: {replan_budget}."
                    ],
                })
                await self._status(status_cb, f"Replanning after {tc.name} failed...")
                continue

            # ── OBSERVE ───────────────────────────────────────────────────────
            messages.append({
                "role": "user",
                "parts": [
                    f"[Observation from '{tc.name}']:\n{obs}\n\n"
                    "Continue with the next step, or emit a final_answer if the task is complete."
                ],
            })

        # ── Max iterations hit — force final answer ───────────────────────────
        await self._status(status_cb, "Summarising results...")
        messages.append({
            "role": "user",
            "parts": ["Maximum steps reached. Summarise everything you've found as a final_answer now."],
        })
        try:
            response = await asyncio.wait_for(
                self._router.complete(messages, system_prompt, tools),
                timeout=STEP_LLM_TIMEOUT,
            )
            if isinstance(response, TextResponse):
                answer = response.text
            elif isinstance(response, ToolCall):
                answer = f"Completed {len(steps)} steps. Last action: {steps[-1].action if steps else 'none'}."
            else:
                answer = str(response)
        except Exception as exc:
            answer = f"Completed {len(steps)} steps but failed to summarise: {exc}"

        return AgentResult(
            answer=answer.strip(),
            steps=steps,
            used_skills=used_skills,
            iterations=MAX_ITERATIONS,
        )
