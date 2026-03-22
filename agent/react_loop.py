"""
agent/react_loop.py — Nexara V1
ReAct engine with:
  • SkillClassifier — selects 4-8 relevant skills before the loop
    (drops token cost from ~2000 to ~150 per call)
  • Dual execution path — native FC for capable models, text injection
    fallback for others (no more leaked JSON blocks)
  • Rich status messages — user sees exactly what the agent is doing
  • Per-user concurrency semaphore
  • JSON retry + replan on failure
  • Per-step LLM and skill timeouts
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any

from agent.llm_router import LLMRouter, LLMResponse, ToolCall, TextResponse, AskUser
from agent.tool_schema import gemini_tools, _SCHEMA_MAP
from utils.skill_classifier import SkillClassifier, skill_label

logger = logging.getLogger("nexara.react")

MAX_ITERATIONS   = 14
MAX_REPLAN_DEPTH = 3
STEP_LLM_TIMEOUT = 45
SKILL_TIMEOUT    = 60
JSON_RETRY_LIMIT = 2

_USER_SEMS: dict[int, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(2))
_classifier = SkillClassifier()


# ── Status message helpers ────────────────────────────────────────────────────

def _think_status(step: int, goal: str) -> str:
    """Generate a contextual thinking status based on the goal."""
    g = goal.lower()

    if any(w in g for w in ("search", "research", "find", "look up", "what is", "who is")):
        return f"🔍 Researching... (step {step})"
    if any(w in g for w in ("pdf", "docx", "document", "report", "file", "generate", "create")):
        return f"📄 Preparing document... (step {step})"
    if any(w in g for w in ("install", "apt", "package", "java", "node", "python")):
        return f"📦 Planning installation... (step {step})"
    if any(w in g for w in ("code", "script", "program", "function", "debug", "fix")):
        return f"⚙️ Writing code... (step {step})"
    if any(w in g for w in ("schedule", "remind", "every", "daily", "timer")):
        return f"📅 Setting up schedule... (step {step})"
    if any(w in g for w in ("download", "fetch", "get file", "youtube", "video")):
        return f"⬇️ Preparing download... (step {step})"
    if any(w in g for w in ("translate", "translation")):
        return f"🌐 Translating... (step {step})"
    if any(w in g for w in ("weather", "temperature", "forecast")):
        return f"🌤️ Checking weather... (step {step})"
    if any(w in g for w in ("speed", "ping", "bandwidth", "internet")):
        return f"🚀 Testing connection... (step {step})"
    if any(w in g for w in ("disk", "storage", "space", "memory", "cpu", "system")):
        return f"🖥️ Checking system... (step {step})"
    if any(w in g for w in ("send", "email", "discord", "slack", "message")):
        return f"📧 Preparing message... (step {step})"
    if any(w in g for w in ("image", "photo", "picture", "resize", "convert")):
        return f"🖼️ Processing image... (step {step})"
    if any(w in g for w in ("database", "sql", "query", "data")):
        return f"🗄️ Querying data... (step {step})"
    if any(w in g for w in ("git", "commit", "push", "pull", "branch")):
        return f"🔧 Git operation... (step {step})"
    if any(w in g for w in ("docker", "container", "image")):
        return f"🐳 Docker operation... (step {step})"

    return f"🧠 Analysing... (step {step})"


def _act_status(skill_name: str, args: dict, thought: str) -> str:
    """Generate a status message for an action being executed."""
    label = skill_label(skill_name, args)
    if thought:
        preview = thought[:50].rstrip()
        return f"{label} — {preview}{'…' if len(thought) > 50 else ''}"
    return label


# ── Data classes ──────────────────────────────────────────────────────────────

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


# ── ReactLoop ─────────────────────────────────────────────────────────────────

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
        goal:           str,
        history:        list[dict],
        system_prompt:  str,
        tools:          list[dict] | None = None,
        active_skills:  list[str] | None  = None,
        user_id:        int = 0,
        status_cb:      Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        sem = _USER_SEMS[user_id]
        async with sem:
            return await self._run_inner(
                goal, history, system_prompt, tools, active_skills, status_cb
            )

    @staticmethod
    async def _status(cb, msg: str):
        """Send a status update. Plain text only — no Markdown."""
        if cb:
            try:
                await cb(msg)
            except Exception:
                pass

    async def _run_inner(
        self, goal, history, system_prompt, tools, active_skills, status_cb
    ) -> AgentResult:

        # ── Skill classification ───────────────────────────────────────────────
        # Select only relevant skills for this request.
        # Falls back to full tool list if no active_skills provided.
        if active_skills and tools:
            selected      = _classifier.select(goal, active_skills)
            filtered_tools = self._filter_tools(tools, selected)
            logger.info("Classifier: '%s' → %d skills: %s",
                        goal[:60], len(selected), selected)
        else:
            filtered_tools = tools
            selected       = []

        steps:         list[Step] = []
        used_skills:   list[str]  = []
        replan_budget              = MAX_REPLAN_DEPTH
        json_retries               = 0

        messages = list(history)
        messages.append({"role": "user", "parts": [goal]})

        for i in range(1, MAX_ITERATIONS + 1):

            # ── THINK ────────────────────────────────────────────────────────
            await self._status(status_cb, _think_status(i, goal))

            try:
                response: LLMResponse = await asyncio.wait_for(
                    self._router.complete(
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=filtered_tools,
                    ),
                    timeout=STEP_LLM_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timed out on step %d", i)
                await self._status(status_cb, "⚠️ LLM timed out — wrapping up...")
                break
            except Exception as exc:
                logger.error("LLM error on step %d: %s", i, exc)
                return AgentResult(
                    answer=f"I ran into an error communicating with the AI: {exc}",
                    steps=steps, used_skills=used_skills, iterations=i,
                )

            # ── TERMINAL: text response ───────────────────────────────────────
            if isinstance(response, TextResponse):
                raw = response.text

                # Empty = suppressed leaked JSON → retry
                if not raw.strip() and json_retries < JSON_RETRY_LIMIT:
                    json_retries += 1
                    messages.append({
                        "role": "user",
                        "parts": [
                            "Your last response could not be parsed. "
                            "Please emit a valid JSON block. "
                            'Use {"action":"final_answer","answer":"..."} if done, '
                            'or {"action":"<skill>","args":{...},"thought":"..."} for a tool.'
                        ],
                    })
                    await self._status(status_cb, f"🔄 Retrying... (attempt {json_retries})")
                    continue

                # Raw JSON action leaked as text → retry
                if (json_retries < JSON_RETRY_LIMIT
                        and raw.strip().startswith("{")
                        and '"action"' in raw
                        and '"final_answer"' not in raw):
                    json_retries += 1
                    messages.append({
                        "role": "user",
                        "parts": [
                            f"You returned a tool call as plain text instead of executing it: {raw[:200]}\n"
                            "Wrap the JSON in a ```json block so it executes properly."
                        ],
                    })
                    await self._status(status_cb, f"🔄 Fixing tool call format... (attempt {json_retries})")
                    continue

                # Clean text answer
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
            json_retries  = 0

            await self._status(status_cb, _act_status(tc.name, tc.args, tc.thought))

            try:
                result = await asyncio.wait_for(
                    self._exec(tc.name, tc.args),
                    timeout=SKILL_TIMEOUT,
                )
                obs = str(result)
                ok  = getattr(result, "success", True)
            except asyncio.TimeoutError:
                obs = f"[{tc.name} timed out after {SKILL_TIMEOUT}s — try a different approach]"
                ok  = False
                logger.warning("Skill '%s' timed out", tc.name)
            except Exception as exc:
                obs = f"[Fatal error in {tc.name}: {exc}]"
                ok  = False
                logger.error("Skill '%s' raised: %s", tc.name, exc)

            used_skills.append(tc.name)
            step = Step(i, tc.thought, tc.name, tc.args, obs, ok)
            steps.append(step)

            logger.info("Step %d/%d: %s → ok=%s | %s",
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
                await self._status(status_cb, f"🔁 Trying alternative approach...")
                continue

            # ── OBSERVE ───────────────────────────────────────────────────────
            # Use proper tool role if the model used native FC
            if tc.tool_call_id:
                # Append assistant message with tool_calls first
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": __import__("json").dumps(tc.args),
                        },
                    }],
                })
                # Then tool result
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.tool_call_id,
                    "content":      obs,
                })
            else:
                # Text injection path — use user role
                messages.append({
                    "role": "user",
                    "parts": [
                        f"[Result from '{tc.name}']:\n{obs}\n\n"
                        "Continue with the next step, or emit a final_answer if complete."
                    ],
                })

            # Update thinking status with what was found
            if ok and len(obs) > 20:
                await self._status(status_cb, f"✅ Got result from {tc.name}. Analysing...")

        # ── Max iterations — force summary ────────────────────────────────────
        await self._status(status_cb, "📝 Summarising results...")
        messages.append({
            "role": "user",
            "parts": ["Maximum steps reached. Summarise everything as a final_answer now."],
        })
        try:
            response = await asyncio.wait_for(
                self._router.complete(messages, system_prompt, filtered_tools),
                timeout=STEP_LLM_TIMEOUT,
            )
            answer = response.text if isinstance(response, TextResponse) else (
                f"Completed {len(steps)} steps. Last: {steps[-1].action if steps else 'none'}."
            )
        except Exception as exc:
            answer = f"Completed {len(steps)} steps but failed to summarise: {exc}"

        return AgentResult(
            answer=answer.strip(),
            steps=steps,
            used_skills=used_skills,
            iterations=MAX_ITERATIONS,
        )

    # ── Tool filtering ────────────────────────────────────────────────────────

    @staticmethod
    def _filter_tools(tools: list[dict], selected_names: list[str]) -> list[dict]:
        """Filter the full tool list down to only selected skill names."""
        if not selected_names:
            return tools
        selected_set = set(selected_names)
        result = []
        for tg in tools:
            filtered_decls = [
                fn for fn in tg.get("function_declarations", [])
                if fn.get("name") in selected_set
            ]
            if filtered_decls:
                result.append({"function_declarations": filtered_decls})
        return result if result else tools  # never return empty
