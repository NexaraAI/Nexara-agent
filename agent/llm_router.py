"""
agent/llm_router.py — Nexara V1
Multi-provider LLM router with:
  • 4 providers: Groq · Gemini · NVIDIA NIM · Ollama
  • Configurable primary provider (PRIMARY_PROVIDER in .env)
  • Per-provider token usage tracking (pre-emptive 429 avoidance)
  • Proactive rate-limit switching with user notification
  • Runtime model switching without restart
  • Live model listing from provider APIs (/switchmodel)
  • Response metadata: provider, model, elapsed time
"""

import asyncio
import json
import logging
import re as _re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger("nexara.llm_router")


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name:    str
    args:    dict
    thought: str = ""

@dataclass
class TextResponse:
    text: str

@dataclass
class AskUser:
    question: str

LLMResponse = ToolCall | TextResponse | AskUser


@dataclass
class ResponseMeta:
    provider: str
    model:    str
    elapsed:  float
    tokens:   int

    def badge(self) -> str:
        icons = {"groq": "⚡", "gemini": "🔵", "nvidia": "🟢", "ollama": "📡"}
        icon  = icons.get(self.provider, "🤖")
        return f"`{icon} {self.provider} · {self.model} · {self.elapsed:.1f}s · ~{self.tokens}tok`"


# ── Providers ─────────────────────────────────────────────────────────────────

class Provider(Enum):
    GROQ   = "groq"
    GEMINI = "gemini"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"

_LIMITS = {
    Provider.GROQ:   6_000,
    Provider.GEMINI: 32_000,
    Provider.NVIDIA: 4_000,
    Provider.OLLAMA: 999_999,
}

_SOFT_PCT = 0.80


@dataclass
class ProviderHealth:
    provider:        Provider
    failures:        int   = 0
    last_failure:    float = 0.0
    cooldown_s:      float = 5.0
    tokens_this_min: int   = 0
    min_start:       float = field(default_factory=time.time)

    def is_available(self) -> bool:
        if self.failures == 0:
            return True
        return (time.time() - self.last_failure) > self.cooldown_s

    def is_near_limit(self) -> bool:
        self._maybe_reset_minute()
        limit = _LIMITS.get(self.provider, 4_000)
        return self.tokens_this_min >= int(limit * _SOFT_PCT)

    def add_tokens(self, n: int):
        self._maybe_reset_minute()
        self.tokens_this_min += n

    def _maybe_reset_minute(self):
        if time.time() - self.min_start > 60:
            self.tokens_this_min = 0
            self.min_start       = time.time()

    def record_failure(self):
        self.failures    += 1
        self.last_failure = time.time()
        # 1 fail=5s, 2=10s, 3=20s, 4=40s … cap 300s — much less aggressive
        self.cooldown_s   = min(300.0, 5.0 * (2 ** (self.failures - 1)))
        logger.warning("Provider %s failed (%d×). Cooldown %.0fs",
                       self.provider.value, self.failures, self.cooldown_s)

    def record_success(self):
        self.failures   = 0
        self.cooldown_s = 5.0


# ── Router ────────────────────────────────────────────────────────────────────

class LLMRouter:

    def __init__(self, config):
        self._cfg = config

        self._active: dict[Provider, str] = {
            Provider.GROQ:   config.GROQ_MODEL,
            Provider.GEMINI: config.LLM_MODEL,
            Provider.NVIDIA: config.NVIDIA_MODEL,
            Provider.OLLAMA: config.OLLAMA_MODEL,
        }

        self._health: dict[Provider, ProviderHealth] = {
            p: ProviderHealth(p) for p in Provider
        }

        self._gemini_client = None
        self._groq_client   = None
        self._nvidia_client = None
        self._init_clients()

        self.on_switch_cb               = None
        self.last_meta: ResponseMeta | None = None

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_clients(self):
        try:
            import google.generativeai as genai
            if self._cfg.LLM_API_KEY:
                genai.configure(api_key=self._cfg.LLM_API_KEY)
                self._gemini_client = genai
                logger.info("Gemini ready (%s)", self._active[Provider.GEMINI])
        except Exception as exc:
            logger.warning("Gemini init failed: %s", exc)

        try:
            if self._cfg.GROQ_API_KEY:
                import openai
                self._groq_client = openai.AsyncOpenAI(
                    api_key=self._cfg.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                logger.info("Groq ready (%s)", self._active[Provider.GROQ])
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)

        try:
            if self._cfg.NVIDIA_API_KEY:
                import openai
                self._nvidia_client = openai.AsyncOpenAI(
                    api_key=self._cfg.NVIDIA_API_KEY,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
                logger.info("NVIDIA NIM ready (%s)", self._active[Provider.NVIDIA])
        except Exception as exc:
            logger.warning("NVIDIA NIM init failed: %s", exc)

    # ── Provider chain ────────────────────────────────────────────────────────

    def _chain(self) -> list[Provider]:
        primary_name = getattr(self._cfg, "PRIMARY_PROVIDER", "groq").lower()
        mapping      = {p.value: p for p in Provider}
        primary      = mapping.get(primary_name, Provider.GROQ)
        rest         = [p for p in [Provider.GROQ, Provider.GEMINI, Provider.NVIDIA, Provider.OLLAMA]
                        if p != primary]
        return [primary] + rest

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages:         list[dict],
        system_prompt:    str,
        tools:            list[dict] | None = None,
        estimated_tokens: int = 0,
    ) -> LLMResponse:
        chain     = self._chain()
        prev_prov = None

        for provider in chain:
            h = self._health[provider]

            if not h.is_available():
                logger.debug("Skipping %s (cooldown)", provider.value)
                continue

            if h.is_near_limit():
                logger.info("%s near token limit — skipping", provider.value)
                if self.on_switch_cb and prev_prov:
                    await self.on_switch_cb(
                        f"⚠️ `{prev_prov}` near rate limit — switching to `{provider.value}`"
                    )
                continue

            t0 = time.time()
            try:
                result = await asyncio.wait_for(
                    self._call(provider, messages, system_prompt, tools),
                    timeout=90,
                )
                elapsed = time.time() - t0
                h.record_success()
                h.add_tokens(estimated_tokens or 500)

                self.last_meta = ResponseMeta(
                    provider=provider.value,
                    model=self._active[provider],
                    elapsed=elapsed,
                    tokens=estimated_tokens or 500,
                )

                if prev_prov and self.on_switch_cb:
                    await self.on_switch_cb(
                        f"↩️ Switched from `{prev_prov}` → `{provider.value}` "
                        f"(`{self._active[provider]}`)"
                    )

                logger.debug("Response from %s in %.1fs", provider.value, elapsed)
                return result

            except asyncio.TimeoutError:
                logger.warning("%s timed out", provider.value)
                h.record_failure()
                prev_prov = provider.value
                if self.on_switch_cb:
                    await self.on_switch_cb(
                        f"⏱ `{provider.value}` timed out — trying next provider…"
                    )

            except Exception as exc:
                logger.warning("%s error: %s", provider.value, exc)
                h.record_failure()
                prev_prov = provider.value
                if "429" in str(exc) or "rate" in str(exc).lower():
                    if self.on_switch_cb:
                        await self.on_switch_cb(
                            f"🚫 `{provider.value}` rate limited — switching provider…"
                        )

        raise RuntimeError("All LLM providers exhausted. Check API keys and network.")

    # ── Model switching ───────────────────────────────────────────────────────

    def switch_model(self, provider: str, model: str) -> str:
        p = {p.value: p for p in Provider}.get(provider.lower())
        if not p:
            return f"❌ Unknown provider `{provider}`"
        self._active[p] = model
        return f"✅ `{provider}` → `{model}`"

    def set_primary(self, provider: str) -> str:
        if provider.lower() not in {p.value for p in Provider}:
            return f"❌ Unknown provider `{provider}`"
        self._cfg.PRIMARY_PROVIDER = provider.lower()
        return f"⭐ `{provider}` set as primary provider"

    def active_models(self) -> dict[str, str]:
        return {p.value: m for p, m in self._active.items()}

    # ── Live model fetching ───────────────────────────────────────────────────

    async def fetch_gemini_models(self) -> list[str]:
        if not self._cfg.LLM_API_KEY:
            return []
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={self._cfg.LLM_API_KEY}"
            )
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            models = [
                m["name"].replace("models/", "")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            def _rank(n):
                if "flash" in n: return 0
                if "pro"   in n: return 1
                return 2
            return sorted(models, key=_rank)
        except Exception as exc:
            logger.warning("fetch_gemini_models: %s", exc)
            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]

    async def fetch_groq_models(self) -> list[str]:
        if not self._groq_client:
            return []
        try:
            resp = await self._groq_client.models.list()
            skip = ("whisper", "tts", "embed", "guard")
            return sorted(m.id for m in resp.data if not any(s in m.id for s in skip))
        except Exception as exc:
            logger.warning("fetch_groq_models: %s", exc)
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    async def fetch_nvidia_models(self) -> list[str]:
        if not self._nvidia_client:
            return []
        try:
            resp = await self._nvidia_client.models.list()
            skip = ("embed", "rerank", "guardrail", "vision")
            return sorted(m.id for m in resp.data if not any(s in m.id for s in skip))
        except Exception as exc:
            logger.warning("fetch_nvidia_models: %s", exc)
            return [
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-405b-instruct",
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "mistralai/mixtral-8x7b-instruct-v0.1",
            ]

    async def fetch_ollama_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(f"{self._cfg.OLLAMA_URL}/api/tags")
                resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> str:
        primary = getattr(self._cfg, "PRIMARY_PROVIDER", "groq")
        lines   = ["🤖 **LLM Router Status**\n"]
        icons   = {Provider.GROQ: "⚡", Provider.GEMINI: "🔵",
                   Provider.NVIDIA: "🟢", Provider.OLLAMA: "📡"}

        for p in self._chain():
            h     = self._health[p]
            avail = "🟢" if h.is_available() else "🔴"
            near  = " ⚠️ _near limit_" if h.is_near_limit() else ""
            star  = " ⭐" if p.value == primary else ""
            model = self._active[p]
            lines.append(
                f"{avail} {icons[p]} `{p.value}`{star} — `{model}`"
                + (f"  _{h.tokens_this_min} tok/min_" if h.tokens_this_min else "")
                + near
                + (f"  _(cooldown {h.cooldown_s:.0f}s)_" if not h.is_available() else "")
            )
        return "\n".join(lines)

    def token_usage(self) -> dict[str, int]:
        return {p.value: self._health[p].tokens_this_min for p in Provider}

    # ── Provider dispatch ─────────────────────────────────────────────────────

    async def _call(self, provider, messages, system_prompt, tools) -> LLMResponse:
        if provider == Provider.GEMINI: return await self._gemini(messages, system_prompt, tools)
        if provider == Provider.GROQ:   return await self._groq(messages, system_prompt, tools)
        if provider == Provider.NVIDIA: return await self._nvidia(messages, system_prompt, tools)
        if provider == Provider.OLLAMA: return await self._ollama(messages, system_prompt, tools)
        raise ValueError(f"Unknown provider: {provider}")

    # ── Gemini ────────────────────────────────────────────────────────────────

    async def _gemini(self, messages, system_prompt, tools) -> LLMResponse:
        if not self._gemini_client:
            raise RuntimeError("Gemini not configured — set LLM_API_KEY")
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=self._active[Provider.GEMINI],
            system_instruction=system_prompt,
        )
        gemini_msgs = [
            {"role": "model" if m["role"] == "assistant" else m["role"],
             "parts": m.get("parts", [m.get("content", "")])}
            for m in messages
        ]
        kwargs: dict[str, Any] = {"contents": gemini_msgs}
        if tools:
            from google.generativeai import protos
            fn_decls = [
                protos.FunctionDeclaration(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=fn.get("parameters", {}),
                )
                for tg in tools for fn in tg.get("function_declarations", [])
            ]
            kwargs["tools"] = [protos.Tool(function_declarations=fn_decls)]
        response = await asyncio.to_thread(lambda: model.generate_content(**kwargs))
        try:
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    return ToolCall(name=fc.name, args=dict(fc.args) if fc.args else {})
            text = response.text or ""
        except ValueError as exc:
            fb     = getattr(response, "prompt_feedback", None)
            reason = str(fb) if fb else str(exc)
            logger.warning("Gemini blocked: %s", reason)
            text = f"[Blocked: {reason}]"
        return _parse_text(text)

    # ── Groq ──────────────────────────────────────────────────────────────────

    async def _groq(self, messages, system_prompt, tools) -> LLMResponse:
        if not self._groq_client:
            raise RuntimeError("Groq not configured — set GROQ_API_KEY")
        return await self._openai_compat(
            self._groq_client, self._active[Provider.GROQ],
            messages, system_prompt, tools,
        )

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────

    async def _nvidia(self, messages, system_prompt, tools) -> LLMResponse:
        if not self._nvidia_client:
            raise RuntimeError("NVIDIA NIM not configured — set NVIDIA_API_KEY")
        return await self._openai_compat(
            self._nvidia_client, self._active[Provider.NVIDIA],
            messages, system_prompt, tools,
        )

    # ── Ollama ────────────────────────────────────────────────────────────────

    async def _ollama(self, messages, system_prompt, tools) -> LLMResponse:
        msgs = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role    = m.get("role", "user")
            content = (" ".join(m.get("parts", [])) if isinstance(m.get("parts"), list)
                       else m.get("content", ""))
            msgs.append({"role": role, "content": content})
        if tools:
            msgs[0]["content"] += f"\n\n{_tools_to_text(tools)}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._cfg.OLLAMA_URL}/api/chat",
                json={"model": self._active[Provider.OLLAMA], "messages": msgs, "stream": False},
            )
            resp.raise_for_status()
        return _parse_text(resp.json().get("message", {}).get("content", ""))

    # ── OpenAI-compatible (Groq + NVIDIA) ─────────────────────────────────────
    # IMPORTANT: We do NOT use native function-calling/tool_choice here.
    # Groq returns 400 "tool_use_failed" when the model tries to emit a tool
    # call for a function not explicitly declared in request_tools — which
    # happens when the system prompt (ReAct JSON) and native tools conflict.
    # Solution: inject tool descriptions as plain text (same as Ollama).
    # Gemini is unaffected since it uses its own function_declarations API.

    async def _openai_compat(self, client, model, messages, system_prompt, tools) -> LLMResponse:
        sp = system_prompt + (f"\n\n{_tools_to_text(tools)}" if tools else "")
        msgs: list[dict] = [{"role": "system", "content": sp}]
        for m in messages:
            role    = m.get("role", "user")
            if role == "model": role = "assistant"
            content = (" ".join(m.get("parts", [])) if isinstance(m.get("parts"), list)
                       else m.get("content", ""))
            msgs.append({"role": role, "content": content})
        response = await client.chat.completions.create(model=model, messages=msgs)
        choice   = response.choices[0]
        return _parse_text(choice.message.content or "")


# ── Shared text helpers ───────────────────────────────────────────────────────

_JSON_RE = _re.compile(r"```json\s*(\{.*?\})\s*```", _re.DOTALL)


def _extract_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1: return None
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:                escape = False; continue
        if ch == "\\" and in_str: escape = True;  continue
        if ch == '"':             in_str = not in_str; continue
        if in_str:                continue
        if ch == "{":             depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return text[start:i + 1]
    return None


def _parse_text(text: str) -> LLMResponse:
    m   = _JSON_RE.search(text)
    raw = m.group(1) if m else _extract_json(text)
    if raw:
        try:
            obj    = json.loads(raw)
            action = obj.get("action", "")
            if action == "final_answer": return TextResponse(text=obj.get("answer", text))
            if action == "ask_user":     return AskUser(question=obj.get("question", ""))
            if action:
                return ToolCall(name=action, args=obj.get("args", {}), thought=obj.get("thought", ""))
        except json.JSONDecodeError:
            pass
    return TextResponse(text=text.strip())


def _tools_to_text(tools: list[dict]) -> str:
    lines = [
        "## Available Tools\nRespond with ONLY this JSON when using a tool:\n"
        '```json\n{"action":"<tool>","args":{<args>},"thought":"why"}\n```\n'
        'When done: ```json\n{"action":"final_answer","answer":"<answer>"}\n```\n\n'
        "### Tool list:"
    ]
    for tg in tools:
        for fn in tg.get("function_declarations", []):
            props = fn.get("parameters", {}).get("properties", {})
            args  = ", ".join(f"{k}: {v.get('description','')}" for k, v in props.items())
            lines.append(f"- **{fn['name']}**({args}) — {fn['description']}")
    return "\n".join(lines)
