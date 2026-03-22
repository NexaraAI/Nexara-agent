"""
agent/llm_router.py — Nexara V1
Multi-provider LLM router with:
  • 4 providers: Groq · Gemini · NVIDIA NIM · Ollama
  • Per-model function-calling capability detection
    - Known FC-capable models → native tools API (clean, typed, reliable)
    - Unknown / incapable models → text injection fallback (safe default)
  • Single-provider retry with backoff (no false "all exhausted" errors)
  • Configurable primary provider
  • Per-provider token tracking and rate-limit pre-emption
  • Runtime model switching + live model listing
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


# ── Function-calling capability registry ─────────────────────────────────────
# Maps known model names to FC support. Unknown models default to False (safe).
# Groq: https://console.groq.com/docs/tool-use
# NVIDIA: varies by model — only llama-3.x and mistral-large are reliable

FC_CAPABLE_MODELS: frozenset[str] = frozenset({
    # Groq — llama-3.x are reliable; others less so
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "qwen-qwq-32b",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "mistral-saba-24b",
    "compound-beta",
    "compound-beta-mini",
    # NVIDIA — llama-3.x and mistral-large only
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "qwen/qwen2.5-72b-instruct",
    # Gemini — always supports native FC, handled separately
})


def supports_function_calling(provider: str, model: str) -> bool:
    """
    Returns True if the provider+model combination reliably supports
    native OpenAI-style function calling.

    Gemini always uses its own native FC API.
    Ollama never supports FC — text injection only.
    Groq/NVIDIA: only known-good models get native FC; unknown models
    fall back to text injection to avoid 400 errors and JSON leaks.
    """
    if provider == "gemini":
        return True   # Gemini always uses its own native FC
    if provider == "ollama":
        return False  # Ollama has no function calling support
    return model in FC_CAPABLE_MODELS


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name:         str
    args:         dict
    thought:      str  = ""
    tool_call_id: str  = ""   # set when using native FC

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
    native_fc: bool = False

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

_SOFT_PCT    = 0.80
_MAX_RETRIES = 3   # single-provider retry attempts before giving up


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
                logger.info("Groq ready (%s) FC=%s",
                            self._active[Provider.GROQ],
                            supports_function_calling("groq", self._active[Provider.GROQ]))
        except Exception as exc:
            logger.warning("Groq init failed: %s", exc)

        try:
            if self._cfg.NVIDIA_API_KEY:
                import openai
                self._nvidia_client = openai.AsyncOpenAI(
                    api_key=self._cfg.NVIDIA_API_KEY,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
                logger.info("NVIDIA NIM ready (%s) FC=%s",
                            self._active[Provider.NVIDIA],
                            supports_function_calling("nvidia", self._active[Provider.NVIDIA]))
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

    def _configured_providers(self) -> list[Provider]:
        """Return only providers that actually have credentials."""
        result = []
        if self._groq_client:   result.append(Provider.GROQ)
        if self._gemini_client: result.append(Provider.GEMINI)
        if self._nvidia_client: result.append(Provider.NVIDIA)
        result.append(Provider.OLLAMA)  # Ollama is always "configured" (local)
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages:         list[dict],
        system_prompt:    str,
        tools:            list[dict] | None = None,
        estimated_tokens: int = 0,
    ) -> LLMResponse:
        """
        Try providers in priority order until one succeeds.

        Single-provider resilience: if only one provider is configured
        and it hits a transient error, retry with exponential backoff
        instead of immediately returning "all providers exhausted".
        """
        chain      = self._chain()
        configured = set(self._configured_providers())
        prev_prov  = None

        # Filter chain to configured providers only
        active_chain = [p for p in chain if p in configured]

        if not active_chain:
            raise RuntimeError("No LLM providers configured. Add at least one API key.")

        single_provider = len(active_chain) == 1

        for provider in active_chain:
            h = self._health[provider]

            if not h.is_available():
                if single_provider:
                    # Wait out the cooldown instead of giving up
                    wait = h.cooldown_s - (time.time() - h.last_failure)
                    if wait > 0:
                        logger.info("Single provider %s in cooldown — waiting %.0fs",
                                    provider.value, wait)
                        await asyncio.sleep(min(wait, 30))
                else:
                    logger.debug("Skipping %s (cooldown)", provider.value)
                    continue

            if h.is_near_limit() and not single_provider:
                logger.info("%s near token limit — skipping", provider.value)
                if self.on_switch_cb and prev_prov:
                    await self.on_switch_cb(
                        f"⚠️ `{prev_prov}` near rate limit — switching to `{provider.value}`"
                    )
                continue

            # Retry loop for single-provider setups
            max_attempts = _MAX_RETRIES if single_provider else 1

            for attempt in range(1, max_attempts + 1):
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
                        native_fc=supports_function_calling(
                            provider.value, self._active[provider]
                        ),
                    )

                    if prev_prov and self.on_switch_cb:
                        await self.on_switch_cb(
                            f"↩️ Switched from `{prev_prov}` → `{provider.value}` "
                            f"(`{self._active[provider]}`)"
                        )

                    return result

                except asyncio.TimeoutError:
                    logger.warning("%s timed out (attempt %d/%d)",
                                   provider.value, attempt, max_attempts)
                    h.record_failure()
                    prev_prov = provider.value
                    if self.on_switch_cb:
                        await self.on_switch_cb(
                            f"⏱ `{provider.value}` timed out — trying next…"
                        )
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)

                except Exception as exc:
                    logger.warning("%s error (attempt %d/%d): %s",
                                   provider.value, attempt, max_attempts, exc)
                    h.record_failure()
                    prev_prov = provider.value
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        if self.on_switch_cb:
                            await self.on_switch_cb(
                                f"🚫 `{provider.value}` rate limited — switching…"
                            )
                        break  # Rate limited — skip retries, try next provider
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            "All LLM providers exhausted or unavailable. "
            "Check your API keys and network connection."
        )

    # ── Model switching ───────────────────────────────────────────────────────

    def switch_model(self, provider: str, model: str) -> str:
        p = {p.value: p for p in Provider}.get(provider.lower())
        if not p:
            return f"❌ Unknown provider `{provider}`"
        self._active[p] = model
        fc = supports_function_calling(provider.lower(), model)
        fc_note = " (native FC)" if fc else " (text injection)"
        return f"✅ `{provider}` → `{model}`{fc_note}"

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
            return ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro-exp-03-25", "gemini-3.1-pro-preview"]

    async def fetch_groq_models(self) -> list[str]:
        if not self._groq_client:
            return []
        try:
            resp = await self._groq_client.models.list()
            skip = ("whisper", "tts", "embed", "guard")
            return sorted(m.id for m in resp.data if not any(s in m.id for s in skip))
        except Exception as exc:
            logger.warning("fetch_groq_models: %s", exc)
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

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
            near  = " ⚠️ near limit" if h.is_near_limit() else ""
            star  = " ⭐" if p.value == primary else ""
            model = self._active[p]
            fc    = " [FC]" if supports_function_calling(p.value, model) else " [text]"
            configured = p in set(self._configured_providers())
            cfg_note   = "" if configured else " (no key)"
            lines.append(
                f"{avail} {icons[p]} `{p.value}`{star} — `{model}`{fc}{cfg_note}"
                + (f"  {h.tokens_this_min} tok/min" if h.tokens_this_min else "")
                + near
                + (f"  (cooldown {h.cooldown_s:.0f}s)" if not h.is_available() else "")
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

    # ── Gemini (always native FC) ─────────────────────────────────────────────

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
            if m.get("role") not in ("tool",)  # skip tool-role messages for Gemini
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
            self._groq_client, Provider.GROQ,
            messages, system_prompt, tools,
        )

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────

    async def _nvidia(self, messages, system_prompt, tools) -> LLMResponse:
        if not self._nvidia_client:
            raise RuntimeError("NVIDIA NIM not configured — set NVIDIA_API_KEY")
        return await self._openai_compat(
            self._nvidia_client, Provider.NVIDIA,
            messages, system_prompt, tools,
        )

    # ── Ollama (always text injection) ────────────────────────────────────────

    async def _ollama(self, messages, system_prompt, tools) -> LLMResponse:
        sp = system_prompt + (f"\n\n{_tools_to_text(tools)}" if tools else "")
        msgs = [{"role": "system", "content": sp}]
        for m in messages:
            role = m.get("role", "user")
            if role in ("model", "tool"): role = "assistant"
            content = (" ".join(m.get("parts", [])) if isinstance(m.get("parts"), list)
                       else m.get("content", ""))
            msgs.append({"role": role, "content": content})
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._cfg.OLLAMA_URL}/api/chat",
                json={"model": self._active[Provider.OLLAMA], "messages": msgs, "stream": False},
            )
            resp.raise_for_status()
        return _parse_text(resp.json().get("message", {}).get("content", ""))

    # ── OpenAI-compatible (Groq + NVIDIA) ─────────────────────────────────────
    # Per-model FC detection:
    #   - Known FC-capable models → native tools API (clean, typed, no JSON parsing)
    #   - Unknown / unreliable models → text injection (safe fallback, always works)
    # This fixes the original Groq 400 error (was caused by passing ALL 78 tools
    # to models that don't support them). With 4-8 filtered tools + known-good
    # models, native FC works correctly.

    async def _openai_compat(
        self, client, provider: Provider, messages, system_prompt, tools
    ) -> LLMResponse:
        model     = self._active[provider]
        use_fc    = tools and supports_function_calling(provider.value, model)

        if use_fc:
            return await self._openai_native_fc(client, model, messages, system_prompt, tools)
        else:
            return await self._openai_text_inject(client, model, messages, system_prompt, tools)

    async def _openai_native_fc(
        self, client, model, messages, system_prompt, tools
    ) -> LLMResponse:
        """Use OpenAI native function calling for FC-capable models."""
        msgs: list[dict] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m.get("role", "user")
            if role == "model": role = "assistant"
            if role == "tool":
                # Proper tool result message
                msgs.append({
                    "role":         "tool",
                    "tool_call_id": m.get("tool_call_id", "unknown"),
                    "content":      m.get("content", ""),
                })
                continue
            content = (" ".join(m.get("parts", [])) if isinstance(m.get("parts"), list)
                       else m.get("content", ""))
            # If this assistant message had a tool call, include it
            if role == "assistant" and m.get("tool_calls"):
                msgs.append({"role": "assistant", "tool_calls": m["tool_calls"],
                             "content": content or ""})
            else:
                msgs.append({"role": role, "content": content})

        # Build OpenAI-format tool schemas
        oa_tools = []
        for tg in (tools or []):
            for fn in tg.get("function_declarations", []):
                oa_tools.append({"type": "function", "function": fn})

        response = await client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=oa_tools,
            tool_choice="auto",
        )
        choice = response.choices[0]

        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, Exception):
                args = {}
            return ToolCall(
                name=tc.function.name,
                args=args,
                tool_call_id=tc.id,
            )
        return _parse_text(choice.message.content or "")

    async def _openai_text_inject(
        self, client, model, messages, system_prompt, tools
    ) -> LLMResponse:
        """Text injection fallback for models without reliable FC support."""
        sp = system_prompt + (f"\n\n{_tools_to_text(tools)}" if tools else "")
        msgs: list[dict] = [{"role": "system", "content": sp}]
        for m in messages:
            role = m.get("role", "user")
            if role in ("model", "tool", "assistant"):
                role = "assistant"
            content = (" ".join(m.get("parts", [])) if isinstance(m.get("parts"), list)
                       else m.get("content", ""))
            if content:
                msgs.append({"role": role, "content": content})
        response = await client.chat.completions.create(model=model, messages=msgs)
        choice   = response.choices[0]
        return _parse_text(choice.message.content or "")


# ── Text parsing helpers ──────────────────────────────────────────────────────
# Handles: fenced ```json blocks, bare JSON objects, two concatenated objects.

_JSON_RE = _re.compile(r"```(?:json)?\s*(.*?)\s*```", _re.DOTALL)


def _extract_json(text: str) -> str | None:
    """Extract first complete JSON object from text."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:                escape = False; continue
        if ch == "\\" and in_str: escape = True;  continue
        if ch == '"':             in_str = not in_str; continue
        if in_str:                continue
        if ch == "{":             depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_parse_action(raw: str) -> LLMResponse | None:
    """Parse a raw JSON string into a structured response. Returns None on failure."""
    if not raw:
        return None
    first = _extract_json(raw)
    if not first:
        return None
    try:
        obj    = json.loads(first)
        action = obj.get("action", "")
        if action == "final_answer":
            return TextResponse(text=obj.get("answer", ""))
        if action == "ask_user":
            return AskUser(question=obj.get("question", ""))
        if action:
            return ToolCall(
                name=action,
                args=obj.get("args", {}),
                thought=obj.get("thought", ""),
            )
    except (json.JSONDecodeError, Exception):
        pass
    return None


def _parse_text(text: str) -> LLMResponse:
    """
    Parse LLM text output into a structured response.
    1. Try all fenced code blocks
    2. Try bare JSON anywhere in text
    3. Suppress leaked JSON tool calls (return empty → retry in react_loop)
    4. Return as plain text
    """
    # Try every fenced code block
    for m in _JSON_RE.finditer(text):
        result = _try_parse_action(m.group(1))
        if result is not None:
            return result

    # Try bare JSON
    result = _try_parse_action(text)
    if result is not None:
        return result

    # Suppress leaked JSON action blocks — react_loop will retry
    stripped = text.strip()
    if stripped.startswith("{") and '"action"' in stripped:
        logger.warning("Suppressing leaked JSON action in TextResponse: %s", stripped[:100])
        return TextResponse(text="")

    return TextResponse(text=stripped)


def _tools_to_text(tools: list[dict]) -> str:
    """Format tools as plain text for injection into system prompt."""
    lines = [
        "## Available Tools\n"
        "To use a tool, respond with EXACTLY this JSON in a ```json block:\n"
        '```json\n{"action":"<tool_name>","args":{<args>},"thought":"why"}\n```\n'
        'When done: ```json\n{"action":"final_answer","answer":"<answer>"}\n```\n\n'
        "### Tools:"
    ]
    for tg in tools:
        for fn in tg.get("function_declarations", []):
            props = fn.get("parameters", {}).get("properties", {})
            req   = fn.get("parameters", {}).get("required", [])
            args  = ", ".join(
                f"{k}{'*' if k in req else ''}: {v.get('description','')}"
                for k, v in props.items()
            )
            lines.append(f"- **{fn['name']}**({args}) — {fn['description']}")
    return "\n".join(lines)
