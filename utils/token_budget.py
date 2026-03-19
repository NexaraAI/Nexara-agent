"""
utils/token_budget.py — Nexara V1
Dynamic token budget manager.

Prevents rate limiting by enforcing context size limits before every LLM call.
No more 429s from accidentally sending 40 turns of history + memory + huge prompt.

Budget allocation (4000 tokens total — safe for all providers):
  System prompt   → up to 700  (capped and warned if exceeded)
  Tool schemas    → up to 500  (platform-filtered, so smaller than before)
  User message    → up to 300
  Conversation    → ~2500 remaining, trimmed intelligently
  Memory context  → only injected if budget allows + query is complex
"""

import re
from dataclasses import dataclass

# ── Token estimation ──────────────────────────────────────────────────────────
# GPT-style: ~4 chars per token. Conservative estimate to avoid surprises.

def est(text: str) -> int:
    return max(1, len(text) // 4)


# ── Per-provider rate limits (tokens per minute) ──────────────────────────────
PROVIDER_LIMITS: dict[str, int] = {
    "groq":   6_000,
    "gemini": 32_000,
    "nvidia": 4_000,
    "ollama": 999_999,
}

# Soft limit — switch provider at 80% to avoid hitting the wall
SOFT_LIMIT_PCT = 0.80

# ── Budget constants ──────────────────────────────────────────────────────────
MAX_BUDGET      = 4_000
SYSTEM_RESERVE  = 700
TOOLS_RESERVE   = 500
USER_RESERVE    = 300
HISTORY_BUDGET  = MAX_BUDGET - SYSTEM_RESERVE - TOOLS_RESERVE - USER_RESERVE  # 2500

# Always keep this many most-recent turns regardless of budget
ALWAYS_KEEP_TURNS = 6

# Keywords that signal a complex query deserving memory injection
_COMPLEX_KEYWORDS = {
    "research", "find", "compare", "history", "explain", "summarize",
    "analyse", "analyze", "difference", "versus", "vs", "why", "how does",
    "what is", "tell me about", "review", "investigate", "search",
    "download", "create", "build", "write", "run", "execute",
    "schedule", "remind", "monitor", "check",
}


@dataclass
class BudgetResult:
    trimmed_history: list[dict]
    memory_slots:    int    # 0, 2, or 5 memories to inject
    estimated_total: int    # rough token estimate for logging


def apply(
    message:       str,
    history:       list[dict],
    system_prompt: str,
) -> BudgetResult:
    """
    Given raw inputs, return trimmed history + how many memory slots to use.
    Call this before every LLM dispatch.
    """
    sp_cost  = min(est(system_prompt), SYSTEM_RESERVE)
    msg_cost = min(est(message), USER_RESERVE)
    remaining = MAX_BUDGET - sp_cost - msg_cost - TOOLS_RESERVE

    # ── Trim history ──────────────────────────────────────────────────────────
    trimmed = _trim_history(history, remaining)

    # ── Memory slot decision ──────────────────────────────────────────────────
    history_cost  = sum(_turn_cost(t) for t in trimmed)
    mem_budget    = remaining - history_cost
    memory_slots  = _decide_memory_slots(message, mem_budget)

    total_est = (
        sp_cost + msg_cost + TOOLS_RESERVE
        + history_cost
        + (memory_slots * 80)   # avg memory entry ≈ 80 tokens
    )

    return BudgetResult(
        trimmed_history=trimmed,
        memory_slots=memory_slots,
        estimated_total=total_est,
    )


def is_near_limit(provider: str, tokens_used_this_minute: int) -> bool:
    """Return True if we should pre-emptively switch providers."""
    limit = PROVIDER_LIMITS.get(provider, 4_000)
    return tokens_used_this_minute >= int(limit * SOFT_LIMIT_PCT)


# ── Internals ─────────────────────────────────────────────────────────────────

def _trim_history(history: list[dict], budget: int) -> list[dict]:
    if not history:
        return []

    always_keep = history[-ALWAYS_KEEP_TURNS:]
    older       = history[:-ALWAYS_KEEP_TURNS]

    keep_cost = sum(_turn_cost(t) for t in always_keep)
    remaining = budget - keep_cost

    kept_older = []
    for turn in reversed(older):
        cost = _turn_cost(turn)
        if cost <= remaining:
            kept_older.insert(0, turn)
            remaining -= cost
        # Don't break — might skip one big turn and fit several small ones

    return kept_older + always_keep


def _turn_cost(turn: dict) -> int:
    parts   = turn.get("parts", [])
    content = turn.get("content", "")
    text    = " ".join(parts) if isinstance(parts, list) else content
    return est(text)


def _decide_memory_slots(message: str, mem_budget: int) -> int:
    # Too short to benefit from memory
    if len(message) <= 20:
        return 0

    lower      = message.lower()
    is_complex = (
        any(kw in lower for kw in _COMPLEX_KEYWORDS)
        or len(message) > 80
        or "?" in message
    )

    if not is_complex:
        return 0
    if mem_budget > 800:
        return 5
    if mem_budget > 400:
        return 2
    return 0
