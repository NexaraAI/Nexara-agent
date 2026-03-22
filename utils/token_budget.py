"""
utils/token_budget.py — Nexara V1
Dynamic token budget manager.

Now classifier-aware: when the skill classifier reduces the tool set,
the tools reserve shrinks proportionally, freeing tokens for history
and memory. Gemini 2.5 Flash has 1M TPM so we raise the budget ceiling.

Also provides per-provider budget caps so the same code works whether
someone is on Groq (6k TPM) or Gemini (1M TPM).
"""

import re
from dataclasses import dataclass

# ── Token estimation ──────────────────────────────────────────────────────────
def est(text: str) -> int:
    return max(1, len(text) // 4)

# ── Per-provider soft limits ──────────────────────────────────────────────────
PROVIDER_LIMITS: dict[str, int] = {
    "groq":   6_000,
    "gemini": 1_000_000,
    "nvidia": 4_000,
    "ollama": 999_999,
}
SOFT_LIMIT_PCT = 0.80

# ── Budget constants ──────────────────────────────────────────────────────────
# Conservative base budget — works on ALL providers including Groq 6k TPM.
# With classifier: tools shrink from ~2000 to ~150 tokens, freeing ~1850
# extra tokens for history and memory.
MAX_BUDGET          = 4_000
SYSTEM_RESERVE      = 800    # system prompt + identity + behaviour rules
TOOLS_RESERVE_FULL  = 2_000  # all 78 skills as text
TOOLS_RESERVE_SMALL = 150    # 4-8 classified skills
USER_RESERVE        = 300
ALWAYS_KEEP_TURNS   = 6

_COMPLEX_KEYWORDS = {
    "research", "find", "compare", "history", "explain", "summarize",
    "analyse", "analyze", "difference", "versus", "vs", "why", "how does",
    "what is", "tell me about", "review", "investigate", "search",
    "download", "create", "build", "write", "run", "execute",
    "schedule", "remind", "monitor", "check", "generate", "make",
}


@dataclass
class BudgetResult:
    trimmed_history: list[dict]
    memory_slots:    int
    estimated_total: int
    tools_tokens:    int   # actual tools token cost used for this call


def apply(
    message:          str,
    history:          list[dict],
    system_prompt:    str,
    tools_count:      int = 78,   # number of tools being passed this call
) -> BudgetResult:
    """
    Calculate trimmed history and memory slots given the tool count.

    tools_count = number of skills the classifier selected.
    78 = no classifier (full tool set)
    4-8 = classifier active (much smaller)
    """
    # Tools cost estimate — each tool description ≈ 25 tokens avg
    tools_tokens = min(tools_count * 25, TOOLS_RESERVE_FULL)
    if tools_count <= 10:
        tools_tokens = TOOLS_RESERVE_SMALL  # flat small cost for filtered set

    sp_cost   = min(est(system_prompt), SYSTEM_RESERVE)
    msg_cost  = min(est(message), USER_RESERVE)
    remaining = MAX_BUDGET - sp_cost - msg_cost - tools_tokens

    trimmed      = _trim_history(history, remaining)
    history_cost = sum(_turn_cost(t) for t in trimmed)
    mem_budget   = remaining - history_cost
    memory_slots = _decide_memory_slots(message, mem_budget)

    total_est = (
        sp_cost + msg_cost + tools_tokens
        + history_cost
        + (memory_slots * 80)
    )

    return BudgetResult(
        trimmed_history=trimmed,
        memory_slots=memory_slots,
        estimated_total=total_est,
        tools_tokens=tools_tokens,
    )


def is_near_limit(provider: str, tokens_used_this_minute: int) -> bool:
    limit = PROVIDER_LIMITS.get(provider, 4_000)
    return tokens_used_this_minute >= int(limit * SOFT_LIMIT_PCT)


# ── Internals ─────────────────────────────────────────────────────────────────

def _trim_history(history: list[dict], budget: int) -> list[dict]:
    if not history:
        return []
    always_keep = history[-ALWAYS_KEEP_TURNS:]
    older       = history[:-ALWAYS_KEEP_TURNS]
    keep_cost   = sum(_turn_cost(t) for t in always_keep)
    remaining   = budget - keep_cost
    kept_older  = []
    for turn in reversed(older):
        cost = _turn_cost(turn)
        if cost <= remaining:
            kept_older.insert(0, turn)
            remaining -= cost
    return kept_older + always_keep


def _turn_cost(turn: dict) -> int:
    parts   = turn.get("parts", [])
    content = turn.get("content", "")
    text    = " ".join(parts) if isinstance(parts, list) else content
    return est(text)


def _decide_memory_slots(message: str, mem_budget: int) -> int:
    if len(message) <= 20:
        return 0
    lower      = message.lower()
    is_complex = (
        any(kw in lower for kw in _COMPLEX_KEYWORDS)
        or len(message) > 60
        or "?" in message
    )
    if not is_complex:
        return 0
    if mem_budget > 1200:
        return 8   # classifier freed up lots of tokens — inject more memories
    if mem_budget > 800:
        return 5
    if mem_budget > 400:
        return 2
    return 0
