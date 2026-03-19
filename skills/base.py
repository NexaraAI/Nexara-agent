"""
skills/base.py — Nexara V1
Base class and metaclass for all Nexara skills.

V1 additions:
  • BaseSkill.platforms — list of platform tags this skill supports
    "all"     → works everywhere
    "android" → Termux/Android only
    "linux"   → Linux, Codespace, WSL, Docker
    "macos"   → macOS only
    "windows" → Windows only
    "core"    → alias for "all"
  • SkillMeta inherits ABCMeta (fixes metaclass conflict with ABC)
"""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Standardised return value for every skill."""
    success: bool
    output:  str
    data:    dict[str, Any] = field(default_factory=dict)
    error:   str = ""

    def __str__(self) -> str:
        return self.output if self.success else f"[Error] {self.error}"


class SkillMeta(ABCMeta):
    """
    Metaclass that auto-registers concrete skill classes into a global registry.
    Inherits ABCMeta so it stays compatible with ABC.
    """
    _registry: dict[str, type] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if bases and not getattr(cls, "__abstractmethods__", None):
            skill_name = getattr(cls, "name", name.lower())
            if skill_name:
                mcs._registry[skill_name] = cls
        return cls

    @classmethod
    def get_registry(mcs) -> dict[str, type]:
        return dict(mcs._registry)


class BaseSkill(ABC, metaclass=SkillMeta):
    """
    Abstract base all skills must extend.

    Required class attributes:
        name        : str        — short identifier used by the router
        description : str        — shown to the LLM in the system prompt
        platforms   : list[str]  — which platforms this skill runs on
                                   default "all" = everywhere

    Must implement:
        execute(self, **kwargs) -> SkillResult
    """

    name:        str       = ""
    description: str       = ""
    platforms:   list[str] = ["all"]   # override in subclass if platform-specific

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        ...

    def __repr__(self) -> str:
        return f"<Skill:{self.name} platforms={self.platforms}>"
