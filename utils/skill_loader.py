"""
utils/skill_loader.py — Nexara V1
Dynamic skill loader with AST Security Scanner + platform filtering.

Flow:
  1. Platform detected.
  2. Local skills loaded if they match platform tags.
  3. Remote manifest parsed.
  4. Missing/updated skills fetched.
  5. SHA256 checksum verified.
  6. AST Security Scan blocks RCE (exec, subprocess, socket, etc.).
  7. Safe skills injected into memory.
"""

import ast
import asyncio
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from utils.platform import PlatformContext

logger = logging.getLogger("nexara.skill_loader")

WAREHOUSE_BASE  = "https://raw.githubusercontent.com/NexaraAI/nexara-skills/main"
LOCAL_SKILL_DIR = Path(__file__).parent.parent / "skills"
CACHE_DIR       = Path.home() / ".nexara" / "skills_cache"


# ── AST Security Scanner ──────────────────────────────────────────────────────

class SecurityViolation(Exception):
    pass


class SkillSafetyScanner(ast.NodeVisitor):
    """
    Walks the AST of downloaded skills to block dangerous Python patterns
    before the code is ever loaded into the interpreter.
    """
    
    BANNED_IMPORTS = {"socket", "subprocess", "importlib", "ctypes", "pty", "builtins"}
    BANNED_CALLS   = {"exec", "eval", "compile", "__import__"}
    BANNED_ATTRS   = {"system", "popen", "spawn"}  # used with os.system etc.

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in self.BANNED_IMPORTS:
                raise SecurityViolation(f"Banned import detected: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in self.BANNED_IMPORTS:
                raise SecurityViolation(f"Banned from-import detected: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check direct calls like exec() or eval()
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BANNED_CALLS:
                raise SecurityViolation(f"Banned function call detected: {node.func.id}()")
        
        # Check attribute calls like os.system()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.BANNED_ATTRS:
                raise SecurityViolation(f"Banned method call detected: .{node.func.attr}()")
                
        self.generic_visit(node)


def _scan_code_safety(code_text: str, skill_name: str):
    """Parses code text into an AST and runs the safety scanner."""
    try:
        tree = ast.parse(code_text)
        scanner = SkillSafetyScanner()
        scanner.visit(tree)
    except SyntaxError as e:
        raise SecurityViolation(f"Syntax error in skill {skill_name}: {e}")


# ── Skill Loader ──────────────────────────────────────────────────────────────

class SkillLoader:

    def __init__(
        self,
        platform_ctx:  "PlatformContext",
        channel:       str = "stable",
        warehouse_url: str = "",
    ):
        self._ctx       = platform_ctx
        self._channel   = channel
        self._warehouse = warehouse_url or WAREHOUSE_BASE
        self._loaded:   list[str] = []

    def _matches(self, platforms: list[str]) -> bool:
        if "all" in platforms or "core" in platforms:
            return True
        return self._ctx.platform.value in platforms

    def load_local(self):
        """Phase 1: Load pre-packaged skills from local directory."""
        if not LOCAL_SKILL_DIR.exists():
            return
            
        for path in LOCAL_SKILL_DIR.rglob("*.py"):
            if path.name.startswith("__") or path.name == "base.py":
                continue
            self._import_file(path)

    async def fetch_remote(self, manifest: dict):
        """Phase 2: Fetch missing/updated skills securely from warehouse."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for name, meta in manifest.get("skills", {}).items():
            if not self._matches(meta.get("platforms", ["all"])):
                continue
            if name in self._loaded:
                continue

            cached   = CACHE_DIR / f"{name}.py"
            expected = meta.get("checksum", "")

            # Check local cache first
            if cached.exists() and self._sha256(cached) == expected:
                if self._safe_load(name, cached.read_text(encoding="utf-8"), cached):
                    continue

            # Fetch from warehouse
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(f"{self._warehouse}/{meta['file']}")
                    r.raise_for_status()

                code_text = r.text
                actual_hash = "sha256:" + hashlib.sha256(code_text.encode()).hexdigest()
                
                if expected and actual_hash != expected:
                    logger.warning("Checksum mismatch for '%s' — skipping", name)
                    continue

                if self._safe_load(name, code_text, cached):
                    # Only write to disk if the AST scan passed
                    cached.write_text(code_text, encoding="utf-8")
                    logger.info("Fetched and verified skill '%s' from warehouse", name)

            except Exception as exc:
                logger.warning("Failed to fetch '%s': %s", name, exc)

    def _safe_load(self, name: str, code_text: str, file_path: Path) -> bool:
        """Runs the AST scan before loading into memory."""
        try:
            _scan_code_safety(code_text, name)
            self._import_file(file_path, code_text=code_text)
            self._loaded.append(name)
            return True
        except SecurityViolation as exc:
            logger.error("🛡️ Security block on skill '%s': %s", name, exc)
            return False
        except Exception as exc:
            logger.error("Failed to load skill '%s': %s", name, exc)
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def _import_file(self, path: Path, code_text: str | None = None):
        module_name = f"nexara.dynamic_skills.{path.stem}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            if code_text:
                # Execute from memory buffer if we have it (remote fetch)
                exec(compile(code_text, str(path), 'exec'), module.__dict__)
            else:
                # Standard file load (local phase)
                spec.loader.exec_module(module)
                
            return module
        raise ImportError(f"Could not load {path}")
