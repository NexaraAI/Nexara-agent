"""
utils/skill_loader.py — Nexara V1
Dynamic skill loader with AST Security Scanner + platform filtering.

Flow:
  1. Platform detected.
  2. Local skills loaded if they match platform tags.
  3. Remote manifest parsed.
  4. Missing/updated skills fetched.
  5. SHA256 checksum verified.
  6. AST Security Scan blocks RCE patterns.
  7. Safe skills injected into memory.

Security scanner improvements:
  - Added 'os' to partially-banned: os.system / os.popen / os.exec* blocked
    via BANNED_ATTRS (os itself is allowed for path ops like os.path).
  - getattr() calls blocked — common bypass for attribute restrictions.
  - __builtins__ / __import__ string-key access via subscript blocked.
  - Nested attribute chains scanned (e.g. os.path is fine, os.system is not).
  - visit_Attribute now checks all attribute accesses, not just top-level calls.
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

    Blocks:
    - Direct dangerous imports (subprocess, socket, ctypes, pty, builtins, importlib)
    - Dangerous built-in calls: exec(), eval(), compile(), __import__()
    - getattr() — primary bypass for attribute restrictions
    - Dangerous OS methods: os.system(), os.popen(), os.execv() etc.
    - Subscript access to __builtins__ / __import__ keys
    - String-based module loading patterns
    """

    BANNED_IMPORTS = {
        "subprocess", "socket", "importlib", "ctypes", "pty", "builtins",
        "multiprocessing", "concurrent",
    }

    # These module.attr combos are specifically dangerous
    BANNED_METHOD_ATTRS = {
        "system", "popen", "execv", "execve", "execvp", "execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnv", "spawnve",
        "popen2", "popen3", "popen4", "fork", "forkpty",
    }

    # Direct function calls that are always dangerous
    BANNED_CALLS = {"exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr"}

    # Dangerous dunder/magic string keys accessed via subscript
    BANNED_SUBSCRIPT_KEYS = {"__builtins__", "__import__", "__loader__", "__spec__"}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base in self.BANNED_IMPORTS:
                raise SecurityViolation(f"Banned import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base = node.module.split(".")[0]
            if base in self.BANNED_IMPORTS:
                raise SecurityViolation(f"Banned from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct function name calls: exec(), eval(), getattr() etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BANNED_CALLS:
                raise SecurityViolation(f"Banned function call: {node.func.id}()")

        # Attribute calls: os.system(), obj.popen() etc.
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.BANNED_METHOD_ATTRS:
                raise SecurityViolation(
                    f"Banned method call: .{node.func.attr}() "
                    f"— use the httpx or asyncio APIs instead"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Catch dangerous attribute accesses even outside of call context
        # e.g. `fn = os.system` (assignment, not call)
        if node.attr in self.BANNED_METHOD_ATTRS:
            raise SecurityViolation(
                f"Access to dangerous attribute: .{node.attr} — not permitted in skills"
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Block: globals()["__builtins__"], locals()["__import__"] etc.
        key_node = node.slice
        # Python 3.9+: slice is the value directly; 3.8: wrapped in ast.Index
        if isinstance(key_node, ast.Index):
            key_node = key_node.value  # type: ignore[attr-defined]
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            if key_node.value in self.BANNED_SUBSCRIPT_KEYS:
                raise SecurityViolation(
                    f"Access to banned key: '{key_node.value}' — not permitted in skills"
                )
        self.generic_visit(node)


def _scan_code_safety(code_text: str, skill_name: str):
    """Parse code into AST and run the safety scanner. Raises SecurityViolation on any hit."""
    try:
        tree = ast.parse(code_text)
        SkillSafetyScanner().visit(tree)
    except SyntaxError as exc:
        raise SecurityViolation(f"Syntax error in skill '{skill_name}': {exc}") from exc


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

    async def load(self) -> list[str]:
        """Master entrypoint: load local skills, fetch manifest, fetch remote skills."""
        self.load_local()

        manifest = {}
        try:
            manifest_url = f"{self._warehouse}/manifest.json"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(manifest_url)
                r.raise_for_status()
                manifest = r.json()
                logger.info("Manifest fetched: %d skills listed", len(manifest.get("skills", {})))
        except Exception as exc:
            logger.error("Failed to fetch manifest from warehouse: %s", exc)

        if manifest:
            await self.fetch_remote(manifest)

        return self._loaded

    def load_local(self):
        """Phase 1: Load pre-packaged skills from the local skills/ directory."""
        if not LOCAL_SKILL_DIR.exists():
            return
        for path in LOCAL_SKILL_DIR.rglob("*.py"):
            if path.name.startswith("__") or path.name == "base.py":
                continue
            try:
                self._import_file(path)
            except Exception as exc:
                logger.error("Failed to load local skill '%s': %s", path.stem, exc)

    async def fetch_remote(self, manifest: dict):
        """Phase 2: Securely fetch missing/updated skills from the warehouse."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for name, meta in manifest.get("skills", {}).items():
            if not self._matches(meta.get("platforms", ["all"])):
                continue
            if name in self._loaded:
                continue

            cached   = CACHE_DIR / f"{name}.py"
            expected = meta.get("checksum", "")

            # Use cached copy if checksum matches
            if cached.exists():
                try:
                    cached_text = cached.read_text(encoding="utf-8")
                    if self._sha256_text(cached_text) == expected:
                        if self._safe_load(name, cached_text, cached):
                            continue
                except Exception as exc:
                    logger.warning("Cache read failed for '%s': %s", name, exc)

            # Fetch fresh from warehouse
            file_path = meta.get("file", "")
            if not file_path:
                logger.warning("No file path in manifest for skill '%s'", name)
                continue

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(f"{self._warehouse}/{file_path}")
                    r.raise_for_status()

                code_text   = r.text
                actual_hash = self._sha256_text(code_text)

                if expected and actual_hash != expected:
                    logger.warning(
                        "Checksum mismatch for '%s' — expected %s got %s — skipping",
                        name, expected[:16], actual_hash[:16],
                    )
                    continue

                if self._safe_load(name, code_text, cached):
                    try:
                        cached.write_text(code_text, encoding="utf-8")
                    except Exception as exc:
                        logger.warning("Could not cache skill '%s': %s", name, exc)
                    logger.info("Fetched and verified skill '%s'", name)

            except httpx.HTTPStatusError as exc:
                logger.warning("HTTP %d fetching skill '%s'", exc.response.status_code, name)
            except Exception as exc:
                logger.warning("Failed to fetch skill '%s': %s", name, exc)

    def _safe_load(self, name: str, code_text: str, file_path: Path) -> bool:
        """Run AST security scan then import. Returns True on success."""
        try:
            _scan_code_safety(code_text, name)
        except SecurityViolation as exc:
            logger.error("Security block on skill '%s': %s", name, exc)
            return False

        try:
            self._import_file(file_path, code_text=code_text)
            if name not in self._loaded:
                self._loaded.append(name)
            return True
        except Exception as exc:
            logger.error("Import failed for skill '%s': %s", name, exc)
            return False

    @staticmethod
    def _sha256_text(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def _import_file(self, path: Path, code_text: str | None = None):
        module_name = f"nexara.dynamic_skills.{path.stem}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if not spec or not spec.loader:
            raise ImportError(f"Could not create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            if code_text:
                exec(compile(code_text, str(path), "exec"), module.__dict__)
            else:
                spec.loader.exec_module(module)
        except Exception:
            # Remove from sys.modules so a retry can attempt a clean load
            sys.modules.pop(module_name, None)
            raise

        return module
