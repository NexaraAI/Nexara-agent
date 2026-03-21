"""
agent/memory.py — Nexara V1
Persistent memory with semantic vector search.

Storage  : SQLite (facts, tasks, downloads, conversations)
Search   : sentence-transformers embeddings -> cosine similarity
           (falls back to FTS on overlayfs / Codespaces / Docker)
Injection: Top-N most relevant memories for current query only.

SIGBUS root cause & fix:
  PyTorch lazily mmap-loads model weights. On overlayfs (Codespaces, Docker
  overlay2) the kernel cannot back those mmap pages and raises SIGBUS on first
  access — usually inside _memory.remember() after a skill run completes.
  SIGBUS is signal 7 — it terminates the process, it cannot be caught by
  try/except.

  Fix: _overlayfs_environment() uses THREE independent detection methods:
    1. Env vars  — CODESPACES, GITHUB_CODESPACE_TOKEN (fastest)
    2. /.dockerenv — Docker overlay2
    3. /proc/mounts — looks for 'overlay' filesystem type (catches edge cases
       where env vars are absent but overlayfs is still in use)
  If ANY method fires, sentence-transformers is never imported and the FTS
  keyword fallback is used instead. No mmap, no SIGBUS.
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("nexara.memory")

DB_PATH   = Path.home() / ".nexara" / "memory.db"
EMBED_DIM = 384

# Prevent tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Hint to PyTorch not to use CUDA memory caching (extra safety on VMs)
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")


# ── Overlayfs detection — three independent methods ───────────────────────────

def _overlayfs_environment() -> bool:
    """
    Returns True if the process is running on a filesystem where PyTorch
    mmap of weight files would cause SIGBUS.

    Method 1 — environment variables (fastest, set by Codespaces/CI):
    """
    if (os.getenv("CODESPACES") == "true"
            or os.getenv("GITHUB_CODESPACE_TOKEN")
            or os.getenv("GITHUB_ACTIONS") == "true"):
        return True

    # Method 2 — Docker sentinel file
    if Path("/.dockerenv").exists():
        return True

    # Method 3 — /proc/mounts overlay filesystem type
    # This catches cases where env vars are not set but overlayfs is active,
    # e.g. nested containers, custom Codespace images, or Gitpod workspaces.
    try:
        mounts = Path("/proc/mounts").read_text()
        for line in mounts.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("overlay", "overlayfs"):
                return True
    except Exception:
        pass

    # Method 4 — WSL has its own mmap quirks
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return True
    except Exception:
        pass

    return False


_EMBEDDINGS_DISABLED = _overlayfs_environment()
if _EMBEDDINGS_DISABLED:
    logger.info(
        "sentence-transformers DISABLED — overlayfs/Codespaces/Docker detected. "
        "Using FTS keyword fallback (stable, no mmap, no SIGBUS)."
    )


# ── Embedding engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers with platform-aware safe fallback.

    On overlayfs platforms: _load() returns False immediately — PyTorch is
    never imported, no mmap, no SIGBUS possible.

    On safe platforms (native Linux, macOS, Windows):
      - asyncio.Lock prevents concurrent coroutine encode() calls
      - threading.Lock prevents any residual thread-pool races
    """
    _model:    Any            = None
    _available: bool | None   = None
    _aio_lock: asyncio.Lock   = asyncio.Lock()
    _thr_lock: threading.Lock = threading.Lock()

    @classmethod
    def _load(cls) -> bool:
        if _EMBEDDINGS_DISABLED:
            cls._available = False
            return False

        if cls._available is not None:
            return cls._available

        try:
            from sentence_transformers import SentenceTransformer
            cls._model     = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",   # explicit cpu avoids CUDA mmap paths
            )
            cls._available = True
            logger.info("sentence-transformers loaded (semantic search enabled)")
        except Exception as exc:
            logger.warning("sentence-transformers unavailable: %s — FTS fallback", exc)
            cls._available = False
        return cls._available

    @classmethod
    async def embed(cls, text: str) -> np.ndarray | None:
        if not await asyncio.to_thread(cls._load):
            return None
        try:
            async with cls._aio_lock:
                def _enc():
                    with cls._thr_lock:
                        return cls._model.encode(text, normalize_embeddings=True)
                vec = await asyncio.to_thread(_enc)
            return vec.astype(np.float32)
        except Exception as exc:
            logger.error("embed() failed: %s — falling back to FTS", exc)
            # Disable permanently so we don't keep failing
            cls._available = False
            return None

    @classmethod
    async def embed_batch(cls, texts: list[str]) -> list[np.ndarray] | None:
        if not await asyncio.to_thread(cls._load):
            return None
        try:
            async with cls._aio_lock:
                def _enc():
                    with cls._thr_lock:
                        return cls._model.encode(
                            texts, normalize_embeddings=True, show_progress_bar=False
                        )
                vecs = await asyncio.to_thread(_enc)
            return [v.astype(np.float32) for v in vecs]
        except Exception as exc:
            logger.error("embed_batch() failed: %s — falling back to FTS", exc)
            cls._available = False
            return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    id:         int
    kind:       str
    content:    str
    tags:       list[str]
    metadata:   dict
    timestamp:  float
    importance: int
    score:      float = 1.0


# ── AgentMemory ───────────────────────────────────────────────────────────────

class AgentMemory:

    def __init__(self, db_path: Path = DB_PATH):
        self._db = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    tags        TEXT    DEFAULT '[]',
                    metadata    TEXT    DEFAULT '{}',
                    timestamp   REAL    NOT NULL,
                    importance  INTEGER DEFAULT 3,
                    embedding   BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_mem_ts   ON memories(timestamp DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, content='memories', content_rowid='id');

                CREATE TABLE IF NOT EXISTS tasks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal         TEXT    NOT NULL,
                    steps        TEXT    DEFAULT '[]',
                    status       TEXT    DEFAULT 'pending',
                    result       TEXT    DEFAULT '',
                    created_at   REAL    NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS downloads (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    url          TEXT    NOT NULL,
                    filename     TEXT,
                    save_path    TEXT,
                    status       TEXT    DEFAULT 'queued',
                    size_bytes   INTEGER DEFAULT 0,
                    started_at   REAL,
                    completed_at REAL,
                    error        TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   INTEGER NOT NULL,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, timestamp DESC);
            """)

    # ── Store ─────────────────────────────────────────────────────────────────

    async def remember(
        self,
        content:    str,
        kind:       str = "fact",
        tags:       list[str] | None = None,
        metadata:   dict | None = None,
        importance: int = 3,
    ) -> int:
        tags     = tags or []
        metadata = metadata or {}
        now      = time.time()
        vec      = await EmbeddingEngine.embed(content)
        blob     = vec.tobytes() if vec is not None else None

        def _w():
            with sqlite3.connect(self._db) as conn:
                cur = conn.execute(
                    "INSERT INTO memories (kind,content,tags,metadata,timestamp,importance,embedding) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (kind, content, json.dumps(tags), json.dumps(metadata), now, importance, blob),
                )
                rowid = cur.lastrowid
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content) VALUES (?,?)",
                    (rowid, content),
                )
                return rowid

        row_id = await asyncio.to_thread(_w)
        logger.debug("Memory #%d [%s]: %s", row_id, kind, content[:60])
        return row_id

    # ── Retrieve ──────────────────────────────────────────────────────────────

    async def recall(
        self,
        query:          str = "",
        kind:           str | None = None,
        limit:          int = 8,
        min_importance: int = 1,
    ) -> list[MemoryEntry]:
        def _load():
            with sqlite3.connect(self._db) as conn:
                conn.row_factory = sqlite3.Row
                clauses: list[str] = ["importance >= ?"]
                params:  list[Any] = [min_importance]
                if kind:
                    clauses.append("kind = ?")
                    params.append(kind)
                return conn.execute(
                    f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                    f"ORDER BY timestamp DESC LIMIT 500",
                    params,
                ).fetchall()

        rows = await asyncio.to_thread(_load)
        if not rows:
            return []

        entries = [
            MemoryEntry(
                id=r["id"], kind=r["kind"], content=r["content"],
                tags=json.loads(r["tags"]), metadata=json.loads(r["metadata"]),
                timestamp=r["timestamp"], importance=r["importance"],
                score=float(r["importance"]) / 5.0,
            )
            for r in rows
        ]

        if query:
            q_vec = await EmbeddingEngine.embed(query)
            if q_vec is not None:
                for e, r in zip(entries, rows):
                    blob = r["embedding"]
                    if blob:
                        try:
                            e_vec   = np.frombuffer(blob, dtype=np.float32).copy()
                            e.score = _cosine(q_vec, e_vec) * (e.importance / 5.0)
                        except Exception:
                            pass
                entries = sorted(entries, key=lambda x: x.score, reverse=True)
            else:
                def _fts():
                    with sqlite3.connect(self._db) as conn:
                        conn.row_factory = sqlite3.Row
                        safe_q = " ".join(
                            w for w in query.split() if w.isalnum() or len(w) > 2
                        ) or query
                        try:
                            return conn.execute(
                                "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                                (safe_q, limit * 2),
                            ).fetchall()
                        except Exception:
                            return []
                fts_rows = await asyncio.to_thread(_fts)
                fts_ids  = {r["rowid"] for r in fts_rows}
                entries  = [e for e in entries if e.id in fts_ids] or entries

        return entries[:limit]

    async def relevant_context(self, query: str, limit: int = 5) -> str:
        entries = await self.recall(query=query, limit=limit, min_importance=2)
        if not entries:
            return ""
        from datetime import datetime
        lines = ["Relevant memories:\n"]
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%b %d")
            lines.append(f"- [{e.kind.upper()}] {e.content}  ({ts})")
        return "\n".join(lines)

    # ── Conversation persistence ───────────────────────────────────────────────

    async def save_turn(self, user_id: int, role: str, content: str):
        now = time.time()
        def _w():
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "INSERT INTO conversations (user_id,role,content,timestamp) VALUES (?,?,?,?)",
                    (user_id, role, content, now),
                )
        await asyncio.to_thread(_w)

    async def load_history(self, user_id: int, limit: int = 40) -> list[dict]:
        def _r():
            with sqlite3.connect(self._db) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT role, content FROM conversations "
                    "WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
                return list(reversed(rows))
        rows = await asyncio.to_thread(_r)
        return [{"role": r["role"], "parts": [r["content"]]} for r in rows]

    async def clear_history(self, user_id: int):
        def _w():
            with sqlite3.connect(self._db) as conn:
                conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
        await asyncio.to_thread(_w)

    # ── Task log ──────────────────────────────────────────────────────────────

    async def log_task(self, goal: str, steps: list[str]) -> int:
        now = time.time()
        def _w():
            with sqlite3.connect(self._db) as conn:
                return conn.execute(
                    "INSERT INTO tasks (goal,steps,status,created_at) VALUES (?,?,?,?)",
                    (goal, json.dumps(steps), "running", now),
                ).lastrowid
        return await asyncio.to_thread(_w)

    async def complete_task(self, task_id: int, result: str, success: bool = True):
        now = time.time()
        def _w():
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "UPDATE tasks SET status=?,result=?,completed_at=? WHERE id=?",
                    ("done" if success else "failed", result, now, task_id),
                )
        await asyncio.to_thread(_w)

    # ── Download log ──────────────────────────────────────────────────────────

    async def log_download(self, url: str, filename: str, save_path: str) -> int:
        now = time.time()
        def _w():
            with sqlite3.connect(self._db) as conn:
                return conn.execute(
                    "INSERT INTO downloads (url,filename,save_path,status,started_at) VALUES (?,?,?,?,?)",
                    (url, filename, save_path, "running", now),
                ).lastrowid
        return await asyncio.to_thread(_w)

    async def update_download(self, dl_id: int, status: str, size_bytes: int = 0, error: str = ""):
        now = time.time()
        def _w():
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "UPDATE downloads SET status=?,size_bytes=?,error=?,completed_at=? WHERE id=?",
                    (status, size_bytes, error, now, dl_id),
                )
        await asyncio.to_thread(_w)

    async def download_history(self, limit: int = 10) -> str:
        def _r():
            with sqlite3.connect(self._db) as conn:
                conn.row_factory = sqlite3.Row
                return conn.execute(
                    "SELECT * FROM downloads ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
        rows = await asyncio.to_thread(_r)
        if not rows:
            return "No downloads recorded."
        icon_map = {"done": "✅", "running": "⏳", "failed": "❌", "queued": "🕐"}
        lines = ["📥 Download History\n"]
        for r in rows:
            sz  = f"{r['size_bytes']/1_048_576:.1f} MB" if r["size_bytes"] else "?"
            ico = icon_map.get(r["status"], "•")
            lines.append(f"{ico} {r['filename'] or r['url'][:40]} — {r['status']} ({sz})")
        return "\n".join(lines)

    # ── Memory skill wrappers ─────────────────────────────────────────────────

    async def skill_remember(self, content: str, kind: str = "fact", importance: int = 3, **_) -> str:
        mid = await self.remember(content, kind=kind, importance=importance)
        return f"Stored memory #{mid}"

    async def skill_recall(self, query: str, kind: str | None = None, **_) -> str:
        entries = await self.recall(query=query, kind=kind, limit=6)
        if not entries:
            return "No relevant memories found."
        mode  = "FTS" if _EMBEDDINGS_DISABLED else "semantic"
        lines = [f"Memory Recall ({mode}) for '{query}'\n"]
        for e in entries:
            lines.append(f"- [{e.kind}] {e.content}  (score: {e.score:.2f})")
        return "\n".join(lines)
