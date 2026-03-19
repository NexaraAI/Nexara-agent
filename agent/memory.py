"""
agent/memory.py  — V1
Persistent memory with semantic vector search.

Storage  : SQLite (facts, tasks, downloads, conversations)
Search   : sentence-transformers embeddings → cosine similarity
           (falls back to FTS if transformers not installed)
Injection: Top-N most relevant memories for current query only.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("nexara.memory")

DB_PATH    = Path.home() / ".nexara" / "memory.db"
EMBED_DIM  = 384   # all-MiniLM-L6-v2 dimension


# ── Embedding engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers. Falls back gracefully if not installed.
    Runs all heavy work in a thread pool so it never blocks the event loop.
    """
    _model = None
    _available = None

    @classmethod
    def _load(cls):
        if cls._available is not None:
            return cls._available
        try:
            from sentence_transformers import SentenceTransformer
            cls._model     = SentenceTransformer("all-MiniLM-L6-v2")
            cls._available = True
            logger.info("Sentence-transformers loaded (semantic search enabled)")
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s) — using FTS fallback", exc)
            cls._available = False
        return cls._available

    @classmethod
    async def embed(cls, text: str) -> np.ndarray | None:
        if not await asyncio.to_thread(cls._load):
            return None
        vec = await asyncio.to_thread(cls._model.encode, text, normalize_embeddings=True)
        return vec.astype(np.float32)

    @classmethod
    async def embed_batch(cls, texts: list[str]) -> list[np.ndarray] | None:
        if not await asyncio.to_thread(cls._load):
            return None
        vecs = await asyncio.to_thread(
            cls._model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.astype(np.float32) for v in vecs]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))          # both already L2-normalised


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    id: int
    kind: str
    content: str
    tags: list[str]
    metadata: dict
    timestamp: float
    importance: int
    score: float = 1.0   # relevance score when returned from recall()


# ── AgentMemory ───────────────────────────────────────────────────────────────

class AgentMemory:

    def __init__(self, db_path: Path = DB_PATH):
        self._db = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

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
        content: str,
        kind: str = "fact",
        tags: list[str] | None = None,
        metadata: dict | None = None,
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
                # Update FTS index
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
        query: str = "",
        kind: str | None = None,
        limit: int = 8,
        min_importance: int = 1,
    ) -> list[MemoryEntry]:
        """
        Semantic recall: embed query, score all stored embeddings,
        return top-N by cosine similarity.
        Falls back to FTS keyword search if embeddings unavailable.
        """
        # ── Load all candidate rows ───────────────────────────────────────────
        def _load():
            with sqlite3.connect(self._db) as conn:
                conn.row_factory = sqlite3.Row
                clauses = ["importance >= ?"]
                params: list[Any] = [min_importance]
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

        # ── Semantic scoring ──────────────────────────────────────────────────
        if query:
            q_vec = await EmbeddingEngine.embed(query)
            if q_vec is not None:
                scored = []
                for e, r in zip(entries, rows):
                    blob = r["embedding"]
                    if blob:
                        try:
                            e_vec  = np.frombuffer(blob, dtype=np.float32)
                            e.score = _cosine(q_vec, e_vec) * (e.importance / 5.0)
                        except Exception:
                            pass
                    scored.append(e)
                entries = sorted(scored, key=lambda x: x.score, reverse=True)
            else:
                # FTS fallback
                def _fts():
                    with sqlite3.connect(self._db) as conn:
                        conn.row_factory = sqlite3.Row
                        return conn.execute(
                            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                            (query, limit * 2),
                        ).fetchall()
                fts_rows = await asyncio.to_thread(_fts)
                fts_ids  = {r["rowid"] for r in fts_rows}
                entries  = [e for e in entries if e.id in fts_ids] or entries

        return entries[:limit]

    # ── Relevance-scored memory injection ─────────────────────────────────────

    async def relevant_context(self, query: str, limit: int = 5) -> str:
        """
        Returns only the top-N memories most relevant to `query`.
        Used for LLM system prompt injection.
        """
        entries = await self.recall(query=query, limit=limit, min_importance=2)
        if not entries:
            return ""
        from datetime import datetime
        lines = ["📚 Relevant memories:\n"]
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%b %d")
            lines.append(f"• [{e.kind.upper()}] {e.content}  ({ts})")
        return "\n".join(lines)

    # ── Conversation persistence ───────────────────────────────────────────────

    async def save_turn(self, user_id: int, role: str, content: str):
        await asyncio.to_thread(
            lambda: sqlite3.connect(self._db).execute(
                "INSERT INTO conversations (user_id,role,content,timestamp) VALUES (?,?,?,?)",
                (user_id, role, content, time.time()),
            ).connection.commit()
        )

    async def load_history(self, user_id: int, limit: int = 40) -> list[dict]:
        """Load the last `limit` turns for a user from SQLite."""
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
        await asyncio.to_thread(
            lambda: sqlite3.connect(self._db).execute(
                "DELETE FROM conversations WHERE user_id=?", (user_id,)
            ).connection.commit()
        )

    # ── Task log ──────────────────────────────────────────────────────────────

    async def log_task(self, goal: str, steps: list[str]) -> int:
        def _w():
            with sqlite3.connect(self._db) as conn:
                return conn.execute(
                    "INSERT INTO tasks (goal,steps,status,created_at) VALUES (?,?,?,?)",
                    (goal, json.dumps(steps), "running", time.time()),
                ).lastrowid
        return await asyncio.to_thread(_w)

    async def complete_task(self, task_id: int, result: str, success: bool = True):
        await asyncio.to_thread(
            lambda: sqlite3.connect(self._db).execute(
                "UPDATE tasks SET status=?,result=?,completed_at=? WHERE id=?",
                ("done" if success else "failed", result, time.time(), task_id),
            ).connection.commit()
        )

    # ── Download log ──────────────────────────────────────────────────────────

    async def log_download(self, url: str, filename: str, save_path: str) -> int:
        def _w():
            with sqlite3.connect(self._db) as conn:
                return conn.execute(
                    "INSERT INTO downloads (url,filename,save_path,status,started_at) VALUES (?,?,?,?,?)",
                    (url, filename, save_path, "running", time.time()),
                ).lastrowid
        return await asyncio.to_thread(_w)

    async def update_download(self, dl_id: int, status: str, size_bytes: int = 0, error: str = ""):
        await asyncio.to_thread(
            lambda: sqlite3.connect(self._db).execute(
                "UPDATE downloads SET status=?,size_bytes=?,error=?,completed_at=? WHERE id=?",
                (status, size_bytes, error, time.time(), dl_id),
            ).connection.commit()
        )

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
        lines = ["📥 **Download History**\n"]
        for r in rows:
            sz  = f"{r['size_bytes']/1_048_576:.1f} MB" if r["size_bytes"] else "?"
            ico = icon_map.get(r["status"], "•")
            lines.append(f"{ico} `{r['filename'] or r['url'][:40]}` — {r['status']} ({sz})")
        return "\n".join(lines)

    # ── Memory skill wrappers (called by agent via skill router) ──────────────

    async def skill_remember(self, content: str, kind: str = "fact", importance: int = 3, **_) -> str:
        mid = await self.remember(content, kind=kind, importance=importance)
        return f"✅ Stored memory #{mid}"

    async def skill_recall(self, query: str, kind: str | None = None, **_) -> str:
        entries = await self.recall(query=query, kind=kind, limit=6)
        if not entries:
            return "No relevant memories found."
        lines = [f"🧠 **Memory Recall** for _'{query}'_\n"]
        for e in entries:
            lines.append(f"• [{e.kind}] {e.content}  (score: {e.score:.2f})")
        return "\n".join(lines)
