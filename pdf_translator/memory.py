"""Persistent, context-sensitive translation memory and curated training corpus."""
import hashlib
import json
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .domain import TranslationError


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def memory_key(source, context, language="auto"):
    return hashlib.sha256(json.dumps([normalized(source), normalized(context), language], ensure_ascii=False).encode()).hexdigest()


class TranslationMemory:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "memory.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS pairs (
                  id TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL,
                  context TEXT NOT NULL, language TEXT NOT NULL, quality TEXT NOT NULL,
                  document_id TEXT, hits INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def put(self, source, target, context, language="auto", quality="candidate", document_id=None):
        id_ = memory_key(source, context, language)
        with self.lock, self.connect() as db:
            old = db.execute("SELECT quality FROM pairs WHERE id=?", (id_,)).fetchone()
            if old and old["quality"] == "approved" and quality != "approved":
                return id_
            db.execute("""INSERT INTO pairs(id,source,target,context,language,quality,document_id,updated_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET target=excluded.target,
                quality=excluded.quality, document_id=excluded.document_id, updated_at=excluded.updated_at""",
                       (id_, source, target, context, language, quality, document_id, datetime.now(timezone.utc).isoformat()))
        return id_

    def lookup(self, source, context, language="auto", count=False):
        id_ = memory_key(source, context, language)
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM pairs WHERE id=? AND quality='approved'", (id_,)).fetchone()
            if row and count:
                db.execute("UPDATE pairs SET hits=hits+1 WHERE id=?", (id_,))
            return dict(row) if row else None

    def list(self, quality=None, limit=200, offset=0):
        with self.connect() as db:
            if quality:
                rows = db.execute("SELECT * FROM pairs WHERE quality=? ORDER BY updated_at DESC LIMIT ? OFFSET ?", (quality, limit, offset))
            else:
                rows = db.execute("SELECT * FROM pairs ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset))
            return [dict(row) for row in rows]

    def approve(self, id_, target=None):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM pairs WHERE id=?", (id_,)).fetchone()
            if not row:
                raise KeyError(id_)
            db.execute("UPDATE pairs SET quality='approved',target=?,updated_at=? WHERE id=?",
                       (target if target is not None else row["target"], datetime.now(timezone.utc).isoformat(), id_))
            return dict(row)

    def delete(self, id_):
        with self.lock, self.connect() as db:
            return db.execute("DELETE FROM pairs WHERE id=?", (id_,)).rowcount

    def approved(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM pairs WHERE quality='approved' ORDER BY id")]

    def count(self, name, amount=1):
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO counters VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (name, amount))

    def stats(self):
        with self.connect() as db:
            counts = {row["quality"]: row["n"] for row in db.execute("SELECT quality,COUNT(*) n FROM pairs GROUP BY quality")}
            counters = {row["name"]: row["value"] for row in db.execute("SELECT * FROM counters")}
            return {"approved": counts.get("approved", 0), "candidates": counts.get("candidate", 0),
                    "reused": db.execute("SELECT COALESCE(SUM(hits),0) FROM pairs").fetchone()[0],
                    "local_segments": counters.get("local_segments", 0), "api_segments": counters.get("api_segments", 0)}


def block_context(block):
    return " ".join(run["text"] for run in block["runs"])


class AdaptiveGateway:
    """Reuses only approved pairs; unknown text is sent to the explicitly selected engine."""
    def __init__(self, memory, fallback, language="auto", use_memory=True):
        self.memory, self.fallback, self.language = memory, fallback, language
        self.use_memory = use_memory
        self.origins, self.contexts = {}, {}

    def translate(self, blocks):
        result, unknown = {}, []
        for block in blocks:
            context = block_context(block)
            missing = []
            for run in block["runs"]:
                self.contexts[run["id"]] = context
                row = self.memory.lookup(run["text"], context, self.language, count=True) if self.use_memory else None
                if row:
                    result[run["id"]] = row["target"]
                    self.origins[run["id"]] = "memory"
                else:
                    missing.append(run)
            if missing:
                unknown.append({**block, "context": context, "runs": missing})
        if unknown:
            if self.fallback is None:
                raise TranslationError(
                    "Bellekte olmayan metinler var. API’siz yeni metin çevirmek için önce yerel modeli indir veya bir yerel AI sunucusu bağla.")
            translated = self.fallback.translate(unknown)
            result.update(translated)
            local = getattr(self.fallback, "local", False)
            for id_ in translated:
                self.origins[id_] = "local" if local else "api"
            self.memory.count("local_segments" if local else "api_segments", len(translated))
        return result

    def review(self, entries):
        result, unknown = {}, []
        for entry in entries:
            if self.origins.get(entry["id"]) == "memory":
                result[entry["id"]] = {"id": entry["id"], "status": "pass", "reason": "Aynı bağlamda kullanıcı onaylı çeviri belleğinden.", "corrected_text": None}
            else:
                unknown.append(entry)
        if unknown:
            result.update({row["id"]: row for row in self.fallback.review(unknown)})
        return [result[entry["id"]] for entry in entries]

    def close(self):
        if self.fallback:
            self.fallback.close()
