"""Companion memory: small, explicit, consented, deletable.

* **Nothing is stored without consent.**  Consent is off by default.
  Without it, facts live only in the running session and vanish on exit.
* **Only things the user asks Buddy to remember are saved.**  Conversation
  text, audio and transcripts are never written here automatically.
* **Credentials and payment/ID numbers are refused**, consented or not.
* The user can list (``describe``), correct, delete one item, delete
  everything, export, and withdraw consent.  Deletion uses SQLite
  ``secure_delete`` plus ``VACUUM``.
* Size limits keep it small: number of facts, value length, note count.
* A corrupted database is moved aside and recreated rather than crashing.

The database file is a local runtime file and must stay out of git.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SENSITIVE_WORDS = re.compile(
    r"\b(?:password|passcode|passphrase|pin number|cvv|cvc|ssn|"
    r"social security|credit card|debit card|bank account|"
    r"account number|secret key|api key)\b"
)
LONG_NUMBER = re.compile(r"(?:\d[ -]?){12,}")


def looks_sensitive(text: str) -> bool:
    """True if ``text`` looks like a credential or payment/ID number."""

    lower = text.lower()

    return bool(SENSITIVE_WORDS.search(lower) or LONG_NUMBER.search(lower))


def slug(key: str) -> str:
    """Normalise a memory key."""

    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")[:40]


@dataclass(frozen=True)
class MemoryConfig:
    """Limits."""

    enabled: bool = True
    max_facts: int = 100
    max_value_chars: int = 200
    max_notes: int = 30
    note_retention_days: int = 365


@dataclass(frozen=True)
class MemoryResult:
    """Outcome of a write."""

    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class Fact:
    """One remembered item."""

    key: str
    value: str
    category: str
    updated: float


class CompanionMemoryManager:
    """SQLite-backed memory with consent."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        config: MemoryConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = str(path)
        self.config = config or MemoryConfig()
        self._clock = clock
        self._lock = threading.RLock()
        self._session: dict[str, str] = {}
        self.degraded = False
        self.recovered_from: str | None = None
        self._db = self._open()

    # ----- setup -----

    def _open(self) -> sqlite3.Connection:
        try:
            return self._connect()
        except sqlite3.DatabaseError:
            return self._recover()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, check_same_thread=False)
        db.execute("PRAGMA secure_delete=ON")
        db.executescript(
            "CREATE TABLE IF NOT EXISTS facts("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " category TEXT NOT NULL, updated REAL NOT NULL);"
            "CREATE TABLE IF NOT EXISTS meta("
            " k TEXT PRIMARY KEY, v TEXT NOT NULL);"
        )
        db.execute("SELECT count(*) FROM facts").fetchone()

        return db

    def _recover(self) -> sqlite3.Connection:
        self.degraded = True

        if self.path != ":memory:":
            stamp = int(self._clock())
            target = f"{self.path}.corrupt-{stamp}"

            try:
                Path(self.path).rename(target)
                self.recovered_from = target
            except OSError:
                pass

        try:
            return self._connect()
        except sqlite3.DatabaseError:
            self.path = ":memory:"

            return self._connect()

    def close(self) -> None:
        """Close the database."""

        with self._lock:
            self._db.close()

    # ----- consent -----

    @property
    def consent(self) -> bool:
        """True if the user has allowed saving to disk."""

        with self._lock:
            row = self._db.execute(
                "SELECT v FROM meta WHERE k='consent'").fetchone()

        return bool(row and row[0] == "1") and self.config.enabled

    def grant_consent(self) -> None:
        """The user allowed saving."""

        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES('consent','1')")
            self._db.commit()

    def revoke_consent(self) -> None:
        """Stop saving.  Existing data is kept until the user deletes it."""

        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES('consent','0')")
            self._db.commit()

    # ----- writing -----

    def remember(self, key: str, value: str, category: str = "fact"
                 ) -> MemoryResult:
        """Save ``value`` under ``key`` if allowed."""

        key, value = slug(key), value.strip()

        if not key or not value:
            return MemoryResult(False, "empty")

        if looks_sensitive(value) or looks_sensitive(key):
            return MemoryResult(False, "sensitive")

        if len(value) > self.config.max_value_chars:
            return MemoryResult(False, "too long")

        self._session[key] = value

        if not self.config.enabled:
            return MemoryResult(False, "memory disabled")

        if not self.consent:
            return MemoryResult(False, "no consent")

        with self._lock:
            exists = self._db.execute(
                "SELECT 1 FROM facts WHERE key=?", (key,)).fetchone()
            count = self._db.execute(
                "SELECT count(*) FROM facts").fetchone()[0]

            if not exists and count >= self.config.max_facts:
                return MemoryResult(False, "memory full")

            if category == "note" and not exists:
                notes = self._db.execute(
                    "SELECT count(*) FROM facts WHERE category='note'"
                ).fetchone()[0]

                if notes >= self.config.max_notes:
                    return MemoryResult(False, "too many notes")

            self._db.execute(
                "INSERT OR REPLACE INTO facts VALUES(?,?,?,?)",
                (key, value, category, self._clock()))
            self._db.commit()

        return MemoryResult(True)

    def add_note(self, text: str) -> MemoryResult:
        """Save a free-text note."""

        with self._lock:
            n = self._db.execute(
                "SELECT count(*) FROM facts WHERE category='note'"
            ).fetchone()[0]

        return self.remember(f"note_{int(self._clock())}_{n}", text, "note")

    def correct(self, key: str, value: str) -> MemoryResult:
        """Replace a remembered value."""

        return self.remember(key, value)

    # ----- reading -----

    def recall(self, key: str) -> str | None:
        """Value for ``key`` from disk or, failing that, this session."""

        key = slug(key)

        with self._lock:
            row = self._db.execute(
                "SELECT value FROM facts WHERE key=?", (key,)).fetchone()

        if row:
            return row[0]

        return self._session.get(key)

    def all_facts(self) -> list[Fact]:
        """Everything stored on disk."""

        with self._lock:
            rows = self._db.execute(
                "SELECT key,value,category,updated FROM facts "
                "ORDER BY updated").fetchall()

        return [Fact(*r) for r in rows]

    def describe(self) -> str:
        """Plain-language list of what is saved, for the user."""

        facts = self.all_facts()
        session_only = {k: v for k, v in self._session.items()
                        if k not in {f.key for f in facts}}

        if not facts and not session_only:
            return "I'm not remembering anything about you right now."

        parts = [f"{f.key.replace('_', ' ')}: {f.value}"
                 for f in facts[:8]]
        text = "I remember " + "; ".join(parts) + "."

        if len(facts) > 8:
            text += f" Plus {len(facts) - 8} more."

        if session_only:
            text += (" I also know a few things from this chat that "
                     "I won't keep.")

        return text

    # ----- deleting / exporting -----

    def forget(self, key: str) -> bool:
        """Delete one item.  Returns True if something was removed."""

        key = slug(key)
        removed_session = self._session.pop(key, None) is not None

        with self._lock:
            cur = self._db.execute("DELETE FROM facts WHERE key=?", (key,))
            self._db.commit()
            removed = cur.rowcount > 0
            self._vacuum()

        return removed or removed_session

    def forget_all(self) -> int:
        """Delete everything on disk and in the session."""

        self._session.clear()

        with self._lock:
            cur = self._db.execute("DELETE FROM facts")
            self._db.commit()
            count = cur.rowcount
            self._vacuum()

        return count

    def _vacuum(self) -> None:
        try:
            self._db.execute("VACUUM")
        except sqlite3.Error:
            pass

    def purge_expired(self) -> int:
        """Delete old notes beyond the retention period."""

        cutoff = self._clock() - self.config.note_retention_days * 86400

        with self._lock:
            cur = self._db.execute(
                "DELETE FROM facts WHERE category='note' AND updated<?",
                (cutoff,))
            self._db.commit()

        return cur.rowcount

    def export(self) -> dict:
        """Everything Buddy holds about the user, as plain data."""

        return {
            "consent": self.consent,
            "facts": [
                {"key": f.key, "value": f.value, "category": f.category,
                 "updated": f.updated}
                for f in self.all_facts()
            ],
            "session_only": dict(self._session),
        }

    def export_to(self, path: str | Path) -> Path:
        """Write the export as JSON and return the path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export(), indent=2,
                                     ensure_ascii=False),
                          encoding="utf-8")

        return target
