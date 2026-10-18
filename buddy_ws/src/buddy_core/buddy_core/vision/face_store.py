"""Local, consent-gated storage of face profiles.

What is stored: a display name, the time and method of consent, a handful
of face *embeddings* (numbers, not photos), per-person settings, and a
small audit trail.  Photos are never stored here.

Safety properties:

* a person cannot be added without a recorded consent by the subject;
* everything is deletable (one person, or all) with SQLite secure_delete
  and a VACUUM so deleted rows do not linger in the file;
* the database file is created with owner-only permissions;
* nothing leaves the machine from this module.

Not provided: encryption at rest.  Embeddings are biometric data; protect
the disk (and the SD card) accordingly.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from buddy_core.vision.embedding import l2_normalize


class ConsentError(ValueError):
    """Tried to store a person without valid consent."""


class StoreError(ValueError):
    """Invalid request (limit reached, duplicate name, bad input)."""


@dataclass(frozen=True)
class Consent:
    """Record of the subject agreeing to be remembered."""

    given_at: float
    method: str                 # e.g. "spoken_yes"
    subject_confirmed: bool     # the enrolled person themself said yes
    prompt: str = ""


DEFAULT_SETTINGS = {
    "greet": True,               # may Buddy greet this person by name
    "proactive": False,          # may Buddy start check-ins
    "quiet": False,              # prefers little interaction
    "recognition": True,         # False: do not match this person at all
    "preferred_greeting": "",
    "max_checkins_per_day": 2,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people(
    person_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created REAL NOT NULL,
    consent_at REAL NOT NULL,
    consent_method TEXT NOT NULL,
    last_seen REAL,
    last_greeted REAL,
    settings TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    vec BLOB NOT NULL,
    dim INTEGER NOT NULL,
    quality REAL,
    yaw REAL,
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS negatives(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    vec BLOB NOT NULL,
    dim INTEGER NOT NULL,
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv(
    person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(person_id, key)
);
CREATE TABLE IF NOT EXISTS audit(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    person_id TEXT,
    detail TEXT
);
"""

_NAME_RE = re.compile(r"[^\w '\-]", re.UNICODE)


def clean_name(name: str) -> str:
    name = _NAME_RE.sub("", str(name or "")).strip()
    name = re.sub(r"\s+", " ", name)[:40]

    return name.title() if name.islower() else name


class FaceProfileStore:
    def __init__(self, path: str = ":memory:", max_people: int = 20,
                 max_templates: int = 24, dim: int = 128,
                 clock: Callable[[], float] = time.time) -> None:
        self.path, self.max_people = path, max_people
        self.max_templates, self.dim, self._clock = \
            max_templates, dim, clock
        self._lock = threading.RLock()

        if path != ":memory:":
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, mode=0o700, exist_ok=True)
            os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))
            os.chmod(path, 0o600)

        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA secure_delete=ON")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ----- helpers -----

    def _audit(self, event: str, person_id: str | None = None,
               detail: str = "") -> None:
        self._db.execute(
            "INSERT INTO audit(ts,event,person_id,detail) VALUES(?,?,?,?)",
            (self._clock(), event, person_id, detail))

    def _vec_blob(self, vec: np.ndarray) -> bytes:
        v = l2_normalize(vec)

        if v.size != self.dim:
            raise StoreError(f"embedding must have {self.dim} values")

        return v.astype(np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    # ----- people -----

    def add_person(self, display_name: str, consent: Consent) -> str:
        if not isinstance(consent, Consent) or not consent.subject_confirmed:
            raise ConsentError(
                "consent must come from the person being enrolled")

        name = clean_name(display_name)

        if not name:
            raise StoreError("a name is required")

        with self._lock:
            if self._db.execute(
                    "SELECT 1 FROM people WHERE lower(display_name)=?",
                    (name.lower(),)).fetchone():
                raise StoreError(f"someone named {name} is already saved")

            count = self._db.execute(
                "SELECT COUNT(*) FROM people").fetchone()[0]

            if count >= self.max_people:
                raise StoreError("profile limit reached")

            person_id = "p_" + uuid.uuid4().hex[:10]
            self._db.execute(
                "INSERT INTO people VALUES(?,?,?,?,?,?,?,?)",
                (person_id, name, self._clock(), consent.given_at,
                 consent.method, None, None,
                 json.dumps(DEFAULT_SETTINGS)))
            self._audit("enrolled", person_id, consent.method)
            self._db.commit()

            return person_id

    def add_embedding(self, person_id: str, vec: np.ndarray,
                      quality: float | None = None,
                      yaw: float | None = None) -> None:
        with self._lock:
            if not self._exists(person_id):
                raise StoreError("unknown person")

            n = self._db.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id=?",
                (person_id,)).fetchone()[0]

            if n >= self.max_templates:
                # Keep the newest: drop the oldest template.
                self._db.execute(
                    "DELETE FROM embeddings WHERE id=(SELECT MIN(id) "
                    "FROM embeddings WHERE person_id=?)", (person_id,))

            self._db.execute(
                "INSERT INTO embeddings(person_id,vec,dim,quality,yaw,"
                "added) VALUES(?,?,?,?,?,?)",
                (person_id, self._vec_blob(vec), self.dim, quality, yaw,
                 self._clock()))
            self._db.commit()

    def _exists(self, person_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM people WHERE person_id=?",
                                (person_id,)).fetchone() is not None

    def gallery(self) -> dict[str, np.ndarray]:
        """Templates of every person who has recognition switched on."""

        with self._lock:
            out: dict[str, list[np.ndarray]] = {}
            rows = self._db.execute(
                "SELECT e.person_id, e.vec, p.settings FROM embeddings e "
                "JOIN people p USING(person_id) ORDER BY e.id").fetchall()

            for pid, blob, settings in rows:
                if json.loads(settings).get("recognition", True):
                    out.setdefault(pid, []).append(self._from_blob(blob))

            return {k: np.stack(v) for k, v in out.items()}

    def template_count(self, person_id: str) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id=?",
                (person_id,)).fetchone()[0]

    def get_person(self, person_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT person_id,display_name,created,consent_at,"
                "consent_method,last_seen,last_greeted,settings "
                "FROM people WHERE person_id=?", (person_id,)).fetchone()

        if row is None:
            return None

        return {"person_id": row[0], "display_name": row[1],
                "created": row[2], "consent_at": row[3],
                "consent_method": row[4], "last_seen": row[5],
                "last_greeted": row[6],
                "settings": {**DEFAULT_SETTINGS, **json.loads(row[7])},
                "templates": self.template_count(row[0])}

    def find_by_name(self, name: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT person_id FROM people WHERE lower(display_name)=?",
                (clean_name(name).lower(),)).fetchone()

        return row[0] if row else None

    def list_people(self) -> list[dict]:
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT person_id FROM people ORDER BY created")]

        return [self.get_person(i) for i in ids]

    def set_setting(self, person_id: str, key: str, value) -> None:
        if key not in DEFAULT_SETTINGS:
            raise StoreError(f"unknown setting {key!r}")

        with self._lock:
            person = self.get_person(person_id)

            if person is None:
                raise StoreError("unknown person")

            settings = person["settings"]
            settings[key] = value
            self._db.execute("UPDATE people SET settings=? WHERE "
                             "person_id=?", (json.dumps(settings),
                                             person_id))
            self._audit("setting", person_id, key)
            self._db.commit()

    def rename(self, person_id: str, new_name: str) -> None:
        name = clean_name(new_name)

        with self._lock:
            if not name or not self._exists(person_id):
                raise StoreError("bad rename")

            other = self.find_by_name(name)

            if other and other != person_id:
                raise StoreError(f"someone named {name} is already saved")

            self._db.execute("UPDATE people SET display_name=? WHERE "
                             "person_id=?", (name, person_id))
            self._audit("renamed", person_id)
            self._db.commit()

    def mark_seen(self, person_id: str, when: float | None = None) -> None:
        with self._lock:
            self._db.execute("UPDATE people SET last_seen=? WHERE "
                             "person_id=?", (when or self._clock(),
                                             person_id))
            self._db.commit()

    def mark_greeted(self, person_id: str,
                     when: float | None = None) -> None:
        with self._lock:
            self._db.execute("UPDATE people SET last_greeted=? WHERE "
                             "person_id=?", (when or self._clock(),
                                             person_id))
            self._db.commit()

    # ----- small per-person state (cooldowns, counters) -----

    def kv_get(self, person_id: str, key: str, default=None):
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM kv WHERE person_id=? AND key=?",
                (person_id, key)).fetchone()

        return json.loads(row[0]) if row else default

    def kv_set(self, person_id: str, key: str, value) -> None:
        with self._lock:
            if not self._exists(person_id):
                raise StoreError("unknown person")

            self._db.execute(
                "INSERT INTO kv(person_id,key,value) VALUES(?,?,?) "
                "ON CONFLICT(person_id,key) DO UPDATE SET value=excluded."
                "value", (person_id, key, json.dumps(value)))
            self._db.commit()

    # ----- corrections -----

    def add_negative(self, person_id: str, vec: np.ndarray,
                     max_self_similarity: float = 0.85) -> None:
        """Remember a face that is NOT this person (user correction).

        Refused when the face is nearly identical to the person's own
        saved looks: that would lock the real person out.
        """

        with self._lock:
            if not self._exists(person_id):
                raise StoreError("unknown person")

            own = self.gallery().get(person_id)

            if own is not None and float(np.max(own @ l2_normalize(vec))
                                         ) >= max_self_similarity:
                raise StoreError(
                    "that face looks almost the same as the saved person; "
                    "remove or re-enrol the profile instead")

            self._db.execute(
                "INSERT INTO negatives(person_id,vec,dim,added) "
                "VALUES(?,?,?,?)",
                (person_id, self._vec_blob(vec), self.dim, self._clock()))
            self._audit("correction", person_id, "not this person")
            self._db.commit()

    def negatives(self) -> dict[str, np.ndarray]:
        with self._lock:
            out: dict[str, list[np.ndarray]] = {}

            for pid, blob in self._db.execute(
                    "SELECT person_id, vec FROM negatives"):
                out.setdefault(pid, []).append(self._from_blob(blob))

            return {k: np.stack(v) for k, v in out.items()}

    # ----- deletion, retention, export -----

    def _vacuum(self) -> None:
        self._db.commit()
        self._db.execute("VACUUM")

    def delete_person(self, person_id: str) -> bool:
        with self._lock:
            if not self._exists(person_id):
                return False

            self._db.execute("DELETE FROM embeddings WHERE person_id=?",
                             (person_id,))
            self._db.execute("DELETE FROM negatives WHERE person_id=?",
                             (person_id,))
            self._db.execute("DELETE FROM kv WHERE person_id=?",
                             (person_id,))
            self._db.execute("DELETE FROM people WHERE person_id=?",
                             (person_id,))
            # The trail keeps that a deletion happened, not who.
            self._db.execute("UPDATE audit SET person_id=NULL, detail='' "
                             "WHERE person_id=?", (person_id,))
            self._audit("deleted_person")
            self._vacuum()

            return True

    def delete_all(self) -> int:
        with self._lock:
            n = self._db.execute(
                "SELECT COUNT(*) FROM people").fetchone()[0]

            for table in ("embeddings", "negatives", "kv", "people",
                          "audit"):
                self._db.execute(f"DELETE FROM {table}")

            self._audit("deleted_all")
            self._vacuum()

            return n

    def purge_not_seen_since(self, cutoff: float) -> int:
        """Retention: delete people not seen since ``cutoff`` (and never
        newly enrolled after it).  Returns how many were removed."""

        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT person_id FROM people WHERE "
                "COALESCE(last_seen, created) < ?", (cutoff,))]

        for pid in ids:
            self.delete_person(pid)

        return len(ids)

    def export_person(self, person_id: str,
                      include_vectors: bool = False) -> dict | None:
        person = self.get_person(person_id)

        if person is None:
            return None

        if include_vectors:
            with self._lock:
                rows = self._db.execute(
                    "SELECT vec FROM embeddings WHERE person_id=?",
                    (person_id,)).fetchall()

            person["vectors"] = [self._from_blob(r[0]).tolist()
                                 for r in rows]

        return person

    def audit_log(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts,event,person_id,detail FROM audit "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

        return [{"ts": r[0], "event": r[1], "person_id": r[2],
                 "detail": r[3]} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
