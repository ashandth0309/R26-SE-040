"""Where things were last seen, and which room Buddy is looking at.

Privacy first: this is OFF until the user turns it on by voice.  It
remembers *where and when* a taught object (or a chosen everyday class)
was last seen, never an image and never a person.  Old entries expire
(seven days by default) and the user can forget one thing or everything.

Honesty about what it knows:

* It only knows what the camera has seen.  "Last seen on the kitchen
  table 20 minutes ago" is a report, not a promise the thing is still
  there.
* Without localisation, a "place" is either a room/spot the user taught
  by name (scene memory) or a coarse position in the camera view.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

_SCHEMA = """
PRAGMA secure_delete=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sightings(
    subject TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT, name TEXT,
    place TEXT, region TEXT, first_seen REAL NOT NULL,
    last_seen REAL NOT NULL, seen_count INTEGER NOT NULL,
    score REAL NOT NULL, prev_place TEXT, moved_at REAL);
CREATE TABLE IF NOT EXISTS places(
    name TEXT PRIMARY KEY, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS place_vecs(
    name TEXT NOT NULL REFERENCES places(name) ON DELETE CASCADE,
    vec BLOB NOT NULL);
"""


def region_of(box, frame_shape) -> str:
    h, w = frame_shape[:2]
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "middle"
    row = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "centre"

    return f"{row} {col}" if row != "centre" else f"centre {col}" \
        if col != "middle" else "centre"


def ago(seconds: float) -> str:
    s = max(0.0, seconds)

    if s < 90:
        return "a moment ago"

    if s < 3600:
        return f"about {round(s / 60)} minutes ago"

    if s < 86400:
        h = round(s / 3600)

        return f"about {h} hour{'s' if h != 1 else ''} ago"

    d = round(s / 86400)

    return f"about {d} day{'s' if d != 1 else ''} ago"


@dataclass(frozen=True)
class WhereAnswer:
    found: bool
    text: str
    age_s: float | None = None
    stale: bool = False


class ObjectMemory:
    def __init__(self, path: str = ":memory:", retention_s: float = 7 * 86400,
                 max_rows: int = 500, min_update_s: float = 5.0,
                 stale_after_s: float = 6 * 3600) -> None:
        self.path = path
        self.retention_s, self.max_rows = retention_s, max_rows
        self.min_update_s, self.stale_after_s = min_update_s, stale_after_s
        self._lock = threading.RLock()

        if path != ":memory:":
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            os.chmod(path, 0o600)

        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    # ----- switch -----

    @property
    def enabled(self) -> bool:
        with self._lock:
            r = self._db.execute(
                "SELECT value FROM meta WHERE key='enabled'").fetchone()

        return bool(r and r[0] == "1")

    def enable(self) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES"
                             "('enabled','1')")
            self._db.commit()

    def disable(self, forget: bool = True) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES"
                             "('enabled','0')")
            self._db.commit()

        if forget:
            self.forget_all()

    # ----- recording -----

    def observe(self, subject: str, kind: str, label: str | None,
                name: str | None, place: str | None, box, frame_shape,
                now: float, score: float = 1.0) -> bool:
        """Record a *stable* sighting.  Does nothing while disabled."""

        if not self.enabled or kind not in ("taught", "generic"):
            return False

        if label == "person":
            return False

        region = region_of(box, frame_shape) if box is not None else None

        with self._lock:
            r = self._db.execute(
                "SELECT place,last_seen,seen_count FROM sightings "
                "WHERE subject=?", (subject,)).fetchone()

            if r is None:
                if self._db.execute("SELECT COUNT(*) FROM sightings"
                                    ).fetchone()[0] >= self.max_rows:
                    self._db.execute(
                        "DELETE FROM sightings WHERE subject IN (SELECT "
                        "subject FROM sightings ORDER BY last_seen LIMIT 1)")

                self._db.execute(
                    "INSERT INTO sightings VALUES(?,?,?,?,?,?,?,?,?,?,NULL,"
                    "NULL)", (subject, kind, label, name, place, region,
                              now, now, 1, score))
                self._db.commit()

                return True

            old_place, last, count = r
            changed = bool(place and old_place and place != old_place)

            if now - last < self.min_update_s and not changed:
                return False

            if changed:
                self._db.execute(
                    "UPDATE sightings SET prev_place=?, moved_at=? "
                    "WHERE subject=?", (old_place, now, subject))

            self._db.execute(
                "UPDATE sightings SET place=COALESCE(?,place), region=?, "
                "last_seen=?, seen_count=?, score=?, name=COALESCE(?,name) "
                "WHERE subject=?",
                (place, region, now, count + 1, score, name, subject))
            self._db.commit()

        return True

    # ----- asking -----

    def where_is(self, name: str, now: float) -> WhereAnswer:
        if not self.enabled:
            return WhereAnswer(
                False, "I'm not keeping track of where things are. You can "
                "ask me to start.")

        key = (name or "").lower().strip()

        with self._lock:
            rows = self._db.execute(
                "SELECT name,label,place,region,last_seen,prev_place FROM "
                "sightings WHERE lower(name)=? OR (name IS NULL AND "
                "lower(label)=?) ORDER BY last_seen DESC",
                (key, key)).fetchall()

        if not rows:
            return WhereAnswer(False, f"I haven't seen your {name}.")

        nm, label, place, region, last, prev = rows[0]
        age = now - last
        where = f"on the {place}" if place and not place.startswith(
            ("in ", "on ", "at ")) else (place or f"in the {region} of "
                                         "my view")
        text = f"I last saw your {nm or label} {where}, {ago(age)}."
        stale = age > self.stale_after_s

        if stale:
            text += " It may have been moved since."
        else:
            text += " I can't tell whether it is still there."

        return WhereAnswer(True, text, age, stale)

    def seen_at(self, place: str, now: float, within_s: float = 1800.0
                ) -> list[str]:
        if not self.enabled:
            return []

        with self._lock:
            rows = self._db.execute(
                "SELECT COALESCE(name,label) FROM sightings WHERE place=? "
                "AND last_seen>=? ORDER BY last_seen DESC",
                (place, now - within_s)).fetchall()

        return [r[0] for r in rows]

    def moved_recently(self, name: str, now: float, within_s: float = 3600.0
                       ) -> tuple[str, str] | None:
        with self._lock:
            r = self._db.execute(
                "SELECT prev_place,place,moved_at FROM sightings WHERE "
                "lower(name)=?", ((name or "").lower().strip(),)).fetchone()

        if r and r[2] and now - r[2] <= within_s:
            return r[0], r[1]

        return None

    # ----- forgetting -----

    def forget(self, name: str) -> int:
        with self._lock:
            c = self._db.execute(
                "DELETE FROM sightings WHERE lower(name)=? OR "
                "(name IS NULL AND lower(label)=?)",
                ((name or "").lower().strip(),) * 2).rowcount
            self._db.commit()

        return c

    def forget_all(self) -> int:
        with self._lock:
            c = self._db.execute("DELETE FROM sightings").rowcount
            self._db.commit()

            if self.path != ":memory:":
                self._db.execute("VACUUM")

        return c

    def purge(self, now: float) -> int:
        with self._lock:
            c = self._db.execute("DELETE FROM sightings WHERE last_seen<?",
                                 (now - self.retention_s,)).rowcount
            self._db.commit()

        return c

    def export(self) -> str:
        with self._lock:
            rows = self._db.execute(
                "SELECT subject,kind,label,name,place,region,first_seen,"
                "last_seen,seen_count FROM sightings").fetchall()

        keys = ("subject", "kind", "label", "name", "place", "region",
                "first_seen", "last_seen", "seen_count")

        return json.dumps([dict(zip(keys, r, strict=True)) for r in rows],
                          indent=2)

    # ----- places (scene memory) -----

    def teach_place(self, name: str, vecs, now: float | None = None
                    ) -> None:
        name = (name or "").strip().lower()
        vecs = [np.asarray(v, np.float32) for v in vecs]

        if not name or len(name) > 40:
            raise ValueError("bad place name")

        if len(vecs) < 3:
            raise ValueError("need at least three views of the place")

        with self._lock:
            self._db.execute("DELETE FROM places WHERE name=?", (name,))
            self._db.execute("INSERT INTO places VALUES(?,?)",
                             (name, now or time.time()))

            for v in vecs[:20]:
                self._db.execute("INSERT INTO place_vecs VALUES(?,?)",
                                 (name, v.tobytes()))

            self._db.commit()

    def forget_place(self, name: str) -> bool:
        with self._lock:
            c = self._db.execute("DELETE FROM places WHERE name=?",
                                 ((name or "").strip().lower(),)).rowcount
            self._db.commit()

        return bool(c)

    def places(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT name FROM places ORDER BY name")]

    def recognise_place(self, vec, accept: float = 0.80,
                        margin: float = 0.04) -> tuple[str | None, float]:
        """Which taught place does this whole-frame embedding look like?"""

        with self._lock:
            rows = self._db.execute(
                "SELECT name,vec FROM place_vecs").fetchall()

        best: dict[str, float] = {}

        for nm, b in rows:
            s = float(np.frombuffer(b, np.float32) @ vec)
            best[nm] = max(best.get(nm, -1.0), s)

        if not best:
            return None, 0.0

        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        top_name, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        if top >= accept and top - second >= margin:
            return top_name, top

        return None, top

    def close(self) -> None:
        self._db.close()
