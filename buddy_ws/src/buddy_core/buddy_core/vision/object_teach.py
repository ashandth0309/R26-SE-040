"""Teaching Buddy your own objects, and recognising them again.

Rules that shape the design:

* Only embeddings are stored (no photos).  Nothing is stored until the
  person confirms the name out loud.
* People are not objects: a crop that contains a face is refused.
* A taught object is *your* thing by name ("my blue mug"); Buddy never
  claims more than "I think".  When unsure it says so and offers to learn.
* Corrections are kept: "no, that's not my mug" stops that object from
  matching that look again.
* Thresholds are uncalibrated starting values (see MatchConfig).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class ObjectStoreError(ValueError):
    pass


def norm_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 '\-]", "", (name or "").lower())
    n = re.sub(r"^(my|the|a|an)\s+", "", n.strip())

    return re.sub(r"\s+", " ", n).strip()


def _display(name: str) -> str:
    """Name as spoken back: 'my pen' is remembered as 'pen'."""

    return re.sub(r"^(?i:my|the|a|an)\s+", "", name.strip())


_SCHEMA = """
PRAGMA secure_delete=ON;
CREATE TABLE IF NOT EXISTS objects(
    object_id TEXT PRIMARY KEY, name TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
    class_hint TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS samples(
    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    vec BLOB NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS negatives(
    object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    vec BLOB NOT NULL, created REAL NOT NULL);
"""


class ObjectStore:
    def __init__(self, path: str = ":memory:", max_objects: int = 200,
                 max_samples: int = 12) -> None:
        self.path, self.max_objects, self.max_samples = (
            path, max_objects, max_samples)
        self._lock = threading.RLock()

        if path != ":memory:":
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            os.chmod(path, 0o600)

        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        self._n = self._db.execute(
            "SELECT COUNT(*) FROM objects").fetchone()[0]

    @staticmethod
    def _blob(v) -> bytes:
        return np.asarray(v, np.float32).tobytes()

    def add(self, name: str, vecs, class_hint: str | None = None) -> str:
        key = norm_name(name)

        if not key or len(key) > 40:
            raise ObjectStoreError("bad name")

        with self._lock:
            if self._db.execute("SELECT 1 FROM objects WHERE key=?",
                                (key,)).fetchone():
                raise ObjectStoreError("name already used")

            if self._db.execute("SELECT COUNT(*) FROM objects"
                                ).fetchone()[0] >= self.max_objects:
                raise ObjectStoreError("object limit reached")

            self._n += 1
            oid = f"object_{int(time.time())}_{self._n}"
            now = time.time()
            self._db.execute("INSERT INTO objects VALUES(?,?,?,?,?)",
                             (oid, _display(name), key, class_hint, now))
            self.add_samples(oid, vecs)

            return oid

    def add_samples(self, oid: str, vecs) -> None:
        with self._lock:
            have = self._db.execute(
                "SELECT COUNT(*) FROM samples WHERE object_id=?",
                (oid,)).fetchone()[0]

            for v in list(vecs)[:max(0, self.max_samples - have)]:
                self._db.execute("INSERT INTO samples VALUES(?,?,?)",
                                 (oid, self._blob(v), time.time()))

            self._db.commit()

    def add_negative(self, oid: str, vec) -> None:
        with self._lock:
            self._db.execute("INSERT INTO negatives VALUES(?,?,?)",
                             (oid, self._blob(vec), time.time()))
            self._db.execute(
                "DELETE FROM negatives WHERE object_id=? AND rowid NOT IN "
                "(SELECT rowid FROM negatives WHERE object_id=? "
                "ORDER BY created DESC LIMIT 20)", (oid, oid))
            self._db.commit()

    def find(self, name: str) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT object_id FROM objects WHERE key=?",
                                 (norm_name(name),)).fetchone()

        return r[0] if r else None

    def get(self, oid: str) -> dict | None:
        with self._lock:
            r = self._db.execute(
                "SELECT object_id,name,class_hint,created FROM objects "
                "WHERE object_id=?", (oid,)).fetchone()

        return dict(zip(("object_id", "name", "class_hint", "created"), r,
                        strict=True)) if r else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT object_id,name,class_hint,created FROM objects "
                "ORDER BY name").fetchall()

        return [dict(zip(("object_id", "name", "class_hint", "created"), r,
                         strict=True)) for r in rows]

    def rename(self, oid: str, new_name: str) -> None:
        key = norm_name(new_name)

        if not key or len(key) > 40:
            raise ObjectStoreError("bad name")

        with self._lock:
            hit = self.find(new_name)

            if hit and hit != oid:
                raise ObjectStoreError("name already used")

            self._db.execute("UPDATE objects SET name=?, key=? WHERE "
                             "object_id=?", (new_name.strip(), key, oid))
            self._db.commit()

    def gallery(self) -> dict[str, np.ndarray]:
        with self._lock:
            rows = self._db.execute(
                "SELECT object_id,vec FROM samples").fetchall()

        g: dict[str, list] = {}

        for oid, b in rows:
            g.setdefault(oid, []).append(np.frombuffer(b, np.float32))

        return {k: np.stack(v) for k, v in g.items()}

    def negatives(self) -> dict[str, np.ndarray]:
        with self._lock:
            rows = self._db.execute(
                "SELECT object_id,vec FROM negatives").fetchall()

        g: dict[str, list] = {}

        for oid, b in rows:
            g.setdefault(oid, []).append(np.frombuffer(b, np.float32))

        return {k: np.stack(v) for k, v in g.items()}

    def sample_count(self, oid: str) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM samples WHERE object_id=?",
                (oid,)).fetchone()[0]

    def delete(self, oid: str) -> bool:
        with self._lock:
            c = self._db.execute("DELETE FROM objects WHERE object_id=?",
                                 (oid,)).rowcount
            self._db.commit()

            if c and self.path != ":memory:":
                self._db.execute("VACUUM")

        return bool(c)

    def delete_all(self) -> int:
        with self._lock:
            c = self._db.execute("DELETE FROM objects").rowcount
            self._db.commit()

            if c and self.path != ":memory:":
                self._db.execute("VACUUM")

        return c

    def close(self) -> None:
        self._db.close()


@dataclass
class MatchConfig:
    accept: float = 0.80       # uncalibrated starting values
    margin: float = 0.05
    negative_sim: float = 0.92
    duplicate: float = 0.88    # "I already know this" while teaching


@dataclass(frozen=True)
class ObjectMatch:
    object_id: str | None
    name: str | None
    score: float
    margin: float
    state: str                  # "match" | "uncertain" | "unknown"


class ObjectRecognizer:
    def __init__(self, store: ObjectStore, cfg: MatchConfig | None = None
                 ) -> None:
        self.store, self.cfg = store, cfg or MatchConfig()

    def match(self, vec, class_hint: str | None = None) -> ObjectMatch:
        gal, neg = self.store.gallery(), self.store.negatives()
        scores: list[tuple[float, str]] = []

        for oid, m in gal.items():
            info = self.store.get(oid)

            if info is None:
                continue

            hint = info["class_hint"]

            if class_hint and hint and hint != class_hint:
                continue

            s = float((m @ vec).max())
            n = neg.get(oid)

            if n is not None and float((n @ vec).max()) >= \
                    self.cfg.negative_sim and float((n @ vec).max()) > s:
                continue

            scores.append((s, oid))

        if not scores:
            return ObjectMatch(None, None, 0.0, 0.0, "unknown")

        scores.sort(reverse=True)
        best, oid = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        margin = best - second
        name = self.store.get(oid)["name"]

        if best >= self.cfg.accept and margin >= self.cfg.margin:
            return ObjectMatch(oid, name, best, margin, "match")

        if best >= self.cfg.accept - 0.10:
            return ObjectMatch(oid, name, best, margin, "uncertain")

        return ObjectMatch(None, None, best, margin, "unknown")


# ---------- choosing and checking what is being shown ----------

@dataclass(frozen=True)
class Target:
    box: tuple[int, int, int, int]
    label: str | None           # COCO label if the detector had one
    source: str                 # "detector" | "center_crop"


def select_target(dets, frame_shape, allow_center_crop=True
                  ) -> Target | None:
    """The object the person is most likely showing."""

    h, w = frame_shape[:2]
    best, best_s = None, 0.0

    for d in dets:
        x1, y1, x2, y2 = d.box
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        central = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** .5
                            / 0.7)
        s = (d.area / (w * h)) ** 0.5 * 0.6 + central * 0.4 + d.score * .2

        if s > best_s:
            best, best_s = d, s

    if best is not None:
        return Target(tuple(int(v) for v in best.box), best.label,
                      "detector")

    if not allow_center_crop:
        return None

    return Target((int(w * .2), int(h * .15), int(w * .8), int(h * .85)),
                  None, "center_crop")


@dataclass
class QualityConfig:
    min_side_px: int = 64
    min_sharpness: float = 40.0     # variance of Laplacian, 224 px crop
    min_brightness: float = 40.0
    max_brightness: float = 225.0


def crop_quality(bgr, box, cfg: QualityConfig | None = None
                 ) -> tuple[bool, str]:
    cfg = cfg or QualityConfig()
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    if min(x2 - x1, y2 - y1) < cfg.min_side_px:
        return False, "too_small"

    c = cv2.resize(bgr[y1:y2, x1:x2], (224, 224))
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    b = float(g.mean())

    if b < cfg.min_brightness:
        return False, "too_dark"

    if b > cfg.max_brightness:
        return False, "too_bright"

    if float(cv2.Laplacian(g, cv2.CV_64F).var()) < cfg.min_sharpness:
        return False, "blurry"

    return True, "ok"


def overlaps_face(box, faces) -> bool:
    """True if a face sits inside (or mostly inside) the object box."""

    x1, y1, x2, y2 = box

    for f in faces or ():
        fx = (f[0] + f[2]) / 2
        fy = (f[1] + f[3]) / 2

        if x1 <= fx <= x2 and y1 <= fy <= y2:
            return True

    return False


def crop(bgr, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = bgr.shape[:2]

    return bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


# ---------- the teaching conversation ----------

class Phase(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    CONFIRM_DUPLICATE = "confirm_duplicate"
    CONFIRM_SAVE = "confirm_save"


@dataclass
class TeachConfig:
    samples_needed: int = 6
    max_similarity: float = 0.97       # newer sample must differ a little
    min_gap_s: float = 0.4
    timeout_s: float = 60.0
    max_bad_frames: int = 40


@dataclass(frozen=True)
class TeachReply:
    text: str
    phase: Phase
    saved: bool = False
    object_id: str | None = None


def parse_yes_no(text: str) -> bool | None:
    t = (text or "").lower().strip()

    if re.search(r"\b(no|nope|cancel|stop|don'?t|never mind)\b", t):
        return False

    if re.search(r"\b(yes|yeah|yep|sure|correct|right|ok|okay|please)\b",
                 t):
        return True

    return None


class TeachSession:
    def __init__(self, store: ObjectStore, embedder, cfg=None,
                 match_cfg: MatchConfig | None = None,
                 quality: QualityConfig | None = None) -> None:
        self.store, self.embedder = store, embedder
        self.cfg = cfg or TeachConfig()
        self.recognizer = ObjectRecognizer(store, match_cfg)
        self.quality = quality or QualityConfig()
        self.phase = Phase.IDLE
        self._reset()

    def _reset(self) -> None:
        self.name = ""
        self.class_hint: str | None = None
        self.vecs: list[np.ndarray] = []
        self._started = 0.0
        self._last = -1e9
        self._bad = 0
        self._dup_id: str | None = None
        self._last_hint = ""

    @property
    def active(self) -> bool:
        return self.phase is not Phase.IDLE

    def start(self, name: str, now: float) -> TeachReply:
        key = norm_name(name)

        if not key or len(key) > 40:
            return TeachReply("What should I call it?", self.phase)

        self._reset()
        self.name = name.strip()
        self._started = now
        self.phase = Phase.CAPTURING

        existing = self.store.find(name)
        extra = (" I already know that name, so I will add to what I "
                 "know." if existing else "")
        self._dup_id = existing

        return TeachReply(
            f"Okay. Hold it up in front of me and turn it slowly.{extra}",
            self.phase)

    def cancel(self, why: str = "Okay, never mind.") -> TeachReply:
        self._reset()
        self.phase = Phase.IDLE

        return TeachReply(why, self.phase)

    def feed(self, bgr, dets, now: float, faces=()) -> TeachReply | None:
        """Give one camera frame.  Returns a reply when Buddy should speak."""

        if self.phase is not Phase.CAPTURING:
            return None

        c = self.cfg

        if now - self._started > c.timeout_s:
            return self.cancel("That took too long, so I stopped. Nothing "
                               "was saved.")

        if now - self._last < c.min_gap_s:
            return None

        t = select_target(dets, bgr.shape)

        if t is None:
            return None

        if overlaps_face(t.box, faces):
            return self.cancel("That includes a person's face. I only "
                               "learn objects, so I stopped. Nothing was "
                               "saved.")

        ok, why = crop_quality(bgr, t.box, self.quality)

        if not ok:
            self._bad += 1
            hint = {"too_small": "Bring it a little closer.",
                    "too_dark": "It is too dark here.",
                    "too_bright": "There is too much glare.",
                    "blurry": "Hold it still for a second."}[why]

            if self._bad > c.max_bad_frames:
                return self.cancel("I can't see it clearly enough. "
                                   "Nothing was saved.")

            if hint != self._last_hint:
                self._last_hint = hint

                return TeachReply(hint, self.phase)

            return None

        v = self.embedder.embed(crop(bgr, t.box))

        if self.vecs and max(float(v @ u) for u in self.vecs) > \
                c.max_similarity:
            if len(self.vecs) < c.samples_needed and \
                    self._last_hint != "turn":
                self._last_hint = "turn"

                return TeachReply("Turn it a little so I see another side.",
                                  self.phase)

            return None

        if self.class_hint is None:
            self.class_hint = t.label

        self.vecs.append(v)
        self._last = now
        self._last_hint = ""

        if len(self.vecs) < c.samples_needed:
            return None

        m = self.recognizer.match(np.mean(self.vecs, axis=0),
                                  self.class_hint)

        if m.object_id and m.score >= self.recognizer.cfg.duplicate and \
                m.object_id != self._dup_id:
            self._dup_id = m.object_id
            self.phase = Phase.CONFIRM_DUPLICATE

            return TeachReply(
                f"I already know this as {m.name}. Is it the same thing?",
                self.phase)

        self.phase = Phase.CONFIRM_SAVE

        return TeachReply(f"Got it. Shall I remember this as "
                          f"{self.name}?", self.phase)

    def answer(self, text: str) -> TeachReply:
        yn = parse_yes_no(text)

        if self.phase is Phase.CONFIRM_DUPLICATE:
            if yn is None:
                return TeachReply("Is it the same thing, yes or no?",
                                  self.phase)

            if yn:
                info = self.store.get(self._dup_id)
                self.name = info["name"]
                self.phase = Phase.CONFIRM_SAVE

                return TeachReply(f"Shall I add this to what I know about "
                                  f"{self.name}?", self.phase)

            self._dup_id = self.store.find(self.name)
            self.phase = Phase.CONFIRM_SAVE

            return TeachReply(f"Okay, a different thing. Shall I remember "
                              f"it as {self.name}?", self.phase)

        if self.phase is Phase.CONFIRM_SAVE:
            if yn is None:
                return TeachReply("Should I remember it, yes or no?",
                                  self.phase)

            if not yn:
                return self.cancel("Okay, I won't remember it. Nothing "
                                   "was saved.")

            try:
                if self._dup_id and self.store.get(self._dup_id) and \
                        norm_name(self.store.get(self._dup_id)["name"]) \
                        == norm_name(self.name):
                    oid = self._dup_id
                    self.store.add_samples(oid, self.vecs)
                else:
                    oid = self.store.add(self.name, self.vecs,
                                         self.class_hint)
            except ObjectStoreError as exc:
                return self.cancel(f"I couldn't save it ({exc}).")

            name = self.name
            self._reset()
            self.phase = Phase.IDLE

            return TeachReply(f"Done. I will remember {name}.",
                              self.phase, True, oid)

        return TeachReply("", self.phase)


# ---------- answering "what is this?" and corrections ----------

@dataclass(frozen=True)
class Identification:
    kind: str                    # taught | generic | uncertain | unknown
    text: str
    name: str | None = None
    label: str | None = None
    offer_teach: bool = False
    confidence: float = 0.0      # cosine similarity (or detector score for
    #                              "generic"); NOT a calibrated probability


def identify(bgr, dets, embedder, recognizer: ObjectRecognizer,
             min_generic_score: float = 0.45,
             quality: QualityConfig | None = None,
             faces=()) -> Identification:
    t = select_target(dets, bgr.shape, allow_center_crop=True)

    if overlaps_face(t.box, faces):
        return Identification(
            "unknown", "I only name objects, not people. Show me the "
            "object on its own.")

    ok, why = crop_quality(bgr, t.box, quality)

    if not ok:
        return Identification(
            "unknown", "I can't see it clearly. Could you hold it closer "
            "and still?")

    v = embedder.embed(crop(bgr, t.box))
    m = recognizer.match(v, t.label)

    if m.state == "match":
        return Identification(
            "taught", f"I think this is your {m.name}. It looks about "
            f"{round(m.score * 100)} percent similar to what you showed me.",
            m.name, t.label, False, float(m.score))

    if m.state == "uncertain":
        return Identification(
            "uncertain", f"It might be your {m.name}, but I am not sure. "
            f"It looks about {round(m.score * 100)} percent similar.",
            m.name, t.label, True, float(m.score))

    if t.label and t.source == "detector":
        score = max((d.score for d in dets if d.label == t.label),
                    default=0.0)

        if score >= min_generic_score:
            return Identification(
                "generic", f"It looks like a {t.label}. I don't know if "
                "it is one of yours.", None, t.label, True, float(score))

    return Identification(
        "unknown", "I'm not sure what this is. Would you like to teach me?",
        None, None, True)


def correct(store: ObjectStore, wrong_name: str, vec,
            right_name: str | None = None) -> str:
    """'No, that's not my mug' (and optionally 'it's my cup')."""

    wid = store.find(wrong_name)

    if wid is None:
        return f"I don't have anything called {wrong_name}."

    store.add_negative(wid, vec)

    if right_name:
        rid = store.find(right_name)

        if rid is None:
            return ("Thanks, I'll remember it isn't "
                    f"{wrong_name}. If you'd like, teach me the new one.")

        store.add_samples(rid, [vec])

        return (f"Thanks. I'll remember it is {right_name}, not "
                f"{wrong_name}.")

    return f"Thanks, I'll remember that is not {wrong_name}."


def forget(store: ObjectStore, name: str) -> str:
    oid = store.find(name)

    if oid is None:
        return f"I don't have anything called {name}."

    shown = store.get(oid)["name"]
    store.delete(oid)

    return f"Okay, I forgot {shown}."


def forget_everything(store: ObjectStore) -> str:
    n = store.delete_all()

    return f"Okay, I forgot all {n} things you taught me." if n else \
        "I don't have anything taught to forget."
