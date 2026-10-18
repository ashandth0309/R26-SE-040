"""Consent-based face enrollment, driven by speech and face samples.

The flow (voice wording is chosen by the voice engine; the texts here are
suggestions):

    begin(name)            -> "Would you like me to remember your face?"
    on_utterance("yes")    -> starts collecting (only if the *person* said
                              it; "yes they do" is refused)
    on_face(...)* -> progress, several poses, one face in view
    on_utterance("yes")    -> final confirmation, then it is saved

Nothing touches the store until the final confirmation.  Samples live only
in memory until then, and are thrown away on refusal, timeout or any change
of the person in front of the camera.

Limits that this module cannot remove: the system cannot prove *who* spoke
the consent word.  It refuses third-person answers and requires one stable
face in view, which is the strongest check available without speaker
verification.  A physical confirm button would be stronger.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from buddy_core.vision.embedding import l2_normalize
from buddy_core.vision.face_store import (
    Consent,
    FaceProfileStore,
    StoreError,
    clean_name,
)
from buddy_core.vision.identity import IdentityState, MatchConfig, match


class Consent3(str, Enum):
    YES = "yes"
    NO = "no"
    THIRD_PARTY = "third_party"
    UNCLEAR = "unclear"


_YES = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "yes please",
        "i agree", "i consent", "go ahead", "that is fine",
        "that's fine", "yes you can", "i am fine with that",
        "i'm fine with that", "i am okay with that", "i'm okay with that",
        "yes i do", "yes i would", "yes i am"}
_YES_LEAD = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok"}
_YES_FILL = {"please", "i", "do", "would", "am", "you", "can", "agree",
             "consent", "that", "is", "fine", "it", "s", "thanks", "thank",
             "go", "ahead", "of", "course", "really", "definitely", "sure",
             "okay", "ok", "yes", "yeah"}
_NO_WORDS = {"no", "nope", "nah", "dont", "don't", "not", "never",
             "cancel", "stop", "refuse", "decline", "cant", "can't"}
_THIRD = {"they", "he", "she", "him", "her", "them", "their", "his",
          "hers", "friend", "mom", "dad", "mum"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def parse_consent(text: str) -> Consent3:
    """Read a yes/no answer.  Anything unsure is UNCLEAR, never yes."""

    words = _tokens(text)

    if not words:
        return Consent3.UNCLEAR

    phrase = " ".join(w for w in words if w != "buddy")

    if any(w in _NO_WORDS for w in words):
        return Consent3.NO

    if any(w in _THIRD for w in words):
        return Consent3.THIRD_PARTY

    rest = [w for w in phrase.split()[1:]]

    if phrase in _YES or (phrase.split() and phrase.split()[0] in
                          _YES_LEAD and all(w in _YES_FILL for w in rest)):
        return Consent3.YES

    return Consent3.UNCLEAR


class EnrollState(str, Enum):
    IDLE = "idle"
    AWAITING_CONSENT = "awaiting_consent"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    DONE = "done"
    CANCELLED = "cancelled"
    REFUSED = "refused"


@dataclass
class EnrollmentConfig:
    min_samples: int = 5
    target_samples: int = 8
    min_gap_s: float = 0.4
    near_duplicate_cos: float = 0.985
    same_person_cos: float = 0.40
    consent_timeout_s: float = 30.0
    capture_timeout_s: float = 60.0
    confirm_timeout_s: float = 30.0
    min_pose_bins: int = 2
    match: MatchConfig = field(default_factory=MatchConfig)


@dataclass(frozen=True)
class FaceSample:
    track_id: int
    embedding: np.ndarray
    usable: bool                 # passed the quality checks
    yaw_hint: float | None
    faces_in_view: int
    quality: float | None = None


@dataclass(frozen=True)
class EnrollReply:
    state: EnrollState
    say: str
    done: bool = False
    person_id: str | None = None


def pose_bin(yaw: float | None) -> str:
    if yaw is None:
        return "front"

    return "left" if yaw < -0.15 else "right" if yaw > 0.15 else "front"


class EnrollmentManager:
    def __init__(self, store: FaceProfileStore,
                 cfg: EnrollmentConfig | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.store, self.cfg = store, cfg or EnrollmentConfig()
        self._clock, self._wall = clock, wall
        self._reset()

    def _reset(self) -> None:
        self.state = EnrollState.IDLE
        self.name = ""
        self.track_id: int | None = None
        self._deadline = 0.0
        self._samples: list[tuple[np.ndarray, float | None, float | None]] \
            = []
        self._last_sample_t = -1e9
        self._asked_again = False
        self._existing: str | None = None
        self.consent: Consent | None = None

    # ----- control -----

    def begin(self, name: str, track_id: int | None,
              faces_in_view: int) -> EnrollReply:
        """Start.  ``track_id`` is the person who will be enrolled."""

        if self.state not in (EnrollState.IDLE, EnrollState.DONE,
                              EnrollState.CANCELLED, EnrollState.REFUSED):
            return EnrollReply(self.state, "I'm already in the middle of "
                               "remembering someone.")

        self._reset()
        clean = clean_name(name)

        if not clean:
            return self._refuse("I didn't catch a name.")

        if track_id is None or faces_in_view == 0:
            return self._refuse("I can't see anyone yet. Please stand in "
                                "front of me.")

        if faces_in_view != 1:
            return self._refuse("I can see more than one person. Only the "
                                "person who wants to be remembered should "
                                "be in front of me.")

        existing = self.store.find_by_name(clean)
        self.name, self.track_id = clean, track_id
        self._existing = existing
        self.state = EnrollState.AWAITING_CONSENT
        self._deadline = self._clock() + self.cfg.consent_timeout_s

        return EnrollReply(
            self.state,
            f"{clean}, would you like me to remember your face so I can "
            "recognise you? You can ask me to forget it any time. "
            "Please answer yourself.")

    def cancel(self) -> EnrollReply:
        had = self.state not in (EnrollState.IDLE, EnrollState.DONE)
        self._reset()
        self.state = EnrollState.CANCELLED

        return EnrollReply(self.state, "Okay, I won't remember anyone."
                           if had else "Nothing to cancel.")

    def _refuse(self, say: str) -> EnrollReply:
        self._reset()
        self.state = EnrollState.REFUSED

        return EnrollReply(self.state, say)

    def tick(self) -> EnrollReply | None:
        """Call periodically; ends a stalled enrollment."""

        if self.state in (EnrollState.AWAITING_CONSENT,
                          EnrollState.COLLECTING,
                          EnrollState.CONFIRMING) and \
                self._clock() > self._deadline:
            self._reset()
            self.state = EnrollState.CANCELLED

            return EnrollReply(self.state, "That took too long, so I "
                               "didn't save anything.")

        return None

    # ----- speech -----

    def on_utterance(self, text: str) -> EnrollReply | None:
        if self.state == EnrollState.AWAITING_CONSENT:
            return self._on_consent(text)

        if self.state == EnrollState.CONFIRMING:
            return self._on_confirm(text)

        if self.state == EnrollState.COLLECTING and \
                parse_consent(text) == Consent3.NO:
            return self.cancel()

        return None

    def _on_consent(self, text: str) -> EnrollReply:
        answer = parse_consent(text)

        if answer == Consent3.NO:
            self._reset()
            self.state = EnrollState.CANCELLED

            return EnrollReply(self.state, "Okay, I won't remember you.")

        if answer == Consent3.YES:
            self.consent = Consent(self._wall(), "spoken_yes", True,
                                   "remember your face?")
            self.state = EnrollState.COLLECTING
            self._deadline = self._clock() + self.cfg.capture_timeout_s

            return EnrollReply(
                self.state,
                "Thank you. Please look at me and slowly turn your head "
                "a little to each side.")

        if answer == Consent3.THIRD_PARTY:
            return EnrollReply(
                self.state,
                "I need to hear it from the person being remembered. "
                "Could you say yes or no yourself?")

        if not self._asked_again:
            self._asked_again = True

            return EnrollReply(self.state, "Sorry, was that a yes or a "
                               "no?")

        self._reset()
        self.state = EnrollState.CANCELLED

        return EnrollReply(self.state, "I wasn't sure, so I won't save "
                           "anything.")

    def _on_confirm(self, text: str) -> EnrollReply:
        answer = parse_consent(text)

        if answer == Consent3.YES:
            return self._save()

        if answer in (Consent3.NO, Consent3.THIRD_PARTY):
            self._reset()
            self.state = EnrollState.CANCELLED

            return EnrollReply(self.state, "Okay, I threw those away. "
                               "Nothing was saved.")

        return EnrollReply(self.state, "Should I save this? Yes or no.")

    # ----- vision -----

    def on_face(self, sample: FaceSample) -> EnrollReply | None:
        if self.state != EnrollState.COLLECTING:
            return None

        if sample.faces_in_view != 1 or sample.track_id != self.track_id:
            self._reset()
            self.state = EnrollState.CANCELLED

            return EnrollReply(
                self.state, "Someone else came into view, so I stopped. "
                "I haven't saved anything.")

        if not sample.usable:
            return None

        now = self._clock()

        if now - self._last_sample_t < self.cfg.min_gap_s:
            return None

        vec = l2_normalize(sample.embedding)

        if self._samples:
            first = self._samples[0][0]

            if float(first @ vec) < self.cfg.same_person_cos:
                self._reset()
                self.state = EnrollState.CANCELLED

                return EnrollReply(
                    self.state, "The face changed, so I stopped without "
                    "saving anything.")

            if max(float(s[0] @ vec) for s in self._samples) > \
                    self.cfg.near_duplicate_cos:
                return None

        self._samples.append((vec, sample.yaw_hint, sample.quality))
        self._last_sample_t = now

        if self._ready():
            self.state = EnrollState.CONFIRMING
            self._deadline = now + self.cfg.confirm_timeout_s

            return EnrollReply(
                self.state,
                f"I have {len(self._samples)} good looks. Shall I save "
                f"you as {self.name}? Yes or no.")

        return None

    def _bins(self) -> int:
        return len({pose_bin(y) for _, y, _ in self._samples})

    def _ready(self) -> bool:
        n = len(self._samples)

        return n >= self.cfg.target_samples or (
            n >= self.cfg.min_samples and
            self._bins() >= self.cfg.min_pose_bins)

    @property
    def progress(self) -> dict:
        return {"state": self.state.value, "samples": len(self._samples),
                "needed": self.cfg.min_samples,
                "pose_bins": self._bins() if self._samples else 0}

    # ----- save -----

    def _save(self) -> EnrollReply:
        mean = l2_normalize(np.mean([v for v, _, _ in self._samples],
                                    axis=0))
        gallery = self.store.gallery()
        found = match(mean, gallery, self.cfg.match)
        existing_id = self._existing

        if found.state is IdentityState.RECOGNIZED and \
                found.person_id != existing_id:
            other = self.store.get_person(found.person_id)
            self._reset()
            self.state = EnrollState.REFUSED

            return EnrollReply(
                self.state, "I think I already know you as "
                f"{other['display_name']}, so I didn't create a second "
                "profile.")

        try:
            if existing_id:
                if found.person_id != existing_id:
                    self._reset()
                    self.state = EnrollState.REFUSED

                    return EnrollReply(
                        self.state, "That name is already saved for "
                        "someone who doesn't look like you, so I didn't "
                        "change it.")

                pid = existing_id
            else:
                pid = self.store.add_person(self.name, self.consent)

            for vec, yaw, quality in self._samples:
                self.store.add_embedding(pid, vec, quality, yaw)
        except StoreError as exc:
            self._reset()
            self.state = EnrollState.REFUSED

            return EnrollReply(self.state, f"I couldn't save that: {exc}.")

        name, n = self.name, len(self._samples)
        self._reset()
        self.state = EnrollState.DONE

        return EnrollReply(
            self.state, f"Done. I'll remember you, {name}, from {n} "
            "pictures' worth of face data. I don't keep the photos. Just "
            "say 'forget me' to remove it.", True, pid)
