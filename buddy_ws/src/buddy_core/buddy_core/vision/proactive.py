"""Deciding whether Buddy should offer company, and how often.

This is a *policy*.  It never moves the robot and never speaks.  It returns
a suggestion (an offer, or a short acknowledgement) that the voice engine
may use or ignore.  A face is a weak hint, so an offer is always a question
("Would you like some company?"), never a statement about how someone
feels, and it is made rarely.

Defaults are deliberately conservative:

* strangers are never approached with emotional offers;
* a recognised person must have switched proactive check-ins on;
* a refusal starts a long cooldown, and a second refusal in a day a longer
  one, and "stop asking" turns the feature off for that person;
* silence is treated as "no";
* nothing is offered while someone is talking, the robot is busy, quiet
  mode is on, the person looks busy, or the camera data is stale.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from buddy_core.vision.multimodal import Agreement, FusedObservation


class PState(str, Enum):
    IDLE = "idle"
    PERSON_OBSERVED = "person_observed"
    CONTEXT_EVALUATION = "context_evaluation"
    WAITING_FOR_INVITATION = "waiting_for_invitation"
    INTERACTION_OFFERED = "interaction_offered"
    CONVERSATION_ACTIVE = "conversation_active"
    USER_DECLINED = "user_declined"
    COOLDOWN = "cooldown"
    SUSPENDED = "suspended"


class Reply(str, Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    SPACE = "space"
    STOP_ASKING = "stop_asking"
    UNCLEAR = "unclear"


_STOP_ASKING = re.compile(
    r"\b(stop (?:asking|checking)|don'?t (?:ask|check)|never ask|"
    r"quit asking|stop that)\b")
_SPACE = re.compile(r"\b(space|alone|by myself|leave me|not now|later)\b")
_DECLINE = re.compile(
    r"\b(no|nope|nah|i'?m (?:okay|ok|fine|good|all right)|"
    r"i am (?:okay|ok|fine|good)|not really|no thanks|don'?t)\b")
_ACCEPT = re.compile(
    r"\b(yes|yeah|yep|sure|okay then|ok then|please|i'?d like|"
    r"that would be nice|sounds good|come over|let'?s talk)\b")


def parse_reply(text: str) -> Reply:
    """Read the answer to an offer.  Refusals beat acceptances."""

    t = (text or "").lower().strip()

    if not t:
        return Reply.UNCLEAR

    if _STOP_ASKING.search(t):
        return Reply.STOP_ASKING

    if _SPACE.search(t):
        return Reply.SPACE

    if _DECLINE.search(t):
        return Reply.DECLINE

    if _ACCEPT.search(t):
        return Reply.ACCEPT

    return Reply.UNCLEAR


OFFERS = (
    "Hey, you seem a little quiet. Would you like some company?",
    "Hi. Would you like to chat for a bit, or would you rather have "
    "some space?",
    "Is everything okay? I'm here if you'd like to talk.",
)
ACK = {
    Reply.DECLINE: "Of course.",
    Reply.SPACE: "Okay, I'll give you some space.",
    Reply.STOP_ASKING: "Okay, I won't check in like that anymore.",
    Reply.ACCEPT: "I'm glad. I'm listening.",
}


class StateBackend(Protocol):
    def get(self, person_id: str, key: str, default=None): ...

    def set(self, person_id: str, key: str, value) -> None: ...


class MemoryState:
    def __init__(self) -> None:
        self._d: dict[tuple[str, str], object] = {}

    def get(self, person_id: str, key: str, default=None):
        return self._d.get((person_id, key), default)

    def set(self, person_id: str, key: str, value) -> None:
        self._d[(person_id, key)] = value


class StoreState:
    """Persist cooldowns in the face store (deleted with the profile)."""

    def __init__(self, store) -> None:
        self.store = store

    def get(self, person_id: str, key: str, default=None):
        return self.store.kv_get(person_id, key, default)

    def set(self, person_id: str, key: str, value) -> None:
        self.store.kv_set(person_id, key, value)


@dataclass
class ProactiveConfig:
    sustain_s: float = 60.0           # looked low this long before an offer
    min_gap_s: float = 1800.0         # between any two offers (one person)
    decline_cooldown_s: float = 7200.0
    repeat_decline_cooldown_s: float = 86400.0
    space_cooldown_s: float = 14400.0
    silence_cooldown_s: float = 3600.0
    accepted_cooldown_s: float = 900.0
    response_timeout_s: float = 20.0
    conversation_idle_s: float = 180.0
    default_max_per_day: int = 2
    max_observation_age_s: float = 5.0
    min_confidence: float = 0.30


@dataclass
class ProactiveContext:
    now: float
    day: int                                  # local calendar day number
    person_key: str | None                    # persistent id, else None
    person_settings: dict = field(default_factory=dict)
    recognized: bool = False
    person_present: bool = False
    observation: FusedObservation | None = None
    observation_stamp: float = 0.0
    camera_fresh: bool = True
    quiet_mode: bool = False                  # voice silence mode / button
    someone_speaking: bool = False
    robot_busy: bool = False
    person_busy_hint: bool = False
    invited: bool = False                     # the person asked for company
    emergency: bool = False                   # fall response etc. active


@dataclass(frozen=True)
class Suggestion:
    kind: str                 # "offer" | "ack" | "none"
    text: str
    person_key: str | None
    state: PState
    reason: str = ""
    needs_approach: bool = False   # never True here; approach is separate


NONE = Suggestion("none", "", None, PState.IDLE)


class ProactiveInteractionManager:
    def __init__(self, cfg: ProactiveConfig | None = None,
                 backend: StateBackend | None = None,
                 on_stop_asking: Callable[[str], None] | None = None
                 ) -> None:
        self.cfg = cfg or ProactiveConfig()
        self.backend = backend or MemoryState()
        self.on_stop_asking = on_stop_asking
        self.state = PState.IDLE
        self.reason = "start"
        self._since_low: float | None = None
        self._offered_to: str | None = None
        self._offer_time = 0.0
        self._conv_last = 0.0
        self._until = 0.0
        self._offer_index = 0
        self.offers_made = 0

    # ----- helpers -----

    def _get(self, key: str, k: str, default=None):
        return self.backend.get(key, k, default)

    def _set(self, key: str, k: str, v) -> None:
        self.backend.set(key, k, v)

    def _go(self, state: PState, reason: str) -> None:
        self.state, self.reason = state, reason

    def _blocked(self, ctx: ProactiveContext) -> str | None:
        if ctx.emergency:
            return "emergency_in_progress"

        if not ctx.camera_fresh:
            return "camera_not_fresh"

        if ctx.quiet_mode or ctx.person_settings.get("quiet"):
            return "quiet_mode"

        if ctx.someone_speaking:
            return "someone_speaking"

        if ctx.robot_busy:
            return "robot_busy"

        if ctx.person_busy_hint:
            return "person_busy"

        return None

    # ----- main -----

    def tick(self, ctx: ProactiveContext) -> Suggestion:
        """Call whenever perception updates.  Returns what to do now."""

        now, cfg = ctx.now, self.cfg

        if ctx.emergency:
            self._go(PState.SUSPENDED, "emergency_in_progress")

            return Suggestion("none", "", ctx.person_key, self.state,
                              self.reason)

        if self.state is PState.SUSPENDED:
            self._go(PState.IDLE, "resumed")

        if self.state is PState.INTERACTION_OFFERED:
            if now - self._offer_time > cfg.response_timeout_s:
                key = self._offered_to
                self._set(key, "cooldown_until",
                          now + cfg.silence_cooldown_s)
                self._go(PState.COOLDOWN, "no_response")

            return NONE if self.state is PState.INTERACTION_OFFERED else \
                Suggestion("none", "", key, self.state, self.reason)

        if self.state is PState.CONVERSATION_ACTIVE:
            if now - self._conv_last > cfg.conversation_idle_s or \
                    not ctx.person_present:
                self._end_conversation(now)

            return Suggestion("none", "", self._offered_to, self.state,
                              self.reason)

        if self.state in (PState.COOLDOWN, PState.USER_DECLINED):
            # A cooldown belongs to the person who caused it (it is also
            # stored per person); someone else walking in is a new case.
            if now >= self._until or (
                    ctx.person_key is not None
                    and ctx.person_key != self._offered_to):
                self._go(PState.IDLE, "cooldown_over")
            else:
                return Suggestion("none", "", ctx.person_key, self.state,
                                  self.reason)

        if not ctx.person_present:
            self._since_low = None
            self._go(PState.IDLE, "nobody_here")

            return NONE

        self._go(PState.PERSON_OBSERVED, "person_present")

        if not ctx.recognized and not ctx.invited:
            self._since_low = None
            self._go(PState.IDLE, "stranger_no_invitation")

            return Suggestion("none", "", None, self.state, self.reason)

        key = ctx.person_key

        if key is None:
            self._go(PState.IDLE, "no_persistent_identity")

            return NONE

        if not ctx.invited and not ctx.person_settings.get("proactive",
                                                           False):
            self._go(PState.IDLE, "not_opted_in")

            return Suggestion("none", "", key, self.state, self.reason)

        self._go(PState.CONTEXT_EVALUATION, "evaluating")
        blocked = self._blocked(ctx)

        if blocked:
            self._since_low = None
            self._go(PState.IDLE, blocked)

            return Suggestion("none", "", key, self.state, blocked)

        if self._get(key, "cooldown_until", 0.0) > now:
            self._go(PState.COOLDOWN, "person_cooldown")

            return Suggestion("none", "", key, self.state, self.reason)

        if ctx.invited:
            return self._make_offer(ctx, key, "invited")

        if not self._reliable_low(ctx):
            self._since_low = None
            self._go(PState.WAITING_FOR_INVITATION, "no_reliable_signal")

            return Suggestion("none", "", key, self.state, self.reason)

        if self._since_low is None:
            self._since_low = now

        if now - self._since_low < cfg.sustain_s:
            self._go(PState.WAITING_FOR_INVITATION, "not_sustained")

            return Suggestion("none", "", key, self.state, self.reason)

        if now - self._get(key, "last_offer", -1e12) < cfg.min_gap_s:
            self._go(PState.WAITING_FOR_INVITATION, "too_soon")

            return Suggestion("none", "", key, self.state, self.reason)

        limit = int(ctx.person_settings.get("max_checkins_per_day",
                                            cfg.default_max_per_day))

        if self._get(key, "day", None) == ctx.day and \
                self._get(key, "count_today", 0) >= limit:
            self._go(PState.WAITING_FOR_INVITATION, "daily_limit")

            return Suggestion("none", "", key, self.state, self.reason)

        return self._make_offer(ctx, key, "low_expression_sustained")

    def _reliable_low(self, ctx: ProactiveContext) -> bool:
        obs = ctx.observation

        if obs is None or obs.group != "negative":
            return False

        if obs.agreement not in (Agreement.CONSISTENT,
                                 Agreement.VISUAL_ONLY):
            return False

        if obs.confidence < self.cfg.min_confidence:
            return False

        return ctx.now - ctx.observation_stamp <= \
            self.cfg.max_observation_age_s

    def _make_offer(self, ctx: ProactiveContext, key: str, why: str
                    ) -> Suggestion:
        text = OFFERS[self._offer_index % len(OFFERS)]
        self._offer_index += 1
        day = ctx.day

        if self._get(key, "day", None) != day:
            self._set(key, "day", day)
            self._set(key, "count_today", 0)

        self._set(key, "count_today", self._get(key, "count_today", 0) + 1)
        self._set(key, "last_offer", ctx.now)
        self._offered_to, self._offer_time = key, ctx.now
        self._since_low = None
        self.offers_made += 1
        self._go(PState.INTERACTION_OFFERED, why)

        return Suggestion("offer", text, key, self.state, why)

    # ----- the person's answer -----

    def on_reply(self, text: str, now: float, day: int) -> Suggestion:
        """The person answered (or spoke while an offer was open)."""

        key = self._offered_to

        if self.state not in (PState.INTERACTION_OFFERED,
                              PState.CONVERSATION_ACTIVE) or key is None:
            return NONE

        reply = parse_reply(text)
        cfg = self.cfg

        if self.state is PState.CONVERSATION_ACTIVE:
            self._conv_last = now

            if reply in (Reply.STOP_ASKING,):
                return self._decline(key, reply, now, day)

            return NONE

        if reply is Reply.UNCLEAR:
            self._set(key, "cooldown_until", now + cfg.silence_cooldown_s)
            self._until = now + cfg.silence_cooldown_s
            self._go(PState.COOLDOWN, "unclear_answer")

            return Suggestion("none", "", key, self.state, self.reason)

        if reply is Reply.ACCEPT:
            self._conv_last = now
            self._go(PState.CONVERSATION_ACTIVE, "accepted")

            return Suggestion("ack", ACK[Reply.ACCEPT], key, self.state,
                              "accepted")

        return self._decline(key, reply, now, day)

    def _decline(self, key: str, reply: Reply, now: float, day: int
                 ) -> Suggestion:
        cfg = self.cfg
        same_day = self._get(key, "decline_day", None) == day
        count = (self._get(key, "declines_today", 0) if same_day else 0) + 1
        self._set(key, "decline_day", day)
        self._set(key, "declines_today", count)

        if reply is Reply.SPACE:
            wait = cfg.space_cooldown_s
        elif count >= 2:
            wait = cfg.repeat_decline_cooldown_s
        else:
            wait = cfg.decline_cooldown_s

        if reply is Reply.STOP_ASKING:
            wait = cfg.repeat_decline_cooldown_s * 30

            if self.on_stop_asking:
                self.on_stop_asking(key)

        self._set(key, "cooldown_until", now + wait)
        self._until = now + wait
        self._go(PState.USER_DECLINED, reply.value)

        return Suggestion("ack", ACK[reply], key, self.state, reply.value)

    def conversation_finished(self, now: float) -> None:
        if self.state is PState.CONVERSATION_ACTIVE:
            self._end_conversation(now)

    def _end_conversation(self, now: float) -> None:
        key = self._offered_to

        if key:
            self._set(key, "cooldown_until",
                      now + self.cfg.accepted_cooldown_s)

        self._until = now + self.cfg.accepted_cooldown_s
        self._go(PState.COOLDOWN, "conversation_over")

    def suspend(self, reason: str) -> None:
        self._go(PState.SUSPENDED, reason)

    def reset_person(self, key: str) -> None:
        """Forget the person's cooldown data (used when they are deleted)."""

        if self._offered_to == key:
            self._offered_to = None
            self._go(PState.IDLE, "person_removed")
