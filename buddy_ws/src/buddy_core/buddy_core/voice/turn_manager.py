"""Turn-taking state machine.

States follow the conversation, not the audio hardware:

SLEEPING            Not listening at all (privacy mute or put to sleep).
                    Only an explicit resume (button, ROS service) leaves it.
LISTENING_FOR_WAKE  Idle; waits for the wake word.  Stop is still heard.
AWAKE               Ready for a command or conversation.
USER_SPEAKING       The user is talking.
PROCESSING          Deciding what to say or do.
BUDDY_SPEAKING      Buddy is talking.  Barge-in is possible.
INTERRUPTED         The user talked over Buddy; speech is being cancelled.
COOLDOWN            Short pause after Buddy finishes, so its tail is not
                    heard as the user.
ERROR_RECOVERY      Something failed; wait, then return to listening.

Illegal events are counted and ignored rather than raising, because
a voice loop must never crash on a surprising event order.  ``stop`` is
legal in every state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class TurnState(str, Enum):
    """Conversation states."""

    SLEEPING = "SLEEPING"
    LISTENING_FOR_WAKE = "LISTENING_FOR_WAKE"
    AWAKE = "AWAKE"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    BUDDY_SPEAKING = "BUDDY_SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    COOLDOWN = "COOLDOWN"
    ERROR_RECOVERY = "ERROR_RECOVERY"


S = TurnState

TRANSITIONS: dict[str, dict[TurnState, TurnState]] = {
    "wake": {
        S.LISTENING_FOR_WAKE: S.AWAKE,
        S.COOLDOWN: S.AWAKE,
        S.AWAKE: S.AWAKE,
    },
    "sleep": {
        s: S.LISTENING_FOR_WAKE
        for s in (S.AWAKE, S.COOLDOWN, S.USER_SPEAKING, S.PROCESSING,
                  S.BUDDY_SPEAKING, S.INTERRUPTED, S.ERROR_RECOVERY,
                  S.LISTENING_FOR_WAKE)
    },
    "mute": {s: S.SLEEPING for s in S},
    "unmute": {S.SLEEPING: S.LISTENING_FOR_WAKE},
    "user_started": {
        S.AWAKE: S.USER_SPEAKING,
        S.COOLDOWN: S.USER_SPEAKING,
        S.INTERRUPTED: S.USER_SPEAKING,
        S.LISTENING_FOR_WAKE: S.USER_SPEAKING,
    },
    "user_ended": {S.USER_SPEAKING: S.PROCESSING},
    "user_ended_empty": {S.USER_SPEAKING: S.AWAKE},
    "wake_utterance_ended": {S.USER_SPEAKING: S.LISTENING_FOR_WAKE},
    "buddy_started": {
        S.PROCESSING: S.BUDDY_SPEAKING,
        S.AWAKE: S.BUDDY_SPEAKING,
        S.LISTENING_FOR_WAKE: S.BUDDY_SPEAKING,
        S.COOLDOWN: S.BUDDY_SPEAKING,
    },
    "buddy_finished": {S.BUDDY_SPEAKING: S.COOLDOWN},
    "nothing_to_say": {S.PROCESSING: S.AWAKE},
    "barge_in": {S.BUDDY_SPEAKING: S.INTERRUPTED},
    "error": {s: S.ERROR_RECOVERY for s in S if s is not S.SLEEPING},
}


@dataclass(frozen=True)
class TurnConfig:
    """Timing for the state machine."""

    awake_timeout_s: float = 60.0
    cooldown_s: float = 0.4
    processing_timeout_s: float = 10.0
    error_recovery_s: float = 1.5
    user_speaking_timeout_s: float = 25.0
    interrupt_timeout_s: float = 1.0


class TurnManager:
    """Conversation state machine."""

    def __init__(
        self,
        config: TurnConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Callable[[TurnState, TurnState, str], None]
        | None = None,
    ) -> None:
        self.config = config or TurnConfig()
        self._clock = clock
        self.on_transition = on_transition
        self.state = TurnState.LISTENING_FOR_WAKE
        self._awake = False
        self._entered = clock()
        self._last_activity = clock()
        self.illegal_events = 0
        self.history: list[tuple[str, TurnState]] = []

    # ----- queries -----

    @property
    def ear_open(self) -> bool:
        """True if microphone audio may be given to the recogniser."""

        return self.state not in (S.SLEEPING, S.BUDDY_SPEAKING)

    @property
    def awake(self) -> bool:
        """True if the user woke Buddy and has not put it back to sleep.

        This is a flag, not a function of the state, because Buddy can be
        speaking (a stop acknowledgement, say) while still asleep, and
        that must not wake it.
        """

        return self._awake

    @property
    def wants_wake_word(self) -> bool:
        """True if only the wake grammar should be active."""

        return not self._awake and self.state is not S.SLEEPING

    @property
    def time_in_state(self) -> float:
        """Seconds since the last transition."""

        return self._clock() - self._entered

    # ----- events -----

    def _go(self, event: str, *, force: TurnState | None = None) -> bool:
        target = force or TRANSITIONS.get(event, {}).get(self.state)

        if target is None:
            self.illegal_events += 1

            return False

        old = self.state
        now = self._clock()

        self._last_activity = now

        self.state = target

        if target is S.AWAKE:
            self._awake = True
        elif target in (S.LISTENING_FOR_WAKE, S.SLEEPING):
            self._awake = False

        self._entered = now
        self.history.append((event, target))

        if len(self.history) > 200:
            del self.history[:100]

        if self.on_transition is not None and old != target:
            self.on_transition(old, target, event)

        return True

    def wake(self) -> bool:
        """Wake word heard."""

        return self._go("wake")

    def sleep(self) -> bool:
        """Return to waiting for the wake word."""

        return self._go("sleep")

    def mute(self) -> bool:
        """Stop listening entirely (privacy)."""

        return self._go("mute")

    def unmute(self) -> bool:
        """Resume listening after a mute."""

        return self._go("unmute")

    def user_started(self) -> bool:
        """VAD says the user began speaking."""

        return self._go("user_started")

    def user_ended(self, has_speech: bool = True) -> bool:
        """VAD says the user stopped."""

        return self._go("user_ended" if has_speech else "user_ended_empty")

    def wake_utterance_ended(self) -> bool:
        """A wake-word attempt ended without waking Buddy."""

        return self._go("wake_utterance_ended")

    def buddy_started(self) -> bool:
        """Buddy began to speak."""

        return self._go("buddy_started")

    def buddy_finished(self) -> bool:
        """Buddy finished (or was cut off)."""

        return self._go("buddy_finished")

    def nothing_to_say(self) -> bool:
        """Processing produced no reply."""

        return self._go("nothing_to_say")

    def barge_in(self) -> bool:
        """The user talked over Buddy."""

        return self._go("barge_in")

    def error(self) -> bool:
        """A component failed."""

        return self._go("error")

    def stop(self) -> bool:
        """Spoken or physical stop: legal everywhere.

        Cancels whatever is in progress and returns to listening.
        """

        if self.state is S.SLEEPING:
            return False

        target = S.AWAKE if self._awake else S.LISTENING_FOR_WAKE

        return self._go("stop", force=target)

    # ----- time -----

    def tick(self) -> None:
        """Apply timeouts."""

        cfg = self.config
        elapsed = self.time_in_state
        state = self.state

        if state is S.COOLDOWN and elapsed >= cfg.cooldown_s:
            self._go(
                "cooldown_done",
                force=S.AWAKE if self._awake else S.LISTENING_FOR_WAKE,
            )
        elif state is S.PROCESSING and elapsed >= cfg.processing_timeout_s:
            self._go("processing_timeout", force=S.ERROR_RECOVERY)
        elif state is S.ERROR_RECOVERY and elapsed >= cfg.error_recovery_s:
            self._go("recovered", force=S.LISTENING_FOR_WAKE)
        elif state is S.INTERRUPTED and elapsed >= cfg.interrupt_timeout_s:
            self._go("interrupt_timeout", force=S.AWAKE)
        elif (
            state is S.USER_SPEAKING
            and elapsed >= cfg.user_speaking_timeout_s
        ):
            self._go("speech_timeout", force=S.PROCESSING)
        elif (
            state is S.AWAKE
            and self._clock() - self._last_activity >= cfg.awake_timeout_s
        ):
            self._go("awake_timeout", force=S.LISTENING_FOR_WAKE)

    def touch(self) -> None:
        """Count something as activity (keeps Buddy awake)."""

        self._last_activity = self._clock()
