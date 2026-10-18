"""Spoken controls for memory, privacy and silence.

Everything here is plain-language and confirmable:

* Saving anything needs a spoken "yes".  Deleting everything needs one too.
* Online mode is OFF by default and needs an explicit, confirmed request.
* "mute" stops the microphone and the robot.  Only the button or the app
  can unmute, because a muted microphone cannot hear "unmute".
* "be quiet" and "be silent" change how much Buddy talks; they never
  change what safety acknowledgements are spoken.

Actions that would cut off the spoken confirmation (mute, silent mode) run
a couple of seconds later so the user hears what happened.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from buddy_core.voice.dialogue_manager import (
    ConversationContext,
    DialogueReply,
)
from buddy_core.voice.memory_store import CompanionMemoryManager


class SilenceMode(str, Enum):
    """How much Buddy talks."""

    NORMAL = "normal"
    QUIET = "quiet"     # no backchannels or proactive talk
    SILENT = "silent"   # only command and safety acknowledgements


NAME_BLOCK = frozenset({
    "later", "back", "maybe", "when", "if", "now", "tomorrow", "please",
    "anything", "whatever", "crazy", "sir", "tired", "fine", "good",
})
_NAME = re.compile(
    r"\b(?:my name is|call me|i am called|i'm called|my name's)\s+"
    r"([a-z][a-z'\-]{1,24}(?:\s+[a-z][a-z'\-]{1,24})?)\s*$"
)
_YES = frozenset({"yes", "yeah", "yep", "sure", "okay", "ok", "please do",
                  "yes please", "go ahead", "do it", "i am sure"})
_NO = frozenset({"no", "nope", "no thanks", "not now", "don't", "cancel",
                 "never mind", "no thank you"})
_MINUTES = re.compile(r"for (\d{1,3}) minutes?")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


@dataclass
class _Pending:
    kind: str
    payload: str = ""
    expires: float = 0.0


@dataclass
class CompanionControls:
    """Dialogue hook for memory, privacy and silence commands."""

    memory: CompanionMemoryManager
    set_muted: Callable[[bool], None] = lambda muted: None
    set_silence: Callable[[SilenceMode], None] = lambda mode: None
    set_cloud: Callable[[bool], None] = lambda enabled: None
    cloud_available: Callable[[], bool] = lambda: False
    cloud_enabled: Callable[[], bool] = lambda: False
    export_dir: Path = Path("buddy_exports")
    clock: Callable[[], float] = time.monotonic
    confirm_window_s: float = 30.0
    action_delay_s: float = 2.5
    silence_mode: SilenceMode = SilenceMode.NORMAL
    _pending: _Pending | None = None
    _deferred: list = field(default_factory=list)
    _silence_until: float | None = None

    # ----- time -----

    def tick(self) -> None:
        """Run due deferred actions and expire timed silence."""

        now = self.clock()
        due = [d for d in self._deferred if d[0] <= now]
        self._deferred = [d for d in self._deferred if d[0] > now]

        for _, action in due:
            action()

        if self._silence_until is not None and now >= self._silence_until:
            self._silence_until = None
            self._apply_silence(SilenceMode.NORMAL)

    def _later(self, action: Callable[[], None]) -> None:
        self._deferred.append((self.clock() + self.action_delay_s, action))

    def _apply_silence(self, mode: SilenceMode) -> None:
        self.silence_mode = mode
        self.set_silence(mode)

    # ----- hook -----

    def hook(self, text: str, context: ConversationContext
             ) -> DialogueReply | None:
        """Handle control phrases; return None for ordinary talk."""

        norm = _norm(text)
        reply = self._pending_answer(norm, context)

        if reply is not None:
            return reply

        for handler in (self._silence, self._privacy, self._memory):
            reply = handler(norm, text, context)

            if reply is not None:
                return reply

        return None

    def _say(self, text: str, intent: str) -> DialogueReply:
        return DialogueReply(text, "control", intent)

    # ----- confirmations -----

    def _ask(self, kind: str, payload: str, question: str, intent: str
             ) -> DialogueReply:
        self._pending = _Pending(kind, payload,
                                 self.clock() + self.confirm_window_s)

        return self._say(question, intent)

    def _pending_answer(self, norm: str, context: ConversationContext
                        ) -> DialogueReply | None:
        pending, self._pending = self._pending, None

        if pending is None or self.clock() > pending.expires:
            return None

        if norm in _NO:
            return self._say("Okay, I won't.", "declined")

        if norm not in _YES:
            return None

        if pending.kind == "memory_consent":
            self.memory.grant_consent()
            key, _, value = pending.payload.partition("=")
            result = self.memory.remember(key, value,
                                          "name" if key == "name"
                                          else "note")

            if result.ok:
                return self._say("Okay, I'll remember that.", "saved")

            return self._say("I could not save that.", "save_failed")

        if pending.kind == "forget_all":
            count = self.memory.forget_all()
            context.user_name = None

            return self._say(
                f"Done. I deleted everything I had saved, "
                f"{count} items.", "forgot_all")

        if pending.kind == "cloud":
            self.set_cloud(True)

            return self._say(
                "Online mode is on. What you say may be sent to an "
                "online service. Say 'go offline' to turn it off.",
                "cloud_on")

        return None

    # ----- silence -----

    def _silence(self, norm: str, raw: str, context: ConversationContext
                 ) -> DialogueReply | None:
        minutes = _MINUTES.search(norm)
        until = (self.clock() + int(minutes.group(1)) * 60
                 if minutes else None)

        if re.match(r"^(?:be quiet|quiet mode|shush|hush)\b", norm):
            self._silence_until = until
            self._apply_silence(SilenceMode.QUIET)

            return self._say("Okay, I'll keep quiet unless you ask me "
                             "something.", "quiet")

        if re.match(r"^(?:be silent|silent mode|don't talk|do not talk)",
                    norm):
            self._silence_until = until
            self._later(lambda: self._apply_silence(SilenceMode.SILENT))

            return self._say("Okay, going silent. I'll still confirm "
                             "commands.", "silent")

        if norm in ("normal mode", "you can talk now",
                    "you can talk again", "talk normally",
                    "you can speak now"):
            self._silence_until = None
            self._apply_silence(SilenceMode.NORMAL)

            return self._say("Okay, I'm back to normal.", "normal")

        return None

    # ----- privacy -----

    def _privacy(self, norm: str, raw: str, context: ConversationContext
                 ) -> DialogueReply | None:
        if norm in ("mute", "mute yourself", "privacy mode",
                    "turn off your microphone"):
            self._later(lambda: self.set_muted(True))

            return self._say(
                "Okay, I'm muting my microphone. I'll stop moving too. "
                "Use the button or the app to wake me.", "mute")

        if norm in ("go offline", "work offline", "disable cloud",
                    "turn off online mode", "stay offline"):
            self.set_cloud(False)

            return self._say("Okay, I'm staying offline. Nothing leaves "
                             "this robot.", "cloud_off")

        if norm in ("go online", "enable online mode", "enable cloud",
                    "turn on online mode"):
            if not self.cloud_available():
                return self._say("Online mode isn't set up on this robot.",
                                 "cloud_unavailable")

            return self._ask(
                "cloud", "",
                "Online mode sends what you say to an online service. "
                "Say yes to turn it on.", "cloud_confirm")

        if norm in ("are you recording", "are you recording me",
                    "do you record me", "are you listening"):
            online = self.cloud_enabled()

            return self._say(
                "I only keep audio in memory while you are speaking, and "
                "I don't save recordings. "
                + ("Online mode is on, so your words may be sent to an "
                   "online service." if online else
                   "I'm offline, so nothing leaves this robot."),
                "recording")

        return None

    # ----- memory -----

    def _memory(self, norm: str, raw: str, context: ConversationContext
                ) -> DialogueReply | None:
        mem = self.memory
        name = _NAME.search(raw.lower().strip(" .!"))

        if name:
            return self._learn_name(name.group(1), context)

        if norm in ("what is my name", "what's my name",
                    "do you know my name", "remember my name", "who am i"):
            known = mem.recall("name")

            if known:
                return self._say(f"Your name is {known.title()}.", "name")

            return self._say("You haven't told me your name yet.", "name")

        if norm in ("what do you remember", "what do you remember about me",
                    "what do you know about me"):
            return self._say(mem.describe(), "describe")

        if norm in ("forget my name",):
            mem.forget("name")
            context.user_name = None

            return self._say("Okay, I've forgotten your name.", "forgot")

        if norm in ("forget everything", "delete everything",
                    "delete my data", "delete everything you know about me",
                    "forget everything about me"):
            return self._ask(
                "forget_all", "",
                "Are you sure? Say yes and I'll delete everything I've "
                "saved about you.", "forget_confirm")

        if norm in ("stop remembering", "don't remember anything",
                    "turn off memory", "stop remembering things"):
            mem.revoke_consent()

            return self._say(
                "Okay, I won't save anything new. Say 'forget "
                "everything' if you want me to delete what I have.",
                "revoked")

        if norm in ("you can remember things", "you can remember me",
                    "you can remember things about me", "remember me"):
            mem.grant_consent()

            return self._say("Thanks. I'll remember what you ask me to, "
                             "and you can make me forget any time.",
                             "consent")

        if norm in ("export my data", "export what you remember"):
            path = mem.export_to(self.export_dir / "buddy_memory.json")

            return self._say(
                f"I saved a copy of everything I hold about you as "
                f"{path.name}.", "export")

        note = re.match(r"^(?:please )?remember that (.+)$", norm)

        if note:
            return self._remember_note(note.group(1))

        return None

    def _learn_name(self, name: str, context: ConversationContext
                    ) -> DialogueReply:
        words = name.split()

        if any(w in NAME_BLOCK for w in words):
            return self._say("Sorry, I didn't catch your name. Could you "
                             "say it again?", "name_unclear")

        name = " ".join(w.capitalize() for w in words)
        context.user_name = name
        result = self.memory.remember("name", name, "name")

        if result.ok:
            return self._say(f"Nice to meet you, {name}. I'll remember "
                             f"that.", "name_saved")

        if result.reason == "no consent":
            return self._ask(
                "memory_consent", f"name={name}",
                f"Nice to meet you, {name}. Want me to remember that for "
                f"next time? Say yes or no.", "consent_ask")

        return self._say(f"Nice to meet you, {name}.", "name_session")

    def _remember_note(self, text: str) -> DialogueReply:
        result = self.memory.add_note(text)

        if result.ok:
            return self._say("Okay, I'll remember that.", "saved")

        if result.reason == "sensitive":
            return self._say(
                "I can't keep passwords, card numbers or similar, "
                "sorry.", "refused_sensitive")

        if result.reason == "no consent":
            return self._ask(
                "memory_consent", f"note_{int(time.time())}={text}",
                "Do you want me to save that for next time? Say yes or "
                "no.", "consent_ask")

        if result.reason in ("memory full", "too many notes"):
            return self._say("My memory is full. You could ask me to "
                             "forget something first.", "full")

        return self._say("I couldn't save that.", "save_failed")


def bind_engine(controls: CompanionControls, engine) -> None:
    """Connect the controls to a ``VoiceEngine``."""

    from buddy_core.voice.speaker_manager import Priority

    def apply(mode: SilenceMode) -> None:
        engine.set_quiet(mode is not SilenceMode.NORMAL)
        engine.speaker.set_floor(
            Priority.HIGH if mode is SilenceMode.SILENT else Priority.LOW)

    controls.set_muted = engine.set_muted
    controls.set_silence = apply
    controls.clock = engine._clock
    engine.tick_hooks.append(controls.tick)
