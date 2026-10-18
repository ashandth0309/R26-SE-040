"""Layered dialogue: rules first, then intent model, then optional LLM.

Order of authority (earlier layers win):

1. Conversation meta ("repeat that").
2. Optional hooks (emotional support, memory commands) supplied by the
   engine; they may answer first.
3. Deterministic rules (time, date, identity, greetings, thanks, jokes).
4. A trained intent classifier, if one can be loaded.
5. An optional language model through a provider interface.
6. A varied offline fallback.

The LLM can only *talk*.  If it suggests an action, that suggestion is
returned as ``suggested_action`` text for the caller to confirm with the
user; nothing here ever moves the robot.  Movement is authorised only by
``voice_commands.parse_command`` and ``CommandRouter``.
"""

from __future__ import annotations

import datetime
import random
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from buddy_core.voice.variation import VariantPicker

MAX_REPLY_CHARS = 320


@dataclass(frozen=True)
class DialogueReply:
    """An answer and where it came from."""

    text: str
    source: str
    intent: str = ""
    suggested_action: str | None = None
    language: str = "en"


@dataclass
class ConversationContext:
    """Short-term conversation state."""

    turns: deque = field(default_factory=lambda: deque(maxlen=12))
    user_name: str | None = None
    last_reply: str = ""
    last_intent: str = ""

    def add_user(self, text: str) -> None:
        """Record what the user said."""

        self.turns.append(("user", text))

    def add_buddy(self, text: str, intent: str = "") -> None:
        """Record what Buddy said."""

        self.turns.append(("buddy", text))
        self.last_reply = text
        self.last_intent = intent

    def clear(self) -> None:
        """Forget the conversation (not stored memory)."""

        self.turns.clear()
        self.last_reply = ""
        self.last_intent = ""

    def as_messages(self) -> list[dict]:
        """Turns in chat-message form for an LLM."""

        return [
            {"role": "user" if who == "user" else "assistant",
             "content": text}
            for who, text in self.turns
        ]


class IntentClassifier(Protocol):
    """Maps text to an intent label with a confidence."""

    def predict(self, text: str) -> tuple[str, float]:
        """Return (label, confidence); label 'unknown' if unsure."""


class LlmProvider(Protocol):
    """A text generator.  Must honour ``timeout_s`` and raise on failure."""

    def complete(self, messages: list[dict], timeout_s: float) -> str:
        """Return the reply text."""


class MlIntentClassifier:
    """Adapter for the existing ``ml_brain`` TF-IDF model.

    ``ml_brain`` loads its files relative to the working directory, so it
    may be unavailable; that is reported, not raised.
    """

    def __init__(self, threshold: float = 0.30, module=None) -> None:
        self.threshold = threshold
        self._module = module
        self.available = module is not None

        if module is None:
            try:
                import ml_brain

                self._module = ml_brain
                self.available = True
            except Exception:
                self._module = None
                self.available = False

    def predict(self, text: str) -> tuple[str, float]:
        if not self.available:
            return "unknown", 0.0

        try:
            label = self._module.predict_intent(
                text, threshold=self.threshold)
        except Exception:
            return "unknown", 0.0

        return str(label), 1.0 if label != "unknown" else 0.0


def clean_llm_text(raw: str) -> tuple[str, str | None]:
    """Make model output safe to speak; pull out any suggested action."""

    text = raw or ""
    action = None
    match = re.search(r"ACTION:\s*([A-Za-z_ ]{1,40})", text)

    if match:
        action = match.group(1).strip().lower()
        text = text.replace(match.group(0), "")

    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_REPLY_CHARS:
        cut = text[:MAX_REPLY_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[:end + 1] if end > 60 else cut.rstrip() + "."

    return text, action


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


JOKES = (
    "Why did the computer go to the doctor? It had a virus.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "What do you call a bear with no teeth? A gummy bear.",
    "Why did the robot go on holiday? It needed to recharge.",
    "I tried to catch fog yesterday. I mist.",
)

_RULES: list[tuple[str, frozenset]] = [
    ("repeat", frozenset({
        "repeat that", "say that again", "what did you say",
        "can you repeat that", "come again", "pardon",
    })),
    ("greeting", frozenset({
        "hello", "hi", "hey", "hello buddy", "hi buddy", "hey buddy",
        "good morning", "good afternoon", "good evening",
    })),
    ("how_are_you", frozenset({
        "how are you", "how are you doing", "how is it going",
        "how's it going", "are you okay", "how are you buddy",
    })),
    ("thanks", frozenset({
        "thank you", "thanks", "thanks buddy", "thank you buddy",
        "cheers",
    })),
    ("sorry", frozenset({"sorry", "i am sorry", "i'm sorry", "my bad"})),
    ("goodbye", frozenset({
        "bye", "goodbye", "see you", "see you later", "good night",
    })),
    ("time", frozenset({
        "what time is it", "what is the time", "tell me the time",
        "current time", "what's the time",
    })),
    ("date", frozenset({
        "what date is it", "what is the date", "what day is it",
        "what day is today", "what's the date", "tell me the date",
    })),
    ("identity", frozenset({
        "what is your name", "what's your name", "who are you",
        "tell me your name",
    })),
    ("capabilities", frozenset({
        "what can you do", "what are your capabilities",
        "tell me what you can do", "help",
    })),
    ("creator", frozenset({
        "who made you", "who created you", "who built you",
    })),
    ("joke", frozenset({
        "tell me a joke", "say a joke", "make me laugh", "joke",
        "tell me another joke", "another joke",
    })),
]

_VARIANTS: dict[str, tuple[str, ...]] = {
    "greeting": ("Hey! Good to hear you.", "Hi there.", "Hello!"),
    "greeting_named": ("Hey {name}! Good to hear you.", "Hi {name}.",
                       "Hello {name}!"),
    "how_are_you": (
        "I'm doing well. How about you?",
        "Pretty good, thanks for asking. How are you?",
        "All good here. How is your day going?",
    ),
    "thanks": ("You're welcome!", "Anytime.", "Happy to help."),
    "sorry": ("No worries.", "That's okay.", "Don't worry about it."),
    "goodbye": ("See you later!", "Bye for now.", "Take care."),
    "identity": ("I'm Buddy, your robot companion.",
                 "My name is Buddy. I'm your robot dog."),
    "capabilities": (
        "I can chat, answer simple questions, tell the time and date, "
        "and follow a few movement commands.",
        "Mostly I keep you company and follow simple voice commands "
        "like going forward or turning, always through my safety system.",
    ),
    "creator": ("I was built as part of the Buddy robot project.",),
    "fallback": (
        "I'm not sure I got that. Could you say it another way?",
        "Sorry, I missed that. Can you try again?",
        "Hmm, I didn't quite follow. What do you mean?",
    ),
    "listening": (
        "I'm listening. Tell me more.",
        "Go on, I'm here.",
        "Mm. What happened next?",
    ),
}


class DialogueManager:
    """Produce a reply for what the user said."""

    def __init__(
        self,
        *,
        classifier: IntentClassifier | None = None,
        llm: LlmProvider | None = None,
        llm_timeout_s: float = 6.0,
        llm_allowed: Callable[[], bool] = lambda: True,
        hooks: list[Callable] | None = None,
        rng: random.Random | None = None,
        now: Callable[[], datetime.datetime] = datetime.datetime.now,
        min_intent_confidence: float = 0.5,
    ) -> None:
        self.classifier = classifier
        self.llm = llm
        self.llm_timeout_s = llm_timeout_s
        self.llm_allowed = llm_allowed
        self.hooks = list(hooks or [])
        self._picker = VariantPicker(rng)
        self._rng = rng or random.Random()
        self._now = now
        self.min_intent_confidence = min_intent_confidence
        self.llm_failures = 0

    def _say(self, key: str, **fmt) -> str:
        return self._picker.pick(key, _VARIANTS[key]).format(**fmt)

    def respond(self, text: str, context: ConversationContext
                ) -> DialogueReply:
        """Answer ``text``.  Always returns something speakable."""

        text = text.strip()
        context.add_user(text)
        reply = self._decide(text, context)
        context.add_buddy(reply.text, reply.intent)

        return reply

    def _decide(self, text: str, context: ConversationContext
                ) -> DialogueReply:
        norm = _norm(text)

        if not norm:
            return DialogueReply(self._say("fallback"), "fallback")

        for hook in self.hooks:
            result = hook(text, context)

            if result is not None:
                return result

        rule = self._rule(norm, context)

        if rule is not None:
            return rule

        intent = self._intent(text, context)

        if intent is not None:
            return intent

        llm = self._ask_llm(context)

        if llm is not None:
            return llm

        words = norm.split()

        if len(words) >= 6:
            return DialogueReply(self._say("listening"), "fallback",
                                 "share")

        return DialogueReply(self._say("fallback"), "fallback")

    # ----- layers -----

    def _rule(self, norm: str, context: ConversationContext
              ) -> DialogueReply | None:
        for name, phrases in _RULES:
            if norm in phrases:
                return self._rule_reply(name, context)

        return None

    def _rule_reply(self, name: str, context: ConversationContext
                    ) -> DialogueReply:
        if name == "repeat":
            if context.last_reply:
                return DialogueReply(context.last_reply, "rule", name)

            return DialogueReply("I haven't said anything yet.", "rule",
                                 name)

        if name == "greeting":
            if context.user_name:
                text = self._say("greeting_named", name=context.user_name)
            else:
                text = self._say("greeting")

            return DialogueReply(text, "rule", name)

        if name == "time":
            now = self._now()

            return DialogueReply(
                f"It's {now.strftime('%I:%M %p').lstrip('0')}.", "rule",
                name)

        if name == "date":
            now = self._now()

            return DialogueReply(
                now.strftime("Today is %A, %B %d, %Y."), "rule", name)

        if name == "joke":
            return DialogueReply(self._picker.pick("joke", JOKES), "rule",
                                 name)

        return DialogueReply(self._say(name), "rule", name)

    def _intent(self, text: str, context: ConversationContext
                ) -> DialogueReply | None:
        if self.classifier is None:
            return None

        label, conf = self.classifier.predict(text)

        if label == "unknown" or conf < self.min_intent_confidence:
            return None

        mapping = {
            "greeting": "greeting", "farewell": "goodbye",
            "time": "time", "date": "date", "joke": "joke",
            "ask_wellbeing": "how_are_you", "thanks": "thanks",
            "apology": "sorry", "ask_name_bot": "identity",
            "ask_capabilities": "capabilities", "ask_creator": "creator",
        }
        rule = mapping.get(label)

        if rule is None:
            return None

        reply = self._rule_reply(rule, context)

        return DialogueReply(reply.text, "intent", label)

    def _ask_llm(self, context: ConversationContext
                 ) -> DialogueReply | None:
        if self.llm is None or not self.llm_allowed():
            return None

        try:
            raw = self.llm.complete(context.as_messages(),
                                    self.llm_timeout_s)
        except Exception:
            self.llm_failures += 1

            return None

        text, action = clean_llm_text(raw)

        if not text:
            return None

        return DialogueReply(text, "llm", "llm", suggested_action=action)
