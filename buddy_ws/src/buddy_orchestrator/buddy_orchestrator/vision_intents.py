"""Spoken requests that belong to the vision module.

The voice engine handles motion, stop and sleep itself.  Everything else it
hears reaches the coordinator, which uses this module to find the requests
that need the camera: "what is this?", "this is my red pen", "where did you
last see my book?", "who is this?", "come here".

Rules, same spirit as the voice command parser:

* The whole utterance must match after polite filler words are removed.
  There is no substring matching, so a sentence that merely contains the
  words is conversation, not a command.
* Negated, conditional or past-tense speech is never a command.
* "This is my <person word>" is an introduction, not an object lesson.

Nothing here talks to a device.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    IDENTIFY_OBJECT = "identify_object"
    TEACH_OBJECT = "teach_object"
    WHERE_IS = "where_is"
    WHO_IS_THIS = "who_is_this"
    CHECK_FLOOR = "check_floor"
    PERSON_INTRO = "person_intro"
    COME_HERE = "come_here"


@dataclass(frozen=True)
class VisionIntent:
    kind: Intent
    text: str
    name: str = ""          # object name for TEACH_OBJECT / WHERE_IS


FILLER = frozenset({
    "buddy", "hey", "please", "okay", "ok", "now", "just", "kindly",
    "so", "well", "um", "uh",
})

# "can you tell me ..." in front of a request.
_LEAD = r"^(?:(?:can|could|would|will) you )?(?:tell me )?"

# Anything that turns a request into talk about a request.
NEGATIONS = frozenset({
    "don't", "dont", "not", "never", "no", "didn't", "didnt", "if",
    "when", "whenever", "unless", "should", "shouldn't", "wasn't",
    "isn't", "wouldn't", "won't", "cannot", "can't", "cant", "used",
    "maybe", "might", "wonder", "wondering", "thinking", "think",
})

PERSON_WORDS = frozenset({
    "friend", "brother", "sister", "mom", "mum", "mother", "dad",
    "father", "wife", "husband", "son", "daughter", "colleague",
    "teacher", "boss", "neighbour", "neighbor", "cousin", "uncle",
    "aunt", "grandma", "grandmother", "grandpa", "grandfather",
    "partner", "roommate", "classmate", "guest", "visitor", "person",
    "girlfriend", "boyfriend", "baby", "kid", "child", "buddy's",
})

_WORD = re.compile(r"[a-z']+")
_NAME = r"(?P<name>[a-z][a-z'-]*(?: [a-z][a-z'-]*){0,2})"
_TIME_WORDS = frozenset({
    "yesterday", "today", "tonight", "tomorrow", "morning", "afternoon",
    "evening", "night", "earlier", "ago", "week", "month", "year", "last",
})

_PATTERNS: list[tuple[Intent, re.Pattern]] = [
    (Intent.IDENTIFY_OBJECT, re.compile(
        _LEAD + r"(?:what(?: is|'s| s) (?:this|that|it)(?: object| thing| "
        r"item)?|what (?:this|that) is"
        r"|what am i (?:holding|showing you)"
        r"|what(?: is|'s) (?:this|that) (?:called|in my hand)"
        r"|do (?:you )?(?:know|recognize|recognise|remember) "
        r"(?:what )?(?:this|that)(?: is)?"
        r"|do (?:you )?(?:know|recognize|recognise|remember) "
        r"(?:this|that)(?: object| item| thing)?)$")),
    (Intent.WHO_IS_THIS, re.compile(
        _LEAD + r"(?:who(?: is|'s) (?:this|that|there|here)"
        r"|do (?:you )?(?:know|recognize|recognise) "
        r"(?:who this is|this person|me|who i am)"
        r"|who am i)$")),
    (Intent.CHECK_FLOOR, re.compile(
        _LEAD + r"(?:is|are) (?:someone|somebody|anyone|anybody|there anyone|"
        r"a person|any person|there someone) "
        r"(?:lying|laying) (?:down )?(?:on the floor|on the ground|down)$"
        r"|" + _LEAD + r"(?:is|are) (?:someone|somebody|anyone|anybody) "
        r"on the (?:floor|ground)$"
        r"|^is there (?:someone|somebody|anyone|a person) "
        r"(?:lying|laying|on the floor|on the ground)"
        r"(?: down| on the floor| on the ground)?$")),
    (Intent.WHERE_IS, re.compile(
        _LEAD + r"where (?:did you (?:last )?see|did i (?:leave|put)|is|are|"
        r"was|were) (?:my|the) " + _NAME + r"(?: last)?$")),
    (Intent.COME_HERE, re.compile(
        _LEAD + r"(?:come (?:here|over here|to me|over)|approach me"
        r"|come closer)$")),
    (Intent.TEACH_OBJECT, re.compile(
        _LEAD + r"(?:this|that) is (?:my|a|an|the) " + _NAME + r"$"
        r"|" + _LEAD + r"(?:remember|learn|memorize|memorise) (?:this|that)"
        r"(?: object| item| thing)?(?: as (?:my |a |an |the )?"
        + _NAME.replace("?P<name>", "?P<alt>") + r")?$")),
]


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("’", "'"))


def _clean(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FILLER]


def parse_vision_intent(text: str) -> VisionIntent | None:
    """Return the vision request in ``text``, or None."""

    raw = _tokens(text)

    if not raw or any(t in NEGATIONS for t in raw):
        return None

    sentence = " ".join(_clean(raw))

    if not sentence or len(sentence) > 80:
        return None

    for kind, pattern in _PATTERNS:
        match = pattern.match(sentence)

        if not match:
            continue

        groups = match.groupdict()
        name = (groups.get("name") or groups.get("alt") or "").strip()

        if _TIME_WORDS.intersection(name.split()):
            return None

        if kind is Intent.TEACH_OBJECT:
            words = name.split()

            # "my friend Sam", "my mum": a person, not an object to teach.
            # Erring this way is the safe direction: no object is saved
            # and Buddy asks the person themselves.
            if any(w in PERSON_WORDS for w in words):
                return VisionIntent(Intent.PERSON_INTRO, sentence, name)

        if kind in (Intent.WHERE_IS, Intent.TEACH_OBJECT) and name:
            name = re.sub(r"^(?:my|a|an|the) ", "", name)

        return VisionIntent(kind, sentence, name)

    return None
