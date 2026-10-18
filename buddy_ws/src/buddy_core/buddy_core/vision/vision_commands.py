"""Spoken settings and privacy requests for the vision subsystem.

Examples it understands: "don't greet me", "you can check on me",
"stop reading my expression", "what do you know about me", "forget my
face", "turn off the camera", "remember where things are".

It only edits vision settings and the people the user consented to store.
It does not open the camera, move the robot or touch the voice engine;
the voice side may call ``handle`` and speak the returned text.

Rules:
* Choices about *me* apply to the person Buddy currently recognises.  If
  nobody is recognised, Buddy says so instead of guessing.
* Things that remove protection or data ask for a yes first (forgetting a
  face, switching off fall detection).  Switching things off never needs
  confirmation; switching on needs only the clear request.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from buddy_core.vision.vision_settings import (
    PROFILES, SettingsError, VisionSettings)


@dataclass(frozen=True)
class Command:
    kind: str
    value: object = None
    scope: str = "person"          # person | global


_ME = r"(?:me|my|myself|i)"

RULES: list[tuple[str, str, object, str]] = [
    # (regex, kind, value, scope)
    (r"\b(what (?:do|did) you (?:know|have|store|remember) about me|"
     r"what data (?:do you|have you) (?:have|stored?)(?: on| about) me|"
     r"show me my data)\b", "about_me", None, "person"),
    (r"\b(forget|delete|erase|remove) (?:me|my face|my face data|"
     r"everything about me|my profile|who i am)\b", "forget_me", None,
     "person"),
    (r"\b(stop|don'?t|do not|no more) (?:greet(?:ing)? me|saying hi to me|"
     r"say(?:ing)? hello to me)\b", "greet", False, "person"),
    (r"\b(?:you can |please |do )?greet me\b(?! with)", "greet", True,
     "person"),
    (r"\b(stop|don'?t|do not|no more) (?:check(?:ing)? (?:in )?on me|"
     r"check(?:ing)? in with me|asking if i'?m ok(?:ay)?)\b", "proactive",
     False, "person"),
    (r"\b(?:you can|please|feel free to|it'?s ok(?:ay)? to) "
     r"check(?:ing)? (?:in )?on me\b", "proactive", True, "person"),
    (r"\b(stop|don'?t|do not|no more) (?:read(?:ing)?|look(?:ing)? at|"
     r"watch(?:ing)?|analy[sz]ing) my (?:face|expression|facial expression)"
     r"\b", "expression", False, "person"),
    (r"\b(?:you can|please|it'?s ok(?:ay)? to) (?:read|look at|watch|"
     r"analy[sz]e) my (?:face|expression|facial expression)\b",
     "expression", True, "person"),
    (r"\b(stop|don'?t|do not) (?:recogni[sz](?:e|ing)|identify(?:ing)?) "
     r"me\b", "recognition", False, "person"),
    (r"\b(?:you can|please|can you) recogni[sz]e me\b", "recognition",
     True, "person"),
    (r"\b(?:i want|i'?d like|give me) (?:some )?(?:quiet|peace|space)\b",
     "quiet", True, "person"),
    (r"\b(?:you can talk|talk to me) (?:more|again)\b", "quiet", False,
     "person"),
    (r"\bgreet me with (.+)$", "greeting_text", None, "person"),
    (r"\b(?:limit|at most|only) (?:check(?:ing)?[- ]?ins? (?:to )?)?"
     r"(\d|one|two|three|four|five|zero|none) (?:times )?(?:check[- ]?ins? )?"
     r"(?:a|per) day\b", "max_checkins", None, "person"),
    (r"\b(turn off|switch off|close|stop using|disable) (?:the |your )?"
     r"(?:camera|eyes|vision)\b|\bstop looking\b|\bclose your eyes\b",
     "camera", False, "global"),
    (r"\b(turn on|switch on|open|start using|enable) (?:the |your )?"
     r"(?:camera|eyes|vision)\b|\bopen your eyes\b", "camera", True,
     "global"),
    (r"\b(stop|don'?t|do not|turn off|disable) (?:recogni[sz]ing|"
     r"identifying) (?:faces|people|anyone)\b", "global_recognition",
     False, "global"),
    (r"\b(turn on|start|enable) (?:recogni[sz]ing|identifying) "
     r"(?:faces|people)\b", "global_recognition", True, "global"),
    (r"\b(?:remember|keep track of) where (?:things|my things|objects) "
     r"(?:are|were)\b", "object_memory", True, "global"),
    (r"\b(?:stop|don'?t|no more) (?:remembering|keeping track of) where "
     r"(?:things|my things|objects) (?:are|were)\b|\bforget where "
     r"(?:things|my things|objects) (?:are|were)\b", "object_memory", False,
     "global"),
    (r"\b(turn off|disable|stop) (?:the )?fall (?:detection|checking)\b",
     "fall_detection", False, "global"),
    (r"\b(turn on|enable|start) (?:the )?fall (?:detection|checking)\b",
     "fall_detection", True, "global"),
    (r"\b(lightweight|balanced|enhanced) (?:mode|profile)\b|"
     r"\bswitch to (lightweight|balanced|enhanced)\b", "profile", None,
     "global"),
]
_NUM = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5}


def parse_command(text: str) -> Command | None:
    t = (text or "").lower().strip()

    if not t:
        return None

    for pat, kind, value, scope in RULES:
        m = re.search(pat, t)

        if not m:
            continue

        if kind == "greeting_text":
            g = re.sub(r"https?://\S+|[<>{}\[\]]", "", m.group(1)).strip(
                " .!?\"'")

            if not g or len(g) > 40:
                return Command("greeting_text_bad", None, scope)

            return Command(kind, g, scope)

        if kind == "max_checkins":
            n = m.group(1)

            return Command(kind, int(n) if n.isdigit() else _NUM[n], scope)

        if kind == "profile":
            return Command(kind, next(g for g in m.groups() if g), scope)

        return Command(kind, value, scope)

    return None


YES = re.compile(r"\b(yes|yeah|yep|sure|confirm|do it|go ahead)\b")
NO = re.compile(r"\b(no|nope|cancel|never mind|don'?t|stop)\b")


class VisionController:
    def __init__(self, settings: VisionSettings,
                 apply_settings: Callable[[VisionSettings], None],
                 store=None, memory=None, proactive=None, greeting=None,
                 confirm_timeout_s: float = 20.0) -> None:
        self.settings = settings
        self._apply = apply_settings
        self.store, self.memory = store, memory
        self.proactive, self.greeting = proactive, greeting
        self.timeout = confirm_timeout_s
        self._pending: tuple[Command, str | None, float] | None = None

    @property
    def awaiting_confirmation(self) -> bool:
        return self._pending is not None

    def _set(self, **kw) -> None:
        new = replace(self.settings, **kw)
        new.validate()
        self._apply(new)
        self.settings = new

    def handle(self, text: str, me_person_id: str | None, now: float
               ) -> str | None:
        """Return the reply to speak, or None if this is not for us."""

        if self._pending is not None:
            cmd, pid, t0 = self._pending

            if now - t0 > self.timeout:
                self._pending = None
            elif YES.search((text or "").lower()) and not NO.search(
                    (text or "").lower()):
                self._pending = None

                return self._execute(cmd, pid, now)
            elif NO.search((text or "").lower()):
                self._pending = None

                return "Okay, I won't do that."

        cmd = parse_command(text)

        if cmd is None:
            return None

        if cmd.scope == "person" and cmd.kind != "greeting_text_bad":
            if me_person_id is None or self.store is None or \
                    self.store.get_person(me_person_id) is None:
                return ("I don't recognise you right now, so I can't "
                        "change settings for you. Please show me your face "
                        "and try again.")

        if cmd.kind == "greeting_text_bad":
            return "Please give me a short greeting, under forty letters."

        if cmd.kind == "forget_me":
            self._pending = (cmd, me_person_id, now)

            return ("This will delete your face data and your settings "
                    "from me for good. Should I go ahead?")

        if cmd.kind == "fall_detection" and cmd.value is False:
            self._pending = (cmd, None, now)

            return ("If I stop checking for falls I can't notice if you "
                    "fall. Are you sure?")

        return self._execute(cmd, me_person_id, now)

    # ----- actions -----

    def _execute(self, c: Command, pid: str | None, now: float) -> str:
        k, v = c.kind, c.value

        try:
            if k == "about_me":
                return self._about(pid)

            if k == "forget_me":
                name = self.store.get_person(pid)["display_name"]
                self.store.delete_person(pid)

                if self.greeting is not None:
                    self.greeting.forget(pid)

                if self.proactive is not None:
                    self.proactive.reset_person(pid)

                return (f"Done, {name}. I deleted your face data and "
                        "settings.")

            if k in ("greet", "proactive", "expression", "recognition",
                     "quiet"):
                self.store.set_setting(pid, k, bool(v))

                if k == "proactive" and v:
                    self.store.set_setting(pid, "expression", True)

                return {
                    ("greet", True): "Okay, I'll greet you.",
                    ("greet", False): "Okay, I won't greet you.",
                    ("proactive", True): "Okay, I may check in on you "
                    "now and then, and you can tell me to stop.",
                    ("proactive", False): "Okay, I won't check in on you.",
                    ("expression", True): "Okay, I may look at your "
                    "expression. It's only a rough guess.",
                    ("expression", False): "Okay, I'll stop looking at "
                    "your expression.",
                    ("recognition", True): "Okay, I'll recognise you.",
                    ("recognition", False): "Okay, I won't try to "
                    "recognise you.",
                    ("quiet", True): "Okay, I'll keep quiet.",
                    ("quiet", False): "Okay, I'll chat again.",
                }[(k, bool(v))]

            if k == "greeting_text":
                self.store.set_setting(pid, "preferred_greeting", v)

                return f"Okay, I'll greet you with {v}."

            if k == "max_checkins":
                n = max(0, min(5, int(v)))
                self.store.set_setting(pid, "max_checkins_per_day", n)

                return ("Okay, no check-ins." if n == 0 else
                        f"Okay, at most {n} check-in"
                        f"{'s' if n != 1 else ''} a day.")

            if k == "camera":
                self._set(camera_enabled=bool(v))

                return ("Okay, my camera is on." if v else
                        "Okay, my camera is off. I can't see until you "
                        "turn it on.")

            if k == "global_recognition":
                self._set(recognition=bool(v))

                return ("Okay, I'll recognise people who agreed to it."
                        if v else "Okay, I'll stop recognising faces.")

            if k == "object_memory":
                self._set(object_memory=bool(v))

                if self.memory is not None:
                    self.memory.enable() if v else self.memory.disable(True)

                return ("Okay, I'll remember where I last see things."
                        if v else "Okay, I forgot where things were and "
                        "I won't keep track.")

            if k == "fall_detection":
                self._set(fall_detection=bool(v))

                return ("Okay, I'll watch for falls again." if v else
                        "Okay, fall checking is off.")

            if k == "profile":
                if v not in PROFILES:
                    return "I don't know that mode."

                self._set(profile=v)

                return f"Okay, {v} mode."
        except SettingsError as exc:
            return f"I couldn't change that ({exc})."

        return "Sorry, I didn't understand that."

    def _about(self, pid: str) -> str:
        p = self.store.get_person(pid)
        n = self.store.template_count(pid)
        st = p["settings"]
        on = [name for name, key in (("greeting", "greet"),
                                     ("check-ins", "proactive"),
                                     ("expression reading", "expression"))
              if st.get(key)]

        return (f"I keep {n} face templates for {p['display_name']}, no "
                "photos. "
                + ("These are on: " + ", ".join(on) + "." if on else
                   "Greeting, check-ins and expression reading are off.")
                + " You can ask me to forget you any time.")
