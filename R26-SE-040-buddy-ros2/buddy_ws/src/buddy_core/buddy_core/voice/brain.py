from __future__ import annotations

import datetime
import json
import random
import re
from pathlib import Path

import ml_brain


MEMORY_PATH = Path.home() / ".buddy_memory.json"


def _load_memory():
    if not MEMORY_PATH.exists():
        return {
            "name": None,
            "conversation_count": 0,
            "last_user_text": "",
            "last_buddy_response": "",
        }

    try:
        return json.loads(
            MEMORY_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {
            "name": None,
            "conversation_count": 0,
            "last_user_text": "",
            "last_buddy_response": "",
        }


memory = _load_memory()


def _save_memory():
    try:
        MEMORY_PATH.write_text(
            json.dumps(
                memory,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            "Memory save failed:",
            exc
        )


def _remember_interaction(
    user_text,
    response,
):
    memory["conversation_count"] = (
        memory.get(
            "conversation_count",
            0,
        )
        + 1
    )

    memory["last_user_text"] = user_text
    memory["last_buddy_response"] = response

    _save_memory()


def _extract_name(text):
    patterns = [
        r"\bmy name is\s+(.+)",
        r"\bcall me\s+(.+)",
        r"\bi am called\s+(.+)",
        r"\bi'm called\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if not match:
            continue

        name = match.group(1).strip()

        name = re.sub(
            r"[^a-zA-Z0-9 '\-]",
            "",
            name,
        ).strip()

        if name:
            return name.title()

    return None


def _deterministic_response(text):
    lower = text.lower().strip()

    # -----------------------------
    # NAME MEMORY
    # -----------------------------

    name = _extract_name(text)

    if name:
        memory["name"] = name
        _save_memory()

        return (
            f"Got it, {name}. "
            "I'll remember your name."
        )

    if lower in {
        "what is my name",
        "what's my name",
        "do you know my name",
        "remember my name",
        "who am i",
    }:
        saved_name = memory.get("name")

        if saved_name:
            return (
                f"Your name is "
                f"{saved_name}."
            )

        return (
            "You haven't told me "
            "your name yet."
        )

    # -----------------------------
    # TIME / DATE
    # -----------------------------

    if lower in {
        "what time is it",
        "what is the time",
        "tell me the time",
        "current time",
        "time now",
        "time",
    }:
        now = datetime.datetime.now()

        return (
            "It's "
            + now.strftime("%I:%M %p")
            + "."
        )

    if lower in {
        "what date is it",
        "what is the date",
        "what day is it",
        "what day is today",
        "today date",
        "tell me the date",
    }:
        now = datetime.datetime.now()

        return now.strftime(
            "Today is %A, %B %d, %Y."
        )

    # -----------------------------
    # BASIC CHAT
    # -----------------------------

    if lower in {
        "hello",
        "hi",
        "hey",
        "hello buddy",
        "hi buddy",
        "hey buddy",
    }:
        saved_name = memory.get("name")

        if saved_name:
            return (
                f"Hey {saved_name}! "
                "How's it going?"
            )

        return "Hey! How's it going?"

    if lower in {
        "how are you",
        "how are you doing",
        "how is it going",
        "are you okay",
    }:
        return (
            "I'm doing good. "
            "How are you feeling?"
        )

    if lower in {
        "thank you",
        "thanks",
        "thanks buddy",
        "thank you buddy",
    }:
        return "You're welcome!"

    if lower in {
        "sorry",
        "i am sorry",
        "my bad",
    }:
        return "No worries. It's okay."

    if lower in {
        "bye",
        "goodbye",
        "good night",
        "see you",
        "see you later",
    }:
        return "See you later!"

    # -----------------------------
    # BUDDY INFO
    # -----------------------------

    if lower in {
        "what is your name",
        "what's your name",
        "who are you",
        "tell me your name",
    }:
        return (
            "I'm Buddy, your robot "
            "companion."
        )

    if lower in {
        "what can you do",
        "what are your capabilities",
        "what can you do buddy",
        "tell me what you can do",
    }:
        return (
            "I can talk with you, "
            "remember things, tell the "
            "time and date, respond to "
            "voice commands, and control "
            "my movement through my "
            "safety system."
        )

    if lower in {
        "who made you",
        "who created you",
        "who built you",
        "who is your creator",
    }:
        return (
            "I was created as the "
            "Buddy robot project."
        )

    # -----------------------------
    # JOKES
    # -----------------------------

    if lower in {
        "tell me a joke",
        "say a joke",
        "make me laugh",
        "joke",
    }:
        return random.choice(
            [
                (
                    "Why did the computer "
                    "go to the doctor? "
                    "It had a virus."
                ),
                (
                    "Why don't scientists "
                    "trust atoms? "
                    "Because they make up "
                    "everything."
                ),
                (
                    "What do you call a "
                    "bear with no teeth? "
                    "A gummy bear."
                ),
            ]
        )

    # -----------------------------
    # SIMPLE WELLBEING
    # -----------------------------

    if any(
        phrase in lower
        for phrase in [
            "i am tired",
            "i'm tired",
            "i am sad",
            "i'm sad",
            "feeling down",
            "i am stressed",
            "i'm stressed",
            "i feel lonely",
        ]
    ):
        return (
            "That sounds like a rough "
            "moment. I'm here with you."
        )

    if any(
        phrase in lower
        for phrase in [
            "i am good",
            "i'm good",
            "i am fine",
            "i'm fine",
            "i am happy",
            "i'm happy",
        ]
    ):
        return (
            "That's good to hear!"
        )

    return None


def _ml_response(text):
    intent = ml_brain.predict_intent(
        text,
        threshold=0.30,
    )

    if intent == "unknown":
        return None

    saved_name = memory.get("name")

    if intent == "greeting":
        if saved_name:
            return (
                f"Hey {saved_name}! "
                "How are you?"
            )

        return "Hey! How are you?"

    if intent == "farewell":
        return "See you later!"

    if intent == "time":
        return (
            "It's "
            + datetime.datetime.now().strftime(
                "%I:%M %p"
            )
            + "."
        )

    if intent == "date":
        return datetime.datetime.now().strftime(
            "Today is %A, %B %d, %Y."
        )

    if intent == "joke":
        return (
            "Why did the computer "
            "go to the doctor? "
            "It had a virus."
        )

    if intent == "ask_wellbeing":
        return (
            "I'm doing good. "
            "How are you?"
        )

    if intent == "positive_wellbeing":
        return (
            "That's great to hear!"
        )

    if intent == "negative_wellbeing":
        return (
            "I'm here with you. "
            "Want to talk about it?"
        )

    if intent == "ask_name_bot":
        return (
            "I'm Buddy, your robot "
            "companion."
        )

    if intent == "ask_capabilities":
        return (
            "I can talk with you, "
            "remember things, answer "
            "simple questions, and "
            "respond to movement commands."
        )

    if intent == "thanks":
        return "You're welcome!"

    if intent == "apology":
        return "No worries."

    if intent == "compliment_bot":
        return (
            "Thanks! "
            "That means a lot."
        )

    if intent == "ask_age":
        return (
            "I'm still pretty new, "
            "but I'm learning fast."
        )

    if intent == "ask_creator":
        return (
            "I was created as part of "
            "the Buddy robot project."
        )

    if intent == "help":
        return (
            "Sure. Tell me what you "
            "need help with."
        )

    if intent == "get_name":
        if saved_name:
            return (
                f"Your name is "
                f"{saved_name}."
            )

        return (
            "You haven't told me "
            "your name yet."
        )

    if intent == "set_name":
        name = _extract_name(text)

        if name:
            memory["name"] = name
            _save_memory()

            return (
                f"Got it, {name}. "
                "I'll remember that."
            )

    return None


def get_response(text):
    text = text.strip()

    if not text:
        return None

    response = _deterministic_response(
        text
    )

    if response is None:
        response = _ml_response(
            text
        )

    if response is None:
        response = (
            "I'm listening. "
            "Tell me a little more."
        )

    _remember_interaction(
        text,
        response,
    )

    return response
