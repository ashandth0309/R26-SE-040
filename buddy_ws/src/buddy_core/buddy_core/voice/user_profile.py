from __future__ import annotations

import json
from pathlib import Path


PROFILE_PATH = Path.home() / ".buddy_user_profile.json"


DEFAULT_PROFILE = {
    "preferred_name": "Ashandth",
    "robot_name": "Buddy",

    "preferences": {
        "conversation_style": "friendly, calm, natural",
        "likes_short_spoken_answers": True,
    },

    "relationship": {
        "description": "Buddy is a friendly robot companion trained by Ashandth."
    },

    "notes": []
}


def load_profile():
    if not PROFILE_PATH.exists():
        save_profile(DEFAULT_PROFILE.copy())

    try:
        return json.loads(
            PROFILE_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return DEFAULT_PROFILE.copy()


def save_profile(profile):
    PROFILE_PATH.write_text(
        json.dumps(
            profile,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def remember_note(note):
    note = str(note).strip()

    if not note:
        return

    profile = load_profile()

    notes = profile.setdefault(
        "notes",
        []
    )

    if note not in notes:
        notes.append(note)

    notes[:] = notes[-30:]

    save_profile(profile)


def get_preferred_name():
    profile = load_profile()

    return profile.get(
        "preferred_name",
        "Ashandth",
    )
