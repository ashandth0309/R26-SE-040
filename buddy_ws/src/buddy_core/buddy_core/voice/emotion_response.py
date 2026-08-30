from __future__ import annotations

import random
import time


EMOTION_COOLDOWN_SECONDS = 90.0

_last_emotion = None
_last_response_time = 0.0


def get_emotion_response(
    emotion,
    confidence=1.0,
):
    global _last_emotion
    global _last_response_time

    emotion = (
        str(emotion)
        .lower()
        .strip()
    )

    if confidence < 0.60:
        return None

    now = time.monotonic()

    if (
        emotion == _last_emotion
        and (
            now
            - _last_response_time
        )
        < EMOTION_COOLDOWN_SECONDS
    ):
        return None

    responses = {
        "sad": [
            (
                "You look a little sad. "
                "Do you want to tell me "
                "what happened?"
            ),
            (
                "Hey, you seem a bit down. "
                "I'm here if you want to talk."
            ),
        ],

        "happy": [
            "You look happy today!",
            (
                "You seem happy. "
                "That's nice to see."
            ),
        ],

        "angry": [
            (
                "You seem a little upset. "
                "Is everything okay?"
            ),
        ],

        "tired": [
            (
                "You look tired. "
                "Have you had enough rest?"
            ),
        ],
    }

    choices = responses.get(
        emotion
    )

    if not choices:
        return None

    _last_emotion = emotion
    _last_response_time = now

    return random.choice(
        choices
    )
