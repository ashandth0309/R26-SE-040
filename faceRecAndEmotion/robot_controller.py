"""
Robot Controller for dog robot movement, tail, head, and eye reactions.

Face system calls react_to_emotion().
Voice system calls movement commands through robot_bridge.py.

Currently this prints actions. Add real motor/servo code in the TODO areas.
"""

import time
import threading

try:
    from faceRecAndEmotion.config import ROBOT_REACTIONS
except Exception:
    ROBOT_REACTIONS = {
        "happy": {"tail_wag": "fast", "speed": "normal", "sound": "happy"},
        "sad": {"tail_wag": "slow", "speed": "slow", "sound": "sad"},
        "angry": {"tail_wag": "none", "speed": "cautious", "sound": "alert"},
        "fear": {"tail_wag": "none", "speed": "backward", "sound": "gentle"},
        "surprise": {"tail_wag": "fast", "speed": "stop", "sound": "surprised"},
        "neutral": {"tail_wag": "normal", "speed": "normal", "sound": "normal"},
    }


class RobotController:
    def __init__(self, shared_state=None):
        self.shared_state = shared_state

        self.current_emotion = "neutral"
        self.current_person = None
        self.tail_wag_state = "normal"
        self.speed_state = "normal"
        self.is_moving = False

        self.tail_angle = 0
        self.head_angle = 0
        self.left_eye = "normal"
        self.right_eye = "normal"

        self.reactions = ROBOT_REACTIONS

        self.running = True
        self.animation_thread = threading.Thread(target=self._animation_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()

        print("🤖 Robot Controller initialized")

    def react_to_emotion(self, emotion, person_name=None):
        self.current_emotion = emotion or "neutral"
        self.current_person = person_name

        if emotion in self.reactions:
            reaction = self.reactions[emotion]
            self.tail_wag_state = reaction["tail_wag"]
            self.speed_state = reaction["speed"]

        self._set_eye_expression(emotion)

        print(
            f"🤖 Reacting | person={person_name} | emotion={emotion} | "
            f"tail={self.tail_wag_state} | speed={self.speed_state} | "
            f"eyes={self.left_eye}/{self.right_eye}"
        )

    def _set_eye_expression(self, emotion):
        expressions = {
            "happy": ("wide", "wide"),
            "sad": ("droopy", "droopy"),
            "angry": ("narrow", "narrow"),
            "fear": ("wide", "wide"),
            "surprise": ("very_wide", "very_wide"),
            "disgust": ("narrow", "narrow"),
            "neutral": ("normal", "normal"),
        }

        self.left_eye, self.right_eye = expressions.get(
            emotion,
            ("normal", "normal")
        )

    def _animation_loop(self):
        wag_direction = 1
        wag_speed_map = {
            "fast": 0.05,
            "normal": 0.1,
            "slow": 0.2,
            "none": None,
        }

        while self.running:
            if self.tail_wag_state != "none":
                wag_speed = wag_speed_map.get(self.tail_wag_state, 0.1)

                if wag_speed:
                    self.tail_angle += wag_direction * 15

                    if abs(self.tail_angle) > 45:
                        wag_direction *= -1

                    # TODO: Add real tail servo command here.
                    # Example:
                    # servo_tail.write_angle(90 + self.tail_angle)

                    time.sleep(wag_speed)
            else:
                self.tail_angle = 0
                time.sleep(0.1)

    def move_forward(self):
        self.is_moving = True
        print("🤖 Moving forward")
        # TODO: motor forward code

    def move_backward(self):
        self.is_moving = True
        print("🤖 Moving backward")
        # TODO: motor backward code

    def turn_left(self):
        self.is_moving = True
        print("🤖 Turning left")
        # TODO: left motor turn code

    def turn_right(self):
        self.is_moving = True
        print("🤖 Turning right")
        # TODO: right motor turn code

    def stop(self):
        self.is_moving = False
        print("🤖 Stopped")
        # TODO: stop motors

    def sit(self):
        print("🤖 Sitting")
        # TODO: servo pose

    def stand(self):
        print("🤖 Standing")
        # TODO: servo pose

    def wag_tail(self):
        self.tail_wag_state = "fast"
        print("🤖 Wagging tail")

    def turn_head(self, angle):
        self.head_angle = max(-90, min(90, int(angle)))
        print(f"🤖 Turning head to {self.head_angle} degrees")
        # TODO: head servo command

    def get_status(self):
        return {
            "emotion": self.current_emotion,
            "person": self.current_person,
            "tail_wag": self.tail_wag_state,
            "speed": self.speed_state,
            "tail_angle": self.tail_angle,
            "head_angle": self.head_angle,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "is_moving": self.is_moving,
        }

    def cleanup(self):
        self.running = False

        if self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1.0)

        print("🤖 Robot Controller cleaned up")