"""
Shared state between Ashandth speech.py voice system
and final-clean face/eye system.
"""

import threading
import time


class SharedRobotState:
    def __init__(self):
        self.lock = threading.Lock()

        self.running = True

        self.current_person = None
        self.current_emotion = "neutral"
        self.current_face_source = "none"
        self.current_face_confidence = 0.0

        self.last_seen_time = None
        self.last_voice_command = None
        self.last_voice_response = None
        self.last_voice_time = None

    def set_face_status(self, person=None, emotion="neutral", confidence=0.0, source="unknown"):
        with self.lock:
            self.current_person = person
            self.current_emotion = emotion or "neutral"
            self.current_face_confidence = float(confidence or 0.0)
            self.current_face_source = source or "unknown"
            self.last_seen_time = time.time()

    def set_voice_command(self, command):
        with self.lock:
            self.last_voice_command = command
            self.last_voice_time = time.time()

    def set_voice_response(self, response):
        with self.lock:
            self.last_voice_response = response

    def get_snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "current_person": self.current_person,
                "current_emotion": self.current_emotion,
                "current_face_source": self.current_face_source,
                "current_face_confidence": self.current_face_confidence,
                "last_seen_time": self.last_seen_time,
                "last_voice_command": self.last_voice_command,
                "last_voice_response": self.last_voice_response,
                "last_voice_time": self.last_voice_time,
            }

    def stop(self):
        with self.lock:
            self.running = False