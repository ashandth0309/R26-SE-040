"""
Smart AI Dog Robot - FULL FIXED MAIN SYSTEM
✔ Fixed enrollment 50-frame issue
✔ Fixed indentation bug
✔ Emotion detection kept
✔ Face recognition kept
✔ Frame manager + voice + robot unchanged
"""

import cv2
import numpy as np
import time
import argparse
import os

from config import *
from face_recognition_module import FaceRecognition
from emotion_detection import EmotionDetector
from database_manager import DatabaseManager
from frame_manager import FrameManager
from display_manager import DisplayManager
from robot_controller import RobotController
from emotion_logger import EmotionLogger
from speech import speak

class SmartAIDogRobot:

    def __init__(self):
        print("=" * 50)
        print("🐶 SMART AI DOG ROBOT - INITIALIZING")
        print("=" * 50)

        # Modules
        self.face_recognizer = FaceRecognition()
        self.emotion_detector = EmotionDetector()
        self.database = DatabaseManager()
        self.frame_manager = FrameManager()
        self.display = DisplayManager()
       
        self.robot = RobotController()
        self.emotion_logger = EmotionLogger()

        # Camera
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # State
        self.enrollment_mode = False
        self.enrollment_name = None
        self.enrollment_frames = []
        self.enrollment_bbox = None

        self.recognized_faces = {}
        self.recognition_cooldown = 2

        self.last_process_time = 0
        self.process_interval = 0.1

        print("✅ System Ready!")

    # ---------------- CAMERA ----------------
    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("❌ Camera failed")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return True

    def stop_camera(self):
        if self.cap:
            self.cap.release()

    # ---------------- FRAME PROCESS ----------------
    def process_frame(self, frame):

        faces = self.face_recognizer.detect_faces(frame)

        for bbox in faces:

            face_img = self.face_recognizer.extract_face(frame, bbox)
            if face_img is None:
                continue

            embedding = self.face_recognizer.get_embedding(face_img)
            if embedding is None:
                continue

            name, similarity = self.database.find_person(embedding)

            emotion, probs = self.emotion_detector.detect_emotion(face_img)

            # ---------------- KNOWN PERSON ----------------
            if name:

                frame = self.display.draw_face_info(
                    frame, bbox, name, emotion,
                    probs.get(emotion, 0), True
                )

                # robot + voice + log
                if name not in self.recognized_faces:
                    self.recognized_faces[name] = time.time()

                    self.emotion_logger.log_emotion(
                        name, emotion, probs.get(emotion, 0)
                    )

                    self.robot.react_to_emotion(emotion, name)
                    speak(f"Hello {name}, I sense you are feeling {emotion}")

            # ---------------- UNKNOWN PERSON ----------------
            else:

                if not self.frame_manager.is_capturing:
                    self.frame_manager.start_capture()

                self.frame_manager.add_frame(frame, bbox)

                frame = self.display.draw_face_info(
                    frame, bbox, "UNKNOWN", emotion,
                    probs.get(emotion, 0), False
                )

        return frame

    # ---------------- MAIN LOOP ----------------
    def run(self):

        if not self.start_camera():
            return

        print("🚀 System Running...")

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            # ---------------- ENROLLMENT MODE ----------------
            if self.enrollment_mode:

                display_frame = self.display.draw_enrollment_mode(
                    display_frame,
                    self.enrollment_name,
                    len(self.enrollment_frames) / 50 * 100
                )

                faces = self.face_recognizer.detect_faces(frame)

                # ⭐ FIXED CORE BUG HERE
                if len(faces) > 0 and len(self.enrollment_frames) < 50:

                    bbox = faces[0]
                    self.enrollment_bbox = bbox

                    face_img = self.face_recognizer.extract_face(frame, bbox)

                    if face_img is not None and self.face_recognizer.validate_face(face_img):

                        self.enrollment_frames.append(frame.copy())

                        print(f"📸 Frame {len(self.enrollment_frames)}/50", end="\r")

                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                # complete
                if len(self.enrollment_frames) >= 50:
                    self.complete_enrollment()

            # ---------------- NORMAL MODE ----------------
            else:
                display_frame = self.process_frame(display_frame)

            # FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.fps = 30 / (time.time() - self.start_time)

            display_frame = self.display.draw_status_bar(
                display_frame,
                "Enrollment" if self.enrollment_mode else "Recognition",
                self.fps
            )

            cv2.imshow("Smart AI Dog Robot", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('e'):
                self.start_enrollment()

        self.cleanup()

    # ---------------- ENROLLMENT ----------------
    def start_enrollment(self):
        name = input("Enter name: ").strip()

        if name:
            self.enrollment_mode = True
            self.enrollment_name = name
            self.enrollment_frames = []
            print(f"🎯 Enrollment started for {name}")

    def complete_enrollment(self):

        print("\n📊 Processing enrollment...")

        embeddings = self.face_recognizer.get_embeddings_batch(
            self.enrollment_frames
        )

        if len(embeddings) > 5:
            self.database.add_person(self.enrollment_name, embeddings)
            print("✅ Enrollment Success!")
        else:
            print("❌ Not enough good frames")

        self.enrollment_mode = False
        self.enrollment_frames = []

    # ---------------- CLEANUP ----------------
    def cleanup(self):
        self.stop_camera()
        cv2.destroyAllWindows()
        print("👋 System Shutdown")


if __name__ == "__main__":
    robot = SmartAIDogRobot()
    robot.run()