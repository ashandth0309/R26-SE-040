"""
Smart AI Dog Robot - Integrated Face/Eye System

Face/eye side from final-clean.
Voice side stays in root speech.py from Ashandth.
"""

import cv2
import time
import threading
import tkinter as tk
from tkinter import simpledialog

from faceRecAndEmotion.config import *
from faceRecAndEmotion.face_recognition_module import FaceRecognition
from faceRecAndEmotion.emotion_detection import EmotionDetector
from faceRecAndEmotion.database_manager import DatabaseManager
from faceRecAndEmotion.frame_manager import FrameManager
from faceRecAndEmotion.robot_controller import RobotController
from faceRecAndEmotion.emotion_logger import EmotionLogger


class SmartAIDogRobot:
    def __init__(self, shared_state=None, robot=None):
        print("=" * 50)
        print("🐶 SMART AI DOG ROBOT - FACE / EYE SYSTEM")
        print("=" * 50)

        self.shared_state = shared_state

        self.face_recognizer = FaceRecognition()
        self.emotion_detector = EmotionDetector()
        self.database = DatabaseManager()
        self.frame_manager = FrameManager()
        self.robot = robot if robot is not None else RobotController(shared_state=shared_state)
        self.emotion_logger = EmotionLogger()

        telegram_thread = threading.Thread(
            target=self.emotion_logger.listen_telegram,
            daemon=True
        )
        telegram_thread.start()

        print("📩 Background services active")

        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.frame_counter = 0

        self.cached_faces = []
        self.cached_names = []
        self.cached_emotions = []
        self.cached_confidences = []
        self.cached_all_emotions = []
        self.cached_sources = []

        self.enrollment_mode = False
        self.enrollment_name = None
        self.enrollment_frames = []

        self.last_recognition_time = {}
        self.recognition_cooldown = 2.0

        self.show_details = True

        print("✅ Face/Eye System Ready!")

    def start_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("❌ Cannot access camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        print(f"✅ Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return True

    def draw_confidence_bar(self, frame, x, y, confidence, color, width=60, height=6):
        confidence = max(0, min(100, confidence))

        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)

        filled_width = int(width * confidence / 100)

        if filled_width > 0:
            cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)

        percent_text = f"{confidence:.0f}%"

        cv2.putText(
            frame,
            percent_text,
            (x + width + 5, y + height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

        return frame

    def draw_emotion_details(self, frame, x, y, all_emotions):
        if not all_emotions:
            return frame

        sorted_emotions = sorted(
            all_emotions.items(),
            key=lambda item: item[1],
            reverse=True
        )

        start_y = y + 25

        for emotion, conf in sorted_emotions[:3]:
            if conf > 5:
                emoji = self.emotion_detector.get_emotion_emoji(emotion)
                color = self.emotion_detector.get_emotion_color(emotion)
                text = f"{emoji} {emotion}: {conf:.1f}%"

                text_size = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    1
                )[0]

                cv2.rectangle(
                    frame,
                    (x, start_y - 12),
                    (x + text_size[0] + 4, start_y),
                    (0, 0, 0),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x, start_y - 12),
                    (x + text_size[0] + 4, start_y),
                    color,
                    1
                )

                cv2.putText(
                    frame,
                    text,
                    (x + 2, start_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                start_y += 18

        return frame

    def recognize_hybrid(self, face_img):
        name, score, source = self.face_recognizer.predict_person(face_img)

        if name:
            return name, score, source

        embedding = self.face_recognizer.get_embedding(face_img)
        db_name, db_score = self.database.find_person(embedding)

        if db_name:
            return db_name, db_score * 100, "enrolled_database"

        return None, max(score, db_score * 100), "unknown"

    def process_frame(self, frame):
        self.frame_counter += 1
        display_frame = frame.copy()

        if self.frame_counter % FACE_DETECTION_INTERVAL == 0:
            self.cached_faces = self.face_recognizer.detect_faces(frame)
            self.cached_names = []
            self.cached_emotions = []
            self.cached_confidences = []
            self.cached_all_emotions = []
            self.cached_sources = []

            for bbox in self.cached_faces:
                face_img = self.face_recognizer.extract_face(frame, bbox)

                if face_img is not None and self.face_recognizer.validate_face(face_img):
                    if self.frame_counter % EMBEDDING_INTERVAL == 0:
                        name, recognition_score, source = self.recognize_hybrid(face_img)

                        self.cached_names.append(name)
                        self.cached_sources.append(source)

                        print(
                            f"👤 Recognition: "
                            f"{name if name else 'UNKNOWN'} | "
                            f"score={recognition_score:.1f}% | "
                            f"source={source}"
                        )
                    else:
                        name = self.cached_names[-1] if self.cached_names else None
                        source = self.cached_sources[-1] if self.cached_sources else "cached"

                        self.cached_names.append(name)
                        self.cached_sources.append(source)

                    if self.frame_counter % EMOTION_DETECTION_INTERVAL == 0:
                        emotion, confidence, all_emotions = self.emotion_detector.detect_emotion(face_img)

                        self.cached_emotions.append(emotion)
                        self.cached_confidences.append(confidence)
                        self.cached_all_emotions.append(all_emotions)

                        if confidence > 60:
                            print(
                                f"🎭 {name if name else 'UNKNOWN'} - "
                                f"Emotion: {emotion.upper()} ({confidence:.1f}%)"
                            )
                    else:
                        emotion = self.cached_emotions[-1] if self.cached_emotions else "neutral"
                        confidence = self.cached_confidences[-1] if self.cached_confidences else 0
                        all_emotions = self.cached_all_emotions[-1] if self.cached_all_emotions else {}

                        self.cached_emotions.append(emotion)
                        self.cached_confidences.append(confidence)
                        self.cached_all_emotions.append(all_emotions)

                    if self.shared_state:
                        self.shared_state.set_face_status(
                            person=name,
                            emotion=emotion,
                            confidence=confidence,
                            source=source
                        )

                    if name:
                        current_time = time.time()

                        if (
                            name not in self.last_recognition_time
                            or current_time - self.last_recognition_time[name] > self.recognition_cooldown
                        ):
                            self.last_recognition_time[name] = current_time

                            def async_handler(n, e, c):
                                self.emotion_logger.log_emotion(n, e, c / 100)
                                self.robot.react_to_emotion(e, n)

                            threading.Thread(
                                target=async_handler,
                                args=(name, emotion, confidence),
                                daemon=True
                            ).start()
                    else:
                        if not self.frame_manager.is_capturing:
                            self.frame_manager.start_capture()

                        self.frame_manager.add_frame(frame, bbox)

                else:
                    self.cached_names.append(None)
                    self.cached_sources.append("invalid_face")
                    self.cached_emotions.append("neutral")
                    self.cached_confidences.append(0)
                    self.cached_all_emotions.append({})

        for idx, bbox in enumerate(self.cached_faces):
            x, y, w, h = bbox

            name = self.cached_names[idx] if idx < len(self.cached_names) else None
            emotion = self.cached_emotions[idx] if idx < len(self.cached_emotions) else "neutral"
            confidence = self.cached_confidences[idx] if idx < len(self.cached_confidences) else 0
            all_emotions = self.cached_all_emotions[idx] if idx < len(self.cached_all_emotions) else {}
            source = self.cached_sources[idx] if idx < len(self.cached_sources) else "unknown"

            emotion_color = self.emotion_detector.get_emotion_color(emotion)

            if name:
                color = BOX_COLOR
                label = f"{name} [{emotion.upper()}] {confidence:.0f}%"
            else:
                color = UNKNOWN_BOX_COLOR
                label = f"UNKNOWN [{emotion.upper()}] {confidence:.0f}%"

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            label_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                FONT_THICKNESS
            )[0]

            label_y = y - 10 if y - 10 > 10 else y + h + 20
            label_y = max(20, min(label_y, display_frame.shape[0] - 10))

            cv2.rectangle(
                display_frame,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0] + 5, label_y),
                color,
                -1
            )

            cv2.putText(
                display_frame,
                label,
                (x + 2, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS
            )

            source_text = f"src:{source}"

            cv2.putText(
                display_frame,
                source_text,
                (x, label_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

            bar_y = y + h + 5

            if bar_y + 10 < display_frame.shape[0]:
                self.draw_confidence_bar(
                    display_frame,
                    x,
                    bar_y,
                    confidence,
                    emotion_color,
                    width=80,
                    height=6
                )

            if self.show_details and confidence > 30:
                details_y = y + h + 15

                if details_y + 60 < display_frame.shape[0]:
                    self.draw_emotion_details(
                        display_frame,
                        x,
                        details_y,
                        all_emotions
                    )

        return display_frame

    def run(self):
        if not self.start_camera():
            return

        print("🚀 Running face/eye system")
        print(f"   Face detection: every {FACE_DETECTION_INTERVAL} frames")
        print(f"   Emotion detection: every {EMOTION_DETECTION_INTERVAL} frames")
        print("   Press 'e' to enroll new person")
        print("   Press 'd' to toggle detailed emotion display")
        print("   Press 'q' to quit\n")

        while True:
            if self.shared_state and not self.shared_state.get_snapshot().get("running", True):
                break

            ret, frame = self.cap.read()

            if not ret or frame is None:
                continue

            self.frame_count += 1
            self.fps_counter += 1

            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()

            if self.enrollment_mode:
                display_frame = frame.copy()

                progress = len(self.enrollment_frames) / 50.0 * 100

                cv2.putText(
                    display_frame,
                    f"Enrolling: {self.enrollment_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Progress: {int(progress)}% ({len(self.enrollment_frames)}/50)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    "Look at camera and move your head slightly",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

                faces = self.face_recognizer.detect_faces(frame)

                if faces and len(self.enrollment_frames) < 50:
                    bbox = faces[0]
                    face_img = self.face_recognizer.extract_face(frame, bbox)

                    if face_img is not None and self.face_recognizer.validate_face(face_img):
                        self.enrollment_frames.append(face_img)

                        x, y, w, h = bbox

                        cv2.rectangle(
                            display_frame,
                            (x, y),
                            (x + w, y + h),
                            (255, 255, 0),
                            2
                        )

                        cv2.putText(
                            display_frame,
                            "CAPTURING...",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1
                        )

                if len(self.enrollment_frames) >= 50:
                    self.complete_enrollment()
            else:
                display_frame = self.process_frame(frame)

            cv2.rectangle(
                display_frame,
                (0, 0),
                (display_frame.shape[1], 25),
                (0, 0, 0),
                -1
            )

            status = "ENROLLMENT" if self.enrollment_mode else "ACTIVE"
            details_status = "DETAILS:ON" if self.show_details else "DETAILS:OFF"

            cv2.putText(
                display_frame,
                f"{status} | {details_status} | FPS: {self.fps}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Smart AI Dog Robot - Face/Eye System",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("e") and not self.enrollment_mode:
                self.start_enrollment()

            elif key == ord("d"):
                self.show_details = not self.show_details
                print(
                    f"📊 Detailed emotion display: "
                    f"{'ON' if self.show_details else 'OFF'}"
                )

        self.cleanup()

    def start_enrollment(self):
        def gui_prompt():
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            name = simpledialog.askstring(
                "Enroll Person",
                "Enter person's name:",
                parent=root
            )

            root.destroy()

            if name and name.strip():
                self.enrollment_name = name.strip()
                self.enrollment_frames = []
                self.enrollment_mode = True

                print(f"\n📝 Starting enrollment for: {self.enrollment_name}")
                print("   Look at the camera and move your head slightly...")
                print("   I will capture 50 face samples...")

        threading.Thread(target=gui_prompt, daemon=True).start()

    def complete_enrollment(self):
        print(f"\n📊 Processing {len(self.enrollment_frames)} face samples...")

        embeddings = self.face_recognizer.get_embeddings_batch(
            self.enrollment_frames
        )

        if len(embeddings) >= 10:
            self.database.add_person(
                self.enrollment_name,
                embeddings
            )

            print(f"✅ Successfully enrolled '{self.enrollment_name}'!")
            print(f"   Total embeddings created: {len(embeddings)}")
            print("   This person will be recognized through enrolled_database source.")
        else:
            print(
                f"❌ Enrollment failed - only {len(embeddings)} valid faces detected"
            )
            print("   Please try again with better lighting and face visibility")

        self.enrollment_mode = False
        self.enrollment_frames = []
        self.enrollment_name = None

    def cleanup(self):
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("\n👋 Face/Eye system shutdown complete")


if __name__ == "__main__":
    app = SmartAIDogRobot()
    app.run()