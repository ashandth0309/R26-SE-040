"""
Smart AI Dog Robot - Integrated Face/Eye, Object Learning, Speech & Mobile App Sync System
"""

import base64
import cv2
import glob
import os
import re
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# 1. Internal Folder Paths Setup (Nested strictly inside faceRecAndEmotion)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Speech module import - buddy_speak connect aagum
try:
    from speech import buddy_speak
except Exception:
    try:
        from speech import speak as buddy_speak
    except Exception:
        def buddy_speak(text):
            print(f"🗣️ BUDDY: {text}")

from faceRecAndEmotion.config import *
from faceRecAndEmotion.database_manager import DatabaseManager
from faceRecAndEmotion.emotion_detection import EmotionDetector
from faceRecAndEmotion.emotion_logger import EmotionLogger
from faceRecAndEmotion.face_recognition_module import FaceRecognition
from faceRecAndEmotion.frame_manager import FrameManager
from faceRecAndEmotion.robot_controller import RobotController

try:
    from faceRecAndEmotion.simple_face_memory import SimpleFaceMemory
except Exception:
    SimpleFaceMemory = None

try:
    from faceRecAndEmotion.object_learning_module import ObjectLearningModule
except Exception:
    from object_learning_module import ObjectLearningModule

global_robot_app = None
learner = ObjectLearningModule(threshold=0.70)
flask_app = Flask(__name__)
CORS(flask_app)

# Global Recording Variables
is_recording = False
video_writer = None
current_record_filename = None

latest_unknown_alert = {
    "has_alert": False,
    "timestamp": None,
    "image_base64": None,
    "raw_frame": None,
}

last_unknown_sent_time = 0
ALERT_COOLDOWN_SECONDS = 20

robot_telemetry = {
    "battery": 92,
    "state": "Patrolling",
    "last_face": "Sobiya",
    "last_emotion": "Neutral",
    "confidence": 94,
    "cpu_load": 0.28,
    "memory_load": 0.54,
    "temp": "38°C (Normal)",
    "storage": "64GB / 128GB",
    "patrol_active": True,
    "patrol_progress": 0.65,
    "patrol_elapsed": "Elapsed: 2h 15m",
    "threats_count": 0,
    "max_speed": 1.2,
    "emotion_detection": True,
    "face_recognition": True,
    "robot_name": "BUDDY",
}


# ==========================================
# 📷 CAMERA, RECORDING & PERSISTENT MEDIA API
# ==========================================

@flask_app.route("/video_call")
def video_call():
    """Smooth, non-blocking MJPEG live stream generator with Object Learning overlay"""
    def generate():
        global global_robot_app, learner
        while True:
            if global_robot_app is None or global_robot_app.cap is None:
                time.sleep(0.05)
                continue

            ret, frame = global_robot_app.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Original Object Learning module process frame integration
            frame, box, text, obj = learner.process_frame(frame)

            x1, y1, x2, y2 = box
            color = (0, 0, 255) if learner.is_learning else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            ret_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret_enc:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.03) # Prevents video lag and stream freezing

    return Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@flask_app.route("/api/camera/snapshot", methods=["POST"])
def camera_snapshot():
    global global_robot_app
    if global_robot_app and global_robot_app.current_live_frame is not None:
        filename = f"snapshot_{int(time.time())}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)
        cv2.imwrite(filepath, global_robot_app.current_live_frame)
        print(f"📸 Snapshot saved inside captures: {filepath}")
        return jsonify({
            "status": "success",
            "file": filename,
            "url": f"/captures/{filename}"
        }), 200
    return jsonify({"status": "failed", "message": "Frame not available"}), 400


@flask_app.route("/api/camera/record/start", methods=["POST"])
def start_recording():
    global is_recording, video_writer, current_record_filename
    if is_recording:
        return jsonify({"status": "already_recording"}), 200

    filename = f"record_{int(time.time())}.avi"
    filepath = os.path.join(CAPTURES_DIR, filename)
    current_record_filename = filename

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
    is_recording = True
    print(f"🔴 Video recording started: {filepath}")
    return jsonify({"status": "recording_started", "file": filename}), 200


@flask_app.route("/api/camera/record/stop", methods=["POST"])
def stop_recording():
    global is_recording, video_writer, current_record_filename
    if not is_recording:
        return jsonify({"status": "not_recording"}), 200

    is_recording = False
    if video_writer is not None:
        video_writer.release()
        video_writer = None

    saved_file = current_record_filename
    print(f"⏹️ Video recording completed and saved: {saved_file}")
    return jsonify({
        "status": "recording_stopped",
        "file": saved_file,
        "url": f"/captures/{saved_file}"
    }), 200


@flask_app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@flask_app.route("/api/camera/media_list", methods=["GET"])
def get_media_list():
    files = []
    if os.path.exists(CAPTURES_DIR):
        for f in sorted(os.listdir(CAPTURES_DIR), reverse=True):
            if f.endswith(('.jpg', '.avi', '.mp4')):
                filepath = os.path.join(CAPTURES_DIR, f)
                mod_time = os.path.getmtime(filepath)
                time_str = datetime.fromtimestamp(mod_time).strftime("%I:%M %p")
                files.append({
                    "name": f,
                    "title": "Front Cam" if f.endswith('.jpg') else "Record Video",
                    "time": time_str,
                    "type": "photo" if f.endswith('.jpg') else "video",
                    "url": f"/captures/{f}"
                })
    return jsonify(files), 200


@flask_app.route("/api/camera/delete_media", methods=["POST"])
def delete_media():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    target_path = os.path.join(CAPTURES_DIR, filename)
    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"🗑️ Deleted media: {filename}")
        return jsonify({"status": "success", "message": "File deleted"}), 200
    return jsonify({"status": "failed", "message": "File not found"}), 404


@flask_app.route("/api/camera/torch", methods=["POST"])
def camera_torch():
    data = request.get_json(silent=True) or {}
    state = data.get("state", False)
    print(f"🔦 Robot Light State: {state}")
    return jsonify({"status": "success", "torch": state}), 200


@flask_app.route("/api/camera/reset", methods=["POST"])
def camera_reset():
    global global_robot_app
    if global_robot_app and hasattr(global_robot_app, 'robot'):
        try:
            global_robot_app.robot.pan_camera(0)
            global_robot_app.robot.tilt_camera(0)
        except Exception:
            pass
    print("🔄 Camera pan-tilt reset to center.")
    return jsonify({"status": "success", "message": "Camera reset to center"}), 200


# =======================================================
# 👤 ENROLLMENT & OBJECT LEARNING API ENDPOINTS
# =======================================================

@flask_app.route("/api/enroll_unknown_person", methods=["POST"])
def enroll_unknown_person():
    global latest_unknown_alert, global_robot_app, robot_telemetry
    try:
        data = request.get_json(silent=True) or {}
        person_name = data.get("name", "").strip().lower()

        if not person_name:
            return jsonify({"status": "failed", "message": "Name is empty"}), 400

        # Normalization: "sobiya a" -> "sobiya" to avoid duplicate folders
        clean_folder_name = person_name.split()[0]

        person_dataset_dir = os.path.join(DATASET_DIR, clean_folder_name)
        os.makedirs(person_dataset_dir, exist_ok=True)

        face_img = latest_unknown_alert.get("raw_frame")
        if face_img is None and data.get("image"):
            b64_str = data.get("image")
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if face_img is None:
            return jsonify({"status": "failed", "message": "No face image available"}), 400

        img_filename = f"{clean_folder_name}_{int(time.time())}.jpg"
        save_path = os.path.join(person_dataset_dir, img_filename)
        cv2.imwrite(save_path, face_img)
        print(f"📁 Image stored cleanly inside: {save_path}")

        if global_robot_app and hasattr(global_robot_app, 'face_recognizer'):
            emb = global_robot_app.face_recognizer.get_embedding(face_img)
            if emb is not None and hasattr(global_robot_app, 'database'):
                if hasattr(global_robot_app.database, 'add_person'):
                    global_robot_app.database.add_person(clean_folder_name, emb)
                elif hasattr(global_robot_app.database, 'save_person'):
                    global_robot_app.database.save_person(clean_folder_name, emb)
                print(f"🎉 Enrolled & linked into known_faces.json: {clean_folder_name}")

        latest_unknown_alert["has_alert"] = False
        latest_unknown_alert["image_base64"] = None
        latest_unknown_alert["raw_frame"] = None
        latest_unknown_alert["timestamp"] = None
        robot_telemetry["threats_count"] = 0

        return jsonify({
            "status": "success",
            "message": f"{clean_folder_name} registered successfully!"
        }), 200

    except Exception as err:
        print(f"❌ Critical Enrollment Error: {err}")
        return jsonify({"status": "failed", "message": str(err)}), 500


@flask_app.route("/api/get_unknown_alert", methods=["GET"])
def get_unknown_alert():
    global latest_unknown_alert
    if latest_unknown_alert["has_alert"] and latest_unknown_alert["image_base64"]:
        return jsonify(
            {
                "status": "alert",
                "time": latest_unknown_alert["timestamp"],
                "image": latest_unknown_alert["image_base64"],
            }
        )
    return jsonify({"status": "idle", "image": None})


@flask_app.route("/api/clear_unknown_alert", methods=["POST"])
@flask_app.route("/api/clear_alert", methods=["POST"])
def clear_alert():
    global latest_unknown_alert
    latest_unknown_alert["has_alert"] = False
    latest_unknown_alert["image_base64"] = None
    latest_unknown_alert["raw_frame"] = None
    robot_telemetry["threats_count"] = 0
    return jsonify({"status": "cleared", "success": True})


@flask_app.route('/learn/<obj_name>', methods=['GET', 'POST'])
def api_learn_object(obj_name):
    global learner
    if learner:
        learner.start_learning(obj_name)
        print(f"🎯 [API Received]: Learning started for '{obj_name}'")
        return {"status": "success", "object": obj_name}, 200
    return {"status": "error", "message": "Learner not found"}, 500


@flask_app.route('/identify', methods=['GET'])
def api_identify_object():
    global learner
    name = learner.last_detected_name if learner else "unknown"
    return {"detected": name}


# ==========================================
# 🎙️ VOICE COMMANDS & TELEMETRY
# ==========================================

@flask_app.route("/api/control/voice_command", methods=["POST"])
def voice_command():
    global global_robot_app
    data = request.get_json(silent=True) or {}
    cmd = data.get("command", "").strip()
    print(f"🎙️ Mobile Voice Command Received: '{cmd}'")

    response_text = "Command executed."
    cmd_lower = cmd.lower()

    if "come here" in cmd_lower or "move forward" in cmd_lower:
        response_text = "Motors active. Rolling towards you now."
        if global_robot_app and hasattr(global_robot_app, 'robot'):
            try:
                global_robot_app.robot.move_forward()
            except Exception as e:
                print(f"Motor error: {e}")

    elif "stop" in cmd_lower:
        response_text = "Brakes applied. Robot stationary."
        if global_robot_app and hasattr(global_robot_app, 'robot'):
            try:
                global_robot_app.robot.stop()
            except Exception as e:
                print(f"Motor stop error: {e}")

    elif "patrol" in cmd_lower:
        robot_telemetry["patrol_active"] = True
        robot_telemetry["state"] = "Patrolling"
        response_text = "Patrol route started. Monitoring room."

    elif "scan" in cmd_lower:
        response_text = "Starting camera sensor sweep to scan surroundings."
        if global_robot_app and hasattr(global_robot_app, 'robot'):
            try:
                global_robot_app.robot.pan_camera(0)
            except Exception:
                pass
    else:
        response_text = f"Executing {cmd} now."

    threading.Thread(target=lambda: buddy_speak(response_text), daemon=True).start()

    return jsonify({
        "status": "success",
        "response": response_text
    }), 200


@flask_app.route("/api/control/action", methods=["POST"])
def control_action():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")
    print(f"🎮 Mobile Command: {action}")
    if action == "RETURN_BASE":
        response_msg = "Returning to docking base."
    elif action == "SPEAK":
        response_msg = "Hello, I am Buddy your companion."
    else:
        response_msg = f"Action {action} dispatched."

    threading.Thread(target=lambda: buddy_speak(response_msg), daemon=True).start()
    return jsonify({"status": "success", "action": action, "response": response_msg})


@flask_app.route("/api/robot_status", methods=["GET"])
def get_robot_status():
    return jsonify({
        "status": "online",
        "battery": robot_telemetry["battery"],
        "state": robot_telemetry["state"],
        "last_face": robot_telemetry["last_face"],
        "last_emotion": robot_telemetry["last_emotion"],
        "has_unknown": latest_unknown_alert["has_alert"],
    })


@flask_app.route("/api/household_members", methods=["GET"])
def get_household_members():
    members = [
        {"name": "Sobiya", "role": "Owner", "status": "RECOGNIZED"},
        {"name": "Rayan", "role": "Member", "status": "RECOGNIZED"},
    ]
    if latest_unknown_alert["has_alert"] and latest_unknown_alert["image_base64"]:
        members.append({
            "name": "Unknown Visitor",
            "role": "Guest",
            "status": "UNREGISTERED",
            "image": latest_unknown_alert["image_base64"]
        })
    return jsonify(members)


@flask_app.route("/api/device_diagnostics", methods=["GET"])
def get_device_diagnostics():
    return jsonify({
        "battery": robot_telemetry["battery"],
        "connection": "Strong (Wi-Fi)",
        "behavior": robot_telemetry["state"],
        "temp": robot_telemetry["temp"],
        "storage": robot_telemetry["storage"],
        "cpu_load": robot_telemetry["cpu_load"],
        "memory_load": robot_telemetry["memory_load"],
        "last_checked": datetime.now().strftime("%I:%M %p"),
        "sensors": {
            "Camera Module": True,
            "LiDAR / Ultrasonic": True,
            "Microphone Array": True,
            "IMU Accelerometer": True,
        }
    })


@flask_app.route("/api/security/status", methods=["GET"])
def get_security_status():
    return jsonify({
        "patrol_active": robot_telemetry["patrol_active"],
        "progress": robot_telemetry["patrol_progress"],
        "elapsed": robot_telemetry["patrol_elapsed"],
        "threats_count": 1 if latest_unknown_alert["has_alert"] else 0,
    })


@flask_app.route("/api/security/toggle_patrol", methods=["POST"])
def toggle_patrol():
    data = request.get_json(silent=True) or {}
    robot_telemetry["patrol_active"] = data.get("active", not robot_telemetry["patrol_active"])
    robot_telemetry["state"] = "Patrolling" if robot_telemetry["patrol_active"] else "Standby"
    return jsonify({"success": True, "patrol_active": robot_telemetry["patrol_active"]})


@flask_app.route("/api/control/move", methods=["POST"])
def control_move():
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "STOP")
    return jsonify({"status": "moving", "direction": direction})


@flask_app.route("/api/settings/get", methods=["GET"])
def get_settings():
    return jsonify({
        "max_speed": robot_telemetry["max_speed"],
        "emotion_detection": robot_telemetry["emotion_detection"],
        "face_recognition": robot_telemetry["face_recognition"],
        "robot_name": robot_telemetry["robot_name"],
    })


@flask_app.route("/api/settings/update", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True) or {}
    for k, v in data.items():
        if k in robot_telemetry:
            robot_telemetry[k] = v
    return jsonify({"success": True})


@flask_app.route("/api/notifications", methods=["GET"])
def get_notifications():
    notifications = []
    if latest_unknown_alert["has_alert"] and latest_unknown_alert["image_base64"]:
        notifications.append({
            "id": "1",
            "title": "CRITICAL: Unregistered Person",
            "desc": "Unknown face confirmed. Click to verify & enroll into household.",
            "time": latest_unknown_alert["timestamp"] or "Just now",
            "isUnread": True,
            "category": "Security",
            "hasAction": True,
            "actionText": "Register Face",
            "actionType": "enroll",
            "image": latest_unknown_alert["image_base64"]
        })
    notifications.append({
        "id": "2",
        "title": "INFO: Patrol Route Active",
        "desc": "Perimeter Alpha monitoring running normally.",
        "time": "15m ago",
        "isUnread": False,
        "category": "System",
        "hasAction": False,
        "image": None
    })
    return jsonify(notifications)


# ==========================================
# 🐶 SMART AI DOG ROBOT CLASS
# ==========================================

class SmartAIDogRobot:

    def __init__(self, shared_state=None, robot=None):
        print("=" * 50)
        print("🐶 SMART AI DOG ROBOT - FACE / EYE SYSTEM")
        print("=" * 50)

        self.shared_state = shared_state
        self.face_recognizer = FaceRecognition()
        self.emotion_detector = EmotionDetector()
        self.database = DatabaseManager()
        self.already_greeted_people = set()

        self.unknown_consecutive_count = 0
        self.object_module = ObjectLearningModule(threshold=0.70)
        self.latest_object_status = ""
        self.current_live_frame = None

        if SimpleFaceMemory is not None:
            try:
                self.simple_face_memory = SimpleFaceMemory()
            except Exception as error:
                self.simple_face_memory = None
        else:
            self.simple_face_memory = None

        self.frame_manager = FrameManager()
        self.robot = robot if robot is not None else RobotController(shared_state=shared_state)
        self.emotion_logger = EmotionLogger()

        telegram_thread = threading.Thread(
            target=self.emotion_logger.listen_telegram, daemon=True
        )
        telegram_thread.start()

        self.cap = None
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
        print("✅ System Ready!")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("❌ Cannot access camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def start_enrollment(self):
        root = tk.Tk()
        root.withdraw()
        name = simpledialog.askstring("Enroll Person", "Enter Name:")
        root.destroy()

        if name:
            self.enrollment_name = name.strip()
            self.enrollment_frames = []
            self.enrollment_mode = True
            print(f"📸 Starting face enrollment for: {self.enrollment_name}")

    def recognize_hybrid(self, face_img):
        score = 0
        db_score = 0

        try:
            name, score, source = self.face_recognizer.predict_person(face_img)
            if name:
                return name, score, source
        except Exception:
            pass

        try:
            embedding = self.face_recognizer.get_embedding(face_img)
            db_name, db_score = self.database.find_person(embedding)
            if db_name:
                return db_name, db_score * 100, "enrolled_database"
        except Exception:
            pass

        try:
            if self.simple_face_memory is not None:
                lbph_name, lbph_confidence = self.simple_face_memory.predict(face_img)
                if lbph_name:
                    return lbph_name, lbph_confidence, "lbph_voice_memory"
        except Exception:
            pass

        return None, max(score, db_score * 100), "unknown"

    def process_frame(self, frame):
        self.current_live_frame = frame.copy()

        # Video recording pipeline frame writer
        global is_recording, video_writer
        if is_recording and video_writer is not None:
            resized_frame = cv2.resize(frame, (640, 480))
            video_writer.write(resized_frame)

        if 'learner' in globals() and learner is not None:
            learner.current_frame = frame.copy()

        global latest_unknown_alert, robot_telemetry, last_unknown_sent_time
        self.frame_counter += 1
        display_frame = frame.copy()

        # Low-light auto enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 80:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            display_frame = frame.copy()

        # Offline enrollment mode capture
        if self.enrollment_mode:
            faces = self.face_recognizer.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                face_img = self.face_recognizer.extract_face(frame, (x, y, w, h))
                if face_img is not None:
                    self.enrollment_frames.append(face_img)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"Capturing: {len(self.enrollment_frames)}/50",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    if len(self.enrollment_frames) >= 50:
                        embeddings = [self.face_recognizer.get_embedding(f) for f in self.enrollment_frames]
                        valid_embeddings = [e for e in embeddings if e is not None]
                        if valid_embeddings:
                            mean_embedding = np.mean(valid_embeddings, axis=0)
                            self.database.add_person(self.enrollment_name, mean_embedding)
                            print(f"🎉 Enrolled {self.enrollment_name} successfully!")
                        self.enrollment_mode = False
                        self.enrollment_frames = []
            return display_frame

        # Face detection interval
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
                    else:
                        emotion = self.cached_emotions[-1] if self.cached_emotions else "neutral"
                        confidence = self.cached_confidences[-1] if self.cached_confidences else 0
                        all_emotions = self.cached_all_emotions[-1] if self.cached_all_emotions else {}
                        self.cached_emotions.append(emotion)
                        self.cached_confidences.append(confidence)
                        self.cached_all_emotions.append(all_emotions)

                    if name:
                        robot_telemetry["last_face"] = name
                    robot_telemetry["last_emotion"] = emotion.capitalize()
                    robot_telemetry["confidence"] = int(confidence)

                    if self.shared_state:
                        self.shared_state.set_face_status(
                            person=name,
                            emotion=emotion,
                            confidence=confidence,
                            source=source,
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
                                daemon=True,
                            ).start()
                else:
                    self.cached_names.append(None)
                    self.cached_sources.append("invalid_face")
                    self.cached_emotions.append("neutral")
                    self.cached_confidences.append(0)
                    self.cached_all_emotions.append({})

        found_unknown_this_frame = False

        for idx, bbox in enumerate(self.cached_faces):
            x, y, w, h = bbox
            name = self.cached_names[idx] if idx < len(self.cached_names) else None
            emotion = self.cached_emotions[idx] if idx < len(self.cached_emotions) else "neutral"
            confidence = self.cached_confidences[idx] if idx < len(self.cached_confidences) else 0

            conf_percent = confidence * 100.0 if confidence <= 1.0 else float(confidence)

            # Default box_color assignment prevents UnboundLocalError
            box_color = BOX_COLOR

            # ==========================================================
            # 🛡️ STRICT 70% THRESHOLD FILTER
            # ==========================================================
            if not name or str(name).strip().upper() == "UNKNOWN" or conf_percent < 70.0:
                name = "UNKNOWN"
                box_color = UNKNOWN_BOX_COLOR
                found_unknown_this_frame = True
                self.unknown_consecutive_count += 1

                if self.unknown_consecutive_count >= 8:
                    current_ts = time.time()
                    if (not latest_unknown_alert["has_alert"]) and (current_ts - last_unknown_sent_time > ALERT_COOLDOWN_SECONDS):
                        face_crop = self.face_recognizer.extract_face(frame, bbox)
                        if face_crop is not None and face_crop.shape[0] > 50 and face_crop.shape[1] > 50:

                            is_registered = False
                            try:
                                emb = self.face_recognizer.get_embedding(face_crop)
                                if emb is not None:
                                    db_name, db_score = self.database.find_person(emb)
                                    if db_name and db_score >= 0.70:
                                        is_registered = True
                                        name = db_name
                            except Exception:
                                pass

                            if not is_registered:
                                name = "UNKNOWN"
                                box_color = UNKNOWN_BOX_COLOR
                                _, buffer = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                img_b64 = base64.b64encode(buffer).decode("utf-8")
                                latest_unknown_alert["has_alert"] = True
                                latest_unknown_alert["timestamp"] = time.strftime("%I:%M %p")
                                latest_unknown_alert["image_base64"] = img_b64
                                latest_unknown_alert["raw_frame"] = face_crop
                                last_unknown_sent_time = current_ts
                                robot_telemetry["threats_count"] = 1
                                print(f"🚨 CONFIRMED UNKNOWN VISITOR: Sent snapshot to app ({latest_unknown_alert['timestamp']})")
                                self.unknown_consecutive_count = 0
                            else:
                                box_color = BOX_COLOR
            else:
                box_color = BOX_COLOR
                self.unknown_consecutive_count = 0

            label = f"{name or 'UNKNOWN'} [{emotion.upper()}] {conf_percent:.0f}%"
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            label_y = max(20, y - 10 if y - 10 > 10 else y + h + 20)
            cv2.putText(display_frame, label, (x + 2, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

            if name and name != "UNKNOWN":
                if name not in self.already_greeted_people:
                    self.already_greeted_people.add(name)
                    greeting_text = f"Hi {name}, what's up!"
                    print(f"🗣️ Buddy Greeting: {greeting_text}")
                    threading.Thread(target=lambda: buddy_speak(greeting_text), daemon=True).start()

        if not found_unknown_this_frame:
            self.unknown_consecutive_count = 0

        if self.latest_object_status:
            cv2.putText(display_frame, f"Obj: {self.latest_object_status}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display_frame

    def run(self):
        global global_robot_app, learner
        global_robot_app = self

        if not self.start_camera():
            return

        print("🚀 Face/Eye & Telemetry streaming system active (Optimized 30 FPS)")

        while True:
            if self.shared_state and not self.shared_state.get_snapshot().get("running", True):
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            if self.shared_state:
                self.shared_state.set_latest_frame(frame)

            display_frame = self.process_frame(frame)

            # Original Object Learning processing integration in run loop
            display_frame, box, text, obj = learner.process_frame(display_frame)

            x1, y1, x2, y2 = box
            color = (0, 0, 255) if learner.is_learning else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Smart AI Dog Robot - Face/Eye System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e") and not self.enrollment_mode:
                self.start_enrollment()
            elif key == ord("o"):
                learner.start_learning("pen")

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def start_flask_server():
    print("🌐 API Server online on http://0.0.0.0:5000")
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)


def run_robot_voice():
    try:
        import speech
        speech.start_conversation()
    except Exception as e:
        print(f"⚠️ Voice loop status: {e}")


if __name__ == "__main__":
    stream_thread = threading.Thread(target=start_flask_server, daemon=True)
    stream_thread.start()

    voice_thread = threading.Thread(target=run_robot_voice, daemon=True)
    voice_thread.start()

    app = SmartAIDogRobot()
    app.run()
