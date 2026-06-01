from faceRecAndEmotion.simple_face_memory import SimpleFaceMemory

"""
Bridge used by root speech.py from Ashandth branch.

speech.py calls:
    handle_robot_voice_command(text)

This connects voice commands to:
- robot movement
- eye/head/tail commands
- face recognition status
- emotion status
- fluent voice-based face enrollment:
  "this is me my name is Ashandth"
"""

import os
import re
import json
import time
from datetime import datetime

import cv2


_shared_state = None
_robot = None
_face_app = None

_simple_face_memory = None


def get_simple_face_memory():
    global _simple_face_memory

    if _simple_face_memory is None:
        _simple_face_memory = SimpleFaceMemory()

    return _simple_face_memory


def init_bridge(shared_state, robot, face_app=None):
    global _shared_state, _robot, _face_app
    _shared_state = shared_state
    _robot = robot
    _face_app = face_app
    print("✅ Robot bridge initialized")


def set_face_app(face_app):
    global _face_app
    _face_app = face_app
    print("✅ Face app connected to robot bridge")


def get_shared_state():
    return _shared_state


def get_robot():
    return _robot


def get_face_app():
    return _face_app


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def _save_name_to_buddy_memory(name):
    """
    Save recognized/enrolled name into root buddy_memory.json used by speech.py.
    """
    memory_path = "buddy_memory.json"

    default = {
        "name": None,
        "conversation_count": 0,
        "likes": [],
        "dislikes": [],
        "recent_topics": [],
        "emotion_pattern": [],
        "last_user_question": "",
        "last_bot_answer": "",
        "last_topic": "",
        "saved_questions": []
    }

    try:
        if os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as file:
                memory = json.load(file)
        else:
            memory = default

        for key, value in default.items():
            memory.setdefault(key, value)

        memory["name"] = name

        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(memory, file, indent=2, ensure_ascii=False)

        print(f"✅ Saved name to Buddy memory: {name}")

    except Exception as error:
        print(f"⚠️ Could not save Buddy memory name: {error}")


def _extract_self_enrollment_name(text):
    """
    Extract name from natural phrases.

    Supported:
    - this is me my name is Ashandth
    - buddy this is me my name is Ashandth
    - remember my face my name is Ashandth
    - save my face call me Ashandth
    - look at me I am Ashandth
    - I am Ashandth remember my face
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    enrollment_triggers = [
        "this is me",
        "remember my face",
        "save my face",
        "learn my face",
        "register my face",
        "enroll me",
        "recognize me",
        "look at me",
        "see me",
        "scan my face",
        "save me",
        "remember me"
    ]

    has_trigger = any(trigger in text_lower for trigger in enrollment_triggers)

    name_patterns = [
        r"my name is\s+(.+)",
        r"call me\s+(.+)",
        r"i am\s+(.+)",
        r"i'm\s+(.+)"
    ]

    extracted_name = None

    for pattern in name_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted_name = match.group(1).strip()
            break

    if not extracted_name:
        return None

    # Allow "I am Ashandth remember my face" even if trigger appears after name.
    if not has_trigger:
        return None

    # Remove trailing filler words.
    extracted_name = re.sub(
        r"\b(please|now|buddy|okay|ok|save it|remember it|this is me|remember my face|save my face|learn my face|register my face|enroll me|recognize me|look at me|scan my face|save me|remember me)\b",
        "",
        extracted_name
    ).strip()

    extracted_name = re.sub(r"[^a-zA-Z\s]", "", extracted_name).strip()

    bad_names = {
        "good", "fine", "okay", "ok", "happy", "sad", "ready",
        "here", "me", "my", "your", "the", "a", "an"
    }

    if not extracted_name:
        return None

    if extracted_name.lower() in bad_names:
        return None

    return extracted_name.title()


def _save_enrollment_photos(name, frames_and_faces):
    try:
        safe_name = name.replace(" ", "_")
        base_dir = os.path.join(
            "faceRecAndEmotion",
            "database",
            "voice_enrolled_faces",
            safe_name
        )
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, item in enumerate(frames_and_faces):
            frame = item.get("frame")
            face_img = item.get("face")

            if frame is not None:
                full_path = os.path.join(base_dir, f"{timestamp}_{idx:02d}_full.jpg")
                cv2.imwrite(full_path, frame)

            if face_img is not None:
                face_path = os.path.join(base_dir, f"{timestamp}_{idx:02d}_face.jpg")
                cv2.imwrite(face_path, face_img)

        print(f"✅ Saved enrollment photos for {name}")

    except Exception as error:
        print(f"⚠️ Could not save enrolled face photos: {error}")


def enroll_current_face_from_voice(name, seconds=4, target_samples=20):
    """
    Video-call style enrollment.

    Captures multiple fresh camera frames for a few seconds while user looks at camera.
    Detects face repeatedly.
    Saves photos.
    Creates embeddings.
    Saves to face database.
    Saves name to Buddy memory.
    """
    print(f"🎙️ Voice enrollment started for: {name}")

    if _shared_state is None:
        return False, "My eye system is not connected yet."

    if _face_app is None:
        return False, "My face learning system is not ready yet."

    start_time = time.time()
    face_samples = []
    frames_and_faces = []

    last_capture_time = 0

    while time.time() - start_time < seconds and len(face_samples) < target_samples:
        frame = _shared_state.get_latest_frame(max_age_seconds=2)

        if frame is None:
            time.sleep(0.1)
            continue

        # Avoid capturing the exact same frame too quickly.
        if time.time() - last_capture_time < 0.15:
            time.sleep(0.05)
            continue

        last_capture_time = time.time()

        faces = _face_app.face_recognizer.detect_faces(frame)

        if not faces:
            print("👀 Enrollment: no face in this frame")
            time.sleep(0.1)
            continue

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        bbox = faces[0]

        face_img = _face_app.face_recognizer.extract_face(frame, bbox)

        if face_img is None:
            print("👀 Enrollment: face crop failed")
            time.sleep(0.1)
            continue

        if not _face_app.face_recognizer.validate_face(face_img):
            print("👀 Enrollment: face not valid/clear")
            time.sleep(0.1)
            continue

        face_samples.append(face_img)
        frames_and_faces.append({
            "frame": frame,
            "face": face_img
        })

        print(f"📸 Enrollment sample {len(face_samples)}/{target_samples}")

        time.sleep(0.08)

    if len(face_samples) < 3:
        return False, (
            "I can hear you, but I could not capture enough clear face samples. "
            "Please keep your face in the camera with better light and say it again."
        )

    
    # First try old embedding database if available.
    saved_with_embedding_db = False

    try:
        embeddings = _face_app.face_recognizer.get_embeddings_batch(face_samples)
    except Exception as error:
        print(f"⚠️ Embedding creation failed, using LBPH fallback: {error}")
        embeddings = []

    if len(embeddings) >= 1:
        try:
            _face_app.database.add_person(name, embeddings)
            saved_with_embedding_db = True
            print("✅ Saved face using embedding database")
        except Exception as error:
            print(f"⚠️ Embedding database save failed, using LBPH fallback: {error}")

        # Always save to LBPH too. This works without face_model.keras.
    try:
        simple_memory = get_simple_face_memory()
        saved_count = simple_memory.add_person(name, face_samples)

        if saved_count < 1 and not saved_with_embedding_db:
            return False, "I saw your face, but I could not save enough clear face samples."

        print(f"✅ Saved {saved_count} face samples using LBPH fallback")

    except Exception as error:
        print(f"⚠️ LBPH save failed: {error}")

        if not saved_with_embedding_db:
            return False, (
                "I saw your face, but I could not save it. "
                "Please install opencv-contrib-python and try again."
            )


    _save_enrollment_photos(name, frames_and_faces)
    _save_name_to_buddy_memory(name)

    if _shared_state:
        _shared_state.set_face_status(
            person=name,
            emotion="neutral",
            confidence=100,
            source="voice_enrollment"
        )

    if _robot:
        _robot.react_to_emotion("happy", name)

    return True, (
        f"Nice to meet you {name}. I saved your name and face. "
        f"I captured {len(face_samples)} face samples, so next time I see you, I will try to recognize you."
    )


def handle_robot_voice_command(text):
    """
    Returns a response string if this was a robot/vision command.
    Returns None if speech.py should continue normal Buddy brain logic.
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    print(f"🔗 Bridge heard: {text_lower}")

    # ============================================================
    # VOICE FACE ENROLLMENT
    # ============================================================

    enrollment_name = _extract_self_enrollment_name(text)

    if enrollment_name:
        print(f"🔗 Enrollment command detected. Name: {enrollment_name}")
        success, response = enroll_current_face_from_voice(enrollment_name)
        return response

    if _robot is None:
        return None

    # ============================================================
    # MOVEMENT COMMANDS
    # ============================================================

    if text_lower in [
        "move forward", "go forward", "walk forward", "come forward",
        "buddy move forward", "buddy go forward"
    ]:
        _robot.move_forward()
        return "Moving forward."

    if text_lower in [
        "move backward", "go back", "back up", "move back",
        "buddy move backward", "buddy go back"
    ]:
        _robot.move_backward()
        return "Moving backward."

    if text_lower in ["turn left", "go left", "buddy turn left"]:
        _robot.turn_left()
        return "Turning left."

    if text_lower in ["turn right", "go right", "buddy turn right"]:
        _robot.turn_right()
        return "Turning right."

    if text_lower in [
        "robot stop", "stop robot", "stop moving", "freeze",
        "buddy stop moving", "stop movement"
    ]:
        _robot.stop()
        return "Robot stopped."

    # ============================================================
    # DOG POSE / EXPRESSION COMMANDS
    # ============================================================

    if text_lower in ["sit", "sit down", "buddy sit", "buddy sit down"]:
        _robot.sit()
        return "Sitting down."

    if text_lower in ["stand", "stand up", "buddy stand", "buddy stand up"]:
        _robot.stand()
        return "Standing up."

    if text_lower in [
        "wag tail", "wag your tail", "be happy",
        "buddy wag tail", "buddy wag your tail"
    ]:
        _robot.wag_tail()
        return "Wagging my tail."

    if "turn your head left" in text_lower or "look left" in text_lower:
        _robot.turn_head(-45)
        return "Turning my head left."

    if "turn your head right" in text_lower or "look right" in text_lower:
        _robot.turn_head(45)
        return "Turning my head right."

    if "look forward" in text_lower or "center your head" in text_lower or "look straight" in text_lower:
        _robot.turn_head(0)
        return "Looking forward."

    # ============================================================
    # FACE / VISION QUESTIONS
    # ============================================================

    if _contains_any(text_lower, [
        "who do you see",
        "can you see me",
        "who is in front",
        "do you recognize me",
        "who am i"
    ]):
        if _shared_state is None:
            return "My vision system is not connected."

        snapshot = _shared_state.get_snapshot()
        person = snapshot.get("current_person")
        emotion = snapshot.get("current_emotion")
        confidence = snapshot.get("current_face_confidence", 0)

        if person:
            return f"I can see {person}. They look {emotion}, with about {confidence:.0f} percent confidence."

        return f"I can see someone, but I do not recognize them. The emotion looks {emotion}."

    if _contains_any(text_lower, [
        "what emotion",
        "how do i look",
        "am i happy",
        "am i sad",
        "what is my emotion",
        "detect my emotion"
    ]):
        if _shared_state is None:
            return "My emotion vision system is not connected."

        snapshot = _shared_state.get_snapshot()
        emotion = snapshot.get("current_emotion")
        confidence = snapshot.get("current_face_confidence", 0)

        return f"You look {emotion}, with about {confidence:.0f} percent confidence."

    if _contains_any(text_lower, [
        "robot status",
        "your status",
        "dog status",
        "what is your status"
    ]):
        status = _robot.get_status()
        person = status.get("person") or "no known person"
        return (
            f"My current emotion is {status['emotion']}. "
            f"I see {person}. "
            f"My eyes are {status['left_eye']} and {status['right_eye']}. "
            f"My tail is {status['tail_wag']}. "
            f"Moving is {status['is_moving']}."
        )

    return None