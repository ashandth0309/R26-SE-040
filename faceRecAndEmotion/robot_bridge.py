"""
Bridge used by root speech.py from Ashandth branch.

speech.py calls:
    handle_robot_voice_command(text)

This connects voice commands to:
- robot movement
- eye/head/tail commands
- face recognition status
- emotion status
- voice-based face enrollment:
  "this is me my name is Ashandth"
"""

import os
import re
from datetime import datetime

import cv2


_shared_state = None
_robot = None
_face_app = None


def init_bridge(shared_state, robot, face_app=None):
    global _shared_state, _robot, _face_app
    _shared_state = shared_state
    _robot = robot
    _face_app = face_app


def set_face_app(face_app):
    global _face_app
    _face_app = face_app


def get_shared_state():
    return _shared_state


def get_robot():
    return _robot


def get_face_app():
    return _face_app


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def _extract_self_enrollment_name(text):
    """
    Extract name from:
    - this is me my name is Ashandth
    - this is me I am Ashandth
    - buddy this is me call me Ashandth
    - remember my face my name is Ashandth
    """
    text_lower = text.lower().strip()

    enrollment_triggers = [
        "this is me",
        "remember my face",
        "save my face",
        "learn my face",
        "register my face",
        "enroll me",
        "recognize me"
    ]

    if not any(trigger in text_lower for trigger in enrollment_triggers):
        return None

    patterns = [
        r"my name is\s+(.+)",
        r"call me\s+(.+)",
        r"i am\s+(.+)",
        r"i'm\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).strip()

            # Remove trailing filler words
            name = re.sub(
                r"\b(please|now|buddy|okay|ok|save it|remember it|this is me)\b",
                "",
                name
            ).strip()

            name = re.sub(r"[^a-zA-Z\s]", "", name).strip()

            if name:
                return name.title()

    return None


def enroll_current_face_from_voice(name):
    """
    Capture the current camera frame, detect face, save photos,
    create embeddings, and add the person to the database.
    """
    if _shared_state is None:
        return False, "My eye system is not connected yet."

    if _face_app is None:
        return False, "My face learning system is not ready yet."

    frame = _shared_state.get_latest_frame(max_age_seconds=5)

    if frame is None:
        return False, "I cannot see a fresh camera image right now. Please look at the camera and try again."

    faces = _face_app.face_recognizer.detect_faces(frame)

    if not faces:
        return False, "I cannot see your face clearly. Please look at the camera with better light."

    # Pick the largest visible face
    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    bbox = faces[0]

    face_img = _face_app.face_recognizer.extract_face(frame, bbox)

    if face_img is None or not _face_app.face_recognizer.validate_face(face_img):
        return False, "I found a face, but it is not clear enough. Please face the camera and try again."

    # Capture multiple samples from the same latest frame with slight copies.
    # If you want stronger accuracy, say the command several times from different angles.
    face_samples = [face_img]

    # Try to create extra samples from recent frame crop by slight image changes.
    try:
        for scale in [1.02, 0.98, 1.04, 0.96]:
            resized = cv2.resize(face_img, None, fx=scale, fy=scale)
            resized = cv2.resize(resized, (face_img.shape[1], face_img.shape[0]))
            face_samples.append(resized)
    except Exception:
        pass

    embeddings = _face_app.face_recognizer.get_embeddings_batch(face_samples)

    if len(embeddings) < 1:
        return False, "I saw you, but I could not create a face memory. Please try again with better lighting."

    _face_app.database.add_person(name, embeddings)

    # Save photo evidence
    try:
        base_dir = os.path.join("faceRecAndEmotion", "database", "voice_enrolled_faces", name.replace(" ", "_"))
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(base_dir, f"{timestamp}_full.jpg")
        face_path = os.path.join(base_dir, f"{timestamp}_face.jpg")

        cv2.imwrite(full_path, frame)
        cv2.imwrite(face_path, face_img)
    except Exception as error:
        print(f"Could not save enrolled face photo: {error}")

    if _shared_state:
        _shared_state.set_face_status(
            person=name,
            emotion="neutral",
            confidence=100,
            source="voice_enrollment"
        )

    if _robot:
        _robot.react_to_emotion("happy", name)

    return True, f"Nice to meet you {name}. I saved your name and face. Next time I see you, I will try to recognize you."


def handle_robot_voice_command(text):
    """
    Returns a response string if this was a robot/vision command.
    Returns None if speech.py should continue normal Buddy brain logic.
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    # ============================================================
    # VOICE FACE ENROLLMENT
    # ============================================================

    enrollment_name = _extract_self_enrollment_name(text)

    if enrollment_name:
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