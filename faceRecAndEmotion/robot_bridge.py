"""
Bridge used by root speech.py from Ashandth branch.

speech.py calls:
    handle_robot_voice_command(text)

This connects voice commands to:
- robot movement
- eye/head/tail commands
- face recognition status
- emotion status
"""

_shared_state = None
_robot = None


def init_bridge(shared_state, robot):
    global _shared_state, _robot
    _shared_state = shared_state
    _robot = robot


def get_shared_state():
    return _shared_state


def get_robot():
    return _robot


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def handle_robot_voice_command(text):
    """
    Returns a response string if this was a robot/vision command.
    Returns None if speech.py should continue normal Buddy brain logic.
    """
    if not text:
        return None

    text_lower = text.lower().strip()

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