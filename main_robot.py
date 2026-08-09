"""
Main integrated launcher for Smart AI Dog Robot.

Voice:
    speech.py from Ashandth branch

Face/Eye:
    faceRecAndEmotion from final-clean branch
"""

import threading
import time

from faceRecAndEmotion.shared_state import SharedRobotState
from faceRecAndEmotion.robot_controller import RobotController
from faceRecAndEmotion.robot_bridge import init_bridge, set_face_app

import speech


def run_face_system(face_app):
    try:
        face_app.run()
    except Exception as error:
        print(f"❌ Face system crashed: {error}")


def run_voice_system():
    speech.start_conversation()


def main():
    print("=" * 60)
    print("🐶 SMART AI DOG ROBOT - FULL INTEGRATION")
    print("Voice: Ashandth speech.py")
    print("Face/Eye: final-clean faceRecAndEmotion")
    print("=" * 60)

    shared_state = SharedRobotState()
    robot = RobotController(shared_state=shared_state)

    face_app = None

    try:
        from faceRecAndEmotion.main import SmartAIDogRobot
        face_app = SmartAIDogRobot(shared_state=shared_state, robot=robot)
        init_bridge(shared_state, robot, face_app)
        set_face_app(face_app)
    except Exception as error:
        print(f"❌ Could not start face app: {error}")
        print("⚠️ Voice will still run, but camera/eye commands will not work.")
        init_bridge(shared_state, robot, None)

    if face_app is not None:
        face_thread = threading.Thread(
            target=run_face_system,
            args=(face_app,),
            daemon=True
        )
        face_thread.start()
        time.sleep(2)

    voice_thread = threading.Thread(
        target=run_voice_system,
        daemon=False
    )

    voice_thread.start()
    voice_thread.join()

    shared_state.stop()
    robot.cleanup()

    print("👋 Full robot system stopped")


if __name__ == "__main__":
    main()