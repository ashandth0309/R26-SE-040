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
from faceRecAndEmotion.robot_bridge import init_bridge
from faceRecAndEmotion.main import SmartAIDogRobot

import speech


def run_face_system(shared_state, robot):
    app = SmartAIDogRobot(shared_state=shared_state, robot=robot)
    app.run()


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

    init_bridge(shared_state, robot)

    face_thread = threading.Thread(
        target=run_face_system,
        args=(shared_state, robot),
        daemon=True
    )

    voice_thread = threading.Thread(
        target=run_voice_system,
        daemon=False
    )

    face_thread.start()
    time.sleep(2)

    voice_thread.start()
    voice_thread.join()

    shared_state.stop()
    robot.cleanup()

    print("👋 Full robot system stopped")


if __name__ == "__main__":
    main()