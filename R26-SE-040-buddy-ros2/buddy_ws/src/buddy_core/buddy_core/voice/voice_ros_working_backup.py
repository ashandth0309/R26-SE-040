import json
import queue
import subprocess
import threading
import time
import wave

import numpy as np
import rclpy
import sounddevice as sd
from piper import PiperVoice
from rclpy.node import Node
from vosk import KaldiRecognizer, Model

from buddy_interfaces.msg import MovementCommand


VOSK_MODEL = "/home/ashandth/R26-SE-040/models/vosk-model-small-en-us-0.15"
PIPER_MODEL = "/home/ashandth/R26-SE-040/models/piper/en_US-lessac-medium.onnx"

MIC_DEVICE = 3
MIC_RATE = 48000
VOSK_RATE = 16000
CHANNELS = 2

MOVE_SPEED = 0.20
TURN_SPEED = 0.60

COMMANDS = [
    "buddy",
    "buddy hello",
    "buddy go forward",
    "buddy come back",
    "buddy go back",
    "buddy go backward",
    "buddy go backwards",
    "buddy reverse",
    "buddy go reverse",
    "buddy move back",
    "buddy move backward",
    "buddy turn left",
    "buddy turn right",
    "buddy stop",
    "hello",
    "go forward",
    "come back",
    "go back",
    "go backward",
    "go backwards",
    "reverse",
    "go reverse",
    "move back",
    "move backward",
    "turn left",
    "turn right",
    "stop",
    "[unk]"
]


class BuddyVoiceNode(Node):
    def __init__(self):
        super().__init__("buddy_voice")

        self.publisher = self.create_publisher(
            MovementCommand,
            "/movement/request",
            10
        )

        self.sequence_id = 0
        self.active_command = MovementCommand.COMMAND_STOP
        self.is_speaking = False

        self.audio_queue = queue.Queue()

        print("Loading BUDDY ears...")
        self.vosk_model = Model(VOSK_MODEL)

        self.recognizer = KaldiRecognizer(
            self.vosk_model,
            VOSK_RATE,
            json.dumps(COMMANDS)
        )

        print("Loading BUDDY voice...")
        self.voice = PiperVoice.load(PIPER_MODEL)

        self.create_timer(
            0.10,
            self.publish_active_command
        )

        print("BUDDY READY")


    def publish_command(
        self,
        command,
        linear_speed=0.0,
        angular_speed=0.0
    ):
        self.sequence_id += 1

        msg = MovementCommand()

        msg.stamp = self.get_clock().now().to_msg()
        msg.sequence_id = self.sequence_id
        msg.source = "buddy_voice"
        msg.command = command

        msg.linear_speed = float(linear_speed)
        msg.angular_speed = float(angular_speed)

        msg.valid_for.sec = 0
        msg.valid_for.nanosec = 300_000_000

        msg.emergency = False

        self.publisher.publish(msg)


    def publish_active_command(self):
        if self.active_command == MovementCommand.COMMAND_STOP:
            return

        if self.active_command == MovementCommand.COMMAND_FORWARD:
            self.publish_command(
                MovementCommand.COMMAND_FORWARD,
                linear_speed=MOVE_SPEED
            )

        elif self.active_command == MovementCommand.COMMAND_BACKWARD:
            self.publish_command(
                MovementCommand.COMMAND_BACKWARD,
                linear_speed=MOVE_SPEED
            )

        elif self.active_command == MovementCommand.COMMAND_ROTATE_LEFT:
            self.publish_command(
                MovementCommand.COMMAND_ROTATE_LEFT,
                angular_speed=TURN_SPEED
            )

        elif self.active_command == MovementCommand.COMMAND_ROTATE_RIGHT:
            self.publish_command(
                MovementCommand.COMMAND_ROTATE_RIGHT,
                angular_speed=TURN_SPEED
            )


    def stop_motion(self):
        self.active_command = MovementCommand.COMMAND_STOP

        self.publish_command(
            MovementCommand.COMMAND_STOP
        )


    def speak(self, text):
        print("BUDDY:", text)

        self.is_speaking = True

        try:
            wav_path = "/tmp/buddy_voice_reply.wav"

            with wave.open(wav_path, "wb") as wav_file:
                self.voice.synthesize_wav(
                    text,
                    wav_file
                )

            subprocess.run(
                [
                    "aplay",
                    "-q",
                    "-D",
                    "plughw:1,0",
                    wav_path
                ],
                check=False
            )

        finally:
            self.is_speaking = False

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break


    def handle_text(self, text):
        text = text.lower().strip()

        print("YOU:", text)

        if text in (
            "buddy stop",
            "stop",
        ):
            self.stop_motion()
            self.speak("Stopping.")
            return

        if text in (
            "buddy go forward",
            "go forward",
        ):
            self.active_command = MovementCommand.COMMAND_FORWARD
            self.speak("Going forward.")
            return

        if text in (
            "buddy come back",
            "buddy go back",
            "buddy go backward",
            "buddy go backwards",
            "buddy reverse",
            "buddy go reverse",
            "buddy move back",
            "buddy move backward",
            "come back",
            "go back",
            "go backward",
            "go backwards",
            "reverse",
            "go reverse",
            "move back",
            "move backward",
        ):
            self.active_command = MovementCommand.COMMAND_BACKWARD
            self.speak("Coming back.")
            return

        if text in (
            "buddy turn left",
            "turn left",
        ):
            self.active_command = MovementCommand.COMMAND_ROTATE_LEFT
            self.speak("Turning left.")
            return

        if text in (
            "buddy turn right",
            "turn right",
        ):
            self.active_command = MovementCommand.COMMAND_ROTATE_RIGHT
            self.speak("Turning right.")
            return

        if text in (
            "buddy hello",
            "hello",
        ):
            self.speak("Hello!")
            return

        if text == "buddy":
            self.speak("Yes?")


    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):
        if self.is_speaking:
            return

        self.audio_queue.put(
            indata.copy()
        )


    def listen_loop(self):
        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=MIC_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=12000,
            callback=self.audio_callback,
        ):
            while rclpy.ok():

                block = self.audio_queue.get()

                mono = (
                    block.astype(np.int32)
                    .mean(axis=1)
                )

                mono = mono[::3]

                mono = np.clip(
                    mono,
                    -32768,
                    32767
                ).astype(np.int16)

                if self.recognizer.AcceptWaveform(
                    mono.tobytes()
                ):
                    result = json.loads(
                        self.recognizer.Result()
                    )

                    text = result.get(
                        "text",
                        ""
                    ).strip()

                    if (
                        not text
                        or text == "[unk]"
                    ):
                        continue

                    self.handle_text(text)


def main():
    rclpy.init()

    node = BuddyVoiceNode()

    listener = threading.Thread(
        target=node.listen_loop,
        daemon=True
    )

    listener.start()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_motion()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
