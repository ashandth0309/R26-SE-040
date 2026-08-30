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
from buddy_core.voice.brain import get_response


VOSK_MODEL = "/home/ashandth/R26-SE-040/models/vosk-model-small-en-us-0.15"
PIPER_MODEL = "/home/ashandth/R26-SE-040/models/piper/en_US-lessac-medium.onnx"

def find_buddy_microphone():
    """Automatically find the AB13X USB microphone that has input channels."""
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        name = str(device["name"])

        if "AB13X USB Audio" in name and device["max_input_channels"] >= 2:
            print(
                f"MICROPHONE: using device {index} - {name} "
                f"({device['max_input_channels']} input channels)"
            )
            return index

    raise RuntimeError(
        "No usable AB13X USB microphone found."
    )


MIC_DEVICE = find_buddy_microphone()
MIC_RATE = 48000
VOSK_RATE = 16000
CHANNELS = 2

MOVE_SPEED = 0.20
TURN_SPEED = 0.60

AWAKE_TIMEOUT_SECONDS = 60.0


COMMANDS = [
    "buddy",
    "buddy stop",
    "stop",

    "buddy go forward",
    "buddy move forward",
    "buddy forward",
    "go forward",
    "move forward",
    "forward",

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

    "buddy turn left",
    "turn left",

    "buddy turn right",
    "turn right",

    "[unk]",
]


WAKE_COMMANDS = [
    "buddy",
    "hey buddy",
    "hello buddy",
    "buddy wake up",
    "wake up buddy",
    "buddy stop",
    "stop",
]


class BuddyVoiceNode(Node):
    def __init__(self):
        super().__init__("buddy_voice")

        self.publisher = self.create_publisher(
            MovementCommand,
            "/movement/request",
            10,
        )

        self.sequence_id = 0
        self.active_command = MovementCommand.COMMAND_STOP

        self.is_speaking = False
        self.is_awake = False
        self.last_activity_time = 0.0
        self.last_command_time = 0.0

        self.audio_queue = queue.Queue()

        print("Loading BUDDY ears...")
        self.vosk_model = Model(VOSK_MODEL)

        self.wake_recognizer = None
        self.command_recognizer = None
        self.general_recognizer = None

        self.reset_recognizers()

        print("Loading BUDDY voice...")
        self.voice = PiperVoice.load(PIPER_MODEL)

        self.create_timer(
            0.10,
            self.publish_active_command,
        )

        self.create_timer(
            1.0,
            self.check_sleep_timeout,
        )

        print("BUDDY READY")


    def reset_recognizers(self):
        self.wake_recognizer = KaldiRecognizer(
            self.vosk_model,
            VOSK_RATE,
            json.dumps(WAKE_COMMANDS),
        )

        self.command_recognizer = KaldiRecognizer(
            self.vosk_model,
            VOSK_RATE,
            json.dumps(COMMANDS),
        )

        self.general_recognizer = KaldiRecognizer(
            self.vosk_model,
            VOSK_RATE,
        )


    def clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


    def publish_command(
        self,
        command,
        linear_speed=0.0,
        angular_speed=0.0,
    ):
        if not rclpy.ok():
            print("ROS context closed - command skipped")
            return

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
                linear_speed=MOVE_SPEED,
            )

        elif self.active_command == MovementCommand.COMMAND_BACKWARD:
            self.publish_command(
                MovementCommand.COMMAND_BACKWARD,
                linear_speed=MOVE_SPEED,
            )

        elif self.active_command == MovementCommand.COMMAND_ROTATE_LEFT:
            self.publish_command(
                MovementCommand.COMMAND_ROTATE_LEFT,
                angular_speed=TURN_SPEED,
            )

        elif self.active_command == MovementCommand.COMMAND_ROTATE_RIGHT:
            self.publish_command(
                MovementCommand.COMMAND_ROTATE_RIGHT,
                angular_speed=TURN_SPEED,
            )


    def stop_motion(self):
        self.active_command = MovementCommand.COMMAND_STOP

        self.publish_command(
            MovementCommand.COMMAND_STOP,
        )


    def wake_up(self):
        self.is_awake = True
        self.last_activity_time = time.monotonic()

        print("STATE: AWAKE")


    def go_to_sleep(self):
        if not self.is_awake:
            return

        self.stop_motion()

        self.is_awake = False
        self.last_activity_time = 0.0

        self.clear_audio_queue()
        self.reset_recognizers()

        print("STATE: SLEEPING")


    def check_sleep_timeout(self):
        if not self.is_awake:
            return

        if self.is_speaking:
            return

        idle_time = (
            time.monotonic()
            - self.last_activity_time
        )

        if idle_time >= AWAKE_TIMEOUT_SECONDS:
            self.go_to_sleep()


    def speak(self, text):
        if not text:
            return

        print("BUDDY:", text)

        self.is_speaking = True

        try:
            wav_path = "/tmp/buddy_voice_reply.wav"

            with wave.open(
                wav_path,
                "wb",
            ) as wav_file:
                self.voice.synthesize_wav(
                    text,
                    wav_file,
                )

            subprocess.run(
                [
                    "aplay",
                    "-q",
                    "-D",
                    "plughw:1,0",
                    wav_path,
                ],
                check=False,
            )

        finally:
            self.is_speaking = False

            self.clear_audio_queue()
            self.reset_recognizers()

            if self.is_awake:
                self.last_activity_time = time.monotonic()


    def handle_movement_command(self, text):
        text = text.lower().strip()

        if text in (
            "buddy stop",
            "stop",
        ):
            self.stop_motion()
            self.last_command_time = time.monotonic()

            if not self.is_awake:
                self.wake_up()

            self.speak("Stopping.")
            return True

        if text in (
            "buddy go forward",
            "buddy move forward",
            "buddy forward",
            "go forward",
            "move forward",
            "forward",
        ):
            if not self.is_awake:
                return False

            self.active_command = MovementCommand.COMMAND_FORWARD
            self.last_command_time = time.monotonic()

            self.speak("Going forward.")
            return True

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
            if not self.is_awake:
                return False

            self.active_command = MovementCommand.COMMAND_BACKWARD
            self.last_command_time = time.monotonic()

            self.speak("Coming back.")
            return True

        if text in (
            "buddy turn left",
            "turn left",
        ):
            if not self.is_awake:
                return False

            self.active_command = MovementCommand.COMMAND_ROTATE_LEFT
            self.last_command_time = time.monotonic()

            self.speak("Turning left.")
            return True

        if text in (
            "buddy turn right",
            "turn right",
        ):
            if not self.is_awake:
                return False

            self.active_command = MovementCommand.COMMAND_ROTATE_RIGHT
            self.last_command_time = time.monotonic()

            self.speak("Turning right.")
            return True

        return False


    def handle_general_text(self, text):
        text = text.strip()

        if not text:
            return

        print("YOU:", text)

        self.last_activity_time = time.monotonic()

        response = get_response(text)

        if response:
            self.speak(response)


    def handle_sleeping_audio(self, audio_bytes):
        if not self.wake_recognizer.AcceptWaveform(
            audio_bytes
        ):
            return

        result = json.loads(
            self.wake_recognizer.Result()
        )

        text = result.get(
            "text",
            "",
        ).strip().lower()

        if not text:
            return

        print("WAKE HEARD:", text)

        if text in (
            "buddy stop",
            "stop",
        ):
            self.handle_movement_command(text)
            return

        if text in (
            "buddy",
            "hey buddy",
            "hello buddy",
            "buddy wake up",
            "wake up buddy",
        ):
            self.wake_up()
            self.speak("Yes?")


    def handle_awake_audio(self, audio_bytes):
        command_finished = (
            self.command_recognizer.AcceptWaveform(
                audio_bytes
            )
        )

        general_finished = (
            self.general_recognizer.AcceptWaveform(
                audio_bytes
            )
        )

        if command_finished:
            result = json.loads(
                self.command_recognizer.Result()
            )

            command_text = result.get(
                "text",
                "",
            ).strip().lower()

            if (
                command_text
                and command_text != "[unk]"
            ):
                print(
                    "COMMAND HEARD:",
                    command_text,
                )

                if self.handle_movement_command(
                    command_text
                ):
                    return

        if general_finished:
            result = json.loads(
                self.general_recognizer.Result()
            )

            general_text = result.get(
                "text",
                "",
            ).strip()

            if not general_text:
                return

            if (
                time.monotonic()
                - self.last_command_time
                < 1.0
            ):
                return

            lower = general_text.lower().strip()

            if lower in (
                "buddy",
                "hey buddy",
                "hello buddy",
            ):
                self.last_activity_time = time.monotonic()
                self.speak("Yes?")
                return

            self.handle_general_text(
                general_text
            )


    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
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
                    32767,
                ).astype(np.int16)

                audio_bytes = mono.tobytes()

                if self.is_speaking:
                    continue

                if self.is_awake:
                    self.handle_awake_audio(
                        audio_bytes
                    )
                else:
                    self.handle_sleeping_audio(
                        audio_bytes
                    )


def main():
    rclpy.init()

    node = BuddyVoiceNode()

    listener = threading.Thread(
        target=node.listen_loop,
        daemon=True,
    )

    listener.start()

    try:
        node.speak(
            "Hi there. Buddy is ready."
        )

        print("STATE: SLEEPING")

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            try:
                node.stop_motion()
            except Exception as exc:
                print("Shutdown STOP skipped:", exc)

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
