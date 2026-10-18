"""ROS 2 node for the voice companion engine.

Run as ``ros2 run buddy_core voice_companion`` (opt-in; the validated
``buddy_voice.py`` demo node is untouched and must NOT run at the same time:
two nodes would share one microphone and both publish ``/movement/request``).

Topics (names are parameters):

* publishes  ``/movement/request``     MovementCommand  (via the safety
  supervisor, never ``/movement/safe``)
* publishes  ``/navigation/command``   only if navigation is enabled and
  the navigation package is installed
* publishes  ``/voice/command``        VoiceCommand (command intents only,
  no audio)
* publishes  ``/voice/status``         JSON status (no speech text)
* subscribes ``/safety_state``         emergency stop vetoes new motion
* subscribes ``/emotion/observation``  facial-expression hint (name is an
  assumption; set the parameter to match the vision node)
* subscribes ``/navigation/status``    feedback for spoken answers
* subscribes ``/voice/say``            SpeechRequest; the only way other
  modules make Buddy speak (this node owns the speaker)
* service    ``/voice/mute``           std_srvs/SetBool; the only way to
  unmute
* client     ``/coordinator/route_utterance``  optional (``enable_gateway``);
  bounded and fails open, see ``RosUtteranceGateway``

Integration: with the coordinator running, launch this node with
``movement_topic`` and ``navigation_topic`` pointing at the coordinator's
intake topics.  STOP is then also published directly on
``direct_stop_topic`` / ``direct_nav_stop_topic`` so stopping never
depends on the coordinator being alive.

This file is UNVERIFIED under real ROS 2; tests use strict stubs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
from rclpy.node import Node

from buddy_interfaces.msg import (
    EmotionObservation,
    MovementCommand,
    SafetyState,
    SpeechRequest,
    VoiceCommand,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool

from buddy_core.voice.config import VoiceSettings, apply_profile, load_settings
from buddy_core.voice.engine_factory import (
    ModelMissing,
    build_companion,
    real_backends,
    simulated_backends,
)
from buddy_core.voice.runner import EngineRunner
from buddy_core.voice.speaker_manager import Priority
from buddy_core.voice.voice_engine import GatewayAnswer

# NavigationStatus.msg: IDLE=0 RUNNING=1 PAUSED=2 SUCCEEDED=3 FAILED=4
# CANCELLED=5.  A test compares this table with the .msg file.
NAV_STATE_NAMES = {0: "idle", 1: "active", 2: "paused", 3: "done",
                   4: "failed", 5: "cancelled"}

MOVE_STOP = 0
NAV_STOP = 0
DEFAULT_SAY_TTL_S = 5.0


class RosMovementSink:
    """Publishes ``MovementCommand`` to ``/movement/request``.

    ``stop_publisher`` is an independent route straight to the safety
    supervisor.  A STOP always goes there too, so it works even if the
    coordinator that normally relays movement is down.
    """

    def __init__(self, node: Node, publisher, stop_publisher=None) -> None:
        self._node = node
        self._pub = publisher
        self._stop_pub = stop_publisher

    def send_movement(self, command, linear, angular, valid_for_s,
                      emergency, sequence_id, source) -> None:
        msg = MovementCommand()
        msg.stamp = self._node.get_clock().now().to_msg()
        msg.sequence_id = int(sequence_id)
        msg.source = source
        msg.command = int(command)
        msg.linear_speed = float(linear)
        msg.angular_speed = float(angular)
        msg.valid_for.sec = int(valid_for_s)
        msg.valid_for.nanosec = int((valid_for_s % 1.0) * 1e9)
        msg.emergency = bool(emergency)
        self._pub.publish(msg)

        if int(command) == MOVE_STOP and self._stop_pub is not None \
                and self._stop_pub is not self._pub:
            self._stop_pub.publish(msg)


class RosNavigationSink:
    """Publishes ``NavigationCommand`` (from the navigation package)."""

    def __init__(self, node: Node, publisher, msg_type,
                 stop_publisher=None) -> None:
        self._node = node
        self._pub = publisher
        self._type = msg_type
        self._stop_pub = stop_publisher

    def send_navigation(self, command, request_id, source, duration_s,
                        location_name) -> None:
        msg = self._type()
        msg.stamp = self._node.get_clock().now().to_msg()
        msg.request_id = request_id
        msg.source = source
        msg.command = int(command)
        msg.duration_s = float(duration_s)
        msg.location_name = location_name
        self._pub.publish(msg)

        if int(command) == NAV_STOP and self._stop_pub is not None \
                and self._stop_pub is not self._pub:
            self._stop_pub.publish(msg)


class RosUtteranceGateway:
    """Asks the coordinator whether it will handle an utterance.

    Called on the engine's executor thread, never on the audio path.  It
    fails open: if the service is not up, errors, or does not answer
    within ``timeout_s``, it returns None and the voice engine answers
    the utterance as ordinary conversation.
    """

    def __init__(self, client, request_type, timeout_s: float) -> None:
        self._client = client
        self._type = request_type
        self._timeout = timeout_s
        self._seq = 0
        self.timeouts = 0
        self.unavailable = 0

    def __call__(self, text: str, confidence: float | None):
        if not self._client.service_is_ready():
            self.unavailable += 1

            return None

        self._seq += 1
        request = self._type.Request()
        request.request_id = f"voice-{self._seq}"
        request.text = text
        request.confidence = float(confidence or 0.0)
        request.speaker_id = ""
        done = threading.Event()
        future = self._client.call_async(request)
        future.add_done_callback(lambda _f: done.set())

        if not done.wait(self._timeout):
            self.timeouts += 1
            future.cancel()

            return None

        response = future.result()

        if response is None:
            return None

        return GatewayAnswer(bool(response.claimed), str(response.reply),
                             str(response.intent))


def _default_config_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("buddy_core")) / \
            "config" / "voice_companion.yaml"

        return str(path) if path.exists() else ""
    except Exception:
        return ""


class VoiceCompanionNode(Node):
    """Thin ROS wrapper around the pure-Python companion."""

    def __init__(self) -> None:
        super().__init__("buddy_voice_companion")
        p = self._param

        config_path = p("config_path", "") or _default_config_path()
        settings: VoiceSettings = load_settings(config_path or None)
        profile = p("profile", "")

        if profile:
            settings.profile = profile

        settings.models.vosk_model_path = p(
            "vosk_model_path", settings.models.vosk_model_path)
        settings.models.piper_model_path = p(
            "piper_model_path", settings.models.piper_model_path)
        settings.memory.db_path = p("memory_db_path",
                                    settings.memory.db_path)
        settings.navigation.enabled = bool(
            p("enable_navigation", settings.navigation.enabled))
        settings.online.cloud_enabled = bool(
            p("enable_cloud", settings.online.cloud_enabled))
        apply_profile(settings)
        self.settings = settings

        if not p("allow_coexist", False) and \
                "buddy_voice" in self._other_node_names():
            raise RuntimeError(
                "The 'buddy_voice' node is running. Stop it first: two "
                "voice nodes cannot share the microphone and "
                "/movement/request.")

        movement_topic = p("movement_topic", "/movement/request")
        direct_stop_topic = p("direct_stop_topic", "/movement/request")
        self._movement_pub = self.create_publisher(
            MovementCommand, movement_topic, 10)
        self._stop_pub = self._movement_pub \
            if direct_stop_topic == movement_topic \
            else self.create_publisher(
                MovementCommand, direct_stop_topic, 10)
        self._command_pub = self.create_publisher(
            VoiceCommand, p("voice_command_topic", "/voice/command"), 10)
        self._status_pub = self.create_publisher(
            String, p("status_topic", "/voice/status"), 10)
        nav_sink = self._make_navigation_sink(settings)

        backends = simulated_backends() if p("dry_run", False) \
            else real_backends(settings)
        self._publish_commands = bool(p("publish_voice_command", True))
        self.executor_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="voice_reply")
        self.gateway = self._make_gateway()
        self.say_dropped = {"stale": 0, "quiet": 0}
        self.companion = build_companion(
            settings, backends,
            RosMovementSink(self, self._movement_pub, self._stop_pub),
            nav_sink, executor=self.executor_pool,
            command_observer=self._on_command,
            utterance_gateway=self.gateway)
        self.runner = EngineRunner(self.companion)

        self.create_subscription(
            SafetyState, p("safety_topic", "/safety_state"),
            self._on_safety, 10)
        self.create_subscription(
            EmotionObservation, p("emotion_topic", "/emotion/observation"),
            self._on_emotion, 10)
        self.create_subscription(
            SpeechRequest, p("say_topic", "/voice/say"), self._on_say, 10)
        self.create_service(SetBool, p("mute_service", "/voice/mute"),
                            self._on_mute)
        self.create_timer(float(p("status_period_s", 1.0)),
                          self._publish_status)

        for note in self.companion.notes:
            self.get_logger().info(note)

        if p("autostart_runner", True):
            self.runner.start()

        self.get_logger().info(
            f"voice companion ready (profile={settings.profile}, "
            f"cloud={'on' if self.companion.cloud_state['on'] else 'off'}, "
            f"navigation={'on' if nav_sink else 'off'})")

    # ----- setup helpers -----

    def _param(self, name, default):
        self.declare_parameter(name, default)

        return self.get_parameter(name).value

    def _other_node_names(self) -> list[str]:
        try:
            return list(self.get_node_names())
        except Exception:
            return []

    def _make_navigation_sink(self, settings):
        if not settings.navigation.enabled:
            return None

        try:
            from buddy_nav_interfaces.msg import NavigationCommand
            from buddy_nav_interfaces.msg import NavigationStatus
        except ImportError:
            self.get_logger().warn(
                "navigation requested but buddy_nav_interfaces is not "
                "installed; navigation commands stay disabled")
            settings.navigation.enabled = False

            return None

        nav_topic = self._param("navigation_topic", "/navigation/command")
        nav_stop_topic = self._param(
            "direct_nav_stop_topic", "/navigation/command")
        pub = self.create_publisher(NavigationCommand, nav_topic, 10)
        stop_pub = pub if nav_stop_topic == nav_topic \
            else self.create_publisher(
                NavigationCommand, nav_stop_topic, 10)
        self.create_subscription(
            NavigationStatus, self._param(
                "navigation_status_topic", "/navigation/status"),
            self._on_nav_status, 10)

        return RosNavigationSink(self, pub, NavigationCommand, stop_pub)

    def _make_gateway(self):
        if not self._param("enable_gateway", False):
            return None

        try:
            from buddy_interfaces.srv import RouteUtterance
        except ImportError:
            self.get_logger().warn(
                "enable_gateway requested but RouteUtterance is not "
                "installed; the gateway stays off")

            return None

        client = self.create_client(
            RouteUtterance, self._param(
                "gateway_service", "/coordinator/route_utterance"))

        return RosUtteranceGateway(
            client, RouteUtterance,
            float(self._param("gateway_timeout_s", 3.0)))

    # ----- callbacks (hand work to the engine thread) -----

    def _on_safety(self, msg) -> None:
        inhibit = bool(msg.emergency_stop_active) or not bool(
            msg.movement_allowed)
        self.runner.submit(
            lambda: self.companion.router.set_inhibited(inhibit))

    def _on_emotion(self, msg) -> None:
        expression, conf = str(msg.expression), float(msg.confidence)
        self.runner.submit(
            lambda: self.companion.emotion.observe_vision(
                expression, conf))

    def _on_say(self, msg) -> None:
        """Speak a request from another module through the one speaker."""

        text = str(msg.text).strip()

        if not text:
            return

        priority = Priority(min(max(int(msg.priority), 0), 3))
        ttl = float(msg.ttl_s) or DEFAULT_SAY_TTL_S
        received = time.monotonic()
        interrupt = bool(msg.interrupt) and priority <= Priority.HIGH

        def speak() -> None:
            engine = self.companion.engine

            if time.monotonic() - received > ttl:
                self.say_dropped["stale"] += 1

                return

            # "Let's be quiet" silences everything except safety speech.
            if engine.quiet_mode and priority > Priority.HIGH:
                self.say_dropped["quiet"] += 1

                return

            if interrupt:
                self.companion.speaker.cancel_low_priority()

            engine.say(text, priority)

        self.runner.submit(speak)

    def _on_nav_status(self, msg) -> None:
        # Field names mirror the navigation package; read defensively
        # because they are unverified against this branch.
        request_id = str(getattr(msg, "request_id", ""))
        state = NAV_STATE_NAMES.get(int(getattr(msg, "state", -1)),
                                    "unknown")
        detail = str(getattr(msg, "reason", ""))
        self.runner.submit(
            lambda: self.companion.router.on_navigation_status(
                request_id, state, detail))

    def _on_mute(self, request, response):
        muted = bool(request.data)
        self.runner.submit(lambda: self.companion.engine.set_muted(muted))
        response.success = True
        response.message = "muted" if muted else "unmuted"

        return response

    def _on_command(self, parsed, result, transcript) -> None:
        if not self._publish_commands:
            return

        msg = VoiceCommand()
        msg.stamp = self.get_clock().now().to_msg()
        msg.text = parsed.text
        msg.intent = parsed.kind.value
        msg.confidence = float(transcript.confidence or 0.0)
        msg.speaker_id = ""
        msg.wake_word_detected = False
        msg.request_id = result.command_id or ""
        self._command_pub.publish(msg)

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self.companion.status())
        self._status_pub.publish(msg)

    def destroy_node(self):
        try:
            self.runner.shutdown()
            self.executor_pool.shutdown(wait=False, cancel_futures=True)
        finally:
            return super().destroy_node()


def main(args=None) -> int:
    """Entry point.  Returns a process exit code."""

    rclpy.init(args=args)
    node = None
    code = 0

    try:
        node = VoiceCompanionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ModelMissing as exc:
        print(f"voice companion cannot start: {exc}", file=sys.stderr)
        code = 2
    except Exception as exc:
        print(f"voice companion failed: {exc}", file=sys.stderr)
        code = 1
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()

    return code


if __name__ == "__main__":
    sys.exit(main())
