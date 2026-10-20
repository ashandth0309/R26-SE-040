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
* service    ``/voice/mute``           std_srvs/SetBool; the only way to
  unmute

This file is UNVERIFIED under real ROS 2; tests use strict stubs.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
from rclpy.node import Node

from buddy_interfaces.msg import (
    EmotionObservation,
    MovementCommand,
    SafetyState,
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
from buddy_core.voice.buddy_object_voice_bridge import ObjectConversation

NAV_STATE_NAMES = {0: "idle", 1: "active", 2: "done", 3: "rejected",
                   4: "failed", 5: "paused"}


class RosMovementSink:
    """Publishes ``MovementCommand`` to ``/movement/request``."""

    def __init__(self, node: Node, publisher) -> None:
        self._node = node
        self._pub = publisher

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


class RosNavigationSink:
    """Publishes ``NavigationCommand`` (from the navigation package)."""

    def __init__(self, node: Node, publisher, msg_type) -> None:
        self._node = node
        self._pub = publisher
        self._type = msg_type

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
        self._movement_pub = self.create_publisher(
            MovementCommand, movement_topic, 10)
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
        self.companion = build_companion(
            settings, backends, RosMovementSink(self, self._movement_pub),
            nav_sink, executor=self.executor_pool,
            command_observer=self._on_command)
        # Object commands are handled only in ordinary dialogue, after STOP/motion.
        self.companion.engine.dialogue = ObjectConversation(
            self.companion.engine.dialogue)
        self.runner = EngineRunner(self.companion)

        self.create_subscription(
            SafetyState, p("safety_topic", "/safety_state"),
            self._on_safety, 10)
        self.create_subscription(
            EmotionObservation, p("emotion_topic", "/emotion/observation"),
            self._on_emotion, 10)
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

        pub = self.create_publisher(
            NavigationCommand, self._param(
                "navigation_topic", "/navigation/command"), 10)
        self.create_subscription(
            NavigationStatus, self._param(
                "navigation_status_topic", "/navigation/status"),
            self._on_nav_status, 10)

        return RosNavigationSink(self, pub, NavigationCommand)

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

    def _on_nav_status(self, msg) -> None:
        # Field names mirror the navigation package; read defensively
        # because they are unverified against this branch.
        request_id = str(getattr(msg, "request_id", ""))
        state = NAV_STATE_NAMES.get(int(getattr(msg, "state", -1)),
                                    "unknown")
        self.runner.submit(
            lambda: self.companion.router.on_navigation_status(
                request_id, state))

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
