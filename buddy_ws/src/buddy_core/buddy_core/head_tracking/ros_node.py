"""ROS 2 node for head tracking (thin wrapper around the bridge).

NOT built or run under ROS 2 in the environment this was written in; it
is exercised only against a stub (test_head_ros_node.py).  See
docs/head_tracking_physical_validation.md for the build and run steps.

The node owns NO camera and NO microphone.  It subscribes to results.
"""

from __future__ import annotations

from buddy_core.head_tracking.config import (
    HeadTrackingConfigError,
    load_head_tracking_config,
)
from buddy_core.head_tracking.factory import build_head_system
from buddy_core.head_tracking.ros_adapter import HeadTrackingBridge, \
    event_json


def _try_import(module: str, name: str):
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


def main(args=None) -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class HeadTrackingNode(Node):
        def __init__(self) -> None:
            super().__init__("head_tracking")
            p = self._param
            ht_path = p("head_config_path", "")
            buddy_path = p("buddy_config_path", "")
            from buddy_core.config_loader import load_buddy_config
            self.ht_cfg = load_head_tracking_config(ht_path or None)
            buddy = load_buddy_config(buddy_path or None)
            self._events = self.create_publisher(
                String, p("events_topic", "/head/events"), 20)
            try:
                self.system = build_head_system(
                    buddy, self.ht_cfg, on_event=self._on_event)
            except HeadTrackingConfigError as exc:
                self.get_logger().error(f"head tracking disabled: {exc}")
                raise
            self.get_logger().info(
                f"head tracking mode: {self.system.mode}")
            cam = buddy.get("camera", {})
            self.bridge = HeadTrackingBridge(
                self.system.manager,
                int(p("frame_width", cam.get("width", 640))),
                int(p("frame_height", cam.get("height", 480))))
            self._status = self.create_publisher(
                String, p("status_topic", "/head/status"), 5)

            self._wire(p, String)
            if not self.system.start(self._now()):
                self.get_logger().error("neck failed to start (fault)")
            self.create_timer(1.0 / self.ht_cfg.tick_rate_hz, self._tick)
            self.create_timer(1.0, self._publish_status)

        def _param(self, name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _wire(self, p, String) -> None:
            b = self.bridge
            from buddy_interfaces.msg import SafetyState
            self.create_subscription(
                SafetyState, p("safety_topic", "/safety_state"),
                lambda m: b.on_safety(m, self._now()), 10)
            people = _try_import("buddy_interfaces.msg", "VisionPeople")
            if people is not None:
                self.create_subscription(
                    people, p("people_topic", "/vision/people"),
                    lambda m: b.on_people(m, self._now()), 5)
            else:
                self.get_logger().warn(
                    "VisionPeople not available (Sobiya's vision branch "
                    "not integrated): face and gesture input disabled")
            voice = _try_import("buddy_interfaces.msg", "VoiceCommand")
            if voice is not None:
                self.create_subscription(
                    voice, p("voice_command_topic", "/voice/command"),
                    lambda m: b.on_voice_command(m, self._now()), 10)
            for topic, handler, name in (
                    ("/voice/status", b.on_voice_status, "voice_status"),
                    ("/head/command", b.on_command, "command"),
                    ("/head/hand_observation", b.on_hand_json, "hand"),
                    ("/head/sound_direction", b.on_sound_json, "sound")):
                self.create_subscription(
                    String, p(f"{name}_topic", topic),
                    lambda m, h=handler: h(m.data, self._now()), 10)

        def _on_event(self, ev) -> None:
            msg = __import__("std_msgs.msg", fromlist=["String"]).String()
            msg.data = event_json(ev)
            self._events.publish(msg)

        def _tick(self) -> None:
            self.system.manager.tick(self._now())

        def _publish_status(self) -> None:
            msg = __import__("std_msgs.msg", fromlist=["String"]).String()
            msg.data = self.bridge.status_json(self._now())
            self._status.publish(msg)

        def destroy_node(self):
            try:
                self.system.shutdown()
            finally:
                return super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = HeadTrackingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
