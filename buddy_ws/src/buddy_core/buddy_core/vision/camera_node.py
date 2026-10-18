"""ROS 2 camera node: the one owner of the Pi camera.

Publishes  /camera/image_raw  sensor_msgs/Image (bgr8)
           /diagnostics       diagnostic_msgs/DiagnosticArray (about 1 Hz)
Service    /camera/enable     std_srvs/SetBool  (privacy: off releases the
                                                 device)

Same topic, encoding and node name as the navigation branch's camera node,
so the two are interchangeable.  Run only one of them: this node refuses to
start if a ``buddy_camera`` node already exists, and a lock file stops a
second process from opening the device.

UNVERIFIED under real ROS 2 and with the real Pi camera.
"""

from __future__ import annotations

import json

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

from buddy_core.vision.camera_service import CameraConfig, CameraService


def frame_to_image_fields(frame) -> dict:
    """Plain fields for a ``sensor_msgs/Image`` (testable without ROS)."""

    sec = int(frame.stamp)
    nanosec = int(round((frame.stamp - sec) * 1e9))

    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000

    return {
        "sec": sec, "nanosec": nanosec,
        "height": frame.height, "width": frame.width,
        "encoding": "bgr8", "is_bigendian": 0,
        "step": frame.width * 3, "data": frame.image.tobytes(),
    }


class CameraNode(Node):
    """Thin ROS wrapper around ``CameraService``."""

    def __init__(self) -> None:
        super().__init__("buddy_camera")
        p = self._param
        config = CameraConfig(
            backend=str(p("backend", "mock")),
            width=int(p("width", 640)),
            height=int(p("height", 480)),
            fps=float(p("fps", 10.0)),
            device_index=int(p("device_index", 0)),
            video_path=str(p("video_path", "")),
            pixel_format=str(p("pixel_format", "RGB888")),
            enabled=bool(p("enabled", True)),
            lock_path=str(p("lock_path", "/tmp/buddy_camera.lock")),
        )
        self._frame_id = str(p("frame_id", "camera_link"))

        if not p("allow_coexist", False) and \
                "buddy_camera" in self._other_nodes():
            raise RuntimeError(
                "a 'buddy_camera' node is already running; only one "
                "process may own the camera")

        self.service = CameraService(config)
        self._sub = self.service.bus.subscribe("ros_publisher", maxlen=1)
        self._image_pub = self.create_publisher(Image, "/camera/image_raw", 2)
        self._diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10)
        self.create_service(SetBool, p("enable_service", "/camera/enable"),
                            self._on_enable)
        self.create_timer(1.0 / max(1.0, config.fps), self._publish_image)
        self.create_timer(1.0, self._publish_diagnostics)

        if p("autostart", True):
            self.service.start()

        self.get_logger().info(
            f"camera node ready backend={config.backend} "
            f"{config.width}x{config.height}@{config.fps} "
            f"enabled={config.enabled}")

    def _param(self, name, default):
        self.declare_parameter(name, default)

        return self.get_parameter(name).value

    def _other_nodes(self) -> list[str]:
        try:
            return list(self.get_node_names())
        except Exception:
            return []

    def _on_enable(self, request, response):
        self.service.set_enabled(bool(request.data))
        response.success = True
        response.message = "camera on" if request.data else "camera off"

        return response

    def _publish_image(self) -> None:
        frame = self._sub.latest()

        if frame is None:
            return

        fields = frame_to_image_fields(frame)
        msg = Image()
        msg.header.stamp.sec = fields["sec"]
        msg.header.stamp.nanosec = fields["nanosec"]
        msg.header.frame_id = self._frame_id
        msg.height = fields["height"]
        msg.width = fields["width"]
        msg.encoding = fields["encoding"]
        msg.is_bigendian = fields["is_bigendian"]
        msg.step = fields["step"]
        msg.data = fields["data"]
        self._image_pub.publish(msg)

    def _publish_diagnostics(self) -> None:
        info = self.service.health()
        status = DiagnosticStatus()
        status.name = "buddy_camera"
        status.hardware_id = f"camera_{info['backend']}"
        ok = info["state"] == "streaming"
        off = info["state"] == "disabled"
        status.level = DiagnosticStatus.OK if ok else (
            DiagnosticStatus.WARN if off else DiagnosticStatus.ERROR)
        status.message = info["state"] if not info["last_error"] or ok \
            else f"{info['state']}: {info['last_error']}"
        status.values = [KeyValue(key=str(k), value=json.dumps(v)
                                  if isinstance(v, dict) else str(v))
                         for k, v in info.items()]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self._diag_pub.publish(array)

    def destroy_node(self) -> bool:
        self.service.stop()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
