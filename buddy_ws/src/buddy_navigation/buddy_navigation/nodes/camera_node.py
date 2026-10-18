"""ROS 2 camera node: owns the Pi camera and publishes frames.

Publishes
  /camera/image_raw   sensor_msgs/Image (bgr8)
  /diagnostics        diagnostic_msgs/DiagnosticArray (about 1 Hz)

Only one process may own the camera.  Other modules (for example the face
module) should subscribe to ``/camera/image_raw`` instead of opening the
device.  Do not run this node at the same time as another camera owner.
The default backend is ``mock`` so nothing touches hardware by accident.
"""

from __future__ import annotations

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Image

from buddy_navigation.perception.camera import (
    CameraConfig,
    CameraInterface,
    frame_to_image_fields,
)


class CameraNode(Node):
    """Capture frames and publish them."""

    def __init__(self) -> None:
        super().__init__("buddy_camera")

        self.declare_parameter("backend", "mock")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("device_index", 0)
        self.declare_parameter("frame_id", "camera_link")

        config = CameraConfig(
            backend=str(self.get_parameter("backend").value),
            width=int(self.get_parameter("width").value),
            height=int(self.get_parameter("height").value),
            fps=float(self.get_parameter("fps").value),
            device_index=int(self.get_parameter("device_index").value),
            frame_id=str(self.get_parameter("frame_id").value),
        )
        self.camera = CameraInterface(
            config,
            clock=lambda: self.get_clock().now().nanoseconds * 1e-9,
        )

        self.image_publisher = self.create_publisher(
            Image, "/camera/image_raw", 2
        )
        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )

        if self.camera.start():
            self.get_logger().info(
                f"Camera started: backend={config.backend} "
                f"{config.width}x{config.height}@{config.fps}"
            )
        else:
            self.get_logger().error(
                f"Camera failed to start: {self.camera.last_error}"
            )

        self.create_timer(1.0 / config.fps, self._grab)
        self.create_timer(1.0, self._diagnostics)

    def _grab(self) -> None:
        frame = self.camera.grab()

        if frame is None:
            return

        fields = frame_to_image_fields(frame)
        message = Image()
        message.header.stamp.sec = fields["sec"]
        message.header.stamp.nanosec = fields["nanosec"]
        message.header.frame_id = fields["frame_id"]
        message.height = fields["height"]
        message.width = fields["width"]
        message.encoding = fields["encoding"]
        message.is_bigendian = fields["is_bigendian"]
        message.step = fields["step"]
        message.data = fields["data"]
        self.image_publisher.publish(message)

    def _diagnostics(self) -> None:
        info = self.camera.diagnostics()
        status = DiagnosticStatus()
        status.name = "buddy_camera"
        status.hardware_id = f"camera_{info['backend']}"

        if info["healthy"]:
            status.level = DiagnosticStatus.OK
            status.message = "OK"
        else:
            status.level = DiagnosticStatus.ERROR
            status.message = str(info["last_error"] or "Camera unhealthy")

        status.values = [
            KeyValue(key=str(k), value=str(v)) for k, v in info.items()
        ]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self.diagnostics_publisher.publish(array)

    def destroy_node(self) -> bool:
        self.camera.stop()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
