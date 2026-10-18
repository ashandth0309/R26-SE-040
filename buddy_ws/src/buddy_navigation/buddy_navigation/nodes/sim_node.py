"""Simulated world for running the navigation stack without hardware.

Replaces the physical layer only: it publishes the same topics the real
ultrasonic node does and moves a kinematic robot from ``/movement/safe``.
Run the REAL ``safety_supervisor`` and ``navigation`` nodes against it.

Publishes
  /ultrasonic/<name>  sensor_msgs/Range
  /obstacle_status    buddy_interfaces/ObstacleStatus (same rules as the
                      real node, so the real safety supervisor works)
  /sim/pose           geometry_msgs/PoseStamped (ground truth)

Subscribes
  /movement/safe      buddy_interfaces/MovementCommand

Parameters: ``fault_sensor`` and ``fault_mode`` inject a sensor failure
(no_response, stuck_high, specular, frozen) at start-up.  SIMULATION ONLY.
"""

from __future__ import annotations

import math

import rclpy
from buddy_interfaces.msg import MovementCommand, ObstacleStatus
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Range

from buddy_navigation.core.commands import MoveCode, MoveRequest
from buddy_navigation.core.geometry import Pose2D
from buddy_navigation.core.presets import (
    simulation_footprint,
    simulation_sensor_specs,
)
from buddy_navigation.core.sensor_model import SensorStatus
from buddy_navigation.sim.mock_robot import MockRobot
from buddy_navigation.sim.mock_ultrasonic import MockUltrasonicArray
from buddy_navigation.sim.world import six_by_six_world

STOP_DISTANCE_M = 0.25
DT = 0.01


def range_value(sample) -> float:
    """REP-117 encoding of a simulated sample."""

    if sample.status is SensorStatus.OK:
        return float(sample.distance_m)

    if sample.status is SensorStatus.CLEAR:
        return math.inf

    if sample.status is SensorStatus.TOO_CLOSE:
        return -math.inf

    return math.nan


class SimNode(Node):
    """Simulated sensors and robot."""

    def __init__(self) -> None:
        super().__init__("buddy_navigation_sim")

        self.declare_parameter("start_x", 0.5)
        self.declare_parameter("start_y", 0.5)
        self.declare_parameter("start_yaw", 0.0)
        self.declare_parameter("fault_sensor", "")
        self.declare_parameter("fault_mode", "no_response")

        specs = simulation_sensor_specs()
        self.world = six_by_six_world(1.0)
        self.array = MockUltrasonicArray(
            self.world, specs, noise_std_m=0.01
        )
        self.robot = MockRobot(
            self.world,
            simulation_footprint(),
            Pose2D(
                float(self.get_parameter("start_x").value),
                float(self.get_parameter("start_y").value),
                float(self.get_parameter("start_yaw").value),
            ),
        )
        fault = str(self.get_parameter("fault_sensor").value)

        if fault:
            self.array.inject_fault(
                fault, str(self.get_parameter("fault_mode").value)
            )
            self.get_logger().warning(
                f"Injected fault '{self.get_parameter('fault_mode').value}'"
                f" on {fault}."
            )

        self.latest = {}
        self._reported_collision = False
        self.range_publishers = {
            spec.name: self.create_publisher(
                Range, f"/ultrasonic/{spec.name}", 10
            )
            for spec in specs
        }
        self.specs = {spec.name: spec for spec in specs}
        self.status_publisher = self.create_publisher(
            ObstacleStatus, "/obstacle_status", 10
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, "/sim/pose", 10
        )
        self.create_subscription(
            MovementCommand, "/movement/safe", self._on_safe, 10
        )
        self.create_timer(DT, self._step)
        self.create_timer(0.05, self._publish_pose)

        self.get_logger().warning(
            "SIMULATION ONLY: no hardware is being driven."
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_safe(self, msg) -> None:
        try:
            code = MoveCode(int(msg.command))
            speed = max(float(msg.linear_speed), float(msg.angular_speed))
            valid = float(msg.valid_for.sec) + msg.valid_for.nanosec * 1e-9
            request = MoveRequest(
                code, min(speed, 1.0), max(valid, 0.05), str(msg.source)
            )
        except ValueError:
            return

        self.robot.apply(request, self._now())

    def _step(self) -> None:
        now = self._now()
        sample = self.array.step(self.robot.pose, now)

        if sample is not None:
            self.latest[sample.name] = sample
            message = Range()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = f"ultrasonic_{sample.name}_link"
            message.radiation_type = Range.ULTRASOUND
            message.field_of_view = float(self.specs[sample.name].fov_rad)
            message.min_range = float(self.specs[sample.name].min_range_m)
            message.max_range = float(self.specs[sample.name].max_range_m)
            message.range = range_value(sample)
            self.range_publishers[sample.name].publish(message)

            if len(self.latest) == len(self.specs):
                self._publish_status()

        self.robot.step(now, DT)

    def _legacy(self, name: str):
        sample = self.latest.get(name)

        if sample is not None and sample.status is SensorStatus.OK:
            return sample.distance_m

        return None

    def _publish_status(self) -> None:
        def pair(a, b):
            values = [v for v in (self._legacy(a), self._legacy(b)) if v]
            return (min(values), True) if values else (math.nan, False)

        front, fv = pair("front_left", "front_right")
        rear, rv = pair("rear_left", "rear_right")
        left, lv = pair("front_left", "rear_left")
        right, rtv = pair("front_right", "rear_right")
        message = ObstacleStatus()
        message.stamp = self.get_clock().now().to_msg()
        message.front_distance_m = float(front)
        message.rear_distance_m = float(rear)
        message.left_distance_m = float(left)
        message.right_distance_m = float(right)
        message.front_valid, message.rear_valid = fv, rv
        message.left_valid, message.right_valid = lv, rtv
        message.front_blocked = fv and front <= STOP_DISTANCE_M
        message.rear_blocked = rv and rear <= STOP_DISTANCE_M
        message.left_blocked = lv and left <= STOP_DISTANCE_M
        message.right_blocked = rtv and right <= STOP_DISTANCE_M
        message.degraded = any(
            self._legacy(n) is None for n in self.specs
        )
        self.status_publisher.publish(message)

    def _publish_pose(self) -> None:
        pose = self.robot.pose
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.position.x = float(pose.x)
        message.pose.position.y = float(pose.y)
        message.pose.orientation.z = math.sin(pose.yaw / 2.0)
        message.pose.orientation.w = math.cos(pose.yaw / 2.0)
        self.pose_publisher.publish(message)

        if self.robot.collided and not self._reported_collision:
            self._reported_collision = True
            self.get_logger().error("Simulated robot COLLIDED.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = SimNode()
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
