"""ROS 2 obstacle sensing node for BUDDY."""

from __future__ import annotations

import math
import time

import rclpy
from buddy_interfaces.msg import ObstacleStatus
from rclpy.node import Node

from buddy_core.config_loader import load_buddy_config
from buddy_core.hardware.factory import get_gpio_backend
from buddy_core.sensors.ultrasonic import UltrasonicSensor


class ObstacleSensorNode(Node):
    """Publish directional obstacle information from four HC-SR04 sensors."""

    def __init__(self) -> None:
        super().__init__("buddy_obstacle_sensor")

        self.config = load_buddy_config()

        self.hardware_config = self.config["hardware"]
        self.ultrasonic_config = self.hardware_config["ultrasonic"]
        self.safety_config = self.config["safety"]

        self.stop_distance = self.safety_config[
            "obstacle_stop_distance_m"
        ]

        self.inter_sensor_delay = self.ultrasonic_config[
            "inter_sensor_delay_seconds"
        ]

        self.gpio = get_gpio_backend(self.config)

        self.sensors = self._create_sensors()

        self.publisher = self.create_publisher(
            ObstacleStatus,
            "/obstacle_status",
            10,
        )

        # One complete four-sensor scan every 0.30 seconds.
        self.timer = self.create_timer(
            0.30,
            self._scan_and_publish,
        )

        self.get_logger().info(
            "BUDDY obstacle sensor node started."
        )

        if not self.hardware_config["enabled"]:
            self.get_logger().warning(
                "Physical hardware is disabled. "
                "Obstacle node is using the mock GPIO backend."
            )

    def _create_sensor(
        self,
        sensor_config: dict,
    ) -> UltrasonicSensor:
        return UltrasonicSensor(
            self.gpio,
            sensor_config["trigger"],
            sensor_config["echo"],
            timeout_seconds=self.ultrasonic_config[
                "timeout_seconds"
            ],
            minimum_distance_m=self.ultrasonic_config[
                "minimum_distance_m"
            ],
            maximum_distance_m=self.ultrasonic_config[
                "maximum_distance_m"
            ],
            filter_window=self.ultrasonic_config[
                "filter_window"
            ],
        )

    def _create_sensors(self) -> dict[str, UltrasonicSensor]:
        return {
            "front_left": self._create_sensor(
                self.ultrasonic_config["front_left"]
            ),
            "front_right": self._create_sensor(
                self.ultrasonic_config["front_right"]
            ),
            "rear_left": self._create_sensor(
                self.ultrasonic_config["rear_left"]
            ),
            "rear_right": self._create_sensor(
                self.ultrasonic_config["rear_right"]
            ),
        }

    @staticmethod
    def _nearest_valid(
        first: float | None,
        second: float | None,
    ) -> tuple[float, bool]:
        values = [
            value
            for value in (first, second)
            if value is not None
        ]

        if not values:
            return math.nan, False

        return min(values), True

    def _read_sensor(
        self,
        name: str,
    ) -> float | None:
        distance = self.sensors[name].read_distance_m()

        time.sleep(self.inter_sensor_delay)

        return distance

    def _scan_and_publish(self) -> None:
        readings = {
            "front_left": self._read_sensor("front_left"),
            "front_right": self._read_sensor("front_right"),
            "rear_left": self._read_sensor("rear_left"),
            "rear_right": self._read_sensor("rear_right"),
        }

        front, front_valid = self._nearest_valid(
            readings["front_left"],
            readings["front_right"],
        )

        rear, rear_valid = self._nearest_valid(
            readings["rear_left"],
            readings["rear_right"],
        )

        left, left_valid = self._nearest_valid(
            readings["front_left"],
            readings["rear_left"],
        )

        right, right_valid = self._nearest_valid(
            readings["front_right"],
            readings["rear_right"],
        )

        message = ObstacleStatus()

        message.stamp = self.get_clock().now().to_msg()

        message.front_distance_m = float(front)
        message.rear_distance_m = float(rear)
        message.left_distance_m = float(left)
        message.right_distance_m = float(right)

        message.front_valid = front_valid
        message.rear_valid = rear_valid
        message.left_valid = left_valid
        message.right_valid = right_valid

        message.front_blocked = (
            front_valid
            and front <= self.stop_distance
        )

        message.rear_blocked = (
            rear_valid
            and rear <= self.stop_distance
        )

        message.left_blocked = (
            left_valid
            and left <= self.stop_distance
        )

        message.right_blocked = (
            right_valid
            and right <= self.stop_distance
        )

        message.degraded = any(
            value is None
            for value in readings.values()
        )

        self.publisher.publish(message)

    def destroy_node(self) -> bool:
        for sensor in self.sensors.values():
            try:
                sensor.shutdown()
            except Exception:
                pass

        try:
            self.gpio.cleanup()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = ObstacleSensorNode()
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
