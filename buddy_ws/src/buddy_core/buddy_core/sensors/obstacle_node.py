"""ROS 2 obstacle sensing node for BUDDY."""

from __future__ import annotations

import math

import rclpy
from buddy_interfaces.msg import ObstacleStatus
from diagnostic_msgs.msg import (
    DiagnosticArray,
    DiagnosticStatus,
    KeyValue,
)
from rclpy.node import Node
from sensor_msgs.msg import Range

from buddy_core.config_loader import load_buddy_config
from buddy_core.hardware.factory import get_gpio_backend
from buddy_core.sensors.ultrasonic import (
    UltrasonicMeasurement,
    UltrasonicSensor,
    UltrasonicStatus,
)
from buddy_core.sensors.ultrasonic_array import UltrasonicArrayScanner

SENSOR_ORDER = ("front_left", "front_right", "rear_left", "rear_right")

FAULT_STATUSES = frozenset(
    {
        UltrasonicStatus.NO_RESPONSE,
        UltrasonicStatus.STUCK_HIGH,
        UltrasonicStatus.ERROR,
    }
)


def range_value_for(measurement: UltrasonicMeasurement) -> float:
    """Map a measurement to a ``sensor_msgs/Range`` value (REP-117).

    * a real distance is reported as-is;
    * ``-inf`` means an object too close to detect;
    * ``+inf`` means nothing detected within range;
    * ``NaN`` means the reading is invalid (sensor fault).
    """

    status = measurement.status

    if status is UltrasonicStatus.OK:
        return float(measurement.raw_distance_m)

    if status is UltrasonicStatus.TOO_CLOSE:
        return -math.inf

    if status is UltrasonicStatus.CLEAR:
        return math.inf

    return math.nan


class ObstacleSensorNode(Node):
    """Publish directional obstacle information from four HC-SR04 sensors.

    ``/obstacle_status`` keeps exactly the semantics the safety supervisor
    already relies on.  In addition, each sensor is published on its own
    ``/ultrasonic/<name>`` ``sensor_msgs/Range`` topic and sensor health
    on ``/diagnostics`` so the navigation stack can tell "nothing in
    range" from "sensor fault".
    """

    def __init__(self) -> None:
        super().__init__("buddy_obstacle_sensor")

        self.declare_parameter("publish_per_sensor_ranges", True)
        self.declare_parameter("field_of_view_rad", 0.2618)

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

        self.publish_ranges = bool(
            self.get_parameter("publish_per_sensor_ranges").value
        )
        self.field_of_view = float(
            self.get_parameter("field_of_view_rad").value
        )

        self.gpio = get_gpio_backend(self.config)

        self.sensors = self._create_sensors()

        # Only this scanner may trigger sensors, one at a time.
        self.scanner = UltrasonicArrayScanner(
            self.sensors,
            order=SENSOR_ORDER,
            min_trigger_interval_seconds=self.inter_sensor_delay,
        )

        self.latest: dict[str, UltrasonicMeasurement] = {}
        self.last_status: dict[str, UltrasonicStatus] = {}
        self.consecutive_faults: dict[str, int] = {
            name: 0 for name in SENSOR_ORDER
        }

        self.publisher = self.create_publisher(
            ObstacleStatus,
            "/obstacle_status",
            10,
        )

        self.range_publishers = {}

        if self.publish_ranges:
            for name in SENSOR_ORDER:
                self.range_publishers[name] = self.create_publisher(
                    Range,
                    f"/ultrasonic/{name}",
                    10,
                )

        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            "/diagnostics",
            10,
        )

        # The scanner refuses to fire before the inter-sensor delay has
        # elapsed, so this fast timer never breaks the spacing rule.  Each
        # tick measures at most one sensor and is bounded by the sensor
        # timeout.
        self.timer = self.create_timer(
            0.01,
            self._tick,
        )

        self.create_timer(
            1.0,
            self._publish_diagnostics,
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
            name: self._create_sensor(self.ultrasonic_config[name])
            for name in SENSOR_ORDER
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

    @staticmethod
    def _legacy_distance(
        measurement: UltrasonicMeasurement,
    ) -> float | None:
        """Return the distance the original driver API would have given.

        That is the median-filtered distance for a valid reading and
        ``None`` for every other outcome.
        """

        if measurement.status is UltrasonicStatus.OK:
            return measurement.filtered_distance_m

        return None

    def _tick(self) -> None:
        result = self.scanner.step()

        if result is None:
            return

        self.latest[result.name] = result.measurement
        self._track_health(result.name, result.measurement)

        if self.publish_ranges:
            self._publish_range(result.name, result.measurement)

        if result.scan_complete:
            self._publish_status()

    def _track_health(
        self,
        name: str,
        measurement: UltrasonicMeasurement,
    ) -> None:
        status = measurement.status

        if status in FAULT_STATUSES:
            self.consecutive_faults[name] += 1
        else:
            self.consecutive_faults[name] = 0

        previous = self.last_status.get(name)

        if previous is not status:
            if status in FAULT_STATUSES:
                self.get_logger().warning(
                    f"Ultrasonic sensor '{name}' fault: {status.value}."
                )
            elif previous in FAULT_STATUSES:
                self.get_logger().info(
                    f"Ultrasonic sensor '{name}' recovered."
                )

        self.last_status[name] = status

    def _publish_range(
        self,
        name: str,
        measurement: UltrasonicMeasurement,
    ) -> None:
        message = Range()

        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"ultrasonic_{name}_link"
        message.radiation_type = Range.ULTRASOUND
        message.field_of_view = self.field_of_view
        message.min_range = float(
            self.ultrasonic_config["minimum_distance_m"]
        )
        message.max_range = float(
            self.ultrasonic_config["maximum_distance_m"]
        )
        message.range = float(range_value_for(measurement))

        self.range_publishers[name].publish(message)

    def _publish_status(self) -> None:
        readings = {
            name: self._legacy_distance(self.latest[name])
            for name in SENSOR_ORDER
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

    def _publish_diagnostics(self) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()

        for name in SENSOR_ORDER:
            status = self.last_status.get(name)
            diagnostic = DiagnosticStatus()
            diagnostic.name = f"buddy_ultrasonic/{name}"
            diagnostic.hardware_id = f"ultrasonic_{name}"

            if status is None:
                diagnostic.level = DiagnosticStatus.STALE
                diagnostic.message = "No measurement yet"
            elif status in FAULT_STATUSES:
                diagnostic.level = DiagnosticStatus.ERROR
                diagnostic.message = f"Sensor fault: {status.value}"
            elif status is UltrasonicStatus.CLEAR:
                diagnostic.level = DiagnosticStatus.OK
                diagnostic.message = "Nothing detected in range"
            elif status is UltrasonicStatus.TOO_CLOSE:
                diagnostic.level = DiagnosticStatus.WARN
                diagnostic.message = "Obstacle closer than minimum range"
            else:
                diagnostic.level = DiagnosticStatus.OK
                diagnostic.message = "OK"

            diagnostic.values = [
                KeyValue(
                    key="last_status",
                    value=status.value if status else "none",
                ),
                KeyValue(
                    key="consecutive_faults",
                    value=str(self.consecutive_faults[name]),
                ),
            ]

            array.status.append(diagnostic)

        self.diagnostics_publisher.publish(array)

    def destroy_node(self) -> bool:
        try:
            self.scanner.shutdown()
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
