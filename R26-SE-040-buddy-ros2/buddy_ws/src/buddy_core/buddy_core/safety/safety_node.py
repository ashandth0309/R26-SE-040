"""ROS 2 deterministic safety supervisor for BUDDY."""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from buddy_interfaces.msg import (
    MovementCommand,
    ObstacleStatus,
    SafetyState,
)
from buddy_interfaces.srv import (
    EmergencyStop,
    ResetEmergencyStop,
)

from buddy_core.safety.supervisor import SafetySupervisor


class SafetySupervisorNode(Node):
    """Validate movement requests before motor execution."""

    def __init__(self) -> None:
        super().__init__("safety_supervisor")

        self.supervisor = SafetySupervisor()
        self.obstacle_status = None
        self.last_obstacle_time = None
        self.obstacle_timeout_seconds = 1.0
        self.active_command = MovementCommand.COMMAND_STOP

        self.safe_command_publisher = self.create_publisher(
            MovementCommand,
            "/movement/safe",
            10,
        )

        self.safety_state_publisher = self.create_publisher(
            SafetyState,
            "/safety_state",
            10,
        )

        self.create_subscription(
            MovementCommand,
            "/movement/request",
            self._movement_callback,
            10,
        )

        self.create_subscription(
            ObstacleStatus,
            "/obstacle_status",
            self._obstacle_callback,
            10,
        )

        self.create_service(
            EmergencyStop,
            "/emergency_stop",
            self._emergency_stop_callback,
        )

        self.create_service(
            ResetEmergencyStop,
            "/reset_emergency_stop",
            self._reset_emergency_stop_callback,
        )

        self.create_timer(
            0.5,
            self._publish_safety_state,
        )

        self.create_timer(
            0.1,
            self._obstacle_watchdog,
        )

        self.get_logger().info(
            "BUDDY safety supervisor started."
        )

    def _movement_callback(
        self,
        message: MovementCommand,
    ) -> None:
        decision = self._evaluate(message.command)

        if message.emergency:
            self.supervisor.activate_emergency_stop()
            self._publish_stop(
                "Movement request contained emergency flag."
            )
            return

        if decision.allowed:
            self.active_command = message.command
            self.safe_command_publisher.publish(message)

            self.get_logger().info(
                f"Movement accepted: command={message.command}"
            )
            return

        self.get_logger().warning(
            f"Movement rejected: {decision.reason}"
        )

        self._publish_stop(decision.reason)

    def _obstacle_callback(
        self,
        message: ObstacleStatus,
    ) -> None:
        self.obstacle_status = message
        self.last_obstacle_time = self.get_clock().now()

        if self.active_command == MovementCommand.COMMAND_STOP:
            return

        decision = self._evaluate(self.active_command)

        if not decision.allowed:
            self.get_logger().warning(
                "Active movement became unsafe: "
                f"{decision.reason}"
            )
            self._publish_stop(decision.reason)

    def _obstacle_data_is_fresh(self) -> bool:
        if self.obstacle_status is None:
            return False

        if self.last_obstacle_time is None:
            return False

        age_seconds = (
            self.get_clock().now() - self.last_obstacle_time
        ).nanoseconds / 1_000_000_000.0

        return age_seconds <= self.obstacle_timeout_seconds

    def _obstacle_watchdog(self) -> None:
        if self._obstacle_data_is_fresh():
            return

        if self.active_command == MovementCommand.COMMAND_STOP:
            return

        self.get_logger().error(
            "Obstacle data stale while movement active."
        )

        self._publish_stop(
            "Obstacle data stale or unavailable."
        )

    def _evaluate(self, command: int):
        status = self.obstacle_status

        if not self._obstacle_data_is_fresh():
            return self.supervisor.evaluate(
                command,
                obstacle_data_available=False,
            )

        return self.supervisor.evaluate(
            command,
            front_blocked=status.front_blocked,
            rear_blocked=status.rear_blocked,
            left_blocked=status.left_blocked,
            right_blocked=status.right_blocked,
            obstacle_data_available=True,
        )

    def _publish_stop(self, reason: str) -> None:
        message = MovementCommand()

        message.stamp = self.get_clock().now().to_msg()
        message.sequence_id = 0
        message.source = "safety_supervisor"
        message.command = MovementCommand.COMMAND_STOP
        message.linear_speed = 0.0
        message.angular_speed = 0.0
        message.valid_for.sec = 1
        message.valid_for.nanosec = 0
        message.emergency = self.supervisor.emergency_stop_active

        self.active_command = MovementCommand.COMMAND_STOP

        self.safe_command_publisher.publish(message)

        self.get_logger().warning(
            f"Safety STOP published: {reason}"
        )

    def _emergency_stop_callback(
        self,
        request,
        response,
    ):
        self.supervisor.activate_emergency_stop()

        self._publish_stop(
            f"Emergency stop requested by {request.source}: "
            f"{request.reason}"
        )

        response.accepted = True
        response.emergency_stop_active = True
        response.message = "Emergency stop activated."

        return response

    def _reset_emergency_stop_callback(
        self,
        request,
        response,
    ):
        self.supervisor.reset_emergency_stop()

        response.accepted = True
        response.emergency_stop_active = False
        response.message = "Emergency stop reset."

        self.get_logger().info(
            f"Emergency stop reset by {request.source}."
        )

        return response

    def _publish_safety_state(self) -> None:
        message = SafetyState()

        message.stamp = self.get_clock().now().to_msg()
        message.emergency_stop_active = (
            self.supervisor.emergency_stop_active
        )

        if self.supervisor.emergency_stop_active:
            message.movement_allowed = False
            message.degraded_mode = False
            message.reason = "Emergency stop active."
            message.safety_level = SafetyState.SAFETY_EMERGENCY

        elif not self._obstacle_data_is_fresh():
            message.movement_allowed = False
            message.degraded_mode = True
            message.reason = "Obstacle data stale or unavailable."
            message.safety_level = SafetyState.SAFETY_RESTRICTED

        elif self.obstacle_status.degraded:
            message.movement_allowed = True
            message.degraded_mode = True
            message.reason = "Obstacle sensing degraded."
            message.safety_level = SafetyState.SAFETY_CAUTION

        elif (
            self.obstacle_status.front_blocked
            or self.obstacle_status.rear_blocked
            or self.obstacle_status.left_blocked
            or self.obstacle_status.right_blocked
        ):
            message.movement_allowed = True
            message.degraded_mode = False
            message.reason = "Directional movement restrictions active."
            message.safety_level = SafetyState.SAFETY_RESTRICTED

        else:
            message.movement_allowed = True
            message.degraded_mode = False
            message.reason = "Safety conditions normal."
            message.safety_level = SafetyState.SAFETY_NORMAL

        self.safety_state_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = SafetySupervisorNode()

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
