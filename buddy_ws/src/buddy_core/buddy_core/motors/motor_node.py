"""ROS 2 safety-approved motor executor for BUDDY."""

from __future__ import annotations

import time

import rclpy
from buddy_interfaces.msg import MovementCommand
from rclpy.node import Node

from buddy_core.config_loader import load_buddy_config
from buddy_core.motors.factory import create_motor_system


class MotorExecutorNode(Node):
    """Execute only movement commands approved by the safety layer."""

    def __init__(self) -> None:
        super().__init__("motor_executor")

        self.config = load_buddy_config()
        self.safety_config = self.config["safety"]

        self.default_timeout = float(
            self.safety_config["command_timeout_seconds"]
        )

        self.motor_system = create_motor_system(self.config)
        self.motor_system.initialize()
        self.motor_system.enable_drivers()

        self.last_command_time = None
        self.command_timeout = self.default_timeout
        self.motion_active = False

        self.create_subscription(
            MovementCommand,
            "/movement/safe",
            self._command_callback,
            10,
        )

        self.create_timer(
            0.05,
            self._watchdog_callback,
        )

        self.get_logger().info(
            "BUDDY motor executor started. "
            "Listening only to /movement/safe."
        )

    @staticmethod
    def _clamp_speed(value: float) -> float:
        """Clamp an absolute normalized speed into [0.0, 1.0]."""
        return min(max(abs(float(value)), 0.0), 1.0)

    def _validity_seconds(
        self,
        message: MovementCommand,
    ) -> float:
        seconds = (
            float(message.valid_for.sec)
            + float(message.valid_for.nanosec) / 1_000_000_000.0
        )

        if seconds <= 0.0:
            return self.default_timeout

        return min(seconds, self.default_timeout)

    def _turn_speed(
        self,
        message: MovementCommand,
    ) -> float:
        speed = self._clamp_speed(message.angular_speed)

        if speed == 0.0:
            speed = self._clamp_speed(message.linear_speed)

        return speed

    def _command_callback(
        self,
        message: MovementCommand,
    ) -> None:
        self.last_command_time = time.monotonic()
        self.command_timeout = self._validity_seconds(message)

        if message.emergency:
            self._stop("Emergency flag received.")
            return

        command = message.command

        try:
            if command == MovementCommand.COMMAND_STOP:
                self._stop("Safety-approved STOP received.")

            elif command == MovementCommand.COMMAND_FORWARD:
                speed = self._clamp_speed(message.linear_speed)

                if speed == 0.0:
                    self._stop("Forward speed is zero.")
                    return

                self.motor_system.drive.forward(speed)
                self.motion_active = True

                self.get_logger().info(
                    f"Executing FORWARD at {speed:.2f}."
                )

            elif command == MovementCommand.COMMAND_BACKWARD:
                speed = self._clamp_speed(message.linear_speed)

                if speed == 0.0:
                    self._stop("Backward speed is zero.")
                    return

                self.motor_system.drive.reverse(speed)
                self.motion_active = True

                self.get_logger().info(
                    f"Executing BACKWARD at {speed:.2f}."
                )

            elif command in (
                MovementCommand.COMMAND_TURN_LEFT,
                MovementCommand.COMMAND_ROTATE_LEFT,
            ):
                speed = self._turn_speed(message)

                if speed == 0.0:
                    self._stop("Left turn speed is zero.")
                    return

                self.motor_system.drive.turn_left(speed)
                self.motion_active = True

                self.get_logger().info(
                    f"Executing LEFT rotation at {speed:.2f}."
                )

            elif command in (
                MovementCommand.COMMAND_TURN_RIGHT,
                MovementCommand.COMMAND_ROTATE_RIGHT,
            ):
                speed = self._turn_speed(message)

                if speed == 0.0:
                    self._stop("Right turn speed is zero.")
                    return

                self.motor_system.drive.turn_right(speed)
                self.motion_active = True

                self.get_logger().info(
                    f"Executing RIGHT rotation at {speed:.2f}."
                )

            else:
                self._stop(
                    f"Unknown movement command {command}."
                )

        except Exception as exc:
            self._stop(
                f"Motor command failed: {exc}"
            )
            self.get_logger().error(
                f"Motor execution error: {exc}"
            )

    def _watchdog_callback(self) -> None:
        if not self.motion_active:
            return

        if self.last_command_time is None:
            self._stop("No command timestamp available.")
            return

        elapsed = time.monotonic() - self.last_command_time

        if elapsed > self.command_timeout:
            self._stop(
                "Movement command expired; watchdog STOP."
            )

    def _stop(self, reason: str) -> None:
        try:
            self.motor_system.drive.stop()
        finally:
            if self.motion_active:
                self.get_logger().warning(reason)

            self.motion_active = False

    def destroy_node(self) -> bool:
        try:
            self.motor_system.cleanup()
        except Exception as exc:
            self.get_logger().error(
                f"Motor cleanup error: {exc}"
            )

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = MotorExecutorNode()
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
