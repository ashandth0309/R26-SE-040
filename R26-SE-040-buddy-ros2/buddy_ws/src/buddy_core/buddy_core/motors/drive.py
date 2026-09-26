"""Safe four-wheel drive controller for BUDDY."""

from buddy_core.motors.tb6612 import TB6612Motor


class BuddyDrive:
    """Coordinate BUDDY's four independent DC drive motors."""

    def __init__(
        self,
        front_left: TB6612Motor,
        front_right: TB6612Motor,
        rear_left: TB6612Motor,
        rear_right: TB6612Motor,
    ) -> None:
        self.front_left = front_left
        self.front_right = front_right
        self.rear_left = rear_left
        self.rear_right = rear_right

        self._motors = (
            self.front_left,
            self.front_right,
            self.rear_left,
            self.rear_right,
        )

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Return whether all drive motors were initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize all four motors in the stopped state."""
        initialized_motors = []

        try:
            for motor in self._motors:
                motor.initialize()
                initialized_motors.append(motor)
        except Exception:
            for motor in initialized_motors:
                motor.stop()
            raise

        self._initialized = True

    def set_wheel_speeds(
        self,
        front_left: float,
        front_right: float,
        rear_left: float,
        rear_right: float,
    ) -> None:
        """Set normalized speed independently for all four wheels."""
        self._require_initialized()

        speeds = (
            front_left,
            front_right,
            rear_left,
            rear_right,
        )

        for speed in speeds:
            self._validate_speed(speed)

        self.front_left.set_speed(front_left)
        self.front_right.set_speed(front_right)
        self.rear_left.set_speed(rear_left)
        self.rear_right.set_speed(rear_right)

    def forward(self, speed: float) -> None:
        """Drive all four wheels forward."""
        self._validate_positive_speed(speed)
        self.set_wheel_speeds(speed, speed, speed, speed)

    def reverse(self, speed: float) -> None:
        """Drive all four wheels in reverse."""
        self._validate_positive_speed(speed)
        self.set_wheel_speeds(-speed, -speed, -speed, -speed)

    def turn_left(self, speed: float) -> None:
        """Turn left in place."""
        self._validate_positive_speed(speed)
        self.set_wheel_speeds(-speed, speed, -speed, speed)

    def turn_right(self, speed: float) -> None:
        """Turn right in place."""
        self._validate_positive_speed(speed)
        self.set_wheel_speeds(speed, -speed, speed, -speed)

    def stop(self) -> None:
        """Stop all four motors."""
        for motor in self._motors:
            motor.stop()

    def cleanup(self) -> None:
        """Safely stop and clean up every drive motor."""
        self.stop()

        for motor in self._motors:
            motor.cleanup()

        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "BUDDY drive must be initialized before use."
            )

    @staticmethod
    def _validate_speed(speed: float) -> None:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("Wheel speed must be a numeric value.")

        if not -1.0 <= float(speed) <= 1.0:
            raise ValueError(
                "Wheel speed must be between -1.0 and 1.0."
            )

    @staticmethod
    def _validate_positive_speed(speed: float) -> None:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("Drive speed must be a numeric value.")

        if not 0.0 <= float(speed) <= 1.0:
            raise ValueError(
                "Drive speed must be between 0.0 and 1.0."
            )
