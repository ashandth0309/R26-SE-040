"""Safe single-motor abstraction for a TB6612FNG motor channel."""

from buddy_core.hardware.base import GPIOBackend


class TB6612Motor:
    """Control one DC motor through one TB6612FNG H-bridge channel."""

    def __init__(
        self,
        gpio: GPIOBackend,
        in1_pin: int,
        in2_pin: int,
        pwm_pin: int,
        pwm_frequency_hz: float = 1000.0,
    ) -> None:
        self._gpio = gpio
        self._in1_pin = in1_pin
        self._in2_pin = in2_pin
        self._pwm_pin = pwm_pin
        self._pwm_frequency_hz = pwm_frequency_hz
        self._initialized = False
        self._speed = 0.0

    @property
    def speed(self) -> float:
        """Return the currently requested normalized motor speed."""
        return self._speed

    def initialize(self) -> None:
        """Configure the motor pins and leave the motor stopped."""
        self._gpio.configure_output(self._in1_pin, initial=False)
        self._gpio.configure_output(self._in2_pin, initial=False)

        self._gpio.setup_pwm(
            self._pwm_pin,
            frequency_hz=self._pwm_frequency_hz,
            duty_cycle=0.0,
        )

        self._speed = 0.0
        self._initialized = True

    def set_speed(self, speed: float) -> None:
        """Set normalized motor speed from -1.0 to +1.0."""
        if not self._initialized:
            raise RuntimeError("Motor must be initialized before use.")

        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("Motor speed must be a numeric value.")

        speed = float(speed)

        if not -1.0 <= speed <= 1.0:
            raise ValueError("Motor speed must be between -1.0 and 1.0.")

        if speed > 0.0:
            self._gpio.write(self._in1_pin, True)
            self._gpio.write(self._in2_pin, False)

        elif speed < 0.0:
            self._gpio.write(self._in1_pin, False)
            self._gpio.write(self._in2_pin, True)

        else:
            self._gpio.write(self._in1_pin, False)
            self._gpio.write(self._in2_pin, False)

        duty_cycle = abs(speed) * 100.0
        self._gpio.update_pwm(self._pwm_pin, duty_cycle)

        self._speed = speed

    def stop(self) -> None:
        """Stop the motor safely."""
        if not self._initialized:
            return

        self._gpio.update_pwm(self._pwm_pin, 0.0)
        self._gpio.write(self._in1_pin, False)
        self._gpio.write(self._in2_pin, False)

        self._speed = 0.0

    def cleanup(self) -> None:
        """Return this motor to a safe stopped state."""
        self.stop()


class TB6612Standby:
    """Safely control one TB6612FNG STBY pin."""

    def __init__(
        self,
        gpio: GPIOBackend,
        standby_pin: int,
    ) -> None:
        self._gpio = gpio
        self._standby_pin = standby_pin
        self._initialized = False
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Return whether the driver is enabled."""
        return self._enabled

    def initialize(self) -> None:
        """Configure STBY LOW so the driver starts disabled."""
        self._gpio.configure_output(
            self._standby_pin,
            initial=False,
        )

        self._initialized = True
        self._enabled = False

    def enable(self) -> None:
        """Enable the TB6612 driver."""
        if not self._initialized:
            raise RuntimeError(
                "TB6612 standby control must be "
                "initialized before use."
            )

        self._gpio.write(
            self._standby_pin,
            True,
        )

        self._enabled = True

    def disable(self) -> None:
        """Disable the TB6612 driver."""
        if not self._initialized:
            return

        self._gpio.write(
            self._standby_pin,
            False,
        )

        self._enabled = False

    def cleanup(self) -> None:
        """Return STBY to the safe disabled state."""
        self.disable()
