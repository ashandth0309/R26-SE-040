"""Tests for BUDDY motor-control abstractions."""

import pytest


from buddy_core.hardware.mock_gpio import MockGPIOBackend
from buddy_core.motors.tb6612 import TB6612Motor


IN1 = 10
IN2 = 11
PWM = 12


def create_motor():
    gpio = MockGPIOBackend()

    motor = TB6612Motor(
        gpio=gpio,
        in1_pin=IN1,
        in2_pin=IN2,
        pwm_pin=PWM,
    )

    return gpio, motor


def test_motor_initializes_stopped():
    gpio, motor = create_motor()

    motor.initialize()

    assert gpio.pin_values[IN1] is False
    assert gpio.pin_values[IN2] is False
    assert gpio.pwm_channels[PWM].duty_cycle == 0.0
    assert motor.speed == 0.0


def test_forward_direction_and_pwm():
    gpio, motor = create_motor()
    motor.initialize()

    motor.set_speed(0.5)

    assert gpio.pin_values[IN1] is True
    assert gpio.pin_values[IN2] is False
    assert gpio.pwm_channels[PWM].duty_cycle == 50.0
    assert motor.speed == 0.5


def test_reverse_direction_and_pwm():
    gpio, motor = create_motor()
    motor.initialize()

    motor.set_speed(-0.25)

    assert gpio.pin_values[IN1] is False
    assert gpio.pin_values[IN2] is True
    assert gpio.pwm_channels[PWM].duty_cycle == 25.0
    assert motor.speed == -0.25


def test_zero_speed_stops_motor():
    gpio, motor = create_motor()
    motor.initialize()

    motor.set_speed(0.75)
    motor.set_speed(0.0)

    assert gpio.pin_values[IN1] is False
    assert gpio.pin_values[IN2] is False
    assert gpio.pwm_channels[PWM].duty_cycle == 0.0
    assert motor.speed == 0.0


@pytest.mark.parametrize("speed", [-1.01, 1.01, -2.0, 2.0])
def test_invalid_speed_range_is_rejected(speed):
    _, motor = create_motor()
    motor.initialize()

    with pytest.raises(ValueError):
        motor.set_speed(speed)


@pytest.mark.parametrize("speed", [None, "fast", True])
def test_invalid_speed_type_is_rejected(speed):
    _, motor = create_motor()
    motor.initialize()

    with pytest.raises(ValueError):
        motor.set_speed(speed)


def test_motor_requires_initialization():
    _, motor = create_motor()

    with pytest.raises(RuntimeError):
        motor.set_speed(0.5)


def test_stop_is_safe_before_initialization():
    _, motor = create_motor()

    motor.stop()

    assert motor.speed == 0.0


def test_cleanup_stops_motor():
    gpio, motor = create_motor()
    motor.initialize()
    motor.set_speed(1.0)

    motor.cleanup()

    assert gpio.pin_values[IN1] is False
    assert gpio.pin_values[IN2] is False
    assert gpio.pwm_channels[PWM].duty_cycle == 0.0
    assert motor.speed == 0.0

from buddy_core.motors.drive import BuddyDrive


FL_IN1 = 20
FL_IN2 = 21
FL_PWM = 22

FR_IN1 = 23
FR_IN2 = 24
FR_PWM = 25

RL_IN1 = 30
RL_IN2 = 31
RL_PWM = 32

RR_IN1 = 33
RR_IN2 = 34
RR_PWM = 35


def create_drive():
    gpio = MockGPIOBackend()

    front_left = TB6612Motor(
        gpio,
        FL_IN1,
        FL_IN2,
        FL_PWM,
    )

    front_right = TB6612Motor(
        gpio,
        FR_IN1,
        FR_IN2,
        FR_PWM,
    )

    rear_left = TB6612Motor(
        gpio,
        RL_IN1,
        RL_IN2,
        RL_PWM,
    )

    rear_right = TB6612Motor(
        gpio,
        RR_IN1,
        RR_IN2,
        RR_PWM,
    )

    drive = BuddyDrive(
        front_left,
        front_right,
        rear_left,
        rear_right,
    )

    return gpio, drive


def test_drive_initializes_all_motors_stopped():
    gpio, drive = create_drive()

    drive.initialize()

    assert drive.initialized is True

    for pin in (
        FL_PWM,
        FR_PWM,
        RL_PWM,
        RR_PWM,
    ):
        assert (
            gpio.pwm_channels[pin].duty_cycle
            == 0.0
        )


def test_drive_forward():
    _, drive = create_drive()
    drive.initialize()

    drive.forward(0.4)

    assert drive.front_left.speed == 0.4
    assert drive.front_right.speed == 0.4
    assert drive.rear_left.speed == 0.4
    assert drive.rear_right.speed == 0.4


def test_drive_reverse():
    _, drive = create_drive()
    drive.initialize()

    drive.reverse(0.4)

    assert drive.front_left.speed == -0.4
    assert drive.front_right.speed == -0.4
    assert drive.rear_left.speed == -0.4
    assert drive.rear_right.speed == -0.4


def test_drive_turn_left():
    _, drive = create_drive()
    drive.initialize()

    drive.turn_left(0.3)

    assert drive.front_left.speed == -0.3
    assert drive.rear_left.speed == -0.3

    assert drive.front_right.speed == 0.3
    assert drive.rear_right.speed == 0.3


def test_drive_turn_right():
    _, drive = create_drive()
    drive.initialize()

    drive.turn_right(0.3)

    assert drive.front_left.speed == 0.3
    assert drive.rear_left.speed == 0.3

    assert drive.front_right.speed == -0.3
    assert drive.rear_right.speed == -0.3


def test_independent_wheel_speeds():
    _, drive = create_drive()
    drive.initialize()

    drive.set_wheel_speeds(
        0.1,
        0.2,
        0.3,
        0.4,
    )

    assert drive.front_left.speed == 0.1
    assert drive.front_right.speed == 0.2
    assert drive.rear_left.speed == 0.3
    assert drive.rear_right.speed == 0.4


def test_drive_stop():
    _, drive = create_drive()
    drive.initialize()

    drive.forward(0.8)
    drive.stop()

    assert drive.front_left.speed == 0.0
    assert drive.front_right.speed == 0.0
    assert drive.rear_left.speed == 0.0
    assert drive.rear_right.speed == 0.0


def test_drive_requires_initialization():
    _, drive = create_drive()

    with pytest.raises(RuntimeError):
        drive.forward(0.5)


@pytest.mark.parametrize(
    "speed",
    [-0.1, 1.1, None, "fast", True],
)
def test_drive_rejects_invalid_command_speed(speed):
    _, drive = create_drive()
    drive.initialize()

    with pytest.raises(ValueError):
        drive.forward(speed)


def test_drive_cleanup_stops_every_motor():
    _, drive = create_drive()
    drive.initialize()

    drive.forward(1.0)
    drive.cleanup()

    assert drive.initialized is False

    assert drive.front_left.speed == 0.0
    assert drive.front_right.speed == 0.0
    assert drive.rear_left.speed == 0.0
    assert drive.rear_right.speed == 0.0


from buddy_core.motors.tb6612 import TB6612Standby


STBY = 40


def test_standby_initializes_disabled():
    gpio = MockGPIOBackend()
    standby = TB6612Standby(gpio, STBY)

    standby.initialize()

    assert gpio.pin_values[STBY] is False
    assert standby.enabled is False


def test_standby_enable():
    gpio = MockGPIOBackend()
    standby = TB6612Standby(gpio, STBY)

    standby.initialize()
    standby.enable()

    assert gpio.pin_values[STBY] is True
    assert standby.enabled is True


def test_standby_disable():
    gpio = MockGPIOBackend()
    standby = TB6612Standby(gpio, STBY)

    standby.initialize()
    standby.enable()
    standby.disable()

    assert gpio.pin_values[STBY] is False
    assert standby.enabled is False


def test_standby_requires_initialization_before_enable():
    gpio = MockGPIOBackend()
    standby = TB6612Standby(gpio, STBY)

    with pytest.raises(RuntimeError):
        standby.enable()


def test_standby_cleanup_disables_driver():
    gpio = MockGPIOBackend()
    standby = TB6612Standby(gpio, STBY)

    standby.initialize()
    standby.enable()
    standby.cleanup()

    assert gpio.pin_values[STBY] is False
    assert standby.enabled is False


from copy import deepcopy

from buddy_core.hardware import BuddyHardwareError
from buddy_core.hardware.mock_gpio import MockGPIOBackend
from buddy_core.motors.factory import create_motor_system


def motor_factory_config():
    return {
        "hardware": {
            "enabled": False,
            "simulation": True,
            "gpio": {
                "numbering_mode": "BCM",
            },
            "motors": {
                "enabled": True,
                "pwm_frequency_hz": 1000,
                "driver_1": {
                    "standby": 1,
                    "front_left": {
                        "in1": 2,
                        "in2": 3,
                        "pwm": 4,
                    },
                    "front_right": {
                        "in1": 5,
                        "in2": 6,
                        "pwm": 7,
                    },
                },
                "driver_2": {
                    "standby": 8,
                    "rear_left": {
                        "in1": 9,
                        "in2": 10,
                        "pwm": 11,
                    },
                    "rear_right": {
                        "in1": 12,
                        "in2": 13,
                        "pwm": 14,
                    },
                },
            },
        },
    }


def test_motor_factory_rejects_disabled_motor_control():
    config = motor_factory_config()
    config["hardware"]["motors"]["enabled"] = False

    with pytest.raises(BuddyHardwareError):
        create_motor_system(config)


def test_motor_factory_rejects_missing_pin():
    config = motor_factory_config()
    config["hardware"]["motors"]["driver_1"][
        "front_left"
    ]["in1"] = None

    with pytest.raises(BuddyHardwareError):
        create_motor_system(config)


def test_motor_factory_rejects_duplicate_pins():
    config = motor_factory_config()

    config["hardware"]["motors"]["driver_2"][
        "rear_right"
    ]["pwm"] = 1

    with pytest.raises(BuddyHardwareError):
        create_motor_system(config)


def test_motor_factory_uses_mock_when_hardware_disabled():
    system = create_motor_system(
        motor_factory_config()
    )

    assert isinstance(
        system.gpio,
        MockGPIOBackend,
    )


def test_motor_system_initializes_with_standby_disabled():
    system = create_motor_system(
        motor_factory_config()
    )

    system.initialize()

    gpio = system.gpio

    assert gpio.pin_values[1] is False
    assert gpio.pin_values[8] is False

    assert (
        system.driver_1_standby.enabled
        is False
    )
    assert (
        system.driver_2_standby.enabled
        is False
    )


def test_motor_system_enable_and_disable_drivers():
    system = create_motor_system(
        motor_factory_config()
    )

    system.initialize()
    system.enable_drivers()

    assert system.gpio.pin_values[1] is True
    assert system.gpio.pin_values[8] is True

    system.disable_drivers()

    assert system.gpio.pin_values[1] is False
    assert system.gpio.pin_values[8] is False


def test_motor_system_cleanup_is_safe():
    system = create_motor_system(
        motor_factory_config()
    )

    system.initialize()
    system.enable_drivers()
    system.drive.forward(0.5)

    system.cleanup()

    assert system.initialized is False
    assert system.gpio.cleaned_up is True

    assert system.drive.front_left.speed == 0.0
    assert system.drive.front_right.speed == 0.0
    assert system.drive.rear_left.speed == 0.0
    assert system.drive.rear_right.speed == 0.0
