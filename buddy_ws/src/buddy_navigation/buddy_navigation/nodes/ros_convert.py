"""Conversions between ROS messages and the ROS-free navigation types."""

from __future__ import annotations

import math

from buddy_navigation.core.commands import MoveRequest
from buddy_navigation.core.sensor_model import RangeSample, SensorStatus


def range_to_sample(
    name: str,
    value: float,
    min_range: float,
    max_range: float,
    timestamp_s: float,
) -> RangeSample:
    """Decode a ``sensor_msgs/Range`` value (REP-117 encoding)."""

    if math.isnan(value):
        return RangeSample(name, SensorStatus.FAULT, None, timestamp_s)

    if value == math.inf or value > max_range:
        return RangeSample(name, SensorStatus.CLEAR, None, timestamp_s)

    if value == -math.inf or value < min_range:
        return RangeSample(name, SensorStatus.TOO_CLOSE, None, timestamp_s)

    return RangeSample(name, SensorStatus.OK, float(value), timestamp_s)


def fill_movement_command(message, request: MoveRequest, sequence: int, stamp):
    """Populate a ``buddy_interfaces/MovementCommand``."""

    message.stamp = stamp
    message.sequence_id = sequence
    message.source = request.source
    message.command = int(request.code)
    message.linear_speed = float(request.speed)
    message.angular_speed = float(request.speed)
    seconds = float(request.valid_for_s)
    message.valid_for.sec = int(seconds)
    message.valid_for.nanosec = int((seconds % 1.0) * 1e9)
    message.emergency = bool(request.emergency)

    return message
