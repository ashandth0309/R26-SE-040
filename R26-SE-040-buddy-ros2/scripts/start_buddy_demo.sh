#!/usr/bin/env bash


REPO="/home/ashandth/R26-SE-040"
VENV="/home/ashandth/buddy-servo-env"

cd "$REPO"

source /opt/ros/jazzy/setup.bash

if [ -f "$REPO/buddy_ws/install/setup.bash" ]; then
    source "$REPO/buddy_ws/install/setup.bash"
fi

source "$VENV/bin/activate"

export PYTHONPATH="$REPO:$REPO/buddy_ws/src/buddy_core:${PYTHONPATH:-}"

sleep 12

cleanup() {
    kill "$VOICE_PID" 2>/dev/null || true
    kill "$MOTOR_PID" 2>/dev/null || true
    kill "$ULTRA_PID" 2>/dev/null || true
    kill "$SAFETY_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

unset BUDDY_CONFIG_OVERRIDE

ros2 run buddy_core safety_supervisor \
    >> "$REPO/logs/safety_supervisor.log" 2>&1 &
SAFETY_PID=$!

sleep 2

BUDDY_CONFIG_OVERRIDE="$REPO/config/buddy_ultrasonic_physical.yaml" \
ros2 run buddy_core obstacle_sensor \
    >> "$REPO/logs/obstacle_sensor.log" 2>&1 &
ULTRA_PID=$!

sleep 2

BUDDY_CONFIG_OVERRIDE="$REPO/config/buddy_motion_physical.yaml" \
ros2 run buddy_core motor_executor \
    >> "$REPO/logs/motor_executor.log" 2>&1 &
MOTOR_PID=$!

sleep 2

unset BUDDY_CONFIG_OVERRIDE

python3 \
"$REPO/buddy_ws/src/buddy_core/buddy_core/voice/buddy_voice.py" \
    >> "$REPO/logs/buddy_voice.log" 2>&1 &
VOICE_PID=$!

wait "$VOICE_PID"
