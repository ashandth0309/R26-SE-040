#!/usr/bin/env bash
# Start BUDDY with the voice COMPANION engine instead of the old demo voice.
# Does not replace scripts/start_buddy_demo.sh (the validated demo).
#
# Required environment (set in the service file or your shell):
#   BUDDY_VOSK_MODEL   path to an unpacked Vosk model directory
#   BUDDY_PIPER_MODEL  path to a Piper .onnx voice
# Optional:
#   BUDDY_REPO         repo path (default /home/ashandth/R26-SE-040)
#   BUDDY_VENV         python venv (default /home/ashandth/buddy-servo-env)
#   BUDDY_VOICE_ARGS   extra ros2 args, e.g. "-p profile:=lightweight"
#
# UNVERIFIED: not yet run on the robot.

REPO="${BUDDY_REPO:-/home/ashandth/R26-SE-040}"
VENV="${BUDDY_VENV:-/home/ashandth/buddy-servo-env}"

cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

source /opt/ros/jazzy/setup.bash

if [ -f "$REPO/buddy_ws/install/setup.bash" ]; then
    source "$REPO/buddy_ws/install/setup.bash"
fi

source "$VENV/bin/activate"

export PYTHONPATH="$REPO:$REPO/buddy_ws/src/buddy_core:${PYTHONPATH:-}"

if [ -z "$BUDDY_VOSK_MODEL" ] || [ -z "$BUDDY_PIPER_MODEL" ]; then
    echo "Set BUDDY_VOSK_MODEL and BUDDY_PIPER_MODEL first." >&2
    exit 2
fi

if ros2 node list 2>/dev/null | grep -q "/buddy_voice$"; then
    echo "The old buddy_voice node is running. Stop it first." >&2
    exit 3
fi

sleep "${BUDDY_START_DELAY:-12}"

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

# shellcheck disable=SC2086
ros2 run buddy_core voice_companion --ros-args \
    -p "vosk_model_path:=$BUDDY_VOSK_MODEL" \
    -p "piper_model_path:=$BUDDY_PIPER_MODEL" \
    $BUDDY_VOICE_ARGS \
    >> "$REPO/logs/voice_companion.log" 2>&1 &
VOICE_PID=$!

wait "$VOICE_PID"
