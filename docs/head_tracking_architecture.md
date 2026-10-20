# BUDDY intelligent head tracking - architecture

Branch: `feature/buddy-intelligent-head-tracking` (from `origin/Yohan_Mapping`).
Package: `buddy_ws/src/buddy_core/buddy_core/head_tracking/`.
Status of every claim below is in [head_tracking_status.md](head_tracking_status.md).
**Nothing here has been run on the robot, on a Raspberry Pi, or under a real
ROS 2 build.** Tests are unit, mock-integration and ROS-stub tests only.

## 1. What it does

Connects three inputs to Buddy's existing two-axis neck:

1. **Voice direction** - after a wake word, estimate where the sound came from
   (two microphones, GCC-PHAT) and turn the head roughly that way.
2. **Face following** - keep one person's face near the image centre.
3. **Hand following** - on an explicit activation, follow one hand.
   **Today this follows the WRIST, not the fingertip** (section 7).

It reuses `NeckController` / `get_neck_backend` / `PCA9685NeckBackend`
unchanged. It never touches locomotion.

## 2. Data flow

```
 wake word ─┐                              ┌────────────────────────────┐
 sound est ─┼─> HeadTrackingManager ──────>│ HeadMotionCoordinator      │──> NeckController ──> PCA9685 / mock
 faces ─────┤   (state machine, who owns   │  priority arbiter          │     (clamps to
 hand/gest ─┤    the head right now)       │  slew / velocity / accel   │      calibrated limits)
 safety ────┤                              │  dead zone, min step       │
 commands ──┘                              │  e-stop / fault latches    │
                                           └────────────────────────────┘
   FaceFollower  HandFollower  SoundDirectionEstimator/Filter  (pure logic, no I/O)
   ros_adapter.py (messages/JSON -> calls)    ros_node.py (thin rclpy wrapper)
```

Only the coordinator talks to the neck. The manager submits targets for the
current owner only; the coordinator arbitrates again as a second line of defence.

| Module | Role |
|---|---|
| `models.py` | plain dataclasses/enums (no ROS, numpy or other subsystem imports) |
| `config.py` | validated config; unknown keys rejected |
| `motion.py` | the only neck writer: arbitration, limits, latches, shutdown |
| `servo.py` | image error -> angle correction (P/PI/PID, dead zone, smoothing) |
| `face_follow.py` | target lock, no target jumping, caller preference |
| `hand_follow.py` | explicit-activation hand follower, wrist fallback |
| `sound_direction.py` | GCC-PHAT, rejection logic, agreement filter |
| `manager.py` | state machine and policy |
| `factory.py` | builds the system; **hardware gating** |
| `ros_adapter.py`, `ros_node.py` | transport |
| `bringup.py`, `mic_check.py` | physical validation tools |

## 3. State machine

`IDLE -> LISTENING -> SOUND_ORIENTING -> FACE_TRACKING -> HAND_TRACKING -> IDLE`
plus three latched states that can be entered from anywhere: `STOPPED`
(explicit user stop), `EMERGENCY_STOP`, `FAULT`.

| From | Allowed next |
|---|---|
| IDLE | LISTENING, FACE_TRACKING, HAND_TRACKING |
| LISTENING | IDLE, SOUND_ORIENTING, FACE_TRACKING, HAND_TRACKING |
| SOUND_ORIENTING | IDLE, FACE_TRACKING, LISTENING, HAND_TRACKING |
| FACE_TRACKING | IDLE, LISTENING, SOUND_ORIENTING, HAND_TRACKING |
| HAND_TRACKING | FACE_TRACKING, IDLE |
| STOPPED / EMERGENCY_STOP / FAULT | IDLE (only after the latch clears) |

Illegal transitions are refused and reported as `COMMAND_REJECTED`.
When two latches are set, `EMERGENCY_STOP` is shown over `FAULT` over `STOPPED`,
and clearing one never clears another.

Typical flow: wake word -> LISTENING; two agreeing, confident sound estimates ->
`HEAD_TURN_REQUESTED`, SOUND_ORIENTING; turn reaches target and a face is seen ->
FACE_TRACKING (caller preferred if association is reliable); activation gesture
or command -> HAND_TRACKING; stop gesture / hand lost / 30 s -> back to FACE_TRACKING
or IDLE; face lost -> IDLE and a slewed return to neutral.

## 4. Priorities

| # | Source | How it is enforced |
|---|---|---|
| 1 | emergency stop / hardware fault | latched flags in the coordinator; blocks every submit; clears pending targets; optional servo release |
| 2 | servo safety limits | always-on clamp (calibrated limits **plus** `limit_margin_deg`), on top of `NeckController`'s own clamp |
| 3 | explicit user stop | latched flag; `resume` required |
| 4 | hand following | `Priority.HAND` |
| 5 | active speaker (caller) face / sound turn | `Priority.SPEAKER_FACE`, only while the caller association is reliable and within `speaker_priority_s` |
| 6 | passive face tracking | `Priority.PASSIVE_FACE` |
| 7 | idle / neutral | `Priority.IDLE` |

The lowest number that has an unexpired target wins; others are ignored and
counted in `suppressed_conflicts`. The manager also stops submitting from lower
owners (e.g. face targets are not sent while HAND_TRACKING), so in normal operation
that counter stays 0. Targets expire after `request_ttl_s`: if a producer dies the
head holds, and after `neutral_after_s` returns to neutral.

## 5. Motion safety (Feature D)

Per axis: dead zone (only gates *starting* a move), velocity limit, acceleration limit
with a braking curve (no overshoot, ends exactly on the target), minimum write step,
time-step clamp (`max_dt_s`) so a stalled loop cannot cause a jump, inward limit
margin, per-axis limits from the existing `hardware.neck` calibration.
Rejected without motion: NaN/inf, bool, strings, `None`, unknown sources, bad speed scale.
A servo write error (`BuddyNeckError`, e.g. I2C unplugged) latches `FAULT`, stops all
writes (no retry storm), tries to release the servos, and requires an operator
`reset_fault`. `start()` commands the neutral pose - **servos have no position
feedback, so that first command is an unramped step from wherever the head is.**
`shutdown()` slews to neutral, releases and cleans up; it is idempotent and moves
nothing while a latch is set.

Known property of the *current* calibration (`config/buddy.yaml`): tilt
`center_angle_deg` equals `max_angle_deg` (180), so there is no tilt travel past
centre in one direction, and the 3 degree margin pulls neutral to 177. Re-measure.

## 6. Hardware gating

Real servos move only if **all** are true; otherwise `build_head_system` raises
`HeadTrackingConfigError` listing every missing gate (nothing falls back silently):

1. `head_tracking.hardware.mode: hardware`
2. `hardware_arm: true`
3. `calibration_confirmed: true`
4. `pan_sign` and `tilt_sign` set to +1/-1
5. `buddy.yaml`: `hardware.enabled: true`, `hardware.simulation: false`,
   `hardware.neck.enabled: true`
6. environment variable `BUDDY_HEAD_HARDWARE_ARMED=1` (never set by any launch file)

Mock mode (the default) always uses `MockNeckBackend`, even if `buddy.yaml`
enables the physical neck. Each gate has a test that removes it alone.

## 7. Voice direction, face and hand: what is real

**Sound.** `sound_direction.py` implements GCC-PHAT (band-limited, interpolated, with
calibration offset, spacing, channel swap). It refuses to estimate unless the caller
says the two channels share one clock (`synchronized=True`) and rejects: silent or
clipped audio, short windows, weak correlation peak, low peak-to-background ratio,
two competing sources, delays larger than the microphone geometry allows, directions
near the microphone axis, and (via `SoundDirectionFilter`) estimates that do not agree.
Front/back cannot be told apart with two microphones (`front_back_ambiguous` is always
true); the camera decides. Resolution at 16 kHz and 15 cm is about 8 degrees per raw
sample (`resolution_deg` is reported). **Accuracy on the real microphones is unknown.**
Two lavalier microphones are normally two *separate* capture devices with independent
clocks; Ashandth's `audio_devices.py` calls that topology `TWO_DEVICES` and says their
streams "must never be summed sample by sample". They only work here if they are
wired into one stereo input (e.g. a 2-channel ADC/sound card). If not, the documented
fallback is: wake word + camera only (face acquisition decides the direction); the
`SOUND_DIRECTION_UNCERTAIN` event is emitted with reason `microphones_not_synchronized`.
Synthetic numbers (noise-only peak <= 0.05, coherent signal to about -5 dB SNR, equal
two-source ratio about 1.0) justify the *thresholds*, not a real-world accuracy.

**Face.** Inputs are `FaceFrame` (boxes in pixels + opaque track ids). Target lock is
sticky (no jumping to a larger or more central newcomer), needs 3 sightings to acquire,
holds the pose for `lost_hold_s` when briefly missing, then goes LOST. A caller
association is made only when exactly one face is valid, or the most central face is
`assoc_margin` times closer to the centre than the next; otherwise no guess.
Camera latency is compensated approximately: correction is applied to the pose at
frame time (`position_at`). The real capture time is not available from Sobiya's
message (see integration guide).

**Hand.** Sobiya's code has body pose (wrist landmarks) and arm gestures only; there is
**no hand-landmark or fingertip model integrated** (her `gestures.py` says so; the palm
detector is staged in her model manifest but unused). Therefore hand following
labels every result `wrist_fallback`; `HAND_ACTIVATED` events carry
`accurate_fingertip: false`. A wrist can be 20-30 cm from the pointing finger. If a
future model fills `HandObservation.fingertips`, the same code uses `index_fingertip`.
Activation is explicit (command or configured gesture, default `raised_hand`, only
from the person already being tracked), one person and one hand, 30 s maximum.

## 8. ROS 2 interfaces

Only interfaces that exist are used. Table of what is present where:

| Topic | Type | In base branch? | Direction |
|---|---|---|---|
| `/safety_state` | `buddy_interfaces/SafetyState` | yes | in (e-stop) |
| `/vision/people` | `buddy_interfaces/VisionPeople` | **no** (Sobiya) | in, optional |
| `/voice/command` | `buddy_interfaces/VoiceCommand` | message yes; publisher on Ashandth | in, optional |
| `/voice/status` | `std_msgs/String` JSON | no (Ashandth) | in, optional |
| `/head/command` | `std_msgs/String` | new | in |
| `/head/hand_observation` | `std_msgs/String` JSON | new | in |
| `/head/sound_direction` | `std_msgs/String` JSON | new | in |
| `/head/status` | `std_msgs/String` JSON, 1 Hz | new | out |
| `/head/events` | `std_msgs/String` JSON per event | new | out |

No camera or microphone subscription exists: **the node owns neither**, so it cannot
compete for them. Frame size comes from `buddy.yaml` `camera.width/height` (or
parameters). No new `.msg` files were added (avoids clashing with the other branches);
typed messages can replace the JSON topics later.

JSON schemas: see `ros_adapter.py` docstrings. `/head/command` text: `stop`,
`resume`, `reset_fault`, `hand_on [left|right]`, `hand_off`, `wake` (test).
Published status/events never contain track ids or names.

Parameters of `head_tracking_node`: `head_config_path`, `buddy_config_path`,
`frame_width`, `frame_height`, topic names (`*_topic`). Everything else is in
`config/head_tracking.yaml`. Launch: `launch/head_tracking.launch.py` (mock by default).
Entry point `head_tracking_node` was added to `setup.py`; `package.xml` gained
`std_msgs`, `python3-numpy`, `launch_ros` exec dependencies. **Not built with colcon.**

## 9. Resource use

Pure Python + numpy. Per tick: a few dozen arithmetic operations. Sound estimation
runs only when a window is supplied (one FFT of ~0.25 s of audio). No image is ever
handled here. CPU/RAM on the Pi 4B were **not measured**.
