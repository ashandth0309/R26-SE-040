# Head tracking - what is implemented, simulated, unverified and blocked

Be strict: **no item below is hardware-verified.** "Tested" means automated tests with mocks,
synthetic audio, a simulated camera, or ROS stand-ins in a Linux sandbox without ROS 2,
a Raspberry Pi, a camera, servos or microphones.

## Test results (this sandbox)

| Suite | Result |
|---|---|
| `test_head_*.py` (8 files) | 238 passed |
| whole `buddy_core/test` (excluding the three ament lint tests, which need `ament_*` not installed here) | 382 passed |
| `buddy_navigation/test` (Yohan's navigation, untouched) | 171 passed |
| `flake8 --max-line-length 99` on all new Python | clean (pre-existing `setup.py` E122 unchanged) |
| mutation check | 35 deliberate breakages of safety logic (limits, velocity/acceleration, dead zone, priority inversion, e-stop, NaN guard, fault latch, time-step clamp, sound sign, sync check, ambiguity, calibration offset, endfire, target jumping, servo signs, activation gate, fingertip mislabel, hand ownership, speaking gate, turn cooldown, hardware gates, config validation, adapter checks) - all caught by at least one test |
| ROS 2 build / `colcon test` | **NOT RUN** (no ROS 2 here) |
| Raspberry Pi run, CPU/RAM | **NOT RUN** |
| Physical servos / PCA9685 / camera / microphones | **NOT RUN** |

## Feature status

| Feature | Implemented in code | Verified how | Still unverified / blocked |
|---|---|---|---|
| A. Voice direction (GCC-PHAT, rejection, filter, events) | yes | synthetic stereo audio; rejection cases; manager rules (speaking gate, cooldown, wake word required) | Accuracy on the real mics unknown. **Blocked** unless the two lavalier mics reach the Pi as ONE two-channel device (separate devices are unsynchronized). Needs a mic-owning publisher (Ashandth's node). Documented fallback: wake word + camera. |
| A. Wake-word interface | yes (`/voice/command` flag, `/head/command wake`, `WakeWordEvent`) | adapter tests | Ashandth's node currently never sets the flag (change #1 in the integration guide). |
| B. Face following | yes | unit + closed-loop simulated camera incl. latency 0-200 ms, inverted servos, moving/crossing targets | Real camera FOV, latency, `VisionPeople` bbox kind (face vs body), servo directions, gains. Sobiya's vision not present in this base branch. |
| C. Hand following | yes, as **wrist fallback** | unit + simulated end-to-end | **No fingertip model exists in the repo.** Fingertip path is code-complete but untested with real data. Wrist positions are not published by Sobiya yet (change #3). |
| D. Servo motion safety | yes | unit + mock PCA9685 (fake `adafruit_servokit`) incl. disconnection | Real mechanical limits, direction signs, centre, speed/accel on real MG90s, brown-out behaviour. Neutral command at start is an unramped step. |
| E. Priority state machine | yes | transition tests, e-stop/stop/fault latches, conflict tests | Real `/safety_state` end to end under ROS. |
| ROS 2 adapter, node, launch, setup entry | yes | adapter unit tests + strict stub node tests | **Never built or run under rclpy/DDS.** `package.xml`/`setup.py` edits untested by colcon. |
| Hardware gating | yes | each gate removed in turn | n/a |
| Idle head movement | yes, off by default | mock tests | not evaluated physically |
| Bring-up and mic-check tools | yes | mock/synthetic tests | never run against hardware |

## Findings about the existing repository (worth acting on)

1. `config/buddy.yaml` neck calibration has tilt `center_angle_deg == max_angle_deg == 180` - no
   upward/downward travel past centre in one direction; likely an uncalibrated placeholder.
2. Sobiya's code has **no fingertip/hand-landmark detection**; wrist and whole-arm gestures only.
3. Ashandth's `voice_node` always publishes `wake_word_detected = False`.
4. `VisionPeople.stamp` is publish time, not capture time; `bbox` face-vs-body is unconfirmed.
5. Base branch has no vision/voice topics at all, only `/camera/image_raw` (`camera_node`) and `/safety_state`.
6. Two lavalier microphones are, by default, two unsynchronized capture devices
   (Ashandth's own `MicTopology.TWO_DEVICES`).

## Remaining hardware dependencies

Measured servo travel/centre/signs; camera FOV and mounting; a synchronized two-channel mic path
and measured spacing (or accept the fallback); colcon build and run on the Pi; Sobiya's vision and
Ashandth's voice integrated; a fingertip model if finger tracking is wanted; power budget check
for two MG90s on the PCA9685 supply.

## Definition of done - honest assessment

Implemented and simulated: turning toward a localized call (given valid stereo audio), visually
acquiring the caller, keeping the face centred, switching to intentional hand following, following
horizontally and vertically, respecting configured limits, recovering safely when the target
disappears. **Not claimed:** any of this working on the real robot. That requires completing
[head_tracking_physical_validation.md](head_tracking_physical_validation.md).
