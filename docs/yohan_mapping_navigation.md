# Autonomous Mapping, Obstacle Avoidance, Navigation and Security Patrol

Author: S. S. Y. Wickramasinghe (branch `Yohan_Mapping`, from `buddy-ros2`)

This document describes the navigation component of BUDDY: what exists, how
it is wired into the existing ROS 2 stack, how to run it in simulation, and,
just as important, what it cannot do yet. It separates three things
throughout:

* **Verified here**: executed in the development sandbox, which has no ROS 2
  and no Raspberry Pi.
* **Simulation only**: behaviour shown against the mock world.
* **Not yet verified on hardware**: anything that needs the robot.

Nothing in this document claims SLAM, natural-language navigation or
validated physical patrol.

---

## 1. Architecture

```
 4 x HC-SR04 --> ultrasonic_array (sequential) --> obstacle_node
                                                     |  /ultrasonic/<name> (Range, REP-117)
                                                     |  /obstacle_status   (unchanged semantics)
                                                     |  /diagnostics
                     voice / vision                  v
                 /navigation/command  --> navigation_node ---> /movement/request
                                            |  (only while a goal runs)     |
                                            |                               v
                      /navigation/status <--+                      safety_supervisor
                      /navigation/local_map                         (existing, unchanged)
                      /patrol_route (action)                                |
                                                                            v /movement/safe
                                                                      motor_executor
                                                                       (existing, unchanged)
```

Design rules:

* One motor path only. Navigation publishes `MovementCommand` on
  `/movement/request`. The existing safety supervisor and motor executor are
  untouched and still validate everything. There is no second motor driver.
* One ultrasonic driver only. The new `UltrasonicArrayScanner` and the
  `measure()` method are additions to `buddy_core.sensors.ultrasonic`; the
  legacy `read_distance_m()` API is unchanged.
* Navigation adds stricter checks in front of the supervisor rather than
  bypassing it. It never sets the `emergency` flag.
* Pure-Python core (`buddy_navigation/core`, `sim`, `perception`) with thin
  ROS wrappers (`nodes`), so the logic is unit tested without ROS.

Packages touched:

| Package | Change |
| --- | --- |
| `buddy_core` | Additive: `UltrasonicMeasurement`, `UltrasonicSensor.measure()`, `UltrasonicArrayScanner`, extended `obstacle_node` (per-sensor Range, diagnostics). `/obstacle_status` output unchanged. |
| `buddy_navigation` | New: obstacle assessment, local map, grid and Dijkstra planner, coordinator, patrol, service, config, camera, simulator, nodes, launch. |
| `buddy_nav_interfaces` | New: `NavigationCommand.msg`, `NavigationStatus.msg`. |
| `buddy_interfaces` | Unchanged. `PatrolRoute.action` is reused as is. |

Voice, face, emotion and ML modules were not edited.

## 2. Hardware and GPIO assumptions

Fixed hardware: Raspberry Pi 4, four ultrasonic sensors (front left, front
right, rear left, rear right), Pi Camera Module 2, existing motor and servo
electronics. No LiDAR, ToF, depth camera, encoders or IMU are assumed.

Ultrasonic wiring as it appears in `config/buddy.yaml` (BCM numbering):

| Sensor | Trigger | Echo |
| --- | --- | --- |
| front_left | 21 | 22 |
| front_right | 23 | 24 |
| rear_left | 25 | 26 |
| rear_right | 27 | 14 |

Points to confirm on the robot:

* HC-SR04 modules usually drive echo at 5 V. Each echo line needs a divider
  or level shifter before the 3.3 V GPIO. This cannot be checked from
  software.
* BCM 14 is the default UART TX pin on a Raspberry Pi. Make sure the serial
  console is disabled on that pin, or move the rear-right echo.
* `docs/gpio_plan.md` still lists the obstacle sensor pins as TBD while
  `config/buddy.yaml` assigns them. The two should be reconciled by the
  hardware owner. This branch does not edit that file.
* Echo wiring was not measured or probed; the pin numbers are taken from the
  repository config.

## 3. Sensor placement

Sensor mounting poses, fields of view and the body footprint are
**configuration, not facts**. `config/navigation.yaml` ships placeholder
values (corner mounts, 20 degree toe-out, 30 degree beam, 30 x 20 cm body)
and sets `geometry_confirmed: false`.

While `geometry_confirmed` is false the loader forces `allow_motion` to
false, whatever the file says. To enable autonomous motion:

1. Measure each sensor position (x forward, y left, metres from the body
   centre) and its yaw.
2. Measure the body length and width.
3. Measure the real beam width (it is not guaranteed to be 30 degrees).
4. Decide the coverage thresholds consciously (section 7).
5. Set `geometry_confirmed: true` and `allow_motion: true`.

Important finding (simulation of the assumed geometry, not a measurement):
four corner-mounted beams of about 30 degrees cover roughly 85 percent of the
forward stop zone and roughly 40 percent of the area swept by pivoting on the
spot, largely regardless of toe angle. With the strict default thresholds the
assessor therefore answers STOP. This is deliberate: the robot refuses to
guess about space it cannot observe. Adding a fifth sensor, a centre-front
sensor or a sweeping mount would change this; that is a hardware decision.

## 4. Camera pipeline

`buddy_navigation/perception/camera.py`:

* Backends: `picamera2` (libcamera, preferred on the Pi), `opencv`
  (`cv2.VideoCapture`), `mock` (default, synthetic frames).
* Configurable resolution and fps, monotonic timestamps, measured fps,
  consecutive-failure tracking, automatic release after repeated failures,
  idempotent `stop()`, diagnostics dictionary.
* `FrameBus`: one process owns the camera and any number of consumers
  subscribe. A failing consumer cannot stop capture or starve the others.
* `camera_node` publishes `/camera/image_raw` (`bgr8`, built without
  `cv_bridge`) and `/diagnostics`.
* `landmarks.py`: optional ORB feature extraction and matching, plus a pinhole
  calibration container. Experimental. Camera detection is **not** used for
  distance sensing or for any safety decision.

Camera ownership: libcamera normally allows a single owner. The face module
(Sobiya) already uses the camera. Agree one owner and have everyone else
subscribe to `/camera/image_raw`. `camera_node` must not run next to another
camera owner. This integration was not done; it needs a team decision.

## 5. Mapping approach

There is no reliable pose on the current hardware (no encoders, no IMU), so
the mapping is staged honestly:

1. **Robot-relative local map (real robot)**. `LocalObstacleMap` is a
   log-odds grid (4 m square, 5 cm cells). An echo does not locate an object;
   it says the nearest reflector inside a cone is at range r. The map
   therefore:
   * adds only moderate occupancy belief on the whole arc at range r, never a
     single precise point;
   * adds free evidence inside the cone in front of the arc;
   * leaves unobserved cells UNKNOWN and never treats unknown as free;
   * lets every cell expire back to UNKNOWN (default 3 s);
   * ignores faulty, stale or missing sensors instead of reading them as
     free;
   * handles "too close" by marking the zone next to the sensor occupied.
   Published as `/navigation/local_map` in the `base_link` frame.
2. **Pose-based map (simulation only)**. The same class can be fed a pose
   and then acts as a world-fixed map. Only the simulator provides a pose.
3. **Visual landmarks (experimental)**. ORB features exist as a building
   block. No landmark map, loop closure or localisation is implemented.

This is not SLAM and must not be described as such.

## 6. Dijkstra planner

`core/grid.py` and `core/planner.py`:

* `OccupancyGrid2D` follows the `nav_msgs/OccupancyGrid` convention
  (-1 unknown, 0..100 percent).
* `PlanningGrid` builds a blocked-cell grid from an occupancy grid with
  circular inflation by the robot radius and a policy for unknown cells
  (blocked by default).
* `dijkstra()` supports 4 or 8 connectivity, forbids diagonal corner cutting
  by default, reconstructs the path, and returns a status instead of raising:
  `OK`, `START_OUT_OF_BOUNDS`, `GOAL_OUT_OF_BOUNDS`, `START_BLOCKED`,
  `GOAL_BLOCKED`, `NO_PATH`.
* The planner knows nothing about motors; it works on cells.

The test scenario is a simple 6x6 grid inspired by the university
robotics-assignment exercise, with 1 m cells. It is not a map of BUDDY's real
environment.

## 7. Obstacle decision logic

`core/sensor_model.py` and `core/obstacle_assessment.py`:

* `SensorHealthMonitor` keeps per-sensor state: OK, CLEAR, TOO_CLOSE, FAULT,
  STALE (older than 0.8 s), NO_DATA. Distances use the smaller of the latest
  value and a 3-sample median. A CLEAR that follows a recent near reading is
  not trusted until three consecutive CLEARs.
* HC-SR04 semantics used: no rising edge means a fault; a rising edge with no
  falling edge inside the timeout means nothing in range; a pulse shorter than
  the minimum means too close; an echo stuck high means a fault. Soft or
  angled surfaces can also look like "nothing in range", so CLEAR is trusted
  only with confirmation.
* `ObstacleAssessor` samples the area the robot would sweep (forward,
  backward, rotate left, rotate right). Each sample point is FREE, OCCUPIED
  (on an echo arc, within tolerance) or UNKNOWN (not covered, or shadowed by a
  nearer echo). A minimum obstacle width probe (default 12 cm) stops thin
  objects slipping between beams.
* Decision order per motion: TOO_CLOSE gives REPLAN; an obstacle in the stop
  zone (0.25 m from the body) gives REPLAN; coverage below the required
  fraction gives STOP; an obstacle in the slow zone (0.60 m) gives SLOW (half
  speed); partial coverage gives SLOW; otherwise PROCEED.
* It also reports front blocked, reverse safe, left/right clearance, the side
  with more room, missing and unreliable sensors. It does not implement
  "front blocked means turn right": a turn is itself assessed as a motion and
  is refused if its swept area is not observed.
* Strict coverage defaults of 1.0 mean the physical profile will STOP until
  thresholds are chosen. The relaxed thresholds (0.80 linear, 0.35 rotation)
  live only in the simulation preset.

## 8. Emergency stop and safety behaviour

Layers, from outermost to innermost:

1. `NavigationConfig.allow_motion` is false by default and is forced false
   until the geometry is confirmed.
2. The coordinator returns STOP and fails the goal on an emergency stop
   (mirrored from `/safety_state`) or a manual override.
3. Missing, stale or faulty sensors, or insufficient coverage, stop motion.
4. Path and patrol goals are refused without a trusted pose.
5. Speeds are clamped (defaults: 0.2 linear, 0.3 rotation, the previously
   calibrated demo values); SLOW halves them.
6. Requests are valid for 0.3 s and are republished continuously, so a stalled
   or crashed navigation node stops the robot within about a second through
   the existing motor executor expiry.
7. The existing `safety_supervisor` still evaluates every request, still owns
   `/emergency_stop` and `/reset_emergency_stop`, and navigation never calls
   them or clears the latch.
8. On node shutdown an explicit STOP is published.

Manual override: the navigation node is silent when idle so it cannot fight
the voice module. If any other source publishes a non-STOP request on
`/movement/request` while a goal runs, the goal is aborted.

Known gaps in the existing supervisor (documented, deliberately not changed
because that would alter the validated voice demo):

* When both sensors of a pair return no distance, the legacy
  `/obstacle_status` pair is "invalid" and not "blocked", and the supervisor
  fails open for it. A characterisation test records this. Navigation's own
  assessment handles the case, but direct voice commands still pass through
  the original supervisor behaviour.
* The supervisor does not clamp speed.

Unsafe situations the design cannot fix: a sensor that silently reports
"clear" while an obstacle is present (specular reflection, soft fabric,
sensor drifting). The simulator can inject this (`specular`); with a single
covering sensor it is undetectable. Stairs and drops are out of scope; use on
flat, controlled floors only.

## 9. ROS 2 nodes, topics and interfaces

| Node | Executable | Role |
| --- | --- | --- |
| `buddy_obstacle_sensor` | `obstacle_sensor` (buddy_core) | Sequential ultrasonic scan, legacy status, per-sensor Range, diagnostics |
| `buddy_navigation` | `navigation_node` | Commands, guarded motion, patrol, local map, status |
| `buddy_camera` | `camera_node` | Camera ownership, `/camera/image_raw`, diagnostics |
| `buddy_navigation_sim` | `sim_node` | Simulated sensors and robot (simulation only) |

Topics:

| Topic | Type | Direction (navigation) |
| --- | --- | --- |
| `/ultrasonic/{front_left,front_right,rear_left,rear_right}` | `sensor_msgs/Range` (NaN = fault, +inf = clear, -inf = too close) | in |
| `/obstacle_status` | `buddy_interfaces/ObstacleStatus` | unchanged, consumed by supervisor |
| `/navigation/command` | `buddy_nav_interfaces/NavigationCommand` | in |
| `/navigation/status` | `buddy_nav_interfaces/NavigationStatus` | out |
| `/navigation/local_map` | `nav_msgs/OccupancyGrid` | out |
| `/movement/request` | `buddy_interfaces/MovementCommand` | out, plus override detection in |
| `/safety_state` | `buddy_interfaces/SafetyState` | in |
| `/sim/pose` | `geometry_msgs/PoseStamped` | in (simulation only) |
| `/camera/image_raw` | `sensor_msgs/Image` | camera node out |
| `/patrol_route` | `buddy_interfaces/action/PatrolRoute` | action server |

### Interface for future voice and vision requests

Publish a `NavigationCommand` on `/navigation/command`:

| Spoken or visual request | `command` | Fields |
| --- | --- | --- |
| "stop" | `COMMAND_STOP` | none |
| "move forward" | `COMMAND_MOVE_FORWARD` | `duration_s` (0 = default 1 s, capped at 5 s) |
| "turn left / right" | `COMMAND_TURN_LEFT` / `COMMAND_TURN_RIGHT` | `duration_s` |
| "patrol" | `COMMAND_START_PATROL` | `route_id`, `repeat` |
| "go to the kitchen" | `COMMAND_GO_TO_LOCATION` | `location_name` (must exist in config) |
| cancel, pause, resume | `COMMAND_CANCEL`, `COMMAND_PAUSE`, `COMMAND_RESUME` | none |

Replies arrive on `/navigation/status` (state, reason, `request_id`,
`motion_enabled`, `pose_trusted`, `sensors_healthy`, `emergency_stop_active`,
progress), which gives the voice module a reason it can read out when a
request is refused. Timed moves are open loop until the chassis is
calibrated. This is an interface, not natural-language understanding: the
voice and vision modules must still turn speech or images into these
commands. No end-to-end voice navigation is claimed.

Patrol: `PatrolRoute.action` goal `route_id` / `repeat`; feedback `state`,
`current_waypoint`, `total_waypoints`, `progress`; result `success`,
`reason`. Routes and locations are user-defined in the YAML config. Patrol
only starts when the pose is trusted and all four sensors are healthy, plans
each leg with Dijkstra, stops on obstacles, supports pause, resume, cancel
and repeat, and writes structured JSON event logs (`patrol_started`,
`leg_started`, `waypoint_reached`, `lap_completed`, `patrol_failed`, ...).

## 10. Simulation and demo instructions

No hardware is needed. On a machine with ROS 2 Jazzy:

```bash
cd buddy_ws
colcon build --packages-select buddy_interfaces buddy_nav_interfaces buddy_core buddy_navigation
source install/setup.bash
ros2 launch buddy_navigation simulation.launch.py
```

This starts the simulator, the real safety supervisor and the navigation node
(simulation profile). In another terminal:

```bash
# go to a named location (needs the simulated trusted pose)
ros2 topic pub --once /navigation/command buddy_nav_interfaces/msg/NavigationCommand \
  "{command: 5, location_name: corner, request_id: demo1, source: manual}"

# patrol the demo route
ros2 topic pub --once /navigation/command buddy_nav_interfaces/msg/NavigationCommand \
  "{command: 4, route_id: demo_loop, request_id: demo2, source: manual}"

ros2 topic echo /navigation/status
```

Fault injection: `ros2 launch buddy_navigation simulation.launch.py
fault_sensor:=front_left fault_mode:=no_response` (other modes: `stuck_high`,
`specular`, `frozen`).

Without ROS, the same logic runs in Python:

```bash
cd buddy_ws/src/buddy_navigation
python -m pytest test -p no:logging
```

`buddy_navigation/sim/harness.py` (`SimHarness`) runs the coordinator against
the mock world and the real `SafetySupervisor` class in a closed loop.

## 11. Raspberry Pi deployment

Do these in order and keep the robot on blocks (wheels off the ground) until
step 6.

1. Build the workspace as above on the Pi (ROS 2 Jazzy, Ubuntu 24.04).
2. Verify the echo level shifting and the BCM 14 caveat (section 2).
3. Run the obstacle node with hardware enabled in `config/buddy.yaml` and
   check `/ultrasonic/*` and `/diagnostics` with objects at known distances.
   Record the real beam width and minimum range.
4. Measure the geometry (section 3) and fill in `navigation.yaml`.
5. Characterise speed: metres per second and radians per second at the
   configured speeds, and the stopping distance. The simulator's motion
   model (0.35 m/s and 2 rad/s at speed 1.0) is an assumption, not a
   measurement.
6. Only then set `geometry_confirmed: true` and `allow_motion: true`, start
   with `navigation.launch.py`, and try a short timed forward motion in an
   empty, flat, supervised area with a hand on the emergency stop.
7. Camera: choose the single owner (section 4) before starting `camera_node`.

Path and patrol goals cannot run on the real robot until a trusted pose
source exists (`pose_source: odom` is reserved and not implemented).

## 12. Test results

Executed in the development sandbox (Python, no ROS 2, no hardware):

| Suite | Result |
| --- | --- |
| `buddy_core` unit tests (ament lint tests excluded) | 144 passed (120 existing plus 24 new ultrasonic tests) |
| `buddy_navigation` tests | 171 passed |
| `flake8` on new and changed files | clean |
| `git diff --check` | clean |

`buddy_navigation` tests by file: sensor model 20, obstacle assessment 24,
obstacle node (stubbed) 10, planner 16, local map 15, simulation blocks 17,
closed-loop coordinator 15, patrol 10, config 10, service 12, camera 12, node
wrappers (stubbed) 10.

What the tests cover: sequential triggering and minimum spacing, invalid and
timed-out readings, stuck echo, stale data, REP-117 encoding, local map arcs
and expiry, Dijkstra optimality (cross-checked against breadth-first search on
random grids), blocked and unreachable goals, cancellation, pause and resume,
emergency stop, manual override, unsafe-motion rejection, speed limits,
motion disabled by default, sensor timeout stopping the robot, and an
obstacle appearing mid-route.

Mutation checks: for the key safety rules (guard removal, e-stop,
motion-disabled default, pose gating, override detection, speed scaling,
precondition checks, planner corner cutting) the rule was temporarily broken
and the suite was confirmed to fail. One mutant (ignoring the `usable` flag
in the assessor) survived as equivalent behaviour. The slow-zone speed scaling
had no test until a mutation exposed it; a test was added.

**Not run, because this environment has no ROS 2:**

* `colcon build`, the `rosidl` message generation for `buddy_nav_interfaces`,
  ament lint tests, `ros2 launch`, DDS/QoS behaviour, the executor, the
  patrol action server, and any `launch_testing`.
* Node wrapper tests use strict stubs generated from the repo's `.msg` files.
  They catch wrong field names and topics but are logic checks, not proof of
  ROS behaviour.
* Nothing was run on the Raspberry Pi or with real sensors, camera or motors.

## 13. Limitations

* No pose source. Path following, go-to-location and patrol work only with the
  simulator's ground-truth pose.
* No real measurements: sensor geometry, beam width, speeds, stopping
  distance and footprint are placeholders. Motion is disabled until they are
  confirmed.
* Four corner beams cannot observe the whole swept area. The strict physical
  defaults will refuse most autonomous motion; relaxing them is a safety
  decision for the team.
* Ultrasonic limitations: wide beams (bearing ambiguity), soft or angled
  surfaces that return no echo, crosstalk risk (mitigated by sequential
  triggering), temperature dependence of the speed of sound (not
  compensated), and a roughly 2 cm blind zone.
* Silent "clear" failures cannot be detected.
* The simulator is a 2D analytic ray model with simple noise. It does not
  model real acoustics, wheel slip or tilt.
* The local map is robot-relative and short lived; it is not a persistent
  map and it is not shared with the planner on the real robot.
* The camera path is not used for safety. Landmarks are experimental.
* The legacy supervisor gaps in section 8 remain.
* The patrol action server and ROS wrappers have not run under real ROS 2.
* No natural-language handling.
* Flat, controlled environments only; no stairs or drops.

## 14. Future improvements

* Add wheel encoders and an IMU, then implement `pose_source: odom` with
  explicit pose-quality reporting; only then enable path and patrol goals.
* Add a centre-front or sweeping ultrasonic sensor, or a downward sensor, to
  raise coverage (hardware decision).
* Calibrate speed and turn rate; replace the timed primitives with
  distance-based motions.
* Feed the local map into replanning (currently a blocked route fails with a
  reason instead of replanning).
* Temperature compensation and crosstalk tests on the real array.
* Camera calibration, a shared camera owner, and a visual landmark map
  (experimental) once a pose exists.
* ROS integration tests under `launch_testing` on a ROS machine, and a field
  test log with measured results to replace the placeholders above.
* Close the supervisor fail-open gap in coordination with its owner, with
  re-validation of the voice demo.
