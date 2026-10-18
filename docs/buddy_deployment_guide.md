# BUDDY integrated deployment guide

> **None of these commands has been run by the author of this branch.** The
> build sandbox has no ROS 2, no Raspberry Pi and no robot. Treat every step
> as a checklist to verify, and record what actually happens. Stop at the first
> step that fails.

## 0. Before anything else

- Robot on a stand, wheels off the ground, until Level 7 begins.
- Physical emergency stop within reach and tested.
- Stop any older nodes: `buddy_voice` (the legacy voice node) must not run
  next to `buddy_voice_companion`, and only one camera node may run.

## 1. Get the branch

```bash
git fetch origin
git checkout testing-integration
```

## 2. Python environment and models (not in git)

Same as the individual components:

- Voice: `numpy pyyaml sounddevice vosk piper-tts`; set `BUDDY_VOSK_MODEL`
  and `BUDDY_PIPER_MODEL` (see `docs/ashandth_voice_companion_engine.md`).
- Vision: `python3 scripts/fetch_vision_models.py`
  (see `docs/sobiya_advanced_vision.md`).
- Navigation: no extra models.

## 3. Build (Level 5, first time)

```bash
cd buddy_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select buddy_orchestrator buddy_core buddy_navigation
colcon test-result --verbose
```

Expected, if the sandbox results hold: the orchestrator, core and navigation
Python tests pass. `ament_copyright`, `ament_flake8` and `ament_pep257` may
report issues in older files that this branch did not touch.

Record: build time, warnings, failing tests.

## 4. Simulation first (no hardware)

```bash
ros2 launch buddy_orchestrator full_simulation.launch.py
```

In another terminal:

```bash
ros2 node list                         # expect 7 nodes, none duplicated
ros2 topic echo /coordinator/status    # robot_state IDLE, ready_for_motion true
ros2 topic hz /safety_state            # about 2 Hz
ros2 topic info /voice/say             # exactly one publisher
ros2 topic info /movement/safe         # exactly one publisher
```

Then exercise the Test A-P inputs from `buddy_testing_matrix.md`. The
simulation profile uses a mock camera and a dry-run voice backend, so for
spoken tests use the offline profile on the Pi.

## 5. Raspberry Pi bench (Level 6): wheels off the ground

```bash
ros2 launch buddy_orchestrator full_offline.launch.py
```

Check, one at a time, and write the result down:

1. `ros2 node list`: `buddy_voice_companion`, `buddy_camera`, `buddy_vision`,
   `buddy_obstacle_sensor`, `safety_supervisor`, `motor_executor`,
   `buddy_navigation`, `buddy_coordinator`. No `buddy_voice`, no second
   camera.
2. `ros2 topic echo /coordinator/status`: `hardware_conflicts` empty.
3. Two microphones, one speaker, camera and four ultrasonic sensors report
   OK (voice status, vision status, `/obstacle_status`).
4. Cover each ultrasonic sensor in turn: `ready_for_motion` becomes false.
   Uncover: it recovers.
5. Say "Buddy, move forward": motion stays **disabled** by default
   (`allow_motion false` in `navigation.yaml`), so Buddy should say it is
   switched off. Do not change this until step 7.
6. Say "what is this?" with an object, and with the lens covered.
7. CPU, RAM and temperature with everything running for at least 10 minutes:
   `top`, `vcgencmd measure_temp`. **No Pi measurements exist yet.**
8. Ctrl-C the launch: all nodes stop, nothing keeps driving.

If CPU or temperature is too high, use the voice `lightweight` profile and
the vision `lightweight` profile first (`docs/` of each component).

## 6. Other launch files

| File | Starts | Use |
|---|---|---|
| `full_offline` | everything, real hardware, no cloud | the robot |
| `full_hybrid` | same, cloud allowed (voice still needs endpoint and confirmation) | optional |
| `full_simulation` | everything with simulator, mock camera, dry-run voice | development |
| `voice_only` | voice node alone, no navigation | microphone/speaker bench |
| `vision_only` | camera + vision | camera bench |
| `navigation_only` | sensing, safety, navigation (simulation by default) | navigation bench |

## 7. Controlled physical motion (Level 7)

Only after steps 3-6 passed and are recorded. Edit the navigation config (`allow_motion` and `geometry_confirmed`) to
enable motion (see `docs/yohan_mapping_navigation.md`), keep the wheels off
the ground first, then short timed moves, then obstacle stopping, then the
emergency stop, then floor tests with someone holding the stop. Spoken
"patrol" needs a route in the physical config and `default_patrol_route`;
without them it is refused in words. "Come here" is refused on hardware by
design.

## 8. Rollback

`git checkout Sobiya_AdvancedVision` (or the branch you came from) and
rebuild. The integration changes are additive; the voice node behaves as
before when none of the new parameters are set.
