# BUDDY testing matrix

**Evidence legend** (never mixed in a claim):

| Label | Meaning |
|---|---|
| **static** | syntax, lint, manifests, imports, cross-file consistency checked by tests |
| **unit** | one module, pure Python |
| **contract** | two components against real `.msg`/`.srv` files or each other's real code |
| **mock-e2e** | whole system in one process: real coordinator, voice engine, vision logic and navigation service; fake audio, camera, models, motors and no ROS transport |
| **ros-stub** | ROS node wrapper against strict stubs generated from the repo's real message files |
| **NOT RUN** | needs ROS 2, a Raspberry Pi or the robot; no result exists |

Nothing below is a Raspberry Pi or physical result.

## Levels (spec section 21)

| Level | Status | Evidence |
|---|---|---|
| 1 Static | done | flake8 (max 99) clean on every file added or changed here; launch files parsed; entry points, node names, parameters and config paths cross-checked (`test_launch.py`); contract constants vs `.msg` (`test_contracts.py`). Lint debt in teammates' older files is untouched. |
| 2 Unit | done | see counts below |
| 3 Contract | done (mock) | `test_contracts.py`, `test_coordinator_node.py`, `test_voice_integration_hooks.py`, cross-component cases in `test_e2e_mock.py` |
| 4 Mock end-to-end | done | `test_e2e_mock.py` (below) |
| 5 ROS 2 integration | **NOT RUN** | no ROS 2 in the build sandbox: no `colcon build`, node start, QoS, DDS, shutdown/restart test |
| 6 Pi bench | **NOT RUN** | |
| 7 Physical | **NOT RUN** | |

## Counts (this sandbox, Python 3, no ROS)

| Suite | Result |
|---|---|
| `buddy_orchestrator` | 288 passed in 28.91s |
| `buddy_core` | 1033 passed, 11 skipped in 5.86s |
| `buddy_navigation` | 173 passed in 9.48s |

The three `ament_*` lint tests cannot import here and are excluded.
`test_logging_utils` fails under pytest log capture and passes with
`-p no:logging` (pre-existing).

## Acceptance scenarios (spec section 22)

All results are **mock-e2e** unless stated.

| Test | Scenario | Result | What the test checks | Caveat |
|---|---|---|---|---|
| A | normal conversation | partial | Buddy speaks, nothing moves | the offline dialogue has no language model in this harness; the wording ("I'm not sure I got that") is a fallback, so reply *quality* is **not** evaluated |
| B | emotional conversation | partial | same | follow-up question quality **not** evaluated |
| C | backchannel "Mm-hmm" | unit only | backchannel tests in `test_voice_engine.py` and `test_voice_integration.py` | not re-run in the integrated harness |
| D | face recognition | pass | known person named, unknown not named, nobody in view | fake face detector/recogniser; real recognition accuracy is Sobiya's dataset results, not this |
| E | "what is this?" | pass | detected object named; no vision → "I can't see right now"; camera off/lost → says so; never an invented label | fake detector |
| F | teach and retrieve | pass | taught frames recognised, other frames not | fake embedder: identical pixels give identical vectors, so "100 percent similar" means nothing about a real model |
| G | voice/visual emotion fusion | unit only | `test_vision_multimodal.py`, `test_voice_emotion*.py` | not exercised end to end |
| H | proactive companion | pass | one offer within cooldown, LOW priority, none in quiet mode or during conversation | trigger conditions of the offer itself are Sobiya's unit tests |
| I | obstacle avoidance | pass | static and appearing obstacle: no collision; failed sensors: refused in words | mock ultrasonics and mock robot |
| J | navigation | pass | destination reached, result spoken, no collision | 6×6 teaching world |
| K | patrol | pass | configured route completes and is announced; cancel stops | simulated route |
| L | emergency stop | pass | spoken stop halts within 0.6 s of simulated time even with the coordinator dead; stop clears queued speech; latched e-stop blocks new motion and recovers to IDLE | simulated time, not real latency |
| M | no internet | pass | conversation, vision question and motion work with cloud off | the offline path is also the default |
| N | hardware failure | partial | ultrasonic failure blocks motion and voice continues; camera stops delivering → vision questions say so, voice and navigation continue; navigation status lost → DEGRADED + stop | microphone loss: unit tests only (`test_voice_audio.py` failover, `test_voice_integration.py::test_losing_every_microphone_stops_a_moving_robot`) |
| O | multi-person | partial | coordinator unit tests: with more than one person nobody is named aloud; mock-e2e: the same person is not greeted repeatedly | the multi-face tracker is Sobiya's unit tests |
| P | possible fall | pass | mock fall → spoken check-in "Are you okay?", no movement, no navigation command, and none after 60 s of no answer | fall classifier is a mock pose sequence, not a validated detector |
| Q | long-running stability | 30.3 simulated min, no crash, collision or violation; unexplained heap growth ~144 KB/cycle (`buddy_final_integration.md`) | `tools/soak.py` | sandbox process, **not** a Pi |

Ambiguous language (spec section 6): eight sentences containing "forward" or
"turn left" in non-command use, a low-confidence command, a negated command and
an unknown destination all produce no motion request (mock-e2e, and the
parser itself in `test_vision_intents.py` and the voice command tests).

## Safety regression tests

| Defect | Test |
|---|---|
| unmeasured side allowed | `buddy_core/test/test_safety_supervisor.py`, `test_safety_node_validity.py` |
| two camera owners | `test_vision_camera.py`, `buddy_navigation/test/test_nodes_stubbed.py` |
| mutation check | disabling the motion gate, ignoring the e-stop, or letting a camera question fall through each makes the suite fail (run by hand during development; not automated) |

## What a real ROS 2 session must still run

`colcon build`, then `ros2 launch buddy_orchestrator full_simulation.launch.py`,
check `ros2 topic echo /coordinator/status`, say or inject the Test A-P
inputs, then repeat with `full_offline` on the Pi. See
`buddy_deployment_guide.md`.
