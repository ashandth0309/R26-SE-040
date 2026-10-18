# BUDDY integration interfaces

**Interface version: 1.0.0** (`buddy_orchestrator.contracts.INTERFACE_VERSION`).
The machine-readable half of this document is
`buddy_ws/src/buddy_orchestrator/buddy_orchestrator/contracts.py`; unit tests
check it against the real `.msg`/`.srv` files and the code.

Change rule: adding a topic, service or optional field raises the minor
version; renaming or removing one, or changing a meaning, raises the major
version and needs all three component owners to agree.

> Status of this document: it describes what the code does. None of it has
> been run under a real ROS 2 graph (see `buddy_known_limitations.md`).

## Topics

| Topic | Type | Publishers | Subscribers | Notes |
|---|---|---|---|---|
| `/movement/request` | buddy_interfaces/MovementCommand | buddy_navigation; coordinator (relayed voice movement and STOP); voice (STOP only) | safety_supervisor, buddy_navigation (backs off for 1 s if another source moves the robot) | Everything that wants to move the robot ends here. Only the supervisor may turn it into `/movement/safe`. |
| `/movement/safe` | buddy_interfaces/MovementCommand | safety_supervisor ONLY | motor_executor (or the navigation simulator) | At most one publisher, audited at runtime. |
| `/obstacle_status` | buddy_interfaces/ObstacleStatus | buddy_obstacle_sensor (or the simulator) | safety_supervisor, buddy_coordinator | One ultrasonic driver. Per-side `*_valid` flags are honoured by the supervisor. |
| `/safety_state` | buddy_interfaces/SafetyState | safety_supervisor | buddy_voice_companion, buddy_navigation, buddy_coordinator | Latched emergency stop and `movement_allowed`. |
| `/robot_status` | buddy_interfaces/RobotStatus | buddy_coordinator | apps, tools | Derived from the central robot state. |
| `/navigation/command` | buddy_nav_interfaces/NavigationCommand | buddy_coordinator; voice (STOP only) | buddy_navigation |  |
| `/navigation/status` | buddy_nav_interfaces/NavigationStatus | buddy_navigation | buddy_coordinator, voice |  |
| `/navigation/local_map` | nav_msgs/OccupancyGrid | buddy_navigation | tools |  |
| `/voice/status` | std_msgs/String (JSON) | buddy_voice_companion | buddy_coordinator, buddy_vision | Keys: state, awake, muted, silence_mode, moving, mic_active, speaker_failed, cloud, errors, profile. |
| `/voice/command` | buddy_interfaces/VoiceCommand | buddy_voice_companion | tools | Parsed command observations. |
| `/voice/say` | buddy_interfaces/SpeechRequest | buddy_coordinator ONLY | buddy_voice_companion | The only way for another module to make Buddy speak. |
| `/camera/image_raw` | sensor_msgs/Image | buddy_camera ONLY | buddy_vision | One camera owner. |
| `/vision/events` | buddy_interfaces/VisionEvent | buddy_vision | buddy_coordinator | greeting_due, proactive_offer, fall_suspected, fall_alert, fall_recovered, camera_lost, camera_restored, teach_prompt, ... |
| `/vision/people` | buddy_interfaces/VisionPeople | buddy_vision | buddy_coordinator |  |
| `/vision/objects` | buddy_interfaces/VisionObjects | buddy_vision | tools |  |
| `/vision/status` | std_msgs/String (JSON) | buddy_vision | buddy_coordinator | Keys include camera_enabled, camera_ok, teaching, awaiting_reply, unavailable. |
| `/person/recognized` | buddy_interfaces/RecognizedPerson | buddy_vision | tools |  |
| `/emotion/observation` | buddy_interfaces/EmotionObservation | buddy_vision | buddy_voice_companion | Visual cue only; voice fuses it with what was said. |
| `/security/event` | buddy_interfaces/SecurityEvent | buddy_vision | tools | No image is attached. |
| `/coordinator/status` | std_msgs/String (JSON) | buddy_coordinator | tools | State, readiness, capabilities, conflicts, counters. |
| `/coordinator/voice_movement` | buddy_interfaces/MovementCommand | buddy_voice_companion | buddy_coordinator | Voice movement is relayed, not published to `/movement/request`. |
| `/coordinator/voice_navigation` | buddy_nav_interfaces/NavigationCommand | buddy_voice_companion | buddy_coordinator |  |

Not coordinated by the coordinator but part of the running system:
`/ultrasonic/<sensor>` (`sensor_msgs/Range`, from the one sensor driver to
`buddy_navigation`), `/sim/pose` (simulation only).

QoS: every publisher and subscriber in the code uses the ROS 2 default
(reliable, keep-last) with the depths written in the node sources. QoS
compatibility has been read, not exercised on a real DDS.

## Services

| Service | Type | Server | Clients | Notes |
|---|---|---|---|---|
| `/coordinator/route_utterance` | buddy_interfaces/RouteUtterance | buddy_coordinator | buddy_voice_companion | Bounded, fails open: no answer means ordinary conversation. |
| `/emergency_stop` | buddy_interfaces/EmergencyStop | safety_supervisor | operators, tools | Latches. The coordinator does not call it. |
| `/reset_emergency_stop` | buddy_interfaces/ResetEmergencyStop | safety_supervisor | operators, tools |  |
| `/vision/command` | buddy_interfaces/VisionCommand | buddy_vision | buddy_coordinator | Settings, privacy and replies to offers. |
| `/vision/identify_object` | buddy_interfaces/IdentifyObject | buddy_vision | buddy_coordinator | "What is this?" |
| `/vision/teach_object` | buddy_interfaces/TeachObject | buddy_vision | buddy_coordinator | start / confirm / cancel / list / forget |
| `/vision/where_is` | buddy_interfaces/WhereIs | buddy_vision | buddy_coordinator | Needs object memory switched on. |
| `/vision/enable` | std_srvs/SetBool | buddy_vision | tools |  |
| `/camera/enable` | std_srvs/SetBool | buddy_camera | tools |  |
| `/voice/mute` | std_srvs/SetBool | buddy_voice_companion | tools |  |

## Error codes (`RouteUtterance.error_code`)

| Code | Name | Meaning |
|---|---|---|
| 0 | `OK` | handled |
| 1 | `UNAVAILABLE` | a module needed for this is not running |
| 2 | `TIMEOUT` | a module did not answer in time |
| 3 | `REFUSED_SAFETY` | robot state, sensors or the supervisor said no |
| 4 | `REFUSED_CONSENT` | needs a consent or invitation that is missing |
| 5 | `NOT_UNDERSTOOD` | not understood |
| 6 | `STALE` | data needed is too old to act on |
| 7 | `BUSY` | another task owns the resource |
| 8 | `CANCELLED` | cancelled |
| 9 | `HARDWARE_CONFLICT` | two owners for one device |

## Staleness limits (seconds)

Data older than this is treated as missing, and missing means *no motion*.

- `safety_state`: 2.0
- `navigation_status`: 3.0
- `obstacle_status`: 1.0
- `vision_people`: 2.0
- `vision_status`: 3.0
- `voice_status`: 3.0

## Speech priorities

The integration spec lists six speech classes; the speaker queue has four
levels, so they map as follows (order within a level is arrival order).

| Class | `/voice/say` priority | Notes |
|---|---|---|
| emergency and safety warnings | 0 CRITICAL | may interrupt current speech |
| user stop acknowledgement | 0 CRITICAL | spoken by the voice node itself |
| important robot status, refusals | 1 HIGH | |
| direct answers to the user | 2 NORMAL | |
| proactive companion speech | 3 LOW | gated, see below |
| listening backchannel | 3 LOW | voice-internal, dropped if anything else is playing |

The coordinator's arbiter drops speech before publishing when: Buddy is shutting
down (everything); quiet mode (replies and proactive); the emergency stop is
active (proactive); the user or Buddy is speaking or a conversation is in
progress (proactive); within 20 s of the previous proactive line; the same
words were sent within 5 s. The voice node then applies its own TTL and quiet
rules.

## Hardware ownership

| Resource | Sole owner | Enforced by |
|---|---|---|
| two microphones | `buddy_voice_companion` | voice node refuses to start next to the legacy `buddy_voice`; coordinator audit |
| speaker and speech queue | `buddy_voice_companion` | one `/voice/say` publisher (audited) |
| camera | `buddy_camera` | camera nodes count peers; `/camera/image_raw` limited to one publisher |
| four ultrasonic sensors | `buddy_obstacle_sensor` (or the simulator) | `/obstacle_status` limited to one publisher |
| motors | `motor_executor` | only `/movement/safe`, which only the supervisor publishes |

The coordinator owns **no** hardware. It audits the graph every 2 s and
reports every conflict in `/coordinator/status`; any conflict makes
`ready_for_motion` false.

## Motion path

```
voice ──(relay)──► coordinator ──► /navigation/command ──► buddy_navigation
voice ──(relay)──► coordinator ──► /movement/request  ───┐
buddy_navigation ────────────────► /movement/request  ───┤
voice ── STOP only (direct) ─────► /movement/request  ───┤
voice ── STOP only (direct) ─────► /navigation/command   │
                                                         ▼
                                   safety_supervisor ─► /movement/safe ─► motor_executor
```

- STOP is never delayed or gated by the coordinator, and it has a direct
  route that does not need the coordinator to be running.
- A spoken stop cancels motion and navigation. It does **not** set the
  supervisor's latched emergency stop; that comes from the hardware button
  or the `/emergency_stop` service.
- Every other motion request is checked by the coordinator against robot
  state, fresh safety/obstacle/navigation status, `movement_allowed`,
  navigation `motion_enabled`, `sensors_healthy` and (for patrol and
  destinations) `pose_trusted`. The supervisor and the navigation node then
  check again.

## Coordinator parameters

`profile` (offline | hybrid | simulation), `enable_vision`,
`vision_timeout_s`, `init_timeout_s`, `person_distance_validated`
(keep false), `default_patrol_route`, `config_file`.

## Voice-node parameters added for integration

`movement_topic`, `navigation_topic`, `direct_stop_topic`,
`direct_nav_stop_topic` (interposition), `enable_gateway`,
`gateway_service`, `gateway_timeout_s`, `say_topic`. The launch files set
them; with none set the voice node behaves exactly as before.
