# BUDDY system architecture (integrated)

Branch `testing-integration` = `Sobiya_AdvancedVision` (which already
contains the voice companion) + `Yohan_Mapping` (navigation), plus the
integration work described here. Nothing in this branch has run on a Raspberry
Pi or a real ROS 2 graph; see `buddy_known_limitations.md`.

## Components

```
                       ┌────────────────────────── buddy_coordinator ──────────────────────────┐
                       │  robot state machine   conversation state machine   speech arbiter    │
                       │  motion gate           utterance router             readiness/audit   │
                       └──▲──────────▲───────────────▲──────────────▲───────────────┬───────────┘
 microphones ─► voice ────┘          │               │              │ /voice/say    │ /navigation/command
 speaker     ◄─ voice ◄──────────────┘               │              │               ▼ /movement/request (relay, STOP)
                  ▲  /coordinator/route_utterance ───┘              │         buddy_navigation ─► /movement/request
 camera ─► buddy_camera ─► /camera/image_raw ─► buddy_vision ───────┘                                   │
                                  events, people, status, 4 services                                   ▼
 4× ultrasonic ─► buddy_obstacle_sensor ─► /obstacle_status ─────────────► safety_supervisor ─► /movement/safe ─► motor_executor
```

| Package | Node(s) | Role |
|---|---|---|
| `buddy_core` | `voice_companion`, `vision_camera`, `vision_node`, `obstacle_sensor`, `safety_supervisor`, `motor_executor` | hardware owners, voice engine, vision pipeline, the safety path |
| `buddy_navigation` | `navigation_node`, `sim_node` | guarded navigation, patrol, simulator |
| `buddy_orchestrator` | `buddy_coordinator` | central coordination; owns no hardware |
| `buddy_interfaces`, `buddy_nav_interfaces` | — | messages and services |

## What the coordinator decides

1. **Whether motion may start.** Voice movement and navigation requests
   arrive on intake topics, not on the motion path. They go on only if the
   robot state is IDLE (or PAUSED for resume), safety/obstacle/navigation
   status is fresh, the supervisor allows movement, navigation motion is
   enabled and its sensors are healthy, and (patrol, destinations) the pose is
   trusted. Otherwise Buddy says why. STOP skips all of this.
2. **Whether an utterance is a camera request.** The voice node asks
   `/coordinator/route_utterance` for each finished utterance that is not a
   command. "What is this?", "this is my red pen", "who is this", "is anyone
   on the floor" and "come here" are claimed; the answer comes from vision's
   own result or an honest "I can't see right now". A claimed camera question
   is never handed to the language model, so a model cannot invent an object.
   The call is bounded and fails open.
3. **What may be spoken unprompted.** Every non-voice module speaks through
   `/voice/say`, whose only publisher is the coordinator's arbiter.
4. **Whether the system is healthy.** Stale or missing safety, obstacle or
   navigation status moves the robot to DEGRADED_MODE with a direct STOP.

## Robot state machine

`STARTING → INITIALIZING → IDLE ⇄ NAVIGATION_ACTIVE | PATROL_ACTIVE | PAUSED`
`→ DEGRADED_MODE` (motion inputs lost) and `EMERGENCY_STOP` (latched
supervisor stop, reachable from every live state, leavable only to IDLE or
DEGRADED_MODE after the supervisor clears it) and `SHUTTING_DOWN` (final).
Camera, microphone and internet losses are capability flags, not robot states.

Conversation axis: `LISTENING, USER_SPEAKING, PROCESSING, VISION_REQUEST,
BUDDY_SPEAKING`, derived from the voice status and the routing in progress.
Illegal transitions are refused and counted
(`state_violations` in `/coordinator/status`; tests require zero).

## Profiles

| Profile | Cloud | Hardware | Notes |
|---|---|---|---|
| `offline` | off | real | default for the robot |
| `hybrid` | allowed (voice node still needs endpoint + user confirmation) | real | |
| `simulation` | off | mock (simulator, mock camera, dry-run voice) | the only profile that demonstrates person approach, and only against a mock |

## Safety layers for motion

coordinator gate → navigation node guards (sensor health, obstacle
assessment, pose trust, `allow_motion`) → safety supervisor (latched e-stop,
stale/invalid sensor checks, per-side blocking) → motor executor (command
timeout). A fault in any layer is caught by the next; none is bypassed by
the coordinator.

## Defects found and fixed during integration

| Where | Defect | Fix |
|---|---|---|
| `safety/supervisor.py` | movement toward a side with no valid reading was allowed | refuse and say so (`*_valid` now used) |
| `vision/camera_node.py`, `buddy_navigation/.../camera_node.py` | two camera nodes could start; the guard counted wrongly | one owner, tested |
| `voice_node.py` | navigation state table disagreed with `NavigationStatus.msg` (a finished goal looked rejected) | table corrected, test reads the `.msg` |
| voice → navigation | spoken "start patrol" carries no route, navigation refuses silently while Buddy says "Starting patrol" | coordinator fills `default_patrol_route`, or says there is none; a request navigation reports idle is announced as not started |
| `vision_intents` | "that is a good idea" was parsed as teaching an object called "good idea"; "my friend Sam" was an object | abstract-word and person-word rules, "this is" only |
| coordinator | quiet mode read from stale cached state; illegal VISION_REQUEST→PROCESSING transition | read live, transition fixed |

## Not integrated, on purpose

- Person approach on hardware ("come here"): no person-distance estimate has
  been validated, so it is refused. Simulation demonstrates the decision
  logic against a mock navigation adapter that moves nothing.
- Face enrolment by voice: vision exposes no enrolment service; "this is my
  friend Sam" only explains that the person must agree themselves.
