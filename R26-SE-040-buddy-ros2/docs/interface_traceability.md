# BUDDY ROS 2 Interface Traceability

## Task

TASK 05 — Define and Freeze Core ROS 2 Interfaces

## Purpose

This document provides initial traceability between the BUDDY ROS 2 interface contracts and future subsystem responsibilities.

Implementation modules and exact future task numbers remain TBD until formally assigned.

| Interface | Type | Primary Future Producer | Primary Future Consumer | Requirement Area | Implementation Task |
|---|---|---|---|---|---|
| RobotStatus | Message | Central controller / health subsystem | Mobile app / diagnostics / logger | Robot state, mobile status, diagnostics | TBD |
| MovementCommand | Message | Mobile control / navigation / deterministic command router | Arbitration / safety controller | Movement and safety | TBD |
| ObstacleStatus | Message | Obstacle sensor subsystem | Safety / navigation / patrol | Obstacle detection and collision prevention | TBD |
| SafetyState | Message | Safety controller | Movement / navigation / mobile status | Emergency stop and movement safety | TBD |
| RecognizedPerson | Message | Vision subsystem | Security / memory / following | Face recognition and person tracking | TBD |
| EmotionObservation | Message | Emotion-recognition subsystem | Memory / reporting / interaction controller | Emotion observation and history | TBD |
| VoiceCommand | Message | Voice subsystem | Command router / central controller | Wake word, STT, intent recognition | TBD |
| SecurityEvent | Message | Security / patrol / vision subsystem | Local storage / cloud sync / mobile app | Security monitoring and incident recording | TBD |
| SetRobotMode | Service | Mobile app / controller / validated voice command | State-machine controller | Robot operating modes | TBD |
| EmergencyStop | Service | Mobile app / safety trigger / authorized controller | Safety controller | Emergency stop | TBD |
| ResetEmergencyStop | Service | Authorized controller / mobile app | Safety controller | Safe E-stop reset | TBD |
| FollowPerson | Action | Controller / validated user command | Following subsystem | Person following | TBD |
| ApproachCaller | Action | Controller / validated voice command | Caller-approach subsystem | Sound direction and caller approach | TBD |
| PatrolRoute | Action | Controller / mobile app / schedule | Patrol/navigation subsystem | Autonomous patrol | TBD |

## Traceability Model

Each interface should eventually support:

Requirement ID
→ Implementation Task
→ Module
→ Test
→ Result

Current implementation links:

TBD

Current test links:

TBD

Current result links:

TBD

The interface definitions themselves are verified in Task 05 through:

- clean ROS 2 build
- ROS interface discovery
- ROS interface inspection
- Python import testing
- message-instantiation testing
- Task 04 smoke-test regression

## Freeze Rule

BUDDY ROS Interface Contract v1 becomes frozen after Task 05 passes.

Any later interface modification requires explicit traceability updates and regression testing.
