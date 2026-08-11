# BUDDY ROS 2 Interface Contracts

## Task

TASK 05 — Define and Freeze Core ROS 2 Interfaces

## Purpose

This document defines the first stable ROS 2 communication-contract baseline for BUDDY.

These interfaces are shared contracts between future BUDDY subsystems such as movement, safety, voice, vision, navigation, patrol, security, and mobile control.

Task 05 defines interfaces only. It does not implement the subsystem behaviour behind them.

---

## Communication Rules

### Topics

Use topics for continuous state and event streams.

### Services

Use services for short request-response operations.

### Actions

Use actions for long-running and cancellable robot behaviours.

---

# Messages

## RobotStatus

Purpose:

Provides high-level observable BUDDY system status.

Possible future producers:

- central controller
- system-health module

Possible future consumers:

- mobile app
- dashboard
- logger
- diagnostics
- security subsystem

Important notes:

- Battery percentage may be unavailable until battery measurement hardware is implemented.
- Producers must not fabricate unavailable sensor values.
- Robot operating mode is represented using defined mode constants.

---

## MovementCommand

Purpose:

Represents a requested physical movement command before safety arbitration.

Possible future producers:

- manual mobile control
- deterministic voice-command router
- navigation subsystem
- patrol subsystem
- following subsystem

Possible future consumers:

- command arbitration layer
- safety layer

Safety meaning:

MovementCommand is a REQUEST.

It does not directly authorize motor movement.

Required future flow:

Command Source
→ MovementCommand
→ Arbitration
→ Safety Validation
→ Motor Controller

MovementCommand must never directly bypass safety and operate GPIO.

The emergency field does not replace the dedicated EmergencyStop service.

---

## ObstacleStatus

Purpose:

Represents processed obstacle-distance and blocked-state information.

Future producer:

- obstacle sensor subsystem

Future consumers:

- safety subsystem
- navigation
- patrol
- manual-control safety

Important notes:

Reading validity is separate from blocked state.

Distances are represented in meters.

Hardware-specific GPIO information is intentionally excluded.

---

## SafetyState

Purpose:

Represents the global movement-safety state.

Future producer:

- deterministic safety controller

Future consumers:

- movement arbitration
- navigation
- central controller
- mobile app
- diagnostics

Safety levels:

- NORMAL
- CAUTION
- RESTRICTED
- EMERGENCY

Emergency stop has the highest authority.

---

## RecognizedPerson

Purpose:

Represents a face/person recognition observation.

Future producer:

- vision subsystem

Future consumers:

- central controller
- security subsystem
- user-memory subsystem
- person-following subsystem

Bounding-box convention:

- normalized coordinates from 0.0 to 1.0
- origin is the top-left of the image
- bbox_x and bbox_y represent the normalized top-left position
- bbox_width and bbox_height are normalized to image dimensions

No image bytes are embedded in this message.

---

## EmotionObservation

Purpose:

Represents one significant facial-expression observation.

Future producer:

- emotion-recognition subsystem

Future consumers:

- memory subsystem
- reporting subsystem
- interaction controller

Important note:

EmotionObservation is an observation only.

It is not a medical diagnosis, mental-health diagnosis, or statement of psychological certainty.

---

## VoiceCommand

Purpose:

Represents a processed voice command/intention after speech recognition and intent parsing.

Future producer:

- voice subsystem

Future consumers:

- deterministic command router
- central controller
- conversation subsystem

Important safety note:

VoiceCommand does not directly control motors or other physical hardware.

Physical actions require deterministic validation.

Raw microphone audio is not embedded in this message.

---

## SecurityEvent

Purpose:

Represents a significant security event.

Future producers:

- security subsystem
- vision subsystem
- patrol subsystem
- system controller

Future consumers:

- event logger
- local storage
- cloud sync
- mobile app

Important notes:

- image_path and video_path are local media references.
- Binary image/video data is not stored in this message.
- The base interface is not coupled to any cloud provider.

---

# Services

## SetRobotMode

Purpose:

Requests a high-level BUDDY operating-mode change.

The service reports whether the transition request was accepted and what the active mode is.

Actual transition validation belongs to future deterministic state-machine logic.

---

## EmergencyStop

Purpose:

Provides an explicit emergency-stop request interface.

Emergency stop has the highest control authority.

Task 05 defines only the service contract.

Future safety implementation must ensure E-stop overrides all normal behaviour.

---

## ResetEmergencyStop

Purpose:

Requests clearing of an active emergency-stop state.

Calling this service does not guarantee reset.

Future safety validation may reject the request when movement cannot safely resume.

---

# Actions

## FollowPerson

Purpose:

Represents long-running, cancellable person-following behaviour.

Future implementation must handle:

- target identity
- target visibility
- distance control
- target loss
- obstacle safety
- cancellation
- timeout

Task 05 does not implement this behaviour.

---

## ApproachCaller

Purpose:

Represents the long-running behaviour associated with commands such as:

"Buddy, come here."

The goal accepts an upstream sound-direction estimate and confidence.

Important limitation:

This interface does not imply that two independent USB microphones can provide perfect caller localization.

The future localization subsystem must handle uncertainty.

---

## PatrolRoute

Purpose:

Represents long-running, cancellable patrol-route execution.

Route geometry and map representation are intentionally not part of Task 05.

They belong to later navigation/mapping tasks.

---

# Safety Rules

1. MovementCommand is a request only.
2. Movement commands never bypass arbitration.
3. Movement commands never bypass safety validation.
4. Emergency stop has highest authority.
5. ResetEmergencyStop is only a request.
6. Safety logic may reject an emergency-stop reset.
7. VoiceCommand does not directly operate hardware.
8. LLM output does not directly operate motors or safety-critical systems.
9. Physical-control execution must remain deterministic.
10. Safety interfaces remain independent of mobile UI implementation.

---

# Units

Distance:

meters

Linear speed:

meters per second

Movement angular speed:

radians per second

Approach-caller direction:

degrees

Timestamps:

ROS `builtin_interfaces/Time`

Durations:

ROS `builtin_interfaces/Duration`

---

# Coordinate Convention

RecognizedPerson bounding boxes use normalized image coordinates.

Range:

0.0 to 1.0

Origin:

top-left

bbox_x:

normalized x coordinate of the left edge

bbox_y:

normalized y coordinate of the top edge

bbox_width:

normalized bounding-box width

bbox_height:

normalized bounding-box height

---

# Interface Baseline

Version:

BUDDY ROS Interface Contract v1

This baseline contains:

- 8 messages
- 3 services
- 3 actions

Total:

14 interfaces

---

# Freeze Policy

After TASK 05 passes, BUDDY ROS Interface Contract v1 is considered frozen.

Future tasks may consume these interfaces.

Future tasks may not silently rewrite them.

A future interface change requires:

1. documented reason
2. affected consumers identified
3. requirement impact identified
4. documentation updated
5. dependent modules updated
6. regression testing performed
7. explicit interface-change commit

Interface compatibility is treated as a project constraint.

---

# Result

TASK 05 PASS
