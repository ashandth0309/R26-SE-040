# BUDDY v1.0 — Project Scope

**Project:** BUDDY — Affordable AI-Powered Intelligent Robot Dog
**Platform:** Raspberry Pi 4B 4 GB / Ubuntu 24.04 / ROS 2 Jazzy
**Status:** Task 01 Scope Baseline

---

# 1. Purpose

This document establishes the boundaries of the Buddy final-year research project.

It prevents future development tasks or AI coding assistants from silently converting optional, experimental or hardware-dependent features into mandatory architectural dependencies.

The project follows:

**ONE TASK → IMPLEMENT → TEST → VERIFY → COMMIT → NEXT TASK**

---

# 2. IN SCOPE FOR BUDDY v1.0

The following capabilities form the intended Buddy v1.0 research system.

## Platform

- Raspberry Pi 4 Model B — 4 GB
- Ubuntu 24.04
- ROS 2 Jazzy
- modular ROS architecture
- local configuration
- local logging
- automatic startup/deployment
- graceful failure handling

## Voice

- microphone input
- wake word “Buddy”
- voice activity detection where appropriate
- offline STT for core commands
- deterministic intent recognition
- local TTS
- basic conversational interaction
- user-specific responses where practical

## Vision

- Raspberry Pi Camera V2
- camera health/configuration
- face detection
- known-user face recognition
- user enrollment
- unknown-person result
- facial-expression estimation
- identity + expression association
- local security capture

## Memory

- SQLite-based local structured storage
- user profiles
- face profile metadata
- emotion history
- security history
- robot events
- relevant interaction history
- persistent state

## Movement

- four-wheel drive control
- forward
- reverse
- left
- right
- rotation
- stop
- bounded speed control
- command timeout
- deterministic safety validation
- emergency stop

## Obstacle Safety

- front HC-SR04
- rear HC-SR04
- left HC-SR04
- right HC-SR04
- filtered distance readings
- invalid-reading handling
- sensor health
- collision prevention
- coordinated triggering

## Mobile Control

- React Native preferred
- local network connection
- robot status
- manual driving
- stop
- emergency stop
- camera viewing
- mode control
- security-event display

## Offline Operation

Core operation shall not require internet for:

- movement
- emergency stop
- obstacle safety
- local camera
- core voice
- local STT/TTS where practical
- face recognition
- basic expression estimation
- local memory
- local recording
- same-network control

## Security

- known/unknown-person handling
- security-event creation
- timestamps
- local image capture
- local event storage
- duplicate-alert control
- owner inspection where connectivity permits

## Reliability

- error handling
- log rotation
- CPU/RAM/temperature monitoring
- sustained runtime tests
- reconnection testing
- emergency-stop regression

---

# 3. CONDITIONAL / HARDWARE-DEPENDENT

These capabilities are desired but depend on hardware feasibility.

## Accurate Environment Mapping

Reliable SLAM is conditional on adequate localization/pose sensing.

Current hardware may lack:

- wheel encoders
- IMU
- LiDAR
- depth sensing

A feasibility checkpoint must occur before accurate mapping is claimed.

## No-Entry Zones

Map-based no-entry zones depend on sufficiently reliable mapping/localization.

## Autonomous Patrol Routes

Sophisticated map-based patrol depends on adequate localization.

Simpler bounded patrol behaviour may still be possible without full SLAM.

## Battery Percentage

Reliable battery reporting depends on appropriate voltage/current sensing hardware.

## Camera Pan/Tilt

Depends on:

- MG90S compatibility
- pan/tilt bracket
- PCA9685
- safe power supply
- verified mechanical limits

## Motor Performance

Depends on confirmation of:

- motor voltage
- motor current
- stall current
- TB6612FNG suitability
- battery capability
- chassis mechanics

## LCD

Integration depends on identifying the exact LCD model and interface.

---

# 4. EXPERIMENTAL

The following are research features and are not guaranteed to reach ideal performance.

## Two-USB-Microphone Sound Localization

Research includes:

- RMS comparison
- synchronization analysis
- cross-correlation
- TDOA experiments
- left/right estimation
- confidence estimation
- placement calibration

Two independent USB microphones may not provide sufficiently synchronized samples for reliable TDOA.

Perfect 360-degree localization is not guaranteed.

Front/back ambiguity may remain.

A synchronized microphone array may ultimately be required.

## Caller Approach

The desired behaviour:

**“Buddy, come here.”**

may eventually combine:

voice detection
→ direction estimation
→ camera orientation
→ person detection
→ caller selection
→ body orientation
→ safe approach.

Its final accuracy depends on sound localization, vision and navigation performance.

## Mapping With Current Hardware

Mapping without wheel encoders, IMU, LiDAR or depth hardware is experimental.

No claim of accurate SLAM shall be made unless experimentally justified.

## Local LLM

Running Ollama/Llama or another local LLM simultaneously with the full robotics stack on Raspberry Pi 4B 4 GB is experimental.

The LLM may be disabled if it compromises:

- safety responsiveness
- camera processing
- voice latency
- temperature
- RAM
- system stability

## Advanced Visual Learning

Voice-assisted object learning from user examples is experimental and shall not be described as universal one-shot semantic learning.

---

# 5. OPTIONAL ENHANCEMENTS

These are desirable if time and resources permit.

- advanced object learning
- sophisticated Buddy personality
- online weather
- online news
- web search
- music services
- richer conversational AI
- rolling pre-event video buffer
- advanced user preference learning
- detailed monthly behaviour reports
- richer LCD interface
- additional cloud backup
- additional sensors
- synchronized microphone array
- IMU
- wheel encoders
- LiDAR
- depth/ToF sensing
- enhanced map visualization
- advanced patrol planning
- advanced body/head coordination

Optional enhancements shall not become dependencies for core movement or safety.

---

# 6. OUT OF SCOPE FOR CURRENT RESEARCH

Unless the project scope is formally changed, Buddy v1.0 does not attempt to guarantee:

- commercial product certification
- medical diagnosis
- mental-health diagnosis
- psychological diagnosis
- professional security-system certification
- safety-critical industrial certification
- military operation
- weaponization
- unrestricted autonomous outdoor navigation
- perfect face recognition
- perfect emotion recognition
- perfect speech recognition
- perfect 360-degree sound localization using two independent USB microphones
- guaranteed accurate SLAM without adequate localization hardware
- universal object understanding from one example
- human-level general intelligence
- unrestricted LLM control of physical hardware
- direct public internet exposure of Raspberry Pi control services
- unlimited cloud storage
- unlimited recording storage

---

# 7. Core Safety Scope

Safety is always in scope.

The following cannot be removed to save development time:

- emergency stop
- startup motor safety
- stale-command rejection
- manual connection-loss stop
- collision protection
- deterministic physical-command validation
- servo limits
- critical sensor failure handling
- credential protection

A feature that cannot operate safely shall be disabled rather than bypassing the safety architecture.

---

# 8. Offline-First Scope

Cloud services are enhancements.

Buddy's essential operation shall be designed around local availability.

Loss of internet shall not disable:

- movement safety
- E-stop
- core voice
- local camera
- core vision
- local memory
- obstacle detection
- local manual control

---

# 9. Scope Change Control

Future tasks shall not silently modify this scope.

Any significant scope change must identify:

1. requested change
2. reason
3. affected requirements
4. affected modules
5. new hardware/software dependencies
6. safety impact
7. Raspberry Pi performance impact
8. regression tests required

Experimental functionality shall not be promoted to mandatory core functionality without evidence and an explicit scope update.

---

# 10. Legacy Code Boundary

Existing Buddy voice/face functionality is considered prototype/legacy source.

Task 01 does not:

- rewrite it
- delete it
- refactor it
- rename it
- migrate it

Migration shall occur only in explicitly assigned future tasks.
