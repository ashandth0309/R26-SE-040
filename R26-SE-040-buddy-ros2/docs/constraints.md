# BUDDY v1.0 — Engineering Constraints

**Project:** BUDDY — Affordable AI-Powered Intelligent Robot Dog
**Platform:** Raspberry Pi 4B 4 GB / Ubuntu 24.04 / ROS 2 Jazzy
**Status:** Task 01 Constraint Baseline

---

# 1. Purpose

This document records known technical, hardware, performance, safety, networking and research constraints.

For each major constraint:

**Constraint → Impact → Mitigation**

---

# 2. Raspberry Pi Compute Constraint

## Constraint

Buddy uses a Raspberry Pi 4 Model B with 4 GB RAM.

## Impact

Buddy cannot assume desktop/server-class compute.

Simultaneously running:

- speech recognition
- camera capture
- face recognition
- expression recognition
- tracking
- navigation
- streaming
- LLM inference

may cause excessive latency, memory pressure or thermal throttling.

## Mitigation

- modular ROS nodes
- benchmark every heavy subsystem
- avoid running every model on every frame
- use bounded queues
- drop stale frames
- select lightweight models
- control inference frequency
- disable optional AI under resource pressure
- prioritize safety processes

---

# 3. RAM Constraint

## Constraint

Available RAM is 4 GB and must also support the operating system and ROS.

## Impact

Large models and unbounded queues may cause swapping, OOM termination or severe latency.

## Mitigation

- lightweight models
- bounded queues
- controlled caching
- memory monitoring
- avoid unnecessary duplicate frame buffers
- long-duration leak testing

---

# 4. Thermal Constraint

## Constraint

Sustained AI processing may heat the Raspberry Pi.

## Impact

Thermal throttling may reduce inference and control performance.

## Mitigation

- cooling hardware
- temperature monitoring
- workload scheduling
- performance benchmarking
- reduce optional inference when necessary

---

# 5. Power Constraint

## Constraint

Buddy combines Raspberry Pi, motors, servos, sensors, microphones, speakers and other electronics.

The exact battery/power architecture is not yet fully verified.

## Impact

Voltage drop, motor stall current or servo load may reset/damage electronics or destabilize the Pi.

## Mitigation

Verify before integration:

- cell chemistry
- BMS
- battery voltage
- capacity
- discharge rating
- motor voltage/current/stall current
- regulator capacity
- wiring
- grounding
- fuse/protection
- separate/appropriate power rails

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

---

# 6. TB6612FNG Constraint

## Constraint

Two TB6612FNG boards are planned.

Exact motor electrical characteristics remain to be confirmed.

## Impact

Drivers may be unsuitable if motor stall current exceeds safe capability.

## Mitigation

Measure/confirm motor specifications before connecting final motor load.

Do not assume compatibility solely because the driver physically connects.

---

# 7. HC-SR04 Logic-Level Constraint

## Constraint

HC-SR04 modules commonly use 5 V signalling while Raspberry Pi GPIO is 3.3 V logic.

## Impact

Unsafe direct echo connection may damage the Raspberry Pi.

## Mitigation

Use verified voltage-divider/level-shifting/protection circuitry.

Do not connect an unverified 5 V echo signal directly to Pi GPIO.

---

# 8. Ultrasonic Cross-Talk Constraint

## Constraint

Four ultrasonic sensors may interfere when triggered simultaneously.

## Impact

False distance measurements may occur.

## Mitigation

- sequential triggering
- timing separation
- filtering
- invalid-reading rejection
- sensor-health monitoring

---

# 9. Independent USB Microphone Constraint

## Constraint

Buddy currently uses two separate USB microphone devices.

They may use independent audio clocks.

## Impact

Samples may not be sufficiently synchronized for reliable time-difference-of-arrival localization.

Clock drift and USB buffering may affect correlation.

## Mitigation

Experimentally evaluate:

- device timing
- correlation stability
- RMS difference
- synchronization drift
- microphone spacing
- orientation
- distance
- environmental noise

If inadequate, recommend a synchronized microphone array.

---

# 10. Two-Microphone Geometry Constraint

## Constraint

Two microphones provide limited spatial information.

## Impact

Front/back ambiguity may occur.

Reliable full 360-degree localization is not guaranteed.

## Mitigation

Combine available audio cues with:

- head scanning
- visual person detection
- confidence thresholds
- active reorientation

Consider additional synchronized microphones if required.

---

# 11. Camera Constraint

## Constraint

Buddy uses Raspberry Pi Camera Module V2 / IMX219.

## Impact

Resolution, FPS, low-light performance and AI inference must fit Pi resources.

## Mitigation

- benchmark resolutions
- benchmark FPS
- resize inference frames
- separate capture resolution from inference resolution where useful
- avoid unnecessary full-frame heavy inference

---

# 12. Servo Mechanical Constraint

## Constraint

Physical pan/tilt limits depend on the assembled bracket and camera wiring.

## Impact

Excessive servo commands may jam hardware, strain wiring or damage servos/bracket.

## Mitigation

Calibrate physical limits before enabling automatic tracking.

Enforce software limits permanently.

---

# 13. Missing Wheel Encoder Constraint

## Constraint

Confirmed wheel encoders are currently unavailable.

## Impact

Buddy may lack reliable wheel odometry.

Distance travelled and orientation estimates may drift significantly.

## Mitigation

Do not claim accurate odometry/SLAM.

Evaluate encoder-capable motors or external encoder hardware if required.

---

# 14. Missing IMU Constraint

## Constraint

No confirmed IMU is currently available.

## Impact

Orientation/turn estimation may be unreliable.

## Mitigation

Evaluate addition of an IMU during mapping/navigation feasibility testing.

---

# 15. Missing LiDAR Constraint

## Constraint

No confirmed LiDAR is currently available.

## Impact

High-quality 2D mapping/SLAM may be difficult using only ultrasonic/camera data.

## Mitigation

Treat accurate mapping as experimental/hardware-dependent.

Evaluate LiDAR or suitable depth/ToF hardware if required.

---

# 16. Mapping Constraint

## Constraint

Accurate SLAM normally requires sufficiently reliable perception plus pose estimation.

## Impact

The current sensor set may not support research-quality accurate mapping.

## Mitigation

Perform a formal mapping feasibility checkpoint before implementing or claiming full SLAM.

---

# 17. Network Availability Constraint

## Constraint

Internet access may disappear.

## Impact

Cloud APIs, remote telepresence and cloud synchronization may become unavailable.

## Mitigation

Buddy shall be offline-first.

Essential operation remains local.

Pending cloud work is queued.

---

# 18. Wi-Fi Reliability Constraint

## Constraint

Local Wi-Fi may disconnect or experience latency.

## Impact

Remote/manual commands may stop arriving.

## Mitigation

- command expiration
- heartbeat/session monitoring
- connection-loss motor stop
- reconnection logic

Never continue stale manual movement indefinitely.

---

# 19. Remote Internet Security Constraint

## Constraint

Directly exposing Raspberry Pi control ports creates unacceptable security risk.

## Impact

Unauthorized users could potentially reach robot control systems.

## Mitigation

Use authenticated, encrypted, authorized remote-access architecture.

Never expose raw GPIO/motor interfaces publicly.

---

# 20. Video Bandwidth Constraint

## Constraint

Live video can consume significant network and Pi resources.

## Impact

High resolution/bitrate may increase latency and interfere with other processing.

## Mitigation

Evaluate WebRTC and adaptive media settings.

Measure latency and CPU load.

---

# 21. Two-Way Audio Constraint

## Constraint

Buddy's microphone may capture its own speaker.

## Impact

Feedback and false wake/STT activation may occur.

## Mitigation

Initially prefer controlled push-to-talk where appropriate.

Evaluate:

- microphone/speaker placement
- muting rules
- echo cancellation
- assistant listening-state coordination

---

# 22. Local LLM Constraint

## Constraint

Raspberry Pi 4B 4 GB has limited resources for LLM inference.

## Impact

Local LLM execution may produce unacceptable latency or compete with safety/vision/audio.

## Mitigation

Treat local LLM as optional/experimental.

Deterministic offline commands remain available without the LLM.

---

# 23. Storage Constraint

## Constraint

microSD/local storage is finite.

## Impact

Video, images, logs and databases may fill storage.

## Mitigation

Implement:

- log rotation
- recording quotas
- retention policies
- cleanup
- storage monitoring
- cloud synchronization where selected

---

# 24. microSD Reliability Constraint

## Constraint

Repeated writes and unexpected power loss may affect microSD reliability.

## Impact

Database or filesystem corruption is possible.

## Mitigation

- graceful shutdown
- bounded logging
- appropriate SQLite transaction handling
- backups where appropriate
- avoid unnecessary writes

---

# 25. Facial Recognition Constraint

## Constraint

Recognition performance varies with:

- lighting
- pose
- distance
- occlusion
- camera quality

## Impact

Known users may occasionally be missed or incorrectly matched.

## Mitigation

- multiple enrollment samples
- confidence thresholds
- unknown result
- evaluation across conditions

---

# 26. Facial Expression Constraint

## Constraint

Computer vision estimates facial expressions; it does not know a person's psychological state with certainty.

## Impact

Expression labels may be incorrect and must not be interpreted clinically.

## Mitigation

Use careful terminology and confidence.

Never claim medical or mental-health diagnosis.

---

# 27. Object Learning Constraint

## Constraint

A small number of user-provided examples does not provide universal semantic understanding.

## Impact

Recognition may work only for visually similar examples.

## Mitigation

Describe initial functionality as example/embedding-based learning and measure its performance.

---

# 28. Safety Timing Constraint

## Constraint

AI workloads may introduce latency.

## Impact

Safety decisions could be delayed if implemented in the same blocking execution path.

## Mitigation

Safety-critical processing must remain independent/responsive.

AI must never block emergency stop or collision protection.

---

# 29. Development Deadline Constraint

## Constraint

Buddy is a final-year research project with limited development time.

## Impact

Attempting every advanced feature at production quality may prevent completion of core research functionality.

## Mitigation

Use:

- MUST
- SHOULD
- COULD
- EXPERIMENTAL

prioritization.

Complete and verify core functionality before optional enhancements.

---

# 30. Research Scope Constraint

## Constraint

The project must remain academically defensible.

## Impact

Unsupported claims about AI, localization, emotion or mapping would weaken the research.

## Mitigation

Measure performance experimentally and clearly report limitations.

Never fabricate benchmark results.

---

# 31. Legacy Code Constraint

## Constraint

Existing Buddy voice/face code already represents working/prototype functionality.

## Impact

Large uncontrolled rewrites could introduce regressions.

## Mitigation

Existing source is preserved until explicit migration tasks.

Future integration shall use stable interfaces and regression testing.

---

# 32. Configuration Constraint

## Constraint

Hardware pin assignments, thresholds, paths and credentials may vary between development environments.

## Impact

Hard-coded values reduce portability and increase safety/security risk.

## Mitigation

Future architecture shall centralize configuration appropriately and keep secrets outside Git.

---

# 33. Final Constraint Principle

When hardware, software or research evidence cannot support a requested feature reliably:

**Buddy shall degrade gracefully or disable that feature rather than pretend the capability works.**
