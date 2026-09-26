# BUDDY v1.0 — System Requirements Specification

**Project:** BUDDY — Affordable AI-Powered Intelligent Robot Dog
**Platform:** Raspberry Pi 4 Model B — 4 GB
**Operating System:** Ubuntu 24.04 LTS
**Robotics Middleware:** ROS 2 Jazzy
**Document:** System Requirements Specification
**Status:** Task 01 — Requirements Baseline

---

# 1. Project Overview

BUDDY is an affordable AI-powered intelligent robot dog developed as a final-year research project.

Buddy is intended to combine:

- voice interaction
- computer vision
- facial recognition
- facial-expression estimation
- user memory
- autonomous and manual movement
- obstacle detection
- person tracking
- sound-direction estimation
- security monitoring
- autonomous patrol
- mobile remote control
- live video/audio telepresence
- offline-first intelligence
- optional online AI and cloud services

The system shall be developed as a modular ROS 2 architecture rather than as one monolithic Python application.

Development follows:

**ONE TASK → IMPLEMENT → TEST → VERIFY → COMMIT → NEXT TASK**

Existing Buddy source code is treated as legacy/prototype functionality until explicitly migrated by a later task.

---

# 2. Product Vision

Buddy shall behave as one coordinated robotic companion rather than as a collection of unrelated AI demonstrations.

The long-term interaction model is:

**Hear → Understand → Observe → Decide → Validate → Act → Remember**

Physical actions shall always pass deterministic safety validation.

Conversational AI shall never have unrestricted physical control.

---

# 3. Requirement Priority Classification

Every requirement uses one of the following priorities.

## MUST

Required for Buddy v1.0 or required for system/safety integrity.

## SHOULD

Important functionality that should be implemented, but reduced capability may be accepted when hardware, time, or Raspberry Pi resource limitations prevent the ideal implementation.

## COULD

Enhancement that may be implemented if time and resources permit.

## EXPERIMENTAL

Research capability whose feasibility or performance must be experimentally determined.

Experimental functionality must never be represented as guaranteed.

---

# 4. Hardware Baseline

### BUD-HW-001 — Primary Computer — MUST

Buddy shall use a Raspberry Pi 4 Model B with 4 GB RAM as its primary onboard computer.

### BUD-HW-002 — Camera — MUST

Buddy shall use a Raspberry Pi Camera Module V2 / IMX219 as the baseline vision sensor.

### BUD-HW-003 — Microphones — MUST

Buddy shall support two separate USB lavalier microphones.

Exact microphone model, USB audio chipset, clock behaviour, sample-rate support, and synchronization capability:

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

### BUD-HW-004 — Speakers — MUST

Buddy shall support onboard speaker output.

Exact speaker amplifier and electrical power requirements:

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

### BUD-HW-005 — Drive Platform — MUST

Buddy shall use a four-wheel chassis with four DC geared motors.

Motor rated voltage, operating current and stall current:

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

### BUD-HW-006 — Motor Drivers — MUST

Two TB6612FNG motor-driver boards are planned for motor control.

Electrical suitability shall be verified before connecting motors.

### BUD-HW-007 — Obstacle Sensors — MUST

Buddy shall use four HC-SR04 ultrasonic sensors intended for:

- front
- rear
- left
- right

Raspberry Pi GPIO shall not receive unsafe 5 V echo signals.

Required level conversion/protection shall be verified before connection.

### BUD-HW-008 — Camera Servos — SHOULD

Buddy shall use two MG90S metal-gear servos for camera pan and tilt.

### BUD-HW-009 — Camera Mount — SHOULD

Buddy shall use a two-axis pan/tilt camera bracket.

### BUD-HW-010 — Servo Controller — SHOULD

Buddy shall use a PCA9685 controller for servo control where appropriate.

### BUD-HW-011 — Power Conversion — MUST

An LM2596 buck converter is part of the planned power architecture.

Final voltage/current configuration:

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

### BUD-HW-012 — Battery — MUST

The current planned battery arrangement uses three 18650 cells.

The following shall be verified before integration:

- cell chemistry
- capacity
- discharge capability
- battery configuration
- BMS/protection
- charging method
- fuse/protection
- output voltage
- motor power requirements
- Raspberry Pi power requirements

### BUD-HW-013 — USB Hub — MUST

Buddy shall support the required USB devices through an appropriate USB hub.

Hub power and bandwidth requirements shall be tested.

### BUD-HW-014 — Cooling — MUST

The Raspberry Pi shall use appropriate cooling to support sustained AI/robotics workloads.

### BUD-HW-015 — Storage — MUST

Buddy shall use microSD or another supported local storage medium for operating system, software, database, logs, models and local event data.

### BUD-HW-016 — LCD — COULD

An LCD display may be integrated.

Exact model/interface:

**TBD — MUST VERIFY BEFORE HARDWARE INTEGRATION**

---

# 5. Software Platform Requirements

### BUD-SYS-001 — Operating System — MUST

Buddy shall use Ubuntu 24.04 LTS.

### BUD-SYS-002 — Robotics Middleware — MUST

Buddy shall use ROS 2 Jazzy.

### BUD-SYS-003 — Modular Architecture — MUST

Buddy shall not be implemented as one giant Python program.

Major functionality shall be divided into independently testable modules/nodes.

### BUD-SYS-004 — Stable Interfaces — MUST

Subsystems shall communicate through defined interfaces.

Future tasks shall not silently rewrite previously verified functionality.

### BUD-SYS-005 — Resource Awareness — MUST

All production components shall respect Raspberry Pi 4 limitations involving:

- CPU
- RAM
- temperature
- storage
- USB bandwidth
- camera bandwidth
- inference latency
- battery/power

---

# 6. ROS 2 Requirements

### BUD-ROS-001 — Topics — MUST

ROS 2 topics shall be used for appropriate continuous or streaming state data.

### BUD-ROS-002 — Services — MUST

ROS 2 services shall be used for appropriate short request/response operations.

### BUD-ROS-003 — Actions — MUST

ROS 2 actions shall be used for long-running cancellable behaviours where appropriate.

Candidate future actions include:

- ApproachCaller
- FollowPerson
- FollowVisualTarget
- PatrolRoute
- NavigateToGoal
- ExploreArea

Exact interfaces shall be defined in a later task.

### BUD-ROS-004 — Safety Isolation — MUST

Failure of experimental AI nodes shall not disable critical movement safety.

### BUD-ROS-005 — Node Failure Handling — MUST

Critical node failure shall be detectable and shall trigger an appropriate safe/degraded state.

---

# 7. Voice and Audio Requirements

### BUD-AUD-001 — Microphone Detection — MUST

Buddy shall detect and identify available microphone devices.

### BUD-AUD-002 — Independent Testing — MUST

Each microphone shall be independently testable.

### BUD-AUD-003 — Simultaneous Capture — MUST

Buddy shall attempt simultaneous capture from both USB microphones.

### BUD-AUD-004 — Audio Measurement — MUST

Buddy shall support measurement of audio levels such as RMS for microphone testing and calibration.

### BUD-AUD-005 — Voice Activity Detection — MUST

Buddy shall detect periods of likely human speech before invoking expensive speech processing where practical.

### BUD-AUD-006 — Wake Word — MUST

Buddy shall support the wake word:

**Buddy**

### BUD-AUD-007 — Speech Capture — MUST

After valid activation, Buddy shall capture the relevant spoken command.

### BUD-AUD-008 — Offline STT — MUST

Core voice commands shall support offline speech-to-text.

### BUD-AUD-009 — Sri Lankan English — SHOULD

Speech-recognition performance with Sri Lankan English shall be experimentally evaluated and improved where practical.

### BUD-AUD-010 — Intent Recognition — MUST

Recognized text shall be passed through a deterministic command/intent layer.

### BUD-AUD-011 — TTS — MUST

Buddy shall provide local text-to-speech for core responses.

### BUD-AUD-012 — Self-Speech Protection — SHOULD

Buddy shall reduce or prevent accidental recognition of speech generated by its own speakers.

### BUD-AUD-013 — Interruption — SHOULD

Voice architecture should support safe interruption/cancellation where appropriate.

---

# 8. Sound Direction Requirements

### BUD-SND-001 — Direction Research — EXPERIMENTAL

Buddy shall investigate sound-direction estimation using the two available USB microphones.

### BUD-SND-002 — RMS Comparison — EXPERIMENTAL

Relative microphone energy shall be evaluated as one possible direction cue.

### BUD-SND-003 — Cross-Correlation — EXPERIMENTAL

Cross-correlation/time-delay approaches shall be experimentally evaluated.

### BUD-SND-004 — Synchronization Validation — MUST

The timing relationship between both USB microphones shall be experimentally measured before claiming TDOA accuracy.

### BUD-SND-005 — Direction Confidence — SHOULD

Any direction estimate shall include a confidence/reliability measure where practical.

### BUD-SND-006 — Calibration — MUST

Microphone placement and environmental calibration shall be performed.

### BUD-SND-007 — Limitation Disclosure — MUST

Buddy shall not claim perfect 360-degree localization from two independent USB microphones.

Potential limitations include:

- front/back ambiguity
- reflections
- environmental noise
- USB latency
- independently clocked devices
- synchronization drift

### BUD-SND-008 — Future Array — COULD

A synchronized microphone array may be recommended if the current hardware cannot provide adequate localization.

---

# 9. Conversation and Intelligence Requirements

### BUD-AI-001 — Deterministic Commands — MUST

Safety-relevant commands shall use deterministic command routing.

### BUD-AI-002 — Conversation Manager — SHOULD

Buddy shall maintain conversational context where appropriate.

### BUD-AI-003 — Personality — SHOULD

Buddy may provide a consistent companion-style personality.

### BUD-AI-004 — User-Specific Responses — SHOULD

Responses may be adapted using known user information and preferences.

### BUD-AI-005 — Offline Intelligence — MUST

Essential commands and responses shall remain available without internet connectivity.

### BUD-AI-006 — Existing AI Techniques — SHOULD

Existing prototype techniques including TF-IDF, Logistic Regression, Vosk and related components may be evaluated and migrated where appropriate.

### BUD-AI-007 — Optional LLM — COULD

An optional conversational LLM may be integrated.

Potential technologies include Ollama/Llama or another justified local/remote model.

### BUD-AI-008 — LLM Resource Feasibility — EXPERIMENTAL

Running an onboard LLM simultaneously with the full vision/audio/navigation workload on Raspberry Pi 4B 4 GB requires benchmarking.

### BUD-AI-009 — LLM Physical Isolation — MUST

An LLM shall never directly control:

- GPIO
- motors
- navigation
- unsafe servo movement
- emergency systems
- security-critical actions

An LLM may only propose an intent that passes deterministic validation.

---

# 10. Computer Vision Requirements

### BUD-VIS-001 — Camera Initialization — MUST

Buddy shall detect and initialize the Raspberry Pi Camera V2.

### BUD-VIS-002 — Camera Health — MUST

Camera availability/failure shall be detectable.

### BUD-VIS-003 — Camera Configuration — MUST

Resolution and FPS shall be configurable.

### BUD-VIS-004 — Face Detection — MUST

Buddy shall detect human faces.

### BUD-VIS-005 — Face Recognition — MUST

Buddy shall recognize enrolled known users.

### BUD-VIS-006 — Unknown Person — MUST

Buddy shall support an unknown-person result when identification confidence is insufficient.

### BUD-VIS-007 — Facial Expression — MUST

Buddy shall estimate supported facial-expression classes.

### BUD-VIS-008 — Identity + Expression — MUST

When confidence permits, expression observations shall be associated with a recognized user.

### BUD-VIS-009 — Person Tracking — SHOULD

Buddy shall support camera-based person tracking.

### BUD-VIS-010 — Hand Tracking — SHOULD

Buddy should support hand detection/tracking.

### BUD-VIS-011 — Finger Tracking — SHOULD

Buddy should support index-fingertip tracking.

### BUD-VIS-012 — Learned Objects — COULD

Buddy may support recognition of user-taught visual examples.

### BUD-VIS-013 — Security Capture — MUST

Buddy shall support local security image/video capture.

### BUD-VIS-014 — Pi Optimization — MUST

Heavy computer-vision inference shall be scheduled/optimized rather than necessarily running every model on every frame.

---

# 11. Camera Head Requirements

### BUD-HEAD-001 — Pan — SHOULD

Buddy's camera shall be capable of looking left and right.

### BUD-HEAD-002 — Tilt — SHOULD

Buddy's camera shall be capable of looking up and down.

### BUD-HEAD-003 — Centre — SHOULD

Buddy shall support returning the camera to a defined centre position.

### BUD-HEAD-004 — Smooth Motion — SHOULD

Pan/tilt motion should be sufficiently smooth for tracking.

### BUD-HEAD-005 — Servo Limits — MUST

Mechanical/software servo limits shall prevent commands outside verified safe ranges.

### BUD-HEAD-006 — Target Tracking — SHOULD

The camera head should track selected visual targets.

### BUD-HEAD-007 — Head Before Body — SHOULD

For suitable tracking behaviours, Buddy should first use head movement.

When the target leaves practical head range, a validated body-rotation request may be issued.

---

# 12. Person Learning Requirements

### BUD-PLR-001 — Voice Enrollment — SHOULD

Buddy should support commands similar to:

**“Buddy, this is Robert.”**

### BUD-PLR-002 — Multiple Samples — SHOULD

Person enrollment shall capture multiple useful face samples/representations.

### BUD-PLR-003 — Name Association — SHOULD

Samples shall be associated with a user-confirmed identity/name.

### BUD-PLR-004 — Confirmation — MUST

Enrollment shall require appropriate confirmation before permanently creating/changing identity data.

### BUD-PLR-005 — Cancellation — MUST

Enrollment shall support cancellation.

### BUD-PLR-006 — Duplicate Detection — SHOULD

Buddy should detect likely duplicate enrollment.

### BUD-PLR-007 — Persistence — MUST

Enrolled user information shall persist across restarts.

---

# 13. Object Learning Requirements

### BUD-OLR-001 — Voice-Assisted Learning — COULD

Buddy may support commands such as:

**“Buddy, this is a chair.”**

### BUD-OLR-002 — Multiple Views — COULD

Several useful views/examples should be captured.

### BUD-OLR-003 — Label Association — COULD

Captured visual representations shall be associated with a user-confirmed label.

### BUD-OLR-004 — Realistic Capability — MUST

The system shall not claim that one image teaches Buddy the universal semantic concept of an object category.

Initial implementation shall be treated as example/embedding-based visual learning.

---

# 14. Emotion History Requirements

### BUD-EMO-001 — Supported Classes — MUST

Buddy may estimate supported expression classes such as:

- happy
- sad
- angry
- surprised
- neutral
- other outputs supported by the selected model

### BUD-EMO-002 — Event Deduplication — MUST

Every video frame shall not be counted as a separate emotional event.

### BUD-EMO-003 — Confidence — MUST

Expression observations shall use confidence thresholds/quality validation.

### BUD-EMO-004 — Time Windows — MUST

Sampling and time windows shall reduce duplicate observations.

### BUD-EMO-005 — Daily Summary — SHOULD

Buddy shall support daily expression-history summaries.

### BUD-EMO-006 — Weekly Summary — SHOULD

Buddy shall support weekly summaries.

### BUD-EMO-007 — Monthly Summary — SHOULD

Buddy shall support monthly summaries.

### BUD-EMO-008 — Adaptive Interaction — COULD

Historical expression observations may influence non-medical interaction style.

### BUD-EMO-009 — No Diagnosis — MUST

Expression recognition shall not be represented as:

- medical diagnosis
- mental-health diagnosis
- psychological certainty

---

# 15. Memory Requirements

### BUD-MEM-001 — SQLite — MUST

SQLite shall be the preferred structured local database.

### BUD-MEM-002 — User Profiles — MUST

Buddy shall persist appropriate user profiles.

### BUD-MEM-003 — Face Profiles — MUST

Face identity metadata shall be persistently managed.

### BUD-MEM-004 — Preferences — SHOULD

Buddy may store user preferences.

### BUD-MEM-005 — Interaction History — SHOULD

Relevant interaction history may be stored.

### BUD-MEM-006 — Emotion History — MUST

Expression observations and summaries shall be stored where required.

### BUD-MEM-007 — Learned Objects — COULD

Learned-object metadata may be stored.

### BUD-MEM-008 — Security History — MUST

Security-event metadata shall be stored.

### BUD-MEM-009 — Patrol History — SHOULD

Patrol execution/history should be stored.

### BUD-MEM-010 — Robot Events — MUST

Important robot/system events shall be persistently recordable.

### BUD-MEM-011 — Mapping Metadata — COULD

Map/restricted-zone metadata may be stored if mapping becomes feasible.

### BUD-MEM-012 — Cloud Sync State — SHOULD

Pending/synchronized cloud items shall be persistently tracked.

### BUD-MEM-013 — Media Files — MUST

Large images/videos shall normally be stored as files with database metadata/path references.

### BUD-MEM-014 — Offline Availability — MUST

Core local memory shall work without cloud access.

---

# 16. Movement Requirements

### BUD-MOV-001 — Forward — MUST

Buddy shall support validated forward motion.

### BUD-MOV-002 — Reverse — MUST

Buddy shall support validated reverse motion.

### BUD-MOV-003 — Left — MUST

Buddy shall support left turning.

### BUD-MOV-004 — Right — MUST

Buddy shall support right turning.

### BUD-MOV-005 — Rotation — MUST

Buddy shall support in-place or practical chassis rotation where hardware permits.

### BUD-MOV-006 — Stop — MUST

Buddy shall provide immediate software stop capability.

### BUD-MOV-007 — Speed — MUST

Movement architecture shall support bounded speed control.

### BUD-MOV-008 — Acceleration — SHOULD

Controlled acceleration/deceleration should be used where practical.

### BUD-MOV-009 — Control Pipeline — MUST

Physical movement shall follow:

**Command Source → Arbitration → Safety Validation → Motor Controller**

### BUD-MOV-010 — No AI Bypass — MUST

No AI subsystem may bypass the movement safety pipeline.

---

# 17. Emergency Stop and Movement Safety

### BUD-SAF-001 — E-Stop Priority — MUST

Emergency stop has absolute highest authority.

### BUD-SAF-002 — E-Stop Latching — MUST

After emergency stop, intentional movement shall remain disabled until an explicit safe reset.

### BUD-SAF-003 — Command Timeout — MUST

Movement commands shall expire.

### BUD-SAF-004 — Stale Command Rejection — MUST

Expired/stale commands shall be rejected.

### BUD-SAF-005 — Connection Loss — MUST

Loss of manual remote-control connection shall stop manual motion.

### BUD-SAF-006 — Speed Limits — MUST

Configured speed limits shall be enforced.

### BUD-SAF-007 — Invalid Command — MUST

Invalid movement commands shall be rejected.

### BUD-SAF-008 — Obstacle Override — MUST

Collision/obstacle protection may reject a movement request from any source.

### BUD-SAF-009 — Startup Safety — MUST

Buddy shall boot into a non-moving state.

### BUD-SAF-010 — Shutdown Safety — MUST

Shutdown/crash shall result in motor stop where technically possible.

### BUD-SAF-011 — Sensor Failure — MUST

Failure of critical safety sensing shall cause appropriately restricted/degraded operation.

---

# 18. Ultrasonic Requirements

### BUD-USS-001 — Four Directions — MUST

Ultrasonic sensing shall support front, rear, left and right coverage.

### BUD-USS-002 — Distance Measurement — MUST

Valid distance measurements shall be made available to safety/navigation.

### BUD-USS-003 — Invalid Measurement — MUST

Invalid/out-of-range sensor measurements shall be detected.

### BUD-USS-004 — Filtering — MUST

Sensor noise shall be filtered appropriately.

### BUD-USS-005 — Health — MUST

Sensor health/failure shall be detectable.

### BUD-USS-006 — Cross-Talk — MUST

Multiple ultrasonic sensors shall be scheduled/coordinated to reduce acoustic interference.

---

# 19. Navigation Requirements

### BUD-NAV-001 — Obstacle Avoidance — SHOULD

Buddy shall support safe obstacle avoidance.

### BUD-NAV-002 — Autonomous Movement — SHOULD

Buddy shall support bounded autonomous movement.

### BUD-NAV-003 — Recovery — SHOULD

Navigation shall support recovery from recoverable blocked/lost situations.

### BUD-NAV-004 — Caller Approach — SHOULD

Buddy should eventually approach a detected caller safely.

### BUD-NAV-005 — Person Following — SHOULD

Buddy should support following a selected person.

### BUD-NAV-006 — Patrol — SHOULD

Buddy should support autonomous patrol when localization/navigation capability permits.

### BUD-NAV-007 — Exploration — COULD

Buddy may explore an environment where safe and technically feasible.

### BUD-NAV-008 — Restricted Areas — COULD

Navigation may respect map-based restricted zones when reliable mapping/localization exists.

---

# 20. Person Following Requirements

### BUD-FOL-001 — Target Identity — SHOULD

Following shall operate on an explicitly selected/validated target.

### BUD-FOL-002 — Visual Tracking — SHOULD

The selected person shall be visually tracked.

### BUD-FOL-003 — Distance Control — MUST

Following shall maintain a safe target distance.

### BUD-FOL-004 — Obstacle Safety — MUST

Obstacle safety shall override following movement.

### BUD-FOL-005 — Target Loss — MUST

Buddy shall stop or enter bounded reacquisition when target confidence is inadequate.

### BUD-FOL-006 — Cancellation — MUST

Following shall be cancellable.

### BUD-FOL-007 — Timeout — MUST

Unsuccessful reacquisition shall eventually timeout.

---

# 21. Finger Following Requirements

### BUD-FNG-001 — Hand Detection — COULD

Buddy may detect a user's hand.

### BUD-FNG-002 — Index Fingertip — COULD

Buddy may estimate the index-fingertip position.

### BUD-FNG-003 — Target Lock — COULD

The fingertip may be used as a visual tracking target.

### BUD-FNG-004 — Camera Tracking — COULD

The camera head may track the fingertip.

### BUD-FNG-005 — Body Rotation — COULD

Body rotation may be requested when head range becomes insufficient.

### BUD-FNG-006 — Safety — MUST

Any physical movement caused by finger tracking shall pass normal obstacle/movement safety.

### BUD-FNG-007 — Target Loss — MUST

Physical finger-follow behaviour shall stop when target confidence becomes inadequate.

---

# 22. Mapping Requirements

### BUD-MAP-001 — Mapping Research — EXPERIMENTAL

Buddy shall investigate generation of a room/environment map.

### BUD-MAP-002 — Pose Requirement — MUST

The project shall acknowledge that reliable mapping/SLAM generally requires adequate pose estimation.

### BUD-MAP-003 — Hardware Feasibility Check — MUST

Before claiming accurate SLAM, the project shall evaluate the availability/need for:

- wheel encoders
- IMU
- 2D LiDAR
- ToF/depth sensing
- other justified localization hardware

### BUD-MAP-004 — Current Limitation — MUST

Software shall not claim accurate physical position when available hardware cannot support it.

### BUD-MAP-005 — Future Hardware — COULD

Additional localization sensors may be added if required.

---

# 23. Restricted Zone Requirements

### BUD-ZON-001 — App Selection — COULD

If mapping becomes reliable, the owner may define map regions as restricted/no-entry areas.

### BUD-ZON-002 — Navigation Enforcement — MUST

If restricted zones are enabled, autonomous navigation shall not intentionally enter them.

### BUD-ZON-003 — Patrol Enforcement — MUST

Patrol shall respect active restricted zones.

### BUD-ZON-004 — Following Enforcement — MUST

Autonomous following shall respect active restricted zones.

---

# 24. Patrol Requirements

### BUD-PAT-001 — Start — SHOULD

Buddy shall support starting a configured patrol.

### BUD-PAT-002 — Stop — MUST

Patrol shall be safely stoppable.

### BUD-PAT-003 — Pause — SHOULD

Patrol should support pause/resume where appropriate.

### BUD-PAT-004 — Status — SHOULD

Patrol progress/status shall be observable.

### BUD-PAT-005 — History — SHOULD

Patrol history should be recorded.

### BUD-PAT-006 — Obstacle Safety — MUST

Obstacle safety shall override patrol movement.

### BUD-PAT-007 — Manual Override — MUST

Manual remote control shall safely suspend/override patrol.

---

# 25. Security Requirements

### BUD-SEC-001 — Known Person — MUST

Buddy shall identify enrolled users where recognition confidence permits.

### BUD-SEC-002 — Unknown Person — MUST

Buddy shall produce an unknown-person security result where appropriate.

### BUD-SEC-003 — Event Creation — MUST

Significant security detections shall create timestamped events.

### BUD-SEC-004 — Image Capture — MUST

Security events shall support image capture.

### BUD-SEC-005 — Video — SHOULD

Significant events should support bounded video recording.

### BUD-SEC-006 — Local First — MUST

Security data shall be stored locally before optional cloud synchronization.

### BUD-SEC-007 — Notification — SHOULD

When connectivity exists, Buddy should notify the owner of significant events.

### BUD-SEC-008 — Remote Inspection — SHOULD

The owner should be able to open live control after an alert.

### BUD-SEC-009 — Alert Deduplication — MUST

A continuously visible unknown person shall not create uncontrolled repeated alert flooding.

---

# 26. Incident Recording Requirements

### BUD-REC-001 — Media Capture — MUST

Significant incidents shall support relevant media capture.

### BUD-REC-002 — Metadata — MUST

Incident metadata shall include timestamps and relevant event information.

### BUD-REC-003 — Local Storage — MUST

Incident media shall be stored locally first.

### BUD-REC-004 — Rolling Buffer — COULD

A bounded rolling camera buffer may support pre-trigger and post-trigger incident footage.

### BUD-REC-005 — Quota — MUST

Recording storage shall have defined quotas.

### BUD-REC-006 — Retention — MUST

Retention/cleanup policies shall prevent unbounded storage growth.

---

# 27. Offline Requirements

### BUD-OFF-001 — Movement — MUST
Movement shall not require internet.

### BUD-OFF-002 — Safety — MUST
Safety and E-stop shall not require internet.

### BUD-OFF-003 — Sensors — MUST
Obstacle detection shall not require internet.

### BUD-OFF-004 — Vision — MUST
Core camera/known-user recognition/basic expression estimation shall work locally where technically practical.

### BUD-OFF-005 — Voice — MUST
Wake word, core STT, deterministic commands and local TTS shall work offline where technically practical.

### BUD-OFF-006 — Memory — MUST
Core memory shall work locally.

### BUD-OFF-007 — Local Control — MUST
Same-network manual control shall not require external internet connectivity.

### BUD-OFF-008 — Recording — MUST
Security recording shall remain available locally.

### BUD-OFF-009 — Internet Loss — MUST
Internet loss shall not crash Buddy.

---

# 28. Cloud Synchronization Requirements

### BUD-CLD-001 — Persistent Queue — SHOULD

Important cloud-bound events shall enter a persistent synchronization queue.

### BUD-CLD-002 — Reconnect Upload — SHOULD

Pending data may synchronize when connectivity returns.

### BUD-CLD-003 — Retry — MUST

Cloud synchronization shall use bounded retry/backoff behaviour.

### BUD-CLD-004 — Duplicate Avoidance — MUST

Synchronization shall prevent uncontrolled duplicate uploads.

### BUD-CLD-005 — Persistent State — MUST

Sync state shall survive restart where required.

### BUD-CLD-006 — Provider — COULD

A future task may select Google Drive, Firebase, Supabase, or another justified service.

Task 01 does not select the provider.

---

# 29. Local Mobile Control Requirements

### BUD-LOC-001 — Same Network — MUST

The mobile app shall support local communication with Buddy on the same network.

### BUD-LOC-002 — Manual Movement — MUST

The app shall provide validated manual movement commands.

### BUD-LOC-003 — E-Stop — MUST

The app shall provide accessible emergency stop functionality.

### BUD-LOC-004 — Camera — MUST

The app shall support live camera viewing when streaming functionality is available.

### BUD-LOC-005 — Head Control — SHOULD

The app should support camera pan/tilt controls.

### BUD-LOC-006 — Audio — SHOULD

The app should support live listening and push-to-talk.

### BUD-LOC-007 — Robot Status — MUST

Robot health/state shall be visible.

---

# 30. Internet Remote Control Requirements

### BUD-REM-001 — Secure Remote Access — MUST

Remote access shall use a secure architecture.

### BUD-REM-002 — No Direct Public Ports — MUST

Raspberry Pi GPIO/control services shall never be directly exposed to the public internet.

### BUD-REM-003 — Remote Status — SHOULD

The owner should be able to check Buddy status remotely.

### BUD-REM-004 — Remote Camera — SHOULD

The owner should be able to view live video remotely.

### BUD-REM-005 — Remote Audio — SHOULD

The owner should be able to hear Buddy's environment and speak through Buddy.

### BUD-REM-006 — Remote Driving — SHOULD

Authorized owners should be able to drive Buddy remotely subject to all movement safety rules.

### BUD-REM-007 — Remote Stop — MUST

Remote control shall always provide stop capability.

---

# 31. Live Video and Audio Requirements

### BUD-MED-001 — WebRTC Evaluation — SHOULD

WebRTC shall be seriously evaluated for low-latency live video and two-way audio.

### BUD-MED-002 — Mobile Compatibility — MUST

Selected media technology shall support the chosen mobile application architecture.

### BUD-MED-003 — Pi Compatibility — MUST

Selected media technology shall be feasible on Raspberry Pi 4.

### BUD-MED-004 — Reconnection — MUST

Media sessions shall support recovery/reconnection.

### BUD-MED-005 — Security — MUST

Media sessions shall be authenticated/encrypted appropriately.

### BUD-MED-006 — Push-to-Talk — SHOULD

Push-to-talk may be used initially to reduce feedback and implementation complexity.

---

# 32. Control and Telemetry Networking

### BUD-NET-001 — Separate Control Channel — MUST

Robot control shall use a suitable mechanism such as WebSocket rather than embedding physical commands into the media stream.

### BUD-NET-002 — Authorization — MUST

Movement commands shall be authorized.

### BUD-NET-003 — Validation — MUST

Remote movement commands shall pass deterministic validation.

### BUD-NET-004 — Freshness — MUST

Commands shall include sufficient timestamp/sequence/freshness information to reject stale/replayed movement instructions.

### BUD-NET-005 — Connection Failsafe — MUST

Loss of the active manual-control connection shall stop manual movement.

---

# 33. Network Security Requirements

### BUD-AUTH-001 — Device Identity — MUST
Buddy shall possess an appropriate device identity.

### BUD-AUTH-002 — Authentication — MUST
Owner access shall require authentication.

### BUD-AUTH-003 — Authorization — MUST
Authenticated users shall only access authorized capabilities.

### BUD-AUTH-004 — Encryption — MUST
Sensitive network communication shall use encrypted transport.

### BUD-AUTH-005 — Token Security — MUST
Tokens/credentials shall be securely handled.

### BUD-AUTH-006 — Token Expiration — MUST
Authentication/session tokens shall expire appropriately.

### BUD-AUTH-007 — Session Termination — MUST
Sessions shall be terminable/revocable where appropriate.

### BUD-AUTH-008 — Replay Protection — SHOULD
Sensitive commands should include replay resistance.

### BUD-AUTH-009 — Rate Limiting — SHOULD
Externally reachable APIs should use appropriate rate limiting.

### BUD-AUTH-010 — Brute Force Protection — SHOULD
Authentication should include appropriate brute-force mitigation.

### BUD-AUTH-011 — Git Secrets — MUST
Passwords, API keys, private credentials and tokens shall never be committed to Git.

---

# 34. Mobile Application Requirements

### BUD-APP-001 — Technology — MUST

React Native is the preferred application technology unless later engineering benchmarking provides a strong reason to change.

### BUD-APP-002 — Dashboard — MUST

Dashboard shall eventually expose:

- Buddy online/offline
- current mode
- battery where measurable
- network state
- camera status
- microphone status
- activity
- robot health

### BUD-APP-003 — Live Control — MUST

Live Control shall eventually provide:

- live video
- forward
- backward
- left
- right
- stop
- bounded speed control
- emergency stop

### BUD-APP-004 — Camera Head — SHOULD

App should provide:

- left
- right
- up
- down
- centre

### BUD-APP-005 — Telepresence — SHOULD

App should support:

- listen
- mute
- push-to-talk
- speaker control

### BUD-APP-006 — Modes — MUST

App shall provide safe switching between manual and supported autonomous modes.

### BUD-APP-007 — Patrol — SHOULD

App should expose patrol start/stop/status/history.

### BUD-APP-008 — Security — SHOULD

App should expose alerts, timestamps, event images/video and a Live Control shortcut.

### BUD-APP-009 — Map — COULD

If mapping becomes reliable, app may display map, estimated robot position, patrol routes and restricted zones.

### BUD-APP-010 — Users — SHOULD

App should support management/display of known users and learned profiles.

### BUD-APP-011 — Buddy Information — SHOULD

App should expose recognized user, current supported expression, emotion history/reports and interaction history where appropriate.

---

# 35. State Machine Requirements

### BUD-STA-001 — States — MUST

Buddy shall define at least:

- STARTING
- IDLE
- LISTENING
- INTERACTING
- MANUAL_CONTROL
- FOLLOWING
- AUTONOMOUS
- PATROL
- SECURITY_ALERT
- ERROR
- EMERGENCY_STOP
- SHUTTING_DOWN

### BUD-STA-002 — Valid Transitions — MUST

Only explicitly permitted state transitions shall be accepted.

### BUD-STA-003 — Conflict Prevention — MUST

Conflicting movement behaviours shall not simultaneously control the robot.

### BUD-STA-004 — Priority — MUST

System authority shall follow the safety principle:

**EMERGENCY_STOP**
>
**Safety**
>
**Authorized Manual Control**
>
**Validated Higher-Priority Robot Behaviour**
>
**Following**
>
**Autonomous/Patrol Behaviour**
>
**Idle Behaviour**

Exact arbitration rules shall be frozen later.

---

# 36. Concurrency Requirements

### BUD-CON-001 — Concurrent Subsystems — MUST

Architecture shall support concurrent operation of appropriate:

- camera capture
- microphone capture
- wake detection
- STT
- vision inference
- sensor polling
- motor control
- network communication
- media streaming

### BUD-CON-002 — Safety Responsiveness — MUST

CPU-heavy AI processing shall not block safety processing.

### BUD-CON-003 — Bounded Queues — MUST

Queues shall be bounded where unbounded growth could exhaust memory.

### BUD-CON-004 — Stale Media — MUST

Stale camera/audio processing data may be dropped instead of accumulating indefinitely.

### BUD-CON-005 — Failure Isolation — MUST

Experimental AI functionality shall not share unnecessary failure domains with safety-critical components.

---

# 37. Failure Handling Requirements

### BUD-FLT-001 — Camera Failure — MUST

Camera failure shall not unnecessarily disable healthy voice, manual movement or ultrasonic safety functionality.

### BUD-FLT-002 — One Microphone Failure — MUST

Buddy should continue single-microphone voice interaction where possible while disabling/degrading sound localization.

### BUD-FLT-003 — Internet Failure — MUST

Core offline functionality shall remain operational.

### BUD-FLT-004 — Cloud Failure — MUST

Pending data shall remain locally queued.

### BUD-FLT-005 — LLM Failure — MUST

Deterministic commands shall remain operational.

### BUD-FLT-006 — Sensor Failure — MUST

Critical obstacle-sensor failure shall produce an appropriately restricted movement mode.

### BUD-FLT-007 — Database Failure — MUST

Database failure shall be detected/logged and shall not result in unsafe physical behaviour.

### BUD-FLT-008 — API Failure — MUST

External API failure shall not crash core robot functionality.

---

# 38. Logging Requirements

### BUD-LOG-001 — Central Logging — MUST

Buddy shall log significant:

- startup
- shutdown
- node initialization/health
- microphone state
- voice commands
- recognition events
- movement
- safety blocks
- security events
- network sessions
- errors
- performance warnings

### BUD-LOG-002 — Rotation — MUST

Logs shall rotate.

### BUD-LOG-003 — Storage Protection — MUST

Logs shall not grow indefinitely.

### BUD-LOG-004 — Secret Protection — MUST

Passwords/tokens shall never be written to logs.

---

# 39. Performance Requirements

### BUD-PER-001 — CPU — MUST
CPU usage shall be measurable.

### BUD-PER-002 — RAM — MUST
RAM usage shall be measurable.

### BUD-PER-003 — Temperature — MUST
Raspberry Pi temperature shall be measurable.

### BUD-PER-004 — Camera FPS — MUST
Camera processing FPS shall be measured.

### BUD-PER-005 — Vision Latency — MUST
Face and expression inference latency shall be measured.

### BUD-PER-006 — Voice Latency — MUST
Wake-word/STT latency shall be measured.

### BUD-PER-007 — Motion Latency — MUST
Command-to-movement latency shall be measured.

### BUD-PER-008 — Video Latency — MUST
Live-video latency shall be measured.

### BUD-PER-009 — Audio Latency — MUST
Telepresence audio latency shall be measured.

### BUD-PER-010 — Remote Latency — MUST
Remote-control latency shall be measured.

### BUD-PER-011 — Storage Growth — MUST
Database/log/media growth shall be measured.

### BUD-PER-012 — Battery Runtime — SHOULD
Battery runtime shall be measured when final power hardware permits meaningful measurement.

---

# 40. Reliability Requirements

### BUD-REL-001 — 30-Minute Test — MUST
Buddy shall undergo a 30-minute sustained test.

### BUD-REL-002 — One-Hour Test — MUST
Buddy shall undergo a one-hour sustained test.

### BUD-REL-003 — Multi-Hour Test — MUST
Buddy shall undergo multi-hour testing.

### BUD-REL-004 — Reconnection — MUST
Wi-Fi/app/internet recovery shall be tested.

### BUD-REL-005 — Camera Stability — MUST
Camera stability shall be tested.

### BUD-REL-006 — Microphone Stability — MUST
Microphone stability shall be tested.

### BUD-REL-007 — Memory Leak — MUST
Long-running memory behaviour shall be monitored.

### BUD-REL-008 — Temperature — MUST
Thermal behaviour shall be monitored.

### BUD-REL-009 — E-Stop Regression — MUST
Emergency stop shall be repeatedly regression-tested after movement-related changes.

---

# 41. Deployment Requirements

### BUD-DEP-001 — Automatic Startup — MUST
Final Buddy shall support automatic startup.

### BUD-DEP-002 — ROS Environment — MUST
Required ROS environment shall load correctly.

### BUD-DEP-003 — Configuration — MUST
Runtime configuration shall load predictably.

### BUD-DEP-004 — Models — MUST
Required AI models shall be initialized with failure handling.

### BUD-DEP-005 — Database — MUST
Database initialization/migration shall be controlled.

### BUD-DEP-006 — Permissions — MUST
Hardware/software permissions shall be configured appropriately.

### BUD-DEP-007 — Service Manager — MUST
systemd or a justified equivalent shall manage production startup/recovery.

### BUD-DEP-008 — Graceful Shutdown — MUST
Shutdown shall safely stop physical movement and close resources.

### BUD-DEP-009 — Crash Recovery — MUST
Critical services shall have an appropriate recovery strategy.

---

# 42. Research Evaluation Requirements

### BUD-RES-001 — Voice Evaluation — MUST

Evaluate:

- STT accuracy
- intent accuracy
- wake-word success
- latency

**Threshold TBD during benchmarking task.**

### BUD-RES-002 — Sound Localization Evaluation — MUST

Evaluate:

- direction accuracy
- confidence
- effect of distance
- effect of environmental noise
- limitations of independent USB microphones

**Threshold TBD during benchmarking task.**

### BUD-RES-003 — Vision Evaluation — MUST

Evaluate:

- face detection
- face recognition
- unknown-person behaviour
- expression-model performance
- tracking performance

**Threshold TBD during benchmarking task.**

### BUD-RES-004 — Movement Evaluation — MUST

Evaluate:

- command response
- calibration
- obstacle stopping
- safety response

### BUD-RES-005 — Navigation Evaluation — SHOULD

Where implemented, evaluate:

- target success
- collision avoidance
- patrol completion
- recovery performance

### BUD-RES-006 — Networking Evaluation — MUST

Evaluate:

- local command latency
- remote command latency
- video latency
- audio latency
- reconnection behaviour

### BUD-RES-007 — Pi Evaluation — MUST

Evaluate:

- CPU
- RAM
- temperature
- sustained runtime

### BUD-RES-008 — User Evaluation — SHOULD

If permitted by research procedures, evaluate:

- perceived responsiveness
- usability
- interaction quality

No experimental results are claimed by this Task 01 document.

---

# 43. NON-NEGOTIABLE SAFETY INVARIANTS

The following rules are permanent project constraints.

1. Emergency stop overrides everything.
2. No LLM can directly operate motors.
3. No mobile/network command can bypass deterministic validation.
4. Collision safety may reject movement from any source.
5. Stale movement commands expire.
6. Connection loss stops manual remote movement.
7. Startup shall not cause unexpected motion.
8. Shutdown/crash shall result in motor stop where technically possible.
9. Failed critical safety sensors cause restricted/degraded operation.
10. Servo mechanical limits shall be enforced.
11. Credentials shall never be committed to Git.
12. Unknown/unvalidated external commands shall be rejected.
13. Experimental AI failure shall not disable critical movement safety.
14. Autonomous behaviour shall not override emergency stop or collision protection.
15. Future modules shall integrate through defined interfaces rather than bypassing existing safety mechanisms.

---

# 44. Requirement Traceability

All requirements shall eventually follow:

**Requirement ID → Implementation Task → Module → Test → Result**

At Task 01:

| Field | Value |
|---|---|
| Implementation Task | TBD |
| Module | TBD |
| Test | TBD |
| Result | TBD |

Future tasks shall update traceability as implementation progresses.

No future task may silently change an established MUST requirement without documenting the architectural dependency, justification and required regression testing.
