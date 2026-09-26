# BUDDY v1.0 — System Acceptance Criteria

**Project:** BUDDY — Affordable AI-Powered Intelligent Robot Dog
**Platform:** Raspberry Pi 4B 4 GB / Ubuntu 24.04 / ROS 2 Jazzy
**Status:** Task 01 Acceptance Baseline

---

# 1. Purpose

This document defines system-level conditions that shall eventually be used to determine whether Buddy functionality is acceptable.

Task 01 defines criteria only.

It does not claim that these criteria have already passed.

Where research benchmarking is required:

**Threshold TBD during benchmarking task**

---

# 2. Startup

### AC-START-001

Buddy shall boot without causing unintended motor movement.

**Expected:** PASS/FAIL

### AC-START-002

Required core ROS 2 components shall initialize or report a controlled degraded/error state.

### AC-START-003

Hardware unavailable during startup shall be identified rather than causing an unexplained whole-system crash where graceful degradation is possible.

---

# 3. Voice

### AC-VOICE-001

Buddy shall detect the configured wake word:

**Buddy**

Performance threshold:

**TBD during benchmarking task**

### AC-VOICE-002

Buddy shall execute supported core commands without requiring internet connectivity.

### AC-VOICE-003

Offline STT performance shall be measured using representative commands and speakers.

Accuracy threshold:

**TBD during benchmarking task**

### AC-VOICE-004

Intent-classification accuracy shall be experimentally measured.

Threshold:

**TBD during benchmarking task**

### AC-VOICE-005

Buddy shall produce local TTS responses for supported core interactions.

### AC-VOICE-006

Speaker output shall not cause uncontrolled repeated self-triggering.

---

# 4. Microphones and Sound Direction

### AC-SND-001

Both USB microphones shall be independently detectable.

### AC-SND-002

Each microphone shall successfully capture audio independently.

### AC-SND-003

The system shall attempt simultaneous capture and measure synchronization characteristics.

### AC-SND-004

Sound-direction performance shall be tested at multiple physical positions.

### AC-SND-005

Direction accuracy shall be reported honestly.

Threshold:

**TBD during benchmarking task**

### AC-SND-006

The research report shall explicitly document front/back ambiguity and synchronization limitations if observed.

### AC-SND-007

If the hardware cannot provide reliable localization, Buddy shall degrade the feature rather than falsely report precise direction.

---

# 5. Camera

### AC-CAM-001

Raspberry Pi Camera V2 shall initialize successfully on target hardware.

### AC-CAM-002

The system shall detect/report camera failure.

### AC-CAM-003

Camera FPS shall be measured at selected operating resolutions.

Threshold:

**TBD during benchmarking task**

### AC-CAM-004

Camera processing shall remain stable during sustained testing.

---

# 6. Face Recognition

### AC-FACE-001

Buddy shall detect faces under representative supported conditions.

### AC-FACE-002

Buddy shall enroll a known user using the defined enrollment process.

### AC-FACE-003

Enrolled user identity shall persist after restart.

### AC-FACE-004

Buddy shall attempt recognition of enrolled users.

Accuracy threshold:

**TBD during benchmarking task**

### AC-FACE-005

Insufficient-confidence identities shall be allowed to remain unknown rather than being forcibly assigned to a known user.

### AC-FACE-006

Duplicate/cancelled enrollment shall be handled appropriately.

---

# 7. Facial Expression History

### AC-EMO-001

Buddy shall produce supported expression estimates.

### AC-EMO-002

Expression observations shall include sufficient confidence/quality handling.

### AC-EMO-003

Every camera frame shall not be stored as an independent emotional event.

### AC-EMO-004

Daily summaries shall be derivable from stored significant observations.

### AC-EMO-005

Weekly/monthly summaries shall be derivable when enough history exists.

### AC-EMO-006

The UI/reports shall not present facial-expression estimation as medical or psychological diagnosis.

---

# 8. Camera Pan/Tilt

### AC-HEAD-001

Camera shall move left/right within calibrated safe limits.

### AC-HEAD-002

Camera shall move up/down within calibrated safe limits.

### AC-HEAD-003

Camera shall return to a configured centre position.

### AC-HEAD-004

Software commands outside mechanical limits shall be rejected/clamped safely.

### AC-HEAD-005

Automatic tracking shall never intentionally command an unsafe servo position.

---

# 9. Motors

### AC-MOT-001

Each required motor/motor group shall be independently validated before full chassis movement.

### AC-MOT-002

Buddy shall move forward on a valid safe command.

### AC-MOT-003

Buddy shall reverse on a valid safe command.

### AC-MOT-004

Buddy shall turn left/right.

### AC-MOT-005

Buddy shall stop on command.

### AC-MOT-006

Bounded speed control shall function.

### AC-MOT-007

Invalid movement commands shall not cause arbitrary motor behaviour.

---

# 10. Emergency Stop

### AC-ESTOP-001

Emergency stop shall override manual control.

### AC-ESTOP-002

Emergency stop shall override autonomous movement.

### AC-ESTOP-003

Emergency stop shall override patrol.

### AC-ESTOP-004

Emergency stop shall override following.

### AC-ESTOP-005

Emergency stop shall override voice-triggered physical behaviour.

### AC-ESTOP-006

After E-stop activation:

**Intentional movement must remain disabled until explicit safe reset.**

This is a strict acceptance criterion.

### AC-ESTOP-007

E-stop shall be regression-tested after movement-related architectural changes.

---

# 11. Movement Failsafe

### AC-SAFE-001

Stale movement commands shall expire.

### AC-SAFE-002

Loss of active manual-control connection shall result in manual motion stopping.

### AC-SAFE-003

Safety validation shall be able to reject a command regardless of whether its source is:

- app
- voice
- autonomous navigation
- following
- patrol
- AI intent

### AC-SAFE-004

No LLM output shall directly reach motor GPIO/control hardware.

---

# 12. Ultrasonic Protection

### AC-USS-001

Front sensor shall provide valid readings under supported conditions.

### AC-USS-002

Rear sensor shall provide valid readings.

### AC-USS-003

Left sensor shall provide valid readings.

### AC-USS-004

Right sensor shall provide valid readings.

### AC-USS-005

Invalid readings shall be detectable.

### AC-USS-006

Sensor triggering shall avoid uncontrolled simultaneous cross-talk.

### AC-USS-007

Unsafe movement toward a detected obstacle shall be blocked according to the final calibrated safety policy.

Distance threshold:

**TBD during hardware calibration task**

### AC-USS-008

Failure of a critical obstacle sensor shall trigger an appropriately restricted movement mode.

---

# 13. Person Following

### AC-FOLLOW-001

Buddy shall follow only a selected/validated target.

### AC-FOLLOW-002

Obstacle protection shall remain active while following.

### AC-FOLLOW-003

Buddy shall not continue blindly when target confidence is lost.

### AC-FOLLOW-004

Following shall support cancellation.

### AC-FOLLOW-005

Unsuccessful target reacquisition shall timeout.

### AC-FOLLOW-006

Following distance performance shall be measured.

Threshold:

**TBD during benchmarking task**

---

# 14. Finger Tracking

### AC-FINGER-001

If implemented, Buddy shall detect a supported hand/index-finger target under representative conditions.

### AC-FINGER-002

Camera pan/tilt may track the target without exceeding mechanical limits.

### AC-FINGER-003

Body movement caused by finger tracking shall still pass obstacle/safety validation.

### AC-FINGER-004

Loss of target shall stop physical finger-follow behaviour.

---

# 15. Memory

### AC-MEM-001

SQLite database shall initialize successfully.

### AC-MEM-002

Required user data shall persist across restart.

### AC-MEM-003

Security-event metadata shall persist.

### AC-MEM-004

Expression-history data shall persist where configured.

### AC-MEM-005

Large media shall not cause uncontrolled database growth.

### AC-MEM-006

Database failure shall not create unsafe motor behaviour.

---

# 16. Object/Person Learning

### AC-LRN-001

Person enrollment shall support confirmation.

### AC-LRN-002

Person enrollment shall support cancellation.

### AC-LRN-003

Successful enrolled identity shall remain available after restart.

### AC-LRN-004

If object learning is implemented, stored labels/examples shall persist.

### AC-LRN-005

Object-learning documentation shall not claim universal one-shot object understanding.

---

# 17. Mapping

### AC-MAP-001

Before full SLAM is claimed, a hardware feasibility assessment shall be completed.

### AC-MAP-002

The assessment shall explicitly evaluate whether adequate pose/localization information exists.

### AC-MAP-003

If current hardware is inadequate, accurate SLAM shall remain experimental rather than being falsely presented as completed.

### AC-MAP-004

If mapping is implemented, map quality shall be experimentally evaluated.

Threshold:

**TBD during mapping benchmark task**

---

# 18. Restricted Zones

### AC-ZONE-001

If map-based restricted zones are implemented, a configured forbidden region shall be represented in the navigation system.

### AC-ZONE-002

Autonomous navigation shall not intentionally enter an active forbidden zone.

### AC-ZONE-003

Patrol shall respect forbidden zones.

### AC-ZONE-004

Following shall respect forbidden zones.

---

# 19. Patrol

### AC-PAT-001

Buddy shall start an accepted patrol configuration.

### AC-PAT-002

Patrol shall be stoppable.

### AC-PAT-003

Manual takeover shall suspend autonomous patrol movement safely.

### AC-PAT-004

Emergency stop shall immediately override patrol.

### AC-PAT-005

Patrol events/history shall be recordable.

### AC-PAT-006

Patrol completion/recovery performance shall be measured.

Threshold:

**TBD during benchmarking task**

---

# 20. Security Events

### AC-SEC-001

Unknown-person detection shall be capable of creating a timestamped event.

### AC-SEC-002

Security event shall support local image capture.

### AC-SEC-003

Event metadata shall persist locally.

### AC-SEC-004

Repeated observation of the same continuous unknown-person event shall not produce uncontrolled alert flooding.

### AC-SEC-005

When networking is available, an eligible event shall be capable of reaching the owner notification system.

### AC-SEC-006

Owner shall be able to enter supported Live Control from security workflow when implemented.

---

# 21. Recording

### AC-REC-001

Significant incident media shall be stored locally.

### AC-REC-002

Storage quotas shall prevent unlimited recording growth.

### AC-REC-003

Retention/cleanup shall remove data according to configured policy.

### AC-REC-004

Failure to upload media shall not immediately destroy the local event.

### AC-REC-005

If rolling recording is implemented, pre/post-trigger behaviour shall be tested.

---

# 22. Offline Behaviour

### AC-OFF-001

Disconnecting internet shall not disable emergency stop.

### AC-OFF-002

Disconnecting internet shall not disable local obstacle safety.

### AC-OFF-003

Disconnecting internet shall not disable core movement.

### AC-OFF-004

Core supported voice commands shall remain available offline.

### AC-OFF-005

Local face recognition shall remain available where the local model is loaded.

### AC-OFF-006

Local memory shall remain available.

### AC-OFF-007

Local recording shall remain available.

### AC-OFF-008

Same-network local control shall not depend on an external cloud service where technically practical.

---

# 23. Cloud Synchronization

### AC-CLOUD-001

Eligible offline-created events shall be able to remain in a persistent pending queue.

### AC-CLOUD-002

After internet recovery, pending events shall be retried.

### AC-CLOUD-003

Repeated retry shall not produce uncontrolled duplicate cloud items.

### AC-CLOUD-004

Failed uploads shall retain appropriate failure/pending state.

### AC-CLOUD-005

Cloud-service failure shall not crash core robot operation.

---

# 24. Mobile Local Control

### AC-APP-001

The app shall discover/connect to Buddy using the selected local architecture.

### AC-APP-002

Authorized local movement commands shall reach the deterministic control pipeline.

### AC-APP-003

App stop shall stop commanded manual movement.

### AC-APP-004

App E-stop shall invoke the highest-priority emergency-stop mechanism.

### AC-APP-005

Robot status shall update in the app.

### AC-APP-006

Local control shall recover appropriately after temporary connection loss.

---

# 25. Live Video

### AC-VIDEO-001

The app shall display Buddy's live camera when media connection is active.

### AC-VIDEO-002

End-to-end video latency shall be measured.

Threshold:

**TBD during media benchmarking task**

### AC-VIDEO-003

Video disconnection shall not affect movement safety.

### AC-VIDEO-004

Media reconnection shall be tested.

---

# 26. Telepresence Audio

### AC-AUDIO-001

Owner shall be able to hear Buddy's microphone when authorized telepresence is active.

### AC-AUDIO-002

Owner shall be able to send voice to Buddy's speaker using the selected communication mode.

### AC-AUDIO-003

Push-to-talk shall be tested if used.

### AC-AUDIO-004

Audio feedback/self-triggering behaviour shall be evaluated.

### AC-AUDIO-005

Audio latency shall be measured.

Threshold:

**TBD during media benchmarking task**

---

# 27. Internet Remote Control

### AC-REMOTE-001

Remote control shall require authenticated access.

### AC-REMOTE-002

Remote commands shall require authorization.

### AC-REMOTE-003

Remote movement shall pass the same deterministic safety pipeline as local movement.

### AC-REMOTE-004

Stale/replayed remote movement commands shall be rejected where applicable.

### AC-REMOTE-005

Loss of remote manual-control connection shall stop manual motion.

### AC-REMOTE-006

Raspberry Pi GPIO/motor control ports shall not be directly exposed to the public internet.

---

# 28. Authentication and Security

### AC-AUTH-001

Unauthorized users shall not receive movement authority.

### AC-AUTH-002

Sensitive communication shall use encrypted transport.

### AC-AUTH-003

Credentials shall not exist in committed Git source.

### AC-AUTH-004

Session expiration/termination shall be supported according to the final authentication architecture.

### AC-AUTH-005

Externally reachable interfaces shall validate incoming commands/data.

---

# 29. State Machine

### AC-STATE-001

Buddy shall reject invalid state transitions.

### AC-STATE-002

Emergency stop shall be reachable from every movement-capable operating mode.

### AC-STATE-003

Manual mode and autonomous movement shall not simultaneously possess uncontrolled movement authority.

### AC-STATE-004

ERROR state shall produce defined safe behaviour.

### AC-STATE-005

SHUTTING_DOWN shall prevent initiation of new unsafe physical actions.

---

# 30. Failure Handling

### AC-FAIL-001

Camera disconnection shall be detected.

### AC-FAIL-002

Loss of one microphone shall be detected where technically observable.

### AC-FAIL-003

Loss of one microphone shall disable/degrade localization rather than necessarily disabling all voice interaction.

### AC-FAIL-004

Internet loss shall be detected without crashing core functionality.

### AC-FAIL-005

LLM failure shall not disable deterministic core commands.

### AC-FAIL-006

Critical obstacle-sensor failure shall cause restricted/degraded movement.

### AC-FAIL-007

Database failure shall be logged and shall not produce uncontrolled physical movement.

---

# 31. Logging

### AC-LOG-001

Startup/shutdown shall produce appropriate logs.

### AC-LOG-002

Movement/safety events shall be loggable.

### AC-LOG-003

Security events shall be loggable.

### AC-LOG-004

Errors shall be loggable.

### AC-LOG-005

Logs shall rotate.

### AC-LOG-006

Logs shall not grow without bound.

### AC-LOG-007

Passwords/tokens shall not be written into normal logs.

---

# 32. Performance

### AC-PERF-001

CPU utilization shall be measured during representative operation.

### AC-PERF-002

RAM utilization shall be measured.

### AC-PERF-003

Raspberry Pi temperature shall be measured.

### AC-PERF-004

Camera FPS shall be measured.

### AC-PERF-005

Vision inference latency shall be measured.

### AC-PERF-006

Voice latency shall be measured.

### AC-PERF-007

Command-to-movement latency shall be measured.

### AC-PERF-008

Video/audio latency shall be measured.

Final thresholds:

**TBD during subsystem benchmarking tasks**

---

# 33. Long-Duration Reliability

### AC-REL-001

Buddy shall undergo a continuous 30-minute test.

### AC-REL-002

Buddy shall undergo a continuous one-hour test.

### AC-REL-003

Buddy shall undergo multi-hour testing.

### AC-REL-004

During sustained tests monitor:

- CPU
- RAM
- temperature
- camera
- microphones
- ROS nodes
- database
- networking
- logs

### AC-REL-005

No uncontrolled movement shall occur following software/network failure.

### AC-REL-006

Memory growth shall be investigated for leaks.

### AC-REL-007

Wi-Fi/application reconnection shall be tested.

### AC-REL-008

Internet loss/recovery shall be tested.

---

# 34. Deployment

### AC-DEP-001

Final Buddy shall automatically initialize required production services after boot.

### AC-DEP-002

Startup shall leave motors stopped until valid movement authority exists.

### AC-DEP-003

Required configuration shall load successfully or produce a controlled error.

### AC-DEP-004

Required models/database shall initialize successfully or cause controlled degradation.

### AC-DEP-005

Graceful shutdown shall stop physical movement.

### AC-DEP-006

Configured critical services shall recover appropriately from recoverable crashes.

---

# 35. Research Acceptance

The final research evaluation shall report measured performance rather than fabricated claims.

Required evaluation areas include:

- STT
- wake word
- intent classification
- sound localization
- face recognition
- expression estimation
- tracking
- obstacle safety
- movement
- navigation where implemented
- patrol where implemented
- local networking
- remote networking
- live video
- telepresence audio
- CPU
- RAM
- temperature
- sustained runtime

Where a feature fails to achieve the desired capability, the result and limitation shall be documented.

A negative experimental result is acceptable.

A fabricated successful result is not.

---

# 36. Task 01 Traceability Baseline

Each system requirement shall eventually map to:

| Requirement ID | Implementation Task | Module | Test | Result |
|---|---|---|---|---|
| Defined in requirements.md | TBD | TBD | TBD | TBD |

Task 01 does not assign implementation modules/tests prematurely.

---

# 37. Task 01 Documentation Acceptance

Task 01 passes when:

1. `docs/requirements.md` exists and contains the authoritative requirements.
2. `docs/scope.md` exists and defines scope boundaries.
3. `docs/constraints.md` exists and documents engineering constraints.
4. `docs/acceptance_criteria.md` exists.
5. Requirements use unique IDs.
6. Requirement priorities are defined.
7. Experimental features are identified.
8. Raspberry Pi limitations are acknowledged.
9. Two-microphone localization limitations are acknowledged.
10. Mapping hardware limitations are acknowledged.
11. ROS 2 Jazzy is established as the robotics middleware.
12. Offline-first operation is required.
13. Safety invariants are established.
14. LLM direct motor control is prohibited.
15. Remote movement cannot bypass safety.
16. Traceability structure exists.
17. Existing Buddy source has not been rewritten by Task 01.
18. No implementation code has been created by Task 01.
19. No experimental results have been fabricated.

After verification, Task 01 may be committed using:

**docs: freeze Buddy v1.0 requirements and scope**
