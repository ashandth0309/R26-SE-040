# BUDDY Hardware Architecture

## Task

TASK 08 — Hardware and Electrical Architecture

## Purpose

This document defines the preliminary hardware integration architecture for the BUDDY intelligent robot dog.

Task 08 establishes system boundaries and safety constraints before hardware-control software is introduced.

---

## Compute

BUDDY uses:

Raspberry Pi 4 Model B — 4 GB

The Raspberry Pi acts as the central robot computer for future:

- ROS 2 nodes
- voice processing
- vision processing
- movement coordination
- navigation
- safety supervision
- networking
- mobile communication

The Raspberry Pi power supply must be stable and electrically isolated from high-current actuator disturbances through appropriate power architecture.

---

## Vision

BUDDY includes a camera intended for:

- face recognition
- emotion observation
- security/event capture
- possible live mobile viewing

Exact model:

TBD — physically verify

Interface:

TBD — USB or CSI

Task 08 does not activate or implement camera functionality.

---

## Audio

BUDDY currently has two USB microphones.

Primary intended functions:

- voice capture
- speech-recognition input
- wake-word input
- experimental sound-direction estimation

Interface:

USB

Because the microphones use USB audio interfaces, they do not require GPIO logic-level conversion.

Microphone placement, synchronization, calibration, and localization algorithms belong to later tasks.

---

## Obstacle Detection

BUDDY currently has four obstacle sensors.

Planned logical locations:

- front
- rear
- left
- right

Exact sensor model:

TBD — physically verify

Required electrical information:

- supply voltage
- output voltage
- interface
- timing
- current consumption

GPIO assignments are not finalized until this information is verified.

---

## Actuation

BUDDY requires multiple actuators for robotic movement.

Exact actuator model:

TBD

Exact quantity:

TBD

Required verification:

- rated voltage
- control signal
- no-load current
- operating current
- stall current
- simultaneous peak demand

Actuator power must be separated from Raspberry Pi GPIO.

No actuator-control software is created during Task 08.

---

## Power Conversion

BUDDY currently has an LM2596 adjustable DC-DC buck converter.

Its role is voltage step-down.

Its eventual use depends on:

- battery voltage
- required output voltage
- required current
- thermal capability
- output stability
- ripple performance

Task 08 does not declare the LM2596 suitable for the Raspberry Pi or actuator rail until those requirements are verified.

---

## Logic-Level Conversion

BUDDY has a 4-channel logic-level converter.

Its intended purpose is signal-level voltage translation.

Example:

5 V logic
↔
3.3 V logic

It must not be used for:

- battery voltage conversion
- Raspberry Pi power regulation
- servo power
- motor power

Use is determined by the actual signal-interface requirements.

---

## USB Architecture

Known USB-connected hardware includes:

- microphone 1
- microphone 2
- USB hub

The camera may also be USB depending on its final verified model.

The USB hub is currently:

TBD — powered or bus-powered

The total USB peripheral power budget must be verified before relying on Raspberry Pi USB power for all peripherals simultaneously.

---

## Conceptual System Architecture

BUDDY is conceptually divided into:

### Compute Domain

Raspberry Pi 4B

### Vision Domain

Camera

### Audio Domain

Two USB microphones

### Obstacle Detection Domain

Four distance/obstacle sensors

### Actuation Domain

Multiple servos/actuators

### Power Domain

Battery
→ protection
→ regulated compute supply
→ actuator supply
→ accessory supply where required

### Signal Interface Domain

Raspberry Pi 3.3 V GPIO
↔
verified peripheral interfaces

with logic-level conversion only when required.

---

## Grounding Strategy

Subsystems exchanging logic signals generally require a shared electrical reference.

However:

- high-current actuator current must not be routed through Raspberry Pi GPIO wiring
- actuator return paths must be suitable for their current
- logic-reference grounding and high-current routing must be planned separately

---

## Hardware Safety

Before connecting any new component:

1. identify the exact component
2. verify nominal supply voltage
3. verify maximum voltage
4. verify logic voltage
5. verify current requirement
6. verify peak/stall current where applicable
7. verify connector pinout
8. verify polarity
9. verify grounding requirements
10. verify regulator capability
11. verify GPIO compatibility

Do not connect hardware based only on approximate voltage similarity.

---

## Electrical Noise and Brownout Risk

Multiple actuators can cause sudden current demand.

Possible consequences include:

- voltage sag
- Raspberry Pi undervoltage
- Raspberry Pi resets
- USB instability
- sensor errors
- electrical noise
- storage corruption

Future implementation must validate supply stability under realistic actuator loading.

---

## Unknown / Pending Specifications

The following remain unresolved:

- battery chemistry
- battery cell count
- battery voltage range
- battery capacity
- battery discharge rating
- battery BMS/protection status
- actuator model
- actuator quantity
- actuator voltage
- actuator stall current
- obstacle sensor model
- camera model/interface
- USB hub power configuration
- LM2596 practical safe current for the intended use
- dedicated actuator controller availability/model

These unknowns must not be converted into assumed specifications.

---

## Hardware Activation Block

The software architecture may continue to be developed.

However, hardware activation involving unverified electrical interfaces is blocked until the relevant component information is confirmed.

This particularly applies to:

- battery-to-regulator connections
- Raspberry Pi power wiring
- actuator power
- obstacle sensor GPIO wiring
- 5 V to 3.3 V signal interfaces

---

## Task Boundary

Task 08 performs:

- hardware inventory
- electrical architecture
- power planning
- GPIO planning
- integration-risk documentation

Task 08 does not perform:

- GPIO activation
- motor operation
- servo operation
- obstacle sensor measurement
- camera implementation
- microphone implementation
- navigation implementation
- movement implementation

No hardware driver implementation occurred.

---

## Result

TASK 08 ARCHITECTURE PASS — HARDWARE ACTIVATION BLOCKED

Critical physical specifications must be verified before the relevant hardware implementation task begins.
