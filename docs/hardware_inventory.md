# BUDDY Hardware Inventory

## Task

TASK 08 — Hardware and Electrical Architecture

## Purpose

This document records the currently known BUDDY hardware and identifies specifications that must be physically verified before hardware activation.

Unknown electrical values are deliberately marked as TBD rather than guessed.

---

## Current Hardware Inventory

| Component | Qty | Model | Function | Supply | Logic | Interface | Current | Verification |
|---|---:|---|---|---|---|---|---|---|
| Raspberry Pi | 1 | Raspberry Pi 4 Model B, 4 GB | Main robot computer | 5 V regulated input required | 3.3 V GPIO logic | GPIO / USB / CSI / network | TBD system load | Verified project hardware |
| Robot chassis / body | 1 | BUDDY dog robot chassis | Mechanical structure | N/A | N/A | Mechanical | N/A | Present |
| Obstacle sensors | 4 | TBD — physically verify | Front/rear/left/right obstacle detection | TBD | TBD | TBD | TBD | Quantity known; model unverified |
| Camera | 1 | TBD — physically verify | Face recognition, emotion observation, security capture, possible live view | TBD | N/A | USB or CSI — TBD | TBD | Model/interface unverified |
| USB microphones | 2 | USB lavalier microphones | Voice capture and experimental sound direction | USB power | USB digital interface | USB | TBD | Present |
| USB hub | 1 | TBD | Connect multiple USB peripherals | TBD | USB | USB | TBD | Power type unverified |
| Buck converter | 1 | LM2596 adjustable DC-DC buck converter | DC voltage step-down / regulation | Input TBD | N/A | Power | Safe current TBD | Present |
| Logic-level converter | 1 | 4-channel logic-level converter | Translate logic voltage levels where required | TBD | 3.3 V / 5 V logic use case | Digital signals | Signal-level only | Present |
| Rechargeable battery | 1 | TBD — physically verify | Main portable energy source | TBD | N/A | Power | Capacity/discharge rating TBD | Present; electrical specifications unverified |
| External charger | 1 | TBD | Charge removable battery | TBD | N/A | Battery charging | TBD | Present |
| Servo motors / actuators | Multiple | TBD — physically verify | Robot leg/body movement | TBD | Control interface TBD | PWM / controller TBD | Stall current TBD | Present/expected; exact specification required |
| Actuator control board | TBD | TBD | Possible centralized servo control | TBD | TBD | TBD | TBD | Do not assume present |

---

## Confirmed Logical Sensor Placement

BUDDY currently has four obstacle sensors available.

Planned logical positions:

- front
- rear
- left
- right

Physical wiring and GPIO assignments remain TBD until the exact sensor model and electrical interface are verified.

---

## Camera Intended Roles

The camera is intended for:

- face recognition
- emotion observation
- security/event capture
- possible mobile live viewing

The exact camera model and USB/CSI interface must be verified before implementation.

---

## Microphone Intended Roles

Two USB microphones are intended for:

- voice capture
- speech recognition input
- experimental sound-direction estimation

Because they use USB audio interfaces, they do not require Raspberry Pi GPIO logic-level conversion.

Accurate 360-degree sound localization is not assumed.

Placement and calibration belong to a later task.

---

## Information Required Before Hardware Activation

The following information must be physically verified before any hardware implementation task that depends on it:

### Battery

- chemistry
- cell count
- nominal voltage
- fully charged voltage
- capacity
- continuous discharge rating
- connector type
- connector polarity
- integrated protection/BMS status

### Actuators

- exact servo/actuator model
- quantity
- rated voltage
- control signal type
- no-load current
- operating current
- stall current
- simultaneous peak-current requirement

### Obstacle Sensors

- exact model
- supply voltage
- output/logic voltage
- interface type
- current requirement

### Camera

- exact model
- USB or CSI interface
- supply/current requirement if USB

### USB Hub

- powered or bus-powered
- input rating if powered
- supported current budget

### LM2596 Module

- input-voltage range
- practical safe output current
- cooling/thermal capability
- output-voltage setting before use

### Logic-Level Converter

- board type
- supported voltage domains
- directionality
- suitability for the intended signals

### Actuator Controller

If a separate PWM controller, PCA9685, servo board, H-bridge, or motor driver is present:

- exact model
- supply voltage
- logic voltage
- current capability
- interface

If none is physically present, it must not be recorded as existing hardware.

---

## Hardware Activation Status

Hardware activation is currently blocked for any subsystem whose voltage/current/interface requirements remain unverified.

No electrical specification in this document should be treated as confirmed unless explicitly marked verified.

---

## Task Boundary

Task 08 documents hardware architecture only.

No motors, servos, GPIO outputs, sensors, camera, or microphone hardware were activated as part of creating this inventory.
