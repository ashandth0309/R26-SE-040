# BUDDY GPIO Plan

## Task

TASK 08 — Hardware and Electrical Architecture

## Purpose

This document defines a reservation-based GPIO planning strategy for the BUDDY robot dog.

The purpose is to prevent future pin conflicts and unsafe electrical connections before the exact hardware interfaces are verified.

Task 08 does not finalize GPIO assignments.

---

## Raspberry Pi GPIO Safety

The Raspberry Pi uses 3.3 V GPIO logic.

Important rules:

- Never connect an unverified 5 V signal directly to a Raspberry Pi GPIO input.
- GPIO pins are intended for logic/control signals only.
- Motors and servos must not receive operating power through GPIO.
- Sensor output voltage must be verified before connection.
- Physical header pin numbers and BCM GPIO numbers must not be confused.
- Final GPIO assignments require hardware-interface verification.

---

## Numbering Convention

Future BUDDY software should clearly distinguish between:

- physical Raspberry Pi header pin number
- BCM GPIO number

Documentation and source code must state which numbering scheme is used.

Do not mix numbering schemes within a subsystem.

---

## Current GPIO Reservation Plan

| Function | Interface | Raspberry Pi Pin / GPIO | Voltage | Status | Notes |
|---|---|---|---|---|---|
| Front obstacle sensor trigger | TBD | TBD | TBD | Reserved conceptually | Exact sensor model required |
| Front obstacle sensor return/echo | TBD | TBD | TBD | Reserved conceptually | Verify output voltage before GPIO connection |
| Rear obstacle sensor trigger | TBD | TBD | TBD | Reserved conceptually | Exact sensor model required |
| Rear obstacle sensor return/echo | TBD | TBD | TBD | Reserved conceptually | Verify output voltage before GPIO connection |
| Left obstacle sensor trigger | TBD | TBD | TBD | Reserved conceptually | Exact sensor model required |
| Left obstacle sensor return/echo | TBD | TBD | TBD | Reserved conceptually | Verify output voltage before GPIO connection |
| Right obstacle sensor trigger | TBD | TBD | TBD | Reserved conceptually | Exact sensor model required |
| Right obstacle sensor return/echo | TBD | TBD | TBD | Reserved conceptually | Verify output voltage before GPIO connection |
| Future actuator controller SDA | I2C only if supported | TBD | 3.3 V logic | Reserved only | Do not assume actuator board exists |
| Future actuator controller SCL | I2C only if supported | TBD | 3.3 V logic | Reserved only | Do not assume actuator board exists |
| Emergency-stop input | Digital input | TBD | 3.3 V maximum GPIO input | Future reservation | Hardware design TBD |
| Future status LED | Digital output | TBD | 3.3 V GPIO logic | Future reservation | Current limiting required if implemented |
| UART TX | UART | TBD | 3.3 V | Reserved bus | Only if future device requires UART |
| UART RX | UART | TBD | 3.3 V | Reserved bus | Never accept unsafe input voltage |
| SPI bus | SPI | TBD | 3.3 V | Reserved bus | Assign only if a future device requires SPI |

No physical GPIO numbers are finalized during Task 08.

---

## Obstacle Sensor Planning

BUDDY currently has four obstacle sensors intended for:

- front
- rear
- left
- right

The exact sensor model remains unverified.

Therefore:

- trigger pin requirement is TBD
- return/echo pin requirement is TBD
- supply voltage is TBD
- logic voltage is TBD
- timing requirements are TBD

If the sensor is a device such as an ultrasonic module whose output exceeds Raspberry Pi GPIO voltage limits, appropriate voltage translation or division must be used.

This statement is conditional on the actual sensor model.

---

## Logic-Level Conversion

BUDDY has a 4-channel logic-level converter.

Its possible future use is translation between different digital logic voltage domains.

Example:

5 V device logic
↔
3.3 V Raspberry Pi logic

The converter must only be used when compatible with the actual signal type and direction.

It is not a substitute for:

- motor drivers
- servo power supplies
- voltage regulators
- battery protection

---

## Reserved Communication Buses

Where appropriate, future BUDDY hardware may use shared Raspberry Pi buses rather than consuming individual GPIO pins unnecessarily.

Potential buses include:

### I2C

Useful for devices that explicitly support I2C.

Possible future uses may include:

- PWM/servo controller
- sensors
- monitoring hardware

No device is assigned to I2C until its hardware capability is verified.

### SPI

Reserved for future hardware that explicitly requires SPI.

No current Task 08 device is assigned to SPI.

### UART

Reserved for future serial devices if required.

Any UART-connected device must use Raspberry Pi-compatible logic levels.

---

## Pin Conflict Prevention

Before assigning a GPIO pin in a future task, verify whether it is used or reserved for:

- I2C
- SPI
- UART
- PWM
- camera/display interfaces
- boot-sensitive behaviour
- system functions
- another BUDDY subsystem

The GPIO plan must remain centralized rather than allowing individual subsystem tasks to select pins independently.

---

## PWM / Actuator Planning

BUDDY's actuator architecture is not finalized.

Before assigning PWM pins, verify:

- actuator type
- number of channels
- actuator controller availability
- whether direct GPIO PWM is appropriate
- whether a dedicated PWM controller is required
- control logic voltage
- timing requirements

A dedicated multi-channel PWM controller may be preferable for multiple servos, but Task 08 does not claim that such a board is currently available.

---

## Camera

The BUDDY camera may use:

- USB
- CSI

depending on the actual camera model.

Neither interface requires arbitrary GPIO assignment.

The exact camera interface remains TBD.

---

## USB Microphones

The two microphones use USB.

They therefore do not require Raspberry Pi GPIO pins or logic-level conversion.

---

## Emergency Stop Planning

A future physical emergency-stop input may be added.

Its electrical implementation is currently TBD.

Before assigning a GPIO:

- determine switch/contact type
- determine pull-up/pull-down strategy
- verify fail-safe behaviour
- verify logic voltage
- determine whether hardware-level motor power interruption is also required

A software GPIO emergency-stop signal must not automatically be treated as equivalent to a hardware power-disconnect safety system.

---

## GPIO Activation Rule

A GPIO assignment may move from TBD to final only when:

1. the connected hardware is physically identified
2. its interface is verified
3. its voltage level is verified
4. its pinout is verified
5. conflicting Raspberry Pi functions are checked
6. the centralized GPIO plan is updated
7. the implementation task explicitly authorizes activation

---

## Current Result

GPIO functions have been reserved conceptually.

No final Raspberry Pi GPIO numbers are assigned.

No GPIO output/input hardware was activated.

GPIO IMPLEMENTATION BLOCKED UNTIL HARDWARE INTERFACES ARE VERIFIED.
