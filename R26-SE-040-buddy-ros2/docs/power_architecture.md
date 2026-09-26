# BUDDY Power Architecture

## Task

TASK 08 — Hardware and Electrical Architecture

## Purpose

This document defines the preliminary electrical power architecture for the BUDDY robot dog.

The purpose is to prevent unsafe assumptions before the battery, actuators, regulators, sensors, and peripheral power requirements are physically verified.

Unknown electrical specifications are deliberately marked as TBD.

---

# Power Domains

BUDDY must be treated as having multiple electrical power domains.

Conceptual architecture:

Battery
│
├── Protection / Fuse / Master Switch
│
├── Actuator Power Rail
│   └── Servos / Motors
│
├── Regulated Compute Power Rail
│   └── Raspberry Pi 4 Model B
│
└── Accessory Power Rail
    ├── Sensors
    ├── Camera
    └── Other peripherals as required

The final electrical wiring must not be implemented until the required voltage and current specifications are verified.

---

# Battery

The BUDDY robot uses a removable rechargeable battery.

The following specifications are currently TBD and must be physically verified:

- battery chemistry
- cell count
- nominal voltage
- maximum fully charged voltage
- capacity
- continuous discharge rating
- peak discharge capability
- connector type
- connector polarity
- integrated protection/BMS status

No battery voltage or chemistry is assumed in this architecture.

---

# Battery Protection and BMS

Charging protection and operating/discharge protection are separate concerns.

Removing the battery from BUDDY and charging it using a separate charger does not automatically guarantee that the battery is protected while powering the robot.

If the battery pack already contains an integrated protection circuit or BMS, an additional external BMS may not be required.

If the battery is an unprotected multi-cell lithium battery pack, suitable pack-level protection may be required.

The final BMS requirement cannot be determined until the battery chemistry, cell configuration, and existing protection circuitry are physically verified.

Current status:

BMS REQUIREMENT — TBD

---

# Master Protection

The future BUDDY power system should consider:

- master power switch
- appropriately rated fuse or resettable protection
- correct connector polarity
- reverse-polarity awareness
- insulated electrical connections
- suitable wire gauge
- strain relief
- battery protection appropriate to the battery type

Fuse rating must not be guessed.

Current fuse rating:

TBD — calculate after maximum current demand is known.

---

# Raspberry Pi Power

The Raspberry Pi 4 Model B requires a stable regulated power supply.

The Raspberry Pi must not be connected directly to an unverified battery output.

Conceptual path:

Battery
→ Protection
→ DC-DC Regulation
→ Stable Raspberry Pi Supply
→ Raspberry Pi 4B

The final regulator selection and wiring depend on:

- battery voltage range
- Raspberry Pi load
- USB peripheral load
- regulator current capability
- regulator thermal performance
- voltage stability
- electrical noise

The Raspberry Pi power rail must be protected from voltage sag caused by high-current actuators.

---

# LM2596 Buck Converter

BUDDY currently has an LM2596 adjustable DC-DC buck converter.

Its purpose is DC voltage step-down.

Conceptually:

Higher DC Voltage
→ LM2596
→ Lower Regulated DC Voltage

The LM2596 is not a logic-level converter.

It is not a motor controller.

It is not automatically considered suitable for powering the Raspberry Pi merely because its output can be adjusted to the required voltage.

Before using the module, verify:

- input-voltage range
- output-voltage setting
- practical continuous-current capability
- thermal performance
- voltage stability
- ripple under load

The regulator output must be measured before connecting sensitive electronics.

Current planned role:

TBD pending battery and load verification.

---

# Actuator Power

Servos and other high-current actuators must not be powered from Raspberry Pi GPIO pins.

The actuator power architecture must remain separate from GPIO signal control.

Conceptual architecture:

Battery
→ Protection
→ Suitable Actuator Power Rail
→ Servos / Actuators

The final architecture depends on:

- actuator model
- actuator quantity
- rated voltage
- normal operating current
- startup current
- stall current
- simultaneous actuator demand

These specifications are currently TBD.

No final actuator regulator or power supply is selected by Task 08.

---

# Logic-Level Converter

BUDDY has a 4-channel logic-level converter.

Its purpose is logic signal voltage translation where required.

Example:

5 V logic
↔
3.3 V logic

The logic-level converter must NOT be used as:

- a battery regulator
- Raspberry Pi power supply
- servo power supply
- motor power supply

It handles signal-level translation only.

It should only be inserted where the actual device electrical interface requires voltage translation.

---

# Raspberry Pi GPIO Electrical Safety

Raspberry Pi GPIO uses 3.3 V logic.

Important rules:

- Never intentionally apply 5 V directly to a Raspberry Pi GPIO input.
- GPIO pins are for control/data signals, not motor power.
- Motors and servos must not draw operating power through GPIO.
- Sensor output voltage must be verified before connection.
- GPIO assignments must not be finalized before interface verification.

---

# Obstacle Sensor Power

BUDDY currently has four obstacle sensors intended for:

- front
- rear
- left
- right

Their exact model and electrical characteristics remain TBD.

The following must be verified:

- supply voltage
- logic/output voltage
- current requirement
- interface type

If the sensors are a type that produces a 5 V logic output, that output must not be connected directly to Raspberry Pi GPIO.

Suitable level conversion or voltage division would be required.

This requirement is conditional on the actual sensor model.

---

# Camera Power

The camera is intended for:

- face recognition
- emotion observation
- security capture
- possible live viewing

The exact camera model and interface remain TBD.

If USB:

USB power consumption must be included in the Raspberry Pi/USB power budget.

If CSI:

the appropriate Raspberry Pi camera interface and power characteristics must be followed.

No camera implementation occurs in Task 08.

---

# USB Microphones

BUDDY currently uses two USB microphones.

Their intended functions include:

- voice capture
- speech-recognition input
- experimental sound-direction estimation

They receive power through their USB interface.

Their combined USB power consumption must be considered when calculating the Raspberry Pi peripheral power budget.

---

# USB Hub

The USB hub power configuration is currently:

TBD — powered or bus-powered must be physically verified.

The combined power demand of:

- microphone 1
- microphone 2
- camera if USB
- future USB peripherals

must be checked.

The Raspberry Pi USB subsystem must not automatically be assumed capable of powering every peripheral simultaneously.

---

# Common Ground Strategy

Electrical subsystems exchanging logic signals normally require a common electrical reference.

Conceptually:

Raspberry Pi Ground
↔
Sensor / Controller Ground

However, a common ground does not mean high-current actuator current should be routed through Raspberry Pi wiring.

High-current actuator return paths must be designed appropriately.

Signal grounding and high-current power routing must be considered separately.

---

# Preliminary Power Budget

| Load | Voltage | Typical Current | Peak/Stall Current | Planned Source | Status |
|---|---|---:|---:|---|---|
| Raspberry Pi 4B | Regulated supply required | TBD | TBD | Compute rail | Pending verification |
| USB hub | TBD | TBD | TBD | USB / external — TBD | Pending verification |
| Camera | TBD | TBD | TBD | USB/CSI — TBD | Pending verification |
| USB microphone 1 | USB | TBD | TBD | USB | Pending measurement/specification |
| USB microphone 2 | USB | TBD | TBD | USB | Pending measurement/specification |
| Front obstacle sensor | TBD | TBD | TBD | Accessory rail | Model TBD |
| Rear obstacle sensor | TBD | TBD | TBD | Accessory rail | Model TBD |
| Left obstacle sensor | TBD | TBD | TBD | Accessory rail | Model TBD |
| Right obstacle sensor | TBD | TBD | TBD | Accessory rail | Model TBD |
| Servos / actuators | TBD | TBD | TBD | Actuator rail | Critical specification pending |

Because several major loads remain unknown, no total-current figure is calculated.

POWER CAPACITY VERIFICATION PENDING.

---

# Brownout and Electrical Noise Risks

High-current actuators can cause:

- sudden current demand
- battery voltage sag
- regulator voltage drop
- electrical noise
- Raspberry Pi undervoltage
- Raspberry Pi reset
- USB instability
- sensor errors
- storage corruption

The Raspberry Pi power rail must therefore be designed so actuator activity does not cause unacceptable compute-supply instability.

Future testing should monitor Raspberry Pi undervoltage/throttling status using appropriate diagnostics such as:

`vcgencmd get_throttled`

Supply voltage should also be measured under realistic load.

No monitoring node is implemented in Task 08.

---

# Voltage Verification Procedure

Before connecting sensitive electronics to an adjustable regulator:

1. Disconnect the load.
2. Verify regulator input requirements.
3. Verify polarity.
4. Apply power safely.
5. Measure the regulator output using a multimeter.
6. Adjust the output to the required voltage.
7. Measure again.
8. Verify output stability.
9. Switch power off.
10. Connect the intended load only after the output has been verified.

Do not connect the Raspberry Pi first and adjust the regulator afterward.

---

# Current Verification Procedure

Before actuator implementation:

1. Identify the exact actuator model.
2. Obtain manufacturer specifications where available.
3. Determine rated voltage.
4. Determine normal operating current.
5. Determine peak/stall current.
6. Determine the number of actuators that may operate simultaneously.
7. Calculate realistic peak demand.
8. Select the power system and wiring accordingly.
9. Validate the system under controlled load.

The actuator supply must not be sized using idle current alone.

---

# Future Electrical Bring-Up Order

Hardware should later be activated incrementally.

## Stage 1

Verify battery and protection architecture without connecting the Raspberry Pi.

## Stage 2

Verify regulated outputs using a multimeter.

## Stage 3

Power the Raspberry Pi without actuators.

## Stage 4

Connect verified low-current sensors.

## Stage 5

Connect verified control/interface hardware.

## Stage 6

Test one actuator under controlled conditions.

## Stage 7

Increase actuator load gradually.

## Stage 8

Perform full-load power and brownout testing.

Task 08 documents this procedure only.

It does not execute these stages.

---

# Current Power Architecture Result

The conceptual power architecture is defined.

However, the following critical specifications remain unresolved:

- battery electrical specifications
- battery protection/BMS status
- actuator voltage/current requirements
- obstacle sensor electrical specifications
- USB hub power configuration
- camera interface/power requirements
- LM2596 practical suitability for the intended rail

Therefore:

POWER CAPACITY VERIFICATION PENDING

and

HARDWARE ACTIVATION REMAINS BLOCKED

until the relevant electrical specifications are verified.
