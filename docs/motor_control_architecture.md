# BUDDY Motor Control Architecture

## Task

TASK 10 — Safe Four-Wheel Motor Control Foundation

## Purpose

Task 10 establishes the software foundation required to safely control BUDDY's four DC gear motors through two TB6612FNG dual H-bridge motor drivers.

Physical motor hardware remains disabled by default.

## Architecture

```text
ROS 2 / Future Navigation Nodes
            |
            v
       BuddyDrive
            |
    +-------+-------+
    |               |
    v               v
TB6612 Driver 1  TB6612 Driver 2
    |               |
 +--+--+          +--+--+
 |     |          |     |
FL    FR          RL    RR
```

## Software Layers

```text
Central Configuration
        |
        v
Task 09 GPIO Factory
        |
        v
Motor System Factory
        |
        +----------------------+
        |                      |
        v                      v
TB6612Standby             BuddyDrive
                               |
                     +---------+---------+
                     |    |         |    |
                     v    v         v    v
                    FL   FR        RL   RR
```

## Safety Model

Motor hardware is disabled by default:

```yaml
hardware:
  enabled: false
  simulation: true

  motors:
    enabled: false
```

Physical GPIO pin assignments remain null until the wiring design has been reviewed and approved.

Creating the motor subsystem does not automatically activate physical motors.

Each TB6612FNG driver has explicit STBY control.

STBY initializes LOW so the motor driver starts disabled.

Drivers must be explicitly enabled before physical motor activation.

Cleanup stops all motors and returns both driver boards to the disabled state.

## Motor Commands

Normalized motor speed:

```text
-1.0 = full reverse
 0.0 = stopped
+1.0 = full forward
```

The software supports:

- Forward movement
- Reverse movement
- Pivot left
- Pivot right
- Independent wheel speeds
- Safe stop
- Safe cleanup

## Drive Layout

```text
              FRONT

     Front Left     Front Right
          FL             FR

          |               |
          |     BUDDY     |
          |               |

     Rear Left      Rear Right
          RL             RR

               REAR
```

## Driver Allocation

TB6612FNG Driver 1:

```text
Channel A/B -> Front Left / Front Right
```

TB6612FNG Driver 2:

```text
Channel A/B -> Rear Left / Rear Right
```

The final A/B assignment will be confirmed during physical wiring.

## Current Hardware State

No physical GPIO assignments have been activated.

No motor-driver wiring is assumed by the software configuration.

Motor control remains in safe simulation mode until physical wiring, power distribution, common grounding, GPIO selection, motor polarity, and driver connections have been verified.

## Verification

Task 10 was verified on the BUDDY Raspberry Pi ROS 2 environment.

Results:

```text
Full regression suite: 92 passed
Motor tests:           40 passed
ROS 2 packages built:   3
Configuration smoke:   PASS
Logging smoke:         PASS
Core heartbeat:        PASS
```

## Physical Activation

Physical activation must only occur after:

1. Raspberry Pi GPIO assignments are finalized.
2. Both TB6612FNG boards are identified and oriented.
3. Four motor terminal pairs are identified.
4. Motor power voltage is verified.
5. Logic power is verified.
6. Common ground is established.
7. STBY wiring is verified.
8. Battery/BMS/fuse/power path is verified.
9. Wheels are lifted from the ground for first testing.
10. One motor is tested at a time before enabling the complete 4WD base.
