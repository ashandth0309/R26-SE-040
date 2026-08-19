# BUDDY Hardware Abstraction Architecture

## Task

TASK 09 --- Hardware Activation Foundation and Safe Hardware Abstraction
Layer

## Purpose

Task 09 establishes a reusable and safe software layer between future
BUDDY ROS 2 nodes and physical Raspberry Pi GPIO hardware.

The goal is to prevent future subsystems from accessing Raspberry Pi
GPIO directly and independently.

Instead, future hardware-related modules will use a centralized BUDDY
hardware abstraction.

------------------------------------------------------------------------

## Architecture

``` text
Future ROS 2 Nodes
        │
        ▼
BUDDY Hardware API
        │
        ├─────────────────────┐
        ▼                     ▼
Mock / Simulation      Raspberry Pi Backend
Backend                       │
        │                     ▼
        │                  GPIO
        │                     │
        └─────────────┐       ▼
                      │   Physical Hardware
                      │
                      ▼
               Safe Development
```

------------------------------------------------------------------------

## Safety Model

Physical hardware is disabled by default.

The configuration contains:

``` yaml
hardware:
  enabled: false
  simulation: true

  gpio:
    numbering_mode: "BCM"
```

When:

``` text
hardware.enabled = false
```

BUDDY uses the mock GPIO backend.

No physical GPIO hardware is accessed.

When:

``` text
hardware.enabled = true
hardware.simulation = true
```

BUDDY still uses the mock backend.

Physical GPIO is selected only when:

``` text
hardware.enabled = true
hardware.simulation = false
```

and the Raspberry Pi GPIO dependency is available.

------------------------------------------------------------------------

## GPIO Abstraction

The hardware abstraction currently provides primitive operations for:

-   configuring digital inputs
-   configuring digital outputs
-   writing digital output values
-   reading digital input values
-   creating PWM channels
-   changing PWM duty cycle
-   safe cleanup

Task 09 intentionally does not implement:

-   motor control
-   servo movement
-   ultrasonic ranging
-   obstacle avoidance
-   navigation
-   patrol
-   camera control

Those features will use the abstraction in later tasks.

------------------------------------------------------------------------

## Mock GPIO Backend

The mock backend provides an in-memory representation of GPIO state.

It records:

-   pin modes
-   input/output values
-   pull configuration
-   PWM frequency
-   PWM duty cycle
-   cleanup state

This allows BUDDY hardware software to be developed and tested on a
normal Ubuntu development computer without connecting physical Raspberry
Pi hardware.

------------------------------------------------------------------------

## Raspberry Pi GPIO Backend

The Raspberry Pi backend is isolated from the rest of BUDDY.

The Raspberry Pi-specific GPIO package is imported only when physical
hardware mode is explicitly requested.

Therefore, importing the BUDDY package on a non-Raspberry-Pi development
machine does not require Raspberry Pi GPIO libraries.

If the dependency is unavailable while physical hardware mode is
requested, BUDDY raises a controlled hardware error instead of silently
continuing.

------------------------------------------------------------------------

## Backend Selection

Backend selection is performed centrally.

Conceptually:

``` text
Load configuration
        │
        ▼
hardware.enabled?
        │
     NO ├──────────────→ Mock Backend
        │
       YES
        │
        ▼
hardware.simulation?
        │
     YES ├──────────────→ Mock Backend
        │
       NO
        │
        ▼
Validate physical GPIO support
        │
        ▼
Raspberry Pi Backend
```

This provides fail-safe startup behaviour.

------------------------------------------------------------------------

## Configuration

Task 09 extends the existing Task 06 centralized configuration
architecture.

No separate configuration loader is introduced.

The base configuration contains:

``` yaml
hardware:
  enabled: false
  simulation: true

  gpio:
    numbering_mode: "BCM"
```

Development configuration also explicitly disables physical hardware.

No actual physical GPIO pin numbers are assigned in Task 09.

Pin assignments remain subject to Task 08 hardware verification.

------------------------------------------------------------------------

## GPIO Numbering

The abstraction currently accepts:

-   BCM
-   BOARD

The BUDDY project currently plans to use BCM numbering through
centralized configuration.

The physical GPIO mapping for motors, ultrasonic sensors, emergency stop
and other hardware remains TBD until formally verified.

------------------------------------------------------------------------

## Initialization Safety

Physical hardware cannot be activated merely by importing a BUDDY
hardware module.

Physical GPIO requires explicit configuration.

If configuration is missing or invalid, initialization fails safely.

------------------------------------------------------------------------

## Shutdown and Cleanup

All GPIO backends expose:

``` text
cleanup()
```

The mock backend returns recorded outputs and PWM channels to safe
inactive states.

The Raspberry Pi backend attempts to:

-   set configured outputs LOW
-   set PWM duty cycles to zero
-   stop PWM channels
-   release GPIO resources

This architecture allows future ROS 2 nodes to perform controlled
cleanup during shutdown.

------------------------------------------------------------------------

## Logging

Task 09 reuses the centralized Task 07 logging architecture.

Hardware utilities may use:

``` text
buddy_core.logging_utils
```

Backend-selection events may record:

-   mock backend selected
-   simulation mode selected
-   physical GPIO requested
-   configuration failures
-   GPIO dependency failures
-   cleanup

No new logging system is introduced.

------------------------------------------------------------------------

## Relationship to Future ROS 2 Nodes

Future BUDDY nodes may include:

``` text
obstacle_node
motor_driver_node
movement_controller_node
navigation_node
patrol_node
safety_node
neck_controller_node
buddy_state_node
system_monitor_node
```

These nodes should use the BUDDY hardware abstraction instead of
directly importing Raspberry Pi GPIO libraries.

Example:

``` text
navigation_node
       │
       ▼
movement_controller_node
       │
       ▼
motor_driver_node
       │
       ▼
BUDDY Hardware API
       │
       ▼
Raspberry Pi GPIO Backend
       │
       ▼
Motor Driver Hardware
```

------------------------------------------------------------------------

## Task Boundary

Task 09 creates the hardware software foundation only.

No physical motors were activated.

No servos were moved.

No ultrasonic sensors were operated.

No autonomous movement was implemented.

No GPIO pin assignments were finalized.

No Task 10 functionality was implemented.

------------------------------------------------------------------------

## Testing

Task 09 automated tests verify:

-   mock backend initialization
-   digital output configuration
-   output state changes
-   digital input simulation
-   PWM setup
-   PWM updates
-   cleanup
-   hardware-disabled behaviour
-   simulation backend selection
-   invalid configuration handling
-   unavailable Raspberry Pi GPIO dependency handling

All existing configuration and logging functionality must continue to
pass regression testing before Task 09 can be committed.

------------------------------------------------------------------------

## Result

TASK 09 hardware abstraction foundation implemented with physical
hardware disabled by default.

Final Task 09 PASS is granted only after full project regression
testing, ROS 2 build verification, smoke testing, documentation review
and Git verification.
