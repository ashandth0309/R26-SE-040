# BUDDY – Intelligent Companion Robot Dog

**Research Project ID:** R26-SE-040  
**Programme:** BSc (Hons) in Information Technology – Software Engineering  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)

BUDDY is an affordable intelligent companion robot-dog research prototype that integrates offline voice interaction, visual intelligence, obstacle-aware navigation, human following, and safety-supervised physical movement on Raspberry Pi-class hardware.

The project is divided into three individual research components and integrated through shared interfaces and ROS 2-based communication.

---

## Research Team

### Ashandth Uthayashankar – IT22272522
**Research Area:** Offline Voice Intelligence, Personalized Interaction, and Safety-Aware System Integration

Main contributions:
- Wake-word interaction using “Buddy”
- Offline speech recognition using Vosk
- Intent recognition and deterministic command processing
- Piper text-to-speech integration
- Voice-controlled physical movement commands
- ROS 2 voice and movement integration
- Safety-aware movement integration
- Raspberry Pi deployment and hardware integration support

### Sobiya Anton Suresh – IT22203694
**Research Area:** Visual Intelligence for Face Recognition, Emotion-Aware Interaction, Human Fall Detection, and Adaptive Object Learning

Main contributions:
- Known and unknown person recognition
- Local face enrolment
- Facial-expression cue analysis
- Adaptive object learning
- Human fall-detection logic
- Pan/tilt face tracking
- Camera streaming and media handling
- Flutter companion application

### S. S. Y. Wickramasinghe – IT21816086
**Research Area:** Autonomous Navigation, Obstacle Avoidance, Human Following, and 2D Path Planning

Main contributions:
- Obstacle-aware navigation research
- A* 2D path planning
- PID-style human following
- Field-of-view and occlusion reasoning
- Search and recovery behaviour
- Dynamic replanning
- Ursina-based 3D-visualised 2D navigation simulation

---

## Supervisors

**Supervisor:** Pradeep K. W. Abeygunawardhana  
**Co-Supervisor:** Samantha Rajapakse

---

## System Architecture

BUDDY follows a modular architecture in which high-level intelligent components do not directly control the motors.

```text
Voice / Vision / Navigation
            ↓
          ROS 2
            ↓
   High-Level Commands
            ↓
    Safety Supervisor
            ↓
   Safe Movement Command
            ↓
      Motor Executor
            ↓
       Motor Drivers
            ↓
          Motors
```

The safety supervisor checks current obstacle information before movement commands are allowed to reach the motor-control layer.

---

## Main Technologies

### Voice and Integration
- Python
- Vosk
- Piper
- ROS 2 Jazzy
- scikit-learn

### Visual Intelligence
- OpenCV
- DeepFace
- MediaPipe
- MobileNetV2
- Flask
- Flutter

### Navigation
- Python
- A*
- PID control
- Ursina
- Ultrasonic sensing

### Platform
- Raspberry Pi 4B – 4 GB
- Ubuntu 24.04 ARM64
- ROS 2 Jazzy

---

## Hardware Platform

The BUDDY physical prototype includes:

- Raspberry Pi 4B – 4 GB
- Raspberry Pi Camera Module / IMX219
- AB13X USB microphone
- 4 geared DC motors
- 2 × TB6612FNG motor drivers
- 4 × HC-SR04 ultrasonic sensors
- MG90-class servos
- PCA9685 servo controller
- Protected 3 × 18650 battery power system
- Voltage regulation and protection circuitry

---

## Repository Development

Development was carried out using separate Git branches for the individual research components and integration work.

Important development branches include:

```text
buddy-ros2
sobiya-object-fall-detect
integration/sobiya-vision-app
Yohan
voice-+-face
```

This branch-based workflow allowed individual components to be developed and tested separately before integration.

---

## Basic ROS 2 Setup

Install ROS 2 Jazzy on Ubuntu 24.04 and source the environment:

```bash
source /opt/ros/jazzy/setup.bash
```

Build the required ROS 2 workspace:

```bash
cd buddy_ws
colcon build
source install/setup.bash
```

Actual launch commands depend on the component being tested.

---

## Voice Runtime

The validated core voice path uses:

```text
AB13X USB Microphone
        ↓
Audio preprocessing
        ↓
Vosk offline speech recognition
        ↓
Wake word / intent processing
        ↓
ROS 2 movement request
        ↓
Safety supervisor
        ↓
Motor executor
```

The wake word is:

```text
Buddy
```

Validated physical movement commands include:

```text
Forward
Backward
Left
Right
Stop
```

---

## Autonomous Navigation and Patrol (work in progress)

The mapping, obstacle-avoidance, navigation and security-patrol component
lives in `buddy_ws/src/buddy_navigation` (with message definitions in
`buddy_nav_interfaces`). It reuses the existing safety supervisor and motor
executor, ships with motion disabled by default, and is validated so far in
simulation and unit tests only. See
[docs/yohan_mapping_navigation.md](docs/yohan_mapping_navigation.md) for the
architecture, simulation instructions, test results and limitations.

---

## Validation Status

### Physically Validated
- Cold boot to interactive operation
- Offline voice interaction
- Wake-word activation
- Forward, backward, left, right and stop movement
- Four-direction ultrasonic obstacle sensing
- Dynamic obstacle stop/resume
- Camera-based face tracking

### Subsystem Validated
- Known and unknown person recognition
- Face enrolment
- Facial-expression cues
- Adaptive object learning
- Camera and media functionality

### Simulation Validated
- A* path planning
- PID-style human following
- Field-of-view reasoning
- Dynamic replanning
- Search and recovery behaviour

### Continuing Work
- Full physical patrol and localisation
- Full-body physical fall validation
- Dual-microphone acoustic localisation
- Fully concurrent multimodal operation

---

## Project Status

BUDDY is a final-year research prototype.

The repository distinguishes between physically validated functionality, subsystem-level validation, simulation results and continuing research work. Features are not presented as production-ready where full validation has not been completed.

---

## License / Academic Use

This repository contains academic research work developed for the SLIIT final-year research project **R26-SE-040**.
