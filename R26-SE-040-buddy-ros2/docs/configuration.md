# BUDDY Configuration Architecture

## TASK 06 — Centralized Configuration and Secrets Architecture

### Purpose

This document defines the centralized configuration architecture used by the BUDDY ROS 2 migration.

The goal is to prevent future BUDDY modules from hardcoding robot parameters, file paths, credentials, thresholds, and environment-specific values independently.

---

## 1. Configuration Layers

BUDDY uses three configuration layers.

### 1.1 Version-Controlled YAML

Version-controlled YAML files are used for non-secret robot and runtime configuration.

Primary files:

```text
config/buddy.yaml
config/development.yaml
```

These files may be committed to Git because they must not contain passwords, API keys, tokens, or private credentials.

### 1.2 Environment / Private Secrets

Sensitive values are provided through environment variables.

Examples include:

- Telegram credentials
- Future cloud credentials
- Future remote-authentication secrets

Real secrets must never be stored in version-controlled YAML files.

### 1.3 ROS 2 Runtime Parameters

Future subsystem ROS 2 nodes may expose only the parameters they actually require.

Task 06 does not create a global parameter server.

ROS 2 node parameters will be introduced incrementally by later subsystem tasks.

---

# 2. Configuration Files

## 2.1 `config/buddy.yaml`

This is the base BUDDY robot configuration.

It currently defines configuration sections for:

- Robot identity
- Runtime
- Timing
- Movement
- Safety
- Camera
- Audio
- Vision
- Network
- Storage
- Feature flags

Hardware-sensitive values that have not yet been calibrated remain `null` where appropriate.

The presence of configuration fields does not imply that the associated subsystem is already implemented.

Movement, navigation, patrol, security, and mobile control remain disabled until their future implementation tasks.

---

## 2.2 `config/development.yaml`

This file provides safe development-machine overrides.

Development mode currently disables hardware-dependent functionality, including:

- Camera
- Audio
- Voice
- Vision
- Movement
- Navigation
- Patrol
- Security
- Mobile control

This allows BUDDY software infrastructure to run safely on the Ubuntu development environment without requiring robot hardware.

---

# 3. Configuration Loading

Configuration loading is implemented in:

```text
buddy_ws/src/buddy_core/buddy_core/config_loader.py
```

The configuration loader:

1. Loads the base YAML configuration.
2. Optionally loads an override YAML configuration.
3. Recursively merges dictionaries.
4. Validates the resulting configuration.
5. Raises explicit errors for invalid configuration.

A nested override changes only the supplied fields.

### Example Base Configuration

```yaml
camera:
  enabled: true
  width: 640
  height: 480
```

### Example Override

```yaml
camera:
  enabled: false
```

### Merged Result

```yaml
camera:
  enabled: false
  width: 640
  height: 480
```

Unrelated nested values remain intact.

---

# 4. Validation

Configuration validation follows a fail-fast approach.

Invalid configuration should be detected during initialization rather than silently accepted.

Validation includes:

- Required sections
- Non-empty robot ID
- Non-empty robot name
- Valid logging level
- Positive timing values
- Non-negative movement limits when provided
- Non-negative safety distances when provided
- Valid camera dimensions when camera is enabled
- Valid audio sample rate when audio is enabled
- Boolean feature flags
- Valid storage path strings

Configuration values that depend on future hardware calibration may remain `null` until their subsystem requires them.

Configuration errors use:

```text
BuddyConfigError
```

---

# 5. Secrets

Secret access is implemented in:

```text
buddy_ws/src/buddy_core/buddy_core/secrets.py
```

The helper reads approved secret values from environment variables.

### Rules

- No hardcoded secrets
- No credentials in YAML
- No credentials in Git
- No secret values in logs
- Required missing secrets raise `BuddySecretError`
- Optional missing secrets return `None`

Task 06 does not implement Telegram, cloud, or remote-authentication functionality.

It establishes only the safe secret-access foundation.

---

# 6. `.env`

A real `.env` file is private local configuration.

It is ignored by Git.

```text
.env
```

The following example file contains placeholders only:

```text
.env.example
```

GitHub Personal Access Tokens are not BUDDY application configuration and must not be placed in `.env` or `.env.example`.

---

# 7. Path Resolution

The configuration architecture does not hardcode a machine-specific home path such as:

```text
/home/ashandth/...
```

Configuration lookup supports machine-independent path resolution.

The loader can locate the project configuration independently of the current shell working directory.

The following environment variables may be used when an explicit configuration location is required:

```text
BUDDY_PROJECT_ROOT
BUDDY_CONFIG_PATH
BUDDY_CONFIG_OVERRIDE
```

This allows future Raspberry Pi deployment without depending on the Ubuntu development username.

---

# 8. Configuration Smoke Test

The configuration smoke-test executable is:

```text
config_smoke_test
```

Command:

```bash
ros2 run buddy_core config_smoke_test
```

The smoke test safely reports only non-secret configuration values, including:

- Robot ID
- Robot name
- Runtime environment
- Log level
- Camera status
- Audio status
- Movement feature status
- Navigation feature status

No secret values are printed.

The smoke test was also verified from:

```text
/tmp
```

This demonstrates that configuration resolution does not depend on running the command from the repository directory.

---

# 9. Automated Tests

Task 06 includes focused tests for:

1. Valid YAML loading
2. Recursive override merging
3. Base dictionary immutability
4. Malformed YAML
5. Missing required sections
6. Invalid logging levels
7. Negative timing values
8. Invalid camera resolution
9. Base + override loading
10. Optional secret handling
11. Required missing secret handling
12. Required present secret handling

# Logging Configuration

Task 07 extends the centralized configuration architecture with logging settings.

The base configuration contains:

```yaml
logging:
  enabled: true
  console_enabled: true
  file_enabled: true
  file_name: "buddy.log"
  max_bytes: 5242880
  backup_count: 5
  include_context: true

### Test Result

```text
12 tests passed.
```

---

# 10. Future ROS Parameter Integration

Future subsystem nodes may declare ROS 2 parameters for values that they require at runtime.

Task 06 intentionally does not convert every configuration value into a ROS parameter.

The intended future flow is:

```text
Versioned YAML
      ↓
Validated Configuration
      ↓
Subsystem-Specific ROS Parameters
      ↓
Subsystem
```

Secrets remain outside ROS parameter files unless a future security review explicitly justifies otherwise.

---

# 11. Task 05 Interface Protection

The BUDDY ROS Interface Contract v1 remains frozen.

Task 06 does not modify:

```text
buddy_interfaces/msg/
buddy_interfaces/srv/
buddy_interfaces/action/
```

Future interface changes require explicit documented interface-version work.

---

# 12. Legacy Code

The legacy/non-ROS prototype configuration and source remain untouched during Task 06.

This includes:

```text
config.py
speech.py
main_robot.py
ml_brain.py
faceRecAndEmotion/
```

Migration will occur only in later controlled tasks.

---

# 13. Task 06 Summary

Task 06 establishes a centralized and secure configuration foundation for the BUDDY ROS 2 architecture.

The completed architecture provides:

- Centralized non-secret YAML configuration
- Development-specific configuration overrides
- Recursive configuration merging
- Fail-fast configuration validation
- Environment-based secret handling
- Protection against credentials entering Git
- Machine-independent configuration path resolution
- A safe ROS 2 configuration smoke test
- Automated configuration and secret-handling tests
- A foundation for future subsystem-specific ROS 2 parameters
- Continued protection of the frozen BUDDY ROS Interface Contract v1
- Separation between the ROS 2 migration and legacy BUDDY prototype

### Verification Status

```text
Configuration loading:       PASSED
Recursive override merging:  PASSED
Configuration validation:    PASSED
Secret handling:             PASSED
Path-independent loading:    PASSED
ROS 2 smoke test:            PASSED
Automated tests:             12 PASSED
Interface Contract v1:       UNCHANGED
Legacy code:                 UNCHANGED
```

**TASK 06 — Centralized Configuration and Secrets Architecture: COMPLETE**
