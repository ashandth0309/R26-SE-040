# BUDDY Logging Architecture

## Task

TASK 07 — Centralized Logging, Rotation, Context, and ROS 2 Integration

## Purpose

This document defines the centralized logging architecture for the BUDDY ROS 2 migration.

The logging system provides consistent runtime diagnostics before future BUDDY subsystems such as movement, safety, audio, vision, navigation, security, and networking are implemented.

The goals are:

- consistent logging
- bounded disk usage
- useful debugging information
- machine-independent log paths
- safe handling of sensitive information
- compatibility with ROS 2 logging
- prevention of duplicate handlers

---

## Logging Layers

BUDDY uses two complementary logging mechanisms.

### ROS 2 Logging

ROS 2 nodes should continue to use:

```python
self.get_logger()
```

for normal ROS runtime information.

This includes:

- node startup
- node shutdown
- ROS communication state
- warnings
- runtime errors

Task 07 does not replace ROS 2 logging.

### BUDDY Shared Python Logging

Shared Python utilities and non-node modules may use:

```python
buddy_core.logging_utils
```

This provides:

- console logging
- rotating file logging
- consistent formatting
- contextual metadata support
- redaction helpers
- safe exception logging

The shared logger complements ROS 2 logging rather than replacing it.

---

# Configuration

The logging system uses the existing centralized configuration architecture.

## Log Level

Configured using:

```text
runtime.log_level
```

Supported values:

- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

The development environment currently uses:

```text
DEBUG
```

## Logging Settings

Configured under:

```text
logging
```

Available settings include:

- enabled
- console_enabled
- file_enabled
- file_name
- max_bytes
- backup_count
- include_context

Current base defaults:

- logging enabled: true
- console logging enabled: true
- file logging enabled: true
- file name: buddy.log
- maximum file size: 5,242,880 bytes
- retained backups: 5
- optional context enabled: true

---

# Log Storage

The log directory is configured using:

```text
storage.log_directory
```

Current value:

```text
logs
```

The default runtime log is therefore:

```text
logs/buddy.log
```

Relative paths are resolved against the BUDDY project root.

Machine-specific paths such as:

```text
/home/ashandth/...
```

are not hardcoded.

The logging smoke test was successfully executed from `/tmp`, while the actual runtime log continued to resolve to the BUDDY project directory.

---

# Log Rotation

BUDDY uses Python's standard:

```python
logging.handlers.RotatingFileHandler
```

This prevents uncontrolled growth of a single log file.

Current base configuration:

```text
max_bytes = 5242880
backup_count = 5
```

When the active log reaches the configured size, backup logs may be created such as:

- `buddy.log.1`
- `buddy.log.2`
- `buddy.log.3`

Rotation behaviour was verified using a temporary test configuration with a deliberately small file-size limit.

Production defaults were not changed merely for testing.

---

# Log Format

The centralized Python log format is:

```text
timestamp | level | logger | message
```

Example:

```text
2026-08-11T19:13:16Z | INFO | buddy.core.logging_smoke_test | BUDDY logging configured successfully.
```

Timestamps use UTC and include the `Z` suffix.

---

# Logger Naming

BUDDY uses hierarchical logger names.

Examples:

- `buddy.core`
- `buddy.audio`
- `buddy.vision`
- `buddy.movement`
- `buddy.navigation`
- `buddy.security`
- `buddy.network`

Task 07 implements the common infrastructure only.

Future subsystem loggers will be introduced by their own implementation tasks.

---

# Contextual Logging

Optional contextual metadata may be supplied when retrieving a BUDDY logger.

Possible future context fields include:

- robot_id
- request_id
- event_id
- module

Context remains optional.

Modules should include only information useful for debugging or traceability.

Sensitive values must not be included as context.

---

# Secret Safety

Secrets must never intentionally be written to logs.

Examples include:

- Telegram bot tokens
- Telegram private identifiers where treated as sensitive
- cloud API keys
- cloud API secrets
- remote authentication secrets
- GitHub Personal Access Tokens
- authorization headers
- bearer tokens
- passwords

The logging architecture provides:

```python
redact_mapping()
```

Known sensitive key names are replaced with:

```text
[REDACTED]
```

Nested mappings are supported.

The original input mapping is not modified.

Important:

Redaction is a safety helper, not a guarantee that arbitrary secrets can always be detected automatically.

The primary rule remains:

**Do not pass secrets to logging APIs.**

---

# Exception Logging

Unexpected exceptions should be recorded with useful diagnostic information and stack traces.

The logging utility provides:

```python
log_exception()
```

Future code should not use patterns such as:

```python
except Exception:
    pass
```

when failures require diagnosis.

---

# Duplicate Handler Protection

Repeated calls to logging configuration do not accumulate additional BUDDY handlers.

This prevents duplicated output such as one log message appearing two, three, or more times after reconfiguration.

Automated tests verify duplicate-handler protection.

---

# Runtime Logs and Git

Runtime logs are not source code.

The project `.gitignore` excludes:

```text
logs/
```

Therefore:

```text
logs/buddy.log
```

and rotated backups are not committed to Git.

---

# Development Safety

Task 07 introduces logging infrastructure only.

It does not:

- access GPIO
- operate motors
- access cameras
- access microphones
- implement voice recognition
- implement face recognition
- implement navigation
- implement patrol
- implement networking
- implement mobile control

---

# Automated Verification

Task 07 logging tests verify:

- configured log level
- console handler creation
- rotating file handler creation
- log-directory creation
- log-file creation
- duplicate-handler protection
- token redaction
- password redaction
- nested redaction
- redaction immutability
- invalid `max_bytes` handling
- invalid `backup_count` handling
- disabled file logging
- project-relative log paths
- log rotation
- contextual logger creation

Task 06 configuration and secret tests were also rerun.

Combined targeted result:

```text
32 tests passed
```

---

# Regression Verification

Task 07 preserves previous project functionality.

Verified:

- ROS workspace build succeeds
- configuration smoke test remains available
- core smoke test remains available
- logging smoke test is available
- Task 05 ROS interfaces remain frozen
- runtime logs remain ignored by Git

---

# Result

**TASK 07 PASS**
