# BUDDY Development Workflow

## Development Rule

ONE TASK → IMPLEMENT → TEST → VERIFY → COMMIT → NEXT TASK

## Branch Responsibilities

### main
Research release/reference branch. Do not develop directly here.

### voice-+-face
Legacy working prototype/reference branch. Preserve this branch during ROS 2 migration.

### buddy-ros2
Active ROS 2 integration and development branch.

### Temporary Feature Branches
Use temporary branches for risky development, major refactoring, experiments, or work requiring several commits.

Naming examples:

- feature/task-XX-short-name
- fix/short-name
- docs/short-name

## Commit Policy

Recommended prefixes:

- docs:
- chore:
- build:
- feat:
- fix:
- test:
- refactor:
- perf:
- security:
- deploy:
- research:

Avoid vague commit messages such as `update`, `final`, `changes`, `new`, or `working`.

## Task Completion Policy

A task is complete only when:

1. Implementation is complete.
2. Task-specific tests pass.
3. Regression tests pass where applicable.
4. Git diff has been reviewed.
5. Generated files are not accidentally tracked.
6. Secrets are not committed.
7. The task is committed.
8. The branch is pushed.

## Stable Interface Policy

Once ROS 2 interfaces are frozen, they are treated as project contracts.

If an interface must change:

1. Document why.
2. Identify affected consumers.
3. Update documentation.
4. Update dependent modules.
5. Run regression tests.
6. Commit the interface change explicitly.

No future task may silently rewrite an established interface.
