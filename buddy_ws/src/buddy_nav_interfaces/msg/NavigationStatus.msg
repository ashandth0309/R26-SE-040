# Navigation state, published about once per second and on every change.

uint8 STATE_IDLE=0
uint8 STATE_RUNNING=1
uint8 STATE_PAUSED=2
uint8 STATE_SUCCEEDED=3
uint8 STATE_FAILED=4
uint8 STATE_CANCELLED=5

builtin_interfaces/Time stamp

uint8 state
string request_id
string active_command
string reason

# Why motion may be refused, so callers can explain it to the user.
bool motion_enabled
bool pose_trusted
bool sensors_healthy
bool emergency_stop_active

float32 progress
