uint8 MODE_STARTING=0
uint8 MODE_IDLE=1
uint8 MODE_LISTENING=2
uint8 MODE_INTERACTING=3
uint8 MODE_MANUAL_CONTROL=4
uint8 MODE_FOLLOWING=5
uint8 MODE_AUTONOMOUS=6
uint8 MODE_PATROL=7
uint8 MODE_SECURITY_ALERT=8
uint8 MODE_ERROR=9
uint8 MODE_EMERGENCY_STOP=10
uint8 MODE_SHUTTING_DOWN=11

builtin_interfaces/Time stamp
string robot_id
uint8 mode

bool online
bool camera_available
bool microphone_available
bool movement_available
bool obstacle_sensors_available
bool internet_available

float32 battery_percent
string activity
string health_summary
