uint8 COMMAND_STOP=0
uint8 COMMAND_FORWARD=1
uint8 COMMAND_BACKWARD=2
uint8 COMMAND_TURN_LEFT=3
uint8 COMMAND_TURN_RIGHT=4
uint8 COMMAND_ROTATE_LEFT=5
uint8 COMMAND_ROTATE_RIGHT=6

builtin_interfaces/Time stamp

uint32 sequence_id
string source
uint8 command

float32 linear_speed
float32 angular_speed

builtin_interfaces/Duration valid_for

bool emergency
