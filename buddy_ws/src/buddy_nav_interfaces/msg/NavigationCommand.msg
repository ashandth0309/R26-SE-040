# High-level navigation request for BUDDY.
#
# Intended publishers: the voice module and the vision module, after they
# have turned a user request into one of these commands.  Nothing here
# makes natural language work end to end by itself: the sender is
# responsible for understanding the request.
#
# Every command is still filtered by buddy_navigation's own checks and by
# the existing safety_supervisor.  Physical motion also needs the
# navigation node to be explicitly enabled.

uint8 COMMAND_STOP=0
uint8 COMMAND_MOVE_FORWARD=1
uint8 COMMAND_TURN_LEFT=2
uint8 COMMAND_TURN_RIGHT=3
uint8 COMMAND_START_PATROL=4
uint8 COMMAND_GO_TO_LOCATION=5
uint8 COMMAND_CANCEL=6
uint8 COMMAND_PAUSE=7
uint8 COMMAND_RESUME=8

builtin_interfaces/Time stamp

string request_id
string source
uint8 command

# MOVE_FORWARD / TURN_*: how long to move.  Zero selects the node default.
# Motion is timed and open loop until the chassis is calibrated.
float32 duration_s

# START_PATROL: id of a route defined in the navigation config.
string route_id
bool repeat

# GO_TO_LOCATION: name of a location defined in the navigation config.
# Needs a trusted pose; refused otherwise.
string location_name
