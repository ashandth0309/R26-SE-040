# A request for Buddy to say something.
#
# The voice node owns the speaker and the only speech queue.  Any other
# module (the coordinator, the vision module through the coordinator,
# navigation status) asks for speech with this message on /voice/say.
# Nothing else may play audio.
#
# priority matches the voice engine's queue: lower plays first.

uint8 PRIORITY_CRITICAL=0   # emergency and safety warnings
uint8 PRIORITY_HIGH=1       # stop acknowledgements, important status
uint8 PRIORITY_NORMAL=2     # direct answers to the user
uint8 PRIORITY_LOW=3        # proactive speech, optional remarks

builtin_interfaces/Time stamp

string request_id
string source
string text
uint8 priority

# Drop the request if it has not started within this time. Zero keeps the
# receiver's default.
float32 ttl_s

# Stop whatever is being said first. Only honoured for CRITICAL and HIGH.
bool interrupt
