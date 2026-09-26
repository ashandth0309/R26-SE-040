uint8 EVENT_UNKNOWN_PERSON=0
uint8 EVENT_KNOWN_PERSON=1
uint8 EVENT_MOTION=2
uint8 EVENT_MANUAL=3
uint8 EVENT_SYSTEM=4

uint8 SEVERITY_INFO=0
uint8 SEVERITY_LOW=1
uint8 SEVERITY_MEDIUM=2
uint8 SEVERITY_HIGH=3
uint8 SEVERITY_CRITICAL=4

builtin_interfaces/Time stamp

string event_id

uint8 event_type
uint8 severity

string person_id
string description

string image_path
string video_path

bool uploaded
