# What is visible about one person right now. Nothing here is stored.
# display_name is filled only for people who agreed to be remembered.
# Expression fields are empty unless that person opted in, and are a rough
# guess about a visible expression, never a statement about feelings.
string person_key
string display_name
bool known
string identity_state

string posture
string motion

string expression_group
float32 expression_confidence
string expression_agreement

string fall_state
string[] gestures

float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height
