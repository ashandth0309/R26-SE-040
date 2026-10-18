# Something worth telling the rest of Buddy about. Events are advice: the
# voice engine decides whether to speak and the safety supervisor still
# decides whether to move. suggested_text may be empty.
builtin_interfaces/Time stamp

# person_appeared person_left greeting_due expression_changed
# proactive_offer proactive_ack gesture object_appeared object_disappeared
# fall_suspected fall_alert fall_recovered resting_lying lost_after_descent
# teach_prompt
string kind

string person_key
string severity
string suggested_text
bool needs_voice_check

# Extra detail as JSON (no images, no audio)
string data_json
