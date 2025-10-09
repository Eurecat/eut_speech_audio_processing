# This message encodes the live result of a speech recognition process.
# A series of incremental results might be provided, until a final recognition
# hypothesis is returned.

std_msgs/Header header

string transcript # incremental recognition results
float64 transcript_confidence # confidence of the recognised speech transcript, [0-1]


# The detected language code, e.g. "en", "es", "zh", ...
string language_code