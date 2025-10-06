# Diarization message containing speaker information
# Timestamp when this diarization was performed
builtin_interfaces/Time timestamp

# Current active speaker (e.g., "speaker1", "speaker2", etc.)
string current_speaker

# List of all speakers detected in this audio segment
string[] active_speakers

# Confidence score for the current speaker detection (0.0 to 1.0)
float32 confidence