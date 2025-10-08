# This message contains speaker diarization results for audio stream analysis.
# The message provides real-time information about who is speaking and when

std_msgs/Header header

# Primary speaker currently active in the audio stream
# Format: "speaker1", "speaker2", "speaker3", etc.
string current_speaker

# Confidence score for the current speaker
float64 current_speaker_confidence

# List of all speakers detected as active in the current audio segment
# Format: ["speaker1", "speaker2"] for multiple simultaneous speakers
string[] active_speakers