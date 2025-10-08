# This message combines audio data with device information for audio streaming applications.
# It contains a chunk of audio samples along with metadata about the audio capture device.

std_msgs/Header header

# Audio data chunk containing float32 samples
# Contains audio samples captured from the input device
# Data represents audio amplitude values in the range [-1.0, 1.0]
float32[] audio

# Human-readable name of the audio capture device
string device_name

# Unique identifier for the audio device
# This ID corresponds to the system's audio device index
uint8 device_id

# Sample rate of the audio device in Hz
# Common values: 16000, 44100, 48000
# Determines the temporal resolution of the audio data
# Higher sample rates provide better audio quality but more data
float32 device_samplerate