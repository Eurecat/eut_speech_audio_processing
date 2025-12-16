# Audio Stream Manager

This package provides audio capturing and processing utilities for ROS2, with support for automatic device management, manual device selection, and audio recording capabilities.

This package is designed to be hardware-agnostic and to integrate with speech-recognition processing pipelines. It aims to work across different environments and microphone types (USB, built-in, wireless), and to be reusable by downstream speech-processing nodes.

Ideal for production environments where audio devices may connect or disconnect (e.g., USB microphones, wireless devices) and you want the system to continue operating without manual intervention.


## Components

### 1. Audio Capturing Automatic Device (`audio_capturing_automatic_device.py`)

**Purpose**: Automatically detects and manages audio input devices with fallback support and device recovery.

**Key Features**:
- **Automatic Device Discovery**: Searches for audio devices by name (configurable via `device_name` parameter)
- **Device Fallback**: Automatically switches to any available working device if the primary device disconnects
- **Primary Device Recovery**: Periodically checks if the primary device becomes available again and switches back to it
- **Disconnection Detection**: Monitors audio stream callbacks and detects device disconnection automatically
- **Audio Resampling**: Resamples audio to a target sample rate (default: 16kHz) using librosa
- **Consistent Chunk Size**: Buffers audio data to maintain consistent chunk sizes after resampling
- **Silent Device Detection**: Tests devices to ensure they are receiving audio (RMS > 0) before using them

**Published Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Audio data with device metadata

	Typical message fields published on `/audio_and_device_info`:
	- `header` (std_msgs/Header): message timestamp
	- `audio` (float32[]): audio samples (normalized float32)
	- `device_name` (string): human-readable device name
	- `device_id` (int32): device index on the host system
	- `device_samplerate` (float32): samplerate of the published audio (Hz)

**Workflow**:
1. On startup, searches for devices matching `device_name`
2. Tests each matching device to ensure it's receiving audio
3. If no matching device is found or working, falls back to any available device
4. Continuously monitors the audio stream for disconnections
5. When disconnection is detected, automatically searches for a new working device
6. Periodically checks if the primary device is available again and switches back when possible

**Diagram:** [Open audio capturing workflow](workflow_audio_capturing.mmd)



*Note about fallback behavior:*

If the node starts while the preferred device is unavailable it will connect to a fallback device so the pipeline remains operational. The node periodically checks for the preferred (primary) device and will attempt to switch back when it becomes available. Depending on system drivers and device enumeration, switching may not always be seamless; if you see issues where the primary device is ignored after being plugged-in, restarting the node can help.

---

### 2. Save MP3 (`save_mp3.py`)

**Purpose**: Records audio from ROS2 topic and saves it as an MP3 file when the node is stopped. Simply run the node while audio is being published, then stop it to get an MP3 file.

**Key Features**:
- **Continuous Buffering**: Accumulates all audio data received during node execution
- **WAV Intermediate**: Converts float32 audio to 16-bit PCM WAV format
- **MP3 Conversion**: Uses FFmpeg to convert WAV to MP3 with high quality VBR encoding
- **Automatic Cleanup**: Removes temporary WAV file after conversion

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Audio stream to record

**Workflow**:
1. Subscribes to `/audio_and_device_info` topic
2. Accumulates all received audio chunks in memory
3. When node is stopped (Ctrl+C), converts accumulated float32 samples to int16 PCM
4. Writes PCM data to temporary WAV file
5. Uses FFmpeg to convert WAV to MP3 with libmp3lame codec (quality level 2)
6. Cleans up temporary WAV file

**Diagram:** [Open save MP3 workflow](workflow_save_mp3.mmd)
