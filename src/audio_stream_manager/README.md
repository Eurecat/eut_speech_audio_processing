# Audio Stream Manager

This package provides audio capturing and processing utilities for ROS2, with support for automatic device management, manual device selection, and audio recording capabilities.

This package is designed to be hardware-agnostic and to integrate with speech-recognition processing pipelines. It aims to work across different environments and microphone types (USB, built-in, wireless), and to be reusable by downstream speech-processing nodes. It seamlessly handles environments where audio devices may connect or disconnect by automatically selecting the next functioning device without requiring manual intervention.


## Architecture

The package follows a **decoupled design**: ROS2 nodes are thin wrappers that only handle parameters and publishing, while all audio logic lives in pure-Python classes with no ROS2 dependencies.

```
audio_stream_manager/
├── audio_capturing.py        # ROS2 node (thin wrapper)
├── audio_capture_engine.py   # All audio logic: discovery, streaming, watchdog
├── audio_to_mp3.py           # ROS2 node (thin wrapper)
└── utils/
    ├── audio_models.py        # ActiveDevice dataclass
    ├── audio_utils.py         # compute_rms, resample_audio
    ├── sound_device_manager.py # SoundDeviceManager class
    └── audio_to_mp3_utils.py  # save_to_wav, convert_wav_to_mp3
```

---

## Components

### 1. Audio Capturing (`audio_capturing.py` + `audio_capture_engine.py`)

**Purpose**: Automatically detects and manages audio input devices with fallback support and device recovery.

#### `AudioCapturing` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `audio_params.yaml`
- Instantiate and start `AudioCaptureEngine`
- Stamp and publish `AudioAndDeviceInfo` messages via the `on_chunk_ready` callback
- Update the `device_name` parameter when the engine reports a device change via `on_device_changed`

#### `AudioCaptureEngine` (pure Python)
Owns all audio logic with zero ROS2 dependencies:
- **Device discovery**: queries `SoundDeviceManager` to find devices matching `device_name`
- **Device testing**: verifies a device actually produces audio (RMS > 0) before using it
- **Fallback**: switches to any available working device if the primary is unavailable
- **Audio resampling**: resamples chunks to `target_samplerate` (default 16 kHz) via `resample_audio`
- **Consistent chunk size**: buffers resampled data to emit chunks of a fixed size
- **Stream management**: opens/closes `sounddevice.InputStream` via `SoundDeviceManager`

#### `DeviceWatchdog` (inside `audio_capture_engine.py`)
Runs two background daemon threads:
- **Disconnection monitor**: detects when the stream stops delivering callbacks and triggers `on_disconnected`
- **Primary device recovery**: periodically checks whether the preferred device is available again and triggers `on_check_recovery`

#### `SoundDeviceManager` (`utils/sound_device_manager.py`)
Stateless helper class that wraps `sounddevice` queries:
- `query_input_devices()` — returns all available input devices
- `find_by_name(name, ...)` — filters devices by name substring
- `test_device(...)` — opens a short trial stream and verifies audio flow
- `open_stream(...)` / `stop_stream(...)` — manages `InputStream` lifecycle

#### `ActiveDevice` (`utils/audio_models.py`)
Dataclass holding the state of the currently active device: `device`, `name`, `index`, `samplerate`, `channels`.

**Published Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Audio data with device metadata

	Typical message fields published on `/audio_and_device_info`:
	- `header` (std_msgs/Header): message timestamp
	- `audio` (float32[]): audio samples (normalized float32)
	- `device_name` (string): human-readable device name
	- `device_id` (int32): device index on the host system
	- `device_samplerate` (float32): samplerate of the published audio (Hz)

**Workflow**:
1. On startup, `AudioCapturing` reads params and calls `engine.start(device_name)`
2. Engine searches for devices matching `device_name` via `SoundDeviceManager`
3. Tests each candidate; falls back to any available working device if none match
4. Opens stream and starts `DeviceWatchdog` threads
5. For each audio callback, resamples and buffers data; emits fixed-size chunks via `on_chunk_ready`
6. Watchdog detects disconnection → engine finds a new device and reopens the stream
7. Watchdog periodically checks for primary device recovery → engine switches back when found

**Diagram:** [Open audio capturing workflow](workflow_audio_capturing.mmd)

*Note about fallback behavior:*

If the node starts while the preferred device is unavailable it will connect to a fallback device so the pipeline remains operational. The node periodically checks for the preferred (primary) device and will attempt to switch back when it becomes available. Depending on system drivers and device enumeration, switching may not always be seamless; if you see issues where the primary device is ignored after being plugged-in, restarting the node can help.

---

### 2. Audio to MP3 (`audio_to_mp3.py`)

**Purpose**: Records audio from the ROS2 topic and saves it as an MP3 file when the node is stopped. Simply run the node while audio is being published, then stop it to get an MP3 file.

**Key Features**:
- **Continuous Buffering**: Accumulates all audio data received during node execution
- **WAV Intermediate**: Converts float32 audio to 16-bit PCM WAV format via `save_to_wav`
- **MP3 Conversion**: Uses FFmpeg to convert WAV to MP3 with high quality VBR encoding via `convert_wav_to_mp3`
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
