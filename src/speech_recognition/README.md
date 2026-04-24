# Speech Recognition Package

This package provides ROS2 nodes and helpers for end-to-end speech processing: voice activity detection, speaker diarization, wake-word spotting, and automatic speech recognition. It follows the same **decoupled design** as `audio_stream_manager`: ROS2 nodes are thin wrappers that only handle parameters, subscriptions, and publications, while all AI/signal-processing logic lives in pure-Python engine classes with zero ROS2 dependencies.

---

## Architecture

```
speech_recognition/
├── vad.py                  # ROS2 node (thin wrapper)
├── vad_engine.py           # All VAD logic: Silero model loading, inference
├── wake_word.py            # ROS2 node (thin wrapper)
├── wake_word_engine.py     # All wake-word logic: OpenWakeWord, sliding-window inference
├── diarization.py          # ROS2 node (thin wrapper)
├── diarization_engine.py   # All diarization logic: diart pipeline, observer, speaker mapping
├── asr.py                  # ROS2 node (thin wrapper)
├── asr_engine.py           # All ASR logic: Whisper model, VAD state machine, buffering
├── ros_audio_source.py     # AudioSource adapter: bridges ROS audio chunks to diart
└── utils/
    └── database_utils.py   # DataBaseManager: MongoDB speaker embedding persistence
```

---

## Components

### 1. VAD — Voice Activity Detection (`vad.py` + `vad_engine.py`)

**Purpose**: Receives raw audio chunks and publishes a per-chunk speech probability using the Silero VAD model.

#### `VAD` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `vad_params.yaml`
- Instantiate `VADEngine`
- Forward audio chunks from the subscription to the engine
- Stamp and publish `Vad` messages with the probability returned by the engine

#### `VADEngine` (pure Python)
Owns all VAD logic with zero ROS2 dependencies:
- **Model loading**: downloads and caches the Silero VAD model via `torch.hub`
- **Device selection**: runs inference on CUDA if available, otherwise CPU
- **Inference**: `predict(audio_data, sample_rate)` returns speech probability in `[0.0, 1.0]`
- Validates chunk size (expected 512 samples) and returns `0.0` for unexpected sizes

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Raw audio from the capture node

**Published Topics**:
- `/vad` (`hri_msgs/Vad`): Per-chunk speech probability

**Diagram**: `vad_workflow.mmd`

---

### 2. Wake Word Detection (`wake_word.py` + `wake_word_engine.py`)

**Purpose**: Runs a sliding-window keyword detector over incoming audio using OpenWakeWord models and publishes a `WakeWord` message on each detection.

#### `WakeWordDetectorNode` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `wake_word_params.yaml`
- Instantiate `WakeWordEngine` and provide `_publish_wake_word` as the detection callback
- Feed audio chunks into the engine
- Stamp and publish `WakeWord` messages when the callback fires

#### `WakeWordEngine` (pure Python)
Owns all wake-word logic with zero ROS2 dependencies:
- **Model loading**: loads one or more `.onnx`/`.tflite` model files from `model_base_path`
- **Sliding window**: accumulates audio in a deque buffer, processes windows at each `step_duration` interval
- **Inference thread**: runs model inference in a background thread via an internal audio queue
- **Detection callback**: calls `on_wake_word_detected(probability)` whenever a non-zero confidence score is produced

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Raw audio from the capture node

**Published Topics**:
- `/wake_word` (`hri_msgs/WakeWord`): Wake-word detection events

**Weights**: `.onnx` model files in `weights_openwakeword/` (e.g. `hey_jana.onnx`, `hey_robot.onnx`)

**Diagram**: `wake_word_workflow.mmd`

---

### 3. Speaker Diarization (`diarization.py` + `diarization_engine.py`)

**Purpose**: Identifies and segments different speakers in the audio stream using the diart streaming pipeline (pyannote-based). Speaker embeddings are persisted in MongoDB for re-identification across sessions.

#### `DiarizationNode` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `diarization_params.yaml`
- Instantiate `DiarizationEngine` with detection/update callbacks
- Feed audio chunks and VAD probabilities into the engine
- Own VAD buffering and publish `SpeechActivityDetection` messages on speaker-state changes
- Manage ROS4HRI voice publisher lifecycle (create, cleanup, publish)

#### `DiarizationEngine` (pure Python)
Owns all diarization logic with zero ROS2 dependencies:
- **Model loading**: loads `pyannote/segmentation` and `pyannote/embedding` models (requires `HF_TOKEN`)
- **Streaming pipeline**: runs `diart.SpeakerDiarization` with `StreamingInference`
- **`DiarizationObserver`**: reactive observer attached to the pipeline that processes `Annotation` frames and maps short-lived segment labels to stable speaker IDs
- **Speaker mapping**: assigns consistent speaker IDs across pipeline restarts and segments
- **Database integration**: queries `DataBaseManager` to match new embeddings against known speakers and stores new ones

#### `ROSAudioSource` (pure Python adapter)
Bridges incoming ROS audio chunks to the `AudioSource` interface expected by diart:
- Implements `diart.sources.AudioSource` using reactive programming patterns (`Subject`)
- Buffers incoming 512-sample chunks into fixed-size blocks (`block_duration` seconds)
- Thread-safe queue with configurable backpressure
- Used internally by `DiarizationEngine`; has no ROS2 imports

#### `DataBaseManager` (`utils/database_utils.py`)
Manages speaker embedding persistence in MongoDB:
- `save_speaker(name, embedding)` — upserts a speaker record
- `find_speaker(embedding, threshold)` — cosine-similarity search against all stored embeddings
- Enables seamless speaker re-identification across container restarts

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Raw audio chunks
- `/vad` (`hri_msgs/Vad`): VAD probabilities (gates the diarization pipeline)

**Published Topics**:
- `/voice_activity` (`hri_msgs/SpeechActivityDetection`): Per-speaker speech activity

**Requires**:
- `HF_TOKEN` environment variable or `huggingface-cli login` for gated pyannote models
- MongoDB running and accessible (see root `README.md` for setup)

**Diagram**: `diarization_workflow.mmd`

---

### 4. ASR — Automatic Speech Recognition (`asr.py` + `asr_engine.py`)

**Purpose**: Buffers incoming audio, uses VAD probabilities to detect speech segments, and transcribes them with a Whisper model. Publishes `SpeechResult` and `LiveSpeech` messages, with optional ROS4HRI-compatible per-speaker publication.

#### `ASRNode` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `asr_params.yaml`
- Validate the model size early (fail fast with a clear message) via `ASREngine.validate_model_size`
- Instantiate `ASREngine` and provide publishing callbacks
- Feed audio chunks, VAD probabilities, and speaker IDs into the engine
- Manage ROS4HRI voice publisher lifecycle (create, cleanup)

#### `ASREngine` (pure Python)
Owns all ASR logic with zero ROS2 dependencies:
- **Model registry**: maps short names (`turbo`, `large-v3`, `distil-large-v3`, …) to HuggingFace model IDs
- **Model loading**: downloads and caches a `faster_whisper.WhisperModel`; supports optional `BatchedInferencePipeline`
- **VAD state machine**: tracks `speech` / `silence` states, manages pre-buffer for leading audio capture
- **Silence timer thread**: triggers transcription after `min_silence_duration` of silence
- **Chunk splitting**: splits long utterances at `max_chunk_duration` to avoid latency spikes
- **Transcription callback**: calls `on_transcript(text, speaker_id)` — the ROS2 node stamps and publishes from this callback

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Raw audio chunks
- `/vad` (`hri_msgs/Vad`): VAD probabilities
- `/voice_activity` (`hri_msgs/SpeechActivityDetection`): Speaker ID from diarization (optional)

**Published Topics**:
- `/speech` (`hri_msgs/SpeechResult`): Final transcription results
- `/live_speech` (`hri_msgs/LiveSpeech`): Intermediate (live) transcription results

**Diagram**: `asr_workflow.mmd`

---

## Quick Notes

- **Audio chunk size**: audio arrives as fixed 512-sample chunks (~0.032 s at 16 kHz) on `/audio_and_device_info`.
- **Pipeline dependencies**: Diarization requires VAD. ASR works best with both VAD and Diarization enabled.
- **HuggingFace auth**: set `HF_TOKEN` in the environment or run `huggingface-cli login` for gated pyannote/Whisper models.
- **MongoDB**: required for speaker persistence. See the root `README.md` for configuration and troubleshooting.

## Diagrams

All workflows are documented as Mermaid diagrams alongside this file:

| Component | Diagram |
|-----------|---------|
| VAD | `vad_workflow.mmd` |
| Wake Word | `wake_word_workflow.mmd` |
| Diarization | `diarization_workflow.mmd` |
| ASR | `asr_workflow.mmd` |
| Database | `database_workflow.mmd` |

To render diagrams, install a Mermaid renderer or use the VS Code Mermaid Preview extension.
