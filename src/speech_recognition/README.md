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
├── diarization_engine.py   # diart backend (legacy): diart pipeline, observer, speaker mapping
├── diart_identity_engine.py # diart backend using VoiceIdentityManager (diart_use_voice_identity_manager)
├── redi_voice_engine.py    # redimnet2 backend, no diart: VAD turns -> ReDimNet2 -> VoiceIdentityManager
├── voice_identity_manager.py # Shared speaker identities: matching, merging, persistence
├── asr.py                  # ROS2 node (thin wrapper)
├── asr_engine.py           # All ASR logic: Whisper model, VAD state machine, buffering
├── parakeet_asr_engine.py  # asr_backend=parakeet: NeMo Parakeet TDT model on the same ASREngine pipeline
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

**Diagram**: [Open VAD workflow](vad_workflow.mmd)

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

**Diagram**: [Open wake word workflow](wake_word_workflow.mmd)

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
- `/speech_activity_detection` (`hri_msgs/SpeechActivityDetection`): Per-speaker speech activity

**Requires**:
- `HF_TOKEN` environment variable or `huggingface-cli login` for gated pyannote models
- MongoDB running and accessible (see root `README.md` for setup)

#### Selecting the backend

`diarization_backend` chooses the engine; the node logs
`Selected diarization backend: <backend> (engine=<class>)` at startup, check it.

| `diarization_backend` | `diart_use_voice_identity_manager` | Engine |
|---|---|---|
| `redimnet2` (launch default) | n/a | `RediVoiceEngine` |
| `diart` | `False` (default) | `DiarizationEngine` (legacy, unchanged) |
| `diart` | `True` | `DiartManagedIdentityEngine` |

`docker-compose_mp3.yaml` passes `diarization_backend:=${DIARIZATION_BACKEND:-diart}`
explicitly, which overrides the launch default: set `DIARIZATION_BACKEND` in `Docker/.env`.

#### REDI backend (`redi_voice_engine.py` + `voice_identity_manager.py`)

Independent of diart: no pyannote, no online clustering. Modelled on
`EutHRIFaces/face_recognition/identity_manager.py`, with a voice *turn* playing the role of a
tracked face.

```
/vad -> speech turns (split on pauses)
     -> ReDimNet2 embedding of the most recent 2 s of the turn, refreshed every 0.5 s
     -> VoiceIdentityManager: match / create / learn / merge EUT_speakerN
     -> /speech_activity_detection
```

- **Speaker change without a pause:** each refresh also embeds the last 1.0 s (the *probe*). If
  it scores below `redi_change_threshold` (0.35) against the current speaker, the turn is split
  there and the newcomer is labelled from the probe, about 1 s after they start.
- **Matching:** cosine score plus a top-1/top-2 margin, exclusive assignment, stickiness.
  Bars: 0.55 for a confirmed speaker, 0.40 for a young one or for windows under 1.5 s.
- **Creating:** only from ≥1.5 s of speech, or from a probe after a detected change. A speaker
  created from a short window is reseeded from its longer windows as it keeps talking.
- **Learning:** only from stable stretches (consecutive windows on the same speaker), at most once
  per 2 s of speech, never from a window that crosses a speaker change.
- **Merging:** identities that turn out to be the same person are merged, keeping the lower number.

Model: ReDimNet2 `b6` / `lm` / `vb2+vox2+cnc2_v0`, loaded through torch hub from a pinned commit
of `PalabraAI/redimnet2` (the `v1.0.0` tag cannot load this checkpoint). Internet is needed on the
first load. Every threshold is documented in `config/diarization_params.yaml`; measurements and
history are in `plans/plan_REDI_fixes.md`.

Known limit: a line shorter than ~1 s, by someone not heard yet, followed with no pause by another
speaker, keeps the previous label.

#### Speaker persistence in MongoDB (REDI)

Like EutHRIFaces, REDI saves speakers and reloads them at startup, so after a restart the same
voice keeps the same `EUT_speakerN` and new speakers continue the numbering.

| | |
|---|---|
| On/off | `REDI_USE_DATABASE=true` / `false` in `Docker/.env` (passed as the `redi_use_database` launch argument by `docker-compose.yaml`, `android-docker-compose.yaml` and `docker-compose_mp3.yaml`). Without the launch argument the yaml value `redi_use_database: True` applies. |
| Server | The same `mongodb` container and `coghri_speakers_mongodb_data` volume as diart |
| Location | database `speaker_recognition`, collection `voice_identities`, one `model_key` per ReDimNet2 checkpoint (`redimnet2:b6:lm:vb2+vox2+cnc2_v0`). Legacy diart uses collection `speakers`, so the two backends never mix embeddings. |
| Saved | Confirmed speakers, and unconfirmed ones with ≥2 embeddings from ≥3 s of speech. A single short burst is never saved. Up to the last 20 embeddings plus the mean. |
| When | When a speaker first qualifies, at every turn end, and at shutdown. Saving never waits for a clean shutdown. |
| Loaded | At startup, into memory. MongoDB is not queried per turn. Loaded speakers are never dropped by inactive cleanup. |
| Errors | If MongoDB is unreachable the node logs a warning and runs with session-only speakers. |

The legacy diart backend is unaffected: it keeps its own `use_database` parameter (default `False`).

Startup log lines to check:

```
Voice identities persisted in MongoDB (model_key=redimnet2:b6:lm:vb2+vox2+cnc2_v0)
Voice identity manager ready with 2 persistent identities: EUT_speaker2, EUT_speaker3
Persisted voice identity EUT_speaker4 (2 embeddings, 4.0s)
```

Forget all REDI speakers (the MongoDB container must be running):

```bash
docker exec mongodb mongosh -u eurecat -p cerdanyola --authenticationDatabase admin --eval \
  'db.getSiblingDB("speaker_recognition").voice_identities.deleteMany({model_key: "redimnet2:b6:lm:vb2+vox2+cnc2_v0"})'
```

For ground-truth mp3 tests set `REDI_USE_DATABASE=false`, otherwise speakers saved from the
microphone or earlier runs are loaded and the ids differ from a fresh run.

**Diagram**: [Open diarization workflow](diarization_workflow.mmd)

---

### 4. ASR — Automatic Speech Recognition (`asr.py` + `asr_engine.py`)

**Purpose**: Buffers incoming audio, uses VAD probabilities to detect speech segments, and transcribes them with a Whisper or Parakeet model. Publishes `SpeechResult` and `LiveSpeech` messages, with optional ROS4HRI-compatible per-speaker publication.

#### Selecting the backend

`asr_backend` in `asr_params.yaml` chooses the model; the node logs
`Selected ASR backend: <backend>` at startup. Both backends use the same
`ASREngine` pipeline (VAD segmentation, speaker-change flushes, per-speaker
sentence grouping) and publish the same `/speech_result`, so diarization and
downstream consumers do not change.

| `asr_backend` | Engine | Languages | Notes |
|---|---|---|---|
| `whisper` (default) | `ASREngine` (faster-whisper) | all Whisper languages, **including Catalan** | `model_size`, `compute_type`, batched inference apply |
| `parakeet` | `ParakeetASREngine` (NVIDIA NeMo, `parakeet_model_name`) | 25 European languages, Spanish yes, **Catalan no** | always float32 on GPU; `language_code` from a separate LangID model (see below) |

Override without editing the yaml: `ASR_BACKEND=parakeet` in `Docker/.env` (passed as the
`asr_backend` launch argument by `docker-compose.yaml`, `android-docker-compose.yaml` and
`docker-compose_mp3.yaml`). Empty keeps the yaml value.

**Parakeet language code.** `parakeet-tdt-0.6b-v3` neither outputs nor accepts a language id
(confirmed by NVIDIA in NeMo issues #14799 and #15097). `language_id.py` runs NVIDIA's
`langid_ambernet` (`parakeet_language_id_model`, 107 languages including Catalan, ~6 ms per chunk)
on every published chunk and picks the most likely language from `language`, with the same meaning
as for Whisper (`"auto"` = en/es/ca, one code = no detection, a list = choose from it). Below
`parakeet_language_id_min_confidence` (0.9) or for chunks under 0.25 s the last detected language is
kept, so short replies do not flip it. Measured on Jetson Thor, choosing from en/es/ca:

| Test | Whisper turbo detection (Whisper backend) | AmberNet + confidence gate (Parakeet backend) |
|---|---|---|
| FLEURS, 1 s / 2 s / 3 s of speech | 82% / 91% / 98% (Catalan at 1 s: 48%) | 88% / 96% / 99% without gate |
| Spanish/Catalan/English sequence, real chunk lengths | 82% (Catalan 58%) | 97% (ROS pipeline: 46/46) |
| English movie mp3 (music, short chunks) | 98% | 98% (ROS pipeline: ~97%) |

The Parakeet checkpoint (`nvidia/parakeet-tdt-0.6b-v3`, ~2.5 GB) downloads once to
`speech_recognition/weights/`, next to the Whisper weights. NeMo (`nemo_toolkit[asr]==2.4.0`)
is installed in the ARM / Jetson Thor image (`requirements_arm.txt`).

#### `ASRNode` (ROS2 node)
Thin node whose only responsibilities are:
- Declare and read ROS2 parameters from `asr_params.yaml`
- Select the backend (`asr_backend`) and validate the model size early (fail fast with a clear message) via `validate_model_size`
- Instantiate `ASREngine` or `ParakeetASREngine` and provide publishing callbacks
- Feed audio chunks, VAD probabilities, and speaker IDs into the engine
- Manage ROS4HRI voice publisher lifecycle (create, cleanup)

#### `ASREngine` (pure Python)
Owns all ASR logic with zero ROS2 dependencies:
- **Model registry**: maps short names (`turbo`, `large-v3`, `distil-large-v3`, …) to HuggingFace model IDs
- **Model loading**: downloads and caches a `faster_whisper.WhisperModel`; supports optional `BatchedInferencePipeline`
- **Backend hooks**: `_load_model()`, `_resolve_language()` and `_run_transcription()` are the only methods a backend overrides (`ParakeetASREngine`)
- **VAD state machine**: tracks `speech` / `silence` states, manages pre-buffer for leading audio capture
- **Silence timer thread**: triggers transcription after `min_silence_duration` of silence
- **Chunk splitting**: splits long utterances at `max_chunk_duration` to avoid latency spikes
- **Transcription callback**: calls `on_transcript(text, speaker_id)` — the ROS2 node stamps and publishes from this callback

**Subscribed Topics**:
- `/audio_and_device_info` (`hri_msgs/AudioAndDeviceInfo`): Raw audio chunks
- `/vad` (`hri_msgs/Vad`): VAD probabilities
- `/voice_activity` (`hri_msgs/SpeechActivityDetection`): Speaker ID from diarization (optional)

**Published Topics**:
- `/speech_result` (`hri_msgs/SpeechResult`): Final transcription results
- `/humans/voices/{speaker_id}/speech` (`hri_msgs/LiveSpeech`): Intermediate (live) transcription results

**Diagram**: [Open ASR workflow](asr_workflow.mmd)

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
| VAD | [vad_workflow.mmd](vad_workflow.mmd) |
| Wake Word | [wake_word_workflow.mmd](wake_word_workflow.mmd) |
| Diarization | [diarization_workflow.mmd](diarization_workflow.mmd) |
| ASR | [asr_workflow.mmd](asr_workflow.mmd) |
| Database | [database_workflow.mmd](database_workflow.mmd) |

To render diagrams, install a Mermaid renderer or use the VS Code Mermaid Preview extension.
