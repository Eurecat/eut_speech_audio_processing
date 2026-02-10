# eut_speech_audio_processing

[![Build Status](https://github.com/Eurecat/eut_speech_audio_processing/actions/workflows/ci-cd.yml/badge.svg?branch=jazzy-devel)](https://github.com/Eurecat/eut_speech_audio_processing/actions/workflows/ci-cd.yml?query=branch%3Ajazzy-devel)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Eurecat/eut_speech_audio_processing/badges/jazzy-devel/test-badge.json)](https://github.com/Eurecat/eut_speech_audio_processing/actions/workflows/ci-cd.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Eurecat/eut_speech_audio_processing/badges/jazzy-devel/coverage-badge.json)](https://github.com/Eurecat/eut_speech_audio_processing/actions/workflows/ci-cd.yml)

## What This Repository Does

**eut_speech_audio_processing** provides comprehensive audio perception capabilities for robotic systems, enabling robots to hear, understand, and interact with their acoustic environment. It processes audio streams through a sophisticated pipeline that detects speech activity, identifies speakers, recognizes wake words, and transcribes spoken language into text for natural human-robot interaction.

<p align="center">
  <img src="Docker/imgs/perceptionstack_diagram.jpeg" alt="Audio Processing Architecture" width="800"/>
</p>

## Key Characteristics

- 🎤 **Audio Stream Management**: Hardware-isolated audio capture with robust error handling
- 🗣️ **Voice Activity Detection (VAD)**: Real-time detection of speech segments in audio streams
- 👥 **Speaker Diarization**: Multi-speaker identification and temporal segmentation
- 📝 **Speech Transcription**: High-accuracy ASR using state-of-the-art Whisper models
- 🔊 **Wake Word Detection**: Configurable keyword spotting for voice activation
- 🗄️ **Speaker Database**: MongoDB-based speaker embedding storage and recognition
- 🐳 **Docker Containerization**: Fully containerized with decoupled services for reliability
- ⚙️ **Modular Design**: Enable/disable components independently based on deployment needs

## Overview

This repository contains the **speech and audio processing module** for the perception layer of robotic systems. The module provides comprehensive audio processing capabilities designed to enable robots to understand and interact with their acoustic environment.

### Architecture

The system is designed with two **decoupled components** to ensure robust error handling and system reliability:

#### **Audio Stream Manager**
This submodule interfaces directly with audio hardware devices to capture raw audio streams, isolating hardware-related issues from the speech processing logic.

#### **Speech Recognition Pipeline**
This submodule contains several audio processing and understanding capabilities that enable real-time analysis of acoustic environments, including **speech and wake word detection**, **speaker identification**, and  **transcription**. 

This module receives as input the processed audio output from the *Audio Stream Manager*. 

It is composed of:
  - **Voice Activity Detection (VAD)**: Detects when speech is present in the audio stream
  - **Speaker Diarization**: Identifies and segments different speakers in multi-speaker scenarios
  - **Speech Transcription**: Converts spoken language into text using automatic speech recognition (ASR)
  - **Wake Word Detection**: Keyword spotting for voice activation

### Installation & Setup

#### Step 0: Build Base Image
First, build the required base Docker image from [EutRobAIDockers](https://github.com/Eurecat/EutRobAIDockers).
```bash
git clone git@github.com:Eurecat/EutRobAIDockers.git
cd EutRobAIDockers
./build_container.sh 
# Defaults to ROS2 Jazzy and GPU
# Optionally, use --clean-rebuild to force a complete rebuild without cached layers. --cpu flag can be used to build a CPU-only image if needed. etc.
```

#### Step 1: Clone Repository
```bash
git clone git@github.com:Eurecat/eut_speech_audio_processing.git
cd eut_speech_audio_processing
```

**📖 For detailed setup, VS Code integration, and troubleshooting, see [PRECOMMIT.md](PRECOMMIT.md)**

#### Step 2: Build Application Image

For Vulcanexus-based installations:
```bash
cd Docker && ./build_container.sh --vulcanexus
```

For standard installations:
```bash
cd Docker && ./build_container.sh
```

**Build Options:**
- Use `--clean-rebuild` flag to force a complete rebuild without cached layers

### Configuration Parameters

Configure your Hugging Face token in the `.env` file to access the required models (take a look at the `.env.example` and use the same variable name). Ensure the token has permission for:

- `openai/whisper`
- `pyannote/embedding`
- `pyannote/segmentation`




## Usage

### Docker Compose (Recommended)

Navigate to the Docker directory and launch both services simultaneously:

```bash
cd Docker
docker compose up
```

This command will initialize both the **Audio Stream Manager** and the **Speech Recognition Pipeline** services automatically.

**Microphone Selection:**
1. Check detected audio devices:
   ```bash
   docker logs audio_device_manager
   ```
   Example output shows available devices with their hardware IDs.

2. Modify device_name with the desired one in [audio_params.yaml](./src/audio_stream_manager/config/audio_params.yaml)

3. Restart only the audio service:
   ```bash
   docker restart audio_device_manager
   ```

#### Service Configuration

The Docker Compose setup includes two main services:

1. **Audio Device Manager Service**: Handles audio input device selection and stream management
2. **Speech Recognition Service**: Provides VAD, diarization, wake word and ASR capabilities

#### Enabling/Disabling Components

You can selectively enable or disable speech recognition components by editing the `command` section in the `dev-docker-compose.yaml` file. Modify the speech recognition service command as follows:

```bash
# Example: Disable diarization and ASR, keep only VAD
command: bash -c "source /workspace/install/setup.bash && ros2 launch speech_recognition speech_recognition.launch.py enable_diarization:=false enable_asr:=false"
```

**Available options:**
- `enable_vad:=true/false` - Voice Activity Detection
- `enable_diarization:=true/false` - Speaker Diarization  
- `enable_wake_word:=true/false`- Wake Word
- `enable_asr:=true/false` - Automatic Speech Recognition

**Important Dependencies:**
- **Diarization** requires **VAD** to work properly
- **ASR** requires both **VAD** and **Diarization** for optimal performance



### Managing the Speaker Recognition Database

To query the database:
```bash
mongosh
use speaker_recognition
db.speakers.find()
```

To delete the database, remove the associated Docker volume.

You can also manage entries via the web interface at [http://0.0.0.0:8081/db/speaker_recognition/speakers](http://0.0.0.0:8081/db/speaker_recognition/speakers).

#### Formatting code - Pre-commit Hooks (Optional but Recommended)

This repository uses **Ruff** for automatic Python code formatting via pre-commit hooks.

**Quick Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks 
pre-commit install # Runs on changed files only by default when git commit

# (Optional) Run on all existing files
pre-commit run --all-files

#If you need to commit urgently and skip the pre-commit checks
git commit -m "urgent fix" --no-verify
```
Now Ruff will automatically format your code before each commit. If formatting changes are made, review them with `git diff`, then stage and commit again.

Follow [PRECOMMIT.md](./PRECOMMIT.md) for detailed instructions and troubleshooting tips related to pre-commit hooks.

## Troubleshooting

### Port 27017 Already in Use

If you encounter the error `failed to bind host port for 0.0.0.0:27017:172.21.0.2:27017/tcp: address already in use`, this means another service is already occupying port 27017. The docker-compose MongoDB service cannot start because the port is blocked. To resolve this, identify and stop the conflicting service with `sudo lsof -i :27017` and kill the process if needed, then restart docker-compose. 

```bash
sudo lsof -ti:27017 | xargs -r sudo kill -9
```
### Container Name Conflicts

If you switch between `dev-docker-compose.yaml` and `docker-compose.yaml`, you may encounter errors like `Conflict. The container name "/mongodb_faces" is already in use`. This happens because containers from the previous compose file are still running. To resolve this, remove all containers and restart: 
```bash
docker rm -f $(docker ps -aq)
```
then run `docker compose up` again. This cleanly removes all existing containers and allows the new composition to start fresh.


### Setup for Local Testing

1. **Configure secrets** (if needed for your workflow):
   ```bash
   # Create a secrets file
   touch .secrets
   
   # Add your secrets (example):
   echo "HF_TOKEN=your_huggingface_token_here" >> .secrets
   ```

   ⚠️ **Important**: Don't commit the `.secrets` file to GitHub! Add it to `.gitignore`:
   ```bash
   echo ".secrets" >> .gitignore
   ```

### Running CI/CD Locally

Follow [CI CD Readme](CI_CD_SETUP.md) for detailed instructions on how to run GitHub Actions workflows locally.

### Authors
- [Josep Bravo](https://github.com/LeBrav)
- [Joan Omedes](https://github.com/joan-omedes)
- [Devis Dal Moro](https://github.com/devis12)