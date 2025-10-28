# eut_speech_audio_processing

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

**Prerequisites:**
- Your default SSH keys will be used during the image build process
- Eurecat VPN access may be required to pull dependencies from private GitLab repositories

#### Step 0: Build Base Image
First, build the required base Docker image from [EutRobAIDockers](https://github.com/Eurecat/EutRobAIDockers).

#### Step 1: Clone Repository
```bash
git clone git@github.com:Eurecat/eut_speech_audio_processing.git
cd eut_speech_audio_processing
```

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
