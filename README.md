# eut_speech_audio_processing

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

#### Step 1.5: Setup Pre-commit Hooks (Optional but Recommended)

This repository uses **Ruff** for automatic Python code formatting via pre-commit hooks.

**Quick Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

Now Ruff will automatically format your code before each commit. If formatting changes are made, review them with `git diff`, then stage and commit again.

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

## CI/CD Testing with Act

To test the GitHub Actions CI/CD pipeline locally without pushing to GitHub, you can use [Act](https://github.com/nektos/act), which runs GitHub Actions locally using Docker.

### Prerequisites

1. **Install Go** (if not already installed):
   - Follow the installation guide at: https://go.dev/doc/install

2. **Install Act**:
   ```bash
   # Clone and install Act
   git clone https://github.com/nektos/act.git
   cd act
   ./install.sh
   ```

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

1. **Run the complete workflow**:
   ```bash
   ./act/bin/act
   ```

2. **Select runner size** when prompted:
   - Choose **Medium** for most cases (recommended)
   - Use **Large** for resource-intensive builds
   - Use **Micro** for simple tests

3. **Run specific job**:
   ```bash
   # List available jobs first
   ./act/bin/act --list
   
   # Run only the test job
   ./act/bin/act -j test
   
   # Run only the deploy job
   ./act/bin/act -j deploy
   
   # Run with secrets file
   ./act/bin/act --secret-file .secrets
   
   # Run specific job with secrets
   ./act/bin/act -j test --secret-file .secrets
   ```

### Benefits of Local Testing

- **Faster feedback**: Test your CI/CD changes without pushing to GitHub
- **Cost-effective**: No GitHub Actions minutes consumed
- **Debugging**: Easier to debug workflow issues locally
- **Iterative development**: Quickly iterate on workflow changes

This allows you to validate your CI/CD pipeline changes before committing and pushing to the repository.


