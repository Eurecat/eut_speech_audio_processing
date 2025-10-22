# eut_speech_audio_processing

## Overview

This repository contains the **speech and audio processing module** for the perception layer of robotic systems. The module provides comprehensive audio processing capabilities designed to enable robots to understand and interact with their acoustic environment.

### Architecture

The system is designed with two **decoupled components** to ensure robust error handling and system reliability:

#### **Audio Stream Manager**
This submodule interfaces directly with audio hardware devices to capture raw audio streams, isolating hardware-related issues from the speech processing logic.

#### **Speech Recognition Pipeline**
This submodule contains several audio processing and understanding capabilities that enable real-time analysis of acoustic environments, including **speech detection**, **speaker identification**, and  **transcription**. 

This module receives as input the processed audio output from the *Audio Stream Manager*. 

It is composed of:
  - **Voice Activity Detection (VAD)**: Detects when speech is present in the audio stream
  - **Speaker Diarization**: Identifies and segments different speakers in multi-speaker scenarios
  - **Speech Transcription**: Converts spoken language into text using automatic speech recognition (ASR)
  - **Wake Word Detection**: *(Future enhancement)* Keyword spotting for voice activation

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

Configure your Hugging Face token in the `.env` file with appropriate permissions for the required models. 

**For quick testing**, you can add this token to your `.env` file: `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`

## Usage

### Launch Audio Stream Manager
Execute the following command to initialize the audio input device selection module:

```bash
ros2 launch audio_stream_manager audio_stream_manager.launch.py
```

### Launch Speech Recognition Pipeline
In a separate terminal, execute the following command to start the voice activity detection, speaker diarization, and automatic speech recognition modules:

```bash
ros2 launch speech_recognition speech_recognition.launch.py
```
