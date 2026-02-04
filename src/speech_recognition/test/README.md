# Testing Guide for speech_recognition Package

This directory contains comprehensive unit and integration tests for the `speech_recognition` package, demonstrating ROS 2 testing with AI/ML models using pytest and launch_pytest frameworks.

## Test Structure

The package implements a **four-layer testing approach**:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Pipeline Integration Tests                                           │
│  (test_speech_pipeline_integration.py)                               │
│  - Complete speech processing pipeline                               │
│  - VAD + ASR + Diarization coordination                              │
│  - End-to-end speech-to-text validation                              │
│  - Uses launch_pytest framework                                      │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   │ Validates Complete Pipeline
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Unit Tests                                                           │
│  (test_vad.py, test_asr.py, test_diarization.py)                    │
│  - Individual component testing                                      │
│  - AI/ML model validation                                            │
│  - Pure logic without ROS dependencies                               │
│  - Fast, isolated validation                                         │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   │ Supports
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Smoke Tests                                                          │
│  (test_smoke.py)                                                     │
│  - Basic import and instantiation tests                              │
│  - Quick validation of package integrity                             │
│  - Model loading verification                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Test Files

### 1. **`test_vad.py`** (Voice Activity Detection Unit Tests)
Tests voice activity detection functionality:

- **Node Initialization**: VAD node setup and parameter validation
- **Model Loading**: Silero VAD model integration and configuration
- **Audio Processing**: Voice activity detection on various audio samples
- **Confidence Thresholding**: Speech/non-speech classification logic
- **Device Management**: GPU/CPU device selection and optimization

Key test methods:
```python
def test_node_initialization()          # Basic VAD node setup
def test_parameter_declarations()       # Configuration validation  
def test_vad_audio_processing()         # Voice detection logic
def test_confidence_thresholding()      # Classification accuracy
def test_device_selection()             # Hardware optimization
```

### 2. **`test_asr.py`** (Automatic Speech Recognition Unit Tests)
Tests speech-to-text functionality:

- **Node Setup**: ASR node initialization and model loading
- **Whisper Integration**: Faster-whisper model configuration
- **Audio Chunking**: Audio segmentation and preprocessing
- **Language Detection**: Multi-language speech recognition
- **Transcription Logic**: Speech-to-text conversion accuracy

Key test methods:
```python
def test_node_initialization()          # ASR node setup
def test_model_loading()                # Whisper model initialization
def test_audio_chunking()               # Audio preprocessing
def test_transcription()                # Speech-to-text conversion
def test_language_detection()           # Multi-language support
```

### 3. **`test_diarization.py`** (Speaker Diarization Unit Tests)
Tests speaker identification and tracking:

- **Node Initialization**: Diarization node setup and configuration
- **Speaker Modeling**: Speaker embedding extraction and clustering
- **Multi-Speaker Handling**: Speaker separation and identification
- **Temporal Segmentation**: Speaker timeline tracking
- **Performance Optimization**: Real-time processing validation

Key test methods:
```python
def test_node_initialization()          # Diarization node setup
def test_speaker_embedding()            # Speaker feature extraction
def test_multi_speaker_detection()      # Speaker separation
def test_speaker_tracking()             # Temporal consistency
```

### 4. **`test_speech_pipeline_integration.py`** (Complete Pipeline Tests)
Tests the integrated speech processing system:

- **Node Coordination**: Multi-node launch and communication
- **Data Flow**: Audio → VAD → ASR → Diarization pipeline
- **Message Validation**: ROS message format and timing
- **End-to-End Testing**: Complete speech processing workflow
- **Performance Testing**: Real-time processing capabilities

Uses launch_pytest framework:
```python
@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # Launch VAD, ASR, and diarization nodes
    vad_node = launch_ros.actions.Node(package='speech_recognition', executable='vad')
    asr_node = launch_ros.actions.Node(package='speech_recognition', executable='asr')
    diarization_node = launch_ros.actions.Node(package='speech_recognition', executable='diarization')
```

### 5. **`test_smoke.py`** (Basic Smoke Tests)
Quick validation tests:

- **Import Tests**: Verify all modules can be imported
- **Model Availability**: Check AI/ML models are accessible
- **Dependencies**: Validate PyTorch, Whisper, Silero installations
- **Basic Instantiation**: Node creation without errors

### 6. **`conftest.py`** (Test Configuration)
Configures the AI/ML testing environment:

- **Environment Setup**: Configures diarization Python environment
- **Model Mocking**: Mocks heavy AI dependencies for fast testing
- **Path Configuration**: Ensures proper AI package importing
- **Early Mock Setup**: Prevents import errors during test collection

```python
DIARIZATION_ENV_PATH = "/opt/ros_python_diarization_env"

# Mock heavy AI/ML dependencies
mock_silero = MagicMock()
sys.modules['silero_vad'] = mock_silero
sys.modules['silero_vad.data'] = mock_silero.data
```

## Environment Requirements

### Prerequisites

The tests require access to AI/ML processing libraries:

```bash
# Set the diarization environment path  
export AI_VENV="/opt/ros_python_diarization_env"

# Ensure AI dependencies are installed in the environment:
# - torch (PyTorch)
# - torchaudio
# - faster-whisper
# - silero-vad
# - diart (speaker diarization)
# - pyannote.audio
```

### Environment Dependencies

The speech recognition package uses specialized AI environment:
- **PyTorch**: Deep learning framework
- **Whisper**: Speech recognition models
- **Silero VAD**: Voice activity detection
- **Diart**: Speaker diarization toolkit
- **PyAnnote**: Audio processing and segmentation

## Running the Tests

### Build the package first
```bash
cd /path/to/workspace

# Ensure dependencies are built first
colcon build --packages-select hri_msgs audio_common_msgs

# Build speech recognition package
colcon build --packages-select speech_recognition
```

### Run all tests for speech_recognition
```bash
colcon test --packages-select speech_recognition
colcon test-result --verbose
```

### Run with detailed output
```bash
colcon test --packages-select speech_recognition --event-handlers console_direct+ --pytest-args '-v'
```

### Run specific test categories

#### Unit tests only (fast)
```bash
colcon test --packages-select speech_recognition --pytest-args 'test/test_vad.py test/test_asr.py test/test_diarization.py -v'
```

#### Integration tests only
```bash
colcon test --packages-select speech_recognition --pytest-args 'test/test_speech_pipeline_integration.py -v'
```

#### Smoke tests only (fastest)
```bash
colcon test --packages-select speech_recognition --pytest-args 'test/test_smoke.py -v'
```

### Run specific test functions
```bash
colcon test --packages-select speech_recognition --pytest-args 'test/test_vad.py::TestVADNode::test_node_initialization -v'
```

### Run tests with coverage
```bash
colcon test --packages-select speech_recognition --pytest-args '--cov=speech_recognition --cov-report=html'
```

### Debug mode with AI model output
```bash
colcon test --packages-select speech_recognition --pytest-args '-v -s --log-cli-level=DEBUG'
```

### Run tests with specific markers
```bash
# Run only unit tests
colcon test --packages-select speech_recognition --pytest-args '-m unit -v'

# Run only integration tests
colcon test --packages-select speech_recognition --pytest-args '-m integration -v'

# Skip slow model loading tests
colcon test --packages-select speech_recognition --pytest-args '-m "not slow" -v'
```

## Test Coverage

The test suite validates:

### ✅ **Voice Activity Detection (VAD)**
- Silero VAD model loading and initialization
- Audio preprocessing and normalization
- Voice activity classification accuracy
- Confidence score thresholding
- Real-time VAD processing pipeline

### ✅ **Automatic Speech Recognition (ASR)**
- Faster-Whisper model integration
- Multi-language speech recognition
- Audio chunking and preprocessing
- Transcription accuracy and confidence
- Language detection capabilities

### ✅ **Speaker Diarization**
- Speaker identification and embedding extraction
- Multi-speaker scenario handling
- Speaker tracking across time segments
- Clustering and classification accuracy
- Real-time diarization performance

### ✅ **ROS Integration**
- Node lifecycle management (launch, shutdown)
- Topic registration and message publication
- Inter-node communication validation
- Message format compatibility
- Parameter configuration and updates

### ✅ **AI/ML Model Integration**
- Model loading and initialization
- GPU/CPU device selection
- Memory management for large models
- Inference performance optimization
- Error handling for model failures

### ✅ **Performance Validation**
- Real-time processing capabilities
- Latency and throughput measurements
- Memory usage monitoring
- CPU/GPU utilization optimization

## Testing Best Practices

### AI/ML Model Mocking
Tests extensively use mocking for AI models to enable:
- **Fast test execution** without loading heavy models
- **CI/CD compatibility** in environments without GPU
- **Deterministic results** independent of model versions
- **Isolation** of logic from model implementation

### Parametrized Testing
Many tests use pytest parametrization for comprehensive coverage:
```python
@pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium"])
def test_different_whisper_models(self, model_size):
    # Test with different Whisper model sizes

@pytest.mark.parametrize("language", ["en", "es", "fr", "de"])
def test_multi_language_recognition(self, language):
    # Test speech recognition in different languages
```

### Performance Testing
Dedicated performance validation includes:
- Audio processing latency measurements
- Real-time factor calculations
- Memory usage monitoring
- Throughput testing with various audio lengths

### Error Simulation
Comprehensive error handling tests:
- Malformed audio data scenarios
- Model loading failures
- GPU memory exhaustion
- Network connectivity issues
- Invalid configuration parameters

## Continuous Integration

The test suite is designed for CI/CD environments:

- **Fast unit tests** (< 30 seconds with mocking)
- **Moderate integration tests** (< 2 minutes per component)
- **AI model independence** (mocked for CI, real for development)
- **Parallel execution** support
- **Deterministic results** in containerized environments

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Ensure AI environment is configured
   export AI_VENV="/opt/ros_python_diarization_env"
   
   # Check PyTorch installation
   python3 -c "import torch; print(torch.__version__)"
   ```

2. **GPU/CUDA Issues**
   ```bash
   # Tests should work with CPU fallback
   # Check CUDA availability
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Missing AI Dependencies**
   ```bash
   # Install missing packages in AI environment
   /opt/ros_python_diarization_env/bin/pip install faster-whisper diart pyannote.audio
   ```

4. **Import Errors**
   ```bash
   # Clear Python cache
   find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
   
   # Rebuild package
   colcon build --packages-select speech_recognition --cmake-clean-cache
   ```

### Debug AI Model Issues
```bash
# Test individual model loading
python3 -c "
import torch
from faster_whisper import WhisperModel
model = WhisperModel('tiny')
print('Whisper model loaded successfully')
"

# Check Silero VAD
python3 -c "
import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
print('Silero VAD loaded successfully')
"
```

## Integration with Other Packages

This package integrates with:
- **audio_stream_manager**: Receives audio data from microphones
- **hri_msgs**: Uses standard HRI message formats
- **audio_common_msgs**: Compatible with ROS audio ecosystem

## Performance Benchmarks

Typical performance on standard hardware:
- **VAD Processing**: < 10ms per audio chunk
- **ASR Processing**: Real-time factor < 0.3 (faster than real-time)
- **Diarization**: Real-time factor < 1.0 for up to 4 speakers
- **Memory Usage**: < 2GB for all models combined

## Additional Resources

- [ROS 2 Testing with Python](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Testing/Python.html)
- [pytest Documentation](https://docs.pytest.org/)
- [launch_pytest](https://github.com/ros2/launch/tree/rolling/launch_pytest)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/testing.html)
- [Faster-Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD Models](https://github.com/snakers4/silero-vad)

## Summary

This comprehensive test suite provides:

- ✅ **Layered testing approach**: Unit → Integration → Pipeline
- ✅ **AI/ML integration**: Proper PyTorch and model testing
- ✅ **ROS 2 best practices**: launch_pytest and message validation
- ✅ **Performance validation**: Real-time processing capabilities
- ✅ **CI/CD compatibility**: Fast execution with model mocking
- ✅ **Multi-language support**: International speech recognition
- ✅ **Error resilience**: Comprehensive failure mode testing

The testing framework ensures both **algorithmic correctness** and **ROS communication reliability** across the complete speech processing pipeline from voice detection to transcription and speaker identification.