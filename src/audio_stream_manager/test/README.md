# Testing Guide for audio_stream_manager Package

This directory contains unit and integration tests for the `audio_stream_manager` package, demonstrating ROS 2 testing with audio processing capabilities using pytest and launch_pytest frameworks.

## Test Structure

The package implements a **three-layer testing approach**:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Integration Tests                                                    │
│  (test_audio_stream_integration.py)                                  │
│  - Launch actual audio capturing nodes                               │
│  - End-to-end audio streaming validation                             │
│  - Uses launch_pytest framework                                      │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   │ Validates ROS Interface
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Unit Tests                                                           │
│  (test_audio_capturing_node.py, test_save_mp3.py)                   │
│  - Pure logic testing without ROS dependencies                       │
│  - Audio device management and processing                            │
│  - File I/O and format conversion                                    │
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
└──────────────────────────────────────────────────────────────────────┘
```

## Test Files

### 1. **`test_audio_capturing_node.py`** (Pure Unit Tests)
Tests the core audio capturing functionality without ROS dependencies:

- **SuppressStderr Tests**: Context manager for suppressing audio device errors
- **AudioCapturingNode Tests**: Audio device detection and management
- **Audio Processing**: Format conversion and buffer management  
- **Device Handling**: Connection/disconnection scenarios
- **Error Cases**: Malformed audio data and device failures

Key test methods:
```python
def test_node_initialization()          # Basic node setup
def test_audio_device_detection()       # Device enumeration
def test_audio_format_conversion()      # Audio data processing
def test_device_disconnection()         # Error handling
```

### 2. **`test_save_mp3.py`** (Audio File Processing Unit Tests)
Tests audio recording and file conversion capabilities:

- **Recording Logic**: Audio buffer management and timing
- **MP3 Conversion**: Audio format conversion and encoding
- **File Operations**: Audio file saving and validation
- **Edge Cases**: Empty buffers, invalid formats

### 3. **`test_audio_stream_integration.py`** (ROS Integration Tests)
Tests the complete audio streaming pipeline with ROS:

- **Node Lifecycle**: Launch and shutdown of audio nodes
- **Topic Validation**: Audio message publication on topics
- **Message Flow**: End-to-end audio data streaming
- **ROS Parameters**: Configuration and parameter handling

Uses launch_pytest framework:
```python
@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # Launch audio capturing nodes
```

### 4. **`test_smoke.py`** (Basic Smoke Tests)
Quick validation tests:

- **Import Tests**: Verify all modules can be imported
- **Instantiation**: Basic object creation without errors
- **Dependencies**: Check required packages are available

### 5. **`conftest.py`** (Test Configuration)
Configures the testing environment:

- **Environment Setup**: Configures AI/audio Python environment
- **Mock Setup**: Mocks audio dependencies when not available  
- **Path Configuration**: Ensures proper module importing

```python
VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")
```

## Environment Requirements

### Prerequisites

The tests require access to audio processing libraries:

```bash
# Set the AI environment path
export AI_VENV="/opt/ros_python_env"  # or your audio processing venv path

# Ensure audio dependencies are installed in the environment:
# - sounddevice
# - numpy 
# - scipy (for audio processing)
```

## Running the Tests

### Build the package first
```bash
cd /path/to/workspace
colcon build --packages-select audio_stream_manager
```

### Run all tests for audio_stream_manager
```bash
colcon test --packages-select audio_stream_manager
colcon test-result --verbose
```

### Run with detailed output
```bash
colcon test --packages-select audio_stream_manager --event-handlers console_direct+ --pytest-args '-v'
```

### Run specific test categories

#### Unit tests only
```bash
colcon test --packages-select audio_stream_manager --pytest-args 'test/test_audio_capturing_node.py test/test_save_mp3.py -v'
```

#### Integration tests only  
```bash
colcon test --packages-select audio_stream_manager --pytest-args 'test/test_audio_stream_integration.py -v'
```

#### Smoke tests only
```bash
colcon test --packages-select audio_stream_manager --pytest-args 'test/test_smoke.py -v'
```

### Run specific test functions
```bash
colcon test --packages-select audio_stream_manager --pytest-args 'test/test_audio_capturing_node.py::TestAudioCapturingNode::test_node_initialization -v'
```

### Run tests with coverage
```bash
colcon test --packages-select audio_stream_manager --pytest-args '--cov=audio_stream_manager --cov-report=html'
```

### Debug mode with output
```bash
colcon test --packages-select audio_stream_manager --pytest-args '-v -s --log-cli-level=DEBUG'
```

## Test Coverage

The test suite validates:

### ✅ **Audio Device Management**
- Audio device detection and enumeration
- Device connection and disconnection handling
- Audio format support validation
- Error recovery mechanisms

### ✅ **Audio Processing**
- Real-time audio capture and buffering
- Audio format conversion (sample rates, channels)
- Audio data validation and processing
- Buffer management and overflow handling

### ✅ **File Operations**
- Audio recording to various formats
- MP3 encoding and conversion
- File I/O error handling
- Audio metadata processing

### ✅ **ROS Integration**
- Node lifecycle management
- Topic registration and publishing
- Audio message format validation
- Parameter configuration and updates
- Inter-node communication

### ✅ **Error Handling**
- Device unavailable scenarios
- Invalid audio format handling
- Network and I/O error recovery
- Graceful degradation

## Testing Best Practices

### Mock Usage
Tests extensively use mocking for:
- Audio device simulation (when hardware unavailable)
- File system operations
- ROS node dependencies
- Error condition simulation

### Parametrized Testing
Many tests use pytest parametrization:
```python
@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100, 48000])
def test_different_sample_rates(self, sample_rate):
    # Test with various audio sample rates
```

### Edge Case Coverage
- Empty audio buffers
- Malformed audio data  
- Device disconnection during recording
- Insufficient disk space scenarios
- Invalid configuration parameters

## Continuous Integration

The test suite is designed for CI/CD:

- **Fast unit tests** (< 15 seconds total)
- **Audio integration tests** (< 1 minute)
- **Smoke tests** (< 5 seconds)
- **No hardware dependencies** (mocked when unavailable)
- **Deterministic results** in containerized environments

## Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   ```bash
   # Tests should work with mocked devices
   # For real device tests, ensure audio hardware is available
   ```

2. **Missing Audio Dependencies**
   ```bash
   # Ensure AI environment is set correctly
   export AI_VENV="/path/to/your/audio/env"
   ```

3. **Permission Errors**
   ```bash
   # Audio device access might require permissions
   # Tests use mocks by default to avoid this
   ```

### Debug Audio Issues
```bash
# Check available audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Run with detailed audio logging
colcon test --packages-select audio_stream_manager --pytest-args '-v -s --log-cli-level=DEBUG'
```

## Integration with Other Packages

This package tests integrate with:
- **speech_recognition**: Audio data flows to speech processing
- **hri_msgs**: Message format compatibility
- **audio_common_msgs**: Standard audio message types

## Additional Resources

- [ROS 2 Testing with Python](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Testing/Python.html)
- [pytest Documentation](https://docs.pytest.org/)
- [launch_pytest](https://github.com/ros2/launch/tree/rolling/launch_pytest)
- [SoundDevice Documentation](https://python-sounddevice.readthedocs.io/)

## Summary

This comprehensive test suite provides:

- ✅ **Layered testing approach**: Unit → Integration → System
- ✅ **Audio processing validation**: Device management and data processing
- ✅ **ROS 2 integration**: Complete node lifecycle and communication testing
- ✅ **Hardware independence**: Mocked dependencies for CI/CD
- ✅ **Performance validation**: Real-time audio processing capabilities
- ✅ **Error resilience**: Edge cases and failure mode coverage

The testing framework ensures both **audio processing correctness** and **ROS communication reliability** for real-time audio streaming applications.