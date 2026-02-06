"""
Test configuration for speech_recognition package.
Ensures proper Python environment setup for speech/AI dependencies.
"""
import os
import sys
from unittest.mock import MagicMock

# Mock silero_vad.data immediately to prevent import errors during collection
mock_silero_data = MagicMock()
sys.modules['silero_vad.data'] = mock_silero_data

# Setup environment for speech_recognition (uses unified ros_python_env)
DIARIZATION_ENV_PATH = "/opt/ros_python_env"
if os.path.exists(DIARIZATION_ENV_PATH):
    # Add diarization venv site-packages to Python path
    diarization_site_packages = os.path.join(DIARIZATION_ENV_PATH, "lib/python3.12/site-packages")
    if os.path.exists(diarization_site_packages) and diarization_site_packages not in sys.path:
        sys.path.insert(0, diarization_site_packages)
        print(f"[pytest] Using unified environment: {DIARIZATION_ENV_PATH}")

# Add src to Python path for tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Mock heavy AI/ML dependencies for testing
def setup_ml_mocks():
    """Setup mock dependencies for ML/AI testing without requiring full models"""
    from unittest.mock import MagicMock, patch
    
    # Mock torch and torchaudio
    try:
        import torch
        import torchaudio
        print(f"[pytest] Using real PyTorch: {torch.__version__}")
    except ImportError:
        print("[pytest] Mocking PyTorch dependencies")
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0+mock"
        mock_torch.hub.set_dir = MagicMock()
        mock_torch.hub.load = MagicMock()
        mock_torch.tensor = MagicMock()
        mock_torch.jit.load = MagicMock()
        
        mock_torchaudio = MagicMock()
        mock_torchaudio.functional.resample = MagicMock()
        
        sys.modules['torch'] = mock_torch
        sys.modules['torchaudio'] = mock_torchaudio
    
    # Mock faster_whisper
    try:
        import faster_whisper
    except ImportError:
        print("[pytest] Mocking faster_whisper")
        mock_whisper = MagicMock()
        mock_whisper.WhisperModel = MagicMock()
        mock_whisper.BatchedInferencePipeline = MagicMock()
        sys.modules['faster_whisper'] = mock_whisper
    
    # Mock diart
    try:
        import diart
    except ImportError:
        print("[pytest] Mocking diart")
        mock_diart = MagicMock()
        mock_diart.SpeakerDiarization = MagicMock()
        mock_diart.SpeakerDiarizationConfig = MagicMock()
        mock_diart.models = MagicMock()
        sys.modules['diart'] = mock_diart
        sys.modules['diart.models'] = mock_diart.models
    
    # Mock pyannote
    try:
        import pyannote
    except ImportError:
        print("[pytest] Mocking pyannote")
        mock_pyannote = MagicMock()
        mock_pyannote.core.Annotation = MagicMock()
        sys.modules['pyannote'] = mock_pyannote
        sys.modules['pyannote.core'] = mock_pyannote.core
    
    # Mock openwakeword
    try:
        import openwakeword
    except ImportError:
        print("[pytest] Mocking openwakeword")
        sys.modules['openwakeword'] = MagicMock()
    
    # Mock silero_vad module that's causing the import error
    mock_silero = MagicMock()
    mock_silero.model.load_silero_vad = MagicMock()
    mock_silero.utils_vad.init_jit_model = MagicMock()
    mock_silero.utils_vad.OnnxWrapper = MagicMock()
    mock_silero.data = MagicMock()  # Mock the missing data submodule
    sys.modules['silero_vad'] = mock_silero
    sys.modules['silero_vad.model'] = mock_silero.model
    sys.modules['silero_vad.utils_vad'] = mock_silero.utils_vad
    sys.modules['silero_vad.data'] = mock_silero.data
    
    print("[pytest] ML dependency mocking completed")


# Setup mocks before any imports
setup_ml_mocks()


# Basic pytest configuration
def pytest_configure(config):
    """Configure pytest for speech recognition tests"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for speech components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with ROS"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that require model loading"
    )