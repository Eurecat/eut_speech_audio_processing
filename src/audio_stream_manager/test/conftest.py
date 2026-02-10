"""
Test configuration for audio_stream_manager package.
Ensures proper Python environment setup for audio dependencies.
"""

import gc
import os
import sys

import pytest

# Setup environment for audio_stream_manager (uses ros_python_env)
VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")
if os.path.exists(VENV_PATH):
    # Add venv site-packages to Python path
    venv_site_packages = os.path.join(VENV_PATH, "lib/python3.12/site-packages")
    if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
        print(f"[pytest] Using audio environment: {VENV_PATH}")

# Add src to Python path for tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Mock audio dependencies if not available
def setup_audio_mocks():
    """Setup mock dependencies for audio testing"""
    from unittest.mock import MagicMock

    # Mock sounddevice if not available
    try:
        import sounddevice
    except ImportError:
        sys.modules["sounddevice"] = MagicMock()
        print("[pytest] Mocked sounddevice")

    # Mock numpy if not available (should be available but just in case)
    try:
        import numpy
    except ImportError:
        sys.modules["numpy"] = MagicMock()
        print("[pytest] Mocked numpy")

    # Mock ROS if not available
    try:
        import rclpy
    except ImportError:
        sys.modules["rclpy"] = MagicMock()
        sys.modules["rclpy.node"] = MagicMock()
        sys.modules["rclpy.parameter"] = MagicMock()
        print("[pytest] Mocked rclpy")


# Setup mocks
setup_audio_mocks()


@pytest.fixture(scope="session", autouse=True)
def setup_tests():
    """Setup test environment once per session"""
    # Suppress ALSA errors in CI environment
    os.environ["ALSA_PCM_CARD"] = "0"
    os.environ["ALSA_PCM_DEVICE"] = "0"
    yield


# Basic pytest configuration
def pytest_configure(config):
    """Configure pytest for audio stream manager tests"""
    config.addinivalue_line("markers", "unit: Unit tests for audio components")
    config.addinivalue_line("markers", "integration: Integration tests with ROS")


# Memory optimization hooks
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Force garbage collection after each test to prevent memory buildup"""
    yield
    gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """Force garbage collection between tests to free memory"""
    gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup after all tests complete"""
    gc.collect()
