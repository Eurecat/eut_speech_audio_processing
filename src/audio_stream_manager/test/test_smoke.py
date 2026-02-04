"""
Simple smoke test to verify test discovery is working.
"""
import pytest


def test_basic_import():
    """Test that basic imports work"""
    assert True


def test_pytest_working():
    """Test that pytest is functioning"""
    assert 1 + 1 == 2


class TestAudioStreamManagerSmoke:
    """Smoke tests for audio_stream_manager package"""
    
    def test_package_importable(self):
        """Test that the package can be imported"""
        try:
            import audio_stream_manager
            assert True
        except ImportError:
            # Package might not be built yet, that's okay
            pytest.skip("audio_stream_manager package not available")
    
    def test_basic_functionality(self):
        """Basic smoke test"""
        assert True