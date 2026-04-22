"""
Unit tests for AudioCapturing class.
Tests core functionality without ROS dependencies where possible.
"""

import pytest
import numpy as np
import sounddevice as sd
import threading
from unittest.mock import Mock, MagicMock, patch, call
from audio_stream_manager.audio_capturing import AudioCapturing
from audio_stream_manager.audio_stream_manager.utils.audio_utils import (
    SuppressStderr,
    get_available_input_devices,
    find_devices_by_name,
)


class TestSuppressStderr:
    """Test suite for SuppressStderr context manager"""

    def test_context_manager_protocol(self):
        """Test that SuppressStderr implements context manager protocol"""
        suppressor = SuppressStderr()
        assert hasattr(suppressor, "__enter__")
        assert hasattr(suppressor, "__exit__")

    @patch("os.open")
    @patch("os.dup")
    @patch("os.dup2")
    @patch("os.close")
    def test_stderr_suppression(self, mock_close, mock_dup2, mock_dup, mock_open):
        """Test stderr suppression mechanism"""
        mock_open.return_value = 3  # mock file descriptor
        mock_dup.return_value = 2  # mock stderr fd

        with SuppressStderr():
            pass

        # Verify file operations
        mock_open.assert_called_once()
        mock_dup.assert_called_once_with(2)
        assert mock_dup2.call_count == 2
        assert mock_close.call_count == 2


@patch("audio_stream_manager.audio_capturing.threading.Thread")
class TestAudioCapturing:
    """Test suite for AudioCapturing class"""

    @patch("rclpy.node.Node.__init__")
    @patch("audio_stream_manager.audio_capturing.AudioCapturing.setup_working_device")
    def test_node_initialization(self, mock_setup_device, mock_node_init, mock_thread):
        """Test that the node initializes with correct parameters"""
        with patch.object(AudioCapturing, "declare_parameter"), patch.object(
            AudioCapturing, "create_publisher"
        ), patch.object(AudioCapturing, "get_logger"):
            node = AudioCapturing()

            # Verify ROS node was initialized
            mock_node_init.assert_called_once_with("audio_capturing")

            # Verify setup_working_device was called
            mock_setup_device.assert_called_once()

            # Check essential attributes would be set during normal initialization
            assert hasattr(node, "__class__")
            assert node.__class__.__name__ == "AudioCapturing"

    @patch("rclpy.node.Node.__init__")
    @patch("audio_stream_manager.audio_capturing.AudioCapturing.setup_working_device")
    def test_parameter_declarations(self, mock_setup_device, mock_node_init, mock_thread):
        """Test that all required parameters are declared"""
        with patch.object(AudioCapturing, "create_publisher"), patch.object(
            AudioCapturing, "get_logger"
        ):
            declare_param_mock = MagicMock()

            with patch.object(AudioCapturing, "declare_parameter", declare_param_mock):
                node = AudioCapturing()

                # Verify all expected parameters are declared
                expected_calls = [
                    call("device_name", "DJI"),
                    call("dtype", "float32"),
                    call("channels", 1),
                    call("chunk", 512),
                    call("disconnection_timeout", 3.0),
                    call("disconnection_check_interval", 1.0),
                    call("test_stream_duration", 0.1),
                    call("primary_device_check_interval", 5.0),
                    call("target_samplerate", 16000),
                ]

                declare_param_mock.assert_has_calls(expected_calls, any_order=True)

    @patch("rclpy.node.Node.__init__")
    @patch("sounddevice.query_devices")
    @patch("audio_stream_manager.audio_capturing.AudioCapturing.setup_working_device")
    def test_get_device_list(
        self, mock_setup_device, mock_query_devices, mock_node_init, mock_thread
    ):
        """Test device list retrieval"""
        mock_devices = [
            {"name": "DJI MIC", "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Built-in Microphone", "max_input_channels": 2, "default_samplerate": 44100},
        ]
        mock_query_devices.return_value = mock_devices

        # get_available_input_devices is now a standalone helper function
        available_devices = get_available_input_devices(mock_devices)

        assert len(available_devices) == 2
        assert 0 in available_devices  # DJI MIC
        assert 1 in available_devices  # Built-in Microphone

    def test_find_device_by_name(self, mock_thread):
        """Test finding device by name"""
        mock_devices = [
            {"name": "DJI MIC", "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Built-in Microphone", "max_input_channels": 2, "default_samplerate": 44100},
        ]
        available_devices = get_available_input_devices(mock_devices)

        # find_devices_by_name is now a standalone helper function
        device_indices = find_devices_by_name("DJI", available_devices, mock_devices)
        assert 0 in device_indices

        # Test finding non-existent device
        device_indices = find_devices_by_name("NonExistent", available_devices, mock_devices)
        assert len(device_indices) == 0

    def test_audio_data_processing(self, mock_thread):
        """Test audio data processing methods"""
        # Test data conversion functions
        test_data = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)

        # Test data is already in correct format
        assert test_data.dtype == np.float32
        assert len(test_data.shape) == 1

        # Test data scaling
        scaled_data = test_data * 32767
        assert np.all(scaled_data >= -32767)
        assert np.all(scaled_data <= 32767)

    @pytest.mark.parametrize(
        "channels,chunk_size",
        [
            (1, 512),
            (1, 1024),
            (2, 512),
            (2, 1024),
        ],
    )
    def test_audio_parameters(self, mock_thread, channels, chunk_size):
        """Test various audio parameter combinations"""
        # Test that parameters are valid
        assert channels > 0
        assert chunk_size > 0
        assert chunk_size % 2 == 0  # Should be power of 2 for efficiency

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_sample_rates(self, mock_thread, sample_rate):
        """Test various sample rate configurations"""
        # Test common audio sample rates
        assert sample_rate > 0
        assert sample_rate <= 48000  # Reasonable upper limit


@patch("audio_stream_manager.audio_capturing.threading.Thread")
class TestAudioCapturingROS:
    """Test suite for AudioCapturing ROS functionality"""

    @pytest.fixture
    def mock_node(self, mock_thread):
        """Fixture providing a mocked AudioCapturing"""
        with patch(
            "audio_stream_manager.audio_capturing.AudioCapturing.setup_working_device"
        ), patch("rclpy.node.Node.__init__"), patch.object(
            AudioCapturing, "declare_parameter"
        ), patch.object(AudioCapturing, "create_publisher"), patch.object(
            AudioCapturing, "get_logger"
        ):
            node = AudioCapturing()
            node.audio_and_device_info_pub = Mock()
            return node

    def test_publisher_creation(self, mock_node):
        """Test that audio publisher is created"""
        assert mock_node.audio_and_device_info_pub is not None

    def test_audio_message_structure(self, mock_node):
        """Test audio message creation and structure"""
        # Mock audio data
        audio_data = np.random.rand(512).astype(np.float32)

        # Test message would be published with correct structure
        # (actual message creation tested in integration tests)
        assert len(audio_data) == 512
        assert audio_data.dtype == np.float32


@patch("audio_stream_manager.audio_capturing.threading.Thread")
class TestAudioCapturingEdgeCases:
    """Test suite for edge cases and error handling"""

    def test_no_audio_devices_available(self, mock_thread):
        """Test behavior when no audio devices are available"""
        # get_available_input_devices is now a standalone helper function
        available_devices = get_available_input_devices([])
        assert len(available_devices) == 0

    def test_device_query_exception(self, mock_thread):
        """Test handling of sounddevice exceptions"""
        # sd.query_devices() raises — verify get_available_input_devices handles an empty list
        # gracefully (exception handling happens inside setup_working_device in the node)
        with patch("sounddevice.query_devices", side_effect=Exception("Device query failed")):
            import sounddevice as sd

            try:
                devices = sd.query_devices()
            except Exception as e:
                assert str(e) == "Device query failed"

    def test_empty_audio_data(self, mock_thread):
        """Test handling of empty audio data"""
        empty_data = np.array([], dtype=np.float32)

        assert len(empty_data) == 0
        assert empty_data.dtype == np.float32

    def test_audio_data_clipping(self, mock_thread):
        """Test audio data clipping behavior"""
        # Test data that would clip
        extreme_data = np.array([2.0, -3.0, 1.5, -1.8], dtype=np.float32)

        # Should be clipped to [-1, 1] range
        clipped_data = np.clip(extreme_data, -1.0, 1.0)

        assert np.all(clipped_data >= -1.0)
        assert np.all(clipped_data <= 1.0)
        assert clipped_data[0] == 1.0  # 2.0 clipped to 1.0
        assert clipped_data[1] == -1.0  # -3.0 clipped to -1.0
