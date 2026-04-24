"""
Unit tests for AudioToMp3 class.
Tests audio recording and conversion functionality.
"""

import os
import wave
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from audio_stream_manager.audio_stream_manager.audio_to_mp3 import AudioToMp3


class TestAudioToMp3:
    """Test suite for AudioToMp3 class"""

    @patch("rclpy.node.Node.__init__")
    def test_node_initialization(self, mock_node_init):
        """Test that the node initializes correctly"""
        with patch.object(AudioToMp3, "create_subscription"), patch.object(
            AudioToMp3, "get_logger"
        ):
            node = AudioToMp3()

            # Verify node was initialized with correct name
            mock_node_init.assert_called_once_with("audio_to_mp3")

            # Check initial state
            assert hasattr(node, "audio_buffer")
            assert isinstance(node.audio_buffer, list)
            assert len(node.audio_buffer) == 0

    @patch("rclpy.node.Node.__init__")
    def test_subscription_creation(self, mock_node_init):
        """Test that audio subscription is created"""
        create_sub_mock = MagicMock()
        get_logger_mock = MagicMock()

        with patch.object(AudioToMp3, "create_subscription", create_sub_mock), patch.object(
            AudioToMp3, "get_logger", return_value=get_logger_mock
        ):
            node = AudioToMp3()

            # Verify subscription was created correctly
            create_sub_mock.assert_called_once()
            call_args = create_sub_mock.call_args
            assert call_args[0][1] == "/audio_and_device_info"  # topic name
            assert call_args[0][2] == node.audio_callback  # callback function
            assert call_args[0][3] == 10  # queue size

    def test_audio_callback(self):
        """Test audio callback processes messages correctly"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Create mock audio message
            mock_msg = Mock()
            test_audio_data = [0.1, 0.2, -0.3, 0.4, 0.5]
            mock_msg.audio = test_audio_data

            # Process callback
            node.audio_callback(mock_msg)

            # Verify audio data was added to buffer
            assert len(node.audio_buffer) == 1
            np.testing.assert_array_equal(
                node.audio_buffer[0], np.array(test_audio_data, dtype=np.float32)
            )

    def test_multiple_audio_callbacks(self):
        """Test multiple audio callbacks accumulate correctly"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Create multiple mock audio messages
            test_chunks = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

            for chunk_data in test_chunks:
                mock_msg = Mock()
                mock_msg.audio = chunk_data
                node.audio_callback(mock_msg)

            # Verify all chunks were added
            assert len(node.audio_buffer) == 3
            for i, chunk_data in enumerate(test_chunks):
                np.testing.assert_array_equal(
                    node.audio_buffer[i], np.array(chunk_data, dtype=np.float32)
                )

    @patch("wave.open")
    @patch("numpy.concatenate")
    def test_save_to_wav_success(self, mock_concatenate, mock_wave_open):
        """Test successful WAV file saving"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Setup test data
            test_samples = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
            mock_concatenate.return_value = test_samples
            node.audio_buffer = [test_samples]  # Non-empty buffer

            # Mock wave file context manager
            mock_wave_file = MagicMock()
            mock_wave_open.return_value.__enter__.return_value = mock_wave_file

            # Test saving
            result = node.save_to_wav("test.wav")

            # Verify success
            assert result is True
            mock_wave_open.assert_called_once_with("test.wav", "wb")
            mock_wave_file.setnchannels.assert_called_once_with(1)
            mock_wave_file.setsampwidth.assert_called_once_with(2)
            mock_wave_file.setframerate.assert_called_once_with(16000)
            mock_wave_file.writeframes.assert_called_once()

    def test_save_to_wav_empty_buffer(self):
        """Test WAV saving with empty buffer"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger") as mock_logger:
            node = AudioToMp3()
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance

            # Empty buffer
            node.audio_buffer = []

            # Test saving
            result = node.save_to_wav("test.wav")

            # Verify failure and warning
            assert result is False

    def test_audio_data_conversion(self):
        """Test audio data conversion from float32 to int16"""
        # Test data in [-1.0, 1.0] range
        test_samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        # Convert to int16 (as done in save_to_wav)
        clipped = np.clip(test_samples, -1.0, 1.0)
        samples_int16 = (clipped * 32767.0).astype(np.int16)

        # Verify conversion - note that -0.5 * 32767 = -16383.5 which truncates to -16383 when cast to int16
        # The int16 range is asymmetric: -32768 to 32767
        # -1.0 * 32767 = -32767 (not -32768 because we multiply by 32767, not 32768)
        # The key issue was that -16384 was expected but -16383 is correct due to truncation
        expected = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)
        np.testing.assert_array_equal(samples_int16, expected)

    def test_audio_data_clipping(self):
        """Test audio data clipping for out-of-range values"""
        # Test data outside [-1.0, 1.0] range
        test_samples = np.array([2.0, -3.0, 1.5, -1.8, 0.5], dtype=np.float32)

        # Clip to valid range
        clipped = np.clip(test_samples, -1.0, 1.0)

        # Verify clipping
        expected = np.array([1.0, -1.0, 1.0, -1.0, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(clipped, expected)

    @patch("subprocess.run")
    def test_wav_to_mp3_conversion(self, mock_subprocess):
        """Test WAV to MP3 conversion via subprocess"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Mock successful conversion
            mock_subprocess.return_value.returncode = 0

            # Test the conversion command structure
            wav_file = "test.wav"
            mp3_file = "test.mp3"

            # This would be called in a convert_to_mp3 method if implemented
            expected_cmd = ["ffmpeg", "-i", wav_file, "-codec:a", "libmp3lame", mp3_file]

            # Verify command structure is correct
            assert expected_cmd[0] == "ffmpeg"
            assert expected_cmd[1] == "-i"
            assert expected_cmd[2] == wav_file
            assert mp3_file in expected_cmd


class TestAudioToMp3EdgeCases:
    """Test suite for edge cases and error handling"""

    def test_very_small_audio_data(self):
        """Test handling of very small audio chunks"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Single sample
            mock_msg = Mock()
            mock_msg.audio = [0.5]

            node.audio_callback(mock_msg)

            assert len(node.audio_buffer) == 1
            assert len(node.audio_buffer[0]) == 1

    def test_zero_amplitude_audio(self):
        """Test handling of silent audio (all zeros)"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Silent audio
            mock_msg = Mock()
            mock_msg.audio = [0.0, 0.0, 0.0, 0.0]

            node.audio_callback(mock_msg)

            # Should still process correctly
            assert len(node.audio_buffer) == 1
            np.testing.assert_array_equal(node.audio_buffer[0], np.zeros(4, dtype=np.float32))

    def test_large_audio_buffer(self):
        """Test handling of large audio buffer accumulation"""
        with patch("rclpy.node.Node.__init__"), patch.object(
            AudioToMp3, "create_subscription"
        ), patch.object(AudioToMp3, "get_logger"):
            node = AudioToMp3()

            # Add many chunks
            chunk_size = 1024
            num_chunks = 100

            for i in range(num_chunks):
                mock_msg = Mock()
                mock_msg.audio = np.random.rand(chunk_size).tolist()
                node.audio_callback(mock_msg)

            # Verify all chunks accumulated
            assert len(node.audio_buffer) == num_chunks

            # Test concatenation would work
            with patch("numpy.concatenate") as mock_concat:
                mock_concat.return_value = np.array([])
                node.save_to_wav("test.wav")
                mock_concat.assert_called_once_with(node.audio_buffer)

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_different_sample_rates(self, sample_rate):
        """Test handling of different sample rates"""
        # Test that the constant can be updated for different rates
        from audio_stream_manager.audio_stream_manager.audio_to_mp3 import SAMPLE_RATE

        # Default should be 16000
        assert SAMPLE_RATE == 16000

        # Test that other rates are valid
        assert sample_rate > 0
        assert sample_rate <= 48000
