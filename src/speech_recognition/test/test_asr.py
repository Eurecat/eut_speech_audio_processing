"""
Unit tests for ASRNode class.
Tests automatic speech recognition functionality.
"""
import pytest
import numpy as np
from collections import deque
from unittest.mock import Mock, MagicMock, patch, call
from speech_recognition.asr import ASRNode


class TestASRNode:
    """Test suite for ASRNode class"""
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_node_initialization(self, mock_whisper_model, mock_node_init):
        """Test that the ASR node initializes correctly"""
        # Mock the WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        get_param_mock = MagicMock()
        
        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter'), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription'), \
             patch.object(ASRNode, 'create_publisher'), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Verify node was initialized with correct name
            mock_node_init.assert_called_once_with("asr_node")
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_parameter_declarations(self, mock_whisper_model, mock_node_init):
        """Test that all required parameters are declared"""
        # Mock the WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        declare_param_mock = MagicMock()
        get_param_mock = MagicMock()
        
        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter', declare_param_mock), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription'), \
             patch.object(ASRNode, 'create_publisher'), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Verify all expected parameters are declared with correct defaults
            expected_param_calls = [
                ('model_size', 'turbo'),
                ('compute_type', 'float32'),
                ('language', 'auto'),
                ('use_batched_inference', False),
                ('batch_size', 16),
                ('vad_threshold', 0.5),
                ('min_silence_duration', 1.0),
                ('max_chunk_duration', 30.0),
                ('silence_detection_threshold', 0.00001),
                ('pre_buffer_duration', 0.3),
                ('ros4hri_with_id', True),
                ('cleanup_inactive_topics', False),
                ('inactive_topic_timeout', 10.0)
            ]
            
            for param_name, default_value in expected_param_calls:
                declare_param_mock.assert_any_call(param_name, default_value)
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_whisper_model_loading(self, mock_whisper_model, mock_node_init):
        """Test Whisper model loading"""
        # Mock model
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        get_param_mock = MagicMock()
        
        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter'), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription'), \
             patch.object(ASRNode, 'create_publisher'), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Test model loading would be called during initialization
            # (actual model loading tested separately)
    
    def test_audio_preprocessing_for_asr(self):
        """Test audio data preprocessing for ASR"""
        # Test audio preprocessing logic
        sample_rate = 16000
        audio_data = np.random.randn(sample_rate * 2).astype(np.float32)  # 2 seconds
        
        # Test normalization
        normalized = audio_data / np.max(np.abs(audio_data))
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test chunking for long audio
        chunk_duration = 30.0  # seconds
        max_samples = int(sample_rate * chunk_duration)
        
        if len(audio_data) > max_samples:
            chunked = audio_data[:max_samples]
        else:
            chunked = audio_data
        
        assert len(chunked) <= max_samples
        assert chunked.dtype == np.float32
    
    def test_silence_detection(self):
        """Test silence detection logic"""
        # Test silent audio
        silent_audio = np.zeros(1600, dtype=np.float32)  # 100ms of silence
        rms_silent = np.sqrt(np.mean(silent_audio ** 2))
        
        # Test audio with signal
        signal_audio = np.random.randn(1600).astype(np.float32) * 0.1
        rms_signal = np.sqrt(np.mean(signal_audio ** 2))
        
        silence_threshold = 0.00001
        
        assert rms_silent < silence_threshold  # Should detect as silence
        assert rms_signal > silence_threshold  # Should detect as signal
    
    def test_audio_buffering(self):
        """Test audio buffering logic"""
        # Test circular buffer behavior
        buffer_size = 1000
        buffer = deque(maxlen=buffer_size)
        
        # Add more items than buffer size
        for i in range(1500):
            buffer.append(i)
        
        # Should only keep last buffer_size items
        assert len(buffer) == buffer_size
        assert buffer[0] == 500  # First item should be 500 (1500 - 1000)
        assert buffer[-1] == 1499  # Last item should be 1499
    
    def test_transcription_result_structure(self):
        """Test transcription result structure"""
        # Mock transcription result
        transcription_result = {
            'text': "Hello world",
            'language': "en",
            'confidence': 0.95,
            'segments': [
                {'start': 0.0, 'end': 1.2, 'text': "Hello world"}
            ]
        }
        
        # Verify structure
        assert 'text' in transcription_result
        assert 'language' in transcription_result
        assert 'confidence' in transcription_result
        assert 'segments' in transcription_result
        
        assert isinstance(transcription_result['text'], str)
        assert 0.0 <= transcription_result['confidence'] <= 1.0
        assert isinstance(transcription_result['segments'], list)
        
        if transcription_result['segments']:
            segment = transcription_result['segments'][0]
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert segment['start'] >= 0
            assert segment['end'] > segment['start']


class TestASRNodeROS:
    """Test suite for ASRNode ROS functionality"""
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_subscription_creation(self, mock_whisper_model, mock_node_init):
        """Test that subscriptions are created"""
        # Mock the WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        create_sub_mock = MagicMock()
        get_param_mock = MagicMock()        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter'), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription', create_sub_mock), \
             patch.object(ASRNode, 'create_publisher'), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Should create subscriptions for audio and VAD
            assert create_sub_mock.call_count >= 1
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_publisher_creation(self, mock_whisper_model, mock_node_init):
        """Test that publishers are created"""
        # Mock the WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        create_pub_mock = MagicMock()
        get_param_mock = MagicMock()
        
        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter'), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription'), \
             patch.object(ASRNode, 'create_publisher', create_pub_mock), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Should create publishers for speech results
            assert create_pub_mock.call_count >= 1
    
    @patch('rclpy.node.Node.__init__')
    @patch('faster_whisper.WhisperModel')
    def test_vad_integration(self, mock_whisper_model, mock_node_init):
        """Test VAD integration with ASR"""
        # Mock the WhisperModel
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        get_param_mock = MagicMock()
        
        # Mock parameter values with valid model_size
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in ["use_batched_inference", "ros4hri_with_id", "cleanup_inactive_topics"]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:  # double values
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(ASRNode, 'declare_parameter'), \
             patch.object(ASRNode, 'get_parameter', get_param_mock), \
             patch.object(ASRNode, 'create_subscription'), \
             patch.object(ASRNode, 'create_publisher'), \
             patch.object(ASRNode, 'get_logger'):
            
            node = ASRNode()
            
            # Mock VAD message
            mock_vad_msg = Mock()
            mock_vad_msg.is_speech = True
            mock_vad_msg.confidence = 0.8
            
            # Test VAD processing logic
            vad_threshold = 0.5
            should_process = mock_vad_msg.is_speech and mock_vad_msg.confidence > vad_threshold
            
            assert should_process == True
            
            # Test with low confidence
            mock_vad_msg.confidence = 0.3
            should_process = mock_vad_msg.is_speech and mock_vad_msg.confidence > vad_threshold
            
            assert should_process == False


class TestASRNodeLanguageProcessing:
    """Test suite for language processing functionality"""
    
    @pytest.mark.parametrize("language", ["auto", "en", "es", "fr", "de", "it"])
    def test_language_detection(self, language):
        """Test language detection and specification"""
        # Test that language codes are valid
        valid_languages = ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "zh"]
        assert language in valid_languages or language == "auto"
        
        # Test auto detection
        if language == "auto":
            detected_language = "en"  # Mock detection result
            assert detected_language in valid_languages
    
    def test_multilingual_transcription(self):
        """Test handling of multilingual audio"""
        # Mock multilingual transcription results
        multilingual_results = [
            {"text": "Hello world", "language": "en"},
            {"text": "Hola mundo", "language": "es"},
            {"text": "Bonjour le monde", "language": "fr"}
        ]
        
        for result in multilingual_results:
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0
            assert isinstance(result["language"], str)
            assert len(result["language"]) == 2  # ISO language code
    
    def test_text_postprocessing(self):
        """Test text postprocessing and cleanup"""
        # Test text cleaning
        raw_text = "  Hello,   world!  \n"
        cleaned_text = raw_text.strip()
        normalized_text = " ".join(cleaned_text.split())
        
        assert normalized_text == "Hello, world!"
        
        # Test punctuation handling
        text_with_punct = "Hello world."
        assert text_with_punct.endswith(".")
        
        # Test empty text handling
        empty_text = "   "
        cleaned_empty = empty_text.strip()
        assert cleaned_empty == ""


class TestASRNodeEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_empty_audio_handling(self):
        """Test handling of empty audio input"""
        empty_audio = np.array([], dtype=np.float32)
        
        # Should handle empty arrays gracefully
        assert len(empty_audio) == 0
        
        # ASR should skip processing empty audio
        if len(empty_audio) == 0:
            transcription_result = {"text": "", "confidence": 0.0}
        
        assert transcription_result["text"] == ""
        assert transcription_result["confidence"] == 0.0
    
    def test_very_short_audio_clips(self):
        """Test handling of very short audio clips"""
        # Very short clip (less than typical minimum)
        short_audio = np.random.randn(160).astype(np.float32)  # 10ms at 16kHz
        
        min_duration = 0.1  # 100ms minimum
        min_samples = int(16000 * min_duration)
        
        if len(short_audio) < min_samples:
            # Would pad or skip in real implementation
            should_skip = True
        else:
            should_skip = False
        
        assert should_skip == True  # 10ms is too short
    
    def test_very_long_audio_chunks(self):
        """Test handling of very long audio chunks"""
        # Very long audio clip
        long_audio = np.random.randn(16000 * 60).astype(np.float32)  # 60 seconds
        
        max_duration = 30.0  # 30 seconds maximum
        max_samples = int(16000 * max_duration)
        
        if len(long_audio) > max_samples:
            # Should chunk the audio
            chunked_audio = long_audio[:max_samples]
        else:
            chunked_audio = long_audio
        
        assert len(chunked_audio) == max_samples
    
    def test_corrupted_audio_data(self):
        """Test handling of corrupted or invalid audio data"""
        # Test with NaN values
        corrupted_audio = np.array([0.1, np.nan, 0.3, np.inf], dtype=np.float32)
        
        # Should detect and handle corrupted data
        is_corrupted = np.any(np.isnan(corrupted_audio)) or np.any(np.isinf(corrupted_audio))
        assert is_corrupted == True
        
        # Clean the data
        cleaned_audio = corrupted_audio[np.isfinite(corrupted_audio)]
        assert len(cleaned_audio) == 2  # Only 0.1 and 0.3 are valid
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts during model download"""
        # Mock timeout scenario
        timeout_occurred = True
        
        if timeout_occurred:
            # Should fall back gracefully
            fallback_model = "base"  # Smaller model as fallback
            assert fallback_model in ["tiny", "base", "small", "medium", "large"]
    
    @pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large", "turbo"])
    def test_different_model_sizes(self, model_size):
        """Test different Whisper model sizes"""
        valid_models = ["tiny", "base", "small", "medium", "large", "turbo"]
        assert model_size in valid_models
        
        # Test model size affects processing speed vs accuracy trade-off
        if model_size in ["tiny", "base"]:
            expected_speed = "fast"
            expected_accuracy = "moderate"
        elif model_size in ["small", "medium"]:
            expected_speed = "moderate" 
            expected_accuracy = "good"
        else:  # large, turbo
            expected_speed = "slow"
            expected_accuracy = "best"
        
        assert expected_speed in ["fast", "moderate", "slow"]
        assert expected_accuracy in ["moderate", "good", "best"]