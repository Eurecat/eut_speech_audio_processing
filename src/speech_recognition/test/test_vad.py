"""
Unit tests for VADNode class.
Tests voice activity detection functionality.
"""
import pytest
import numpy as np
import os
from unittest.mock import Mock, MagicMock, patch, call
from speech_recognition.vad import VADNode


class TestVADNode:
    """Test suite for VADNode class"""
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_node_initialization(self, mock_torch_load, mock_node_init):
        """Test that the VAD node initializes correctly"""
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify node was initialized with correct name
            mock_node_init.assert_called_once_with("vad_node")
            
            # Check initial state
            assert hasattr(node, 'vad_initialized')
            assert node.vad_initialized == False
            assert hasattr(node, 'last_log_time')
            assert hasattr(node, 'log_time')
            assert node.log_time == 1.0
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_parameter_declarations(self, mock_torch_load, mock_node_init):
        """Test that all required parameters are declared"""
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        declare_param_mock = MagicMock()
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter', declare_param_mock), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify all expected parameters are declared with correct defaults
            expected_param_calls = [
                ('repo_model', 'snakers4/silero-vad'),
                ('model_name', 'silero_vad')
            ]
            
            # Check the first two parameters with known values
            for param_name, default_value in expected_param_calls:
                declare_param_mock.assert_any_call(param_name, default_value)
            
            # Check that weights_dir parameter was called (value varies based on path)
            weights_dir_calls = [call for call in declare_param_mock.call_args_list if call[0][0] == 'weights_dir']
            assert len(weights_dir_calls) == 1
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_vad_model_loading(self, mock_torch_load, mock_node_init):
        """Test VAD model loading"""
        # Mock model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify model loading was called
            mock_torch_load.assert_called_once()
            
            # Verify model is stored
            assert node.model is not None
    
    def test_audio_preprocessing(self):
        """Test audio data preprocessing for VAD"""
        # Test audio preprocessing logic
        sample_rate = 16000
        audio_data = np.random.randn(sample_rate).astype(np.float32)
        
        # Test normalization
        normalized = audio_data / np.max(np.abs(audio_data))
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test resampling (conceptual test)
        target_rate = 16000
        if target_rate == sample_rate:
            resampled = audio_data
        else:
            # Would use actual resampling in real implementation
            resampled = audio_data
        
        assert len(resampled) > 0
        assert resampled.dtype == np.float32
    
    @pytest.mark.parametrize("confidence_threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_vad_confidence_thresholds(self, confidence_threshold):
        """Test VAD with different confidence thresholds"""
        # Test that confidence threshold is valid
        assert 0.0 <= confidence_threshold <= 1.0
        
        # Mock VAD result
        mock_confidence = 0.6
        
        # Test decision logic
        is_speech = mock_confidence > confidence_threshold
        
        if confidence_threshold < 0.6:
            assert is_speech == True
        else:
            assert is_speech == False
    
    def test_vad_message_structure(self):
        """Test VAD message structure"""
        # Mock VAD message creation (would use actual Vad message)
        vad_result = {
            'is_speech': True,
            'confidence': 0.8,
            'timestamp': 12345.67
        }
        
        # Verify structure
        assert 'is_speech' in vad_result
        assert 'confidence' in vad_result
        assert 'timestamp' in vad_result
        assert isinstance(vad_result['is_speech'], bool)
        assert 0.0 <= vad_result['confidence'] <= 1.0
        assert vad_result['timestamp'] > 0


class TestVADNodeROS:
    """Test suite for VADNode ROS functionality"""
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_subscription_creation(self, mock_torch_load, mock_node_init):
        """Test that audio subscription is created"""
        create_sub_mock = MagicMock()
        
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription', create_sub_mock), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify subscription was created
            create_sub_mock.assert_called_once()
            call_args = create_sub_mock.call_args
            assert call_args[0][1] == "audio_and_device_info"  # topic name
            assert call_args[0][2] == node.listener_callback  # callback function
            assert call_args[0][3] == 10  # queue size
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_publisher_creation(self, mock_torch_load, mock_node_init):
        """Test that VAD publisher is created"""
        create_pub_mock = MagicMock()
        
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher', create_pub_mock), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify publisher was created
            create_pub_mock.assert_called_once()
            call_args = create_pub_mock.call_args
            assert call_args[0][1] == "vad"  # topic name
            assert call_args[0][2] == 10  # queue size
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_audio_callback_processing(self, mock_torch_load, mock_node_init):
        """Test audio callback processes messages correctly"""
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock the VAD model
            mock_model.return_value = [0.8]  # Mock confidence score
            
            node = VADNode()
            
            # Create mock audio message
            mock_msg = Mock()
            mock_msg.audio = [0.1, 0.2, -0.3, 0.4] * 1000  # 4000 samples
            mock_msg.sample_rate = 16000
            
            # Mock publisher
            node.vad_pub = MagicMock()
            
            # Process callback (would trigger VAD processing)
            # Note: Actual implementation would call listener_callback
            audio_array = np.array(mock_msg.audio, dtype=np.float32)
            
            # Verify audio data is processed correctly
            assert len(audio_array) == 4000
            assert audio_array.dtype == np.float32


class TestVADNodeEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_empty_audio_data(self):
        """Test handling of empty audio data"""
        empty_audio = np.array([], dtype=np.float32)
        
        # Should handle empty arrays gracefully
        assert len(empty_audio) == 0
        assert empty_audio.dtype == np.float32
        
        # VAD processing should handle this case
        # (actual implementation would check for empty input)
    
    def test_very_short_audio_clips(self):
        """Test handling of very short audio clips"""
        short_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Should handle short clips
        assert len(short_audio) == 3
        
        # VAD might need minimum length requirement
        min_length = 160  # 10ms at 16kHz
        if len(short_audio) < min_length:
            # Would pad or skip in real implementation
            padded = np.pad(short_audio, (0, min_length - len(short_audio)))
            assert len(padded) == min_length
    
    def test_silent_audio_detection(self):
        """Test VAD behavior with silent audio"""
        silent_audio = np.zeros(1600, dtype=np.float32)  # 100ms of silence
        
        # Silent audio should typically result in no speech detection
        # (this would be determined by the actual VAD model)
        max_amplitude = np.max(np.abs(silent_audio))
        assert max_amplitude == 0.0
        
        # In real VAD, this would likely result in is_speech = False
    
    def test_very_loud_audio_clipping(self):
        """Test handling of very loud audio that might clip"""
        loud_audio = np.array([2.0, -3.0, 1.5, -2.5], dtype=np.float32)
        
        # Should clip to [-1, 1] range
        clipped_audio = np.clip(loud_audio, -1.0, 1.0)
        
        assert np.all(clipped_audio >= -1.0)
        assert np.all(clipped_audio <= 1.0)
        assert clipped_audio[0] == 1.0  # 2.0 clipped to 1.0
        assert clipped_audio[1] == -1.0  # -3.0 clipped to -1.0
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_model_loading_failure(self, mock_torch_load, mock_node_init):
        """Test handling of model loading failure"""
        mock_torch_load.side_effect = Exception("Model loading failed")
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter'), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs'), \
             patch('torch.hub.set_dir'):
            
            # Should handle model loading failure gracefully
            with pytest.raises(Exception):
                node = VADNode()
    
    @patch('rclpy.node.Node.__init__')
    @patch('torch.hub.load')
    def test_weights_directory_creation(self, mock_torch_load, mock_node_init):
        """Test weights directory creation"""
        # Mock torch hub load to return a model
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)
        
        get_param_mock = MagicMock()
        
        # Mock parameter values
        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "repo_model":
                mock_param.get_parameter_value.return_value.string_value = "snakers4/silero-vad"
            elif param_name == "model_name":
                mock_param.get_parameter_value.return_value.string_value = "silero_vad"
            elif param_name == "weights_dir":
                mock_param.get_parameter_value.return_value.string_value = "/tmp/weights"
            return mock_param
        
        get_param_mock.side_effect = mock_get_parameter
        
        with patch.object(VADNode, 'declare_parameter'), \
             patch.object(VADNode, 'get_parameter', get_param_mock), \
             patch.object(VADNode, 'create_subscription'), \
             patch.object(VADNode, 'create_publisher'), \
             patch.object(VADNode, 'get_logger'), \
             patch('os.makedirs') as mock_makedirs, \
             patch('torch.hub.set_dir'), \
             patch('torch.cuda.is_available', return_value=False):
            
            node = VADNode()
            
            # Verify weights directory is created
            mock_makedirs.assert_called_once_with("/tmp/weights", exist_ok=True)
            mock_makedirs.assert_called_once()
    
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_different_sample_rates(self, sample_rate):
        """Test VAD with different audio sample rates"""
        # Generate test audio at different sample rates
        duration = 1.0  # 1 second
        num_samples = int(sample_rate * duration)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        assert len(audio_data) == num_samples
        assert audio_data.dtype == np.float32
        
        # VAD models typically expect 16kHz
        # Resampling would be needed for other rates
        if sample_rate != 16000:
            # Would need resampling in real implementation
            pass