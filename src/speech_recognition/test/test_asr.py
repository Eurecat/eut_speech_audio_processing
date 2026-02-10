"""
Minimal unit tests for ASRNode class - ESSENTIAL TESTS ONLY
"""

from unittest.mock import MagicMock, patch

from speech_recognition.asr import ASRNode


class TestASRNode:
    """Essential ASR tests only"""

    @patch("rclpy.node.Node.__init__")
    @patch("faster_whisper.WhisperModel")
    def test_node_initialization(self, mock_whisper_model, mock_node_init):
        """Test that the ASR node initializes correctly"""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        get_param_mock = MagicMock()

        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in [
                "use_batched_inference",
                "ros4hri_with_id",
                "cleanup_inactive_topics",
            ]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param

        get_param_mock.side_effect = mock_get_parameter

        with patch.object(ASRNode, "declare_parameter"), patch.object(
            ASRNode, "get_parameter", get_param_mock
        ), patch.object(ASRNode, "create_subscription"), patch.object(
            ASRNode, "create_publisher"
        ), patch.object(ASRNode, "get_logger"):
            node = ASRNode()

            mock_node_init.assert_called_once_with("asr_node")

    @patch("rclpy.node.Node.__init__")
    @patch("faster_whisper.WhisperModel")
    def test_whisper_model_loading(self, mock_whisper_model, mock_node_init):
        """Test Whisper model loading"""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        get_param_mock = MagicMock()

        def mock_get_parameter(param_name):
            mock_param = MagicMock()
            if param_name == "model_size":
                mock_param.get_parameter_value.return_value.string_value = "turbo"
            elif param_name == "compute_type":
                mock_param.get_parameter_value.return_value.string_value = "float32"
            elif param_name == "language":
                mock_param.get_parameter_value.return_value.string_value = "auto"
            elif param_name in [
                "use_batched_inference",
                "ros4hri_with_id",
                "cleanup_inactive_topics",
            ]:
                mock_param.get_parameter_value.return_value.bool_value = False
            elif param_name == "batch_size":
                mock_param.get_parameter_value.return_value.integer_value = 16
            else:
                mock_param.get_parameter_value.return_value.double_value = 0.5
            return mock_param

        get_param_mock.side_effect = mock_get_parameter

        with patch.object(ASRNode, "declare_parameter"), patch.object(
            ASRNode, "get_parameter", get_param_mock
        ), patch.object(ASRNode, "create_subscription"), patch.object(
            ASRNode, "create_publisher"
        ), patch.object(ASRNode, "get_logger"):
            node = ASRNode()

            mock_whisper_model.assert_called_once()
            assert node.model is not None
