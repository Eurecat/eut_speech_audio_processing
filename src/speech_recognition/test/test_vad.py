"""
Minimal unit tests for VADNode class - ESSENTIAL TESTS ONLY
"""

from unittest.mock import MagicMock, patch

from speech_recognition.vad import VADNode


class TestVADNode:
    """Essential VAD tests only"""

    @patch("rclpy.node.Node.__init__")
    @patch("torch.hub.load")
    def test_node_initialization(self, mock_torch_load, mock_node_init):
        """Test that the VAD node initializes correctly"""
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)

        get_param_mock = MagicMock()

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

        with patch.object(VADNode, "declare_parameter"), patch.object(
            VADNode, "get_parameter", get_param_mock
        ), patch.object(VADNode, "create_subscription"), patch.object(
            VADNode, "create_publisher"
        ), patch.object(VADNode, "get_logger"), patch("os.makedirs"), patch(
            "torch.hub.set_dir"
        ), patch("torch.cuda.is_available", return_value=False):
            node = VADNode()

            mock_node_init.assert_called_once_with("vad_node")
            assert hasattr(node, "vad_initialized")

    @patch("rclpy.node.Node.__init__")
    @patch("torch.hub.load")
    def test_vad_model_loading(self, mock_torch_load, mock_node_init):
        """Test VAD model loading"""
        mock_model = MagicMock()
        mock_torch_load.return_value = (mock_model, None)

        get_param_mock = MagicMock()

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

        with patch.object(VADNode, "declare_parameter"), patch.object(
            VADNode, "get_parameter", get_param_mock
        ), patch.object(VADNode, "create_subscription"), patch.object(
            VADNode, "create_publisher"
        ), patch.object(VADNode, "get_logger"), patch("os.makedirs"), patch(
            "torch.hub.set_dir"
        ), patch("torch.cuda.is_available", return_value=False):
            node = VADNode()

            mock_torch_load.assert_called_once()
            assert node.model is not None
