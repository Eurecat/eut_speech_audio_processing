import numpy as np
import rclpy
import torch
from rclpy.node import Node
from hri_msgs.msg import AudioAndDeviceInfo, Vad


class VADNode(Node):
    def __init__(self):
        super().__init__("vad_node")

        # Declare parameters
        self.declare_parameter("repo_model", "snakers4/silero-vad")
        self.declare_parameter("model_name", "silero_vad")

        # Get parameter values
        self.repo_model = (
            self.get_parameter("repo_model").get_parameter_value().string_value
        )
        self.model_name = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )

        # Flag to track if first callback has been processed
        self.vad_initialized = False

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioAndDeviceInfo, "audio_and_device_info", self.listener_callback, 10
        )

        # Publishers
        self.vad_pub = self.create_publisher(Vad, "vad", 10)

        # Load pre-trained VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir=self.repo_model, model=self.model_name, trust_repo=True
        )

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.model.to(self.device)

        self.get_logger().info(
            "VAD node initialized, waiting for audio device information..."
        )

    def listener_callback(self, msg):
        # Initialize VAD on first callback
        if not self.vad_initialized:
            self.get_logger().info(
                f"VAD initialized with audio device: {msg.device_name} (Sample rate: {msg.device_samplerate} Hz)"
            )
            self.vad_initialized = True

        audio_data = np.array(msg.audio, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_data).to(self.device)
        sample_rate = int(msg.device_samplerate)

        # Apply VAD
        if int(audio_data.size) == 512:
            with torch.no_grad():
                prob = self.model(audio_tensor, sr=sample_rate).item()
        else:
            self.get_logger().warn(
                f"Unexpected audio chunk size: {int(audio_data.size)}. Expected 512 samples."
            )
            prob = 0.0
        # Publish VAD result
        vad_msg = Vad()
        vad_msg.header.stamp = self.get_clock().now().to_msg()
        vad_msg.vad_probability = prob
        self.vad_pub.publish(vad_msg)


def main(args=None):
    rclpy.init(args=args)
    vad_node = VADNode()

    try:
        rclpy.spin(vad_node)
    except KeyboardInterrupt:
        vad_node.get_logger().info("Shutting down VAD node.")
    finally:
        vad_node.destroy_node()
        rclpy.shutdown()
