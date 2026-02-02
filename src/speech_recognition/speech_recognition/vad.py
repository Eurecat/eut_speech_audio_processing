import os
import time

import numpy as np
import rclpy
import torch
from hri_msgs.msg import AudioAndDeviceInfo, Vad
from rclpy.node import Node


class VADNode(Node):
    def __init__(self):
        super().__init__("vad_node")

        # Declare parameters
        self.declare_parameter("repo_model", "snakers4/silero-vad")
        self.declare_parameter("model_name", "silero_vad")
        self.declare_parameter(
            "weights_dir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "weights")),
        )

        # Get parameter values
        self.repo_model = self.get_parameter("repo_model").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.weights_dir = os.path.abspath(
            self.get_parameter("weights_dir").get_parameter_value().string_value
        )

        # Flag to track if first callback has been processed
        self.vad_initialized = False

        # Logging control
        self.last_log_time = 0  # For 1Hz logging
        self.log_time = 1.0  # Log every 1 second

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioAndDeviceInfo, "audio_and_device_info", self.listener_callback, 10
        )

        # Publishers
        self.vad_pub = self.create_publisher(Vad, "vad", 10)

        # Load VAD model from local weights or download if not present
        os.makedirs(self.weights_dir, exist_ok=True)

        # Set torch hub directory to weights folder
        torch.hub.set_dir(str(self.weights_dir))

        self.get_logger().info(f"Loading VAD model from: {self.weights_dir}")
        self.model, _ = torch.hub.load(
            repo_or_dir=self.repo_model, model=self.model_name, trust_repo=True
        )

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        light_green = "\033[38;5;82m"
        reset = "\033[0m"
        self.get_logger().info(f"{light_green}Using device on VAD: {self.device}{reset}")
        self.model.to(self.device)
        green = "\033[32m"
        reset = "\033[0m"
        self.get_logger().info(
            f"{green}VAD node initialized, waiting for audio device information...{reset}"
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

        current_time = time.time()
        if current_time - self.last_log_time >= self.log_time:
            self.get_logger().debug(f"VAD probability: {prob}")
            self.last_log_time = current_time

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
