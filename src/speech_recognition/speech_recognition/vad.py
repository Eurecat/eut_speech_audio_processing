import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray

REPO_MODEL = "snakers4/silero-vad"
MODEL_NAME = "silero_vad"
SAMPLERATE = 16000


class VADNode(Node):
    def __init__(self):
        super().__init__("vad_node")

        # Subscribers
        self.audio_sub = self.create_subscription(
            Float32MultiArray, "audio", self.listener_callback, 10
        )

        # Publishers
        self.vad_pub = self.create_publisher(Bool, "vad", 10)

        # Load pre-trained VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir=REPO_MODEL, model=MODEL_NAME, trust_repo=True
        )

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.model.to(self.device)

    def listener_callback(self, msg):
        audio_data = np.array(msg.data, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        # Apply VAD
        with torch.no_grad():
            prob = self.model(audio_tensor, sr=SAMPLERATE).item()
            speech_detected = prob > 0.5

        # Publish VAD result
        vad_msg = Bool()
        vad_msg.data = speech_detected
        self.vad_pub.publish(vad_msg)
        self.get_logger().info(
            f"VAD Probability: {prob:.4f}, Speech Detected: {speech_detected}"
        )


def main(args=None):
    rclpy.init(args=args)
    vad_node = VADNode()

    try:
        rclpy.spin(vad_node)
    except KeyboardInterrupt:
        vad_node.get_logger().info("Shutting down node.")
    finally:
        vad_node.destroy_node()
        rclpy.shutdown()
