import os

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo, Vad
from rclpy.node import Node

from speech_recognition.vad_engine import VADEngine


class VAD(Node):
    def __init__(self):
        super().__init__("vad_node")

        self.declare_parameter("repo_model", "snakers4/silero-vad")
        self.declare_parameter("model_name", "silero_vad")
        self.declare_parameter(
            "weights_dir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "weights")),
        )

        self.engine = VADEngine(
            repo_model=self.get_parameter("repo_model").get_parameter_value().string_value,
            model_name=self.get_parameter("model_name").get_parameter_value().string_value,
            weights_dir=os.path.abspath(
                self.get_parameter("weights_dir").get_parameter_value().string_value
            ),
            logger=self.get_logger(),
        )

        self.vad_initialized = False

        self.audio_sub = self.create_subscription(
            AudioAndDeviceInfo, "audio_and_device_info", self.listener_callback, 10
        )
        self.vad_pub = self.create_publisher(Vad, "vad", 10)

        self.get_logger().info("VAD node initialized, waiting for audio device information...")

    def listener_callback(self, msg: AudioAndDeviceInfo) -> None:
        if not self.vad_initialized:
            self.get_logger().info(
                f"VAD initialized with device: {msg.device_name} "
                f"(Sample rate: {msg.device_samplerate} Hz)"
            )
            self.vad_initialized = True

        audio_data = np.array(msg.audio, dtype=np.float32)
        prob = self.engine.predict(audio_data, int(msg.device_samplerate))

        vad_msg = Vad()
        vad_msg.header.stamp = self.get_clock().now().to_msg()
        vad_msg.vad_probability = prob
        self.vad_pub.publish(vad_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VAD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VAD node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
