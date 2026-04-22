import os
import subprocess

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node

from audio_stream_manager.utils.audio_to_mp3_utils import (
    convert_wav_to_mp3,
    save_to_wav,
)


class AudioToMp3(Node):
    def __init__(self):
        super().__init__("audio_to_mp3")

        # Declare parameters (values come from audio_params.yaml)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("temp_wav", "recording_temp.wav")
        self.declare_parameter("output_mp3", "/workspace/src/audio_stream_manager/recording.mp3")

        self.sample_rate = self.get_parameter("sample_rate").get_parameter_value().integer_value
        self.temp_wav = self.get_parameter("temp_wav").get_parameter_value().string_value
        self.output_mp3 = self.get_parameter("output_mp3").get_parameter_value().string_value

        self.subscription = self.create_subscription(
            AudioAndDeviceInfo,
            "/audio_and_device_info",
            self.audio_callback,
            10,
        )
        self.audio_buffer = []  # list of numpy float32 arrays
        self.get_logger().info("AudioToMp3 node started, subscribing to /audio_and_device_info")

    def audio_callback(self, msg):
        chunk = np.array(msg.audio, dtype=np.float32)
        self.audio_buffer.append(chunk)


def main(args=None):
    rclpy.init(args=args)
    node = AudioToMp3()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping recording...")
    finally:
        write_wav = save_to_wav(
            node.audio_buffer,
            node.temp_wav,
            node.sample_rate,
            logger=node.get_logger(),
        )
        if write_wav:
            try:
                convert_wav_to_mp3(node.temp_wav, node.output_mp3, logger=node.get_logger())
            except subprocess.CalledProcessError as e:
                node.get_logger().error(f"ffmpeg failed: {e}")

        if os.path.exists(node.temp_wav):
            os.remove(node.temp_wav)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
