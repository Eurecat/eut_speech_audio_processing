import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node
from rclpy.parameter import Parameter

from audio_stream_manager.audio_capture_engine import AudioCaptureEngine


class AudioCapturing(Node):
    """
    This node's only jobs are:
      1. Read ROS2 parameters and pass them to the engine.
      2. Stamp and publish AudioAndDeviceInfo messages when the engine has a chunk ready.
      3. Update the device_name parameter when the engine reports a device change.

    All audio logic (device discovery, streaming, buffering, resampling,watchdog threads) lives in AudioCaptureEngine.
    """

    def __init__(self):
        super().__init__("audio_capturing")

        # Declare ROS2 parameters
        self.declare_parameter("device_name", "DJI")
        self.declare_parameter("dtype", "float32")
        self.declare_parameter("channels", 1)
        self.declare_parameter("chunk", 512)
        self.declare_parameter("disconnection_timeout", 3.0)
        self.declare_parameter("disconnection_check_interval", 1.0)
        self.declare_parameter("primary_device_check_interval", 5.0)
        self.declare_parameter("target_samplerate", 16000)

        # Publisher
        self.audio_and_device_info_pub = self.create_publisher(
            AudioAndDeviceInfo, "audio_and_device_info", 10
        )

        # Build and start the engine, passing all audio config and the two callbacks
        self.engine = AudioCaptureEngine(
            dtype=self.get_parameter("dtype").get_parameter_value().string_value,
            channels=self.get_parameter("channels").get_parameter_value().integer_value,
            chunk=self.get_parameter("chunk").get_parameter_value().integer_value,
            target_samplerate=self.get_parameter("target_samplerate")
            .get_parameter_value()
            .integer_value,
            disconnection_timeout=self.get_parameter("disconnection_timeout")
            .get_parameter_value()
            .double_value,
            disconnection_check_interval=self.get_parameter("disconnection_check_interval")
            .get_parameter_value()
            .double_value,
            primary_device_check_interval=self.get_parameter("primary_device_check_interval")
            .get_parameter_value()
            .double_value,
            on_chunk_ready=self._publish_audio,
            on_device_changed=self._on_device_changed,
            logger=self.get_logger(),
        )

        device_name_param = self.get_parameter("device_name").get_parameter_value().string_value
        self.engine.start(device_name_param)

    # ------------------------------------------------------------------
    # Engine callbacks
    # ------------------------------------------------------------------

    def _publish_audio(
        self,
        chunk: np.ndarray,
        device_name: str,
        device_id: int,
        samplerate: float,
    ) -> None:
        """Called by the engine for every ready chunk. Stamps and publishes the message."""
        msg = AudioAndDeviceInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.audio = chunk.tolist()
        msg.device_name = device_name
        msg.device_id = device_id
        msg.device_samplerate = samplerate
        self.audio_and_device_info_pub.publish(msg)

    def _on_device_changed(self, device_name: str) -> None:
        """Called by the engine when the active device changes (including disconnection)."""
        self.set_parameters([Parameter("device_name", Parameter.Type.STRING, device_name)])


def main(args=None):
    rclpy.init(args=args)
    node = AudioCapturing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.engine.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
