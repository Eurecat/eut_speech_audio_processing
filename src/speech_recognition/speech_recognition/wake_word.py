import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo, WakeWord
from rclpy.node import Node

from speech_recognition.wake_word_engine import WakeWordEngine


class WakeWordDetectorNode(Node):
    """All audio logic (model loading, buffering, sliding window, inference,
    processing thread) lives in WakeWordEngine. This node's only jobs are:
      1. Read ROS2 parameters and pass them to the engine.
      2. Stamp and publish WakeWord messages when the engine detects a hit.
    """

    def __init__(self):
        super().__init__("wake_word_node")

        # Declare parameters
        self.declare_parameter("wake_word_model_names", ["hey_jana"])
        self.declare_parameter(
            "model_base_path", "/workspace/src/speech_recognition/weights_openwakeword"
        )
        self.declare_parameter("window_duration", 2.0)
        self.declare_parameter("step_duration", 0.5)

        # Read wake_word_model_names — string arrays need special handling
        wake_word_models = self._get_model_names_param()
        self.get_logger().info(f"Using wake word models: {wake_word_models}")

        self.engine = WakeWordEngine(
            wake_word_models=wake_word_models,
            model_base_path=self.get_parameter("model_base_path")
            .get_parameter_value()
            .string_value,
            window_duration=self.get_parameter("window_duration")
            .get_parameter_value()
            .double_value,
            step_duration=self.get_parameter("step_duration").get_parameter_value().double_value,
            on_wake_word_detected=self._publish_wake_word,
            logger=self.get_logger(),
        )

        self._device_logged = False

        self.audio_subscription = self.create_subscription(
            AudioAndDeviceInfo,
            "audio_and_device_info",
            self._audio_callback,
            10,
        )
        self.wake_word_publisher = self.create_publisher(WakeWord, "wake_word", 10)

        self.get_logger().info("Wake word node initialized, waiting for audio...")

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_model_names_param(self):
        try:
            value = self.get_parameter("wake_word_model_names").value
            if value and isinstance(value, list):
                return value
            self.get_logger().warn(
                "wake_word_model_names is empty or invalid, using default ['hey_jana']."
            )
        except Exception as e:
            self.get_logger().error(f"Error reading wake_word_model_names: {e}")
        return ["hey_jana"]

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _audio_callback(self, msg: AudioAndDeviceInfo) -> None:
        if not self._device_logged:
            self.get_logger().info(
                f"First audio received from device: {msg.device_name} "
                f"(ID: {msg.device_id}, Rate: {msg.device_samplerate} Hz)"
            )
            self._device_logged = True

        audio_data = np.array(msg.audio, dtype=np.float32)
        self.engine.push_audio(audio_data, msg.device_samplerate)

    def _publish_wake_word(self, probability: float, winning_model: str) -> None:
        """Called by the engine when a wake word is detected."""
        msg = WakeWord()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.wake_word_probability = float(probability)
        msg.winning_model = winning_model
        self.wake_word_publisher.publish(msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down wake word node...")
        self.engine.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WakeWordDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
