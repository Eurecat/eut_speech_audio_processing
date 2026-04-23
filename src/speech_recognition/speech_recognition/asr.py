import os
import time
from typing import Dict

import numpy as np
import rclpy
from hri_msgs.msg import (
    AudioAndDeviceInfo,
    LiveSpeech,
    SpeechActivityDetection,
    SpeechResult,
    Vad,
)
from rclpy.node import Node

from speech_recognition.asr_engine import ASREngine


class ASRNode(Node):
    """
    All ASR logic (model loading, buffering, VAD state machine, chunk
    splitting, transcription) lives in ASREngine. This node's jobs are:
      1. Read ROS2 parameters and pass them to the engine.
      2. Feed audio chunks, VAD probabilities, and speaker IDs into the engine.
      3. Stamp and publish SpeechResult and LiveSpeech when the engine
         produces a transcript.
      4. Manage ROS4HRI voice publisher lifecycle (create, cleanup).
    """

    def __init__(self):
        super().__init__("asr_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_size", "turbo")
        self.declare_parameter("compute_type", "float32")
        self.declare_parameter("language", "auto")
        self.declare_parameter("use_batched_inference", False)
        self.declare_parameter("batch_size", 16)
        self.declare_parameter("vad_threshold", 0.5)
        self.declare_parameter("min_silence_duration", 1.0)
        self.declare_parameter("max_chunk_duration", 30.0)
        self.declare_parameter("silence_detection_threshold", 0.00001)
        self.declare_parameter("pre_buffer_duration", 0.3)
        self.declare_parameter("ros4hri_with_id", True)
        self.declare_parameter("cleanup_inactive_topics", False)
        self.declare_parameter("inactive_topic_timeout", 10.0)

        model_size = self.get_parameter("model_size").get_parameter_value().string_value

        # Validate model size early so the node fails fast with a clear message
        try:
            ASREngine.validate_model_size(model_size)
        except ValueError as e:
            self.get_logger().error(str(e))
            raise

        self.ros4hri_enabled = (
            self.get_parameter("ros4hri_with_id").get_parameter_value().bool_value
        )
        self.cleanup_inactive_topics = (
            self.get_parameter("cleanup_inactive_topics").get_parameter_value().bool_value
        )
        self.inactive_topic_timeout = (
            self.get_parameter("inactive_topic_timeout").get_parameter_value().double_value
        )

        weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))

        # ------------------------------------------------------------------
        # Engine
        # ------------------------------------------------------------------
        self.engine = ASREngine(
            model_size=model_size,
            compute_type=self.get_parameter("compute_type").get_parameter_value().string_value,
            language=self.get_parameter("language").get_parameter_value().string_value,
            use_batched_inference=self.get_parameter("use_batched_inference")
            .get_parameter_value()
            .bool_value,
            batch_size=self.get_parameter("batch_size").get_parameter_value().integer_value,
            vad_threshold=self.get_parameter("vad_threshold").get_parameter_value().double_value,
            min_silence_duration=self.get_parameter("min_silence_duration")
            .get_parameter_value()
            .double_value,
            max_chunk_duration=self.get_parameter("max_chunk_duration")
            .get_parameter_value()
            .double_value,
            silence_detection_threshold=self.get_parameter("silence_detection_threshold")
            .get_parameter_value()
            .double_value,
            pre_buffer_duration=self.get_parameter("pre_buffer_duration")
            .get_parameter_value()
            .double_value,
            weights_dir=weights_dir,
            on_transcript_ready=self._publish_transcript,
            logger=self.get_logger(),
        )

        # ------------------------------------------------------------------
        # ROS4HRI voice publisher registry
        # ------------------------------------------------------------------
        self.voice_publishers: Dict[str, object] = {}
        self.voice_publishers_activity: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.asr_pub = self.create_publisher(SpeechResult, "speech_result", 10)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(
            AudioAndDeviceInfo,
            "audio_and_device_info",
            self._audio_callback,
            10,
        )
        self.create_subscription(Vad, "vad", self._vad_callback, 10)
        self.create_subscription(
            SpeechActivityDetection,
            "speech_activity_detection",
            self._speech_activity_callback,
            10,
        )

        # ------------------------------------------------------------------
        # Optional cleanup timer
        # ------------------------------------------------------------------
        if self.cleanup_inactive_topics:
            self.create_timer(1.0, self._cleanup_topics_callback)
            self.get_logger().info(
                f"Topic cleanup enabled with timeout: {self.inactive_topic_timeout}s"
            )

        self.get_logger().info("ASR node initialized, waiting for audio...")

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _audio_callback(self, msg: AudioAndDeviceInfo) -> None:
        if self.engine.sample_rate is None:
            self.engine.set_sample_rate(int(msg.device_samplerate))
            self.get_logger().info(
                f"ASR initialized with device: {msg.device_name} "
                f"(Sample rate: {msg.device_samplerate} Hz)"
            )
        self.engine.push_audio(np.array(msg.audio, dtype=np.float32))

    def _vad_callback(self, msg: Vad) -> None:
        self.engine.update_vad(msg.vad_probability)

    def _speech_activity_callback(self, msg: SpeechActivityDetection) -> None:
        self.engine.update_speaker(msg.speaker_id)

        # Create ROS4HRI speech publisher for this speaker if needed
        if self.ros4hri_enabled and msg.speaker_id and msg.speaker_id != "unknown":
            if msg.speaker_id not in self.voice_publishers:
                self._create_voice_publisher(msg.speaker_id)
            self.voice_publishers_activity[msg.speaker_id] = time.time()

    # ------------------------------------------------------------------
    # Engine callback
    # ------------------------------------------------------------------

    def _publish_transcript(self, transcript: str, speaker_id: str, language_code: str) -> None:
        """Called by the engine when a transcript is ready. Stamps and publishes."""
        msg = SpeechResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.transcript = transcript
        msg.speaker_id = speaker_id
        msg.language_code = language_code
        msg.transcript_confidence = 0.0
        msg.speaker_id_confidence = 0.0
        self.asr_pub.publish(msg)

        self.get_logger().info(
            f"Published transcript: '{transcript}' (lang: {language_code}, speaker: {speaker_id})"
        )

        if self.ros4hri_enabled and speaker_id and speaker_id != "unknown":
            self._publish_ros4hri_speech(speaker_id, transcript, language_code)

    # ------------------------------------------------------------------
    # ROS4HRI voice publisher lifecycle
    # ------------------------------------------------------------------

    def _create_voice_publisher(self, speaker_id: str) -> None:
        topic = f"/humans/voices/{speaker_id}/speech"
        self.voice_publishers[speaker_id] = self.create_publisher(LiveSpeech, topic, 10)
        self.voice_publishers_activity[speaker_id] = time.time()
        self.get_logger().debug(f"Created speech publisher for speaker: {speaker_id}")

    def _publish_ros4hri_speech(self, speaker_id: str, transcript: str, language_code: str) -> None:
        if speaker_id not in self.voice_publishers:
            self._create_voice_publisher(speaker_id)
        self.voice_publishers_activity[speaker_id] = time.time()

        msg = LiveSpeech()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.final = transcript
        msg.incremental = ""
        msg.confidence = 0.0
        msg.locale = language_code
        self.voice_publishers[speaker_id].publish(msg)

    def _cleanup_topics_callback(self) -> None:
        current_time = time.time()
        to_remove = [
            sid
            for sid, last_active in self.voice_publishers_activity.items()
            if current_time - last_active > self.inactive_topic_timeout
        ]
        for sid in to_remove:
            self.get_logger().info(f"Destroying inactive publisher for speaker: {sid}")
            self.destroy_publisher(self.voice_publishers.pop(sid))
            del self.voice_publishers_activity[sid]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.engine.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ASR node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
