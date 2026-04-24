import signal
import time
from typing import Dict, Optional, Set

import numpy as np
import rclpy
import torch
from audio_common_msgs.msg import AudioData
from hri_msgs.msg import AudioAndDeviceInfo, IdsList, SpeechActivityDetection, Vad
from rclpy.node import Node
from std_msgs.msg import Bool

from speech_recognition.diarization_engine import DiarizationEngine


class DiarizationNode(Node):
    """Thin ROS2 node: declares parameters, owns all publishers/subscribers,
    and wires the DiarizationEngine.

    All diarization logic (model loading, pipeline, observer, embedding
    extraction, speaker mapping, database) lives in DiarizationEngine.

    This node's jobs are:
      1. Read ROS2 parameters and pass them to the engine.
      2. Feed audio chunks and VAD probabilities into the engine.
      3. Own VAD buffering and publish SpeechActivityDetection on state changes.
      4. Manage ROS4HRI voice publisher lifecycle (create, cleanup, publish).
    """

    def __init__(self):
        super().__init__("diarization_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("chunk_duration", 2.0)
        self.declare_parameter("overlap_duration", 0.5)
        self.declare_parameter("segmentation_model_name", "pyannote/segmentation")
        self.declare_parameter("embedding_model_name", "pyannote/embedding")
        self.declare_parameter("vad_threshold", 0.5)
        self.declare_parameter("use_database", True)
        self.declare_parameter("similarity_threshold", 0.3)
        self.declare_parameter("ros4hri_with_id", True)
        self.declare_parameter("cleanup_inactive_topics", False)
        self.declare_parameter("inactive_topic_timeout", 10.0)
        self.declare_parameter("init_retry_backoff_sec", 10.0)

        self.vad_threshold = self.get_parameter("vad_threshold").get_parameter_value().double_value
        self.ros4hri_enabled = (
            self.get_parameter("ros4hri_with_id").get_parameter_value().bool_value
        )
        self.cleanup_inactive_topics = (
            self.get_parameter("cleanup_inactive_topics").get_parameter_value().bool_value
        )
        self.inactive_topic_timeout = (
            self.get_parameter("inactive_topic_timeout").get_parameter_value().double_value
        )
        self.init_retry_backoff_sec = (
            self.get_parameter("init_retry_backoff_sec").get_parameter_value().double_value
        )

        # ------------------------------------------------------------------
        # Engine
        # ------------------------------------------------------------------
        self.engine = DiarizationEngine(
            chunk_duration=self.get_parameter("chunk_duration").get_parameter_value().double_value,
            overlap_duration=self.get_parameter("overlap_duration")
            .get_parameter_value()
            .double_value,
            segmentation_model_name=self.get_parameter("segmentation_model_name")
            .get_parameter_value()
            .string_value,
            embedding_model_name=self.get_parameter("embedding_model_name")
            .get_parameter_value()
            .string_value,
            vad_threshold=self.vad_threshold,
            similarity_threshold=self.get_parameter("similarity_threshold")
            .get_parameter_value()
            .double_value,
            use_database=self.get_parameter("use_database").get_parameter_value().bool_value,
            ros4hri_enabled=self.ros4hri_enabled,
            on_eut_speaker_changed=self._on_eut_speaker_changed,
            on_voice_update=self._on_voice_update,
            logger=self.get_logger(),
        )

        # ------------------------------------------------------------------
        # State owned by the node
        # ------------------------------------------------------------------
        self._device_initialized: bool = False
        self._next_init_retry_at: float = 0.0
        self._init_attempts: int = 0
        self._eut_speaker_id: Optional[str] = None
        self._speaker_activated: bool = False

        # VAD buffering — the node decides when to publish active/inactive
        self._vad_buffer: list = []
        self._vad_buffer_size: int = 30
        self._vad_rms_threshold: float = 0.5

        # ROS4HRI voice publisher registry
        self.voice_publishers: Dict[str, Dict] = {}
        self.voice_publishers_activity: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.speech_activity_pub = self.create_publisher(
            SpeechActivityDetection, "speech_activity_detection", 10
        )
        if self.ros4hri_enabled:
            self.ids_pub = self.create_publisher(IdsList, "/humans/voices/tracked", 1)

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

        # ------------------------------------------------------------------
        # Optional cleanup timer
        # ------------------------------------------------------------------
        if self.cleanup_inactive_topics:
            self.create_timer(1.0, self._cleanup_topics_callback)
            self.get_logger().info(
                f"Topic cleanup enabled with timeout: {self.inactive_topic_timeout}s"
            )

        self.get_logger().info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        self.get_logger().info("Diarization node initialized, waiting for audio and VAD info...")

    # ------------------------------------------------------------------
    # Audio subscriber — initializes engine on first message, then feeds audio
    # ------------------------------------------------------------------

    def _audio_callback(self, msg: AudioAndDeviceInfo) -> None:
        if not self._device_initialized:
            now = time.time()
            if now < self._next_init_retry_at:
                return

            self._init_attempts += 1
            self.get_logger().info(
                f"Listening audio from device: {msg.device_name} "
                f"(Sample rate: {msg.device_samplerate} Hz) "
                f"[attempt {self._init_attempts}]"
            )
            if not msg.device_samplerate or msg.device_samplerate <= 0:
                self.get_logger().error(f"Invalid sample rate: {msg.device_samplerate}")
                return

            success = self.engine.initialize(int(msg.device_samplerate))
            if success:
                self._device_initialized = True
            else:
                self._next_init_retry_at = now + max(self.init_retry_backoff_sec, 0.0)
                self.get_logger().warn(
                    f"Diarization init failed. Retrying in {self.init_retry_backoff_sec:.1f}s."
                )
                return

        audio_data = np.array(msg.audio, dtype=np.float32)
        self.engine.push_audio(audio_data)

    # ------------------------------------------------------------------
    # VAD subscriber — buffers probabilities, publishes speech activity
    # ------------------------------------------------------------------

    def _vad_callback(self, msg: Vad) -> None:
        if not hasattr(self, "_vad_info_logged"):
            self.get_logger().info("VAD information received.")
            self._vad_info_logged = True

        probability = msg.vad_probability
        self.engine.update_vad_probability(probability)

        self._vad_buffer.append(probability)
        if len(self._vad_buffer) > self._vad_buffer_size:
            self._vad_buffer.pop(0)

        vad_rms = 0.0
        if len(self._vad_buffer) >= self._vad_buffer_size:
            arr = np.array(self._vad_buffer)
            vad_rms = float(np.sqrt(np.mean(arr**2)))

        sustained_silence = (
            len(self._vad_buffer) >= self._vad_buffer_size and vad_rms <= self._vad_rms_threshold
        )

        if sustained_silence and self._eut_speaker_id is not None and self._speaker_activated:
            self._publish_speech_activity(self._eut_speaker_id, active=False)
            self.get_logger().info(
                f"Speech ended: speaker={self._eut_speaker_id.replace('EUT_', '')}"
            )
            self._eut_speaker_id = None
            self._speaker_activated = False

        elif probability > self.vad_threshold and self._eut_speaker_id is not None:
            self._publish_speech_activity(self._eut_speaker_id, active=True)
            self._speaker_activated = True

    # ------------------------------------------------------------------
    # Engine callbacks
    # ------------------------------------------------------------------

    def _on_eut_speaker_changed(self, eut_speaker_id: Optional[str]) -> None:
        """Called by the engine when the active EUT speaker changes."""
        self._eut_speaker_id = eut_speaker_id

    def _on_voice_update(
        self,
        active_eut_speakers: Set[str],
        audio_block: Optional[np.ndarray],
    ) -> None:
        """Called by the engine each diarization step with active speakers + audio."""
        current_ros4hri_ids = {s.replace("EUT_", "") for s in active_eut_speakers}

        # Publish tracked voices list
        ids_msg = IdsList()
        ids_msg.header.stamp = self.get_clock().now().to_msg()
        ids_msg.ids = list(current_ros4hri_ids)
        self.ids_pub.publish(ids_msg)

        # Per-voice publishers
        for eut_id in active_eut_speakers:
            ros4hri_id = eut_id.replace("EUT_", "")
            if ros4hri_id not in self.voice_publishers:
                self._create_voice_publishers(ros4hri_id)
            self.voice_publishers_activity[ros4hri_id] = time.time()

            bool_msg = Bool()
            bool_msg.data = True
            self.voice_publishers[ros4hri_id]["is_speaking"].publish(bool_msg)

            if audio_block is not None:
                audio_int16 = (audio_block * 32767).clip(-32768, 32767).astype(np.int16)
                audio_msg = AudioData()
                audio_msg.data = audio_int16.tobytes()
                self.voice_publishers[ros4hri_id]["audio"].publish(audio_msg)

        # Speakers that stopped
        prev_eut_speakers = {
            f"EUT_{rid}" for rid in self.voice_publishers if rid not in current_ros4hri_ids
        }
        for eut_id in prev_eut_speakers:
            ros4hri_id = eut_id.replace("EUT_", "")
            if ros4hri_id in self.voice_publishers:
                bool_msg = Bool()
                bool_msg.data = False
                self.voice_publishers[ros4hri_id]["is_speaking"].publish(bool_msg)
                self.voice_publishers_activity[ros4hri_id] = time.time()

    # ------------------------------------------------------------------
    # Speech activity publishing
    # ------------------------------------------------------------------

    def _publish_speech_activity(self, eut_speaker_id: str, active: bool) -> None:
        msg = SpeechActivityDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.speaker_id = eut_speaker_id.replace("EUT_", "")
        msg.active = active
        self.speech_activity_pub.publish(msg)

    # ------------------------------------------------------------------
    # ROS4HRI voice publisher lifecycle
    # ------------------------------------------------------------------

    def _create_voice_publishers(self, voice_id: str) -> None:
        base_topic = f"/humans/voices/{voice_id}"
        self.voice_publishers[voice_id] = {
            "audio": self.create_publisher(AudioData, f"{base_topic}/audio", 10),
            "is_speaking": self.create_publisher(Bool, f"{base_topic}/is_speaking", 10),
        }
        self.voice_publishers_activity[voice_id] = time.time()
        self.get_logger().info(f"Created ROS4HRI publishers for {voice_id}")

    def _cleanup_topics_callback(self) -> None:
        current_time = time.time()
        to_remove = [
            vid
            for vid, last_active in self.voice_publishers_activity.items()
            if current_time - last_active > self.inactive_topic_timeout
        ]
        for vid in to_remove:
            self.get_logger().info(f"Destroying inactive publishers for {vid}")
            for pub in self.voice_publishers[vid].values():
                self.destroy_publisher(pub)
            del self.voice_publishers[vid]
            del self.voice_publishers_activity[vid]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down diarization node...")
        try:
            self.engine.stop()
        except Exception as e:
            self.get_logger().warn(f"Error during engine shutdown: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DiarizationNode()

    def shutdown_handler(signum, frame):
        node.get_logger().info(f"Signal {signum} received, shutting down...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down diarization node.")
    except SystemExit:
        node.get_logger().info("SystemExit caught, cleaning up...")
    finally:
        try:
            if rclpy.ok():
                node.destroy_node()
                rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()
