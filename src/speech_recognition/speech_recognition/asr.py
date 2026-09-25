import json
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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from speech_recognition.asr_engine import ASREngine
from speech_recognition.model_weights import weights_dir

ASR_BACKENDS = ("whisper", "parakeet")


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
        self.declare_parameter("diarization_offset", 0.0)
        self.declare_parameter("min_speaker_run_tokens", 2)
        self.declare_parameter("min_speaker_run_duration", 0.4)
        self.declare_parameter("snap_splits_to_sentences", True)
        self.declare_parameter("speaker_interval_tolerance", 3.0)
        self.declare_parameter("min_speaker_chunk_duration", 0.3)
        self.declare_parameter("unknown_speaker_grace", 0.5)
        self.declare_parameter("asr_backend", "whisper")
        self.declare_parameter("parakeet_model_name", "nvidia/parakeet-tdt-0.6b-v3")
        self.declare_parameter("parakeet_language_id_model", "langid_ambernet")
        self.declare_parameter("parakeet_language_id_min_confidence", 0.9)
        self.declare_parameter("ros4hri_with_id", True)
        self.declare_parameter("cleanup_inactive_topics", False)
        self.declare_parameter("inactive_topic_timeout", 10.0)

        model_size = self.get_parameter("model_size").get_parameter_value().string_value

        # Backend selection: both backends share the ASREngine pipeline and
        # publish the same SpeechResult; only the model differs.
        backend = self.get_parameter("asr_backend").get_parameter_value().string_value
        backend = backend.strip().lower()
        if backend not in ASR_BACKENDS:
            raise ValueError(f"Unsupported asr_backend '{backend}'. Use one of {ASR_BACKENDS}.")

        engine_class = ASREngine
        backend_options = {}
        if backend == "parakeet":
            # Imported only when selected: the Whisper backend never loads NeMo.
            from speech_recognition.parakeet_asr_engine import ParakeetASREngine

            engine_class = ParakeetASREngine
            backend_options["parakeet_model_name"] = (
                self.get_parameter("parakeet_model_name").get_parameter_value().string_value
            )
            backend_options["language_id_model"] = (
                self.get_parameter("parakeet_language_id_model").get_parameter_value().string_value
            )
            backend_options["language_id_min_confidence"] = (
                self.get_parameter("parakeet_language_id_min_confidence")
                .get_parameter_value()
                .double_value
            )
        self.get_logger().info(f"Selected ASR backend: {backend}")

        # Validate model size early so the node fails fast with a clear message
        try:
            engine_class.validate_model_size(model_size)
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

        # ------------------------------------------------------------------
        # Engine
        # ------------------------------------------------------------------
        self.engine = engine_class(
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
            diarization_offset=self.get_parameter("diarization_offset")
            .get_parameter_value()
            .double_value,
            min_speaker_run_tokens=self.get_parameter("min_speaker_run_tokens")
            .get_parameter_value()
            .integer_value,
            min_speaker_run_duration=self.get_parameter("min_speaker_run_duration")
            .get_parameter_value()
            .double_value,
            snap_splits_to_sentences=self.get_parameter("snap_splits_to_sentences")
            .get_parameter_value()
            .bool_value,
            speaker_interval_tolerance=self.get_parameter("speaker_interval_tolerance")
            .get_parameter_value()
            .double_value,
            min_speaker_chunk_duration=self.get_parameter("min_speaker_chunk_duration")
            .get_parameter_value()
            .double_value,
            unknown_speaker_grace=self.get_parameter("unknown_speaker_grace")
            .get_parameter_value()
            .double_value,
            weights_dir=weights_dir(),
            on_transcript_ready=self._publish_transcript,
            logger=self.get_logger(),
            **backend_options,
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
        # SpeechResult (hri_msgs) has no field for edge timing diagnostics: its
        # only string/float slots are transcript_confidence and locale, and both
        # are now used for what they actually mean. Processing/audio/realtime
        # timing rides a sibling std_msgs/String topic instead, stamped
        # identically to the SpeechResult it belongs with so a subscriber (the
        # Android bridge, capture_hyp.py) can pair the two by header.stamp.
        self.asr_timing_pub = self.create_publisher(String, "speech_result_timing", 10)

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
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.engine.update_speaker(
            msg.speaker_id,
            bool(msg.active),
            float(msg.speaker_id_confidence),
            stamp=stamp if stamp > 0 else None,
        )

        # Create ROS4HRI speech publisher for this speaker if needed
        if self.ros4hri_enabled and msg.speaker_id and msg.speaker_id != "unknown":
            if msg.speaker_id not in self.voice_publishers:
                self._create_voice_publisher(msg.speaker_id)
            self.voice_publishers_activity[msg.speaker_id] = time.time()

    # ------------------------------------------------------------------
    # Engine callback
    # ------------------------------------------------------------------

    def _publish_transcript(
        self,
        transcript: str,
        speaker_id: str,
        language_code: str,
        transcript_confidence: float,
        processing_ms: int,
        vad_wait_ms: int,
        audio_duration_ms: int,
        realtime_factor: float,
        speaker_confidence: float = -1.0,
    ) -> None:
        """Called by the engine when a transcript is ready. Stamps and publishes."""
        msg = SpeechResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.transcript = transcript
        msg.speaker_id = speaker_id
        msg.language_code = language_code
        # Real ASR confidence in [0, 1] (per hri_msgs/SpeechResult.msg): 0.0 for
        # Whisper, which has no well-calibrated per-word score to report; the
        # NeMo confidence estimator's mean word score for Parakeet. See
        # ASREngine._chunk_confidence / ParakeetASREngine._hypothesis_confidence.
        msg.transcript_confidence = float(transcript_confidence)
        # How much this attribution is worth: the identity match score weighted by
        # how much of the utterance that speaker actually held. -1.0 means the
        # backend reported no score, which downstream must read as "unavailable",
        # never as "low" — EutPersonManager weighs a voice link by this value.
        msg.speaker_id_confidence = float(speaker_confidence)
        # Bare ISO 639-1 code (no region: neither backend detects one). Empty
        # when the backend published without a language at all.
        msg.locale = language_code or ""

        # Published just ahead of SpeechResult (same stamp, so a subscriber can
        # pair them). A subscriber must still not assume arrival order across
        # the two topics — see android_transcript_bridge.py's _take_timing().
        #
        # processing_ms is pure model compute time; vad_wait_ms is how long the
        # engine sat on the VAD silence timer before that started (0 for a
        # forced max-duration split or a speaker-change flush, neither of which
        # waits on VAD). Kept apart rather than blended into one number.
        timing = String()
        timing.data = json.dumps(
            {
                "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
                "processing_ms": int(processing_ms),
                "vad_wait_ms": int(vad_wait_ms),
                "audio_duration_ms": int(audio_duration_ms),
                "realtime_factor": float(realtime_factor),
            }
        )
        self.asr_timing_pub.publish(timing)
        self.asr_pub.publish(msg)

        self.get_logger().info(
            f"Published transcript: '{transcript}' (lang: {language_code}, speaker: {speaker_id}"
            f"@{speaker_confidence:.2f}, transcript_conf={transcript_confidence:.2f}, "
            f"proc={processing_ms}ms, vad_wait={vad_wait_ms}ms, audio={audio_duration_ms}ms, x{realtime_factor:.2f})"
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
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutting down ASR node.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
