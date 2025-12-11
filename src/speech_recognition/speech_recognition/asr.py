import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
import torch
from faster_whisper import WhisperModel
from rclpy.node import Node

from hri_msgs.msg import AudioAndDeviceInfo, SpeechActivityDetection, SpeechResult, Vad, LiveSpeech


class ASRNode(Node):
    def __init__(self):
        super().__init__("asr_node")

        # Declare parameters
        self.declare_parameter("model_size", "turbo")
        self.declare_parameter("vad_threshold", 0.5)
        self.declare_parameter("min_silence_duration", 1.0)  # seconds
        self.declare_parameter("max_chunk_duration", 30.0)  # seconds
        self.declare_parameter(
            "silence_detection_threshold", 0.00001
        )  # RMS threshold for silence detection
        self.declare_parameter(
            "pre_buffer_duration", 0.3
        )  # seconds of audio to prepend
        self.declare_parameter("ros4hri_with_id", True)
        self.declare_parameter("cleanup_inactive_topics", False)
        self.declare_parameter("inactive_topic_timeout", 10.0)

        # Get parameter values
        self.model_size = (
            self.get_parameter("model_size").get_parameter_value().string_value
        )
        self.vad_threshold = (
            self.get_parameter("vad_threshold").get_parameter_value().double_value
        )
        self.min_silence_duration = (
            self.get_parameter("min_silence_duration")
            .get_parameter_value()
            .double_value
        )
        self.max_chunk_duration = (
            self.get_parameter("max_chunk_duration").get_parameter_value().double_value
        )
        self.silence_detection_threshold = (
            self.get_parameter("silence_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.pre_buffer_duration = (
            self.get_parameter("pre_buffer_duration").get_parameter_value().double_value
        )
        self.ros4hri_enabled = (
            self.get_parameter("ros4hri_with_id").get_parameter_value().bool_value
        )
        self.cleanup_inactive_topics = (
            self.get_parameter("cleanup_inactive_topics").get_parameter_value().bool_value
        )
        self.inactive_topic_timeout = (
            self.get_parameter("inactive_topic_timeout").get_parameter_value().double_value
        )

        self.get_logger().info(f"Using VAD: {self.vad_threshold}")

        self.possible_model_sizes = {
            "tiny.en": "Systran/faster-whisper-tiny.en",
            "tiny": "Systran/faster-whisper-tiny",
            "base.en": "Systran/faster-whisper-base.en",
            "base": "Systran/faster-whisper-base",
            "small.en": "Systran/faster-whisper-small.en",
            "small": "Systran/faster-whisper-small",
            "medium.en": "Systran/faster-whisper-medium.en",
            "medium": "Systran/faster-whisper-medium",
            "large-v1": "Systran/faster-whisper-large-v1",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "Systran/faster-whisper-large-v3",
            "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
            "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
            "distil-small.en": "Systran/faster-distil-whisper-small.en",
            "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
            "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
            "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        }

        model_path = None

        if self.model_size not in self.possible_model_sizes:
            self.get_logger().error(
                f"Invalid model_size parameter: {self.model_size}. Available models: {list(self.possible_model_sizes.keys())}"
            )
            raise ValueError(f"Invalid model_size: {self.model_size}")
        else:
            self.model_dir = self.possible_model_sizes[self.model_size]
            # To load the model from local, we need to convert the directory name to the model path
            model_path = "models--" + self.model_dir.replace("/", "--")

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Search model inside "weights" folder (next to this file)
        # If "weights" folder does not exist, it will be created automatically by faster-whisper
        weights_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "weights")
        )

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            self.get_logger().info(f"Weights directory created: {weights_dir}")

        # Model setup - try to find a local snapshot under weights/ and use it,
        # otherwise fall back to using the HF repo id with download_root.
        resolved_model_path = None
        for name in os.listdir(weights_dir):
            if name == model_path:
                resolved_model_path = os.path.join(weights_dir, name)
                self.get_logger().info(
                    f"Found local snapshot for model: {resolved_model_path}"
                )
                # Walk through subdirectories to find the actual model folder (model.bin)
                for root, _, files in os.walk(resolved_model_path):
                    if "model.bin" in files:
                        resolved_model_path = root
                        break

        if resolved_model_path and os.path.isdir(resolved_model_path):
            self.get_logger().info(
                f"Using local snapshot for model: {resolved_model_path}"
            )
            model_arg = resolved_model_path
            # when passing a snapshot folder, no need to set download_root
            self.model = WhisperModel(
                model_arg, device=self.device, compute_type="float32"
            )
        else:
            self.get_logger().info(
                f"No local snapshot found, using HuggingFace repo id: {self.model_size} (download_root={weights_dir})"
            )
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float32",
                download_root=weights_dir,
            )
        green = "\033[92m"
        reset = "\033[0m"
        self.get_logger().info(f"{green}Model {self.model_size} loaded.{reset}")

        # Audio processing variables
        self.sample_rate = None
        self.audio_buffer = deque()
        self.vad_state = False
        self.last_vad_change_time = 0.0
        self.speech_start_time = 0.0
        self.last_silence_time = 0.0
        self.speech_interrupted = False

        self.speaker_id = None
        self.speaker_history = deque()  # Store speaker_id with timestamps

        # Thread safety
        self.buffer_lock = threading.RLock()  # Use RLock to prevent deadlock
        self.processing_thread = None
        self.should_stop = False
        self.speech_cancelled = threading.Event()  # Signal to cancel speech processing

        # Subscribers
        self.audio_and_device_info_sub = self.create_subscription(
            AudioAndDeviceInfo,
            "audio_and_device_info",
            self.audio_and_device_info_callback,
            10,
        )
        self.vad_sub = self.create_subscription(
            Vad,
            "vad",
            self.vad_callback,
            10,
        )
        self.speech_activity_sub = self.create_subscription(
            SpeechActivityDetection,
            "speech_activity_detection",
            self.speech_activity_callback,
            10,
        )

        # Publishers
        self.asr_pub = self.create_publisher(SpeechResult, "speech_result", 10)
        
        # ROS4HRI Publishers
        self.voice_publishers = {} # Map speaker_id to dict of publishers
        self.voice_publishers_activity = {} # Map speaker_id to last activity timestamp

        # Cleanup timer
        if self.cleanup_inactive_topics:
            self.create_timer(1.0, self.cleanup_topics_callback)
            self.get_logger().info(f"Topic cleanup enabled with timeout: {self.inactive_topic_timeout}s")

        self.get_logger().info("ASR Node initialized, waiting for audio...")

    def cleanup_topics_callback(self):
        """Check for inactive topics and destroy them"""
        current_time = time.time()
        speakers_to_remove = []

        for speaker_id, last_active in self.voice_publishers_activity.items():
            if current_time - last_active > self.inactive_topic_timeout:
                speakers_to_remove.append(speaker_id)

        for speaker_id in speakers_to_remove:
            self.get_logger().info(f"Destroying inactive publishers for {speaker_id}")
            if speaker_id in self.voice_publishers:
                self.destroy_publisher(self.voice_publishers[speaker_id])
                del self.voice_publishers[speaker_id]
            
            del self.voice_publishers_activity[speaker_id]

    def audio_and_device_info_callback(self, msg: AudioAndDeviceInfo):
        """Process incoming audio data"""
        if self.sample_rate is None:
            self.sample_rate = int(msg.device_samplerate)
            self.get_logger().info(
                f"ASR initialized with audio device: {msg.device_name} (Sample rate: {msg.device_samplerate} Hz)"
            )

        # Convert audio to numpy array.
        audio_data = np.array(msg.audio, dtype=np.float32)

        with self.buffer_lock:
            # Add audio data with timestamp
            current_time = time.time()
            self.audio_buffer.append({"audio": audio_data, "timestamp": current_time})

            # Keep only last 35 seconds of audio (5s buffer beyond max chunk).
            # Additional 5 seconds for safety margin in case that we need to split chunks.
            buffer_duration = 35.0
            cutoff_time = current_time - buffer_duration
            while self.audio_buffer and self.audio_buffer[0]["timestamp"] < cutoff_time:
                self.audio_buffer.popleft()

    def vad_callback(self, msg: Vad):
        current_time = time.time()
        new_vad_state = msg.vad_probability > self.vad_threshold

        if new_vad_state != self.vad_state:
            self.get_logger().debug(
                f"VAD state changed: {self.vad_state} -> {new_vad_state}"
            )

            if new_vad_state:
                # Speech started or reactivated
                self.speech_cancelled.set()  # Cancel any pending processing

                if self.speech_start_time == 0:
                    # New speech started
                    self.speech_start_time = current_time
                    self.get_logger().debug("Speech started")
                else:
                    # Speech continued after brief pause
                    self.get_logger().debug("Speech continued from previous segment")

                self.speech_interrupted = False
            else:
                # Speech ended
                self.last_silence_time = current_time
                self.get_logger().debug("Speech ended, starting silence timer")

                # Clear the cancel flag for new processing
                self.speech_cancelled.clear()

                # Start processing thread if not already running
                if (
                    self.processing_thread is None
                    or not self.processing_thread.is_alive()
                ):
                    self.processing_thread = threading.Thread(
                        target=self._process_speech_end
                    )
                    self.processing_thread.daemon = True
                    self.processing_thread.start()

        self.vad_state = new_vad_state
        self.last_vad_change_time = current_time

        # Check for long speech chunks that need to be split
        if self.vad_state and self.speech_start_time > 0:
            speech_duration = current_time - self.speech_start_time
            if speech_duration >= self.max_chunk_duration:
                self.get_logger().debug(
                    f"Speech duration exceeded {self.max_chunk_duration}s, forcing chunk split"
                )
                self._force_chunk_split()

    def speech_activity_callback(self, msg: SpeechActivityDetection):
        """Process speech activity detection results"""
        self.speaker_id = msg.speaker_id
        
        # Create speech publisher for ROS4HRI when a new speaker is detected
        if self.ros4hri_enabled and self.speaker_id and self.speaker_id != "unknown":
            if self.speaker_id not in self.voice_publishers:
                topic = f"/humans/voices/{self.speaker_id}/speech"
                self.voice_publishers[self.speaker_id] = self.create_publisher(LiveSpeech, topic, 10)
                self.get_logger().debug(f"Created speech publisher for speaker: {self.speaker_id}")
            self.voice_publishers_activity[self.speaker_id] = time.time()

    def _process_speech_end(self):
        """Process speech when VAD goes from on to off"""
        self.get_logger().debug(
            f"Started silence timer, waiting {self.min_silence_duration}s"
        )

        # Wait for min_silence_duration or until cancelled (VAD reactivated)
        was_cancelled = self.speech_cancelled.wait(timeout=self.min_silence_duration)

        if was_cancelled:
            # VAD reactivated before timeout - cancel processing
            self.speech_interrupted = False
            self.get_logger().debug("VAD reactivated, canceling speech processing")
            return

        # Timeout reached - check if still in silence and process
        if not self.vad_state and self.last_silence_time > 0:
            self.get_logger().debug("Processing speech chunk after silence timeout")
            self._transcribe_speech_chunk()
            self.speech_interrupted = False
        else:
            self.get_logger().debug(
                "Silence timeout reached but VAD state changed, skipping transcription"
            )

    def _extract_audio_data(self, end_time):
        """Extract audio data from buffer (call this INSIDE the lock)"""
        start_time = self.speech_start_time
        if start_time <= 0:
            return None

        # Instead of start_time, use an earlier time to prevent cutting the beginning
        actual_start_time = start_time - self.pre_buffer_duration

        # Collect audio data for the speech chunk
        audio_chunks = []
        for audio_chunk in self.audio_buffer:
            chunk_time = audio_chunk["timestamp"]
            if actual_start_time <= chunk_time <= end_time:
                audio_chunks.append(audio_chunk["audio"])

        if not audio_chunks:
            return None

        # Concatenate audio chunks
        return np.concatenate(audio_chunks)

    def _force_chunk_split(self):
        """Force a chunk split due to maximum duration"""
        audio_data = None
        split_time = None

        with self.buffer_lock:
            if not self.audio_buffer:
                return

            # Find the best split point (silence) in the middle portion of the chunk
            current_time = time.time()
            chunk_start_time = self.speech_start_time

            # Look for silence in the middle 50% of the chunk
            middle_start = chunk_start_time + (current_time - chunk_start_time) * 0.25
            middle_end = chunk_start_time + (current_time - chunk_start_time) * 0.75

            best_split_time = None
            min_rms = float("inf")

            # Find the quietest moment in the middle section
            for audio_chunk in self.audio_buffer:
                chunk_time = audio_chunk["timestamp"]
                if middle_start <= chunk_time <= middle_end:
                    rms = np.sqrt(np.mean(audio_chunk["audio"] ** 2))
                    if rms < min_rms:
                        min_rms = rms
                        best_split_time = chunk_time

            # Determine split time and extract audio data
            if best_split_time and min_rms < self.silence_detection_threshold:
                split_time = best_split_time
                self.get_logger().debug(
                    f"Found silence at {best_split_time}, splitting chunk"
                )
            else:
                split_time = current_time
                self.get_logger().debug("No silence found, splitting at current time")

            # Extract audio data INSIDE the lock
            audio_data = self._extract_audio_data(split_time)

        # Process transcription OUTSIDE the lock
        if audio_data is not None:
            self._transcribe_speech_chunk_with_data(audio_data, split_time)
            self.speech_start_time = split_time

    def _transcribe_speech_chunk(self, end_time=None):
        """Transcribe the accumulated speech chunk"""
        if end_time is None:
            end_time = (
                self.last_silence_time if self.last_silence_time > 0 else time.time()
            )

        # Extract audio data with lock
        with self.buffer_lock:
            audio_data = self._extract_audio_data(end_time)

        # Process transcription without lock
        if audio_data is not None:
            self._transcribe_speech_chunk_with_data(audio_data, end_time)

    def _transcribe_speech_chunk_with_data(self, audio_data, end_time):
        """Transcribe with pre-extracted audio data (NO lock needed)"""
        if audio_data is None or len(audio_data) == 0:
            self.get_logger().warn("No audio data found for transcription")
            return

        # Check if we have enough audio (at least 0.01 seconds)
        min_duration = 0.01

        if self.sample_rate is None or len(audio_data) < int(
            self.sample_rate * min_duration
        ):
            self.get_logger().warn(
                "Audio chunk too short for transcription or sample rate not set"
            )
            return

        self.get_logger().info(
            f"Transcribing {len(audio_data) / self.sample_rate:.2f}s of audio"
        )

        # Perform transcription
        try:
            segments, info = self.model.transcribe(
                audio_data,
                vad_filter=True,
                word_timestamps=False,
                language="en",
            )

            # Combine all segments into one transcript
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())

            transcript = " ".join(transcript_parts).strip()

            if transcript:
                # Publish ASR result
                asr_msg = SpeechResult()
                asr_msg.header.stamp = self.get_clock().now().to_msg()
                asr_msg.transcript = transcript
                asr_msg.speaker_id = self.speaker_id if self.speaker_id else "unknown"
                asr_msg.language_code = (
                    info.language if hasattr(info, "language") else "unknown"
                )
                # Set default confidence since faster-whisper might not return per-transcript confidence easily aggregated
                # Or we can take average of segment avg_logprob converted to prob, but 0.0 is consistent with existing code
                asr_msg.transcript_confidence = 0.0 
                asr_msg.speaker_id_confidence = 0.0

                self.asr_pub.publish(asr_msg)

                # Handle ROS4HRI LiveSpeech
                if self.ros4hri_enabled and asr_msg.speaker_id and asr_msg.speaker_id != "unknown":
                    self._publish_ros4hri_speech(asr_msg.speaker_id, transcript, asr_msg.language_code)

                # Print info log in yellow color
                yellow = "\033[93m"
                reset = "\033[0m"
                self.get_logger().info(
                    f"{yellow}Published transcript: '{transcript}' (lang: {asr_msg.language_code}, conf: {asr_msg.transcript_confidence:.2f}){reset}"
                )
            else:
                self.get_logger().info("Empty transcript, not publishing")

        except Exception as e:
            self.get_logger().error(f"Transcription failed: {e}")

        # Reset speech timing
        self.speech_start_time = 0.0
        self.last_silence_time = 0.0

    def _publish_ros4hri_speech(self, speaker_id, transcript, language_code):
        """Publish LiveSpeech message for ROS4HRI"""
        # Ensure publisher exists
        if speaker_id not in self.voice_publishers:
            topic = f"/humans/voices/{speaker_id}/speech"
            self.voice_publishers[speaker_id] = self.create_publisher(LiveSpeech, topic, 10)
        
        self.voice_publishers_activity[speaker_id] = time.time()
        
        msg = LiveSpeech()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.final = transcript
        msg.incremental = "" # We only have final result here
        msg.confidence = 0.0 # Placeholder
        msg.locale = language_code # Using language code as locale for now
        
        self.voice_publishers[speaker_id].publish(msg)

    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
        self.should_stop = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    asr_node = ASRNode()

    try:
        rclpy.spin(asr_node)
    except KeyboardInterrupt:
        asr_node.get_logger().info("Shutting down ASR node.")
    finally:
        asr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
