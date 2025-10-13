import rclpy
from rclpy.node import Node
import torch
import time
import numpy as np
import threading
from collections import deque
from faster_whisper import WhisperModel
from audio_stream_manager_interfaces.msg import Asr, AudioAndDeviceInfo, Vad

MODEL_SIZE = "medium"
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION = 1.0  # seconds
MAX_CHUNK_DURATION = 30.0  # seconds
SILENCE_DETECTION_THRESHOLD = 0.01  # RMS threshold for silence detection
PRE_BUFFER_DURATION = (
    1.0  # seconds of audio to prepend the loss of informationbefore VAD start
)


class ASRNode(Node):
    def __init__(self):
        super().__init__("asr_node")
        self.get_logger().info("ASR Node has been started.")

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load Whisper model
        self.model = WhisperModel(
            MODEL_SIZE, device=self.device, compute_type="float32"
        )
        self.get_logger().info(f"Model {MODEL_SIZE} loaded.")

        # Audio processing variables
        self.sample_rate = None
        self.audio_buffer = deque()
        self.vad_state = False
        self.last_vad_change_time = 0.0
        self.speech_start_time = 0.0
        self.last_silence_time = 0.0
        self.speech_interrupted = False

        # Thread safety
        self.buffer_lock = threading.Lock()
        self.processing_thread = None
        self.should_stop = False

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

        # Publishers
        self.asr_pub = self.create_publisher(Asr, "asr", 10)

        self.get_logger().info("ASR Node initialized, waiting for audio...")

    def audio_and_device_info_callback(self, msg: AudioAndDeviceInfo):
        """Process incoming audio data"""
        if self.sample_rate is None:
            self.sample_rate = int(msg.device_samplerate)
            self.get_logger().info(f"Audio sample rate: {self.sample_rate} Hz")

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
        """Process VAD state changes"""
        current_time = time.time()
        new_vad_state = msg.vad_probability > VAD_THRESHOLD

        # Detect VAD state change
        if new_vad_state != self.vad_state:
            self.get_logger().info(
                f"VAD state changed: {self.vad_state} -> {new_vad_state}"
            )

            if new_vad_state and not self.speech_interrupted:
                # Speech started
                self.speech_start_time = current_time
                self.get_logger().info("Speech started")
            else:
                # Speech ended
                self.last_silence_time = current_time
                self.get_logger().info("Speech ended, starting silence timer")

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
            if speech_duration >= MAX_CHUNK_DURATION:
                self.get_logger().info(
                    f"Speech duration exceeded {MAX_CHUNK_DURATION}s, forcing chunk split"
                )
                self._force_chunk_split()

    def _process_speech_end(self):
        """Process speech when VAD goes from on to off"""
        while not self.should_stop:
            current_time = time.time()

            self.get_logger().info(
                f"Processing speech end, current time: {current_time}, "
                f"time since silence: {current_time - self.last_silence_time}, vad_state: {self.vad_state}"
            )

            # Check if we're still in silence and enough time has passed
            if (
                not self.vad_state
                and self.last_silence_time > 0
                and current_time - self.last_silence_time >= MIN_SILENCE_DURATION
            ):
                self.get_logger().info("Processing speech chunk after silence timeout")
                self._transcribe_speech_chunk()
                self.speech_interrupted = False
                return

            # Check if VAD became active again
            if self.vad_state:
                self.speech_interrupted = True
                self.get_logger().info("VAD reactivated, canceling speech processing")
                return

            # Wait a bit before checking again
            time.sleep(0.1)

    def _force_chunk_split(self):
        """Force a chunk split due to maximum duration"""
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

            # If we found a quiet moment, use it as split point
            if best_split_time and min_rms < SILENCE_DETECTION_THRESHOLD:
                self.get_logger().info(
                    f"Found silence at {best_split_time}, splitting chunk"
                )
                self._transcribe_speech_chunk(end_time=best_split_time)
                self.speech_start_time = best_split_time
            else:
                # No good split point found, just split at current time
                self.get_logger().info("No silence found, splitting at current time")
                self._transcribe_speech_chunk()
                self.speech_start_time = current_time

    def _transcribe_speech_chunk(self, end_time=None):
        """Transcribe the accumulated speech chunk"""
        if end_time is None:
            end_time = (
                self.last_silence_time if self.last_silence_time > 0 else time.time()
            )

        start_time = self.speech_start_time

        if start_time <= 0:
            self.get_logger().warn("No valid speech start time")
            return

        # Instead of start_time, use an earlier time to prevent cutting the beginning
        actual_start_time = start_time - PRE_BUFFER_DURATION

        with self.buffer_lock:
            # Collect audio data for the speech chunk
            audio_chunks = []
            for audio_chunk in self.audio_buffer:
                chunk_time = audio_chunk["timestamp"]
                if actual_start_time <= chunk_time <= end_time:
                    audio_chunks.append(audio_chunk["audio"])

            if not audio_chunks:
                self.get_logger().warn("No audio data found for transcription")
                return

            # Concatenate audio chunks
            audio_data = np.concatenate(audio_chunks)

        # Check if we have enough audio (at least 0.01 seconds)
        min_duration = 0.01
        if self.sample_rate is None or len(audio_data) < int(
            self.sample_rate * min_duration
        ):
            self.get_logger().info(
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
                asr_msg = Asr()
                asr_msg.header.stamp = self.get_clock().now().to_msg()
                asr_msg.transcript = transcript
                asr_msg.transcript_confidence = (
                    float(info.language_probability)
                    if hasattr(info, "language_probability")
                    else 0.8
                )
                asr_msg.language_code = (
                    info.language if hasattr(info, "language") else "unknown"
                )

                self.asr_pub.publish(asr_msg)
                self.get_logger().info(
                    f"Published transcript: '{transcript}' (lang: {asr_msg.language_code}, conf: {asr_msg.transcript_confidence:.2f})"
                )
            else:
                self.get_logger().info("Empty transcript, not publishing")

        except Exception as e:
            self.get_logger().error(f"Transcription failed: {e}")

        # Reset speech timing
        self.speech_start_time = 0.0
        self.last_silence_time = 0.0

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
