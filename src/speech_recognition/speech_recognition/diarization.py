import rclpy
import torch
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from audio_stream_manager_interfaces.msg import AudioDeviceInfo, Diarization
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from collections import deque
import time
import diart.models as m

# Audio processing constants
SAMPLERATE = 16000  # Hz
CHUNK_DURATION = 3.0  # seconds - duration of audio chunks for diarization
OVERLAP_DURATION = 0.5  # seconds - overlap between consecutive chunks
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)
OVERLAP_SIZE = int(SAMPLERATE * OVERLAP_DURATION)
SEGMENTATION_MODEL_NAME = "pyannote/segmentation-3.0"
EMBEDDING_MODEL_NAME = "pyannote/embedding"


class DiarizationNode(Node):
    def __init__(self):
        super().__init__("diarization_node")

        # Initialize device info
        self.device_info = None
        self.current_samplerate = SAMPLERATE

        # Subscribers
        self.audio_sub = self.create_subscription(
            Float32MultiArray, "audio", self.listener_callback, 10
        )

        self.device_info_sub = self.create_subscription(
            AudioDeviceInfo, "audio_device_info", self.device_info_callback, 10
        )

        # Publishers
        self.diarization_pub = self.create_publisher(Diarization, "diarization", 10)

        # Select device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Configuration for the diarization model
        segmentation = m.SegmentationModel.from_pretrained(SEGMENTATION_MODEL_NAME)
        embedding = m.EmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.config = SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            device=self.device,
            sample_rate=SAMPLERATE,  # Use default sample rate, will be updated when device info arrives
            duration=CHUNK_DURATION,
            step=0.5,  # Step size for sliding window
            tau_active=0.7,  # Lower threshold for speaker change detection
            delta_new=0.83,  # Lower threshold for new speaker detection
        )

        # Load pre-trained diarization model
        self.model = SpeakerDiarization(self.config)

        # Audio buffer to accumulate chunks for processing
        self.audio_buffer = deque(maxlen=CHUNK_SIZE * 2)  # Buffer for 2x chunk size

        # Speaker tracking
        self.speaker_mapping = {}  # Maps model speaker IDs to consistent speaker numbers
        self.next_speaker_id = 1
        self.current_speaker = None
        self.last_process_time = time.time()

        self.get_logger().info("Diarization node initialized")

    def device_info_callback(self, msg):
        """Callback for audio device info updates"""
        self.device_info = msg
        self.current_samplerate = int(msg.device_samplerate)
        self.get_logger().info(
            f"Updated device info: {msg.device_name} (ID: {msg.device_id}, "
            f"Sample rate: {msg.device_samplerate} Hz)"
        )

    def map_speaker_id(self, model_speaker_id):
        """Map model speaker ID to consistent speaker number"""
        if model_speaker_id not in self.speaker_mapping:
            self.speaker_mapping[model_speaker_id] = self.next_speaker_id
            self.next_speaker_id += 1
            self.get_logger().info(
                f"New speaker detected: speaker{self.speaker_mapping[model_speaker_id]}"
            )

        return self.speaker_mapping[model_speaker_id]

    def listener_callback(self, msg):
        try:
            # Convert audio data to numpy array
            audio_data = np.array(msg.data, dtype=np.float32)

            # Add to buffer
            self.audio_buffer.extend(audio_data)

            # Process when we have enough audio data
            if len(self.audio_buffer) >= CHUNK_SIZE:
                self.process_audio_chunk()

        except Exception as e:
            self.get_logger().error(f"Error in listener_callback: {e}")

    def process_audio_chunk(self):
        """Process accumulated audio for speaker diarization"""
        try:
            # Get audio chunk from buffer
            chunk_data = np.array(list(self.audio_buffer)[:CHUNK_SIZE])

            # Remove processed data (keeping overlap)
            for _ in range(CHUNK_SIZE - OVERLAP_SIZE):
                if self.audio_buffer:
                    self.audio_buffer.popleft()

            # Check if we have enough data
            if len(chunk_data) == 0:
                self.get_logger().warn("No audio data to process")
                return

            # Check if samplerate is valid before processing
            if self.current_samplerate <= 0:
                self.get_logger().warn(
                    f"Invalid sample rate: {self.current_samplerate}. Waiting for device info..."
                )
                return

            # Reshape audio for diart (needs to be 2D: samples x channels)
            audio_chunk = chunk_data.reshape(-1, 1)  # Convert to 2D array

            # Calculate actual duration based on audio length and sample rate
            actual_duration = len(chunk_data) / self.current_samplerate

            # Ensure we have a minimum duration to avoid division by zero
            if actual_duration <= 0:
                self.get_logger().warn(f"Invalid audio duration: {actual_duration}")
                return

            # Create SlidingWindowFeature for diart
            from pyannote.core import SlidingWindowFeature, SlidingWindow

            sliding_window = SlidingWindow(
                duration=actual_duration, step=actual_duration, start=0.0
            )

            audio_feature = SlidingWindowFeature(audio_chunk, sliding_window)

            # Perform diarization - pass as a list of SlidingWindowFeature
            with torch.no_grad():
                diarization_results = self.model([audio_feature])

            # Process diarization result (first result from the list)
            if diarization_results:
                annotation, _ = diarization_results[
                    0
                ]  # Get annotation and ignore audio
                speaker_info = self.extract_speaker_info(annotation)

                # Publish result
                if speaker_info:
                    diarization_msg = Diarization()
                    diarization_msg.current_speaker = speaker_info
                    diarization_msg.active_speakers = [
                        speaker_info
                    ]  # For now, just the current speaker
                    diarization_msg.confidence = (
                        0.8  # Default confidence, could be improved
                    )
                    diarization_msg.timestamp = self.get_clock().now().to_msg()

                    self.diarization_pub.publish(diarization_msg)
                    self.get_logger().info(f"Diarization: {speaker_info}")

        except Exception as e:
            self.get_logger().error(f"Error in process_audio_chunk: {e}")

    def extract_speaker_info(self, annotation):
        """Extract speaker information from pyannote Annotation object and format as speakerX"""
        try:
            # annotation is a pyannote.core.Annotation object
            if not annotation:
                return None

            # Get all speakers in this annotation
            speakers_in_chunk = []

            # Iterate through all segments and speakers
            for segment, track, speaker_label in annotation.itertracks(
                yield_label=True
            ):
                # Map model speaker label to consistent speaker number
                speaker_num = self.map_speaker_id(speaker_label)
                speaker_name = f"speaker{speaker_num}"

                if speaker_name not in speakers_in_chunk:
                    speakers_in_chunk.append(speaker_name)

            if speakers_in_chunk:
                # For simplicity, return the first speaker found
                # In a more sophisticated implementation, you might want to:
                # - Return the speaker with the longest duration in this chunk
                # - Return all active speakers
                # - Use some confidence scoring
                current_speaker = speakers_in_chunk[0]

                if current_speaker != self.current_speaker:
                    self.current_speaker = current_speaker
                    return current_speaker
                return current_speaker

            return None

        except Exception as e:
            self.get_logger().error(f"Error extracting speaker info: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    diarization_node = DiarizationNode()

    try:
        rclpy.spin(diarization_node)
    except KeyboardInterrupt:
        diarization_node.get_logger().info("Shutting down node.")
    finally:
        diarization_node.destroy_node()
        rclpy.shutdown()
