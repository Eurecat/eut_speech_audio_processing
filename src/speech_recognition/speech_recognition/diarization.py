# import warnings

# # Matplotlib 3D warning
# warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# # Torchaudio deprecation
# warnings.filterwarnings(
#     "ignore", message="torchaudio._backend.set_audio_backend has been deprecated"
# )

# # PyTorch Lightning checkpoint upgrades
# warnings.filterwarnings(
#     "ignore", message="Lightning automatically upgraded your loaded checkpoint"
# )
# warnings.filterwarnings("ignore", message="multiple `ModelCheckpoint` callback states")

# # PyAnnote frame mismatch
# warnings.filterwarnings(
#     "ignore",
#     message="Mismatch between frames",
#     module="pyannote.audio.models.blocks.pooling",
# )

import threading
import time
from typing import Dict

import diart.models as m
import numpy as np
import rclpy
import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from pyannote.core import Annotation
from rclpy.node import Node
from rx.core.observer.observer import Observer

from hri_msgs.msg import AudioAndDeviceInfo, SpeechActivityDetection, Vad

from .ros_audio_source import ROSAudioSource
from .database import DataBaseManager


class DiarizationObserver(Observer):
    """Custom observer that processes diarization results and publishes them"""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.known_speakers = set()
        self.speaker_mapping = {}  # Maps model speaker IDs to consistent speaker numbers
        self.next_speaker_id = 1
        self.current_speaker = None
        self.previous_speaker = None
        self.last_process_time = time.time()

        # Data base variables
        self.db = DataBaseManager()
        self.number_of_speakers = self.db.number_speakers()

    def _extract_prediction(self, value):
        """Extract prediction annotation from the value"""
        if isinstance(value, tuple):
            return value[0]  # Assuming prediction is the first element
        elif isinstance(value, Annotation):
            return value
        else:
            return None

    def on_next(self, value):
        """Process new diarization result and publish speaker status"""

        prediction = self._extract_prediction(value)
        if prediction is None:
            self.node.get_logger().warn(
                "No prediction extracted from diarization value"
            )
            return

        current_time = time.time()
        if current_time - self.last_process_time < 0.5:
            return  # Limit processing to once per 0.5 seconds
        self.last_process_time = current_time

        # Extract current active speakers from the annotation
        active_speakers = set()
        for track_tuple in prediction.itertracks(yield_label=True):
            self.node.get_logger().debug(f"Track tuple: {track_tuple}")
            if len(track_tuple) == 3:
                segment, track, speaker = track_tuple
                self.node.get_logger().debug(
                    f"Found speaker: {speaker} in segment: {segment}"
                )
                # Print what is inside the track_tuple for debugging
                self.node.get_logger().debug(f"Track tuple contents: {track_tuple}")
                self.node.get_logger().debug(
                    f"Segment: {segment}, Track: {track}, Speaker: {speaker}"
                )
            elif len(track_tuple) == 2:
                segment, track = track_tuple
                speaker = None
                self.node.get_logger().debug(
                    f"Found track without speaker: {track} in segment: {segment}"
                )
            else:
                continue
            if speaker is not None:
                active_speakers.add(speaker)
                if speaker not in self.known_speakers:
                    self.known_speakers.add(speaker)
                    self.speaker_mapping[speaker] = self.next_speaker_id
                    self.next_speaker_id += 1

        # Determine current speaker (take the first one if multiple)
        current_speaker = list(active_speakers)[0] if active_speakers else None

        # Publish speech activity detection message only if VAD probability > threshold
        if current_speaker is None:
            return

        # Check VAD threshold before publishing
        if self.node.current_vad_probability <= self.node.vad_threshold:
            return

        self.node.real_speaker = f"speaker{self.speaker_mapping[current_speaker]}"

    def on_error(self, error: Exception):
        self.node.get_logger().error(f"DiarizationObserver error: {error}")
        self.db.close()

    def process_embeddings(self, pipeline_embeddings: Dict):
        """
        Process embeddings extracted from the pipeline.
        Depending if they are known or new, show and save them.
        """
        if not pipeline_embeddings:
            self.node.get_logger().warn("No embeddings detected in the pipeline")
            return

        self.node.get_logger().info(
            f"Embeddings detected: {len(pipeline_embeddings)} speaker(s)"
        )

        for speaker_id, embedding in pipeline_embeddings.items():
            # Convert to numpy if necessary
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Check if it already exists in the database
            result = self.db.find_speaker(embedding)

            if result:
                name, distance = result
                self.node.get_logger().info(f"Recognized as: {name}")
                self.node.get_logger().info(f"Cosine distance: {distance:.4f}")
            else:
                # New speaker, save it
                new_number_of_speakers = self.number_of_speakers + 1
                name = f"Speaker_{new_number_of_speakers}"
                self.db.save_speaker(name, embedding)
                self.node.get_logger().info(f"New speaker saved as: {name}")
                self.node.get_logger().info("Embedding saved in MongoDB")


class DiarizationNode(Node):
    def __init__(self):
        super().__init__("diarization_node")

        # Declare parameters
        self.declare_parameter(
            "chunk_duration", 2.0
        )  # seconds - duration of audio chunks for diarization
        self.declare_parameter(
            "overlap_duration", 0.5
        )  # seconds - overlap between consecutive chunks
        self.declare_parameter("segmentation_model_name", "pyannote/segmentation")
        self.declare_parameter("embedding_model_name", "pyannote/embedding")
        self.declare_parameter(
            "vad_threshold", 0.5
        )  # VAD threshold for publishing speech activity

        # Get parameter values
        self.chunk_duration = (
            self.get_parameter("chunk_duration").get_parameter_value().double_value
        )
        self.overlap_duration = (
            self.get_parameter("overlap_duration").get_parameter_value().double_value
        )
        self.segmentation_model_name = (
            self.get_parameter("segmentation_model_name")
            .get_parameter_value()
            .string_value
        )
        self.embedding_model_name = (
            self.get_parameter("embedding_model_name")
            .get_parameter_value()
            .string_value
        )

        # Initialize device info
        self.device_name = None
        self.device_id = None
        self.device_samplerate = None
        self.chunk_size = None
        self.overlap_size = None
        self.source = None
        self.model = None
        self.config = None
        self.inference = None
        self.observer = None

        # Flag to track if device info has been received
        self.device_info_received = False
        self.diarization_started = False

        # VAD tracking
        self.current_vad_probability = 0.0
        self.vad_threshold = (
            self.get_parameter("vad_threshold").get_parameter_value().double_value
        )

        # VAD buffer for less reactive active=False detection
        self.vad_buffer = []
        self.vad_buffer_size = 30
        self.vad_rms_threshold = 0.5

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
        self.speech_activity_pub = self.create_publisher(
            SpeechActivityDetection, "speech_activity_detection", 10
        )

        # Select device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Speaker tracking
        self.speaker_mapping = {}  # Maps model speaker IDs to consistent speaker numbers
        self.next_speaker_id = 1
        self.current_speaker = None
        self.previous_speaker = None
        self.last_process_time = time.time()
        self.real_speaker = None
        self.speaker_activated = False

        # Logging control
        self.last_log_time = 0  # For 1Hz logging
        self.log_time = 1.0  # Log every 1 second

        self.get_logger().info(
            "Diarization node initialized, waiting for device and VAD info..."
        )

    def audio_and_device_info_callback(self, msg: AudioAndDeviceInfo):
        """Callback for audio device info updates"""

        # Only initialize once
        if not self.device_info_received:
            # Get device info from message
            self.device_name = msg.device_name
            self.device_id = msg.device_id
            self.device_samplerate = msg.device_samplerate

            self.get_logger().info(
                f"Diarization initialized with audio device: {msg.device_name} (Sample rate: {msg.device_samplerate} Hz)"
            )

            # Validate that we have all required information
            if self.device_samplerate is None or self.device_samplerate <= 0:
                self.get_logger().error(
                    f"Invalid or missing sample rate: {self.device_samplerate}"
                )
                return

            try:
                self.chunk_size = int(self.device_samplerate * self.chunk_duration)
                self.overlap_size = int(self.device_samplerate * self.overlap_duration)

                self.get_logger().info("Loading segmentation and embedding models.")
                # Get Hugging Face token from environment variable or set it here
                import os

                hf_token = os.environ.get("HF_TOKEN", None)
                if hf_token is None:
                    self.get_logger().warn(
                        "No Hugging Face token found in environment variable HF_TOKEN. Public models may work, but private/gated models will fail."
                    )

                # Load segmentation and embedding models
                segmentation = m.SegmentationModel.from_pretrained(
                    self.segmentation_model_name, use_hf_token=hf_token
                )
                embedding = m.EmbeddingModel.from_pretrained(
                    self.embedding_model_name, use_hf_token=hf_token
                )

                # Calculate step size to align with block duration
                step_duration = 0.5  # Match with ROSAudioSource block_duration

                self.config = SpeakerDiarizationConfig(
                    segmentation=segmentation,
                    embedding=embedding,
                    device=self.device,
                    sample_rate=int(self.device_samplerate),
                    duration=self.chunk_duration,
                    step=step_duration,  # Align with audio source block duration
                    tau_active=0.7,  # Lower threshold for speaker activity detection
                    delta_new=0.95,  # Lower threshold for new speaker detection
                    # gamma=3.0,  # Scale for speaker change detection
                    # beta=10.0,  # Beta parameter for speaker change
                    max_speakers=10,  # Maximum number of speakers
                )

                # Load pre-trained diarization model
                self.model = SpeakerDiarization(self.config)

                # Create custom ROS audio source with aligned block duration
                try:
                    # Use step_duration to ensure alignment
                    self.source = ROSAudioSource(
                        sample_rate=int(self.device_samplerate),
                        block_duration=step_duration,  # Match step size
                    )
                    # Start reading from the source (starts the internal thread)
                    self.source.read()

                except Exception as audio_error:
                    self.get_logger().error(
                        f"Failed to initialize ROS audio source: {audio_error}"
                    )
                    return

                # Mark device info as received
                self.device_info_received = True

                # Start diarization in a separate thread
                self.diarization_procedure = threading.Thread(
                    target=self.run_diarization
                )
                self.diarization_procedure.daemon = True
                self.diarization_procedure.start()

                self.get_logger().debug("Diarization system initialized and started")

            except Exception as e:
                self.get_logger().error(f"Failed to initialize diarization: {e}")
                self.device_info_received = False

        if self.source is not None and self.device_info_received:
            # Convert message data to numpy array (just in case)
            audio_data = np.array(msg.audio, dtype=np.float32)

            # Feed the audio to our custom source
            self.source.add_audio_chunk(audio_data)

    def vad_callback(self, msg: Vad):
        """Process VAD messages to track speech probability"""

        # Print once VAD information received
        if not hasattr(self, "vad_info_logged"):
            self.get_logger().info("VAD information received.")
            self.vad_info_logged = True

        self.current_vad_probability = msg.vad_probability

        # Add current VAD probability to buffer
        self.vad_buffer.append(self.current_vad_probability)

        # Keep only the last vad_buffer_size values
        if len(self.vad_buffer) > self.vad_buffer_size:
            self.vad_buffer.pop(0)

        # Calculate RMS of VAD buffer for less reactive active=False detection
        vad_rms = 0.0
        if len(self.vad_buffer) >= self.vad_buffer_size:
            vad_array = np.array(self.vad_buffer)
            vad_rms = float(np.sqrt(np.mean(vad_array**2)))

        if (
            len(self.vad_buffer) >= self.vad_buffer_size
            and vad_rms <= self.vad_rms_threshold
            and self.real_speaker is not None
            and self.speaker_activated
        ):
            # If VAD RMS indicates sustained silence, publish inactive status for current speaker
            speech_activity_msg = SpeechActivityDetection()
            speech_activity_msg.header.stamp = self.get_clock().now().to_msg()
            speech_activity_msg.speaker_id = self.real_speaker
            speech_activity_msg.active = False
            self.speech_activity_pub.publish(speech_activity_msg)
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f"Published speech activity: speaker={speech_activity_msg.speaker_id}"
                )
                self.last_log_time = current_time
            self.real_speaker = None
            # A speaker cannot be deactivated if they were not activated before
            self.speaker_activated = False

        elif (
            self.current_vad_probability > self.vad_threshold
            and self.real_speaker is not None
        ):
            speech_activity_msg = SpeechActivityDetection()
            speech_activity_msg.header.stamp = self.get_clock().now().to_msg()
            speech_activity_msg.speaker_id = self.real_speaker
            speech_activity_msg.active = True
            self.speaker_activated = True

            self.speech_activity_pub.publish(speech_activity_msg)

    def run_diarization(self):
        """Run diarization in a separate thread"""

        # This method is now only called after device info is received
        if not self.device_info_received or self.source is None or self.model is None:
            self.get_logger().error(
                "Cannot start diarization: missing device info or model"
            )
            return

        if self.diarization_started:
            self.get_logger().warn("Diarization already started")
            return

        self.get_logger().debug("Starting diarization process")

        try:
            # Create streaming inference
            self.inference = StreamingInference(self.model, self.source, do_plot=False)

            # Create custom observer to handle diarization results
            self.observer = DiarizationObserver(self)

            # Attach observer to the inference pipeline using correct method
            self.inference.attach_observers(self.observer)

            # Mark as started
            self.diarization_started = True

            # Start processing audio stream
            self.inference()

            # Try to extract embeddings from the pipeline
            pipeline_embeddings = self._extract_embeddings_from_pipeline(self.model)

            if pipeline_embeddings:
                self.observer.process_embeddings(pipeline_embeddings)
            else:
                self.get_logger().warn(
                    "No embeddings could be extracted from the pipeline"
                )

        except Exception as e:
            self.get_logger().error(f"Failed to start diarization: {e}")
            self.diarization_started = False

    def _extract_embeddings_from_pipeline(self, pipeline: SpeakerDiarization):
        """
        Extract embeddings from the diarization pipeline.
        """
        embeddings = {}

        if hasattr(pipeline, "clustering"):
            clustering = pipeline.clustering

            # Extract embeddings from clustering centers
            if hasattr(clustering, "centers") and clustering.centers is not None:
                centers = clustering.centers
                # Only extract the active centers if available
                active_centers = clustering.active_centers
                for center_idx in active_centers:
                    embedding = centers[center_idx]
                    embeddings[f"speaker_{center_idx}"] = embedding

            else:
                self.get_logger().error("clustering.centers is empty or None")

        if not embeddings:
            self.get_logger().warn("No embeddings found")

        return embeddings

    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
        try:
            # Mark as not started to stop the diarization thread
            self.diarization_started = False

            # Close the audio source
            if hasattr(self, "source") and self.source is not None:
                self.source.close()

        except Exception as e:
            self.get_logger().warn(f"Error during cleanup: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    diarization_node = DiarizationNode()

    try:
        rclpy.spin(diarization_node)
    except KeyboardInterrupt:
        diarization_node.get_logger().info("Shutting down DIARIZATION node.")
    finally:
        diarization_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
