import warnings

# Matplotlib 3D warning
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# Torchaudio deprecation
warnings.filterwarnings(
    "ignore", message="torchaudio._backend.set_audio_backend has been deprecated"
)

# PyTorch Lightning checkpoint upgrades
warnings.filterwarnings(
    "ignore", message="Lightning automatically upgraded your loaded checkpoint"
)
warnings.filterwarnings("ignore", message="multiple `ModelCheckpoint` callback states")

# PyAnnote frame mismatch
warnings.filterwarnings(
    "ignore",
    message="Mismatch between frames",
    module="pyannote.audio.models.blocks.pooling",
)

import signal
import sys
import threading
import time
from typing import Dict

import diart.models as m
import numpy as np
import rclpy
import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from hri_msgs.msg import AudioAndDeviceInfo, SpeechActivityDetection, Vad
from pyannote.core import Annotation
from rclpy.node import Node
from rx.core.observer.observer import Observer

from .database import DataBaseManager
from .ros_audio_source import ROSAudioSource


class DiarizationObserver(Observer):
    """Custom observer that processes diarization results and publishes them"""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.known_diart_speakers = set()  # Track DIART's internal speaker labels
        self.diart_to_eut_mapping = {}  # Maps DIART speaker IDs to EUT speaker IDs
        self.next_eut_speaker_id = 1
        self.current_diart_speaker = None
        self.previous_diart_speaker = None
        self.last_process_time = time.time()
        self.last_merge_check_time = time.time()
        self.merge_check_interval = 10.0  # Check for similar speakers every 10 seconds

        # Get use_database parameter from node
        self.use_database = self.node.use_database
        
        # Data base variables - only initialize if enabled
        if self.use_database:
            self.db = DataBaseManager()
            self.number_of_speakers = self.db.number_speakers()
            self.node.get_logger().info(f"Database enabled - {self.number_of_speakers} speakers loaded")
        else:
            self.db = None
            self.number_of_speakers = 0
            self.node.get_logger().info("Database disabled - using session-only speaker tracking")

        # Store embeddings in memory to save on shutdown
        self.pending_embeddings = {}
        
        # Threshold for considering speakers as the same person (lower = stricter)
        self.similarity_threshold = 0.3

    def _extract_prediction(self, value):
        """Extract prediction annotation from the value"""
        if isinstance(value, tuple):
            return value[0]  # Assuming prediction is the first element
        elif isinstance(value, Annotation):
            return value
        else:
            return None

    def _extract_embeddings_from_pipeline(self, pipeline: SpeakerDiarization):
        """
        Extract embeddings from the diarization pipeline.
        """
        embeddings = {}

        self.node.get_logger().info("=" * 80)
        self.node.get_logger().info("EXTRACTING EMBEDDINGS FROM PIPELINE")
        
        if hasattr(pipeline, "clustering"):
            clustering = pipeline.clustering
            self.node.get_logger().info(f"Clustering object found: {type(clustering)}")
            
            # Log clustering state
            if hasattr(clustering, "num_clusters"):
                self.node.get_logger().info(f"  Number of clusters: {clustering.num_clusters}")
            
            # Log active centers information
            if hasattr(clustering, "active_centers"):
                active_centers = clustering.active_centers
                self.node.get_logger().info(f"  Active centers: {active_centers}")
                self.node.get_logger().info(f"  Number of active centers: {len(active_centers)}")
            else:
                self.node.get_logger().warn("  No 'active_centers' attribute found")
                active_centers = []

            # Extract embeddings from clustering centers
            if hasattr(clustering, "centers") and clustering.centers is not None:
                centers = clustering.centers
                self.node.get_logger().info(f"  Centers shape: {centers.shape if hasattr(centers, 'shape') else 'N/A'}")
                self.node.get_logger().info(f"  Centers type: {type(centers)}")
                self.node.get_logger().info(f"  Total centers available: {len(centers)}")
                
                # Only extract the active centers if available
                for idx, center_idx in enumerate(active_centers):
                    embedding = centers[center_idx]
                    diart_speaker_id = f"speaker{center_idx}"
                    embeddings[diart_speaker_id] = embedding
                    
                    # Log embedding details
                    if hasattr(embedding, 'shape'):
                        self.node.get_logger().info(f"  [{idx}] DIART Speaker ID: {diart_speaker_id}, Center index: {center_idx}, Embedding shape: {embedding.shape}")
                    else:
                        self.node.get_logger().info(f"  [{idx}] DIART Speaker ID: {diart_speaker_id}, Center index: {center_idx}, Embedding length: {len(embedding)}")
                    
                    # Log embedding statistics
                    if isinstance(embedding, np.ndarray) or hasattr(embedding, '__array__'):
                        emb_array = np.array(embedding)
                        self.node.get_logger().info(f"      Embedding mean: {emb_array.mean():.6f}, std: {emb_array.std():.6f}, min: {emb_array.min():.6f}, max: {emb_array.max():.6f}")
            else:
                self.node.get_logger().warn("  clustering.centers is empty or None")
                
            # Log additional clustering attributes if available
            if hasattr(clustering, 'assignment'):
                self.node.get_logger().info(f"  Current assignment: {clustering.assignment}")
            if hasattr(clustering, 'min_size'):
                self.node.get_logger().info(f"  Min cluster size: {clustering.min_size}")
                
        else:
            self.node.get_logger().warn("No 'clustering' attribute in pipeline")

        if not embeddings:
            self.node.get_logger().warn("No embeddings found in pipeline")
        else:
            self.node.get_logger().info(f"Successfully extracted {len(embeddings)} embedding(s)")
            
        self.node.get_logger().info("=" * 80)
        return embeddings

    def on_next(self, value):
        """Process new diarization result and publish speaker status"""

        self.node.get_logger().info("+" * 80)
        self.node.get_logger().info("NEW DIARIZATION RESULT RECEIVED")
        
        prediction = self._extract_prediction(value)
        if prediction is None:
            self.node.get_logger().warn(
                "No prediction extracted from diarization value"
            )
            return

        self.node.get_logger().info(f"Prediction type: {type(prediction)}")
        self.node.get_logger().info(f"Prediction labels (DIART): {prediction.labels()}")
        self.node.get_logger().info(f"Number of tracks: {len(list(prediction.itertracks()))}")

        current_time = time.time()
        if current_time - self.last_process_time < 0.5:
            self.node.get_logger().info("Skipping processing (too soon, < 0.5s since last)")
            return  # Limit processing to once per 0.5 seconds
        self.last_process_time = current_time


        # Extract current active speakers from the annotation
        active_diart_speakers = set()
        self.node.get_logger().info("Iterating through prediction tracks:")
        
        for track_tuple in prediction.itertracks(yield_label=True):
            self.node.get_logger().info(f"  Track tuple length: {len(track_tuple)}, content: {track_tuple}")
            
            if len(track_tuple) == 3:
                segment, track, diart_speaker = track_tuple
                self.node.get_logger().info(
                    f"  ✓ Found DIART speaker: {diart_speaker} in segment: {segment} (duration: {segment.duration:.2f}s)"
                )
                self.node.get_logger().info(f"    Track ID: {track}")
                
            elif len(track_tuple) == 2:
                segment, track = track_tuple
                diart_speaker = None
                self.node.get_logger().info(
                    f"  ✗ Found track WITHOUT speaker label: {track} in segment: {segment}"
                )
            else:
                self.node.get_logger().warn(f"  ? Unexpected track tuple format: {track_tuple}")
                continue
                
            if diart_speaker is not None:
                active_diart_speakers.add(diart_speaker)
                if diart_speaker not in self.known_diart_speakers:
                    self.known_diart_speakers.add(diart_speaker)
                    # Don't create mapping here anymore - let _process_embeddings handle it
                    self.node.get_logger().info(f"  NEW DIART SPEAKER DISCOVERED: {diart_speaker}")

        # Determine current speaker (take the first one if multiple)
        current_diart_speaker = list(active_diart_speakers)[0] if active_diart_speakers else None
        self.node.get_logger().info(f"Active DIART speakers: {active_diart_speakers}")
        self.node.get_logger().info(f"Current DIART speaker selected: {current_diart_speaker}")
        self.node.get_logger().info(f"All known DIART speakers: {self.known_diart_speakers}")
        self.node.get_logger().info(f"DIART->EUT speaker mapping: {self.diart_to_eut_mapping}")

        # Publish speech activity detection message only if VAD probability > threshold
        if current_diart_speaker is None:
            self.node.get_logger().info("No current speaker detected, skipping")
            self.node.get_logger().info("+" * 80)
            return

        # Check VAD threshold before publishing
        self.node.get_logger().info(f"Current VAD probability: {self.node.current_vad_probability:.4f}, threshold: {self.node.vad_threshold}")
        
        if self.node.current_vad_probability <= self.node.vad_threshold:
            self.node.get_logger().info("VAD probability below threshold, skipping embedding extraction")
            self.node.get_logger().info("+" * 80)
            return

        # Extract and process embeddings when a real speaker is detected
        if self.node.model is not None:
            try:
                self.node.get_logger().info("Extracting embeddings from pipeline...")
                pipeline_embeddings = self._extract_embeddings_from_pipeline(
                    self.node.model
                )
                if pipeline_embeddings:
                    self.node.get_logger().info(f"Processing {len(pipeline_embeddings)} embeddings...")
                    self._process_embeddings(pipeline_embeddings, current_diart_speaker)
                else:
                    self.node.get_logger().warn("No embeddings extracted from pipeline")
            except Exception as ex:
                self.node.get_logger().error(f"Error extracting embeddings: {ex}")
                import traceback
                self.node.get_logger().error(traceback.format_exc())
        
        self.node.get_logger().info("+" * 80)

    def on_error(self, error: Exception):
        self.node.get_logger().error(f"DiarizationObserver error: {error}")
        if self.db:
            self.db.close()

    def _process_embeddings(self, pipeline_embeddings: Dict, current_diart_speaker):
        """
        Process embeddings extracted from the pipeline.
        Store them in memory to be saved on shutdown instead of saving immediately.
        """
        self.node.get_logger().info("*" * 80)
        self.node.get_logger().info("PROCESSING EMBEDDINGS")
        
        if not pipeline_embeddings:
            self.node.get_logger().warn("No embeddings detected in the pipeline")
            return

        self.node.get_logger().info(f"Total embeddings to process: {len(pipeline_embeddings)}")
        self.node.get_logger().info(f"Current DIART speaker: {current_diart_speaker}")
        self.node.get_logger().info(f"Pending embeddings in memory: {list(self.pending_embeddings.keys())}")
        self.node.get_logger().info(f"Speakers in database: {self.number_of_speakers}")

        for diart_speaker_id, embedding in pipeline_embeddings.items():
            self.node.get_logger().info(f"--- Processing DIART speaker: {diart_speaker_id} ---")
            self.node.get_logger().info(f"    Embedding shape/length: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}")
            
            if diart_speaker_id == current_diart_speaker:
                self.node.get_logger().info(f"    ✓ This is the CURRENT ACTIVE DIART speaker")
                
                # Convert to numpy if necessary
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                    self.node.get_logger().info(f"    Converted to numpy array")

                # Check if it already exists in the database
                if self.use_database:
                    self.node.get_logger().info(f"    Searching database for matching speaker...")
                    result = self.db.find_speaker(embedding)

                    if result:
                        eut_speaker_name, distance = result
                        self.node.get_logger().info(f"    ✓✓✓ RECOGNIZED as existing EUT speaker: {eut_speaker_name}")
                        self.node.get_logger().info(f"    Cosine distance: {distance:.4f}")
                        self.node.get_logger().info(f"    Setting eut_speaker_id to: {eut_speaker_name}")
                        self.node.eut_speaker_id = eut_speaker_name
                        
                        # Find if the diart_speaker_id has been included in pending embeddings and remove it
                        if diart_speaker_id in self.pending_embeddings:
                            del self.pending_embeddings[diart_speaker_id]
                            self.node.get_logger().info(
                                f"    Removed DIART {diart_speaker_id} from pending embeddings (already in DB)"
                            )
                    else:
                        # Assign EUT speaker ID based on total speakers (DB + pending)
                        total_speakers = self.number_of_speakers + len(self.pending_embeddings)
                        new_eut_speaker_number = total_speakers + 1
                        self.node.eut_speaker_id = f"EUT_speaker{new_eut_speaker_number}"
                        
                        self.node.get_logger().info(f"    ✗✗✗ NEW SPEAKER detected")
                        self.node.get_logger().info(f"    Total speakers in DB: {self.number_of_speakers}")
                        self.node.get_logger().info(f"    Pending embeddings: {len(self.pending_embeddings)}")
                        self.node.get_logger().info(f"    Assigned new EUT speaker number: {new_eut_speaker_number}")
                        self.node.get_logger().info(f"    Setting eut_speaker_id to: {self.node.eut_speaker_id}")
                        
                        # Store embedding in memory, don't save yet
                        self.pending_embeddings[diart_speaker_id] = embedding
                        self.node.get_logger().info(
                            f"    Stored in pending embeddings (will be saved on shutdown)"
                        )
                else:
                    # Database disabled - use clustering speaker_id with EUT prefix
                    self.node.get_logger().info("    Database disabled - using clustering speaker ID with EUT prefix")
                    
                    # Check if we've already mapped this DIART speaker
                    if diart_speaker_id in self.diart_to_eut_mapping:
                        # Use existing mapping
                        self.node.eut_speaker_id = self.diart_to_eut_mapping[diart_speaker_id]
                        self.node.get_logger().info(f"    Using existing mapping: DIART {diart_speaker_id} -> {self.node.eut_speaker_id}")
                    else:
                        # Create new EUT mapping for this DIART speaker
                        new_eut_speaker_number = self.next_eut_speaker_id
                        self.node.eut_speaker_id = f"EUT_speaker{new_eut_speaker_number}"
                        self.diart_to_eut_mapping[diart_speaker_id] = self.node.eut_speaker_id
                        self.next_eut_speaker_id += 1
                        self.node.get_logger().info(f"    Created new mapping: DIART {diart_speaker_id} -> {self.node.eut_speaker_id} (number: {new_eut_speaker_number})")
            else:
                self.node.get_logger().info(f"    ✗ Not the current active DIART speaker, skipping")
        
        self.node.get_logger().info(f"Final eut_speaker_id value: {self.node.eut_speaker_id}")
        self.node.get_logger().info("*" * 80)

    def _save_pending_embeddings(self):
        """Save all pending embeddings to the database. Called on node shutdown."""
        if not self.use_database:
            self.node.get_logger().info("Database disabled, skipping saving embeddings")
            return

        if not self.pending_embeddings:
            self.node.get_logger().info("No new embeddings to save")
            return

        self.node.get_logger().info(
            f"Saving {len(self.pending_embeddings)} new speaker(s) to database..."
        )

        for diart_speaker_id, embedding in self.pending_embeddings.items():
            new_number_of_speakers = self.number_of_speakers + 1
            eut_speaker_name = f"EUT_speaker{new_number_of_speakers}"
            self.db.save_speaker(eut_speaker_name, embedding)
            self.node.get_logger().info(f"Saved DIART {diart_speaker_id} as: {eut_speaker_name} in MONGODB")
            self.number_of_speakers = new_number_of_speakers

        self.node.get_logger().info("All embeddings saved to MongoDB")


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
        self.declare_parameter(
            "use_database", False
        )  # Whether to use the database for speaker tracking

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
        self.use_database = (
            self.get_parameter("use_database").get_parameter_value().bool_value
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
        self.diart_to_eut_mapping = {}  # Maps DIART speaker IDs to EUT speaker IDs
        self.next_eut_speaker_id = 1
        self.current_diart_speaker = None
        self.previous_diart_speaker = None
        self.last_process_time = time.time()
        self.eut_speaker_id = None  # Current EUT speaker ID
        self.speaker_activated = False

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
                    delta_new=0.92,  # Lower threshold for new speaker detection
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
            and self.eut_speaker_id is not None
            and self.speaker_activated
        ):
            # If VAD RMS indicates sustained silence, publish inactive status for current speaker
            speech_activity_msg = SpeechActivityDetection()
            speech_activity_msg.header.stamp = self.get_clock().now().to_msg()
            speech_activity_msg.speaker_id = self.eut_speaker_id.replace("EUT_", "")
            speech_activity_msg.active = False
            self.speech_activity_pub.publish(speech_activity_msg)
            self.get_logger().info(
                f"Finish publishing speech activity: speaker={speech_activity_msg.speaker_id}"
            )
            self.eut_speaker_id = None
            # A speaker cannot be deactivated if they were not activated before
            self.speaker_activated = False

        elif (
            self.current_vad_probability > self.vad_threshold
            and self.eut_speaker_id is not None
        ):
            speech_activity_msg = SpeechActivityDetection()
            speech_activity_msg.header.stamp = self.get_clock().now().to_msg()
            speech_activity_msg.speaker_id = self.eut_speaker_id.replace("EUT_", "")
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

        except Exception as e:
            self.get_logger().error(f"Failed to start diarization: {e}")
            self.diarization_started = False

    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
        try:
            self.get_logger().info("Shutting down diarization node...")

            # Mark as not started to stop the diarization thread
            self.diarization_started = False

            # Save pending embeddings to database before shutdown
            if hasattr(self, "observer") and self.observer is not None:
                self.observer._save_pending_embeddings()
                # Close database connection
                if self.observer.db:
                    self.observer.db.close()

            # Close the audio source
            if hasattr(self, "source") and self.source is not None:
                self.source.close()

        except Exception as e:
            self.get_logger().warn(f"Error during cleanup: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    diarization_node = DiarizationNode()

    def shutdown_handler(signum, frame):
        diarization_node.get_logger().info(
            f"Signal {signum} received, shutting down..."
        )
        # Raise KeyboardInterrupt instead of calling sys.exit
        raise KeyboardInterrupt

    # Capture both signals
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(diarization_node)
    except KeyboardInterrupt:
        diarization_node.get_logger().info("Shutting down DIARIZATION node.")
    except SystemExit:
        # Handle SystemExit without propagating it
        diarization_node.get_logger().info("SystemExit caught, cleaning up...")
    finally:
        # Ensure cleanup always happens
        try:
            if rclpy.ok():
                diarization_node.destroy_node()
                rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()
