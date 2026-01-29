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
from hri_msgs.msg import AudioAndDeviceInfo, SpeechActivityDetection, Vad, IdsList
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Bool
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
        self.highest_eut_speaker_number = (
            0  # Track the highest EUT speaker number ever assigned
        )
        self.current_diart_speaker = None
        self.previous_diart_speaker = None
        self.last_process_time = time.time()
        self.last_merge_check_time = time.time()

        # Get use_database parameter from node
        self.use_database = self.node.use_database

        # Data base variables - only initialize if enabled
        if self.use_database:
            try:
                self.db = DataBaseManager()
                self.number_of_speakers = self.db.number_speakers()
                # Initialize highest_eut_speaker_number from database
                self.highest_eut_speaker_number = self.number_of_speakers
                self.next_eut_speaker_id = self.number_of_speakers + 1
                self.node.get_logger().info(
                    f"Database enabled - {self.number_of_speakers} speakers loaded"
                )
                self.node.get_logger().info(
                    f"Highest EUT speaker number initialized to: {self.highest_eut_speaker_number}"
                )
            except Exception as e:
                self.node.get_logger().warn(f"Failed to connect to database: {e}")
                self.node.get_logger().warn(
                    "Disabling database usage and falling back to session-only tracking."
                )
                self.use_database = False
                self.db = None

        if not self.use_database:
            self.db = None
            self.number_of_speakers = 0
            self.node.get_logger().info(
                "Database disabled - using session-only speaker tracking"
            )

        # ROS4HRI tracking with ID approach
        self.ros4hri_enabled = self.node.ros4hri_enabled
        self.active_voices = set()  # Set of currently active EUT speaker IDs

        # Store embeddings in memory to save on shutdown
        self.pending_embeddings = {}  # Maps DIART speaker ID to embedding

        # Threshold for considering speakers as the same person (lower = stricter)
        self.similarity_threshold = 0.5

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

        self.node.get_logger().debug("=" * 80)
        self.node.get_logger().debug("EXTRACTING EMBEDDINGS FROM PIPELINE")

        if hasattr(pipeline, "clustering"):
            clustering = pipeline.clustering
            self.node.get_logger().debug(f"Clustering object found: {type(clustering)}")

            # Log clustering state
            if hasattr(clustering, "num_clusters"):
                self.node.get_logger().debug(
                    f"  Number of clusters: {clustering.num_clusters}"
                )

            # Log active centers information
            if hasattr(clustering, "active_centers"):
                active_centers = clustering.active_centers
                self.node.get_logger().debug(f"  Active centers: {active_centers}")
                self.node.get_logger().debug(
                    f"  Number of active centers: {len(active_centers)}"
                )
            else:
                self.node.get_logger().debug("  No 'active_centers' attribute found")
                active_centers = []

            # Extract embeddings from clustering centers
            if hasattr(clustering, "centers") and clustering.centers is not None:
                centers = clustering.centers
                self.node.get_logger().debug(
                    f"  Centers shape: {centers.shape if hasattr(centers, 'shape') else 'N/A'}"
                )
                self.node.get_logger().debug(f"  Centers type: {type(centers)}")
                self.node.get_logger().debug(
                    f"  Total centers available: {len(centers)}"
                )

                # Only extract the active centers if available
                for idx, center_idx in enumerate(active_centers):
                    embedding = centers[center_idx]
                    diart_speaker_id = f"speaker{center_idx}"
                    embeddings[diart_speaker_id] = embedding

                    # Log embedding details
                    if hasattr(embedding, "shape"):
                        self.node.get_logger().debug(
                            f"  [{idx}] DIART Speaker ID: {diart_speaker_id}, Center index: {center_idx}, Embedding shape: {embedding.shape}"
                        )
                    else:
                        self.node.get_logger().debug(
                            f"  [{idx}] DIART Speaker ID: {diart_speaker_id}, Center index: {center_idx}, Embedding length: {len(embedding)}"
                        )

                    # Log embedding statistics
                    if isinstance(embedding, np.ndarray) or hasattr(
                        embedding, "__array__"
                    ):
                        emb_array = np.array(embedding)
                        self.node.get_logger().debug(
                            f"      Embedding mean: {emb_array.mean():.6f}, std: {emb_array.std():.6f}, min: {emb_array.min():.6f}, max: {emb_array.max():.6f}"
                        )
            else:
                self.node.get_logger().debug("  clustering.centers is empty or None")

            # Log additional clustering attributes if available
            if hasattr(clustering, "assignment"):
                self.node.get_logger().debug(
                    f"  Current assignment: {clustering.assignment}"
                )
            if hasattr(clustering, "min_size"):
                self.node.get_logger().debug(
                    f"  Min cluster size: {clustering.min_size}"
                )

        else:
            self.node.get_logger().debug("No 'clustering' attribute in pipeline")

        if not embeddings:
            self.node.get_logger().debug("No embeddings found in pipeline")
        else:
            self.node.get_logger().debug(
                f"Successfully extracted {len(embeddings)} embedding(s)"
            )

        self.node.get_logger().debug("=" * 80)
        return embeddings

    def _merge_similar_speakers(self, pipeline_embeddings: Dict):
        """
        Compare all speaker embeddings and merge similar ones by updating the mapping.
        Uses cosine distance to detect if two DIART speakers are actually the same person.
        """
        if len(pipeline_embeddings) < 2:
            return  # Nothing to merge

        self.node.get_logger().debug("=" * 80)
        self.node.get_logger().debug("CHECKING FOR SIMILAR SPEAKERS TO MERGE")

        # Convert all embeddings to numpy arrays for comparison
        speaker_ids = list(pipeline_embeddings.keys())
        embeddings_list = []

        for speaker_id in speaker_ids:
            emb = pipeline_embeddings[speaker_id]
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            # Normalize the embedding for cosine similarity
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings_list.append(emb_norm)

        self.node.get_logger().debug(f"Comparing {len(speaker_ids)} speaker embeddings")
        self.node.get_logger().debug(
            f"Current DIART->EUT mapping: {self.diart_to_eut_mapping}"
        )

        # Compare each pair of speakers
        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                speaker_i = speaker_ids[i]
                speaker_j = speaker_ids[j]

                # Get current EUT mappings
                eut_i = self.diart_to_eut_mapping.get(speaker_i)
                eut_j = self.diart_to_eut_mapping.get(speaker_j)

                # Skip if already mapped to the same EUT speaker
                if eut_i and eut_j and eut_i == eut_j:
                    self.node.get_logger().debug(
                        f"  {speaker_i} ({eut_i}) and {speaker_j} ({eut_j}) already mapped to same EUT speaker"
                    )
                    continue

                # Compute cosine distance (1 - cosine similarity)
                emb_i = embeddings_list[i]
                emb_j = embeddings_list[j]
                cosine_similarity = np.dot(emb_i, emb_j)
                cosine_distance = 1.0 - cosine_similarity

                # Show EUT IDs in comparison
                eut_i_str = f"({eut_i})" if eut_i else "(unmapped)"
                eut_j_str = f"({eut_j})" if eut_j else "(unmapped)"
                self.node.get_logger().debug(
                    f"  Comparing {speaker_i} {eut_i_str} vs {speaker_j} {eut_j_str}: cosine_distance = {cosine_distance:.4f}"
                )

                # If distance is below threshold, they're the same person
                if cosine_distance < self.similarity_threshold:
                    self.node.get_logger().debug(
                        f"  ✓✓✓ DETECTED DUPLICATE: {speaker_i} and {speaker_j} are the same person!"
                    )

                    # Decide which EUT speaker ID to use
                    if eut_i and eut_j:
                        # Both already mapped - use the lower numbered one
                        keep_eut = min(eut_i, eut_j)
                        discard_eut = max(eut_i, eut_j)
                        self.node.get_logger().debug(
                            f"    Both mapped: keeping {keep_eut}, merging {discard_eut}"
                        )

                        # Update all mappings that point to discard_eut to point to keep_eut
                        for diart_id, eut_id in self.diart_to_eut_mapping.items():
                            if eut_id == discard_eut:
                                self.diart_to_eut_mapping[diart_id] = keep_eut
                                self.node.get_logger().debug(
                                    f"    Updated {diart_id}: {discard_eut} -> {keep_eut}"
                                )

                    elif eut_i:
                        # Only i is mapped, map j to same EUT speaker
                        self.diart_to_eut_mapping[speaker_j] = eut_i
                        self.node.get_logger().debug(
                            f"    Mapped {speaker_j} to existing {eut_i}"
                        )

                    elif eut_j:
                        # Only j is mapped, map i to same EUT speaker
                        self.diart_to_eut_mapping[speaker_i] = eut_j
                        self.node.get_logger().debug(
                            f"    Mapped {speaker_i} to existing {eut_j}"
                        )

                    else:
                        # Neither mapped yet - will be handled in normal flow
                        # Just log that they should be merged
                        self.node.get_logger().debug(
                            f"    Both unmapped - will assign same EUT ID when processed"
                        )
                else:
                    self.node.get_logger().debug(
                        f"    Different speakers (distance > {self.similarity_threshold:.4f})"
                    )

        self.node.get_logger().debug(
            f"Updated DIART->EUT mapping: {self.diart_to_eut_mapping}"
        )
        self.node.get_logger().debug("=" * 80)

    def on_next(self, value):
        """Process new diarization result and publish speaker status"""

        self.node.get_logger().debug("+" * 80)
        self.node.get_logger().debug("NEW DIARIZATION RESULT RECEIVED")

        prediction = self._extract_prediction(value)
        if prediction is None:
            self.node.get_logger().debug(
                "No prediction extracted from diarization value"
            )
            return

        self.node.get_logger().debug(f"Prediction type: {type(prediction)}")
        self.node.get_logger().debug(
            f"Prediction labels (DIART): {prediction.labels()}"
        )
        self.node.get_logger().debug(
            f"Number of tracks: {len(list(prediction.itertracks()))}"
        )

        current_time = time.time()
        if current_time - self.last_process_time < 0.5:
            self.node.get_logger().debug(
                "Skipping processing (too soon, < 0.5s since last)"
            )
            return  # Limit processing to once per 0.5 seconds
        self.last_process_time = current_time

        # Extract current active speakers from the annotation
        active_diart_speakers = set()
        self.node.get_logger().debug("Iterating through prediction tracks:")

        for track_tuple in prediction.itertracks(yield_label=True):
            self.node.get_logger().debug(
                f"  Track tuple length: {len(track_tuple)}, content: {track_tuple}"
            )

            if len(track_tuple) == 3:
                segment, track, diart_speaker = track_tuple
                self.node.get_logger().debug(
                    f"  ✓ Found DIART speaker: {diart_speaker} in segment: {segment} (duration: {segment.duration:.2f}s)"
                )
                self.node.get_logger().debug(f"    Track ID: {track}")

            elif len(track_tuple) == 2:
                segment, track = track_tuple
                diart_speaker = None
                self.node.get_logger().debug(
                    f"  ✗ Found track WITHOUT speaker label: {track} in segment: {segment}"
                )
            else:
                self.node.get_logger().debug(
                    f"  ? Unexpected track tuple format: {track_tuple}"
                )
                continue

            if diart_speaker is not None:
                active_diart_speakers.add(diart_speaker)
                if diart_speaker not in self.known_diart_speakers:
                    self.known_diart_speakers.add(diart_speaker)
                    # Don't create mapping here anymore - let _process_embeddings handle it
                    self.node.get_logger().debug(
                        f"  NEW DIART SPEAKER DISCOVERED: {diart_speaker}"
                    )

        # Determine current speaker (take the first one if multiple)
        current_diart_speaker = (
            list(active_diart_speakers)[0] if active_diart_speakers else None
        )
        self.node.get_logger().debug(f"Active DIART speakers: {active_diart_speakers}")
        self.node.get_logger().debug(
            f"Current DIART speaker selected: {current_diart_speaker}"
        )
        self.node.get_logger().debug(
            f"All known DIART speakers: {self.known_diart_speakers}"
        )
        self.node.get_logger().debug(
            f"DIART->EUT speaker mapping: {self.diart_to_eut_mapping}"
        )

        # Publish speech activity detection message only if VAD probability > threshold
        if current_diart_speaker is None:
            self.node.get_logger().debug("No current speaker detected, skipping")
            self.node.get_logger().debug("+" * 80)
            return

        # Check VAD threshold before publishing
        self.node.get_logger().debug(
            f"Current VAD probability: {self.node.current_vad_probability:.4f}, threshold: {self.node.vad_threshold}"
        )

        if self.node.current_vad_probability <= self.node.vad_threshold:
            self.node.get_logger().debug(
                "VAD probability below threshold, skipping embedding extraction"
            )
            self.node.get_logger().debug("+" * 80)
            return

        # Extract and process embeddings when a real speaker is detected
        if self.node.model is not None:
            try:
                self.node.get_logger().debug("Extracting embeddings from pipeline...")
                pipeline_embeddings = self._extract_embeddings_from_pipeline(
                    self.node.model
                )
                if pipeline_embeddings:
                    self.node.get_logger().debug(
                        f"Processing {len(pipeline_embeddings)} embeddings..."
                    )
                    self._process_embeddings(
                        pipeline_embeddings,
                        current_diart_speaker,
                        active_diart_speakers,
                    )
                else:
                    self.node.get_logger().warn("No embeddings extracted from pipeline")
            except Exception as ex:
                self.node.get_logger().error(f"Error extracting embeddings: {ex}")
                import traceback

                self.node.get_logger().error(traceback.format_exc())

        self.node.get_logger().debug("+" * 80)

    def on_error(self, error: Exception):
        self.node.get_logger().error(f"DiarizationObserver error: {error}")
        if self.db:
            self.db.close()

    def _process_embeddings(
        self,
        pipeline_embeddings: Dict,
        current_diart_speaker,
        active_diart_speakers: set,
    ):
        """
        Process embeddings extracted from the pipeline.
        Store them in memory to be saved on shutdown instead of saving immediately.
        """
        self.node.get_logger().debug("*" * 80)
        self.node.get_logger().debug("PROCESSING EMBEDDINGS")

        if not pipeline_embeddings:
            self.node.get_logger().warn("No embeddings detected in the pipeline")
            return

        self.node.get_logger().debug(
            f"Total embeddings to process: {len(pipeline_embeddings)}"
        )
        self.node.get_logger().debug(f"Current DIART speaker: {current_diart_speaker}")
        self.node.get_logger().debug(
            f"Pending embeddings in memory: {list(self.pending_embeddings.keys())}"
        )
        self.node.get_logger().debug(f"Speakers in database: {self.number_of_speakers}")

        # First, check for similar speakers and merge them
        # self._merge_similar_speakers(pipeline_embeddings)

        for diart_speaker_id, embedding in pipeline_embeddings.items():
            self.node.get_logger().debug(
                f"--- Processing DIART speaker: {diart_speaker_id} ---"
            )
            self.node.get_logger().debug(
                f"    Embedding shape/length: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}"
            )

            if diart_speaker_id == current_diart_speaker:
                self.node.get_logger().debug(
                    f"    ✓ This is the CURRENT ACTIVE DIART speaker"
                )

                # Convert to numpy if necessary
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                    self.node.get_logger().debug(f"    Converted to numpy array")

                # Check if it already exists in the database
                if self.use_database:
                    self.node.get_logger().debug(
                        "\033[92m    Searching database for matching speaker...\033[0m"
                    )
                    result = self.db.find_speaker(embedding, self.node.get_logger())

                    if result:
                        eut_speaker_name, distance = result
                        self.node.get_logger().info(
                            f"    ✓✓✓ RECOGNIZED as existing EUT speaker: {eut_speaker_name}"
                        )
                        self.node.get_logger().info(
                            f"    Cosine distance: {distance:.4f}"
                        )
                        self.node.get_logger().info(
                            f"    Setting eut_speaker_id to: {eut_speaker_name}"
                        )

                        # Set the EUT speaker ID
                        self.node.eut_speaker_id = eut_speaker_name

                        # Find if the diart_speaker_id has been included in pending embeddings and remove it
                        if diart_speaker_id in self.pending_embeddings:
                            del self.pending_embeddings[diart_speaker_id]
                            self.node.get_logger().debug(
                                f"    Removed DIART {diart_speaker_id} from pending embeddings (already in DB)"
                            )

                        # Update mapping
                        self.diart_to_eut_mapping[diart_speaker_id] = eut_speaker_name
                    else:
                        # Check if this DIART speaker already has an EUT ID assigned
                        if diart_speaker_id in self.diart_to_eut_mapping:
                            # Use existing mapping
                            self.node.eut_speaker_id = self.diart_to_eut_mapping[
                                diart_speaker_id
                            ]
                            self.node.get_logger().debug(
                                f"    Using existing EUT mapping: {self.node.eut_speaker_id}"
                            )
                        else:
                            # Assign NEW EUT speaker ID
                            new_eut_speaker_number = self.highest_eut_speaker_number + 1
                            self.highest_eut_speaker_number = new_eut_speaker_number
                            self.node.eut_speaker_id = (
                                f"EUT_speaker{new_eut_speaker_number}"
                            )

                            # Create mapping
                            self.diart_to_eut_mapping[diart_speaker_id] = (
                                self.node.eut_speaker_id
                            )

                            self.node.get_logger().debug(
                                f"    ✗✗✗ NEW SPEAKER detected"
                            )
                            self.node.get_logger().debug(
                                f"    Total speakers in DB: {self.number_of_speakers}"
                            )
                            self.node.get_logger().debug(
                                f"    Pending embeddings: {len(self.pending_embeddings)}"
                            )
                            self.node.get_logger().debug(
                                f"    Assigned new EUT speaker number: {new_eut_speaker_number}"
                            )
                            self.node.get_logger().debug(
                                f"    Setting eut_speaker_id to: {self.node.eut_speaker_id}"
                            )

                        # Store embedding in memory if not already there
                        if diart_speaker_id not in self.pending_embeddings:
                            self.pending_embeddings[diart_speaker_id] = embedding
                            self.node.get_logger().debug(
                                f"    Stored in pending embeddings (will be saved on shutdown)"
                            )
                        else:
                            # Update the embedding (it may have changed)
                            self.pending_embeddings[diart_speaker_id] = embedding
                            self.node.get_logger().debug(
                                f"    Updated embedding in pending (already existed)"
                            )
                else:
                    # Database disabled - use clustering speaker_id with EUT prefix
                    self.node.get_logger().debug(
                        "    Database disabled - using clustering speaker ID with EUT prefix"
                    )

                    # Check if we've already mapped this DIART speaker
                    if diart_speaker_id in self.diart_to_eut_mapping:
                        # Use existing mapping
                        self.node.eut_speaker_id = self.diart_to_eut_mapping[
                            diart_speaker_id
                        ]
                        self.node.get_logger().debug(
                            f"    Using existing mapping: DIART {diart_speaker_id} -> {self.node.eut_speaker_id}"
                        )
                    else:
                        # Create new EUT mapping for this DIART speaker
                        new_eut_speaker_number = self.next_eut_speaker_id
                        self.node.eut_speaker_id = (
                            f"EUT_speaker{new_eut_speaker_number}"
                        )
                        self.diart_to_eut_mapping[diart_speaker_id] = (
                            self.node.eut_speaker_id
                        )
                        self.next_eut_speaker_id += 1
                        self.node.get_logger().debug(
                            f"    Created new mapping: DIART {diart_speaker_id} -> {self.node.eut_speaker_id} (number: {new_eut_speaker_number})"
                        )
            else:
                self.node.get_logger().debug(
                    f"    ✗ Not the current active DIART speaker, skipping"
                )
        self._merge_similar_speakers(pipeline_embeddings)

        self.node.get_logger().info(
            f"Final eut_speaker_id value: {self.node.eut_speaker_id}"
        )

        if self.ros4hri_enabled:
            self._handle_ros4hri_publishing(active_diart_speakers)

        self.node.get_logger().debug("*" * 80)

    def _handle_ros4hri_publishing(self, active_diart_speakers):
        """Handle ROS4HRI standard publishing for voices"""
        current_eut_speakers = set()
        current_ros4hri_ids = set()  # IDs without EUT_ prefix for ROS4HRI

        # Map active DIART speakers to EUT speakers
        for d_speaker in active_diart_speakers:
            if d_speaker in self.diart_to_eut_mapping:
                eut_id = self.diart_to_eut_mapping[d_speaker]
                current_eut_speakers.add(eut_id)
                # Remove EUT_ prefix for ROS4HRI standard
                ros4hri_id = eut_id.replace("EUT_", "")
                current_ros4hri_ids.add(ros4hri_id)

        # Publish tracked voices list (without EUT_ prefix)
        ids_msg = IdsList()
        ids_msg.header.stamp = self.node.get_clock().now().to_msg()
        ids_msg.ids = list(current_ros4hri_ids)
        self.node.ids_pub.publish(ids_msg)

        # Handle individual voice publishers
        for eut_id in current_eut_speakers:
            # Remove EUT_ prefix for ROS4HRI topic names
            ros4hri_id = eut_id.replace("EUT_", "")

            # Create publishers if they don't exist (using ros4hri_id for topics)
            if ros4hri_id not in self.node.voice_publishers:
                self.node.create_voice_publishers(ros4hri_id)

            # Update activity timestamp
            self.node.voice_publishers_activity[ros4hri_id] = time.time()

            # Publish is_speaking = True
            bool_msg = Bool()
            bool_msg.data = True
            self.node.voice_publishers[ros4hri_id]["is_speaking"].publish(bool_msg)

            # Publish audio
            # Get last emitted block from source
            if self.node.source and self.node.source.last_emitted_block is not None:
                audio_data = self.node.source.last_emitted_block

                # Convert float32 audio to int16 for AudioData (assuming standard PCM)
                # Float32 is usually -1.0 to 1.0
                audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

                audio_msg = AudioData()
                audio_msg.data = audio_int16.tobytes()
                self.node.voice_publishers[ros4hri_id]["audio"].publish(audio_msg)

        # Handle voices that stopped speaking
        for eut_id in self.active_voices:
            if eut_id not in current_eut_speakers:
                ros4hri_id = eut_id.replace("EUT_", "")
                if ros4hri_id in self.node.voice_publishers:
                    bool_msg = Bool()
                    bool_msg.data = False
                    self.node.voice_publishers[ros4hri_id]["is_speaking"].publish(
                        bool_msg
                    )
                    # Update activity timestamp
                    self.node.voice_publishers_activity[ros4hri_id] = time.time()

        self.active_voices = current_eut_speakers

    def _save_pending_embeddings(self):
        """Save all pending embeddings to the database. Called on node shutdown.
        If multiple DIART speakers map to the same EUT speaker, merge their embeddings by averaging.
        """
        if not self.use_database:
            self.node.get_logger().debug(
                "Database disabled, skipping saving embeddings"
            )
            return

        if not self.pending_embeddings:
            self.node.get_logger().debug("No new embeddings to save")
            return

        self.node.get_logger().debug("=" * 80)
        self.node.get_logger().debug("SAVING PENDING EMBEDDINGS TO DATABASE")
        self.node.get_logger().debug(
            f"Total pending DIART speakers: {len(self.pending_embeddings)}"
        )
        self.node.get_logger().debug(f"DIART->EUT mapping: {self.diart_to_eut_mapping}")

        # Group embeddings by EUT speaker ID
        eut_to_embeddings = {}  # Maps EUT speaker ID to list of embeddings

        for diart_speaker_id, embedding in self.pending_embeddings.items():
            # Get the EUT speaker ID for this DIART speaker
            eut_speaker_id = self.diart_to_eut_mapping.get(diart_speaker_id)

            if eut_speaker_id:
                # Add embedding to the list for this EUT speaker
                if eut_speaker_id not in eut_to_embeddings:
                    eut_to_embeddings[eut_speaker_id] = []
                eut_to_embeddings[eut_speaker_id].append((diart_speaker_id, embedding))
                self.node.get_logger().debug(
                    f"  Grouping DIART {diart_speaker_id} under {eut_speaker_id}"
                )
            else:
                # No mapping found - create a new EUT speaker
                new_eut_speaker_number = self.highest_eut_speaker_number + 1
                self.highest_eut_speaker_number = new_eut_speaker_number
                eut_speaker_id = f"EUT_speaker{new_eut_speaker_number}"
                eut_to_embeddings[eut_speaker_id] = [(diart_speaker_id, embedding)]
                self.node.get_logger().debug(
                    f"  Creating new EUT speaker {eut_speaker_id} for unmapped DIART {diart_speaker_id}"
                )

        self.node.get_logger().debug(
            f"Unique EUT speakers to save: {len(eut_to_embeddings)}"
        )

        # Save each EUT speaker with merged embeddings if necessary
        for eut_speaker_id, embeddings_list in eut_to_embeddings.items():
            self.node.get_logger().debug("-" * 80)
            self.node.get_logger().debug(f"Processing EUT speaker: {eut_speaker_id}")
            self.node.get_logger().debug(
                f"  Number of DIART speakers: {len(embeddings_list)}"
            )

            if len(embeddings_list) == 1:
                # Only one DIART speaker - save directly
                diart_id, embedding = embeddings_list[0]
                self.node.get_logger().debug(f"  Single DIART speaker: {diart_id}")
                self.node.get_logger().debug(
                    f"  Saving to database as: {eut_speaker_id}"
                )
                self.db.save_speaker(eut_speaker_id, embedding)
                self.number_of_speakers += 1
            else:
                # Multiple DIART speakers - merge embeddings by averaging
                self.node.get_logger().debug(
                    f"  Multiple DIART speakers detected - merging embeddings:"
                )
                diart_ids = [diart_id for diart_id, _ in embeddings_list]
                for diart_id in diart_ids:
                    self.node.get_logger().debug(f"    - {diart_id}")

                # Convert all embeddings to numpy arrays and stack them
                embeddings_array = []
                for diart_id, emb in embeddings_list:
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb)
                    embeddings_array.append(emb)

                # Average the embeddings
                embeddings_stack = np.stack(embeddings_array, axis=0)
                merged_embedding = np.mean(embeddings_stack, axis=0)

                self.node.get_logger().debug(
                    f"  Merged embedding shape: {merged_embedding.shape}"
                )
                self.node.get_logger().debug(
                    f"  Merged embedding stats: mean={merged_embedding.mean():.6f}, std={merged_embedding.std():.6f}"
                )
                self.node.get_logger().debug(
                    f"  Saving merged embedding to database as: {eut_speaker_id}"
                )

                # Save the merged embedding
                self.db.save_speaker(eut_speaker_id, merged_embedding)
                self.number_of_speakers += 1

        self.node.get_logger().debug("=" * 80)
        self.node.get_logger().debug(
            f"Successfully saved {len(eut_to_embeddings)} unique speaker(s) to MongoDB"
        )
        self.node.get_logger().debug(
            f"Total speakers in database: {self.number_of_speakers}"
        )


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
            "use_database", True
        )  # Whether to use the database for speaker tracking
        self.declare_parameter(
            "ros4hri_with_id", True
        )  # Enable ROS4HRI standard publishing with ID approach
        self.declare_parameter("cleanup_inactive_topics", False)
        self.declare_parameter("inactive_topic_timeout", 10.0)

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
        self.ros4hri_enabled = (
            self.get_parameter("ros4hri_with_id").get_parameter_value().bool_value
        )
        self.cleanup_inactive_topics = (
            self.get_parameter("cleanup_inactive_topics")
            .get_parameter_value()
            .bool_value
        )
        self.inactive_topic_timeout = (
            self.get_parameter("inactive_topic_timeout")
            .get_parameter_value()
            .double_value
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

        # ROS4HRI Publishers
        self.voice_publishers = {}  # Map eut_id to dict of publishers
        self.voice_publishers_activity = {}  # Map eut_id to last activity timestamp
        if self.ros4hri_enabled:
            self.ids_pub = self.create_publisher(IdsList, "/humans/voices/tracked", 1)

        # Cleanup timer
        if self.cleanup_inactive_topics:
            self.create_timer(1.0, self.cleanup_topics_callback)
            self.get_logger().info(
                f"Topic cleanup enabled with timeout: {self.inactive_topic_timeout}s"
            )

        # Select device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        light_green = "\033[38;5;82m"
        reset = "\033[0m"
        self.get_logger().info(f"{light_green}Using device on diarization: {self.device}{reset}")
        # Add ANSI color helpers for logging
        self._COLOR_GREEN = "\033[92m"
        self._COLOR_RESET = "\033[0m"

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

    def create_voice_publishers(self, voice_id):
        """Create publishers for a specific voice ID"""
        base_topic = f"/humans/voices/{voice_id}"
        self.voice_publishers[voice_id] = {
            "audio": self.create_publisher(AudioData, f"{base_topic}/audio", 10),
            "is_speaking": self.create_publisher(Bool, f"{base_topic}/is_speaking", 10),
            # Note: 'speech' publisher is created in ASR node when needed
            # 'features': TODO
        }
        self.voice_publishers_activity[voice_id] = time.time()
        self.get_logger().info(f"Created ROS4HRI publishers for {voice_id}")

    def cleanup_topics_callback(self):
        """Check for inactive topics and destroy them"""
        current_time = time.time()
        speakers_to_remove = []

        for voice_id, last_active in self.voice_publishers_activity.items():
            if current_time - last_active > self.inactive_topic_timeout:
                speakers_to_remove.append(voice_id)

        for voice_id in speakers_to_remove:
            self.get_logger().info(f"Destroying inactive publishers for {voice_id}")
            if voice_id in self.voice_publishers:
                publishers = self.voice_publishers[voice_id]
                for pub in publishers.values():
                    self.destroy_publisher(pub)
                del self.voice_publishers[voice_id]

            del self.voice_publishers_activity[voice_id]

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

                # Enhanced step-by-step logging for model loading
                self.get_logger().info(
                    "[Diarization] Starting model initialization sequence"
                )
                self.get_logger().info(
                    f"[Diarization] Segmentation model name: {self.segmentation_model_name}"
                )
                self.get_logger().info(
                    f"[Diarization] Embedding model name: {self.embedding_model_name}"
                )
                self.get_logger().info(
                    "[Diarization] Retrieving HuggingFace token from env"
                )
                import os

                hf_token = os.environ.get("HF_TOKEN", None)
                if hf_token is None:
                    self.get_logger().warn(
                        "[Diarization] No HF_TOKEN found. Public models ok; gated models will fail."
                    )
                else:
                    self.get_logger().info("[Diarization] HuggingFace token detected")

                # Load segmentation model
                self.get_logger().info("[Diarization] Loading segmentation model ...")
                segmentation = m.SegmentationModel.from_pretrained(
                    self.segmentation_model_name, use_hf_token=hf_token
                )
                self.get_logger().info(
                    f"[Diarization] Segmentation model loaded: {type(segmentation).__name__}"
                )

                # Load embedding model
                self.get_logger().info("[Diarization] Loading embedding model ...")
                embedding = m.EmbeddingModel.from_pretrained(
                    self.embedding_model_name, use_hf_token=hf_token
                )
                self.get_logger().info(
                    f"[Diarization] Embedding model loaded: {type(embedding).__name__}"
                )

                # Calculate step size to align with block duration
                step_duration = 0.5  # Match with ROSAudioSource block_duration
                self.get_logger().info(
                    f"[Diarization] Using streaming step duration: {step_duration}s"
                )

                # Build config
                self.get_logger().info(
                    "[Diarization] Building SpeakerDiarizationConfig ..."
                )
                self.config = SpeakerDiarizationConfig(
                    segmentation=segmentation,
                    embedding=embedding,
                    device=self.device,
                    sample_rate=int(self.device_samplerate),
                    duration=self.chunk_duration,
                    step=step_duration,
                    tau_active=0.7,
                    delta_new=0.90,
                    # gamma=3.0, #Scale for speaker change detection
                    # beta=10.0, #Beta parameter for speaker change
                    max_speakers=10,
                )
                self.get_logger().info("[Diarization] Config object created")

                # Instantiate pipeline
                self.get_logger().info(
                    "[Diarization] Instantiating SpeakerDiarization pipeline ..."
                )
                self.model = SpeakerDiarization(self.config)
                self.get_logger().info(
                    f"{self._COLOR_GREEN}[Diarization] Pipeline instantiated: {type(self.model).__name__}{self._COLOR_RESET}"
                )

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

                # Mark device info as received and start thread
                self.device_info_received = True
                self.diarization_procedure = threading.Thread(
                    target=self.run_diarization
                )
                self.diarization_procedure.daemon = True
                self.diarization_procedure.start()
                self.get_logger().info("Diarization system initialized and started")

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
