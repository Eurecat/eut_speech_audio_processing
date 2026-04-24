import os
import threading
import time
import warnings
from typing import Callable, Dict, List, Optional, Set

import diart.models as m
import numpy as np
import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from pyannote.core import Annotation
from rx.core.observer.observer import Observer

from speech_recognition.ros_audio_source import ROSAudioSource
from speech_recognition.utils.database_utils import DataBaseManager

# ---------------------------------------------------------------------------
# Suppress noisy third-party warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings(
    "ignore", message="torchaudio._backend.set_audio_backend has been deprecated"
)
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")
warnings.filterwarnings("ignore", message="multiple `ModelCheckpoint` callback states")
warnings.filterwarnings(
    "ignore", message="Mismatch between frames", module="pyannote.audio.models.blocks.pooling"
)


# ---------------------------------------------------------------------------
# DiarizationObserver — communicates via callbacks only with the ros2 node
# ---------------------------------------------------------------------------


class DiarizationObserver(Observer):
    """
    Observer attached to the diarization inference that processes predictions and
    manages speaker ID assignment and tracking. Communicates outward via callbacks
    to the ROS2 node, which handles all ROS interactions and VAD logic.
    """

    def __init__(
        self,
        *,
        use_database: bool,
        ros4hri_enabled: bool,
        vad_threshold: float,
        similarity_threshold: float,
        get_current_vad_probability: Callable[[], float],
        get_pipeline: Callable[[], Optional["SpeakerDiarization"]],
        get_last_audio_block: Callable[[], Optional[np.ndarray]],
        on_eut_speaker_changed: Callable[[Optional[str]], None],
        on_voice_update: Callable[[Set[str], Optional[np.ndarray]], None],
        logger,
    ):
        super().__init__()
        self._logger = logger
        self.use_database = use_database
        self.ros4hri_enabled = ros4hri_enabled
        self.vad_threshold = vad_threshold
        self.similarity_threshold = similarity_threshold

        self._get_current_vad_probability = get_current_vad_probability
        self._get_pipeline = get_pipeline
        self._get_last_audio_block = get_last_audio_block
        self._on_eut_speaker_changed = on_eut_speaker_changed
        self._on_voice_update = on_voice_update

        # Speaker tracking
        self.known_diart_speakers: Set[str] = set()
        self.diart_to_eut_mapping: Dict[str, str] = {}
        self.next_eut_speaker_id = 1
        self.highest_eut_speaker_number = 0
        self.active_voices: Set[str] = set()
        self.pending_embeddings: Dict[str, np.ndarray] = {}

        self.last_process_time = time.time()
        self.last_confidence_log_time = time.time()

        # Database
        if self.use_database:
            try:
                self.db = DataBaseManager()
                self.number_of_speakers = self.db.number_speakers()
                self.highest_eut_speaker_number = self.number_of_speakers
                self.next_eut_speaker_id = self.number_of_speakers + 1
                self._logger.info(f"Database enabled — {self.number_of_speakers} speakers loaded")
            except Exception as e:
                self._logger.warn(
                    f"Failed to connect to database: {e}. Falling back to session-only tracking."
                )
                self.use_database = False
                self.db = None
                self.number_of_speakers = 0
        else:
            self.db = None
            self.number_of_speakers = 0
            self._logger.info("Database disabled — using session-only speaker tracking")

    # ------------------------------------------------------------------
    # Observer interface
    # ------------------------------------------------------------------

    # Function called automatically by the diarization inference on every new prediction
    def on_next(self, value) -> None:
        prediction = self._extract_prediction(value)
        if prediction is None:
            return

        current_time = time.time()
        if current_time - self.last_process_time < 0.5:
            return
        self.last_process_time = current_time

        active_diart_speakers: Set[str] = set()
        for track_tuple in prediction.itertracks(yield_label=True):
            if len(track_tuple) == 3:
                _segment, _track, diart_speaker = track_tuple
                active_diart_speakers.add(diart_speaker)

                # Detect new speakers in the during the same session
                if diart_speaker not in self.known_diart_speakers:
                    self.known_diart_speakers.add(diart_speaker)
                    self._logger.debug(f"New DIART speaker discovered: {diart_speaker}")

        # For simplicity, we focus on the first active speaker for embedding extraction and EUT ID assignment.
        # TODO: In a multi-speaker scenario, we would need to extract and track embeddings for all active speakers,
        # and handle cases where speakers overlap, enter, or leave the conversation at different times.
        current_diart_speaker = next(iter(active_diart_speakers), None)

        if current_diart_speaker is None:
            return

        if self._get_current_vad_probability() <= self.vad_threshold:
            self._logger.debug("VAD probability below threshold, skipping embedding extraction")
            return

        pipeline = self._get_pipeline()
        if pipeline is not None:
            try:
                pipeline_embeddings = self._extract_embeddings_from_pipeline(pipeline)
                if pipeline_embeddings:
                    self._process_embeddings(
                        pipeline_embeddings, current_diart_speaker, active_diart_speakers
                    )
                else:
                    self._logger.warn("No embeddings extracted from pipeline")
            except Exception as ex:
                import traceback

                self._logger.error(f"Error extracting embeddings: {ex}")
                self._logger.error(traceback.format_exc())

    def on_error(self, error: Exception) -> None:
        self._logger.error(f"DiarizationObserver error: {error}")
        if self.db:
            self.db.close()

    def on_completed(self) -> None:
        self._logger.info("Diarization stream completed.")

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_prediction(self, value):
        if isinstance(value, tuple):
            return value[0]
        if isinstance(value, Annotation):
            return value
        return None

    def _extract_embeddings_from_pipeline(self, pipeline: "SpeakerDiarization") -> Dict:
        embeddings = {}
        if not hasattr(pipeline, "clustering"):
            return embeddings

        clustering = pipeline.clustering
        active_centers = getattr(clustering, "active_centers", [])
        centers = getattr(clustering, "centers", None)

        if centers is None:
            return embeddings

        for center_idx in active_centers:
            diart_speaker_id = f"speaker{center_idx}"
            embeddings[diart_speaker_id] = centers[center_idx]

        return embeddings

    def _merge_similar_speakers(self, pipeline_embeddings: Dict) -> None:
        if len(pipeline_embeddings) < 2:
            return

        speaker_ids = list(pipeline_embeddings.keys())
        normalized: List[np.ndarray] = []
        for sid in speaker_ids:
            emb = np.array(pipeline_embeddings[sid])
            normalized.append(emb / (np.linalg.norm(emb) + 1e-8))

        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                sid_i, sid_j = speaker_ids[i], speaker_ids[j]
                eut_i = self.diart_to_eut_mapping.get(sid_i)
                eut_j = self.diart_to_eut_mapping.get(sid_j)

                if eut_i and eut_j and eut_i == eut_j:
                    continue

                cosine_distance = 1.0 - float(np.dot(normalized[i], normalized[j]))
                if cosine_distance < self.similarity_threshold:
                    if eut_i and eut_j:
                        keep, discard = min(eut_i, eut_j), max(eut_i, eut_j)
                        for diart_id, eut_id in self.diart_to_eut_mapping.items():
                            if eut_id == discard:
                                self.diart_to_eut_mapping[diart_id] = keep
                    elif eut_i:
                        self.diart_to_eut_mapping[sid_j] = eut_i
                    elif eut_j:
                        self.diart_to_eut_mapping[sid_i] = eut_j

    # ------------------------------------------------------------------
    # Embedding processing and speaker ID assignment
    # ------------------------------------------------------------------

    def _process_embeddings(
        self,
        pipeline_embeddings: Dict,
        current_diart_speaker: str,
        active_diart_speakers: Set[str],
    ) -> None:
        for diart_speaker_id, embedding in pipeline_embeddings.items():
            if diart_speaker_id != current_diart_speaker:
                continue

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            eut_speaker_id = self._resolve_eut_speaker(diart_speaker_id, embedding)
            self._on_eut_speaker_changed(eut_speaker_id)
            self._logger.info(f"Final eut_speaker_id: {eut_speaker_id}")

        self._merge_similar_speakers(pipeline_embeddings)

        if self.ros4hri_enabled:
            current_eut_speakers: Set[str] = set()
            for d_speaker in active_diart_speakers:
                if d_speaker in self.diart_to_eut_mapping:
                    current_eut_speakers.add(self.diart_to_eut_mapping[d_speaker])

            audio_block = self._get_last_audio_block()
            self._on_voice_update(current_eut_speakers, audio_block)
            self.active_voices = current_eut_speakers

    def _resolve_eut_speaker(self, diart_speaker_id: str, embedding: np.ndarray) -> str:
        """Return the EUT speaker ID for a DIART speaker, creating one if needed."""
        if self.use_database:
            result = self.db.find_speaker(embedding, self._logger)
            if result:
                eut_speaker_name, distance = result
                self._logger.info(
                    f"Recognized existing speaker: {eut_speaker_name} (distance: {distance:.4f})"
                )
                self.diart_to_eut_mapping[diart_speaker_id] = eut_speaker_name
                self.pending_embeddings.pop(diart_speaker_id, None)
                return eut_speaker_name

        # Not found in DB or DB disabled — use existing mapping or create new
        if diart_speaker_id in self.diart_to_eut_mapping:
            return self.diart_to_eut_mapping[diart_speaker_id]

        if self.use_database:
            new_number = self.highest_eut_speaker_number + 1
            self.highest_eut_speaker_number = new_number
        else:
            new_number = self.next_eut_speaker_id
            self.next_eut_speaker_id += 1

        eut_speaker_id = f"EUT_speaker{new_number}"
        self.diart_to_eut_mapping[diart_speaker_id] = eut_speaker_id
        self.pending_embeddings[diart_speaker_id] = embedding
        self._logger.debug(f"New speaker assigned: DIART {diart_speaker_id} -> {eut_speaker_id}")
        return eut_speaker_id

    # ------------------------------------------------------------------
    # Shutdown persistence
    # ------------------------------------------------------------------

    def save_pending_embeddings(self) -> None:
        """Save all pending embeddings to the database. Call on shutdown."""
        if not self.use_database or not self.pending_embeddings:
            return

        # Group embeddings by EUT speaker ID
        eut_to_embeddings: Dict[str, List] = {}
        for diart_id, embedding in self.pending_embeddings.items():
            eut_id = self.diart_to_eut_mapping.get(diart_id)
            if not eut_id:
                new_number = self.highest_eut_speaker_number + 1
                self.highest_eut_speaker_number = new_number
                eut_id = f"EUT_speaker{new_number}"
            eut_to_embeddings.setdefault(eut_id, []).append(embedding)

        for eut_id, embs in eut_to_embeddings.items():
            emb_arrays = [np.array(e) for e in embs]
            merged = (
                np.mean(np.stack(emb_arrays, axis=0), axis=0)
                if len(emb_arrays) > 1
                else emb_arrays[0]
            )
            self.db.save_speaker(eut_id, merged)
            self.number_of_speakers += 1
            self._logger.info(f"Saved speaker {eut_id} to database.")


# ---------------------------------------------------------------------------
# DiarizationEngine — owns model loading, pipeline, source, and threading
# ---------------------------------------------------------------------------


class DiarizationEngine:
    """Owns the full diarization pipeline with zero ROS2 dependencies.

    Communicates outward via three callbacks:
      - on_eut_speaker_changed(eut_speaker_id: Optional[str])
          Fired when the active speaker changes. Engine passes None on silence.
      - on_voice_update(active_eut_speakers: Set[str], audio_block: Optional[np.ndarray])
          Fired each diarization step with the current set of active speakers
          and the latest raw audio block, for ROS4HRI voice publishing.
      - on_speech_activity(speaker_id: str, active: bool)
          NOT fired here — the node owns VAD buffering logic and decides
          when to publish active/inactive. The engine only fires on_eut_speaker_changed.
    """

    def __init__(
        self,
        *,
        chunk_duration: float,
        overlap_duration: float,
        segmentation_model_name: str,
        embedding_model_name: str,
        vad_threshold: float,
        similarity_threshold: float,
        use_database: bool,
        ros4hri_enabled: bool,
        on_eut_speaker_changed: Callable[[Optional[str]], None],
        on_voice_update: Callable[[Set[str], Optional[np.ndarray]], None],
        logger,
    ):
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.segmentation_model_name = segmentation_model_name
        self.embedding_model_name = embedding_model_name
        self.vad_threshold = vad_threshold
        self.similarity_threshold = similarity_threshold
        self.use_database = use_database
        self.ros4hri_enabled = ros4hri_enabled
        self._on_eut_speaker_changed = on_eut_speaker_changed
        self._on_voice_update = on_voice_update
        self._logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f"Using device on diarization: {self.device}")

        self.source: Optional[ROSAudioSource] = None
        self.model: Optional[SpeakerDiarization] = None
        self.inference: Optional[StreamingInference] = None
        self.observer: Optional[DiarizationObserver] = None

        self._current_vad_probability: float = 0.0
        self._initialized: bool = False
        self._diarization_started: bool = False
        self._diarization_thread: Optional[threading.Thread] = None

    def update_vad_probability(self, probability: float) -> None:
        """Called by the node on each VAD message."""
        self._current_vad_probability = probability

    def push_audio(self, audio_data: np.ndarray) -> None:
        """Feed a new audio chunk from the ROS subscriber into the audio source."""
        if self.source is not None and self._initialized:
            self.source.add_audio_chunk(audio_data)

    def initialize(self, sample_rate: int) -> bool:
        """Load models and set up the pipeline. Returns True on success."""
        if self._initialized:
            return True
        try:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token is None:
                self._logger.warn(
                    "No Hugging Face token found (HF_TOKEN/HUGGINGFACE_HUB_TOKEN). "
                    "Gated models like pyannote/segmentation will fail unless weights are cached locally."
                )

            self._logger.info(f"Loading segmentation model: {self.segmentation_model_name}")
            segmentation = m.SegmentationModel.from_pretrained(
                self.segmentation_model_name, use_hf_token=hf_token
            )
            if segmentation is None:
                raise RuntimeError(
                    f"Segmentation model '{self.segmentation_model_name}' could not be loaded "
                    "(wrapper or internal model is None). Accept its conditions at "
                    "https://hf.co/pyannote/segmentation and set HF_TOKEN."
                )
            self._logger.info(f"Segmentation model loaded: {type(segmentation.model).__name__}")

            self._logger.info(f"Loading embedding model: {self.embedding_model_name}")
            embedding = m.EmbeddingModel.from_pretrained(
                self.embedding_model_name, use_hf_token=hf_token
            )
            if embedding is None:
                raise RuntimeError(
                    f"Embedding model '{self.embedding_model_name}' could not be loaded "
                    "(wrapper or internal model is None). Accept its conditions at "
                    "https://hf.co/pyannote/embedding and set HF_TOKEN."
                )
            self._logger.info(f"Embedding model loaded: {type(embedding.model).__name__}")

            step_duration = 0.5
            self.config = SpeakerDiarizationConfig(
                segmentation=segmentation,
                embedding=embedding,
                device=self.device,
                sample_rate=sample_rate,
                duration=self.chunk_duration,
                step=step_duration,
                tau_active=0.7,
                delta_new=0.90,
                max_speakers=10,
            )
            self.model = SpeakerDiarization(self.config)
            self._logger.info(f"Pipeline instantiated: {type(self.model).__name__}")

            self.source = ROSAudioSource(sample_rate=sample_rate, block_duration=step_duration)
            self.source.read()

            self.observer = DiarizationObserver(
                use_database=self.use_database,
                ros4hri_enabled=self.ros4hri_enabled,
                vad_threshold=self.vad_threshold,
                similarity_threshold=self.similarity_threshold,
                get_current_vad_probability=lambda: self._current_vad_probability,
                get_pipeline=lambda: self.model,
                get_last_audio_block=lambda: self.source.last_emitted_block
                if self.source
                else None,
                on_eut_speaker_changed=self._on_eut_speaker_changed,
                on_voice_update=self._on_voice_update,
                logger=self._logger,
            )

            self._initialized = True
            self._diarization_thread = threading.Thread(target=self._run_diarization, daemon=True)
            self._diarization_thread.start()
            self._logger.info("Diarization engine initialized and running.")
            return True

        except Exception as e:
            import traceback

            self._logger.error(f"Failed to initialize diarization engine: {e}")
            self._logger.error(traceback.format_exc())
            return False

    def stop(self) -> None:
        """Save embeddings and close resources cleanly."""
        self._diarization_started = False
        if self.observer is not None:
            self.observer.save_pending_embeddings()
            if self.observer.db:
                self.observer.db.close()
        if self.source is not None:
            self.source.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_diarization(self) -> None:
        if not self._initialized or self.source is None or self.model is None:
            self._logger.error("Cannot start diarization: engine not initialized.")
            return
        if self._diarization_started:
            self._logger.warn("Diarization already running.")
            return
        try:
            self.inference = StreamingInference(self.model, self.source, do_plot=False)
            self.inference.attach_observers(self.observer)
            self._diarization_started = True
            self.inference()
        except Exception as e:
            self._logger.error(f"Diarization pipeline error: {e}")
            self._diarization_started = False
