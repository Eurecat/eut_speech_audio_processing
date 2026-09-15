"""REDI backend: VAD speech turns -> ReDimNet2 embeddings -> VoiceIdentityManager.

Independent of DIART. There is no segmentation model and no online clustering:
a *turn* (contiguous speech between pauses) is the voice equivalent of a tracked
face in the face recognition package. Each turn is embedded with ReDimNet2 and
handed to :class:`VoiceIdentityManager`, which owns every identity decision.

Implements the same small contract the diarization node uses for the DIART
engines: ``initialize``, ``push_audio``, ``update_vad_probability``,
``speaker_confidence`` and ``stop``.

A turn may still contain several speakers when nobody pauses: the label follows
whoever dominates the most recent ``max_embed_seconds`` window, and identities
only learn from stretches where consecutive windows agree on the speaker.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

import numpy as np

from speech_recognition.voice_identity_manager import (
    MongoVoiceIdentityStore,
    VoiceIdentityManager,
)


@dataclass
class TurnObservation:
    """A request to embed a turn's speech so far."""

    turn_id: str
    audio: np.ndarray
    mean_vad: float
    final: bool
    turn_speech_seconds: float = 0.0  # speech accumulated in the turn when observed


class TurnSegmenter:
    """Cut a VAD-labelled audio stream into speech turns.

    Only speech chunks are accumulated, so each observation is clean speech and
    never includes the pauses around it. A turn produces a provisional
    observation once it holds ``min_embed_seconds`` of speech, a refreshed one
    every ``embed_interval_seconds`` after that, and a final one when a pause of
    ``turn_silence_seconds`` closes it. Provisional observations give a speaker
    label while the person is still talking instead of only after they stop.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        vad_threshold: float,
        turn_silence_seconds: float = 0.35,
        min_embed_seconds: float = 0.8,
        embed_interval_seconds: float = 0.5,
        max_embed_seconds: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.turn_silence_seconds = turn_silence_seconds
        self.min_embed_samples = int(min_embed_seconds * sample_rate)
        self.embed_interval_samples = int(embed_interval_seconds * sample_rate)
        self.max_embed_samples = int(max_embed_seconds * sample_rate)

        self._turn_number = 0
        self._chunks: List[np.ndarray] = []
        self._vad_values: List[float] = []
        self._speech_samples = 0
        self._next_emit_at = 0
        self._silence_seconds = 0.0
        self._in_turn = False

    @property
    def current_turn_id(self) -> Optional[str]:
        return f"turn{self._turn_number}" if self._in_turn else None

    def push(self, chunk: np.ndarray, vad_probability: float) -> List[TurnObservation]:
        seconds = len(chunk) / float(self.sample_rate)
        observations: List[TurnObservation] = []

        if vad_probability > self.vad_threshold:
            if not self._in_turn:
                self._start_turn()
            self._silence_seconds = 0.0
            self._chunks.append(np.asarray(chunk, dtype=np.float32))
            self._vad_values.append(float(vad_probability))
            self._speech_samples += len(chunk)
            if self._speech_samples >= self._next_emit_at:
                observations.append(self._observe(final=False))
                self._next_emit_at = self._speech_samples + self.embed_interval_samples
            return observations

        if self._in_turn:
            self._silence_seconds += seconds
            if self._silence_seconds >= self.turn_silence_seconds:
                observations.extend(self.close_turn())
        return observations

    def close_turn(self) -> List[TurnObservation]:
        if not self._in_turn:
            return []
        final = []
        if self._speech_samples >= self.min_embed_samples:
            final.append(self._observe(final=True))
        self._in_turn = False
        self._chunks = []
        self._vad_values = []
        self._speech_samples = 0
        return final

    def _start_turn(self) -> None:
        self._turn_number += 1
        self._in_turn = True
        self._chunks = []
        self._vad_values = []
        self._speech_samples = 0
        self._next_emit_at = self.min_embed_samples

    def _observe(self, *, final: bool) -> TurnObservation:
        audio = np.concatenate(self._chunks)
        # The most recent speech only: bounds compute on long turns, and lets a
        # different speaker who takes over without pausing eventually dominate.
        audio = audio[-self.max_embed_samples :]
        return TurnObservation(
            turn_id=f"turn{self._turn_number}",
            audio=audio,
            mean_vad=float(np.mean(self._vad_values)) if self._vad_values else 0.0,
            final=final,
            turn_speech_seconds=self._speech_samples / float(self.sample_rate),
        )


class RediVoiceEngine:
    def __init__(
        self,
        *,
        vad_threshold: float,
        use_database: bool,
        ros4hri_enabled: bool,
        on_eut_speaker_changed: Callable[[Optional[str]], None],
        on_voice_update: Callable[[Set[str], Optional[np.ndarray]], None],
        logger,
        redi_repository: str,
        redi_model_name: str,
        redi_train_type: str,
        redi_dataset: str,
        redi_mongo_uri: str = "",
        min_create_seconds: float = 1.5,
        turn_options: Optional[dict] = None,
        identity_options: Optional[dict] = None,
        **_diart_options,  # the node passes DIART-only settings to every engine
    ) -> None:
        self.vad_threshold = vad_threshold
        self.use_database = use_database
        self.ros4hri_enabled = ros4hri_enabled
        self._on_eut_speaker_changed = on_eut_speaker_changed
        self._on_voice_update = on_voice_update
        self._logger = logger
        self._repository = redi_repository
        self._model_name = redi_model_name
        self._train_type = redi_train_type
        self._dataset = redi_dataset
        self._mongo_uri = redi_mongo_uri or os.environ.get(
            "MONGODB_URI",
            "mongodb://eurecat:cerdanyola@localhost:27017/?authSource=admin",
        )
        self._turn_options = turn_options or {}
        self._min_create_seconds = min_create_seconds
        self._identity_options = identity_options or {}

        self._vad_probability = 0.0
        self._initialized = False
        self._model = None
        self._device = None
        self._segmenter: Optional[TurnSegmenter] = None
        self._manager: Optional[VoiceIdentityManager] = None
        self._queue: "queue.Queue[Optional[TurnObservation]]" = queue.Queue(maxsize=16)
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_speaker: Optional[str] = None
        self._last_confidence = 0.0
        self._sample_rate = 16000
        self._turns: dict = {}

    # ------------------------------------------------------------------
    # Node contract
    # ------------------------------------------------------------------

    @property
    def speaker_confidence(self) -> float:
        return self._last_confidence

    def update_vad_probability(self, probability: float) -> None:
        self._vad_probability = float(probability)

    def initialize(self, sample_rate: int) -> bool:
        if self._initialized:
            return True
        if sample_rate != 16000:
            self._logger.error(f"ReDimNet2 requires 16 kHz audio, received {sample_rate} Hz")
            return False
        try:
            import torch

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._logger.info(
                f"Loading ReDimNet2 {self._model_name}/{self._train_type}/{self._dataset} "
                f"on {self._device}"
            )
            self._model = torch.hub.load(
                self._repository,
                "redimnet2",
                model_name=self._model_name,
                train_type=self._train_type,
                dataset=self._dataset,
                pretrained=True,
                trust_repo=True,
            )
            self._model.eval().to(self._device)
        except Exception as error:
            self._logger.error(f"Failed to load ReDimNet2: {error}")
            return False

        store = None
        if self.use_database:
            model_key = f"redimnet2:{self._model_name}:{self._train_type}:{self._dataset}"
            try:
                store = MongoVoiceIdentityStore(self._mongo_uri, model_key)
            except Exception as error:
                self._logger.warn(
                    f"Voice identity MongoDB unavailable ({error}); using session-only identities"
                )

        self._sample_rate = sample_rate
        self._manager = VoiceIdentityManager(
            logger=self._logger, store=store, **self._identity_options
        )
        self._segmenter = TurnSegmenter(
            sample_rate=sample_rate, vad_threshold=self.vad_threshold, **self._turn_options
        )
        self._worker = threading.Thread(target=self._run, name="redi-voice", daemon=True)
        self._worker.start()
        self._initialized = True
        self._logger.info("REDI voice engine running (VAD turns + ReDimNet2, no DIART)")
        return True

    def push_audio(self, audio_data: np.ndarray) -> None:
        if not self._initialized or self._segmenter is None:
            return
        with self._lock:
            observations = self._segmenter.push(audio_data, self._vad_probability)
        for observation in observations:
            self._enqueue(observation)

    def stop(self) -> None:
        if not self._initialized:
            return
        self._initialized = False
        with self._lock:
            for observation in self._segmenter.close_turn():
                self._enqueue(observation)
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        if self._manager is not None:
            self._manager.close()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _enqueue(self, observation: TurnObservation) -> None:
        try:
            self._queue.put_nowait(observation)
        except queue.Full:
            # Falling behind: a provisional observation is superseded by the next
            # one, so drop it. A final observation carries the learning step and
            # must not be lost.
            if observation.final:
                self._queue.put(observation)

    def _run(self) -> None:
        while True:
            observation = self._queue.get()
            if observation is None:
                return
            try:
                self._handle(observation)
            except Exception as error:
                self._logger.error(f"REDI voice engine error on {observation.turn_id}: {error}")

    def _handle(self, observation: TurnObservation) -> None:
        embedding = self._embed(observation.audio)
        window_seconds = len(observation.audio) / float(self._sample_rate)
        state = self._turns.setdefault(
            observation.turn_id, {"label": None, "learned_at": float("-inf")}
        )

        # Learn only inside a stable stretch of one speaker: the previous window of
        # this turn landed on the same speaker, and enough new speech has passed
        # that this window is not a near-copy of the last one learned. A window
        # straddling a speaker change lands on a different speaker than its
        # predecessor and is not learned. Learning only from whole unmixed turns
        # starved identities, because conversational turns are almost always mixed.
        learn_interval = self._segmenter.max_embed_samples / float(self._sample_rate)
        expected = state["label"]
        spaced = observation.turn_speech_seconds - state["learned_at"] >= learn_interval
        learn_as = expected if expected is not None and spaced else None

        results = self._manager.process_new_embedding_batch(
            {observation.turn_id: embedding},
            speech_seconds=window_seconds,
            quality=observation.mean_vad,
            learn=learn_as is not None,
            learn_if_assigned_to={observation.turn_id: learn_as} if learn_as else None,
            # A short window is too noisy to define a new speaker; it may still
            # recognise one that already exists.
            allow_create=window_seconds >= self._min_create_seconds,
        )
        if observation.final:
            self._turns.pop(observation.turn_id, None)

        kind = "final" if observation.final else "provisional"
        if observation.turn_id not in results:
            self._logger.info(
                f"REDI turn {observation.turn_id} ({kind}, {window_seconds:.2f}s) -> "
                f"no known speaker, too short to create one"
            )
            return

        speaker, confidence = results[observation.turn_id]
        created = state["label"] is None and confidence >= 1.0
        learned = learn_as is not None and speaker == learn_as
        if learned or created:
            state["learned_at"] = observation.turn_speech_seconds
        state["label"] = speaker
        self._last_confidence = confidence

        self._logger.info(
            f"REDI turn {observation.turn_id} ({kind}, {window_seconds:.2f}s) -> {speaker} "
            f"score={confidence:.3f}{' learned' if learned else ''}"
        )
        if speaker != self._last_speaker:
            self._logger.info(f"Active eut_speaker_id: {speaker}")
            self._last_speaker = speaker
        self._on_eut_speaker_changed(speaker)

        if self.ros4hri_enabled:
            self._on_voice_update({speaker}, None)

        if observation.final:
            # A closed turn never produces another observation; drop its mapping so
            # it cannot pin a later turn's speaker.
            with self._lock:
                current = self._segmenter.current_turn_id
            self._manager.cleanup_inactive_track_mappings({current} if current else set())

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
            vector = self._model(tensor.to(self._device)).squeeze(0)
        return vector.detach().float().cpu().numpy()
