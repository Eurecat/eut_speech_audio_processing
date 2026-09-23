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
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Set

import numpy as np

from speech_recognition.voice_identity_manager import (
    NO_MATCH_SCORE,
    MongoVoiceIdentityStore,
    VoiceIdentityManager,
    normalize_embedding,
)


@dataclass
class TurnObservation:
    """A request to embed a turn segment's speech so far."""

    turn_id: str  # track id of the segment: "turn7", then "turn7.2" after a split
    audio: np.ndarray
    mean_vad: float
    final: bool
    turn_speech_seconds: float = 0.0  # speech accumulated in the segment when observed
    probe: Optional[np.ndarray] = None  # most recent speech only, to detect a speaker change
    end_sample: int = 0  # speech samples in the whole turn when observed


class TurnSegmenter:
    """Cut a VAD-labelled audio stream into speech turns.

    Only speech chunks are accumulated, so each observation is clean speech and
    never includes the pauses around it. A turn produces a provisional
    observation once it holds ``min_embed_seconds`` of speech, a refreshed one
    every ``embed_interval_seconds`` after that, and a final one when a pause of
    ``turn_silence_seconds`` closes it. Provisional observations give a speaker
    label while the person is still talking instead of only after they stop.

    When someone takes over without pausing, the engine calls :meth:`split` and
    the turn continues as a new *segment* with its own track id, whose windows
    only contain speech from the split point on. Once a segment is longer than
    its probe, each observation also carries ``probe``: the last
    ``probe_seconds`` of speech, short enough to reveal a change long before the
    newcomer dominates the full window.
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
        probe_seconds: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.turn_silence_seconds = turn_silence_seconds
        self.min_embed_samples = int(min_embed_seconds * sample_rate)
        self.embed_interval_samples = int(embed_interval_seconds * sample_rate)
        self.max_embed_samples = int(max_embed_seconds * sample_rate)
        self.probe_samples = int(probe_seconds * sample_rate)

        self._turn_number = 0
        self._segment_number = 1
        self._segment_start = 0
        self._chunks: List[np.ndarray] = []
        self._vad_values: List[float] = []
        self._speech_samples = 0
        self._next_emit_at = 0
        self._silence_seconds = 0.0
        self._in_turn = False

    @property
    def current_turn_id(self) -> Optional[str]:
        if not self._in_turn:
            return None
        if self._segment_number == 1:
            return f"turn{self._turn_number}"
        return f"turn{self._turn_number}.{self._segment_number}"

    def split(self, track_id: str, at_sample: int) -> Optional[str]:
        """Start a new segment of the open turn at ``at_sample`` (turn speech samples).

        Returns the new segment's track id, or ``None`` when ``track_id`` is no
        longer the open segment (the turn ended or was split meanwhile).
        """
        if track_id != self.current_turn_id or at_sample <= self._segment_start:
            return None
        self._segment_start = min(at_sample, self._speech_samples)
        self._segment_number += 1
        return self.current_turn_id

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
        if self._speech_samples - self._segment_start >= self.min_embed_samples:
            final.append(self._observe(final=True))
        self._in_turn = False
        self._chunks = []
        self._vad_values = []
        self._speech_samples = 0
        return final

    def _start_turn(self) -> None:
        self._turn_number += 1
        self._segment_number = 1
        self._segment_start = 0
        self._in_turn = True
        self._chunks = []
        self._vad_values = []
        self._speech_samples = 0
        self._next_emit_at = self.min_embed_samples

    def _observe(self, *, final: bool) -> TurnObservation:
        segment = np.concatenate(self._chunks)[self._segment_start :]
        # The most recent speech only: bounds compute on long turns, and lets a
        # different speaker who takes over without pausing eventually dominate.
        audio = segment[-self.max_embed_samples :]
        probe = None
        if self.probe_samples and len(segment) >= self.probe_samples + self.embed_interval_samples:
            probe = segment[-self.probe_samples :]
        return TurnObservation(
            turn_id=self.current_turn_id,
            audio=audio,
            mean_vad=float(np.mean(self._vad_values)) if self._vad_values else 0.0,
            final=final,
            turn_speech_seconds=len(segment) / float(self.sample_rate),
            probe=probe,
            end_sample=self._speech_samples,
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
        change_threshold: float = 0.35,
        create_on_ambiguous_probe: bool = True,
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
        self._change_threshold = change_threshold
        self._create_on_ambiguous_probe = create_on_ambiguous_probe
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
        self._retired_tracks: Deque[str] = deque(maxlen=64)
        # A turn that ends too short to CREATE a speaker on its own (see
        # _min_create_seconds) is kept here instead of just discarded. If the
        # *next* such orphan sounds like the same voice, that is much stronger
        # evidence than either turn alone, and the pair creates the identity
        # together. Cleared the moment any turn resolves normally, so it only
        # ever bridges two consecutive, otherwise-unidentified turns.
        self._pending_orphan: Optional[tuple[np.ndarray, float]] = None

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
                self._logger.info(f"Voice identities persisted in MongoDB (model_key={model_key})")
            except Exception as error:
                self._logger.warn(
                    f"Voice identity MongoDB unavailable ({error}); using session-only identities"
                )
        else:
            self._logger.info("Voice identity database disabled: session-only identities")

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

    @staticmethod
    def _new_state() -> dict:
        return {"label": None, "learned_at": float("-inf"), "seeded": None, "seed_seconds": 0.0}

    def _handle(self, observation: TurnObservation) -> None:
        if observation.turn_id in self._retired_tracks:
            # Queued before its segment was split: it mixes the old speaker's audio
            # with the newcomer's, and the new segment already covers what follows.
            return
        state = self._turns.setdefault(observation.turn_id, self._new_state())
        if observation.probe is not None and state["label"] is not None:
            probe_embedding = self._embed(observation.probe)
            probe_score = self._manager.score(state["label"], probe_embedding)
            if probe_score is not None and probe_score < self._change_threshold:
                self._handle_change(observation, state, probe_embedding, probe_score)
                return

        embedding = self._embed(observation.audio)
        window_seconds = len(observation.audio) / float(self._sample_rate)

        # Learn only inside a stable stretch of one speaker: the previous window of
        # this segment landed on the same speaker, and enough new speech has passed
        # that this window is not a near-copy of the last one learned. Learning only
        # from whole unmixed turns starved identities, because conversational turns
        # are almost always mixed.
        learn_interval = self._segmenter.max_embed_samples / float(self._sample_rate)
        expected = state["label"]
        spaced = observation.turn_speech_seconds - state["learned_at"] >= learn_interval
        learn_as = expected if expected is not None and spaced else None
        # A speaker this segment created is still defined by its short seed: grow
        # the seed with the segment instead of averaging a longer window into it.
        reseed = state["seeded"] is not None and learn_as is None

        known = set(self._manager.identities)
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
        if observation.turn_id not in results:
            if observation.final:
                self._turns.pop(observation.turn_id, None)
                confirmed = self._try_confirm_orphan(observation.turn_id, embedding, window_seconds)
                if confirmed is not None:
                    speaker, note = confirmed
                    state["label"], state["seeded"], state["seed_seconds"] = speaker, speaker, window_seconds
                    state["learned_at"] = observation.turn_speech_seconds
                    self._publish(observation, state, speaker, NO_MATCH_SCORE, window_seconds, note)
                    return
            kind = "final" if observation.final else "provisional"
            self._logger.info(
                f"REDI turn {observation.turn_id} ({kind}, {window_seconds:.2f}s) -> "
                f"no known speaker, too short to create one"
            )
            return

        self._pending_orphan = None
        speaker, confidence = results[observation.turn_id]
        created = speaker not in known
        learned = learn_as is not None and speaker == learn_as
        note = " learned" if learned else ""
        if reseed and speaker == state["seeded"] and window_seconds > state["seed_seconds"]:
            if self._manager.reseed_identity(speaker, embedding, window_seconds):
                state["seed_seconds"] = window_seconds
                note = " reseeded"
        if created:
            state["seeded"], state["seed_seconds"] = speaker, window_seconds
        if learned or created:
            state["learned_at"] = observation.turn_speech_seconds
        if learned or speaker != state["seeded"]:
            state["seeded"] = None
        self._publish(observation, state, speaker, confidence, window_seconds, note)

    def _try_confirm_orphan(
        self, turn_id: str, embedding: np.ndarray, window_seconds: float
    ) -> Optional[tuple[str, str]]:
        """A turn just ended too short to create a speaker on its own (see
        ``_min_create_seconds``). Compare it against the last such orphan, if one
        is still pending: two independent turns that both match nobody known and
        sound like each other are much stronger evidence than either alone, so
        the pair creates the identity together, seeded from their combined
        audio instead of either turn's noisy-on-its-own embedding.

        Returns ``(speaker, log note)`` once confirmed, else stashes this turn
        as the new pending orphan (replacing any older one) and returns None.
        """
        try:
            vector = normalize_embedding(embedding)
        except ValueError:
            return None

        pending = self._pending_orphan
        self._pending_orphan = (embedding, window_seconds)
        if pending is None:
            return None

        prev_embedding, prev_seconds = pending
        similarity = float(normalize_embedding(prev_embedding) @ vector)
        if similarity < self._manager.merge_threshold:
            return None

        combined_seconds = prev_seconds + window_seconds
        combined_embedding = (
            prev_embedding * prev_seconds + embedding * window_seconds
        ) / combined_seconds
        results = self._manager.process_new_embedding_batch(
            {turn_id: combined_embedding},
            speech_seconds=combined_seconds,
            learn=True,
            allow_create=True,
        )
        self._pending_orphan = None
        if turn_id not in results:
            return None
        speaker, _ = results[turn_id]
        self._logger.info(
            f"Two unidentified turns matched each other (similarity={similarity:.3f}) -> "
            f"{speaker} from their combined {combined_seconds:.2f}s"
        )
        return speaker, " confirmed from prior orphan"

    def _handle_change(
        self,
        observation: TurnObservation,
        state: dict,
        probe_embedding: np.ndarray,
        probe_score: float,
    ) -> None:
        """The latest speech no longer sounds like the segment's speaker: split there.

        Only the probe is labelled now. It is too short to learn from, but it may
        create a speaker, because a change was detected: waiting for a full window
        of the newcomer is exactly the lag this avoids. Such a speaker is reseeded
        from the longer windows of its own segment as they arrive.
        """
        at_sample = observation.end_sample - len(observation.probe)
        with self._lock:
            track_id = self._segmenter.split(observation.turn_id, at_sample)
        # The turn already moved on (ended or split): still label the probe, under
        # a track id no later observation will reuse.
        track_id = track_id or f"{observation.turn_id}@{at_sample}"
        self._turns.pop(observation.turn_id, None)
        self._retired_tracks.append(observation.turn_id)

        probe_seconds = len(observation.probe) / float(self._sample_rate)
        known = set(self._manager.identities)
        results = self._manager.process_new_embedding_batch(
            {track_id: probe_embedding},
            speech_seconds=probe_seconds,
            quality=observation.mean_vad,
            learn=False,
            # Whether a near-tie against two known speakers may still create a
            # third. Suppressing it here looks obviously right and was measured:
            # it does not help, because the same ambiguity reappears on the full
            # window a moment later and creates the identity anyway. Default
            # keeps the behaviour that measured better. See the config comment.
            allow_create_when_ambiguous=self._create_on_ambiguous_probe,
        )
        self._logger.info(
            f"REDI change in {observation.turn_id}: last {probe_seconds:.2f}s scored "
            f"{probe_score:.3f} against {state['label']}, new segment {track_id}"
        )
        if track_id not in results:
            return
        speaker, confidence = results[track_id]
        new_state = self._new_state()
        # Learning resumes only after a full window beyond the probe, like after creation.
        new_state["learned_at"] = probe_seconds
        if speaker not in known:
            new_state["seeded"], new_state["seed_seconds"] = speaker, probe_seconds
        if observation.final or "@" in track_id:
            observation = TurnObservation(**{**observation.__dict__, "final": True})
        else:
            self._turns[track_id] = new_state
        observation = TurnObservation(**{**observation.__dict__, "turn_id": track_id})
        self._publish(observation, new_state, speaker, confidence, probe_seconds, " probe")

    def _publish(
        self,
        observation: TurnObservation,
        state: dict,
        speaker: str,
        confidence: float,
        window_seconds: float,
        note: str,
    ) -> None:
        state["label"] = speaker
        self._last_confidence = confidence
        if observation.final:
            self._turns.pop(observation.turn_id, None)

        kind = "final" if observation.final else "provisional"
        self._logger.info(
            f"REDI turn {observation.turn_id} ({kind}, {window_seconds:.2f}s) -> {speaker} "
            f"score={confidence:.3f}{note}"
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
            # Save at every turn end rather than only at shutdown: a container that
            # is killed never runs the shutdown flush.
            self._manager.flush()

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
            vector = self._model(tensor.to(self._device)).squeeze(0)
        return vector.detach().float().cpu().numpy()
