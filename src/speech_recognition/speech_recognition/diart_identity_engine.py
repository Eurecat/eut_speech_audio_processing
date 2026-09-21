"""DIART backend with speaker identities owned by :class:`VoiceIdentityManager`.

DIART still does segmentation and online clustering, exactly as in the legacy
backend. Only the step that turns DIART's transient local tracks into persistent
``EUT_speakerN`` ids is replaced.

The legacy step maps a DIART track to an identity once and then keeps it forever,
updating that identity's centroid on every call without any similarity, overlap
or quality check. If DIART later reuses a track index for someone else, that
person silently inherits the identity and poisons it. The shared manager instead
re-matches every batch with exclusive assignment, a stickiness margin, gated
updates and identity merging, which is also what the REDI backend uses.
"""

from __future__ import annotations

import os
from typing import Dict, Set

import numpy as np

from speech_recognition.diarization_engine import DiarizationEngine, DiarizationObserver
from speech_recognition.voice_identity_manager import (
    MongoVoiceIdentityStore,
    VoiceIdentityManager,
)


class ManagedIdentityObserver(DiarizationObserver):
    def __init__(
        self,
        *,
        identity_manager: VoiceIdentityManager,
        chunk_step_seconds: float,
        **kwargs,
    ) -> None:
        # The manager owns persistence; the legacy per-track Mongo lookup must
        # not run alongside it.
        kwargs["use_database"] = False
        super().__init__(**kwargs)
        self.identity_manager = identity_manager
        self.chunk_step_seconds = chunk_step_seconds
        self.last_assignment_confidence = 0.0

    def _process_embeddings(
        self,
        pipeline_embeddings: Dict,
        current_diart_speaker: str,
        active_diart_speakers: Set[str],
    ) -> None:
        results = self.identity_manager.process_new_embedding_batch(
            {track: np.asarray(embedding) for track, embedding in pipeline_embeddings.items()},
            speech_seconds=self.chunk_step_seconds,
            quality=float(self._get_current_vad_probability()),
            overlapped=len(active_diart_speakers) > 1,
        )
        for track, (unique_id, _confidence) in results.items():
            self.diart_to_eut_mapping[track] = unique_id
        self.identity_manager.cleanup_inactive_track_mappings(set(pipeline_embeddings))

        primary = self.diart_to_eut_mapping.get(current_diart_speaker)
        if primary is None:
            return
        if current_diart_speaker in results:
            self.last_assignment_confidence = results[current_diart_speaker][1]

        if primary != self._last_active_eut_speaker:
            self._logger.info(f"Active eut_speaker_id: {primary}")
            self._last_active_eut_speaker = primary
        self._on_eut_speaker_changed(primary)

        if self.ros4hri_enabled:
            active = {
                self.diart_to_eut_mapping[track]
                for track in active_diart_speakers
                if track in self.diart_to_eut_mapping
            }
            self._on_voice_update(active, self._get_last_audio_block())
            self.active_voices = active

    def _merge_similar_speakers(self, pipeline_embeddings: Dict) -> None:
        # Merging is done by the manager, against whole identity populations.
        return

    def save_pending_embeddings(self) -> None:
        self.identity_manager.flush()


class DiartManagedIdentityEngine(DiarizationEngine):
    def __init__(
        self,
        *,
        identity_options: Dict,
        mongo_uri: str = "",
        **kwargs,
    ) -> None:
        self._identity_options = identity_options
        self._mongo_uri = mongo_uri or os.environ.get(
            "MONGODB_URI",
            "mongodb://eurecat:cerdanyola@localhost:27017/?authSource=admin",
        )
        super().__init__(**kwargs)

    @property
    def speaker_confidence(self) -> float:
        if isinstance(self.observer, ManagedIdentityObserver):
            return self.observer.last_assignment_confidence
        return 0.0

    def _create_observer(self) -> DiarizationObserver:
        store = None
        if self.use_database:
            # Namespaced by model so pyannote vectors are never compared with
            # ReDimNet2 vectors stored by the REDI backend.
            model_key = f"diart:{self.embedding_model_name}"
            try:
                store = MongoVoiceIdentityStore(self._mongo_uri, model_key)
            except Exception as error:
                self._logger.warn(
                    f"Voice identity MongoDB unavailable ({error}); using session-only identities"
                )

        manager = VoiceIdentityManager(logger=self._logger, store=store, **self._identity_options)
        return ManagedIdentityObserver(
            identity_manager=manager,
            chunk_step_seconds=self.step_duration,
            use_database=False,
            ros4hri_enabled=self.ros4hri_enabled,
            vad_threshold=self.vad_threshold,
            similarity_threshold=self.similarity_threshold,
            get_current_vad_probability=lambda: self._current_vad_probability,
            get_pipeline=lambda: self.model,
            get_last_audio_block=lambda: self.source.last_emitted_block if self.source else None,
            on_eut_speaker_changed=self._on_eut_speaker_changed,
            on_voice_update=self._on_voice_update,
            logger=self._logger,
        )
