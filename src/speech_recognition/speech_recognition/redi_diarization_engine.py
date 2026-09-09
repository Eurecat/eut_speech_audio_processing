"""ReDimNet2-backed variant of the existing DIART streaming pipeline.

DIART remains responsible for overlap-aware segmentation and online temporal
tracking. Its embedding model is replaced by ReDimNet2, while persistent EUT
speaker identities are assigned by :class:`SpeakerIdentityManager` in RAM.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Set

import numpy as np
import torch
import torch.nn.functional as functional

import diart.models as diart_models

from speech_recognition.diarization_engine import DiarizationEngine, DiarizationObserver
from speech_recognition.redi_speaker_identity import (
    MongoSpeakerIdentityStore,
    SpeakerIdentityManager,
)


class ReDimNet2EmbeddingAdapter(torch.nn.Module):
    """Adapt ReDimNet2 waveform inference to DIART's weighted embedding API."""

    def __init__(
        self,
        *,
        repository: str,
        model_name: str,
        train_type: str,
        dataset: str,
        sample_rate: int = 16000,
        min_speech_seconds: float = 0.40,
        weight_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_seconds * sample_rate)
        self.weight_threshold = weight_threshold
        self.model = torch.hub.load(
            repository,
            "redimnet2",
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
            pretrained=True,
            trust_repo=True,
        )
        self.model.eval()
        self.embedding_dimension = int(getattr(self.model, "embed_dim", 192))

    def forward(
        self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3:
            raise ValueError(
                f"Expected waveform shape (batch, channels, samples), got {waveform.shape}"
            )

        mono = waveform.mean(dim=1)
        if weights is None:
            embeddings = self.model(mono)
            return functional.normalize(embeddings, dim=-1)

        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        sample_weights = functional.interpolate(
            weights.unsqueeze(1).to(dtype=mono.dtype),
            size=mono.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        outputs = []
        for signal, mask in zip(mono, sample_weights):
            clean_signal = signal[mask >= self.weight_threshold]
            if clean_signal.numel() < self.min_speech_samples:
                outputs.append(
                    torch.full(
                        (self.embedding_dimension,),
                        float("nan"),
                        dtype=mono.dtype,
                        device=mono.device,
                    )
                )
                continue
            embedding = self.model(clean_signal.unsqueeze(0)).squeeze(0)
            outputs.append(functional.normalize(embedding, dim=-1))
        return torch.stack(outputs)


class RediDiarizationObserver(DiarizationObserver):
    """Map transient DIART tracks onto guarded, persistent speaker identities."""

    def __init__(
        self,
        *,
        identity_manager: SpeakerIdentityManager,
        chunk_step_seconds: float,
        **kwargs,
    ) -> None:
        # The legacy observer's Mongo lookup is intentionally disabled. The REDI
        # manager owns persistence and performs all hot-path matching in RAM.
        kwargs["use_database"] = False
        super().__init__(**kwargs)
        self.identity_manager = identity_manager
        self.chunk_step_seconds = chunk_step_seconds
        self.last_assignment_confidence = 0.0
        self._active_speaker_count = 1

    def _process_embeddings(
        self,
        pipeline_embeddings: Dict,
        current_diart_speaker: str,
        active_diart_speakers: Set[str],
    ) -> None:
        self._active_speaker_count = len(active_diart_speakers)
        super()._process_embeddings(
            pipeline_embeddings, current_diart_speaker, active_diart_speakers
        )

    def _resolve_eut_speaker(self, diart_speaker_id: str, embedding: np.ndarray) -> str:
        assignment = self.identity_manager.assign(
            diart_speaker_id,
            embedding,
            speech_seconds=self.chunk_step_seconds,
            quality=float(self._get_current_vad_probability()),
            overlapped=self._active_speaker_count > 1,
        )
        self.diart_to_eut_mapping[diart_speaker_id] = assignment.speaker_id
        self.last_assignment_confidence = assignment.confidence
        self._logger.info(
            "REDI assignment: "
            f"track={diart_speaker_id} identity={assignment.speaker_id} "
            f"score={assignment.confidence:.3f} state={assignment.state} "
            f"updated={assignment.accepted_for_update}"
        )
        return assignment.speaker_id

    def _merge_similar_speakers(self, pipeline_embeddings: Dict) -> None:
        # Identity merging is intentionally not performed from one window. Stable
        # assignment is handled by top-1/top-2 margin checks and track hysteresis.
        return

    def save_pending_embeddings(self) -> None:
        self.identity_manager.flush()

    def on_error(self, error: Exception) -> None:
        self._logger.error(f"REDI diarization observer error: {error}")


class RediDiarizationEngine(DiarizationEngine):
    """Selectable DIART + ReDimNet2 + in-memory identity backend."""

    def __init__(
        self,
        *,
        redi_repository: str,
        redi_model_name: str,
        redi_train_type: str,
        redi_dataset: str,
        redi_match_threshold: float,
        redi_match_margin: float,
        redi_continue_threshold: float,
        redi_switch_threshold: float,
        redi_update_threshold: float,
        redi_min_update_quality: float,
        redi_min_confirm_embeddings: int,
        redi_min_confirm_seconds: float,
        redi_max_prototypes: int,
        redi_mongo_uri: str,
        **kwargs,
    ) -> None:
        self.redi_repository = redi_repository
        self.redi_model_name = redi_model_name
        self.redi_train_type = redi_train_type
        self.redi_dataset = redi_dataset
        self.redi_mongo_uri = redi_mongo_uri or os.environ.get(
            "MONGODB_URI",
            "mongodb://eurecat:cerdanyola@localhost:27017/?authSource=admin",
        )
        self._identity_options = {
            "match_threshold": redi_match_threshold,
            "match_margin": redi_match_margin,
            "continue_threshold": redi_continue_threshold,
            "switch_threshold": redi_switch_threshold,
            "update_threshold": redi_update_threshold,
            "min_update_quality": redi_min_update_quality,
            "min_confirm_embeddings": redi_min_confirm_embeddings,
            "min_confirm_seconds": redi_min_confirm_seconds,
            "max_prototypes": redi_max_prototypes,
        }
        super().__init__(**kwargs)

    @property
    def speaker_confidence(self) -> float:
        if isinstance(self.observer, RediDiarizationObserver):
            return self.observer.last_assignment_confidence
        return 0.0

    def initialize(self, sample_rate: int) -> bool:
        if sample_rate != 16000:
            self._logger.error(
                f"ReDimNet2 requires mono 16 kHz audio, received {sample_rate} Hz"
            )
            return False
        return super().initialize(sample_rate)

    def _create_embedding_model(self, hf_token: Optional[str]):
        del hf_token  # ReDimNet2 weights are public GitHub release assets.
        self._logger.info(
            "Loading ReDimNet2 embedding model: "
            f"{self.redi_model_name}/{self.redi_train_type}/{self.redi_dataset}"
        )
        adapter = ReDimNet2EmbeddingAdapter(
            repository=self.redi_repository,
            model_name=self.redi_model_name,
            train_type=self.redi_train_type,
            dataset=self.redi_dataset,
        )
        return diart_models.EmbeddingModel(lambda: adapter)

    def _create_observer(self) -> DiarizationObserver:
        store = None
        if self.use_database:
            model_key = (
                f"redimnet2:{self.redi_model_name}:"
                f"{self.redi_train_type}:{self.redi_dataset}:192"
            )
            try:
                store = MongoSpeakerIdentityStore(self.redi_mongo_uri, model_key)
            except Exception as error:
                self._logger.warn(
                    f"REDI MongoDB unavailable ({error}); using session-only identities"
                )

        manager = SpeakerIdentityManager(
            logger=self._logger,
            store=store,
            **self._identity_options,
        )
        return RediDiarizationObserver(
            identity_manager=manager,
            chunk_step_seconds=0.5,
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

    def stop(self) -> None:
        super().stop()
        if isinstance(self.observer, RediDiarizationObserver):
            self.observer.identity_manager.close()
