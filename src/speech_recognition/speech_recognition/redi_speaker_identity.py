"""In-memory speaker identity clustering with optional MongoDB persistence.

ReDimNet2 produces embeddings, not diarization labels. This module turns normalized
embeddings into stable ``EUT_speakerN`` identities. MongoDB is deliberately kept
out of the real-time matching path: identities are loaded once, matched in RAM,
and only confirmed/dirty identities are checkpointed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass
class SpeakerIdentity:
    speaker_id: str
    prototypes: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    sample_count: int = 0
    clean_speech_seconds: float = 0.0
    confirmed: bool = False
    last_seen: float = 0.0
    accepted_since_save: int = 0


@dataclass(frozen=True)
class SpeakerAssignment:
    speaker_id: str
    confidence: float
    state: str
    accepted_for_update: bool


class SpeakerIdentityStore(Protocol):
    def load(self) -> List[SpeakerIdentity]: ...

    def save(self, identity: SpeakerIdentity) -> None: ...

    def close(self) -> None: ...


class MongoSpeakerIdentityStore:
    """Persistence adapter for ReDimNet2 identities.

    Documents are namespaced by ``model_key`` so 192-dimensional ReDimNet2
    embeddings are never compared with embeddings from the legacy pyannote model.
    """

    def __init__(
        self,
        mongo_uri: str,
        model_key: str,
        database_name: str = "speaker_recognition",
        collection_name: str = "speaker_identities",
    ) -> None:
        from pymongo import MongoClient

        self._model_key = model_key
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._collection = self._client[database_name][collection_name]
        self._collection.create_index(
            [("model_key", 1), ("speaker_id", 1)], unique=True
        )

    def load(self) -> List[SpeakerIdentity]:
        identities = []
        for document in self._collection.find({"model_key": self._model_key}):
            prototypes = [
                _normalize(np.asarray(item, dtype=np.float32))
                for item in document.get("prototypes", [])
            ]
            centroid_value = document.get("centroid")
            centroid = (
                _normalize(np.asarray(centroid_value, dtype=np.float32))
                if centroid_value is not None
                else None
            )
            if centroid is None and prototypes:
                centroid = _normalize(np.mean(np.stack(prototypes), axis=0))
            if centroid is None:
                continue
            identities.append(
                SpeakerIdentity(
                    speaker_id=document["speaker_id"],
                    prototypes=prototypes or [centroid.copy()],
                    centroid=centroid,
                    sample_count=int(document.get("sample_count", len(prototypes))),
                    clean_speech_seconds=float(document.get("clean_speech_seconds", 0.0)),
                    confirmed=True,
                    last_seen=float(document.get("last_seen", 0.0)),
                )
            )
        return identities

    def save(self, identity: SpeakerIdentity) -> None:
        if identity.centroid is None:
            return
        document = {
            "model_key": self._model_key,
            "speaker_id": identity.speaker_id,
            "centroid": identity.centroid.tolist(),
            "prototypes": [prototype.tolist() for prototype in identity.prototypes],
            "sample_count": identity.sample_count,
            "clean_speech_seconds": identity.clean_speech_seconds,
            "last_seen": identity.last_seen,
            "updated_at": time.time(),
        }
        self._collection.update_one(
            {"model_key": self._model_key, "speaker_id": identity.speaker_id},
            {"$set": document},
            upsert=True,
        )

    def close(self) -> None:
        self._client.close()


def _normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Speaker embedding must contain a finite, non-zero vector")
    return vector / norm


class SpeakerIdentityManager:
    """Online speaker assignment with margin checks, hysteresis and safe updates."""

    def __init__(
        self,
        *,
        logger,
        store: Optional[SpeakerIdentityStore] = None,
        match_threshold: float = 0.68,
        match_margin: float = 0.08,
        continue_threshold: float = 0.58,
        switch_threshold: float = 0.74,
        update_threshold: float = 0.72,
        min_update_quality: float = 0.70,
        min_update_seconds: float = 0.40,
        min_confirm_embeddings: int = 3,
        min_confirm_seconds: float = 1.5,
        max_prototypes: int = 6,
        persist_every: int = 5,
    ) -> None:
        self._logger = logger
        self._store = store
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.continue_threshold = continue_threshold
        self.switch_threshold = switch_threshold
        self.update_threshold = update_threshold
        self.min_update_quality = min_update_quality
        self.min_update_seconds = min_update_seconds
        self.min_confirm_embeddings = min_confirm_embeddings
        self.min_confirm_seconds = min_confirm_seconds
        self.max_prototypes = max(1, max_prototypes)
        self.persist_every = max(1, persist_every)

        self.identities: Dict[str, SpeakerIdentity] = {}
        self.track_assignments: Dict[str, str] = {}
        self._next_speaker_number = 1

        if self._store is not None:
            for identity in self._store.load():
                self.identities[identity.speaker_id] = identity
                self._next_speaker_number = max(
                    self._next_speaker_number,
                    self._speaker_number(identity.speaker_id) + 1,
                )
        self._logger.info(
            f"ReDimNet2 identity manager loaded {len(self.identities)} persistent identities"
        )

    @staticmethod
    def _speaker_number(speaker_id: str) -> int:
        match = re.search(r"(\d+)$", speaker_id)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _score(identity: SpeakerIdentity, embedding: np.ndarray) -> float:
        if identity.centroid is None:
            return -1.0
        centroid_score = float(np.dot(identity.centroid, embedding))
        prototype_score = max(
            (float(np.dot(prototype, embedding)) for prototype in identity.prototypes),
            default=centroid_score,
        )
        return 0.4 * centroid_score + 0.6 * prototype_score

    def _rank(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        scores = [
            (speaker_id, self._score(identity, embedding))
            for speaker_id, identity in self.identities.items()
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def assign(
        self,
        track_id: str,
        embedding: np.ndarray,
        *,
        speech_seconds: float,
        quality: float,
        overlapped: bool,
    ) -> SpeakerAssignment:
        vector = _normalize(embedding)
        ranked = self._rank(vector)
        best_id, best_score = ranked[0] if ranked else (None, -1.0)
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score

        previous_id = self.track_assignments.get(track_id)
        chosen_id: Optional[str] = None
        chosen_score = best_score

        if previous_id in self.identities:
            previous = self.identities[previous_id]
            previous_score = self._score(previous, vector)
            strong_switch = (
                best_id is not None
                and best_id != previous_id
                and best_score >= self.switch_threshold
                and best_score - previous_score >= self.match_margin
            )
            if not strong_switch and (
                previous_score >= self.continue_threshold or not previous.confirmed
            ):
                chosen_id = previous_id
                chosen_score = previous_score

        if chosen_id is None and best_id is not None:
            candidate = self.identities[best_id]
            # A provisional identity is backed by as little as one noisy
            # embedding. Holding it to the same strict bar as a confirmed,
            # multi-sample centroid means a physical speaker who alternates
            # quickly with someone else (so their identity never accumulates
            # enough samples to confirm) keeps failing the match and spawns a
            # new sibling identity every time diart hands out a new local
            # track for them. Confirmed identities keep the strict threshold
            # since merging two different real speakers is the costlier
            # mistake there.
            required_threshold = (
                self.match_threshold if candidate.confirmed else self.continue_threshold
            )
            if best_score >= required_threshold and margin >= self.match_margin:
                chosen_id = best_id
                chosen_score = best_score

        if chosen_id is None:
            chosen_id = f"EUT_speaker{self._next_speaker_number}"
            self._next_speaker_number += 1
            identity = SpeakerIdentity(
                speaker_id=chosen_id,
                prototypes=[vector.copy()],
                centroid=vector.copy(),
                sample_count=1,
                clean_speech_seconds=speech_seconds if not overlapped else 0.0,
                last_seen=time.time(),
            )
            self.identities[chosen_id] = identity
            self.track_assignments[track_id] = chosen_id
            return SpeakerAssignment(chosen_id, 1.0, "PROVISIONAL", True)

        self.track_assignments[track_id] = chosen_id
        identity = self.identities[chosen_id]
        accepted = self._may_update(
            identity,
            chosen_score,
            speech_seconds=speech_seconds,
            quality=quality,
            overlapped=overlapped,
        )
        if accepted:
            self._update(identity, vector, speech_seconds, quality)
        else:
            identity.last_seen = time.time()

        state = "CONFIRMED" if identity.confirmed else "PROVISIONAL"
        return SpeakerAssignment(chosen_id, max(0.0, chosen_score), state, accepted)

    def _may_update(
        self,
        identity: SpeakerIdentity,
        score: float,
        *,
        speech_seconds: float,
        quality: float,
        overlapped: bool,
    ) -> bool:
        threshold = self.update_threshold if identity.confirmed else self.continue_threshold
        return (
            not overlapped
            and speech_seconds >= self.min_update_seconds
            and quality >= self.min_update_quality
            and score >= threshold
        )

    def _update(
        self,
        identity: SpeakerIdentity,
        embedding: np.ndarray,
        speech_seconds: float,
        quality: float,
    ) -> None:
        identity.sample_count += 1
        identity.clean_speech_seconds += speech_seconds
        identity.last_seen = time.time()
        identity.accepted_since_save += 1

        # A bounded prototype bank preserves different acoustic conditions.
        max_similarity = max(
            (float(np.dot(item, embedding)) for item in identity.prototypes),
            default=-1.0,
        )
        if max_similarity < 0.97:
            identity.prototypes.append(embedding.copy())
            if len(identity.prototypes) > self.max_prototypes:
                identity.prototypes.pop(0)

        # Quality-weighted conservative centroid update. Re-normalize every time.
        alpha = min(0.20, max(0.03, 0.05 + 0.10 * quality))
        identity.centroid = _normalize((1.0 - alpha) * identity.centroid + alpha * embedding)

        was_confirmed = identity.confirmed
        identity.confirmed = (
            identity.sample_count >= self.min_confirm_embeddings
            and identity.clean_speech_seconds >= self.min_confirm_seconds
        )
        if identity.confirmed and (
            not was_confirmed or identity.accepted_since_save >= self.persist_every
        ):
            self._save(identity)

    def _save(self, identity: SpeakerIdentity) -> None:
        if self._store is None or not identity.confirmed:
            return
        self._store.save(identity)
        identity.accepted_since_save = 0

    def flush(self) -> None:
        if self._store is None:
            return
        for identity in self.identities.values():
            if identity.confirmed:
                self._save(identity)

    def close(self) -> None:
        self.flush()
        if self._store is not None:
            self._store.close()
