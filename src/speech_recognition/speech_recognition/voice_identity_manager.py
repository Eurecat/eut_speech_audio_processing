"""Persistent voice identity management, shared by every diarization backend.

Modelled on ``EutHRIFaces/face_recognition/face_recognition/identity_manager.py``:
embeddings arrive keyed by a transient track id, and this class decides which
persistent ``EUT_speakerN`` each one belongs to. It owns the speaker population,
the matching rules, merging, cleanup and MongoDB persistence.

It knows nothing about DIART, ReDimNet2, pyannote or ROS. A backend only has to
produce ``{track_id: embedding}`` and call :meth:`process_new_embedding_batch`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class VoiceIdentityCluster:
    """One persistent speaker: the population of embeddings plus its statistics."""

    unique_id: str
    creation_timestamp: float
    last_seen_timestamp: float

    all_embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_confidences: List[float] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None

    associated_track_ids: Set[str] = field(default_factory=set)
    current_track_id: Optional[str] = None

    total_detections: int = 0
    clean_speech_seconds: float = 0.0
    confirmed: bool = False
    quality_score: float = 0.0

    custom_name: Optional[str] = None
    unsaved_updates: int = 0


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Speaker embedding must be a finite, non-zero vector")
    return vector / norm


class MongoVoiceIdentityStore:
    """MongoDB persistence for :class:`VoiceIdentityCluster`.

    Documents are namespaced by ``model_key`` so embeddings from different
    models (e.g. 192-dim ReDimNet2 vs 512-dim pyannote) are never compared.
    """

    def __init__(
        self,
        mongo_uri: str,
        model_key: str,
        database_name: str = "speaker_recognition",
        collection_name: str = "voice_identities",
        save_last_n_embeddings: int = 20,
    ) -> None:
        from pymongo import MongoClient

        self._model_key = model_key
        self._save_last_n = max(1, save_last_n_embeddings)
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._collection = self._client[database_name][collection_name]
        self._collection.create_index([("model_key", 1), ("unique_id", 1)], unique=True)

    def load(self) -> List[VoiceIdentityCluster]:
        identities = []
        for document in self._collection.find({"model_key": self._model_key}):
            embeddings = [
                normalize_embedding(np.asarray(item, dtype=np.float32))
                for item in document.get("embeddings", [])
            ]
            mean = document.get("mean_embedding")
            if mean is None and not embeddings:
                continue
            mean_vector = (
                normalize_embedding(np.asarray(mean, dtype=np.float32))
                if mean is not None
                else normalize_embedding(np.mean(np.stack(embeddings), axis=0))
            )
            identities.append(
                VoiceIdentityCluster(
                    unique_id=document["unique_id"],
                    creation_timestamp=float(document.get("creation_timestamp", 0.0)),
                    last_seen_timestamp=float(document.get("last_seen_timestamp", 0.0)),
                    all_embeddings=embeddings or [mean_vector.copy()],
                    embedding_confidences=[
                        float(c) for c in document.get("embedding_confidences", [])
                    ]
                    or [1.0] * max(1, len(embeddings)),
                    mean_embedding=mean_vector,
                    total_detections=int(document.get("total_detections", 0)),
                    clean_speech_seconds=float(document.get("clean_speech_seconds", 0.0)),
                    confirmed=True,
                    quality_score=float(document.get("quality_score", 0.0)),
                    custom_name=document.get("custom_name"),
                )
            )
        return identities

    def save(self, identity: VoiceIdentityCluster) -> None:
        if identity.mean_embedding is None:
            return
        recent = identity.all_embeddings[-self._save_last_n :]
        self._collection.update_one(
            {"model_key": self._model_key, "unique_id": identity.unique_id},
            {
                "$set": {
                    "model_key": self._model_key,
                    "unique_id": identity.unique_id,
                    "creation_timestamp": float(identity.creation_timestamp),
                    "last_seen_timestamp": float(identity.last_seen_timestamp),
                    "total_detections": int(identity.total_detections),
                    "clean_speech_seconds": float(identity.clean_speech_seconds),
                    "quality_score": float(identity.quality_score),
                    "custom_name": identity.custom_name,
                    "embeddings": [e.astype(float).tolist() for e in recent],
                    "embedding_confidences": [
                        float(c) for c in identity.embedding_confidences[-self._save_last_n :]
                    ],
                    "mean_embedding": identity.mean_embedding.astype(float).tolist(),
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )

    def delete(self, unique_id: str) -> None:
        self._collection.delete_one({"model_key": self._model_key, "unique_id": unique_id})

    def close(self) -> None:
        self._client.close()


class VoiceIdentityManager:
    """Assign transient tracks to persistent speakers.

    The matching rules are deliberately stricter than a plain cosine threshold:

    * an absolute score **and** a top1-top2 margin must both be satisfied, so a
      near-tie between two speakers never silently picks one;
    * assignment across a batch is exclusive, so two tracks that are active at
      the same moment can never collapse onto one speaker;
    * a track stays on its previous speaker unless another beats it by
      ``stickiness_margin``, which absorbs frame-to-frame noise;
    * an identity is only updated from audio that is long, clean and confidently
      matched, because a single bad update is permanent.
    """

    def __init__(
        self,
        *,
        logger,
        store=None,
        similarity_threshold: float = 0.55,
        young_identity_threshold: float = 0.40,
        match_margin: float = 0.06,
        stickiness_margin: float = 0.25,
        merge_threshold: float = 0.80,
        update_threshold: float = 0.55,
        min_update_quality: float = 0.70,
        min_update_seconds: float = 0.40,
        min_confirm_embeddings: int = 3,
        min_confirm_seconds: float = 1.5,
        max_embeddings_per_identity: int = 50,
        min_embeddings_for_merge: int = 4,
        identity_timeout: float = 60.0,
        min_embeddings_for_identity: int = 3,
        use_ewma_for_mean: bool = False,
        ewma_alpha: float = 0.6,
        persist_every: int = 5,
    ) -> None:
        self._logger = logger
        self._store = store
        self.similarity_threshold = similarity_threshold
        self.young_identity_threshold = min(young_identity_threshold, similarity_threshold)
        self.match_margin = match_margin
        self.stickiness_margin = stickiness_margin
        self.merge_threshold = merge_threshold
        self.update_threshold = update_threshold
        self.min_update_quality = min_update_quality
        self.min_update_seconds = min_update_seconds
        self.min_confirm_embeddings = min_confirm_embeddings
        self.min_confirm_seconds = min_confirm_seconds
        self.max_embeddings_per_identity = max(1, max_embeddings_per_identity)
        self.min_embeddings_for_merge = min_embeddings_for_merge
        self.identity_timeout = identity_timeout
        self.min_embeddings_for_identity = min_embeddings_for_identity
        self.use_ewma_for_mean = use_ewma_for_mean
        self.ewma_alpha = min(0.99, max(0.01, ewma_alpha))
        self.persist_every = max(1, persist_every)

        self.identities: Dict[str, VoiceIdentityCluster] = {}
        self.track_to_identity: Dict[str, str] = {}
        self._next_speaker_number = 1
        self._last_rejection: Dict[str, Tuple[str, float, float, float]] = {}

        self.total_created = 0
        self.total_merges = 0

        if self._store is not None:
            for identity in self._store.load():
                self.identities[identity.unique_id] = identity
                self._next_speaker_number = max(
                    self._next_speaker_number, self._speaker_number(identity.unique_id) + 1
                )
        self._logger.info(
            f"Voice identity manager ready with {len(self.identities)} persistent identities"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_new_embedding_batch(
        self,
        track_embeddings: Dict[str, np.ndarray],
        *,
        speech_seconds: Dict[str, float] | float = 0.0,
        quality: Dict[str, float] | float = 1.0,
        overlapped: bool = False,
        learn: bool = True,
        allow_create: bool = True,
        learn_if_assigned_to: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Tuple[str, float]]:
        """Assign every track in this batch to a speaker.

        Returns ``{track_id: (EUT_speakerN, confidence)}``.

        ``speech_seconds`` and ``quality`` may be a single value applied to every
        track, or a per-track mapping. ``overlapped`` marks the whole batch as
        containing simultaneous speech, which blocks identity updates.
        ``learn=False`` matches without adding the embeddings to any identity, for
        provisional observations that will be followed by a better one.
        ``allow_create=False`` leaves an unmatched track out of the result instead of
        creating a speaker from it, for observations too short to seed a reliable one.
        ``learn_if_assigned_to`` restricts learning for a track to the case where it
        was assigned to the given identity, so a window that crossed a speaker
        change is never learned into whoever it happened to land on.
        """
        if not track_embeddings:
            return {}

        self._merge_similar_identities()
        self._absorb_stray_identities()

        vectors: Dict[str, np.ndarray] = {}
        for track_id, embedding in track_embeddings.items():
            try:
                vectors[track_id] = normalize_embedding(embedding)
            except ValueError:
                continue  # a silent or masked-out track produces no usable vector
        if not vectors:
            return {}

        track_ids = list(vectors)
        self._last_rejection.clear()
        assignments = self._assign_batch(track_ids, vectors)

        now = time.time()
        results: Dict[str, Tuple[str, float]] = {}
        for track_id in track_ids:
            unique_id, score = assignments[track_id]
            vector = vectors[track_id]
            track_seconds = self._per_track(speech_seconds, track_id)
            track_quality = self._per_track(quality, track_id)

            if unique_id is None:
                if not allow_create:
                    self._last_rejection.pop(track_id, None)
                    continue
                unique_id = self._create_identity(track_id, vector, track_seconds, now)
                results[track_id] = (unique_id, 1.0)
                continue

            identity = self.identities[unique_id]
            self.track_to_identity[track_id] = unique_id
            identity.associated_track_ids.add(track_id)
            identity.current_track_id = track_id
            identity.last_seen_timestamp = now
            identity.total_detections += 1

            expected = (learn_if_assigned_to or {}).get(track_id)
            allowed = learn and (expected is None or expected == unique_id)
            if allowed and self._may_update(identity, score, track_seconds, track_quality, overlapped):
                self._add_embedding(identity, vector, score, track_seconds)
            results[track_id] = (unique_id, max(0.0, score))

        self.cleanup_inactive_identities()
        return results

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _assign_batch(
        self, track_ids: Sequence[str], vectors: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[Optional[str], float]]:
        """Exclusive assignment of tracks to identities, best matches first."""
        result: Dict[str, Tuple[Optional[str], float]] = {
            track_id: (None, 0.0) for track_id in track_ids
        }
        identity_ids = [
            unique_id
            for unique_id, identity in self.identities.items()
            if identity.mean_embedding is not None
        ]
        if not identity_ids:
            return result

        representations = np.stack(
            [self._representation(self.identities[uid]) for uid in identity_ids]
        )
        queries = np.stack([vectors[track_id] for track_id in track_ids])
        similarity = queries @ representations.T  # both sides are L2-normalised

        # Resolve the most confident tracks first so a strong match claims its
        # speaker before a weaker, more ambiguous track can take it.
        order = sorted(
            range(len(track_ids)), key=lambda i: float(np.max(similarity[i])), reverse=True
        )
        claimed: Set[int] = set()

        for row in order:
            track_id = track_ids[row]
            scores = similarity[row]
            available = [j for j in range(len(identity_ids)) if j not in claimed]
            if not available:
                break

            ranked = sorted(available, key=lambda j: float(scores[j]), reverse=True)
            best = ranked[0]
            best_score = float(scores[best])
            second_score = float(scores[ranked[1]]) if len(ranked) > 1 else -1.0
            margin = best_score - second_score

            chosen = None
            previous = self.track_to_identity.get(track_id)
            if previous in identity_ids:
                previous_index = identity_ids.index(previous)
                if previous_index not in claimed:
                    previous_score = float(scores[previous_index])
                    # Stay on the previous speaker unless another one is clearly
                    # better. The floor matters when the speaker changes inside one
                    # track and the newcomer is unknown: the previous identity is
                    # then also the best candidate, so the margin alone would keep
                    # the track pinned to it no matter how low it scores.
                    if (
                        previous_score >= self.young_identity_threshold
                        and previous_score >= best_score - self.stickiness_margin
                    ):
                        chosen, best_score = previous_index, previous_score

            required = self._required_score(self.identities[identity_ids[best]])
            if chosen is None and best_score >= required:
                # A near-tie means we cannot tell the speakers apart; treating
                # that as a match is how two people end up sharing an identity.
                if margin >= self.match_margin or len(ranked) == 1:
                    chosen = best

            if chosen is not None:
                claimed.add(chosen)
                result[track_id] = (identity_ids[chosen], best_score)
            else:
                # Kept so a new identity can report *why* it was created; logging
                # only successful matches hides the scores that need calibrating.
                self._last_rejection[track_id] = (identity_ids[best], best_score, margin, required)
        return result

    def _required_score(self, identity: VoiceIdentityCluster) -> float:
        """Minimum similarity to join an identity, relaxed while it is young.

        A new identity's mean is a single noisy utterance, so a second utterance
        of the same speaker scores only the pairwise similarity against it, well
        below what it would score against a converged mean. Holding young
        identities to the full threshold means the same speaker keeps spawning
        fresh identities that each stay too thin to ever be merged. The top1-top2
        margin still applies, which is what stops two different people merging.
        """
        if identity.confirmed:
            return self.similarity_threshold
        return self.young_identity_threshold

    def _representation(self, identity: VoiceIdentityCluster) -> np.ndarray:
        """Blend the mean with recent history so one drifting vector cannot define a speaker."""
        mean = identity.mean_embedding
        if len(identity.all_embeddings) < 4:
            return mean
        recent = np.mean(np.stack(identity.all_embeddings[-10:]), axis=0)
        return normalize_embedding(0.6 * mean + 0.4 * recent)

    # ------------------------------------------------------------------
    # Identity lifecycle
    # ------------------------------------------------------------------

    def _create_identity(
        self, track_id: str, vector: np.ndarray, speech_seconds: float, now: float
    ) -> str:
        unique_id = f"EUT_speaker{self._next_speaker_number}"
        self._next_speaker_number += 1
        identity = VoiceIdentityCluster(
            unique_id=unique_id,
            creation_timestamp=now,
            last_seen_timestamp=now,
            all_embeddings=[vector.copy()],
            embedding_confidences=[1.0],
            mean_embedding=vector.copy(),
            associated_track_ids={track_id},
            current_track_id=track_id,
            total_detections=1,
            clean_speech_seconds=speech_seconds,
        )
        self.identities[unique_id] = identity
        self.track_to_identity[track_id] = unique_id
        self.total_created += 1
        rejected = self._last_rejection.pop(track_id, None)
        if rejected is None:
            self._logger.info(f"New voice identity {unique_id} (track {track_id}, no candidates)")
        else:
            nearest, score, margin, required = rejected
            self._logger.info(
                f"New voice identity {unique_id} (track {track_id}): nearest={nearest} "
                f"score={score:.3f} required={required:.3f} margin={margin:.3f}"
            )
        return unique_id

    def _may_update(
        self,
        identity: VoiceIdentityCluster,
        score: float,
        speech_seconds: float,
        quality: float,
        overlapped: bool,
    ) -> bool:
        # Learn at the same bar used to match. A stricter update bar starves the
        # identity: it keeps being matched but never gains samples, so its mean
        # never converges and later utterances keep failing to match it. The
        # margin, duration, quality and overlap gates are what prevent poisoning.
        threshold = self.update_threshold if identity.confirmed else self.young_identity_threshold
        return (
            not overlapped
            and speech_seconds >= self.min_update_seconds
            and quality >= self.min_update_quality
            and score >= threshold
        )

    def _add_embedding(
        self,
        identity: VoiceIdentityCluster,
        vector: np.ndarray,
        confidence: float,
        speech_seconds: float,
    ) -> None:
        identity.all_embeddings.append(vector.copy())
        identity.embedding_confidences.append(float(confidence))
        if len(identity.all_embeddings) > self.max_embeddings_per_identity:
            identity.all_embeddings.pop(0)
            identity.embedding_confidences.pop(0)

        if self.use_ewma_for_mean and identity.mean_embedding is not None:
            blended = (
                self.ewma_alpha * vector + (1.0 - self.ewma_alpha) * identity.mean_embedding
            )
            identity.mean_embedding = normalize_embedding(blended)
        else:
            identity.mean_embedding = normalize_embedding(
                np.mean(np.stack(identity.all_embeddings), axis=0)
            )

        identity.clean_speech_seconds += speech_seconds
        identity.unsaved_updates += 1
        identity.quality_score = self._quality_score(identity)

        was_confirmed = identity.confirmed
        identity.confirmed = (
            len(identity.all_embeddings) >= self.min_confirm_embeddings
            and identity.clean_speech_seconds >= self.min_confirm_seconds
        )
        if identity.confirmed and (
            not was_confirmed or identity.unsaved_updates >= self.persist_every
        ):
            self._save(identity)

    def _quality_score(self, identity: VoiceIdentityCluster) -> float:
        if len(identity.all_embeddings) < 2 or identity.mean_embedding is None:
            return 0.0
        recent = np.stack(identity.all_embeddings[-10:])
        consistency = float(np.mean(recent @ identity.mean_embedding))
        population = min(len(identity.all_embeddings) / float(self.max_embeddings_per_identity), 1.0)
        return 0.7 * consistency + 0.3 * population

    # ------------------------------------------------------------------
    # Merging and cleanup
    # ------------------------------------------------------------------

    def _merge_similar_identities(self) -> None:
        """Fold together identities that turned out to be the same person.

        Fragmentation is otherwise permanent: a speaker who spawned two
        identities early, before either had enough samples to match reliably,
        would keep both forever.
        """
        candidates = [
            unique_id
            for unique_id, identity in self.identities.items()
            if identity.mean_embedding is not None
            and len(identity.all_embeddings) >= self.min_embeddings_for_merge
        ]
        if len(candidates) < 2:
            return

        matrix = np.stack([self._representation(self.identities[uid]) for uid in candidates])
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -1.0)

        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if similarity[i, j] >= self.merge_threshold:
                    pairs.append((float(similarity[i, j]), candidates[i], candidates[j]))
        if not pairs:
            return

        pairs.sort(reverse=True)
        merged: Set[str] = set()
        for score, first, second in pairs:
            if first in merged or second in merged:
                continue
            if first not in self.identities or second not in self.identities:
                continue
            # Always keep the older speaker number so ids stay stable for anyone
            # who already saw them.
            keep, drop = sorted(
                (first, second), key=lambda uid: self._speaker_number(uid)
            )
            self.merge_identities(keep, drop)
            merged.add(drop)
            self._logger.info(f"Merged {drop} into {keep} (similarity {score:.3f})")

    def _absorb_stray_identities(self) -> None:
        """Fold young identities into a speaker that has since matured.

        A noisy utterance can miss its true speaker while that speaker's mean is
        still built from one or two samples, and so spawns a stray single-sample
        identity. Strays are too thin for :meth:`_merge_similar_identities`,
        which only compares well-populated identities, so without this they
        persist forever. Once the true speaker has converged, the stray's vector
        scores much higher against it, so it is re-checked here using the normal
        full match threshold and margin rather than a looser rule.
        """
        mature = [
            uid
            for uid, identity in self.identities.items()
            if identity.confirmed and identity.mean_embedding is not None
        ]
        if not mature:
            return
        young = [
            uid
            for uid, identity in self.identities.items()
            if not identity.confirmed and identity.mean_embedding is not None
        ]
        if not young:
            return

        references = np.stack([self._representation(self.identities[uid]) for uid in mature])
        for uid in young:
            if uid not in self.identities:
                continue
            scores = references @ self.identities[uid].mean_embedding
            order = np.argsort(scores)[::-1]
            best_score = float(scores[order[0]])
            second_score = float(scores[order[1]]) if len(order) > 1 else -1.0
            if best_score < self.similarity_threshold:
                continue
            if len(order) > 1 and best_score - second_score < self.match_margin:
                continue  # ambiguous between two known speakers: leave it alone
            keep = mature[int(order[0])]
            self.merge_identities(keep, uid)
            self._logger.info(f"Absorbed stray {uid} into {keep} (similarity {best_score:.3f})")

    def merge_identities(self, keep_id: str, drop_id: str) -> bool:
        if keep_id not in self.identities or drop_id not in self.identities:
            return False
        keep = self.identities[keep_id]
        drop = self.identities[drop_id]

        keep.all_embeddings.extend(drop.all_embeddings)
        keep.embedding_confidences.extend(drop.embedding_confidences)
        if len(keep.all_embeddings) > self.max_embeddings_per_identity:
            keep.all_embeddings = keep.all_embeddings[-self.max_embeddings_per_identity :]
            keep.embedding_confidences = keep.embedding_confidences[
                -self.max_embeddings_per_identity :
            ]
        keep.mean_embedding = normalize_embedding(
            np.mean(np.stack(keep.all_embeddings), axis=0)
        )
        keep.associated_track_ids.update(drop.associated_track_ids)
        keep.total_detections += drop.total_detections
        keep.clean_speech_seconds += drop.clean_speech_seconds
        keep.creation_timestamp = min(keep.creation_timestamp, drop.creation_timestamp)
        keep.last_seen_timestamp = max(keep.last_seen_timestamp, drop.last_seen_timestamp)
        keep.confirmed = keep.confirmed or drop.confirmed
        keep.quality_score = self._quality_score(keep)
        keep.unsaved_updates += 1

        for track_id, unique_id in list(self.track_to_identity.items()):
            if unique_id == drop_id:
                self.track_to_identity[track_id] = keep_id

        del self.identities[drop_id]
        if self._store is not None:
            self._store.delete(drop_id)
        self._save(keep)
        self.total_merges += 1
        return True

    def cleanup_inactive_track_mappings(self, active_track_ids: Set[str]) -> None:
        """Forget mappings for tracks that are gone.

        A backend may reuse a track id for a different person later; without
        this, the new speaker would silently inherit the old one's identity.
        """
        for track_id in [t for t in self.track_to_identity if t not in active_track_ids]:
            unique_id = self.track_to_identity.pop(track_id)
            identity = self.identities.get(unique_id)
            if identity is not None and identity.current_track_id == track_id:
                identity.current_track_id = None

    def cleanup_inactive_identities(self) -> None:
        """Drop identities that never gathered enough evidence and went quiet."""
        now = time.time()
        for unique_id, identity in list(self.identities.items()):
            stale = now - identity.last_seen_timestamp > self.identity_timeout
            thin = len(identity.all_embeddings) < self.min_embeddings_for_identity
            if stale and thin and not identity.confirmed:
                del self.identities[unique_id]
                for track_id, mapped in list(self.track_to_identity.items()):
                    if mapped == unique_id:
                        del self.track_to_identity[track_id]

    # ------------------------------------------------------------------
    # Persistence and helpers
    # ------------------------------------------------------------------

    def _save(self, identity: VoiceIdentityCluster) -> None:
        if self._store is None or not identity.confirmed:
            return
        self._store.save(identity)
        identity.unsaved_updates = 0

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

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_identities": len(self.identities),
            "total_created": self.total_created,
            "total_merges": self.total_merges,
            "active_tracks": len(self.track_to_identity),
        }

    @staticmethod
    def _per_track(value: Dict[str, float] | float, track_id: str) -> float:
        if isinstance(value, dict):
            return float(value.get(track_id, 0.0))
        return float(value)

    @staticmethod
    def _speaker_number(unique_id: str) -> int:
        match = re.search(r"(\d+)$", unique_id)
        return int(match.group(1)) if match else 0
