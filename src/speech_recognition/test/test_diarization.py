from unittest.mock import Mock

import numpy as np

from speech_recognition.redi_speaker_identity import SpeakerIdentity, SpeakerIdentityManager


class MemoryStore:
    def __init__(self, identities=None):
        self.identities = identities or []
        self.saved = []
        self.closed = False

    def load(self):
        return self.identities

    def save(self, identity):
        self.saved.append(identity.speaker_id)

    def close(self):
        self.closed = True


def vector(*values):
    result = np.asarray(values, dtype=np.float32)
    return result / np.linalg.norm(result)


def identity(speaker_id, embedding):
    return SpeakerIdentity(
        speaker_id=speaker_id,
        prototypes=[embedding.copy()],
        centroid=embedding.copy(),
        sample_count=5,
        clean_speech_seconds=5.0,
        confirmed=True,
    )


def manager(store=None, **kwargs):
    defaults = {
        "logger": Mock(),
        "store": store,
        "match_threshold": 0.70,
        "match_margin": 0.10,
        "continue_threshold": 0.55,
        "switch_threshold": 0.80,
        "update_threshold": 0.75,
        "min_update_quality": 0.70,
        "min_update_seconds": 0.40,
        "min_confirm_embeddings": 3,
        "min_confirm_seconds": 1.0,
        "persist_every": 2,
    }
    defaults.update(kwargs)
    return SpeakerIdentityManager(**defaults)


def test_loads_identities_and_matches_in_memory():
    known = identity("EUT_speaker7", vector(1.0, 0.0, 0.0))
    subject = manager(MemoryStore([known]))

    assignment = subject.assign(
        "track0",
        vector(0.99, 0.05, 0.0),
        speech_seconds=0.5,
        quality=0.9,
        overlapped=False,
    )

    assert assignment.speaker_id == "EUT_speaker7"
    assert assignment.state == "CONFIRMED"
    assert subject._next_speaker_number == 8


def test_ambiguous_top_two_scores_create_provisional_identity():
    store = MemoryStore(
        [
            identity("EUT_speaker1", vector(1.0, 0.0, 0.0)),
            identity("EUT_speaker2", vector(0.98, 0.20, 0.0)),
        ]
    )
    subject = manager(store)

    assignment = subject.assign(
        "new-track",
        vector(0.99, 0.10, 0.0),
        speech_seconds=0.5,
        quality=0.9,
        overlapped=False,
    )

    assert assignment.speaker_id == "EUT_speaker3"
    assert assignment.state == "PROVISIONAL"


def test_hysteresis_keeps_previous_identity_near_boundary():
    first = identity("EUT_speaker1", vector(1.0, 0.0, 0.0))
    second = identity("EUT_speaker2", vector(0.0, 1.0, 0.0))
    subject = manager(MemoryStore([first, second]))
    subject.track_assignments["track0"] = first.speaker_id

    assignment = subject.assign(
        "track0",
        vector(0.65, 0.76, 0.0),
        speech_seconds=0.5,
        quality=0.9,
        overlapped=False,
    )

    assert assignment.speaker_id == first.speaker_id


def test_overlap_does_not_poison_identity():
    known = identity("EUT_speaker1", vector(1.0, 0.0, 0.0))
    original_centroid = known.centroid.copy()
    subject = manager(MemoryStore([known]))

    assignment = subject.assign(
        "track0",
        vector(0.9, 0.4, 0.0),
        speech_seconds=0.5,
        quality=0.95,
        overlapped=True,
    )

    assert not assignment.accepted_for_update
    np.testing.assert_allclose(known.centroid, original_centroid)


def test_provisional_identity_is_confirmed_then_persisted():
    store = MemoryStore()
    subject = manager(store)
    embedding = vector(1.0, 0.0, 0.0)

    first = subject.assign(
        "track0", embedding, speech_seconds=0.5, quality=0.95, overlapped=False
    )
    second = subject.assign(
        "track0", embedding, speech_seconds=0.5, quality=0.95, overlapped=False
    )
    third = subject.assign(
        "track0", embedding, speech_seconds=0.5, quality=0.95, overlapped=False
    )

    assert first.state == "PROVISIONAL"
    assert second.state == "PROVISIONAL"
    assert third.state == "CONFIRMED"
    assert store.saved == ["EUT_speaker1"]
