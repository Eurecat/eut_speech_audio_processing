"""The identity manager must not invent a speaker out of an ambiguous match.

A near-tie between two known voices means the speech belongs to one of them.
Creating a third identity there is the one answer that cannot be right, and it is
what produced ten speaker ids for a two-speaker CALLHOME call in the Sprint 4
benchmark: nine of the ten came from the 1 s speaker-change probe, six of those
after failing only the top1-top2 margin.
"""

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "speech_recognition")
)

from voice_identity_manager import VoiceIdentityManager, normalize_embedding  # noqa: E402


def manager(**kwargs):
    return VoiceIdentityManager(
        logger=logging.getLogger("test"),
        store=None,
        similarity_threshold=0.55,
        young_identity_threshold=0.40,
        match_margin=0.06,
        **kwargs,
    )


def seed(vm, vector, track):
    """Create an identity from `vector` and confirm it so the full bar applies."""
    result = vm.process_new_embedding_batch({track: vector}, speech_seconds=2.0, quality=1.0)
    unique_id = result[track][0]
    identity = vm.identities[unique_id]
    identity.confirmed = True
    return unique_id


def two_similar_identities():
    """Two known speakers placed symmetrically around a shared direction.

    Both share component 0 and differ on orthogonal components, so a probe along
    component 0 alone scores *identically* against the two of them: above the
    match bar, with a top1-top2 margin of zero. That is the real ambiguity the
    guard is about, and it cannot be produced by simply putting one vector near
    another. Their mutual similarity stays below `merge_threshold` so the two do
    not fold into one before the probe is tested.
    """
    vm = manager()
    # The offset is chosen so the two seeds score 0.37 against each other, below
    # the young match bar, and so the second seed really creates a second
    # identity instead of being absorbed by the first. With a smaller offset the
    # fixture silently collapses to one speaker and every test below passes
    # while proving nothing.
    offset = 1.3
    a = np.zeros(32, dtype=np.float32)
    a[0], a[1] = 1.0, offset
    b = np.zeros(32, dtype=np.float32)
    b[0], b[2] = 1.0, offset
    first = seed(vm, normalize_embedding(a), "t_a")
    second = seed(vm, normalize_embedding(b), "t_b")
    assert first != second, "fixture collapsed: the second seed matched the first"
    # Detach the tracks so stickiness cannot decide the next assignment for us.
    vm.track_to_identity.clear()
    return vm, first, second


def ambiguous_probe():
    """Scores ~0.61 against both identities: above the bar, margin ~0."""
    probe = np.zeros(32, dtype=np.float32)
    probe[0] = 1.0
    return probe


def test_the_fixture_really_is_ambiguous():
    """Guard the guard: without a genuine near-tie the other tests prove nothing."""
    vm, first, second = two_similar_identities()
    probe = ambiguous_probe()
    assert first != second, "there must be two distinct identities to be ambiguous between"
    assert len(vm.identities) == 2, f"expected exactly two identities, got {list(vm.identities)}"
    scores = sorted(vm.score(uid, probe) for uid in (first, second))
    assert scores[0] > vm.similarity_threshold, f"probe must clear the match bar: {scores}"
    assert scores[1] - scores[0] < vm.match_margin, f"probe must be a near-tie: {scores}"


def test_ambiguous_probe_creates_a_speaker_when_allowed():
    """The previous behaviour, kept reachable so the change can be A/B tested."""
    vm, first, second = two_similar_identities()
    before = set(vm.identities)

    result = vm.process_new_embedding_batch(
        {"probe": ambiguous_probe()},
        speech_seconds=1.0,
        learn=False,
        allow_create_when_ambiguous=True,
    )

    created = set(vm.identities) - before
    assert created, "expected the ambiguous probe to invent a third speaker"
    assert result["probe"][0] in created


def test_ambiguous_probe_is_left_unresolved_by_default():
    vm, first, second = two_similar_identities()
    before = set(vm.identities)

    result = vm.process_new_embedding_batch(
        {"probe": ambiguous_probe()},
        speech_seconds=1.0,
        learn=False,
        allow_create_when_ambiguous=False,
    )

    assert set(vm.identities) == before, "no third speaker may be invented from a near-tie"
    assert "probe" not in result, "an unresolved probe must not be reported as a speaker"


def test_a_genuinely_unknown_voice_still_creates_a_speaker():
    """The guard must only suppress near-ties, never a real new person.

    A voice that scores *below* the match bar is unlike everything known, so the
    probe path keeps its low-latency creation: that is the case it exists for.
    """
    vm, first, second = two_similar_identities()
    before = set(vm.identities)

    stranger = np.zeros(32, dtype=np.float32)
    stranger[7] = 1.0  # orthogonal to both seeds

    result = vm.process_new_embedding_batch(
        {"probe": stranger},
        speech_seconds=1.0,
        learn=False,
        allow_create_when_ambiguous=False,
    )

    created = set(vm.identities) - before
    assert created, "an unfamiliar voice must still be able to create a speaker"
    assert result["probe"][0] in created


def test_a_clear_match_is_unaffected_by_the_guard():
    vm, first, second = two_similar_identities()
    before = set(vm.identities)

    exact = vm.identities[first].mean_embedding.copy()
    result = vm.process_new_embedding_batch(
        {"probe": exact},
        speech_seconds=1.0,
        learn=False,
        allow_create_when_ambiguous=False,
    )

    assert set(vm.identities) == before
    assert result["probe"][0] == first


def test_transient_identities_never_reach_the_store():
    """A speaker created from one short probe is not written to MongoDB.

    This is what makes the transient hypotheses a ROS-visibility problem rather
    than database pollution: persistence needs either confirmation or two
    embeddings and `min_persist_seconds` of clean speech.
    """
    saved = []

    class RecordingStore:
        def load(self):
            return []

        def save(self, identity):
            saved.append(identity.unique_id)

        def delete(self, unique_id):
            pass

        def close(self):
            pass

    vm = VoiceIdentityManager(
        logger=logging.getLogger("test"), store=RecordingStore(), min_persist_seconds=3.0
    )
    vector = np.zeros(32, dtype=np.float32)
    vector[3] = 1.0
    vm.process_new_embedding_batch({"t": vector}, speech_seconds=1.0, quality=1.0)

    assert saved == [], "a one-second seed must not be persisted"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
