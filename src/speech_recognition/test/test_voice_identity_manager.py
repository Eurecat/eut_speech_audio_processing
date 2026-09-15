"""Unit tests for VoiceIdentityManager, using synthetic embeddings only.

Each speaker is a fixed random direction in 192-dim space; an "utterance" is that
direction plus Gaussian noise. ``noise`` controls how far same-speaker
utterances drift apart, which is the property that broke the DIART-coupled
approach.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "speech_recognition" / "voice_identity_manager.py"
)
_spec = importlib.util.spec_from_file_location("voice_identity_manager", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(_module)
VoiceIdentityManager = _module.VoiceIdentityManager

DIM = 192


class _Store:
    def __init__(self):
        self.saved = {}
        self.deleted = []

    def load(self):
        return []

    def save(self, identity):
        self.saved[identity.unique_id] = identity

    def delete(self, unique_id):
        self.deleted.append(unique_id)

    def close(self):
        pass


def _speaker(rng):
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def _utterance(rng, speaker, noise):
    v = speaker + rng.normal(scale=noise, size=DIM)
    return v / np.linalg.norm(v)


def _manager(**overrides):
    return VoiceIdentityManager(logger=logging.getLogger("test"), **overrides)


def _feed(manager, track_id, embedding):
    return manager.process_new_embedding_batch(
        {track_id: embedding}, speech_seconds=1.0, quality=1.0
    )[track_id][0]


def test_alternating_speakers_get_two_stable_identities():
    rng = np.random.default_rng(0)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()

    labels = []
    for turn in range(20):
        speaker = a if turn % 2 == 0 else b
        labels.append(_feed(manager, f"turn{turn}", _utterance(rng, speaker, noise=0.05)))

    assert len(set(labels)) == 2
    assert len(set(labels[0::2])) == 1, "all of A's turns must share one identity"
    assert len(set(labels[1::2])) == 1, "all of B's turns must share one identity"
    assert labels[0] != labels[1]


def test_short_interjection_is_not_absorbed_by_the_dominant_speaker():
    """The GT case DIART got wrong: B, B, A, B — A must keep its own identity."""
    rng = np.random.default_rng(1)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()

    sequence = [a, b, b, a, b, b, b]
    labels = [
        _feed(manager, f"turn{i}", _utterance(rng, s, noise=0.05))
        for i, s in enumerate(sequence)
    ]

    assert labels[0] == labels[3], "A's two turns must match"
    assert labels[3] != labels[1], "A's interjection must not take B's identity"
    assert len({labels[i] for i in (1, 2, 4, 5, 6)}) == 1, "B stays one identity"


def test_simultaneous_tracks_never_share_an_identity():
    rng = np.random.default_rng(2)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()
    _feed(manager, "seed_a", _utterance(rng, a, 0.05))
    _feed(manager, "seed_b", _utterance(rng, b, 0.05))

    for step in range(10):
        result = manager.process_new_embedding_batch(
            {"x": _utterance(rng, a, 0.05), "y": _utterance(rng, a, 0.05)},
            speech_seconds=1.0,
            quality=1.0,
        )
        assert result["x"][0] != result["y"][0], "exclusive assignment violated"


def test_fragmented_speaker_is_merged_back():
    rng = np.random.default_rng(3)
    a = _speaker(rng)
    manager = _manager(
        similarity_threshold=0.99,
        young_identity_threshold=0.99,
        merge_threshold=0.80,
        min_embeddings_for_merge=4,
    )

    # Impossible match thresholds force every early turn to spawn a new
    # identity, reproducing the fragmentation seen on real audio.
    for i in range(6):
        _feed(manager, f"early{i}", _utterance(rng, a, 0.03))
    assert len(manager.identities) > 1

    manager.similarity_threshold = 0.55
    manager.young_identity_threshold = 0.40
    for i in range(30):
        _feed(manager, f"early{i % 6}", _utterance(rng, a, 0.03))

    assert len(manager.identities) == 1, manager.get_statistics()


def test_near_tie_does_not_silently_pick_a_speaker():
    manager = _manager(similarity_threshold=0.5, match_margin=0.06)
    rng = np.random.default_rng(4)
    a, b = _speaker(rng), _speaker(rng)
    _feed(manager, "a", a)
    _feed(manager, "b", b)

    between = (a + b) / np.linalg.norm(a + b)  # equidistant from both
    assigned = _feed(manager, "ambiguous", between)
    assert assigned not in {manager.track_to_identity["a"], manager.track_to_identity["b"]}


def test_overlapped_audio_never_updates_an_identity():
    rng = np.random.default_rng(5)
    a = _speaker(rng)
    manager = _manager()
    uid = _feed(manager, "t", _utterance(rng, a, 0.05))
    before = len(manager.identities[uid].all_embeddings)

    manager.process_new_embedding_batch(
        {"t": _utterance(rng, a, 0.05)}, speech_seconds=1.0, quality=1.0, overlapped=True
    )
    assert len(manager.identities[uid].all_embeddings) == before


def test_learn_false_matches_without_growing_the_identity():
    rng = np.random.default_rng(9)
    a = _speaker(rng)
    manager = _manager()
    uid = _feed(manager, "t", _utterance(rng, a, 0.05))
    before = len(manager.identities[uid].all_embeddings)

    result = manager.process_new_embedding_batch(
        {"t": _utterance(rng, a, 0.05)}, speech_seconds=1.0, quality=1.0, learn=False
    )
    assert result["t"][0] == uid
    assert len(manager.identities[uid].all_embeddings) == before


def test_speaker_change_within_one_track_escapes_stickiness():
    """One VAD turn with no pause: A talks, then B takes over on the same track."""
    rng = np.random.default_rng(11)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()
    for _ in range(4):
        _feed(manager, "other_turn_a", _utterance(rng, a, 0.05))
    _feed(manager, "other_turn_b", _utterance(rng, b, 0.05))

    uid_a = _feed(manager, "turn", _utterance(rng, a, 0.05))
    uid_after_change = _feed(manager, "turn", _utterance(rng, b, 0.05))
    assert uid_after_change != uid_a, "stickiness pinned a track after its speaker changed"


def test_new_speaker_within_one_track_gets_a_new_identity():
    rng = np.random.default_rng(12)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()
    uid_a = _feed(manager, "turn", _utterance(rng, a, 0.05))
    uid_b = _feed(manager, "turn", _utterance(rng, b, 0.05))  # B never heard before
    assert uid_b != uid_a


def test_allow_create_false_defers_instead_of_inventing_a_speaker():
    rng = np.random.default_rng(13)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()
    _feed(manager, "seed", _utterance(rng, a, 0.05))

    result = manager.process_new_embedding_batch(
        {"short": _utterance(rng, b, 0.05)}, speech_seconds=0.8, quality=1.0, allow_create=False
    )
    assert "short" not in result
    assert len(manager.identities) == 1


def test_allow_create_false_still_matches_a_known_speaker():
    rng = np.random.default_rng(14)
    a = _speaker(rng)
    manager = _manager()
    uid = _feed(manager, "seed", _utterance(rng, a, 0.05))
    result = manager.process_new_embedding_batch(
        {"short": _utterance(rng, a, 0.05)}, speech_seconds=0.8, quality=1.0, allow_create=False
    )
    assert result["short"][0] == uid


def test_learning_is_skipped_when_assigned_to_an_unexpected_speaker():
    rng = np.random.default_rng(15)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager()
    uid_a = _feed(manager, "ta", _utterance(rng, a, 0.05))
    uid_b = _feed(manager, "tb", _utterance(rng, b, 0.05))
    sizes = {u: len(i.all_embeddings) for u, i in manager.identities.items()}

    result = manager.process_new_embedding_batch(
        {"t": _utterance(rng, b, 0.05)},
        speech_seconds=1.0,
        quality=1.0,
        learn_if_assigned_to={"t": uid_a},
    )
    assert result["t"][0] == uid_b
    assert {u: len(i.all_embeddings) for u, i in manager.identities.items()} == sizes


def test_track_mapping_cleanup_prevents_identity_inheritance():
    rng = np.random.default_rng(6)
    a, b = _speaker(rng), _speaker(rng)
    manager = _manager(stickiness_margin=1.0)  # maximal stickiness
    uid_a = _feed(manager, "reused", _utterance(rng, a, 0.05))

    manager.cleanup_inactive_track_mappings(active_track_ids=set())
    uid_b = _feed(manager, "reused", _utterance(rng, b, 0.05))
    assert uid_b != uid_a, "a reused track id must not inherit the old speaker"


def test_confirmed_identity_is_persisted_and_merge_deletes_loser():
    rng = np.random.default_rng(7)
    a = _speaker(rng)
    store = _Store()
    manager = _manager(store=store, min_confirm_embeddings=3, min_confirm_seconds=1.5)
    for i in range(5):
        _feed(manager, "t", _utterance(rng, a, 0.05))
    assert store.saved, "confirmed identity should be written to the store"


def test_persisted_identities_are_loaded_and_matched_on_startup():
    rng = np.random.default_rng(10)
    a = _speaker(rng)
    VoiceIdentityCluster = _module.VoiceIdentityCluster

    class _LoadedStore(_Store):
        def load(self):
            return [
                VoiceIdentityCluster(
                    unique_id="EUT_speaker7",
                    creation_timestamp=0.0,
                    last_seen_timestamp=0.0,
                    all_embeddings=[a.copy()],
                    embedding_confidences=[1.0],
                    mean_embedding=a.copy(),
                    confirmed=True,
                )
            ]

    manager = _manager(store=_LoadedStore())
    assert _feed(manager, "t", _utterance(rng, a, 0.05)) == "EUT_speaker7"
    b = _speaker(rng)
    assert _feed(manager, "u", _utterance(rng, b, 0.05)) == "EUT_speaker8", (
        "new ids must continue after the highest persisted number"
    )


def _run_conversation(noise, seed=8, turns=60, speakers=3):
    rng = np.random.default_rng(seed)
    voices = [_speaker(rng) for _ in range(speakers)]
    manager = _manager()
    truth, live = [], []
    for i in range(turns):
        s = int(rng.integers(0, speakers))
        truth.append(s)
        live.append(_feed(manager, f"turn{i}", _utterance(rng, voices[s], noise)))
    return manager, truth, live


def _purity(truth, labels, speakers=3):
    worst = 1.0
    for s in range(speakers):
        assigned = [labels[i] for i in range(len(truth)) if truth[i] == s]
        dominant = max(set(assigned), key=assigned.count)
        worst = min(worst, assigned.count(dominant) / len(assigned))
    return worst


@pytest.mark.parametrize(
    "noise",
    [
        0.05,  # same-speaker pair similarity ~0.68 — matches measured ReDimNet2 on the mp3
        0.08,  # pair ~0.46 — noisier than measured
        pytest.param(
            0.11,
            marks=pytest.mark.xfail(
                reason="pair similarity ~0.29, worse than any measured ReDimNet2 audio; "
                "a stress case, deliberately not a calibration target",
                strict=False,
            ),
        ),
    ],
)
def test_conversation_converges_to_the_true_speakers(noise):
    """After the conversation every turn resolves, through merges, to its speaker."""
    manager, truth, _ = _run_conversation(noise)

    assert len(manager.identities) == 3, manager.get_statistics()
    final = [manager.track_to_identity[f"turn{i}"] for i in range(len(truth))]
    assert _purity(truth, final) == 1.0


def test_live_labels_are_exact_on_realistic_audio():
    """What /speech_result publishes in the moment, at measured ReDimNet2 quality."""
    _, truth, live = _run_conversation(noise=0.05)
    assert _purity(truth, live) == 1.0


def test_live_labels_on_noisy_audio_leak_only_transient_strays():
    """Noisier than measured audio: some turns are published under a stray id.

    Each stray is absorbed into its true speaker once that speaker's mean has
    converged (covered by the convergence test), but the label already published
    for that turn is not revised. This bounds that cost rather than hiding it.

    The bound is a regression guard set from measurement, not a target: with this
    seed the worst speaker is 13/16 = 0.81, because it has the fewest turns and so
    proportionally more of them arrive before its mean converges. That is inherent
    to online assignment; removing it means delaying labels, which costs latency.
    """
    _, truth, live = _run_conversation(noise=0.08)
    assert _purity(truth, live) >= 0.80


def test_short_window_matches_a_confirmed_speaker_at_the_short_window_bar():
    manager = VoiceIdentityManager(
        logger=logging.getLogger("test"), short_window_seconds=1.5, short_window_threshold=0.45
    )
    rng = np.random.default_rng(3)
    voice = rng.normal(size=192)
    voice /= np.linalg.norm(voice)
    for i in range(4):
        manager.process_new_embedding_batch({f"t{i}": voice}, speech_seconds=2.0)
    (uid,) = manager.identities
    assert manager.identities[uid].confirmed

    # A vector scoring 0.50 against the speaker: below the confirmed bar, above the short one.
    other = rng.normal(size=192)
    other -= (other @ voice) * voice
    other /= np.linalg.norm(other)
    probe = 0.5 * voice + np.sqrt(1 - 0.25) * other
    long_result = manager.process_new_embedding_batch(
        {"long": probe}, speech_seconds=2.0, learn=False, allow_create=False
    )
    short_result = manager.process_new_embedding_batch(
        {"short": probe}, speech_seconds=1.0, learn=False, allow_create=False
    )
    assert "long" not in long_result
    assert short_result["short"][0] == uid


def test_reseed_replaces_only_a_single_seed():
    manager = VoiceIdentityManager(logger=logging.getLogger("test"))
    rng = np.random.default_rng(4)
    first, second = (rng.normal(size=192) for _ in range(2))
    (uid, _), = manager.process_new_embedding_batch({"t": first}, speech_seconds=1.0).values()
    assert manager.reseed_identity(uid, second, 2.0)
    identity = manager.identities[uid]
    assert np.allclose(identity.mean_embedding, second / np.linalg.norm(second))
    assert identity.clean_speech_seconds == 2.0
    identity.all_embeddings.append(identity.mean_embedding.copy())
    assert not manager.reseed_identity(uid, first, 2.0)
