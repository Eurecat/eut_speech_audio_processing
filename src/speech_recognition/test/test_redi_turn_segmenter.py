"""Unit tests for TurnSegmenter — pure Python, no model, no ROS, no GPU."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1] / "speech_recognition"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("speech_recognition.voice_identity_manager", "voice_identity_manager.py")
TurnSegmenter = _load("redi_voice_engine", "redi_voice_engine.py").TurnSegmenter

SR = 16000
CHUNK = 512  # 32 ms, what the audio publisher sends


def _segmenter(**overrides):
    options = dict(
        sample_rate=SR,
        vad_threshold=0.5,
        turn_silence_seconds=0.35,
        min_embed_seconds=0.8,
        embed_interval_seconds=1.0,
        max_embed_seconds=6.0,
    )
    options.update(overrides)
    return TurnSegmenter(**options)


def _run(segmenter, pattern):
    """pattern: list of (seconds, is_speech). Returns all emitted observations."""
    out = []
    for seconds, speech in pattern:
        for _ in range(int(round(seconds * SR / CHUNK))):
            chunk = np.full(CHUNK, 0.1 if speech else 0.0, dtype=np.float32)
            out.extend(segmenter.push(chunk, 0.9 if speech else 0.05))
    return out


def test_pause_splits_speech_into_separate_turns():
    obs = _run(_segmenter(), [(2.0, True), (0.6, False), (2.0, True), (0.6, False)])
    finals = [o for o in obs if o.final]
    assert [o.turn_id for o in finals] == ["turn1", "turn2"]


def test_short_pause_inside_a_sentence_does_not_split_the_turn():
    obs = _run(_segmenter(), [(1.0, True), (0.2, False), (1.0, True), (0.6, False)])
    assert len({o.turn_id for o in obs}) == 1


def test_provisional_label_arrives_before_the_turn_ends():
    obs = _run(_segmenter(), [(3.0, True)])  # still talking, no pause yet
    assert obs, "a speaker must be labelled while still speaking"
    assert all(not o.final for o in obs)
    first = obs[0]
    assert abs(len(first.audio) / SR - 0.8) < CHUNK / SR + 1e-9


def test_blip_too_short_to_embed_emits_nothing():
    assert _run(_segmenter(), [(0.3, True), (0.6, False)]) == []


def test_observations_contain_speech_only_never_the_pauses():
    obs = _run(_segmenter(), [(1.0, True), (0.2, False), (1.0, True), (0.6, False)])
    final = [o for o in obs if o.final][0]
    assert np.all(final.audio == np.float32(0.1)), "silence leaked into the embedding audio"


def test_long_turn_embeds_only_the_most_recent_window():
    obs = _run(_segmenter(max_embed_seconds=2.0), [(5.0, True), (0.6, False)])
    final = [o for o in obs if o.final][0]
    assert len(final.audio) <= 2 * SR


def test_close_turn_flushes_the_open_turn_on_shutdown():
    segmenter = _segmenter()
    _run(segmenter, [(1.5, True)])
    flushed = segmenter.close_turn()
    assert len(flushed) == 1 and flushed[0].final


# ---------------------------------------------------------------------------
# RediVoiceEngine decision logic, with a fake embedder (no model, no GPU)
# ---------------------------------------------------------------------------

import logging  # noqa: E402

_engine_module = sys.modules["redi_voice_engine"]
_vim = sys.modules["speech_recognition.voice_identity_manager"]
DIM = 192


def _voice(seed):
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def _engine(voices):
    """Engine wired without initialize(); audio[0] selects which voice it 'contains'."""
    changes = []
    engine = _engine_module.RediVoiceEngine(
        vad_threshold=0.5,
        use_database=False,
        ros4hri_enabled=False,
        on_eut_speaker_changed=changes.append,
        on_voice_update=lambda *_: None,
        logger=logging.getLogger("test"),
        redi_repository="",
        redi_model_name="",
        redi_train_type="",
        redi_dataset="",
    )
    engine._manager = _vim.VoiceIdentityManager(logger=logging.getLogger("test"))
    engine._segmenter = _segmenter(max_embed_seconds=2.0, embed_interval_seconds=0.5)
    rng = np.random.default_rng(0)

    def fake_embed(audio):
        voice = voices[int(audio[0])]
        noisy = voice + rng.normal(scale=0.05, size=DIM)
        return noisy / np.linalg.norm(noisy)

    engine._embed = fake_embed
    return engine, changes


def _obs(turn, voice, final, at, seconds=2.0):
    """A window of `seconds` of `voice`, observed when the turn holds `at` seconds of speech."""
    audio = np.full(int(seconds * SR), float(voice), dtype=np.float32)
    return _engine_module.TurnObservation(
        turn_id=turn, audio=audio, mean_vad=0.9, final=final, turn_speech_seconds=at
    )


def _sizes(engine):
    return {uid: len(i.all_embeddings) for uid, i in engine._manager.identities.items()}


def test_short_window_cannot_create_a_speaker():
    engine, changes = _engine({0: _voice(1)})
    engine._handle(_obs("turn1", 0, final=False, at=0.8, seconds=0.8))
    assert engine._manager.identities == {}
    assert changes == []


def test_short_window_still_recognises_a_known_speaker():
    engine, changes = _engine({0: _voice(1)})
    engine._handle(_obs("turn1", 0, final=True, at=2.0))
    known = changes[-1]
    engine._handle(_obs("turn2", 0, final=False, at=0.8, seconds=0.8))
    assert changes[-1] == known


def test_long_clean_turn_keeps_learning_at_spaced_intervals():
    engine, _ = _engine({0: _voice(1)})
    for i, at in enumerate([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]):
        engine._handle(_obs("turn1", 0, final=False, at=at))
    (identity,) = engine._manager.identities.values()
    # created at 1.5s, then learned at 3.5s and 5.5s: every 2s of speech, never
    # from each overlapping 0.5s refresh.
    assert len(identity.all_embeddings) == 3


def test_window_crossing_a_speaker_change_is_not_learned():
    engine, _ = _engine({0: _voice(1), 1: _voice(2)})
    engine._handle(_obs("seed_b", 1, final=True, at=2.0))  # B known
    engine._handle(_obs("turn1", 0, final=False, at=2.0))  # A starts, creates A
    engine._handle(_obs("turn1", 0, final=False, at=4.0))  # A stable -> learned
    before = _sizes(engine)
    engine._handle(_obs("turn1", 1, final=False, at=6.0))  # B takes over: label switches
    assert _sizes(engine) == before, "the boundary window must not be learned"


def test_speaker_returning_after_an_interruption_keeps_their_original_id():
    """Reproduces the reported failure: B, A interrupts without a pause, B again."""
    engine, changes = _engine({0: _voice(1), 1: _voice(2)})
    t = 0.0
    for _ in range(4):  # B talks for a while in one turn
        t += 2.0
        engine._handle(_obs("turn1", 1, final=False, at=t))
    b_id = changes[-1]
    for _ in range(2):  # A interrupts, same turn, no pause
        t += 2.0
        engine._handle(_obs("turn1", 0, final=False, at=t))
    a_id = changes[-1]
    t += 2.0
    engine._handle(_obs("turn1", 0, final=True, at=t))
    engine._handle(_obs("turn2", 1, final=False, at=2.0))  # B returns, new turn

    assert a_id != b_id
    assert changes[-1] == b_id, "B came back under a new id instead of B's own"
    assert len(engine._manager.identities) == 2


def test_engine_reports_speaker_change_to_the_node():
    engine, changes = _engine({0: _voice(1), 1: _voice(2)})
    engine._handle(_obs("turn1", 0, final=True, at=2.0))
    engine._handle(_obs("turn2", 1, final=True, at=2.0))
    assert changes[0] != changes[-1]


# ---------------------------------------------------------------------------
# Speaker change inside one turn: segments and the probe
# ---------------------------------------------------------------------------


def _voice_chunks(segmenter, voice, seconds):
    """Push `seconds` of speech whose samples all hold `voice`; return the observations."""
    out = []
    for _ in range(int(round(seconds * SR / CHUNK))):
        out.extend(segmenter.push(np.full(CHUNK, float(voice), dtype=np.float32), 0.9))
    return out


def test_split_starts_a_new_segment_holding_only_later_speech():
    segmenter = _segmenter(max_embed_seconds=2.0, embed_interval_seconds=0.5, probe_seconds=1.0)
    first = _voice_chunks(segmenter, 1, 2.0)[-1]
    assert first.turn_id == "turn1"
    new_id = segmenter.split("turn1", segmenter._speech_samples)
    assert new_id == "turn1.2"
    later = _voice_chunks(segmenter, 2, 1.0)
    assert later and all(o.turn_id == "turn1.2" for o in later)
    assert all(np.all(o.audio == np.float32(2.0)) for o in later), "old speaker leaked into segment"


def test_split_of_a_segment_that_moved_on_is_refused():
    segmenter = _segmenter(probe_seconds=1.0)
    _voice_chunks(segmenter, 1, 2.0)
    assert segmenter.split("turn1", 8000) == "turn1.2"
    assert segmenter.split("turn1", 16000) is None  # already split
    _run(segmenter, [(0.6, False)])
    assert segmenter.split("turn1.2", 16000) is None  # turn ended


def test_probe_is_the_most_recent_speech_and_only_once_longer_than_it():
    segmenter = _segmenter(max_embed_seconds=2.0, embed_interval_seconds=0.5, probe_seconds=1.0)
    obs = _voice_chunks(segmenter, 1, 3.0)
    assert obs[0].probe is None  # 0.8 s segment: the probe would be the whole window
    probed = [o for o in obs if o.probe is not None]
    assert probed
    assert all(abs(len(o.probe) / SR - 1.0) < CHUNK / SR + 1e-9 for o in probed)


def _mixing_engine(voices, probe_seconds=1.0):
    """Engine fed by a real segmenter; an embedding blends the voices its samples hold."""
    engine, changes = _engine(voices)
    engine._segmenter = _segmenter(
        max_embed_seconds=2.0, embed_interval_seconds=0.5, probe_seconds=probe_seconds
    )
    engine._change_threshold = 0.35
    rng = np.random.default_rng(0)

    def fake_embed(audio):
        values, counts = np.unique(audio.astype(int), return_counts=True)
        mix = sum(c * voices[v] for v, c in zip(values, counts))
        noisy = mix / np.linalg.norm(mix) + rng.normal(scale=0.05, size=DIM)
        return noisy / np.linalg.norm(noisy)

    engine._embed = fake_embed
    return engine, changes


def _speak(engine, voice, seconds):
    """Feed speech and return (seconds of speech fed so far in this call, label) per decision."""
    decisions = []
    for i in range(int(round(seconds * SR / CHUNK))):
        chunk = np.full(CHUNK, float(voice), dtype=np.float32)
        for observation in engine._segmenter.push(chunk, 0.9):
            engine._handle(observation)
            decisions.append(((i + 1) * CHUNK / SR, engine._last_speaker))
    return decisions


def test_label_switches_about_one_probe_after_an_unknown_speaker_takes_over():
    engine, _ = _mixing_engine({1: _voice(1), 2: _voice(2)})
    _speak(engine, 1, 4.0)
    a_id = engine._last_speaker
    decisions = _speak(engine, 2, 3.0)  # B takes over, no pause
    switched = next(t for t, speaker in decisions if speaker != a_id)
    assert switched <= 1.0, f"label switched {switched:.2f}s after the change"  # 1.41 s without the probe
    b_id = decisions[-1][1]
    assert b_id != a_id and len(engine._manager.identities) == 2


def test_known_speaker_taking_over_is_recognised_from_the_probe_without_a_new_id():
    engine, _ = _mixing_engine({1: _voice(1), 2: _voice(2)})
    _speak(engine, 2, 4.0)
    b_id = engine._last_speaker
    _run(engine._segmenter, [(0.6, False)])
    _speak(engine, 1, 4.0)
    a_id = engine._last_speaker
    decisions = _speak(engine, 2, 3.0)
    switched = next(t for t, speaker in decisions if speaker != a_id)
    assert switched <= 1.0
    assert decisions[-1][1] == b_id
    assert len(engine._manager.identities) == 2


def test_one_speaker_talking_on_never_triggers_a_change():
    engine, _ = _mixing_engine({1: _voice(1)})
    _speak(engine, 1, 12.0)
    assert len(engine._manager.identities) == 1
    assert engine._segmenter.current_turn_id == "turn1"


def test_speaker_created_from_a_probe_is_reseeded_from_its_longer_windows():
    engine, _ = _mixing_engine({1: _voice(1), 2: _voice(2)})
    _speak(engine, 1, 4.0)
    a_id = engine._last_speaker
    _speak(engine, 2, 3.0)
    (b_id,) = [uid for uid in engine._manager.identities if uid != a_id]
    b = engine._manager.identities[b_id]
    assert b.clean_speech_seconds >= 2.0, "seed still the 1.0 s probe"


def test_observation_queued_before_its_segment_was_split_is_ignored():
    engine, changes = _mixing_engine({1: _voice(1), 2: _voice(2)})
    _speak(engine, 1, 4.0)
    stale = engine._segmenter._observe(final=False)  # still "turn1", queued late
    _speak(engine, 2, 2.0)  # the change splits turn1
    before = (len(changes), _sizes(engine))
    engine._handle(stale)
    assert (len(changes), _sizes(engine)) == before


def test_probe_disabled_keeps_the_previous_behaviour():
    engine, _ = _mixing_engine({1: _voice(1), 2: _voice(2)}, probe_seconds=0.0)
    _speak(engine, 1, 4.0)
    a_id = engine._last_speaker
    _speak(engine, 2, 3.0)
    assert engine._segmenter._observe(final=False).probe is None
    assert engine._segmenter.current_turn_id == "turn1", "split without a probe"
    assert engine._last_speaker != a_id  # the full window still switches, only later
