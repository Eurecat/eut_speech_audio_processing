"""Audio must not be repeated across a mid-speech chunk split.

Heavy dependencies (torch, ctranslate2, faster_whisper) are stubbed: only the
buffer bookkeeping is under test, not transcription.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

for _name in ("torch", "ctranslate2", "faster_whisper"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["faster_whisper"].BatchedInferencePipeline = object
sys.modules["faster_whisper"].WhisperModel = object

_PATH = Path(__file__).resolve().parents[1] / "speech_recognition" / "asr_engine.py"
_spec = importlib.util.spec_from_file_location("asr_engine_under_test", _PATH)
_asr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _asr
_spec.loader.exec_module(_asr)
ASREngine = _asr.ASREngine

CHUNK_SECONDS = 0.032


class _Captured:
    """Stands in for the transcription thread: records what would be transcribed."""

    def __init__(self):
        self.calls = []

    def __call__(self, audio, start, stop, **_kwargs):
        self.calls.append(audio.copy())


def _engine(pre_buffer=0.5, min_speaker_chunk=1.0):
    engine = ASREngine.__new__(ASREngine)  # skip model loading
    engine.pre_buffer_duration = pre_buffer
    engine.min_speaker_chunk_duration = min_speaker_chunk
    engine.sample_rate = 16000
    engine.audio_buffer = []
    engine.speech_start_time = 0.0
    engine.vad_state = True
    engine._segment_has_onset = True
    engine._buffer_lock = _asr.threading.Lock()
    engine._transcribe_with_data = _Captured()
    return engine


def _fill(engine, start, seconds):
    """One chunk per 32 ms; each chunk's samples hold its own index, so overlaps are visible."""
    n = int(round(seconds / CHUNK_SECONDS))
    for i in range(n):
        index = len(engine.audio_buffer)
        engine.audio_buffer.append(
            {"timestamp": start + i * CHUNK_SECONDS, "audio": np.full(512, index, dtype=np.float32)}
        )


def _chunk_ids(audio):
    return set(np.unique(audio).astype(int))


def test_speaker_change_split_does_not_repeat_audio(monkeypatch):
    engine = _engine()
    t0 = 1000.0
    _fill(engine, t0 - 1.0, 6.0)  # includes 1s of audio before speech started
    engine.speech_start_time = t0
    engine._segment_has_onset = True

    # Run the flush synchronously so the test can inspect what it would publish.
    monkeypatch.setattr(
        _asr.threading,
        "Thread",
        lambda target, args, kwargs, daemon: types.SimpleNamespace(
            start=lambda: target(*args, **kwargs)
        ),
    )
    change = t0 + 2.0
    engine._flush_chunk_on_speaker_change(change)
    first = engine._transcribe_with_data.calls[0]

    with engine._buffer_lock:
        second, _, _ = engine._extract_audio_data(t0 + 4.0)

    overlap = _chunk_ids(first) & _chunk_ids(second)
    assert not overlap, f"{len(overlap)} chunks published twice across the split"


def test_pre_buffer_still_applies_at_a_genuine_speech_onset():
    engine = _engine(pre_buffer=0.5)
    t0 = 1000.0
    _fill(engine, t0 - 1.0, 3.0)
    engine.speech_start_time = t0
    engine._segment_has_onset = True

    with engine._buffer_lock:
        audio, start, _ = engine._extract_audio_data(t0 + 1.0)
    assert start == t0 - 0.5, "word onsets before VAD fired must still be captured"
    assert len(audio) >= int(round(1.5 / CHUNK_SECONDS)) * 512


def test_chunk_confidence_defaults_to_zero_when_backend_reports_nothing():
    """Whisper's _collect_segments() never sets "confidence", so this is what
    keeps SpeechResult.transcript_confidence at 0.0 for that backend."""
    segments = [{"text": "hello"}, {"text": "world"}]
    assert ASREngine._chunk_confidence(segments) == 0.0


def test_chunk_confidence_averages_present_values_and_ignores_missing_ones():
    segments = [{"confidence": 0.8}, {"confidence": None}, {"confidence": 0.4}]
    assert round(ASREngine._chunk_confidence(segments), 6) == 0.6
