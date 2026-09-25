"""Parakeet output must reach the shared speaker grouping in Whisper's format.

Heavy dependencies (torch, ctranslate2, faster_whisper, nemo) are stubbed when
missing: only the NeMo hypothesis -> segment adapter and the published language
choice are under test, not the models.
"""

import importlib.util
import sys
import threading
import types
from collections import deque
from pathlib import Path

import numpy as np

for _name in ("torch", "ctranslate2", "faster_whisper"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
        if _name == "faster_whisper":
            sys.modules[_name].BatchedInferencePipeline = object
            sys.modules[_name].WhisperModel = object

_PKG = Path(__file__).resolve().parents[1] / "speech_recognition"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "speech_recognition.asr_engine" not in sys.modules:
    _load("speech_recognition.asr_engine", "asr_engine.py")
if "speech_recognition.language_id" not in sys.modules:
    _load("speech_recognition.language_id", "language_id.py")
ParakeetASREngine = _load("parakeet_asr_engine_under_test", "parakeet_asr_engine.py").ParakeetASREngine


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _word(text, start, end):
    return {"word": text, "start": start, "end": end}


# What NeMo 2.4 returns for "You're a jerk, Tom. Look, Celia." after
# process_timestamp_outputs() (seconds relative to the chunk).
_HYPOTHESIS = types.SimpleNamespace(
    text="You're a jerk, Tom. Look, Celia.",
    timestamp={
        "word": [
            _word("You're", 0.0, 0.4),
            _word("a", 0.4, 0.6),
            _word("jerk,", 0.6, 0.9),
            _word("Tom.", 1.0, 1.3),
            _word("Look,", 2.0, 2.3),
            _word("Celia.", 2.4, 2.9),
        ],
        "segment": [
            {"segment": "You're a jerk, Tom.", "start": 0.0, "end": 1.3},
            {"segment": "Look, Celia.", "start": 2.0, "end": 2.9},
        ],
    },
)


def _engine(language="en, es, ca"):
    engine = ParakeetASREngine.__new__(ParakeetASREngine)  # skip model loading
    engine._logger = _Logger()
    engine.language = language
    engine.parakeet_model_name = "nvidia/parakeet-tdt-0.6b-v3"
    engine.speaker_id = None
    engine.speaker_timeline = deque()
    engine._speaker_lock = threading.RLock()
    engine.speaker_interval_tolerance = 3.0
    engine.snap_splits_to_sentences = True
    engine.min_speaker_run_tokens = 2
    engine.min_speaker_run_duration = 0.4
    engine._language_id = None
    engine._last_language = None
    engine.language_id_min_confidence = 0.9
    return engine


class _ScriptedLanguageId:
    """Stands in for AmberNet: returns the next scripted probability dict."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def probabilities(self, audio, languages):
        self.calls.append(list(languages))
        return self.results.pop(0)


_SPEECH = np.zeros(16000, dtype=np.float32)  # 1 s: long enough for language identification


def test_segments_keep_whisper_word_format():
    segments = _engine()._to_segments(_HYPOTHESIS)

    assert [s["text"] for s in segments] == ["You're a jerk, Tom.", "Look, Celia."]
    assert segments[0]["words"][0] == (0.0, 0.4, " You're")
    assert [w[2] for w in segments[1]["words"]] == [" Look,", " Celia."]


def test_to_segments_attaches_the_same_chunk_confidence_to_every_segment():
    """The NeMo confidence estimator scores the whole decode, not sub-spans of
    it, so every segment split out of one hypothesis gets the same number."""
    hypothesis = types.SimpleNamespace(
        text=_HYPOTHESIS.text,
        timestamp=_HYPOTHESIS.timestamp,
        word_confidence=[0.9, 0.8, 0.2, 0.5, 0.7, 0.3],  # mean = 0.5666...
    )
    segments = _engine()._to_segments(hypothesis)
    assert [s["confidence"] for s in segments] == [
        segments[0]["confidence"],
        segments[0]["confidence"],
    ]
    assert round(segments[0]["confidence"], 4) == round(sum(hypothesis.word_confidence) / 6, 4)


def test_hypothesis_confidence_is_none_without_word_confidence():
    """Whisper's shared ASREngine._chunk_confidence() reads this as 0.0: see
    test_chunk_confidence_defaults_to_zero_when_backend_reports_nothing in
    test_asr_chunk_boundaries.py."""
    segments = _engine()._to_segments(_HYPOTHESIS)  # no word_confidence attribute at all
    assert all(s["confidence"] is None for s in segments)


def test_speaker_change_inside_chunk_splits_sentences():
    engine = _engine()
    t0 = 1000.0
    engine.speaker_timeline.extend(
        [
            {"start": t0 - 1.0, "end": t0 + 1.5, "speaker": "speaker1", "active": True},
            {"start": t0 + 1.5, "end": t0 + 3.0, "speaker": "speaker2", "active": True},
        ]
    )

    groups = engine._group_segments_by_speaker(engine._to_segments(_HYPOTHESIS), t0, t0 + 3.0)

    assert [(g["speaker"], g["text"]) for g in groups] == [
        ("speaker1", "You're a jerk, Tom."),
        ("speaker2", "Look, Celia."),
    ]


def test_empty_hypothesis_publishes_nothing():
    assert _engine()._to_segments(types.SimpleNamespace(text="", timestamp={})) == []


def test_text_without_timestamps_is_one_segment():
    segments = _engine()._to_segments(types.SimpleNamespace(text="Hello.", timestamp=None))
    assert segments == [
        {"start": 0.0, "end": 0.0, "text": "Hello.", "words": [], "confidence": None}
    ]


def test_confident_detection_is_published_and_remembered():
    engine = _engine("en, es, ca")
    engine._language_id = _ScriptedLanguageId(
        {"en": 0.02, "es": 0.95, "ca": 0.03},
        {"en": 0.40, "es": 0.35, "ca": 0.25},
    )

    assert engine._identify_language(_SPEECH) == "es"
    # Low confidence ("Yeah.") keeps the last confident language instead of flipping.
    assert engine._identify_language(_SPEECH) == "es"
    assert engine._language_id.calls[0] == ["en", "es", "ca"]


def test_low_confidence_before_any_detection_uses_first_configured_language():
    engine = _engine("ca, es")
    engine._language_id = _ScriptedLanguageId({"ca": 0.6, "es": 0.4})
    assert engine._identify_language(_SPEECH) == "ca"
    assert engine._last_language is None


def test_short_chunk_skips_language_identification():
    engine = _engine("en, es, ca")
    engine._last_language = "ca"
    engine._language_id = _ScriptedLanguageId()
    assert engine._identify_language(_SPEECH[:1600]) == "ca"
    assert engine._language_id.calls == []


def test_single_language_and_auto_match_whisper_semantics():
    assert _engine("es")._candidate_languages() == ["es"]
    assert _engine("es")._identify_language(_SPEECH) == "es"
    assert _engine("auto")._candidate_languages() == ["en", "es", "ca"]
    # No LangID model loaded: first candidate.
    assert _engine("en, es, ca")._identify_language(_SPEECH) == "en"


def test_noise_chunk_does_not_change_language():
    engine = _engine("en, es, ca")
    engine._last_language = "es"
    engine._language_id = _ScriptedLanguageId({"en": 0.99, "es": 0.005, "ca": 0.005})
    engine.sample_rate = 16000
    engine._transcribe_hypotheses = lambda audio: [types.SimpleNamespace(text="", timestamp={})]

    assert engine._run_transcription(_SPEECH, "en, es, ca") == ([], "es")
    assert engine._language_id.calls == []
