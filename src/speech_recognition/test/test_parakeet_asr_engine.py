"""Parakeet output must reach the shared speaker grouping in Whisper's format.

Heavy dependencies (torch, ctranslate2, faster_whisper, nemo) are stubbed when
missing: only the NeMo hypothesis -> segment adapter is under test, not the model.
"""

import importlib.util
import sys
import threading
import types
from collections import deque
from pathlib import Path

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
    return engine


def test_segments_keep_whisper_word_format():
    segments = _engine()._to_segments(_HYPOTHESIS)

    assert [s["text"] for s in segments] == ["You're a jerk, Tom.", "Look, Celia."]
    assert segments[0]["words"][0] == (0.0, 0.4, " You're")
    assert [w[2] for w in segments[1]["words"]] == [" Look,", " Celia."]


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
    assert segments == [{"start": 0.0, "end": 0.0, "text": "Hello.", "words": []}]


def test_reported_language_is_first_configured_code():
    assert _engine("en, es, ca")._reported_language() == "en"
    assert _engine("es")._reported_language() == "es"
    assert _engine("auto")._reported_language() == "unknown"
