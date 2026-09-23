"""Which speaker a transcribed word is attributed to, from SpeechActivityDetection.

Heavy dependencies are stubbed; only the speaker timeline is under test.
"""

import importlib.util
import logging
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

for _name in ("torch", "ctranslate2", "faster_whisper"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["faster_whisper"].BatchedInferencePipeline = object
sys.modules["faster_whisper"].WhisperModel = object

_PATH = Path(__file__).resolve().parents[1] / "speech_recognition" / "asr_engine.py"
_spec = importlib.util.spec_from_file_location("asr_engine_attribution", _PATH)
_asr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _asr
_spec.loader.exec_module(_asr)


def _engine():
    engine = _asr.ASREngine.__new__(_asr.ASREngine)  # skip model loading
    engine._logger = logging.getLogger("test")
    engine.diarization_offset = -1.0
    engine.speaker_interval_tolerance = 3.0
    engine.speaker_timeline = deque()
    engine._speaker_lock = threading.RLock()
    engine.speaker_id = None
    engine.flushes = []
    engine._flush_chunk_on_speaker_change = engine.flushes.append
    return engine


def test_speaker_who_stopped_is_not_held_over_the_next_speech():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 10.0)
    engine.update_speaker("speaker1", False, stamp=now - 8.0)
    assert engine._resolve_speaker_for_interval(now - 5.0, now - 4.0) == "unknown"


def test_speaker_still_talking_is_held_while_diarization_catches_up():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 10.0)
    assert engine._resolve_speaker_for_interval(now - 0.5, now - 0.2) == "speaker1"


def test_label_covers_its_whole_segment_from_the_stamped_start():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 10.0)
    engine.update_speaker("speaker1", False, stamp=now - 8.0)
    # Decided now, after ~1.8s of speech, but it describes speech from 3s ago.
    engine.update_speaker("speaker2", True, -1.0, stamp=now - 3.0)
    assert engine._resolve_speaker_for_interval(now - 2.9, now - 2.6) == "speaker2"


def test_new_speech_after_a_stop_is_not_split():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 10.0)
    engine.update_speaker("speaker1", False, stamp=now - 8.0)
    engine.update_speaker("speaker2", True, -1.0, stamp=now - 3.0)
    assert engine.flushes == []


def test_switch_while_the_previous_speaker_talks_splits_at_the_new_segment_start():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 10.0)
    engine.update_speaker("speaker2", True, 0.8, stamp=now - 2.0)
    assert engine.flushes == [now - 2.0]
    assert engine._resolve_speaker_for_interval(now - 4.0, now - 3.0) == "speaker1"
    assert engine._resolve_speaker_for_interval(now - 1.9, now - 1.5) == "speaker2"


def test_revised_label_for_the_same_stretch_replaces_the_first_one():
    engine, now = _engine(), time.time()
    engine.update_speaker("speaker1", True, 0.9, stamp=now - 3.0)
    engine.update_speaker("speaker2", True, 0.9, stamp=now - 3.0)
    assert engine._resolve_speaker_for_interval(now - 2.9, now - 2.5) == "speaker2"
    assert [s["speaker"] for s in engine.speaker_timeline] == ["speaker2"]


def test_unstamped_events_still_get_the_diarization_offset():
    engine = _engine()
    before = time.time()
    engine.update_speaker("speaker1", True, 0.9)
    start = engine.speaker_timeline[-1]["start"]
    assert before - 1.0 - 0.01 <= start <= time.time() - 1.0 + 0.01
