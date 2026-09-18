"""`speaker_id_confidence` on SpeechResult must mean something, or stay -1.

The field existed and was pinned to 0.0, which downstream cannot tell apart from
"identified this speaker with zero confidence". EutPersonManager weighs a voice
link by this number, so a wrong value there is worse than no value.

The logic under test is pure: it reads the speaker timeline and an interval. It
is exercised without importing the ASR engine's model dependencies by binding the
unbound method onto a stub that owns only the two attributes it touches.
"""

import os
import sys
import threading
import types

import pytest

ENGINE = os.path.join(
    os.path.dirname(__file__), "..", "speech_recognition", "asr_engine.py"
)


def load_method():
    """Extract `speaker_confidence_for_interval` without importing the module.

    Importing asr_engine pulls in Whisper/NeMo; this test is about arithmetic on a
    timeline, so the source is compiled in isolation instead.
    """
    import ast
    import typing

    tree = ast.parse(open(ENGINE).read())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "speaker_confidence_for_interval"
    )
    # The signature is annotated with typing names, which are evaluated when the
    # function object is created, so they have to exist in the namespace.
    namespace: dict = {"Optional": typing.Optional, "float": float, "str": str}
    exec(compile(ast.Module([node], []), ENGINE, "exec"), namespace)
    return namespace["speaker_confidence_for_interval"]


class Stub:
    def __init__(self, timeline):
        self.speaker_timeline = timeline
        self._speaker_lock = threading.Lock()


@pytest.fixture(scope="module")
def confidence():
    method = load_method()
    return lambda timeline, start, end, speaker: method(Stub(timeline), start, end, speaker)


def segment(start, end, speaker, conf):
    return {"start": start, "end": end, "speaker": speaker, "active": True, "confidence": conf}


def test_full_coverage_and_certain_match_is_the_match_score(confidence):
    timeline = [segment(0.0, 10.0, "EUT_speaker1", 0.9)]
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == pytest.approx(0.9)


def test_partial_coverage_lowers_the_confidence(confidence):
    """A speaker who held a third of the utterance is a weak label for it."""
    timeline = [segment(0.0, 3.0, "EUT_speaker1", 0.9)]
    # 3s of a 9s utterance at match 0.9 -> 0.3 * 0.9
    assert confidence(timeline, 0.0, 9.0, "EUT_speaker1") == pytest.approx(0.3)


def test_weak_match_lowers_the_confidence_even_at_full_coverage(confidence):
    timeline = [segment(0.0, 10.0, "EUT_speaker1", 0.42)]
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == pytest.approx(0.42)


def test_several_stretches_are_weighted_by_their_duration(confidence):
    timeline = [
        segment(0.0, 8.0, "EUT_speaker1", 1.0),
        segment(8.0, 10.0, "EUT_speaker1", 0.5),
    ]
    # full coverage, mean match = (8*1.0 + 2*0.5) / 10
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == pytest.approx(0.9)


def test_other_speakers_do_not_contribute(confidence):
    timeline = [
        segment(0.0, 5.0, "EUT_speaker1", 0.8),
        segment(5.0, 10.0, "EUT_speaker2", 1.0),
    ]
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == pytest.approx(0.4)


def test_missing_score_reports_unavailable_not_zero(confidence):
    """A backend that reports no score must not be turned into low confidence."""
    timeline = [segment(0.0, 10.0, "EUT_speaker1", -1.0)]
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == -1.0


def test_unknown_speaker_or_empty_timeline_is_unavailable(confidence):
    assert confidence([], 0.0, 10.0, "EUT_speaker1") == -1.0
    assert confidence(
        [segment(0.0, 10.0, "EUT_speaker2", 0.9)], 0.0, 10.0, "EUT_speaker1"
    ) == -1.0


def test_degenerate_intervals_are_unavailable(confidence):
    timeline = [segment(0.0, 10.0, "EUT_speaker1", 0.9)]
    assert confidence(timeline, None, 10.0, "EUT_speaker1") == -1.0
    assert confidence(timeline, 5.0, 5.0, "EUT_speaker1") == -1.0
    assert confidence(timeline, 8.0, 2.0, "EUT_speaker1") == -1.0


def test_result_never_leaves_the_unit_range(confidence):
    """Coverage is clamped: a timeline longer than the utterance must not exceed 1."""
    timeline = [segment(-100.0, 100.0, "EUT_speaker1", 1.0)]
    assert confidence(timeline, 0.0, 10.0, "EUT_speaker1") == pytest.approx(1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
