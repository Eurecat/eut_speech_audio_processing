#!/usr/bin/env python3
"""Build a *silver* utterance ground truth: GT speaker turns + offline Whisper text.

The speaker turns are real ground truth (RTTM from the AIMARA suite, or a known
speaker order for scripted files). The text is NOT: each turn is transcribed
offline with faster-whisper large-v3-turbo (a different model than the Parakeet
streaming pipeline), and a person must review it. Edit ``text`` in the output
JSON and set ``verified: true`` on every utterance you checked.

Usage (inside the eut_audio image, where faster_whisper is installed):

    # CallHome excerpt: speaker turns come from the RTTM
    python3 make_silver_gt.py --audio X.wav --rttm X.rttm --out gt/X.json

    # Scripted file with no RTTM: VAD segments, merged until they match the order
    python3 make_silver_gt.py --audio weather.wav --speaker-order A,B,B,C,C,B,A,C,B,A --out gt/weather.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
PAD_SEC = 0.1  # context around each turn so Whisper does not clip the first/last word
DEFAULT_MODEL = "/workspace/weights/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"


def read_rttm(path: Path) -> list[dict]:
    turns = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        turns.append({"start": start, "end": start + dur, "speaker": parts[7]})
    return sorted(turns, key=lambda t: (t["start"], t["end"]))


def vad_turns(audio: np.ndarray, order: list[str]) -> list[dict]:
    """Silero VAD segments, merged across the smallest gaps until one per speaker turn."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))
    segs = [[s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE] for s in stamps]
    if len(segs) < len(order):
        raise SystemExit(f"VAD found {len(segs)} segments, fewer than the {len(order)} turns in --speaker-order")
    while len(segs) > len(order):
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        i = int(np.argmin(gaps))
        segs[i][1] = segs[i + 1][1]
        del segs[i + 1]
    return [{"start": s, "end": e, "speaker": spk} for (s, e), spk in zip(segs, order)]


def overlaps_other(turn: dict, turns: list[dict]) -> bool:
    return any(
        o is not turn and o["speaker"] != turn["speaker"]
        and min(o["end"], turn["end"]) - max(o["start"], turn["start"]) > 0.0
        for o in turns
    )


def resolve_model(path: str) -> str:
    snaps = Path(path) / "snapshots"
    if snaps.is_dir():
        found = sorted(p for p in snaps.iterdir() if (p / "model.bin").exists())
        if found:
            return str(found[-1])
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--rttm", type=Path)
    ap.add_argument("--speaker-order", help="comma list, one label per utterance (no-RTTM files)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--language", default="es")
    ap.add_argument("--model", default=os.environ.get("SILVER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--min-turn", type=float, default=0.3, help="drop GT turns shorter than this (s)")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SAMPLE_RATE, f"{args.audio} is {sr} Hz, expected {SAMPLE_RATE}"
    duration = len(audio) / SAMPLE_RATE

    if args.rttm:
        turns, source = read_rttm(args.rttm), f"rttm:{args.rttm.name}"
    elif args.speaker_order:
        order = [s.strip() for s in args.speaker_order.split(",") if s.strip()]
        turns, source = vad_turns(audio, order), "vad+speaker_order"
    else:
        raise SystemExit("give --rttm or --speaker-order")
    all_turns = list(turns)
    turns = [t for t in turns if t["end"] - t["start"] >= args.min_turn]

    import ctranslate2
    from faster_whisper import WhisperModel

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(resolve_model(args.model), device=device,
                         compute_type="float16" if device == "cuda" else "float32")

    utterances = []
    for idx, t in enumerate(turns):
        a = int(max(0.0, t["start"] - PAD_SEC) * SAMPLE_RATE)
        b = int(min(duration, t["end"] + PAD_SEC) * SAMPLE_RATE)
        segments, _ = model.transcribe(
            audio[a:b], language=args.language, beam_size=5,
            vad_filter=False, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        utterances.append({
            "id": f"u{idx:03d}",
            "start": round(t["start"], 3),
            "end": round(t["end"], 3),
            "speaker": t["speaker"],
            "overlap": overlaps_other(t, all_turns),
            "text": text,
            "text_silver": text,
            "verified": False,
        })
        print(f"{t['start']:7.2f}-{t['end']:7.2f} {t['speaker']:>3} | {text}", flush=True)

    out = {
        "schema_version": 1,
        "audio": args.audio.name,
        "duration_sec": round(duration, 3),
        "language": args.language,
        "speakers": sorted({u["speaker"] for u in utterances}),
        "turn_source": source,
        "text_source": f"silver: faster-whisper {Path(args.model).name} per GT turn, beam 5",
        "review_note": "text is machine-made; edit 'text', keep 'text_silver', set verified=true when checked",
        "utterances": utterances,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(utterances)} utterances, {len(all_turns) - len(turns)} short turns dropped)")


if __name__ == "__main__":
    main()
