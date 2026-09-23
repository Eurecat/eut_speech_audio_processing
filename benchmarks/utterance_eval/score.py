#!/usr/bin/env python3
"""Score captured pipeline output against utterance ground truth.

For every file that has both gt/<stem>.json and hyp/<stem>.json:

* WER: all GT words (in time order) are aligned to all published words (in
  publish order) with one Levenshtein alignment. Each GT utterance then gets the
  hypothesis words that landed on it, so segmentation differences between the
  streaming ASR and the GT turns do not count as errors.
* Speaker attribution: the pipeline's ids (speaker1, speaker7, ...) are mapped
  one-to-one onto GT labels (A, B, ...) by the Hungarian method over aligned
  word counts. An utterance is "ok" when the mapped majority speaker of its
  words is its GT speaker; "unknown" is counted apart, never as ok.
* DER (pyannote.metrics, collar 0.25 s): from /speech_activity_detection
  events, and from the ASR utterances themselves.

Writes results/<stem>.json, results/REPORT.md and results/report.html.

    python3 score.py            # every stem with gt + hyp
    python3 score.py wer_es__es_es_weather_wer
"""
from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UNKNOWN = {"", "unknown"}
SILENCE_GATE_SEC = 0.25  # asr min_silence_duration: speech ended this long before the transcript was made
DER_COLLAR = 0.25


# ----------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------

def _num_to_words(match: re.Match, language: str) -> str:
    try:
        from num2words import num2words

        return num2words(int(match.group(0)), lang=language)
    except Exception:
        return match.group(0)


def normalise(text: str, language: str = "es") -> list[str]:
    """Lowercase, digits to words, no accents, no punctuation."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"\d+", lambda m: _num_to_words(m, language), text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s']", " ", text)
    words = text.split()
    if language == "es":
        # "veintiún grados" vs num2words' "veintiuno": fold the apocope on both sides.
        words = [w[:-1] if w.endswith("uno") else w for w in words]
    return words


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment -> [(op, ref_index, hyp_index)], op in ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def hyp_segments_from_results(results: list[dict]) -> list[dict]:
    segs = []
    for r in results:
        end = r["t_pub"] - r["proc_ms"] / 1000.0 - SILENCE_GATE_SEC
        segs.append({"start": max(0.0, end - r["audio_ms"] / 1000.0), "end": max(0.0, end), "speaker": r["speaker"]})
    return segs


def hyp_segments_from_activity(activity: list[dict], duration: float) -> list[dict]:
    segs, cur, since = [], None, 0.0
    for ev in sorted(activity, key=lambda e: e["t"]):
        if cur is not None and ev["t"] > since:
            segs.append({"start": since, "end": ev["t"], "speaker": cur})
        cur, since = (ev["speaker"] if ev["active"] else None), ev["t"]
    if cur is not None and duration > since:
        segs.append({"start": since, "end": duration, "speaker": cur})
    return [s for s in segs if s["speaker"] not in UNKNOWN]


def der(gt_utts: list[dict], hyp_segs: list[dict], duration: float) -> dict | None:
    try:
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None
    ref, hyp = Annotation(), Annotation()
    for k, u in enumerate(gt_utts):
        ref[Segment(u["start"], u["end"]), k] = u["speaker"]
    for k, s in enumerate(hyp_segs):
        if s["end"] > s["start"]:
            hyp[Segment(s["start"], s["end"]), k] = s["speaker"]
    metric = DiarizationErrorRate(collar=DER_COLLAR, skip_overlap=False)
    detail = metric(ref, hyp, uem=Timeline([Segment(0, duration)]), detailed=True)
    total = detail["total"] or 1.0
    return {
        "der": round(detail["diarization error rate"], 4),
        "missed": round(detail["missed detection"] / total, 4),
        "false_alarm": round(detail["false alarm"] / total, 4),
        "confusion": round(detail["confusion"] / total, 4),
        "collar": DER_COLLAR,
    }


def score_file(stem: str) -> dict:
    gt = json.loads((ROOT / "gt" / f"{stem}.json").read_text())
    hyp = json.loads((ROOT / "hyp" / f"{stem}.json").read_text())
    lang = gt.get("language", "es")
    utts = sorted(gt["utterances"], key=lambda u: (u["start"], u["end"]))
    results = hyp["results"]

    ref_words, ref_utt = [], []
    for k, u in enumerate(utts):
        w = normalise(u["text"], lang)
        ref_words += w
        ref_utt += [k] * len(w)
    hyp_words, hyp_res = [], []
    for k, r in enumerate(results):
        w = normalise(r["text"], lang)
        hyp_words += w
        hyp_res += [k] * len(w)

    ops = align(ref_words, hyp_words)

    # Every op belongs to one GT utterance; an insertion joins the utterance of the last ref word seen.
    per_utt: dict[int, list] = defaultdict(list)
    last = 0
    for op, i, j in ops:
        if i is not None:
            last = ref_utt[i]
        per_utt[last].append((op, i, j))

    # Speaker mapping from words that were aligned to a GT word (ok or sub).
    pair = Counter()
    for op, i, j in ops:
        if i is not None and j is not None:
            spk = results[hyp_res[j]]["speaker"]
            if spk not in UNKNOWN:
                pair[(utts[ref_utt[i]]["speaker"], spk)] += 1
    gt_spk = sorted({u["speaker"] for u in utts})
    hyp_spk = sorted({s for _, s in pair})
    mapping: dict[str, str] = {}
    if gt_spk and hyp_spk:
        from scipy.optimize import linear_sum_assignment

        cost = np.array([[-pair[(g, h)] for h in hyp_spk] for g in gt_spk])
        for gi, hi in zip(*linear_sum_assignment(cost)):
            if pair[(gt_spk[gi], hyp_spk[hi])] > 0:
                mapping[hyp_spk[hi]] = gt_spk[gi]

    rows, tot = [], Counter()
    for k, u in enumerate(utts):
        uops = per_utt.get(k, [])
        c = Counter(op for op, _, _ in uops)
        n_ref = c["ok"] + c["sub"] + c["del"]
        errs = c["sub"] + c["del"] + c["ins"]
        tot.update(c)
        spk_votes = Counter(results[hyp_res[j]]["speaker"] for op, _, j in uops if j is not None)
        res_ids = sorted({hyp_res[j] for _, _, j in uops if j is not None})
        if spk_votes:
            raw = spk_votes.most_common(1)[0][0]
            mapped = "unknown" if raw in UNKNOWN else mapping.get(raw, f"extra:{raw}")
            status = "unknown" if raw in UNKNOWN else ("ok" if mapped == u["speaker"] else "wrong")
        else:
            raw, mapped, status = "", "", "missed"
        rows.append({
            "id": u["id"], "start": u["start"], "end": u["end"],
            "gt_speaker": u["speaker"], "gt_text": u["text"], "overlap": u.get("overlap", False),
            "verified": u.get("verified", False),
            "hyp_text": " ".join(hyp_words[j] for _, _, j in uops if j is not None),
            "hyp_speaker_raw": raw, "hyp_speaker": mapped, "speaker_status": status,
            "hyp_result_ids": res_ids,
            "words_ref": n_ref, "sub": c["sub"], "del": c["del"], "ins": c["ins"],
            "wer": round(errs / n_ref, 4) if n_ref else None,
            "diff": [[op, ref_words[i] if i is not None else "", hyp_words[j] if j is not None else ""]
                     for op, i, j in uops],
        })

    n_ref = tot["ok"] + tot["sub"] + tot["del"]
    status = Counter(r["speaker_status"] for r in rows)
    scored = [r for r in rows if r["speaker_status"] != "missed"]
    word_ok = sum(v for (g, h), v in pair.items() if mapping.get(h) == g)
    word_all = sum(v for (g, h), v in pair.items())

    # Confusion: GT speaker -> mapped hyp speaker, over utterances.
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        confusion[r["gt_speaker"]][r["hyp_speaker"] or "missed"] += 1

    duration = gt["duration_sec"]
    return {
        "stem": stem,
        "audio": gt["audio"],
        "duration_sec": duration,
        "gt_text_source": gt.get("text_source", ""),
        "gt_verified": f"{sum(u.get('verified', False) for u in utts)}/{len(utts)}",
        "gt_speakers": gt_spk,
        "hyp_speakers": sorted({r['speaker'] for r in results}),
        "speaker_mapping": mapping,
        "summary": {
            "gt_utterances": len(utts),
            "hyp_utterances": len(results),
            "wer": round((tot["sub"] + tot["del"] + tot["ins"]) / n_ref, 4) if n_ref else None,
            "ref_words": n_ref, "sub": tot["sub"], "del": tot["del"], "ins": tot["ins"],
            "speaker_ok_utt": status["ok"], "speaker_wrong_utt": status["wrong"],
            "speaker_unknown_utt": status["unknown"], "missed_utt": status["missed"],
            "speaker_acc_utt": round(status["ok"] / len(scored), 4) if scored else None,
            "speaker_acc_words": round(word_ok / word_all, 4) if word_all else None,
            "der_activity": der(utts, hyp_segments_from_activity(hyp.get("activity", []), duration), duration),
            "der_asr_utterances": der(utts, hyp_segments_from_results(results), duration),
        },
        "confusion": {g: dict(c) for g, c in sorted(confusion.items())},
        "utterances": rows,
    }


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------

def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def der_str(d) -> str:
    return "—" if not d else f"{pct(d['der'])} (miss {pct(d['missed'])}, FA {pct(d['false_alarm'])}, conf {pct(d['confusion'])})"


def write_markdown(reports: list[dict], path: Path) -> None:
    out = ["# Utterance benchmark: WER + speaker attribution", "",
           "GT text is **silver** (offline Whisper per GT turn) until `verified` is true in `gt/*.json`.",
           "Speaker ids are mapped to GT labels by Hungarian matching on aligned words.", "",
           "| file | GT verified | WER | speaker acc (utt) | speaker acc (words) | ok / wrong / unknown / missed | DER activity | DER ASR utt |",
           "|---|---|---|---|---|---|---|---|"]
    for r in reports:
        s = r["summary"]
        out.append(f"| {r['stem']} | {r['gt_verified']} | {pct(s['wer'])} | {pct(s['speaker_acc_utt'])} | "
                   f"{pct(s['speaker_acc_words'])} | {s['speaker_ok_utt']} / {s['speaker_wrong_utt']} / "
                   f"{s['speaker_unknown_utt']} / {s['missed_utt']} | {der_str(s['der_activity'])} | "
                   f"{der_str(s['der_asr_utterances'])} |")
    for r in reports:
        out += ["", f"## {r['stem']}", "",
                f"Mapping: `{json.dumps(r['speaker_mapping'])}`  ", f"GT text: {r['gt_text_source']}", "",
                "| id | time | GT spk | pred spk | | WER | GT text | ASR text |", "|---|---|---|---|---|---|---|---|"]
        for u in r["utterances"]:
            mark = {"ok": "✅", "wrong": "❌", "unknown": "❔", "missed": "∅"}[u["speaker_status"]]
            pred = u["hyp_speaker"] + (f" ({u['hyp_speaker_raw']})" if u["hyp_speaker_raw"] else "")
            out.append(f"| {u['id']} | {u['start']:.1f}-{u['end']:.1f} | {u['gt_speaker']} | {pred} | {mark} | "
                       f"{pct(u['wer'])} | {u['gt_text'].replace('|', '/')} | {u['hyp_text']} |")
    path.write_text("\n".join(out) + "\n")


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Utterance Benchmark</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--ok:#15803d;--bad:#b91c1c;--unk:#a16207;--okbg:#dcfce7;--badbg:#fee2e2;--unkbg:#fef9c3}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--card:#292524;--ok:#4ade80;--bad:#f87171;--unk:#facc15;--okbg:#14532d;--badbg:#7f1d1d;--unkbg:#713f12}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 6px}.mut{color:var(--mut)}
.tiles{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.tile b{display:block;font-size:18px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--card)}
td.n{white-space:nowrap;font-variant-numeric:tabular-nums}
.ok{background:var(--okbg)}.wrong{background:var(--badbg)}.unknown,.missed{background:var(--unkbg)}
.sub{color:var(--bad);font-weight:600}.ins{color:var(--bad);text-decoration:underline}.del{color:var(--bad);text-decoration:line-through;opacity:.7}
select{font:inherit;padding:4px}
</style></head><body>
<h1>Utterance benchmark</h1>
<div class="mut">GT text is silver (offline Whisper) until verified. Red bold = substitution, underline = inserted, strike = deleted GT word.</div>
<p><label>File <select id="f"></select></label> <label><input type="checkbox" id="bad"> only errors</label></p>
<div id="out"></div>
<script>
const R = __DATA__;
const pct = v => v == null ? "—" : (100 * v).toFixed(1) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const derS = d => d ? pct(d.der) + ` <span class=mut>(miss ${pct(d.missed)}, FA ${pct(d.false_alarm)}, conf ${pct(d.confusion)})</span>` : "—";
const sel = document.getElementById("f"), bad = document.getElementById("bad");
R.forEach((r, i) => sel.add(new Option(r.stem, i)));
function diff(u) {
  return u.diff.map(([op, r, h]) => op == "ok" ? esc(h) : op == "sub" ? `<span class=sub title="GT: ${esc(r)}">${esc(h)}</span>`
    : op == "ins" ? `<span class=ins>${esc(h)}</span>` : `<span class=del>${esc(r)}</span>`).join(" ");
}
function draw() {
  const r = R[sel.value], s = r.summary;
  const tiles = [["WER", pct(s.wer)], ["Speaker acc (utt)", pct(s.speaker_acc_utt)], ["Speaker acc (words)", pct(s.speaker_acc_words)],
    ["ok / wrong / unknown / missed", `${s.speaker_ok_utt} / ${s.speaker_wrong_utt} / ${s.speaker_unknown_utt} / ${s.missed_utt}`],
    ["GT / ASR utterances", `${s.gt_utterances} / ${s.hyp_utterances}`], ["GT verified", r.gt_verified],
    ["DER activity", derS(s.der_activity)], ["DER ASR utt", derS(s.der_asr_utterances)]];
  const conf = Object.entries(r.confusion).map(([g, c]) => `${g}: ` + Object.entries(c).map(([h, n]) => `${h}×${n}`).join(", ")).join(" · ");
  const rows = r.utterances.filter(u => !bad.checked || u.speaker_status != "ok" || (u.wer || 0) > 0).map(u => `<tr class="${u.speaker_status}">
    <td class=n>${u.id}${u.overlap ? " ⧉" : ""}${u.verified ? " ✔" : ""}</td><td class=n>${u.start.toFixed(1)}–${u.end.toFixed(1)}</td>
    <td>${esc(u.gt_speaker)}</td><td>${esc(u.hyp_speaker)} <span class=mut>${esc(u.hyp_speaker_raw)}</span></td>
    <td class=n>${pct(u.wer)}</td><td>${esc(u.gt_text)}</td><td>${diff(u)}</td></tr>`).join("");
  document.getElementById("out").innerHTML = `<h2>${esc(r.audio)}</h2>
    <div class=tiles>${tiles.map(([k, v]) => `<div class=tile><span class=mut>${k}</span><b>${v}</b></div>`).join("")}</div>
    <div class=mut>Mapping ${esc(JSON.stringify(r.speaker_mapping))} · Confusion ${esc(conf)} · ⧉ = GT overlap</div>
    <div class=wrap><table><thead><tr><th>id</th><th>time</th><th>GT spk</th><th>pred spk</th><th>WER</th><th>GT text</th><th>ASR text (diff)</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}
sel.onchange = bad.onchange = draw; draw();
</script></body></html>
"""


def main() -> None:
    stems = sys.argv[1:] or sorted(p.stem for p in (ROOT / "hyp").glob("*.json") if (ROOT / "gt" / p.name).exists())
    if not stems:
        raise SystemExit("no stem has both gt/<stem>.json and hyp/<stem>.json")
    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    reports = []
    for stem in stems:
        rep = score_file(stem)
        (res_dir / f"{stem}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
        s = rep["summary"]
        print(f"{stem}: WER {pct(s['wer'])}, speaker acc {pct(s['speaker_acc_utt'])} "
              f"(ok {s['speaker_ok_utt']} wrong {s['speaker_wrong_utt']} unknown {s['speaker_unknown_utt']} "
              f"missed {s['missed_utt']}), DER activity {der_str(s['der_activity'])}")
        reports.append(rep)
    # Reports always cover every scored file, not only the ones passed on the command line.
    all_reports = [json.loads(p.read_text()) for p in sorted(res_dir.glob("*.json"))]
    write_markdown(all_reports, res_dir / "REPORT.md")
    (res_dir / "report.html").write_text(HTML.replace("__DATA__", json.dumps(all_reports, ensure_ascii=False)
                                                      .replace("</", "<\\/")))
    print(f"wrote {res_dir / 'REPORT.md'} and {res_dir / 'report.html'}")


if __name__ == "__main__":
    main()
