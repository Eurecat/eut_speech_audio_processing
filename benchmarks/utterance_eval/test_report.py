#!/usr/bin/env python3
"""Visual HTML report for one set of scored files (no A/B comparison).

Reads results/<stem>.json for the given stems (default: the 3 files in
TEST_compose_mp3.md) and writes one page: metric bars per file + pooled,
a stacked ok/wrong/unknown/missed bar per file, and a full per-utterance
table with a word-level diff (substitution/insertion/deletion highlighted).

    python3 test_report.py                       # the 3 TEST_compose_mp3 files -> TEST_compose_mp3.html
    python3 test_report.py --out X.html s1 s2 s3
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from compare import STATUSES, esc, metric, pct, pooled, status_counts  # reuse the compare.py building blocks

ROOT = Path(__file__).resolve().parent
DEFAULT_STEMS = [
    "wer_es__es_es_weather_wer",
    "callhome_spa_snr20__spa_0019_4spk_snr20",
    "callhome_spa_snr20__spa_0018_2spk_snr20",
]
# der_fair/der_strict match ~/aimara-bench/benchmarks/scoring/metrics.py: fair = collar 0.25s, overlap
# skipped (comparable to published CALLHOME/AMI numbers); strict = collar 0s, overlap scored.
METRICS = [
    ("wer", "WER all", False),
    ("wer_long", "WER long", False),
    ("speaker_acc_utt", "Speaker acc", True),
    ("speaker_acc_long", "Speaker acc long", True),
    ("der_fair", "DER fair (activity)", False),
    ("der_strict", "DER strict (activity)", False),
]

CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e0;
--hue:#2a78d6;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--none:#b9b8b1;--track:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--card:#232322;--ink:#fff;
--ink2:#c3c2b7;--muted:#8f8e86;--grid:#383835;--hue:#3987e5;--track:#2e2e2c;--none:#6b6a64}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#232322;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8e86;--grid:#383835;
--hue:#3987e5;--track:#2e2e2c;--none:#6b6a64}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif;padding:24px 16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}p.sub{color:var(--ink2);margin:0 0 16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:8px 0 16px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:4px 0;overflow-x:auto;margin-bottom:20px}
table{border-collapse:collapse;width:100%}
th{font-weight:600;color:var(--ink2);font-size:12px;text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid var(--grid);vertical-align:middle}
tr:last-child td{border-bottom:0}
tr.grp td{background:var(--track);font-weight:600;padding-top:9px}
tr.pooled td{background:color-mix(in srgb,var(--hue) 10%,transparent);font-weight:600}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:flex;align-items:center;gap:8px;min-width:200px}
.track{flex:1;height:8px;background:var(--track);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:0 4px 4px 0;background:var(--hue)}
.val{width:52px;text-align:right;font-variant-numeric:tabular-nums}
.stack{display:flex;gap:2px;height:22px;min-width:260px}
.seg{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;min-width:0;overflow:hidden}
.seg:first-child{border-radius:4px 0 0 4px}.seg:last-child{border-radius:0 4px 4px 0}
.seg.ok{background:var(--good)}.seg.wrong{background:var(--crit)}.seg.unknown{background:var(--warn);color:#0b0b0b}.seg.missed{background:var(--none);color:#0b0b0b}
.chip{display:inline-flex;align-items:center;gap:4px;border-radius:999px;padding:1px 8px;font-size:12px;font-weight:600;white-space:nowrap}
.chip.ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--ink)}
.chip.wrong{background:color-mix(in srgb,var(--crit) 22%,transparent);color:var(--ink)}
.chip.unknown{background:color-mix(in srgb,var(--warn) 30%,transparent);color:var(--ink)}
.chip.missed{background:var(--track);color:var(--ink2)}
.chip .s{font-weight:800}
.chip.ok .s{color:var(--good)}.chip.wrong .s{color:var(--crit)}
.mut{color:var(--muted)}
.txt{min-width:220px;max-width:420px}
.sub{color:var(--crit);font-weight:600}.ins{color:var(--crit);text-decoration:underline}.del{color:var(--crit);text-decoration:line-through;opacity:.7}
.long{font-size:11px;color:var(--muted);border:1px solid var(--grid);border-radius:4px;padding:0 4px;margin-left:4px}
"""


def metric_val(summary: dict, key: str):
    return metric(summary, key) if key.startswith("der_") else summary.get(key)


def bar_cell(v) -> str:
    if v is None:
        return '<td><div class="bar"><div class="track"></div><span class="val mut">—</span></div></td>'
    w = max(0.0, min(1.0, v)) * 100
    return (f'<td><div class="bar" title="{pct(v)}"><div class="track"><div class="fill" style="width:{w:.1f}%">'
            f'</div></div><span class="val">{pct(v)}</span></div></td>')


def stack(summary: dict) -> str:
    c = status_counts(summary)
    n = sum(c.values()) or 1
    segs = "".join(
        f'<div class="seg {st}" style="flex:{c[st]} 1 0" title="{label}: {c[st]} of {n}">{sym} {c[st]}</div>'
        for st, sym, label in STATUSES if c[st])
    return f'<div class="stack">{segs}</div>' if segs else '<span class="mut">—</span>'


def chip(u: dict) -> str:
    sym = {st: s for st, s, _ in STATUSES}[u["speaker_status"]]
    who = u["hyp_speaker"] or "—"
    raw = f' title="pipeline id: {esc(u["hyp_speaker_raw"])}"' if u["hyp_speaker_raw"] else ""
    return f'<span class="chip {u["speaker_status"]}"{raw}><span class="s">{sym}</span>{esc(who)}</span>'


def diff_html(u: dict) -> str:
    parts = []
    for op, r, h in u["diff"]:
        if op == "ok":
            parts.append(esc(h))
        elif op == "sub":
            parts.append(f'<span class="sub" title="GT: {esc(r)}">{esc(h)}</span>')
        elif op == "ins":
            parts.append(f'<span class="ins">{esc(h)}</span>')
        elif op == "del":
            parts.append(f'<span class="del">{esc(r)}</span>')
    return " ".join(parts) or '<i class="mut">—</i>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", default=DEFAULT_STEMS)
    ap.add_argument("--out", type=Path, default=ROOT / "TEST_compose_mp3.html")
    ap.add_argument("--title", default="Test: docker-compose_mp3 pipeline (snr20)")
    args = ap.parse_args()

    reports = []
    for stem in args.stems:
        p = ROOT / "results" / f"{stem}.json"
        if not p.exists():
            raise SystemExit(f"missing {p} — run score.py (or run_test_mp3.sh) first")
        reports.append(json.loads(p.read_text()))

    pool = pooled(reports) if len(reports) > 1 else None

    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(args.title)}</title>'
         f'<style>{CSS}</style></head><body><main>',
         f'<h1>{esc(args.title)}</h1>',
         '<p class="sub">WER long only counts GT utterances with more than 6 normalised words. '
         'Speaker accuracy excludes missed (untranscribed) utterances. GT text is silver-reviewed by the user.</p>']

    h.append('<h2>Summary</h2><div class="card"><table><thead><tr><th>file</th>'
             + "".join(f'<th>{label}</th>' for _, label, _ in METRICS) + '<th>GT verified</th></tr></thead><tbody>')
    for r in reports:
        s = r["summary"]
        h.append(f'<tr><td>{esc(r["stem"])}<div class="mut" style="font-size:12px">'
                 f'{s["gt_utterances"]} GT utt, {s.get("long_utterances", 0)} long</div></td>'
                 + "".join(bar_cell(metric_val(s, k)) for k, _, _ in METRICS)
                 + f'<td class="num">{esc(r["gt_verified"])}</td></tr>')
    if pool:
        h.append(f'<tr class="pooled"><td><b>all (pooled)</b><div class="mut" style="font-size:12px">'
                 f'{pool["gt_utterances"]} GT utt, {pool.get("long_utterances", 0)} long</div></td>'
                 + "".join(bar_cell(metric_val(pool, k)) for k, _, _ in METRICS) + '<td></td></tr>')
    h.append('</tbody></table></div>')

    h.append('<h2>Who got each utterance</h2><div class="legend">'
             + "".join(f'<span><span class="sw" style="background:var(--{c})"></span>{sym} {label}</span>'
                       for (st, sym, label), c in zip(STATUSES, ["good", "crit", "warn", "none"]))
             + '</div><div class="card"><table><thead><tr><th>file</th><th></th></tr></thead><tbody>')
    for r in reports:
        h.append(f'<tr><td>{esc(r["stem"])}</td><td>{stack(r["summary"])}</td></tr>')
    if pool:
        h.append(f'<tr class="pooled"><td><b>all (pooled)</b></td><td>{stack(pool)}</td></tr>')
    h.append('</tbody></table></div>')

    for r in reports:
        h.append(f'<h2>{esc(r["stem"])}: utterance by utterance</h2>'
                 f'<p class="sub">{esc(r["audio"])} · mapping <code>{esc(json.dumps(r["speaker_mapping"]))}</code> · '
                 f'{esc(r["gt_text_source"])}</p>'
                 '<div class="card"><table><thead><tr><th>id</th><th>time</th><th>GT</th><th>GT text</th>'
                 '<th>pred</th><th>WER</th><th>ASR text (diff)</th></tr></thead><tbody>')
        for u in r["utterances"]:
            long_mark = '<span class="long">long</span>' if u.get("long") else ""
            h.append(
                f'<tr><td class="num">{esc(u["id"])}</td><td class="num mut">{u["start"]:.1f}–{u["end"]:.1f}</td>'
                f'<td><b>{esc(u["gt_speaker"])}</b></td>'
                f'<td class="txt">{esc(u["gt_text"])}{long_mark}</td>'
                f'<td>{chip(u)}</td><td class="num">{pct(u["wer"])}</td>'
                f'<td class="txt">{diff_html(u)}</td></tr>')
        h.append('</tbody></table></div>')

    h.append('</main></body></html>')
    args.out.write_text("".join(h))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
