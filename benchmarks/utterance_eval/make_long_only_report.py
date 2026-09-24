#!/usr/bin/env python3
"""Derive a 'long utterances only' speaker-attribution report from the Jetson's
TEST_compose_mp3 results (results/<stem>.json, written by score.py) -- the last
HTML the Jetson-side benchmark produced (confirmed by mtime: TEST_compose_mp3.html
17:00:19 > compare.html 17:00:17 > results/report.html 17:00:15).

Filter: GT utterances with >= 6 normalised words (the user's own threshold for this
request). Note this is a half-step broader than the site's own "WER long" definition
(score.py: LONG_MORE_THAN = 6, i.e. STRICTLY more than 6 / 7+ words) -- adds 6
borderline 6-word utterances across the 3 files (4 in wer_es, 2 in spa_0018, 0 in
spa_0019) that the published WER-long/speaker-acc-long numbers do not include.

Reuses each utterance's already-computed speaker_status (ok/wrong/unknown/missed)
and word diff, unchanged -- this page only changes which utterances are shown and
how accuracy is aggregated (>=6-word GT utterances only, missed excluded from
accuracy, same as the site's own rule).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEMS = [
    "wer_es__es_es_weather_wer",
    "callhome_spa_snr20__spa_0019_4spk_snr20",
    "callhome_spa_snr20__spa_0018_2spk_snr20",
]
THRESHOLD = 6  # words_ref >= THRESHOLD counts as "long" for this page

STATUSES = [("ok", "✓", "right speaker"), ("wrong", "✗", "wrong speaker"),
            ("unknown", "?", "unknown"), ("missed", "∅", "not transcribed")]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def diff_html(u):
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


def chip(u):
    sym = {st: s for st, s, _ in STATUSES}[u["speaker_status"]]
    who = u["hyp_speaker"] or "—"
    raw = f' title="pipeline id: {esc(u["hyp_speaker_raw"])}"' if u["hyp_speaker_raw"] else ""
    return f'<span class="chip {u["speaker_status"]}"{raw}><span class="s">{sym}</span>{esc(who)}</span>'


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
.badge{font-size:11px;color:var(--muted);border:1px solid var(--grid);border-radius:4px;padding:0 4px;margin-left:4px}
"""


def main():
    reports = [json.loads((HERE / "results" / f"{s}.json").read_text()) for s in STEMS]

    per_file = []
    all_long = []
    for r in reports:
        long_rows = [u for u in r["utterances"] if u["words_ref"] >= THRESHOLD]
        for u in long_rows:
            u["_stem"] = r["stem"]
        all_long += long_rows
        scored = [u for u in long_rows if u["speaker_status"] != "missed"]
        status = {st: sum(1 for u in long_rows if u["speaker_status"] == st) for st, _, _ in STATUSES}
        ref = sum(u["words_ref"] for u in long_rows)
        err = sum(u["sub"] + u["del"] + u["ins"] for u in long_rows)
        per_file.append({
            "stem": r["stem"], "n_long": len(long_rows), "n_gt": len(r["utterances"]),
            "wer_long": round(err / ref, 4) if ref else None,
            "acc_long": round(status["ok"] / len(scored), 4) if scored else None,
            "status": status,
        })

    scored_all = [u for u in all_long if u["speaker_status"] != "missed"]
    status_all = {st: sum(1 for u in all_long if u["speaker_status"] == st) for st, _, _ in STATUSES}
    ref_all = sum(u["words_ref"] for u in all_long)
    err_all = sum(u["sub"] + u["del"] + u["ins"] for u in all_long)
    pool = {
        "n_long": len(all_long),
        "wer_long": round(err_all / ref_all, 4) if ref_all else None,
        "acc_long": round(status_all["ok"] / len(scored_all), 4) if scored_all else None,
        "status": status_all,
    }

    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width,initial-scale=1">'
         '<title>Long utterances only — speaker attribution</title>'
         f'<style>{CSS}</style></head><body><main>',
         '<h1>Long utterances only — speaker attribution accuracy</h1>',
         '<p class="sub">Derived from <code>TEST_compose_mp3.html</code> (the last HTML this benchmark produced — '
         f'{", ".join(s for s in STEMS)}), filtered to GT utterances with <b>&ge;{THRESHOLD} normalised words</b>. '
         'This threshold is half a word broader than the site\'s own "WER long" rule '
         f'(<code>score.py</code>: strictly more than {THRESHOLD}), so it includes a handful of exactly-{THRESHOLD}-word '
         'utterances the published WER-long/speaker-acc-long numbers exclude. Speaker accuracy excludes '
         '<code>missed</code> (untranscribed) utterances, same rule as the source page. Jetson pipeline only — '
         'no Android numbers are mixed in here.</p>']

    h.append('<h2>Summary — long utterances (&ge;6 GT words)</h2><div class="card"><table><thead><tr>'
              '<th>file</th><th>long / GT utt</th><th>WER (long)</th><th>speaker acc (long)</th>'
              '<th>ok / wrong / unknown / missed</th></tr></thead><tbody>')
    for f in per_file:
        h.append(f'<tr><td>{esc(f["stem"])}</td><td class="num">{f["n_long"]} / {f["n_gt"]}</td>'
                  f'<td class="num">{pct(f["wer_long"])}</td><td class="num">{pct(f["acc_long"])}</td>'
                  f'<td class="num">{f["status"]["ok"]} / {f["status"]["wrong"]} / {f["status"]["unknown"]} / {f["status"]["missed"]}</td></tr>')
    h.append(f'<tr class="pooled"><td><b>all (pooled)</b></td><td class="num">{pool["n_long"]}</td>'
              f'<td class="num"><b>{pct(pool["wer_long"])}</b></td><td class="num"><b>{pct(pool["acc_long"])}</b></td>'
              f'<td class="num">{pool["status"]["ok"]} / {pool["status"]["wrong"]} / {pool["status"]["unknown"]} / {pool["status"]["missed"]}</td></tr>')
    h.append('</tbody></table></div>')

    h.append('<h2>Who got each long utterance</h2><div class="legend">'
              + "".join(f'<span><span class="sw" style="background:var(--{c})"></span>{sym} {label}</span>'
                        for (st, sym, label), c in zip(STATUSES, ["good", "crit", "warn", "none"]))
              + '</div><div class="card"><table><thead><tr><th>file</th><th></th></tr></thead><tbody>')
    for f in per_file:
        s = f["status"]
        n = sum(s.values()) or 1
        segs = "".join(f'<div class="seg {st}" style="flex:{s[st]} 1 0" title="{label}: {s[st]} of {n}">{sym} {s[st]}</div>'
                        for st, sym, label in STATUSES if s[st])
        h.append(f'<tr><td>{esc(f["stem"])}</td><td><div class="stack">{segs}</div></td></tr>')
    ps = pool["status"]; pn = sum(ps.values()) or 1
    psegs = "".join(f'<div class="seg {st}" style="flex:{ps[st]} 1 0" title="{label}: {ps[st]} of {pn}">{sym} {ps[st]}</div>'
                     for st, sym, label in STATUSES if ps[st])
    h.append(f'<tr class="pooled"><td><b>all (pooled)</b></td><td><div class="stack">{psegs}</div></td></tr>')
    h.append('</tbody></table></div>')

    for r in reports:
        long_rows = [u for u in r["utterances"] if u["words_ref"] >= THRESHOLD]
        if not long_rows:
            continue
        h.append(f'<h2>{esc(r["stem"])}: long utterances only ({len(long_rows)} of {len(r["utterances"])} GT)</h2>'
                  f'<p class="sub">{esc(r["audio"])} · mapping <code>{esc(json.dumps(r["speaker_mapping"]))}</code></p>'
                  '<div class="card"><table><thead><tr><th>id</th><th>time</th><th>words</th><th>GT</th><th>GT text</th>'
                  '<th>pred</th><th>WER</th><th>ASR text (diff)</th></tr></thead><tbody>')
        for u in long_rows:
            h.append(
                f'<tr><td class="num">{esc(u["id"])}</td><td class="num mut">{u["start"]:.1f}–{u["end"]:.1f}</td>'
                f'<td class="num">{u["words_ref"]}</td>'
                f'<td><b>{esc(u["gt_speaker"])}</b></td>'
                f'<td class="txt">{esc(u["gt_text"])}</td>'
                f'<td>{chip(u)}</td><td class="num">{pct(u["wer"])}</td>'
                f'<td class="txt">{diff_html(u)}</td></tr>')
        h.append('</tbody></table></div>')

    h.append('</main></body></html>')
    out_path = HERE / "TEST_compose_mp3_long_only.html"  # written next to TEST_compose_mp3.html
    out_path.write_text("".join(h))
    print(f"wrote {out_path}")
    for f in per_file:
        print(f'{f["stem"]}: {f["n_long"]} long utt, WER long {pct(f["wer_long"])}, speaker acc long {pct(f["acc_long"])}, '
              f'ok/wrong/unknown/missed {f["status"]["ok"]}/{f["status"]["wrong"]}/{f["status"]["unknown"]}/{f["status"]["missed"]}')
    print(f'ALL POOLED: {pool["n_long"]} long utt, WER long {pct(pool["wer_long"])}, speaker acc long {pct(pool["acc_long"])}, '
          f'ok/wrong/unknown/missed {pool["status"]["ok"]}/{pool["status"]["wrong"]}/{pool["status"]["unknown"]}/{pool["status"]["missed"]}')


if __name__ == "__main__":
    main()
