#!/usr/bin/env python3
"""Compare scored runs of the same excerpt under different audio conditions.

Reads results/*.json (written by score.py), groups them by excerpt (spa_0019_4spk,
spa_0018_2spk, ...) and condition (snr20, clean, ...), and writes:

    COMPARE.md     tables with WER all, WER long, speaker accuracy, DER and the change
    compare.html   the same as a visual page, plus utterance-by-utterance side by side

    python3 compare.py                      # conditions snr20 vs clean
    python3 compare.py --a snr10 --b clean
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM_RE = re.compile(r"^(?P<corpus>.+?)_(?P<cond>clean|snr\d+)__(?P<base>.+)_(?P=cond)$")
STATUSES = [("ok", "✓", "right speaker"), ("wrong", "✗", "wrong speaker"),
            ("unknown", "?", "unknown"), ("missed", "∅", "not transcribed")]

# (key, label, higher_is_better); values are fractions 0..1
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


def split_stem(stem: str) -> tuple[str, str]:
    m = STEM_RE.match(stem)
    return (m["base"], m["cond"]) if m else (stem, "clean")


def metric(summary: dict, key: str):
    if key.startswith("der_"):
        return ((summary.get("der_activity") or {}).get(key.removeprefix("der_")) or {}).get("der")
    return summary.get(key)


def pooled(reports: list[dict]) -> dict:
    """Pool several files into one summary (word-weighted WER, utterance-weighted accuracy)."""
    s = [r["summary"] for r in reports]
    ref = sum(x["ref_words"] for x in s)
    lref = sum(x.get("long_ref_words", 0) for x in s)
    utts = [u for r in reports for u in r["utterances"]]
    scored = [u for u in utts if u["speaker_status"] != "missed"]
    lscored = [u for u in scored if u.get("long")]
    dur = sum(r["duration_sec"] for r in reports)
    der_activity = {}
    for variant in ("fair", "strict"):
        ders = [(metric(x, f"der_{variant}"), r["duration_sec"]) for x, r in zip(s, reports)
                if metric(x, f"der_{variant}") is not None]
        if ders and dur:
            der_activity[variant] = {"der": sum(d * w for d, w in ders) / dur}
    out = {
        "ref_words": ref,
        "wer": sum(x["sub"] + x["del"] + x["ins"] for x in s) / ref if ref else None,
        "wer_long": sum(x.get("long_errors", 0) for x in s) / lref if lref else None,
        "long_utterances": sum(x.get("long_utterances", 0) for x in s),
        "speaker_acc_utt": sum(u["speaker_status"] == "ok" for u in scored) / len(scored) if scored else None,
        "speaker_acc_long": sum(u["speaker_status"] == "ok" for u in lscored) / len(lscored) if lscored else None,
        "der_activity": der_activity or None,
        "gt_utterances": len(utts),
        "hyp_utterances": sum(x["hyp_utterances"] for x in s),
    }
    for st, _, _ in STATUSES:
        out[f"n_{st}"] = sum(u["speaker_status"] == st for u in utts)
    return out


def status_counts(summary: dict) -> dict:
    if "n_ok" in summary:
        return {st: summary[f"n_{st}"] for st, _, _ in STATUSES}
    return {"ok": summary["speaker_ok_utt"], "wrong": summary["speaker_wrong_utt"],
            "unknown": summary["speaker_unknown_utt"], "missed": summary["missed_utt"]}


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def delta(va, vb, higher_better: bool) -> tuple[str, str]:
    """Change from A to B in percentage points -> (text, better|worse|same|na)."""
    if va is None or vb is None:
        return "—", "na"
    d = 100 * (vb - va)
    if abs(d) < 0.05:
        return "0.0 pts", "same"
    better = (d > 0) == higher_better
    return f"{'+' if d > 0 else '−'}{abs(d):.1f} pts", "better" if better else "worse"


# ----------------------------------------------------------------------------

def build(a: str, b: str) -> tuple[list[dict], list[str]]:
    reports = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
    groups: dict[str, dict[str, dict]] = {}
    for r in reports:
        base, cond = split_stem(r["stem"])
        groups.setdefault(base, {})[cond] = r
    rows = []
    for base, by_cond in sorted(groups.items(), key=lambda kv: (len(kv[1]) < 2, kv[0])):
        if a not in by_cond and b not in by_cond:
            continue
        rows.append({"name": base, "a": by_cond.get(a), "b": by_cond.get(b),
                     "sa": by_cond[a]["summary"] if a in by_cond else None,
                     "sb": by_cond[b]["summary"] if b in by_cond else None})
    paired = [r for r in rows if r["a"] and r["b"]]
    if len(paired) > 1:
        rows.append({"name": f"all paired ({len(paired)} files)", "a": None, "b": None, "pooled": True,
                     "sa": pooled([r["a"] for r in paired]), "sb": pooled([r["b"] for r in paired])})
    return rows, [a, b]


def write_md(rows: list[dict], conds: list[str], path: Path) -> None:
    a, b = conds
    arrow = {"better": "🟢", "worse": "🔴", "same": "⚪", "na": ""}
    out = [f"# Comparison: `{a}` vs `{b}`", "",
           f"Change = `{b}` minus `{a}`, in percentage points. 🟢 better, 🔴 worse.", "",
           "| excerpt | metric | " + f"{a} | {b} | change |", "|---|---|---|---|---|"]
    for r in rows:
        for key, label, hb in METRICS:
            va = metric(r["sa"], key) if r["sa"] else None
            vb = metric(r["sb"], key) if r["sb"] else None
            txt, kind = delta(va, vb, hb)
            out.append(f"| {r['name'] if key == 'wer' else ''} | {label} | {pct(va)} | {pct(vb)} | {arrow[kind]} {txt} |")
    out += ["", "## Speaker status per utterance (✓ ok / ✗ wrong / ? unknown / ∅ missed)", "",
            f"| excerpt | {a} | {b} |", "|---|---|---|"]
    for r in rows:
        cells = []
        for s in (r["sa"], r["sb"]):
            cells.append("—" if not s else " ".join(f"{sym} {status_counts(s)[st]}" for st, sym, _ in STATUSES))
        out.append(f"| {r['name']} | {cells[0]} | {cells[1]} |")
    for r in rows:
        if not (r.get("a") and r.get("b")):
            continue
        out += ["", f"## {r['name']}: utterance by utterance", "",
                f"| id | GT spk | GT text | {a} spk | {a} WER | {b} spk | {b} WER | change |",
                "|---|---|---|---|---|---|---|---|"]
        for ua, ub in zip(r["a"]["utterances"], r["b"]["utterances"]):
            sym = {st: s for st, s, _ in STATUSES}
            chg = ("fixed" if ub["speaker_status"] == "ok" and ua["speaker_status"] != "ok" else
                   "broke" if ua["speaker_status"] == "ok" and ub["speaker_status"] != "ok" else "")
            out.append(f"| {ua['id']} | {ua['gt_speaker']} | {ua['gt_text'].replace('|', '/')} | "
                       f"{sym[ua['speaker_status']]} {ua['hyp_speaker']} | {pct(ua['wer'])} | "
                       f"{sym[ub['speaker_status']]} {ub['hyp_speaker']} | {pct(ub['wer'])} | {chg} |")
    path.write_text("\n".join(out) + "\n")


CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e0;
--a:#2a78d6;--b:#eb6834;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--none:#b9b8b1;--track:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--card:#232322;--ink:#fff;
--ink2:#c3c2b7;--muted:#8f8e86;--grid:#383835;--a:#3987e5;--b:#d95926;--track:#2e2e2c;--none:#6b6a64}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#232322;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8e86;--grid:#383835;
--a:#3987e5;--b:#d95926;--track:#2e2e2c;--none:#6b6a64}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif;padding:24px 16px}
main{max-width:1180px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}p.sub{color:var(--ink2);margin:0 0 16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:8px 0 16px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:4px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%}
th{font-weight:600;color:var(--ink2);font-size:12px;text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid var(--grid);vertical-align:middle}
tr:last-child td{border-bottom:0}
tr.grp td{background:var(--track);font-weight:600;padding-top:9px}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:flex;align-items:center;gap:8px;min-width:170px}
.track{flex:1;height:8px;background:var(--track);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:0 4px 4px 0}
.fill.a{background:var(--a)}.fill.b{background:var(--b)}
.val{width:52px;text-align:right;font-variant-numeric:tabular-nums}
.chg{white-space:nowrap;font-variant-numeric:tabular-nums}
.chg .ic{display:inline-block;width:16px;text-align:center;font-weight:700}
.chg.better .ic{color:var(--good)}.chg.worse .ic{color:var(--crit)}.chg.same .ic,.chg.na .ic{color:var(--muted)}
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
.tag{font-size:11px;font-weight:700;border-radius:4px;padding:1px 6px;white-space:nowrap}
.tag.fixed{color:var(--good);border:1px solid var(--good)}.tag.broke{color:var(--crit);border:1px solid var(--crit)}
.txt{min-width:220px;max-width:360px}.asr{color:var(--ink2);font-size:13px}
.mut{color:var(--muted)}
details{margin-top:12px}summary{cursor:pointer;font-weight:600;padding:6px 0}
.tools{display:flex;gap:12px;align-items:center;margin:8px 0;color:var(--ink2);font-size:13px}
"""


def esc(s) -> str:
    return html.escape(str(s))


def bar_cell(v, cls: str) -> str:
    if v is None:
        return '<td><div class="bar"><div class="track"></div><span class="val mut">—</span></div></td>'
    w = max(0.0, min(1.0, v)) * 100
    return (f'<td><div class="bar" title="{pct(v)}"><div class="track"><div class="fill {cls}" style="width:{w:.1f}%">'
            f'</div></div><span class="val">{pct(v)}</span></div></td>')


def stack(summary: dict | None) -> str:
    if not summary:
        return '<span class="mut">—</span>'
    c = status_counts(summary)
    n = sum(c.values()) or 1
    segs = "".join(
        f'<div class="seg {st}" style="flex:{c[st]} 1 0" title="{label}: {c[st]} of {n}">{sym} {c[st]}</div>'
        for st, sym, label in STATUSES if c[st])
    return f'<div class="stack">{segs}</div>'


def chip(u: dict) -> str:
    sym = {st: s for st, s, _ in STATUSES}[u["speaker_status"]]
    who = u["hyp_speaker"] or "—"
    raw = f' title="pipeline id: {esc(u["hyp_speaker_raw"])}"' if u["hyp_speaker_raw"] else ""
    return f'<span class="chip {u["speaker_status"]}"{raw}><span class="s">{sym}</span>{esc(who)}</span>'


def write_html(rows: list[dict], conds: list[str], path: Path) -> None:
    a, b = conds
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Noise Comparison</title>'
         f'<style>{CSS}</style></head><body><main>',
         f'<h1>{esc(a)} vs {esc(b)}</h1>',
         f'<p class="sub">Same excerpts, same GT, pipeline run twice. Change = {esc(b)} minus {esc(a)}, in percentage points. '
         f'WER long only counts GT utterances with more than 6 words.</p>',
         f'<div class="legend"><span><span class="sw" style="background:var(--a)"></span>{esc(a)}</span>'
         f'<span><span class="sw" style="background:var(--b)"></span>{esc(b)}</span>'
         f'<span><b style="color:var(--good)">▼/▲</b> better</span><span><b style="color:var(--crit)">▼/▲</b> worse</span></div>',
         '<h2>Metrics</h2><div class="card"><table><thead><tr><th>metric</th>'
         f'<th>{esc(a)}</th><th>{esc(b)}</th><th>change</th></tr></thead><tbody>']
    for r in rows:
        n_long = (r["sb"] or r["sa"]).get("long_utterances", 0)
        h.append(f'<tr class="grp"><td colspan="4">{esc(r["name"])} '
                 f'<span class="mut">· {(r["sb"] or r["sa"])["gt_utterances"]} GT utterances, {n_long} long</span></td></tr>')
        for key, label, hb in METRICS:
            va = metric(r["sa"], key) if r["sa"] else None
            vb = metric(r["sb"], key) if r["sb"] else None
            txt, kind = delta(va, vb, hb)
            icon = {"better": "▲" if (vb or 0) > (va or 0) else "▼", "worse": "▲" if (vb or 0) > (va or 0) else "▼"}.get(kind, "·")
            hint = "lower is better" if not hb else "higher is better"
            h.append(f'<tr><td>{label} <span class="mut" title="{hint}">{"↓" if not hb else "↑"}</span></td>'
                     f'{bar_cell(va, "a")}{bar_cell(vb, "b")}'
                     f'<td class="chg {kind}"><span class="ic">{icon}</span>{txt}</td></tr>')
    h.append('</tbody></table></div>')

    h.append('<h2>Who got each utterance</h2><div class="legend">'
             + "".join(f'<span><span class="sw" style="background:var(--{c})"></span>{sym} {label}</span>'
                       for (st, sym, label), c in zip(STATUSES, ["good", "crit", "warn", "none"]))
             + '</div><div class="card"><table><thead><tr><th>excerpt</th>'
             f'<th>{esc(a)}</th><th>{esc(b)}</th></tr></thead><tbody>')
    for r in rows:
        h.append(f'<tr><td>{esc(r["name"])}</td><td>{stack(r["sa"])}</td><td>{stack(r["sb"])}</td></tr>')
    h.append('</tbody></table></div>')

    for r in rows:
        if not (r.get("a") and r.get("b")):
            continue
        ua_list, ub_list = r["a"]["utterances"], r["b"]["utterances"]
        fixed = sum(ub["speaker_status"] == "ok" and ua["speaker_status"] != "ok" for ua, ub in zip(ua_list, ub_list))
        broke = sum(ua["speaker_status"] == "ok" and ub["speaker_status"] != "ok" for ua, ub in zip(ua_list, ub_list))
        h.append(f'<h2>{esc(r["name"])}: utterance by utterance</h2>'
                 f'<p class="sub">Speaker in {esc(b)} vs {esc(a)}: <span class="tag fixed">{fixed} fixed</span> '
                 f'<span class="tag broke">{broke} broke</span> · mapping {esc(a)} '
                 f'<code>{esc(json.dumps(r["a"]["speaker_mapping"]))}</code>, {esc(b)} '
                 f'<code>{esc(json.dumps(r["b"]["speaker_mapping"]))}</code></p>'
                 '<div class="card"><table><thead><tr><th>id</th><th>time</th><th>GT</th><th>GT text</th>'
                 f'<th>{esc(a)}</th><th>WER</th><th>{esc(b)}</th><th>WER</th><th></th></tr></thead><tbody>')
        for ua, ub in zip(ua_list, ub_list):
            tag = ('<span class="tag fixed">fixed</span>' if ub["speaker_status"] == "ok" and ua["speaker_status"] != "ok"
                   else '<span class="tag broke">broke</span>' if ua["speaker_status"] == "ok" and ub["speaker_status"] != "ok"
                   else "")
            long_mark = ' <span class="mut" title="more than 6 words">long</span>' if ua.get("long") else ""
            h.append(
                f'<tr><td class="num">{esc(ua["id"])}</td><td class="num mut">{ua["start"]:.1f}–{ua["end"]:.1f}</td>'
                f'<td><b>{esc(ua["gt_speaker"])}</b></td>'
                f'<td class="txt">{esc(ua["gt_text"])}{long_mark}'
                f'<div class="asr" title="{esc(a)}"><span style="color:var(--a)">■</span> {esc(ua["hyp_text"]) or "<i>—</i>"}</div>'
                f'<div class="asr" title="{esc(b)}"><span style="color:var(--b)">■</span> {esc(ub["hyp_text"]) or "<i>—</i>"}</div></td>'
                f'<td>{chip(ua)}</td><td class="num">{pct(ua["wer"])}</td>'
                f'<td>{chip(ub)}</td><td class="num">{pct(ub["wer"])}</td><td>{tag}</td></tr>')
        h.append('</tbody></table></div>')
    h.append('</main></body></html>')
    path.write_text("".join(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="snr20", help="baseline condition")
    ap.add_argument("--b", default="clean", help="compared condition")
    args = ap.parse_args()
    rows, conds = build(args.a, args.b)
    write_md(rows, conds, ROOT / "COMPARE.md")
    write_html(rows, conds, ROOT / "compare.html")
    print(f"wrote {ROOT / 'COMPARE.md'} and {ROOT / 'compare.html'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
