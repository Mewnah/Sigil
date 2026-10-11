#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own-mic ranking of STT engines per language (P7-A decision rule, user decision 2026-10-09).

A streamer's own captions come from their own microphone, so the defaults are chosen on the clean
results plus the mic-side conditions (``room_light`` and ``mic_codec`` by default). The bleed
conditions (music, effects, babble, VRChat mix) play no part in that choice; they are tabulated
next to it only because the "other players' captions" (loopback) feature hears VRChat's own audio.

Input: the ``items.jsonl`` files that ``score_stt.py --items`` writes (one line per label and
corpus item, with the edit counts). Labels are ``<report>/<engine>``; the engine id after the last
slash names the engine.

    python campaign/own_mic.py --items scores_a/items.jsonl scores_b/items.jsonl \
        [--own clean,room_light,mic_codec] [--bleed music_snr10,babble_snr10,vrchat_mix] \
        [--min-utterances 20] [--md out.md] [--json out.json]

Rate = pooled corpus rate (sum of errors over sum of reference units). The 95 % intervals and the
paired differences resample **utterances** (all conditions of an utterance together), 1000 resamples,
seed 20261002 (the scorer's). The decision rule printed with the table:

1. best = the lowest own-mic rate among the engines that cover the language (engines with fewer than
   ``--min-utterances`` utterances, such as the 12-utterance Nemotron sample, are listed but not ranked);
2. an engine **ties** the best when the paired interval of (engine minus best) contains zero **and** its
   point estimate is at most ``--margin`` (1.5 points WER / CER) above the best (a wide interval on 90
   utterances must not let a clearly worse engine in);
3. among the tied engines the **cheapest** wins (the ``--cost`` order, light to heavy). An engine that is
   significantly worse than the best, or more than the margin worse, is out whatever it costs;
4. the engines of ``--exclude-finals`` (default: Granite, which writes no case or punctuation, and Nemotron, a streaming
   partials engine) are never a caption final: they stay in the tables and in the comparison, but they are removed from
   the tie set, so no ``--cost`` order or measured cost can make one the default. When no final engine ties the best
   (the best is an excluded one by a clear margin), the best-ranked final engine takes the language and the result says so;
5. **incomplete data withholds the default.** An engine (not excluded) that the files hold for a language but that lacks
   one of the ``--own`` conditions is not ranked, so a winner among the rest could be wrong (for example Qwen3-ASR 0.6B in
   Korean when the file with its ``room_light`` / ``mic_codec`` runs was not passed). The language then gets no winner (the
   would-be one is kept as ``provisional_winner``), the tool says so loudly on stderr and exits 2. ``--allow-incomplete``
   keeps the provisional winner and still warns. An engine with all conditions but fewer than ``--min-utterances``
   utterances is only listed (``too_small``): a 12-utterance sample cannot decide anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

DEFAULT_SEED = 20261002
DEFAULT_B = 1000
# Relative cost of one audio second on the CPU, light to heavy (campaign `speed_utt`): the tie-break.
DEFAULT_COST = ["sensevoice", "moonshine", "parakeet-ultra", "parakeet-v3", "granite-cpu", "qwen3-asr-0.6b", "nemotron-560"]
# Engines that are never a caption final, whatever they score or cost: Granite TurboCTC writes lower case without punctuation,
# Nemotron is a streaming partials engine (its own finals are the one-pass route of a 4-core PC, not a second pass).
DEFAULT_EXCLUDE_FINALS = ["granite-cpu", "nemotron-560"]


def load_items(paths: list[str]) -> dict[str, dict[str, dict]]:
    """{engine: {item id: {utterance, language, condition, errors, units}}} over all files (later files win)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                c = r.get("counts")
                if c is None:
                    continue
                engine = r["label"].rsplit("/", 1)[-1]
                item = r["item"]
                utt, _, cond = item.rpartition("/")
                out[engine][item] = {
                    "utterance": utt,
                    "language": r["language"],
                    "condition": r["condition"],
                    "errors": c["errors"],
                    "units": c["ref_units"],
                }
    return out


def pooled(items: list[dict]) -> float | None:
    u = sum(i["units"] for i in items)
    return None if u == 0 else sum(i["errors"] for i in items) / u


def cluster_arrays(items: list[dict], utterances: list[str]):
    """Per-utterance error and unit sums over ``utterances`` (aligned), for the cluster bootstrap."""
    err = defaultdict(float)
    uni = defaultdict(float)
    for i in items:
        err[i["utterance"]] += i["errors"]
        uni[i["utterance"]] += i["units"]
    return np.array([err[u] for u in utterances]), np.array([uni[u] for u in utterances])


def ci_of(items: list[dict], b: int, seed: int):
    utts = sorted({i["utterance"] for i in items})
    e, u = cluster_arrays(items, utts)
    if u.sum() == 0:
        return None, None, None
    rate = float(e.sum() / u.sum())
    if len(utts) < 2:
        return rate, rate, rate
    idx = np.random.default_rng(seed).integers(0, len(utts), size=(b, len(utts)))
    with np.errstate(divide="ignore", invalid="ignore"):
        st = e[idx].sum(axis=1) / u[idx].sum(axis=1)
    st = st[np.isfinite(st)]
    lo, hi = np.percentile(st, [2.5, 97.5])
    return rate, float(lo), float(hi)


def paired(a_items: list[dict], b_items: list[dict], b: int, seed: int):
    """(a minus b) over the utterances both have, cluster bootstrap. Returns (delta, lo, hi, n_utterances)."""
    ua = {i["utterance"] for i in a_items}
    ub = {i["utterance"] for i in b_items}
    utts = sorted(ua & ub)
    if not utts:
        return None, None, None, 0
    ae, au = cluster_arrays([i for i in a_items if i["utterance"] in set(utts)], utts)
    be, bu = cluster_arrays([i for i in b_items if i["utterance"] in set(utts)], utts)
    if au.sum() == 0 or bu.sum() == 0:
        return None, None, None, len(utts)
    delta = float(ae.sum() / au.sum() - be.sum() / bu.sum())
    if len(utts) < 2:
        return delta, delta, delta, len(utts)
    idx = np.random.default_rng(seed).integers(0, len(utts), size=(b, len(utts)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = ae[idx].sum(axis=1) / au[idx].sum(axis=1) - be[idx].sum(axis=1) / bu[idx].sum(axis=1)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return delta, float(lo), float(hi), len(utts)


def analyse(
    by_engine, own: list[str], bleed: list[str], min_utt: int, cost: list[str], b: int, seed: int, names: dict[str, str],
    margin: float = 0.015, exclude_finals: list[str] | None = None, allow_incomplete: bool = False,
) -> dict:
    exclude = set(DEFAULT_EXCLUDE_FINALS if exclude_finals is None else exclude_finals)
    langs = sorted({i["language"] for e in by_engine.values() for i in e.values()})
    result = {
        "own_conditions": own, "bleed_conditions": bleed, "margin": margin, "exclude_finals": sorted(exclude),
        "allow_incomplete": allow_incomplete, "languages": {},
    }
    for lang in langs:
        engines = {}
        for eng, items in by_engine.items():
            mine = [i for i in items.values() if i["language"] == lang]
            if mine:
                engines[eng] = mine
        rows = {}
        for eng, mine in engines.items():
            own_items = [i for i in mine if i["condition"] in own]
            bleed_items = [i for i in mine if i["condition"] in bleed]
            n_utt = len({i["utterance"] for i in own_items})
            rate, lo, hi = ci_of(own_items, b, seed) if own_items else (None, None, None)
            per_cond = {c: pooled([i for i in own_items if i["condition"] == c]) for c in own}
            brate = pooled(bleed_items) if bleed_items else None
            bper = {c: pooled([i for i in bleed_items if i["condition"] == c]) for c in bleed}
            allrate = pooled([i for i in mine if i["condition"] in own or i["condition"] in bleed])
            missing = [c for c in own if per_cond[c] is None]
            rows[eng] = {
                "name": names.get(eng, eng), "utterances": n_utt, "own": rate, "own_ci": [lo, hi], "per_condition": per_cond,
                "bleed": brate, "bleed_per_condition": bper, "all": allrate,
                "missing_conditions": missing, "excluded": eng in exclude,
                "ranked": n_utt >= min_utt and not missing,
            }
        # An engine the data holds for this language but cannot be ranked for lack of an own-mic condition: whoever wins
        # among the rest might be the wrong default, so the language gets no winner (rule 5 of the module docstring).
        incomplete = [{"engine": e, "missing": r["missing_conditions"]} for e, r in rows.items() if r["missing_conditions"] and not r["excluded"]]
        too_small = [e for e, r in rows.items() if not r["missing_conditions"] and r["utterances"] < min_utt and not r["excluded"]]
        ranked = sorted((e for e in rows if rows[e]["ranked"]), key=lambda e: rows[e]["own"])
        if not ranked:
            result["languages"][lang] = {"engines": rows, "winner": None, "provisional_winner": None, "incomplete": incomplete, "too_small": too_small}
            continue
        best = ranked[0]
        own_items = lambda e: [i for i in engines[e] if i["condition"] in own]  # noqa: E731
        for e in ranked:
            if e == best:
                rows[e]["vs_best"] = None
                rows[e]["tie_with_best"] = True
                continue
            d, lo, hi, n = paired(own_items(e), own_items(best), b, seed)
            rows[e]["vs_best"] = {"delta": d, "ci95": [lo, hi], "utterances": n}
            rows[e]["tie_with_best"] = lo is not None and lo <= 0.0 <= hi and d is not None and d <= margin
        tied = [e for e in ranked if rows[e]["tie_with_best"]]
        # rule 4: an excluded engine takes part in the comparison above but is never the default
        tied_finals = [e for e in tied if e not in exclude]
        basis = "tie"
        if tied_finals:
            order = {e: (cost.index(e) if e in cost else len(cost)) for e in tied_finals}
            winner = min(tied_finals, key=lambda e: (order[e], rows[e]["own"]))
        else:
            finals = [e for e in ranked if e not in exclude]
            winner = finals[0] if finals else None
            basis = "best final engine (the best engine overall is never a final)" if winner else None
        # the all-condition ranking (what the first pass of the campaign used) for the record
        by_all = sorted((e for e in ranked if rows[e]["all"] is not None), key=lambda e: rows[e]["all"])
        result["languages"][lang] = {
            "engines": rows, "best_own": best, "tied": tied, "winner": None if incomplete and not allow_incomplete else winner,
            "provisional_winner": winner, "winner_basis": basis, "incomplete": incomplete, "too_small": too_small,
            "ranking_own": ranked, "ranking_all": by_all,
        }
    return result


def pct(v, d=1):
    return "-" if v is None else f"{v * 100:.{d}f}"


def _metric(lang: str) -> str:
    return "CER" if lang in ("ja", "ko", "zh", "yue") else "WER"


def _sorted_engines(data: dict, order: list[str]) -> list[str]:
    return sorted(data["engines"], key=lambda e: (order.index(e) if e in order else 99))


def own_table(res: dict, order: list[str]) -> str:
    """Own-microphone table: the per-condition rates, the pooled rate with its interval, the paired gap to the best."""
    own = res["own_conditions"]
    head = ["Language", "Engine"] + own + ["Own mic, pooled (95 % interval)", "Gap to the best (95 % interval)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|---|" + "---:|" * (len(head) - 2)]
    for lang, data in res["languages"].items():
        winner = data.get("winner")
        for e in _sorted_engines(data, order):
            r = data["engines"][e]
            cells = [pct(r["per_condition"].get(c)) for c in own]
            ownc = f"{pct(r['own'])} [{pct(r['own_ci'][0])}, {pct(r['own_ci'][1])}]"
            if e == winner:
                ownc = f"**{ownc}**"
            vb = r.get("vs_best")
            if not r["ranked"]:
                missing = [c for c in own if r["per_condition"].get(c) is None]
                why = f"no {', '.join(missing)}" if missing else f"{r['utterances']} utterances"
                vsb = f"not ranked ({why})"
            elif vb is None:
                vsb = "best, never a final" if r.get("excluded") else "best"
            else:
                if r["tie_with_best"]:
                    tie = "tie, never a final" if r.get("excluded") else "tie"
                elif vb["ci95"][0] > 0:
                    tie = "worse"
                else:
                    tie = "behind by more than the margin"
                vsb = f"{vb['delta'] * 100:+.1f} [{vb['ci95'][0] * 100:+.1f}, {vb['ci95'][1] * 100:+.1f}], {tie}"
            out.append(f"| {lang} ({_metric(lang)}) | {r['name']} | " + " | ".join(cells) + f" | {ownc} | {vsb} |")
    return "\n".join(out)


def bleed_table(res: dict, order: list[str]) -> str:
    """Bleed table (loopback captions; plays no part in the own-mic choice)."""
    bleed = res["bleed_conditions"]
    head = ["Language", "Engine"] + bleed + ["All bleed"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|---|" + "---:|" * (len(head) - 2)]
    for lang, data in res["languages"].items():
        for e in _sorted_engines(data, order):
            r = data["engines"][e]
            cells = [pct(r["bleed_per_condition"].get(c)) for c in bleed]
            out.append(f"| {lang} ({_metric(lang)}) | {r['name']} | " + " | ".join(cells) + f" | {pct(r['bleed'])} |")
    return "\n".join(out)


def withheld_reason(data: dict) -> str:
    """Why a language has no winner: engines the data lacks conditions for, or nothing ranked (``''`` when it has one)."""
    if data.get("winner"):
        return ""
    parts = [f"{x['engine']} has no {', '.join(x['missing'])}" for x in data.get("incomplete", [])]
    if parts:
        return "withheld, " + "; ".join(parts)
    return "withheld, no final engine has enough utterances"


def ranking_table(res: dict) -> str:
    """Both rankings per language: the own-mic one that sets the default and the all-conditions one."""
    out = ["| Language | Own mic (clean + mic-side): best to worst | All conditions incl. bleed: best to worst | Default |", "|---|---|---|---|"]
    for lang, data in res["languages"].items():
        nm = lambda e: data["engines"][e]["name"]  # noqa: E731
        own_rank = " > ".join(nm(e) for e in data.get("ranking_own", [])) or "-"
        all_rank = " > ".join(nm(e) for e in data.get("ranking_all", [])) or "-"
        if data.get("winner"):
            default = nm(data["winner"])
            if data.get("provisional_winner") and data.get("incomplete"):
                default += " (provisional: " + "; ".join(f"{x['engine']} lacks {', '.join(x['missing'])}" for x in data["incomplete"]) + ")"
        else:
            default = "**" + withheld_reason(data) + "**"
        out.append(f"| {lang} | {own_rank} | {all_rank} | {default} |")
    return "\n".join(out)


def to_markdown(res: dict, order: list[str]) -> str:
    return own_table(res, order) + "\n\n" + bleed_table(res, order) + "\n\n" + ranking_table(res)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", required=True, help="items.jsonl files written by score_stt.py --items")
    ap.add_argument("--own", default="clean,room_light,mic_codec")
    ap.add_argument("--bleed", default="music_snr10,babble_snr10,vrchat_mix")
    ap.add_argument("--min-utterances", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.015, help="largest point-estimate gap (absolute rate) to the best that still counts as a tie")
    ap.add_argument("--cost", default=",".join(DEFAULT_COST), help="engine ids, light to heavy (the tie-break among tied engines)")
    ap.add_argument(
        "--exclude-finals", default=",".join(DEFAULT_EXCLUDE_FINALS),
        help="engine ids that are never a caption final: tabulated and compared, but never the default (empty = none)",
    )
    ap.add_argument(
        "--allow-incomplete", action="store_true",
        help="keep the provisional winner of a language whose data lacks an own-mic condition for some engine (still warns, still exits 2)",
    )
    ap.add_argument("--names", default="", help="id=Display name,... overrides")
    ap.add_argument("--langs", default="")
    ap.add_argument("--resamples", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    names = {
        "parakeet-ultra": "Parakeet Ultra int8", "parakeet-v3": "Parakeet TDT v3 int8", "qwen3-asr-0.6b": "Qwen3-ASR 0.6B int8",
        "sensevoice": "SenseVoice small int8", "nemotron-560": "Nemotron 3.5 streaming 560 ms", "granite-cpu": "Granite TurboCTC Q8_0 (CPU, 4 threads)",
    }
    for kv in filter(None, a.names.split(",")):
        k, _, v = kv.partition("=")
        names[k] = v
    by_engine = load_items(a.items)
    if a.langs:
        keep = set(a.langs.split(","))
        by_engine = {e: {k: v for k, v in items.items() if v["language"] in keep} for e, items in by_engine.items()}
    cost = [c for c in a.cost.split(",") if c]
    exclude = [c for c in a.exclude_finals.split(",") if c]
    res = analyse(by_engine, a.own.split(","), a.bleed.split(","), a.min_utterances, cost, a.resamples, a.seed, names, a.margin, exclude, a.allow_incomplete)
    md = to_markdown(res, cost)
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print(md)
    print()
    incomplete_langs = []
    for lang, d in res["languages"].items():
        if d.get("incomplete"):
            incomplete_langs.append(lang)
        if d.get("winner"):
            note = f" [provisional, {withheld_reason({'incomplete': d['incomplete']}).replace('withheld, ', '')}]" if d.get("incomplete") else ""
            basis = f" [{d['winner_basis']}]" if d.get("winner_basis") not in (None, "tie") else ""
            print(f"{lang}: winner {d['winner']} (best own-mic {d['best_own']}, tied {', '.join(d['tied'])}); ranking own: {' > '.join(d['ranking_own'])}; ranking all: {' > '.join(d['ranking_all'])}{basis}{note}")
        else:
            print(f"{lang}: NO WINNER, {withheld_reason(d)}" + (f"; the ranked engines alone would pick {d['provisional_winner']}" if d.get("provisional_winner") else ""))
        if d.get("too_small"):
            print(f"{lang}: note, listed but not ranked (fewer than {a.min_utterances} utterances): {', '.join(d['too_small'])}", file=sys.stderr)
    if incomplete_langs:
        print(
            "\nWARNING: the input files do not cover every engine on every own-mic condition ("
            + ", ".join(a.own.split(","))
            + ") for: "
            + ", ".join(
                f"{lang} ({'; '.join(x['engine'] + ' lacks ' + ', '.join(x['missing']) for x in res['languages'][lang]['incomplete'])})" for lang in incomplete_langs
            )
            + ".\nA default chosen without those engines can be the wrong one (for example Korean: Qwen3-ASR 0.6B leads SenseVoice only once its mic-side runs are counted)."
            + "\nPass every scores/items.jsonl of the campaign (finals, mic-a, mic-b, mic-c; see campaign/README.md)."
            + (" --allow-incomplete kept the provisional winners." if a.allow_incomplete else " The affected languages have no winner."),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
