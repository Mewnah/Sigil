# SPDX-License-Identifier: AGPL-3.0-or-later
"""The campaign's own-mic decision rule (campaign/own_mic.py): which conditions decide, when two engines
tie, and what the tie-break does. Pure arithmetic on synthetic item lists."""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.helpers import BENCH

CAMPAIGN = os.path.join(BENCH, "campaign")
if CAMPAIGN not in sys.path:
    sys.path.insert(0, CAMPAIGN)

import own_mic  # noqa: E402

OWN = ["clean", "room_light", "mic_codec"]
BLEED = ["music_snr10", "babble_snr10", "vrchat_mix"]


def items(lang, errors_by_condition, n=30, units=10):
    """{item id: item} for one engine: ``errors_by_condition[c]`` is the error count of every utterance of that condition,
    or a list with one count per utterance."""
    out = {}
    for cond, e in errors_by_condition.items():
        for u in range(n):
            errs = e[u] if isinstance(e, list) else e
            utt = f"{lang}/u{u:03d}"
            out[f"{utt}/{cond}"] = {"utterance": utt, "language": lang, "condition": cond, "errors": errs, "units": units}
    return out


def analyse(by_engine, **kw):
    kw.setdefault("own", OWN)
    kw.setdefault("bleed", BLEED)
    return own_mic.analyse(
        by_engine, kw["own"], kw["bleed"], kw.get("min_utt", 20), kw.get("cost", own_mic.DEFAULT_COST),
        200, 20261002, {}, kw.get("margin", 0.015), kw.get("exclude"), kw.get("allow_incomplete", False),
    )["languages"]


def varying(base, pattern):
    """A per-utterance error list: ``base`` plus a repeating pattern (so paired intervals are not degenerate)."""
    return [base + pattern[i % len(pattern)] for i in range(30)]


class OwnMicRule(unittest.TestCase):
    def test_the_bleed_conditions_never_move_the_choice(self):
        # B is far better under bleed but worse on the mic-side conditions: A must win.
        a = items("ko", {c: varying(1, [0, 1, 0, 1]) for c in OWN} | {c: varying(5, [0, 1]) for c in BLEED})
        b = items("ko", {c: varying(3, [0, 1, 0, 1]) for c in OWN} | {c: varying(1, [0, 1]) for c in BLEED})
        res = analyse({"a": a, "b": b}, cost=["b", "a"])["ko"]
        self.assertEqual(res["winner"], "a")
        self.assertEqual(res["ranking_own"], ["a", "b"])
        # the all-conditions ranking (what the first campaign pass used) is kept for the record and may differ
        self.assertEqual(res["ranking_all"], ["b", "a"])

    def test_a_significant_gap_beats_a_cheaper_engine(self):
        best = items("de", {c: varying(1, [0, 1]) for c in OWN})
        cheap = items("de", {c: varying(3, [0, 1]) for c in OWN})
        res = analyse({"best": best, "cheap": cheap}, cost=["cheap", "best"])["de"]
        self.assertEqual(res["winner"], "best")
        self.assertFalse(res["engines"]["cheap"]["tie_with_best"])
        self.assertGreater(res["engines"]["cheap"]["vs_best"]["ci95"][0], 0)

    def test_a_tie_goes_to_the_cheaper_engine(self):
        best = items("en", {c: varying(1, [0, 1, 1, 0]) for c in OWN})
        close = items("en", {c: varying(1, [1, 0, 1, 0]) for c in OWN})  # same total, different utterances
        res = analyse({"best": best, "close": close}, cost=["close", "best"])["en"]
        self.assertTrue(res["engines"]["close"]["tie_with_best"])
        self.assertEqual(res["winner"], "close")

    def test_an_interval_that_spans_zero_is_not_enough_when_the_gap_exceeds_the_margin(self):
        # 3 utterances in 30 carry all the difference: a wide interval that includes zero, a point gap of 2 points > 1.5
        best = items("fr", {c: [0] * 30 for c in OWN})
        noisy = items("fr", {c: [6, 6, 6] + [0] * 27 for c in OWN})
        res = analyse({"best": best, "noisy": noisy}, cost=["noisy", "best"])["fr"]
        e = res["engines"]["noisy"]
        self.assertLessEqual(e["vs_best"]["ci95"][0], 0)  # the interval spans zero
        self.assertGreater(e["vs_best"]["delta"], 0.015)
        self.assertFalse(e["tie_with_best"])
        self.assertEqual(res["winner"], "best")

    def test_small_samples_are_listed_but_not_ranked(self):
        full = items("ru", {c: varying(1, [0, 1]) for c in OWN})
        small = items("ru", {c: [0] * 12 for c in OWN}, n=12)  # the 12-utterance Nemotron sample
        res = analyse({"full": full, "small": small}, min_utt=20)["ru"]
        self.assertFalse(res["engines"]["small"]["ranked"])
        self.assertEqual(res["ranking_own"], ["full"])
        self.assertEqual(res["winner"], "full")

    def test_an_engine_missing_a_mic_condition_is_not_ranked_and_withholds_the_winner(self):
        # clean only: the mic-side conditions are required for the own-mic ranking, and a winner chosen without this engine
        # could be the wrong default (campaign review: the README example left Qwen3-ASR out and Korean flipped to SenseVoice)
        partial = items("es", {"clean": varying(0, [0, 1])})
        full = items("es", {c: varying(1, [0, 1]) for c in OWN})
        res = analyse({"partial": partial, "full": full})["es"]
        self.assertFalse(res["engines"]["partial"]["ranked"])
        self.assertEqual(res["engines"]["partial"]["missing_conditions"], ["room_light", "mic_codec"])
        self.assertIsNone(res["winner"])
        self.assertEqual(res["provisional_winner"], "full")
        self.assertEqual(res["incomplete"], [{"engine": "partial", "missing": ["room_light", "mic_codec"]}])

    def test_the_korean_flip_the_readme_example_produced_is_withheld(self):
        # SenseVoice has all conditions, Qwen3-ASR only the finals-core clean run (its mic-side runs are in another file)
        sense = items("ko", {c: varying(4, [0, 1, 0, 2]) for c in OWN})
        qwen = items("ko", {"clean": varying(2, [0, 1, 0, 2])})
        res = analyse({"sensevoice": sense, "qwen3-asr-0.6b": qwen})["ko"]
        self.assertIsNone(res["winner"])
        self.assertEqual(res["provisional_winner"], "sensevoice")
        self.assertEqual([x["engine"] for x in res["incomplete"]], ["qwen3-asr-0.6b"])
        # with the missing runs added the answer is the opposite, as in the shipped default
        qwen = items("ko", {c: varying(2, [0, 1, 0, 2]) for c in OWN})
        res = analyse({"sensevoice": sense, "qwen3-asr-0.6b": qwen})["ko"]
        self.assertEqual(res["winner"], "qwen3-asr-0.6b")
        self.assertEqual(res["incomplete"], [])

    def test_an_engine_with_only_bleed_items_is_incomplete_not_invisible(self):
        full = items("de", {c: varying(1, [0, 1]) for c in OWN})
        bleed_only = items("de", {c: varying(0, [0, 1]) for c in BLEED})
        res = analyse({"full": full, "bleed_only": bleed_only})["de"]
        self.assertIn("bleed_only", res["engines"])
        self.assertEqual(res["engines"]["bleed_only"]["missing_conditions"], OWN)
        self.assertIsNone(res["winner"])

    def test_allow_incomplete_keeps_the_provisional_winner(self):
        partial = items("es", {"clean": varying(0, [0, 1])})
        full = items("es", {c: varying(1, [0, 1]) for c in OWN})
        res = analyse({"partial": partial, "full": full}, allow_incomplete=True)["es"]
        self.assertEqual(res["winner"], "full")
        self.assertEqual(len(res["incomplete"]), 1)  # still reported, so the markdown and the exit code flag it

    def test_an_excluded_engine_without_the_mic_runs_does_not_block_the_winner(self):
        # Nemotron outside English: 12 utterances on `core` only, by design
        full = items("ja", {c: varying(1, [0, 1]) for c in OWN})
        nemotron = items("ja", {"clean": [0] * 12}, n=12)
        res = analyse({"sensevoice": full, "nemotron-560": nemotron})["ja"]
        self.assertEqual(res["winner"], "sensevoice")
        self.assertEqual(res["incomplete"], [])
        self.assertFalse(res["engines"]["nemotron-560"]["ranked"])  # still listed, not ranked

    def test_a_small_sample_with_every_condition_is_only_listed(self):
        full = items("ru", {c: varying(1, [0, 1]) for c in OWN})
        small = items("ru", {c: [0] * 12 for c in OWN}, n=12)
        res = analyse({"full": full, "small": small}, min_utt=20)["ru"]
        self.assertEqual(res["winner"], "full")
        self.assertEqual(res["too_small"], ["small"])
        self.assertEqual(res["incomplete"], [])

    def test_granite_never_wins_even_when_it_is_the_cheapest_and_the_best(self):
        # the English case: Granite is best by a hair and ties Ultra; no cost order may make it the default
        granite = items("en", {c: varying(1, [0, 1, 0, 1]) for c in OWN})
        ultra = items("en", {c: varying(1, [1, 0, 1, 0]) for c in OWN})
        for cost in (["granite-cpu", "parakeet-ultra"], ["parakeet-ultra", "granite-cpu"], []):
            res = analyse({"granite-cpu": granite, "parakeet-ultra": ultra}, cost=cost)["en"]
            self.assertTrue(res["engines"]["granite-cpu"]["tie_with_best"])
            self.assertIn("granite-cpu", res["tied"])  # it stays in the comparison and the tables
            self.assertEqual(res["winner"], "parakeet-ultra", cost)
        # the default exclusion is the contract of the CLI
        self.assertEqual(own_mic.DEFAULT_EXCLUDE_FINALS, ["granite-cpu", "nemotron-560"])

    def test_nemotron_never_wins_a_language_it_ranks_in(self):
        nemotron = items("en", {c: varying(0, [0, 1, 0, 1]) for c in OWN})
        ultra = items("en", {c: varying(1, [1, 0, 1, 0]) for c in OWN})
        res = analyse({"nemotron-560": nemotron, "parakeet-ultra": ultra}, cost=["nemotron-560", "parakeet-ultra"])["en"]
        self.assertEqual(res["winner"], "parakeet-ultra")

    def test_when_the_best_engine_is_excluded_by_a_clear_margin_the_best_final_wins_and_says_so(self):
        granite = items("en", {c: varying(0, [0, 1]) for c in OWN})
        ultra = items("en", {c: varying(4, [0, 1]) for c in OWN})
        slow = items("en", {c: varying(6, [0, 1]) for c in OWN})
        res = analyse({"granite-cpu": granite, "parakeet-ultra": ultra, "qwen3-asr-0.6b": slow})["en"]
        self.assertEqual(res["best_own"], "granite-cpu")
        self.assertEqual(res["tied"], ["granite-cpu"])
        self.assertEqual(res["winner"], "parakeet-ultra")
        self.assertIn("never a final", res["winner_basis"])

    def test_with_every_ranked_engine_excluded_there_is_no_winner(self):
        granite = items("en", {c: varying(1, [0, 1]) for c in OWN})
        res = analyse({"granite-cpu": granite})["en"]
        self.assertIsNone(res["winner"])

    def test_an_empty_exclude_list_lets_granite_win_on_cost(self):
        granite = items("en", {c: varying(1, [0, 1, 0, 1]) for c in OWN})
        ultra = items("en", {c: varying(1, [1, 0, 1, 0]) for c in OWN})
        res = analyse({"granite-cpu": granite, "parakeet-ultra": ultra}, cost=["granite-cpu", "parakeet-ultra"], exclude=[])["en"]
        self.assertEqual(res["winner"], "granite-cpu")

    def test_the_markdown_marks_a_withheld_language_and_a_never_a_final_engine(self):
        partial = items("es", {"clean": varying(0, [0, 1])})
        full = items("es", {c: varying(1, [0, 1]) for c in OWN})
        granite = items("en", {c: varying(1, [0, 1]) for c in OWN})
        ultra = items("en", {c: varying(1, [1, 0]) for c in OWN})
        res = own_mic.analyse(
            {"partial": partial, "full": full, "granite-cpu": granite, "parakeet-ultra": ultra},
            OWN, BLEED, 20, own_mic.DEFAULT_COST, 100, 20261002, {},
        )
        md = own_mic.to_markdown(res, own_mic.DEFAULT_COST)
        self.assertIn("**withheld, partial has no room_light, mic_codec**", md)
        self.assertIn("never a final", md)

    def test_pooled_rate_is_errors_over_units_not_a_mean_of_rates(self):
        long_ = {"utterance": "x/1", "language": "x", "condition": "clean", "errors": 1, "units": 100}
        short = {"utterance": "x/2", "language": "x", "condition": "clean", "errors": 1, "units": 1}
        self.assertAlmostEqual(own_mic.pooled([long_, short]), 2 / 101)

    def test_the_markdown_has_the_three_tables_and_the_default_in_bold(self):
        a = items("ja", {c: varying(1, [0, 1]) for c in OWN} | {c: varying(2, [0, 1]) for c in BLEED})
        b = items("ja", {c: varying(4, [0, 1]) for c in OWN} | {c: varying(2, [0, 1]) for c in BLEED})
        res = own_mic.analyse({"sensevoice": a, "qwen3-asr-0.6b": b}, OWN, BLEED, 20, own_mic.DEFAULT_COST, 100, 20261002, {})
        md = own_mic.to_markdown(res, own_mic.DEFAULT_COST)
        self.assertIn("Own mic, pooled", md)
        self.assertIn("All bleed", md)
        self.assertIn("best to worst", md)
        self.assertIn("ja (CER)", md)
        self.assertRegex(md, r"\*\*\d+\.\d \[")

    def test_load_items_reads_the_scorer_output_and_names_engines_by_label_suffix(self):
        rows = [
            {"label": "rep-a/parakeet-ultra", "item": "en/123/clean", "language": "en", "condition": "clean", "failed": False,
             "counts": {"errors": 2, "ref_units": 20}},
            {"label": "rep-a/parakeet-ultra", "item": "en/124/clean", "language": "en", "condition": "clean", "failed": True, "counts": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "items.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            got = own_mic.load_items([p])
        self.assertEqual(list(got), ["parakeet-ultra"])
        self.assertEqual(got["parakeet-ultra"]["en/123/clean"]["utterance"], "en/123")
        self.assertEqual(len(got["parakeet-ultra"]), 1)  # the failed (counts: null) item is skipped here; the scorer already counted it


def _write_items(path, by_engine):
    """items.jsonl as score_stt.py --items writes it, from {engine: {item id: item}}."""
    with open(path, "w", encoding="utf-8") as f:
        for engine, its in by_engine.items():
            for item_id, it in its.items():
                f.write(json.dumps({
                    "label": f"rep/{engine}", "item": item_id, "language": it["language"], "condition": it["condition"],
                    "failed": False, "counts": {"errors": it["errors"], "ref_units": it["units"]},
                }) + "\n")


class OwnMicCommandLine(unittest.TestCase):
    """The README example must not hand out a wrong default silently."""

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = own_mic.main(["--resamples", "100", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_missing_mic_runs_exit_2_warn_loudly_and_print_no_winner_line(self):
        with tempfile.TemporaryDirectory() as d:
            finals, mic = os.path.join(d, "finals.jsonl"), os.path.join(d, "mic.jsonl")
            sense = items("ko", {c: varying(4, [0, 1, 0, 2]) for c in OWN})
            qwen = items("ko", {c: varying(2, [0, 1, 0, 2]) for c in OWN})
            _write_items(finals, {"qwen3-asr-0.6b": {k: v for k, v in qwen.items() if v["condition"] == "clean"}})
            _write_items(mic, {"sensevoice": sense})
            code, out, err = self.run_main("--items", finals, mic)
            self.assertEqual(code, 2)
            self.assertIn("WARNING", err)
            self.assertIn("qwen3-asr-0.6b lacks room_light, mic_codec", err)
            self.assertNotRegex(out, r"(?m)^ko: winner")
            self.assertIn("ko: NO WINNER", out)
            # all runs passed: the shipped answer, exit 0, no warning
            qwen_mic = os.path.join(d, "qwen_mic.jsonl")
            _write_items(qwen_mic, {"qwen3-asr-0.6b": {k: v for k, v in qwen.items() if v["condition"] != "clean"}})
            code, out, err = self.run_main("--items", finals, mic, qwen_mic)
            self.assertEqual(code, 0)
            self.assertNotIn("WARNING", err)
            self.assertRegex(out, r"(?m)^ko: winner qwen3-asr-0.6b")

    def test_allow_incomplete_prints_a_flagged_winner_and_still_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            _write_items(p, {
                "sensevoice": items("ko", {c: varying(4, [0, 1]) for c in OWN}),
                "qwen3-asr-0.6b": items("ko", {"clean": varying(2, [0, 1])}),
            })
            code, out, err = self.run_main("--allow-incomplete", "--items", p)
            self.assertEqual(code, 2)
            self.assertRegex(out, r"(?m)^ko: winner sensevoice .*provisional")
            self.assertIn("kept the provisional winners", err)

    def test_the_exclude_finals_flag_defaults_to_granite_and_nemotron_and_can_be_emptied(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            _write_items(p, {
                "granite-cpu": items("en", {c: varying(1, [0, 1, 0, 1]) for c in OWN}),
                "parakeet-ultra": items("en", {c: varying(1, [1, 0, 1, 0]) for c in OWN}),
            })
            code, out, _ = self.run_main("--items", p, "--cost", "granite-cpu,parakeet-ultra")
            self.assertEqual(code, 0)
            self.assertRegex(out, r"(?m)^en: winner parakeet-ultra")
            code, out, _ = self.run_main("--items", p, "--cost", "granite-cpu,parakeet-ultra", "--exclude-finals", "")
            self.assertRegex(out, r"(?m)^en: winner granite-cpu")

    def test_the_readme_example_names_every_scores_file_of_the_decision(self):
        with open(os.path.join(CAMPAIGN, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        m = re.search(r"own_mic\.py\s+--items(.*?)--md", readme, re.S)
        self.assertIsNotNone(m, "README must show the own_mic.py command")
        for name in ("finals", "mic-a", "mic-b", "mic-c"):
            self.assertIn(name, m.group(1))


FOOTPRINT = os.path.join(CAMPAIGN, "footprint")
PWSH = shutil.which("pwsh") if sys.platform == "win32" else None


def _ps(name):
    with open(os.path.join(FOOTPRINT, name), encoding="utf-8") as f:
        return f.read()


class FootprintLaunchRules(unittest.TestCase):
    """The footprint scripts start the real app on a PC the user is working on: never in front, never against a stranger's
    process. (Static checks of the sources; the helper functions are exercised in FootprintLaunchGuard.)"""

    def test_both_launchers_set_the_switch_guard_the_exe_and_restore_the_environment(self):
        for name in ("footprint.ps1", "diagnose.ps1"):
            src = _ps(name)
            with self.subTest(script=name):
                self.assertIn("launch-guard.ps1", src)
                # refused before anything is written or started
                guard, first_start = src.index("Assert-StartsMinimized"), src.index("Start-Process $Exe")
                self.assertLess(guard, first_start)
                for later in ("New-Item", "Set-Content"):
                    if later in src:
                        self.assertLess(guard, src.index(later))
                # the switch is set right before the app starts, and restored with the rest of the environment
                self.assertLess(src.index("Set-StartMinimizedEnv"), first_start)
                self.assertRegex(src, r"\$(saved|savedEnv) = @\{[^\n]*SIGIL_START_MINIMIZED = \$env:SIGIL_START_MINIMIZED")
                self.assertIn("[switch]$ShowWindow", src)

    def test_the_fallback_minimize_never_activates_the_window(self):
        for name in ("footprint.ps1", "diagnose.ps1"):
            src = _ps(name)
            with self.subTest(script=name):
                self.assertRegex(src, r"ShowWindow\(\$app\.MainWindowHandle, 7\)")  # SW_SHOWMINNOACTIVE
                self.assertNotRegex(src, r"ShowWindow\(\$app\.MainWindowHandle, 6\)")  # SW_MINIMIZE hands the focus on

    def test_the_cleanup_stops_only_the_llama_server_of_this_run(self):
        for name in ("footprint.ps1", "diagnose.ps1"):
            src = _ps(name)
            with self.subTest(script=name):
                self.assertIn("Stop-OwnedProcess", src)
                self.assertNotRegex(src, r"Get-Process llama-server[^\n]*Stop-Process")

    @unittest.skipUnless(PWSH, "needs Windows PowerShell 7")
    def test_every_translation_scenario_states_its_interim_policy(self):
        # Run the settings section of footprint.ps1 for each scenario and read what it would write to the profile.
        src = _ps("footprint.ps1")
        a, b = src.index("# --- settings of the scenario"), src.index("$settings = @{ id =")
        section = src[a:b]
        scenarios = ["translate-cpu", "translate-cpu-finals", "translate-cpu-slow", "translate-vulkan", "translate-auto"]
        script = (
            "$Device = ''\n$out = [ordered]@{}\n"
            + "foreach ($Scenario in @(" + ",".join(f"'{s}'" for s in scenarios) + ")) {\n"
            + "  & { param($Scenario, $Device)\n" + section + "\n    $out[$Scenario] = $services.translation.data\n  } $Scenario $Device\n}\n"
            + "$out | ConvertTo-Json -Depth 8 -Compress\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.ps1")
            with open(p, "w", encoding="utf-8") as f:
                f.write(script)
            run = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", p], capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        got = json.loads(run.stdout)
        interim = {k: v["auto"]["interim"] for k, v in got.items()}
        self.assertEqual(interim, {
            "translate-cpu": True, "translate-cpu-finals": False, "translate-cpu-slow": True, "translate-vulkan": True,
            "translate-auto": "auto",
        })
        self.assertEqual(got["translate-cpu-slow"]["partials"]["throttleMs"], 1500)
        self.assertEqual([got[s]["llm"]["serverVariant"] for s in scenarios], ["cpu", "cpu", "cpu", "vulkan", "auto"])


@unittest.skipUnless(PWSH, "needs Windows PowerShell 7")
class FootprintLaunchGuard(unittest.TestCase):
    """footprint/launch-guard.ps1 run for real: the exe check and the pick of the processes the run owns."""

    def run_ps(self, body):
        script = f". '{os.path.join(FOOTPRINT, 'launch-guard.ps1')}'\n$ErrorActionPreference = 'Stop'\n" + body
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.ps1")
            with open(p, "w", encoding="utf-8") as f:
                f.write(script)
            return subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", p], capture_output=True, text=True, timeout=120)

    def test_an_exe_without_the_switch_is_refused_and_one_with_it_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = os.path.join(d, "old.exe"), os.path.join(d, "new.exe")
            with open(old, "wb") as f:
                f.write(b"MZ" + b"\0" * 5000 + b"SIGIL_START_MINIMIZE" + b"\0" * 100)  # one letter short: not the switch
            with open(new, "wb") as f:
                f.write(b"MZ" + b"\0" * (5 * 1024 * 1024) + b"xx SIGIL_START_MINIMIZED xx")  # past the first 4 MB read window
            refused = self.run_ps(f"Assert-StartsMinimized -Exe '{old}'")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("predates SIGIL_START_MINIMIZED", refused.stderr)
            self.assertEqual(self.run_ps(f"Assert-StartsMinimized -Exe '{new}'").returncode, 0)
            # an explicit -ShowWindow is the one way to start an exe that predates the switch
            self.assertEqual(self.run_ps(f"Assert-StartsMinimized -Exe '{old}' -ShowWindow").returncode, 0)

    def test_the_switch_is_set_or_removed_for_the_child_only_by_the_helper(self):
        run = self.run_ps(
            "$env:SIGIL_START_MINIMIZED = $null\n"
            "Set-StartMinimizedEnv; $a = $env:SIGIL_START_MINIMIZED\n"
            "Set-StartMinimizedEnv -ShowWindow; $b = [string]$env:SIGIL_START_MINIMIZED\n"
            "\"$a|$b\"\n"
        )
        self.assertEqual(run.stdout.strip(), "1|")

    def test_only_descendants_created_after_their_parent_are_owned(self):
        run = self.run_ps(
            "$t = [datetime]'2026-10-10T12:00:00'\n"
            "function Row($id, $parent, $name, $at) { [pscustomobject]@{ ProcessId = $id; ParentProcessId = $parent; Name = $name; CreationDate = $at } }\n"
            "$tbl = @(\n"
            "  (Row 100 1 'sigil.exe' $t),\n"
            "  (Row 101 100 'llama-server.exe' $t.AddSeconds(5)),\n"      # the app's own translator
            "  (Row 102 100 'msedgewebview2.exe' $t.AddSeconds(3)),\n"
            "  (Row 103 102 'llama-server.exe' $t.AddSeconds(9)),\n"      # a grandchild
            "  (Row 200 1 'llama-server.exe' $t.AddSeconds(7)),\n"        # the user's own server, started during the run
            "  (Row 201 100 'llama-server.exe' $t.AddHours(-3))\n"        # a stale row naming a recycled parent pid
            ")\n"
            "$mine = @(Select-OwnedProcess -Table $tbl -RootPid 100 -RootStart $t -Name 'llama-server*') | ForEach-Object ProcessId\n"
            "$all = @(Select-OwnedProcess -Table $tbl -RootPid 100 -RootStart $t) | ForEach-Object ProcessId\n"
            "$none = @(Select-OwnedProcess -Table $tbl -RootPid 999 -RootStart $t -Name 'llama-server*').Count\n"
            "\"$($mine -join ',')|$($all -join ',')|$none\"\n"
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), "101,103|101,102,103|0")

    def test_stop_owned_process_kills_the_child_of_the_root_and_spares_a_stranger(self):
        # Real processes: a root pwsh that starts a child sleeper, and a stranger sleeper started by the test itself.
        with tempfile.TemporaryDirectory() as d:
            pidfile = os.path.join(d, "child.pid")
            root_script = os.path.join(d, "root.ps1")
            with open(root_script, "w", encoding="utf-8") as f:
                f.write(
                    "$c = Start-Process pwsh -ArgumentList '-NoProfile','-NonInteractive','-Command','Start-Sleep 60' -PassThru -WindowStyle Hidden\n"
                    f"Set-Content -LiteralPath '{pidfile}' $c.Id\nStart-Sleep 60\n"
                )
            run = self.run_ps(
                "$sleeper = @('-NoProfile','-NonInteractive','-Command','Start-Sleep 60')\n"
                "$stranger = Start-Process pwsh -ArgumentList $sleeper -PassThru -WindowStyle Hidden\n"
                f"$root = Start-Process pwsh -ArgumentList '-NoProfile','-NonInteractive','-File','{root_script}' -PassThru -WindowStyle Hidden\n"
                "$child = 0\n"
                "try {\n"
                "  $rootStart = $root.StartTime\n"
                "  $deadline = (Get-Date).AddSeconds(40)\n"
                f"  while (-not (Test-Path -LiteralPath '{pidfile}') -and (Get-Date) -lt $deadline) {{ Start-Sleep -Milliseconds 200 }}\n"
                f"  Start-Sleep -Milliseconds 300; $child = [int](Get-Content -LiteralPath '{pidfile}')\n"
                "  $known = @(Get-OwnedProcess -RootPid $root.Id -RootStart $rootStart -Name 'pwsh*')\n"
                "  $picked = ($known | ForEach-Object ProcessId) -join ','\n"
                "  Stop-OwnedProcess -RootPid $root.Id -RootStart $rootStart -Name 'pwsh*' -Known $known\n"
                "  Start-Sleep -Seconds 2\n"
                "  $childAlive = [bool](Get-Process -Id $child -ErrorAction SilentlyContinue)\n"
                "  $strangerAlive = [bool](Get-Process -Id $stranger.Id -ErrorAction SilentlyContinue)\n"
                "  \"$($picked -eq [string]$child)|$childAlive|$strangerAlive\"\n"
                "} finally {\n"
                "  foreach ($id in @($stranger.Id, $root.Id, $child)) { if ($id) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue } }\n"
                "}\n"
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip().splitlines()[-1], "True|False|True")


if __name__ == "__main__":
    unittest.main()
