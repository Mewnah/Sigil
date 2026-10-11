# SPDX-License-Identifier: AGPL-3.0-or-later
"""MT-Europe: the table script (campaign/mt_europe_table.py) on synthetic hypotheses, and the flores200-eu set builder."""

import io
import json
import os
import sys
import tarfile
import tempfile
import unittest

from tests.helpers import BENCH

CAMPAIGN = os.path.join(BENCH, "campaign")
if CAMPAIGN not in sys.path:
    sys.path.insert(0, CAMPAIGN)

import mt_europe_table as table  # noqa: E402

A, B = "nllb-beam4", "hy-mt2-1.8b-cpu"


def line(set_name, direction, n, src, ref):
    s, t = direction.split("-")
    return {"id": f"{set_name}:{direction}:{n}", "source": s, "target": t, "src": src, "ref": ref, "refs": [ref], "set": set_name}


def report(items_by_engine):
    """A speech-bench report with one mt_quality run holding ``items_by_engine[engine] = [(line, hyp, latency)]``."""
    results = []
    for engine, items in items_by_engine.items():
        results.append({"engine": engine, "status": "passed", "items": [
            {"item": ln["id"], "direction": f"{ln['source']}-{ln['target']}", "text": hyp, "latency_ms": ms, "error": None} for ln, hyp, ms in items]})
    return {"runs": [{"name": "r", "mode": "mt_quality", "status": "passed", "results": [{**r, "task": "mt"} for r in results]}], "engines": [{"id": e, "task": "mt"} for e in items_by_engine]}


class TableTest(unittest.TestCase):
    def setUp(self):
        self.refs = ["Das ist ein kleiner Test für die Übersetzung.", "Wie geht es dir heute Abend im Stream?", "Ich höre dich gut, danke.",
                     "Die Katze sitzt auf der Matte im Wohnzimmer.", "Wir sehen uns morgen wieder in der Welt."]
        self.lines = [line("sigil-casual", "en-de", i, f"src {i}", r) for i, r in enumerate(self.refs, 1)]
        self.lines += [line("wmt24pp", "en-de", i, f"src {i}", r) for i, r in enumerate(self.refs, 1)]
        self.lines += [line("sigil-casual", "de-en", i, f"quelle {i}", r) for i, r in enumerate(self.refs, 1)]
        self.sets = {"p": {ln["id"]: ln for ln in self.lines}}

    def data(self, a_good=True):
        exact = lambda ln: ln["ref"]  # noqa: E731
        wrong = lambda ln: "Etwas ganz anderes steht hier."  # noqa: E731
        return table.collect(report({
            A: [(ln, exact(ln) if a_good else wrong(ln), 1500.0) for ln in self.lines],
            B: [(ln, wrong(ln), 400.0) for ln in self.lines],
        }), self.sets)

    def test_pair_of_names_the_language_with_english(self):
        self.assertEqual(table.pair_of("en-fr"), "fr")
        self.assertEqual(table.pair_of("pt-en"), "pt")
        self.assertIsNone(table.pair_of("fr-de"))
        self.assertEqual(table.pair_of("en-zh-Hant"), "zh-Hant")

    def test_a_clearly_better_engine_has_an_interval_above_zero_per_direction_and_pair(self):
        res = table.analyse(self.data(), A, B, 200, 1)
        by_dir = {r["direction"]: r for r in res["directions"]}
        self.assertEqual(set(by_dir), {"en-de", "de-en"})
        self.assertEqual(by_dir["en-de"]["lines"], 10)  # two sets of five lines
        self.assertGreater(by_dir["en-de"]["ci95"][0], 0)
        self.assertEqual(by_dir["en-de"]["prob_a_better"], 1.0)
        self.assertEqual(by_dir["en-de"]["latency"][A]["all"]["p50_ms"], 1500.0)
        self.assertEqual(by_dir["en-de"]["latency"][B]["casual"]["lines"], 5)  # only the casual set
        self.assertEqual(len(res["pairs"]), 1)
        pair = res["pairs"][0]
        self.assertEqual((pair["pair"], pair["directions"], pair["lines"]), ("en<->de", ["de-en", "en-de"], 15))
        self.assertGreater(pair["ci95"][0], 0)

    def test_identical_engines_tie_with_an_interval_around_zero(self):
        res = table.analyse(self.data(a_good=False), A, B, 200, 1)
        for r in res["directions"] + res["pairs"]:
            self.assertAlmostEqual(r["delta"], 0.0, places=9)
            self.assertLessEqual(r["ci95"][0], 0.0)
            self.assertGreaterEqual(r["ci95"][1], 0.0)

    def test_the_direction_score_is_the_line_weighted_mean_of_the_set_scores(self):
        # one good line in the casual set (5 lines), five bad ones in wmt24pp: the mean weights the sets by their line counts
        lines = [line("sigil-casual", "en-de", i, "s", self.refs[i - 1]) for i in range(1, 6)] + [line("wmt24pp", "en-de", i, "s", self.refs[i - 1]) for i in range(1, 4)]
        sets = {"p": {ln["id"]: ln for ln in lines}}
        rep = report({A: [(ln, ln["ref"], 1.0) for ln in lines], B: [(ln, "Etwas ganz anderes steht hier.", 1.0) for ln in lines]})
        res = table.analyse(table.collect(rep, sets), A, B, 0, 1)
        row = res["directions"][0]
        self.assertAlmostEqual(row["a"], 100.0, places=6)
        self.assertEqual(row["lines"], 8)
        self.assertAlmostEqual(row["sets"]["sigil-casual"][A], 100.0, places=6)
        self.assertAlmostEqual(row["b"], (row["sets"]["sigil-casual"][B] * 5 + row["sets"]["wmt24pp"][B] * 3) / 8, places=9)

    def test_markdown_has_the_pair_direction_and_set_tables(self):
        md = table.to_markdown(table.analyse(self.data(), A, B, 50, 1))
        self.assertIn("| en<->de |", md)
        self.assertIn("| de-en |", md)
        self.assertIn("| direction | sigil-casual | wmt24pp |", md)


class FloresEuTest(unittest.TestCase):
    def test_the_set_reuses_the_flores200_pick_and_covers_the_european_directions(self):
        sys.path.insert(0, os.path.join(BENCH, "corpus"))
        import build_text_sets as bts

        codes = {"eng_Latn": "en", **{v: k for k, v in bts.FLORES_EU.items()}, **{v: k for k, v in bts.FLORES.items()}}
        n = 40
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "eval", "flores200")
            os.makedirs(folder)
            with tarfile.open(os.path.join(folder, "flores200_dataset.tar.gz"), "w:gz") as tar:
                for code, lang in codes.items():
                    data = "".join(f"{lang} sentence {i}\n" for i in range(n)).encode("utf-8")
                    info = tarfile.TarInfo(f"flores200_dataset/devtest/{code}.devtest")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            plain = bts.flores200(root, 12, 7)
            eu = bts.flores200_eu(root, 12, 7)
        directions = sorted({f"{r['source']}-{r['target']}" for r in eu})
        self.assertEqual(directions, sorted([f"en-{l}" for l in bts.FLORES_EU] + ["it-en", "nl-en"]))
        self.assertTrue(all(r["set"] == "flores200-eu" for r in eu))
        ids = [r["id"] for r in eu]
        self.assertEqual(len(ids), len(set(ids)))
        # the English side is the same sentence list as the X -> en rows of flores200 (the same devtest picks)
        picked = [r["ref"] for r in plain if r["source"] == "de"]
        en_de = [r["src"] for r in eu if (r["source"], r["target"]) == ("en", "de")]
        self.assertEqual(en_de, picked)
        self.assertEqual(json.loads(json.dumps(eu))[0]["license"], "CC-BY-SA-4.0")


if __name__ == "__main__":
    unittest.main()
