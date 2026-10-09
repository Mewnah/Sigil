# SPDX-License-Identifier: AGPL-3.0-or-later
"""tools/listening-test: the MT / TTS builders (blinding, selection, loudness), the aggregator, the
MetricX wrapper's detection, and (when Node + the repo's Playwright are present) both pages driven
headless from file:// up to the exported answers."""

import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import unittest

import numpy as np

from tests.helpers import LISTENING, REPO, load_json, voice_like, write_json, write_text

import validate


def _pairs_file(path, n=12):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            systems = {"alpha": f"Translation {i} by alpha.", "beta": f"Translation {i} by beta." if i % 4 else f"Translation {i} by alpha.", "gamma": None if i == 5 else f"G {i}"}
            f.write(json.dumps({"id": f"bsd:ja-en:{i + 1}", "source": "ja", "target": "en", "src": f"文 {i}", "ref": f"Ref {i}", "systems": systems}, ensure_ascii=False) + "\n")


class MtBuilderTests(unittest.TestCase):
    def test_blind_ab_build(self):
        import make_mt_test

        with tempfile.TemporaryDirectory() as tmp:
            pairs = os.path.join(tmp, "pairs.ja-en.jsonl")
            _pairs_file(pairs)
            ctx = os.path.join(tmp, "set.jsonl")
            with open(ctx, "w", encoding="utf-8") as f:
                for i in range(12):
                    f.write(json.dumps({"id": f"bsd:ja-en:{i + 1}", "source": "ja", "target": "en", "src": f"文 {i}", "ref": "r", "set": "bsd", "doc": f"s1#{i + 1}"}, ensure_ascii=False) + "\n")
            out = os.path.join(tmp, "ab")
            rc = make_mt_test.main(["--pairs", pairs, "--systems", "alpha,beta,gamma", "--per-direction", "20", "--seed", "7", "--context", ctx, "--out", out])
            self.assertEqual(rc, 0)
            manifest = load_json(os.path.join(out, "test", "manifest.json"))
            key = load_json(os.path.join(out, "key.json"))
            self.assertEqual(validate.schema_errors(manifest), [])
            self.assertEqual(validate.schema_errors(key), [])
            with open(os.path.join(out, "test", "manifest.js"), encoding="utf-8") as f:
                text = f.read()
            for sysname in ("alpha", "beta", "gamma"):
                self.assertNotIn(f'"{sysname}"', text, "system names never reach the page")
            sel = key["selection"]
            self.assertEqual(sel["alpha vs beta | ja-en"]["identical_skipped"], 3)
            self.assertEqual(sel["alpha vs gamma | ja-en"]["failed_skipped"], 1)
            self.assertEqual(len(manifest["trials"]), 9 + 11 + 11)
            sides = [t["A"] for t in key["trials"] if {t["A"], t["B"]} == {"alpha", "beta"}]
            self.assertTrue(0 < sides.count("alpha") < len(sides), "A / B sides are randomized")
            with_ctx = [t for t in manifest["trials"] if t["context"]]
            self.assertTrue(with_ctx)
            self.assertTrue(os.path.isfile(os.path.join(out, "test", "index.html")))
            # same seed, same blinding (test id differs: it never derives from the seed)
            out2 = os.path.join(tmp, "ab2")
            make_mt_test.main(["--pairs", pairs, "--systems", "alpha,beta,gamma", "--per-direction", "20", "--seed", "7", "--context", ctx, "--out", out2])
            key2 = load_json(os.path.join(out2, "key.json"))
            self.assertEqual([(t["item"], t["A"]) for t in key["trials"]], [(t["item"], t["A"]) for t in key2["trials"]])
            self.assertNotEqual(key["test_id"], key2["test_id"])


class TtsBuilderTests(unittest.TestCase):
    def test_mushra_and_pairwise_builds_with_loudness_match(self):
        import soundfile as sf

        import make_tts_test

        with tempfile.TemporaryDirectory() as tmp:
            prompts = os.path.join(tmp, "prompts.jsonl")
            with open(prompts, "w", encoding="utf-8") as f:
                for i in range(4):
                    f.write(json.dumps({"id": f"en-casual-{i + 1}", "lang": "en", "text": f"Line {i}", "kind": "casual"}) + "\n")
            report = {"schema": "sigil-speech-bench-report/1", "name": "tts", "engines": [], "runs": [{"name": "r", "mode": "tts_listening", "results": []}]}
            for e, (rate, level) in {"kokoro": (24000, 0.05), "qwen": (24000, 0.4), "magpie": (22050, 0.1)}.items():
                report["engines"].append({"id": e, "task": "tts"})
                items = []
                for i in range(4):
                    rel = f"audio/{e}/en-casual-{i + 1}.wav"
                    os.makedirs(os.path.join(tmp, "audio", e), exist_ok=True)
                    x = voice_like(1.5, seed=i, sr=rate, level=level)
                    if e == "magpie" and i == 3:
                        items.append({"item": f"en-casual-{i + 1}", "language": "en", "audio": None, "error": "failed"})
                        continue
                    sf.write(os.path.join(tmp, rel), x, rate, subtype="PCM_16")
                    items.append({"item": f"en-casual-{i + 1}", "language": "en", "audio": rel, "error": None})
                report["runs"][0]["results"].append({"engine": e, "items": items})
            rpath = os.path.join(tmp, "report.json")
            write_json(rpath, report)
            out = os.path.join(tmp, "t1")
            rc = make_tts_test.main(["--report", rpath, "--prompts", prompts, "--engines", "kokoro,qwen,magpie", "--per-language", "4", "--seed", "3", "--out", out])
            self.assertEqual(rc, 0)
            manifest = load_json(os.path.join(out, "test", "manifest.json"))
            key = load_json(os.path.join(out, "key.json"))
            self.assertEqual(validate.schema_errors(manifest), [])
            self.assertEqual(validate.schema_errors(key), [])
            self.assertEqual(len(manifest["trials"]), 3, "the prompt magpie failed is not a trial")
            self.assertEqual(key["audio"]["rate"], 24000)
            levels = []
            for t in manifest["trials"]:
                self.assertEqual(len(t["stimuli"]), 3)
                for s in t["stimuli"]:
                    x, sr = sf.read(os.path.join(out, "test", s["audio"]))
                    self.assertEqual(sr, 24000)
                    levels.append(make_tts_test.active_dbfs(x, sr))
            self.assertLess(max(levels) - min(levels), 0.2, "renditions are loudness matched")
            out2 = os.path.join(tmp, "t2")
            make_tts_test.main(["--report", rpath, "--prompts", prompts, "--engines", "kokoro,qwen", "--mode", "pairwise", "--seed", "3", "--out", out2])
            m2 = load_json(os.path.join(out2, "test", "manifest.json"))
            self.assertEqual(validate.schema_errors(m2), [])
            self.assertEqual(len(m2["trials"]), 4)


class AggregateTests(unittest.TestCase):
    def _answers(self, tmp, key, who, fn):
        a = {"schema": "sigil-listening-results/1", "test_id": key["test_id"], "kind": key["kind"], "mode": key.get("mode"), "rater": who,
             "exported_unix_ms": 1, "answers": {t["id"]: fn(t) for t in key["trials"]}}
        p = os.path.join(tmp, f"answers-{who}.json")
        write_json(p, a)
        self.assertEqual(validate.schema_errors(a), [])
        return p

    def test_pairwise(self):
        import aggregate

        key = {"schema": "sigil-listening-key/1", "test_id": "0123456789abcdef", "kind": "mt", "mode": "pairwise", "seed": 1, "systems": ["a", "b", "c"],
               "trials": [{"id": f"t{i:03d}", "item": str(i), "direction": "ja-en", "A": ["a", "b"][i % 2], "B": ["b", "a"][i % 2]} for i in range(20)]
               + [{"id": f"t{i:03d}", "item": str(i), "direction": "ja-en", "A": "c", "B": "a"} for i in range(20, 30)]}
        with tempfile.TemporaryDirectory() as tmp:
            prefer_a = lambda t: {"choice": "A" if t["A"] == "a" else ("B" if t["B"] == "a" else "tie")}  # noqa: E731
            paths = [self._answers(tmp, key, w, prefer_a) for w in ("x", "y", "z")]
            paths.append(self._answers(tmp, key, "lazy", lambda t: {} if t["id"] != "t000" else {"choice": "A"}))
            res = aggregate.aggregate(key, paths, min_complete=0.5, b=100)
            self.assertEqual(res["listeners"], ["x", "y", "z"])
            self.assertIn("lazy", res["excluded"])
            rows = {(r["scope"], tuple(r["systems"])): r for r in res["pairwise"]["pairs"]}
            ab = rows[("all", ("a", "b"))]
            self.assertEqual(ab["shares"]["a"]["preferred"], 60)
            self.assertEqual(ab["shares"]["a"]["share"], 1.0)
            self.assertGreater(ab["shares"]["a"]["ci95"][0], 0.9)
            bt = res["pairwise"]["bradley_terry"]
            self.assertGreater(bt["a"], bt["b"])
            self.assertIn("| all | a | b |", aggregate.to_markdown(res))
            other = dict(key, test_id="fedcba9876543210")
            with self.assertRaises(SystemExit):
                aggregate.aggregate(other, paths)

    def test_mushra(self):
        import aggregate

        rng = random.Random(1)
        key = {"schema": "sigil-listening-key/1", "test_id": "0123456789abcdef", "kind": "tts", "mode": "mushra", "seed": 1, "systems": ["k", "q", "m"],
               "trials": [{"id": f"t{i:03d}", "item": f"en-read-{i}", "lang": "en" if i < 6 else "ja", "stimuli": {"s1": "k", "s2": "q", "s3": "m"}} for i in range(10)]}
        base = {"k": 60, "q": 75, "m": 50}
        with tempfile.TemporaryDirectory() as tmp:
            paths = [self._answers(tmp, key, w, lambda t: {"ratings": {s: base[sysn] + rng.randint(-5, 5) + off for s, sysn in t["stimuli"].items()}, "played": ["s1", "s2", "s3"]})
                     for w, off in (("x", 0), ("y", 10), ("z", -10))]
            res = aggregate.aggregate(key, paths, baseline="k", b=200)
            sysres = res["mushra"]["systems"]
            self.assertGreater(sysres["q"]["mean"], sysres["k"]["mean"])
            self.assertGreater(sysres["k"]["mean"], sysres["m"]["mean"])
            self.assertLessEqual(sysres["q"]["ci95"][0], sysres["q"]["mean"])
            d = res["mushra"]["vs_baseline"]["q"]
            self.assertGreater(d["ci95"][0], 0)
            self.assertIn("ja", res["mushra"]["by_language"])
            self.assertIn("| q |", aggregate.to_markdown(res))


class MetricxTests(unittest.TestCase):
    def test_detection_without_files(self):
        from score import metricx

        with tempfile.TemporaryDirectory() as tmp:
            st = metricx.detect("python", os.path.join(tmp, "no-model"), os.path.join(tmp, "no-tok"), os.path.join(tmp, "no-code"))
            self.assertFalse(st["ready"])
            self.assertEqual(sorted(st["missing"]), ["code", "model", "tokenizer"])

    def test_code_archive_is_extracted_once(self):
        from score import metricx

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src", "metricx-abc", "metricx24")
            os.makedirs(src)
            write_text(os.path.join(src, "models.py"), "# fake\n")
            arc = os.path.join(tmp, "code")
            os.makedirs(arc)
            with tarfile.open(os.path.join(arc, "metricx-abc.tar.gz"), "w:gz") as tar:
                tar.add(os.path.join(tmp, "src", "metricx-abc"), arcname="metricx-abc")
            d = metricx.code_dir(arc)
            self.assertTrue(os.path.isfile(os.path.join(d, "metricx24", "models.py")))
            self.assertEqual(metricx.code_dir(arc), d)

    def test_flatten_pairs(self):
        from score import metricx

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "pairs.ja-en.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "1", "source": "ja", "target": "en", "src": "s", "ref": "r", "systems": {"a": "x", "b": None}}) + "\n")
            rows = metricx.flatten_pairs([p])
            self.assertEqual([(r["system"], r["hyp"], r["direction"]) for r in rows], [("a", "x", "ja-en"), ("b", "", "ja-en")])


def _node():
    return shutil.which("node")


@unittest.skipUnless(_node() and os.path.isdir(os.path.join(REPO, "node_modules", "@playwright", "test")), "node / Playwright not installed")
class PageTests(unittest.TestCase):
    """Drives mt-ab.html and tts-listen.html headless (Chromium) from file://; see pages_smoke.mjs."""

    def test_pages_export_answers(self):
        import soundfile as sf

        import make_mt_test
        import make_tts_test

        with tempfile.TemporaryDirectory() as tmp:
            pairs = os.path.join(tmp, "pairs.ja-en.jsonl")
            _pairs_file(pairs, 6)
            make_mt_test.main(["--pairs", pairs, "--systems", "alpha,gamma", "--per-direction", "3", "--seed", "1", "--out", os.path.join(tmp, "mt")])
            prompts = os.path.join(tmp, "prompts.jsonl")
            with open(prompts, "w", encoding="utf-8") as f:
                for i in range(2):
                    f.write(json.dumps({"id": f"en-casual-{i + 1}", "lang": "en", "text": f"Line {i}", "kind": "casual"}) + "\n")
            report = {"schema": "sigil-speech-bench-report/1", "name": "tts", "engines": [], "runs": [{"name": "r", "mode": "tts_listening", "results": []}]}
            for e in ("k", "q"):
                report["engines"].append({"id": e, "task": "tts"})
                items = []
                for i in range(2):
                    rel = f"audio/{e}/{i}.wav"
                    os.makedirs(os.path.join(tmp, "audio", e), exist_ok=True)
                    sf.write(os.path.join(tmp, rel), voice_like(0.6, seed=i, sr=16000), 16000, subtype="PCM_16")
                    items.append({"item": f"en-casual-{i + 1}", "language": "en", "audio": rel, "error": None})
                report["runs"][0]["results"].append({"engine": e, "items": items})
            write_json(os.path.join(tmp, "report.json"), report)
            make_tts_test.main(["--report", os.path.join(tmp, "report.json"), "--prompts", prompts, "--engines", "k,q", "--seed", "1", "--out", os.path.join(tmp, "tts")])
            out = os.path.join(tmp, "results")
            os.makedirs(out)
            script = os.path.join(LISTENING, "tests", "pages_smoke.mjs")
            proc = subprocess.run([_node(), script, os.path.join(tmp, "mt", "test", "index.html"), os.path.join(tmp, "tts", "test", "index.html"), out],
                                  cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
            if proc.returncode == 3:
                self.skipTest("Playwright's Chromium is not installed: " + proc.stderr[-300:])
            self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
            mt = load_json(os.path.join(out, "mt.json"))
            tts = load_json(os.path.join(out, "tts.json"))
            for res, kind in ((mt, "mt"), (tts, "tts")):
                self.assertEqual(validate.schema_errors(res), [], kind)
                self.assertEqual(res["kind"], kind)
            mt_key = load_json(os.path.join(tmp, "mt", "key.json"))
            self.assertEqual(mt["test_id"], mt_key["test_id"])
            self.assertEqual(len([a for a in mt["answers"].values() if a.get("choice")]), len(mt_key["trials"]))
            self.assertEqual(mt["rater"], "smoke")
            tts_key = load_json(os.path.join(tmp, "tts", "key.json"))
            first = tts_key["trials"][0]["id"]
            self.assertEqual(sorted(tts["answers"][first]["ratings"]), ["s1", "s2"])
            import aggregate

            p = os.path.join(out, "mt.json")
            res = aggregate.aggregate(mt_key, [p], min_complete=0.5, b=50)
            self.assertEqual(res["listeners"], ["smoke"])


if __name__ == "__main__":
    unittest.main()
