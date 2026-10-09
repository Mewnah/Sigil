# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden tests of the normalizers, WER / CER alignment, bootstrap and the STT / MT / TTS scorers."""

import os
import unittest

from tests.helpers import EXAMPLES, FIXTURES, load_json

from score.bootstrap import bootstrap_ci, paired_bootstrap, paired_bootstrap_stat
from score.normalize import describe, graphemes, metric_for, normalize, tokens
from score.wer import align_counts, score_pair


class NormalizeTests(unittest.TestCase):
    def test_english_follows_whisper(self):
        cases = {
            "Unfortunately, studying traffic flow is difficult because driver behavior cannot be predicted with one-hundred percent certainty.":
                "unfortunately studying traffic flow is difficult because driver behavior cannot be predicted with 100% certainty",
            "I can't believe it's twenty five dollars, Mr. Smith!": "i can not believe it is $25 mister smith",
            "[laughs] Um, the colour is grey (I think).": "the color is gray",
            "Twenty-five people came.": "25 people came",
            "We won 3,000 games in the 1960s.": "we won 3000 games in the 1960s",
            "<|en|><|HAPPY|>Hello there.": "hello there",
        }
        for raw, want in cases.items():
            self.assertEqual(normalize(raw, "en"), want, raw)

    def test_basic_languages_keep_diacritics(self):
        self.assertEqual(normalize("L'homme a dit: «Ça va?»", "fr"), "l homme a dit ça va")
        self.assertEqual(normalize("¿Dónde está el baño?", "es"), "dónde está el baño")
        self.assertEqual(normalize("Größe, Straße!", "de"), "größe straße")
        self.assertEqual(normalize("Ёлка, ёжик — всё хорошо!", "ru"), "елка ежик все хорошо")

    def test_cjk_drops_punctuation_spaces_and_widths(self):
        self.assertEqual(normalize("「今日は、ＶＲＣｈａｔで遊ぼう！」ｶﾀｶﾅ 123", "ja"), "今日はvrchatで遊ぼうカタカナ123")
        self.assertEqual(normalize("ラーメン、々 （笑）", "ja"), "ラーメン々")
        self.assertEqual(normalize("안녕하세요, 반갑습니다!", "ko"), "안녕하세요반갑습니다")
        self.assertEqual(normalize("<|zh|>你好，世界。", "zh"), "你好世界")
        self.assertEqual(normalize("", "ja"), "")
        self.assertEqual(normalize(None, "en"), "")

    def test_metric_and_units(self):
        self.assertEqual(metric_for("ja-JP"), "cer")
        self.assertEqual(metric_for("zh_Hans"), "cer")
        self.assertEqual(metric_for("ko"), "cer")
        self.assertEqual(metric_for("pt"), "wer")
        self.assertEqual(tokens("今日は", "ja"), ["今", "日", "は"])
        self.assertEqual(tokens("Hello there", "en"), ["hello", "there"])
        self.assertEqual(tokens("Hello", "en", "char"), list("hello"))
        self.assertEqual(graphemes("가 각"), ["가", "각"])

    def test_describe_is_stamped(self):
        d = describe()
        self.assertEqual(d["version"], "sigil-norm/1")
        self.assertIn(d["english_spelling"]["source"], ("english.json", "english_spelling_sigil.json"))
        self.assertEqual(len(d["english_spelling"]["sha256"]), 64)


class AlignmentTests(unittest.TestCase):
    def test_counts(self):
        c = align_counts("a b c d".split(), "a x c d e".split())
        self.assertEqual((c.sub, c.dele, c.ins, c.ref_len), (1, 0, 1, 4))
        self.assertAlmostEqual(c.rate(), 0.5)
        c = align_counts("a b c".split(), "a c".split())
        self.assertEqual((c.sub, c.dele, c.ins), (0, 1, 0))
        c = align_counts([], ["x"])
        self.assertEqual((c.ins, c.rate()), (1, None))
        c = align_counts(["x", "y"], [])
        self.assertEqual((c.dele, c.rate()), (2, 1.0))
        c = align_counts("the cat sat".split(), "the cat sat".split())
        self.assertEqual(c.errors, 0)

    def test_minimum_edit_distance(self):
        # kitten -> sitting: 3 edits (2 substitutions, 1 insertion)
        c = align_counts(list("kitten"), list("sitting"))
        self.assertEqual(c.errors, 3)
        self.assertEqual((c.sub, c.ins), (2, 1))

    def test_score_pair_normalizes_first(self):
        self.assertEqual(score_pair("Twenty five people came.", "25 people came", "en").errors, 0)
        c = score_pair("今日は良い天気", "今日はいい天気", "ja")
        self.assertEqual((c.sub, c.ref_len), (1, 7))


class BootstrapTests(unittest.TestCase):
    def test_ci_brackets_the_rate_and_is_seeded(self):
        errs, units = [1, 0, 3, 2, 0, 1, 5, 0], [10, 8, 12, 9, 11, 10, 15, 7]
        r, lo, hi = bootstrap_ci(errs, units, 500, 1)
        self.assertAlmostEqual(r, 12 / 82)
        self.assertLessEqual(lo, r)
        self.assertGreaterEqual(hi, r)
        self.assertEqual(bootstrap_ci(errs, units, 500, 1), (r, lo, hi))
        self.assertEqual(bootstrap_ci([], [], 10, 1), (None, None, None))

    def test_paired(self):
        a_e, b_e, u = [0, 1, 0, 0, 1, 0], [2, 3, 1, 2, 2, 1], [10] * 6
        res = paired_bootstrap(a_e, u, b_e, u, 500, 3)
        self.assertLess(res["delta"], 0)
        self.assertLess(res["ci95"][1], 0)
        self.assertGreater(res["prob_a_better"], 0.95)
        same = paired_bootstrap(a_e, u, a_e, u, 200, 3)
        self.assertEqual(same["delta"], 0)
        self.assertEqual(same["p_two_sided"], 1.0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1], [1, 2], [1, 2])

    def test_paired_stat_higher_is_better(self):
        a = [[3, 4], [4, 4], [2, 4]]
        b = [[1, 4], [2, 4], [1, 4]]
        res = paired_bootstrap_stat(lambda s: s[0] / s[1], a, b, 300, 1, higher_is_better=True)
        self.assertGreater(res["delta"], 0)
        self.assertGreater(res["prob_a_better"], 0.9)


class ScorerTests(unittest.TestCase):
    def test_stt_scores_of_the_example_report(self):
        from score.score_stt import score, to_csv, to_markdown

        report = load_json(os.path.join(EXAMPLES, "report.example.json"))
        manifest = load_json(os.path.join(FIXTURES, "manifest.small.json"))
        s, items = score([("example", report)], manifest, b=200, seed=1)
        rows = {(r["label"], r["language"], r["condition"]): r for r in s["rows"]}
        # en clean: both engines exact after normalization (digits vs words, punctuation, hyphens)
        self.assertEqual(rows[("qwen3-asr-0.6b", "en", "clean")]["rate"], 0.0)
        self.assertEqual(rows[("parakeet-ultra", "en", "clean")]["rate"], 0.0)
        # en music: qwen3 "to night" (1 sub + 1 ins) and "came to party" (1 del) over 8 + 6 reference words
        self.assertEqual(rows[("qwen3-asr-0.6b", "en", "music_snr10")]["errors"], 3)
        self.assertEqual(rows[("qwen3-asr-0.6b", "en", "music_snr10")]["ref_units"], 14)
        # ja music: one failed item counts as all deletions (failed_as empty)
        ja = rows[("qwen3-asr-0.6b", "ja", "music_snr10")]
        self.assertEqual(ja["failed"], 1)
        self.assertEqual(ja["metric"], "cer")
        self.assertEqual(ja["del"] + ja["sub"] + ja["ins"], ja["errors"])
        deltas = {(d["label"], d["language"], d["condition"]): d for d in s["deltas_vs_clean"]}
        self.assertGreater(deltas[("qwen3-asr-0.6b", "en", "music_snr10")]["delta"], 0)
        self.assertEqual(len(s["comparisons"]), 2)
        self.assertIn("| en | clean |", to_markdown(s))
        self.assertTrue(to_csv(s).startswith("label,"))
        self.assertEqual(len(items), 12)
        excluded, _ = score([("example", report)], manifest, b=50, seed=1, failed_as="exclude")
        ja_ex = {(r["label"], r["language"], r["condition"]): r for r in excluded["rows"]}[("qwen3-asr-0.6b", "ja", "music_snr10")]
        self.assertEqual(ja_ex["excluded"], 1)
        self.assertLess(ja_ex["ref_units"], ja["ref_units"])

    def test_mt_chrf_matches_sacrebleu(self):
        from sacrebleu.metrics import CHRF

        from score.score_mt import score

        systems = {
            "good": {"1": {"direction": "en-de", "hyp": "Die Katze sitzt auf der Matte.", "failed": False, "src": "x", "refs": ["Die Katze sitzt auf der Matte."], "target": "de", "source": "en"},
                     "2": {"direction": "en-de", "hyp": "Guten Morgen", "failed": False, "src": "x", "refs": ["Guten Morgen!"], "target": "de", "source": "en"}},
            "bad": {"1": {"direction": "en-de", "hyp": "Der Hund schläft.", "failed": False, "src": "x", "refs": ["Die Katze sitzt auf der Matte."], "target": "de", "source": "en"},
                    "2": {"direction": "en-de", "hyp": "", "failed": True, "src": "x", "refs": ["Guten Morgen!"], "target": "de", "source": "en"}},
        }
        s = score(systems, b=200, seed=1)
        rows = {r["label"]: r for r in s["rows"]}
        want = CHRF().corpus_score(["Die Katze sitzt auf der Matte.", "Guten Morgen"], [["Die Katze sitzt auf der Matte.", "Guten Morgen!"]]).score
        self.assertAlmostEqual(rows["good"]["chrf"], want, places=6)
        self.assertEqual(rows["bad"]["failed"], 1)
        comp = s["comparisons"][0]
        self.assertEqual((comp["a_label"], comp["b_label"]), ("bad", "good"))
        self.assertLess(comp["delta"], 0)

    def test_mt_metricx_merge_and_pairs(self):
        import tempfile

        from score.score_mt import export_pairs, score

        systems = {
            "a": {"1": {"direction": "ja-en", "hyp": "Hello.", "failed": False, "src": "こんにちは。", "refs": ["Hello."], "target": "en", "source": "ja"}},
            "b": {"1": {"direction": "ja-en", "hyp": "Hi.", "failed": False, "src": "こんにちは。", "refs": ["Hello."], "target": "en", "source": "ja"}},
        }
        s = score(systems, metricx={("a", "1"): 1.5, ("b", "1"): 4.0}, b=50, seed=1)
        rows = {r["label"]: r for r in s["rows"]}
        self.assertEqual(rows["a"]["metricx"], 1.5)
        self.assertTrue(any(c["metric"] == "metricx" for c in s["comparisons"]))
        with tempfile.TemporaryDirectory() as d:
            paths = export_pairs(systems, d)
            self.assertEqual(len(paths), 1)
            import json

            with open(paths[0], encoding="utf-8") as f:
                line = json.loads(f.readline())
            self.assertEqual(line["systems"], {"a": "Hello.", "b": "Hi."})

    def test_merge_exports(self):
        import json
        import tempfile

        from score.merge_exports import main as merge_main

        with tempfile.TemporaryDirectory() as d:
            leak = load_json(os.path.join(EXAMPLES, "export.example.json"))
            leak["environment"]["hostname"] = "PC"
            bad = os.path.join(d, "leak.json")
            with open(bad, "w", encoding="utf-8") as f:
                json.dump(leak, f)
            out = os.path.join(d, "merged")
            self.assertEqual(merge_main([os.path.join(EXAMPLES, "export.example.json"), os.path.join(EXAMPLES, "report.example.json"), bad, "--out", out]), 0)
            merged = load_json(out + ".json")
            self.assertEqual([r["file"] for r in merged["refused"]], ["leak.json"])
            self.assertEqual(len(merged["rows"]), 2 * 4)
            self.assertIn("H2: AMD Ryzen 7 7800X3D", merged["rows"][0]["machine"])

    def test_tts_prescreen(self):
        from score.common import load_jsonl
        from score.score_tts import score

        report = load_json(os.path.join(EXAMPLES, "report.example.json"))
        prompts = {p["id"]: p for p in load_jsonl(os.path.join(FIXTURES, "prompts.small.jsonl"))}
        s = score([("example", report)], prompts, b=50, seed=1)
        rows = {(r["label"], r["language"]): r for r in s["rows"]}
        self.assertEqual(rows[("kokoro", "en")]["intelligibility"], 0.0)
        self.assertEqual(rows[("kokoro", "en")]["ttfa_ms_p50"], 310.0)
        self.assertEqual(rows[("kokoro", "ja")]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
