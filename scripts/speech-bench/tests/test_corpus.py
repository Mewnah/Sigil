# SPDX-License-Identifier: AGPL-3.0-or-later
"""Corpus builder: DSP definitions, deterministic synthesis, an end-to-end build on a tiny
synthetic P7 root (determinism, achieved SNRs, resume, real drop-in clips) and the text sets."""

import argparse
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
import wave

import numpy as np

from tests.helpers import load_json, make_fake_p7, small_recipe, voice_like, wav_bytes

import validate
from corpus import dsp, sources
from corpus.build_corpus import build, effective_recipe, read_pcm16


class DspTests(unittest.TestCase):
    def test_active_level_is_scale_invariant_and_ignores_pauses(self):
        x = voice_like(4.0, seed=1)
        p = dsp.active_power(x)
        self.assertAlmostEqual(dsp.active_power(x * 0.001) / p, 1e-6, delta=1e-9)
        padded = np.concatenate([np.zeros(16000 * 3), x, np.zeros(16000 * 3)])
        self.assertAlmostEqual(dsp.power_to_db(dsp.active_power(padded)), dsp.power_to_db(p), delta=0.05)
        self.assertEqual(dsp.active_power(np.zeros(1000)), 0.0)

    def test_gain_for_snr(self):
        s = voice_like(4.0, seed=2)
        n = np.random.default_rng(3).standard_normal(len(s)) * 0.3
        for snr in (20.0, 10.0, 5.0, 0.0):
            g = dsp.gain_for_snr(dsp.active_power(s), dsp.active_power(n), snr)
            got = dsp.power_to_db(dsp.active_power(s) / dsp.active_power(n * g))
            self.assertAlmostEqual(got, snr, places=6)

    def test_room_drr_and_length(self):
        rng = np.random.default_rng(0)
        h = dsp.synthetic_rir(rng, 0.35, 6.0)
        tail = h[1:]
        self.assertAlmostEqual(10 * np.log10(1.0 / np.sum(tail ** 2)), 6.0, places=6)
        x = voice_like(2.0, seed=4)
        self.assertEqual(len(dsp.convolve_same(x, h)), len(x))

    def test_mu_law_and_filters(self):
        x = np.linspace(-0.9, 0.9, 1001)
        y = dsp.mu_law(x, 8)
        self.assertLess(np.max(np.abs(y - x)), 0.03)
        t = np.arange(16000) / 16000
        low = np.sin(2 * np.pi * 100 * t)
        self.assertLess(np.std(dsp.butter_filter(low, "highpass", 200)[4000:]), 0.5 * np.std(low))
        high = np.sin(2 * np.pi * 6000 * t)
        self.assertLess(np.std(dsp.butter_filter(high, "lowpass", 3800, order=8)[4000:]), 0.05)

    def test_headroom_and_pcm(self):
        y, g = dsp.headroom(np.array([0.5, -2.0, 1.0]))
        self.assertAlmostEqual(float(np.max(np.abs(y))), 0.98)
        self.assertLess(g, 0)
        self.assertEqual(dsp.headroom(np.array([0.1]))[1], 0.0)
        self.assertEqual(dsp.to_pcm16(np.array([1.5, -1.5, 0.0])).tolist(), [32767, -32768, 0])


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        rows = []
        for sid in range(20):
            for k, g in enumerate(("male", "female", "male")):
                rows.append({"sentence_id": sid, "file_name": f"{sid}-{k}.wav", "raw_transcription": f"s{sid}", "transcription": f"s{sid}",
                             "num_samples": 16000 * 5, "gender": g})
        a, pool = sources.select_fleurs(rows, "en", 1, 10, 3, 20, babble_pool=4)
        b, _ = sources.select_fleurs(rows, "en", 1, 10, 3, 20, babble_pool=4)
        self.assertEqual([r["file_name"] for r in a], [r["file_name"] for r in b])
        self.assertEqual(len({r["sentence_id"] for r in a}), 10)
        genders = [r["gender"] for r in a]
        self.assertEqual(genders.count("male"), genders.count("female"))
        self.assertEqual([r["rank"] for r in a], list(range(10)))
        self.assertFalse({r["sentence_id"] for r in pool} & {r["sentence_id"] for r in a})
        c, _ = sources.select_fleurs(rows, "en", 2, 10, 3, 20)
        self.assertNotEqual([r["file_name"] for r in a], [r["file_name"] for r in c])


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sigil-p7-corpus-test-")
        cls.p7 = make_fake_p7(os.path.join(cls.tmp, "p7"))
        cls.recipe_path = os.path.join(cls.tmp, "recipe.json")
        with open(cls.recipe_path, "w", encoding="utf-8") as f:
            json.dump(small_recipe(), f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def args(self, out, **kw):
        a = dict(recipe=self.recipe_path, p7_root=self.p7, out=out, languages=None, per_language=None, conditions=None, workers=1,
                 priority="normal", resume=False, no_real=False, real_only=False, hash_sources=False)
        a.update(kw)
        return argparse.Namespace(**a)

    def test_build_is_deterministic_valid_and_hits_its_snrs(self):
        out1, out2 = os.path.join(self.tmp, "c1"), os.path.join(self.tmp, "c2")
        m1 = build(self.args(out1))
        m2 = build(self.args(out2))
        self.assertEqual(m1["content_sha256"], m2["content_sha256"])
        self.assertEqual(validate.schema_errors(m1), [])
        self.assertEqual(len(m1["items"]), 2 * 3 * 14)
        by = {it["id"]: it for it in m1["items"]}
        for it in m1["items"]:
            path = os.path.join(out1, it["audio"])
            with wave.open(path, "rb") as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
            if it["condition"] in ("music_snr10", "sfx_snr5", "babble_snr20", "babble_xlang_snr10"):
                clean = read_pcm16(os.path.join(out1, by[it["utterance"] + "/clean"]["audio"]))
                y = read_pcm16(path) / 10 ** (it["mix"]["post_gain_db"] / 20)
                snr = dsp.power_to_db(dsp.active_power(clean) / dsp.active_power(y - clean))
                self.assertAlmostEqual(snr, it["mix"]["snr_db"], delta=0.3, msg=it["id"])
            if it["condition"] == "clean":
                self.assertAlmostEqual(dsp.active_dbfs(read_pcm16(path)), -26.0, delta=4.2)
        # every condition of an utterance shares the speech gain (paired)
        gains = {}
        for it in m1["items"]:
            gains.setdefault(it["utterance"], set()).add(it["mix"]["speech_gain_db"])
        self.assertTrue(all(len(v) == 1 for v in gains.values()))

    def test_resume_reuses_finished_items(self):
        out = os.path.join(self.tmp, "c3")
        m1 = build(self.args(out, conditions=["clean", "vrchat_mix"]))
        victim = os.path.join(out, m1["items"][0]["audio"])
        mtime = os.path.getmtime(os.path.join(out, m1["items"][1]["audio"]))
        os.remove(victim)
        m2 = build(self.args(out, conditions=["clean", "vrchat_mix"], resume=True))
        self.assertTrue(os.path.isfile(victim))
        self.assertEqual(m1["content_sha256"], m2["content_sha256"])
        self.assertEqual(os.path.getmtime(os.path.join(out, m1["items"][1]["audio"])), mtime)

    def test_overrides_change_the_identity(self):
        r = small_recipe()
        a = effective_recipe(r, None, None, None)
        b = effective_recipe(r, ["en"], None, None)
        from corpus.build_corpus import canonical_sha256

        self.assertNotEqual(canonical_sha256(a), canonical_sha256(b))
        with self.assertRaises(SystemExit):
            effective_recipe(r, ["xx"], None, None)

    def test_real_drop_in(self):
        out = os.path.join(self.tmp, "c4")
        build(self.args(out, conditions=["clean"]))
        sess = os.path.join(out, "real", "2026-10-05-alice")
        os.makedirs(sess, exist_ok=True)
        with open(os.path.join(sess, "session.json"), "w", encoding="utf-8") as f:
            json.dump({"language": "ja", "consent": {"given": True, "date": "2026-10-05", "scope": "local benchmarking only"}, "speaker": "alice"}, f)
        with open(os.path.join(sess, "clip01.wav"), "wb") as f:
            f.write(wav_bytes(np.repeat(voice_like(2.0, seed=9, sr=16000), 3), 48000))
        with open(os.path.join(sess, "clip01.txt"), "w", encoding="utf-8") as f:
            f.write("ちょっと待ってね\n")
        with open(os.path.join(sess, "clip02.wav"), "wb") as f:
            f.write(wav_bytes(voice_like(1.0, seed=10)))  # no transcript: skipped
        other = os.path.join(out, "real", "no-consent")
        os.makedirs(other, exist_ok=True)
        with open(os.path.join(other, "a.wav"), "wb") as f:
            f.write(wav_bytes(voice_like(1.0, seed=11)))
        with open(os.path.join(other, "a.txt"), "w", encoding="utf-8") as f:
            f.write("hello")
        m = build(self.args(out, real_only=True))
        real = [it for it in m["items"] if it["kind"] == "real"]
        self.assertEqual([it["id"] for it in real], ["real/2026-10-05-alice/clip01"])
        self.assertEqual(real[0]["reference"], "ちょっと待ってね")
        self.assertEqual(real[0]["source"]["original_rate"], 48000)
        self.assertEqual(len([it for it in m["items"] if it["kind"] == "synthetic"]), 2 * 3)
        self.assertIn("real", {c["id"] for c in m["conditions"]})
        self.assertEqual(validate.schema_errors(m), [])
        self.assertTrue(os.path.isfile(os.path.join(out, "real", "README.txt")))


class TextSetTests(unittest.TestCase):
    def test_text_sets_from_fake_sources(self):
        from corpus import build_text_sets as bts

        with tempfile.TemporaryDirectory() as tmp:
            p7 = os.path.join(tmp, "p7")
            wmt = os.path.join(p7, "eval", "wmt24pp")
            os.makedirs(wmt)
            with open(os.path.join(wmt, "en-ja_JP.jsonl"), "w", encoding="utf-8") as f:
                for i in range(6):
                    f.write(json.dumps({"lp": "en-ja_JP", "domain": "social" if i < 4 else "news", "document_id": f"d{i // 2}", "segment_id": i % 2,
                                        "is_bad_source": i == 3, "source": f"line {i}", "target": f"行{i}", "original_target": f"行{i}"}, ensure_ascii=False) + "\n")
            flores = os.path.join(p7, "eval", "flores200")
            os.makedirs(flores)
            with tarfile.open(os.path.join(flores, "flores200_dataset.tar.gz"), "w:gz") as tar:
                for code, lines in {"eng_Latn": ["one", "two", "three"], "jpn_Jpan": ["一", "二", "三"]}.items():
                    data = ("\n".join(lines) + "\n").encode("utf-8")
                    info = tarfile.TarInfo(f"./flores200_dataset/devtest/{code}.devtest")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            bsd = os.path.join(p7, "eval", "bsd-ja-en")
            os.makedirs(bsd)
            with open(os.path.join(bsd, "test.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "s1", "tag": "meeting", "title": "t", "original_language": "ja",
                            "conversation": [{"no": 1, "ja_sentence": "こんにちは", "en_sentence": "Hello"}, {"no": 2, "ja_sentence": "はい", "en_sentence": "Yes"}]}], f, ensure_ascii=False)
            w = bts.wmt24pp(p7, 3, 1)
            self.assertEqual(len(w), 3)
            self.assertTrue(all(r["domain"] == "social" for r in w))
            self.assertNotIn("line 3", [r["src"] for r in w])
            self.assertEqual(w, bts.wmt24pp(p7, 3, 1))
            fl = bts.flores200(p7, 2, 1)
            self.assertEqual({(r["source"], r["target"]) for r in fl}, {("ja", "en")})
            self.assertEqual(len(fl), 2)
            b = bts.bsd(p7, 10, 1)
            self.assertEqual(len(b), 4)
            path = os.path.join(tmp, "set.jsonl")
            bts.write_jsonl(path, w + fl + b + bts.sigil_casual())
            self.assertEqual(validate.validate_jsonl("mt_line", path), [])


if __name__ == "__main__":
    unittest.main()
