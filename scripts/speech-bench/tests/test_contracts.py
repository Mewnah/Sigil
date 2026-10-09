# SPDX-License-Identifier: AGPL-3.0-or-later
"""The frozen formats: every example and fixture validates, the validator catches what it must,
and the reference export derivation follows the privacy rules."""

import copy
import glob
import json
import os
import unittest

from tests.helpers import BENCH, EXAMPLES, FIXTURES, load_json

import validate
from score.common import canonical_sha256
from score.make_export import make_export, model_label, scrub


class SchemaTests(unittest.TestCase):
    def test_every_schema_is_a_valid_draft_2020_12_schema(self):
        import jsonschema

        paths = glob.glob(os.path.join(BENCH, "schemas", "*.json")) + glob.glob(os.path.join(validate.LISTENING_SCHEMAS, "*.json"))
        self.assertGreaterEqual(len(paths), 10)
        for p in paths:
            jsonschema.Draft202012Validator.check_schema(load_json(p))

    def test_examples_and_fixtures_validate(self):
        files = sorted(glob.glob(os.path.join(EXAMPLES, "*.json"))) + [os.path.join(FIXTURES, "manifest.small.json"), os.path.join(BENCH, "corpus", "recipe.json")]
        self.assertGreaterEqual(len(files), 8)
        for p in files:
            self.assertEqual(validate.validate_file(p), [], p)
        self.assertEqual(validate.validate_jsonl("mt_line", os.path.join(FIXTURES, "mt_set.small.jsonl")), [])
        self.assertEqual(validate.validate_jsonl("tts_prompt", os.path.join(FIXTURES, "prompts.small.jsonl")), [])

    def test_spec_schema_rejects_unknown_fields_and_bad_values(self):
        spec = load_json(os.path.join(EXAMPLES, "spec.accuracy.json"))
        bad = copy.deepcopy(spec)
        bad["runs"][0]["moed"] = "accuracy"
        self.assertTrue(validate.schema_errors(bad))
        bad = copy.deepcopy(spec)
        bad["engines"][0]["execution_provider"] = "gpu"
        self.assertTrue(validate.schema_errors(bad))
        bad = copy.deepcopy(spec)
        bad["runs"][0]["mode"] = "wer"
        self.assertTrue(validate.schema_errors(bad))
        bad = copy.deepcopy(spec)
        bad["runs"].append({"name": "mt", "mode": "mt_quality"})
        self.assertTrue(validate.schema_errors(bad), "mt_quality needs mt options")

    def test_spec_cross_references(self):
        spec = load_json(os.path.join(EXAMPLES, "spec.accuracy.json"))
        self.assertEqual(validate.spec_errors(spec), [])
        bad = copy.deepcopy(spec)
        bad["engines"][0]["backend"] = "nllb"
        self.assertTrue(any("not a stt backend" in e for e in validate.spec_errors(bad)))
        bad = copy.deepcopy(spec)
        bad["runs"][0]["engines"] = ["nope"]
        self.assertTrue(any("unknown engine" in e for e in validate.spec_errors(bad)))
        bad = copy.deepcopy(spec)
        bad["engines"].append(dict(bad["engines"][0]))
        self.assertTrue(any("duplicate id" in e for e in validate.spec_errors(bad)))
        bad = copy.deepcopy(spec)
        bad["engines"][0]["options"] = {"apiKey": "x"}
        self.assertTrue(any("secret" in e for e in validate.spec_errors(bad)))
        mt = load_json(os.path.join(EXAMPLES, "spec.mt_quality.json"))
        mt["runs"][0]["engines"] = ["hy-mt2-1.8b"]
        mt["runs"].append({"name": "x", "mode": "accuracy", "engines": ["hy-mt2-1.8b"]})
        self.assertTrue(any("takes ['stt']" in e for e in validate.spec_errors(mt)))

    def test_report_items_are_typed(self):
        report = load_json(os.path.join(EXAMPLES, "report.example.json"))
        bad = copy.deepcopy(report)
        bad["runs"][0]["results"][0]["items"][0]["wer"] = 0.1
        self.assertTrue(validate.schema_errors(bad), "items never carry scores")
        bad = copy.deepcopy(report)
        del bad["environment"]["logical_cores"]
        self.assertTrue(validate.schema_errors(bad))

    def test_spec_hash_is_canonical(self):
        report = load_json(os.path.join(EXAMPLES, "report.example.json"))
        self.assertEqual(report["spec_sha256"], canonical_sha256(report["spec"]))
        shuffled = json.loads(json.dumps(report["spec"]), object_pairs_hook=lambda kv: dict(reversed(kv)))
        self.assertEqual(canonical_sha256(shuffled), report["spec_sha256"])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.report = load_json(os.path.join(EXAMPLES, "report.example.json"))

    def test_example_export_is_the_reference_derivation(self):
        want = load_json(os.path.join(EXAMPLES, "export.example.json"))
        got = make_export(self.report, include_text=False, include_items=True, note="Format example (made-up numbers).", now_ms=want["export"]["created_unix_ms"])
        self.assertEqual(got, want)
        self.assertEqual(validate.privacy_errors(got), [])

    def test_privacy_rules(self):
        exp = make_export(self.report)
        for run in exp["runs"]:
            self.assertNotIn("files", run)
            for res in run["results"]:
                for it in res["items"]:
                    self.assertIsNone(it.get("text"))
                    self.assertIsNone(it.get("audio"))
        with_text = make_export(self.report, include_text=True)
        texts = [it.get("text") for r in with_text["runs"] for res in r["results"] for it in res["items"] if it.get("text")]
        self.assertTrue(texts)
        self.assertEqual(validate.privacy_errors(with_text), [])
        no_items = make_export(self.report, include_items=False)
        self.assertTrue(all(not res["items"] for r in no_items["runs"] for res in r["results"]))

    def test_privacy_checker_catches_leaks(self):
        exp = make_export(self.report)
        leak = copy.deepcopy(exp)
        leak["runs"][0]["results"][0]["error"] = "C:\\Users\\alice\\AppData\\Local\\x.onnx missing"
        self.assertTrue(validate.privacy_errors(leak))
        leak = copy.deepcopy(exp)
        leak["runs"][0]["results"][0]["items"][0]["text"] = "hello"
        self.assertTrue(any("include_text is false" in e for e in validate.privacy_errors(leak)))
        leak = copy.deepcopy(exp)
        leak["export"]["include_text"] = True
        leak["runs"][0]["results"][0]["items"][0]["item"] = "real/s/clip"
        leak["runs"][0]["results"][0]["items"][0]["text"] = "private words"
        self.assertTrue(any("real clip" in e for e in validate.privacy_errors(leak)))
        leak = copy.deepcopy(exp)
        leak["environment"]["hostname"] = "ALICE-PC"
        self.assertTrue(validate.privacy_errors(leak))

    def test_scrub_and_model_labels(self):
        self.assertEqual(scrub("cannot open C:\\Users\\bob\\models\\x.gguf now"), "cannot open <path>/x.gguf now")
        self.assertEqual(scrub({"a": ["/home/bob/m.bin"]}), {"a": ["<path>/m.bin"]})
        self.assertEqual(model_label("stt/qwen3-asr-0.6b-int8"), "stt/qwen3-asr-0.6b-int8")
        self.assertEqual(model_label("D:/models/mine.onnx"), "byo")


if __name__ == "__main__":
    unittest.main()
