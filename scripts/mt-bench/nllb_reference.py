"""NLLB-200 reference fixtures for Sigil's Rust engine (src-tauri/src/services/mt_local/{tokenizer,nllb}.rs).

Writes scripts/mt-bench/fixtures/nllb_parity.json with, per case:
- `ids`: the source token ids: the FLORES language token of tokenizer.json, then the pieces of the *slow* Hugging
  Face NllbTokenizer, i.e. Python sentencepiece over sentencepiece.bpe.model (the reference the Rust `tokenizers`
  port must match), after checking that the fast tokenizer (tokenizer.json, what Sigil ships) agrees;
- `output_ids` / `output_text` / `output_margins` (with --model-dir): greedy decoding of the same ONNX files Sigil
  runs (encoder + merged decoder with KV cache) on Python onnxruntime, with the loop Sigil implements (decoder
  prompt `</s> <tgt_lang>`, argmax, stop at `</s>`), and the top-1 / top-2 logit gap of every step. The Rust
  real-model test must reproduce the ids, except after a near-tie (gap < 0.05) that another ONNX Runtime build
  (Sigil links 1.28; the script's Python package may be older) breaks the other way.

Maintainer tool; nothing here runs in CI. Needs: transformers (NllbTokenizer / NllbTokenizerFast), sentencepiece,
tokenizers, numpy and onnxruntime; the NLLB folder from the model store (`Sigil.exe --headless fetch
mt/nllb-200-distilled-600m-int8`) plus sentencepiece.bpe.model from the same Hugging Face revision
(Xenova/nllb-200-distilled-600M@261c31d, 4.8 MB).

Usage (repo root):
  python scripts/mt-bench/nllb_reference.py --tokenizer-dir <model dir> --spm <sentencepiece.bpe.model> \
      [--model-dir <model dir>] [--out scripts/mt-bench/fixtures/nllb_parity.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# (source text, source FLORES code, target FLORES code). Covers the normalizer (NFKC, full-width forms, repeated
# spaces), every script family Sigil's captions meet, emoji and unknown characters, and short caption-style lines.
CASES = [
    ("Hello, how are you today?", "eng_Latn", "jpn_Jpan"),
    ("I'll be right back, give me five minutes.", "eng_Latn", "deu_Latn"),
    ("That boss fight was insane!!", "eng_Latn", "spa_Latn"),
    ("  Leading and trailing   spaces  ", "eng_Latn", "fra_Latn"),
    ("Numbers: 3.14, 1,000,000 and 42%.", "eng_Latn", "kor_Hang"),
    ("Smart quotes “like this” and an ellipsis…", "eng_Latn", "zho_Hans"),
    ("Ｆｕｌｌ－ｗｉｄｔｈ ｔｅｘｔ １２３", "eng_Latn", "rus_Cyrl"),
    ("Emoji are fine \U0001f600\U0001f44d", "eng_Latn", "ita_Latn"),
    ("今日はいい天気ですね。", "jpn_Jpan", "eng_Latn"),
    ("ちょっと待って、今行くから！", "jpn_Jpan", "eng_Latn"),
    ("我们今天晚上一起玩游戏吧。", "zho_Hans", "eng_Latn"),
    ("我們今天晚上一起玩遊戲吧。", "zho_Hant", "eng_Latn"),
    ("안녕하세요, 만나서 반갑습니다.", "kor_Hang", "eng_Latn"),
    ("¿Dónde está la estación de tren?", "spa_Latn", "eng_Latn"),
    ("Die Straße ist sehr schön.", "deu_Latn", "eng_Latn"),
    ("Привет, как дела?", "rus_Cyrl", "eng_Latn"),
    ("مرحبا بك في البث", "arb_Arab", "eng_Latn"),
    ("नमस्ते, आप कैसे हैं?", "hin_Deva", "eng_Latn"),
    ("สวัสดีครับ", "tha_Thai", "eng_Latn"),
    ("Olá, tudo bem? Vamos jogar.", "por_Latn", "eng_Latn"),
    ("Xin chào các bạn", "vie_Latn", "eng_Latn"),
    ("Merhaba, nasılsın?", "tur_Latn", "eng_Latn"),
    ("Mixed 日本語 and English", "eng_Latn", "fra_Latn"),
    ("A rare symbol ☃ and a control​space", "eng_Latn", "deu_Latn"),
    ("Tab\tand　ideographic no-break  spaces", "eng_Latn", "deu_Latn"),
    ("First line.\nSecond line.", "eng_Latn", "spa_Latn"),
    ("ok", "eng_Latn", "jpn_Jpan"),
]


def clean_source(text: str) -> str:
    """Mirror of Rust `mt_local::tokenizer::clean_source`: every whitespace run becomes one space, then trim."""
    return " ".join(text.split())


def load_tokenizers(tokenizer_dir: str, spm: str):
    from transformers import NllbTokenizer, NllbTokenizerFast

    slow = NllbTokenizer(vocab_file=spm, legacy_behaviour=False)
    fast = NllbTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"), legacy_behaviour=False)
    return slow, fast


def encode(tok, text: str, src: str) -> list[int]:
    tok.src_lang = src
    return list(tok(text).input_ids)


def find(model_dir: str, names: list[str]) -> str:
    for base in (model_dir, os.path.join(model_dir, "onnx")):
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"none of {names} in {model_dir}")


def greedy(enc, dec, src_ids: list[int], tgt_id: int, eos: int, max_new: int, margins: list[float] | None = None) -> list[int]:
    import numpy as np

    ids = np.array([src_ids], dtype=np.int64)
    mask = np.ones_like(ids)
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    past_names = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values.")]
    heads, head_dim = dec.get_inputs()[3].shape[1], dec.get_inputs()[3].shape[3]
    feeds = {n: np.zeros((1, heads, 0, head_dim), dtype=np.float32) for n in past_names}
    step_ids = np.array([[eos, tgt_id]], dtype=np.int64)
    out: list[int] = []
    use_cache = False
    enc_past: dict[str, "np.ndarray"] = {}
    out_names = [o.name for o in dec.get_outputs()]
    for _ in range(max_new):
        feeds.update(
            {
                "encoder_attention_mask": mask,
                "input_ids": step_ids,
                "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([use_cache]),
            }
        )
        results = dict(zip(out_names, dec.run(None, feeds)))
        logits = results["logits"][0, -1]
        nxt = int(np.argmax(logits))
        if margins is not None:
            top2 = np.partition(logits, -2)[-2:]
            margins.append(round(float(top2[1] - top2[0]), 4))
        if nxt == eos:
            break
        out.append(nxt)
        for name, value in results.items():
            if not name.startswith("present."):
                continue
            past = name.replace("present.", "past_key_values.")
            if ".encoder." in name:
                if not use_cache:
                    enc_past[past] = value
                feeds[past] = enc_past[past]
            else:
                feeds[past] = value
        use_cache = True
        step_ids = np.array([[nxt]], dtype=np.int64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True, help="folder with tokenizer.json")
    ap.add_argument("--spm", required=True, help="sentencepiece.bpe.model of the same revision")
    ap.add_argument("--model-dir", help="folder with the ONNX files (adds greedy reference outputs)")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "nllb_parity.json"),
    )
    args = ap.parse_args()

    slow, fast = load_tokenizers(args.tokenizer_dir, args.spm)
    enc = dec = None
    if args.model_dir:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        enc = ort.InferenceSession(find(args.model_dir, ["encoder_model_quantized.onnx"]), so, providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(
            find(args.model_dir, ["decoder_model_merged_quantized.onnx"]), so, providers=["CPUExecutionProvider"]
        )

    cases = []
    mismatches = 0
    lang_offsets = set()
    for text, src, tgt in CASES:
        slow_ids = encode(slow, text, src)
        # Sigil cleans whitespace before tokenizing (clean_source in tokenizer.rs), as sentencepiece's
        # remove_extra_whitespaces does; tokenizer.json alone keeps a trailing "▁".
        fast_ids = encode(fast, clean_source(text), src)
        # The pieces come from sentencepiece; the language token from tokenizer.json. The slow tokenizer built from
        # the .model file alone numbers the FLORES codes one higher than the checkpoint (transformers 4.38-4.39
        # without added_tokens_decoder), while the checkpoint and the ONNX export use tokenizer.json's ids
        # (eng_Latn = 256047), so only the language id may differ.
        if slow_ids[0] != fast_ids[0]:
            lang_offsets.add(slow_ids[0] - fast_ids[0])
        if slow_ids[1:] != fast_ids[1:]:
            mismatches += 1
            print(f"slow/fast mismatch for {text!r}:\n  slow {slow_ids}\n  fast {fast_ids}", file=sys.stderr)
        ids = [fast_ids[0]] + slow_ids[1:]
        case = {"text": text, "source": src, "target": tgt, "ids": ids}
        if enc is not None:
            t0 = time.perf_counter()
            tgt_id = fast.convert_tokens_to_ids(tgt)
            margins: list[float] = []
            out = greedy(enc, dec, ids, tgt_id, fast.eos_token_id, max_new=3 * len(ids) + 10, margins=margins)
            case["output_ids"] = out
            # Logit gap between the best and second-best token at each step (the last one chose </s>): a
            # different ONNX Runtime build may break near-ties differently, so the Rust test accepts a divergence
            # only where this gap is tiny.
            case["output_margins"] = margins
            case["output_text"] = fast.decode(out, skip_special_tokens=True)
            print(f"{src}->{tgt} {1000 * (time.perf_counter() - t0):.0f} ms: {case['output_text']}", file=sys.stderr)
        cases.append(case)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = {
        "$comment": "Generated by scripts/mt-bench/nllb_reference.py (Xenova/nllb-200-distilled-600M@261c31d). "
        "ids = Python sentencepiece (slow NllbTokenizer); output_ids = greedy decoding on onnxruntime "
        "(encoder_model_quantized + decoder_model_merged_quantized). Checked by mt_local::tokenizer and "
        "mt_local::nllb tests.",
        "model": "mt/nllb-200-distilled-600m-int8",
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if lang_offsets:
        print(f"note: slow tokenizer language ids differ by {sorted(lang_offsets)} (known; tokenizer.json wins)", file=sys.stderr)
    print(f"wrote {len(cases)} cases to {args.out} ({mismatches} slow/fast piece mismatches)", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
