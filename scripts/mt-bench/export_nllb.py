"""Build an NLLB-200 model folder / zip for Sigil (catalog entry mt/nllb-200-distilled-1.3b-int8, which stays
'unpublished': install the zip by local import in Models & Runtimes or SIGIL_LOCAL_ASSETS).

Layout (what src-tauri/src/services/mt_local/nllb.rs loads):
  encoder_model_quantized.onnx          input_ids, attention_mask -> last_hidden_state
  decoder_model_merged_quantized.onnx   Optimum's merged decoder (KV cache + use_cache_branch)
  tokenizer.json, config.json, generation_config.json

Steps: Optimum ONNX export (task text2text-generation-with-past, which merges the decoders), 8-bit dynamic
quantization of both graphs with onnxruntime (subgraphs included, so both branches of the merged decoder are
quantized), a quick greedy smoke translation, then the zip plus a .sha256 file with the size and hash to pin in
src-tauri/model-catalog/mt-nllb.json when publishing.

Maintainer tool, run by hand (it downloads the PyTorch checkpoint, 5.5 GB for 1.3B, into the Hugging Face cache):
  pip install "optimum[exporters]" onnx onnxruntime transformers sentencepiece
  python scripts/mt-bench/export_nllb.py --model facebook/nllb-200-distilled-1.3B --work <scratch dir> \
      --out <dir>/nllb-200-distilled-1.3b-int8.zip
The same script reproduces a 600M folder with --model facebook/nllb-200-distilled-600M (Sigil ships Xenova's
export of it instead; see the catalog comments).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

FILES = [
    "encoder_model_quantized.onnx",
    "decoder_model_merged_quantized.onnx",
    "tokenizer.json",
    "config.json",
    "generation_config.json",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def export(model: str, fp32_dir: str) -> None:
    if os.path.isfile(os.path.join(fp32_dir, "decoder_model_merged.onnx")):
        print(f"reusing the export in {fp32_dir}")
        return
    run(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            model,
            "--task",
            "text2text-generation-with-past",
            fp32_dir,
        ]
    )
    if not os.path.isfile(os.path.join(fp32_dir, "decoder_model_merged.onnx")):
        raise SystemExit("the export has no decoder_model_merged.onnx (optimum too old? it needs post-processing on)")


# The loop as it reads inside the dedented function (the function body is indented by 4).
_OLD_CONCAT = """    quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
    for i in range(1, len(quantized_per_channel_data_list)):
        channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
        quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)
"""
_NEW_CONCAT = """    quantized_weights = np.concatenate(
        [np.asarray(c).reshape(reshape_dims) for c in quantized_per_channel_data_list], channel_axis
    )
"""


def speed_up_per_channel_quantization() -> None:
    """onnxruntime 1.19 joins the per-channel results with `np.concatenate((so_far, one_channel))` in a loop, which
    copies the growing array once per channel: quadratic in the channel count. The 1.3B decoder's output projection
    (1024 x 256206) has 256206 channels, which is hours of copying (a 32768-channel weight already takes 100 s); the
    single concatenate below gives the identical array. Applied by rewriting the library function's source; when the
    installed onnxruntime differs, the function is left alone (correct, only slow)."""
    import inspect
    import textwrap

    from onnxruntime.quantization import base_quantizer

    cls = base_quantizer.BaseQuantizer
    fn = cls.quantize_weight_per_channel_impl
    if getattr(fn, "_sigil_fast_concat", False):
        return
    source = textwrap.dedent(inspect.getsource(fn))
    if _OLD_CONCAT not in source:
        print("note: onnxruntime's per-channel loop differs from 1.19; large weights will quantize slowly", flush=True)
        return
    namespace: dict = {}
    exec(compile(source.replace(_OLD_CONCAT, _NEW_CONCAT), inspect.getsourcefile(fn) or "<patched>", "exec"), vars(base_quantizer), namespace)
    patched = namespace["quantize_weight_per_channel_impl"]
    patched._sigil_fast_concat = True
    cls.quantize_weight_per_channel_impl = patched


def quantize(src: str, dst: str, per_channel: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    speed_up_per_channel_quantization()
    print(f"quantizing {os.path.basename(src)} -> {os.path.basename(dst)}", flush=True)
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        # The merged decoder keeps both branches in If subgraphs; without this only the outer graph is quantized.
        extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True},
        # The fp32 1.3B graphs are above protobuf's 2 GB limit and carry external data, so the save never
        # fails on size here; the int8 result is made self-contained again below.
        use_external_data_format=True,
    )
    inline_external_data(dst)


def inline_external_data(path: str) -> None:
    """Fold the external data file quantize_dynamic wrote next to `path` back into the .onnx, so the quantized
    graph is one self-contained file (the zip and Sigil's loader take exactly the five FILES). Fails when the
    int8 graph itself is above protobuf's 2 GB limit."""
    import onnx

    data = path + ".data"
    if not os.path.isfile(data):
        return
    total = os.path.getsize(path) + os.path.getsize(data)
    if total > 1_900_000_000:
        raise SystemExit(f"{os.path.basename(path)} is {total} bytes with its weights: too large for one self-contained .onnx")
    print(f"inlining {os.path.basename(data)} ({os.path.getsize(data)} bytes) into {os.path.basename(path)}", flush=True)
    model = onnx.load(path, load_external_data=True)
    onnx.save(model, path)
    del model
    os.remove(data)


def smoke(folder: str) -> None:
    """Greedy-translate two lines with the quantized graphs (the loop Sigil runs; see nllb_reference.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import onnxruntime as ort
    from nllb_reference import greedy  # type: ignore
    from transformers import NllbTokenizerFast

    tok = NllbTokenizerFast(tokenizer_file=os.path.join(folder, "tokenizer.json"), legacy_behaviour=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    enc = ort.InferenceSession(os.path.join(folder, FILES[0]), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(folder, FILES[1]), so, providers=["CPUExecutionProvider"])
    for text, src, tgt in [
        ("Hello everyone, thanks for coming to the stream!", "eng_Latn", "jpn_Jpan"),
        ("今日はいい天気ですね。", "jpn_Jpan", "eng_Latn"),
    ]:
        tok.src_lang = src
        ids = list(tok(text).input_ids)
        out = greedy(enc, dec, ids, tok.convert_tokens_to_ids(tgt), tok.eos_token_id, 3 * len(ids) + 10)
        print(f"  {text} -> {tok.decode(out, skip_special_tokens=True)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--work", required=True, help="scratch folder (fp32 export and the quantized folder)")
    ap.add_argument("--out", required=True, help="zip to write, e.g. nllb-200-distilled-1.3b-int8.zip")
    ap.add_argument("--no-per-channel", action="store_true", help="per-tensor weight scales (smaller, less accurate)")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    fp32 = os.path.join(args.work, "fp32")
    folder = os.path.join(args.work, "int8")
    os.makedirs(fp32, exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    export(args.model, fp32)
    quantize(os.path.join(fp32, "encoder_model.onnx"), os.path.join(folder, FILES[0]), not args.no_per_channel)
    quantize(os.path.join(fp32, "decoder_model_merged.onnx"), os.path.join(folder, FILES[1]), not args.no_per_channel)
    for name in FILES[2:]:
        shutil.copyfile(os.path.join(fp32, name), os.path.join(folder, name))
    for name in os.listdir(folder):
        if name not in FILES:
            raise SystemExit(f"unexpected file {name} in {folder} (external data? the quantized graphs must be self-contained)")
    if not args.skip_smoke:
        smoke(folder)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in FILES:
            z.write(os.path.join(folder, name), name)
    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    size = os.path.getsize(args.out)
    with open(args.out + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h.hexdigest()}  {os.path.basename(args.out)}  {size}\n")
    print(f"wrote {args.out}: {size} bytes, sha256 {h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
