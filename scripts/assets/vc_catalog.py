#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the voice-changer base-asset catalog fragments from one declaration table.

Fragments (``src-tauri/model-catalog/``, owned by P3-A): ``vc-bases.json`` (pitch models and the MIT
default voice), ``vc-embedders.json``, ``vc-denoise.json``, ``bench-models.json``.

- **Upstream files** (published entries) are pinned here by the facts recorded when they were fetched
  (size, sha256, immutable revision). The store downloads them from those URLs.
- **Sigil-built files** (unpublished entries: the exported embedders, the FCPE adapter, the GPU load
  graph) take their size and sha256 from the staged exports (``--stage``, default
  ``%LOCALAPPDATA%/sigil-dev-cache/vc-assets``: the SIGIL_LOCAL_ASSETS layout the export scripts
  write). They start ``unpublished`` (local import or SIGIL_LOCAL_ASSETS only): the release URL names
  the file in the ``assets-v1`` release. After the files are uploaded
  (``scripts/release/publish-assets.ps1``), ``node scripts/models/publish-entries.mjs --write`` proves
  the uploads and flips the status in the fragment; this script keeps whatever status the committed
  fragment has, so regenerating never takes a published entry back.

Usage (repo root):
    python scripts/assets/vc_catalog.py            # rewrite the four fragments
    python scripts/assets/vc_catalog.py --check    # exit 1 when a fragment differs from what it would write
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CATALOG = os.path.join(REPO, "src-tauri", "model-catalog")
DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
STAGE = os.path.join(DEV_CACHE, "vc-assets")
RELEASE = "https://github.com/Mewnah/Sigil/releases/download/assets-v1/"

HF_RVC = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"  # lj1995/VoiceConversionWebUI
HF_APPLIO = "70ed563897504c756ec94067c12c902c4fd42025"  # IAHispano/Applio
HF_DEFAULT_VOICE = "22e4cd2b759f5da8cbfc2c0b05efc200fbfc4256"  # PhoenixStormJr/RVC-V2-default-voice
HF_DPDFNET = "dd6818d00f50c836fed43a6243ebe49116de5964"  # Ceva-IP/DPDFNet
GH_SWIFTF0 = "d1ba77fe5331310b778ebe126b6cb3656d41d24f"  # lars76/swift-f0
GH_GTCRN = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"  # Xiaobin-Rong/gtcrn
GH_DFN = "d375b2d8309e0935d165700c91da9de862a99c31"  # Rikorose/DeepFilterNet
GH_DNS = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"  # microsoft/DNS-Challenge


def hf(repo, rev, path):
    return {"kind": "huggingface", "url": f"https://huggingface.co/{repo}/resolve/{rev}/{path}", "revision": rev}


def raw(repo, rev, path):
    return {"kind": "direct", "url": f"https://raw.githubusercontent.com/{repo}/{rev}/{path}", "revision": rev}


def release(repo, tag, name):
    return {"kind": "github-release", "url": f"https://github.com/{repo}/releases/download/{tag}/{name}", "revision": tag}


def upstream_file(path, size, sha256, *sources, archive=None):
    f = {"path": path, "size": size, "sha256": sha256, "sources": list(sources)}
    if archive:
        f["archive"] = archive
    return f


def built_status(entry_id):
    """Status of a Sigil-built entry: what the committed fragments say (``unpublished`` until
    ``scripts/models/publish-entries.mjs --write`` flips it after the files are on the assets-v1 release),
    so regenerating a fragment never takes a published entry back."""
    for name in sorted(os.listdir(CATALOG)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        with open(os.path.join(CATALOG, name), encoding="utf-8") as f:
            for e in json.load(f).get("entries", []):
                if e.get("id") == entry_id:
                    return e.get("status", "unpublished")
    return "unpublished"


def staged_file(entry_id, path, stage):
    """A Sigil-built file: size + sha256 from the staged export; the placeholder release name."""
    src = os.path.join(stage, *entry_id.split("/"), *path.split("/"))
    if not os.path.isfile(src):
        raise SystemExit(f"{src} is missing: run the export script that builds {entry_id} first")
    h = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    name = f"{entry_id.replace('/', '-')}-{path.replace('/', '-')}"
    return {"path": path, "size": os.path.getsize(src), "sha256": h.hexdigest(),
            "sources": [{"kind": "github-release", "url": RELEASE + name, "revision": "assets-v1"}]}


def entry(id_, kind, version, title, description, license_, status, files, family=None):
    return {"id": id_, "family": family or id_.split("/")[0], "kind": kind, "version": version, "title": title,
            "description": description, "license": license_, "status": status, "files": files}


# ---------------------------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------------------------

EMBEDDERS = [
    # (name, title, description, license, has int8)
    ("contentvec", "ContentVec (768)", "Content encoder for RVC v2 voices (HuBERT-base ContentVec, layer 12). The default for most voices.", "MIT", True),
    ("contentvec-256", "ContentVec (256, RVC v1)", "Content encoder for RVC v1 voices (ContentVec layer 9 through its 256-channel projection).", "MIT", True),
    ("spin-v2", "SPIN v2", "Content encoder for voices trained with Applio's SPIN v2 embedder.", "MIT", False),
    ("spin", "SPIN", "Content encoder for voices trained with Applio's SPIN embedder.", "MIT", False),
    ("hubert-ja", "Japanese HuBERT", "Content encoder for voices trained with the Japanese HuBERT base embedder (rinna).", "Apache-2.0", False),
    ("hubert-ko", "Korean HuBERT", "Content encoder for voices trained with the Korean HuBERT base embedder (team-lucid).", "Apache-2.0", False),
    ("hubert-zh", "Chinese HuBERT", "Content encoder for voices trained with the Chinese HuBERT base embedder (TencentGameMate).", "MIT", False),
]


def vc_bases(stage):
    return {
        "$schema": "./_schema.json",
        "$comment": [
            "Voice-changer base assets (owned by P3-A; ids and the in-entry layout model.onnx are frozen in src-tauri/src/services/neural_audio/vc/assets.rs). Written by scripts/assets/vc_catalog.py; checks and contracts: scripts/vc-templates/export_pitch.py (reports/pitch.json).",
            "vc/pitch-rmvpe: RVC's rmvpe.onnx (log-mel [1,128,T] -> salience [1,T,360]; rmvpe.rs computes the mel).",
            "vc/pitch-fcpe: Sigil's adapter of the community FCPE export gzivdo/fcpe-onnx@5800a2b1 (torchfcpe 0.0.4 weights): audio [1,N] f32 16 kHz -> f0 [1, N//160+1] Hz, 0 = unvoiced (NaN from quiet frames mapped to 0). The contract fcpe.rs implements. Sigil-built; starts unpublished (becomes published with scripts/models/publish-entries.mjs once its file is on the assets-v1 release).",
            "vc/pitch-swiftf0: SwiftF0 model.onnx (audio [1,N], fmin, fmax scalars -> pitch f64 [1, N//256], confidence [1, N//256]; hop 256 = 16 ms, voiced when confidence >= 0.9). Also Applio's predictors/swift.onnx (identical bytes).",
            "vc/pitch-crepe-tiny: onnxcrepe v1.1.0 tiny.onnx (frames [n,1024] zero-mean / unit-std windows, hop 160 -> probabilities [n,360]; cents = 1997.3794 + 20 * bin).",
            "vc/pitch-pesto: import-only until a maintainer exports PESTO's streaming model with its own realtime/export_onnx.py (pesto@62bc0c97, mir-1k_g7.ckpt) as model.onnx; unpinned.",
            "vc/voice-default: the MIT default voice PhoenixStormJr/RVC-V2-default-voice (v2, 40 kHz, f0, HiFi-GAN) with its feature index; P3-B imports default.pth into a slot on first run and adopts the .index.",
        ],
        "entries": [
            entry("vc/pitch-rmvpe", "model", "1", "RMVPE pitch", "Accurate pitch tracking for the voice changer's GPU presets (RVC's RMVPE). About 360 MB.",
                  "MIT", "published", [upstream_file("model.onnx", 361688443, "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd",
                                                     hf("lj1995/VoiceConversionWebUI", HF_RVC, "rmvpe.onnx"))]),
            entry("vc/pitch-fcpe", "model", "0.0.4-sigil1", "FCPE pitch", "Fast pitch tracking for the voice changer (FCPE). About 44 MB.",
                  "MIT", built_status("vc/pitch-fcpe"), [staged_file("vc/pitch-fcpe", "model.onnx", stage)]),
            entry("vc/pitch-swiftf0", "model", "1", "SwiftF0 pitch", "Very light pitch tracking for CPU voice changing (SwiftF0). About 135 KB.",
                  "MIT", "published", [upstream_file("model.onnx", 135090, "6385e8c2ebc3872e82c9ff5946228de44cd3be77a750ea53698b7dbfe94b0a22",
                                                     raw("lars76/swift-f0", GH_SWIFTF0, "swift_f0/model.onnx"),
                                                     hf("IAHispano/Applio", HF_APPLIO, "Resources/predictors/swift.onnx"))]),
            entry("vc/pitch-crepe-tiny", "model", "1.1.0", "CREPE tiny pitch", "Neural pitch tracking (CREPE, tiny model). About 2 MB.",
                  "MIT", "published", [upstream_file("model.onnx", 1955762, "91fc2a0fd10f965dbf7775995daf50e99273caedd7efd00001f23be649da1bc3",
                                                     release("yqzhishen/onnxcrepe", "v1.1.0", "tiny.onnx"))]),
            entry("vc/pitch-pesto", "model", "2-mir1k-g7", "PESTO pitch", "Streaming pitch tracking (PESTO). Import a model.onnx exported with PESTO's realtime export script.",
                  "LGPL-3.0", "unpublished", [{"path": "model.onnx", "size": 0, "sources": [{"kind": "github-release", "url": RELEASE + "vc-pitch-pesto-model.onnx", "revision": "assets-v1"}]}]),
            entry("vc/voice-default", "model", "1", "Default voice", "A ready-to-use RVC voice (v2, 40 kHz) so the voice changer works before you add your own. About 118 MB.",
                  "MIT", "published", [
                      upstream_file("default.pth", 55223136, "c9d6b0ac7aa8df91917757894561cd690e5c4b66e97a9f92630c0a7257fbfccc",
                                    hf("PhoenixStormJr/RVC-V2-default-voice", HF_DEFAULT_VOICE, "default.pth")),
                      upstream_file("added_IVF511_Flat_nprobe_1_default_v2.index", 63016939, "93e0fbf723992b5ff6a1af3cae4e15f3bbb1dd860880fbd76565be23549aa1b7",
                                    hf("PhoenixStormJr/RVC-V2-default-voice", HF_DEFAULT_VOICE, "added_IVF511_Flat_nprobe_1_default_v2.index")),
                  ]),
        ],
    }


def vc_embedders(stage):
    entries = []
    for name, title, desc, lic, int8 in EMBEDDERS:
        eid = f"vc/embedder-{name}"
        files = [staged_file(eid, "model.fp32.onnx", stage), staged_file(eid, "model.fp16.onnx", stage)]
        if int8:
            files.append(staged_file(eid, "model.int8.onnx", stage))
        files.append(staged_file(eid, "embedder.json", stage))
        entries.append(entry(eid, "model", "1", title, desc, lic, built_status(eid), files))
    return {
        "$schema": "./_schema.json",
        "$comment": [
            "Voice-changer content encoders (owned by P3-A; ids and model.<fp32|fp16|int8>.onnx are frozen in vc/assets.rs). Sigil exports of the Hugging Face-format checkpoints Applio distributes (IAHispano/Applio@" + HF_APPLIO + " Resources/embedders/<name>), built by scripts/vc-templates/export_embedders.py; they start unpublished (local import or SIGIL_LOCAL_ASSETS) and become published with scripts/models/publish-entries.mjs once their files are on the assets-v1 release. Written by scripts/assets/vc_catalog.py.",
            "Contract: audio [1,N] f32 16 kHz (raw, not normalised) -> feats [1,T,C] f32, T = (N - 400) // 320 + 1 (400-sample receptive field, 320 hop, checked per checkpoint); C = 768 (256 for contentvec-256). The contract rvc.rs feeds BYO encoders.",
            "embedder.json in each entry: the contract, parity per variant and device class (CPU: ONNX Runtime CPU; GPU: DirectML) and the recommended variant per class. fp16 = transformer in fp16 (the conv front end stays fp32); it misses the export gate's fp16 parity on DirectML for contentvec and hubert-ja, and embedder.json 'recommended' is the export gate's pick, not the engine's order: the engine runs fp16 first on GPUs because the downstream scorer is neutral (docs/adr/0011-vc-content-encoder-precision.md). int8 (CPU, VNNI) ships only for contentvec / contentvec-256: dynamic int8 of the other encoders misses its gate (activation outliers).",
            "vc/embedder-spin is an addition to the frozen ids (Applio's first SPIN embedder, embedder_model \"spin\"); the engine's EmbedderKind needs a variant for it (P3-C1).",
        ],
        "entries": entries,
    }


def vc_denoise():
    return {
        "$schema": "./_schema.json",
        "$comment": [
            "Voice-changer input denoisers (owned by P3-A; ids in vc/assets.rs). Upstream ONNX files, pinned; checks and contracts: scripts/vc-templates/export_denoise.py (reports/denoise.json). Written by scripts/assets/vc_catalog.py.",
            "vc/denoise-gtcrn: GTCRN streaming (16 kHz): mix [1,257,1,2] (512-point STFT frame of a sqrt-Hann window, hop 256) + conv_cache [2,1,16,16,33], tra_cache [2,3,1,1,16], inter_cache [2,1,33,16] -> enh [1,257,1,2] + the next caches.",
            "vc/denoise-dfn3: DeepFilterNet3 (48 kHz, 960-point STFT, hop 480): enc.onnx, erb_dec.onnx, df_dec.onnx and config.ini (every DSP constant), extracted from the upstream tar.gz (two leading folders stripped).",
            "vc/denoise-dpdfnet: DPDFNet-2 48 kHz HR: spec [1,1,481,2] (960-point STFT frame at 48 kHz, hop 480) + state_in [56436] -> spec_e [1,1,481,2] + state_out.",
        ],
        "entries": [
            entry("vc/denoise-gtcrn", "model", "1", "GTCRN denoiser", "Removes background noise from the voice changer's input with almost no CPU. About 0.5 MB.",
                  "MIT", "published", [upstream_file("model.onnx", 535190, "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
                                                     raw("Xiaobin-Rong/gtcrn", GH_GTCRN, "stream/onnx_models/gtcrn_simple.onnx"))]),
            entry("vc/denoise-dfn3", "model", "3", "DeepFilterNet3 denoiser", "High-quality noise removal for the voice changer's input (48 kHz). About 8 MB.",
                  "MIT OR Apache-2.0", "published", [upstream_file(
                      "DeepFilterNet3_onnx.tar.gz", 7983136, "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
                      raw("Rikorose/DeepFilterNet", GH_DFN, "models/DeepFilterNet3_onnx.tar.gz"),
                      archive={"format": "tar.gz", "strip": 2, "include": ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"]})]),
            entry("vc/denoise-dpdfnet", "model", "2-48k-hr", "DPDFNet denoiser", "High-quality streaming noise removal for the voice changer's input (48 kHz). About 10 MB.",
                  "Apache-2.0", "published", [upstream_file("model.onnx", 10493337, "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
                                                            hf("Ceva-IP/DPDFNet", HF_DPDFNET, "onnx/dpdfnet2_48khz_hr.onnx"))]),
        ],
    }


def bench_models(stage):
    return {
        "$schema": "./_schema.json",
        "$comment": [
            "Benchmark-only models of kind bench (owned by P3-A; ids in vc/assets.rs); never needed by normal users. Checks and contracts: scripts/vc-templates/export_bench.py (reports/bench.json). Written by scripts/assets/vc_catalog.py.",
            "bench/dnsmos-p835: DNSMOS P.835 (model.onnx: input_1 [N,144160] = 9.01 s at 16 kHz -> [N,3] raw SIG/BAK/OVRL, mapped by DNS-Challenge's non-personalised polynomials) and P.808 (p808.onnx: input_1 [N,900,120] log-mel).",
            "bench/campplus: CAM++ speaker embedding (3D-Speaker zh-cn common, sherpa-onnx export): x [N,T,80] Kaldi fbank minus its time mean -> embedding [N,192].",
            "bench/gpu-load: Sigil's GPU hammer for the contention mode (8 x Conv 3x3 32->32 + ReLU, x [1,32,H,W] -> y), built by export_bench.py; starts unpublished.",
            "bench/campplus-exact: the same CAM++ checkpoint re-exported by scripts/vc-templates/export_bench.py with exact segment pooling (>= 100 dB SNR against upstream; P3B-B, ADR 0009); same I/O contract as bench/campplus; starts unpublished (becomes published with scripts/models/publish-entries.mjs once its file is on the assets-v1 release), staged at %LOCALAPPDATA%/sigil-dev-cache/vc-assets/bench/campplus-exact/model.onnx.",
        ],
        "entries": [
            entry("bench/dnsmos-p835", "bench", "1", "DNSMOS P.835", "Speech-quality predictor for the voice-changer benchmark.",
                  "MIT", "published", [
                      upstream_file("model.onnx", 1157965, "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
                                    raw("microsoft/DNS-Challenge", GH_DNS, "DNSMOS/DNSMOS/sig_bak_ovr.onnx")),
                      upstream_file("p808.onnx", 224860, "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
                                    raw("microsoft/DNS-Challenge", GH_DNS, "DNSMOS/DNSMOS/model_v8.onnx")),
                  ]),
            entry("bench/campplus", "bench", "1", "CAM++ speaker embedding", "Speaker-similarity model for the voice-changer benchmark.",
                  "Apache-2.0", "published", [upstream_file("model.onnx", 28281138, "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11",
                                                            release("k2-fsa/sherpa-onnx", "speaker-recongition-models", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"))]),
            entry("bench/campplus-exact", "bench", "1", "CAM++ speaker embedding (exact pooling)", "Speaker-similarity model of the voice-changer benchmark's scorer v2.",
                  "Apache-2.0", built_status("bench/campplus-exact"), [staged_file("bench/campplus-exact", "model.onnx", stage)]),
            entry("bench/gpu-load", "bench", "1", "GPU load generator", "A fixed convolution workload the benchmark runs next to the voice changer to simulate a game.",
                  "AGPL-3.0-or-later", built_status("bench/gpu-load"), [staged_file("bench/gpu-load", "model.onnx", stage)]),
        ],
    }


def render(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fragments = {
        "vc-bases.json": vc_bases(args.stage),
        "vc-embedders.json": vc_embedders(args.stage),
        "vc-denoise.json": vc_denoise(),
        "bench-models.json": bench_models(args.stage),
    }
    stale = []
    for name, obj in fragments.items():
        path = os.path.join(CATALOG, name)
        text = render(obj)
        current = open(path, encoding="utf-8").read() if os.path.isfile(path) else ""
        if current != text:
            stale.append(name)
            if not args.check:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
    if args.check:
        print("fragments up to date" if not stale else f"stale: {', '.join(stale)}")
        sys.exit(1 if stale else 0)
    print(f"wrote {', '.join(stale) or 'nothing (unchanged)'}")


if __name__ == "__main__":
    main()
