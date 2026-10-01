"""Parity vectors for the Rust Sigil-ZS port (P5-B): runs scripts/vc-families/seed_vc/sampler.py on
the CPU EP (ORT 1.28) with fixed draws and writes raw little-endian f32 files plus meta.json.

    python scripts/vc-families/seed_vc/make_vectors.py <out dir>

Then SIGIL_TEST_MODELS_DIR=<dev cache> SIGIL_ZS_VECTORS=<out dir> cargo test
real_model_matches_the_reference_sampler -- --ignored (graphs from export.py in the dev cache)."""
import json, os, sys
import numpy as np
import onnxruntime as ort
import torch, torchaudio, librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sampler.py sits next to this file
import sampler as sp  # noqa: E402

CACHE = os.path.join(os.environ["LOCALAPPDATA"], "sigil-dev-cache")
G = os.path.join(CACHE, "vc-families", "seed_vc")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
names = ["xlsr", "lr", "dit_step", "dit_step1", "hift", "hift_f0", "hift_nsf", "campplus", "mel"]
sess = {n: ort.InferenceSession(os.path.join(G, n + ".onnx"), so, providers=["CPUExecutionProvider"]) for n in names}
zs = sp.SigilZS(sess)

def save(name, a):
    np.ascontiguousarray(a, dtype=np.float32).tofile(os.path.join(OUT, name + ".f32"))

samples = os.path.join(CACHE, "models", "transcribe", "samples")
meta = {}
for preset_name in ("sigil_low", "sigil_lean"):
    preset = sp.PRESETS[preset_name]
    ref22 = librosa.load(os.path.join(samples, "ja-short.wav"), sr=22050)[0].astype(np.float32)[: int(22050 * preset.max_prompt)]
    ref16 = torchaudio.functional.resample(torch.from_numpy(ref22), 22050, 16000).numpy()
    fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(ref16)[None], num_mel_bins=80, dither=0, sample_frequency=16000)
    fb = (fb - fb.mean(dim=0, keepdim=True)).numpy()
    ref = zs.prepare_reference(ref22, ref16, fb)
    src16 = librosa.load(os.path.join(samples, "jfk.wav"), sr=16000)[0].astype(np.float32)
    sh = preset.shapes(len(ref22) / 22050)
    s0 = int(1.0 * 16000)
    window = np.ascontiguousarray(src16[s0: s0 + sh["n16"]])
    rng = np.random.default_rng(7)
    p = ref["prompt_mel"].shape[-1]
    big_l = p + sh["n_mel"]
    z = rng.standard_normal((1, 80, big_l)).astype(np.float32)
    hp = rng.uniform(-np.pi, np.pi, (1, 9, 1)).astype(np.float32); hp[:, 0] = 0.0
    sn = rng.standard_normal((1, 9, sh["n_mel"] * 256)).astype(np.float32)
    out_p, st_p = zs.tick(ref, window, preset, z, hp, sn, hift_mode="partial")
    out_f, st_f = zs.tick(ref, window, preset, z, hp, sn, hift_mode="full")
    pre = preset_name + "."
    for k, v in {"ref22": ref22, "ref16": ref16, "fbank": fb, "style": ref["style"], "prompt_mel": ref["prompt_mel"],
                 "prompt_cond": ref["prompt_cond"], "window": window, "z": z, "harm_phase": hp, "sine_noise": sn,
                 "out_partial": out_p, "out_full": out_f, "mel_out": st_p["mel_out"], "f0": st_p["f0"],
                 "harm_phase_dec": st_p["harm_phase"]}.items():
        save(pre + k, v)
    meta[preset_name] = {"shapes": {k: int(v) for k, v in sh.items()}, "prompt_frames": int(p), "L": int(big_l),
                         "fbank_frames": int(fb.shape[0]), "steps": preset.steps, "cfg_rate": preset.cfg_rate,
                         "frames_per_block": preset.frames_per_block(), "dec_start_partial": int(st_p["dec_start"]),
                         "out_len": int(out_p.size), "units": preset.units()}
    print(preset_name, meta[preset_name])
# linspace and nsf helpers
meta["linspace"] = {str(n): [float(v) for v in sp.linspace_f32(n)] for n in (1, 4, 6, 10, 30)}
f0 = np.array([0, 0, 110.0, 180.5, 220.25, 0, 300.0, 1000.0], np.float32)
hp = np.array([0, 0.5, -1.0, 2.0, -3.0, 1.5, 0.25, -0.75, 3.1], np.float32).reshape(1, 9, 1)
meta["nsf"] = {"f0": f0.tolist(), "harm_phase": hp.reshape(-1).tolist(),
               "start_cycles_3": sp.nsf_start_cycles(f0, 3).tolist(),
               "cycles_at_700": sp.nsf_cycles_at(f0, hp, 700).tolist(),
               "anchor_700": sp.harm_phase_for_anchor(f0, 700, np.linspace(0, 0.9, 9)).reshape(-1).tolist()}
x = (np.sin(np.arange(22050) * 0.013) * 0.3 + np.cos(np.arange(22050) * 0.0007) * 0.2).astype(np.float32)
save("resample_in", x)
save("resample_out", torchaudio.functional.resample(torch.from_numpy(x), 22050, 16000).numpy())
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
