#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Voice-changer input denoisers (``vc/denoise-*``): stage and check the upstream ONNX files.

| id | files | source | rate / frame |
| --- | --- | --- | --- |
| vc/denoise-gtcrn | model.onnx (GTCRN streaming, simplified graph; MIT, 0.5 MB) | Xiaobin-Rong/gtcrn@502ebfab stream/onnx_models/gtcrn_simple.onnx | 16 kHz, 512-point STFT, hop 256 |
| vc/denoise-dfn3 | enc.onnx, erb_dec.onnx, df_dec.onnx, config.ini (DeepFilterNet3; MIT/Apache-2.0, 8 MB tar.gz) | Rikorose/DeepFilterNet@d375b2d8 models/DeepFilterNet3_onnx.tar.gz | 48 kHz, 960-point STFT, hop 480 |
| vc/denoise-dpdfnet | model.onnx (DPDFNet-2 48 kHz HR; Apache-2.0, 10 MB) | Ceva-IP/DPDFNet@dd6818d0 onnx/dpdfnet2_48khz_hr.onnx | 48 kHz, 960-point STFT, hop 480 |

Contracts (what P3-E's ``neural_audio/denoise`` feeds; all frame by frame, streaming):
- **GTCRN**: ``mix`` f32 [1, 257, 1, 2] (one STFT frame, real / imaginary: 512-point FFT of a
  sqrt-Hann-windowed 16 kHz frame, hop 256), caches ``conv_cache`` [2, 1, 16, 16, 33], ``tra_cache``
  [2, 3, 1, 1, 16], ``inter_cache`` [2, 1, 33, 16] (zeros at the start) -> ``enh`` [1, 257, 1, 2] and
  the next caches (``*_out``). Inverse STFT with the same window, overlap-add. Latency: one frame
  (512 samples = 32 ms of analysis window, 16 ms hop).
- **DeepFilterNet3**: the three-graph DeepFilterNet ONNX export (``enc``: ``feat_erb`` [1, 1, S, 32],
  ``feat_spec`` [1, 2, S, 96] -> e0..e3, emb, c0, lsnr; ``erb_dec`` -> ERB gains ``m``; ``df_dec``
  -> deep-filter coefficients [1, S, 96, 10] (order 5)), with DeepFilterNet's DSP around it (vorbis
  window, ERB features with exponential mean normalisation, unit-normalised complex features,
  ``df_lookahead`` 2, ``conv_lookahead`` 2; ``config.ini`` holds every constant).
- **DPDFNet-2 48 kHz HR**: ``spec`` f32 [1, 1, 481, 2] (one 960-point STFT frame at 48 kHz, hop 480)
  and the flat recurrent ``state_in`` f32 [56436] (zeros at the start) -> ``spec_e`` [1, 1, 481, 2]
  and ``state_out``. ~20 ms first-frame latency, 10 ms per hop (Ceva's README).

Checks: GTCRN enhances noisy speech (jfk.wav + white noise at 0 dB SNR) frame by frame with its
caches, and must raise SI-SDR by >= 5 dB; DPDFNet runs the same stream at 48 kHz and must attenuate a
noise-only input by >= 6 dB (its exact analysis window lives in Ceva's Python package, so speech
quality is P3-E's measurement); DeepFilterNet3's three graphs run on features of the right shapes and
return finite outputs (its DSP is P3-E's). Report: reports/denoise.json.

Usage (repo root): ``python scripts/vc-templates/export_denoise.py``.
"""

import argparse
import os
import sys
import tarfile
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigil_assets as sa  # noqa: E402

PINS = {
    "gtcrn": ("vc/denoise-gtcrn", "gtcrn/gtcrn_simple.onnx", "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"),
    "dfn3": ("vc/denoise-dfn3", "dfn3/DeepFilterNet3_onnx.tar.gz", "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"),
    "dpdfnet": ("vc/denoise-dpdfnet", "dpdfnet/dpdfnet2_48khz_hr.onnx", "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"),
}


def noisy_speech(snr_db=0.0, sr=16000):
    clean = sa.speech("jfk.wav", 4.0)
    if clean is None:
        clean = sa.harmonic_tone(np.linspace(120, 220, 64000), 4.0)
    noise = np.random.default_rng(3).standard_normal(clean.size).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2) / 10 ** (snr_db / 10))
    return clean, (clean + noise).astype(np.float32), noise


def check_gtcrn(path, ep):
    sess = sa.session(path, ep)
    clean, noisy, _ = noisy_speech(0.0)
    win = torch.hann_window(512).pow(0.5)
    spec = torch.view_as_real(torch.stft(torch.from_numpy(noisy), 512, 256, 512, win, return_complex=True))  # [257, T, 2]
    caches = {"conv_cache": np.zeros((2, 1, 16, 16, 33), np.float32), "tra_cache": np.zeros((2, 3, 1, 1, 16), np.float32),
              "inter_cache": np.zeros((2, 1, 33, 16), np.float32)}
    frames = []
    for t in range(spec.shape[1]):
        enh, c1, c2, c3 = sess.run(None, {"mix": spec[None, :, t : t + 1, :].numpy(), **caches})
        caches = {"conv_cache": c1, "tra_cache": c2, "inter_cache": c3}
        frames.append(enh[0, :, 0, :])
    out = torch.from_numpy(np.stack(frames, axis=1))  # [257, T, 2]
    y = torch.istft(torch.view_as_complex(out.contiguous()), 512, 256, 512, win, length=noisy.size).numpy()
    before, after = sa.si_sdr(noisy, clean), sa.si_sdr(y, clean)
    ok = bool(np.isfinite(y).all() and after - before >= 5.0)
    return {"si_sdr_in_db": float(before), "si_sdr_out_db": float(after), "improvement_db": float(after - before), "frames": int(spec.shape[1]), "ok": ok}


def vorbis(n):
    k = np.arange(n) + 0.5
    return np.sin(0.5 * np.pi * np.sin(np.pi * k / n) ** 2).astype(np.float32)


def check_dpdfnet(path, ep):
    sess = sa.session(path, ep)
    n_state = [i for i in sess.get_inputs() if i.name == "state_in"][0].shape[0]
    rng = np.random.default_rng(5)
    noise = (0.05 * rng.standard_normal(48000 * 2)).astype(np.float32)
    win = vorbis(960)
    wnorm = 1.0 / (960**2 / (2 * 480))
    state = np.zeros(n_state, np.float32)
    energy_in = energy_out = 0.0
    for t in range(0, noise.size - 960, 480):
        fr = np.fft.rfft(noise[t : t + 960] * win) * wnorm
        spec = np.stack([fr.real, fr.imag], -1).astype(np.float32)[None, None]
        out, state = sess.run(None, {"spec": spec, "state_in": state})
        if t >= 48000 // 2:  # after the recurrent state settled
            energy_in += float(np.sum(spec**2))
            energy_out += float(np.sum(out**2))
    atten = 10 * np.log10(energy_in / max(energy_out, 1e-20))
    return {"state_size": int(n_state), "noise_attenuation_db": float(atten), "ok": bool(np.isfinite(atten) and atten >= 6.0)}


def check_dfn3(folder, ep):
    enc = sa.session(os.path.join(folder, "enc.onnx"), ep)
    erb = sa.session(os.path.join(folder, "erb_dec.onnx"), ep)
    dfd = sa.session(os.path.join(folder, "df_dec.onnx"), ep)
    s = 50
    rng = np.random.default_rng(9)
    feat_erb = (rng.standard_normal((1, 1, s, 32)) * 0.5 - 1).astype(np.float32)
    feat_spec = (rng.standard_normal((1, 2, s, 96)) * 0.3).astype(np.float32)
    e0, e1, e2, e3, emb, c0, lsnr = enc.run(None, {"feat_erb": feat_erb, "feat_spec": feat_spec})
    (m,) = erb.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
    coefs, alpha = dfd.run(None, {"emb": emb, "c0": c0})
    shapes = {"m": list(m.shape), "coefs": list(coefs.shape), "lsnr": list(lsnr.shape)}
    ok = all(np.isfinite(x).all() for x in (m, coefs, lsnr)) and list(m.shape) == [1, 1, s, 32] and list(coefs.shape)[1:] == [s, 96, 10]
    return {"shapes": shapes, "mask_range": [float(m.min()), float(m.max())], "ok": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=("cpu", "dml"))
    ap.add_argument("--stage", default=sa.STAGE)
    args = ap.parse_args()
    out = {"ep": args.ep}
    g = sa.upstream(PINS["gtcrn"][1], PINS["gtcrn"][2])
    out["gtcrn"] = check_gtcrn(g, args.ep)
    out["gtcrn"]["contract"] = sa.io_contract(g)
    out["gtcrn"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["gtcrn"][0], "model.onnx", g, args.stage)))
    d = sa.upstream(PINS["dpdfnet"][1], PINS["dpdfnet"][2])
    out["dpdfnet"] = check_dpdfnet(d, args.ep)
    out["dpdfnet"]["contract"] = sa.io_contract(d)
    out["dpdfnet"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["dpdfnet"][0], "model.onnx", d, args.stage)))
    t = sa.upstream(PINS["dfn3"][1], PINS["dfn3"][2])
    dest = os.path.join(args.stage, *PINS["dfn3"][0].split("/"))
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(t, "r:gz") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        if sorted(os.path.basename(n) for n in names) != ["config.ini", "df_dec.onnx", "enc.onnx", "erb_dec.onnx"] or any(n.count("/") != 2 for n in names):
            raise SystemExit(f"unexpected DeepFilterNet3 archive layout: {names}")
        with tempfile.TemporaryDirectory() as tmp:
            tar.extractall(tmp, filter="data")
            for n in names:  # the catalog strips the two leading components (tmp/export/)
                sa.stage(PINS["dfn3"][0], os.path.basename(n), os.path.join(tmp, n), args.stage)
    out["dfn3"] = check_dfn3(dest, args.ep)
    out["dfn3"]["archive"] = {"bytes": os.path.getsize(t), "sha256": sa.sha256_file(t), "members": names, "strip": 2}
    out["dfn3"]["contract"] = {n: sa.io_contract(os.path.join(dest, n)) for n in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx")}
    print(f"{'PASS' if out['gtcrn']['ok'] else 'FAIL'}  gtcrn: SI-SDR {out['gtcrn']['si_sdr_in_db']:.1f} -> {out['gtcrn']['si_sdr_out_db']:.1f} dB")
    print(f"{'PASS' if out['dpdfnet']['ok'] else 'FAIL'}  dpdfnet: noise attenuated by {out['dpdfnet']['noise_attenuation_db']:.1f} dB (state {out['dpdfnet']['state_size']})")
    print(f"{'PASS' if out['dfn3']['ok'] else 'FAIL'}  dfn3: {out['dfn3']['shapes']}")
    sa.report("denoise" if args.ep == "cpu" else f"denoise_{args.ep}", out)
    raise SystemExit(0 if all(out[k]["ok"] for k in ("gtcrn", "dpdfnet", "dfn3")) else 1)


if __name__ == "__main__":
    main()
