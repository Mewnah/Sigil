# ADR 0006: Zero-shot and singing families run in Sigil's engine (Seed-VC, DDSP-SVC, MeanVC2)

- **Status:** accepted (P3-H spike; every report re-measured on the current exports after the P3-H
  review fixes). Per family:
  - **Seed-VC real-time (Sigil-ZS):** GO in-engine on CUDA and DirectML at fp32. The CPU cannot
    keep up live, so it is not a live mode.
  - **DDSP-SVC 6.3:** GO in-engine on CUDA, from Sigil's own export (not upstream's exporter).
    DirectML passes parity but is too slow in one graph until P5-A restructures it. The CPU cannot
    keep up live.
  - **MeanVC2:** GO as an experimental CPU family, if P5-C's A/B passes. That A/B needs the speaker
    encoder, which is a manual download.
- **P5 (2026-10-01):** all three shipped in the engine: Sigil-ZS on CUDA and DirectML (T10 met on
  H2 with `sigil_low`), DDSP-SVC on CUDA and DirectML (DirectML fixed by P5-A; T10 met with CREPE,
  SwiftF0 or fusion F0), MeanVC2 as an experimental CPU family (real time on one thread; the
  listening A/B waits on the manual speaker encoder). See [P5 outcomes](#p5-outcomes).
- **Date:** 2026-09-30 (evidence re-measured 2026-10-01)
- **Workstream:** P3-H (the scripts are `scripts/vc-families/`; the evidence is `scripts/vc-families/reports/*.json`)
- **Plan decisions:** "Zero-shot and singing inside Sigil's engine"; deferred items "Seed-VC realtime
  in-engine export" and "MeanVC2"
- **Research items:** User adjustment: bring zero-shot and singing into the engine (feasibility); §4.2
  (Seed-VC, DDSP-SVC 6.3, MeanVC2 rows)

## Context

The plan puts zero-shot and singing voices inside Sigil's own engine: one process, no Python, and
the same latency budget, GPU scheduling and TTS mixing as RVC. Sidecars were to be the fallback.
Nobody publishes ONNX graphs for any of the three candidates. DDSP-SVC ships an exporter, but it has
defects (below). The spike therefore had to answer three questions per family, with Sigil's own
numbers:

1. Does the upstream model export to ONNX with the same output?
2. What is the reference loop a Rust port reproduces?
3. What does a step and a tick cost on the dev rig's GPU and CPU?

The gate is SNR ≥ 35 dB against upstream PyTorch fp32, on every stage and on the output audio. A
row whose reference or candidate holds a non-finite value fails.

Constraints:
- Real-time sessions use 1–2 threads with spinning off, and fixed shapes where the EP can capture
  CUDA graphs.
- Every lever is switchable.
- Downloads happen one file at a time, into the dev cache only. Sigil-made exports stay `unpublished`.
- Weights are never committed.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Port each family into the engine: Sigil exports the graphs, and a Rust loop mirrors the reference sampler | Gets the whole harness: budget, fixed shapes, CUDA graphs, anchored noise, priority stream, and TTS through the voice | Export work per family; Sigil owns the parity gate | **Chosen** for all three; measured below |
| B. Sidecar protocol v2 (seed-vc-realtime, upstream Python) | No export work | Python and PyTorch on friends' PCs; upstream runs 430–690 ms live; no fixed shapes or graphs; a second CUDA context | Kept for X-VC and StreamVoiceAnon (plan), not needed for these three |
| C. DDSP-SVC through upstream's own `export_onnx.py` (`encoder.onnx` + `velocity.onnx`) | Already written upstream | Time input is int64 `1000 t`; `sin(πx)/(πx)` is NaN at f0 = 0; no `silence_front` slice or formant shift; the n_spk = 1 branch references an undefined variable | Rejected (rows below) |
| D. Drop the family | No work | Loses zero-shot or singing | Not needed |

## Decision

### Sigil-ZS: Seed-VC real-time (`seed-uvit-tat-xlsr-tiny`), GO on the GPU

**Graphs** (`scripts/vc-families/seed_vc/export.py`, opset 17, weights inside):

| graph | inputs | outputs | notes |
| --- | --- | --- | --- |
| `xlsr` | wav16k [1,N] | feats [1,T,1024] | XLS-R-300M layer 12, normalises inside |
| `lr` | feats [1,T,1024], n_frames i64 [1] | cond [1,n,384] | Length regulator. PyTorch's nearest index is computed explicitly, because ONNX `Resize` rounds differently at exact multiples |
| `dit_step` | x, prompt_x [1,80,L]; cond [1,L,384]; style [1,192]; t, dt, cfg_rate [1]; prompt_len i64 [1] | x_next [1,80,L] | One Euler step. CFG is a batch of 2 inside, and the prompt frames are re-zeroed. Fixed L, so the CUDA EP captures it |
| `dit_step1` | the same without cfg_rate | x_next | The `cfg_rate = 0` step (batch 1) |
| `dit` | the bare estimator | v | Fallback when the update runs on the host |
| `hift` | mel [1,80,M]; harm_phase [1,9,1]; sine_noise [1,9,256M] | audio [1,256M], f0 | CosyVoice HiFT over the whole tick (upstream). Randomness comes in as inputs. The NSF phase is wrapped per frame. STFT and iSTFT are strided and transposed convolutions |
| `hift_f0` | mel [1,80,M] | f0 [1,M] | HiFT's F0 predictor alone (fp32 only), over the whole tick |
| `hift_nsf` | mel [1,80,W]; f0 [1,W]; harm_phase [1,9,1]; sine_noise [1,9,256W] | audio [1,256W] | HiFT synthesis from a given F0: upstream's own `forward(x, f0)` seam. `harm_phase` is the phase at the slice's first sample |
| `campplus` | Kaldi fbank [1,F,80], mean-removed | style [1,192] | Once per voice. Segment pooling is explicit (see cross-cutting finding 1) |
| `mel` | wav22k [1,S] | mel [1,80,M] | Once per voice (the prompt mel) |

**Loop.** The loop is `seed_vc/sampler.py`, which is upstream `real-time-gui.py` `custom_infer`:

- **Once per voice:** `xlsr`, `campplus`, `mel` and `lr` over the reference, cut to `max_prompt`.
  The results are the prompt condition, the prompt mel and the style.
- **Per tick:**
  1. `xlsr` over the whole window, keeping only the last `extra_dit` seconds.
  2. `lr` to the tick's mel length.
  3. Concatenate the prompt condition and the tick's condition.
  4. Start from z ~ N(0,1) with the prompt frames zeroed.
  5. Run `steps` × `dit_step`, from t = 0 with dt taken from the float32 `linspace` (reproduced bit
     for bit).
  6. Vocoder, either way (both gated, table below):
     - **partial decode** (Sigil's lever): `hift_f0` over the tick's mel, then `hift_nsf` over only
       the returned frames plus 12 frames of decoder context, with `harm_phase` carried to the
       slice's first sample or anchored at the join (below). 12 frames reproduce the whole-tick
       decode (> 90 dB; the decoder's receptive field), so `hift_nsf` decodes 35 of 65 frames for
       `upstream_readme` and 28 of 41 for `sigil_low`. That saves 1.0–1.5 ms per tick on the 4090,
       where HiFT is launch-bound, and 32–78 ms on the CPU;
     - `hift` over the whole tick: upstream's path, the fallback.
  7. Take the block plus the SOLA margins.

The random inputs are z, the harmonic phases and the sine noise. The engine feeds its own
counter-based noise, and `hift_f0` gives it the tick's F0 before synthesis, so it can choose the
NSF phase.

**Coherent ticks.** Upstream draws a fresh z, fresh harmonic phases and fresh sine noise every tick,
so the two ticks in a SOLA crossfade are two different renditions of the same audio. `parity.py`
(`stream_join`) runs consecutive ticks and measures each join (table below):
- **Upstream:** the overlapping mel frames of consecutive ticks agree to only 13.9 dB, and their
  waveforms do not line up at any fixed offset (best correlation 0.37, at scattered offsets).
- **Time base.** A tick turns 37 XLS-R frames (0.74 s of content) into 65 mel frames (0.755 s of
  audio) at `upstream_readme`. One 180 ms block therefore moves the content by 15.81 mel frames,
  while the output advances 15.5 frames (3969 samples).
- **Anchored DiT noise** (z, and the sine noise, drawn per absolute mel frame on a 16-frame-per-tick
  grid): the overlapping mels agree to 16.6 dB, and the waveforms line up at a fixed −127 samples
  (3969 − 16 × 256), correlation 0.71.
- **Plus the NSF phase anchored at the join** (`Preset.join_anchor`): correlation 0.76 (0.64 at the
  grid offset, against 0.57 without the phase lever).
- **Upstream's SOLA** searches only 0 to +441 samples, so it never reaches the −127 alignment: within
  its range the best correlation is 0.23 for upstream and 0.42–0.43 with the anchored levers.

On its own, the phase lever cannot make two different renditions agree. On top of anchored noise it
adds 0.07 of correlation at the grid offset, and both pay off only when the join lands on the grid.
P5-B must therefore make the block a whole number of mel frames on the content time base (or widen
the SOLA search to include the grid offset), draw z and the sine noise per absolute mel frame, and
anchor the NSF phase at the join. The benchmark then decides the defaults.

**Precision.**
- Default: fp32 for every graph, except that `hift` / `hift_nsf` may run their decoder in fp16.
  - The fp16 decoder passes: the chain gets 47.2 dB (47.6 dB with partial decode). It saves
    1.0–2.4 ms per tick on the 4090.
  - The benchmark decides whether that becomes the default.
- fp16 for XLS-R or the DiT is a lever, off by default.
  - The fp16 graphs keep upstream autocast's fp32 islands exactly: the DiT's timestep sinusoid takes
    the exact fp32 `freqs` and `t`, and HiFT's Snake activations keep their exact fp32 `alpha`
    (each copied before `.half()` rounds it). An earlier export rounded them first, which put the
    sinusoid's argument up to 0.36 rad off.
  - It passes on its own stages: XLS-R at 50.1 dB, the DiT step at 77.9 dB, the tick mel at
    68.3 dB, which is now closer to fp32 than upstream's own fp16 mode (66.95 dB).
  - Its waveform fails, because HiFT predicts its own F0 from the mel (cross-cutting finding 3).
  - That is as far from fp32 as upstream's own fp16 real-time mode, `real-time-gui --fp16 true`:
    log-mel 24.5 dB against upstream's 24.2 dB; both waveforms are about −3 dB (upstream −3.66).
  - On the 4090, all-fp16 saves 12–20 % of the tick, almost all of it in XLS-R and HiFT. The DiT
    step costs about the same either way (2.1–2.3 ms), because it is launch-bound, not math-bound.

**EPs.**
- CUDA with a captured graph per preset shape for `dit_step`, `xlsr`, `hift` and `hift_nsf`. `lr`
  stays uncaptured; it takes under 1 ms.
- DirectML passes parity at 68.5 dB (68.9 dB with partial decode). With host tensors it runs
  `upstream_readme` at RTF 0.28 and `sigil_lean` at RTF 0.24 (partial decode). DirectML refuses
  the pinned dimensions of both HiFT graphs, which then run with dynamic shapes.
- The CPU cannot keep up live. On ORT 1.28's CPU EP at 2 threads a tick takes 3.2–8.5× its block
  (RTF 3.23 for `sigil_lean` to 8.09 for `upstream_readme`, partial decode). A live stream would
  fall further behind every block, without bound, so no delay setting makes it work. Converting a
  recorded clip offline would take 3–9 s per second of audio.

**Presets to start P5-B from.** P5-B's benchmark picks the defaults.

| preset | block | DiT context | steps | CFG | prompt |
| --- | --- | --- | --- | --- | --- |
| `upstream_readme` (the RTX 3060 Laptop recommendation) | 180 ms | 0.5 s | 10 | 0.7 | 3 s |
| `sigil_low` | 100 ms | 0.3 s | 6 | 0.7 | 3 s |
| `sigil_lean` | 100 ms | 0.3 s | 4 | 0 | 1.5 s |

**Against T10** (p95 ≤ 300 ms end to end on H1 idle). These are estimates; P5-B measures the real
figures on H1.

<!-- tables:t10 -->
*Sigil-ZS against T10* (p95 ≤ 300 ms end to end on H1 idle). Algorithmic latency, worst case: block + right context + SOLA crossfade + SOLA search. Tick on the 4090: per call the larger of the kept block's p90 and the slowest block's median, CUDA fp32 (graphs where captured), whole-tick `hift` / partial decode. H1 = the 4090 tick × 3 (the assumption) or × 5 (pessimistic); the bracket is what is left of 300 ms for the terms not counted (see below):

| preset | algorithmic (ms) | 4090 tick, conservative (ms) | H1 ×3, partial (left of 300) | H1 ×5, partial (left of 300) |
|---|---|---|---|---|
| `upstream_readme` | 180 + 20 + 40 + 20 = 260 | 46.1 / 44.4 | 393 (-93) | 482 (-182) |
| `sigil_low` | 100 + 20 + 40 + 20 = 180 | 29.7 / 29.1 | 267 (+33) | 325 (-25) |
| `sigil_lean` | 100 + 20 + 40 + 20 = 180 | 21.3 / 20.1 | 240 (+60) | 281 (+19) |
<!-- /tables:t10 -->

- The algorithmic latency is what a played sample lags its input before any compute: the block,
  the right context, the SOLA crossfade and up to one SOLA search window (the played block starts
  at the chosen offset).
- Not counted, and to come out of the bracket: resampling the device input to 16 kHz and the
  22.05 kHz output to the device rate (a few ms of filter delay each), the WASAPI capture and render
  buffers (about 10–20 ms each in shared mode), the harness's alignment of blocks to device periods,
  and the H1/4090 factor itself (an assumption until P5-B's H1 run).
- **`upstream_readme` does not fit:** its algorithmic latency alone is 260 ms.
- **`sigil_low` is borderline:** 33 ms are left at ×3 for the uncounted terms, which is about
  what the device buffers alone take, and at ×5 it is 25 ms over.
- **`sigil_lean` fits with a little more room** (60 ms left at ×3, 19 ms at ×5), at the cost of CFG
  and half the prompt.
- The coherent-ticks change (a block of whole mel frames) moves the block length slightly; P5-B
  re-derives these rows for the blocks it picks.
- Speaker similarity per preset is P5-B's measurement.

### DDSP-SVC 6.3 singing, GO on the GPU

**Graphs** (`scripts/vc-families/ddsp_svc/export.py`):

| graph | inputs | outputs | notes |
| --- | --- | --- | --- |
| `ddsp` | hubert [1,U,768], mel2ph i64 [1,T], f0 Hz [1,T], volume [1,T], randn [1,512T], start_frame i64 [1], formant_shift [1,1] | x [1,1,128,R], cond [1,128,R] | Upstream's inference path, including the `silence_front` slice (R = T − start_frame) and the formant shift (`aug_shift`). STFT and iSTFT are convolutions; the sinc has no NaN. The DDSP mel is upstream's `vocoder.extract`: it is built from the vocoder's own STFT config (n_fft, window, hop, bins, band), not the exporter's hard-coded one, and the export refuses a vocoder whose sampling rate or hop does not fit the model |
| `reflow_step` | x, t, dt, cond | x + dt · v(x, 1000 t, cond) | One Euler step with float t, as upstream PyTorch runs it. Captured as a CUDA graph |
| `velocity_f` | x, t, cond | v | The bare LYNXNet2 |
| `vocoder` | mel [1,128,R], f0 [1,R], phase0 [1] | audio [1,512R] | openvpi PC-NSF-HiFiGAN 44.1 kHz with a mini-NSF source. `phase0` (cycles) continues the excitation across windows; 0 means upstream. The F0 is an input, so the engine knows it before synthesis |

**Units.** The units come from Sigil's shared ContentVec embedder (`vc/embedder-contentvec`, P3-A).
It runs twice, on the 16 kHz window and on the window delayed by 160 samples, and the two outputs are
interleaved. This is upstream's `contentvec768l12tta2x`. It matches upstream's encoder at 111–116 dB.

**Loop.** The loop is `ddsp_svc/sampler.py`, which is upstream `gui_reflow.py` `SvcDDSP.infer`
(the model call `Unit2Wav.forward`, then its volume gate):
1. Run `ddsp` over the whole window (upstream's default is 2.5 s).
2. Start the flow at x = t_start · x_ddsp + (1 − t_start) · z, with the default t_start 0.7.
3. Run `steps` × `reflow_step` with a float32 t and dt.
4. Denormalise the mel.
5. Run `vocoder` over the last R frames.
6. Apply the volume gate: frames whose volume is at or below the threshold (the GUI slider, default
   −45 dBFS) are muted. The per-frame 0/1 mask is upsampled linearly over each 512-sample frame
   (`ddsp.core.upsample`) and applied to the output. Without it, silence still comes out as the
   DDSP noise excitation filtered by the flow. The sampler applies it by default
   (`threshold_db=None` returns the bare model output).

**Weights.** DDSP-SVC voices are trained per user, and no public 6.3 checkpoint exists. The DDSP and
flow networks are therefore exported with seeded random weights of the 6.3 architecture
(`configs/reflow.yaml`), as the RVC templates were (ADR 0004). The vocoder is the real one.

P5-A imports users' checkpoints (`model_*.pt` + `config.yaml`, torch zip) with the allow-list
unpickler. It then either writes the weights into per-config templates (the ADR 0004 recipe) or
exports per voice. P5-A decides which. Either way the `ddsp` graph takes its mel settings from the
voice's own vocoder config.

**Range.** The vocoder alone matches upstream at 88.4 dB over a 4 s E2 → D#7 sweep. The mini-NSF
source (5512.5 Hz) keeps D#7 (2489 Hz) under its Nyquist frequency. Gross pitch error needs a trained
voice, so it is P5-A's measurement.

**EPs.**
- CUDA with graphs for `reflow_step`, `vocoder` and ContentVec; `ddsp` cannot be captured.
- DirectML passes parity at 75.3 dB. The flow step, the vocoder and ContentVec are quick there,
  but `ddsp` takes 2.0–4.6 s per call. ORT's own profile (table below) locates it:
  - DirectML's graph fusion refuses the pinned shapes of `ddsp`, so it runs unfused with dynamic
    shapes: 271 DirectML nodes and 10 small shape ops on the CPU.
  - The CPU-placed ops take 0.09 ms per run, and ORT's node times (the CPU-side dispatch of the
    DirectML work) add up to 5–7 ms. CPU fallback and dispatch are therefore not the cost.
  - For the whole call the GPU sits at 100 %, and one CPU thread stays busy. So the time is
    DirectML executing the unfused graph. Which kernel is slow is not visible to ORT's profiler;
    the CUDA EP runs the same graph in about 3 ms.
  - DDSP-SVC on DirectML is therefore NO-GO as exported. P5-A finds the slow kernel (bisect the
    graph or use PIX), then exports `ddsp` with static shapes per preset, or splits out the part
    that breaks fusion.
- The CPU cannot keep up live: on ORT 1.28's CPU EP at 2 threads a tick takes 3.4–5.6× its block.
  The vocoder and ContentVec alone exceed the block.
- The family is therefore CUDA-only for now.

<!-- tables:profile -->
*DDSP-SVC on DirectML, ORT's own profile* (`ddsp_svc/parity.py --ep dml --profile`; median of 5 runs after 2 warm-up runs. Node time = the summed per-node times ORT records, which on DirectML is the CPU-side dispatch of each node, not its GPU execution; process CPU = this process's CPU time per run; GPU = median utilisation sampled every 0.25 s during the runs, shown when the runs last long enough for 4 samples):

| preset | graph | wall (ms) | node time (ms) | process CPU (ms) | GPU (%) | nodes per EP | dims pinned |
|---|---|---|---|---|---|---|---|
| `upstream_gui` | ddsp | 4489.1 | 5.08 | 4622 | 100 | CPU 10, Dml 271 | no (refused) |
| `upstream_gui` | contentvec | 4.2 | 0.10 | 3 | — | Dml 1 | yes |
| `upstream_gui` | reflow_step | 1.0 | 0.09 | 0 | — | Dml 1 | yes |
| `upstream_gui` | vocoder | 5.7 | 0.10 | 6 | — | Dml 1 | yes |
| `sigil_singing` | ddsp | 2016.0 | 6.02 | 2047 | 100 | CPU 10, Dml 271 | no (refused) |
| `sigil_singing` | contentvec | 3.3 | 0.12 | 3 | — | Dml 1 | yes |
| `sigil_singing` | reflow_step | 0.7 | 0.10 | 0 | — | Dml 1 | yes |
| `sigil_singing` | vocoder | 2.0 | 0.10 | 0 | — | Dml 1 | yes |
| `sigil_singing_fast` | ddsp | 2022.7 | 7.48 | 1650 | 100 | CPU 10, Dml 271 | no (refused) |
| `sigil_singing_fast` | contentvec | 3.7 | 0.10 | 3 | — | Dml 1 | yes |
| `sigil_singing_fast` | reflow_step | 0.7 | 0.11 | 3 | — | Dml 1 | yes |
| `sigil_singing_fast` | vocoder | 2.3 | 0.12 | 3 | — | Dml 1 | yes |

Ops the DirectML session placed on the CPU (`ddsp`): Unsqueeze × 3, Concat × 3, Gather × 2, Slice × 1, Squeeze × 1, 0.09 ms per run in all.
<!-- /tables:profile -->

### MeanVC2 (40 ms model), GO as an experimental CPU family

**Graphs** (`scripts/vc-families/meanvc/export.py`, all fixed-shape except `fbank` and `vocos`):

| graph | inputs | outputs | notes |
| --- | --- | --- | --- |
| `fbank` | wav16k | Kaldi fbank [1,F,80] | DC removal, pre-emphasis, the povey window and the DFT fold into one strided convolution |
| `asr` | fbank [1,11,80], offset i64, rcs i64 (4), att_cache [6,4,4,128], cnn_cache [6,1,256,8] | bn [1,2,256] and both caches | The Fast-U2++ TorchScript encoder, exported as is |
| `gtm` | spks [1,256] | k_mem, v_mem [1,32,256] | Once per voice |
| `dit_step` | x [1,8,80], t, r, dt, cond [1,8,256], spks, k_mem, v_mem, offset i64, n_valid i64, cache [4,2,2,12,64] | x_next, cache_out | Mean flow. Upstream's growing per-layer KV cache becomes one fixed 12-frame buffer: front-padded keys are masked to −inf, and RoPE runs over explicit positions. Checked equal to upstream over sliding calls |
| `vocos` | mel [1,80,M] | wav [1,160(M−1)] | Vocos rebuilt from the TorchScript state dict (strict). The ISTFT head is a transposed convolution |

**Loop.** The loop is `meanvc/sampler.py`, which is upstream `run_rt.py` `VCRunner` state for state:
- The fbank remainder carries over between calls.
- ASR runs every 8 fbank frames over an 11-frame window.
- BN frames are upsampled 4× (linear, align_corners).
- The DiT takes 4 frames + a 4-frame future with 2 mean-flow steps (1 → 0.5 → 0).
- Upstream already reuses the future block's noise, which is anchored noise in all but name.
- Vocos runs over 2 + 4 frames, with a 10 ms crossfade.

Upstream processes 160 ms audio chunks and interpolates each chunk's BN to 17 frames. The Rust port
generalises this to 4k + 1 frames for k new BN frames, which is identical in the steady state, and
can then tick every 80 ms.

**Speaker encoder.** The speaker encoder is WavLM-Large plus a fine-tuned ECAPA-TDNN. Its fine-tuned
weights are a Google Drive download that cannot be scripted, so they are a **manual step**. Parity
therefore used a fixed, seeded unit-vector embedding: graph parity does not depend on the voice.
Listening quality is unmeasured until P5-C exports the encoder and runs the A/B.

**EPs.** All of these were measured on ORT 1.28, the engine's runtime (the CPU EP of the
`venv-ort128-cuda` build).
- CPU, at 1 thread: RTF 0.148 with 2 steps and 0.105 with 1 step; the chain passes parity at
  59.2 dB. A second thread does not help, because the graphs are tiny. (On onnxruntime 1.19.2, the
  spike's first runtime, it was RTF 0.176 and 61.3 dB.)
- On CUDA the ASR cannot be captured, because it slices its positional table with a runtime offset.
  The ASR alone is slower there than on one CPU thread (4.2 against 2.4 ms).
- With the DiT and Vocos captured, CUDA reaches RTF 0.094; without graphs it is RTF 0.25.
- The CPU already meets the budget and costs no VRAM, so the family is CPU-first.

## Evidence

Everything runs on the H2 dev rig:
- **CPU:** AMD Ryzen 7 7800X3D (8 cores, 16 threads).
- **GPU:** RTX 4090 24 GB, driver 617.14, CUDA 13.2 with cuDNN 9.26.
- **Reference:** Python 3.12.10 with torch 2.3.1+cu121. Upstream runs in PyTorch fp32 on the CPU
  as the reference.
- **Runtimes:** ONNX Runtime 1.28.0, the engine's version, for the CPU EP and CUDA (the
  `venv-ort128-cuda` build of onnxruntime-gpu; CUDA with TF32 off); DirectML on
  onnxruntime-directml 1.24.4, the newest DirectML wheel.
- **Sessions:** built the way the engine builds them: sequential, spinning off, 2 intra-op threads
  unless stated, and every symbolic dimension pinned. A DirectML session that refused its pinned
  dimensions falls back to dynamic shapes, and the report records it (`session.dims_refused`).
- **Timing:** each call is timed in 3 blocks after warm-up, and the quickest block's median is kept;
  every block's median and the kept block's p90 are in the reports. A block is 100 calls for
  MeanVC2; for Seed-VC and DDSP-SVC it is 30 calls on GPU EPs, 7 for Seed-VC on the CPU and 10 for
  DDSP-SVC on the CPU. Before each GPU measurement the scripts wait for utilisation under 40 %; the
  desktop alone holds 25–30 %, and another benchmark holds the GPU near 100 %. The rig is shared,
  and a first round of GPU numbers taken during another workstream's benchmark read 2–4× high;
  those were discarded. `timing_conditions` and `gpu_before` in each report record the load seen.
- **Identity:** every report section (parity, cost, profile, joins) records the sha256 of each graph
  file it measured (`graphs`). A report that merges sections measured on different exports lists
  them under `graph_mismatch`; none of the current reports has one. All eight reports were
  measured from scratch on the current exports.
- **Commands:** `python scripts/vc-families/<family>/parity.py [--ep cuda|dml]` in the venvs named in
  `scripts/vc-families/README.md`; `ddsp_svc/parity.py --ep dml --profile` adds ORT's profile. The
  tables below are generated from the reports by `python scripts/vc-families/adr_tables.py --write`.
- **Input audio:** `jfk.wav` (11 s).
- **Seed-VC reference voice:** `ja-short.wav`.

**Parity** (SNR in dB against upstream; audio rows show waveform, then log-mel in parentheses):

<!-- tables:parity -->
*Seed-VC* (preset `upstream_readme` unless noted; stage rows feed each graph upstream's own inputs, chain rows run the whole reference branch and one tick):

| stage | CPU fp32 | CUDA fp32 | DirectML fp32 | CUDA hift16 | CUDA xlsr16 | CUDA fp16 (all) |
|---|---|---|---|---|---|---|
| xlsr (tick window) | 109.32 | 105.58 | 108.51 | 105.72 | 50.05 | 49.8 |
| campplus | 108.25 | 107.46 | 116.72 | 107.46 | 107.46 | 107.46 |
| mel (prompt) | 137.36 | 137.28 | 136.67 | 137.28 | 137.28 | 137.28 |
| lr (tick) | 127.87 | 123.92 | 127.5 | 123.92 | 123.92 | 123.92 |
| dit (estimator, CFG batch) | 128.44 | 128.56 | 125.56 | 128.56 | 128.56 | 69.43 |
| dit_step | 132.14 | 132.71 | 125.32 | 132.71 | 132.71 | 77.94 |
| dit_step1 | 135.96 | 137.2 | 136.79 | 137.2 | 137.2 | 79.03 |
| hift (whole tick, on upstream's mel) | 70.4 (102.39) | 78.25 (105.94) | 80.84 (109.93) | 47.05 (55.47) | 78.25 (105.94) | 47.05 (55.58) |
| hift_f0 | 129.93 | 127.99 | 130.4 | 127.99 | 127.99 | 127.99 |
| hift_f0 + hift_nsf, partial decode (returned audio) | 68.73 (94.74) | 77.76 (101.55) | 76.83 (102.03) | 47.36 (50.87) | 77.76 (101.55) | 47.36 (50.87) |
| chain: tick mel | 118.52 | 126.55 | 118.77 | 125.83 | 72.19 | 68.33 |
| chain: tick audio, `hift` | 55.45 (77.97) | 66.76 (87.97) | 68.47 (89.93) | 47.17 (50.5) | 8.05 (32.43) | -2.99 (24.52) |
| chain: tick audio, partial decode | 55.74 (78.2) | 67.16 (88.39) | 68.85 (90.24) | 47.57 (50.75) | 8.05 (32.43) | -2.99 (24.53) |
| chain: tick audio, `sigil_lean`, `hift` | 101.15 (123.85) | 101.22 (119.14) | 99.18 (121.72) | 38.08 (63.22) | 48.9 (69.87) | 37.21 (60.53) |
| chain: tick audio, `sigil_lean`, partial decode | 96.67 (115.45) | 96.98 (114.55) | 96.23 (115.23) | 37.83 (63.04) | 48.9 (69.88) | 37.28 (60.61) |

For scale: upstream's own fp16 real-time mode against upstream fp32 on CUDA (same draws) gives tick audio -3.66 dB (log-mel 24.19 dB) and tick mel 66.95 dB.

*DDSP-SVC 6.3* (preset `upstream_gui`: 2.5 s window, 20 steps from t_start 0.7; F0 sweeps E2 → D#7 over the decoded frames):

| stage | CPU | CUDA | DirectML |
|---|---|---|---|
| units: Sigil ContentVec ONNX × 2 (tta2x) vs upstream encoder | 115.88 | 113.34 | 111.36 |
| ddsp: cond (DDSP mel) | 85.86 | 85.89 | 83.06 |
| ddsp: x (normalised) | 83.3 | 83.33 | 80.5 |
| chain: mel after 20 flow steps | 90.71 | 90.73 | 88.05 |
| chain: tick audio (formant +2 st) | 80.33 (86.64) | 80.2 (86.44) | 75.29 (83.78) |
| chain: tick audio, update on the host (`velocity_f`) | 80.33 (86.64) | 80.2 (86.46) | 75.29 (83.78) |
| chain: tick audio (formant 0) | 84.6 (92.17) | 84.56 (92.17) | 77.46 (88.17) |
| chain: tick audio + `SvcDDSP.infer`'s volume gate | 83.48 (91.05) | 83.32 (90.84) | 75.89 (88.15) |
| vocoder alone: 4 s real mel, F0 sweep E2 → D#7 | 88.44 (111.48) | 88.48 (111.28) | 81.39 (99.29) |
| upstream exporter: `encoder.onnx` cond over the decoded frames | 21.16 | 21.16 | 21.16 |
| upstream exporter: `velocity.onnx`, int64 t rounded / truncated, tick audio | 80.35 / 49.09 | 80.18 / 49.09 | 75.29 / 49.12 |
| upstream exporter: `encoder.onnx` output NaN at f0 = 0 | yes | yes | no |

The gate row mutes 44 % of the decoded audio (threshold -41 dBFS, the median frame volume).

*MeanVC2* (upstream `process_file` on the whole 11 s utterance, 2 ODE steps):

| stage | CPU | CUDA |
|---|---|---|
| fbank (whole utterance) vs torchaudio Kaldi | 109.18 | 109.44 |
| asr: 12 streaming calls (bn) | 121.75 | 119.25 |
| asr: attention cache after 12 calls | 122.93 | 118.04 |
| gtm | 133.4 | 138.52 |
| dit (bare, 5 streaming calls, fixed cache vs upstream's growing one) | 120.76 | 123.51 |
| chain: upsampled BN (69 chunks) | 104.69 | 104.25 |
| chain: mel (552 DiT calls) | 102.48 | 102.73 |
| chain: output audio (11 s) | 59.23 (82.16) | 54.64 (79.81) |
| vocos alone (24 frames) | 88.91 (131.79) | 92.43 (134.29) |
| crossfade `linspace` bit-exact with torch | yes | yes |
<!-- /tables:parity -->

**Joins between Seed-VC ticks** (not gated; `stream_join`: correlation is the normalised correlation of
the current tick against SOLA's crossfade buffer from the previous tick; offsets in samples at
22.05 kHz relative to upstream's join):

<!-- tables:join -->
*Seed-VC joins* (CUDA fp32, `upstream_readme`, 8 consecutive joins from 1 s of `jfk.wav`; the content moves 15.811 mel frames per block, so the grid advances 16 frames and the ticks line up -127 samples from upstream's join):

| levers | mel overlap SNR (dB) | best correlation (offsets) | at the grid offset | best in SOLA's 0…441 |
|---|---|---|---|---|
| upstream: fresh z, harmonic phases and sine noise every tick | 13.93 | 0.372 (128, -501, 6, 66, -87, -262, -512, 512) | -0.071 | 0.228 |
| z and sine noise anchored on the mel-frame grid | 16.584 | 0.71 (-128, -126, -238, -374, -126, -127, -127, -127) | 0.566 | 0.423 |
| … and the NSF phase anchored at the join | 16.584 | 0.756 (-128, -126, -132, -310, -126, -127, -127, -127) | 0.64 | 0.434 |

The CPU fp32 run gives the same means to within 0.001.
<!-- /tables:join -->

**Cost** (ms, median; RTF = tick / block):

<!-- tables:cost -->
*Seed-VC* (ms per call at each preset's shapes; `g` = replayed CUDA graph, `b` = IOBinding, plain = host tensors; tick = xlsr + lr + steps × step + vocoder, with `hift` over the whole tick or `hift_f0` + `hift_nsf` over the returned frames plus 12 frames of context):

| preset (block, steps) | EP / variant | xlsr | lr | step | hift | hift_f0 + hift_nsf | tick, hift | tick, partial | RTF |
|---|---|---|---|---|---|---|---|---|---|
| `upstream_readme` (180 ms, 10) | CUDA fp32 | 5.44 g | 0.61 b | 2.25 g | 7.31 g | 0.38 g + 5.42 g | 35.85 | 34.34 | 0.199 / 0.191 |
| `upstream_readme` (180 ms, 10) | CUDA hift16 | 4.79 g | 0.77 b | 2.15 g | 5.48 g | 0.31 g + 4.62 g | 32.5 | 31.96 | 0.181 / 0.178 |
| `upstream_readme` (180 ms, 10) | CUDA fp16 (all) | 2.90 g | 0.69 b | 2.17 g | 5.51 g | 0.23 g + 4.75 g | 30.84 | 30.3 | 0.171 / 0.168 |
| `upstream_readme` (180 ms, 10) | DirectML fp32 | 6.47 | 0.90 | 3.69 | 9.37 ¹ | 0.23 + 6.48 ¹ | 53.65 | 50.99 | 0.298 / 0.283 |
| `upstream_readme` (180 ms, 10) | CPU fp32, 2 threads | 269.66 | 2.04 | 107.79 | 184.06 | 2.19 + 103.62 | 1533.7 | 1455.46 | 8.521 / 8.086 |
| `sigil_low` (100 ms, 6) | CUDA fp32 | 5.08 g | 0.46 b | 2.01 g | 6.87 g | 0.52 g + 5.30 g | 24.45 | 23.41 | 0.245 / 0.234 |
| `sigil_low` (100 ms, 6) | CUDA hift16 | 5.19 g | 0.63 b | 1.95 g | 5.06 g | 0.42 g + 4.23 g | 22.56 | 22.15 | 0.226 / 0.222 |
| `sigil_low` (100 ms, 6) | CUDA fp16 (all) | 2.19 g | 0.52 b | 1.93 g | 5.18 g | 0.23 g + 4.42 g | 19.47 | 18.94 | 0.195 / 0.189 |
| `sigil_low` (100 ms, 6) | DirectML fp32 | 4.20 | 0.79 | 3.00 | 7.05 ¹ | 0.30 + 6.35 ¹ | 30.07 | 29.67 | 0.301 / 0.297 |
| `sigil_low` (100 ms, 6) | CPU fp32, 2 threads | 121.70 | 1.30 | 98.71 | 126.84 | 1.50 + 93.22 | 842.11 | 809.99 | 8.421 / 8.1 |
| `sigil_lean` (100 ms, 4) | CUDA fp32 | 4.51 g | 0.70 b | 1.20 g | 6.99 g | 0.49 g + 5.25 g | 16.98 | 15.73 | 0.17 / 0.157 |
| `sigil_lean` (100 ms, 4) | CUDA hift16 | 4.51 g | 0.62 b | 1.19 g | 5.18 g | 0.55 g + 4.35 g | 15.07 | 14.78 | 0.151 / 0.148 |
| `sigil_lean` (100 ms, 4) | CUDA fp16 (all) | 2.41 g | 0.39 b | 1.19 g | 5.23 g | 0.54 g + 4.43 g | 12.8 | 12.55 | 0.128 / 0.125 |
| `sigil_lean` (100 ms, 4) | DirectML fp32 | 4.21 | 0.79 | 1.93 | 11.44 ¹ | 0.39 + 10.84 ¹ | 24.15 | 23.94 | 0.242 / 0.239 |
| `sigil_lean` (100 ms, 4) | CPU fp32, 2 threads | 119.43 | 1.30 | 29.92 | 119.84 | 1.37 + 80.91 | 360.28 | 322.72 | 3.603 / 3.227 |

¹ DirectML refused the pinned dimensions; the session runs with dynamic shapes (`session.dims_refused` in the report).

Once per voice (3 s reference): CUDA fp32 xlsr 6.47 b, campplus 12.09 b, mel 0.43 b ms; CPU xlsr 633.50, campplus 29.80, mel 6.33 ms.

*DDSP-SVC 6.3* (tick = 2 × ContentVec (tta2x) + ddsp + steps × reflow_step + vocoder; F0 extraction and resampling not included):

| preset (block, steps) | EP | ContentVec | ddsp | step | vocoder | tick | RTF |
|---|---|---|---|---|---|---|---|
| `upstream_gui` (500 ms, 20) | CUDA | 4.07 g | 2.70 b | 0.49 g | 6.56 g | 27.16 | 0.054 |
| `upstream_gui` (500 ms, 20) | DirectML | 5.64 | 4568.82 ¹ | 1.56 | 5.67 | 4616.97 | 9.234 |
| `upstream_gui` (500 ms, 20) | CPU, 2 threads | 288.05 | 121.92 | 37.11 | 270.21 | 1710.47 | 3.421 |
| `sigil_singing` (100 ms, 10) | CUDA | 2.59 g | 2.61 b | 0.45 g | 4.59 g | 16.85 | 0.169 |
| `sigil_singing` (100 ms, 10) | DirectML | 3.40 | 2065.36 ¹ | 1.16 | 2.43 | 2086.19 | 20.862 |
| `sigil_singing` (100 ms, 10) | CPU, 2 threads | 126.02 | 54.42 | 16.12 | 87.71 | 555.38 | 5.554 |
| `sigil_singing_fast` (100 ms, 5) | CUDA | 2.34 g | 2.69 b | 0.45 g | 4.53 g | 14.14 | 0.141 |
| `sigil_singing_fast` (100 ms, 5) | DirectML | 3.42 | 2052.29 ¹ | 0.67 | 2.87 | 2065.34 | 20.653 |
| `sigil_singing_fast` (100 ms, 5) | CPU, 2 threads | 128.73 | 54.59 | 15.03 | 93.05 | 480.22 | 4.802 |

¹ DirectML refused the pinned dimensions; the session runs with dynamic shapes (`session.dims_refused` in the report).

*MeanVC2* (per 40 ms output chunk = fbank/4 + asr/2 + steps × dit_step + vocos):

| EP | fbank | asr | dit_step | vocos | 1 step: ms (RTF) | 2 steps: ms (RTF) |
|---|---|---|---|---|---|---|
| CPU, 1 thread | 0.08 | 2.43 | 1.71 | 1.27 | 4.21 (0.105) | 5.915 (0.148) |
| CPU, 2 threads | 0.09 | 2.75 | 2.08 | 1.43 | 4.913 (0.123) | 6.994 (0.175) |
| CUDA | 0.13 g | 4.24 b | 0.68 g | 0.25 g | 3.08 (0.077) | 3.757 (0.094) |

MeanVC2 on CUDA without graphs (IOBinding, 2 steps): 10.082 ms per chunk (RTF 0.252).
<!-- /tables:cost -->

## Cross-cutting findings (for the engine, P3-C2, P3-E, P3-F and P5)

1. **CAM++ pooling.** ONNX Runtime's `AveragePool(ceil_mode)` divides the last partial window by the
   full kernel size. PyTorch divides it by the valid frames.
   - A CAM++ export built that way gives 4.5 dB SNR on a 3 s reference. Sigil's graph uses explicit
     segment means and gets 107–117 dB.
   - The bench asset `bench/campplus` (sherpa-onnx's 3D-Speaker export) has the same flaw. It
     contains 52 `AveragePool` nodes with `ceil_mode = 1` and a 100-frame kernel. Against the
     upstream model it gives 4.4 dB SNR (cosine 0.925) on the same fbank, so it is not the same
     function.
   - **P3-F / benchmark v2 must not use it as the T5/T10 speaker-similarity reference as is.**
     Re-export CAM++ the way `seed_vc/export.py` does, or validate the asset first.
   - Separately, the fbank scaling matters slightly: Kaldi's ×32768 against upstream's unscaled input
     gives 29 dB (cosine 0.9994), through the log floor.
2. **fp32 on CUDA** needs `use_tf32 = 0`, or error rises about 10× (ADR 0004).
3. **Vocoders that predict their own F0** (HiFT) turn any upstream difference in the mel into phase
   drift over the window.
   - The waveform then decorrelates, although it sounds the same.
   - So waveform SNR gates fp32 only. A reduced-precision lever gates on its stage outputs, and on
     log-mel against upstream's own reduced-precision mode.
   - For streaming, the F0 has to be known before synthesis so the NSF phase can be chosen: HiFT is
     exported split (`hift_f0` + `hift_nsf`), and DDSP-SVC's vocoder takes its F0 as an input
     (`phase0`). Anchoring the phase alone does not make ticks coherent; it needs anchored noise and
     a block on the mel-frame grid (the join table).
4. **Launch-bound small graphs.** Per-step graphs over a few hundred frames are launch-bound on
   CUDA, so graph capture is mandatory for them. Captured, each step costs 2.5–5× less than with
   IOBinding alone:
   - Seed-VC `dit_step`: 6.0 → 2.3 ms.
   - MeanVC2 `dit_step`: 3.5 → 0.68 ms.
   - DDSP `reflow_step`: 1.2 → 0.49 ms.

   Graphs with runtime-offset slices, or with CPU-placed shape ops, cannot be captured: Seed-VC
   `lr`, MeanVC2 `asr` and DDSP `ddsp`. Such a graph either stays uncaptured (all are small) or
   gets its offset slice moved to the host. For MeanVC2 on CUDA, that is P5-C's option.
5. **DirectML session lifetime** (onnxruntime-directml 1.24.4).
   - Running a session after another DirectML session in the same process had been released crashed
     with an access violation, reproducibly. Sessions that were never released ran fine.
   - DirectML's graph fusion also refused some graphs once their free dimensions were pinned ("The
     parameter is incorrect" in `DmlGraphFusionHelper`): DDSP `ddsp` and both Seed-VC HiFT graphs.
     The unpinned graph ran. The reports record every such fallback (`session.dims_refused`).
   - **P3-C2/P3-E:** check both on ort rc.13 / ORT 1.28. Until then, keep DirectML sessions for the
     life of the engine rather than dropping and rebuilding them, and fall back to unpinned
     dimensions when a pinned DirectML session fails to build.
6. **Export mechanics.**
   - `torch.stft` and `torch.istft` have no portable ONNX export. DFT bases as strided and
     transposed convolutions match to more than 100 dB (`test_families.py`).
   - ONNX `Resize(nearest)` rounds differently from PyTorch at exact multiples, so it becomes an
     explicit gather.
   - `torch.amp.autocast` decorators (x-transformers RoPE) break the tracer.
   - weight_norm at opset 18 does not export under torch 2.3, so it is folded first.
   - A TorchScript archive saved by torch ≥ 2.5 needs one `pad_sequence` call rewritten before
     torch 2.3 loads it.
   - fp16 exports must copy every tensor that upstream's autocast keeps in fp32 (buffers such as
     timestep frequencies, parameters such as Snake alphas, and inputs such as t) before `.half()`
     and put the exact values back: casting back after `.half()` keeps the rounding.
7. **Upstream DDSP-SVC exporter** (why Sigil does not use it):
   - Its `velocity.onnx` takes int64 `1000 t`. Its own loop truncates that and loses 31 dB (49 dB
     against 80 dB rounded).
   - Its `encoder.onnx` is NaN wherever f0 = 0 on CPU and CUDA.
   - It takes the mel of the whole window, so the decoded tail differs from upstream at the slice
     (21 dB).
   - It hard-codes the DDSP mel's STFT (n_fft = win = DDSP `win_length`, 128 bins, 40–16000 Hz),
     which matches upstream only while the vocoder's config happens to agree.
8. **Parity helpers for P5.** `common.compare` / `gate` fail a row when either side is non-finite
   (an upstream path that NaNs on silence, an fp16 reference, a user checkpoint). P5-A/B/C reuse
   them for their own parity.

## Consequences

- **P5-B (Sigil-ZS)** builds on `seed_vc/` graphs and `sampler.py`. No Seed-VC sidecar is needed.
  - Sidecar v2 stays for X-VC and StreamVoiceAnon.
  - Defaults: fp32, CUDA graphs per preset shape, and the reference cache per library voice (style,
    prompt mel, prompt condition; about 25 ms of GPU work once).
  - The vocoder runs as `hift_f0` + `hift_nsf` with partial decode, with `hift` as the fallback;
    the benchmark confirms the default.
  - Coherent ticks: a block of whole mel frames on the content time base, z and the sine noise drawn
    per absolute mel frame, the NSF phase anchored at the join, and a SOLA search that contains the
    expected offset. Measure the join with `stream_join` before and after.
  - P5-B's benchmark picks the preset; `sigil_low` is borderline for T10 on H1. The XLS-R and DiT
    fp16 levers stay off by default.
  - The largest graph is XLS-R at 657 MB fp32 (329 MB fp16). The fp32 set holds about 0.86 GB of
    weights with one step graph and the split vocoder. That is the VRAM floor to plan the T3 budget
    around.
- **P5-A (DDSP-SVC)** uses Sigil's graphs, not upstream's exporter, plus an importer for users'
  checkpoints through the allow-list parser, the shared ContentVec (tta2x), and RMVPE/FCPE F0 with
  upstream's F0 post-processing.
  - The engine applies `SvcDDSP.infer`'s volume gate (threshold setting, default −45 dBFS).
  - Each voice's `ddsp` graph takes its mel settings from that voice's vocoder config.
  - The family is CUDA-only until the `ddsp` graph runs acceptably on DirectML. The UI hides or
    labels it on other PCs.
  - Its graph set is about 330 MB of fp32 weights, most of it the flow network.
- **P5-C (MeanVC2)**, in order:
  1. Export the speaker encoder. Its first step is the manual Google Drive download.
  2. Tick every 80 ms (the 4k + 1 generalisation).
  3. Run on the CPU EP.
  4. A/B against RVC Streamer. On failure it is dropped, per the plan's fallback.
- **P3-F:** the CAM++ bench asset finding (cross-cutting finding 1).
- **P3-C2/P3-E:** the DirectML session lifetime and pinned-dimension findings (finding 5).
- **Catalog.** The graphs stay local-only (status `unpublished`) until the user publishes assets.
  P5's catalog fragments (`vc-zeroshot.json`, `vc-ddsp.json`) record the source revisions listed in
  the export scripts and the graphs' sha256 from the reports.
- **This spike changes no Rust or TypeScript code** and touches no hotspot. The voice-changer
  benchmark (`--headless bench`) is unaffected, and no T1–T9 figure moves. The spike supplies the
  T10 feasibility numbers above.
- **Rollback.** Nothing ships from here. If a P5 port misses T10, the plan's fallback applies:
  sidecar v2 for Seed-VC through seed-vc-realtime, and dropping MeanVC2.

## P5 outcomes

Recorded by P5-I0 (2026-10-01) from the P5-A, P5-B, P5-C and P5-E results. All three families shipped in Sigil's engine as planned; no fallback was needed. Measurements: [VC_BENCHMARKS → P5 voice families](../benchmarks/VC_BENCHMARKS.md#p5-voice-families); code map: [DEVELOPER_GUIDE → Voice families](../DEVELOPER_GUIDE.md#voice-families-p5).

### Sigil-ZS (P5-B): shipped in-engine, T10 met on H2

- **Shipped** as the `sigil_zs` family (`sigil-zs-1`) on CUDA and DirectML at fp32; the CPU is refused. The Rust port matches the reference Python sampler (same graphs and noise, ONNX Runtime CPU): tick mel 127–137 dB, tick audio 83–89 dB (gate 35 dB); reference stage style 102–104 dB, prompt mel 154–155 dB, prompt condition 117–119 dB; resample 142 dB, fbank 112–115 dB.
- **The prompt is chosen from the clip's speech** (`reference::select_prompt`): upstream takes the clip's first seconds, and the test reference starts with 1.7 s of room tone, so the 1.5 s prompt converted toward nobody (cosine −0.02) and the 3 s prompt held 1.3 s of speech. The function drops quiet ends and takes the prompt-length window with the most speech (`sigil_lean` −0.02 → 0.39–0.54).
- **Preset prompts set by measurement:** `sigil_low` 3 → 4 s (offline tick 22.0 → 24.7 ms CUDA, 27.0 → 30.2 ms DirectML), `sigil_lean` 1.5 → 3 s (15.1 → 16.0 / 18.2 → 20.0 ms).
- **Speaker similarity** (exact CAM++ export, three sources, one reference): upstream offline (`inference.py` defaults through our graphs: the whole clip as prompt, 30 steps) 0.559 mean; `sigil_low` (4 s) 0.582; `sigil_low` (5 s) 0.587; `sigil_lean` (3 s) 0.485; `upstream_readme` 0.577. The stream beats its own whole-utterance conversion on every row.
- **T10 on H2:** `sigil_low` with the Stable output buffer (80 / 150 ms; since the P5 final gate the family's minimum is per preset and EP, 95 / 165 ms for `sigil_low` on CUDA, `sigil_zs::tune::min_output_buffer_ms`): paced latency p95 252–256 ms idle (target 300), p99 248–280 ms under the H2 load simulator (target 450), similarity 0.582 ≥ 0.559 upstream offline: **met**. Upstream's live GUI runs 430–690 ms. H1 is not measured; the 4090 × 3 estimate is a 74 ms tick on CUDA (about 273 ms end to end with a 105 ms buffer) and 89 ms on DirectML, where `tune::choose` would pick `sigil_lean`.
- `sigil_lean` meets the delay but not the similarity bar; `upstream_readme` meets similarity but not the delay (p95 about 351 ms) and is a manual choice only.
- **Coherent ticks** as implemented: the block moves the content by 8.91 mel frames, noise is anchored on a 9-frame grid and the phase anchor pairs samples where the grid lines up inside the SOLA search; mean join correlation 0.53–0.65 (two renditions, no audible seam measured; blind listening is P7's).
- fp16 (vocoder + TF32: 15.8 vs 24.6 ms) moves the mel and the HiFT F0, so it stays opt-in with the fp16 XLS-R and DiT switches. TensorRT-RTX requests run on the CUDA sessions. `gpu_priority` and the governor showed no gain and stay off.
- **Finding (affects RVC too):** an idle NVIDIA GPU drops to about 900 MHz for one burst per 100 ms; paced ticks then take 1.4–2.4× the offline time with spikes to 80–105 ms, while a game keeps the clocks up. P3's paced RVC numbers likely carry the same effect.

### DDSP-SVC 6.3 (P5-A): shipped in-engine, DirectML now GO, T10 met with CREPE / SwiftF0 / fusion

- **DirectML is GO:** the `ddsp` graph's inverse STFT became a MatMul plus overlap-add (115 dB against the ConvTranspose form): 3.06 ms per call in Python and 4.2 ms in the engine, down from 2.0–4.6 s. DirectML still refuses the pinned shapes, so the graph runs dynamic; a CUDA request that falls back to DirectML now rebuilds with dynamic shapes (before, `ddsp` landed on the CPU at 32 ms per call).
- **Weights:** per-config template graphs without weights plus `layout.json` and a Rust importer (allow-list unpickler in the import worker): no per-voice export and no Python at import; about 1 s for a 220 MB checkpoint in release.
- **T10** (sung E2 → D#6 sweep at +0 and +12 st, so the output reaches D#7; pitch path = the F0 the vocoder receives):

  | F0 method | Pitch-path GPE | Accuracy | T10 |
  |---|---|---|---|
  | CREPE | 0.00 % | 6.9 cents | met |
  | SwiftF0 | 0.00 % | 5.9 cents | met |
  | Fusion | 0.00 % | 4.5 cents | met |
  | FCPE | 0.00 % | 56 cents (flat at the top) | met on GPE, off-key at the top |
  | RMVPE | 2.36 % | 13.1 cents | missed: notes above about D6 (1.15 kHz input) come out an octave low |

  With CREPE the output scores 0.00 % GPE on CUDA and DirectML (including the frames at ≥ 2 kHz, pYIN-style Rust tracker and librosa pYIN); the offline ceiling is 0.00 %; stream against ceiling F0 16–17 cents, 0 % gross errors, MCD 2.3–2.4 dB (100 ms presets) and 1.0–1.5 dB (`upstream_gui`). New DDSP-SVC voices therefore start with CREPE in the singing range (P5-I0).
- **Cost per 100 ms tick** (release, CREPE): `sigil_singing` CUDA 27.9 ms (content encoder × 2 18.1, F0 0.6, ddsp 2.5, flow 4.7, vocoder 1.8), DirectML 28.4 ms; `sigil_singing_fast` 25.6 ms; `upstream_gui` 35.8 ms CUDA / 103.6 ms DirectML per 500 ms block. VRAM 1028 MB on CUDA with CREPE (1540 MB with RMVPE), 703 MB on DirectML.
- **Delay per preset** (paced): `sigil_singing` about 185 ms p50 with a 35 ms output prefill (which still underruns 0–94 times per 9 s), 215 ms with 70 ms and 0 underruns, so 70 / 140 ms is the family's output-buffer minimum (P5-I0); `upstream_gui` about 614 ms. H1 simulation p99 207–243 ms, 0 blocks over budget.
- The two pooled ContentVec passes (tta2x) are 65 % of the CUDA tick (about 9 ms per pass against 2.6 ms graph-captured in this ADR's spike): putting the shared encoder on the session-plan path is the largest remaining cost cut, for this family and for RVC's T2.
- No public DDSP-SVC 6.3 checkpoint exists: every real-model figure uses a seeded random voice (`export_templates.py --voice`). Pitch does not depend on the voice; timbre and the listening check need the user's own trained voice.

### MeanVC2 (P5-C): GO as an experimental CPU family

- Real time on one CPU thread at 0.2–0.3 cores; the Rust loop matches the Python reference at 85.3 dB SNR (`scripts/vc-families/meanvc/rust_reference.py`). Silence costs 0.0098 cores (T4 silent target met).
- **Split tick:** the 80 ms tick runs as two 40 ms harness blocks, one DiT call per block; latency is unchanged and per-call compute halves (2-step p50 16.6 → 9.6 ms), which removed the output underruns (105 → 0 paced, 243 → 3 under the H3 game load).
- **A/B against the RVC CPU tier** (paced, interleaved): MeanVC2 2 steps 0.28 cores, 292.5 ms engine-added latency, 0 underruns; RVC Streamer RTF 3.4–4.4 (about 14 s of every 20 s dropped); autotuned RVC CPU 2.85 cores and about 1000 underruns per 20 s. On every performance axis MeanVC2 wins. Quality smoke (synthetic speaker embedding): DNSMOS OVRL 3.01–3.03 against RVC 3.11–3.22, and it does not keep the user's intonation (GPE 5–16 %, F0 RMSE 225–267 cents).
- **Still manual:** the speaker encoder (WavLM-Large + ECAPA-TDNN, `wavlm_large_finetune.pth`) is a Google Drive download; a voice gets its speaker from `speaker.npy` or an imported encoder graph. The listening and speaker A/B waits on it, so MeanVC2 stays experimental and is not the CPU default.
- It computes on the CPU on every rung (no CUDA graphs: no VRAM and no GPU context next to a game).

### X-VC and StreamVoiceAnon (P5-E): sidecar protocol v2

- They run only through sidecar protocol v2, wrapped as a voice-family model: frames sent without waiting, a per-frame deadline equal to the budget, cancel, reference upload, pitch per frame. A stalled sidecar never blocks the loop (tests `sidecar_stall_never_blocks_the_loop`, `a_stalled_server_never_blocks_the_caller`, `sidecar_recovers_after_a_stall_and_drops_stale_audio`).
- Their checkpoints (several GB) were not installed in P5, so the adapters are untested against the real models (manual check). The Seed-VC recipe, measured with the real model: about 300 ms heard with `sigil_low`, 400–440 ms with upstream's preset; pinning the graphs' shapes made DirectML real time (168–243 → 36–64 ms per 100 ms frame).
