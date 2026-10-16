# Voice engine roadmap: what would move the engine, not just its settings

Status: proposal of 2026-10-16. Nothing here is built or decided. The question it answers: how do we get better quality, lower latency, higher speaker similarity and lower pitch error together by changing the engine itself, not by tuning presets to a target. The evidence is [VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md), [ADR 0009](adr/0009-vc-scorer-v2.md) (scorer v2 and the ablations) and [ADR 0010](adr/0010-sve-optimization-p3b.md) (P3B optimization). Each number below is quoted from there, or marked derived or estimated, with its basis.

For the core audience (VRChat streamers whose captions matter most) the voice changer is secondary and the GPU is shared with the game, so the plan puts cheap, measurable steps first and makes every XL item conditional on a pilot. Existing RVC voices must keep working; each item says how it treats them.

## Where the engine stands

Sigil's voice engine (SVE) runs an imported RVC voice as split ONNX graphs: a content encoder (ContentVec), an F0 tracker, `enc` (text encoder) and `flowdec` (flow plus NSF-HiFiGAN decoder). It works in short blocks, recomputes the context every block, and joins blocks with SOLA and a crossfade. The execution ladder is CUDA, DirectML, then CPU. Against RVC WebUI's realtime loop on the same voice and clips (RTX 4090, 5 clips, offline tier, one run; [matched comparison](benchmarks/VC_BENCHMARKS.md#matched-comparison-one-run)), Sigil minus RVC:

| Point | DNSMOS OVRL (Sigil / RVC) | Difference | Speaker similarity (Sigil / RVC) | Fine pitch error (Sigil / RVC) |
|---|---|---|---|---|
| About 95 ms: Quality (80 ms) and Quality with Chunk 40 (98 ms), against RVC's 60 ms block (92 ms) | 2.95 and 2.97 / 2.86 | +0.09 [-0.08, +0.26] and +0.10 [-0.13, +0.34] | 0.696 and 0.714 / 0.742 | 16.1 and 16.3 / 20.6 cents |
| About 255 ms: Chunk 200 (258 ms), against RVC's 200 ms block (253 ms) | 3.09 / 3.10 | -0.02 [-0.24, +0.21] | 0.740 / 0.739 | 13.9 / 17.7 cents |
| 60 ms: Low latency, against RVC's 40 ms block (62 ms) | 2.63 / 2.70 | -0.07 [-0.25, +0.10] | not in the run | not in the run |

- **A tie that is not resolved.** Every DNSMOS interval includes 0, and with 5 clips that means "not resolved", not "equal" (about 17-86 clips would resolve a 0.05 difference at 50 % power). The one difference that resolves is speaker similarity: RVC is higher by 0.03-0.05 on all 5 clips at about 95 ms, and level at 255 ms. Part of that may be the reference (see below). Sigil's pitch error is 4-5 cents lower, not resolved.
- **One curve for both engines.** Both gain about 0.3 DNSMOS going from 60-90 ms to about 250 ms. Latency and quality trade along a curve that Sigil and RVC WebUI share.
- **Against its own whole-utterance rendition** ([scorer v2](benchmarks/VC_BENCHMARKS.md#t5-scorer-v2), 18 clips, CUDA, Low latency) the stream misses every limit but clicks: identity delta -0.055 (limit -0.02), fed fine pitch error 22.9 cents (15), fed gross pitch error 1.42 % (0.3 %), fed V/UV error 5.1 % (3 %), envelope MCD 0.60 dB over its floor (limit 0.6, at the limit). DNSMOS 2.78 against 3.08 for the whole-utterance rendition.
- **Latency.** As the app starts Low latency on a GPU: 114.7 ms engine-added p99 (132.6 ms at the reference configuration; 96.0 ms with the opt-in 10 ms device block). Of the 132.6 ms, the model call and compute are 14 ms (derived from the [T1 decomposition](benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults)); the rest is buffers and holds.
- **Compute.** About 0.34 s of GPU time per second of audio at 20 ms blocks (6.9 ms per block, recomputing the context each time), against 0.06-0.18 for RVC WebUI at its longer blocks. A new install runs the CPU Streamer: 265-278 ms median end to end on the 8-core rig (p99 348 ms under a game stand-in), and with 6 or fewer physical cores a longer automatic block ([the CPU Streamer on 4, 5 and 6 cores](benchmarks/VC_BENCHMARKS.md#the-cpu-streamer-on-4-5-and-6-physical-cores)).

## The leading hypothesis, and what it does not explain

Presets (lookahead, chunk size, context, crossfade) slide along one latency-quality frontier, and the measurements point to one cause: **the newest frames of every window have no future.** Twenty milliseconds of real lookahead recovers roughly 85 % of the identity gap (+0.046 on 8 clips against -0.054 on 18) and 0.18 dB of envelope MCD; without it, overlapping ticks agree only to a normalized cross-correlation of 0.77 (0.83 at 10 ms, 0.88 at 20 ms, 0.92 at 40 ms; [context scaling and joins](benchmarks/VC_BENCHMARKS.md#engine-core-context-scaling-and-joins-p3-c1)); onsets after a pause come out softer than in any ceiling. None of the models in the chain was trained for a window with a hard right edge. ContentVec and the F0 trackers learned on whole clips. In RVC training only the decoder sees short random slices (32 frames); `enc_p` (global attention) and the flow (24 frames of right receptive field in the template, 240 ms) see whole slices. To move the frontier, those parts have to be trained for chunked use. That is a hypothesis: [E0](#e0-measure-first-bounds-no-training-probes-and-a-harness-that-can-see-the-effects-cpu-only) tests it before anything expensive starts.

Stateful streaming of the existing generator (caching activations of the flow and the HiFi-GAN) is not the way; see [what is not worth doing](#what-is-not-worth-doing). State pays only for models trained causal, which the content and F0 students would be.

What the hypothesis does not explain, and the plan treats separately:

- **Pitch.** Ceiling A's own streaming tracker (700 ms of context, 80 ms of lookahead) still reads 20.5 cents, so about 15 of the 22.9 cents is the streaming tracker path, not the right edge. The output analyzer reads 33.6 cents on the converted audio against 22.9 fed ([ADR 0009](adr/0009-vc-scorer-v2.md), pitch table), so the decoder (or the analyzer) adds error that nobody has split out. And oracle F0 moves envelope MCD by at most 0.04 dB and cosine by at most 0.01 ([ablation tables](adr/0009-vc-scorer-v2.md)), so pitch accuracy is worth improving for its own sake (intonation, the pitch rows), not as a lever for DNSMOS or similarity.
- **The CPU tier.** FCPE's gain over SwiftF0 (+0.34 DNSMOS, [FCPE against SwiftF0](benchmarks/VC_BENCHMARKS.md#fcpe-against-swiftf0-for-the-streamer-button)) is already banked: the Streamer ships FCPE, and its audio metrics are on par with the GPU path (18 clips: identity -0.052 against -0.055, envelope MCD 2.05 against 2.03 dB, DNSMOS 2.74 against 2.78). What is left there is octave slips and voicing (gross 3.8 %, V/UV 10.3 %) and, above all, compute and delay.
- **MCD against RVC's render.** Even RVC's own streamed output sits 0.8 dB over its seed-to-seed floor against its own offline render (1.97 against 1.18 dB at 200 ms), so part of any streamed-versus-offline MCD exists for every engine.
- **Similarity against RVC.** The `compare` tool scores similarity and MCD against the centroid of RVC's own offline renders (leave one clip out). That favours RVC by construction; how much is unmeasured.

## Ranked items

| Rank | Item | Moves | Effort | Needs |
|---|---|---|---|---|
| 1 | [E0](#e0-measure-first-bounds-no-training-probes-and-a-harness-that-can-see-the-effects-cpu-only) Measure first | decides the rest | M | CPU only |
| 2 | [E8](#e8-device-path-live-loopback-first-then-the-10-ms-device-block-as-default-small-shared-periods-only-if-needed) Device path | latency | M | a real audio device |
| 3 | [E6](#e6-retrieval-measure-it-on-a-voice-with-an-index-and-real-target-audio-time-it-smooth-it) Retrieval | similarity | M | trains one small voice once |
| 4 | [E1a](#e1a-cpu-tier-pitch-run-the-existing-swiftf0-and-fcpe-fusion-through-the-corpus-then-a-small-learned-voicing-head) CPU tier pitch | pitch | M | CPU, a small head (hours) |
| 5 | [E3a](#e3a-right-context-robust-enc_p-and-flow-by-latent-space-self-distillation-pilot-with-a-transfer-test) Generator edge pilot | quality, similarity, latency | M | one GPU training run |
| 6 | [E2](#e2-chunk-trained-contentvec-student-stage-1-same-shapes-small-pilot) Chunk-trained ContentVec, pilot | quality, similarity | L | GPU training, hours to days |
| 7 | [E10](#e10-import-applio-refinegan-and-mrf-voices-and-voices-without-pitch-guidance) Applio voices | coverage | L | none |
| 8 | [E7](#e7-honest-gpu-cost-first-then-fuse-only-where-it-pays) GPU cost readout | compute (visibility) | S | none |
| 9 | [E4](#e4-shorter-crossfade-once-ticks-agree) Shorter crossfade | latency | S | none |
| 10 | [E1b](#e1b-stateful-streaming-f0-student-gated) Stateful F0 student | pitch, VRAM | L | GPU training, hours |
| 11 | [E5](#e5-cpu-decoder-int8-sensitivity-probe-no-training) CPU int8 decoder probe | CPU compute | L | none (post-training quantization) |
| 12 | [E3b](#e3b-ship-the-edge-fix-in-the-lightest-form-that-works) Ship the edge fix | quality, similarity | XL | GPU training |
| 13 | [E2b](#e2b-stateful-content-encoder-stage-2-separate-decision) Stateful content encoder | CPU compute | XL | GPU training |

The order is "what can be done now and cheaply first, then the quality items as soon as E0 releases them". E1b, E3b, E4's wider adoption and E2b are decided only after E3a and E2 report. If E0 says the generator is where the gap is, E3a moves to the front.

### E0. Measure first: bounds, no-training probes and a harness that can see the effects (CPU only)

**What.** Two to three days, no new model, CPU execution provider only.
1. *Reference bias, one hour.* Score Sigil's own whole-utterance rendition against the centroid of RVC's renders and against the centroid of Sigil's renders. The difference is the part of the similarity deficit versus RVC that comes from the reference, not the engine.
2. *Edge probe on speech frames only.* Error of the newest k frames against full-context ContentVec features, for right context R = 0/10/20/40/80 ms and left context 0.3/0.44/1/2 s, with interior frames reported separately, fp32 ONNX, 5-10 clips. Then the same grid through the ceiling harness, so the answer is an envelope-MCD number and not only a feature cosine.
3. *Generator oracle.* Give the true future to `enc_p` alone (the `enc` and `flowdec` graphs are separate: run `enc` on a long window and hand its `m_p`, `logs_p` to the stream's `flowdec`, a probe hook like the existing oracle-content and oracle-F0 ablations), then to the flow alone and the decoder alone in PyTorch (`scripts/vc-templates/margin_probe.py` and `rvc_arch.split` already reproduce the flowdec arithmetic in float64 with the engine's phase carry; the noise tensors are inputs, so the oracle can feed the engine's anchored noise). Score the identity delta (similarity to the target set), because the old oracle tables report cosine to the ceiling, not the identity row.
4. *Controls that need no training.* (a) Pseudo-future padding: extend `m_p`, `logs_p` and the pitch by K frames of the last value before the flow, then drop those frames' audio. (b) Silence compaction in the `enc` window, for the onset problem. (c) A 15 ms crossfade together with the existing 10 ms lookahead on the 18 clips (the premise of E4). (d) Content normalization at 2 s of context (1 s is measured).
5. *Decoder pitch adherence.* The same fed F0, three noise seeds, output F0 against fed F0, plus EGG truth from the CMU ARCTIC recordings that carry an EGG channel (`scripts/vc-bench/egg_f0.py`). Pick one pitch-error definition (fed against input, output analyzer, or `compare`'s fine error) and use it everywhere; three incompatible ones circulate.
6. *Harness, kept small.* The 18 scorer-v2 clips already exist. Add about six noisy-mic or game-bleed clips (existing clips mixed with game audio at 5-15 dB), CER through the existing `stt` option of the bench spec with a multilingual engine, and two more voices on other templates (v1, 32 kHz or 48 kHz) with real recordings of their target speaker and an index (small voices trained once on a public speech corpus with the RVC WebUI already on the dev PC; that training is the only GPU job here and can wait). Keep the clips used for any training disjoint from the clips used for scoring.

**Why.** Every big item below hangs on a number that does not exist yet. The ablations never ran through the set-similarity row, so the -0.054 identity delta is unattributed (oracle content and F0 move only the cosine to the ceiling, by at most 0.012 against the whole-utterance rendition). The `compare` reference is RVC's own renders; the Sigil-only control (two offline Sigil renditions differ by 0.002-0.003, [ADR 0009](adr/0009-vc-scorer-v2.md)) compares Sigil with Sigil, so it does not bound the bias against RVC. The only edge measurement ([P3-D README](../scripts/bench/specs/p3b/d/README.md)) mixes 1.2 s of inserted pause frames with speech and cannot separate left from right context: at a 0.44 s window the newest four frames score cosine 0.435 and all frames 0.494 (so most of the error is window-wide), while at 2 s the interior reaches 0.711 and the newest frames only 0.462 (which does point at the right edge). The CER scorer exists (`fidelity.rs`, first offset, when a spec names an `stt` engine) but the corpus runs do not use it; the one CER on file is a single clip on Whisper tiny.en. Statistically, 5 clips cannot resolve a 0.05 DNSMOS difference, and a 6-trial blind A/B needs 6 of 6 (two-sided p about 0.03) to say anything.

**Expected effect.** No direct metric change. It decides, with rules fixed in advance:
- if pseudo-future padding or silence compaction recover a third of the lookahead-20 effect (+0.015 similarity, -0.06 dB envelope MCD), they ship as small levers and E3a's value is re-estimated before any training;
- if the true-future oracle on `enc_p`, flow and decoder together recovers under 40 % of the identity gap, E3a and E3b are dropped;
- if, on speech frames, the newest-frame error at R = 0 is not at least 0.03 cosine worse than the interior at the same left context, or the downstream envelope MCD barely moves, E2 is dropped;
- if the decoder's pitch adherence error is not below the tracker's, E1b is dropped;
- if the reference bound is 0.02 or more, the similarity gap to RVC is mostly the reference, and E6 and E3 similarity targets are re-set.
Basis: measurement.

**RVC voices.** Works with existing RVC voices as-is.

**GPU or CPU.** CPU only (engine on the CPU provider, PyTorch float64 and float32). The PyTorch flow-versus-decoder split adds one to two days and is done only if the in-engine `enc_p` oracle leaves the question open.

**Effort.** M (two to three days).

**Risk.** Low. The oracle must use the same anchored noise and window geometry as the engine, or the split is noise. Voices trained from a public corpus have clean target recordings but are not the community's voices.

**Lean test.** It is the test: one scripted run on the 5 matched clips and the 18 scorer-v2 clips, reported as one short table with the rules above applied.

**Depends on.** None.

### E8. Device path: live loopback first, then the 10 ms device block as default, small shared periods only if needed

**What.** Step 1 (an hour, manual): run the existing `e2e_loopback` meter through the virtual cable VRChat reads (the dev PC's cable endpoints are disabled today and must be enabled), with the shipped 512-frame block and with `device_block = model_divisor` (10 ms), on a real microphone and headphones; count underruns and read the loopback latency. Step 2: if clean, make the 10 ms block the default for the GPU presets. Step 3, only if the loopback shows the device buffers are still a large share: a Windows IAudioClient3 path (`GetSharedModeEnginePeriod`, `InitializeSharedAudioStream`) for capture and playback next to cpal 0.15, falling back to cpal when the device or the cable driver refuses small periods. Where the freed milliseconds go is a decision, not a side effect: keep them as lower latency, or spend 10 ms on a Low-latency lookahead, which measured +0.032 set similarity [+0.013, +0.055], +0.053 DNSMOS, -3.2 cents and -0.13 dB envelope MCD on 8 clips ([method notes](benchmarks/VC_BENCHMARKS.md#method-notes)).

**Why.** About 89 % of the engine-added latency is structural buffering and 11 % is compute (derived: model 10.4 ms plus compute 3.7 ms of 132.6 ms; the output cushion and callback period alone are 50.6 ms). The 10 ms device block is the one measured free win, but it has never run on a live device ([decision sessions](benchmarks/VC_BENCHMARKS.md#decision-sessions), [still to run](benchmarks/VC_BENCHMARKS.md#still-to-run-manual)). A search of `src-tauri` finds no IAudioClient3 or exclusive-mode code. This item lowers the latency the user feels; it does not move the quality-versus-algorithmic-latency curve.

**Expected effect.** Latency: CUDA -16 % (114.7 to 96.0 ms, interval -20.3 to -7.3 ms), DirectML -9.0 % (124.7 to 113.5 ms), H1 simulation on CUDA -8.4 %; the CPU Streamer gave no information in that session (it ran the one-thread preset, which was not real time). Measured in the harness, not on a device. Small shared periods: another 10-14 ms is an estimate from the 10 ms default period against the roughly 3 ms minimum some drivers offer; it is unverified here and depends on the cable driver. Quality, similarity, pitch: none by themselves; with a 10 ms lookahead, the gains in "What". Compute: slightly more callbacks.

**RVC voices.** Works with existing RVC voices as-is.

**GPU or CPU.** Both. The 10 ms block is measured on the GPU presets only; the period work would help every rung.

**Effort.** M.

**Risk.** Low to medium. Some cable drivers and Bluetooth devices reject small periods or glitch, so the cpal fallback is required. Needs a real device.

**Lean test.** Step 1 is the test: one 10-minute live loopback run on the dev PC, current default against 10 ms, with the 70 % game stand-in running for part of it. Adopt only with no more underruns than the default. IAudioClient3 is written only if step 1 shows the device buffers matter.

**Depends on.** None.

### E6. Retrieval: measure it on a voice with an index and real target audio, time it, smooth it

**What.** (a) Take one voice that has both its `.index` and real recordings of its target speaker held out from training (E0 builds such voices; RVC training also writes the index). Sweep index_rate {0, .25, .5, .75, 1} at Low latency, Quality and Chunk 200, on Sigil and on RVC WebUI with the same index. Score CAM++ against the real recordings (not against any render), plus DNSMOS and CER. (b) Make the kNN stage report its real time: `tick.rs` hard-codes `retrieval_ms: 0.0` in the probe record; compare the CPU IVF with the GPU kNN. (c) If the gain is larger at short chunks, turn index_rate on by default for voices with an index at Low latency, and smooth the per-frame blend over 2-3 content frames. (d) Only if (a)-(c) show a real gain: a LinearVC-style linear map from mic features to the voice's feature distribution, fitted per user, mic and voice from a short enrolment.

**Why.** Quality and Studio ship index_rate 0.5; Streamer and Low latency ship 0, and the kNN block runs only when index_rate is above 0. No Sigil number for retrieval exists: the only benchmark voice has no index, so Quality's 0.5 did nothing in the [matched comparison](benchmarks/VC_BENCHMARKS.md#matched-comparison-one-run) (a control with index 0 scored the same, DNSMOS 2.96 against 2.95). It is the one similarity lever inside the current architecture that nobody has measured, and Sigil already has the stronger search (exact GPU kNN with fp32 re-rank, built but off). The reference matters: retrieval deliberately moves output toward the target's training features, so a reference built from index-free renders is biased against it; only real target recordings score it fairly. Part (d) rests on an abstract (WavLM features, no numbers) and its storage comparison against the index is not like for like (the map needs an enrolment), so it is a conditional spike.

**Expected effect.** Similarity on indexed voices at Low latency: +0.01 to +0.04 CAM++ against real recordings (estimate; no Sigil number exists, and ContentVec is already speaker-perturbed, so expect less than the kNN-VC paper's WavLM gains). Hypothesis: larger at 60-100 ms than at 250 ms, because retrieved rows are offline-quality features. Pitch: none. Compute: the CPU IVF cost on the tick thread becomes visible; (b) turns an unknown into a number.

**RVC voices.** Works with existing RVC voices as-is (uses the voice's own `.index`; community voices often ship one, not counted here). A voice without an index gets nothing.

**GPU or CPU.** Both. GPU kNN is built but off; the CPU path uses IVF.

**Effort.** M.

**Risk.** Low. The index may help little at short chunks; then it is documented and dropped. Smoothing adds a few milliseconds of blend lag.

**Lean test.** One indexed voice with held-out target audio, the 18 clips if the E0 harness exists (else 5), 5 index rates, 3 presets, one run; then a blind A/B of index 0 against the best rate with the prepared kit. The linear-map spike is a separate half-day offline script, run only if the first result is positive.

**Depends on.** E0.

### E1a. CPU tier pitch: run the existing SwiftF0 and FCPE fusion through the corpus, then a small learned voicing head

**What.** (1) Run the CPU fusion that already exists (SwiftF0 pitch with FCPE assisting the voicing, `FusionConfig::cpu` in `f0_fusion.rs`) through the 18-clip scorer-v2 corpus next to FCPE and SwiftF0, paired per clip. It has only been run on the synthetic A/B (CPU only, about a minute per clip on 6 threads). (2) If the fed V/UV error stays above about 5 % or the slip clips (jfk, ko, en) keep their octave errors: a 50-100k-parameter head over SwiftF0's per-frame outputs (salience, confidence, a few frames of context) that predicts voicing and the octave, with whole-clip RMVPE as teacher, on speech with noise, game bleed and creak, plus synthetic octave traps with exact truth. It runs on the CPU in well under a millisecond.

**Why.** New installs run the CPU Streamer, and its F0 is FCPE: fed fine error 35.2 cents, gross 3.8 %, V/UV 10.3 % (limits 15 cents, 0.3 %, 3 %). SwiftF0 has the opposite profile: 13.2 cents and 0.12 % gross, but it voices only part of the frames (V/UV 44.9 %). FCPE won the audio comparison (+0.34 DNSMOS [+0.215, +0.479], higher on 17 of 18 clips) and stays the default; that gain is banked. What remains is its octave slips, concentrated in a few clips (more than 20 % off on 19.6 % of jfk's fed frames, 12.1 % of ko's, 9.0 % of en's), and the synthetic streaming A/B ([pitch bank](benchmarks/VC_BENCHMARKS.md#pitch-bank-retrieval-and-post-stage-p3-d)) shows the fusion cutting FCPE's gross error from 12.7 % to 2.0 % at similar voicing error. The 18-clip corpus has never seen the fusion ([FCPE against SwiftF0](benchmarks/VC_BENCHMARKS.md#fcpe-against-swiftf0-for-the-streamer-button)).

**Expected effect.** CPU tier fed pitch: fine 35 to 15-20 cents, gross 3.8 % to 1 % or less, V/UV 10.3 % to 4-6 % (estimates; basis: SwiftF0's accuracy on the frames it voices and the fusion's synthetic result; the head's voicing accuracy is unknown). DNSMOS, similarity, MCD: about 0 on the corpus mean (oracle F0 recovers 0.03-0.04 dB on the CPU tier), but the slip clips should improve. Compute: about unchanged; the F0 stage costs 2.1 ms with SwiftF0 and 4.4-5.5 ms with FCPE at 6 threads (12 ms on one), so SwiftF0 plus a head in place of FCPE would save 2-3 ms of a 100-150 ms call (derived from the decision-session stage times).

**RVC voices.** Works with existing RVC voices as-is (F0 is a separate input; the target is the RMVPE-style pitch the voices were trained on).

**GPU or CPU.** CPU tier. GPU rungs unchanged.

**Effort.** M (the corpus run is small; the head is hours of training).

**Risk.** Low to medium. Every contour change so far cost 0.014-0.021 cosine on some clips, so cosine and DNSMOS are gates, not just cents. The head inherits SwiftF0's blind spots (singing above D6, creak) unless trained on them.

**Lean test.** Step 1: the corpus run, about 20 minutes of CPU, with the usual paired table. Step 2: train the head in hours; gates are the fed rows on the 18 clips and the 27-clip synthetic exact-truth corpus, cosine not down more than 0.01, DNSMOS not down, and a blind listen of FCPE against the head on the slip clips before it replaces anything.

**Depends on.** E0.

### E3a. Right-context-robust `enc_p` and flow by latent-space self-distillation: pilot, with a transfer test

**What.** In PyTorch on the existing `rvc_arch` port (`scripts/vc-templates/rvc_arch`, which has the training forward path), take the default voice. Teacher: the same weights on a long context with the true future, giving the latent z the flow emits (anchored noise, flow reversed). Student: `enc_p` and the flow only (about 10M parameters, full fine-tune or LoRA, decoder frozen), run on the exact window the engine builds, with the right edge truncated and L = 0/10/20 ms of lookahead. Loss: L2 on z at the frames the engine emits first. Add a multi-resolution STFT term through the frozen decoder only if the latent loss alone does not move the metrics (a 40 kHz HiFi-GAN in the loop is heavy). Mix full-context windows in so behavior does not drift. Data: synthetic, ContentVec features of public speech including onsets after pauses; no original training set and no discriminator. Train on the stock content features; E2 would change the input distribution, so the pilot is re-run on E2's features before anything ships. Then (i) the same recipe, unchanged, on two other voices; (ii) a transfer test: apply voice A's weight delta to voice B and to the pretrained base (RVC voices are fine-tuned from a few pretrained bases), which is arithmetic only and takes an afternoon.

**Why.** Lookahead recovers roughly 85 % of the -0.054 identity delta (20 ms: +0.046 set similarity on 8 clips, [+0.023, +0.078]; the clip sets differ) while oracle content and F0 recover cosine by at most 0.012, and the decoder is the only part trained on random 32-frame slices (`models.py`, `rand_slice_segments`). `enc_p` and the flow (24 frames of right receptive field) train on whole slices and never see a truncated edge. Overlapping ticks agree only to NCC 0.77 at 0 lookahead, which is why the crossfade is 20 ms ([P3-C1](benchmarks/VC_BENCHMARKS.md#engine-core-context-scaling-and-joins-p3-c1)). Onsets after a pause are muted by `enc_p` (stream -4.9 dB against -2.1 dB for the whole-utterance rendition; [the offline ceiling](benchmarks/VC_BENCHMARKS.md#the-offline-ceiling)). The oracle tables cannot say how much of this is `enc_p`, flow or decoder; E0 does.

**Expected effect.** Go or no-go gate, set on the 18 scorer-v2 clips and not on the 5 matched clips (a five-clip lookahead run was neutral, cosine -0.020 to +0.020, [P3 final gate](benchmarks/VC_BENCHMARKS.md#p3-final-gate-pitch-track-fixes)): at 0 lookahead, reach at least half of what 20 ms of lookahead buys: identity delta better by 0.023 with a confidence interval above 0, envelope MCD -0.09 dB, SOLA NCC at least 0.85, CER within +1 point, DNSMOS not lower. Basis: estimate anchored to the measured lookahead-20 effect (+0.046 similarity, -0.18 dB, +0.136 DNSMOS [+0.007, +0.27], 8 clips). Eighteen clips resolve a 0.023 effect (paired SD about 0.033 gives a standard error of 0.008, derived from that interval). Shipped-version estimate: see E3b.

**RVC voices.** Works with the voice unchanged until converted; the pilot produces patched weights for one voice at a time.

**GPU or CPU.** Training on the GPU, minutes to an hour per voice (estimate). Inference graphs and cost unchanged. A heavy job: run it alone while the GPU is free.

**Effort.** M.

**Risk.** Medium to high. It may shift timbre (the flow prior is delicate) or lower similarity; the full-context mix-in limits this. The student also absorbs the content encoder's edge damage, so the result is entangled with E2. Pseudo-future padding (E0) may get a third of the effect for free, which would shrink the case for training.

**Lean test.** One training run on the default voice; then the 18 clips (3 offsets each) scored with set similarity, envelope MCD to the ceiling, DNSMOS, CER and NCC, repeated with oracle (true-future) content to bound the E2 interaction; then the recipe on the two other voices with held-out clips; then a blind A/B (stock against patched, Low latency) with the prepared kit.

**Depends on.** E0.

### E2. Chunk-trained ContentVec student, stage 1: same shapes, small pilot

**What.** Initialize from ContentVec and fine-tune it on the exact windows the engine feeds (0.3-1 s of left context, right edge truncated, random lookahead of 0-4 frames), with the loss on the newest frames against the frozen full-context teacher (layer 12, 768-d; a layer-9 plus `final_proj` head for v1 voices). The graph keeps its input and output shapes, so the fixed-shape CUDA-graph plans, the fp16 and int8 variants, kNN, the parity gates and every RVC voice work unchanged. Add running first-layer normalization statistics as an input, so the first GroupNorm no longer depends on silence. Optional stage 1b: also predict the next N frames, so `enc_p`, flow and decoder see a plausible future instead of a hard edge (compare with E0's pseudo-future padding to see whether learning beats holding the last value). Pilot on 50-100 h of public speech covering the corpus languages. Stage 2 is [E2b](#e2b-stateful-content-encoder-stage-2-separate-decision).

**Why.** Content features are about half of the envelope-MCD gap: oracle content recovers 0.31-0.32 dB of the 0.60-0.64 dB over the floor, oracle F0 about 0 ([ADR 0009](adr/0009-vc-scorer-v2.md), ablation tables). Raw ContentVec on a truncated window is out of distribution: the first GroupNorm normalizes over the window, long contexts mute onsets by 15-35 dB, and the newest frames' cosine to whole-utterance features is 0.435 at 0.44 s and 0.462 at 2 s. Whether that is mostly the right edge or the whole window is the open question (E0, item 2). Related work supports the recipe (causal-WavLM distillation in FocalCodec-Stream, Emformer students in Conan), but no published student emits ContentVec-space features, so it has to be trained here.

**Expected effect.** Capped at 30-50 % of what 20 ms of real future buys, because a student cannot see the future (the real-future effect: -0.18 dB envelope MCD, +0.046 similarity, +0.136 DNSMOS, 8 clips). That gives envelope MCD -0.05 to -0.09 dB, DNSMOS +0.04 to +0.07 at 60-100 ms, similarity +0.01 to +0.02 (estimates). Pitch: about 0. GPU compute: about 0 (launch-bound). CPU: the content stage is 19-21 % of a call and the int8 encoder is already cheap, so count on a few percent at most. Latency: could let Quality shed part of its 20 ms lookahead if stage 1b works (estimate).

**RVC voices.** Works with existing RVC voices as-is (same feature space, same index, same generator). v1 voices need the second head; ja, ko, zh and SPIN encoders keep the stock model until there is demand.

**GPU or CPU.** Both. Training is a one-time job on the dev GPU: hours for the pilot, days for a full run (estimate for a 95M-parameter model).

**Effort.** L.

**Risk.** Medium. Regressing the conditional mean blurs edge features, and DNSMOS and CAM++ can reward blur, so CER is a gate. ContentVec is English-centric and Sigil serves ten languages. The fp16 and int8 parity gates ([ADR 0011](adr/0011-vc-content-encoder-precision.md)) must be re-run. The real gain may sit near the low end.

**Lean test.** Pilot gate, after hours of training on about 100 h: on speech frames only, the newest-frame error at R = 0 and 20 ms must drop at least 30 % against raw ContentVec, the downstream envelope MCD through the ceiling harness must improve, and CER must be no worse; otherwise stop. Then the student's features on the 18 clips (MCD, DNSMOS, set similarity, CER) and a blind A/B at Low latency.

**Depends on.** E0.

### E10. Import Applio RefineGAN and MRF voices, and voices without pitch guidance

**What.** Lift the importer refusal (`import/layout.rs`, layout version 2) for the 15 templates that ship but are refused: Applio's RefineGAN and MRF HiFi-GAN, and voices trained without pitch guidance. Add per-template plan geometry (RefineGAN decodes a 113-117 frame window for 4 returned frames, against 39 for HiFi-GAN), keep RefineGAN in fp32, and add bench rows. GPU rungs first.

**Why.** Only 7 of 22 templates run. A voice trained in Applio with the RefineGAN or MRF vocoder cannot be imported at all, although the templates pass parity on donor builds and on Applio's 32 kHz RefineGAN pretrain ([ADR 0004](adr/0004-rvc-templates.md)); no community-trained voice has gone through that path. This is a compatibility gap, not a metric gap, but a refused voice has no quality, and existing voices must keep working. It also gives a same-engine vocoder comparison for E0's pitch adherence question.

**Expected effect.** Coverage: voices refused today become usable. The four metrics for current voices: none. RefineGAN costs about 2.1 times the flowdec time (14.5 against 6.9 ms on DirectML, 83-105 against 46 ms on the CPU; `scripts/vc-templates/reports/template_timing.json`), so the CPU Streamer will be heavy for it. Pitch: possibly better with MRF's 9-harmonic source (unverified).

**RVC voices.** Works with existing RVC voices as-is; it adds voice classes that are refused today, with no conversion.

**GPU or CPU.** GPU first. RefineGAN on the CPU Streamer will be heavy.

**Effort.** L.

**Risk.** Medium. Window and plan code assumptions; CPU cost; the share of users with such voices is unknown.

**Lean test.** Import one real Applio RefineGAN voice and one MRF voice, run one stream pass each, check parity and underruns, and listen briefly. Estimate the plan changes only after that.

**Depends on.** None.

### E7. Honest GPU cost first, then fuse only where it pays

**What.** (a) Report the GPU busy fraction (NVML utilization, which `diagnostics/gpu_mem.rs` and the resources service already load, or an ETW GPU counter) next to the call time in diagnostics, so users see what Sigil costs their game; today the closest number is call wall time, which is mostly launch and sync waiting (about 1 ms of GPU math per call, [ADR 0010](adr/0010-sve-optimization-p3b.md)). Include a read-only GPU clock and P-state readout, since the clock swing explains most pass-to-pass variation. (b) Compose `enc` and `flowdec` into one graph on DirectML (the deferred follow-up in ADR 0010); on CUDA the gap between them is already 0.08 ms, so measure before building. (c) After E2 lands, compose the content graph with the projection and `enc`.

**Why.** The tick is overhead-bound: CUDA graphs already took the generator from 12 to 3.3 ms. After the device-resident chain, little is left to fuse on CUDA. Most of the tick is content (1.8-2.0 ms), the F0 branch (3.1-3.3 ms, overlapped with content) and flowdec (2.1-2.5 ms) ([integrated smoke](benchmarks/VC_BENCHMARKS.md#integrated-smoke-p3b-i0)). DirectML makes four syncs per call against three.

**Expected effect.** Compute: (a) none, it makes the cost visible; (b) tenths of a millisecond on CUDA (the gap is 0.08 ms), perhaps 1 ms on DirectML (estimate); (c) small. The bigger drops come from other items: with E1b the F0 branch leaves the GPU timeline and the tick shortens by F0 minus content, about 1.3 ms on CUDA and 2.0 ms on DirectML (derived from the stage p50s), which takes 0.34 GPU-s per audio-s at 20 ms blocks to about 0.28. Quality, similarity, pitch: none.

**RVC voices.** Works with existing RVC voices as-is (templates are Sigil-built from the .pth).

**GPU or CPU.** GPU only. The CPU path is MAC-bound.

**Effort.** S (part a); M for (b) and (c).

**Risk.** Low to medium. Merged graphs must keep the fixed-shape plan, the fp16 rules (`enc` stays fp32) and the parity gates. NVML utilization is a coarse measure. Independent reports that TensorRT-RTX is slower than TensorRT 11 are a reason not to switch backends in this item.

**Lean test.** (a) Compare the readout with a game stand-in run. (b) Build the merged graph offline, run `gpu_lever_bench` and the 70 % load row once on DirectML, compare per-call p50 and p95 and underruns.

**Depends on.** None.

### E4. Shorter crossfade once ticks agree

**What.** Let the crossfade follow measured tick-to-tick consistency instead of a fixed 20 ms: a voice (or a stack: lookahead, E3a, E2) whose overlapping ticks reach NCC of at least 0.88 at the chosen join runs a 10-15 ms crossfade. The premise is tested before any training (E0, item 4c).

**Why.** The 10 ms crossfade fails because overlapping ticks decode the same audio differently at the window edge (NCC 0.77), not because blending is poor: it cost cosine -0.040 and DNSMOS -0.162, and 15 ms cost -0.018 and -0.103 ([method notes](benchmarks/VC_BENCHMARKS.md#method-notes)). More future raises NCC: 0.83 at 10 ms, 0.88 at 20 ms, 0.92 at 40 ms. E3a and E2 are meant to supply that consistency without latency.

**Expected effect.** Algorithmic latency: -4 to -9 ms of the 60 ms Low-latency floor (measured: the 10 ms crossfade removed 8.7 ms, [what closes the gaps](benchmarks/VC_BENCHMARKS.md#what-closes-the-gaps); 15 ms would remove about half). Quality and similarity: held at today's level (the gate), not improved. Compute: none. If the consistency comes from 10 ms of lookahead, that lookahead spends the saving (net +1 to +6 ms), so the test is about quality, not a latency win.

**RVC voices.** Works with existing RVC voices as-is for the engine change; only a stack that passes the NCC check uses the shorter crossfade.

**GPU or CPU.** Both.

**Effort.** S.

**Risk.** Low to medium. Join quality is grid-sensitive (a one-sample tail change moved cosine by 0.014); needs the scorer-v2 gate.

**Lean test.** On the 18 clips with scorer v2: adopt a shorter crossfade only if cosine drops under 0.01, DNSMOS under 0.05 and clicks stay 0. Check NCC per stack, and per voice only if voices turn out to differ.

**Depends on.** E0 (the premise test); wider adoption needs E3a or E2.

### E1b. Stateful streaming F0 student (gated)

**What.** A small cached-convolution pitch network (PESTO or SwiftF0 class, 0.1-1M parameters), trained once and shared by all voices. Teacher: whole-clip RMVPE salience and voicing, on speech, singing and augmented audio (noise, music and game bleed, reverb, creak, octave-hard synthetic cases). Student: causal with 0-2 frames of lookahead and carried state, trained with random right-context truncation at 0/10/20/40 ms; 360-bin salience with KL loss plus a voicing BCE, decoded like RMVPE. Exported as stateless ONNX with state in and out; always runs on the CPU, even on GPU rungs. Start only if E0 shows the decoder's pitch adherence error is below the tracker's, and E1a leaves a real gap on the GPU path.

**Why.** The F0 stage is the largest p95 share under GPU load (52-54 %, overlapped with content at 33-40 %) and holds 376 MB of DirectML's 853 MB ([method notes](benchmarks/VC_BENCHMARKS.md#method-notes)). Streaming F0 loses accuracy at the newest frames: fed fine error 22.9 cents against a 5.7-cent floor, V/UV 5.1 % against 3 %, gross 1.42 % against 0.3 %. But ceiling A's own streaming tracker (700 ms context, 80 ms lookahead) reads 20.5 cents, so about 15 cents is the streaming path, not the short right context, and the best geometry hack reached only 15.5-17.8 cents. Voicing onsets cannot be predicted from the past, so a causal student keeps some onset error. Heuristic fixes are exhausted (filled leads, priming and continuity, cross-tick decoding, fusion rules). The output pitch error is 33.6 cents through the output analyzer against 22.9 fed, so the decoder may set a floor that the tracker cannot lower (E0, item 5).

**Expected effect.** Fed fine pitch on the GPU path: 22.9 to 15-20 cents (estimate, anchored to the 17.8 of the best geometry hack and the 20.5 of ceiling A with 80 ms of future; floor 5.7). Gross 1.4 % to under 0.5 %, V/UV 5.1 % to 3-4 % (estimates; lookahead-20 reached 1.8-3.5 %). Quality and similarity on the GPU path: about 0 (oracle F0 recovers at most 0.04 dB and 0.01 cosine). Compute: RMVPE leaves the GPU, about 376 MB less VRAM on DirectML, and the F0 branch leaves the GPU timeline (tick shorter by about 1.3-2.0 ms, derived in E7); CPU cost about 1-2 ms (estimate).

**RVC voices.** Works with existing RVC voices as-is.

**GPU or CPU.** Both: the student always runs on the CPU, which helps a game-shared GPU.

**Effort.** L.

**Risk.** Medium. Truth is whole-clip RMVPE, so a student looks good by construction; judge it on EGG and synthetic exact truth. Earlier F0 changes that cut cents also cost 0.014-0.021 cosine on some clips, so cosine and DNSMOS are gates. Singing above D6 and creak need training coverage.

**Lean test.** First model (hours of training): the 18-clip fed-F0 harness (CPU) against RMVPE streaming, the 27-clip synthetic exact-truth corpus and EGG truth. Then one end-to-end run on the 18 clips: cosine not down more than 0.01 and DNSMOS not down. A blind listen before it replaces RMVPE anywhere.

**Depends on.** E0, E1a.

### E5. CPU decoder: int8 sensitivity probe (no training)

**What.** A post-training static int8 quantization probe of the HiFi-GAN generator (`flowdec`) on one voice, CPU only, on VNNI CPUs (AVX512-VNNI or AVX-VNNI, the same gate as the int8 content encoder): per-channel weights, calibrated activations, mixed precision where it hurts (first and last convolutions and resblock outputs in fp32). Measure the audio change first (DNSMOS, set similarity, SNR against the fp32 render), then the CPU call at 1 and 6 threads. This replaces the earlier idea of distilling a slimmer decoder per voice, which would have needed CPU-only users to train a GAN.

**Why.** On the CPU, convolutions are 91 % of flowdec time and already run at 72-75 % of fp32 FMA peak (57.6 GMAC/s per core, [P3B-G README](../scripts/bench/specs/p3b/g/README.md)), flowdec is 71-77 % of the call, so only fewer or cheaper MACs help. The Streamer needs 120, 160 or 200 ms blocks by core count (265-278 ms median end to end on 8 cores, 348 ms p99 under a game stand-in), 4-core PCs are not real time even idle, and 6 cores drop 9.5 % of the input under a game ([the CPU Streamer on 4, 5 and 6 cores](benchmarks/VC_BENCHMARKS.md#the-cpu-streamer-on-4-5-and-6-physical-cores)). Exact state caching cannot fix this (see [what is not worth doing](#what-is-not-worth-doing)). Int8 is the route [ADR 0010](adr/0010-sve-optimization-p3b.md) deferred as the only one back to 1.5 cores. It changes the audio, which is why it is a probe with a quality gate.

**Expected effect.** CPU: decoder time x0.4 to x0.7 (estimate; int8 VNNI against fp32 FMA, no Sigil number for convolutions; the int8 content encoder gained 13-15 % of its stage), so the 6-thread call goes from 86 ms to about 55-70 ms (derived from the measured 61 ms flowdec share), which might let 6-8 core PCs run shorter blocks. Quality: unknown; HiFi-GAN is sensitive to quantization noise. Gate: DNSMOS within -0.1 and set similarity within -0.01 of fp32. AVX2-only CPUs: nothing. GPU: none (launch-bound).

**RVC voices.** Works with existing RVC voices as-is: a conversion at import, no per-voice training; unconverted voices run fp32.

**GPU or CPU.** CPU rung, VNNI CPUs only.

**Effort.** L (the probe alone is M).

**Risk.** High for quality, low for the probe itself.

**Lean test.** The probe is the test: one voice, 5 clips, SNR and scorer-v2 rows, timing at 1 and 6 threads. The ship path (export, calibration at import, parity gate) is built only if both gates pass.

**Depends on.** E0.

### E3b. Ship the edge fix in the lightest form that works

**What.** Only if E3a passes. In order of preference: (1) a pre-adapted template for the voice the app ships (the MIT default voice), nothing for the user to do; (2) a shared weight delta applied to community voices fine-tuned from the same pretrained base, if E3a's transfer test works; (3) an optional import step that trains on the user's PC in a headless worker (a GPU opt-in runtime pack built on `rvc_arch`), only when not streaming, writing a patched `enc`/`flowdec` overlay beside the original. Skipping any of this leaves the voice running exactly as today, and the re-export and parity gates are reused. Train on whatever features the engine feeds at that time (E2's if E2 ships).

**Why.** Same evidence as E3a. Without it, the generator-side right-context gap (identity -0.054, about 45 % of the MCD excess, onset muting, join NCC 0.77) can only be bought with lookahead at 1 ms per ms, and a per-voice step is the only place weights can change, because no .pth knows about a 0.42 s window with a truncated edge. Option (3) conflicts with the light, few-options release goals (a multi-GB PyTorch pack) and with CPU-only users: decoder forward plus backward is about 137 GMAC per audio second (derived from the template's MAC count), about 2.4 hours of one core per hour of audio per pass at the measured 57.6 GMAC/s, so it needs a GPU and cannot run while a game does.

**Expected effect.** At Low latency: similarity +0.02 to +0.04, DNSMOS +0.08 to +0.2, onset gap 2.8 dB to 1.5 dB or less, SOLA NCC 0.77 to 0.85 or more (all estimates, to be replaced by the E3a result; stacking with E2 is not additive and is measured, not assumed). Latency: enables E4 (-4 to -9 ms) and may let Quality drop part of its 20 ms lookahead. Pitch: none directly. Compute: none (same graphs).

**RVC voices.** (1) and (2) need no user action; (3) needs a one-time conversion per voice. Unconverted voices keep working unchanged.

**GPU or CPU.** Training on a GPU; inference graphs and cost unchanged.

**Effort.** XL.

**Risk.** High. Footprint, training time on weak PCs, a regression that is hard to catch without listening, and unknown transfer across voices. "No voice loses similarity" on two voices times five clips has no power; use the 18-clip set per voice and the blind kit.

**Lean test.** Per voice (three or more, on different templates): the 18 clips with set similarity, DNSMOS, CER and NCC, plus a blind A/B. Ship a voice only if it gains and no tested voice loses similarity.

**Depends on.** E3a (and E2 if it ships).

### E2b. Stateful content encoder, stage 2 (separate decision)

**What.** A causal-chunk ContentVec student with a KV cache and convolution ring buffers that processes only the new frames per tick and keeps seconds of left context.

**Why.** ContentVec is a transformer, so state is natural once the model is trained causal; stateful streaming pays only for models trained for it. On the CPU Streamer the content stage is 19-21 % of the call, and a 120 ms block has only 6 new frames, so the best case is about 4 times fewer content MACs, roughly 15 % of a call (derived), not the 10-20 times that holds at 20 ms blocks, which the CPU never runs. The GPU is launch-bound and gains nothing.

**Expected effect.** CPU call -10 to -15 % at best (derived). Quality: could allow seconds of left context if the student does not mute onsets (unknown). Everything else: none.

**RVC voices.** Works with existing RVC voices as-is (same feature space).

**GPU or CPU.** CPU gain only.

**Effort.** XL.

**Risk.** High. A new model class (chunked, KV-cached), trained from scratch or with a heavy architecture change, untested.

**Lean test.** Decide only after E2 stage 1. First a feasibility spike: does a causal-chunk student reach stage 1's edge error at equal left context?

**Depends on.** E2.

## Recommended first step

Do three things that need no GPU and no training, in parallel:

1. **E0**, two to three days of CPU-only measurement and a small harness upgrade.
2. **E8 step 1**, one hour on a real device: the live loopback through the cable VRChat reads.
3. **E1a step 1**, about 20 minutes of CPU: the existing SwiftF0 and FCPE fusion through the 18-clip corpus.

Together they decide E2, E3a, E1b, E4 and the second half of E8 before any XL work is committed. If E0 says the generator oracle recovers at least 40 % of the gap, start E3a at once: it is one voice and one training run. The E2 pilot follows. GPU training jobs run one at a time, never while a benchmark or the user's game needs the GPU.

## What is not worth doing

- **Exact activation caching of the existing flow and HiFi-GAN** (stateful convolution and WN state). The decoder's right receptive field is 11 frames (`dec_receptive_right_frames` in `layout.json`: `conv_pre` 3 frames plus about 6 for the stage-0 resblocks, from kernel 11 and dilations 1/3/5 at 1 kHz) and the flow's is 24. Whatever the previous tick computed within that distance of its edge used a truncated future and is stale in the next tick. The left margin the engine recomputes (11 frames for the decoder, 24 for the flow) lies inside that stale zone, so almost nothing it recomputes is exact. Only what is older than the stale zone could be cached exactly: a few percent of decoder MACs at most (derived; stage 0 is about 18 % of them, `export_rvc.py`), the flow is only about 3 % of the MACs, and the GPU is launch-bound, so it gains nothing there. Caching stale activations approximately feeds the weights values they never saw. This is derived, not measured; E0 can confirm it with a per-stage receptive-field count (margin_probe style) in an hour.
- **Long left context and `content_norm` alone.** The newest frames do not improve (cosine 0.435 at 0.44 s, 0.462 at 2 s), the measured similarity gain is +0.01, and `content_norm` costs +0.04 dB MCD. Extra context at small chunks even lowered DNSMOS (2000 ms against 192 ms: -0.12 at Chunk 40, -0.15 at Chunk 60, neither resolved). Folding running statistics into E2 is the version that makes sense.
- **KV or convolution caching of stock ContentVec.** Bounded to 2-3 % of a CPU call and zero on the GPU; the useful version is E2b.
- **More sweeps of lookahead, crossfade, SOLA search, aligned tail, F0 fill, priming and extra context.** Measured near break-even; they only slide along the frontier.
- **Heuristic F0 smoothing, cross-tick decoding (the `f0_decode` lever lowered similarity) and new fusion rules** (the existing CPU fusion gets one corpus run in E1a), **F0 forecasting as a lookahead substitute** (oracle F0 recovers at most +0.01 cosine), **a learned crossfade** (joins are already clean), **a neural post-filter** (DNSMOS BAK is already 4.0; the limit is signal distortion).
- **Vocoder swaps for existing voices** (Vocos, BigVGAN-v2, Wavehax, MS-Wavehax, iSTFTNet, FARGAN): all take mel or acoustic features, not RVC's flow latent, so each voice would need retraining. A Wavehax-class model belongs to a possible new family, not this roadmap.
- **Zero-shot families as a similarity fix** (Seed-VC, Vevo, OpenVoice, CosyVoice-VC), **MeanVC2 for pitch** (it has no F0 conditioning), **per-voice LLVC or SynthVC students as the main path**, and **discrete-unit streaming students** (StreamVC, Conan, DualVC3, RT-VC, TVTSyn): none runs existing RVC voices, and most have no released weights.
- **GPU int8, fp8 and TF32 changes** (launch-bound, the math is about 1 ms), **CUDA or D3D priority tricks** (no effect across processes), **a keep-alive kernel** (rvc-realtime removed it because it starves games), and **making TensorRT-RTX the first rung** before a same-GPU measurement (vc-rs found it about 2x slower than classic TensorRT 11).
- **NetEQ-style time-scale playout to shrink the output cushion.** Stretching a standing queue of c ms by a factor s extends its dry-out time by only c times (s - 1). A 15 % stretch lets a 35 ms cushion shrink by about 5 ms, not 15-20; saving 15-20 ms would need a stretch of 1.75-2.3x, which is audible. It also cannot help the real failure, a low GPU clock where the call runs longer than the block (DirectML at a low clock: p95 21.6 ms against a 20 ms block), and the adaptive cushion already underran under load ([ADR 0010](adr/0010-sve-optimization-p3b.md)). The cause worth watching is the GPU clock swing, which E7 (a) makes visible.
- **Distilling a slimmer decoder per voice for CPU-only users.** It needs the users who cannot train a GAN to train one; E5 tests int8 instead.
- **Shipping a multi-GB training pack before E3a shows the recipe transfers.**
- **DirectML replacement** (WebGPU, OpenVINO plugin providers): a hedge, not a quality gain; defer until a non-NVIDIA test GPU is at hand.
- **Swapping ContentVec for another encoder** (SPIN and others): existing RVC voices are trained on ContentVec features.
- **Chasing the 0.03-0.05 similarity deficit versus RVC before E0 fixes the biased reference**, and **treating the 15-cent pitch limit as an audibility threshold** (it is an analyzer-floor rule).

## Changes after the first review

The first draft was reviewed against the repository and this version changed as follows.

- **Framing.** The edge story is now the leading hypothesis that E0 must confirm, with the gaps it does not explain listed separately (pitch, the CPU tier, MCD against RVC's render, the reference).
- **E0** gained the speech-only edge probe (with the 2 s counterpoint to the existing table), the enc_p-first oracle in the engine, the no-training controls (pseudo-future padding, silence compaction, 15 ms crossfade with 10 ms lookahead), CER, extra voices, noisy clips and pre-registered decision rules that match the order. "One day" became two to three.
- **E1** was split. E1a (the CPU tier, cheap, runs the existing fusion on the corpus first) comes early; the stateful student E1b is gated on E0's decoder-adherence result, its pitch estimate was lowered to the evidence (15-20 cents), and its compute claim now says the F0 branch overlaps content.
- **E2** is capped at 30-50 % of the lookahead-20 effect; its smoke gate is now a speech-only edge probe plus a downstream MCD and CER; stage 2 is a separate decision (E2b) with the CPU arithmetic corrected to about 4x fewer content MACs at 120 ms blocks.
- **E3a** is gated on the 18-clip set with CER, tested on three voices, bounded against oracle content, and its cheap no-training controls moved into E0. E3b, E4 and the old E5 no longer assume a training pack on the user's PC: E3b prefers a pre-adapted shipped voice or a shared delta and keeps the user-PC pack as the last resort, E4's premise is tested first, and E5 became a quantization probe.
- **E6** now scores against real target recordings and says Quality already ships index_rate 0.5; the LinearVC spike is conditional and its storage claim is gone.
- **E7** was split and corrected: the kNN skip at index 0 already exists, the duty arithmetic is fixed, and enc/flowdec fusion is worth tenths of a millisecond on CUDA.
- **E8** is live-loopback first, states the gain per execution provider and names where the freed milliseconds go. **E9** (time-scale playout) was removed (arithmetic above). **E10** moved up.
- **Points not adopted as written.**
  - The suggestion that the reference bias is "probably 0.01 or less": the Sigil-only control cannot bound a Sigil-versus-RVC difference, so it stays a one-hour measurement.
  - The suggestion that the CPU tier is tracker-limited by +0.34 DNSMOS: that gain is FCPE over SwiftF0 and already ships; E1a promises pitch rows, not DNSMOS.
  - The estimate that the PyTorch flow-versus-decoder split takes days: the float64 reproduction exists in `margin_probe.py`, so it is one to two days (the whole of E0 is still longer than the first draft said).
  - The "free control" of `content_norm` at 1-2 s: 1 s is already measured (cosine +0.009 and +0.012, MCD +0.04 dB, onsets still muted); only 2 s is new, and it is one row in E0.
  - The claim that scorer v2 has no CER: the code computes it when a spec names an `stt` engine; the corpus runs just do not. E0 turns it on.
  - A reviewer's figure that real lookahead moved similarity by -0.010 on the matched clips could not be reproduced from the benchmark document; the conclusion (a 5-clip gate has no power) stands and the gate moved to 18 clips.
  - A clock-aware rule that picks 40 ms blocks when the GPU clock is low: not adopted without evidence from a run with a game; only the read-only readout (E7 a) is in.

## Limits of the evidence

All GPU numbers come from one RTX 4090 and a Ryzen 7 7800X3D; the 3060 and 4060 records are still pending ([still to run](benchmarks/VC_BENCHMARKS.md#still-to-run-manual)). The Sigil-versus-RVC rows are 5 clips and one run, scorer-v2 rows are 18 clips with bootstrap intervals, and nobody has listened to the matched renditions. Every "estimate" above needs the lean test listed with it. Lean here means: reuse the 18-clip corpus that already exists, one run per question, and a blind A/B only before something ships. The fix for statistical power is not more tests but using the 18-clip set instead of the 5 matched clips for any decision, and treating a 6-trial listen as a smoke check. Students are trained with their own distillation losses and scored on clips they never saw, so DNSMOS and CAM++ are held-out judges and not training targets.
