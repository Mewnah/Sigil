# P3B-F specs

Streaming F0 v2 (cache geometry, edge-correct smoothing, cross-tick decoding, parallel fusion).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-F live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns of one executable (rule R3, R5). Record each spec's command and numbers
in the workstream report.

## Switches (one exe A/Bs every change, rule R5)

| variable | effect |
| --- | --- |
| `SIGIL_VC_F0_GEOMETRY=legacy` / `primed` / `filled` | `legacy` is P3's F0 cache (no priming, no silence continuity, P3's 12-frame lead) and the shipped default (`F0Geometry::DEFAULT`); `primed` (or `v2`) turns priming and continuity on; `filled` is `primed` plus filled calls, the measured combination. The single-part variables below override it (and a preset's base: `F0Geometry::from_env_over`) |
| `SIGIL_VC_F0_LEAD=fill` / `legacy` / `<frames>` | filled calls (the lead fills the extractor's fixed shape: 51 frames on RMVPE's default 64-frame plan, 19 on the 32-frame VRAM saver; plain RMVPE only, not the fusion) / P3's lead / a fixed lead (sweeps). Default `legacy` (fill is opt-in: rule R6, see the results). Alone it changes only the lead: priming and continuity stay as the geometry says (off by default); `SIGIL_VC_F0_GEOMETRY=filled` is the measured combination |
| `SIGIL_VC_F0_PRIME=1` | opt-in (off by default, rule R6): cold starts, gaps and jumps of a bounded extractor (RMVPE's 32-frame saver plan) as slices within the bound instead of one whole-window call (P3, the default); `=0` switches it off over `primed` / `filled` |
| `SIGIL_VC_F0_CONTINUITY=1` | opt-in (off by default, rule R6): silence-skipped ticks append unvoiced frames, so the first voiced tick after a pause is an ordinary incremental call; P3 (the default) leaves a gap in the cache, and the first voiced tick after a pause extracts its whole window. `=0` switches it off over `primed` / `filled` |
| `SIGIL_VC_F0_EDGE=zero` | the streaming smoother's median zero-pads its edges (P3) instead of replicating them (replicating is on by default) |
| `SIGIL_VC_F0_DECODE=1` / `0` | the cross-tick decoder on / off whatever the `f0_decode` lever says (RMVPE only; the lever reaches the F0 cache once `engine/load.rs` calls `F0Cache::set_decode`: a hotspot request of P3B-F; until then this is the only way to turn it on) |
| `SIGIL_VC_FUSION_PARALLEL=0` | the F0 fusion runs SwiftF0 and RMVPE one after the other (P3) instead of concurrently |
| `SIGIL_RMVPE_PLAN_FRAMES=32` | P3B-E's VRAM-saver plan (the 32 vs 64 question) |

## Files

- `fid-f0.template.json` + `run_fidelity.ps1`: fed-F0 fidelity (scorer v2: fine pitch error, GPE,
  V/UV and onset lag of the F0 the engine fed its generator, against whole-clip RMVPE of the input;
  MCD, speaker cosine, DNSMOS) of side A (`-EnvA`) against side B (`-EnvB`) of one executable, on 8
  dev-cache clips at three offsets each (leads of at least 4 s, so the silence skip runs before every
  utterance), ABBA across the clips, under the lock per batch, with the paired t interval over the
  clips and the share of clips where B is not worse.
- `compare_sides.py`: the same paired summary for two side folders of different runs (the engine is
  deterministic, so a side of one run pairs exactly with a side of another: how the 32-frame plan
  compares with the 64-frame plan with a filled lead), with the rows against the whole-utterance
  ceiling. `run_fidelity.ps1 -Preset quality -ExtraMs 192 -BlockMs 80` runs the Quality preset.
- `f0-offline.template.json` + `f0.profile.json`: the F0 stage's cost (offline, DirectML and CUDA,
  the fusion and RMVPE) for `run-perf-gate.ps1 -CompareEnv`: the parallel fusion
  (`SIGIL_VC_FUSION_PARALLEL=0` on side B) and the decoder (`SIGIL_VC_F0_DECODE=1`).
- The dev-rig sweep (CPU, no GPU) is the ignored test
  `neural_audio::vc::engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`
  (`SIGIL_TEST_MODELS_DIR`, `SIGIL_SWEEP_TABLE=geometry|cache|all`, `SIGIL_AB_SEED=1,1001`,
  `SIGIL_SWEEP_OUT=<file.json>`): table geometry (RMVPE at the 32- and 64-frame geometries, real-audio
  leads 12/19/32/51, zero vs replicated median edges, readout lags 0/2/4) and table cache (the
  production cache with the silence skip emulated, P3 vs the v2 parts, the decoder's variants
  replayed from one model pass, per clip / condition rows, the decoder's CPU cost). A second ignored
  test, `pauses_do_not_change_the_fed_f0` (`SIGIL_PAUSE_CLIPS=<wav>|<wav>`), compares P3's cache with
  the opt-in primed geometry over clips that have pauses. The F0 cache's call counters
  (`f0_cache::process_call_counts()`: calls and calls over the extractor's bound, process-wide) are for
  the benchmark engine to read before and after a run.

## Commands

```powershell
# the primed F0 cache (priming, continuity; opt-in) against P3's (the default: side B must switch the parts
# on, an empty side B is P3's cache again), DirectML
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 -Exe <exe> -AppData <appdata> -Ep dml -Out <dir> `
    -EnvA 'SIGIL_VC_F0_GEOMETRY=legacy;SIGIL_VC_F0_EDGE=zero' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed'
# the decoder (f0_decode) on the primed cache (the geometry the results below were measured with), CUDA
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 -Exe <exe> -AppData <appdata> -Ep cuda -Out <dir> `
    -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed;SIGIL_VC_F0_DECODE=1'
# filled calls (the primed cache against the filled one); the VRAM saver's plan with the primed cache
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 ... -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=filled'
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 ... -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed;SIGIL_RMVPE_PLAN_FRAMES=32'
# the Quality preset: filled calls; the 32-frame plan
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 ... -Preset quality -ExtraMs 192 -BlockMs 80 -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=filled'
pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 ... -Preset quality -ExtraMs 192 -BlockMs 80 -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed;SIGIL_RMVPE_PLAN_FRAMES=32'
# a side of one run against a side of another (the engine is deterministic)
python -I scripts/bench/specs/p3b/f/compare_sides.py <run1>/B <run2>/B
# the parallel fusion against the serial one (side B), DirectML and CUDA
pwsh scripts/bench/run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/f/f0.profile.json -Exe <exe> -CompareExe <exe> `
    -Only fusion-dml,fusion-cuda -CompareEnv 'SIGIL_VC_FUSION_PARALLEL=0' -AppData <appdata> `
    -Slot <appdata>/voice-changer/models/default -Clip <wav> -Rounds 6
# the 32-frame plan against the default plan, both with the parallel fusion
pwsh scripts/bench/run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/f/f0.profile.json -Exe <exe> -CompareExe <exe> `
    -Only fusion-dml,fusion-cuda -CompareEnv 'SIGIL_RMVPE_PLAN_FRAMES=32' -AppData <appdata> `
    -Slot <appdata>/voice-changer/models/default -Clip <wav> -Rounds 6
# the dev-rig sweep (CPU; hold the lock: it loads every core)
$env:SIGIL_TEST_MODELS_DIR = "$env:LOCALAPPDATA\sigil-dev-cache"; $env:SIGIL_SWEEP_TABLE = 'cache'; $env:SIGIL_AB_SEED = '1,1001'
cargo test --manifest-path src-tauri/Cargo.toml lead_sweep_against_whole_clip_f0 -- --ignored --nocapture --test-threads=1
```

## Results (P3B-F)

All engine A/Bs are two sides of ONE executable (`exe-f1` = the P3B tree of 2026-10-03 19:20 with P3B-G's
work in progress; rule R5), 8 real speech clips x 3 offsets each, scorer v2 (fed F0 = the pitch the
generator received, against whole-clip RMVPE of the input), under the GPU timing lock, ABBA across the
clips. The engine is deterministic: the default side gave bit-identical numbers in six separate runs
(fed fine error 23.5661 cents on DirectML, 23.537 on CUDA), so sides of different runs pair exactly clip
by clip (the "cross-run" row says so). Verdicts are the paired Student t 95 % interval over the 8 clips;
"k/8" counts clips. The scorer's onset lag is signed (negative: the engine voiced before the reference);
it is scored here by its size.

The runs below were taken while priming and continuity were still on by default (rule R6 has since
turned them off: `F0Geometry::DEFAULT` is P3's cache). "Default" in the tables below therefore means the
**primed** geometry (`SIGIL_VC_F0_GEOMETRY=primed`, or `SIGIL_VC_F0_PRIME=1;SIGIL_VC_F0_CONTINUITY=1`)
and "filled" means `SIGIL_VC_F0_GEOMETRY=filled`; "P3's cache" is `SIGIL_VC_F0_GEOMETRY=legacy`, which is
now what the shipped default does.

### Dev-rig sweep (CPU RMVPE, `lead_sweep_against_whole_clip_f0`)

Fine pitch error FPE in cents / gross pitch error GPE % / V/UV %, readout lag 0 = the newest block, no
lookahead:

| | real speech vs whole-clip RMVPE (7 clips) | synthetic, exact truth (held-out, 21 clips) | synthetic, exact truth (tuning, 21 clips) |
|---|---|---|---|
| 64-frame plan, P3 cache (lead 12) | 32.97 / 2.35 / 4.88 | 13.33 / 6.23 / 4.65 | 11.07 / 5.08 / 4.44 |
| 64, primed (priming, continuity) | 32.97 / 2.35 / 4.85 | 13.33 / 6.23 / 4.67 | 11.08 / 5.05 / 4.43 |
| 64, filled (lead 51) | 32.20 / 1.57 / 4.74 | 16.78 / 6.69 / 5.17 | 11.13 / 5.96 / 4.89 |
| 64, primed + decoder (lever defaults) | 31.74 / 2.19 / 5.27 | 11.76 / 5.58 / 4.75 | 10.84 / 4.83 / 4.36 |
| 32-frame plan, P3 cache | 33.07 / 1.23 / 4.91 (16 calls > 31 frames) | 16.70 / 10.52 / 3.74 (36) | 12.38 / 9.55 / 3.79 (39) |
| 32, primed | 33.06 / 1.23 / 4.91 (0) | 16.70 / 10.52 / 3.77 (0) | 12.38 / 9.57 / 3.79 (0) |
| 32, filled (lead 19) | 32.44 / 1.36 / 4.85 (0) | 16.68 / 8.10 / 4.53 (0) | 11.07 / 6.85 / 4.61 (0) |

Median edges: zero-padded and replicated are bit-identical on RMVPE's cached frames (its contour carries
one extra frame past the call, which absorbs the pad), so the edge fix buys the allocation-free smoother,
not accuracy. The count in brackets is the calls longer than the plan's 31 audio frames: the (opt-in)
cold-start priming takes it to 0.

Per condition (synthetic, exact truth, held-out corpus; GPE % / V/UV %): where the 32-frame plan is
weaker than the 64-frame one, and what a filled lead gives back.

| condition | 64, P3 (lead 12) | 64, filled (51) | 32, primed | 32, filled (19) |
|---|---|---|---|---|
| clean | 0 / 1.18 | 0 / 1.11 | 0 / 0.91 | 0 / 0.91 |
| creak | 11.28 / 1.62 | 11.98 / 1.77 | 21.15 / 1.13 | 14.86 / 1.48 |
| SNR 5 dB | 3.12 / 10.42 | 2.78 / 11.18 | 15.91 / 8.75 | 4.53 / 10.56 |
| SNR 10 dB | 0.09 / 4.28 | 0.35 / 6.25 | 1.38 / 4.07 | 0.35 / 6.32 |
| SNR 20 dB | 0 / 3.05 | 0 / 3.69 | 0.09 / 2.34 | 0 / 2.41 |
| music at 20 dB | 0.37 / 8.75 | 0 / 8.82 | 0.37 / 5.95 | 0 / 6.74 |
| no fundamental | 30.07 / 3.23 | 33.21 / 3.37 | 36.41 / 3.02 | 38.77 / 3.30 |

### Engine, fed-F0 fidelity (Low latency, RMVPE, 8 clips x 3 offsets; A -> B)

| comparison | fed FPE (cents) | fed GPE (%) | fed V/UV (%) | onset lag (size, frames) | speaker cos v2 | env MCD (dB) | DNSMOS |
|---|---|---|---|---|---|---|---|
| primed (`SIGIL_VC_F0_GEOMETRY=primed`, the default when measured) vs P3's cache (`SIGIL_VC_F0_GEOMETRY=legacy;SIGIL_VC_F0_EDGE=zero`), DML | bit-identical on all 8 clips (per-clip median over the offsets; priming does not apply to the unbounded 64-frame plan and the clips hold few pauses, see the pause check below) | | | | | | |
| decoder off -> on (`SIGIL_VC_F0_DECODE=1`), DML | 23.57 -> 23.91 (noise) | 0.71 -> 0.78 (noise) | 3.99 -> 3.82 (noise; better on 5/8) | 0.69 -> 0.25 (better, CI excludes 0) | 0.896 -> 0.882 (worse, CI -0.027 .. -0.001; 1/8 not worse) | +0.005 (noise) | -0.015 (noise) |
| decoder off -> on, CUDA | 23.54 -> 23.88 (noise) | 0.71 -> 0.78 (noise) | 3.99 -> 3.82 (noise) | 0.69 -> 0.25 (better) | 0.893 -> 0.877 (worse, CI -0.024 .. -0.008; 0/8 not worse) | +0.014 (worse, CI +0.001 .. +0.027) | +0.003 (noise) |
| 64-plan: lead 12 -> filled (`SIGIL_VC_F0_LEAD=fill`, lead 51), DML | 23.57 -> 17.76 (better, CI -9.8 .. -1.8; 7/8) | 0.71 -> 0.51 (noise; 6/8) | 3.99 -> 2.33 (better, CI -2.6 .. -0.7; 8/8) | 0.69 -> 0 (noise) | 0.896 -> 0.875 (noise, CI -0.044 .. +0.002; 1/8 not worse) | -0.009 (noise); STFT MCD against the whole-utterance ceiling -0.014 (better, CI excludes 0) | +0.001 (noise) |
| same, CUDA | 23.54 -> 17.76 (better) | 0.71 -> 0.51 (noise) | 3.99 -> 2.33 (better; 8/8) | 0.69 -> 0 (noise) | 0.893 -> 0.877 (noise; 2/8) | -0.010 (noise); against the whole ceiling -0.020 (better, 8/8) | -0.0003 (noise) |
| plan 64 -> 32 (`SIGIL_RMVPE_PLAN_FRAMES=32`, P3B-E's saver), DML | 23.57 -> 18.72 (better, CI -9.1 .. -0.6; 8/8) | 0.71 -> 0.48 (noise) | 3.99 -> 2.97 (better, CI -1.8 .. -0.2; 7/8) | 0.69 -> 0.06 (better) | 0.896 -> 0.888 (noise) | -0.025 (better, CI -0.044 .. -0.005; 7/8) | -0.009 (noise) |
| 32-plan: lead 12 -> filled (lead 19), DML | 18.72 -> 17.87 (noise) | 0.48 -> 0.46 (noise) | 2.97 -> 2.86 (noise) | 0.06 -> 0.13 (noise) | 0.888 -> 0.878 (noise; the yue clip, 0.849 -> 0.757, carries it) | +0.007 (noise) | -0.013 (noise) |
| cross-run, DML: 32-plan (B of the saver run) vs 64-plan filled (B of the filled run) | 18.72 vs 17.76 (noise, CI -2.5 .. +0.6) | 0.48 vs 0.51 (noise) | 2.97 vs 2.33 (the 64-plan filled is better, CI -1.3 .. -0.0; 7/8) | 0.06 vs 0 (noise) | 0.888 vs 0.875 (noise) | 1.960 vs 1.975 (noise) | 2.90 vs 2.91 (noise) |
| Quality preset (80 ms blocks, 192 ms extra context), DML: lead 12 -> filled | 21.52 -> 15.48 (better, CI -10.3 .. -1.8; 7/8) | 0.76 -> 0.44 (noise; 6/8) | 3.83 -> 2.15 (better, CI -2.5 .. -0.9; 8/8) | 1.00 -> 0 (better, CI -1.9 .. -0.1) | 0.939 -> 0.933 (noise; 3/8 not worse; yue 0.908 -> 0.884, zh 0.920 -> 0.908) | -0.010 (noise) | +0.030 (noise) |
| Quality preset, DML: plan 64 -> 32 | 21.52 -> 16.16 (better, CI -9.4 .. -1.3; 7/8) | 0.76 -> 0.45 (noise) | 3.83 -> 2.88 (better, CI -1.7 .. -0.2; 6/8) | 1.00 -> 0.06 (better) | 0.939 -> 0.940 (noise) | -0.006 (noise) | +0.016 (noise) |
| Quality preset, cross-run: 32-plan vs filled 64-plan | 16.16 vs 15.48 (noise) | 0.45 vs 0.44 (noise) | 2.88 vs 2.15 (noise, CI -1.6 .. +0.1) | 0.06 vs 0 | 0.940 vs 0.933 (noise) | 1.770 vs 1.765 (noise) | 3.01 vs 3.03 (noise) |
| copies of the 8 clips at 10 dB SNR (pink noise), DML: plan 64 -> 32 | 28.21 -> 26.87 (noise) | 2.99 -> 2.72 (noise) | 7.26 -> 7.73 (noise) | 0.25 -> 0.06 (noise) | 0.834 -> 0.846 (noise) | -0.003 (noise) | +0.003 (noise) |
| same copies, 32-plan: lead 12 -> filled (19) | 26.87 -> 29.00 (noise) | 2.72 -> 2.76 (noise) | 7.73 -> 7.56 (noise) | 0.06 -> 0.38 (noise) | 0.846 -> 0.850 (noise) | -0.027 (noise) | +0.022 (noise) |

Reading:

1. The parts of the v2 cache (priming and continuity, now opt-in; replicated edges, on) change nothing the
   engine can measure on these clips at the scorer's granularity (per-clip medians), and the CPU sweep finds
   them neutral; their benefit is structural (no extractor call longer than its plan, no whole-window call at
   an onset), and the pause check below shows a few isolated frames that do move.
2. The decoder does not improve real speech and lowers the speaker similarity: it stays off.
3. Fill improves the pitch rows a lot on the 64-frame plan, because the 24-frame incremental call sits
   behind 40 frames of the extractor's silence floor there (the end-aligned padding) and fill replaces
   the floor with real audio.
4. The 32-frame plan gets most of that gain by itself (24 of its 32 frames are real audio), and fill
   adds nothing measurable on top of it on clean speech; on the synthetic corpus fill on the 32-frame
   plan trades gross errors (creak, low SNR: better) for voicing errors (SNR 10 dB: worse).
5. The 32-frame plan is weaker than the 64-frame one under creak and heavy noise (synthetic exact truth:
   GPE 21 % vs 11 % creak, 16 % vs 3 % at SNR 5 dB), which the clean-ish real clips do not show.
6. Every F0 change that moves the contour lowers the speaker cosine v2 a little, mostly on the tonal clips
   (yue, zh); the lead therefore stays an opt-in lever (rule R6).

### Pauses: priming and continuity (opt-in)

Dev-rig check `pauses_do_not_change_the_fed_f0` (CPU RMVPE, the engine's silence skip emulated, readout
lag 0): four clips of 20 to 22 s, the dev-cache utterances joined by 1.2 s of digital silence (`multi-1`,
`multi-3`) or of -52 dBFS room tone (`multi-n1`, `multi-n3`; the gate is -42 dBFS). P3's cache against the
primed geometry (the default when measured), fed frames scored against whole-clip RMVPE as in the engine:

| clip, plan | fed frames that differ | worst difference | extractor calls of 44+ frames (P3 / primed) | calls over the plan (P3 / primed) | FPE c / GPE % / V/UV %, P3 -> primed |
|---|---|---|---|---|---|
| multi-1, 64 | 3 (of 362 converted ticks) | 4.1 cents | 6 / 1 | 0 / 0 | 31.81 / 1.709 / 4.53 -> the same |
| multi-1, 32 | 6 | 6.3 cents | 6 / 0 | 6 / 0 | 29.17 / 0.671 / 4.78 -> 29.16 / 0.671 / 4.78 |
| multi-3, 64 | 22 | one octave flip (1216 cents) | 8 / 1 | 0 / 0 | 34.56 / 2.012 / 5.02 -> 34.56 / 2.117 / 5.11 |
| multi-3, 32 | 21 | 48.8 cents | 8 / 0 | 10 / 0 | 32.14 / 2.128 / 5.11 -> 32.14 / 2.130 / 5.11 |

The room-tone copies give the same differences (within 0.01 cents of FPE). The primed cache's one remaining
44-frame call is the very first window. Priming and continuity therefore leave the contour alone but for
a handful of frames per pause (0.1 to 0.4 % of the fed frames; on `multi-3` at the 64-frame plan 0.1 pt
more gross errors, nowhere else), and they remove every call over the plan: without them each onset on
the 32-frame plan makes RMVPE split a 44-frame call into two plan runs (an extra GPU run on that tick).
In the engine (Low latency, DML, 32-frame plan, `multi-1..3` and the room-tone copies, 3 x 3 offsets;
P3's cache with the zero-padded median against the primed cache with replicated edges) every metric is bit-identical in the per-clip
median over the three offsets (fine error 17.27 cents, GPE 0.42 %, V/UV 2.89 %, cosine v2 0.925, DNSMOS 2.90
on the digital pauses; 17.25 / 0.42 / 2.61 %, 0.923, 2.83 on the room tone): the median hides a few isolated
frames, so this says "no regression at the scorer's granularity", not "no change". Those isolated frames (one of
them an octave) and the lack of any FCPE measurement are why priming and continuity ship off (rule R6).

### F0 stage cost (release engine, offline, `f0.profile.json`, ABBA pairs of one exe)

| gate | DML f0 p50 | CUDA f0 p50 | call p50 / p99 | CPU cores |
|---|---|---|---|---|
| parallel fusion (A, default) vs serial (`SIGIL_VC_FUSION_PARALLEL=0`, B), 6 pairs | 3.94 vs 4.15 ms (serial +5.3 %, CI +5.0 .. +8.7 %) | 3.12 vs 4.66 ms (-33 %) | DML -2 % (noise) / -8.5 %; CUDA -20 % / -20 % | +9 % / +38 % (two threads work at once) |
| decoder off (A) vs on (`SIGIL_VC_F0_DECODE=1`, B), RMVPE alone, 6 pairs | 4.41 vs 4.39 ms (NEUTRAL) | 3.28 vs 3.18 ms (NEUTRAL) | NEUTRAL | NEUTRAL |
| 32-frame plan (B) against the default plan (A), both with the parallel fusion (`SIGIL_RMVPE_PLAN_FRAMES=32`), 6 pairs; the GPU ran at 1470 MHz with 9 % decoder load from another program in this session (2800 MHz in the gate above), so only the ratios carry | 7.00 vs 5.70 ms (-18.7 %, CI -19.0 .. -18.3 %; call p50 -9 %) | 3.87 vs 4.03 ms (+4 %, NEUTRAL) | DML -9 % (p99 neutral); CUDA +2 % / +1 % (NEUTRAL) | NEUTRAL / -4 % |

The decoder's cache update without the model: 0.098 ms per tick at the median (p99 0.16 .. 0.45 ms) in the
unoptimized test build, 6 ABBA rounds over the first clip's ticks.

Pitch A/B (`f0_fusion::tests::pitch_ab_report`, CPU, 27 synthetic clips x 6 s with exact truth, re-run
after the changes; GPE 20 % / V/UV decision error): fusion SwiftF0 + RMVPE streaming 1.56 % / 5.16 %
(offline 0.08 % / 1.09 %), RMVPE alone streaming 8.68 % / 3.87 %. The fusion keeps its large gross-error
advantage, and the parallel fusion is bit-identical to the serial one (unit test), so these rows hold for it.

### Answers and recommendations for P3B-I0

- **Parallel fusion**: bit-identical to the serial one, so it stays on (`SIGIL_VC_FUSION_PARALLEL=0` is
  the switch back). F0 stage on a quiet GPU: DirectML 4.15 -> 3.94 ms, CUDA 4.66 -> 3.12 ms. The plan's
  target of 4.5 -> 3.0 ms on DirectML is **not demonstrated**: the 32-frame plan takes a further 18.7 % off
  in the (throttled) second session, which would put the quiet-machine figure at about 3.2 ms; P7
  re-measures on a quiet machine.
- **Priming and continuity**: opt-in, off by default (`F0Geometry::DEFAULT` is P3's cache; rule R6: a change
  that moves isolated fed frames, one by an octave on the pause clips, and has no FCPE measurement stays a
  switch until per-frame evidence flips it). Turn them on with `SIGIL_VC_F0_GEOMETRY=primed` (or
  `SIGIL_VC_F0_PRIME=1` and `SIGIL_VC_F0_CONTINUITY=1`), which is worth it on the 32-frame plan: every extractor
  call then stays inside its plan (0 calls over the 31-frame bound; without priming 6 to 39 per sweep or pause
  clip) and the contour is unchanged but for a handful of frames per pause. On the default 64-frame plan at Low
  latency priming is a no-op.
- **`f0_decode` lever**: keep it off (`auto_lever` false) and describe it as experimental. In the engine it
  moved no pitch row beyond noise (V/UV -0.16 pt, better on 5 of 8 clips; the plan expected -0.5 .. -1.5 and 70 %
  of the clips), lowered the speaker cosine (-0.014 DML, -0.016 CUDA, CIs exclude 0) and, on CUDA,
  raised the envelope MCD by 0.014 dB; it costs nothing measurable (< 0.1 ms per tick). Its voicing hysteresis moves
  onsets 0.44 frames later, toward the reference's. The lever is not wired in `engine/load.rs` yet (hotspot
  request), and the English string staged by P3B-S0 ("removes octave jumps and late voicing at word
  starts") claims more than the measurements show.
- **Lead fill**: keep P3's lead as the default (rule R6). It is the one switch with a large gain on the
  pitch rows when the plan is 64 frames (V/UV 3.99 -> 2.33 %, fine error 23.6 -> 17.8 cents at Low
  latency, 3.83 -> 2.15 % and 21.5 -> 15.5 cents at Quality), but it lowers the advisory speaker cosine
  (-0.021 / -0.006, within noise, tonal clips first) and the synthetic exact-truth corpora disagree with
  the whole-clip RMVPE reference. **The 32-frame plan gets almost the same pitch gain without the
  cosine cost** (Low latency: 18.7 cents, V/UV 2.97 %, cosine -0.008; Quality: 16.2 cents, V/UV 2.88 %,
  cosine +0.001), at half the VRAM and a faster F0 stage, and fill adds nothing on top of it.
- **32 vs 64 and the Quality preset**: the rule asked whether the 64-frame plan with a 51-frame lead is better
  on fed GPE and V/UV beyond the offset spread and not worse on T5. Against the 32-frame plan: GPE 0.51 vs
  0.48 % and fine error 17.8 vs 18.7 cents (both noise), V/UV 2.33 vs 2.97 % (7 of 8 clips, the interval
  barely excludes 0), cosine 0.875 vs 0.888. The GPE condition fails, so **no 64-frame Quality geometry is
  recommended**; the same holds at the Quality preset (32-plan 16.2 cents / 2.88 % against filled
  15.5 cents / 2.15 %, cosine 0.940 vs 0.933, all noise). One caution for whoever makes the 32-frame plan
  the default: on the synthetic exact-truth corpus it is weaker than the 64-frame plan under creak and heavy
  noise (GPE 21 % vs 11 % creak, 16 % vs 3 % at 5 dB SNR); on real clips at 10 dB SNR the two are equal within
  noise (32 vs 64: fine error 26.9 vs 28.2 cents, V/UV 7.7 vs 7.3 %, cosine 0.846 vs 0.834). A filled lead on the 32-frame
  plan wins that gross-error ground back (creak 15 %, 5 dB 4.5 %) and loses voicing, so it is the right lever
  for a voice flagged as noisy or creaky, not a default.

