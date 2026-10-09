# ADR 0009: Voice-changer fidelity scorer v2 (T5 v2 definitions)

- **Status:** accepted (P3B-B, 2026-10-01). The limits below are pre-registered: fixed from the
  measured floors before the engine workstreams' (P3B-D to P3B-H) final re-score.
- **Date:** 2026-10-01 (limits); corpus runs 2026-10-02
- **Workstream:** P3B-B (the ADR belongs to its workstream; INTEGRATE links it from `docs/adr/README.md`)
- **Plan decision:** P3B (voice-engine optimization phase), target revision T5 (`target_revisions`
  of the P3B plan, accepted by the user)

## Context

P3 scored the Sigil Voice Engine's stream against an offline ceiling with five fidelity metrics
(T5: CAM++ cosine ≥ 0.97, MCD ≤ 0.5 dB, voiced F0 RMSE ≤ 15 cents, V/UV ≤ 3 %, 0 seam clicks). Before
anyone optimizes toward them, P3B checked whether they measure the engine. Measured first (rule R1,
CPU NumPy replicas of the Rust scorer on the saved P3 WAV pairs, `scripts/bench/score_v2/`):

1. **The CAM++ asset is not upstream's function.** `bench/campplus` (sherpa-onnx's export of 3D-Speaker
   `campplus_sv_zh-cn_16k-common`) carries upstream's weights bit-identically (617 / 617 float
   initializers) and matches upstream PyTorch at 106.5-117.4 dB SNR on lengths that fill whole
   100-frame segments, but its 52 `AveragePool(ceil_mode)` segment poolings divide the last, partial
   segment by the full kernel in ONNX Runtime: on every other length it reads 6.5-25.5 dB SNR (mean
   18.1 dB; cosine to upstream 0.908-0.999).
2. **Identical audio scored differently.** The 0.9379 / 0.9412 pair (and 0.9351 / 0.9290, 0.9514 /
   0.9435 in the P3 gates) had bit-identical streams; the ceilings, re-rendered by the other build,
   differed in six 10 ms frames of near-silence (≤ −78 dBFS) inside the scored region. The v1
   embedding includes every frame of the region and normalises the fbank over all of them, so those
   frames moved the cosine by 0.003-0.008. Embedding speech frames only (within 40 dB of the
   loudest), with the exact export, gives bit-identical cosines for both builds' pairs.
3. **The F0 RMSE counts a few octave frames and the analyzer's own error.** Split into fine pitch
   error (non-gross voiced frames) and gross frames, the P3 final gate's rows read: CUDA 52.6 cents =
   44.4 cents fine + 1 gross frame of 449 cents; DirectML 46.9 = 37.5 + 1 × 449; CPU 38.6 = 38.6 + 0;
   DirectML ctx512 61.4 = 38.0 + 3 (349, −475, −479); the pre-fix rows 67.8 = 37.1 + 4 (520, 520, 366,
   −588). The tracker's fine error is 37-45 cents through the output analyzer, which itself moved
   identical CPU audio from 52.6 to 38.6 cents in P3.
4. **The click statistic is ill-conditioned.** The join windows cover about 70 % of a 40 ms block,
   and v1 estimated the background rate from the 3-8 events outside them. Recomputed from the 83
   saved quality reports, v1 reads non-zero on 9 runs; a conditional binomial test (97.5 %) reads 0 on
   7 of them, among them the (30 events, 26 in windows) case (P(X ≥ 26) = 0.032) and the (24, 30)
   ctx512 case, and 2 on the two lead-1 lookahead runs (29 of 32, 37 of 42; P = 0.006). A second
   defect found here: v1 places the windows at the returned span's start in input coordinates, but
   with lookahead the returned audio is the input `lookahead` earlier, so at 20 ms of lookahead the
   windows sat 20 ms late and covered a quarter of the crossfade. Those two "2 clicks" were scored
   on misplaced windows.
5. **The MCD measures excitation as much as envelope, and its floor is near the target.** See the
   floors below.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| Keep v1, relabel | no work | the numbers above stay wrong; the autotuner keeps steering on them | Context 1-5 |
| Replace v1 with v2 | one set of numbers | moves the goalposts; old reports become incomparable | — |
| **v2 next to v1, limits fixed from floors before re-scoring** | interpretable rows, continuity, no goalpost moving | more metrics, longer runs | this ADR |

## Decision

`scoring.analyzer: v2` (opt-in per quality run; `v1` stays the default) adds, next to every v1
metric (`scoring.analyzer: v2+ablations` also runs the oracle ablations at the first offset;
`SIGIL_BENCH_ABLATIONS=1` is the same switch for a `v2` spec; the spec validation accepts both since P3B-I0):

- **Speaker identity:** `bench/campplus-exact`, the same checkpoint re-exported through Seed-VC's
  exact segment pooling (`scripts/vc-templates/export_bench.py`, ≥ 100 dB SNR against upstream;
  measured 104.5-117.4 dB). Embeddings of speech frames only, both renditions on the reference's
  frames. The identity metric is `speaker_set_sim`: the similarity to the centroid of a
  target-speaker set (the ceiling's renditions of *other* clips, ≥ 3 s of speech each), for the
  stream and the whole-utterance ceiling against the same set, and their difference
  `speaker_set_sim_delta`. An RVC voice ships no recordings of its target speaker, so a quality run
  takes the set from `fidelity.reference_clips`, and the corpus measures it with a `rescore` of the
  saved stream / whole-ceiling pairs of every offset, leaving the whole clip out of its own set (its
  other offsets render the same speech) and taking the clip's median over its offsets.
  `speaker_cosine_v2` (against the ceiling) is advisory. The scorer's sha256 and the v1 file's are
  stamped into every report.
- **Pitch:** the F0 the engine fed its generator, captured by the engine probe on the absolute 10 ms
  grid (no output analyzer), against whole-clip analysis of the input: `fed_input_fpe_cents` (fine
  pitch error), `fed_input_gpe`, `fed_input_vuv_error`, the voicing onset lag; the same against the
  ceiling's fed F0 (`fed_*`), ceiling A's streaming tracker against the input
  (`ceiling_fed_input_*`), and the **analyzer floor** `whole_fed_input_*`: the run's tracker over
  the whole clip (the whole ceiling's F0) against the analyzer, i.e. what the same tracker reads on
  the same audio with another context. With RMVPE (the analyzer's own model) that is the floor the
  plan's rule is applied to; with another tracker (FCPE, which the Low-latency preset picks on the
  CPU) it is that tracker's disagreement with the analyzer, a diagnostic, not a floor. A grid check
  (`grid_lag_vs_input`: the frame lag at which the tracks agree best) guards the absolute frame
  grid, and the tracks themselves are in the report. The output-analyzer split
  (`f0_fine_rmse_cents`, `f0_gross_frames` with positions) and a ±1-frame tolerant V/UV error
  (`vuv_error_tolerant`, a SOLA-jitter diagnostic, never a target) stay as diagnostics; the
  analyzer's version (method, model sha256, its output on a fixed probe signal) is stamped into
  every report. The signed onset lag (`fed_input_onset_lag_frames`; negative = the voicing starts
  early) has no better direction, so it is a diagnostic; a before/after gate reads
  `fed_input_onset_lag_abs_frames` (the median of the lags' magnitudes, lower is better), and the
  onset levels `onset_rel_db*` are listed as higher-is-better (closer to 0 dB = a less muted onset).
- **Clicks:** `clicks_binomial` (v1's window shape, placed for the model's delay) and
  `clicks_narrow` (±1.5 ms at both crossfade boundaries, `j − delay + search/2` and that
  `+ crossfade`), each the events beyond the 97.5 % quantile of Binomial(events, coverage).
  Validated by planted pops at both boundaries and a null test (random events at N = 20-40 read 0 in
  ≥ 95 % of draws).
- **Spectral:** `env_mcd_db`, the MCD of the F0-adaptive (CheapTrick-style) envelope, both renditions
  analysed with the reference's F0; the STFT MCD stays. Floors: ceiling A against itself with its
  block grid shifted by 100 ms (`env_mcd_floor_db`, the primary floor: the stream's block grid
  differs from A's), against itself with another noise seed (`env_mcd_floor_seed_db`; the plan
  expected an upper bound, the runs below measured it 0.15 dB *under* the shifted floor, so it is
  reported as a second floor), and against ceiling B (400 ms blocks, 160 ms lookahead, 500 ms
  context: `mcd_offline_floor_db`). `mcd_offset_corrected_db` reads every analysis frame of the
  stream where its block's content is (the block's recorded SOLA offset; a ±6 ms search when a
  report has no call log, and for any call whose join the stream loop did not observe: the harness
  keeps one join per device block, so when a block makes several model calls only the newest is
  attributed and the others read "unobserved", never "joined at the centre";
  `offset_corrected.searched_blocks` and `sola.calls_without_join_info` count them, and a `rescore`
  reads the run's own join geometry and these flags back from the report, so it reproduces a run
  made under a non-default SOLA window). The stream is not spliced: moving the blocks back into place would
  add discontinuities at every join, which the first implementation did and which read 0.3 dB
  *worse* than the raw MCD.
- **Ceilings:** a v2 run of a template voice renders all three: `long_shape` (A, today's), `b` and
  `whole` (the input region with 1 s of reflect padding converted in one pass on a dynamic-shape CPU
  path, at the run's precision, fp32 as an extra row); `ceiling.mode` picks the primary, against
  which the v1 metrics are computed (default A). The ceilings `ceiling.mode` names (and `whole` when
  the ablations run) must render, else the run fails, before the engine loads when the clip or lead
  cannot fit (a `whole` ceiling needs a `lead_s` of at least 0.4 s, the decoder's margin, and a
  speech region of at most 60 s). The others are optional: when one cannot be rendered (region over
  60 s, a short lead, a recording that does not match the input, no fp32 content encoder for the CPU
  EP) the run notes why and omits that rendition's rows and the floors that need it, so a metric that
  needs neither ceiling is never lost to them. Raw MCD against A stays the headline continuity
  number; DNSMOS is reported for
  every rendition, and so are the onset envelopes (`details.offsets[].onset_envelopes`: the energy
  of every rendition around each utterance onset of the input, and `onset_rel_db[_<mode>]`, the
  first 200 ms after the onsets relative to the rendition's speech level), so a ceiling that mutes
  onsets (the failure that rules out 1 s of left context) is visible next to its scores.
- **Hygiene:** the stream is a twin of the voice with the silence skip off (both renditions gated the
  same way); every clip is scored at ≥ 3 offsets (`fidelity.offsets_s`; a v2 run that names none
  scores `lead_s`, `lead_s + 0.27` and `lead_s + 0.61`, and one that names fewer than three says so
  in its notes), headline = median, spread reported; corpus rows are means with a bootstrap 95 % CI, single-clip rows flagged; `rescore`
  re-scores saved WAV pairs (any build's).

**T5 v2 limits** (pre-registered; Low latency / Quality):

| Row | Metric | Limit |
| --- | --- | --- |
| Identity | `speaker_set_sim_delta` = SIM(stream, set) − SIM(whole ceiling, set) | ≥ −0.02 / ≥ −0.01 (corpus mean, CI reported) |
| Fine pitch | `fed_input_fpe_cents` | ≤ 15 cents (the plan's rule, floor + 5 when the analyzer floor exceeds 10 cents, applied once to the floor measured below: the RMVPE analyzer floor read 5.7 cents [3.8, 8.0] over the 18-clip corpus, below 10, so the limit is 15; a fixed number, never a run's own floor) |
| Gross pitch | `fed_input_gpe` | ≤ 0.3 % (corpus level: pooled over every voiced frame, `fed_input_gpe_pooled` in `aggregate.py`) |
| Voicing | `fed_input_vuv_error` | ≤ 3 % |
| Spectral | `env_mcd_over_floor_db` = env MCD vs A − the same-seed, block-shifted floor | ≤ 0.6 dB / ≤ 0.3 dB |
| Seams | `clicks_binomial`, `clicks_narrow` | 0 |

`fidelity::t5_shortfall_v2` scores these rows (Low latency's limits) when a run carries v2 metrics
and the P3 formula otherwise. The constants live in `bench/fidelity.rs` (`T5_*`). The scorecard
applies the limits as fixed numbers. The autotuner's shortfall is the one place a run's own floor
enters: when the run's tracker is not the analyzer's (its `whole_fed_input_fpe_cents` is over 10
cents; FCPE on the CPU preset reads 34.5), `t5_shortfall_v2` charges the pitch rows net of that
run's own `whole_fed_input_*` floors, since the limits are unreachable on such a tracker by
construction and every configuration of the run would otherwise carry the same constant shortfall
(about 15 units on the CPU preset, enough to swamp the autotuner's 25 % relative-improvement rule).
The scorecard labels those rows "tracker disagreement", not streaming loss.

## Evidence

Measure-first (CPU, no GPU; `scripts/bench/score_v2/measure_p3.py` with the dev cache's ONNX
Runtime 1.28 build) and the export check (`python scripts/vc-templates/export_bench.py --only
campplus-exact`; `scripts/vc-templates/reports/bench.json`, section `campplus-exact`):

| Check | Result |
| --- | --- |
| `bench/campplus-exact` vs upstream PyTorch, 16 real fbank lengths (jfk, ja-long; whole-segment and ragged) | min 104.5 dB, max 117.4 dB SNR; deterministic (same input twice bit-identical); sha256 `3d88451a…792e0c`, 28,206,219 bytes |
| `bench/campplus` (legacy) vs upstream | whole segments 106.5-117.4 dB; ragged 6.5-25.5 dB (mean 18.1); 617 / 617 initializers bit-identical (same checkpoint) |
| Calibration, exact export: two different 3.8 s utterances of one speaker | en (JFK) 0.778, ja 0.654; halves of the 11 s JFK clip 0.791; other speakers 0.02-0.26 (margin 0.39) |
| Identical streams across builds, v1 vs v2 cosine | v1 0.9351 / 0.9290 and 0.9514 / 0.9435; exact on speech frames 0.926339 / 0.926339 and 0.948842 / 0.948842 |
| P3 final gate, v1 → v2 cosine (exact, speech frames) | CPU 0.9435 → 0.9488; CUDA 0.9235 → 0.9283; DirectML 0.9254 → 0.9298; ctx512 0.9513 → 0.9574 |
| P3 final gate, F0 RMSE → fine + gross | see Context 3 |
| P3 final gate, envelope MCD (NumPy replica) next to STFT MCD | CPU 2.01 vs 1.58; CUDA 2.04 vs 1.67; DirectML 2.06 vs 1.69; ctx512 2.15 vs 1.72 dB |
| Identity control (CUDA corpus, n 18): offline ceiling A / B in the stream's place, delta against the whole-ceiling set | A −0.003 [−0.013, +0.006]; B −0.002 [−0.010, +0.005] (the stream: −0.054) |

The floors, the whole-utterance ceiling and the oracle ablations (Rust, `Sigil.exe --headless bench`
with `scripts/bench/specs/p3b/b/fid-v2.template.json` under the GPU timing lock) are recorded below
by the run that measured them.

### Corpus runs (P3B-B, 2026-10-01 to 2026-10-02)

Setup: `Sigil.exe --headless bench` through `scripts/bench/specs/p3b/b/run_corpus.ps1`, holding the
GPU timing lock per batch of clips. The quality runs used the release build `exe-b4` (sha256
`7b46677f…`; the scorer in its final state, the engine as P5 left it); the first CPU and DirectML
ablation clips ran on `exe-b3`, whose scorer differs only by the short-clip alignment fix and three
diagnostics added later, which the aggregation reads back from its details. The target-speaker
similarity is a `rescore` (release build `exe-b5`, sha256 `c371ef64…`: the same scorer plus the
per-clip grouping of the rescore) of the saved stream / whole-ceiling pairs of every offset, leaving
each clip out of its own set. Voice: the MIT default voice (`PhoenixStormJr/RVC-V2-default-voice`,
template slot). Stream: the Low-latency preset, 40 ms engine blocks, 512-frame device blocks at 48
kHz, 64 ms extra context, no lookahead (the CPU preset picks FCPE, the GPUs RMVPE). Clips: the 18
local clips of the dev cache (the 13 sherpa-onnx test clips and the 5 transcribe samples; 2.75-28.2
s, ten languages), each at three leads (1.0, 1.27, 1.61 s; the clip's value is the median over
them). Corpus rows are the mean over the clips with a percentile-bootstrap 95 % CI
(`scripts/bench/score_v2/aggregate.py`, 10 000 resamples); GPE is pooled over every voiced frame.
The ablations ran on six clips per EP (`en-jfk`, `en`, `de`, `fr`, `ja`, `zh`). Each clip's
`load.json` records the rustc processes and CPU load when it started: the earlier passes ran while
other agents compiled, the last ones on a quiet machine; no metric here is a timing. Per-EP
summaries: scratchpad `p3b/b/r2/corpus-{cuda,dml,cpu}/summary.{json,md}` (`aggregate.py`).

Spectral (dB; corpus mean [bootstrap 95 % CI]; clip = median over 3 offsets)

| Metric | CUDA (n 18) | DirectML (n 18) | CPU (n 18) |
| --- | --- | --- | --- |
| STFT MCD vs ceiling A (raw, the headline) | 1.91 [1.83, 1.99] | 1.91 [1.83, 1.99] | 1.93 [1.85, 2.01] |
| STFT MCD, offset-corrected | 1.87 [1.79, 1.95] | 1.87 [1.79, 1.95] | 1.89 [1.81, 1.97] |
| STFT floor: A vs A, blocks shifted 100 ms (primary) | 1.33 [1.24, 1.41] | 1.33 [1.24, 1.41] | 1.32 [1.23, 1.40] |
| STFT floor: A vs A, other noise seed | 1.09 [1.04, 1.14] | 1.09 [1.05, 1.14] | 1.09 [1.04, 1.14] |
| STFT floor: A vs ceiling B (`mcd_offline_floor_db`) | 1.35 [1.26, 1.43] | 1.35 [1.26, 1.43] | 1.31 [1.24, 1.38] |
| Envelope MCD vs ceiling A | 2.03 [1.97, 2.08] | 2.03 [1.97, 2.08] | 2.05 [1.99, 2.11] |
| Envelope MCD, offset-corrected | 1.98 [1.92, 2.04] | 1.98 [1.92, 2.04] | 2.00 [1.94, 2.07] |
| Envelope floor: blocks shifted 100 ms (primary) | 1.43 [1.36, 1.49] | 1.43 [1.36, 1.49] | 1.41 [1.35, 1.47] |
| Envelope floor: other noise seed | 1.22 [1.21, 1.24] | 1.22 [1.21, 1.24] | 1.22 [1.20, 1.23] |
| Envelope floor: A vs B | 1.44 [1.37, 1.51] | 1.44 [1.37, 1.51] | 1.41 [1.35, 1.46] |
| **Envelope MCD over its floor (T5 v2 row)** | 0.60 [0.54, 0.65] | 0.60 [0.55, 0.65] | 0.64 [0.58, 0.70] |
| STFT MCD vs ceiling B | 2.03 [1.95, 2.13] | 2.03 [1.94, 2.12] | 2.04 [1.95, 2.13] |
| STFT MCD vs the whole-utterance ceiling | 2.14 [2.03, 2.25] | 2.13 [2.03, 2.25] | 2.16 [2.05, 2.28] |
| STFT MCD vs the whole ceiling, fp32 | 2.13 [2.03, 2.25] | 2.13 [2.03, 2.24] | - (n 0) |

Speaker identity

| Metric | CUDA (n 18) | DirectML (n 18) | CPU (n 18) |
| --- | --- | --- | --- |
| v1 cosine vs A (legacy file, every frame) | 0.9003 [0.8820, 0.9162] | 0.8986 [0.8807, 0.9146] | 0.8986 [0.8820, 0.9133] |
| v2 cosine vs A (exact, speech frames; advisory) | 0.8862 [0.8732, 0.8990] | 0.8799 [0.8612, 0.8971] | 0.8838 [0.8655, 0.9003] |
| SIM(stream, target set) | 0.7149 [0.6808, 0.7455] | 0.7138 [0.6810, 0.7431] | 0.7205 [0.6826, 0.7533] |
| SIM(whole ceiling, target set) | 0.7711 [0.7377, 0.7999] | 0.7711 [0.7377, 0.7999] | 0.7755 [0.7425, 0.8042] |
| **SIM delta (T5 v2 row)** | -0.0542 [-0.0753, -0.0339] | -0.0534 [-0.0794, -0.0262] | -0.0526 [-0.0818, -0.0250] |
| v2 cosine spread over the offsets | 0.0502 [0.0318, 0.0725] | 0.0576 [0.0360, 0.0830] | 0.0426 [0.0286, 0.0581] |

Pitch (cents; V/UV and GPE as shares)

| Metric | CUDA (n 18) | DirectML (n 18) | CPU (n 18) |
| --- | --- | --- | --- |
| Output-analyzer F0 RMSE (v1) | 42.9 [34.5, 53.2] | 43.0 [34.6, 53.5] | 71.9 [38.5, 113.8] |
| Output-analyzer fine error | 33.6 [30.5, 37.1] | 32.7 [30.0, 35.8] | 33.2 [30.1, 36.6] |
| Output-analyzer gross frames | 0.9 [0.3, 1.7] | 0.7 [0.2, 1.2] | 3.2 [0.5, 7.1] |
| **Fed F0 vs input: fine error (T5 v2 row)** | 22.9 [20.6, 25.4] | 22.9 [20.6, 25.4] | 35.2 [30.8, 39.8] |
| **Fed F0 vs input: GPE, pooled (T5 v2 row)** | 0.0142 [0.0142, 0.0142] (n 24889) | 0.0142 [0.0142, 0.0142] (n 24890) | 0.0382 [0.0382, 0.0382] (n 24088) |
| **Fed F0 vs input: V/UV (T5 v2 row)** | 0.0507 [0.0344, 0.0703] | 0.0507 [0.0344, 0.0703] | 0.1029 [0.0726, 0.1342] |
| Fed F0 vs input: onset lag (frames) | -1.06 [-1.75, -0.47] | -1.06 [-1.75, -0.47] | -2.50 [-3.44, -1.56] |
| Fed F0 vs ceiling A's fed F0: fine error | 19.8 [17.7, 22.1] | 19.8 [17.6, 22.1] | 17.5 [13.8, 21.2] |
| Fed F0 vs ceiling A's: V/UV | 0.0408 [0.0295, 0.0540] | 0.0408 [0.0295, 0.0540] | 0.0376 [0.0250, 0.0522] |
| Ceiling A's fed F0 vs input: fine error | 20.5 [18.0, 23.0] | 20.5 [18.0, 23.0] | 34.5 [29.8, 39.3] |
| Analyzer floor: whole-clip tracker vs input, fine | 5.7 [3.8, 8.0] | 5.7 [3.8, 8.0] | 34.5 [29.5, 39.7] |
| Analyzer floor: V/UV | 0.0066 [0.0032, 0.0104] | 0.0066 [0.0032, 0.0104] | 0.0866 [0.0572, 0.1188] |
| Fine error spread over the offsets | 6.4 [5.0, 8.0] | 6.4 [5.0, 8.0] | 4.7 [3.4, 6.2] |

Seams

| Metric | CUDA (n 18) | DirectML (n 18) | CPU (n 18) |
| --- | --- | --- | --- |
| v1 clicks | 0.2 [0.0, 0.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| **Binomial clicks, wide windows (T5 v2 row)** | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| **Binomial clicks, narrow windows (T5 v2 row)** | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.2] | 0.0 [0.0, 0.0] |
| Artifact events | 28.8 [22.2, 36.8] | 29.7 [22.4, 38.3] | 30.6 [23.1, 40.6] |

DNSMOS OVRL and onsets

| Metric | CUDA (n 18) | DirectML (n 18) | CPU (n 18) |
| --- | --- | --- | --- |
| Stream | 2.76 [2.61, 2.90] | 2.79 [2.64, 2.92] | 2.75 [2.60, 2.89] |
| Ceiling A | 3.08 [2.94, 3.20] | 3.07 [2.92, 3.19] | 3.09 [2.98, 3.18] |
| Ceiling B | 3.03 [2.91, 3.14] | 3.03 [2.91, 3.14] | 3.04 [2.92, 3.14] |
| Whole ceiling | 3.12 [3.02, 3.21] | 3.12 [3.02, 3.21] | 3.09 [2.98, 3.19] |
| Whole ceiling fp32 | 3.07 [2.93, 3.19] | 3.07 [2.93, 3.19] | - (n 0) |
| Onset level, stream (dB re speech) | -4.9 [-7.2, -2.6] | -4.9 [-7.2, -2.6] | -5.0 [-7.4, -2.6] |
| Onset level, A | -3.7 [-7.1, -0.7] | -3.7 [-7.1, -0.7] | -3.8 [-7.2, -0.9] |
| Onset level, B | -2.4 [-5.3, -0.0] | -2.4 [-5.3, -0.0] | -3.3 [-7.9, -0.1] |
| Onset level, whole | -2.1 [-9.0, 1.8] | -2.1 [-9.0, 1.8] | -2.2 [-9.2, 1.8] |

CUDA ablations (n = 6 clips): baseline distance, then recovered (positive = closer)

| Ablation | Ceiling | STFT MCD | Env MCD | Cosine v2 | V/UV | Fine F0 (cents) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | A | 1.896 | 2.011 | 0.889 | 0.047 | 33.8 |
| baseline | whole | 2.222 | 2.335 | 0.876 | 0.060 | 31.4 |
| oracle_f0 | A | +0.004 | -0.000 | +0.009 | +0.005 | -1.3 |
| oracle_f0 | whole | +0.013 | +0.009 | +0.010 | +0.015 | +0.8 |
| oracle_content | A | -0.207 | -0.210 | -0.021 | -0.013 | -1.9 |
| oracle_content | whole | +0.376 | +0.379 | +0.004 | -0.003 | -1.7 |
| oracle_both | A | -0.198 | -0.204 | -0.014 | -0.015 | -0.5 |
| oracle_both | whole | +0.400 | +0.397 | +0.003 | +0.004 | +1.3 |
| stream_f0_in_ceiling | A | +0.003 | +0.007 | +0.003 | +0.002 | -1.1 |
| stream_f0_in_ceiling | whole | +0.009 | +0.010 | -0.014 | +0.009 | -2.5 |
| residual | A | +0.315 | +0.324 | +0.013 | +0.003 | +3.3 |
| residual | whole | +0.400 | +0.397 | +0.003 | +0.004 | +1.3 |

DirectML ablations (n = 6 clips): baseline distance, then recovered (positive = closer)

| Ablation | Ceiling | STFT MCD | Env MCD | Cosine v2 | V/UV | Fine F0 (cents) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | A | 1.890 | 2.003 | 0.890 | 0.044 | 34.1 |
| baseline | whole | 2.218 | 2.331 | 0.873 | 0.057 | 30.6 |
| oracle_f0 | A | -0.004 | -0.009 | +0.006 | +0.004 | -1.4 |
| oracle_f0 | whole | +0.012 | +0.009 | +0.009 | +0.014 | -1.2 |
| oracle_content | A | -0.216 | -0.224 | -0.017 | -0.018 | -1.0 |
| oracle_content | whole | +0.369 | +0.368 | +0.012 | -0.008 | -2.2 |
| oracle_both | A | -0.198 | -0.207 | -0.019 | -0.020 | -0.6 |
| oracle_both | whole | +0.400 | +0.396 | +0.004 | +0.001 | +0.7 |
| stream_f0_in_ceiling | A | +0.002 | +0.006 | +0.001 | +0.001 | -0.8 |
| stream_f0_in_ceiling | whole | +0.007 | +0.008 | -0.015 | +0.009 | -2.7 |
| residual | A | +0.313 | +0.320 | +0.011 | -0.001 | +4.3 |
| residual | whole | +0.400 | +0.396 | +0.004 | +0.001 | +0.7 |

CPU ablations (n = 6 clips): baseline distance, then recovered (positive = closer)

| Ablation | Ceiling | STFT MCD | Env MCD | Cosine v2 | V/UV | Fine F0 (cents) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | A | 1.914 | 2.033 | 0.901 | 0.047 | 31.0 |
| baseline | whole | 2.242 | 2.360 | 0.869 | 0.049 | 33.6 |
| oracle_f0 | A | +0.034 | +0.033 | -0.003 | -0.000 | -2.9 |
| oracle_f0 | whole | +0.039 | +0.040 | +0.008 | -0.005 | +1.0 |
| oracle_content | A | -0.199 | -0.198 | -0.027 | -0.022 | -3.1 |
| oracle_content | whole | +0.371 | +0.379 | +0.011 | -0.016 | -3.3 |
| oracle_both | A | -0.175 | -0.174 | -0.042 | -0.015 | -2.4 |
| oracle_both | whole | +0.410 | +0.414 | +0.005 | -0.013 | +0.0 |
| stream_f0_in_ceiling | A | +0.006 | +0.009 | +0.002 | +0.005 | -2.3 |
| stream_f0_in_ceiling | whole | +0.015 | +0.017 | +0.002 | -0.001 | +0.6 |
| residual | A | +0.316 | +0.328 | -0.003 | +0.002 | +1.0 |
| residual | whole | +0.410 | +0.414 | +0.005 | -0.013 | +0.0 |

T5 v2 at the P5 engine, Low latency (corpus mean; met against the Low-latency limit)

| Row | Limit (Low latency / Quality) | CUDA | DirectML | CPU |
| --- | --- | --- | --- | --- |
| Identity: SIM(stream, set) − SIM(whole ceiling, set) | ≥ −0.02 / ≥ −0.01 | −0.054 (missed) | −0.053 (missed) | −0.053 (missed) |
| Fine pitch error of the fed F0 (cents) | ≤ 15 | 22.9 (missed) | 22.9 (missed) | 35.2 (missed; FCPE tracker disagreement, floor 34.5) |
| Gross pitch error of the fed F0 (pooled) | ≤ 0.3 % | 1.42 % (missed) | 1.42 % (missed) | 3.82 % (missed; tracker disagreement) |
| V/UV error of the fed F0 | ≤ 3 % | 5.1 % (missed) | 5.1 % (missed) | 10.3 % (missed; tracker disagreement) |
| Envelope MCD over its floor (dB) | ≤ 0.6 / ≤ 0.3 | 0.60 (missed) | 0.60 (missed) | 0.64 (missed) |
| Seam clicks, wide windows | 0 | 0.0 (met) | 0.0 (met) | 0.0 (met) |
| Seam clicks, narrow windows | 0 | 0.0 (met) | 0.1 (missed) | 0.0 (met) |

What the runs establish:

1. **Floors.** The block-shifted floor (ceiling A against itself, blocks 100 ms apart) reads
   1.32-1.33 dB STFT / 1.41-1.43 dB envelope MCD; ceiling B against A reads about the same
   (1.31-1.35 / 1.41-1.44). The different-seed floor is *lower* (1.09 / 1.22): on a back-to-back
   ceiling, moving the block grid changes more than redrawing the noise, so the plan's expectation
   that it bounds the others from above is refuted. The primary floor stays the block-shifted one
   (the stream's grid differs from A's, so that disagreement is legitimately in the floor); the seed
   floor is reported next to it.
2. **SOLA timing is about 0.04-0.05 dB.** Reading every stream frame where its block's content is
   (the recorded SOLA offsets) lowers the STFT MCD by 0.04 dB and the envelope MCD by 0.04-0.05 dB
   of today's 1.91-1.93 dB. The raw MCD stays the headline.
3. **The gap is the content encoder's context, not the pitch track** (six clips per EP):
   - oracle F0 in the stream recovers −0.00 to +0.03 dB of MCD against A and −0.003 to +0.009 of
     cosine: the F0 is not what separates the stream's spectrum from the ceiling's;
   - oracle content alone moves the stream toward the whole utterance (+0.37 to +0.38 dB closer to
     the whole ceiling) and therefore *away* from A (0.20 to 0.22 dB), whose content is windowed
     too;
   - with the same oracles in the stream and in A (the residual), the stream comes +0.31 to +0.32 dB
     closer to A: the content features' context accounts for about half of the 0.60-0.64 dB over the
     floor; the rest is the generator's right context (enc, flow and decoder margins) and the joins;
   - the stream's F0 decoded by the ceilings (iv) moves the stream's distance to them by +0.00 to
     +0.01 dB.
   This replaces P3's unproven "the gap is the pitch track".
4. **Pitch.** With RMVPE the analyzer floor (the whole-clip tracker against the analyzer) is 5.7
   cents [3.8, 8.0] (n 18; the whole-clip renderer runs on the CPU path, so the CUDA and DirectML
   runs read the same), below 10, so the fine-pitch limit stays 15 cents. The stream's fed F0 reads
   22.9 cents against the input; ceiling A's own streaming tracker (700 ms of context, 80 ms of
   lookahead) reads 20.5, so the loss is in the streaming F0 path itself (cache geometry, smoothing;
   P3B-F), not only the stream's short right context. GPE pooled over every voiced frame: 1.42 % on
   the GPUs, 3.8 % on the CPU (limit 0.3 %); on CUDA `jfk` (195 of 2471 frames), `ja-short` (60 of
   1609 frames) carry 72% of the 354 gross frames, and 5 of 18 clips have none. Fed V/UV 5.1 % on
   the GPUs, 10.3 % on the CPU (limit 3 %; the whole-clip tracker reads 0.7 % on the GPUs). FCPE
   (the CPU preset's tracker) disagrees with the RMVPE analyzer by 34.5 cents even over the whole
   clip, so the CPU rows measure that disagreement more than streaming; which tracker is right needs
   consensus or EGG truth (T6). The fed-F0 tracks sit on the absolute grid: the best frame lag
   against ceiling A's fed F0 is 0 on 162 of 162 offset tracks, and against the input's whole-clip
   analysis on 151 of 162; the other 11 read one frame (FCPE on three CPU clips, the tone-language
   `yue` at two offsets on both GPUs): the stream's tracker follows fast glides a frame behind the
   whole-clip analysis there, while against ceiling A, whose tracker streams too, the lag is 0, so
   it is a tracker lag that the fine-pitch row counts, not a grid error.
5. **Seams.** Both binomial tests read 0 on 53 of 54 clip runs; the exception: DirectML `ja-long`
   reads 0 wide / 1 narrow (11 of 35 events in narrow windows covering 15%, P = 0.011; 11 of 38
   events in narrow windows covering 15%, P = 0.021 at the offsets that fire). The same clip reads 0
   on the other EPs, so this may be a DirectML-specific seam or a chance excess (each test fires on
   about 2.5 % of clean offsets, and a clip's offsets are correlated); by the pre-registered rule (0
   clicks) the narrow-window row is missed on that EP, and a listening check of the flagged seams is
   the follow-up. v1's statistic read 4 on 1 clip (CUDA ko) where both v2 tests read 0 (the
   excess-over-rate estimate of context 4).
6. **Identity.** Leave one clip out (the other clips' whole-ceiling renditions, speech frames; the
   clip's median over its offsets): CUDA: SIM(stream, set) 0.715 [0.681, 0.745] (n 18), SIM(whole
   ceiling, set) 0.771 [0.738, 0.800] (n 18), delta −0.054 [−0.075, −0.034] (n 18); DirectML:
   SIM(stream, set) 0.714 [0.681, 0.743] (n 18), SIM(whole ceiling, set) 0.771 [0.738, 0.800] (n
   18), delta −0.053 [−0.079, −0.026] (n 18); CPU: SIM(stream, set) 0.721 [0.683, 0.753] (n 18),
   SIM(whole ceiling, set) 0.776 [0.743, 0.804] (n 18), delta −0.053 [−0.082, −0.025] (n 18). At Low
   latency the stream sits below the whole-utterance rendition against the same set by more than the
   −0.02 limit on every EP, so the identity row is missed. The set is offline renditions, so the
   delta could also carry whatever separates any streamed rendition from an offline one, and the
   limit would then be unreachable by construction. The control says it is not
   (`scripts/bench/score_v2/make_control_spec.py`, CUDA corpus, the same 18 clips, set, speech
   frames and exact CAM++; an offline rendition takes the stream's place): ceiling A (the long-shape
   ceiling) against the whole-ceiling set reads SIM 0.765 [0.728, 0.796] against the whole
   ceiling's 0.771, delta **−0.003 [−0.013, +0.006]**; ceiling B (400 ms blocks, 160 ms lookahead)
   reads 0.770 [0.739, 0.796], delta **−0.002 [−0.010, +0.005]** (n 18 each). Two offline
   renditions of the same speech differ by less than 0.01 in this row, the −0.02 limit is
   attainable by a long-context rendition, and the stream's −0.054 is about 18 times the
   offline-rendition difference: it is the streaming engine's, not the metric's. The competitor
   clause of T5 (≥ every competitor's stream at matched latency, T6) compares streams with streams.
   The advisory cosine to ceiling A reads 0.880-0.886 (v1's file and frames: 0.899-0.900).
7. **DNSMOS and onsets.** The stream reads 2.75-2.79 OVRL against 3.07-3.09 for A, 3.03-3.04 for B
   and 3.09-3.12 for the whole ceiling. Its first 200 ms after an utterance onset sit −5.0 to −4.9
   dB under its speech level (A −3.8 to −3.7, B −3.3 to −2.4, whole −2.2 to −2.1): the stream's
   onsets are softer than every ceiling's, and no ceiling mutes them.
8. **P3 rows re-scored** (`rescore` of the 28 saved P3 pairs, `scratchpad/p3b/b/r2/rescore-p3-b4`):
   the six pairs scored by P3's final scorer reproduce every v1 metric to 7e-15; the 18 pairs scored
   before the RMVPE mel end-alignment fix reproduce every non-F0 metric to 4e-16 and differ on the
   F0 metrics, whose analyzer changed (the same CPU audio reads 38.6 instead of 52.6 cents); the
   four early P3-F gate reports predate the final MCD definition (18.2 dB then, 2.0 dB now). The v2
   cosine moves the P3 rows by −0.009 to +0.027 (mean +0.002); identical streams that v1 scored
   0.9290 / 0.9351 and 0.9435 / 0.9514 score 0.9270 / 0.9270 and 0.9483 / 0.9483 (the Rust scorer;
   the NumPy replica of the measure-first step read 0.9263 and 0.9488, its fbank and resampler
   differing in the last digits).
9. **Family voices.** A v2 quality run of P5-A's DDSP-SVC synthetic voice (a seeded random-weight
   voice, so its DNSMOS is low by construction) on CUDA, `en-jfk`, three offsets
   (`scripts/bench/specs/p3b/b/fid-v2-family.template.json`): passed, scored against the family's
   own offline ceiling with the v2 metrics that need no engine probe (cosine v2 0.851, envelope MCD
   3.43 dB, offset-corrected 3.42 dB, output-analyzer fine error 68.8 cents with 15 gross frames,
   binomial clicks 0 / 0); ceiling B, the whole-utterance renderer and the ablations are
   template-only (a spec that asks for them on a family voice is refused with a reason), and a
   family engine has no probe, so it reports no fed F0 and its pitch rows fall back to the output
   analyzer's split. The family tests (`vc::families`) pass on the shared harness with the probe in
   the engine.

## Consequences

- Reports of v2 runs carry both metric sets; thresholds can name v2 metrics once `report.rs` lists
  them (`quality::QUALITY_V2_SCALARS`, a P3B-I0 hotspot request).
- `bench/campplus-exact` is a Sigil export: an unpublished `bench` catalog entry (P3B-I0), staged
  locally until the user publishes it. `bench/campplus` stays selectable (`speaker_model: legacy`)
  for one release.
- Quality runs take about 3× (three offsets) plus the ceilings; the corpus runner
  (`scripts/bench/specs/p3b/b/run_corpus.ps1`) holds the GPU timing lock per batch of clips.
- v1 metrics in a v2 report are medians over the offsets of the v2 renditions (silence skip off),
  so they are not identical to a v1 run's; `rescore` reproduces a saved v1 run's numbers exactly.
- The engine probe (`vc::engine::probe`) is bench-only: inert unless a benchmark configures it, and
  the engine's output is bit-identical with it off or tapping.
- An ablation row is only evidence if its oracle reached the generator: every row checks the probe's
  counters (a row with no replaced frame, or with calls whose content grid missed the oracle's
  origin, says `NOT APPLIED` in its `problems` and in the run's notes), and the run reports
  `ablation_rows_unapplied` (0 = all applied). The tick-level plumbing is covered by the engine
  probe's fixture tests.
