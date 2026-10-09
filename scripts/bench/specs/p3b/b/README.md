# P3B-B specs

Bench trust II (fidelity scorers v2, ceilings, oracle ablations; ADR 0009).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-B live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

| file | what |
|---|---|
| `fid-v2.template.json` | One clip scored with `scoring.analyzer: v2` (the ablations through `scoring.analyzer: v2+ablations` or `SIGIL_BENCH_ABLATIONS`): three offsets, ceilings A (long shape, the primary), B and whole, the floors, the saved WAVs for `rescore`. Placeholders `${NAME}`, `${CLIP}`, `${SLOT}`, `${CONTENT_ENCODER}`, `${EP}`, `${PITCH}`, `${PRESET}`, `${EXTRA_MS}`, `${LOOKAHEAD_MS}`, `${ANALYZER}`, `${REFERENCE_CLIPS_JSON}`. |
| `fid-v2-family.template.json` | Scorer v2 on a family voice (e.g. P5-A's DDSP-SVC synthetic voice): the v2 metrics that need no engine probe against the family's own offline ceiling, three offsets. Checks that the v2 path serves non-template voices (ceiling B, the whole-utterance renderer, fed F0 and the ablations are template-only). Placeholders `${NAME}`, `${CLIP}`, `${SLOT}`, `${EP}`. |
| `run_corpus.ps1` | The corpus run: the template over every local clip (18 by default) under the GPU lock, a `rescore` of the stream / whole-ceiling pairs (the leave-one-out target-speaker similarity), and `scripts/bench/score_v2/aggregate.py` (means with bootstrap CIs, the ablation table, the T5 v2 rows). Each clip's folder also gets `load.json` (rustc processes and CPU load when its run started: other agents may be compiling). The lock records the script's pid and is released in `finally`, so a killed run's lock is stale after 30 minutes; `-Batch N` (default 4) runs N clips per hold of the lock (other agents' timing runs hold it 15-30 minutes and the queue alternates owners, so one clip per hold leaves a corpus waiting between 1-2 minute clips). Clips already scored in `-Out` are kept, so a corpus can be run in passes: `-Analyzer v2+ablations -Clips 'a.wav;b.wav' -SkipRescore` for the ablation clips, then `-Analyzer v2` over the rest (the rescore and summary then cover every clip in the folder). |

The scorer-v2 options (spec fields frozen by P3B-S0):
- `scoring.analyzer`: `v1` (default, the P3 scorer), `v2`, or `v2+ablations` (v2 plus the oracle
  ablations at the first offset; `SIGIL_BENCH_ABLATIONS=1` on a `v2` spec is the same switch, which
  `run_corpus.ps1 -Analyzer v2+ablations` sets).
- `ceiling.mode`: `long_shape` (default), `b` or `whole`: the primary ceiling, against which the v1
  metrics are computed. A v2 run of a template voice renders all three (`b` and `whole` need a
  template voice and v2). The ceilings `ceiling.mode` names, and `whole` when the ablations run,
  must render, else the run fails (a `whole` ceiling needs a `lead_s` of at least 0.4 s and a speech
  region of at most 60 s; the run checks both before it loads the engine). The ceilings the run
  renders besides are optional: when one cannot be (region over 60 s, short lead, no fp32 content
  encoder for the CPU EP, a recording that does not match the input) the run notes why and leaves
  that rendition's rows out, and the floors that need it.
- `fidelity.offsets_s`: the leads the clip is scored at (median and spread; v2 only). Empty (the
  default) scores `lead_s`, `lead_s + 0.27` and `lead_s + 0.61`: three anchored-noise realizations,
  because one clip's fine pitch error swings by a factor of two between two leads. Naming fewer than
  three is allowed, with a note in the report.
- `fidelity.reference_clips`: the target-speaker set (WAVs of the target voice, at least 3 s of
  speech each; v2 only).
- `rvc.content_variant`: pins the ceilings' content-encoder variant (`fp32` / `fp16` / `int8`).
- run mode `rescore`: saved stream / ceiling WAV pairs scored again (see
  `scripts/bench/score_v2/make_rescore_spec.py`, which adds every offset of a v2 run). The offsets
  of one run are one clip: its row is the median over them, the run's metrics the mean (and
  bootstrap CI) over the clips, and the target-speaker set of a pair leaves its whole clip out.

CPU-only NumPy prototypes and checks of the v2 scorers on saved WAV pairs are in
`scripts/bench/score_v2/` (`measure_p3.py` is the measure-first step of the workstream).
`make_control_spec.py` writes the identity control of ADR 0009 over a corpus run's saved WAVs: a
`rescore` spec in which an offline ceiling (A or B) takes the stream's place against the
whole-ceiling target set, so the identity row's delta can be read for a rendition that is not
streamed (A reads -0.003, B -0.002, the stream -0.054).
