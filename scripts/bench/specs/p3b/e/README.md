# P3B-E specs

RMVPE right-sized: the plan built once at load (the whole window by default, 32 frames as the opt-in
VRAM saver), never grown (long calls split), exact t32 pruning on 32-frame plans, gated fp16, shared
across voices.

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-E live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

## Switches (one exe can A/B every change, rule R5)

| variable | effect |
| --- | --- |
| `SIGIL_RMVPE_PLAN_FRAMES=32` | the VRAM saver's plan size (32 mel frames, where the pruned t32 graph runs) instead of the default whole-window plan (64 at Low latency, P3's size); any multiple of 32 up to 256 for sweeps |
| `SIGIL_RMVPE_T32=0` | 32-frame plans run the full model, not the pruned t32 graph |
| `SIGIL_RMVPE_FP16=1` / `=0` | fp16 RMVPE on every GPU plan / never (default: only under an explicit `fp16` precision until `FP16_UNDER_AUTO` is flipped) |
| `SIGIL_RMVPE_SHARE=0` | every voice builds its own RMVPE plan |

The default plan is the whole window's (P3's numerics): the 32-frame plan doubles RMVPE's gross
pitch errors on the stream path (`rmvpe_plan_size_f0`), so it is the opt-in VRAM saver
(`RmvpeLoad::vram_saver`), never a default.

## Files

- `onsets.template.json` + `run_onsets.ps1`: measure-first step (1), the F0 calls per minute of
  speech that carry more than 31 audio frames (whole-window calls after a silence skip). One
  offline pass per clip through the engine (direct build: the onset / steady split), under one
  hold of the lock.
- `rmvpe.profile.json` with `rmvpe-gpu.template.json` (offline, paced, the in-process 70 % hammer)
  and `rmvpe-offline.template.json` (offline only): the A/B profile of the switches above, run
  with `run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/e/rmvpe.profile.json`.
- `fid-onsets.template.json` + `run_fidelity.ps1`: whole-clip and onset fidelity (scorer v2, three
  offsets, 4 s leads) of side A against side B (`-EnvB`) on the clips with the most onsets, ABBA
  across the clips, under the lock.
- `run_rig_tests.ps1`: the RMVPE dev-rig tests of `vc/rmvpe.rs` under the lock, with the test
  binary staged next to the ONNX Runtime provider DLLs (CUDA).

## Interleaved runs (P3B-A's gate; A = no variable, B = the variable)

```powershell
# the VRAM saver (32 frames, t32) against the default plan on one exe
pwsh scripts/bench/run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/e/rmvpe.profile.json -Exe <exe-e> `
    -CompareExe <exe-e> -Only rmvpe-dml,rmvpe-cuda-levers -CompareEnv 'SIGIL_RMVPE_PLAN_FRAMES=32' `
    -AppData <appdata> -Slot <appdata>/voice-changer/models/default -Clip <wav>
# fp16 at the default plan size
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe-e> -CompareExe <exe-e> `
    -Only vram-dml,vram-cuda-levers -CompareEnv 'SIGIL_RMVPE_FP16=1' -AppData <appdata> ...
# the whole change against the P3B baseline build
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe-e> -CompareExe scratchpad/p3b/exe-s0/sigil.exe `
    -Only load-dml,load-cuda -Var QOS=none -AppData <appdata> ...
```

The RMVPE-only microbenchmarks and gates are the dev-rig tests in `vc/rmvpe.rs` (hold the lock):
`rmvpe_plan_cost` (32 vs 64 frames, t32, fp16; ABBA rounds), `real_t32_matches_full_on_cpu` and
`rmvpe_fp16_gate` (the stream-path gate; writes `scripts/vc-templates/reports/pitch_rmvpe_fp16_gate.json`).
