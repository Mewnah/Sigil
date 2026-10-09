# P3B-H specs

Load path and stream latency (final shapes, parallel cold start, prewarm, adaptive cushion, SOLA geometry, device block).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-H live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

## Files

| File | What |
|---|---|
| `h.profile.json` | The working profile: `run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/h/h.profile.json`. Rows `h-load-*` (cold loads: `cuda`, `dml`, `cpu`, `quality-cuda` / `quality-dml` with lookahead 20, `xfade10-cuda` / `xfade10-dml` with a 10 ms crossfade) and `h-pace-*` (the reference config paced, under the load simulators, and the 30-minute soak). |
| `load.template.json` | One cold load per process with its wall-clock breakdown (`load_<stage>_ms`, `generator_builds`). Lookahead and crossfade rows use `build: direct` (the factory build refuses `lookahead_ms`), which passes the stream geometry to the engine load like the factory does. |
| `pace.template.json` | The reference config (40 ms engine blocks, RMVPE, Low latency) behind a device block: `BLOCK_FRAMES` (512, or 480 for the 10 ms block), `OUTPUT_CUSHION` (`fixed` / `adaptive`), `CROSSFADE_MS`, the prefill and the runs (`RUNS_JSON`). |
| `fid-h.template.json`, `run_fid.ps1` | The scorer-v2 fidelity corpus (P3B-B's) of one stream geometry: crossfade, lookahead, `SIGIL_VC_SOLA_SEARCH_MS`, `SIGIL_VC_SOLA_SLACK`. |
| `slack_from_timeline.py` | The queue level before each push (the slack to the render deadline) from a run's `*.timeline.csv`. |

## Switches (one exe, ABBA with `-CompareEnv` / `-CompareVars`)

| Switch | Effect |
|---|---|
| `SIGIL_VC_FINAL_SHAPES=0` | The load ignores the request's lookahead and stream geometry (P5's flow: load, `set_lookahead_ms`, `configure_stream`: two generator builds with a lookahead or a crossfade other than 20 ms). |
| `SIGIL_VC_PARALLEL_LOAD=0` / `1` | The cold-start branches (pitch bank, content encoder with its plan, generator, retrieval index) one after the other / beside each other (default on; `1` also turns on P3B-G's `enc` / `flowdec` concurrency). A background hot-switch load is always sequential unless `=1`. |
| `SIGIL_VC_PREWARM=1` | Turns the GPU prewarm on (app start 4 s after the UI is up, and when the panel opens while stopped). Off by default: no end-to-end gain was measured and it holds about 338 MB of VRAM (the CUDA context) from app start. |
| `output_cushion = adaptive` (`OUTPUT_CUSHION=adaptive`) | The adaptive output cushion. `SIGIL_VC_CUSHION_SPREAD=<f>` (default 0.5) and `SIGIL_VC_CUSHION_MARGIN_MS=<ms>` (default 2) are its sweep knobs. |
| `BLOCK_FRAMES=480` | The 10 ms device block (the bench sets the block directly; the app's `device_block = model_divisor` picks it). |
| `SIGIL_VC_SOLA_SEARCH_MS=<ms>`, `SIGIL_VC_SOLA_BIAS=<f>` | The SOLA search window and centre pull of every harness built without an explicit one. |
| `SIGIL_VC_SOLA_SLACK=aligned` | The SOLA tail on the 10 ms frame grid (480 instead of 481 samples at 32 / 40 / 48 kHz with a 20 ms crossfade: one decoded frame less per tick). |

Examples (the voice and clip as in the other workstreams, `-AppData`, `-Slot`, `-Clip`):

    # final shapes: A = P5's flow, B = the load that plans them (ABBA fresh-process pairs)
    $env:SIGIL_VC_FINAL_SHAPES='0'; pwsh scripts/bench/run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/h/h.profile.json -Exe <exe> -CompareExe <exe> -CompareEnv 'SIGIL_VC_FINAL_SHAPES=1' -Only h-load-quality-cuda -Metrics 'engine_load_ms,generator_builds,load_*_ms' ...
    # the adaptive cushion plus the 10 ms device block against the shipped defaults
    pwsh scripts/bench/run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/h/h.profile.json -Exe <exe> -CompareExe <exe> -CompareVars 'OUTPUT_CUSHION=adaptive;BLOCK_FRAMES=480' -Only h-pace-cuda ...
    # the slack to the render deadline of a paced run
    python scripts/bench/specs/p3b/h/slack_from_timeline.py <gate dir>/h-pace-cuda --skip-s 25
