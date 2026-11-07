# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe` with env `SIGIL_CUDA_TUNING=algo=heuristic`; max regression 5 %, rounds 3..5.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-vram-cuda-levers (3 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m1\m3-vram-heur\g-vram-cuda-levers |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-vram-cuda-levers / offline | call_ms_p50 | 3 | 7.1911 | 7.125 | -1.33 % | -6.52 .. 2.82 % | +-4.67 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p95 | 3 | 8.1851 | 8.2209 | 0.25 % | -2.12 .. 1.92 % | +-2.02 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p99 | 3 | 8.5683 | 8.7058 | 1.44 % | -0.13 .. 2.57 % | +-1.35 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | vram_engine_mb | 3 | 1380 | 1380 | 0 % | -0.26 .. 0.16 % | +-0.21 % | t | NEUTRAL |  |

