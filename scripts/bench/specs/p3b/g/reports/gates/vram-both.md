# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe` with env `SIGIL_CUDA_TUNING=maxws=0,algo=heuristic`; max regression 5 %, rounds 3..5.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-vram-cuda-levers (3 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m1\m3-vram-both\g-vram-cuda-levers |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-vram-cuda-levers / offline | call_ms_p50 | 3 | 6.9406 | 6.9384 | -0.03 % | -1.08 .. 1.34 % | +-1.21 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p95 | 3 | 7.7795 | 7.8201 | 0.65 % | 0.09 .. 1.31 % | +-0.61 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p99 | 3 | 8.2169 | 8.3503 | 1.3 % | -0.21 .. 3.52 % | +-1.87 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | vram_engine_mb | 3 | 1380 | 1380 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL |  |

