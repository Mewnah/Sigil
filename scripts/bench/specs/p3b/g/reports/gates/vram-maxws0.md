# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe` with env `SIGIL_CUDA_TUNING=maxws=0`; max regression 5 %, rounds 3..5.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-vram-cuda-levers (3 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m1\m3-vram-maxws0\g-vram-cuda-levers |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-vram-cuda-levers / offline | call_ms_p50 | 3 | 6.5786 | 6.5279 | -0.97 % | -9.67 .. 5.29 % | +-7.48 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p95 | 3 | 7.8006 | 7.3764 | -5.09 % | -8.26 .. -0.26 % | +-4 % | t | IMPROVED |  |
| g-vram-cuda-levers / offline | call_ms_p99 | 3 | 8.2716 | 7.7258 | -4 % | -13.57 .. 4.61 % | +-9.09 % | t | NEUTRAL |  |
| g-vram-cuda-levers / offline | vram_engine_mb | 3 | 1380 | 1380 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL |  |

