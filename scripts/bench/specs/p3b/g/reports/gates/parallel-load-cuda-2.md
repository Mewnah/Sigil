# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_VC_PARALLEL_LOAD=1`; max regression 5 %, rounds 9..9.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-load-cuda (9 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m6\pload-cuda2\g-load-cuda |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-load-cuda / offline | call_ms_p95 | 9 | 8.1535 | 8.3178 | 0.08 % | -0.96 .. 6.36 % | +-3.66 % | wilcoxon | NEUTRAL |  |
| g-load-cuda / offline | engine_load_ms | 9 | 2065.1315 | 2006.2591 | -3.97 % | -25.13 .. -2.26 % | +-11.44 % | wilcoxon | IMPROVED |  |
| g-load-cuda / offline | vram_engine_mb | 9 | 1374 | 1374 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |

