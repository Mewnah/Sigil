# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_VC_PARALLEL_LOAD=1`; max regression 5 %, rounds 7..11.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-load-cuda (7 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m7\pload-cuda\g-load-cuda |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-load-cuda / offline | call_ms_p95 | 7 | 9.5115 | 9.107 | -1.76 % | -3.44 .. -0.03 % | +-1.7 % | wilcoxon | IMPROVED |  |
| g-load-cuda / offline | engine_load_ms | 7 | 2158.8565 | 2025.7602 | -7.63 % | -28.54 .. 2.77 % | +-15.65 % | wilcoxon | NEUTRAL |  |
| g-load-cuda / offline | vram_engine_mb | 7 | 1374 | 1374 | 0 % | -0.07 .. 0 % | +-0.04 % | wilcoxon | NEUTRAL |  |

Warnings:
- g-load-cuda: 4 round(s) reused from an earlier run (-Resume)

