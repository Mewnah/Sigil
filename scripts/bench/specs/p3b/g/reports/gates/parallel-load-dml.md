# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_VC_PARALLEL_LOAD=1`; max regression 5 %, rounds 7..11.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-load-dml (7 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m7\pload-dml\g-load-dml |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-load-dml / offline | call_ms_p95 | 7 | 9.7254 | 9.117 | -0.59 % | -26.39 .. 10.4 % | +-18.39 % | wilcoxon | NEUTRAL |  |
| g-load-dml / offline | engine_load_ms | 7 | 2688.7738 | 2470.6237 | -7.68 % | -15.25 .. -4.09 % | +-5.58 % | wilcoxon | IMPROVED |  |
| g-load-dml / offline | vram_engine_mb | 7 | 854 | 854 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |

