# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_CUDA_TUNING=shrink=1`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-vram-cuda-levers (7 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m6\shrink\g-vram-cuda-levers |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-vram-cuda-levers / offline | call_ms_p50 | 7 | 6.6217 | 6.1318 | 0.89 % | -11.57 .. 6.19 % | +-8.88 % | wilcoxon | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p95 | 7 | 7.5497 | 7.3841 | 2.1 % | -8.8 .. 7.45 % | +-8.13 % | wilcoxon | NEUTRAL |  |
| g-vram-cuda-levers / offline | call_ms_p99 | 7 | 8.0038 | 7.9289 | 2.52 % | -7.78 .. 8 % | +-7.89 % | wilcoxon | NEUTRAL |  |
| g-vram-cuda-levers / offline | vram_engine_mb | 7 | 1380 | 1340 | -2.9 % | -2.9 .. -2.75 % | +-0.07 % | wilcoxon | IMPROVED |  |

Warnings:
- g-vram-cuda-levers: 2 round(s) reused from an earlier run (-Resume)

