# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe`; max regression 5 %, rounds 2..2.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-paced (2 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m0\m1-cuda-paced\g-cuda-paced |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-paced / paced | call_ms_p95 | 2 | 22.3622 | 22.1549 | -0.93 % | -4.94 .. 3.24 % | +-4.09 % | t | NEUTRAL |  |
| g-cuda-paced / paced | call_ms_p99 | 2 | 24.1233 | 24.1339 | 0.03 % | -5.44 .. 5.82 % | +-5.63 % | t | NEUTRAL |  |
| g-cuda-paced / paced | cpu_cores | 2 | 0.6248 | 0.6232 | -0.28 % | -15.38 .. 17.51 % | +-16.44 % | t | NEUTRAL |  |
| g-cuda-paced / paced | latency_ms_p95 | 2 | 132.5913 | 132.5913 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL |  |
| g-cuda-paced / paced | rtf_p95 | 2 | 0.5591 | 0.5539 | -0.93 % | -4.94 .. 3.24 % | +-4.09 % | t | NEUTRAL |  |
| g-cuda-paced / paced | vram_engine_mb | 2 | 1374 | 1374 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL |  |

