# Perf gate: FAILED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe` with env `SIGIL_CUDA_TUNING=wait=spin`; max regression 5 %, rounds 3..3.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-sat (3 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\gfix\sat\gate\g-cuda-sat |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-sat / sat-cpu | call_ms_p50 | 3 | 7.6119 | 7.5218 | -1.18 % | -2.07 .. 0.06 % | +-1.06 % | t | NEUTRAL |  |
| g-cuda-sat / sat-cpu | call_ms_p95 | 3 | 8.6293 | 8.6255 | -1.13 % | -4.22 .. 3.03 % | +-3.63 % | t | NEUTRAL |  |
| g-cuda-sat / sat-cpu | call_ms_p99 | 3 | 9.3073 | 9.3792 | 0.88 % | -0.61 .. 3.17 % | +-1.89 % | t | NEUTRAL |  |
| g-cuda-sat / sat-cpu | cpu_cores | 3 | 0.0655 | 0.1789 | 171.99 % | 153.3 .. 202.45 % | +-24.58 % | t | **REGRESSED** |  |
| g-cuda-sat / sat-cpu | over_budget_ratio | 3 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda-sat / sat-cpu | worker_cpu_cores | 3 | 0.0398 | 0.121 | 199.85 % | 180.16 .. 232.28 % | +-26.06 % | t | **REGRESSED** |  |

Warnings:
- g-cuda-sat: 2 gated row(s) have a 95 % interval wider than +-5 % after 3 rounds (worst: sat-cpu worker_cpu_cores +-26.06 %): the gate cannot resolve a 5 % change there; size the rounds from an A/A session (-AASummary), or read those rows as unresolved

