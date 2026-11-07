# Perf gate: FAILED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\i0\exe-i0\sigil.exe` with env `SIGIL_CUDA_TUNING=wait=spin`; max regression 5 %, rounds 3..3.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-sat-hard (3 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\gfix\sathard\gate\g-cuda-sat-hard |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-sat-hard / sat-hard | call_ms_p50 | 3 | 7.24 | 7.3569 | -1.08 % | -77.86 .. 157.42 % | +-117.64 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (2 pairs): 0.38 %, -16.7 .. 20.96 %, +-18.83 %, NEUTRAL |
| g-cuda-sat-hard / sat-hard | call_ms_p95 | 3 | 11.0856 | 57.4701 | 418.42 % | -62.95 .. 2826.93 % | +-1444.94 % | t | **INCONCLUSIVE** | 1 pair(s) across GPU clock regimes; without them (2 pairs): 446.68 %, 178.53 .. 973.01 %, +-397.24 %, REGRESSED |
| g-cuda-sat-hard / sat-hard | call_ms_p99 | 3 | 82.9996 | 87.3307 | 1.3 % | -44.3 .. 55.81 % | +-50.05 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (2 pairs): 4.82 %, -32.12 .. 61.85 %, +-46.99 %, NEUTRAL |
| g-cuda-sat-hard / sat-hard | cpu_cores | 3 | 0.0576 | 0.1548 | 191.05 % | 110.72 .. 310.02 % | +-99.65 % | t | **REGRESSED** | 1 pair(s) across GPU clock regimes; without them (2 pairs): 195.38 %, -45.99 .. 1515.33 %, +-780.66 %, INCONCLUSIVE |
| g-cuda-sat-hard / sat-hard | over_budget_ratio | 3 | 0.035 | 0.0761 | 68.89 % | -31.8 .. 343.69 % | +-187.74 % | t | **INCONCLUSIVE** | 1 pair(s) across GPU clock regimes; without them (2 pairs): 108.4 %, -85.58 .. 2911.59 %, +-1498.58 %, INCONCLUSIVE |
| g-cuda-sat-hard / sat-hard | worker_cpu_cores | 3 | 0.0355 | 0.0878 | 149.04 % | 80.31 .. 313.63 % | +-116.66 % | t | **REGRESSED** | 1 pair(s) across GPU clock regimes; without them (2 pairs): 185.98 %, -55.71 .. 1746.79 %, +-901.25 %, INCONCLUSIVE |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cuda-sat-hard | A | 2 | sat-hard | state_sm_clock_mhz | 1275 | 2790 | -54.3 % |
| g-cuda-sat-hard | A | 2 | sat-hard | state_mem_clock_mhz | 5001 | 10251 | -51.2 % |

Warnings:
- g-cuda-sat-hard: 6 gated row(s) have a 95 % interval wider than +-5 % after 3 rounds (worst: sat-hard call_ms_p95 +-1444.94 %): the gate cannot resolve a 5 % change there; size the rounds from an A/A session (-AASummary), or read those rows as unresolved

