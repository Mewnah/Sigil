# Perf gate: FAILED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_VC_DML_SHARED=1`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-dml-core (7 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m7\dml-shared-call\g-dml-core |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-dml-core / offline | call_ms_p50 | 7 | 12.3197 | 12.079 | -4.37 % | -11.27 .. 0.06 % | +-5.66 % | wilcoxon | NEUTRAL |  |
| g-dml-core / offline | call_ms_p95 | 7 | 15.4764 | 14.5854 | -7.25 % | -18.01 .. -3.15 % | +-7.43 % | wilcoxon | IMPROVED |  |
| g-dml-core / offline | call_ms_p99 | 7 | 17.2292 | 16.2488 | -7.46 % | -27.47 .. 19.14 % | +-23.31 % | wilcoxon | NEUTRAL |  |
| g-dml-core / offline | content_ms_p95 | 7 | 5.7312 | 4.9027 | -19.51 % | -23.82 .. -9.05 % | +-7.38 % | wilcoxon | IMPROVED |  |
| g-dml-core / offline | f0_ms_p95 | 7 | 8.533 | 7.7205 | -9.77 % | -20.11 .. -7.23 % | +-6.44 % | wilcoxon | IMPROVED |  |
| g-dml-core / offline | over_budget_ratio | 7 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-dml-core / offline | vram_engine_mb | 7 | 854 | 842 | -1.41 % | -1.41 .. -1.41 % | +-0 % | wilcoxon | IMPROVED |  |
| g-dml-core / paced | call_ms_p50 | 7 | 22.917 | 21.445 | -6.42 % | -8.28 .. -4.25 % | +-2.02 % | wilcoxon | IMPROVED |  |
| g-dml-core / paced | call_ms_p95 | 7 | 27.4101 | 24.8359 | -7.1 % | -10.59 .. -4.9 % | +-2.85 % | wilcoxon | IMPROVED |  |
| g-dml-core / paced | call_ms_p99 | 7 | 29.3815 | 26.8944 | -9.37 % | -13.74 .. -5.66 % | +-4.04 % | wilcoxon | IMPROVED |  |
| g-dml-core / paced | content_ms_p95 | 7 | 9.979 | 10.5136 | 6.08 % | 1.13 .. 7.26 % | +-3.06 % | wilcoxon | **REGRESSED** |  |
| g-dml-core / paced | f0_ms_p95 | 7 | 14.9681 | 13.1394 | -10.9 % | -13.98 .. -9.27 % | +-2.36 % | wilcoxon | IMPROVED |  |
| g-dml-core / paced | over_budget_ratio | 7 | 0 | 0 | 0 % | -0.13 .. 0.27 % | +-0.2 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-dml-core / paced | vram_engine_mb | 7 | 854 | 842 | -1.41 % | -1.41 .. -1.41 % | +-0 % | wilcoxon | IMPROVED |  |
| g-dml-core / h2-load70 | call_ms_p50 | 7 | 24.9578 | 24.4053 | -1.11 % | -2.05 .. 0.28 % | +-1.16 % | wilcoxon | NEUTRAL |  |
| g-dml-core / h2-load70 | call_ms_p95 | 7 | 28.1353 | 27.7159 | -1.39 % | -3.39 .. -0.29 % | +-1.55 % | wilcoxon | IMPROVED |  |
| g-dml-core / h2-load70 | call_ms_p99 | 7 | 30.2683 | 29.678 | -3.27 % | -5.59 .. 0.77 % | +-3.18 % | wilcoxon | NEUTRAL |  |
| g-dml-core / h2-load70 | content_ms_p95 | 7 | 9.8067 | 12.0122 | 21.92 % | 19.46 .. 28.94 % | +-4.74 % | wilcoxon | **REGRESSED** |  |
| g-dml-core / h2-load70 | f0_ms_p95 | 7 | 14.8431 | 15.3973 | 3.81 % | 1.14 .. 5.1 % | +-1.98 % | wilcoxon | NEUTRAL |  |
| g-dml-core / h2-load70 | over_budget_ratio | 7 | 0.0007 | 0 | 0 % | -0.4 .. 0 % | +-0.2 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-dml-core / h2-load70 | vram_engine_mb | 7 | 854 | 842 | -1.41 % | -1.41 .. -1.41 % | +-0 % | wilcoxon | IMPROVED |  |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-dml-core | A | 2 | offline | state_sm_clock_mhz | 2002.5 | 1781.25 | 12.4 % |
| g-dml-core | B | 1 | offline | state_sm_clock_mhz | 2010 | 1781.25 | 12.8 % |
| g-dml-core | B | 3 | offline | state_speed_index_ms | 2.796 | 3.32 | -15.8 % |
| g-dml-core | B | 5 | offline | state_speed_index_ms | 2.869 | 3.32 | -13.6 % |
| g-dml-core | A | 2 | paced | state_speed_index_ms | 3.556 | 3.18 | 11.8 % |
| g-dml-core | A | 4 | paced | state_speed_index_ms | 3.738 | 3.18 | 17.5 % |
| g-dml-core | A | 5 | paced | state_speed_index_ms | 2.837 | 3.18 | -10.8 % |
| g-dml-core | A | 6 | paced | state_speed_index_ms | 2.789 | 3.18 | -12.3 % |
| g-dml-core | A | 7 | paced | state_speed_index_ms | 3.511 | 3.18 | 10.4 % |
| g-dml-core | B | 2 | paced | state_speed_index_ms | 3.714 | 3.18 | 16.8 % |
| g-dml-core | B | 3 | paced | state_speed_index_ms | 2.706 | 3.18 | -14.9 % |
| g-dml-core | B | 5 | paced | state_speed_index_ms | 2.791 | 3.18 | -12.2 % |
| g-dml-core | A | 1 | h2-load70 | state_speed_index_ms | 3.684 | 3.174 | 16 % |
| g-dml-core | B | 1 | h2-load70 | state_speed_index_ms | 3.541 | 3.174 | 11.6 % |
| g-dml-core | B | 3 | h2-load70 | state_speed_index_ms | 2.707 | 3.174 | -14.7 % |

