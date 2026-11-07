# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_GPU_PROCESS_PRIORITY=high`; max regression 5 %, rounds 5..9.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-oop-dml (5 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m7\prio-dml\g-oop-dml |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-oop-dml / h2-load70-oop | call_ms_p95 | 5 | 27.9386 | 28.169 | 1.07 % | -1.05 .. 2.5 % | +-1.77 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | call_ms_p99 | 5 | 29.2722 | 29.8681 | 2.88 % | -0.67 .. 4.3 % | +-2.49 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | content_ms_p99 | 5 | 10.4267 | 10.6322 | 0.79 % | -1.48 .. 3.96 % | +-2.72 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | enc_ms_p99 | 5 | 4.8909 | 4.8962 | 1.37 % | -1.87 .. 4.22 % | +-3.05 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | f0_ms_p99 | 5 | 15.9917 | 15.9904 | 1.92 % | -0.94 .. 4.88 % | +-2.91 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | flowdec_ms_p99 | 5 | 11.7333 | 12.1991 | 0.93 % | -1.13 .. 5.81 % | +-3.47 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | gpu_load_duty | 5 | 0.7 | 0.7 | -0 % | -0 .. 0 % | +-0 % | t | NEUTRAL |  |
| g-oop-dml / h2-load70-oop | over_budget_ratio | 5 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-oop-dml | A | 2 | h2-load70-oop | state_mem_clock_mhz | 5001 | 10501 | -52.4 % |
| g-oop-dml | B | 2 | h2-load70-oop | state_mem_clock_mhz | 5001 | 10501 | -52.4 % |
| g-oop-dml | A | 5 | h2-load70-oop | state_speed_index_ms | 3.361 | 3.026 | 11.1 % |

