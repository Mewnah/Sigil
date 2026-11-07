# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_GPU_PROCESS_PRIORITY=high`; max regression 5 %, rounds 5..9.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-oop-cuda (5 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m7\prio-cuda\g-oop-cuda |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-oop-cuda / h2-load70-oop | call_ms_p95 | 5 | 26.8933 | 27.3262 | 1.57 % | -0.12 .. 2.45 % | +-1.28 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | call_ms_p99 | 5 | 29.2796 | 29.4485 | 1.12 % | -1.57 .. 2.94 % | +-2.25 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | content_ms_p99 | 5 | 12.0938 | 12.6886 | 3.54 % | -2.19 .. 6.89 % | +-4.54 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | enc_ms_p99 | 5 | 0.476 | 0.4867 | -0.44 % | -7.79 .. 13.58 % | +-10.69 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | f0_ms_p99 | 5 | 16.6764 | 16.9263 | 0.99 % | 0.09 .. 2.27 % | +-1.09 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | flowdec_ms_p99 | 5 | 13.7975 | 14.1424 | 0.51 % | -1.12 .. 3.72 % | +-2.42 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | gpu_load_duty | 5 | 0.7 | 0.7 | -0 % | -0 .. 0 % | +-0 % | t | NEUTRAL |  |
| g-oop-cuda / h2-load70-oop | over_budget_ratio | 5 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-oop-cuda | A | 5 | h2-load70-oop | state_mem_clock_mhz | 5001 | 10251 | -51.2 % |
| g-oop-cuda | B | 5 | h2-load70-oop | state_mem_clock_mhz | 5001 | 10251 | -51.2 % |

