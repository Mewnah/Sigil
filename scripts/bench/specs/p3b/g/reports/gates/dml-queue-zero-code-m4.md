# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g0\sigil.exe` with vars `LEVERS_JSON={"io_binding": "on", "gpu_priority": "on"}`; max regression 5 %, rounds 5..7.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-load-dml (5 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m0\m4-dmlqueue\g-load-dml |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-load-dml / offline | call_ms_p95 | 5 | 19.4487 | 18.5414 | 0.25 % | -23.2 .. 62.09 % | +-42.64 % | t | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (3 pairs): 0.25 %, -29.68 .. 50.23 %, +-39.95 %, NEUTRAL |
| g-load-dml / offline | engine_load_ms | 5 | 3019.2402 | 2706.5362 | -5.41 % | -15.8 .. 7.09 % | +-11.44 % | t | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (3 pairs): -0.38 %, -15.29 .. 19.9 %, +-17.59 %, NEUTRAL |
| g-load-dml / offline | vram_engine_mb | 5 | 854 | 842 | -1.41 % | -1.41 .. -1.41 % | +-0 % | t | IMPROVED | 2 pair(s) across GPU clock regimes; without them (3 pairs): -1.41 %, -1.41 .. -1.41 %, +-0 %, IMPROVED |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-load-dml | A | 4 | offline | state_sm_clock_mhz | 2790 | 1455 | 91.8 % |
| g-load-dml | A | 5 | offline | state_sm_clock_mhz | 2790 | 1455 | 91.8 % |
| g-load-dml | B | 1 | offline | state_sm_clock_mhz | 975 | 1455 | -33 % |
| g-load-dml | B | 5 | offline | state_sm_clock_mhz | 2790 | 1455 | 91.8 % |
| g-load-dml | A | 4 | offline | state_speed_index_ms | 2.801 | 2.545 | 10 % |
| g-load-dml | B | 2 | offline | state_speed_index_ms | 3.066 | 2.545 | 20.5 % |
| g-load-dml | B | 4 | offline | state_speed_index_ms | 2.952 | 2.545 | 16 % |
| g-load-dml | B | 5 | offline | state_speed_index_ms | 2.962 | 2.545 | 16.4 % |

