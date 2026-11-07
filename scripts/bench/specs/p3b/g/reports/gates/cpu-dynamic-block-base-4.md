# Perf gate: PASSED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe` with vars `DYNAMIC_BLOCK_BASE_JSON=4`; max regression 5 %, rounds 5..9.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cpu (7 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m4\cpu-dbb4\g-cpu |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cpu / offline | call_ms_p95 | 7 | 96.8672 | 95.1018 | -0.85 % | -6.75 .. 5.46 % | +-6.11 % | wilcoxon | NEUTRAL | 7 pair(s) across GPU clock regimes |
| g-cpu / offline | cpu_cores | 7 | 4.6803 | 4.6868 | 0.07 % | -0.77 .. 1.32 % | +-1.05 % | wilcoxon | NEUTRAL | 7 pair(s) across GPU clock regimes |
| g-cpu / offline | rtf_p50 | 7 | 0.6858 | 0.6939 | 3.76 % | -6.07 .. 8.98 % | +-7.53 % | wilcoxon | NEUTRAL | 7 pair(s) across GPU clock regimes |
| g-cpu / offline | rtf_p95 | 7 | 0.8072 | 0.7925 | -0.85 % | -6.75 .. 5.46 % | +-6.11 % | wilcoxon | NEUTRAL | 7 pair(s) across GPU clock regimes |
| g-cpu / offline | rtf_p99 | 7 | 0.8398 | 0.8141 | -0.88 % | -8.75 .. 4.99 % | +-6.87 % | wilcoxon | NEUTRAL | 7 pair(s) across GPU clock regimes |
| g-cpu / paced | call_ms_p95 | 7 | 93.0865 | 95.6228 | 3.27 % | -0.82 .. 7.88 % | +-4.35 % | wilcoxon | NEUTRAL | 6 pair(s) across GPU clock regimes |
| g-cpu / paced | cpu_cores | 7 | 3.2107 | 3.2929 | 2.56 % | -0.27 .. 7.57 % | +-3.92 % | wilcoxon | NEUTRAL | 6 pair(s) across GPU clock regimes |
| g-cpu / paced | rtf_p50 | 7 | 0.6774 | 0.6936 | 2.54 % | -1.28 .. 7.68 % | +-4.48 % | wilcoxon | NEUTRAL | 6 pair(s) across GPU clock regimes |
| g-cpu / paced | rtf_p95 | 7 | 0.7757 | 0.7969 | 3.27 % | -0.82 .. 7.88 % | +-4.35 % | wilcoxon | NEUTRAL | 6 pair(s) across GPU clock regimes |
| g-cpu / paced | rtf_p99 | 7 | 0.8181 | 0.8404 | 4.84 % | -2.13 .. 9.01 % | +-5.57 % | wilcoxon | NEUTRAL | 6 pair(s) across GPU clock regimes |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cpu | A | 1 | offline | state_sm_clock_mhz | 210 | 675 | -68.9 % |
| g-cpu | A | 2 | offline | state_sm_clock_mhz | 885 | 675 | 31.1 % |
| g-cpu | A | 3 | offline | state_sm_clock_mhz | 930 | 675 | 37.8 % |
| g-cpu | A | 4 | offline | state_sm_clock_mhz | 255 | 675 | -62.2 % |
| g-cpu | A | 5 | offline | state_sm_clock_mhz | 780 | 675 | 15.6 % |
| g-cpu | A | 6 | offline | state_sm_clock_mhz | 285 | 675 | -57.8 % |
| g-cpu | B | 4 | offline | state_sm_clock_mhz | 870 | 675 | 28.9 % |
| g-cpu | B | 5 | offline | state_sm_clock_mhz | 480 | 675 | -28.9 % |
| g-cpu | B | 6 | offline | state_sm_clock_mhz | 555 | 675 | -17.8 % |
| g-cpu | B | 7 | offline | state_sm_clock_mhz | 960 | 675 | 42.2 % |
| g-cpu | A | 1 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 3 | offline | state_mem_clock_mhz | 5001 | 810 | 517.4 % |
| g-cpu | A | 4 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 6 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 3 | offline | state_mem_clock_mhz | 607.5 | 810 | -25 % |
| g-cpu | A | 1 | paced | state_sm_clock_mhz | 847.5 | 667.5 | 27 % |
| g-cpu | A | 4 | paced | state_sm_clock_mhz | 780 | 667.5 | 16.9 % |
| g-cpu | A | 5 | paced | state_sm_clock_mhz | 442.5 | 667.5 | -33.7 % |
| g-cpu | A | 6 | paced | state_sm_clock_mhz | 375 | 667.5 | -43.8 % |
| g-cpu | A | 7 | paced | state_sm_clock_mhz | 600 | 667.5 | -10.1 % |
| g-cpu | B | 1 | paced | state_sm_clock_mhz | 480 | 667.5 | -28.1 % |
| g-cpu | B | 2 | paced | state_sm_clock_mhz | 900 | 667.5 | 34.8 % |
| g-cpu | B | 3 | paced | state_sm_clock_mhz | 855 | 667.5 | 28.1 % |
| g-cpu | B | 4 | paced | state_sm_clock_mhz | 840 | 667.5 | 25.8 % |
| g-cpu | B | 7 | paced | state_sm_clock_mhz | 1005 | 667.5 | 50.6 % |
| g-cpu | A | 6 | paced | state_mem_clock_mhz | 405 | 810 | -50 % |

