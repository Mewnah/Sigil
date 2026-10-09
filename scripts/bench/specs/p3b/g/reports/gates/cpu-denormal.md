# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe` with env `SIGIL_VC_DENORMAL_ZERO=1`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cpu (9 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m4\cpu-denormal\g-cpu |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cpu / offline | call_ms_p95 | 9 | 97.5074 | 100.2387 | 1.3 % | -3.51 .. 8.51 % | +-6.01 % | wilcoxon | NEUTRAL | 5 pair(s) across GPU clock regimes; without them (4 pairs): 4.01 %, -10.48 .. 17.97 %, +-14.22 %, NEUTRAL |
| g-cpu / offline | cpu_cores | 9 | 4.655 | 4.6481 | -0.03 % | -1.2 .. 0.75 % | +-0.98 % | wilcoxon | NEUTRAL | 5 pair(s) across GPU clock regimes; without them (4 pairs): -0.44 %, -1.96 .. 1.16 %, +-1.56 %, NEUTRAL |
| g-cpu / offline | rtf_p50 | 9 | 0.7106 | 0.7379 | 4.2 % | -3.99 .. 13.66 % | +-8.82 % | wilcoxon | NEUTRAL | 5 pair(s) across GPU clock regimes; without them (4 pairs): 7.28 %, -10.19 .. 29.69 %, +-19.94 %, INCONCLUSIVE |
| g-cpu / offline | rtf_p95 | 9 | 0.8126 | 0.8353 | 1.3 % | -3.51 .. 8.51 % | +-6.01 % | wilcoxon | NEUTRAL | 5 pair(s) across GPU clock regimes; without them (4 pairs): 4.01 %, -10.48 .. 17.97 %, +-14.22 %, NEUTRAL |
| g-cpu / offline | rtf_p99 | 9 | 0.8492 | 0.8796 | 3.58 % | -2.48 .. 12.02 % | +-7.25 % | wilcoxon | NEUTRAL | 5 pair(s) across GPU clock regimes; without them (4 pairs): 5.35 %, -9.19 .. 20.12 %, +-14.66 %, INCONCLUSIVE |
| g-cpu / paced | call_ms_p95 | 9 | 97.559 | 99.5726 | 1.99 % | -3.63 .. 14.49 % | +-9.06 % | wilcoxon | NEUTRAL | 4 pair(s) across GPU clock regimes; without them (5 pairs): 4.31 %, -1.81 .. 15.47 %, +-8.64 %, NEUTRAL |
| g-cpu / paced | cpu_cores | 9 | 3.2982 | 3.3155 | 0.91 % | -5.79 .. 12.45 % | +-9.12 % | wilcoxon | NEUTRAL | 4 pair(s) across GPU clock regimes; without them (5 pairs): 0.91 %, -4.56 .. 12.09 %, +-8.33 %, NEUTRAL |
| g-cpu / paced | rtf_p50 | 9 | 0.6992 | 0.7258 | 1.4 % | -6.2 .. 16.09 % | +-11.15 % | wilcoxon | NEUTRAL | 4 pair(s) across GPU clock regimes; without them (5 pairs): 1.4 %, -5.74 .. 14.12 %, +-9.93 %, NEUTRAL |
| g-cpu / paced | rtf_p95 | 9 | 0.813 | 0.8298 | 1.99 % | -3.63 .. 14.49 % | +-9.06 % | wilcoxon | NEUTRAL | 4 pair(s) across GPU clock regimes; without them (5 pairs): 4.31 %, -1.81 .. 15.47 %, +-8.64 %, NEUTRAL |
| g-cpu / paced | rtf_p99 | 9 | 0.8737 | 0.8904 | 3.66 % | -1.7 .. 16.78 % | +-9.24 % | wilcoxon | NEUTRAL | 4 pair(s) across GPU clock regimes; without them (5 pairs): 6.69 %, -1.58 .. 15.41 %, +-8.49 %, INCONCLUSIVE |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cpu | A | 1 | offline | state_sm_clock_mhz | 435 | 652.5 | -33.3 % |
| g-cpu | A | 2 | offline | state_sm_clock_mhz | 900 | 652.5 | 37.9 % |
| g-cpu | A | 3 | offline | state_sm_clock_mhz | 300 | 652.5 | -54 % |
| g-cpu | A | 4 | offline | state_sm_clock_mhz | 457.5 | 652.5 | -29.9 % |
| g-cpu | A | 5 | offline | state_sm_clock_mhz | 840 | 652.5 | 28.7 % |
| g-cpu | A | 6 | offline | state_sm_clock_mhz | 735 | 652.5 | 12.6 % |
| g-cpu | A | 8 | offline | state_sm_clock_mhz | 210 | 652.5 | -67.8 % |
| g-cpu | B | 1 | offline | state_sm_clock_mhz | 210 | 652.5 | -67.8 % |
| g-cpu | B | 2 | offline | state_sm_clock_mhz | 870 | 652.5 | 33.3 % |
| g-cpu | B | 3 | offline | state_sm_clock_mhz | 217.5 | 652.5 | -66.7 % |
| g-cpu | B | 4 | offline | state_sm_clock_mhz | 990 | 652.5 | 51.7 % |
| g-cpu | B | 5 | offline | state_sm_clock_mhz | 840 | 652.5 | 28.7 % |
| g-cpu | B | 6 | offline | state_sm_clock_mhz | 405 | 652.5 | -37.9 % |
| g-cpu | B | 7 | offline | state_sm_clock_mhz | 900 | 652.5 | 37.9 % |
| g-cpu | B | 8 | offline | state_sm_clock_mhz | 210 | 652.5 | -67.8 % |
| g-cpu | B | 9 | offline | state_sm_clock_mhz | 765 | 652.5 | 17.2 % |
| g-cpu | A | 1 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 3 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 8 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 9 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 1 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 3 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 6 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 8 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 1 | paced | state_sm_clock_mhz | 210 | 566.25 | -62.9 % |
| g-cpu | A | 2 | paced | state_sm_clock_mhz | 855 | 566.25 | 51 % |
| g-cpu | A | 3 | paced | state_sm_clock_mhz | 247.5 | 566.25 | -56.3 % |
| g-cpu | A | 5 | paced | state_sm_clock_mhz | 375 | 566.25 | -33.8 % |
| g-cpu | A | 7 | paced | state_sm_clock_mhz | 750 | 566.25 | 32.5 % |
| g-cpu | A | 9 | paced | state_sm_clock_mhz | 855 | 566.25 | 51 % |
| g-cpu | B | 1 | paced | state_sm_clock_mhz | 787.5 | 566.25 | 39.1 % |
| g-cpu | B | 2 | paced | state_sm_clock_mhz | 825 | 566.25 | 45.7 % |
| g-cpu | B | 4 | paced | state_sm_clock_mhz | 352.5 | 566.25 | -37.7 % |
| g-cpu | B | 5 | paced | state_sm_clock_mhz | 330 | 566.25 | -41.7 % |
| g-cpu | B | 6 | paced | state_sm_clock_mhz | 675 | 566.25 | 19.2 % |
| g-cpu | B | 7 | paced | state_sm_clock_mhz | 660 | 566.25 | 16.6 % |
| g-cpu | B | 8 | paced | state_sm_clock_mhz | 210 | 566.25 | -62.9 % |
| g-cpu | B | 9 | paced | state_sm_clock_mhz | 772.5 | 566.25 | 36.4 % |
| g-cpu | A | 1 | paced | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 3 | paced | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 5 | paced | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | A | 8 | paced | state_mem_clock_mhz | 607.5 | 810 | -25 % |
| g-cpu | B | 5 | paced | state_mem_clock_mhz | 607.5 | 810 | -25 % |
| g-cpu | B | 8 | paced | state_mem_clock_mhz | 405 | 810 | -50 % |

