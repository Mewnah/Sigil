# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe` with vars `CPU_ARENA_JSON=true`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cpu (5 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m4\cpu-arena\g-cpu |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cpu / offline | call_ms_p95 | 5 | 93.9111 | 89.6524 | -4.53 % | -5.53 .. -2.54 % | +-1.5 % | t | IMPROVED | 2 pair(s) across GPU clock regimes; without them (3 pairs): -4.78 %, -5.36 .. -4.18 %, +-0.59 %, IMPROVED |
| g-cpu / offline | cpu_cores | 5 | 4.6561 | 4.6752 | 0.48 % | -0.06 .. 0.74 % | +-0.4 % | t | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (3 pairs): 0.48 %, -0.08 .. 0.94 %, +-0.51 %, NEUTRAL |
| g-cpu / offline | rtf_p50 | 5 | 0.6977 | 0.6558 | -5.88 % | -10.7 .. -2.62 % | +-4.04 % | t | IMPROVED | 2 pair(s) across GPU clock regimes; without them (3 pairs): -4.71 %, -7.02 .. -3.18 %, +-1.92 %, IMPROVED |
| g-cpu / offline | rtf_p95 | 5 | 0.7826 | 0.7471 | -4.53 % | -5.53 .. -2.54 % | +-1.5 % | t | IMPROVED | 2 pair(s) across GPU clock regimes; without them (3 pairs): -4.78 %, -5.36 .. -4.18 %, +-0.59 %, IMPROVED |
| g-cpu / offline | rtf_p99 | 5 | 0.8039 | 0.7857 | -3.6 % | -8.7 .. 3.01 % | +-5.86 % | t | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (3 pairs): -6.5 %, -11.75 .. 1.57 %, +-6.66 %, NEUTRAL |
| g-cpu / paced | call_ms_p95 | 5 | 95.1969 | 90.2428 | -7.38 % | -9.61 .. -4.02 % | +-2.8 % | t | IMPROVED | 3 pair(s) across GPU clock regimes; without them (2 pairs): -5.92 %, -22.92 .. 14.83 %, +-18.88 %, NEUTRAL |
| g-cpu / paced | cpu_cores | 5 | 3.2827 | 3.1223 | -4.63 % | -6.16 .. -3.49 % | +-1.33 % | t | IMPROVED | 3 pair(s) across GPU clock regimes; without them (2 pairs): -5.83 %, -13.77 .. 2.84 %, +-8.31 %, NEUTRAL |
| g-cpu / paced | rtf_p50 | 5 | 0.6965 | 0.6605 | -4.59 % | -6.65 .. -1.1 % | +-2.77 % | t | IMPROVED | 3 pair(s) across GPU clock regimes; without them (2 pairs): -5.86 %, -14.22 .. 3.31 %, +-8.77 %, NEUTRAL |
| g-cpu / paced | rtf_p95 | 5 | 0.7933 | 0.752 | -7.38 % | -9.61 .. -4.02 % | +-2.8 % | t | IMPROVED | 3 pair(s) across GPU clock regimes; without them (2 pairs): -5.92 %, -22.92 .. 14.83 %, +-18.88 %, NEUTRAL |
| g-cpu / paced | rtf_p99 | 5 | 0.8673 | 0.7941 | -9.42 % | -15.61 .. -2.24 % | +-6.68 % | t | IMPROVED | 3 pair(s) across GPU clock regimes; without them (2 pairs): -7.2 %, -36.44 .. 35.48 %, +-35.96 %, NEUTRAL |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cpu | A | 1 | offline | state_sm_clock_mhz | 330 | 735 | -55.1 % |
| g-cpu | A | 3 | offline | state_sm_clock_mhz | 870 | 735 | 18.4 % |
| g-cpu | A | 4 | offline | state_sm_clock_mhz | 915 | 735 | 24.5 % |
| g-cpu | B | 1 | offline | state_sm_clock_mhz | 442.5 | 735 | -39.8 % |
| g-cpu | B | 2 | offline | state_sm_clock_mhz | 622.5 | 735 | -15.3 % |
| g-cpu | B | 4 | offline | state_sm_clock_mhz | 825 | 735 | 12.2 % |
| g-cpu | A | 1 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 1 | offline | state_mem_clock_mhz | 405 | 810 | -50 % |
| g-cpu | B | 2 | offline | state_mem_clock_mhz | 607.5 | 810 | -25 % |
| g-cpu | A | 1 | paced | state_sm_clock_mhz | 210 | 656.25 | -68 % |
| g-cpu | A | 3 | paced | state_sm_clock_mhz | 780 | 656.25 | 18.9 % |
| g-cpu | A | 4 | paced | state_sm_clock_mhz | 465 | 656.25 | -29.1 % |
| g-cpu | A | 5 | paced | state_sm_clock_mhz | 780 | 656.25 | 18.9 % |
| g-cpu | B | 1 | paced | state_sm_clock_mhz | 450 | 656.25 | -31.4 % |
| g-cpu | B | 4 | paced | state_sm_clock_mhz | 855 | 656.25 | 30.3 % |
| g-cpu | B | 5 | paced | state_sm_clock_mhz | 780 | 656.25 | 18.9 % |
| g-cpu | A | 1 | paced | state_mem_clock_mhz | 405 | 607.5 | -33.3 % |
| g-cpu | A | 2 | paced | state_mem_clock_mhz | 810 | 607.5 | 33.3 % |
| g-cpu | A | 3 | paced | state_mem_clock_mhz | 810 | 607.5 | 33.3 % |
| g-cpu | A | 4 | paced | state_mem_clock_mhz | 405 | 607.5 | -33.3 % |
| g-cpu | A | 5 | paced | state_mem_clock_mhz | 810 | 607.5 | 33.3 % |
| g-cpu | B | 1 | paced | state_mem_clock_mhz | 405 | 607.5 | -33.3 % |
| g-cpu | B | 2 | paced | state_mem_clock_mhz | 405 | 607.5 | -33.3 % |
| g-cpu | B | 3 | paced | state_mem_clock_mhz | 405 | 607.5 | -33.3 % |
| g-cpu | B | 4 | paced | state_mem_clock_mhz | 810 | 607.5 | 33.3 % |
| g-cpu | B | 5 | paced | state_mem_clock_mhz | 810 | 607.5 | 33.3 % |

