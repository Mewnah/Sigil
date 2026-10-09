# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe` with env `SIGIL_CUDA_TUNING=wait=hybrid`; max regression 5 %, rounds 3..5.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-silent (3 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m6\w-silent\g-cuda-silent |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-silent / silent | cpu_cores | 3 | 0.0088 | 0.0081 | -8.08 % | -9.93 .. -5.58 % | +-2.18 % | t | IMPROVED |  |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cuda-silent | A | 1 | silent | state_sm_clock_mhz | 637.5 | 727.5 | -12.4 % |
| g-cuda-silent | A | 3 | silent | state_sm_clock_mhz | 915 | 727.5 | 25.8 % |
| g-cuda-silent | B | 1 | silent | state_sm_clock_mhz | 592.5 | 727.5 | -18.6 % |
| g-cuda-silent | B | 1 | silent | state_mem_clock_mhz | 405 | 810 | -50 % |

Warnings:
- g-cuda-silent: 1 round(s) reused from an earlier run (-Resume)

