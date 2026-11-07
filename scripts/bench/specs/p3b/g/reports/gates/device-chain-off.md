# Perf gate: FAILED

Executable: `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`
Interleaved A/B (ABBA): A = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe`, B = `%USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g2\sigil.exe` with env `SIGIL_VC_DEVICE_CHAIN=0`; max regression 5 %, rounds 5..9.
Profile: `%USERPROFILE%\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-core (5 rounds) | PASS (exit 0) | %USERPROFILE%\AppData\Local\Temp\claude\<project>\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m3\chain-off\g-cuda-core |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-core / offline | call_ms_p50 | 5 | 7.0543 | 7.1154 | 0.56 % | 0.31 .. 0.89 % | +-0.29 % | t | NEUTRAL |  |
| g-cuda-core / offline | call_ms_p95 | 5 | 7.6997 | 7.7261 | 0.34 % | -0.86 .. 1.14 % | +-1 % | t | NEUTRAL |  |
| g-cuda-core / offline | call_ms_p99 | 5 | 7.8825 | 7.9543 | -0.45 % | -1.55 .. 1.6 % | +-1.57 % | t | NEUTRAL |  |
| g-cuda-core / offline | content_ms_p95 | 5 | 2.8406 | 2.7825 | 0.81 % | -2.71 .. 2.41 % | +-2.56 % | t | NEUTRAL |  |
| g-cuda-core / offline | cpu_cores | 5 | 1.3122 | 1.3125 | 0.02 % | -1.17 .. 0.8 % | +-0.99 % | t | NEUTRAL |  |
| g-cuda-core / offline | enc_ms_p95 | 5 | 0.2845 | 1.2593 | 298.24 % | 236.97 .. 362.12 % | +-62.58 % | t | **REGRESSED** |  |
| g-cuda-core / offline | f0_ms_p95 | 5 | 4.2117 | 4.2334 | 0.36 % | -0.88 .. 1.09 % | +-0.99 % | t | NEUTRAL |  |
| g-cuda-core / offline | flowdec_ms_p95 | 5 | 3.341 | 2.6607 | -21.17 % | -23.33 .. -19.72 % | +-1.8 % | t | IMPROVED |  |
| g-cuda-core / offline | gpu_enc_ms_p50 | 5 | 0.8376 | 0.8407 | 0.37 % | -0.77 .. 1.72 % | +-1.25 % | t | NEUTRAL |  |
| g-cuda-core / offline | gpu_flowdec_ms_p50 | 5 | 2.1197 | 2.132 | 0.43 % | 0.37 .. 0.57 % | +-0.1 % | t | NEUTRAL |  |
| g-cuda-core / offline | over_budget_ratio | 5 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda-core / offline | syncs_per_call_mean | 5 | 3 | 4 | 33.33 % | 33.33 .. 33.33 % | +-0 % | t | **REGRESSED** |  |
| g-cuda-core / paced | call_ms_p50 | 5 | 18.908 | 18.8699 | 0.1 % | -35.92 .. 23.54 % | +-29.73 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0.03 %, -46.03 .. 38.27 %, +-42.15 %, NEUTRAL |
| g-cuda-core / paced | call_ms_p95 | 5 | 20.9478 | 21.0479 | 0.11 % | -1.28 .. 1.04 % | +-1.16 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0.23 %, -0.59 .. 1.08 %, +-0.83 %, NEUTRAL |
| g-cuda-core / paced | call_ms_p99 | 5 | 21.6636 | 21.7748 | 0.21 % | -0.44 .. 0.82 % | +-0.63 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0.24 %, -0.75 .. 1.11 %, +-0.93 %, NEUTRAL |
| g-cuda-core / paced | content_ms_p95 | 5 | 8.3136 | 8.3131 | -0.9 % | -1.59 .. -0.05 % | +-0.77 % | t | IMPROVED | 1 pair(s) across GPU clock regimes; without them (4 pairs): -1.07 %, -1.99 .. 0.24 %, +-1.11 %, NEUTRAL |
| g-cuda-core / paced | cpu_cores | 5 | 0.6315 | 0.63 | -0.07 % | -16.16 .. 8.96 % | +-12.56 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0.05 %, -21.71 .. 14.11 %, +-17.91 %, NEUTRAL |
| g-cuda-core / paced | enc_ms_p95 | 5 | 0.2717 | 3.5632 | 1122.8 % | 715.35 .. 1334.47 % | +-309.56 % | t | **REGRESSED** | 1 pair(s) across GPU clock regimes; without them (4 pairs): 1150.77 %, 751.84 .. 1475.87 %, +-362.01 %, REGRESSED |
| g-cuda-core / paced | f0_ms_p95 | 5 | 11.3774 | 11.3443 | -0.53 % | -1.51 .. 0.34 % | +-0.93 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): -0.65 %, -1.97 .. 0.76 %, +-1.36 %, NEUTRAL |
| g-cuda-core / paced | flowdec_ms_p95 | 5 | 9.7332 | 7.0748 | -26.49 % | -27.5 .. -25.34 % | +-1.08 % | t | IMPROVED | 1 pair(s) across GPU clock regimes; without them (4 pairs): -26.07 %, -27.9 .. -24.76 %, +-1.57 %, IMPROVED |
| g-cuda-core / paced | gpu_enc_ms_p50 | 5 | 2.4248 | 2.4863 | 0.37 % | -12.42 .. 10.23 % | +-11.33 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): -0.63 %, -17.45 .. 15.7 %, +-16.57 %, NEUTRAL |
| g-cuda-core / paced | gpu_flowdec_ms_p50 | 5 | 5.9914 | 6.0066 | 0.05 % | -12.21 .. 6.48 % | +-9.34 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0.13 %, -16.53 .. 10.28 %, +-13.41 %, NEUTRAL |
| g-cuda-core / paced | latency_ms_p99 | 5 | 132.5913 | 132.5913 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (4 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda-core / paced | over_budget_ratio | 5 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) 1 pair(s) across GPU clock regimes; without them (4 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda-core / paced | syncs_per_call_mean | 5 | 3 | 4 | 33.33 % | 33.33 .. 33.33 % | +-0 % | t | **REGRESSED** | 1 pair(s) across GPU clock regimes; without them (4 pairs): 33.33 %, 33.33 .. 33.33 %, +-0 %, REGRESSED |
| g-cuda-core / h2-load70 | call_ms_p50 | 5 | 16.5185 | 16.7731 | 2.24 % | 0.28 .. 3.3 % | +-1.51 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | call_ms_p95 | 5 | 18.3175 | 18.4518 | 2.1 % | -0.15 .. 3.81 % | +-1.98 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | call_ms_p99 | 5 | 19.2465 | 19.6561 | 2.13 % | -2.22 .. 3.84 % | +-3.03 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | content_ms_p95 | 5 | 6.6242 | 6.5008 | -1.16 % | -3.48 .. 1.68 % | +-2.58 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | cpu_cores | 5 | 0.4876 | 0.4882 | 0.73 % | -1.62 .. 2.76 % | +-2.19 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | enc_ms_p95 | 5 | 1.2465 | 3.5135 | 173.91 % | 50.02 .. 225.9 % | +-87.94 % | t | **REGRESSED** |  |
| g-cuda-core / h2-load70 | f0_ms_p95 | 5 | 10.0142 | 9.9761 | -0.38 % | -2.47 .. 1.29 % | +-1.88 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | flowdec_ms_p95 | 5 | 8.499 | 6.3566 | -24.81 % | -27.89 .. -18.39 % | +-4.75 % | t | IMPROVED |  |
| g-cuda-core / h2-load70 | gpu_enc_ms_p50 | 5 | 1.9272 | 1.9543 | 0.46 % | -3.83 .. 9.04 % | +-6.43 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | gpu_flowdec_ms_p50 | 5 | 5.2035 | 5.1891 | 0.73 % | -0.98 .. 1.46 % | +-1.22 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | gpu_load_duty | 5 | 0.7 | 0.7 | -0 % | -0 .. 0 % | +-0 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | latency_ms_p99 | 5 | 143.258 | 143.258 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL |  |
| g-cuda-core / h2-load70 | over_budget_ratio | 5 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | t | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda-core / h2-load70 | syncs_per_call_mean | 5 | 3 | 3.9993 | 33.31 % | 32.02 .. 33.67 % | +-0.82 % | t | **REGRESSED** |  |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cuda-core | A | 4 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda-core | A | 5 | offline | state_sm_clock_mhz | 2790 | 2535 | 10.1 % |
| g-cuda-core | B | 4 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda-core | B | 5 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda-core | A | 4 | paced | state_sm_clock_mhz | 1335 | 843.75 | 58.2 % |
| g-cuda-core | A | 5 | paced | state_sm_clock_mhz | 1807.5 | 843.75 | 114.2 % |
| g-cuda-core | B | 4 | paced | state_sm_clock_mhz | 1717.5 | 843.75 | 103.6 % |
| g-cuda-core | B | 5 | paced | state_sm_clock_mhz | 1792.5 | 843.75 | 112.4 % |
| g-cuda-core | A | 5 | paced | state_mem_clock_mhz | 810 | 5001 | -83.8 % |
| g-cuda-core | A | 4 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda-core | A | 5 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda-core | B | 4 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda-core | B | 5 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |

