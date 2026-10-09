# Perf gate: PASSED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g3\sigil.exe` with env `SIGIL_CUDA_TUNING=wait=hybrid`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda-levers-core (9 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m6\w-levers-hybrid\g-cuda-levers-core |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda-levers-core / paced | call_ms_p50 | 9 | 12.0993 | 11.5109 | -0.03 % | -8.41 .. 7.88 % | +-8.15 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): -0.93 %, -16.1 .. 13.77 %, +-14.94 %, NEUTRAL |
| g-cuda-levers-core / paced | call_ms_p95 | 9 | 21.6329 | 15.4926 | 1.41 % | -21.66 .. 14.78 % | +-18.22 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): 6.37 %, -30.98 .. 29.91 %, +-30.45 %, INCONCLUSIVE |
| g-cuda-levers-core / paced | call_ms_p99 | 9 | 25.782 | 25.0771 | -13.41 % | -35.08 .. 13.29 % | +-24.19 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): -2.4 %, -35.08 .. 47.03 %, +-41.06 %, NEUTRAL |
| g-cuda-levers-core / paced | content_ms_p95 | 9 | 8.7132 | 6.0677 | 0.98 % | -32.57 .. 33.1 % | +-32.84 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): 5.48 %, -54.98 .. 85.76 %, +-70.37 %, INCONCLUSIVE |
| g-cuda-levers-core / paced | cpu_cores | 9 | 0.4173 | 0.0446 | -89.24 % | -92.73 .. -88.23 % | +-2.25 % | wilcoxon | IMPROVED | 3 pair(s) across GPU clock regimes; without them (6 pairs): -92.24 %, -93.3 .. -87.62 %, +-2.84 %, IMPROVED |
| g-cuda-levers-core / paced | enc_ms_p95 | 9 | 0.3324 | 0.3713 | 8.87 % | -2.92 .. 21.81 % | +-12.37 % | wilcoxon | **INCONCLUSIVE** | 3 pair(s) across GPU clock regimes; without them (6 pairs): 11.12 %, -7.6 .. 35.72 %, +-21.66 %, INCONCLUSIVE |
| g-cuda-levers-core / paced | f0_ms_p95 | 9 | 11.9401 | 8.4839 | 1.05 % | -25.7 .. 21.82 % | +-23.76 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): 7.34 %, -45.38 .. 55.88 %, +-50.63 %, INCONCLUSIVE |
| g-cuda-levers-core / paced | flowdec_ms_p95 | 9 | 9.683 | 6.716 | -5.19 % | -22.04 .. 7.82 % | +-14.93 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): -2.31 %, -12.46 .. 14.78 %, +-13.62 %, NEUTRAL |
| g-cuda-levers-core / paced | gpu_enc_ms_p50 | 9 | 1.6 | 1.4213 | -2.99 % | -9.47 .. 3.46 % | +-6.47 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): -3.86 %, -13.98 .. 0.04 %, +-7.01 %, NEUTRAL |
| g-cuda-levers-core / paced | gpu_flowdec_ms_p50 | 9 | 3.7908 | 3.4406 | -1.9 % | -12.05 .. 3.21 % | +-7.63 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): -3.33 %, -17.03 .. 11.81 %, +-14.42 %, NEUTRAL |
| g-cuda-levers-core / paced | latency_ms_p99 | 9 | 132.5913 | 132.5913 | 0 % | 0 .. 0.25 % | +-0.13 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): 0 %, 0 .. 0.5 %, +-0.25 %, NEUTRAL |
| g-cuda-levers-core / paced | over_budget_ratio | 9 | 0 | 0 | 0 % | -0.27 .. 0.67 % | +-0.47 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) 3 pair(s) across GPU clock regimes; without them (6 pairs): 0 %, -0.53 .. 1.34 %, +-0.94 %, NEUTRAL |
| g-cuda-levers-core / paced | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 3 pair(s) across GPU clock regimes; without them (6 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda-levers-core / h2-load70 | call_ms_p50 | 9 | 8.1365 | 7.7358 | 0.79 % | -4.24 .. 1.51 % | +-2.87 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | call_ms_p95 | 9 | 15.3046 | 9.708 | -0.45 % | -22.31 .. 3.52 % | +-12.91 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | call_ms_p99 | 9 | 17.8346 | 10.2949 | -1.32 % | -26.48 .. 3.52 % | +-15 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | content_ms_p95 | 9 | 4.5324 | 3.457 | -5.97 % | -13.95 .. 3.6 % | +-8.77 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | cpu_cores | 9 | 0.2341 | 0.0518 | -75.97 % | -89.96 .. -73.89 % | +-8.04 % | wilcoxon | IMPROVED |  |
| g-cuda-levers-core / h2-load70 | enc_ms_p95 | 9 | 1.1571 | 1.1745 | -3.59 % | -8.86 .. 5.16 % | +-7.01 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | f0_ms_p95 | 9 | 7.9932 | 5.3207 | -0.52 % | -20.82 .. 2.38 % | +-11.6 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | flowdec_ms_p95 | 9 | 6.399 | 4.1483 | -1.3 % | -21.89 .. 3.58 % | +-12.73 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | gpu_enc_ms_p50 | 9 | 0.9236 | 0.9175 | -0.67 % | -7.74 .. 3.3 % | +-5.52 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | gpu_flowdec_ms_p50 | 9 | 2.2057 | 2.0849 | -0.22 % | -5.54 .. 1.13 % | +-3.33 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | gpu_load_duty | 9 | 0.7 | 0.7 | -0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | latency_ms_p99 | 9 | 142.5913 | 142.5913 | 0 % | -3.57 .. 3.7 % | +-3.64 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h2-load70 | over_budget_ratio | 9 | 0 | 0 | 0 % | -0.13 .. 0.27 % | +-0.2 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda-levers-core / h2-load70 | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0.02 % | +-0.01 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | call_ms_p50 | 9 | 8.4948 | 8.1561 | -2.79 % | -10.6 .. 5.26 % | +-7.93 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | call_ms_p95 | 9 | 14.9899 | 15.6197 | 2.7 % | -18.38 .. 25.35 % | +-21.87 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | call_ms_p99 | 9 | 17.2227 | 18.2139 | -0.89 % | -19.95 .. 29.68 % | +-24.81 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | content_ms_p95 | 9 | 4.3424 | 5.4368 | 0.31 % | -10.38 .. 22.51 % | +-16.45 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | cpu_cores | 9 | 0.238 | 0.0503 | -76.9 % | -89.57 .. -73.97 % | +-7.8 % | wilcoxon | IMPROVED |  |
| g-cuda-levers-core / h1-sim | enc_ms_p95 | 9 | 1.2803 | 1.2482 | 1.8 % | -8.07 .. 11.02 % | +-9.55 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | f0_ms_p95 | 9 | 7.566 | 8.1888 | 2.34 % | -15.56 .. 25.01 % | +-20.28 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | flowdec_ms_p95 | 9 | 5.8885 | 6.716 | 2.33 % | -16.98 .. 24.68 % | +-20.83 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | gpu_enc_ms_p50 | 9 | 0.9574 | 0.9431 | 0.19 % | -5.63 .. 2.07 % | +-3.85 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | gpu_flowdec_ms_p50 | 9 | 2.4095 | 2.1748 | -0.85 % | -12.13 .. 3.54 % | +-7.83 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | gpu_load_duty | 9 | 0.7 | 0.7 | 0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | latency_ms_p99 | 9 | 132.5913 | 132.5913 | 0 % | -0.23 .. 0.23 % | +-0.23 % | wilcoxon | NEUTRAL |  |
| g-cuda-levers-core / h1-sim | over_budget_ratio | 9 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda-levers-core / h1-sim | syncs_per_call_mean | 9 | 2.9993 | 3 | 0.02 % | -0 .. 0.14 % | +-0.07 % | wilcoxon | NEUTRAL |  |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cuda-levers-core | A | 1 | paced | state_sm_clock_mhz | 1717.5 | 1218.75 | 40.9 % |
| g-cuda-levers-core | A | 2 | paced | state_sm_clock_mhz | 1087.5 | 1218.75 | -10.8 % |
| g-cuda-levers-core | A | 3 | paced | state_sm_clock_mhz | 1410 | 1218.75 | 15.7 % |
| g-cuda-levers-core | A | 4 | paced | state_sm_clock_mhz | 1387.5 | 1218.75 | 13.8 % |
| g-cuda-levers-core | A | 5 | paced | state_sm_clock_mhz | 2325 | 1218.75 | 90.8 % |
| g-cuda-levers-core | A | 6 | paced | state_sm_clock_mhz | 885 | 1218.75 | -27.4 % |
| g-cuda-levers-core | A | 7 | paced | state_sm_clock_mhz | 870 | 1218.75 | -28.6 % |
| g-cuda-levers-core | A | 8 | paced | state_sm_clock_mhz | 780 | 1218.75 | -36 % |
| g-cuda-levers-core | A | 9 | paced | state_sm_clock_mhz | 840 | 1218.75 | -31.1 % |
| g-cuda-levers-core | B | 1 | paced | state_sm_clock_mhz | 1357.5 | 1218.75 | 11.4 % |
| g-cuda-levers-core | B | 2 | paced | state_sm_clock_mhz | 1350 | 1218.75 | 10.8 % |
| g-cuda-levers-core | B | 3 | paced | state_sm_clock_mhz | 1470 | 1218.75 | 20.6 % |
| g-cuda-levers-core | B | 4 | paced | state_sm_clock_mhz | 1792.5 | 1218.75 | 47.1 % |
| g-cuda-levers-core | B | 5 | paced | state_sm_clock_mhz | 1440 | 1218.75 | 18.2 % |
| g-cuda-levers-core | B | 6 | paced | state_sm_clock_mhz | 780 | 1218.75 | -36 % |
| g-cuda-levers-core | B | 7 | paced | state_sm_clock_mhz | 855 | 1218.75 | -29.8 % |
| g-cuda-levers-core | B | 8 | paced | state_sm_clock_mhz | 870 | 1218.75 | -28.6 % |
| g-cuda-levers-core | B | 9 | paced | state_sm_clock_mhz | 802.5 | 1218.75 | -34.2 % |
| g-cuda-levers-core | A | 5 | paced | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda-levers-core | A | 6 | h2-load70 | state_sm_clock_mhz | 1200 | 2790 | -57 % |
| g-cuda-levers-core | A | 7 | h2-load70 | state_sm_clock_mhz | 1200 | 2790 | -57 % |
| g-cuda-levers-core | A | 8 | h2-load70 | state_sm_clock_mhz | 1215 | 2790 | -56.5 % |
| g-cuda-levers-core | A | 9 | h2-load70 | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | B | 6 | h2-load70 | state_sm_clock_mhz | 1155 | 2790 | -58.6 % |
| g-cuda-levers-core | B | 7 | h2-load70 | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | B | 8 | h2-load70 | state_sm_clock_mhz | 1215 | 2790 | -56.5 % |
| g-cuda-levers-core | B | 9 | h2-load70 | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | A | 6 | h1-sim | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | A | 7 | h1-sim | state_sm_clock_mhz | 1200 | 2790 | -57 % |
| g-cuda-levers-core | A | 8 | h1-sim | state_sm_clock_mhz | 1200 | 2790 | -57 % |
| g-cuda-levers-core | A | 9 | h1-sim | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | B | 6 | h1-sim | state_sm_clock_mhz | 1200 | 2790 | -57 % |
| g-cuda-levers-core | B | 7 | h1-sim | state_sm_clock_mhz | 1117.5 | 2790 | -59.9 % |
| g-cuda-levers-core | B | 8 | h1-sim | state_sm_clock_mhz | 1215 | 2790 | -56.5 % |
| g-cuda-levers-core | B | 9 | h1-sim | state_sm_clock_mhz | 1185 | 2790 | -57.5 % |
| g-cuda-levers-core | B | 7 | h1-sim | state_mem_clock_mhz | 5001 | 10251 | -51.2 % |

