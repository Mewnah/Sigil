# Perf gate: FAILED

Executable: `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`
Interleaved A/B (ABBA): A = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe`, B = `C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\exe-g1\sigil.exe` with env `SIGIL_CUDA_TUNING=wait=hybrid`; max regression 5 %, rounds 5..9.
Profile: `C:\Users\chris\Desktop\Sigil\scripts\bench\specs\p3b\g\g.profile.json`

| Spec | Harness | Report |
|---|---|---|
| g-cuda (9 rounds) | PASS (exit 0) | C:\Users\chris\AppData\Local\Temp\claude\C--Users-chris-Desktop-Sigil\4d8dbcba-171d-4482-ad29-f10e34d08945\scratchpad\p3b\g\m1\w-hybrid\g-cuda |

| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| g-cuda / offline | call_ms_p50 | 9 | 7.1438 | 7.299 | 1.27 % | 0.99 .. 3.54 % | +-1.27 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | call_ms_p95 | 9 | 7.8352 | 7.9854 | 1.93 % | 0.35 .. 5.07 % | +-2.36 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | call_ms_p99 | 9 | 8.0933 | 8.2904 | 1.94 % | -0.06 .. 4.65 % | +-2.36 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | content_ms_p95 | 9 | 2.7893 | 2.8167 | 1.9 % | 0.19 .. 6.64 % | +-3.23 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | cpu_cores | 9 | 1.3063 | 0.2152 | -83.6 % | -83.99 .. -81.18 % | +-1.4 % | wilcoxon | IMPROVED |  |
| g-cuda / offline | enc_ms_p95 | 9 | 0.2828 | 0.2942 | 0.28 % | -1.62 .. 9.53 % | +-5.58 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | f0_ms_p95 | 9 | 4.3452 | 4.444 | 1.45 % | 0.27 .. 3.3 % | +-1.51 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | flowdec_ms_p95 | 9 | 3.4057 | 3.4418 | 0.42 % | -0.46 .. 4.78 % | +-2.62 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | gpu_enc_ms_p50 | 9 | 0.8387 | 0.8455 | -0.12 % | -1.45 .. 0.65 % | +-1.05 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | gpu_flowdec_ms_p50 | 9 | 2.0818 | 2.1125 | -0.11 % | -0.59 .. 1.06 % | +-0.83 % | wilcoxon | NEUTRAL |  |
| g-cuda / offline | over_budget_ratio | 9 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda / offline | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |
| g-cuda / paced | call_ms_p50 | 9 | 18.9252 | 18.7078 | 0.44 % | -5.34 .. 3.07 % | +-4.21 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.44 %, -2.2 .. 1.25 %, +-1.72 %, NEUTRAL |
| g-cuda / paced | call_ms_p95 | 9 | 22.1781 | 22.1593 | -0.54 % | -1.05 .. 1.73 % | +-1.39 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -0.54 %, -1.05 .. -0.02 %, +-0.51 %, IMPROVED |
| g-cuda / paced | call_ms_p99 | 9 | 24.2133 | 24.3689 | 0.06 % | -0.71 .. 0.54 % | +-0.62 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.06 %, -0.79 .. 0.54 %, +-0.66 %, NEUTRAL |
| g-cuda / paced | content_ms_p95 | 9 | 9.157 | 9.0348 | -1.18 % | -4.98 .. 4.95 % | +-4.96 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -1.18 %, -2.93 .. 0.49 %, +-1.71 %, NEUTRAL |
| g-cuda / paced | cpu_cores | 9 | 0.6286 | 0.0421 | -92.89 % | -93.39 .. -89.74 % | +-1.83 % | wilcoxon | IMPROVED | 2 pair(s) across GPU clock regimes; without them (7 pairs): -93.3 %, -93.42 .. -89.74 %, +-1.84 %, IMPROVED |
| g-cuda / paced | enc_ms_p95 | 9 | 0.2741 | 0.2743 | -2.56 % | -6.27 .. 2.25 % | +-4.26 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -4.64 %, -7.3 .. 3.36 %, +-5.33 %, NEUTRAL |
| g-cuda / paced | f0_ms_p95 | 9 | 12.6895 | 12.5196 | -0.64 % | -2.79 .. 2.82 % | +-2.81 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -0.64 %, -2.77 .. 0.03 %, +-1.4 %, NEUTRAL |
| g-cuda / paced | flowdec_ms_p95 | 9 | 9.729 | 9.7285 | -0.13 % | -1.96 .. 0.66 % | +-1.31 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -0.13 %, -1.98 .. 0.33 %, +-1.15 %, NEUTRAL |
| g-cuda / paced | gpu_enc_ms_p50 | 9 | 2.261 | 2.0808 | -1.67 % | -8.55 .. 1.67 % | +-5.11 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -1.67 %, -7.95 .. 0.11 %, +-4.03 %, NEUTRAL |
| g-cuda / paced | gpu_flowdec_ms_p50 | 9 | 5.994 | 5.8829 | -0.17 % | -7.37 .. 2.97 % | +-5.17 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -0.17 %, -1.86 .. 0.4 %, +-1.13 %, NEUTRAL |
| g-cuda / paced | latency_ms_p99 | 9 | 132.5913 | 132.5913 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / paced | over_budget_ratio | 9 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / paced | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h2-load70 | call_ms_p50 | 9 | 16.3711 | 16.6536 | 0.76 % | -0.77 .. 2.57 % | +-1.67 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | call_ms_p95 | 9 | 18.4972 | 18.2576 | -0.45 % | -2.92 .. 0.49 % | +-1.7 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | call_ms_p99 | 9 | 19.6214 | 19.369 | -1.17 % | -2.97 .. 0.1 % | +-1.54 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | content_ms_p95 | 9 | 6.5182 | 6.4358 | -3.2 % | -5.51 .. -1.01 % | +-2.25 % | wilcoxon | IMPROVED |  |
| g-cuda / h2-load70 | cpu_cores | 9 | 0.4488 | 0.0502 | -88.81 % | -90.24 .. -74.86 % | +-7.69 % | wilcoxon | IMPROVED |  |
| g-cuda / h2-load70 | enc_ms_p95 | 9 | 1.1898 | 1.2298 | 1.04 % | -5.74 .. 4.49 % | +-5.12 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | f0_ms_p95 | 9 | 10.3352 | 10.1892 | -1.11 % | -2.31 .. -0.08 % | +-1.11 % | wilcoxon | IMPROVED |  |
| g-cuda / h2-load70 | flowdec_ms_p95 | 9 | 8.5823 | 8.5338 | -0.45 % | -1.48 .. 0.41 % | +-0.95 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | gpu_enc_ms_p50 | 9 | 1.9011 | 1.9005 | -1.4 % | -2.28 .. -0.14 % | +-1.07 % | wilcoxon | IMPROVED |  |
| g-cuda / h2-load70 | gpu_flowdec_ms_p50 | 9 | 4.8835 | 5.1343 | 0.23 % | -1.06 .. 4.52 % | +-2.79 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | gpu_load_duty | 9 | 0.7 | 0.7 | 0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | latency_ms_p99 | 9 | 142.5913 | 142.5913 | 0 % | -3.79 .. 3.7 % | +-3.75 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70 | over_budget_ratio | 9 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) |
| g-cuda / h2-load70 | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL |  |
| g-cuda / h2-load70-oop | call_ms_p50 | 9 | 21.852 | 21.784 | 0.39 % | -0.22 .. 46.18 % | +-23.2 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 0.37 %, -0.31 .. 6.03 %, +-3.17 %, NEUTRAL |
| g-cuda / h2-load70-oop | call_ms_p95 | 9 | 23.877 | 24.0129 | 1.44 % | -0.24 .. 44.53 % | +-22.38 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 1.2 %, -1.47 .. 11.99 %, +-6.73 %, NEUTRAL |
| g-cuda / h2-load70-oop | call_ms_p99 | 9 | 26.1378 | 25.49 | 0.99 % | -9.17 .. 12.99 % | +-11.08 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 1.27 %, -2.13 .. 14.29 %, +-8.21 %, NEUTRAL |
| g-cuda / h2-load70-oop | content_ms_p95 | 9 | 9.7146 | 9.4382 | 1.12 % | -1.3 .. 35.91 % | +-18.6 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 0.99 %, -1.5 .. 19.49 %, +-10.5 %, NEUTRAL |
| g-cuda / h2-load70-oop | cpu_cores | 9 | 0.7227 | 0.0468 | -93.37 % | -93.87 .. -83.16 % | +-5.36 % | wilcoxon | IMPROVED | 1 pair(s) across GPU clock regimes; without them (8 pairs): -93.39 %, -93.9 .. -83.03 %, +-5.44 %, IMPROVED |
| g-cuda / h2-load70-oop | enc_ms_p95 | 9 | 0.327 | 0.3531 | 9.06 % | 1.78 .. 19.15 % | +-8.69 % | wilcoxon | **REGRESSED** | 1 pair(s) across GPU clock regimes; without them (8 pairs): 8.69 %, 1.08 .. 19.83 %, +-9.37 %, REGRESSED |
| g-cuda / h2-load70-oop | f0_ms_p95 | 9 | 14.1356 | 14.1456 | 1.98 % | -0.57 .. 46.9 % | +-23.74 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 1.81 %, -0.79 .. 15.31 %, +-8.05 %, NEUTRAL |
| g-cuda / h2-load70-oop | flowdec_ms_p95 | 9 | 10.9803 | 11.0842 | 1.86 % | -0.77 .. 49.39 % | +-25.08 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 1.48 %, -1.13 .. 8.89 %, +-5.01 %, NEUTRAL |
| g-cuda / h2-load70-oop | gpu_enc_ms_p50 | 9 | 1.9707 | 1.9358 | -1.77 % | -4.66 .. 28.32 % | +-16.49 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): -2.48 %, -6.31 .. 6.72 %, +-6.51 %, NEUTRAL |
| g-cuda / h2-load70-oop | gpu_flowdec_ms_p50 | 9 | 6.6898 | 6.7205 | 0.58 % | -0.83 .. 59.92 % | +-30.37 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 0.1 %, -0.96 .. 3.04 %, +-2 %, NEUTRAL |
| g-cuda / h2-load70-oop | gpu_load_duty | 9 | 0.7 | 0.7 | 0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): -0 %, -0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h2-load70-oop | latency_ms_p99 | 9 | 143.258 | 143.258 | 0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h2-load70-oop | over_budget_ratio | 9 | 0 | 0 | 0 % | -0.13 .. 0.27 % | +-0.2 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) 1 pair(s) across GPU clock regimes; without them (8 pairs): 0 %, -0.13 .. 0.13 %, +-0.13 %, NEUTRAL |
| g-cuda / h2-load70-oop | syncs_per_call_mean | 9 | 3 | 3 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 1 pair(s) across GPU clock regimes; without them (8 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h1-sim | call_ms_p50 | 9 | 16.8864 | 16.2156 | 1.23 % | -33.2 .. 49.19 % | +-41.19 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 1.23 %, -0.78 .. 2 %, +-1.39 %, NEUTRAL |
| g-cuda / h1-sim | call_ms_p95 | 9 | 18.7499 | 18.5591 | 0.99 % | -32.75 .. 50.78 % | +-41.77 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.99 %, -1.91 .. 2.56 %, +-2.23 %, NEUTRAL |
| g-cuda / h1-sim | call_ms_p99 | 9 | 20.0339 | 20.0134 | 1.71 % | -33.29 .. 58.36 % | +-45.82 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 1.71 %, -1.78 .. 2.77 %, +-2.27 %, NEUTRAL |
| g-cuda / h1-sim | content_ms_p95 | 9 | 6.6646 | 6.4069 | 0.22 % | -35.01 .. 51.22 % | +-43.11 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.22 %, -0.62 .. 1.22 %, +-0.92 %, NEUTRAL |
| g-cuda / h1-sim | cpu_cores | 9 | 0.4627 | 0.0474 | -88.86 % | -90.06 .. -74.74 % | +-7.66 % | wilcoxon | IMPROVED | 2 pair(s) across GPU clock regimes; without them (7 pairs): -89.46 %, -90.41 .. -74.77 %, +-7.82 %, IMPROVED |
| g-cuda / h1-sim | enc_ms_p95 | 9 | 1.2666 | 1.2468 | 0.79 % | -9.74 .. 17.11 % | +-13.42 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.79 %, -2.03 .. 4.11 %, +-3.07 %, NEUTRAL |
| g-cuda / h1-sim | f0_ms_p95 | 9 | 10.3297 | 10.293 | 0.89 % | -33.34 .. 51.89 % | +-42.61 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.89 %, -0.83 .. 2.55 %, +-1.69 %, NEUTRAL |
| g-cuda / h1-sim | flowdec_ms_p95 | 9 | 8.7209 | 8.4936 | 0.41 % | -37.56 .. 58.53 % | +-48.05 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.41 %, -1.79 .. 1.41 %, +-1.6 %, NEUTRAL |
| g-cuda / h1-sim | gpu_enc_ms_p50 | 9 | 2.1002 | 1.8555 | -1.02 % | -34.22 .. 41.43 % | +-37.83 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -1.02 %, -3.96 .. -0.11 %, +-1.93 %, IMPROVED |
| g-cuda / h1-sim | gpu_flowdec_ms_p50 | 9 | 5.2306 | 4.8722 | 0.5 % | -36.93 .. 53.55 % | +-45.24 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0.5 %, -1.56 .. 0.56 %, +-1.06 %, NEUTRAL |
| g-cuda / h1-sim | gpu_load_duty | 9 | 0.7 | 0.7 | -0 % | -0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): -0 %, -0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h1-sim | latency_ms_p99 | 9 | 143.258 | 143.258 | 0 % | -3.79 .. 3.94 % | +-3.87 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, -0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h1-sim | over_budget_ratio | 9 | 0 | 0 | 0 % | 0 .. 0 % | +-0 % | wilcoxon | NEUTRAL | linear (values <= 0; relative to max(|A median|, 0.25)) 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, 0 .. 0 %, +-0 %, NEUTRAL |
| g-cuda / h1-sim | syncs_per_call_mean | 9 | 3 | 3 | 0 % | -0.05 .. 0 % | +-0.02 % | wilcoxon | NEUTRAL | 2 pair(s) across GPU clock regimes; without them (7 pairs): 0 %, -0.02 .. 0 %, +-0.01 %, NEUTRAL |

Runs more than 10 % off the session median of their GPU / host state (stamped into their report.json as `gate_state_flags`):

| Spec | Side | Round | Run | State | Value | Session median | Off |
|---|---|---|---|---|---|---|---|
| g-cuda | A | 6 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda | A | 7 | offline | state_sm_clock_mhz | 2790 | 2535 | 10.1 % |
| g-cuda | A | 8 | offline | state_sm_clock_mhz | 2790 | 2535 | 10.1 % |
| g-cuda | A | 9 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda | B | 6 | offline | state_sm_clock_mhz | 2797.5 | 2535 | 10.4 % |
| g-cuda | B | 7 | offline | state_sm_clock_mhz | 2790 | 2535 | 10.1 % |
| g-cuda | B | 8 | offline | state_sm_clock_mhz | 2805 | 2535 | 10.7 % |
| g-cuda | B | 9 | offline | state_sm_clock_mhz | 2790 | 2535 | 10.1 % |
| g-cuda | A | 6 | paced | state_sm_clock_mhz | 1807.5 | 862.5 | 109.6 % |
| g-cuda | A | 7 | paced | state_sm_clock_mhz | 1470 | 862.5 | 70.4 % |
| g-cuda | A | 8 | paced | state_sm_clock_mhz | 1305 | 862.5 | 51.3 % |
| g-cuda | A | 9 | paced | state_sm_clock_mhz | 1552.5 | 862.5 | 80 % |
| g-cuda | B | 6 | paced | state_sm_clock_mhz | 1357.5 | 862.5 | 57.4 % |
| g-cuda | B | 7 | paced | state_sm_clock_mhz | 1650 | 862.5 | 91.3 % |
| g-cuda | B | 8 | paced | state_sm_clock_mhz | 1785 | 862.5 | 107 % |
| g-cuda | B | 9 | paced | state_sm_clock_mhz | 1305 | 862.5 | 51.3 % |
| g-cuda | A | 6 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | A | 7 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | A | 8 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | A | 9 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | B | 6 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | B | 7 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | B | 8 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | B | 9 | h2-load70 | state_sm_clock_mhz | 2790 | 1200 | 132.5 % |
| g-cuda | A | 6 | h2-load70-oop | state_sm_clock_mhz | 2805 | 1125 | 149.3 % |
| g-cuda | A | 7 | h2-load70-oop | state_sm_clock_mhz | 2752.5 | 1125 | 144.7 % |
| g-cuda | A | 8 | h2-load70-oop | state_sm_clock_mhz | 2602.5 | 1125 | 131.3 % |
| g-cuda | A | 9 | h2-load70-oop | state_sm_clock_mhz | 2287.5 | 1125 | 103.3 % |
| g-cuda | B | 6 | h2-load70-oop | state_sm_clock_mhz | 2340 | 1125 | 108 % |
| g-cuda | B | 7 | h2-load70-oop | state_sm_clock_mhz | 2565 | 1125 | 128 % |
| g-cuda | B | 8 | h2-load70-oop | state_sm_clock_mhz | 2580 | 1125 | 129.3 % |
| g-cuda | A | 6 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | A | 7 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | A | 8 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | B | 6 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | B | 7 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | B | 8 | h2-load70-oop | state_mem_clock_mhz | 10251 | 5001 | 105 % |
| g-cuda | A | 6 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | A | 7 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | A | 8 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | A | 9 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | B | 5 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | B | 6 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | B | 7 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |
| g-cuda | B | 8 | h1-sim | state_sm_clock_mhz | 2790 | 1192.5 | 134 % |

Warnings:
- g-cuda: 4 round(s) reused from an earlier run (-Resume)

