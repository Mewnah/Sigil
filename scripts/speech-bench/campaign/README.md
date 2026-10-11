# P7-A speech campaign: specs and drivers

The measurement campaign behind [`docs/benchmarks/SPEECH_BENCHMARKS.md`](../../../docs/benchmarks/SPEECH_BENCHMARKS.md).
Everything here is a thin layer over the runner (`Sigil.exe --headless speech-bench`) and the scorers in
[`../score/`](../score/); the formats are the frozen P7-S0 ones ([`../README.md`](../README.md)). Re-running a spec on
another PC gives a comparable table (send `export.json` back, [`../score/merge_exports.py`](../score/merge_exports.py)
tabulates the machines).

**How the defaults are chosen (user decision 2026-10-09).** A streamer's own captions come from their own microphone, and with
headphones the game's audio does not reach it. The own-microphone STT defaults are therefore chosen only on `clean` plus the
mic-side conditions `room_light` and `mic_codec`; the bleed conditions (`music`, `babble`, `vrchat_mix`) play no part, not even as a
tie-breaker, and are tabulated apart for the "other players' captions" (loopback) feature. [`own_mic.py`](own_mic.py) is that rule as
code (and prints both rankings); its tests are `scripts/speech-bench/tests/test_campaign.py`. Two guards are part of the rule:
Granite TurboCTC and Nemotron are **never a caption final** (`--exclude-finals`, default `granite-cpu,nemotron-560`: they stay in the
tables and in the comparison but cannot be the default, whatever the `--cost` order says), and a language whose files lack an own-mic
condition for an engine **gets no winner**: the tool prints `NO WINNER`, a warning on stderr and exits 2 (`--allow-incomplete` keeps a
flagged provisional winner), so a partial file list cannot flip a default silently (an earlier version of the tool, given only
`finals-all` and `mic-a`, printed SenseVoice as the Korean winner, the reverse of the shipped default).

User scope the campaign followed (2026-10-09): the main audience streams STT captions into OBS while playing VRChat, so the
question for every candidate is "good enough captions without taking the game's CPU or GPU". Light and mid-weight engines
are screened first (cheap pass over every candidate), the rigorous test runs only for the finalists, and the heavy GPU
models (Cohere Transcribe, Qwen3-ASR 1.7B, Voxtral, R2T2, Hy-MT2 7B, MiLMMT, TranslateGemma, Qwen3-TTS 1.7B, Magpie, Breeze,
OmniVoice, Fish S2) stay opt-in and are **not** measured here.

## Order the campaign ran in

| Step | Spec | Mode | What it answers |
|---|---|---|---|
| 0 | `check_stt.json`, `check_speed.json`, `check_mt_tts.json` | accuracy / cpu_speed / mt_quality / tts_listening | 2-item first-run check of every backend (a backend that was only unit-tested before) |
| 1 | `stt_screen.json` | accuracy | Screening: every light / mid-weight STT engine, `quick` subset (clean + VRChat mix), 10 utterances per language |
| 2 | `finals_a.json` ... `finals_d.json` | accuracy | Finalists on the `core` subset (clean, music 10 dB, babble 10 dB, VRChat mix), 30 utterances per language (12 for Nemotron outside English) |
| 2b | `mic_a.json`, `mic_b.json`, `mic_c.json` | accuracy | The own-microphone decision: the finalists on the mic-side conditions (`room_light`, `mic_codec`, 30 utterances), the contested Korean pair on 100 utterances, Nemotron's English finals; `own_mic.py` ranks |
| 3 | `stt_auto.json` (`stt_auto_pipe.json`) | accuracy | Auto-language mode (no hint): engines alone, and the shipped pipeline with language ID |
| 4 | `speed_utt.json` | cpu_speed | Decode cost of every CPU engine, idle and next to a game, on a 6-core / 12-thread slice |
| 5 | `speed_stream.json` | cpu_speed | Nemotron 560 ms / 160 ms at real time: first partial, final latency, CPU |
| 6 | `speed_pipeline.json`, `speed_pipeline_4c.json` | cpu_speed | The shipped caption pipeline (Silero VAD, router, two pass) at real time on 6 and 4 cores |
| 7 | `mt_screen.json`, `mtspeed_*.json` | mt_quality / cpu_speed / gpu_under_load | Light translators: chrF and latency, then cost per line in the process tree |
| 8 | `tts_speed.json`, `sapi-measure.ps1` | cpu_speed | Kokoro on the CPU; Windows SAPI |
| 9 | `footprint/footprint.ps1` | (the real app) | Sigil's own CPU / RAM / GPU idle, with captions, with the OBS caption page and with translation |

## Running

```powershell
# the model store the campaign used (every prefetched model), and the corpus under %LOCALAPPDATA%\sigil-dev-cache\p7
$env:SIGIL_APP_DATA_DIR = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p7"

# a release build WITH THE SHIPPED GGML FLAGS (see "Building" below)
cd src-tauri; cargo build --release; cd ..

# one run, under the GPU timing lock, on a 6-core slice, with the process tree sampled
pwsh scripts/speech-bench/campaign/lock-run.ps1 -Name speed-utt -Spec scripts/speech-bench/campaign/specs/speed_utt.json `
  -Out bench-results/speech/speed-utt -Cpu -Affinity FFF -Sample bench-results/speech/speed-utt.csv

# scoring
$py = "$env:LOCALAPPDATA\sigil-dev-cache\p7\venv\Scripts\python.exe"
& $py scripts/speech-bench/score/score_stt.py --manifest "$env:LOCALAPPDATA\sigil-dev-cache\p7\corpus\corpus.json" `
  --report bench-results/speech/stt-screen/report.json --out bench-results/speech/stt-screen/scores --items
& $py scripts/speech-bench/campaign/stt_table.py bench-results/speech/stt-screen/scores/scores.json     # compact table
# the own-microphone decision: rank on clean + room_light + mic_codec, the bleed conditions tabulated apart, both rankings printed
#   finals-all = finals_a ... finals_d scored together (score_stt.py --report <each report.json> --items); mic-a, mic-b, mic-c scored one by one.
#   Qwen3-ASR 0.6B's mic-side runs are in mic-b (and the 100-utterance Korean runs in mic-c), so ALL FOUR files are needed.
& $py scripts/speech-bench/campaign/own_mic.py --items bench-results/speech/finals-all/scores/items.jsonl `
  bench-results/speech/mic-a/scores/items.jsonl bench-results/speech/mic-b/scores/items.jsonl bench-results/speech/mic-c/scores/items.jsonl `
  --md bench-results/speech/own_mic.md --json bench-results/speech/own_mic.json
& $py scripts/speech-bench/campaign/summarize_report.py bench-results/speech/speed-utt/report.json     # speed aggregates
& $py scripts/speech-bench/score/score_mt.py --report bench-results/speech/mt-screen/report.json `
  --set "$env:LOCALAPPDATA\sigil-dev-cache\p7\corpus\mt\sigil-casual.jsonl" --out bench-results/speech/mt-screen/scores
& $py scripts/speech-bench/campaign/mt_table.py bench-results/speech/mt-screen/scores/mt_scores.json
```

`run-bench.ps1` is the same without the lock. `-Cpu` hides CUDA from the process (CPU-only rows), `-Affinity` is a
hex mask of logical processors: `FFF` is the first 12 (six cores with SMT, the "mid-range CPU" slice the campaign calls
6C), `FF` is four cores. `-Sample` records CPU, working set and GPU of Sigil **and every process it starts**
(llama-server, WebView2) every 2 s, with a per-process breakdown.

## Things the campaign found about the tools (keep when re-running)

- **Build the release exe from `src-tauri/`** (or export `GGML_NATIVE=OFF` and `GGML_AVX2=ON`). `src-tauri/.cargo/config.toml`
  carries the ggml flags of the shipped build and cargo reads it only when its working directory is under `src-tauri/`.
  `cargo build --release --manifest-path src-tauri/Cargo.toml` from the repository root builds whisper.cpp for plain
  SSE2: Whisper then runs several times slower than in the installer. (whisper-rs-sys does not rerun its build script when
  those variables change: delete `target/release/build/whisper-rs-sys-*` and its `.fingerprint` folders once.)
- **The runner's process figures do not include helper processes.** `cpu_cores_mean` and `ram_peak_mb` of an engine that
  runs in llama-server, crispasr or a TTS helper describe the Sigil process (about nothing); the helper's CPU and RAM come from
  `-Sample` (`per_process` column). VRAM of helpers is read device-wide through NVML as growth over the baseline.
- **`threads` of a `llama-server` engine is not forwarded** (the bundled server plans its own: physical cores - 2, clamped
  to 2..8, on the real machine, whatever `-Affinity` says). A 6-core slice of an 8-core PC therefore runs the CPU translator
  with 6 threads where a real 6-core PC would use 4.
- **First partial** of a `stream` feed is measured from the first fed sample, so it includes the clip's own leading silence
  (FLEURS clips start 0.3-1.2 s in; the docs quote it from speech onset: -40 dBFS in 20 ms frames).
- **A `pipeline` row at `pace: asap` over-runs a streaming engine** (backlog is dropped, latencies include the backlog); use
  `realtime` for latency, `asap` only for accuracy.
- The 4-hour wait of `lock-run.ps1` is the fair queue: a run that starts after a long lock holder simply waits its turn. A run
  killed by hand (Stop-Process of `sigil.exe`) releases the lock through the wrapper and writes no report, so keep a run
  under about 40 minutes.

## Footprint of the real app

`footprint/footprint.ps1 -Scenario <name> -Load <none|game> -Exe <release sigil.exe with the frontend embedded>` starts the
release `Sigil.exe` in a scratch Windows profile (USERPROFILE / APPDATA / LOCALAPPDATA redirected for the child only, restored
afterwards, so the real `%APPDATA%\com.sigil.app` is never touched; the script reports that; `models` and `runtimes` of the profile are
junctions to the P7 store), writes the scenario's settings, minimizes the window, and samples the process tree (WebView2 and
llama-server included) every 2 s for two minutes after a one-minute warm-up.

| Scenario | What runs |
|---|---|
| `idle` | the app, speech-to-text off |
| `captions`, `captions-nointerim`, `captions-auto` | local STT on the CPU tier: English with interim results, without, and the new-install default (language auto) |
| `obs` | captions + the OBS caption page in headless Chromium + simulated caption traffic |
| `translate-cpu`, `translate-cpu-finals`, `translate-cpu-slow`, `translate-vulkan`, `translate-auto` | obs + the bundled Hy-MT2 1.8B translator (English to Japanese): CPU with interim translations **forced on**, with them off, with them throttled to 1.5 s, on Vulkan (interims forced on), and with the server variant and `interim` both left on `auto` (what a new install gets) |

Every translation scenario writes its `interim` setting explicitly. Since P7-I0 the shipped value is `"auto"`, which holds interim
translations off while the bundled translator runs on the CPU, so a scenario that left it alone would measure finals only
(`translate-cpu` would reproduce the `translate-cpu-finals` row). The forced-on rows are the cost of the policy `auto` replaced.

`footprint/driver.mjs` plays the part of the speaker and of OBS: it opens `obs-captions.html` in headless Chromium and feeds the host
about six interim captions a second while "speaking" (40 % of the time) with a final after each sentence, over the host's own pubsub
WebSocket, and prints what it saw (the OBS page's caption changes, and a count of every topic the host broadcast back, so
`text.translation` shows that the translator really worked); the page's Chromium is sampled separately (it stands for OBS's browser
source). The CPU burner of `-Load game` is four Node processes spinning 60 % of every 100 ms (the `h3_cpu_game` profile, outside the
app). The microphone is the default one (or `-Device`) and hears the room, so the engines idle between captions exactly as they do when
nobody speaks; the cost of speech itself is the `speed_pipeline` figure. Run it under the lock (`scripts/bench/gpu-lock.ps1 -Run ...`).
`footprint/diagnose.ps1` opens the WebView's debugging port and asks the page what the host services report: use it first on a new
build (an exe built without the frontend serves a directory listing and every scenario then measures an empty app; build with
`TAURI_CONFIG={"build":{"frontendDist":"<relative path of a vite build>"}}`, a drive-letter path parses as a URL).

**Nothing may take focus.** The scripts start the app with `SIGIL_START_MINIMIZED=1` (set in the child's environment and restored
afterwards) and fall back to `SW_SHOWMINNOACTIVE` for the window handle. `footprint/launch-guard.ps1` (dot-sourced by both) refuses to
start an exe whose binary does not contain the string `SIGIL_START_MINIMIZED` (a build from before commit `dcf3bca`, as the `p1c` and
`p3b-release` lane exes are): such a build would open a normal window in front of whatever you are doing. Rebuild it, or pass
`-ShowWindow` to start it visibly on purpose (both scripts take the switch). The same file selects the processes a run owns:
the clean-up stops a `llama-server` only when it descends from the app process the script started (parent chain, created after its
parent), never one that you or another tool started while the run was going.

`footprint/close-window.ps1` (dot-sourced by both scripts) ends the app the way the close button does: `WM_CLOSE` to the window titled `Sigil` of that process, never `Process.CloseMainWindow()`. Sigil's event loop owns an invisible "Tao Thread Event Target" window that `CloseMainWindow()` picks once the app is minimized (`SIGIL_START_MINIMIZED`, which every automated launch sets); a `WM_CLOSE` there leaves the process lingering until the app's exit guard ends it 20 s later (see `scripts/release/README.md`). Both scripts still `Kill()` a process that is alive after a few seconds.

## Tests

`python -m unittest discover -s scripts/speech-bench -p "test_campaign.py"` (the venv's Python) tests the own-mic rule in `own_mic.py`
(including the exit-2 behaviour of a partial file list and the never-a-final engines) and, with PowerShell 7 on Windows, the footprint
launch rules: both launchers set and restore the minimize switch and refuse an old exe, the translation scenarios write their `interim`
policy, and the process pick of `launch-guard.ps1` is exercised on a synthetic table and on real processes.
