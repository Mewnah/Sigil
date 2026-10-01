# Release and engineering

Release QA checklists, **network threat model**, **dependency audits**, **performance** notes, and **upgrade backlog**.

## Table of contents

- [Release build smoke matrix](#release-build-smoke-matrix)
- [Manual Tauri smoke checklist](#manual-tauri-checklist)
- [Local packaged smoke (manual)](#local-packaged-smoke-manual)
- [P0 exit matrix (speech stack stabilization)](#p0-exit-matrix-speech-stack-stabilization)
- [P1 exit matrix (platform foundations)](#p1-exit-matrix-platform-foundations)
- [P2 exit matrix (speech engine)](#p2-exit-matrix-speech-engine)
- [P3 exit matrix (voice engine)](#p3-exit-matrix-voice-engine)
- [P4 exit matrix (translation stack)](#p4-exit-matrix-translation-stack)
- [P5 exit matrix (voice families)](#p5-exit-matrix-voice-families)
- [Research §6 bug verification](#research-6-bug-verification)
- [Network security and threat model](#network-security-and-threat-model)
- [Dependency audit notes](#dependency-audit-notes)
- [Third-party components](#third-party-components)
- [Performance and connection-timeout audit](#performance-and-connection-timeout-audit)
- [Major upgrades backlog](#major-upgrades-backlog)

---

## Release build smoke matrix

Use this list before shipping a **packaged** Tauri build. Items that need a human (OAuth, second instance, mic in a real browser) cannot be fully automated in CI; run them locally after `pnpm build` and `pnpm tauri build`.

### Automated / agent checks (run in repo root)

- `pnpm typecheck`, `pnpm i18n:check`, `pnpm test:unit` (vitest) — the same gates CI runs.
- `pnpm run build` — TypeScript + Vite production bundle (includes `sync-manual.mjs`).
- `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets` and `cargo test --manifest-path src-tauri/Cargo.toml` (`dist/` must exist).
- `cd src-tauri && cargo check --release` — release-profile Rust compiles (LTO/strip settings apply at link time; full `cargo build --release` is optional for a longer pass).

### Release build (GPU execution providers)

- Build with **`pnpm tauri:release`** (`tauri build --config scripts/release/tauri.release.conf.json`). Its `beforeBuildCommand` runs `pnpm build` and `stage-native-dlls.mjs --prepare` (an empty staging folder, so compilation never depends on the DLLs); `beforeBundleCommand` runs `--stage`, which copies `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` and `DirectML.dll` with hash verification. The overlay also sets NSIS `installMode: currentUser`.
- Then run **`pwsh scripts/release/verify-bundle.ps1`** (Windows PowerShell 5.1 blocks `.ps1` by default: use `pwsh` or `powershell -ExecutionPolicy Bypass -File …`). It checks that both the NSIS and MSI installers put the three DLLs next to `Sigil.exe`, that `Sigil.exe` has no load-time import outside `scripts/release/expected-imports.txt` (no CUDA, cuDNN, ONNX Runtime, ggml or OpenVR), and prints the EP test results from the installed layout.
- Uploads: NSIS (recommended, per-user), MSI and a portable zip. `release.yml` sets `GGML_NATIVE=OFF` and `GGML_AVX2=ON` so whisper.cpp never targets the runner's CPU.
- The MSI also contains ort's unused `onnxruntime_providers_tensorrt.dll` (the WiX bundler adds every `*.dll` next to the built exe); verify-bundle lists it as a known extra.
- **What the installers do not contain:** NVIDIA's CUDA / cuDNN runtime (the opt-in `runtime/nvidia-cuda13` pack, about 1.9 GB) and every runtime pack (sherpa-onnx, transcribe.cpp, llama-server): nothing in the installer runs CUDA by itself. The 62 MB CUDA provider stays bundled until `runtime/ort-cuda13-provider` is published; then, per [ADR 0007](adr/0007-native-runtime-packs.md), set that entry `published`, add it to `runtime/nvidia-cuda13`'s `requires`, and drop `onnxruntime_providers_cuda.dll` from `stage-native-dlls.mjs` and `verify-bundle.ps1` (`onnxruntime_providers_shared.dll` and `DirectML.dll` stay). The NVIDIA pins live in `src-tauri/model-catalog/runtime-nvidia.json` and `diagnostics/nvidia_pack.rs`, kept equal by a test.
- The base `tauri.conf.json` bundles `src-tauri/resources/vc-templates/` (the 22 RVC split-graph templates, 15.5 MiB, graphs only with no weights; checked-in sources, so compile-time copying is safe) and, since P5, the voice changer's sidecar v2 scripts as `resources/neural-sidecar/` (`sigil_vc_v2.py`, the `xvc` / `streamvoiceanon` / `seedvc` recipes, `vc_sidecar.py`, `PROTOCOL_V2.md`, `README.md`, and `scripts/vc-families/seed_vc/sampler.py` as `seed_vc_sampler.py`; about 120 KB of checked-in sources); the release overlay's `target/native-bundle` mapping merges with it.
- **Release catalogs must be fully pinned:** `build.rs` refuses, in release builds, a `published` catalog entry with a file lacking `sha256` or `git_oid`. Pin with `node scripts/models/pin-catalog.mjs --write` (sequential, at least 2 s apart, cached in `%LOCALAPPDATA%\sigil-dev-cache`); hosts without digests (alphacephei) need `--hash-by-download` and consent for large files. Sigil-built packs stay `unpublished` (asset hosting is local only for now).

### Before tagging a release (dependency hygiene)

Run before you cut a release candidate; update [Dependency audit notes](#dependency-audit-notes) with a new **Last refresh** date and the pnpm / Rust summary tables (or a short “unchanged since last run” note).

- **`pnpm audit`** — repo root; triage or file follow-up issues. Do not blindly `pnpm audit fix --force` without verifying breaking changes (see recommendation in dependency section).
- **`cargo audit`** — `cd src-tauri` after `cargo install cargo-audit` if needed.

### Manual checks (installed or `tauri build` output)

1. **Startup** — App opens with default embedded server port (3030 unless `--port` is set); no blank `#root` error overlay.
2. **OAuth** — Twitch/Kick: loopback `http://localhost:17890/...` works; packaged webview redirects match [manual/03-integrations.md](manual/03-integrations.md#oauth-redirects-and-ports).
3. **External STT / mic.html** — From STT inspector, open Chrome or Edge (or default browser on Linux/macOS); WebSocket pubsub connects; speech reaches the host.
4. **Client mirror** — Open `/client` with `host`, `port`, `id` query params against the running host; canvas loads; disconnect shows a clear error.
5. **i18n** — Switch languages; translations load from `/i18n/...` without JSON parse errors.
6. **Local speech** — STT **On this PC (automatic)**: Start asks before downloading, then captions arrive; TTS **Kokoro / Piper (local)**: the panel's download, **Preview**, then speech through the chosen device; optional HTTP sidecar (`scripts/neural-sidecar/tts_sidecar.py`) through **OpenAI-compatible (HTTP)**.
7. **Second instance** — Launch a second Sigil while the first holds the port; expect the native port-unavailable dialog.

See also [Manual Tauri checklist](#manual-tauri-checklist) for a broader feature pass.

---

## Manual Tauri checklist

Automated Playwright tests use Vite preview only; native Tauri behavior requires a manual pass before release.

### Host (desktop app)

- [ ] `npm run tauri dev` — app window opens, no startup error overlay in `#root`.
- [ ] **Config / network** — `plugin:web|config` loads; status/footer shows expected host/port.
- [ ] **Whisper** — download or load model path (progress shows under the Start button and in Models & Runtimes, never in captions); start recording; one **final** per spoken sentence (`whisper:segment_transcript`); silence produces nothing.
- [ ] **On this PC (automatic)** — fresh profile: Start asks before downloading (sizes match the route preview); downloads show progress; live partials, then one final per sentence; STT mute drops the open sentence.
- [ ] **One STT path** — e.g. native or engine you ship; Start/Stop from bottom bar or inspector.
- [ ] **One TTS path** — voice selected; Start; speak or trigger text event. Long text starts after its first sentence; **Skip** / **Stop speaking** on the status bar.
- [ ] **Twitch** — `SIGIL_TWITCH_CLIENT_ID` set; Twitch console lists **loopback** URLs: **Public** → `http://localhost:17890/oauth/twitch/implicit`; **Confidential** → `http://localhost:17890/oauth/twitch/callback`; plus **Vite** `http://localhost:1420/oauth/...` if you use `pnpm dev`, and the **packaged app** `https` origin + path if flows hit the webview (see [manual/03-integrations.md](manual/03-integrations.md#oauth-redirects-and-ports)); OAuth completes; optional: IRC chat + post/receive.
- [ ] **Discord** — webhook URL; one post with “post only when live” on and off if you use that gate.
- [ ] **OBS** — WebSocket connect; captions only send when “Enable” is on under native captions.
- [ ] **VRChat OSC** — desktop build only; verify chatbox / OSC target with VRChat running.
- [ ] **Kick** — optional: OAuth only; register `http://localhost:17890/oauth/kick/callback` for desktop and any packaged/dev webview URLs per [manual/03-integrations.md](manual/03-integrations.md#oauth-redirects-and-ports) (chat/emotes not wired yet).

### Remote client (`/client`)

- [ ] Open client URL with `host`, `port`, `id` query params; canvas loads when host is running.
- [ ] Disconnect host — user sees a clear error (not a silent blank screen).

### Packaging

- [ ] `pnpm tauri:release` — NSIS and MSI under `src-tauri/target/release/bundle/` match CI expectations in `.github/workflows/release.yml`; `pwsh scripts/release/verify-bundle.ps1` passes.
- [ ] **Dependency audit log** — `pnpm audit` and `cargo audit` completed; [Dependency audit notes](#dependency-audit-notes) updated (date + findings).

---

## Local packaged smoke (manual)

CI [Playwright](https://playwright.dev/) tests run against **`vite preview`** only (`pnpm build` + static server), not the **installed** Tauri binary. Use this short pass after `pnpm tauri build` when you need confidence in the **packaged** app on your machine.

1. **Build** — `pnpm tauri build` (or your release profile). Note output paths under `src-tauri/target/release/` (`.exe` / installer per platform).
2. **Launch** — Start the installed or freshly built executable (not `pnpm tauri dev`).
3. **Window** — Main window appears; no persistent error overlay on `#root`.
4. **Embedded server** — Status or footer shows expected host/port (default **3030** unless `--port`); no immediate bind error.
5. **Optional** — Open **Help** (book icon) and confirm manual markdown loads; spot-check one inspector if you changed native code.

For deeper coverage, continue with the [Manual Tauri checklist](#manual-tauri-checklist) on the same build.

### Packaged STT smoke (Whisper + Vosk)

Run on the **installed** or freshly built exe (`pnpm tauri build` → `src-tauri/target/release/Sigil.exe`), not `pnpm tauri dev`.

| Step | Whisper | Vosk |
|------|---------|------|
| Mic | STT inspector **input device**, or **Settings → Audio → Default input device** (native) when the inspector is on default | same |
| Start | Status line: download → “Loading Whisper model…” → **Connected** | Status line: ensure zip → WASM load → **Connected** |
| Speak | One **final** per sentence after the pause that ends speech | Partials + finals |
| Stop during load | Stop → restart immediately; no ghost status / stale events | Stop during download → restart |
| DevTools | No CSP `worker-src` / `connect-src` (`https://asset.localhost`) errors | same |

Also on the same build:

- **On this PC (automatic)** — English and Japanese: consent dialog, downloads, live partials replaced by one final per sentence; Stop during a download cancels it (the next Start resumes).
- **Groq** (real key) — captions arrive; the status line shows a rate-limit wait when you talk continuously; a bad key stops with a readable message; nothing is sent while muted.
- **Local HTTP server on the LAN** (OpenAI-compatible, for example speaches on another PC) — transcribes (the request is sent by Rust, so the webview CSP does not apply).

Optional helper: `scripts/packaged-smoke-stt.ps1` prints this checklist and launches the release exe if present.

### GPU execution provider smoke

On the same release build ([ADR 0003](adr/0003-gpu-ep-delivery.md)):

1. **CPU baseline** — voice changer **ONNX** or **RVC** mode on **CPU**; the status panel says “Running on CPU”.
2. **NVIDIA with the pack** — Models & Runtimes → **NVIDIA GPU runtime (CUDA 13, cuDNN 9)** → **Download** (or Settings → GPU & diagnostics → NVIDIA GPU runtime → **Download and install**; about 1.9 GB, sequential, resumable), stop/start the voice changer, expect **CUDA** in the status panel and the diagnostics panel (**Test GPU acceleration** reports the runtime source). Then `pwsh scripts/release/verify-bundle.ps1 -RequireCuda`. CUDA needs driver R580+ and a Turing or newer GPU.
3. **NVIDIA without the pack** — a CUDA request runs on **DirectML** with a warning box, one toast and a System log line naming the missing DLL (for example `cudnn64_9.dll`).
4. **Without DirectML.dll** (admin-install the MSI, delete the bundled `DirectML.dll`, run `Sigil.exe --headless ep-probe --out probe.json`) — Sigil still starts; the probe notes that the System32 copy is used, or reports the CPU fallback with its reason.
5. **Test GPU acceleration** in the diagnostics panel completes and saves `app_data\diagnostics\ep-probe-deep.json`.

**Future automation (optional):** A `workflow_dispatch` Windows job or a local script that launches the built `exe` with a timeout is possible; tracked as backlog in [Major upgrades backlog](#major-upgrades-backlog) if product priority allows.

---

## P0 exit matrix (speech stack stabilization)

Automated gates for the P0 pass (all green at P0 integration): `pnpm typecheck`, `pnpm i18n:check`, `pnpm test:unit`, `pnpm build`, `cargo clippy --all-targets` (no new warnings), `cargo test` (including every pre-existing Rust test). The checks below need the real desktop app (`pnpm tauri dev` or a packaged build) and real devices.

| Area | Manual check | Expected |
|------|--------------|----------|
| STT silence | Whisper running, 10 minutes of silence plus keyboard noise | No `transcribe utterance` line under `sigil::stt::whisper` in `sigil.log`; no captions |
| STT fresh profile | Fresh profile on desktop, press Start once | Switched to Whisper (small/base Q5_1); download progress under the Start button, never in captions |
| Groq | Groq key set, speak continuously for 2 minutes | At most 20 requests per minute in the log; no lost sentences |
| Shortcuts | Set STT start and mute shortcuts, use them while VRChat has focus | Toggle once per press |
| Deepgram | Only a key and a language set | Starts (Nova-3) |
| VC TTS | 60 s TTS clip routed through the voice changer | Plays intact, mic ducked or muted per setting |
| VC idle | Voice changer running, user silent | Worker below 1% of a core; log shows the one-time “MMCSS Pro Audio: capture registered, output registered” line |
| VC stall | Force a 2 s GPU stall | Backlog stays at or below budget + one block and recovers on the next wake |
| VC persistence | Change settings, devices and a voice's pitch; restart Sigil | Settings, devices, consent and the voice's `slot.json` values come back; nothing auto-starts |
| GPU EPs | [GPU execution provider smoke](#gpu-execution-provider-smoke) steps 2–4 | CUDA with the pack; DirectML without it; CPU with a visible warning without DirectML |
| LAN | Steps below | All pass |
| TTS | Azure: speak `Tom & Jerry <3`; each backend: long line then Stop; AivisSpeech preset with the engine on port 10101 | Spoken correctly; status goes to disconnected and nothing more plays; pitch resets to 0 and Japanese is spoken |

**LAN manual matrix** (LAN security stage 1):

1. Start Sigil; the log shows `local web server listening on 0.0.0.0:3030`.
2. OBS browser source `http://localhost:3030/client` connects.
3. A phone on the LAN loads `http://<LAN IP>:3030/client` and connects.
4. The STT Chrome button opens `mic.html`; speech arrives; after restarting Sigil the page reconnects by itself.
5. `http://localhost:3030/obs-captions.html` shows captions.
6. `http://localhost:3030/inputfield.html` sends text.
7. Link apps between two PCs carry text both ways with no repeats.
8. On `https://example.com`, run `new WebSocket('ws://localhost:3030/pubsub?id=x')` in devtools: it fails (403), Sigil logs the refusal and a “Blocked a connection” toast appears.
9. With **Only allow Sigil pages to connect** off (or `SIGIL_LAN_ORIGIN_CHECK=0`), step 8 connects.

## P1 exit matrix (platform foundations)

Automated gates at P1 integration (2026-09-29, dev rig: RTX 4090 driver 616.56, Ryzen 7 7800X3D): `pnpm typecheck` clean, `pnpm i18n:check` OK, `pnpm test:unit` 49 files / 585 tests, `pnpm build` OK, `cargo clippy --all-targets` 0 warnings, `cargo test` 706 passed / 0 failed / 23 ignored; re-run at the P1 final gate after the late fixes (bench `bench.vc*` EP components, catalog folder check in `build.rs`, P1-B/C/D2 strings merged): 605 unit tests, `cargo test` 788 passed / 0 failed / 24 ignored, the other gates unchanged (the ignored tests need real models, devices or GPU runtimes; see [DEVELOPER_GUIDE → Release / quality](DEVELOPER_GUIDE.md#release--quality)). `kittentts` and `hf-hub` are gone from `Cargo.lock`; `ort` is `2.0.0-rc.13`.

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green on the chosen ORT version ([ADR 0001](adr/0001-ort-upgrade.md)); kittentts and hf-hub gone | Gates above; `Cargo.lock` has `ort 2.0.0-rc.13` and no kittentts / hf-hub | Met |
| `--headless fetch <id>` and the Models tab install, resume, verify, cancel and delete sequentially; whisper/vosk files adopted | `model_store` fixture-server tests (resume after abort and stall, cancel keeps the partial, failover, verify, remove, one download at a time); `Sigil.exe --headless fetch base/test-small` from huggingface.co; `fetch whisper/tiny.en vosk/small-en-us-0.15` adopted existing files with no network; `fetch runtime/nvidia-cuda13` from `SIGIL_LOCAL_ASSETS` (1.4 GB in 6 s); `fetch --list`. Models tab checked against an IPC mock in the browser pane | Met (automated); real-app tab pass open (manual row 1) |
| The capture hub serves Whisper and Vosk from one stream; unplug/replug recovers within 2 s | `audio::hub_tests` (fan-out, one stream per device, restart under the same epoch, recovery < 2 s after a device-change nudge, default follow); ignored real-WASAPI test passed on the dev rig (one 48 kHz stream shared by `default` and the named mic, loopback opened) | Met (automated); physical unplug open (manual row 2) |
| An echo DLL loads lazily and the app starts without it; process_host children die when Sigil is killed | `native_runtime` tests (missing / garbage / foreign DLLs are errors, ABI mismatches refused); `Sigil.exe --headless engine-probe echo --runtime-dir native/out/dev/echo/cpu` ok; `Sigil.exe` load-time imports unchanged (41); `process_host::tests::helpers_die_when_sigil_is_killed` (TerminateProcess on a fake Sigil kills the helper and its child) | Met; Task-Manager check with llama-server open (manual row 3) |
| `--headless bench` runs a stub model through the real harness in paced mode and emits report.json | `Sigil.exe --headless bench scripts/bench/specs/stub-smoke.json` → PASSED, exit 0 (offline, paced: latency 74.37 ms, 0 underruns; stall: back within budget at the next block), `report.json` + `report.md` written | Met |
| ADRs 0002–0005 record measured go/no-go decisions | [0002](adr/0002-sherpa-onnx.md) sherpa go (SenseVoice en/ja, Kokoro, CUDA coexistence measured), [0003b](adr/0003b-transcribe-cpp.md) transcribe go (cuda13 + CPU measured; vulkan unmeasured), [0004](adr/0004-rvc-templates.md) templates go (parity tables CPU/DML/CUDA), [0005](adr/0005-process-host-llama.md) process host go (llama-server live smoke pending), plus [0001](adr/0001-ort-upgrade.md) and [0007](adr/0007-native-runtime-packs.md) | Met (0005's live smoke manual) |

Also verified at integration: `Sigil.exe --headless sysinfo` (RTX 4090 Ada 8.9, CUDA 13.4 driver, DXGI budget 23,370 MB of 24,138 MB, all four GPU EPs eligible, recommendations Hy-MT2 7B / Qwen3-TTS 1.7B / Qwen3-ASR 1.7B BF16); with `runtime/nvidia-cuda13` installed from the local cache into a scratch profile, `Sigil.exe --headless ep-probe` reports `nvidia_runtime_source: model_store`, cuDNN 9.26.0 loaded, CUDA canary ok (327 ms) and recommends CUDA; `scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu` rebuilds a byte-identical pack from a fresh build tree, and `-Engine transcribe -Variant cpu` delegates to the engine's own `build.ps1` (pack zip identical to P1-G2's, licenses included).

**Manual checks still open** (need the real app, devices or permission to run downloaded binaries):

1. **Models tab in `pnpm tauri dev`:** download `base/test-small`; start a Whisper model, cancel, resume; Check files; Delete; Import a local file (a folder for a multi-file entry); set a custom mirror URL.
2. **USB microphone:** start Whisper on a USB mic, unplug it (expect the "Audio device … stopped" toast, STT stays connected), replug (expect "working again" within about 1 s and transcription continuing without an STT restart).
3. **llama-server:** the [ADR 0005](adr/0005-process-host-llama.md) procedure (`cargo test llama_server_smoke -- --ignored --nocapture` with `SIGIL_TEST_LLAMA_SERVER` / `SIGIL_TEST_LLAMA_GGUF`, Hy-MT2 1.8B Q4); then, once P4-B's UI starts it, end `Sigil.exe` in Task Manager and confirm `llama-server.exe` disappears.
4. **NVIDIA runtime pack in the real app:** install **NVIDIA GPU runtime (CUDA 13, cuDNN 9)** from Models & Runtimes, run an RVC voice on CUDA, then remove the pack (CUDA falls back to DirectML until restart). Users who installed the interim Settings pack before P1-I0 must reinstall it (it lacked two cuDNN 9.26 sub-libraries).
5. **Voice changer benchmark with a real RVC voice** on CUDA, DirectML and CPU (the engine baseline for P3), and the `e2e_loopback` meter over an enabled virtual cable.
6. **transcribe `vulkan` variant:** build with the Vulkan SDK and measure on an AMD/Intel GPU.

**Voice-changer change checklist (from P1):** record `Sigil.exe --headless bench <spec>` before and after, and run `pwsh scripts/bench/run-perf-gate.ps1 -Baseline <dir>` on a release build: a p95 regression over 5 % blocks acceptance ([docs/benchmarks/README.md](benchmarks/README.md)).

## P2 exit matrix (speech engine)

Automated gates at P2 integration (2026-09-29, same dev rig): `pnpm typecheck` clean, `pnpm i18n:check` OK (227 staged keys merged, 9 removed), `pnpm test:unit` 58 files / 806 tests, `pnpm build` OK (the Azure speech SDK chunk is now only loaded on demand: no chunk imports it statically), `cargo clippy --all-targets` 0 warnings, `cargo test` 1047 passed / 0 failed / 52 ignored. Real-model and GPU runs (ignored tests, run with the dev caches): sherpa 72 passed, local TTS 87 passed, transcribe GPU tier 6 passed, Whisper through the pipeline and Smart Turn 4 + 2 passed. Reports: `bench-results/p2-exit/` (git-ignored); numbers in [docs/benchmarks/SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md).

Final P2 gate after the review fixes (2026-09-29): `pnpm typecheck` clean, `pnpm i18n:check` OK (4 staged keys added, 2 removed), `pnpm test:unit` 58 files / 841 tests, `pnpm build` OK (speech SDK chunk hash unchanged), `cargo clippy --all-targets` 0 warnings, `cargo test` 1104 passed / 0 failed / 55 ignored. The fixes: the TS router mirror matches Rust again (auto `streamLanguages`, `no_cpu_fallback`, an unknown voice-changer device counts as the CPU); the pipeline enforces an auto plan's stream languages; a first pass removed by the post-filter no longer counts as a first final; `stop_pipeline` takes an optional `epoch`; the sentence's `TextEvent.lang` reaches local TTS voices.

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green; the app starts with no runtime packs installed | Gates above. The debug `sigil.exe` has 38 load-time imports, all in `scripts/release/expected-imports.txt` (no CUDA, ORT, ggml or sherpa DLL); on an empty `SIGIL_APP_DATA_DIR`, `--headless fetch --list` lists every P2 entry as not installed (exit 0) and `--headless engine-probe sherpa stt/sense-voice-small-int8` ends with "the sherpa-onnx runtime pack is not installed" (exit 1, no crash). Every speech engine answers `NotInstalled` naming its catalog id; the sherpa preload runs only when the pack is installed | Met (automated); a `pnpm tauri dev` start on a fresh profile is manual row 1 |
| CPU-only: en gives Nemotron partials, zh SenseVoice finals, ja Qwen3-ASR-0.6B finals; the router never sends ja / zh / ko to Cohere | Shared routing fixture (56 cases) in Rust (`speech::router`) and TS (`router.test.ts`), plus the exhaustive rule test over every language × tier × PC profile. Through the real pipeline with real Silero VAD (`through_the_pipeline::real_sherpa_engines_decode_through_the_pipeline`): Nemotron en 9 partials then the final 160 ms after VAD end; SenseVoice zh final 233 ms with `lang:zh` / `emotion:neutral` tags; Qwen3-ASR-0.6B ja final 1216 ms; Parakeet Ultra de 287 ms | Met |
| On the dev GPU, two-pass finals replace partials under the same segmentId ≤ 800 ms after VAD end for an 8 s utterance | `real_two_pass_ja_with_vram_logged` (RTX 4090, Qwen3-ASR 1.7B Q8_0 through the governor and the real two-pass queue): 199 ms and 194 ms after VAD end, same segment, revision + 1; process VRAM 384 → 3476 MiB. The TS `SegmentPublisher` publishes one final per segment (`localDriver.test.ts`) | Met |
| Kokoro starts playing after the first sentence (≤ 400 ms on CPU); Stop cancels; every PCM backend honours the device and can route into the VC | Kokoro v1.0 on 4 threads: median RTF 0.294 (Sigil's ORT with the CPU arena, now the default; 0.315 inside sherpa-onnx). The sentence pipeline starts sentence 1 before synthesizing sentence 2 (`pipeline.test.ts`); first audio of a long first sentence 656 ms (its opening clause is about 2 s of audio), so ≤ 400 ms holds only for short opening clauses. Cancel stops a long utterance in about 0.6 s; skip / stop cut audio already in the sink ring (`tts_sink` tests). Device field honoured with tolerant names and route `device` / `voiceChanger` / `both` (`tts_sink` and `aux_feed` tests; P2-G and P2-I mocked-IPC smokes) | Met except the 400 ms target for long opening clauses (P7 decides); the real-device checks are manual rows 5–7 |
| New installs default to `local`; native-backend users get a one-time prompt | `stt/migrate.ts` + `tts/defaults.ts` planners and `Service_STT` / `Service_TTS` tests (fresh install → `local` / `localNeural` silently; an existing native profile is asked once, no → Whisper; offers never repeat; web build untouched). Production-bundle smoke with an IPC mock: a fresh profile boots on `local` + `localNeural` with no dialog; an existing profile on Web Speech gets the STT offer, then the TTS offer, and the language carries over (`ja-JP` → `ja`) | Met |

**Manual checks still open** (need the real app, devices, servers or keys):

1. **Fresh profile in `pnpm tauri dev`:** onboarding offers **On this PC (automatic)** first; press Start with English, agree to the download, captions arrive; switch to Japanese (Nemotron partials plus a Qwen3-ASR 0.6B final); an existing profile on Web Speech sees the two one-time offers.
2. **Whisper with a real microphone:** the download row fills and Models & Runtimes shows `whisper/<id>` installed; captions; the STT hard mute drops the open sentence; replugging a USB microphone continues the recording.
3. **Vosk and Groq:** Vosk English captions and mute; Groq with a real key (captions, rate-limit status, bad-key message, nothing sent while muted); an OpenAI-compatible STT server on the LAN; the Moonshine server backend with the server off shows its message.
4. **Local voices:** Kokoro en / ja / zh, Piper ru and Kitten from the panel (download, Preview, the UI never freezes).
5. **VOICEVOX / AivisSpeech:** a long text starts after its first sentence on the chosen device.
6. **TTS into the voice changer:** route **Voice changer** or **Both** with the voice changer running: speech in the converted voice, the microphone ducked or muted per its setting.
7. **Output device unplugged mid-sentence:** that sentence errors once and the next one reopens the device.
8. **OpenAI-compatible TTS presets** against real servers (Kokoro-FastAPI, Chatterbox, vLLM-Omni, qwentts): pick the preset, **Test voice**, or `cargo test openai_speech_live -- --ignored` with `SIGIL_TEST_OPENAI_SPEECH_URL` / `_MODEL` / `_VOICE`.

---

## P3 exit matrix (voice engine)

Gates at P3 integration (2026-09-30, dev rig H2: RTX 4090, Ryzen 7 7800X3D): `pnpm typecheck` clean; `pnpm i18n:check` OK (P3-E, P3-G and P3-I0 staging merged: 276 keys added, 1 removed); `pnpm test:unit` 92 files / 1414 tests; `pnpm build` OK (speech SDK chunk hash unchanged, `speech-sdk-CdZJ51IR.js`; same 46 chunks as P3-G's build); Playwright 8 / 8 (`PW_PREVIEW_ONLY=1`); an IPC-mock host smoke of the production bundle (boots; the voice-changer panel renders the Voices list, the new Hotkeys section and the Lookahead choice; applying **Quality** sends `lookahead_ms: 20`); `cargo clippy --all-targets` **0 warnings** (the two P3 leftovers fixed: the unused pitch-extractor re-exports and the dead `pcm_convert::build_input_stream_f32`; stale dead-code allowances removed where the code is now used); full `cargo test` **1490 passed / 0 failed / 85 ignored**; the release build's one warning (an unused `mut` in `vc::import::default_templates_dir`, release-only) is fixed. Benchmarks: `run-perf-gate.ps1 -Profile p3` before and after the integration changes on release builds ([VC_BENCHMARKS.md → P3 integration](benchmarks/VC_BENCHMARKS.md#p3-integration-p3-i0-the-benchmark-decided-defaults)); the p95 rule found no regression attributable to the changes (the flagged rows came from a slower GPU state that interleaved reruns reproduced on both builds).

**P3 final gate (2026-10-01, after the review fixers' late requests):** `pnpm typecheck` clean; `pnpm i18n:check` OK (the last staging file merged: 2 keys); `pnpm test:unit` 92 files / 1425 tests; `pnpm build` OK; `cargo clippy --all-targets` **0 warnings**; full `cargo test` **1544 passed / 0 failed / 89 ignored**; the release build has no warnings. Applied: RMVPE's end-aligned log-mel (T5), the bring-your-own ONNX anchored-noise rule (T8), the voice's stream report after the harness is built (T7), the held-slot set for the import commands, the rung-aware fusion requirement, the stale-autotune notice, bare state dicts in `parity_test.py`; the RefineGAN two-window and layout-v2 importer guidance went into the docs for the v2 follow-up ([DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md), plan → P3 outcomes). Benchmarks: `run-perf-gate.ps1 -Profile p3` on release builds before and after, then interleaved reruns, five-clip fidelity runs, the lookahead and FCPE A/B ([VC_BENCHMARKS.md → P3 final gate](benchmarks/VC_BENCHMARKS.md#p3-final-gate-pitch-track-fixes)): V/UV against the ceiling 8.0 → 3.8 % (DirectML) and 9.2 → 3.5 % (CUDA), F0 RMSE 68 → 47 / 53 cents, MCD 1.70 → 1.68 / 1.66 dB, clicks 0; no p95 regression reproduced in the interleaved reruns (medians −3.2 % to +2.4 %). Not adopted by measurement: FCPE's 20-frame lead-in (DNSMOS down on CUDA) and 10 ms of lookahead for Low latency (one-clip gain, latency cost; P7 decides).

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green, and every bundled template passes parity | Gates above. P3-A: all 22 templates pass the CPU parity gate (worst fp32 flowdec 1.0e-5; partial decode equals the full-decode tail) and record a DirectML class per variant (16 run fp16 flowdec on GPUs; `scripts/vc-templates/reports/matrix_parity.md`); the bundled set is 15.5 MiB; the 26 cells no trainer produces are listed with reasons in `matrix.json`. The maintainer cross-check against Applio's own RefineGAN / MRF code (`parity_test.py --upstream-applio`) is manual row 9 | Met (the Applio cross-check is manual) |
| The benchmark scorecard on H2 with the load simulator meets T2, T3, T5, T7 and T9; Streamer CPU meets T4 with the GPU EPs disabled; the H1 run is scheduled | [VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md) scorecard. **Met:** T2 under load (DirectML 0 % over budget; CUDA 0 % since the P3-I0 lever defaults), T2 context scaling (512 ms of context costs nothing measurable), T5 clicks (0) and CER (+0 pt), T5 speaker cosine with Quality's 20 ms lookahead on CUDA (0.975), T7 (backlog never over budget, 0 blocks to recover from a 3 s stall), T9 on CUDA (1.94 s with graph capture), T4 for the GPU preset on DirectML (0.10 cores) and the silent voice (0.0095 cores). **Missed:** T2 per-block p95 ≤ 6 ms (CUDA 10.9 ms, DirectML 11.1 ms; the content encoder is the largest cost), T3 with RMVPE (861 MB; SwiftF0 / CREPE-tiny 484–537 MB), T5 MCD (1.40 dB with lookahead vs ≤ 0.5), F0 RMSE (47–53 cents since the final gate vs ≤ 15) and V/UV (3.5–3.8 % since the final gate, without lookahead, vs ≤ 3 %), T9 on DirectML (2.7–2.9 s), T4 Streamer (RTF 3.2 on one thread; the autotuned CPU preset reaches 0.64 at 200 ms blocks). The H1 run needs a friend's live-record (manual row 4) | **Not met**: engine targets T2 (per block), T3 (RMVPE), T4 (Streamer), T5 (MCD, F0) and T9 (DirectML) remain; the fixes are listed in VC_BENCHMARKS "What closes the gaps" |
| The corpus reaches T8 (≥ 95 % import), and the fuzz corpus causes 0 crashes; the MIT default voice works with zero setup | Fuzz: 200 malformed files, 0 crashes or hangs, in-process and through 200 isolated worker processes (`vc::import::corpus_tests`, `worker::tests`). Default voice: imports in 0.84 s (release) with weights byte-identical to the Python writer; `vc/voice-default` is published and hash-pinned, and the first-run card installs and selects it. **T8 corpus not measured:** no 40-voice community corpus exists on the dev rig (manual row 6; `community_corpus_report` is ready). Layout-v2 voices (f0-less, RefineGAN, MRF) are refused as "not supported yet" until the importer and engine support them, so a corpus with such voices cannot reach 95 % today. "Zero setup" still needs a local import of the ContentVec encoder (a Sigil export, `unpublished` while asset hosting is local) | Partly met (fuzz and default voice); T8 open |
| The autotuner returns a working preset on CPU and GPU; ADR 0006 records zero-shot and singing feasibility with parity numbers | Autotuner: GPU → CUDA, SwiftF0, 60 ms blocks, IOBinding + graphs, p99 RTF 0.47; CPU → SwiftF0, 200 ms blocks, the thread pool, RTF 0.64 (keeps up) (VC_BENCHMARKS "The autotuner"). [ADR 0006](adr/0006-vc-zero-shot-singing.md): Seed-VC GO on CUDA / DirectML fp32 (≥ 58 dB tick parity), DDSP-SVC GO on CUDA (80 dB), MeanVC2 GO as an experimental CPU family (61 dB) | Met |

**Manual checks still open** (need the real app, devices, a friend's PC, downloads or maintainer permission):

1. **In-app voice changer** (`pnpm tauri dev`, fresh profile): the Get started card (default voice download, import and selection; Import for ContentVec from a local folder), drag-and-drop import of a `.pth` with its `.index`, a `.zip`, and a refused file (check the translated error), rename / duplicate / merge / delete, a hot switch between two imported voices while running (no gap), voice hotkeys while a game has focus, Quality's lookahead note on DirectML.
2. **Devices:** USB microphone unplug and replug on the same and on another port, output device unplug and replug while running (`device_lost` / `device_restored`).
3. **30-minute soak** under VRChat + OBS with 0 underruns (T7).
4. **H1 live-record** from a friend's RTX 3060 / 4060 PC (T1, T2, T7 authoritative), and a CPU-only run (T4).
5. **True end-to-end latency** with the e2e-loopback meter over a virtual cable (T1).
6. **Corpora:** the 20-clip consented speech corpus (T5) and the user's 40-voice community corpus (`SIGIL_TEST_VC_CORPUS_DIR=<folder> cargo test --manifest-path src-tauri/Cargo.toml community_corpus_report -- --ignored --nocapture`, T8).
7. **T6 head to head:** RVC WebUI realtime and Applio realtime through `scripts/vc-bench`, w-okada through `play_record.py`, scored with a `compare` run and the blind A/B kit (`tools/listening-test`).
8. **TensorRT-RTX (T9):** put a TensorRT-RTX plugin execution provider plus the runtime DLLs its imports name in a folder, set `SIGIL_TRT_DIR` to it, run the staged test executable twice with `--ignored gpu_lever_bench` (`SIGIL_BENCH_CONFIGS=all+accel`): the second run's cold load on the TRT row must be ≤ 3000 ms and the log must say "engine loaded from the cache". The provider pyke ships with ort rc.13 imports TensorRT 10 names, so it needs a matching runtime (or a provider built against TensorRT-RTX).
9. **Maintainer checks:** `python scripts/vc-templates/parity_test.py --upstream-applio` (must print max|d| ≤ 1e-4; running the downloaded Applio code was not permitted to the agent), and a PESTO export with PESTO's own `realtime/export_onnx.py`.
10. **GPU tests on the dev rig** (ignored tests): copy the test executable, `DirectML.dll` and `onnxruntime_providers_{cuda,shared}.dll` into one folder, set `SIGIL_APP_DATA_DIR` to an app-data folder with `runtime/nvidia-cuda13` installed, then run `--ignored gpu::tests::` (device plan, CUDA graphs, priority stream, shared arena, lever benchmark).

## P4 exit matrix (translation stack)

P4 ran in the same working tree as P3 (the voice-changer engine). Gates at P4 integration (2026-09-30, same dev rig): `pnpm typecheck` clean, `pnpm i18n:check` OK (375 keys added: 373 staged plus 2 model-family labels; 2 removed), `pnpm test:unit` 84 files / 1279 tests, `pnpm build` OK (speech SDK chunk hash unchanged, `speech-sdk-CdZJ51IR.js`; the `openai` chunk is gone with the dependency), Playwright 8 / 8 on that build, and an IPC-mock host smoke of the production bundle (boots; the new "Local model server" init task calls `llm_server_status`; the translation provider list, the AI model panel with the Local model server block and the Models & Runtimes family labels render from the merged strings). Rust (run once P3's files compiled again): `cargo clippy --all-targets` exit 0 with **0 warnings in P4 files** (the 34 remaining are in P3's in-progress `bench`, `neural_audio`, `voice_changer` files and the now-unused `pcm_convert::build_input_stream_f32`); full `cargo test` 1433 passed / 14 failed / 82 ignored, where 13 failures are in P3's in-progress files (`bench::{dsp,engine,fidelity,quality,spec}`, `neural_audio::denoise::stft`, `voice_changer::{playback,timeline}`) and 1 is a P2 timing test under load (`speech::two_pass::…lost_engine_stops_decoding`, passes alone); P4-scoped filters all green: `llm_server` 44 passed / 2 ignored, `mt_local` 59 / 5, `translate::` 18 / 1, `process_host` 60 / 4 (including `helpers_die_when_sigil_is_killed`), `model_store` 79, `catalog` 40, `resources` 17, `cli` 45. **Status: green for the P4-owned scope; the full Rust gate waits on P3.** Real-model runs (ignored tests, P4 workstreams): NLLB tokenizer parity 27 / 27 with Python sentencepiece, greedy output 24 / 27 exact (3 near-ties), the worker path; the Bergamot client against the C++ fake helper.

**P4 final gate (2026-09-30, after the late fixes: per-process id nonces, the last i18n merge):** `pnpm typecheck` clean; `pnpm i18n:check` OK (P4-A / P4-F merged: 8 keys added, 3 removed; P3-E left for P3's integration); `pnpm test:unit` 85 files / 1353 tests; `pnpm build` OK; Playwright 8 / 8 (`PW_PREVIEW_ONLY=1`); `cargo clippy --all-targets` exit 0 with 0 warnings in P4 files (the 2 left are P3's: unused pitch-extractor re-exports in `neural_audio/mod.rs`, and `pcm_convert::build_input_stream_f32`, unused since P3's `voice_changer/worker.rs` stopped calling it); full `cargo test` 1483 passed / 0 failed / 85 ignored once `speech::two_pass`'s engine-error test got room for its five jobs (it had failed under load with `queue_full`). **Live llama-server:** `llm_server::tests::real_model_end_to_end` passed with the pinned b11146 Vulkan build and the hash-verified Hy-MT2 1.8B Q4 (RTX 4090 picked as `Vulkan0` over the iGPU, ready in 1.6–2.9 s, an English line translated to Japanese, stop waited for the process to exit); warm lines take about 50–110 ms (mt-bench below).

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green; vitest covers LocalAgreement, router, prompt builder and language mapping | Gates above. `translation/__tests__/`: `localAgreement.test.ts` (tokenizer round trip, CJK word agreement, a 400-run monotonic-prefix property test), `router.test.ts` (chains, breaker, fall-through on throw and on hang, cloud fallback, interim engine), `prompt.test.ts` (every family, a 300-run fencing property test), `languages.test.ts` (198 languages, fixture parity with Rust `flores.rs`), plus `pipeline`, `tracker`, `glossary`, `sequencer`, `llmEngine` and the engine suites; `llm/__tests__/` (client, stream, providers, transport, sanitize) and `transform/__tests__/streaming.test.ts` | Met for P4 scope (see the Rust note above) |
| Hy-MT2 1.8B on the helper-launched llama-server translates finals | Automated: a fake llama-server through the real start path (`llm_server` lifecycle tests: start → ready, key required, reuse, replace, stop during load, crash loop); the plan table (23 rows); the translation LLM engine against the frozen client / host contracts (`llmEngine.test.ts`); the Hy-MT2 1.8B Q4 GGUF downloaded and hash-verified (`dc5f44fc…c06699`). **Live (final gate):** `real_model_end_to_end` started the pinned b11146 Vulkan server through `process_host` with Sigil's flags, device pin and key, translated a line and stopped it; mt-bench translated all 240 corpus lines with 0 errors | Met for the server path; the in-app run (a spoken line in `pnpm tauri dev`) is manual row 1 |
| Interims never flicker (monotonic committed prefix); finals are never dropped | `pipeline.test.ts` property test (80 seeded scenarios: random segments, given-up segments, late interims, Smart Turn verdicts, random engine timing, streamed partials): every STT final translated exactly once and in order; committed interim text prefix-monotonic; revisions strictly rising; nothing after the final. Captions re-render only the changed tail (`revising.test.ts`: node identity kept; animated run never dropped below the common prefix). Browser smokes (P4-C, P4-G) on production bundles | Met |
| Killing Sigil leaves no llama-server running | `process_host::tests::helpers_die_when_sigil_is_killed` (a fake Sigil started through the real process host is killed with `TerminateProcess`; helper and grandchild exit) runs in the normal `cargo test`; llama-server is started only through `process_host::spawn` (Job Object, kill on close) | Met (automated); the Task Manager check with the real server is manual row 1 |
| Stopping the LLM falls back to Bergamot or NLLB within one timeout | `router.test.ts` with fake timers: a hung engine falls through at exactly `attemptTimeoutMs` (the attempt is aborted), two hung engines answer after two timeouts, breaker open / probe, cloud-to-local fallback; P4-C's browser smoke: chain [llm, libretranslate] came up on the fallback when the LLM could not start, translated the final and showed the fallback warning | Met (automated); real engines in manual rows 1–2 |
| The MT chrF table is recorded | NLLB-200 600M greedy and beam 4, and Hy-MT2 1.8B Q4 on llama-server Vulkan (corpus chrF 59.9, p95 ≤ 109 ms), on the 240-line smoke corpus ([SPEECH_BENCHMARKS.md → Machine translation](benchmarks/SPEECH_BENCHMARKS.md#machine-translation)); `mt-bench` chrF equals sacrebleu 2.6.0 on every direction (max difference 1.4e-14) | Met for NLLB and Hy-MT2; the Bergamot row waits on manual row 2 |

**Manual checks still open** (need the real app, a helper executable, keys or servers):

1. **llama-server + Hy-MT2 1.8B in the app.** The `real_model_end_to_end` test and the Hy-MT2 chrF row were done at the final gate; what is left is the in-app part below. Known gap: the VRAM claim does not resize from the logged buffers, because b11146 prints them only at `-lv 4` (the claim stays at the plan's 1600 MB; the logged buffers were 1380 MB + 100 MB base). The verified GGUF and the b11146 Vulkan zip are in `%LOCALAPPDATA%\sigil-dev-cache\p4b-local-assets`. In PowerShell: `$env:SIGIL_TEST_LLM_RUNTIME = '<folder containing llama-server.exe>'`, `$env:SIGIL_TEST_LLM_GGUF = "$env:LOCALAPPDATA\sigil-dev-cache\p4b-local-assets\Hy-MT2-1.8B-Q4_K_M.gguf"`, then `cargo test --manifest-path src-tauri/Cargo.toml real_model_end_to_end -- --ignored --nocapture`. Then in `pnpm tauri dev` (with `SIGIL_LOCAL_ASSETS` pointing at that folder): Translation → AI model (LLM) → This PC, install the requirements, **Test load**, translate a spoken line, try **Automatic** routing, then end `Sigil.exe` in Task Manager and confirm `llama-server.exe` is gone. Also check that the webview reaches the server directly or through the proxy fallback.
2. **Bergamot pack** ([ADR 0008](adr/0008-bergamot.md)): `build.ps1 -FetchOnly -PinArchives`, `build.ps1`, import the zip, download `mt-bergamot/en_ja` and `ja_en`, then `build.ps1 -SelfTest -Models <app data>\models\mt-bergamot` (p95 < 150 ms per line). If the MSVC build is a no-go, take the ADR's fallback (remove Bergamot from the default chain and interim order).
3. **DeepL** with a real key: `SIGIL_DEEPL_KEY=<key> cargo test --manifest-path src-tauri/Cargo.toml deepl_live -- --ignored`, or start the DeepL backend with a Free (`:fx`) key (usage bar, a translated line, a wrong key shows the 403 message).
4. **AI transform and translation against real services:** a Gemini key (the key is sent in `x-goog-api-key`; if Google rejects it, switch Gemini's `keyHeader` to `bearer` in `llm/providers.ts`), and a real Ollama on localhost (the automatic proxy fallback on its 403 for the app's origin).
5. **MADLAD-400 3B** (marked experimental): whether llama-server b11146 serves this encoder-decoder GGUF.

---

## P5 exit matrix (voice families)

P5 added the voice families (DDSP-SVC, Sigil-ZS, MeanVC2, so-vits-svc, sidecar protocol v2), the shared voice library and the family UI ([plan → P5 outcomes](SPEECH_STACK_IMPLEMENTATION_PLAN.md#p5-outcomes-binding-amendments-for-p6p7)). Gates at P5 integration (2026-10-01, dev rig H2, no other job running): `pnpm typecheck` clean; `pnpm i18n:check` OK (229 keys merged: 86 from P5-F, 143 from P5-G, with six family texts corrected to the measured delays first); `pnpm test:unit` 98 files / 1485 tests; `pnpm build` OK; `cargo clippy --all-targets` exit 0 with **0 warnings**; full `cargo test` **1800 passed / 0 failed / 103 ignored**. Playwright not rerun (no route or `index.tsx` change). Benchmarks on the integrated release exe: the main-path A/B and the family runs below. **Final gate** (2026-10-01, after the review fixes and the fixers' late requests: per-voice output-buffer minimums, the sidecar load cancel, the P5-C / P5-F / P5-G i18n merges): `pnpm typecheck` clean; `pnpm i18n:check` OK (15 keys added, `family_meanvc_tick_info` reworded); `pnpm test:unit` 98 files / 1507 tests; `pnpm build` OK; `cargo clippy --all-targets` exit 0 with 0 warnings; full `cargo test` **1844 passed / 0 failed / 105 ignored**. The final gate changed only load-time code (the factory's family arm, the background loader, the worker's first load), not the per-block path, so the main-path A/B below stands.

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green | Above | Met |
| Zero-shot from a recorded 5–10 s clip meets T10 on H2 (H1 scheduled) | P5-B's benchmark (`sigil_low`, CUDA and DirectML, the Stable 80 / 150 ms buffer; since the final gate the family's minimum is per preset and EP, 95 / 165 ms for `sigil_low` on CUDA and 85 / 155 on DirectML, because 80 ms underran 75 times per 20 s at idle-clock spikes): paced p95 252–256 ms idle (target 300), p99 248–280 ms under the H2 load simulator (target 450), similarity 0.582 against 0.559 for upstream offline on the same reference (exact CAM++, three sources). P5-I0's run of the integrated build: offline call p50 24.4 ms (CUDA) / 29.9 ms (DirectML); paced p95 247.9 / 254.6 ms idle; p99 247.9 ms under the H2 load simulator with 0 underruns; load 1.23 / 1.34 s (T9 ≤ 5 s); scored against its one-off ceiling (CAM++ 0.83, 0 clicks). The reference path: a clip saved by the voice library (`a_take_recorded_through_the_capture_hub_is_trimmed_saved_and_listed`), resolved by `voice_library::reference_clip`, cached per library voice (`library_voices_cache_their_reference`); the benchmark used an 11 s clip that starts with 1.7 s of room tone (`ja-short`; the prompt is taken from its speech, as for a library clip). A recording with a real microphone is manual row 1 | Met on H2; H1 scheduled (4090 × 3 estimate in [VC_BENCHMARKS](benchmarks/VC_BENCHMARKS.md#sigil-zs-zero-shot-p5-b)) |
| Singing tracks up to D#7 | P5-A's real-model test (`real_model_singing_range_and_cost`): the sung E2 → D#6 sweep at +12 st, pitch path and output 0.00 % gross pitch error with CREPE, SwiftF0 or fusion F0 on CUDA and DirectML, including the frames at ≥ 2 kHz (pYIN-style tracker and librosa pYIN); RMVPE misses (2.36 %), so new DDSP-SVC voices start with CREPE in the singing range. P5-I0's run of the integrated build: the benchmark's output of the sweep at +12 st scores 0.00 % GPE against the intended notes with librosa pYIN on CUDA and DirectML (9.7 cents, 106 frames at ≥ 2 kHz, `scripts/bench/score_singing.py`), and 0.85 % / 1.33 % against its offline ceiling (CREPE analyzer); paced 214.9 ms with 0 underruns at the family's 70 ms buffer | Met (with a seeded random voice; a real trained voice is manual row 3) |
| Both are scored against their offline ceilings by the benchmark | The `quality` mode builds each family's own ceiling since P5-I0 (`bench::ceiling::family_ceiling`: Sigil-ZS's one-off conversion of the whole recording, DDSP-SVC through a second instance configured for the whole recording); runs on the integrated build in [VC_BENCHMARKS → Family fidelity](benchmarks/VC_BENCHMARKS.md#family-fidelity-against-the-offline-ceiling-p5-i0) | Met |
| A stalled sidecar never blocks the loop, and its stale audio is dropped by the budget | `sidecar_stall_never_blocks_the_loop` (a sidecar that never answers is muted once per frame at the budget; every call returns at once), `a_stalled_server_never_blocks_the_caller` (the cancel is delivered even with a full queue), `sidecar_recovers_after_a_stall_and_drops_stale_audio` (2 s of late replies against a 400 ms budget: each stalled frame fails once, is cancelled, and audio is back afterwards), `sidecar_700_ms_delay_stays_within_budget`; live: the real Seed-VC model through the real helper and harness, 0 muted blocks idle, late frames muted at the budget under 57–98 % GPU load from other jobs, no call over about 20 ms | Met |
| The main engine is not made worse | The integrated build against HEAD (P3) on the P3 reference specs, 4 interleaved rounds: p95 CUDA −0.4 %, DirectML +1.2 %, CUDA with `gpu_priority` on +0.2 % (P5-S0's +7 % flag was noise); [VC_BENCHMARKS → The main path after P5](benchmarks/VC_BENCHMARKS.md#the-main-path-after-p5-p5-i0) | Met |

**Manual checks still open** (need the real app, a microphone, a checkpoint or another PC):

1. **Reference voice with a real microphone.** In `pnpm tauri dev`: record with the meters (or let it stop at 12 s), preview, save with a name; the voice appears in the voice changer's zero-shot picker, plays back, and its transcript arrives. Also `cargo test --manifest-path src-tauri/Cargo.toml real_microphone -- --ignored` (the dev rig had no default input device) and `SIGIL_APP_DATA_DIR=<app data with an STT model> cargo test real_stt -- --ignored`.
2. **Zero-shot in the app.** Install `vc/zs-seed-vc` (Models & Runtimes → Import from the export folder), **New voice… → Sigil-ZS** from the recorded voice, start, switch the reference while running, hot-switch to an RVC voice and back; check the latency readout includes the voice's output buffer (95 ms prefill for `sigil_low` on CUDA, 85 ms on DirectML). Blind-listen to the joins.
3. **DDSP-SVC with a real trained 6.3 voice:** import its `.pt` / `.zip` in the app (and by dropping the `.pt`), listen, and run the T10 sweep through `real_model_singing_range_and_cost` with `SIGIL_TEST_DDSP_VOICE` (or the `-Profile p5` `ddsp-*` templates plus `scripts/bench/score_singing.py`).
4. **so-vits-svc:** export a 4.1 voice with `onnx_export.py`, add it with **New voice… → so-vits-svc**, A/B it against so-vits-svc's own inference on the same clip, then drop its `experimental` flag (`families/sovits.rs` and its test).
5. **MeanVC2:** download `wavlm_large_finetune.pth` by hand (Google Drive), make `speaker.npy` with upstream `preprocess/extract_spk_emb.py` or export the encoder as `vc/meanvc2-speaker`, then the listening and speaker A/B against RVC Streamer (decides whether it can be the CPU default).
6. **Sidecars:** X-VC and StreamVoiceAnon with their checkpoints in a Python environment (several GB each), a sidecar voice in the app, and a release installer that contains `resources/neural-sidecar` (`pnpm tauri:release`, then check the install folder).
7. **H1 live-record** of `sigil_low` / `sigil_lean` and DDSP-SVC on an RTX 3060 / 4060 under VRChat + OBS (with and without NVIDIA's "Prefer maximum performance").

## Research §6 bug verification

Every bug in [SPEECH_STACK_RESEARCH_2026-09.md §6](SPEECH_STACK_RESEARCH_2026-09.md#6-bugs-found-in-the-current-code-verified-in-this-repo) and the extra bugs found while mapping the subsystems, with how the fix is verified.

| Bug | Fix | Verified by |
|-----|-----|-------------|
| On-device translation checked `typeof … === "object"` and never started | `isCtorLike` accepts functions; shared detector; replaced translators destroyed | `translation/__tests__/engines.test.ts` (Translator defined as a function) |
| Fish Speech used `/v1/models` | `/v1/health`, `/v1/references/list`; streamed WAV repair; optional Bearer key and `model` header; hosted API support | `fish_speech` Rust tests (header repair, reference list, hosted models, header validation) |
| Voice-changer aux queue capped at 2 s, draining from the front | Whole-clip `VecDeque` with back-pressure (60 s queued, 600 s per clip), rubato resampling, mic gate | `aux_feed::sixty_second_aux_clip_stays_intact`, `full_queue_refuses_whole_clips_and_recovers`; manual 60 s clip |
| 2 ms sleep-poll and a BELOW_NORMAL worker | park/unpark wake, `worker_priority`, MMCSS on callback threads | `budget` tests; manual idle-CPU and MMCSS log check |
| Whisper RMS gate flushed silence; uncapped buffer | Silero VAD v6.2 segmenter with 18 s forced split, bounded ordered worker | `speech_core::segmenter` and `utterance` tests (10 s of silence emit nothing), `whisper::tests::real_model_transcribes_speech_and_filters_silence` (ignored, real model) |
| VOICEVOX pitch slider could not reach 0 | Step 0.01 plus **Reset pitch**; AivisSpeech preset resets pitch | `tts/__tests__/voicevox.test.ts`; manual check |
| VOICEVOX `parseFloat(x) \|\| 1.0` turned 0 into 1 | `num()` / `finiteOr` helpers | `voicevox.test.ts` (scales), `voicevoxEngine.test.ts`, `backends.test.ts` |
| Deepgram sent `tier=` but no `model=` | `model=nova-3` (Nova-2 selectable); start needs only key and language | `stt/__tests__/sttOptions.test.ts`; manual Deepgram start |
| Azure STT never set `Speech_SegmentationSilenceTimeoutMs` | Phrase end silence setting (default 500 ms) | `sttOptions.test.ts` (validation) |
| Uberduck used undocumented legacy endpoints | v1 API: Bearer key, `GET /v1/voices`, `POST /v1/text-to-speech` | `uberduck_tts` Rust tests |
| Stale `openRouterModel` default | `google/gemini-2.5-flash-lite` | `transform/__tests__/llmRequest.test.ts` |
| Status strings leaked into captions | `onStatus` channel, never published | `stt/__tests__/statusChannel.test.ts`, `service.test.ts` |
| Dead global shortcuts (registered only with background input; toggles fired twice) | Registered in every desktop build; Pressed-only filter | `keyboard/__tests__/accelerator.test.ts`; manual VRChat check |
| Broken `native` STT default on desktop | One-time migration to Whisper | `stt/__tests__/migrate.test.ts`, `service.test.ts` |
| 500 ms polling finals for HTTP STT (no rate limiting) | One request per VAD utterance, Groq-safe limiter | `utterancePipeline.test.ts`, `rateLimiter.test.ts` (never more than 20 in any 60 s) |
| Translation finals dropped by the `#id` gate | `TranslationSequencer` | `translation/__tests__/sequencer.test.ts`, `service.test.ts` |
| AI transform truncated at 100 tokens; `[AI Rate Limited]` published | `maxTokens` (default 1024); original text plus notice on failure | `llmRequest.test.ts`, `ordering.test.ts` |
| Monitor resampler clicks; input drop-newest; model load after streams opened; F32/I16-only capture; sidecar without timeout; `crepe` alias; no VC persistence | Stateful polyphase resampler; drop-oldest budget; load before streams; any sample format; budget-bounded sidecar; `crepe` → `rmvpe`; Rust-side persistence | `voice_changer` tests (`resample`, `budget`, `persist`, `state`, `params`, `contract_tests`), `sidecar_bridge::exchange_times_out_and_resynchronizes` |
| GPU EPs fell back to CPU silently | Bundled provider DLLs, EP ladder with canary and `error_on_failure`, visible outcome | `ort_session` / `diagnostics` tests; verify-bundle; [GPU execution provider smoke](#gpu-execution-provider-smoke) |
| Foreign websites could open the pubsub WebSocket; binary audio injection into Whisper | Origin allow-list, binary frames refused, inbound topic allow-list | `web::origin` / `pubsub` / `peer` Rust tests, `pubsub/__tests__/inbound.test.ts`; LAN manual matrix |
| XSS in the `/client` startup error and `inputfield.html` | React-rendered error; `textContent` rows | Scratch Playwright checks (P0-I); manual |
| Canvas text element parsed caption text, masks and emote URLs as HTML (Twitch chat could run script in the host webview); `/client?host=` could point the mirror at any server | Text nodes, `textContent` masks and `createElement("img")` emotes; `/client` connects only to the server that served it (or loopback); text frames from the host are validated and neutralized on the client | `tests/smoke.spec.ts` (client host pinning), `utils/__tests__/localPages.test.ts`, `pubsub/__tests__/inbound.test.ts`; scratch browser check of the text renderer (P0 gate) |
| HTTP calls without per-request timeouts; Japanese HTML error pages panicked the preview | Timeout classes, UTF-8-safe previews | `http_client` tests (every byte offset of a Japanese page, hung server) |

---

## Network security and threat model

This document expands the short security notes in [ARCHITECTURE.md](../ARCHITECTURE.md) and [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for operators and contributors. It is **not** a formal pen-test report.

### Components

| Surface | Binding | Purpose |
|---------|---------|---------|
| **Embedded HTTP server** (`web` plugin) | `0.0.0.0` — all IPv4 interfaces (configured port), LAN-reachable | Static UI assets, PeerJS signaling HTTP, pubsub WebSocket; `/pubsub` and `/peer` behind an Origin allow-list |
| **PeerJS** | Host advertises LAN IP + port to clients | `/client` loads in a browser; connects to host for Yjs document sync and related data |
| **Linked pubsub** | Outbound WebSocket from host to a user-supplied URL (optional) | Bridges pubsub to another machine or service when “link” mode is enabled |
| **OAuth loopback** | Localhost redirect handlers | Twitch / Kick (and similar) OAuth callbacks |

### Trust assumptions

1. **Local server is LAN-reachable** — The `web` plugin binds `0.0.0.0`, so anything that can open TCP connections to `<host>:<sigil_port>` — other users on a **shared** machine (terminal server, borrowed laptop) **and any host on the same network** — can reach the same HTTP/WebSocket endpoints the webview uses, without your password. Since LAN security stage 1, the WebSocket handshakes on `/pubsub` and `/peer` pass an **Origin allow-list** (below), which blocks browser cross-site WebSocket hijacking and DNS rebinding; a **non-browser** tool can still forge an Origin until stage 2 adds a token. Mitigation: run Sigil only on trusted accounts and networks; firewall the port to only where you intentionally use `/client` / LAN features; do not expose it via SSH port forwarding, proxies, or tunnels unless you understand the impact.

   | Allowed Origin | From |
   |----------------|------|
   | `https://tauri.localhost`, `http://tauri.localhost`, `tauri://localhost` | Anywhere (Sigil webviews: the host app and Link apps from another Sigil) |
   | `http://<this machine>:<port>` (localhost, `*.localhost`, loopback, this PC's interface IPs, `COMPUTERNAME` / `COMPUTERNAME.local`) | Anywhere (pages this server served) |
   | `http://localhost:1420` (Vite dev server) | Debug builds only |
   | No Origin header | Only from this machine (loopback or one of its interface IPs) |

   Everything else is refused with **403**: `null`, `file://`, malformed or repeated headers, other websites, browser extensions and DNS-rebinding host names. Refusals are logged and emitted as `web:origin_rejected` (throttled per origin; the host shows a toast). The kill switch is **Settings → Local web server → Only allow Sigil pages to connect** (`lanOriginCheck`, read at startup and live), overridden by `SIGIL_LAN_ORIGIN_CHECK=0/1`. A bind failure is reported (`web:server_status` and a toast) instead of silently killing the server.

2. **LAN / client mode** — Peers on the same network who know the host address and port can attempt to join as **clients**. They receive document/collab traffic the host shares over PeerJS; they **do not** execute Tauri `invoke` or read arbitrary disk. Still treat the session as **sensitive**: captions, chat-derived text, and canvas content may be visible. Use a trusted LAN, guest Wi‑Fi isolation, or VPN as appropriate.

3. **Linked pubsub** — The host opens a **WebSocket client** to the URL you configure. That endpoint (and anyone controlling DNS/TLS termination in front of it) can receive and inject pubsub-shaped traffic if the protocol is not authenticated. Only link to servers you operate or explicitly trust; prefer TLS (`wss://`) on non-loopback URLs.

4. **Linked pubsub uses `ws://` only** — The implementation in [`src/shared/services/pubsub/index.ts`](../src/shared/services/pubsub/index.ts) builds `ws://${address}/pubsub?…` (no `wss://` option). If the UI is ever served over **HTTPS**, or you terminate TLS on a reverse proxy, **mixed content** or browser **secure context** rules may block or complicate linking. Treat secure remote linking as a future feature requiring explicit `wss` support and certificate handling.

5. **`text.stt` on the local pubsub WebSocket** — Peers attached to the host’s pubsub socket (e.g. **mic.html** for Chrome/Edge external STT) send JSON that the host validates against an **inbound topic allow-list** ([`src/shared/services/pubsub/inbound.ts`](../src/shared/services/pubsub/inbound.ts)): `obs.caption` (schema-checked), `stt.external.state` (local pages only) and the `text.*` caption topics; everything else (bare `text`, `stt.external.command`, `scenes:change`, …) is dropped. The Rust server no longer relays pubsub between peers, messages are capped at 256 KiB, and **binary frames close the socket with 1003**, so the old binary-audio-to-Whisper injection path is gone. Messages carry a `via` hop chain so Link apps cannot loop. A non-browser LAN tool that forges an allowed Origin can still inject STT-shaped text until stage 2; do not expose the port beyond trusted networks.

6. **No token auth on the local server yet** — The design assumes single-user desktop use on a trusted network. Stage 1 (Origin allow-list, binary refusal, topic allow-list, XSS fixes in the `/client` error fallback and `inputfield.html`, bind-failure reporting) closes the browser-reachable attacks; stage 2 (planned) adds a persistent token for non-loopback clients. Adding intentional remote access, tunneling, or public exposure would still require real authentication, rate limits, and TLS review.

7. **NVIDIA runtime pack** — Downloads only the pinned NVIDIA redistributable URLs with pinned sha256, one file at a time with HTTP Range resume, and extracts only the DLLs into a staging folder that is renamed into place (the model-store pack `runtime/nvidia-cuda13` through the store's own safe extractor; the interim Settings installer with Windows' `tar.exe`, refusing `../` entries). The DLLs load by full path right before the first CUDA session.

8. **Diagnostics IPC** — `log_from_frontend` is webview-only, size-capped and rate-limited; the deep EP probe runs as a separate child process (`Sigil.exe --headless ep-probe`).

9. **Model store** — Catalog entries are pinned to immutable revisions with sha256 or git blob ids (release builds refuse unpinned published entries); downloads resume with HTTP Range, fail over on errors, HTML bodies, wrong sizes and hash mismatches, and are verified before an atomic rename. Archives are extracted with traversal, absolute-path, drive-letter, alternate-data-stream, device-name, case-duplicate, link, entry-count, size and compression-ratio checks. Trust-on-first-use exists only in debug builds and never for executables; `unpublished` entries never download from catalog URLs. A custom mirror must be https, or http on the LAN / loopback, and its files are still checked against the pins. `SIGIL_MODEL_CATALOG_OVERRIDE` / `SIGIL_LOCAL_ASSETS` are developer overrides.

10. **Native runtime packs** — Nothing new is load-time linked. Engine DLLs load lazily from `app_data/runtimes/<name>/<version>/<variant>/` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | DEFAULT_DIRS`, after every file matched the sizes and sha256 in the pack's `sigil-pack.json`, the ABI version and the engine info. sherpa-onnx runs a CPU-only ONNX Runtime 1.28.2 inside its pack next to Sigil's static 1.28.0; C++ exceptions do not cross its C API, so a corrupt model would abort Sigil: catalog models are hash-verified and user-supplied folders are test-loaded in a child process (`validate_in_child`) first.

11. **Helper processes** (`process_host`: llama-server, later Bergamot and TTS servers) — Started suspended and joined to a kill-on-close Job Object before they run (they die with Sigil, even when Sigil is killed), `CREATE_NO_WINDOW`, optional committed-memory cap. They must listen on `127.0.0.1` only: after readiness Sigil reads the OS listener table for every process in the job and stops a helper with any non-loopback TCP listener. Each gets a random 256-bit key (from `ProcessPrng`), passed by environment variable, redacted from logs and never sent to the webview. The health client uses no proxy and follows no redirects.

12. **Untrusted voice files** — RVC `.pth` / `.zip` imports are parsed by Sigil's allow-list pickle VM (only `OrderedDict`, torch tensor rebuilders, storages and `_codecs.encode`; legacy pickles and TorchScript refused) behind a zip guard, in the `Sigil.exe --headless vc-import-worker` child (Job Object limits via `process_host::run_worker` in P3-B); the worker enforces a wall-clock limit today.

**PeerJS signaling** — The client uses `secure: false` against the local HTTP signaling server ([`src/shared/services/peer/provider.ts`](../src/shared/services/peer/provider.ts)). Moving signaling to HTTPS would require revisiting the `secure` flag and URL scheme together.

### Hardening checklist (operators)

- Keep the app updated; re-run `pnpm audit` / `cargo audit` before releases (see [Dependency audit notes](#dependency-audit-notes)).
- Firewall: allow inbound TCP on Sigil’s port only where you intentionally use `/client` or LAN features.
- Reverse proxies: if TLS terminates in front of linked pubsub, set WebSocket idle timeouts above app keepalives (see [Performance and connection-timeout audit](#performance-and-connection-timeout-audit)).
- After changing Tauri plugins or capabilities, review `src-tauri/capabilities/` and IPC allowlists.
- Keep **Only allow Sigil pages to connect** on. If a trusted tool is refused, prefer loading it from `http://localhost:<port>/…` over turning the check off; OBS must load `http://localhost:<port>/client` or `/obs-captions.html`, not a local HTML file.
- Keep the **asset protocol** scope (`app.security.assetProtocol.scope` in `src-tauri/tauri.conf.json`) limited to `$APPDATA/vosk-models/**`, its only consumer (Vosk model zips). Widening it to `$APPDATA/**` would expose settings with API keys and every downloaded model to anything running in the webview; `$APPLOCALDATA/**` on Windows is the WebView2 profile (cookies, local storage, IndexedDB).

### Related docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — stub pointing here and to DEVELOPER_GUIDE (served at `/ARCHITECTURE.md` when bundled)
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) (feature ↔ IPC), [Performance…](#performance-and-connection-timeout-audit) (timeouts), [Major upgrades backlog](#major-upgrades-backlog)

---

## Dependency audit notes

Re-run before releases: `pnpm audit` (repo root) and `cargo install cargo-audit` then `cd src-tauri && cargo audit`.

### Last refresh

**Date:** 2026-04-09

### pnpm (`pnpm audit`)

Last run reported **5 vulnerabilities** — **1 high**, **4 moderate**:

| Severity | Package   | Notes |
|----------|-----------|--------|
| High     | lodash    | `_.template` / prototype pollution paths; patched in `>=4.18.0` (direct dep `^4.17.23`) |
| Moderate | esbuild   | Via `vite` — dev-server request issue; patched in `esbuild >=0.25.0` |
| Moderate | nanoid    | Direct `^4.0.2`; patched in `>=5.0.9` |
| Moderate | vite      | Optimized deps `.map` path handling; advisory lists fix in `>=6.4.2` (current line is Vite 5.x) |

**Recommendation:** Plan a dedicated toolchain upgrade (Vite 6+, aligned `esbuild`, lodash 4.18+, nanoid 5+) on a separate PR with full UI and Tauri testing; do not blindly `pnpm audit fix --force` without verifying breaking changes.

### Rust (`cargo audit`)

Install: `cargo install cargo-audit` then `cd src-tauri && cargo audit`.

**Last run:** **0 vulnerabilities** after bumping transitive `bytes` to `>=1.11.1` (RUSTSEC-2026-0007 — `BytesMut::reserve` overflow) via `cargo update -p bytes` in `Cargo.lock`.

**Warnings:** **22** allowed informational items (unmaintained / unsound advisories on transitive crates — mostly Linux GTK / WebKit stack from Tauri, `uwuify` → old `clap`, `tauri-utils` → `urlpattern` / `unic-*`, etc.). Address by upgrading Tauri ecosystem over time, not per-crate overrides unless a concrete CVE applies.

Track `whisper-rs` / `whisper-rs-sys` alignment in `Cargo.toml` when updating whisper.cpp bindings.

**Dependency changes in the P0 speech-stack pass:** added `tracing` 0.1 and `tracing-subscriber` 0.3 (file log; the 10 MB × 5 rotating sink is in-house because `tracing-appender` only rotates by time, so `tracing-appender` was dropped again) and the `windows` features JobObjects, Dxgi, Dxgi_Common, Console and LibraryLoader; `ort` now enables the `cuda` and `directml` features; `vitest` 3.2 as a dev dependency. The existing `rubato` and `rtrb` crates now also serve `speech_core`, audio capture and the voice changer. Re-run `cargo audit` / `pnpm audit` before the next release.

**Dependency changes in the P1 pass:** `ort` 2.0.0-rc.11 → `=2.0.0-rc.13` (ONNX Runtime 1.28.0; features `cuda`, `directml`, `nvrtx`, `tensorrt`, `half`, `preload-dylibs`, `copy-dylibs`, `ndarray`; one-time pyke dist download of about 62 MB); `kittentts` 0.3.0 vendored into `local_tts/kitten/` and removed with its `hf-hub` dependency; added `espeak-ng` 0.1 (`bundled-data`), `anyhow`, `regex`, `fancy-regex`, `once_cell`, `zip` 2 (deflate only), `half` 2, `libloading` 0.9, `tar` 0.4, `bzip2` 0.6 (pure-Rust), `flate2` 1, and `serde_json` as a build dependency (catalog validation). The CUDA provider shrank from 144 MB to 62 MB and imports only cuBLAS / cuBLASLt. `ort_session` refuses a provider DLL whose file version differs from the linked ONNX Runtime. Re-run `cargo audit` / `pnpm audit` before the next release.

**Native pack reproducibility:** `sigil_engine` packs are built locally with `/Brepro`, fixed zip timestamps and sorted entries (clean rebuilds are byte-identical), from sha256-pinned upstream sources (transcribe.cpp's configure step also checks its ABI hash). Packs are hash-verified at load through `sigil-pack.json` and stay `unpublished` until the maintainer publishes them. `pwsh scripts/runtime-packs/sherpa/probe-matrix.ps1` and `scripts/bench/run-perf-gate.ps1` belong to the performance checks before a release that touches those runtimes.

**Known upstream issue:** whisper.cpp (vendored by `whisper-rs-sys` 0.15) reads a stale logits row for its per-segment no-speech probability whenever the prompt has more than one token (every multilingual model, any initial prompt), and used that value to discard windows. Sigil disables the internal skip and computes the value itself; worth reporting upstream.

---

## Third-party components

License facts for components embedded in or downloaded by Sigil (Sigil itself is AGPL-3.0-only).

| Component | Where | License | Notes |
|-----------|-------|---------|-------|
| Silero VAD v6.2 | Embedded (`src-tauri/resources/models/silero_vad_v6_2.onnx`, `include_bytes!`) | MIT, © 2020-present Silero Team | Source tag v6.2; sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`; notice in `src-tauri/resources/models/silero_vad_v6_2.LICENSE.txt` |
| ONNX Runtime provider DLLs, DirectML | Installed next to `Sigil.exe` by the release installers | MIT (ONNX Runtime), Microsoft DirectML redistributable | Staged from the ort build by `scripts/release/stage-native-dlls.mjs` |
| ONNX Runtime 1.28.0 | Linked statically into `Sigil.exe` (ort 2.0.0-rc.13, pyke dist) | MIT | Provider DLLs from the same build are staged into the installers |
| NVIDIA CUDA 13.2 (cudart, cuBLAS 13.3, cuFFT 12.2) / cuDNN 9.26 (`cuda13`) redistributables | Downloaded at runtime on request (`runtime/nvidia-cuda13`, or the interim Settings pack) | NVIDIA redistributable licenses | Pinned URLs and sha256 in `model-catalog/runtime-nvidia.json` and `diagnostics/nvidia_pack.rs` |
| KittenTTS 0.3.0 (Rust port) | Vendored in `src-tauri/src/services/local_tts/kitten/` | Apache-2.0 | SPDX headers, `LICENSE` and `NOTICE` kept in the folder |
| espeak-ng (pure-Rust port, bundled data) | Linked (crate `espeak-ng` 0.1) | GPL-3.0-or-later | Kitten G2P |
| sherpa-onnx 1.13.8 (with its ONNX Runtime 1.28.2, CPU) | Runtime pack `runtime/sherpa-onnx` downloaded on request | Apache-2.0 (sherpa-onnx), MIT (ONNX Runtime) | Official upstream `win-x64-shared-MT` archive, hash-pinned; bindings generated from the pinned `c-api.h` |
| transcribe.cpp 0.2.4 (with its vendored ggml) | Sigil-built runtime packs `runtime/sigil-transcribe-*` (local import) | MIT | Packs ship `LICENSE-transcribe.cpp.txt`, `THIRD-PARTY-LICENSES-transcribe.cpp.md` and `NOTICE.txt` |
| llama.cpp b11146 (`llama-server`) | Runtime packs `runtime/llama-server-*` downloaded on request | MIT | Official ggml-org release assets, pinned from GitHub digests |
| Hugging Face `tokenizers` 0.23 | Linked (crate, `fancy-regex` feature, no default features) | Apache-2.0 | NLLB tokenizer (`mt_local::tokenizer`) |
| Mozilla Bergamot (bergamot-translator, marian, intgemm, sentencepiece, ssplit-cpp, ruy, simd_utils, onnxjs) | Sigil-built runtime pack `runtime/bergamot` (`sigil-bergamot.exe`, local import) | MPL-2.0 (bergamot-translator), MIT (marian, intgemm), Apache-2.0 (sentencepiece, ruy), per-component for the rest | Built by `scripts/runtime-packs/bergamot/build.ps1` from the commits in `helper/UPSTREAM.json`; the pack ships their license files |
| Translation models (GGUF `mt/*`, NLLB-200 ONNX, Firefox Translations `mt-bergamot/*`) | Downloaded on request through the model store | Per catalog entry (for example MPL-2.0 for the Firefox Translations models, CC-BY-NC-4.0 for NLLB-200) | Pinned to Hugging Face commits / Mozilla's registry sha256 in `model-catalog/{llm-gguf,mt-nllb,mt-bergamot}.json` |
| RVC-WebUI model code (commit 81eed5e8) | Vendored in `scripts/vc-templates/rvc_arch/` (maintainer tooling, not compiled into Sigil) | MIT | `LICENSE` kept; the bundled `resources/vc-templates` graphs are exports of this architecture (no weights) |
| Applio model code (IAHispano/Applio, commit c7665ac9) | Vendored in `scripts/vc-templates/applio_arch/` (maintainer tooling, not compiled into Sigil) | MIT | Unchanged, `LICENSE` kept; used only by the maintainer cross-check of the RefineGAN / MRF templates |
| RVC v2 default voice (PhoenixStormJr/RVC-V2-default-voice) | `vc/voice-default`, downloaded on request (first-run card) and imported | MIT | Revision 22e4cd2b, `default.pth` sha256 `c9d6b0ac…bfccc`, plus its `added_IVF511_Flat_nprobe_1_default_v2.index` |
| Content encoders (ContentVec 768 / 256, SPIN, SPIN v2, Japanese / Korean / Chinese HuBERT) | Sigil exports of Applio's Hugging Face checkpoints (`vc/embedder-*`, local import) | MIT (ContentVec, SPIN, SPIN v2, Chinese HuBERT), Apache-2.0 (Japanese, Korean HuBERT) | Built by `scripts/vc-templates/export_embedders.py` (source commit 70ed5638); entries `unpublished` |
| Pitch models: RMVPE, SwiftF0, CREPE tiny (onnxcrepe), FCPE (community ONNX export of torchfcpe), PESTO | `vc/pitch-*`, downloaded on request (FCPE: Sigil adapter, local import; PESTO: import only) | MIT (RMVPE, SwiftF0, onnxcrepe, FCPE), LGPL-3.0 (PESTO) | Upstream files hash-pinned in `model-catalog/vc-bases.json` |
| Denoisers: GTCRN, DeepFilterNet3, DPDFNet | `vc/denoise-*`, downloaded on request | MIT (GTCRN), MIT OR Apache-2.0 (DeepFilterNet3), Apache-2.0 (DPDFNet) | Hash-pinned in `model-catalog/vc-denoise.json` |
| Benchmark models: DNSMOS P.835, CAM++ (3D-Speaker via sherpa-onnx) | `bench/*`, downloaded only for benchmarks | MIT (DNSMOS), Apache-2.0 (CAM++) | Hash-pinned in `model-catalog/bench-models.json` |
| TensorRT-RTX runtime | Optional local pack `runtime/tensorrt-rtx` (not downloaded by Sigil) | NVIDIA TensorRT-RTX license (runtime), MIT (ONNX Runtime provider) | `unpublished`; needed only for the Studio-TRT preset |
| Seed-VC (Plachtaa/seed-vc@51383efd), DDSP-SVC 6.3 (yxlllc/DDSP-SVC@d6dd52fc), MeanVC2 (ASLP-lab/MeanVC2@13acf84c) model code | Vendored in `scripts/vc-families/*/upstream/` (export spike tooling, not compiled into Sigil) | GPL-3.0 (Seed-VC), MIT (DDSP-SVC), Apache-2.0 per the upstream README (MeanVC2) | Unchanged below their provenance headers; [ADR 0006](adr/0006-vc-zero-shot-singing.md) |

---

## Performance and connection-timeout audit

This document captures baseline methodology, pass/fail thresholds (tune to your hardware), operational guidance, and implementation notes from the audit.

### Test scenarios

| # | Scenario | What to verify |
|---|----------|----------------|
| 1 | Host idle 1–5 min after boot | CPU %, private bytes, thread count stable |
| 2 | Host + local STT/TTS backends | No hung `invoke`; errors within timeout |
| 3 | `/client` same machine | PeerJS connects; canvas syncs |
| 4 | `/client` over LAN | Same as 3; tolerate brief packet loss |
| 5 | Linked pubsub (`linkConnect`) | WS survives idle; reconnects after drop |
| 6 | External browser vs Tauri webview | Same URL: comparable WS/Peer stability |

### How to measure

- **Process:** Task Manager or Process Explorer — `Sigil.exe` CPU (average over 60s idle), Memory (private working set), Threads.
- **Main thread:** Chrome DevTools → Performance — record 10s idle; note long tasks &gt; 50ms.
- **WebSocket:** Network → WS — frames/sec when idle (should be near zero except keepalive pings).
- **Bundle:** Set `ANALYZE=true` and run `npm run build` (PowerShell: `$env:ANALYZE="true"; npm run build`), then open `stats/rollup-stats.html` (treemap). The visualizer is skipped by default so normal builds stay faster.

### Suggested pass/fail thresholds (adjust per release)

| Metric | Target (guidance) |
|--------|-------------------|
| Host idle CPU (no chat/OBS) | &lt; 2–5% on a typical laptop |
| Private working set growth | No unbounded climb over 30 min idle |
| Linked pubsub idle | Connection alive ≥ 10 min behind a consumer router, or reconnect ≤ 30s |
| Outbound HTTP (local TTS/STT) | Fails or completes within configured timeout (no indefinite hang) |
| `/client` | Connects to host PeerJS within 30s on LAN under normal conditions |

### Top hotspots (from bundle + code review)

Last measured production build (gzip sizes in build log):

| Asset | Size (min) | Notes |
|-------|------------|--------|
| `index-*.js` (main) | ~438 KB (~135 KB gzip) | After `manualChunks` split |
| `react-vendor-*.js` | ~349 KB (~116 KB gzip) | React |
| `twurple-*.js` | ~232 KB (~54 KB gzip) | Twitch integration |
| `collab-*.js` | ~193 KB (~58 KB gzip) | PeerJS / Yjs |
| `vosk-*.js` | ~5.8 MB (~2.3 MB gzip) | Lazy-loaded with Vosk STT path only |
| Inspector chunks | 1–24 KB each | Already `React.lazy` in [`src/core/ui/inspector/index.tsx`](../src/core/ui/inspector/index.tsx) |

Inspect `stats/rollup-stats.html` after `ANALYZE=true npm run build`. Large deps: Twurple, Microsoft Speech SDK, OpenAI, framer-motion, PeerJS/Yjs, lodash. Inspectors are lazy; further wins come from `manualChunks` (see `vite.config.ts`) and avoiding new eager imports of heavy SDKs.

### Operational guidance

- **Firewalls:** Allow inbound TCP on the configured Sigil port for `/client` and PeerJS (`/peer`).
- **Reverse proxies:** If terminating TLS in front of Sigil, set WebSocket read/write timeouts above the app keepalive interval (`LINK_PUBSUB_KEEPALIVE_MS` in [`src/shared/services/pubsub/index.ts`](../src/shared/services/pubsub/index.ts), default 25s) or disable proxy idle timeout for that route.
- **Background browser tabs:** Chrome may throttle timers; client uses reconnection logic — prefer keeping the client tab focused during live use.
- **Service worker (`/client`):** Workbox precaches the production build. After a host update, clients may need a refresh so the SW picks up new assets; `index.tsx` already listens for `updatefound` and reloads when appropriate.

### Service lifecycle matrix (host)

| Service | Timers / network idle when “off” | Notes |
|---------|-----------------------------------|--------|
| Twitch | No live interval until logged in | `setInterval` only after successful `connect()` |
| Kick | Same | |
| Others | Unchanged from prior behavior | Further lazy-init can follow the same pattern |

### Rust HTTP timeouts

Model-store downloads (`model_download_client()`) have **no total timeout** and a **30 s stall timeout** (`MODEL_STALL_TIMEOUT`); a stalled or dropped transfer resumes with HTTP Range on the same mirror before failing over.

Shared client: [`src-tauri/src/services/http_client.rs`](../src-tauri/src/services/http_client.rs) — connect 10 s, and any connection that sends nothing for **120 s** is dropped (model downloads included; their total cap stays 2 h). Every API call also sets its own timeout:

| Call class | Timeout |
|------------|---------|
| Health checks | 10 s |
| Voice / language / model lists | 20 s |
| Translation | 20 s + 10 ms per character, at most 60 s |
| TTS | 30 s + 100 ms per character, at most 120 s |
| Moonshine STT | 30 s + the clip length, at most 120 s |

Audio responses are read in chunks (30 s stall limit, 64 MB cap) and a 200 response that is JSON/text/HTML is reported as an error. Errors surface as plugin `Result::Err` strings that carry the server's own message, with query strings and credentials removed from URLs.

### Frontend `fetch` inventory

| Area | Pattern |
|------|--------|
| Kick OAuth/API | `fetchWithTimeout` (25–30s) |
| VoiceVox | `fetchWithTimeout` + existing `AbortSignal` for cancel |
| Discord webhooks | `fetchWithTimeout` (25s) |
| Azure translate (languages list + translate API) | `fetchWithTimeout` (20 s) plus the translation sequencer deadlines (15 s interims / 30 s finals) |
| HTTP STT (local server / Groq) | `fetchWithTimeout` (local 120 s / cloud 60 s), one request per VAD utterance, aborted on stop |
| Twitch emote loaders (BTTV/FFZ/7TV) | `fetchWithTimeout` (25s) via [`src/core/services/twitch/emote_loaders.ts`](../src/core/services/twitch/emote_loaders.ts) |
| LLM transform | 15 s per sentence through an `AbortController` (the HTTP request is actually cancelled) |

### Soak checklist (manual)

1. Start host, leave 60+ minutes idle; note memory every 15 min.
2. Open `/client`, disconnect Wi‑Fi 10s, reconnect; confirm recovery without full page reload when possible.
3. Enable linked pubsub, idle 15+ min, confirm state stays connected or shows reconnecting.

---

## Major upgrades backlog

Use this list to plan larger follow-on work after the current release push. Order is suggested priority, not fixed.

### Execution order (integration branch)

Work proceeds on branch `integration/major-upgrades` in phases: **0 baseline → 1 hygiene → 2 architecture (TS) → 3 client → 4 Tauri v2 → 5 security/docs → merge to `master`. Each phase should leave the app buildable. See the Cursor plan “Integration branch major upgrades” for full detail.

### Platform and build

- **Tauri v2** — Migrate app shell, IPC, and plugins when dependencies are ready; re-validate all `invoke` paths and bundling.
- **Release CI** — [`.github/workflows/release.yml`](../.github/workflows/release.yml) uploads NSIS, MSI and a portable zip by file glob (the old exact MSI name would not have matched); add macOS/Linux jobs if you ship those.
- **Rust toolchain** — Periodically `cargo update` / `cargo audit`; keep `whisper-rs` / `whisper-rs-sys` versions aligned.

### Architecture

- **App runtime API** — Narrow use of `window.ApiServer` / `window.ApiClient` behind a small typed facade for tests and clearer init order.
- **Shared layer** — `consumePubSubMessage` routes inbound topics through the pure allow-list in `inbound.ts`; the STT hand-off still goes through a handler registered from core.
- **Reactive state** — Finish moving incidental Valtio UI slices to Zustand where it simplifies React subscriptions; keep Yjs/document on existing stack.

### Product and quality

- **E2E** — Extend Playwright beyond Vite preview (e.g. smoke on packaged app or documented manual checklist for Tauri-only flows).
- **TTS/STT matrix** — Scripted or manual pass per backend (native, Windows, Azure, Whisper, Vosk, etc.) with “minimum config” documented.
- **i18n and legacy strings** — Replace remaining “curses” product strings in `public/i18n` and static HTML where Sigil is the product name.
- **Client mode** — Harden `/client` for production: errors when host unavailable, reconnect, and security notes for LAN use.

### Optional performance

- **Inspector** — Already lazy-loaded; consider lazy routes for heavy dashboard-only views if cold start is still an issue.
- **Bundle** — Revisit Rollup/manual chunks for largest editor chunks; keep `stats/` out of git (see `.gitignore`).

### Security

- **Invoke surface review** — Re-check Tauri allowlist and custom commands after any plugin changes.
- **WebSocket / pubsub** — Threat model for localhost, LAN `/client`, and linked pubsub: [Network security and threat model](#network-security-and-threat-model).

See also [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for how host, client, and native pieces fit together.
