# Release and engineering

Release QA checklists, **network threat model**, **dependency audits**, **performance** notes, and **upgrade backlog**.

## Table of contents

- [Release build smoke matrix](#release-build-smoke-matrix)
- [Manual Tauri smoke checklist](#manual-tauri-checklist)
- [Local packaged smoke (manual)](#local-packaged-smoke-manual)
- [P0 exit matrix (speech stack stabilization)](#p0-exit-matrix-speech-stack-stabilization)
- [P1 exit matrix (platform foundations)](#p1-exit-matrix-platform-foundations)
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
- The base `tauri.conf.json` bundles `src-tauri/resources/vc-templates/` (the RVC split-graph templates, under 1 MB; checked-in sources, so compile-time copying is safe); the release overlay's `target/native-bundle` mapping merges with it.
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
6. **Neural** — ONNX TTS load path and optional sidecar (`http://127.0.0.1:8766`) if you ship those features.
7. **Second instance** — Launch a second Sigil while the first holds the port; expect the native port-unavailable dialog.

See also [Manual Tauri checklist](#manual-tauri-checklist) for a broader feature pass.

---

## Manual Tauri checklist

Automated Playwright tests use Vite preview only; native Tauri behavior requires a manual pass before release.

### Host (desktop app)

- [ ] `npm run tauri dev` — app window opens, no startup error overlay in `#root`.
- [ ] **Config / network** — `plugin:web|config` loads; status/footer shows expected host/port.
- [ ] **Whisper** — download or load model path (progress shows under the Start button, never in captions); start recording; one **final** per spoken sentence (`whisper:segment_transcript`); silence produces nothing.
- [ ] **One STT path** — e.g. native or engine you ship; Start/Stop from bottom bar or inspector.
- [ ] **One TTS path** — voice selected; Start; speak or trigger text event.
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

---

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
| RVC-WebUI model code (commit 81eed5e8) | Vendored in `scripts/vc-templates/rvc_arch/` (maintainer tooling, not compiled into Sigil) | MIT | `LICENSE` kept; the bundled `resources/vc-templates` graphs are exports of this architecture (no weights) |
| RVC v2 default voice (PhoenixStormJr/RVC-V2-default-voice) | Development parity tests only (`SIGIL_TEST_MODELS_DIR`) | MIT | Revision 22e4cd2b, sha256 `c9d6b0ac…bfccc`; the zero-setup default voice lands with P3 |

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
