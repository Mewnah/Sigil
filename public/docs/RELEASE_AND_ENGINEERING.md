# Release and engineering

Release QA checklists, **network threat model**, **dependency audits**, **performance** notes, and **upgrade backlog**.

## Table of contents

- [Release build smoke matrix](#release-build-smoke-matrix)
- [Manual Tauri smoke checklist](#manual-tauri-checklist)
- [Local packaged smoke (manual)](#local-packaged-smoke-manual)
- [Optional checks (verify when convenient)](#optional-checks-verify-when-convenient)
- [P0 exit matrix (speech stack stabilization)](#p0-exit-matrix-speech-stack-stabilization)
- [P1 exit matrix (platform foundations)](#p1-exit-matrix-platform-foundations)
- [P2 exit matrix (speech engine)](#p2-exit-matrix-speech-engine)
- [P3 exit matrix (voice engine)](#p3-exit-matrix-voice-engine)
- [P4 exit matrix (translation stack)](#p4-exit-matrix-translation-stack)
- [P5 exit matrix (voice families)](#p5-exit-matrix-voice-families)
- [P6 exit matrix (streamer features and extended engines)](#p6-exit-matrix-streamer-features-and-extended-engines)
- [P3B exit matrix (voice-engine optimization)](#p3b-exit-matrix-voice-engine-optimization)
- [P7 exit matrix (measure, set defaults, document, release)](#p7-exit-matrix-measure-set-defaults-document-release)
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
- `node docs/check-docs.mjs --no-todo` — dead links, anchors that match no heading, malformed tables, statements that disagree with the code (UI labels, the Streamer thread rule, unrouted models, the headless table) and leftover `TODO(P7-numbers)` placeholders in the documentation.
- `node scripts/release/check-catalog-pins.mjs` — every published catalog entry is hash-pinned and immutable, and the release set (the defaults a new install fetches) is complete ([scripts/release/README.md](../scripts/release/README.md#catalog-pins-step-1)); `--online` re-derives the defaults' sizes and digests from the hosts' metadata.
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
6. **Local speech** — STT **Built-in (runs on your PC)**: Start asks before downloading, then captions arrive; TTS **Kokoro / Piper (local)**: the panel's download, **Preview**, then speech through the chosen device; optional HTTP sidecar (`scripts/neural-sidecar/tts_sidecar.py`) through **OpenAI-compatible (HTTP)**.
7. **Second instance** — Launch a second Sigil while the first holds the port; expect the native port-unavailable dialog.

See also [Manual Tauri checklist](#manual-tauri-checklist) for a broader feature pass.

---

## Manual Tauri checklist

Automated Playwright tests use Vite preview only; native Tauri behavior requires a manual pass before release.

### Host (desktop app)

- [ ] `npm run tauri dev` — app window opens, no startup error overlay in `#root`.
- [ ] **Config / network** — `plugin:web|config` loads; status/footer shows expected host/port.
- [ ] **Whisper** — download or load model path (progress shows under the Start button and in Models & Runtimes, never in captions); start recording; one **final** per spoken sentence (`whisper:segment_transcript`); silence produces nothing.
- [ ] **Built-in (runs on your PC)** — fresh profile: Start asks before downloading (sizes match the route preview); downloads show progress; live partials, then one final per sentence; STT mute drops the open sentence.
- [ ] **One STT path** — e.g. native or engine you ship; Start/Stop from bottom bar or inspector.
- [ ] **One TTS path** — voice selected; Start; speak or trigger text event. Long text starts after its first sentence; **Skip** / **Stop speaking** on the status bar.
- [ ] **Twitch** — `SIGIL_TWITCH_CLIENT_ID` set; Twitch console lists **loopback** URLs: **Public** → `http://localhost:17890/oauth/twitch/implicit`; **Confidential** → `http://localhost:17890/oauth/twitch/callback`; plus **Vite** `http://localhost:1420/oauth/...` if you use `pnpm dev`, and the **packaged app** `https` origin + path if flows hit the webview (see [manual/03-integrations.md](manual/03-integrations.md#oauth-redirects-and-ports)); OAuth completes; optional: IRC chat + post/receive.
- [ ] **Discord** — webhook URL; one post with “post only when live” on and off if you use that gate.
- [ ] **OBS** — WebSocket connect; captions only send when “Enable” is on under native captions.
- [ ] **VRChat, SteamVR and a second PC** — optional; see [Optional checks](#optional-checks-verify-when-convenient).
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

- **Built-in (runs on your PC)** — English and Japanese: consent dialog, downloads, live partials replaced by one final per sentence; Stop during a download cancels it (the next Start resumes).
- **Groq** (real key) — captions arrive; the status line shows a rate-limit wait when you talk continuously; a bad key stops with a readable message; nothing is sent while muted.
- **Local HTTP server on the LAN** (OpenAI-compatible, for example speaches on another PC) — transcribes (the request is sent by Rust, so the webview CSP does not apply).

Optional helper: `scripts/packaged-smoke-stt.ps1` prints this checklist and launches the release exe if present.

### Clean machine: no GPU, no packs

The release must start and work on a PC with no GPU and no runtime pack installed (a fresh profile; a virtual machine, or Windows Sandbox where that feature is already enabled, or a spare PC). On the installed NSIS build:

1. The app starts: no error overlay on `#root`, the status shows the embedded server's port, and **Models & Runtimes** lists every entry as not installed.
2. **Settings → GPU & diagnostics** reports the processor (or DirectML when the machine has a GPU adapter) with the reason, and **Test GPU acceleration** completes.
3. **Built-in (runs on your PC)** with English: **Start** asks before downloading (the dialog shows the size), the download shows progress and resumes after **Stop**, and captions arrive. **Kokoro / Piper (local)** downloads and previews.
4. `Sigil.exe --headless sysinfo` and `Sigil.exe --headless fetch --list` print their reports and exit 0.
5. `pwsh scripts/release/verify-bundle.ps1` finds no load-time import outside `scripts/release/expected-imports.txt`.

The scripted version is [`scripts/release/packaged-smoke-all.ps1`](../scripts/release/README.md#packaged-smoke-steps-4-and-5) (Windows PowerShell 5.1 or PowerShell 7, one file for the dev PC, Windows Sandbox, a VM and a CI runner): `-Mode Extract` (the default) unpacks the NSIS setup and runs the app against a scratch profile, so nothing is installed and your own profile is untouched; `-Mode Sandbox` runs the installer in Windows Sandbox with the virtual GPU off (the Sandbox viewer window cannot be hidden: run it when you are away from the PC); `-Mode Installed -AllowInstall -ExpectNoGpu` is the step list for a VM ([manual VM steps](../scripts/release/README.md#clean-machine-smoke-without-windows-sandbox)). The hosted Windows runner of `.github/workflows/release.yml` has no GPU and covers the no-GPU leg. Every GUI launch is minimized (`SIGIL_START_MINIMIZED=1`) and the `gui-focus` check proves the app never took the foreground; `gui-exit` closes the window by title and checks that the process ends.

### GPU execution provider smoke

On the same release build ([ADR 0003](adr/0003-gpu-ep-delivery.md)):

1. **CPU baseline** — voice changer **ONNX** or **RVC** mode on **CPU**; the status panel says “Running on CPU”.
2. **NVIDIA with the pack** — Models & Runtimes → **NVIDIA GPU runtime (CUDA 13, cuDNN 9)** → **Download** (or Settings → GPU & diagnostics → NVIDIA GPU runtime → **Download and install**; about 1.9 GB, sequential, resumable), stop/start the voice changer, expect **CUDA** in the status panel and the diagnostics panel (**Test GPU acceleration** reports the runtime source). Then `pwsh scripts/release/verify-bundle.ps1 -RequireCuda`. CUDA needs driver R580+ and a Turing or newer GPU.
3. **NVIDIA without the pack** — a CUDA request runs on **DirectML** with a warning box, one toast and a System log line naming the missing DLL (for example `cudnn64_9.dll`).
4. **Without DirectML.dll** (admin-install the MSI, delete the bundled `DirectML.dll`, run `Sigil.exe --headless ep-probe --out probe.json`) — Sigil still starts; the probe notes that the System32 copy is used, or reports the CPU fallback with its reason.
5. **Test GPU acceleration** in the diagnostics panel completes and saves `app_data\diagnostics\ep-probe-deep.json`.

**Future automation (optional):** A `workflow_dispatch` Windows job or a local script that launches the built `exe` with a timeout is possible; tracked as backlog in [Major upgrades backlog](#major-upgrades-backlog) if product priority allows.

---

## Optional checks (verify when convenient)

Nothing in this section blocks a release. These checks need something the release does not depend on: VRChat or SteamVR running on this machine, a headset, a second PC or a phone for the LAN token matrix, or a friend's RTX 3060 / 4060 or CPU-only PC for live recordings. Sigil's main audience is VRChat streamers who run captions into OBS, and the automated tests, the release smoke above and the benchmark's game-load simulator (which stands for a running game; no measurement was made inside VR) are what the release rests on. Run these when you next have the setup in front of you and file anything that fails as an issue.

The checks moved here from the exit matrices below, which now point back to them. The labels (P6-12, P3-3, ...) are the old row numbers of those matrices.

### VRChat and VR

- **VRChat OSC (from the Manual Tauri checklist):** desktop build only; verify the chatbox / OSC target with VRChat running.
- **P6-12.** **Other players:** start VRChat, Other players → default source → Start: only other players are captioned (not your TTS or music). Close and restart VRChat (fallback on: the default output meanwhile, VRChat again within 5 s; fallback off: the panel waits). On a Windows 10 build below 20348 the default output is used. Reverse translation with a local engine; `others.*` reaches `/client` and OBS pages but is refused from Link apps. `cargo test --manifest-path src-tauri/Cargo.toml real_process_capture -- --ignored` passes on Windows 11.
- **P6-13.** **Caption window** in the real app: transparency, click-through, never stealing focus from a game, the unlocked frame moves and resizes, it closes with the main window and restores its position, **Reset position**, and XSOverlay / OVR Toolkit can pin "Sigil captions".
- **P6-14.** **SteamVR overlay** with SteamVR: it connects and shows captions; head / hand / world attach and **Recenter**; quitting and restarting SteamVR; CJK and RTL legibility in the headset; texture updates at most 10 Hz. Without SteamVR running no `vrserver` / `vrmonitor` / `vrstartup` process appears (checked on the dev PC, which has no SteamVR). `Sigil.exe --headless fetch runtime/openvr` verifies the pinned DLL.
- **P6-15.** **VRChat OSC:** enable OSC in VRChat and **Mute captions when I mute in VRChat**: the status shows the port, "VRChat found on this PC" and "Receiving from VRChat"; muting in VRChat stops captions at once, unmuting resumes them; a mute you set in Sigil survives a VRChat mute / unmute; OSCQuery off with another app on 9001 shows "port in use" while sending still works; pages arrive about 1.5 s apart with nothing cut; a mid-thought pause arrives as one chatbox message with Smart Turn and **Wait until I finish speaking**; SenseVoice tags drive `SigilEmotion` / `SigilEvent` on an avatar.

### LAN token matrix (a phone or a second PC)

P6-1 to P6-3 run on this PC alone; the rest need a phone or a second PC. Stage 2 of the LAN security work (the persistent LAN token). The automated part (59 Rust `web::` tests and the vitest token vectors) is in the [P6 exit matrix](#p6-exit-matrix-streamer-features-and-extended-engines); only the two-device behaviour is manual.

- **P6-1.** **LAN, fresh profile:** Settings → LAN access shows a masked token; after one restart the log shows `LAN access: lan (token set)`.
- **P6-2.** **OBS on this PC** with `http://localhost:3030/client` and `/obs-captions.html` connects without a token.
- **P6-3.** **Chrome / Edge mic page:** `mic.html` on 127.0.0.1 works without a token.
- **P6-4.** **Phone on the LAN** with **Copy LAN canvas link** connects; the same URL without `?token=` shows the invalid-key message, and the host shows "Refused a device on your network".
- **P6-5.** **Second PC with OBS** and **Copy LAN captions link** (or the OBS tab's **Copy LAN link**) shows captions; without the token the page says the LAN token is missing or out of date.
- **P6-6.** **Link apps between two PCs:** **Copy my address** on A, paste into B's address field (fills address and remote token), connect both ways, text flows with no repeats.
- **P6-7.** **New token** on A: B's link stops ("refused the link") and the phone mirror disconnects; copying again restores both.
- **P6-8.** **Legacy on:** the phone URL without a token connects; legacy off disconnects it.
- **P6-9.** **This PC only:** the log shows a rebind to 127.0.0.1:3030; other devices are disconnected and cannot connect; OBS here keeps working; back to LAN rebinds 0.0.0.0. A profile set to This PC only starts without a firewall prompt (the port pre-check binds 127.0.0.1).
- **P6-10.** **Foreign origin:** devtools on https://example.com, `new WebSocket('ws://localhost:3030/pubsub?id=x&token=<token>')` fails with 403.
- **P6-11.** **Origin-less tool from another PC** (`websocat 'ws://<ip>:3030/pubsub?id=x&token=<token>'`) connects; without the token it is refused.

### Live sessions and friends' PCs

The voice-engine numbers in [VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md) come from the dev rig with the benchmark's load simulator. A live session confirms them on real hardware.

- **P3-3.** **30-minute soak** under VRChat + OBS with 0 underruns (T7).
- **P3-4 / P5-7 / P3B-1.** **H1 and H3 live records:** a friend's RTX 3060 / 4060 PC under VRChat + OBS (T1, T2, T3, T7 authoritative; also `sigil_low` / `sigil_lean` and DDSP-SVC, with and without NVIDIA's *Prefer maximum performance* for Sigil.exe, which removes the idle-clock paced penalty), and a 6-core CPU with VRChat (T4). The T1 / T2 / T3 / T4 rows are simulator numbers until then.
- **P3B-2.** **Live device for the new pacing controls:** `device_block=model_divisor` on WASAPI (44.1 and 48 kHz devices, shared and exclusive), the adaptive cushion under a real game; P7-B decides their defaults.
- **P7-Q.** **Live device for the CPU output buffer floor:** a processor-only template voice on 120 ms blocks (the Streamer's automatic block on 7 or more physical cores; the 160 ms block of 6 cores takes a 145 / 171 ms buffer and the 200 ms block of 5 or fewer a 180 / 206 ms buffer instead; or Autotune's block, or a **Block size** you set) on a WASAPI device (44.1 and 48 kHz, shared and exclusive; a Bluetooth headset): no dropouts or clicks with the automatic 110 / 136 ms buffer (`SIGIL_VC_BLOCK_BUFFER=0` switches it off), and whether the roughly 270 ms delay is acceptable. The benchmark and the replays of its recordings cannot see device timing; a device period above 10 ms needs a larger buffer of your own.
- **P3-5 / P3B-5.** **True end-to-end latency** with the e2e-loopback meter over a virtual cable (T1; the dev PC's cable endpoints are disabled).


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

Steps 3 and 7 need a phone or a second PC: run them when convenient (the token version of this matrix is in [Optional checks](#optional-checks-verify-when-convenient)).

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
| CPU-only: en gives Nemotron partials, zh SenseVoice finals, ja Qwen3-ASR-0.6B finals (P2; since P7 ja is a SenseVoice final and ru a Parakeet Ultra final, see the [P7 exit matrix](#p7-exit-matrix-measure-set-defaults-document-release)); the router never sends ja / zh / ko to Cohere | Shared routing fixture (56 cases) in Rust (`speech::router`) and TS (`router.test.ts`), plus the exhaustive rule test over every language × tier × PC profile. Through the real pipeline with real Silero VAD (`through_the_pipeline::real_sherpa_engines_decode_through_the_pipeline`): Nemotron en 9 partials then the final 160 ms after VAD end; SenseVoice zh final 233 ms with `lang:zh` / `emotion:neutral` tags; Qwen3-ASR-0.6B ja final 1216 ms; Parakeet Ultra de 287 ms | Met |
| On the dev GPU, two-pass finals replace partials under the same segmentId ≤ 800 ms after VAD end for an 8 s utterance | `real_two_pass_ja_with_vram_logged` (RTX 4090, Qwen3-ASR 1.7B Q8_0 through the governor and the real two-pass queue): 199 ms and 194 ms after VAD end, same segment, revision + 1; process VRAM 384 → 3476 MiB. The TS `SegmentPublisher` publishes one final per segment (`localDriver.test.ts`) | Met |
| Kokoro starts playing after the first sentence (≤ 400 ms on CPU); Stop cancels; every PCM backend honours the device and can route into the VC | Kokoro v1.0 on 4 threads: median RTF 0.294 (Sigil's ORT with the CPU arena, now the default; 0.315 inside sherpa-onnx). The sentence pipeline starts sentence 1 before synthesizing sentence 2 (`pipeline.test.ts`); first audio of a long first sentence 656 ms (its opening clause is about 2 s of audio), so ≤ 400 ms holds only for short opening clauses. Cancel stops a long utterance in about 0.6 s; skip / stop cut audio already in the sink ring (`tts_sink` tests). Device field honoured with tolerant names and route `device` / `voiceChanger` / `both` (`tts_sink` and `aux_feed` tests; P2-G and P2-I mocked-IPC smokes) | Met except the 400 ms target for long opening clauses (P7 decides); the real-device checks are manual rows 5–7 |
| New installs default to `local`; native-backend users get a one-time prompt | `stt/migrate.ts` + `tts/defaults.ts` planners and `Service_STT` / `Service_TTS` tests (fresh install → `local` / `localNeural` silently; an existing native profile is asked once, no → Whisper; offers never repeat; web build untouched). Production-bundle smoke with an IPC mock: a fresh profile boots on `local` + `localNeural` with no dialog; an existing profile on Web Speech gets the STT offer, then the TTS offer, and the language carries over (`ja-JP` → `ja`) | Met |

**Manual checks still open** (need the real app, devices, servers or keys):

1. **Fresh profile in `pnpm tauri dev`** (move `%APPDATA%\com.sigil.app\user\settings` away first; automated runs always seed a settings file and so never see the wizard): the setup wizard opens, five steps, with **Skip setup** on every step but the last. The speech step shows the **Built-in (runs on your PC)** card and asks for your Windows language (not English) as a question; the profile starts on that engine (the one-time desktop-default migration sets it), and finishing puts the Speech to Text card in the bottom bar. Quitting in the middle of the wizard shows it again at the next start, and a profile that has a settings file (even with `onboardingComplete` false) never sees it. Press Start with English, agree to the download, captions arrive; switch to Japanese (Nemotron partials plus a SenseVoice final; Korean is Qwen3-ASR 0.6B, Russian Parakeet Ultra); an existing profile on Web Speech sees the STT offer, one on Chrome or Edge sees the same offer worded for its browser (**Switch to built-in** / **Keep Chrome** or **Keep Edge**, and Keep never asks again), and the Kokoro offer too once the profile uses TTS (auto-start on, a picked voice, or started once); a profile saved on the retired Moonshine server opens on Built-in with the notice and does not auto-start. Then **Rewrite**: start Speech to Text and Rewrite together, the dot on the Rewrite tab lights, the bottom-bar Rewrite card opens that tab, `Ctrl+2` toggles it.
2. **Whisper with a real microphone:** the download row fills and Models & Runtimes shows `whisper/<id>` installed; captions; the STT hard mute drops the open sentence; replugging a USB microphone continues the recording.
3. **Vosk and Groq:** Vosk English captions and mute; Groq with a real key (captions, rate-limit status, bad-key message, nothing sent while muted); an OpenAI-compatible STT server on the LAN.
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
3. **30-minute soak** under VRChat + OBS: moved to [Optional checks](#optional-checks-verify-when-convenient) (P3-3).
4. **H1 live-record** from a friend's PC: moved to [Optional checks](#optional-checks-verify-when-convenient) (P3-4).
5. **True end-to-end latency** over a virtual cable: moved to [Optional checks](#optional-checks-verify-when-convenient) (P3-5).
6. **Corpora:** the 20-clip consented speech corpus (T5) and the user's 40-voice community corpus (`SIGIL_TEST_VC_CORPUS_DIR=<folder> cargo test --manifest-path src-tauri/Cargo.toml community_corpus_report -- --ignored --nocapture`, T8).
7. **T6 head to head:** RVC WebUI realtime and Applio realtime through `scripts/vc-bench`, w-okada through `play_record.py`, scored with a `compare` run and the blind A/B kit (`tools/listening-test`).
8. **TensorRT-RTX (T9):** put a TensorRT-RTX plugin execution provider plus the runtime DLLs its imports name in a folder, set `SIGIL_TRT_DIR` to it, run the staged test executable twice with `--ignored gpu_lever_bench` (`SIGIL_BENCH_CONFIGS=all+accel`): the second run's cold load on the TRT row must be ≤ 3000 ms and the log must say "engine loaded from the cache". The provider pyke ships with ort rc.13 imports TensorRT 10 names, so it needs a matching runtime (or a provider built against TensorRT-RTX).
9. **Maintainer checks:** `python scripts/vc-templates/parity_test.py --upstream-applio` (must print max|d| ≤ 1e-4; running the downloaded Applio code was not permitted to the agent), and a PESTO export with PESTO's own `realtime/export_onnx.py`.
10. **GPU tests on the dev rig** (ignored tests): copy the test executable, `DirectML.dll` and `onnxruntime_providers_{cuda,shared}.dll` into one folder, set `SIGIL_APP_DATA_DIR` to an app-data folder with `runtime/nvidia-cuda13` installed, then run `--ignored gpu::tests::` (device plan, CUDA graphs, priority stream, shared arena, lever benchmark).

## P4 exit matrix (translation stack)

P4 ran in the same working tree as P3 (the voice-changer engine). Gates at P4 integration (2026-09-30, same dev rig): `pnpm typecheck` clean, `pnpm i18n:check` OK (375 keys added: 373 staged plus 2 model-family labels; 2 removed), `pnpm test:unit` 84 files / 1279 tests, `pnpm build` OK (speech SDK chunk hash unchanged, `speech-sdk-CdZJ51IR.js`; the `openai` chunk is gone with the dependency), Playwright 8 / 8 on that build, and an IPC-mock host smoke of the production bundle (boots; the new "Local model server" init task calls `llm_server_status`; the translation provider list, the language model panel with the Local model server block and the Models & Runtimes family labels render from the merged strings). Rust (run once P3's files compiled again): `cargo clippy --all-targets` exit 0 with **0 warnings in P4 files** (the 34 remaining are in P3's in-progress `bench`, `neural_audio`, `voice_changer` files and the now-unused `pcm_convert::build_input_stream_f32`); full `cargo test` 1433 passed / 14 failed / 82 ignored, where 13 failures are in P3's in-progress files (`bench::{dsp,engine,fidelity,quality,spec}`, `neural_audio::denoise::stft`, `voice_changer::{playback,timeline}`) and 1 is a P2 timing test under load (`speech::two_pass::…lost_engine_stops_decoding`, passes alone); P4-scoped filters all green: `llm_server` 44 passed / 2 ignored, `mt_local` 59 / 5, `translate::` 18 / 1, `process_host` 60 / 4 (including `helpers_die_when_sigil_is_killed`), `model_store` 79, `catalog` 40, `resources` 17, `cli` 45. **Status: green for the P4-owned scope; the full Rust gate waits on P3.** Real-model runs (ignored tests, P4 workstreams): NLLB tokenizer parity 27 / 27 with Python sentencepiece, greedy output 24 / 27 exact (3 near-ties), the worker path; the Bergamot client against the C++ fake helper.

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

1. **llama-server + Hy-MT2 1.8B in the app.** The `real_model_end_to_end` test and the Hy-MT2 chrF row were done at the final gate; what is left is the in-app part below. Known gap: the VRAM claim does not resize from the logged buffers, because b11146 prints them only at `-lv 4` (the claim stays at the plan's 1600 MB; the logged buffers were 1380 MB + 100 MB base). The verified GGUF and the b11146 Vulkan zip are in `%LOCALAPPDATA%\sigil-dev-cache\p4b-local-assets`. In PowerShell: `$env:SIGIL_TEST_LLM_RUNTIME = '<folder containing llama-server.exe>'`, `$env:SIGIL_TEST_LLM_GGUF = "$env:LOCALAPPDATA\sigil-dev-cache\p4b-local-assets\Hy-MT2-1.8B-Q4_K_M.gguf"`, then `cargo test --manifest-path src-tauri/Cargo.toml real_model_end_to_end -- --ignored --nocapture`. Then in `pnpm tauri dev` (with `SIGIL_LOCAL_ASSETS` pointing at that folder): Translation → Language model → This PC, install the requirements, **Test load**, translate a spoken line, try **Automatic** routing, then end `Sigil.exe` in Task Manager and confirm `llama-server.exe` is gone. Also check that the webview reaches the server directly or through the proxy fallback.
2. **Bergamot pack** ([ADR 0008](adr/0008-bergamot.md)): `build.ps1 -FetchOnly -PinArchives`, `build.ps1`, import the zip, download `mt-bergamot/en_ja` and `ja_en`, then `build.ps1 -SelfTest -Models <app data>\models\mt-bergamot` (p95 < 150 ms per line). If the MSVC build is a no-go, take the ADR's fallback (remove Bergamot from the default chain and interim order).
3. **DeepL** with a real key: `SIGIL_DEEPL_KEY=<key> cargo test --manifest-path src-tauri/Cargo.toml deepl_live -- --ignored`, or start the DeepL backend with a Free (`:fx`) key (usage bar, a translated line, a wrong key shows the 403 message).
4. **Rewrite and translation against real services:** a Gemini key (the key is sent in `x-goog-api-key`; if Google rejects it, switch Gemini's `keyHeader` to `bearer` in `llm/providers.ts`), and a real Ollama on localhost (the automatic proxy fallback on its 403 for the app's origin).
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
7. **H1 live-record** of the families: moved to [Optional checks](#optional-checks-verify-when-convenient) (P5-7).

## P6 exit matrix (streamer features and extended engines)

P6 added other players' captions, the caption window and SteamVR overlay, VRChat OSC (mute sync, chatbox pacing, Smart-Turn commit, tags), LAN access stage 2, Qwen3-TTS and the experimental ggml TTS helpers, Style-Bert-VITS2 / AIVMX and GPT-SoVITS, Edge and Gemini TTS with the local fallback, Pocket and Chatterbox in process, SAPI into the TTS sink with the mic policy, Gemini Live STT / Live Translate, readings and the music gate ([plan → P6 outcomes](SPEECH_STACK_IMPLEMENTATION_PLAN.md#p6-outcomes-binding-amendments-for-p7)). P6-M (a user-installed Beatrice 2 VST3) was dropped: it is someone else's closed runtime. P6 ran in the same working tree as P3B (voice-engine optimization), so the gates below were run on the shared tree. Gates at P6 integration (2026-10-01, P3B jobs running in other lanes): `pnpm typecheck` clean; `pnpm i18n:check` OK (646 keys merged from 13 staging files, no conflicts); `pnpm test:unit` 130 files / 1919 tests; `pnpm build` OK (Tailwind's scan now excludes `scripts/` and `src-tauri/`, whose backslash paths and Rust tokens produced invalid CSS candidates; the speech-sdk chunk is unchanged by the P6 `index.tsx` route); Playwright (`PW_PREVIEW_ONLY=1`) 9 passed, including the new `/captions` early-route check; `cargo clippy --all-targets` exit 0 with **0 warnings** crate-wide; full `cargo test` **2320 passed / 0 failed / 128 ignored**. No benchmark was run: P6 does not touch the voice-engine path, and GPU timing belongs to P3B while it measures. Final gate after the review fixes (2026-10-02, P3B measuring on the GPU meanwhile): `pnpm typecheck` clean; `pnpm i18n:check` OK (2 more keys from P6-A's fix); `pnpm test:unit` 134 files / 1974 tests; `pnpm build` OK; Playwright 9 passed; `cargo clippy --all-targets` exit 0 with 0 warnings; full `cargo test` **2351 passed / 129 ignored**, plus one failure outside P6: P3B's `neural_audio::vc::test_models` `fake_rvc_slot_converts_on_directml_or_its_fallback` (DirectML "Unspecified error" while another GPU job ran; P3B-S0 recorded it as flaky under parallel load, and it passed when rerun alone under the GPU timing lock).

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green, every feature OFF by default | Above; `src/core/__tests__/p6Scaffold.test.ts` (every P6 setting defaults off, the new backend values are new enum values, `others.*` / `readings.*` stay outside `text.*`); the Kokoro Japanese reader and the TTS turn gate change nothing without their dictionary / setting | Met |
| The LAN token matrix passes: host, LAN `/client`, OBS, mic.html, Link apps; foreign-origin and unauthenticated LAN clients are rejected | Automated: 59 Rust `web::` tests (`web/auth.rs` combined decisions, close frames, revocation on token / mode / legacy / Origin changes, live rebind, exemption only for same-PC peers without forwarding headers), `lanAccess.test.ts` (token vectors shared with Rust), `hostWebServer.test.ts`, the Playwright `/client` pinning tests. The manual two-PC matrix (P6-1 to P6-11 in [Optional checks](#optional-checks-verify-when-convenient)) has not been run yet | Automated part met; manual matrix open |
| Loopback captions appear only on `others.*` | `others/__tests__/topics.test.ts` and `driver.test.ts` (publishes on `others.stt` only, 0 events on `text`, history empty), `inbound.test.ts` (`others.*` never accepted from WebSocket peers), the P6-A smoke run of a production build with an IPC mock; `real_process_capture` (ignored) attached to a process on build 26200 | Met (a real VRChat session is optional check P6-12) |
| Qwen3-TTS clones from a library clip and streams into the sink | The whole Sigil path against a fake server through the real `process_host` (start, warm-up with TTFA / RTF log, model-type refusal, clone registry, speaker-data cache, crash and restart, VRAM claim); the CPU pack builds and reports its pinned version. The real 0.6B / 1.7B weights were not downloaded (over the phase's 50 MB limit) | Code path met; real model manual row 16 |
| An Edge or Gemini outage falls back to Kokoro within 1 s | Rust mock-server tests (`edge_tts::tests`, `gemini_tts::tests`: 429 / 5xx / refused connection / DNS failure classified in under 1 s; a blackholed connect after about 1.05 s on both (the connect bound is a third of **Wait for the service**, at least 1 s; Edge races the service's IPv6 and IPv4 addresses 250 ms apart, so a dead IPv6 path costs 250 ms, not the bound)); vitest `cloudTts.test.ts` (the failed sentence and the rest of the outage on Kokoro within 1 s, through the full sentence pipeline) | Met for error answers and refused connections (at once) and for an unreachable service (switch after about 1.05 s; Kokoro's first audio about 0.5-0.85 s later). A silent service waits the 3 s **Wait for the service** setting by design (the plan's 3 s stall) |

**Manual checks still open** (need the real app, a key, a GPU run or a large download). Rows 1-15, which need a second PC or phone, VRChat or SteamVR, moved to [Optional checks](#optional-checks-verify-when-convenient) and keep their numbers there as P6-1 to P6-15. Rows 16 and 17 cover the opt-in heavy GPU engines, which P7 deliberately did not benchmark; run them before shipping those engines as tested.

16. **Qwen3-TTS** (hold the GPU timing lock while P3B measures): install `tts/qwen3-tts-0.6b-base` and `tts/qwen3-tts-tokenizer`, import the CUDA 13 (or Vulkan) server pack, clone a library clip, and record the warm-up TTFA / RTF from `sigil.log` and the per-sentence `[Perf]` lines; after **Unload** and **Test** the speaker data comes from the cache; the size swaps to 0.6B when the voice changer starts on an 8 GB card. Or `cargo test real_model_clone_end_to_end -- --ignored --nocapture` with `SIGIL_QWENTTS_EXE`, `SIGIL_TEST_MODELS_DIR`, `SIGIL_GPU_LOCK`. Build the five-architecture CUDA 13 pack and a Vulkan pack (needs the Vulkan SDK).
17. **Experimental GPU voices:** per engine install, **Test**, a cloned voice (Breeze, OmniVoice, S2) and **Unload**; `real_ggml_engine_speaks` per engine with the GPU lock (TTFA, RTF, VRAM); run `s2/build.ps1 -PinFirstFetch`, `fetch-magpie-tokenizer.ps1` and `convert-magpie-v2607.ps1`.
18. **Style-Bert-VITS2:** `Sigil.exe --headless fetch text/jpreprocess-naist-jdic tts/sbv2-bert-ja`, drop an `.aivmx` on the panel, preview and speak (or `real_model_sbv2_speaks` with `SIGIL_TEST_MODELS_DIR`). **GPT-SoVITS:** `python api_v2.py -a 127.0.0.1 -p 9880`, a library voice or **Browse…** clip, **Check**, **Test**.
19. **Edge / Gemini:** speak a few sentences in `pnpm tauri dev` (the Edge voice list loads), then cut the network or block `speech.platform.bing.com`: Kokoro takes over with one notice, and the service is used again after the pause. Gemini with a real free key, including a 429 by speaking quickly on Flash. Fish Audio hosted S2.1 Pro free with a real key.
20. **Pocket / Chatterbox:** install the models (and `runtime/sherpa-onnx`), clone a library voice with both in the app, Start / Stop keeping the model, routing into the voice changer; Chatterbox on CUDA, DirectML and CPU with the fallback line; or `cargo test local_tts -- --ignored --test-threads=1` with `SIGIL_TEST_MODELS_DIR`. Release gate: `real_model_chatterbox_clones_a_library_voice` with the default SIGIL_TEST_CHATTERBOX_EP=cpu is the only run of the real CPU flow step (flow_step_slim.onnx, used by every CPU user and every GPU-to-CPU fallback); `real_model_pocket_clones_a_library_voice` prints the real Pocket package file list (record it; it must include test_wavs/*.wav for the built-in voice). On a laptop with integrated + discrete graphics, check that Chatterbox Auto runs on the discrete card (log line "Chatterbox: loading on dml@N").
21. **Windows SAPI through the voice changer:** voice changer on with **Microphone while TTS plays → Mute**, TTS Windows on **Voice changer**, a multi-sentence text: converted speech sentence by sentence, the microphone silent for the whole reply, one `capture mute VoiceChanger = true` / `false` pair per reply in the log, Skip and Stop cut SAPI at once; **Lower** never touches the hub gate; with the voice changer stopped Windows speech plays on the device; a device saved from the old SAPI list still plays there.
22. **Gemini Live** with a real key: Transcribe with **renew every 1 minute**, speaking across several rotations (no missing or repeated words); mute and unmute; pull the network for about 10 s while speaking (a "reconnecting" line, then the speech transcribed late); Live Translate with the Translation service running (not translated twice).
23. **Readings:** Japanese and Chinese lines in `pnpm tauri dev` with the dictionary card; `readings.stt` reaches `/client`.
24. **Music gate A/B:** turn on **Ignore music and singing (experimental)**, install `vad/firered-aed`, play VRChat world music (segments drop with reason `music`), speak over it (captions continue).
25. **Kokoro Japanese:** with the Japanese dictionary installed, a Japanese Kokoro voice reads kanji with Japanese readings (`日本語` as にほんご, not Mandarin).

## P3B exit matrix (voice-engine optimization)

P3B optimized the Sigil Voice Engine after P3 missed several targets (per-call compute, Low-latency delay, VRAM, the CPU generator) and made the benchmark trustworthy enough to say so ([plan → P3B outcomes](SPEECH_STACK_IMPLEMENTATION_PLAN.md#p3b-outcomes-binding-amendments-for-p7), [ADR 0009](adr/0009-vc-scorer-v2.md), [ADR 0010](adr/0010-sve-optimization-p3b.md)). Gates at P3B integration (2026-10-09, dev rig H2, no other job running): `pnpm typecheck` clean; `pnpm i18n:check` OK (23 keys merged from P3B-S0 and P3B-H, `pnpm i18n:sync` run); `pnpm test:unit` 135 files / 1977 tests; `pnpm build` OK; `cargo clippy --all-targets` **0 warnings**; full `cargo test` **2556 passed / 0 failed / 148 ignored** (the first full run had one failure, `process_host::tests::crashed_helper_is_restarted_then_given_up`, a pid-liveness assertion that passes alone 3 of 3 and in the second full run; it is not a P3B file); the release build of the integrated tree has 0 warnings. Final gate after the review fixes (same day, lane `p3b-dev`): `pnpm typecheck` clean; `pnpm i18n:check` OK (`voice_changer.device_block_not_applied` merged); `pnpm test:unit` 135 files / 1982 tests; `pnpm build` OK; `cargo clippy --all-targets` **0 warnings**; full `cargo test` **2621 passed / 0 failed / 151 ignored** on two consecutive runs. The first full run of that gate failed `bench::compare::tests::a_corpus_run_scores_points_and_refuses_unfair_ones` once: its Sigil row is timed on the wall clock against a 10.7 ms block, a stall of the test process (4.7 s p99 in the reproduction under 24 CPU spinners) marks the row real-time-infeasible and the listening pairs are skipped, as designed; it passed alone, in the whole `bench::` module twice and in two more full runs, and its assertion now prints the trials and the skip reasons. By the user's scope change (2026-10-09) there was no regression scorecard; the functional smoke and three integrated confirmations are in [VC_BENCHMARKS.md → Integrated smoke](benchmarks/VC_BENCHMARKS.md#integrated-smoke-p3b-i0).

The single final scorecard of these shipped defaults (T1-T10, 67 target rows) is in [VC_BENCHMARKS.md → P7-B](benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults); the [P7 exit matrix](#p7-exit-matrix-measure-set-defaults-document-release) summarizes it.


| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green | Above | Met |
| Every default flip has recorded interleaved evidence; no fidelity-lowering lever is a default | [ADR 0010](adr/0010-sve-optimization-p3b.md) lists what is on (with the ABBA numbers) and what is a switch (with the reason); `engine/presets.rs` records both next to the code | Met |
| Every new engine path keeps a kill switch | The switch table in [DEVELOPER_GUIDE → Voice-engine optimization](DEVELOPER_GUIDE.md#voice-engine-optimization-p3b); `SIGIL_VC_PARALLEL_LOAD=0` and `SIGIL_CUDA_TUNING=wait=spin` were exercised on the integrated build (the shared DirectML queue is opt-in, `SIGIL_VC_DML_SHARED=1`, since the review) | Met |
| The integrated engine runs on CUDA, DirectML and the CPU without a crash and with sane output | Smoke: 6 runs exit 0 with 0 model errors; EPs as requested, no fallback or warning; output finite, length equal to the input, rms 0.107-0.126 | Met |
| The shipped defaults did not make the engine slower | Integrated smoke: offline call p95 CUDA 7.2, DirectML 7.8 ms (P5 baseline 10.9 / 11.1); hybrid wait against spin NEUTRAL on call p95 / p99 and -85 % / -91 % CPU (the shared DirectML queue measured -11 % call p50 but is opt-in since the review: its content-stage p95 REGRESSED, rule R3) | Met (the final scorecard is P7-B's) |
| Targets as revised | T1 132.6 ms (115.2 opt-in), T2 CUDA met under load / missed idle, T3 missed except the extra voice, T4 met on cores and the GPU rows / missed on Streamer RTF, T5 v2 missed except clicks, T6 undecided, T9 met | **Missed** (reported as misses; [VC_BENCHMARKS → Scorecard v2](benchmarks/VC_BENCHMARKS.md#scorecard-v2-targets-as-revised-v1-rows-above-stay-for-continuity)) |
| The benchmark can decide | Interleaved gate resolution measured (two A/A sessions: 62-64 of 64 rows NEUTRAL, no false REGRESSED), GPU/host state flags, attribution, scorer v2 with the exact CAM++, compare v2 | Met (paced rows resolve to 8-13 %) |

**Manual checks still open** (need the real app, another PC, a headset, a download or a listening session):

1. **H1 and H3 live records:** moved to [Optional checks](#optional-checks-verify-when-convenient) (P3B-1).
2. **Live device for the new pacing controls:** moved to [Optional checks](#optional-checks-verify-when-convenient) (P3B-2).
3. **Competitors and T6:** install RVC WebUI / Applio / w-okada / vc-rs / rvc-realtime on a PC you are happy to put them on, capture their renditions (`scripts/vc-bench/README.md`), fetch the public corpus (`scripts/bench/fetch-public-corpus.py`, about 646 MB), run a `compare` with a corpus manifest, then the blind A/B (`tools/listening-test`, 5+ listeners).
4. **Listening checks:** the 20-trial A/B page for the ablation dry run (time anchoring did not move any automatic metric), and DirectML `ja-long`'s seams (one narrow-window click).
5. **The e2e loopback** through a virtual cable: moved to [Optional checks](#optional-checks-verify-when-convenient) (P3B-5).
6. **The T8 community corpus** (40 voices) and a trained DDSP-SVC voice (P5 items, unchanged).
7. **A packaged build:** `pnpm tauri:release`, `pwsh scripts/release/verify-bundle.ps1`, start the installed app and run a voice on CUDA and DirectML (the engine's CUDA graphs and the hybrid wait need the provider next to the exe).

## P7 exit matrix (measure, set defaults, document, release)

P7 measured the speech stack and the voice engine outside VR (the benchmark's game-load simulator stands for a running game; nothing was tested inside VR or SteamVR), set the shipped defaults from the numbers, finished the documentation and prepared the release ([plan → P7 outcomes](SPEECH_STACK_IMPLEMENTATION_PLAN.md#p7-outcomes-binding-amendments-and-what-is-left)). By the user's scope (2026-10-09) every candidate got a cheap screening pass and only the top engines per category a full test; the heavy GPU models (Voxtral, Cohere Transcribe, Qwen3-ASR 1.7B, R2T2, Hy-MT2 7B, MiLMMT 4B, TranslateGemma, Qwen3-TTS 1.7B, Magpie, Breeze, OmniVoice, Fish S2 Pro) stay opt-in and were neither benchmarked nor tuned. Publishing stays with the user: nothing was uploaded.

| Exit criterion | Evidence | Status |
|---|---|---|
| Gates green | `pnpm typecheck`, `pnpm i18n:check`, `pnpm test:unit`, `pnpm build`, `cargo clippy --all-targets`, `cargo test`, Playwright, `node docs/check-docs.mjs --no-todo` and `node scripts/release/check-catalog-pins.mjs`, all on the final tree. Final gate run (2026-10-10, after the review fixes; Rust on a BelowNormal cargo, `CARGO_BUILD_JOBS=6`): `pnpm typecheck` clean; `pnpm i18n:check` OK; `pnpm test:unit` 138 files, 2105 tests passed; `pnpm build` OK (it refreshed `public/docs` and `public/manual`); `node docs/check-docs.mjs` 35 files, 0 problems, 0 placeholders; `node scripts/release/check-catalog-pins.mjs` OK; `cargo clippy --all-targets` 0 warnings; `cargo test` passed on the rerun: 2646 passed, 0 failed, 151 ignored (they need real models, devices or a GPU), with `--test-threads=6`. The gate's own two full runs had each failed one test, `services::bench::compare::tests::a_corpus_run_scores_points_and_refuses_unfair_ones`. That test is a wall-clock real-time check on an in-process stub row: it read a per-block p99 of 70 ms against a 10.7 ms block while an orphaned filesystem search held a full core, on a PC that was in use, in a debug build at below-normal priority. It passes alone. `compare.rs` is unchanged since P3B, and the test was not weakened. Playwright was not run: no route and not `src/index.tsx` changed in P7. **Gate again after the output queue floor and the Streamer block (`348d1dc`, 2026-10-11):** `pnpm typecheck` clean; `pnpm i18n:check` OK; `pnpm test:unit` 2107 tests; `node docs/check-docs.mjs` clean; `node scripts/release/check-catalog-pins.mjs` OK; `cargo clippy --all-targets` 0 warnings; `cargo test` 2669 passed. The first full `cargo test` run had one failure, a DirectML device error (the commit message does not name the test): it passed alone and in the second full run, which is the 2669. Treat it as a first-run flake of the DirectML device, not a regression. `pytest scripts/vc-bench/tests` 53 passed | Met (one Rust test is sensitive to CPU contention and one DirectML-backed test failed once on a first run; both passed in the rerun) |
| The benchmark pages hold dev-rig numbers, and catalog and preset defaults equal the measured winners | [SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md) (each result cites the corpus `content_sha256` `459c2ff05c92…`, the normalizer stamp `sigil-norm/1` and the rig), [VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md), `stt-routing.json` and its two test suites. Defaults that changed: the CPU route (**ru** Parakeet Ultra instead of Qwen3-ASR 0.6B, **ja** SenseVoice instead of Qwen3-ASR 0.6B; en, the Parakeet languages, ko and zh confirmed), interim translations off while the local language model runs on the processor (`"auto"` switch; stored booleans win), the CPU Streamer's thread count (4-6 threads instead of 1), and, after P7 (MT-Europe, 2026-10-11), a built-in translation route for the processor tier: English with French, Portuguese, Dutch and Spanish goes to NLLB-200 beam 4 first (+2.8 to +6.0 chrF2, 95 % interval above zero; [SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md#european-pairs-on-a-processor-only-pc-mt-europe-2026-10-11)). No persisted enum value was renamed or removed; a routing change is a table, so no profile was rewritten. One pipeline check through the release exe of the final tree (ko / ja / ru, two clips each): Korean runs Nemotron partials with a Qwen3-ASR 0.6B final, Japanese a SenseVoice final, Russian a Parakeet Ultra final. CPU rows ran on a 6-core slice and, for the pipeline, a 4-core one; there is no friend export yet. Not measured and still opt-in: every heavy GPU engine, Bergamot (pack unbuilt), Edge TTS, quantized Whisper | Met (heavy GPU tier unbenchmarked by scope) |
| Targets T1-T10 are stated as met or missed, with remediation for the misses | [VC_BENCHMARKS.md → P7-B](benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults): 67 target rows of the `final` profile, 37 met, 30 missed, none of those 67 unmeasured, no disturbed step. Five further rows of the scorecard in `VC_BENCHMARKS.md` are **pending or not measured** and are outside that count: T1 end to end (needs a virtual cable pair), T6 (competitors), the T8 95 % row (the 40-voice corpus), T9 hot-switch timing and T9 TensorRT-RTX; the revised targets are in the [P3B matrix](#p3b-exit-matrix-voice-engine-optimization). Misses and their reasons: T1 (engine-added p99 115-143 ms against 85), T2a/T2b at idle GPU clocks and on DirectML under load, T3 VRAM with RMVPE, T4 Streamer RTF (the shipped one-thread preset measured 2.7 and was not real time; the P7 thread policy converts at RTF p99 0.79-0.86 offline on the idle dev rig, still above the 0.6 target). The review of the scorecard also found that the benchmark played only 48-63 % of the converted audio at the default 35 / 70 ms output queue; the cause was the output queue's recovery trim, not the 70 ms cap, and the output queue floor for template voices on long blocks (110 / 136 ms at 120 ms blocks, `SIGIL_VC_BLOCK_BUFFER=0` to switch off) fixed it: the three CPU rows re-run play 99.1-99.8 % of the converted audio (worst single pass 98.8 %) at 265-278 ms median. T1 for the Streamer under the H3 game stand-in (p99 at most 280 ms, provisional) is **missed**: 305 ms (the 4-6 thread policy, 6 threads here), 288 ms (pool) and 295 ms (6 threads) with all of the audio counted, and the 6-thread policy still goes over budget in about 2.5 % of the blocks (RTF p99 1.04); with the button's own FCPE on the final build the 8-core rig reads 348 ms p99 under the stand-in (passes 308 / 348 / 419; 99.3 % played, nothing dropped), but **6, 5 and 4 physical cores drop 9.5 / 13.9 / 19.7 % of the input under it** (4 cores do not keep up even idle, RTF p99 1.16), which the inspector now says ([details](benchmarks/VC_BENCHMARKS.md#the-cpu-streamer-on-4-5-and-6-physical-cores)); the Streamer on the processor takes the 120 ms block by itself when Block size is on Auto (the 20 ms the default chunk derives cannot keep up there); see [VC_BENCHMARKS.md → Re-run with the floor](benchmarks/VC_BENCHMARKS.md#re-run-with-the-floor), T5 v2 identity and pitch rows. T6 (competitors) stays pending: no competitor is installed, and the exact steps are in [`scripts/vc-bench/README.md`](../scripts/vc-bench/README.md). H1/H3 live records, the end-to-end loopback, hot-switch timing and TensorRT-RTX were not measured (reasons in the scorecard) | Stated; misses are the P3B ones |
| Every release catalog entry is hash-pinned | `build.rs` refuses a published entry without `sha256` or `git_oid` in release builds, and `node scripts/release/check-catalog-pins.mjs` (offline, in CI) adds immutable sources and the release set: 303 entries (266 published), 18 default entries and 186 light options. Online re-check of the defaults: 27 files verified against the hosts' metadata, the 19 install records of the P7 store match the catalog. Sigil-built packs stay `unpublished` and install by Import; the Vosk archives (md5-only host) and the Firefox Translations models (attested by Mozilla's registry) are the documented exceptions | Met |
| The packaged smoke passes with no GPU and no packs | `pnpm tauri:release` (16 min) and `pwsh scripts/release/verify-bundle.ps1 -Strict` pass (1 warning: the CUDA runtime pack is not installed, as designed; with the pack from the dev store the probe picks CUDA and passes with 0 warnings). `pwsh scripts/release/packaged-smoke-all.ps1 -Mode Extract -PackStore <P7 store> -Online`: 17 of 17 checks pass on the 0.4.0-1 build (layout, sysinfo, catalog 303 entries, ep-probe, real Nemotron French and SenseVoice English decodes, `base/test-small`, the sherpa-onnx pack, server, page, models, idle, focus: 0 foreground hits in 139 samples and the window ended minimized, log, exit in 0 s). **Not run on the dev PC:** the no-GPU leg (this PC has an RTX 4090; the hosted runner of `release.yml` covers it) and Windows Sandbox (its viewer window cannot be hidden; run `-Mode Sandbox -Online` when away from the PC). The smoke exposed one real bug, fixed with an exit guard (see the changelog) | Met on the dev PC; no-GPU leg left to CI / Sandbox |
| The docs are current | The operator reference, the developer guide, the manual and the changelog; `node docs/check-docs.mjs --no-todo` finds no dead link, anchor, malformed table or `TODO(P7-numbers)` placeholder, and `node scripts/sync-manual.mjs` refreshed `public/docs` and `public/manual` | Met |

**Not done, by design:** the in-VR and SteamVR checks and the two-device LAN matrix ([Optional checks](#optional-checks-verify-when-convenient)), the competitor head-to-head (T6), listening studies, benchmarks of the opt-in heavy engines, the 10 ms device block as a default (it needs a live-device record), the GPU speech tier under a mid-range card's game load, and publishing packs and assets. `whisper-rs` stays (see [Why whisper-rs stays](../scripts/release/README.md#why-whisper-rs-stays)).

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
| Rewrite truncated at 100 tokens; `[AI Rate Limited]` published | `maxTokens` (default 1024); original text plus notice on failure | `llmRequest.test.ts`, `ordering.test.ts` |
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
| **Embedded HTTP server** (`web` plugin) | `0.0.0.0` by default (Settings → LAN access → Allowed with the LAN token), or `127.0.0.1` (This PC only); configured port | Static UI assets, PeerJS signaling HTTP, pubsub WebSocket; `/pubsub` and `/peer` behind an Origin allow-list and, for other machines, the persistent LAN token |
| **OSC receiver** (`osc` plugin, P6-C; only while VRChat mute sync is on) | UDP on `127.0.0.1` (OSCQuery: a free port, plus an HTTP description and mDNS records on 127.0.0.1), or `0.0.0.0:<port>` for VRChat on another PC (packets accepted only from that PC and loopback) | VRChat's MuteSelf; only MuteSelf messages are decoded |
| **Helper processes** (`process_host`: llama-server, Bergamot, TTS helpers) | `127.0.0.1` only (enforced by the exposure audit), random port and key | Local engines; keyless TTS servers sit behind Sigil's key-protected adapter |
| **PeerJS** | Host advertises LAN IP + port to clients | `/client` loads in a browser; connects to host for Yjs document sync and related data |
| **Linked pubsub** | Outbound WebSocket from host to a user-supplied URL (optional) | Bridges pubsub to another machine or service when “link” mode is enabled |
| **OAuth loopback** | Localhost redirect handlers | Twitch / Kick (and similar) OAuth callbacks |

### Trust assumptions

1. **Local server is LAN-reachable** — The `web` plugin binds `0.0.0.0`, so anything that can open TCP connections to `<host>:<sigil_port>` — other users on a **shared** machine (terminal server, borrowed laptop) **and any host on the same network** — can reach the same HTTP/WebSocket endpoints the webview uses, without your password. Since LAN security stage 1, the WebSocket handshakes on `/pubsub` and `/peer` pass an **Origin allow-list** (below), which blocks browser cross-site WebSocket hijacking and DNS rebinding. Since **stage 2** (P6-D, `web/auth.rs`) clients on other machines must also present the persistent **LAN token**: `?token=` on `/pubsub` and page URLs, the PeerJS `key` on `/peer`. It is 26 Crockford base32 characters (130 bits) from the OS random generator, persisted in `lanAccess.token` in plaintext like API keys, read at startup, compared in constant time, and never logged or emitted. Same-PC peers (loopback or this PC's own address) are exempt unless the request carries reverse-proxy forwarding headers; an origin-less non-browser tool on another machine needs the token. Refusals after the upgrade close with 4401 (missing or invalid token) or 4403 (This PC only); `/peer` first sends PeerJS `INVALID-KEY`; Origin refusals stay 403. Admitted sockets are closed when the token is regenerated, the mode switches to This PC only, legacy is turned off, or the Origin check is turned back on. **Allow devices without the token (legacy)** is off by default and shown with a caution. Assets stay public, with `Referrer-Policy: same-origin`. Residual risks: the token crosses the LAN in cleartext (no TLS), LAN links store it on the other device (make a new token if one leaks), and a local proxy that strips forwarding headers makes remote clients look local, so never tunnel the port. Mitigation: run Sigil only on trusted accounts and networks; firewall the port to only where you intentionally use `/client` / LAN features; do not expose it via SSH port forwarding, proxies, or tunnels unless you understand the impact.

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

5. **`text.stt` on the local pubsub WebSocket** — Peers attached to the host’s pubsub socket (e.g. **mic.html** for Chrome/Edge external STT) send JSON that the host validates against an **inbound topic allow-list** ([`src/shared/services/pubsub/inbound.ts`](../src/shared/services/pubsub/inbound.ts)): `obs.caption` (schema-checked), `stt.external.state` (local pages only) and the `text.*` caption topics; everything else (bare `text`, `stt.external.command`, `scenes:change`, …) is dropped. The Rust server no longer relays pubsub between peers, messages are capped at 256 KiB, and **binary frames close the socket with 1003**, so the old binary-audio-to-Whisper injection path is gone. Messages carry a `via` hop chain so Link apps cannot loop. A tool on another machine also needs the LAN token (stage 2); do not expose the port beyond trusted networks. The P6 topics `others.*` and `readings.*` are never accepted from peers (`/client` only validates them for display).

6. **Token auth for other machines (stage 2)** — The design assumes single-user desktop use on a trusted network. Stage 1 (Origin allow-list, binary refusal, topic allow-list, XSS fixes in the `/client` error fallback and `inputfield.html`, bind-failure reporting) closes the browser-reachable attacks; stage 2 (P6-D) adds a persistent token for clients on other machines and a bind mode (LAN or this PC only). It is not a substitute for TLS or internet-grade authentication: intentional remote access, tunneling, or public exposure would still require real authentication, rate limits, and TLS review.

7. **NVIDIA runtime pack** — Downloads only the pinned NVIDIA redistributable URLs with pinned sha256, one file at a time with HTTP Range resume, and extracts only the DLLs into a staging folder that is renamed into place (the model-store pack `runtime/nvidia-cuda13` through the store's own safe extractor; the interim Settings installer with Windows' `tar.exe`, refusing `../` entries). The DLLs load by full path right before the first CUDA session.

8. **Diagnostics IPC** — `log_from_frontend` is webview-only, size-capped and rate-limited; the deep EP probe runs as a separate child process (`Sigil.exe --headless ep-probe`).

9. **Model store** — Catalog entries are pinned to immutable revisions with sha256 or git blob ids (release builds refuse unpinned published entries); downloads resume with HTTP Range, fail over on errors, HTML bodies, wrong sizes and hash mismatches, and are verified before an atomic rename. Archives are extracted with traversal, absolute-path, drive-letter, alternate-data-stream, device-name, case-duplicate, link, entry-count, size and compression-ratio checks. Trust-on-first-use exists only in debug builds and never for executables; `unpublished` entries never download from catalog URLs. A custom mirror must be https, or http on the LAN / loopback, and its files are still checked against the pins. `SIGIL_MODEL_CATALOG_OVERRIDE` / `SIGIL_LOCAL_ASSETS` are developer overrides.

10. **Native runtime packs** — Nothing new is load-time linked. Engine DLLs load lazily from `app_data/runtimes/<name>/<version>/<variant>/` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | DEFAULT_DIRS`, after every file matched the sizes and sha256 in the pack's `sigil-pack.json`, the ABI version and the engine info. sherpa-onnx runs a CPU-only ONNX Runtime 1.28.2 inside its pack next to Sigil's static 1.28.0; C++ exceptions do not cross its C API, so a corrupt model would abort Sigil: catalog models are hash-verified and user-supplied folders are test-loaded in a child process (`validate_in_child`) first.

11. **Helper processes** (`process_host`: llama-server, Bergamot and the TTS helpers) — Started suspended and joined to a kill-on-close Job Object before they run (they die with Sigil, even when Sigil is killed), `CREATE_NO_WINDOW`, optional committed-memory cap. They must listen on `127.0.0.1` only: after readiness Sigil reads the OS listener table for every process in the job and stops a helper with any non-loopback TCP listener. Each gets a random 256-bit key (from `ProcessPrng`), passed by environment variable and redacted from logs; llama-server (P4) and the TTS helpers (P6) hand it to the webview for direct loopback calls, and status payloads never carry it. The health client uses no proxy and follows no redirects. **TTS helpers (P6):** Sigil's `sigil-tts-server` (Qwen3-TTS) requires the Bearer key on every route but `/health` (constant-time), refuses a non-loopback host without `--allow-remote`, sends no wildcard CORS, uses `SO_EXCLUSIVEADDRUSE`, and its exe's sha256 is checked against `sigil-pack.json` before every start (as for the Sigil-built s2 packs). NeMo-Speech.cpp and CrispASR take their keys from the environment. Breeze-TTS-2.cpp and s2.cpp have no key, so Sigil speaks to them only through an in-process adapter (127.0.0.1, random port and its own key, any `Origin` header or a non-loopback `Host` refused); their own loopback ports stay reachable to other programs on this PC while they run (WebSocket and CORS off, random port), until Sigil-patched servers with a key exist.

12. **Untrusted voice files** — RVC `.pth` / `.zip` imports are parsed by Sigil's allow-list pickle VM (only `OrderedDict`, torch tensor rebuilders, storages and `_codecs.encode`; legacy pickles and TorchScript refused) behind a zip guard, in the `Sigil.exe --headless vc-import-worker` child (Job Object limits via `process_host::run_worker` in P3-B); the worker enforces a wall-clock limit today. Style-Bert-VITS2 / AIVMX imports (P6-G) read only the ONNX file's top-level `metadata_props` (the AIVM manifest, hyper-parameters, base64 `.npy` style vectors) into a staged copy; nothing is executed or unpickled, and `.aivm` / `.safetensors` / `.pth` files are refused.

13. **Caption window** (P6-B) — The `captions` window has its own capability (`capabilities/captions.json`: event listen / unlisten and `start-dragging` only): no plugin commands, no file-system access, no window creation. It boots no host (early route before `ApiServer`); caption frames are rebuilt field by field and rendered only as text. `/captions` served over the LAN renders nothing outside Tauri.

14. **Cloud speech** (P6-H, P6-K) — **Edge** sends the spoken text to Microsoft's unofficial Read Aloud endpoint (no key). **Gemini TTS** sends text to Google with the user's key, stored in settings like the other API keys and sent only in the `x-goog-api-key` header (never in URLs or logs); a custom Gemini API URL receives the key. **Gemini Live** sends microphone audio to Google: the key goes only in the `x-goog-api-key` header of `POST /v1beta/auth_tokens`, the WebSocket URL carries a single-use ephemeral token (it must start a session within a minute) and is never logged, and close reasons and HTTP errors are key-redacted.

15. **SteamVR** (P6-B) — `openvr_api.dll` is the hash-pinned `runtime/openvr` pack, loaded lazily (never load-time linked) and initialized as a Background app only while `vrserver.exe` runs, so Sigil never starts SteamVR.

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

Re-run before releases: `pnpm audit` (repo root, one registry request) and `cargo audit` with the commands below. Both tools list every advisory against the lockfiles whether or not the code is in the shipped app, so each finding below carries a reachability verdict; fix what is reachable, list the rest.

### Last refresh

**Date:** 2026-10-11 (tree at `755adb7`; advisory database commit `7eebec69c352`, 1,296 advisories, 791 locked Rust crates, 552 npm packages). Network used: one `pnpm audit` request and one advisory-database fetch. **No dependency was changed by this refresh**: no finding is both rated high or critical and reachable in the shipped app.

### pnpm (`pnpm audit`)

The audit reports **43 findings (2 critical, 23 high, 16 moderate, 2 low)** in 41 advisories. The shipped app is the Vite bundle in the Tauri window: only what `src/` imports is in it. The checks: `grep` for the import in `src/`, `scripts/` and `tests/`, and the package's `browser` entry for packages with a Node build.

| Severity | Package (locked version) and advisories | Path | In the shipped app? | Verdict |
|----------|------------------------------------------|------|---------------------|---------|
| Critical (2) | `tinypool` 1.1.1: prototype-pollution gadget to code execution in worker options | `vitest` | No: the test runner, running our own tests | Left. Fixed by vitest 4.x (or `tinypool` 2.1.2+), a major bump of the test toolchain: after the release |
| High | `lodash` 4.17.23: code injection through `_.template` `imports` key names (+ moderate prototype pollution in `_.unset` / `_.omit`) | direct | Yes, the package is bundled, but the vulnerable calls are not made: the app no longer calls `template` at all (event text has its own reader, `src/client/elements/eventText.ts`; see the closed item below the table) and uses `debounce`, `throttle`, `sample`, `chunk`, `groupBy`, `difference`; never `unset`, `omit`, `set` or `merge` | Not reachable, so left. Fixed in 4.18.0, which `^4.17.23` already allows: `pnpm update lodash@^4.18.0 --network-concurrency=1` (two requests) is the whole fix |
| High (6) + moderate | `nanoid` 4.0.2 (direct) and 3.3.11 (via `postcss`): loops with a zero or negative size, integer overflow, non-integer sizes | direct, `postcss` | Bundled, but every call has a constant size: `nanoid()`, `nanoid(12)`, `(16)`, `(64)`, `customAlphabet(urlAlphabet, 42)`; no size comes from input | Not reachable, left. The fix is nanoid 5.1.x (major) |
| High | `js-cookie` 2.2.1: prototype hijack in `assign()` | `react-use` | No: `react-use` is imported for `useEffectOnce`, `useDebounce` and `useCopyToClipboard` only; `js-cookie` is used by `useCookie` | Not bundled |
| High + moderate | `ws` 8.20.0: memory exhaustion from tiny fragments; uninitialized memory | `@twurple/chat` > `ircv3` > `@d-fischer/connection` | No: the webview build takes `@d-fischer/isomorphic-ws`'s browser entry, which is the native `WebSocket`; Node's `ws` is not bundled | Not bundled |
| High (8) + moderate | `fast-uri` 3.1.0 (host confusion, path traversal, SSRF, authority injection) | `ajv` | No: `ajv` is a direct dependency, but nothing in `src/`, `scripts/` or `tests/` imports it | Not bundled. `ajv` (and `@speechly/browser-client`, below) look unused: removing them is the clean fix, after checking that no tool needs `ajv` as a peer |
| High + moderate (2) | `vite` 5.4.21 (`server.fs.deny` bypass on Windows alternate paths, `.map` handling, `launch-editor` NTLM hash), plus moderate `esbuild` 0.21.5 (dev-server requests) | dev | Not in the bundle (no Vite code). The dev server was the exposure: `pnpm dev` listens on localhost only, but `tauri dev` started it with `--host` (all interfaces), so the `fs.deny` bypass could be reached from the LAN while a developer ran it. `beforeDevCommand` in `src-tauri/tauri.conf.json` is now `pnpm dev --host 127.0.0.1` (add `--host` by hand to open the dev UI from a phone) | Left. Fixed by Vite 6.4.3+ (major, with a new `esbuild`): the planned toolchain upgrade |
| High (2) + moderate (2) | `postcss` 8.5.8 (file read through `sourceMappingURL`, `</style>` XSS in stringify) | build | No: build time, over CSS this repository authors | Left. `postcss@^8.5.23` is inside the range, a build-only bump for the toolchain pass |
| High (2) + moderate | `browserslist` 4.28.1, `baseline-browser-mapping` 2.10.12 | `@vitejs/plugin-react` > `@babel/core` | No: build time | Left (with `@babel/core` 7.29.0, low) |
| High | `source-map-js` 1.2.1 | `@tailwindcss/vite` | No: build time | Left |
| Moderate + low | `i18next-http-backend` 2.7.3: path traversal and URL injection through `lng` / `ns`; incomplete URL validation (SSRF) | direct | Yes, but `lng` is restricted by `supportedLngs` to the locale codes of `src/i18n.ts` and `ns` is the constant `translation`, over the app's own `/i18n/{{lng}}/{{ns}}.json` | Not exploitable as configured; the fix (4.0.2+) is a major bump |
| Moderate | `uuid` 8.3.2, 9.0.0 and 9.0.1 (missing buffer bounds check in v3, v5 and v6 when the caller passes a `buf`) | `@speechly/browser-client` (8.3.2), `vosk-browser` (9.0.0), `microsoft-cognitiveservices-speech-sdk` (9.0.1) | The Speechly copy: no, nothing imports the client. The other two are bundled (the Azure Speech SDK chunk and the lazily loaded `vosk.js`), but each calls only `v4()` without a buffer (`Guid.js` in the SDK, `this.id = v4()` in `vosk.js`) | Not reachable, left. The fix is a newer uuid major, which those packages would have to adopt |
| Moderate | `vitest` 3.2.7, `@vitest/mocker` (path traversal through the mock redirect) | test | No | Left with `tinypool` |

**Recommendation:** one toolchain pass after the first release (Vite 6.4.3+ with its `esbuild`, vitest 4, `postcss`, `nanoid` 5, `i18next-http-backend` 4, `lodash` 4.18+, and removal of the unused `ajv` and `@speechly/browser-client`), with the UI and Tauri tests; do not run `pnpm audit fix --force` blindly.

**Closed (not an advisory):** `src/client/elements/utils.ts` used to compile a user-written event text with `_.template`, whose `<% %>` and `${}` delimiters run JavaScript by design, so a scene document that carried such a text ran code in the window that opened it, and that window has Tauri IPC. Event text is now read by `src/client/elements/eventText.ts`, which evaluates nothing: it follows `event` paths (`.name`, `[0]`, `["key"]`) over own data properties only (no prototype walk, no getters), refuses `__proto__`, `constructor` and `prototype`, leaves everything else (including `<% code %>` blocks) as typed text, and reads in linear time. `src/client/elements/__tests__/eventText.test.ts` holds the attack strings, a differential check against lodash for the forms that are kept, the timing cases (200,000 unterminated openers) and a scan that fails when `lodash/template`, `new Function` or `eval(` appears under `src/`. Nothing else under `src/` compiles text to code (searched for `Function(`, `eval(`, `template(`, string timers, dynamic `import()` and script injection; the `innerHTML` writes are `<style>` elements and fixed strings). In this build no element carries event texts and the function had no caller, so no saved document changes. Syntax for users: [manual](manual/02-studio.md#event-text).

### Rust (`cargo audit`)

Run from a scratch folder that has a `.cargo/audit.toml` with `[yanked] enabled = false` and `update_index = false` (so the tool does not query the crates.io index for every crate): `cargo audit --file <repo>/src-tauri/Cargo.lock`. Install: `cargo install cargo-audit` (already installed on the dev PC). To see whether a crate is in the shipped Windows graph, offline: `cargo tree --offline --locked --target x86_64-pc-windows-msvc -e normal,build -i <crate>@<version>` from `src-tauri` ("nothing to print" means it is not compiled for Windows; add `--target all` to see who pulls it elsewhere).

**Last run (2026-10-11):** **9 vulnerability entries across 5 crates (8 advisories; `h2` appears once per locked version); one is rated high (`quick-xml`, CVSS 7.5, not compiled for Windows) and none is reachable as shipped; 17 allowed warnings** (13 unmaintained, 4 unsound; it was 22 in April). Nothing was updated.

| Advisory | Crate (locked) | Rating | In the shipped Windows build? | Verdict |
|----------|----------------|--------|-------------------------------|---------|
| RUSTSEC-2026-0258 (unbounded empty DATA frames) | `h2` 0.3.27 via `hyper` 0.14 via `warp` 0.3.7; `h2` 0.4.13 via `hyper` 1.8 / `reqwest` | Upstream says "Low severity" (denial of service) | Yes. The 0.3 copy serves the LAN web server, and `hyper` 0.14 accepts HTTP/2 on prior knowledge; 0.4.13 is the outbound client | The fix (`h2` 0.4.16) is not available for the 0.3 line: it ends with `warp` 0.3 / `hyper` 0.14. A cheap mitigation would be an HTTP/1-only server, a change to the web plugin's accept path that this refresh did not make. `cargo update -p h2@0.4.13` fixes the 0.4 copy (a bare `-p h2` is ambiguous with two versions locked; the offline dry run resolves 0.4.19) (an outbound client talking to a hostile server, low) |
| RUSTSEC-2026-0194, -0195 (quadratic time, namespace allocation) | `quick-xml` 0.38.4 via `plist` via `tauri` / `tauri-codegen` | CVSS 7.5 (availability, network) | No: absent from the Windows graph; `plist` is the macOS `Info.plist` reader | Not compiled for Windows |
| RUSTSEC-2026-0274 (double free when an element's `Drop` panics in `ReadChunk::commit`) | `rtrb` 0.3.3, direct | Unrated (memory corruption) | Yes, but the only `read_chunk` / `commit` callers (`audio/consumer.rs`, `voice_changer/budget.rs`, `playback.rs`, `worker.rs`, `tts_sink/output.rs`) read rings of `f32` and of `Tagged` (an `f32` and a `u8`): `Copy` data with no `Drop` to panic. The rings of non-`Copy` values (`FanOutCmd` and `Slot` in `audio/hub.rs` and `watchdog.rs`) use plain `push` / `pop`, so a `read_chunk` added on them would need this reviewed again | Not reachable. `cargo update -p rtrb` (0.3.5) is the one-crate fix when a request is available |
| RUSTSEC-2026-0285 (TLS 1.3 handshake across encryption levels) | `rustls` 0.23.37 | CVSS 5.3 (confidentiality low) | No: the app's HTTP stack is `native-tls` (`hyper-tls`); `rustls` is in `Cargo.lock` through `hyper-rustls` / `tokio-rustls` but in no feature set (`cargo tree --all-features --target all` prints nothing) | Not compiled |
| RUSTSEC-2026-0098, -0099, -0104 (name constraints, revocation-list panic) | `rustls-webpki` 0.103.10 | Unrated | No (same as `rustls`) | Not compiled |

Warnings (informational, none new in code Sigil calls): `atty` / `ansi_term` / `instant` / `proc-macro-error` (through `uwuify` 0.2.2 and `clap` 2), `bincode` 2.0.1 (through `jpreprocess`), `paste` (through `tokenizers`), `ttf-parser` (through `cosmic-text`), `fxhash` and the `unic-*` crates (through `tauri-utils` / `urlpattern`), `glib` 0.18.5 (Linux GTK only, not in the Windows graph) and `rand` 0.7.3 / 0.8.5 (unsound only when a custom logger calls `rand::rng()`; both come from `phf` code generation, and Sigil's `tracing` sink does not). Address them by upgrading the Tauri ecosystem and replacing `uwuify` over time, not per-crate overrides unless a concrete CVE applies.

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
| OpenVR `openvr_api.dll` v2.15.6 (P6-B) | Runtime pack `runtime/openvr` downloaded on request | BSD-3-Clause | Pinned by git blob id; hand-written FFI, no `openvr_sys` |
| cosmic-text 0.19, tiny-skia 0.12, wasapi 0.19, mdns-sd 0.21 | Linked (crates) | MIT / Apache-2.0 (cosmic-text, mdns-sd), BSD-3-Clause (tiny-skia), MIT (wasapi) | SteamVR caption rendering, per-process loopback, OSCQuery |
| jpreprocess 0.15 (OpenJTalk and lindera code), wana_kana 5, jieba-rs 0.11, pinyin 0.11 | Linked (crates) | BSD-3-Clause (jpreprocess), MIT (wana_kana, jieba-rs, pinyin) | Readings, the Style-Bert-VITS2 frontend and Kokoro's Japanese reader |
| NAIST-JDIC for jpreprocess 0.15.0 | `text/jpreprocess-naist-jdic`, downloaded on request | BSD-3-Clause | Shared by readings, Style-Bert-VITS2 and Kokoro (catalog `runtime-sbv2.json`) |
| FireRedVAD AED (FireRedTeam/FireRedVAD@c30ec49e) | `vad/firered-aed`, downloaded on request | Apache-2.0 | Music gate |
| qwentts.cpp `6fae929` (ServeurpersoCom) and its ggml fork `40e16e4`; cpp-httplib, yyjson | Statically linked into the Sigil-built `sigil-tts-server.exe` packs (`runtime/qwentts-server-*`, local import) | MIT | `scripts/runtime-packs/qwentts/` (`UPSTREAM.json`, `NOTICE.txt`) |
| Qwen3-TTS 12 Hz models (GGUF, Serveurperso/Qwen3-TTS-GGUF@b7ee2e8) | `tts/qwen3-tts-*`, downloaded on request | Apache-2.0 | Catalog `tts-qwen3.json` |
| NeMo-Speech.cpp v0.1.0, Breeze-TTS-2.cpp v0.1.0, CrispASR v0.8.40 (official Windows zips), s2.cpp `2c33261` (Sigil-built) | Runtime packs in `tts-ggml-experimental.json` | Apache-2.0 (NeMo-Speech.cpp, Breeze-TTS-2.cpp), MIT (CrispASR), per catalog (s2) | Catalog facts per entry; models: NVIDIA Open Model License (Magpie, NanoCodec), BreezeBlue Research and Non-Commercial License v1.1 (Breeze TTS 2), Apache-2.0 (OmniVoice), Fish Audio Research License (S2 Pro) |
| Style-Bert-VITS2 Japanese BERT (tsukumijima/deberta-v2-large-japanese-char-wwm-onnx, fp16) | `tts/sbv2-bert-ja`, downloaded on request | CC-BY-SA-4.0 | `sbv2_core`'s pipeline ported to Sigil's ORT |
| Chatterbox Multilingual V3 ONNX export (KitsuMate, commit 3f44461) | `tts/chatterbox-mtl-v3` (+ `-gpu-flow`), downloaded on request; its `token_embedding.onnx` and `tokenizer.json` are vendored as test data, plus the two flow steps' input/output declarations (`flow_step_*.io.pb`) | MIT (model), BSD-3-Clause and Apache-2.0 (language tables) | `local_tts/chatterbox_ort/testdata/README.md` |
| Kyutai Pocket TTS (sherpa-onnx packages 2026-01-26) | `tts/pocket-tts-en-int8`, `tts/pocket-tts-en`, downloaded on request | CC-BY-4.0 | Runs on the sherpa-onnx runtime pack |
| Seed-VC (Plachtaa/seed-vc@51383efd), DDSP-SVC 6.3 (yxlllc/DDSP-SVC@d6dd52fc), MeanVC2 (ASLP-lab/MeanVC2@13acf84c) model code | Vendored in `scripts/vc-families/*/upstream/` (export spike tooling, not compiled into Sigil) | GPL-3.0 (Seed-VC), MIT (DDSP-SVC), Apache-2.0 per the upstream README (MeanVC2) | Unchanged below their provenance headers; [ADR 0006](adr/0006-vc-zero-shot-singing.md) |
| FLEURS (speech), FMA (commercial-use shard), ESC-50, WMT24++, FLORES-200, BSD corpus, MetricX-24 | Used locally by the P7 campaign tooling (`scripts/speech-bench/`, under `%LOCALAPPDATA%\sigil-dev-cache\p7`); never redistributed and not bundled | CC-BY-4.0 (FLEURS), per-track Creative Commons licenses (FMA shard), CC BY-NC 3.0 (ESC-50), Apache-2.0 (WMT24++, MetricX-24), CC-BY-SA-4.0 (FLORES-200), CC BY-NC-SA 4.0 (BSD) | The corpus identity (`content_sha256`) is in its manifest |
| Whisper English text normalizer | Vendored in `scripts/speech-bench/score/whisper_english.py` (scoring tooling, not compiled into Sigil) | MIT (OpenAI Whisper), Apache-2.0 via transformers 4.39.3 | Scoring only; the normalizer stamp is `sigil-norm/1` |

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
| Moonshine STT (HTTP preset; retired from the engine list) | 30 s + the clip length, at most 120 s |

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
