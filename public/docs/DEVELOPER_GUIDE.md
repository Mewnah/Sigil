# Developer guide

Architecture, **host services**, **connectivity audit**, and **client-layer rules** for contributors.

## Table of contents

- [Stack and runtime](#stack-and-runtime)
- [Layering](#layering)
- [Data flow (host)](#data-flow-host)
- [Native IPC](#native-ipc)
- [Plugin commands, capabilities and i18n staging](#plugin-commands-capabilities-and-i18n-staging)
- [UI state](#ui-state)
- [Release / quality](#release--quality)
- [Security notes (Tauri 2)](#security-notes-tauri-2)
- [Host services, persistence, and UI tabs](#host-services-persistence-and-ui-tabs)
- [Service and feature connectivity audit](#service-and-feature-connectivity-audit)
- [Speech stack internals](#speech-stack-internals)
- [Remote `/client` mode](#remote-client-mode)
- [Client folder (`src/client`)](#client-folder-srcclient)

---

## Stack and runtime

- **Shell:** Tauri 2 (Rust) — native plugins (audio, STT/TTS, OSC, local HTTP server); capabilities in `src-tauri/capabilities/` replace the v1 allowlist.
- **UI:** Vite, React 18, TypeScript.
- **State:** Valtio for backend/service and document sync; Zustand (`src/core/ui/store.ts`) for shell UI (sidebar, stats panel); Yjs + PeerJS for collaborative document sync.

### Runtime modes

| Mode | How | UI root |
|------|-----|---------|
| **Host** | Default path in desktop app | `SigilRoot` — full editor, integrations, inspectors |
| **Client** | Path `/client` + query (`host`, `port`, `id`) | `ClientView` — canvas/elements only; talks to host over PeerJS |

`AppConfiguration` in `src/config.ts` decides mode and loads network settings (Tauri `plugin:web|config` on host).

## Layering

- **`src/client/`** — Document, scenes, elements, files, particles, sound. Must not depend on host-only APIs (see [Client folder](#client-folder-srcclient)).
- **`src/core/`** — Host services (Twitch, Kick, Discord, OBS, VRC, STT/TTS, translation, transform, etc.), persisted settings (`Service_State`), Sigil chrome.
- **`src/shared/`** — PubSub (PubSub-JS + optional WebSocket + Tauri `pubsub` event), Peer bridge.
- **`src/services-registry.ts`** — `Services` enum (no imports from core/client) to avoid circular types.

## Data flow (host)

1. **Text/events:** `Service_PubSub.publish` → local PubSub-JS topics → `invoke("plugin:web|pubsub_broadcast")` → Rust → `emit("pubsub", …)` → JS listener; plus PeerJS broadcast to clients; optional outbound WS when “linked.”
   - The Rust `web` plugin **no longer relays pubsub between WebSocket peers**. Peer text goes only to the host (`pubsub` event), where `consumePubSubMessage` applies the **inbound allow-list** in [`src/shared/services/pubsub/inbound.ts`](../src/shared/services/pubsub/inbound.ts) and re-broadcasts. Allowed today: `obs.caption` (schema-checked), `stt.external.state` (local pages only) and the `text.*` topics `stt`, `textfield`, `translation`, `transform`, `transform_source`, `transform_raw`. **A new topic that must cross WebSockets needs a route in `inbound.ts`, or it is dropped.**
   - Wire messages are `{topic, data, via}`; `via` is the chain of host session ids, so a host drops its own echoes and never sends a relayed message back over the Link socket it arrived on (Link-apps loop prevention). `text.stt` from another Sigil is published as text instead of re-entering the local STT pipeline.
2. **Document:** Yjs doc in `ApiClient.document`; `Service_Peer` syncs over PeerJS (see `src/shared/services/peer/`).
3. **Local server:** Rust `web` plugin binds `0.0.0.0:<port>` (all IPv4 interfaces, reachable across the LAN) — `ping`, PeerJS path, pubsub WebSocket, static assets (`src-tauri/src/services/web/`). `/pubsub` and `/peer` handshakes pass an **Origin allow-list** (`web/origin.rs`; kill switch `lanOriginCheck` / `SIGIL_LAN_ORIGIN_CHECK`); binary frames close with 1003 and messages are capped at 256 KiB. Commands `server_status`, `set_origin_check`; events `web:server_status`, `web:origin_rejected`. Build links to `public/*.html` pages with `localPageUrl()` ([`src/utils/localPages.ts`](../src/utils/localPages.ts)).

## Native IPC

- Generic `invoke` commands: `get_native_features`, `get_port`, `app_close`.
- Feature work is mostly `plugin:<name>|<command>` from the frontend.
- Heavy entry points run through `Sigil.exe --headless <cmd>` (must be the first argument; parsed before the port pre-check; `src-tauri/src/services/cli/`): `ep-probe [--out file]`, `fetch <id>... [--mirror m] [--custom-url url] | fetch --list`, `bench <spec.json> [--out dir]`, `engine-probe <engine> [model] [wav] [--variant v] [--runtime-dir dir] [--out file] [-- extra]`, `vc-import-worker <job.json>`, `sysinfo [--out file]` and `help`. Exit codes: 0 success, 1 command failed, 2 usage error. In development use `src-tauri\target\debug\Sigil.exe` with a scratch `SIGIL_APP_DATA_DIR`.

## Plugin commands, capabilities and i18n staging

These conventions keep the shared registration files from becoming merge magnets.

- **Command lists are data.** `src-tauri/build.rs` reads `src-tauri/plugin-commands/<plugin>.txt` (one snake_case command per line, `#` comments; the file stem is the plugin name) and generates each plugin's `allow-*` / `deny-*` permissions and `<plugin>:default`. A new command needs the `#[command]` fn, its `generate_handler!` entry **and** a line in that file. `src-tauri/gen/schemas/*` are regenerated by the build; keep the regenerated files and never hand-edit them. When a command is removed, its old permission lingers in `acl-manifests.json` until `cargo clean -p sigil` (the default set is still correct).
- **Capabilities per plugin.** A new plugin gets its own `src-tauri/capabilities/<plugin>.json` (see `diagnostics.json`); `tauri.conf.json` does not enumerate capabilities, so every file loads. `migrated.json` / `desktop.json` hold the older plugins.
- **UI strings go to staging.** Workstreams put new English strings in `scripts/i18n-staging/<WS-ID>.json` (same nested shape as `public/i18n/en/translation.json`; a top-level `"$remove": ["section.key"]` deletes keys from every locale). The integrator runs `node scripts/merge-i18n-staging.mjs` (non-destructive; `--dry-run`, `--only`, `--allow-en-changes`), then `pnpm i18n:sync && pnpm i18n:check`, and removes the merged staging files. The old `merge-neural-i18n.mjs` / `merge-http-tts-i18n.mjs` overwrote locale sections and are retired (they exit 1).
- **Architecture decisions** are ADRs in [`docs/adr/`](adr/README.md).

## UI state

Sidebar tab visibility, expand/collapse, and multi-selection live in **Zustand** (`useAppUIStore`) so React and `ApiServer.changeTab` / `closeSidebar` stay in sync without a Valtio `window.ApiServer.ui` object.

## Release / quality

- **Release artifact** — release builds use `pnpm tauri:release` (`tauri build --config scripts/release/tauri.release.conf.json`). The overlay stages `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` (ORT 1.28, 62 MB) and `DirectML.dll` next to `Sigil.exe` in both the NSIS (per-user, recommended) and MSI installers. The installers do **not** contain the NVIDIA CUDA/cuDNN runtime: it is the opt-in model-store pack `runtime/nvidia-cuda13` (about 1.9 GB, [ADR 0003](adr/0003-gpu-ep-delivery.md), [ADR 0007](adr/0007-native-runtime-packs.md)). The CUDA provider stays bundled until `runtime/ort-cuda13-provider` is published; the registered app-data provider path already works. The execution provider is chosen at runtime and the one actually in use is reported. Plain `pnpm tauri build` still works but ships without the staged provider DLLs. The RVC templates (`src-tauri/resources/vc-templates/`, under 1 MB) are bundled in every build.
- **Never list build-generated files in the base `tauri.conf.json` `bundle.resources`:** tauri-build copies resources at compile time and fails when they are missing, which would break `cargo check` / clippy / test on clean checkouts and in CI. Checked-in source files (the `resources/vc-templates` directory mapping) are fine; build outputs use a release overlay with an (initially empty) directory mapping, as `tauri.release.conf.json` does. The base config uses the map form (`{ "<source dir>": "<target dir>" }`) so the overlay's map merges into it instead of replacing it.
- **CI (GitHub Actions)** — on push/PR to `main` / `master`: an ubuntu job runs `pnpm i18n:check`, `pnpm typecheck`, `pnpm test:unit`, `pnpm build` and the Playwright smoke (`PW_PREVIEW_ONLY=1` after build); a `windows-latest` job runs `cargo clippy --all-targets` then `cargo test` in `src-tauri` (empty `dist/` for `generate_context!`).
- **Build targets:** never cargo-build a scratch **copy** of the crate in the shared `src-tauri/target`: cargo hashes the root package by its relative path, so the copy overwrites the real crate's build script and `OUT_DIR` (symptom: "Permission diagnostics:default not found"). Use a separate `CARGO_TARGET_DIR`; recover with `cargo clean -p sigil`. `src-tauri/.cargo/config.toml` sets `GGML_NATIVE=OFF` / `GGML_AVX2=ON` for whisper-rs-sys (read only when cargo runs from `src-tauri/`; release.yml sets them too).
- `pnpm typecheck` — `tsc --noEmit`
- `pnpm test:unit` — vitest (`vitest.config.ts`, node environment, `@` alias) for pure TS logic: STT pipeline and limiter, translation sequencer, transform ordering, TTS helpers, voice-changer contract, pubsub inbound routing, diagnostics, the i18n staging merger, the models service and formatters, sysinfo / resources (including a Rust-id drift test), audio devices, `rustAudioInput` and `scripts/models/pin-catalog.mjs`
- Rust tests stay inline (`cargo test --manifest-path src-tauri/Cargo.toml`). Tests never touch the network (the model store uses a local warp fixture server). Tests that need real models, devices or GPU runtimes are `#[ignore]` and read environment variables: `SIGIL_TEST_MODELS_DIR` (Whisper `ggml-*.bin`, `sherpa-onnx/…`, `models/transcribe/…`, `rvc-default-voice/default.pth` + `vc-parity/…`), `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN` (CUDA 13 / cuDNN 9 folders for CUDA tests), `SIGIL_TEST_ECHO_DIR`, `SIGIL_TEST_TRANSCRIBE_PACKS`, `SIGIL_TEST_LLAMA_SERVER` / `SIGIL_TEST_LLAMA_GGUF`, `SIGIL_TEST_LOOPBACK_OUT` / `_IN` and `SIGIL_TEST_EXPOSURE=1` (opens a non-loopback socket; triggers a firewall prompt). Example: `SIGIL_TEST_MODELS_DIR=<dir> cargo test real_model -- --ignored`.
- Benchmarks: `Sigil.exe --headless bench scripts/bench/specs/stub-smoke.json` (plumbing) and `pwsh scripts/bench/run-perf-gate.ps1` (release builds; see [docs/benchmarks/README.md](benchmarks/README.md)). A voice-changer change records a run before and after; a p95 regression over 5 % blocks it.
- `pnpm test:e2e` — Playwright smoke (build + preview; `/` and `/client` shells mount `#root`; set `PW_PREVIEW_ONLY=1` if `dist/` is already built)
- `pnpm i18n:check` — locale JSON key parity vs `public/i18n/en/translation.json` and registered codes in `src/i18n.ts`
- `pnpm i18n:sync` — merge missing keys from English into other locale files (new keys copy English until translated)
- `pnpm i18n:report` — list keys whose value still matches English (prioritize refresh / human review)
- `pnpm i18n:translate` — regenerate MT locales from English via `scripts/build-translations.py` (slow; excludes `zh` / `zh_cn` / `zh_tw`, which are maintained manually)

### Translation review (human / community)

Priority locales for native or fluent review: **ja**, **ko**, **ar**, **ur**, **zh_cn**, **zh_tw**, then high-traffic EU languages.

Reviewers should preserve **`{{placeholders}}`**, **`<code>...</code>`** spans, product names (**OBS**, **VRChat**, **Twitch**, **Kick**, **Discord**), and abbreviations (**STT**, **TTS**). Prefer domain-correct wording for streaming and voice tooling over literal machine translation.

Use **`pnpm i18n:report`** to list keys that still match English; fix or file follow-up PRs per locale.

### RTL smoke checklist (`ar`, `ur`)

1. Switch UI language to **Arabic** or **Urdu** in Settings and confirm `<html dir="rtl" lang="…">` (handled in `src/i18n.ts`).
2. Open **Settings**, **sidebar**, and at least one **inspector**; confirm no clipped text or overlapping controls.
3. Open a **modal** and a **dropdown**; confirm menus align readably and close affordances are reachable.
4. Trigger a **toast**; confirm it does not sit off-screen or mirror incorrectly.
5. Prefer **logical** Tailwind on shared chrome (`ms-`/`me-`, `ps-`/`pe-`, `border-s`/`border-e`, `start-`/`end-`) instead of physical `ml-`/`mr-`/`pl-`/`pr-` where mirroring matters.

## Security notes (Tauri 2)

- **Capabilities:** IPC and filesystem access are gated by capability JSON (`migrated`, `desktop-capability`, and one file per newer plugin such as `diagnostics`). Prefer tightening `fs:scope` and plugin permissions over broad `*:default` when adding features.
- **Asset protocol:** `convertFileSrc` URLs are served only from `$APPDATA/vosk-models/**` (`app.security.assetProtocol.scope` in `tauri.conf.json`); Vosk is the only consumer. A new asset consumer should add its own narrow folder rather than widening the scope, and list `https://asset.localhost` (Windows, `useHttpsScheme`) / `asset:` in the matching CSP directive.
- **Threat model (short):** The Rust `web` plugin binds its HTTP/WebSocket stack to **`0.0.0.0`** (all IPv4 interfaces), so it is reachable across the **LAN**, not just localhost — a deliberate choice matching upstream Curses (`src-tauri/src/services/web/mod.rs`; a `127.0.0.1`-only bind rejected some localhost paths). It serves PeerJS signaling, the pubsub WebSocket, and static assets with **no authentication or TLS**. Since LAN security stage 1 the WebSocket handshakes pass an **Origin allow-list** (browser cross-site hijacking and DNS rebinding are refused), binary frames are refused and the host allow-lists inbound topics; a non-browser tool on the LAN can still forge an Origin until stage 2 adds a token. Remote clients use the **PeerJS** path and do not execute Tauri commands. Treat that LAN exposure as the real, shipped attack surface, and any change to the bind address, authentication, or TLS model as high-risk.
- **Expanded network model:** [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#network-security-and-threat-model) — LAN `/client`, linked pubsub, and operator hardening.

---

## Host services, persistence, and UI tabs

This section describes how **integrations**, **host-only modules**, and **sidebar/nav tabs** relate in Sigil. It complements the [connectivity audit](#service-and-feature-connectivity-audit) (Rust ↔ TS wiring).

### Three layers

1. **`Services` enum** ([`src/services-registry.ts`](../src/services-registry.ts))  
   Stable string ids used for **inspector navigation** (`InspectorTabPath.tab`), action bar ordering where applicable, and UI routing. Kept free of imports from `src/core` to avoid circular dependencies.

2. **`BackendState.services`** ([`src/core/schema.ts`](../src/core/schema.ts))  
   Zod-backed **persisted** settings per integration. Keys here are **`PersistedHostServiceKey`** (see `PERSISTED_SERVICE_KEYS`, `isPersistedServiceKey`, and `isPersistedServicesTab` in the same file). Use **`window.ApiServer.patchService(key, fn)`** only for these keys.

3. **`ApiServer` fields** ([`src/core/index.ts`](../src/core/index.ts))  
   Live service instances. Some correspond 1:1 with a persisted key (e.g. `stt` ↔ `state.services.stt`). Others are **host-only** (no row in `BackendState.services`).

### Persisted integrations (`BackendState.services`)

| Key | `ApiServer` field | Notes |
|-----|-------------------|-------|
| `vrc` | `vrc` | |
| `stt` | `stt` | |
| `tts` | `tts` | |
| `translation` | `translation` | |
| `transform` | `transform` | |
| `twitch` | `twitch` | |
| `kick` | `kick` | |
| `discord` | `discord` | |
| `obs` | `obs` | |

### UI tab without persisted slice

| `Services` id | `ApiServer` field | Persistence |
|---------------|-------------------|-------------|
| `voice_changer` | `voiceChanger` | Not under `state.services`. **Rust is authoritative**: the plugin persists its settings (`<app_data>/voice-changer/settings.json`, per-voice `<slot>/slot.json`) and the TS service hydrates from `get_voice_changer_params`. Do not use `patchService` for this tab. |

### Host modules (not `Services` tabs)

| Field | Role |
|-------|------|
| `_state` (`Service_State`) | Load/save `BackendState`, debounced writes through `AtomicJsonFile` (temp file + rename, `.bak` of the last good file, a damaged file kept as `settings.corrupt-<time>.bak`) |
| `keyboard` | Shortcuts / input |
| `sound` | Sound effects |
| `history` | Undo/history style state |

These participate in startup **`init()`** where applicable but are **not** inspector tabs in the `Services` enum.

### Central registration (TypeScript)

- **Startup init order and labels**: single registry in [`src/core/index.ts`](../src/core/index.ts) (`getHostServiceInitTasks`).
- **Inspector lazy imports, nav groups, sidebar buttons, header titles**: [`src/core/ui/servicePanels.tsx`](../src/core/ui/servicePanels.tsx).

### Checklist: add a **persisted** integration

1. Add `Services.<id>` in [`src/services-registry.ts`](../src/services-registry.ts) if it needs a sidebar/inspector tab.
2. Add `Service_*_Schema` and register under `BackendSchema.services` in [`src/core/schema.ts`](../src/core/schema.ts). Update **`PERSISTED_SERVICE_KEYS`** (TypeScript will enforce exhaustiveness against `PersistedHostServiceKey`).
3. Implement `src/core/services/<id>/` (`index.ts`, `schema.ts`, optional `services/` backends).
4. Add **`ApiServer`** field and include **`init`** in the registry inside [`src/core/index.ts`](../src/core/index.ts).
5. Add a panel entry in [`src/core/ui/servicePanels.tsx`](../src/core/ui/servicePanels.tsx) (lazy inspector, nav, sidebar, title).
6. If the service should appear on the bottom action bar, add the id to **`DEFAULT_ACTION_BAR_SERVICE_ORDER`** in [`src/core/schema.ts`](../src/core/schema.ts).
7. Optional: Tauri plugin + TS `invoke` (see connectivity audit).

### Checklist: add a **non-persisted** inspector tab

1. Add `Services.<id>` if it is a first-class tab.
2. **Do not** add a `BackendSchema.services` slice unless you intend to persist settings.
3. Implement service module and wire **`ApiServer`** + **init registry** as needed.
4. Register the panel in **`servicePanels.tsx`** only (avoid `patchService` for this id).

### Checklist: host-only module (no inspector)

1. Add **`ApiServer`** field and hook **`init`** in the registry if it needs startup work.
2. No changes to `Services`, `servicePanels`, or `BackendSchema.services` unless you later expose UI or persistence.

---

## Service and feature connectivity audit

This section inventories how host features connect across **UI → `ApiServer` → persisted schema → Tauri plugins** and records verification status.

### Feature matrix

| Feature | `ApiServer` property | `BackendSchema.services` key | `ApiServer.init()` | Left inspector | Primary nav | Tauri plugin(s) / notes |
|--------|----------------------|------------------------------|--------------------|----------------|-------------|-------------------------|
| Speech-to-text | `stt` | `stt` | Yes | `Services.stt` | Sidebar, Sigil nav, action bar | `whisper`, `vosk-stt`, `moonshine-stt`, `audio` (capture and utterance capture), `uwu` (optional); Groq over HTTPS from the webview |
| Text-to-speech | `tts` | `tts` | Yes | `Services.tts` | Sidebar, Sigil nav, action bar | `windows-tts`, `uberduck-tts`, `kokoro-tts`, `melo-tts`, `chatterbox-tts`, `fish-speech`, `neural-onnx-tts`, `neural-sidecar-tts`; native uses Web Speech API |
| Translation | `translation` | `translation` | Yes | `Services.translation` | Sidebar, Sigil nav, action bar | Azure (renderer `fetch`); `translate` plugin for LibreTranslate-compatible HTTP (`translate`, `detect_language`, `get_languages`, `check_availability`, each with per-call `endpoint` / `apiKey`) |
| AI transform | `transform` | `transform` | Yes | `Services.transform` | Sidebar, Sigil nav, action bar | Cloud/API per inspector (no single Rust plugin) |
| Twitch | `twitch` | `twitch` | Yes | `Services.twitch` | Sidebar, Sigil nav, dock, action bar | OAuth + IRC/WebSocket (JS); `web` for browser open |
| Kick | `kick` | `kick` | Yes | `Services.kick` | Sidebar, Sigil nav, action bar | OAuth + chat (JS) |
| Discord | `discord` | `discord` | Yes | `Services.discord` | Sidebar, Sigil nav, dock, action bar | Webhook/HTTP (JS) |
| VRChat | `vrc` | `vrc` | Yes | `Services.vrc` | Sidebar, Sigil nav, action bar | `osc` |
| OBS | `obs` | `obs` | Yes | `Services.obs` | Sidebar, Sigil nav, action bar | WebSocket to OBS (JS `obs-websocket-js`) |
| Keyboard / shortcuts | `keyboard` | — (uses root `shortcuts`) | Yes | Settings, STT inspector | — | `global-shortcut` registers `start`, `muteMic` and `muteSound` in every desktop build (Pressed events only, key repeat ignored; recording in [`accelerator.ts`](../src/core/services/keyboard/accelerator.ts)); `keyboard` (Windows low-level hook) only when the `background_input` **Cargo feature** is enabled |
| Voice changer | `voiceChanger` | — (Rust-persisted; TS hydrates) | Yes | `Services.voice_changer` | Sigil nav; legacy sidebar button added | `voice-changer`: built-in DSP, WebSocket chunk bridge, optional ONNX file (`OnnxVoiceBridge` + `ort`), RVC, TTS aux via `voice_changer_push_aux_wav`; tunable **chunk frames**, gains, buffer ms, **latency budget**, **worker priority**, **aux mic gate**, optional silence truncation, **`get_voice_changer_metrics`** (with the EP in use), optional **second (monitor) output** + **monitor gain**. See [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#voice-changer) vs VCClient-style expectations. |
| GPU / diagnostics | — (`src/core/services/diagnostics/`, bridge started by `initSystemLogListeners()`) | — | via system log init | Settings (`DiagnosticsPanel`) | — | `diagnostics` (registered first in `main.rs`): file log, EP outcomes, GPU adapters, EP probe (reports which NVIDIA runtime install and CUDA provider are in use), interim NVIDIA runtime pack; events `diagnostics:ep_outcome`, `diagnostics:log`, `diagnostics:probe`, `diagnostics:nvidia_pack` |
| Models & Runtimes | `models` | — (root keys `modelMirror`, `modelMirrorCustomUrl`, `modelMaxConcurrentDownloads`) | Yes ("Models") | `InspectorTabPath` `models` (`inspector_models.tsx`) | Sigil nav (above Settings) | `model-store` (catalog, install queue, `models:progress`, `models:state`), `runtime` (runtime packs, legacy folders) |
| System info / resources | `sysinfo`, `resources` | — | Yes ("System info", "Resources") | Models tab (`GpuRuntimeCard`) | — | `sysinfo` (`get_system_info`, `sysinfo:changed`), `resources` (`get_resource_snapshot`, `get_recommendation`, `resources:changed`) |
| Audio devices | `audioDevices` | — (mirrors the STT hard mute) | Yes ("Audio devices") | device pickers (`InputRustAudioInput`) | — | `audio` (`get_device_snapshot`, `set_capture_mute`, `get_capture_status`; events `audio:devices_changed`, `audio:stream_error`, `audio:stream_recovered`, `audio:stream_closed`) |
| Helper processes / benchmark | — (no UI yet: P4-B / P3-G) | — | — | — | — | `process-host` (`list_helpers`, `stop_helper`, `get_helper_logs`, `process_host:state`), `bench` (`run_benchmark`, `cancel_benchmark`, `get_last_report`, `bench:progress`, `bench:done`) |
| Sound / SFX | `sound` | — (uses root `muteSoundEffects`, devices) | **Yes** (awaited with other inits) | — | — | `plugin:audio|play_async` (TTS voice clips); no canvas visualizer |
| Document undo/redo | — | — | — | History panel / toolbar | — | **Yjs / client `documentUndoState`** (not `Service_History`) |
| `Service_History` | `history` | — | **Not called** | — | — | **Stub** — reserved; undo/redo is document-layer |
| PubSub / link | `ApiShared.pubsub` | `linkAddress`, etc. | via `ApiShared.init()` | — | — | `web` (`pubsub_broadcast`, WS, config) |
| Peer / client | `ApiShared.peer` | — | `peer.startServer()` in host init | — | — | `web` (PeerJS signaling) |
| System logs | `systemLog` module | — | `initSystemLogListeners()` after service init | `SystemLogsPanel` | — | Subscribes to service proxies (no IPC) |

### STT backend wiring

| `STT_Backends` | TS implementation (`stt/index.ts`) | Rust / runtime |
|----------------|-------------------------------------|----------------|
| `native` | `STT_NativeService` | Browser Web Speech API |
| `chrome` / `edge` | — (inspector opens external browser) | `web|open_browser` |
| `browser` | **Mapped to `native`** in `Service_STT.start()` (legacy persisted value) | Same as native |
| `azure` | `STT_AzureService` | Azure SDK (JS) |
| `deepgram` | `STT_DeepgramService` | Deepgram (JS) |
| `whisper` | `STT_WhisperService` | `plugin:whisper|*` |
| `vosk` | `STT_VoskService` | `plugin:vosk-stt|*`, `plugin:audio|*` |
| `moonshine` | `STT_MoonshineService` | `plugin:moonshine-stt|*`, utterance capture (`plugin:audio|start_utterance_capture` …) |
| `openai_audio` | `STT_OpenAI_AudioService` (preset `custom` or `groq`) | utterance capture + HTTP to the user server or Groq |

Engine contract (`ISTTReceiver`): `onStart`, `onStop(error?)`, `onInterim`, `onFinal`, **`onStatus(text, {progress, level})`** (UI only, never published; engines must never send status through `onInterim`) and **`isMuted()`**. `Service_STT.serviceState` carries `statusText`, `statusLevel` and `progress`; `setMuted(bool)` and `toggleRunning()` serve the shortcuts. Moonshine and `openai_audio` run `RustUtteranceService` ([`rustCapturePolling.ts`](../src/core/services/stt/rustCapturePolling.ts)) with `UtterancePipeline` (one ordered request per VAD utterance, merged when behind, classified retries) and the shared `GroqRateLimiter` ([`rateLimiter.ts`](../src/core/services/stt/rateLimiter.ts)). Persisted VAD / Whisper settings become Rust options in [`sttOptions.ts`](../src/core/services/stt/sttOptions.ts) (`RustVadConfig`, `RustWhisperDecodeOptions` in `src/types.ts`). The desktop default migration is [`migrate.ts`](../src/core/services/stt/migrate.ts); the model picker catalog [`whisperModels.ts`](../src/core/services/stt/whisperModels.ts) is tested against `whisper.rs`.

### TTS backend wiring

| `TTS_Backends` | TS implementation | Rust / runtime |
|----------------|-------------------|----------------|
| `native` | `TTS_NativeService` | Web Speech API |
| `windows` | `TTS_WindowsService` | `plugin:windows-tts|speak` |
| `azure` | `TTS_AzureService` | Azure (JS) |
| `uberduck` | `TTS_UberduckService` | `plugin:uberduck-tts|speak` |
| `voicevox` | `TTS_VoicevoxService` | HTTP to VOICEVOX |
| `kokoro` / `melo` / `chatterbox` / `fishSpeech` | matching `TTS_*Service` | respective `plugin:*` |
| `neuralOnnx` | `TTS_NeuralOnnxService` | `plugin:neural-onnx-tts` — Hub repo id **or** local **speech package** folder (`speech_package_v1`: `config.json` + ONNX + voices per [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#local-speech-package-onnx)). IPC: `neural_onnx_tts_set_model_source`, `neural_onnx_tts_set_local_package_format`, `neural_onnx_tts_set_execution_hint`; lazy load on `speak`. Additional on-disk formats (e.g. Piper) not implemented yet. |
| `neuralSidecar` | `TTS_NeuralSidecarService` | `plugin:neural-sidecar-tts` (user HTTP server: `/speech`, `/health`) |

Engine contract (`ITTSReceiver`): `onStart`, `onStop(error?)`, `onError(error)` (`onFilePlayRequest` is gone). Backends report missing settings with `TTS_OPTIONS_MISSING`; `stop()` must call `onStop()` and be idempotent; `play()` must not throw. `Service_TTS` gives each backend instance its own receiver, so stale callbacks are ignored. Shared helpers: `ClipTtsBackend` + `WebAudioClipQueue` for clip-based backends, `num.ts` (`finiteOr`, `clampedOr`, `positiveOr`) for numeric settings, `ssml.ts` for Azure, `voicevoxPresets.ts` (presets, migration, scales). `Service_Sound.enqueueVoiceClip` returns a promise that rejects with a localized message; `clearVoiceClips()` drops queued clips. Rust TTS/HTTP commands (`kokoro-tts`, `chatterbox-tts`, `melo-tts`, `neural-sidecar-tts`) accept an optional per-call `endpoint`; `fish-speech` also takes `apiKey` and `model`. All `neural_onnx_tts` commands are async and run on a background thread.

### Translation backend wiring

| `Translation_Backends` | TS implementation | Runtime |
|------------------------|---------------------|---------|
| `azure` | `Translation_AzureService` | Microsoft Translator REST from webview (`fetch`) |
| `libretranslate` | `Translation_LibreTranslateService` | `plugin:translate|*` → user HTTP server ([scripts/translation-sidecar/README.md](../scripts/translation-sidecar/README.md)) |
| `onDevice` | `Translation_OnDeviceService` | Experimental `globalThis.Translator` / `LanguageDetector` when exposed by the WebView |

Engines implement `ITranslationEngine`: `start` and `translate` take an `AbortSignal`, return a Promise and throw `TranslationError` (which says whether a retry makes sense); `stop` releases resources. `Service_Translation` owns epochs, the start abort and `TranslationSequencer` ([`sequencer.ts`](../src/core/services/translation/sequencer.ts)): per-stream utterances, latest-wins interims, ordered finals, deadlines 15 s / 30 s, one retry for fast retryable final failures. LibreTranslate always passes `endpoint` and `apiKey` per call; `set_endpoint` / `set_api_key` are unused by the frontend.

**AI transform:** engines implement `ITransformEngine` (`start(getData, signal)`, `transform(request, signal)`, `stop`) with `TransformNotConfiguredError` / `TransformRequestError`; settings are read per request. `OrderedQueue` ([`ordering.ts`](../src/core/services/transform/ordering.ts)) delivers outcomes in sentence order (ai, fallback or passthrough). Request building is pure in [`llmRequest.ts`](../src/core/services/transform/services/llmRequest.ts): `max_completion_tokens` for OpenAI and `max_tokens` elsewhere; `reasoning.effort` for OpenRouter and `reasoning_effort` elsewhere; `<think>` blocks stripped and key-like strings redacted from errors.

### Tauri plugins registered (`main.rs`)

`diagnostics` (first, so its setup runs before every other plugin), `osc`, `web`, `audio`, `windows-tts`, `uberduck-tts`, `keyboard`, `uwu`, `whisper`, `translate`, `kokoro-tts`, `moonshine-stt`, `melo-tts`, `chatterbox-tts`, `fish-speech`, `neural-onnx-tts`, `neural-sidecar-tts`, `voice-changer`, `vosk-stt`, then the P1 foundation plugins `model-store`, `runtime`, `sysinfo`, `resources`, `process-host`, `bench`. `main.rs` builds the app with `.build(..).run(..)` so `RunEvent::Exit` can call `process_host::shutdown_all()`.

Core commands: `get_port`, `get_native_features`, `app_close`, `oauth_loopback_start`.

### IPC vs capabilities

Permissions in `src-tauri/capabilities/migrated.json` include defaults for each custom plugin above plus `web`, `osc`, `audio`, `fs`, `shell`, `dialog`, `global-shortcut`, etc.; `diagnostics` and every P1 plugin (`model-store`, `runtime`, `sysinfo`, `resources`, `process-host`, `bench`) have their own `capabilities/<plugin>.json`. Each plugin's `default` set comes from `plugin-commands/<plugin>.txt` (see [Plugin commands, capabilities and i18n staging](#plugin-commands-capabilities-and-i18n-staging)).

**Note:** Legacy `plugin:bg_input|start` / `stop` invokes were removed from `Service_Keyboard`; background input uses `plugin:keyboard|start_tracking` / `stop_tracking` via `startBackgroundInput()` / `stopBackgroundInput()`.

### Element types (`ElementType`)

| Type | Left inspector (`Inspector_Project`) | Property panel | Add menus (sidebar / studio / inspector elements) |
|------|--------------------------------------|----------------|-----------------------------------------------------|
| `text` | Yes | Yes | Yes |
| `image` | Yes | Yes | Yes |

Legacy `audioViz` entries in saved documents are **stripped on load** (`stripLegacyAudioVizElements` in [src/client/schema/index.ts](../src/client/schema/index.ts)).

### Voice changer (summary)

Inspector maps to `plugin:voice-changer|set_voice_changer_params` (pitch, formant, devices, mode, sidecar URL, ONNX path, **chunk_frames**, **input/output gain**, **output_prefill_ms** / **output_max_queue_ms**, **latency_preset**, truncation fields) plus the RVC fields (slot and base paths, **`rvc_execution_provider`** (separate from `onnx_execution_provider`), pitch extractor, extra context, crossfade, index rate, protect, GPU device index, CUDA-only VRAM cap). Non-fatal RVC load issues arrive as `voice_changer:error` with code `rvc_warning`. Other codes: `device`, `sidecar_connect`, `onnx_load`, `rvc_load`, `worker_error` and `worker_panic` stop the engine; `monitor_unavailable` only loses the monitor output; `onnx_infer`, `sidecar_length`, `sidecar_exchange` and `rvc_convert` mean that block was muted (never dry input). The full IPC contract is `src-tauri/src/services/voice_changer/params_contract.json`.

Contract ([`params_contract.json`](../src-tauri/src/services/voice_changer/params_contract.json), checked by Rust `contract_tests.rs` and TS `params.test.ts`): P0 fields `latency_budget_ms`, `aux_mic_gate`, `worker_priority`, `input_device`, `rvc_consent_acknowledged` are optional on set (absent keeps the stored value) and always returned by get. `set_voice_changer_params` returns the **effective** params; event `voice_changer:params` fires when Rust changed what was sent (for example a voice's `slot.json` applied on entering it). `start_voice_changer({inputDevice?})` replaces and persists the stored device when given. Metrics add `process_p99_ms`, `backlog_ms`, `dropped_ms`, `underruns` and `ep`. Module layout: `voice_changer/{mod,params,state,persist,worker,budget,resample,aux_feed,playback,metrics,commands,debug,fallback}.rs` (`aux_feed`, because `aux.rs` is a reserved device name on Windows). **Socket.IO / REST** servers from the w-okada stack are **not** interchangeable with Sigil’s WebSocket sidecar without an adapter. Details: [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#voice-changer).

### Follow-up (optional)

- Playwright: open each inspector tab against `vite preview` to catch routing regressions without Tauri (PR CI now runs smoke tests on push; see `.github/workflows/ci.yml`).

---

## Speech stack internals

Rust modules added by the speech-stack phases ([plan](SPEECH_STACK_IMPLEMENTATION_PLAN.md)): P0 stabilization and the P1 platform foundations (runtime policy, model store, capture hub, native runtime packs, sherpa-onnx, transcribe.cpp, process host, system probe and resource governor, RVC templates, benchmark harness).

### Real-time rules

- Audio callbacks never allocate, lock, log or do IPC: they write whole frames into an `rtrb` ring and wake a worker (`unpark`). Conversion from any cpal sample format uses the allocation-free `pcm_convert::build_input_stream_f32`.
- MMCSS "Pro Audio" is registered **once per callback thread** (`rt_thread::register_mmcss_pro_audio_once`); inference workers never register. Worker priority comes from `rt_thread::set_worker_priority` (auto / protect_game / protect_voice). Admin rights are never requested.
- Epoch discipline: engines bump the epoch under the stop lock on every start and stop and emit only while the epoch is current.

### `speech_core` and utterance capture

`src-tauri/src/services/speech_core/`: `silero.rs` (embedded Silero VAD v6.2 on Sigil's ORT, CPU, 1+1 threads, spinning off), `segmenter.rs` (hysteresis 0.5/0.35, 400 ms pre-roll, 500 ms end silence + 200 ms tail, `SpeechStart` only after 250 ms of confirmed speech, forced split at 18 s at the quietest frame), `resample.rs` (streaming downmix + rubato to 16 kHz, exact length, measured group delay), `hallucination.rs` + `phrases/<lang>.txt` (whole-sentence stock phrases, `~` soft phrases, repeat collapsing, marker stripping), `utterance.rs` (`SpeechFrontEnd`, drop-oldest `BoundedQueue`, take-once `UtteranceStore` capped at 8 utterances / 120 s).

`audio/{speech_capture,utterance_capture}.rs` read a `Raw16kMono` consumer of the [capture hub](#capture-hub) on a worker thread and feed the front end; each discontinuity (restart, device move, mute edge, overrun) closes the open utterance first, and on loopback sources silence is filled in so an utterance still closes when playback stops. A lost device no longer ends the capture (the hub reconnects it). IPC contract:

- `plugin:audio|start_utterance_capture { deviceName?, vad?: VadConfig }` → capture id.
- Event `audio:utterance` (tagged by `kind`): `speech_start {capture_id, seq, start_ms}`, `utterance {capture_id, seq, start_ms, end_ms, sample_rate, samples, forced}`, `dropped {capture_id, seq, reason}`, `error {capture_id, message}`. Ignore other capture ids.
- `plugin:audio|take_utterance { captureId, seq }` → binary `ArrayBuffer` of little-endian f32 mono 16 kHz (take once).
- `plugin:audio|stop_utterance_capture`.

The legacy `audio:chunk` stream (Vosk) is batched about every 20 ms from a hub consumer on a worker thread; `start_audio_capture({ sampleRate: 16000 })` delivers 16 kHz mono (about a sixth of the old 48 kHz stereo IPC), any other request the device's own format, and every payload states its `sample_rate` / `channels`.

### Whisper

`whisper.rs`: one ordered worker per recording with a bounded queue (default 3, drop-oldest), one reused `WhisperState` per model, `single_segment`, `no_context`, `suppress_nst`, token cap scaled to utterance length, abort callback tied to the recording epoch. whisper.cpp's own no-speech value is stale for multi-token prompts, so its internal skip is disabled (`no_speech_thold = 1.0`) and `sot_no_speech_prob()` computes OpenAI's definition with one extra decoder step. `start_recording` accepts optional `vad` (`RustVadConfig`) and `decode` (`RustWhisperDecodeOptions`: `n_threads`, `initial_prompt`, `hotwords`, `no_speech_threshold`, `hallucination_filter`, `max_backlog`); the legacy RMS fields are ignored. Events: `whisper:segment_transcript` (string), `whisper:speech_start`, `whisper:utterance_dropped`, `whisper:capture_error`. The model catalog is `CATALOG` / `ALIASES` in `whisper.rs` (mirrored by the `whisper/<id>` entries of `model-catalog/stt-legacy.json`, so the Models tab and `--headless fetch` install into the same `whisper/` folder; `whisper.rs` still has its own downloader until P2-A moves it to `model_store::ensure`). Capture always goes through the capture hub: the JSON-PCM feed (`feed_audio_chunk`, `captureLocal: false`) is removed and `start_recording` rejects `captureLocal: false`. `whisper:capture_error` now means a fatal speech-detection error only; device loss is `audio:stream_error` / `audio:stream_recovered`. Log target `sigil::stt::whisper`.

### ONNX Runtime sessions and diagnostics

All Sigil ORT sessions go through [`neural_audio/ort_session.rs`](../src-tauri/src/services/neural_audio/ort_session.rs): `session_builder_for_vc(ep_hint, opts)?.component("vc.pitch").commit_from_file(p)` (or `build_session(component, ep_hint, opts, ModelSource)` → `(Session, EpOutcome)`). The ladder is CUDA → DirectML → CPU, each GPU step checked by a cached canary model, registered with `error_on_failure`, then an explicit CPU retry; the outcome is recorded with `diagnostics::record_ep_outcome`. Component keys: `vc`, `vc.content`, `vc.pitch`, `vc.onnx`, `tts.kitten`, `stt.vad`. Code that needs raw builder tweaks uses `build_session` or `builder_for_ep`. Since rc.13 (`runtime_version()` = 1.28.0, `CUDA_MAJOR` = 13): before the first CUDA session `native_runtime::nvidia::ensure_preloaded()` loads the NVIDIA runtime pack; the CUDA provider comes from `Sigil.exe`'s folder first, else from the registered app-data pack (`cuda_provider_in_use()` reports which). Test-only builders: `neural_audio::test_onnx::Graph` (hand-encoded ONNX), `vc::test_models` (fake RVC slots), `local_tts::test_model` (fake Kitten package). KittenTTS 0.3.0 is vendored in `local_tts/kitten/` (Apache-2.0 headers kept; `hub.rs` replaces hf-hub and downloads one file at a time into `app_data/models/kitten/`).

The `diagnostics` plugin (`src-tauri/src/services/diagnostics/`) owns the rotating file log (`logging.rs`, 10 MB × 5, `SIGIL_LOG`), the EcoQoS / timer throttling opt-out (`qos.rs`), DXGI adapters and VRAM (`gpu_mem.rs`), the canary (`canary.rs`), the light and deep EP probe (`ep_probe.rs`, deep probe via `--headless ep-probe`) and the NVIDIA runtime pack (`nvidia_pack.rs`). Its commands are listed in `plugin-commands/diagnostics.txt`; TS side in `src/core/services/diagnostics/`.

### Voice changer worker

`voice_changer/worker.rs` loads every model and bridge before any stream opens (the voice changer still opens its own cpal input until P3-E; it reports `resources::set_vc_running` on start and stop), drains the capture ring on every wake, and applies the latency budget (`budget.rs`: drop the oldest whole frames past max(budget, one block) down to max(block, budget/2) with a 5 ms equal-power splice; accounting captured = consumed + dropped + backlog). `aux_feed.rs` queues whole TTS clips (rubato, back-pressure at 60 s) and applies the mic gate; `resample.rs` is the stateful polyphase monitor resampler; `persist.rs` writes settings atomically (tmp, fsync, `.bak`, rename) through a 400 ms debounced writer flushed on exit. The WebSocket sidecar bridge (`neural_audio/sidecar_bridge.rs`) has connect/handshake timeouts and budget-bounded exchanges.

### Runtime policy (what runs where)

Sigil.exe links only its own static ONNX Runtime (ort 2.0.0-rc.13, ORT 1.28.0, [ADR 0001](adr/0001-ort-upgrade.md)) and whisper-rs's static ggml. Everything else is optional and loads lazily, so **the app starts, and every existing feature works, with no pack installed** (`scripts/release/expected-imports.txt` and `verify-bundle.ps1` guard the load-time imports):

| Runtime | How it runs | Where it lives | Owner module |
|---|---|---|---|
| Sigil's ORT (Silero, pitch models, RVC, Kitten, future Smart Turn / NLLB / the voice engine) | in-process, static | `Sigil.exe` | `neural_audio/ort_session.rs` |
| NVIDIA CUDA 13 + cuDNN 9 runtime | DLLs preloaded by full path before the first CUDA session | model-store pack `runtime/nvidia-cuda13`, else the interim Settings pack | `native_runtime::nvidia` |
| ONNX Runtime CUDA provider | next to `Sigil.exe` (installer), else the registered app-data pack `runtime/ort-cuda13-provider` (`register_ep_library`) | installer / app data | `ort_session` (`SIGIL_CUDA_PROVIDER=exe\|pack` dev override) |
| sherpa-onnx 1.13.8 (CPU STT/TTS) | in-process C API, its own CPU-only `onnxruntime.dll` loaded by full path | `runtime/sherpa-onnx` | `speech_rt::sherpa` ([ADR 0002](adr/0002-sherpa-onnx.md)) |
| transcribe.cpp (GGUF STT, GPU tier) | in-process `sigil_transcribe.dll` (static ggml inside, unique name) | `runtime/sigil-transcribe-<variant>` | `speech_rt::transcribe` ([ADR 0003b](adr/0003b-transcribe-cpp.md)) |
| other `sigil_engine` DLLs (echo today) | in-process, lazily loaded, hash-checked | `runtime/sigil-<engine>-<variant>` | `native_runtime` ([ADR 0007](adr/0007-native-runtime-packs.md)) |
| llama-server (and later Bergamot, qwentts, sbv2) | helper process in a kill-on-close Job Object, 127.0.0.1 only | `runtime/llama-server-<variant>` | `process_host` ([ADR 0005](adr/0005-process-host-llama.md)) |

Rules that follow from it:

- **ORT sessions** are built only through `ort_session` after `ensure_ort_environment`, with spinning off. rc.13's plain `SessionBuilder::new()` applies ORT's auto-EP policy, which only an explicit EP overrides, so never build sessions elsewhere. Import `TensorElementType` from `ort::value`; builders are `mut` before `commit_from_*`. Do not use `ort::ep::cuda::preload_dylibs` (its Windows list still names CUDA 12 DLLs). The CUDA EP needs driver R580+ and a Turing or newer GPU; anything else falls back to DirectML with the reason recorded. A provider DLL whose PE version differs from the linked ORT is refused (`provider_version_mismatch`).
- **Loading native code:** `native_runtime::load_library_in_dir` (`LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | DEFAULT_DIRS`) from the folder `model_store::installed_path(id)` returns. Every in-process ggml engine links ggml statically inside a uniquely named DLL built with `/MD`, `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF`; two engines that both loaded a shared `ggml.dll` would bind to whichever loaded first. sherpa's `onnxruntime.dll` stays CPU-only inside its pack and is refused if another `onnxruntime.dll` (for example Windows ML's System32 1.17) is already in the process.
- **Helpers** run through `process_host::spawn` only: `CREATE_NO_WINDOW | CREATE_SUSPENDED`, joined to their own Job Object before they run, `127.0.0.1` with a random 256-bit key passed by environment variable (never on the command line, never serialized), and a listener-table audit that stops a helper exposing any non-loopback port.
- **Loaded native code stays loaded** until exit (the NVIDIA runtime, the registered provider, GPU engine DLLs). Removing such a pack while Sigil runs is reported as removed, and CUDA falls back to DirectML until the next start.

### Model store and catalog authoring

`model_store` (`src-tauri/src/services/model_store/`, plugin `model-store`) replaces the per-plugin downloaders: `catalog.rs` (embedded catalog plus `SIGIL_MODEL_CATALOG_OVERRIDE`, validation, dependency order), `download.rs` (HTTP Range resume into `<sha>.partial` + `.partial.json`, 30 s stall timeout and no total timeout, streaming sha256 or git blob id, failover on connect errors / 403 / 404 / 429 / 5xx / HTML / wrong size / hash mismatch), `mirrors.rs` (catalog order plus an automatic hf-mirror twin; preference; custom mirror), `extract.rs` (zip, tar.gz, tar.bz2 with traversal, link, device-name, case-duplicate, entry, size and ratio caps), `install.rs` (plan, acquire by legacy adoption / `SIGIL_LOCAL_ASSETS` / network, stage, one-rename commit, atomic `installed.json`), `queue.rs` (FIFO, 1–2 jobs, one per host, cooperative cancel), `legacy.rs` (adopts `whisper/` and `vosk-models/` files by hash), `events.rs` (`models:progress` at most 4 Hz, `models:state`) and `commands.rs` (9 IPC commands).

- **Consumers** locate files only with `model_store::installed_path(id)` and install on demand with `model_store::ensure(&[ids]).await` (on a tokio runtime; works headless). Never build install paths by hand.
- **Layout:** ordinary entries `app_data/models/<family>/<id rest>/<version>/`; runtime packs `app_data/runtimes/<id rest>/<version>/<variant|default>/`; families `whisper` and `vosk` install into the legacy `whisper/` and `vosk-models/` folders the plugins read. `models/<family>/<id rest>/installed.json` is the commit marker.
- **Trust:** release catalogs are fully pinned (`build.rs` refuses a `published` entry with an unpinned file in release builds). Trust-on-first-use exists only in debug builds and never for executables. `unpublished` entries (Sigil-built packs, templates, exports; asset hosting is local only for now) come only from Import, `SIGIL_LOCAL_ASSETS` or a custom mirror, never from their catalog URLs.
- **Authoring:** add a fragment `src-tauri/model-catalog/<area>.json` (format, rules, layout and mirrors in [`model-catalog/README.md`](../src-tauri/model-catalog/README.md)); ids are forever (`whisper/<plugin id>`, `vosk/<id without vosk-model->`, `runtime/sigil-<engine>-<variant>`, `runtime/<name>`). Pin with `node scripts/models/pin-catalog.mjs [--fragment f] [--only prefix] [--write] [--hash-by-download]` (sequential, at least 2 s apart, cached in `%LOCALAPPDATA%\sigil-dev-cache`). `"size": 0` on an unpinned file means unknown. `--headless fetch <id>` with a scratch `SIGIL_APP_DATA_DIR` is the quickest end-to-end check.
- **Tests** run against a warp fixture server (resume, stall, failover, zip-slip, adoption, queue limits) and never touch the network.
- **TS:** `window.ApiServer.models` (`src/core/services/models/index.ts`): state `catalog`, `states`, `progress`, `disk`, `loaded`, `pending`, `opErrors`, `configProblem`, `configError`; methods `install`, `cancel`, `remove`, `verify`, `importLocal`, `refresh`, `load`, `pushConfig` (none reject; a reload never overwrites a newer `models:state` event). Pure helpers are in `models/format.ts`. The Models & Runtimes tab is `inspector_models.tsx`; deep-link with `window.ApiServer.changeTab({ tab: "models", value: "<entry id>" })`.
- **`<ModelRequirement ids reason? onReady? hideWhenReady?>`** (`src/core/ui/inspector/components/ModelRequirement.tsx`) is the card consumers mount: it adds requirements automatically, shows what is missing and the download size, downloads only after an explicit click, offers Import for import-only entries, and fires `onReady` once per transition to ready.
- `src/core/services/stt/downloadProgress.ts` formats both the legacy STT `{file, progress}` events and `models:progress` (`normalizeDownloadProgress`, `formatSpeed`, `formatEta`, `formatByteProgress`).

### Capture hub

`audio/{hub,consumer,watchdog,stream,loopback,device_watch}.rs` open **one cpal stream per physical source** and fan it out to consumers:

- **Source ids** are what the pickers store: `default`, an input name, `loopback:default` or `loopback:<output name>`. "default" and the same mic picked by name share one stream; a `default` source follows the Windows default device (epoch kept), a named one stays.
- **Consumers:** `hub::global()?.attach(SourceSpec::parse(device), ConsumerOptions::speech("stt") | ConsumerOptions::device_rate("vc"))`, optionally `.with_mute_group(Some(MuteGroup::…))`. Output is `Raw16kMono` (the speech front end's rubato resampler, exact length) or `DeviceRate`. Each consumer has its own rtrb ring with a header (frames, QPC capture time, discontinuity flags); a lagging consumer drops its own oldest blocks (`max_backlog_ms`), and `wait()` uses park/unpark. A read never mixes audio from both sides of a restart, device move, mute edge or overrun: `ReadInfo.gap_after` / `restarted` tell the caller to close its state first; `end_qpc_ns` stamps latency.
- **The audio callback** only copies into rings and unparks (no allocation, lock, log or IPC); consumers are added and removed through a command ring and their memory is returned through a second ring.
- **Watchdog:** a cpal error or 2.5 s without audio restarts the stream under the same epoch (backoff 100 ms doubling to 3 s); `audio:stream_error` once per outage, `audio:stream_recovered` once audio flows again, or `audio:stream_closed` when the source lost its last consumer before it recovered. Loopback sources never count as stalled.
- **Device watcher:** a hand-written `IMMNotificationClient` (polling fallback, 10 s re-check) emits `audio:devices_changed` with reason `added` / `removed` / `default_changed` / `state_changed`.
- **Mute gate:** `plugin:audio|set_capture_mute({ group, muted })` per `MuteGroup` (`stt`, `voice_changer`, `others`); muted consumers receive nothing.
- Commands: `get_device_snapshot`, `set_capture_mute`, `get_capture_status` (plus the existing capture commands). `SIGIL_CAPTURE_SHARING=0` restores one stream per consumer (soak A/B only). Log target `sigil::audio::hub`.
- Whisper, utterance capture (Moonshine / HTTP / Groq) and `audio:chunk` (Vosk) run on the hub. The voice changer still opens its own cpal input until P3-E moves it to a `DeviceRate` consumer in the `voice_changer` mute group.
- **TS:** `src/core/services/audioDevices.ts` (`ApiServer.audioDevices`) mirrors device lists and defaults, toasts outages and recoveries, mirrors the STT hard mute to the Rust gate, and restarts only browser-captured STT backends (Azure, Deepgram, Web Speech) after a microphone change or a hub backend whose start failed because its device was missing. `InputRustAudioInput` reads it (a saved but disconnected device stays listed as "not connected"; `includeLoopback` adds loopback sources, used only by the future other-player pickers); `src/utils/rustAudioInput.ts` has the source-id helpers.

### Native runtime packs and the `sigil_engine` ABI

`native_runtime` (plugin `runtime`, [ADR 0007](adr/0007-native-runtime-packs.md)) loads engine DLLs that implement **`native/common/sigil_engine.h` ABI v1**: 16 C exports (`sigil_engine_abi_version`, `_get_info`, `_last_error` (thread-local), `_set_log_callback`, `_create`, `_destroy`, `_cancel`, `_query`, `_process`, `_render`, `sigil_stream_open/_push/_finish/_close`, `sigil_free_string`, `sigil_free_audio`), capability bits `SIGIL_CAP_PROCESS | STREAM | RENDER | CANCEL | GPU`. Everything crossing the boundary is JSON plus interleaved f32 audio; memory returned by the engine is freed only by that engine; calls on one instance are serialized and `cancel` may come from any thread; no exception, `abort` or `exit` may cross the boundary.

- **Packs:** catalog id `runtime/sigil-<engine>-<variant>`, variants in preference order `cuda13`, `vulkan`, `cpu-avx512`, `cpu-avx2`, `cpu` (GPU variants need AVX2). Each pack carries `sigil-pack.json` (sizes and sha256 of every file, the engine's own info); the loader checks the hashes, the ABI version first, all 16 exports and the engine info before use. `native_runtime::open_engine("<engine>", &OpenRequest { … })` picks the best installed variant this PC supports and falls through to the next on a load failure or `SIGIL_E_BACKEND_UNAVAILABLE`. A missing, damaged or foreign DLL is an error, never a panic.
- **Writing an engine:** derive from `sigil::Engine` (`native/common/sigil_engine_impl.hpp`; the shared implementation provides the 16 exports, JSON, errors, cancel and logging) and call `sigil_add_engine(<name> VERSION … SOURCES …)` from `native/engines/<name>/CMakeLists.txt`; `native/cmake/SigilVariant.cmake` forces `/MD`, static third-party libraries, `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF` and the variant's instruction sets (MSVC-only flags are limited to C/C++ so nvcc never sees them). `pwsh scripts/native/build-runtime-packs.ps1 -Engine <name> -Variant <v>` builds, checks with `sigil_engine_check`, and writes reproducible zips to `native/out/` (`dev/`, `packs/`, `local-assets/`); an engine with its own `build.ps1` (transcribe) is delegated to it. See [`scripts/native/README.md`](../scripts/native/README.md).
- **Probe:** `Sigil.exe --headless engine-probe <engine> [model] [wav] [--variant v] [--runtime-dir dir] [-- extra]` (`sherpa`, `transcribe`, or any `sigil_engine` such as `echo`).
- **NVIDIA runtime:** `native_runtime::nvidia` preloads `runtime/nvidia-cuda13` (CUDA 13.2, cuBLAS 13.3, cuFFT 12.2, cuDNN 9.26 `cuda13`, plus every other cuDNN 9 sub-library the pack ships) by full path, falling back to the interim `diagnostics::nvidia_pack`. `fingerprint()` changes when a runtime or provider pack is installed or removed: cached GPU canaries and saved deep EP reports are discarded then. `list_runtime_packs` / `remove_legacy_runtime` list and delete `legacy/*` runtime folders of earlier Sigil versions.

### CPU speech runtime: `speech_rt::sherpa`

sherpa-onnx 1.13.8's C API through vendored bindgen output (`bindings.rs`, 46 functions, regenerated reproducibly by `scripts/runtime-packs/sherpa/gen-bindings.ps1`). The runtime pack `runtime/sherpa-onnx` (upstream `win-x64-shared-MT` build, 8 MB download) is loaded once per process and never unloaded: the pack's `onnxruntime.dll` by full path first (C API level 28), then `sherpa-onnx-c-api.dll`, with the version pinned to exactly 1.13.8. Wrappers (`online`, `offline`, `tts`, `lid`, `speaker`) are `Send + Sync` behind a per-object call lock; blocking calls go on `spawn_blocking`. TTS generation is serialized process-wide (espeak-ng keeps global state). Thread defaults: 2 (streaming recognizers 1, because ORT's spin-wait cannot be disabled through the C API), capped at 4. C++ exceptions do not cross the C API, so a corrupt model aborts the process: catalog models are hash-verified and user folders go through `probe::validate_in_child` first. `detect.rs` recognises official package folders. Dev override `SIGIL_SHERPA_RUNTIME_DIR`; real-model tests use `SIGIL_TEST_MODELS_DIR/sherpa-onnx/{pack-1.13.8,models/…}`.

### GPU speech runtime: `speech_rt::transcribe`

transcribe.cpp v0.2.4 (pinned commit, `native/engines/transcribe/UPSTREAM.json`) behind the `sigil_engine` ABI as `sigil_transcribe.dll`, built by `pwsh native/engines/transcribe/build.ps1 [-Variant cpu,cpu-avx2,cpu-avx512,cuda13,vulkan] [-SelfTest] [-FetchOnly] [-CudaArchitectures …]`. `TranscribeModel::open` picks the best installed `runtime/sigil-transcribe-*` variant (or `SIGIL_TRANSCRIBE_RUNTIME_DIR`), falls through GPU failures to the next variant or the same DLL's CPU backend (`fallback_reason()`), and offers `transcribe` / `transcribe_audio` (resampled with rubato), a canceller, `open_stream`, `warmup` and `device_memory`. The `cuda13` variant imports only `cublas64_13.dll` (cudart static) and requires `runtime/nvidia-cuda13`. Qwen3-ASR in 0.2.4 checks for cancel only before its single-shot decode, so callers drop stale results by epoch. Dev variables `SIGIL_TRANSCRIBE_CUDA_BIN`, `SIGIL_TEST_TRANSCRIBE_PACKS`; real-model tests read `SIGIL_TEST_MODELS_DIR/models/transcribe/`.

### Helper processes (`process_host`)

`process_host::spawn(HelperSpec)` → `HelperHandle { name, pid, port, base_url, api_key }` (the key is never serialized): `{port}` / `{api_key}` are substituted in args and environment, `PortRequest::{none, auto, fixed}` (auto binds `127.0.0.1:0`; a startup port collision retries on a new port), TCP then `GET /health` readiness, restart policy with doubling backoff (budget resets after 60 s of stable running), stop = close stdin, wait `stop_grace_ms`, then `TerminateJobObject`. Output goes to tracing target `sigil::helper` (rate-limited) and a 400-line ring (`get_helper_logs`, key redacted). `StdioMode::Protocol` + `take_stdio(name)` serve JSON-lines helpers (Bergamot); `run_worker(&WorkerSpec)` runs one-shot isolated workers (the `vc-import-worker`) with a memory cap, wall-clock limit and bounded output. `recipes::llama_server(exe, model, &LlamaServerOptions)` builds the llama-server spec (`--host 127.0.0.1 --port {port}`, key in `LLAMA_API_KEY`, `LLAMA_ARG_*` removed from the inherited environment). `shutdown_all` runs on `RunEvent::Exit`; `KILL_ON_JOB_CLOSE` kills helpers even when Sigil is killed. IPC: `list_helpers`, `stop_helper`, `get_helper_logs`; event `process_host:state`.

### System probe and resource governor

- **`sysinfo`** (`probe()`, `cached()`, `refresh()`; IPC `get_system_info`; event `sysinfo:changed`; `Sigil.exe --headless sysinfo`): DXGI adapters plus a D3D12 capability test, NVML (loaded from System32, kept for the process) for driver, CUDA version, compute capability and device memory, `cuDriverGetVersion` from nvcuda without `cuInit`, CPUID features and core counts, total RAM. It never loads ORT or a GPU runtime and never creates a CUDA context; the GPU part has a 15 s timeout. EP eligibility (`sysinfo::eps`): CUDA and TensorRT need Turing (compute capability 7.5) and R580; TensorRT-RTX needs Ampere and R580; DirectML needs D3D12 feature level 11_0; unknown facts give `eligible` with code `unverified`. Eligibility covers GPU, driver and OS only, not whether the runtime pack is installed.
- **`resources`**: GPU-model subsystems call `resources::claim_vram(Subsystem, label, estimate_mb)` before loading (`claim_external_vram` for helper processes), `resize` the RAII claim to the measured footprint, claim 0 MB on the CPU, and ask `resources::recommend(topic, &sysinfo::cached(), &resources::snapshot())` (IPC `get_recommendation`) before choosing a model size. The voice changer reports `set_vc_running` on start and stop and `set_vc_gpu` once its models are loaded (until then `resources` infers the device from the `vc` / `vc.*` EP outcomes; benchmark sessions record as `bench.vc*` and never count). The policy is pure and integer-only: room = primary VRAM − reserve (30 %, at least 2 GB) − the voice changer (its claims, or 1.2 GB while running without one) − other subsystems' claims; candidates in quality order, the first that fits wins; live budget numbers are deliberately not an input. Recommendation ids are `resources::policy::ids` (mirrored and drift-tested in `src/core/services/resources/format.ts`); footprints are provisional until the P7 measurement campaign. Event `resources:changed` (coalesced, 250 ms).
- **TS:** `ApiServer.sysinfo` and `ApiServer.resources` (`src/core/services/{sysinfo,resources}/`) feed `GpuRuntimeCard` on the Models & Runtimes tab.

### RVC templates and the importer

`scripts/vc-templates/` (maintainer-only Python: torch 2.3.1, onnx 1.17.0, onnxruntime 1.19.2) exports the RVC architecture as **split graphs** with external initializers ([ADR 0004](adr/0004-rvc-templates.md)): `enc` (phone + pitch → m_p / logs_p), `flowdec` (flow reverse + NSF decoder over a window, inputs z_noise, src_noise, phase0, g) and `knn` (exact retrieval blend), each fp32 and fp16-inside with fp32 I/O, plus `layout.json` mapping every initializer to its checkpoint key or weight_norm fold. `export_rvc.py` is deterministic; `parity_test.py` gates fp32 at 1e-4 and fp16 at 5e-3 against upstream PyTorch (`--write-fixture` regenerates `src-tauri/test-fixtures/vc/tiny`, `--rust-weights` byte-compares the Rust writer). The bundled template `src-tauri/resources/vc-templates/v2-40k-f0-hifigan/` ships as an installer resource (`tauri.conf.json` `bundle.resources`, found at `<exe>/resources/vc-templates`, `SIGIL_VC_TEMPLATES_DIR` for experimental exports). Precision policy from the measurements: `enc` always fp32; `flowdec` and `knn` fp16 on GPU (flowdec's decoder stage 0 stays fp32), fp32 on CPU.

`neural_audio::vc::import` is the importer core behind `Sigil.exe --headless vc-import-worker <job.json>` (job: `version`, `source`, `out_dir`, `templates_dir`, `time_limit_s`, `smoke`, `smoke_fp16`; output `weights.f32.bin`, `speakers.f32.bin`, `sigil-import.json`): zip guard, an arena-based allow-list pickle VM (legacy pickles and TorchScript refused), bounds-checked tensor views, metadata detection, strict template match, f64 weight_norm fold bit-identical to Python, a weights writer that rejects leftover tensors, and an in-memory session builder. Errors carry one of 16 codes. The default-voice parity test is `cargo test vc::import -- --ignored` with `SIGIL_TEST_MODELS_DIR` (`rvc-default-voice/default.pth`, `vc-parity/<template>/reference.*`), optional `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN` / `SIGIL_TEST_SKIP_DML`.

### Benchmark harness

`src-tauri/src/services/bench/` drives the production voice-changer path (`voice_changer::{budget, playback, fallback}`, `VcStreamHarness`, `RvcModel`) in `offline`, `paced` (`wall`, `virtual` or `simulated` device clock) and `e2e_loopback` runs, with a GPU/CPU load simulator and stall injection. Entry points: `Sigil.exe --headless bench <spec.json> [--out dir]` (report.json, report.md, per-run timeline.csv; exit 1 on a failed threshold), IPC `plugin:bench|run_benchmark({ spec, outDir? })` / `cancel_benchmark` / `get_last_report` (specs resolve against `app_data/bench/` and `outDir` must stay inside it; events `bench:progress`, `bench:done`), and cargo tests with stub models. `scripts/bench/specs/stub-smoke.json` is the plumbing smoke; `pwsh scripts/bench/run-perf-gate.ps1 -Baseline <dir>` blocks a p95 regression over 5 % on release builds. Full spec and metric reference: [docs/benchmarks/README.md](benchmarks/README.md).

### HTTP rules (`http_client.rs`)

Every API call sets a per-request timeout from the classes in `http_client.rs` (`HEALTH_TIMEOUT`, `LIST_TIMEOUT`, `translate_timeout`, `tts_timeout`, `stt_timeout_for_wav_bytes`); the shared client drops a connection that sends nothing for 120 s. Model downloads use `model_download_client()` (no total timeout, `MODEL_STALL_TIMEOUT` = 30 s without data). Build URLs with `api_url` / `normalize_base_url`, report failures with `describe_request_error` / `describe_api_error` (UTF-8-safe previews, server message extracted, URLs redacted), read audio with `read_body_capped` + `ensure_audio_body`. Keys go in headers or bodies, never in URLs.

---

## Remote `/client` mode

`src/client/` contains **no** `invoke(` calls — clients rely on the host for Tauri. STT/TTS configuration is host-only ([OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md)).

---

## Client folder (`src/client`)

Everything that needs to work on remote clients goes here.

**This folder should never reference server** (host-only APIs).

- public state (elements, scenes, files)
- client sound (text animation for now)
- particles
- scenes
- files
