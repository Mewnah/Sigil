# Developer guide

Architecture, **host services**, **connectivity audit**, and **client-layer rules** for contributors.

## Table of contents

- [Stack and runtime](#stack-and-runtime)
- [Layering](#layering)
- [Data flow (host)](#data-flow-host)
- [Native IPC](#native-ipc)
- [UI state](#ui-state)
- [Release / quality](#release--quality)
- [Security notes (Tauri 2)](#security-notes-tauri-2)
- [Host services, persistence, and UI tabs](#host-services-persistence-and-ui-tabs)
- [Service and feature connectivity audit](#service-and-feature-connectivity-audit)
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
2. **Document:** Yjs doc in `ApiClient.document`; `Service_Peer` syncs over PeerJS (see `src/shared/services/peer/`).
3. **Local server:** Rust `web` plugin binds `0.0.0.0:<port>` (all IPv4 interfaces, reachable across the LAN) — `ping`, PeerJS path, pubsub WebSocket, static assets (`src-tauri/src/services/web/`).

## Native IPC

- Generic `invoke` commands: `get_native_features`, `get_port`, `app_close`.
- Feature work is mostly `plugin:<name>|<command>` from the frontend.

## UI state

Sidebar tab visibility, expand/collapse, and multi-selection live in **Zustand** (`useAppUIStore`) so React and `ApiServer.changeTab` / `closeSidebar` stay in sync without a Valtio `window.ApiServer.ui` object.

## Release / quality

- **Release artifact** — `pnpm tauri build` produces a **single** Windows exe/MSI with unified ONNX Runtime (CPU + CUDA + DirectML via `ort`); execution provider is chosen at runtime in the voice changer inspector (the neural TTS inspector only stores its choice; local speech runs on the CPU). No separate CPU/GPU downloads.
- **CI (GitHub Actions)** — on push/PR to `main` / `master`: an ubuntu job runs `pnpm i18n:check`, `pnpm typecheck`, `pnpm build` and the Playwright smoke (`PW_PREVIEW_ONLY=1` after build); a `windows-latest` job runs `cargo clippy --all-targets` then `cargo test` in `src-tauri` (empty `dist/` for `generate_context!`).
- `pnpm typecheck` — `tsc --noEmit`
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

- **Capabilities:** IPC and filesystem access are gated by capability JSON (`migrated`, `desktop-capability`). Prefer tightening `fs:scope` and plugin permissions over broad `*:default` when adding features.
- **Asset protocol:** `convertFileSrc` URLs are served only from `$APPDATA/vosk-models/**` (`app.security.assetProtocol.scope` in `tauri.conf.json`); Vosk is the only consumer. A new asset consumer should add its own narrow folder rather than widening the scope, and list `https://asset.localhost` (Windows, `useHttpsScheme`) / `asset:` in the matching CSP directive.
- **Threat model (short):** The Rust `web` plugin binds its HTTP/WebSocket stack to **`0.0.0.0`** (all IPv4 interfaces), so it is reachable across the **LAN**, not just localhost — a deliberate choice matching upstream Curses (`src-tauri/src/services/web/mod.rs`; a `127.0.0.1`-only bind rejected some localhost paths). It serves PeerJS signaling, the pubsub relay, and static assets with **no authentication or TLS**. Remote clients use the **PeerJS** path and do not execute Tauri commands, but anyone who can reach the port on the LAN can hit those endpoints. Treat that LAN exposure as the real, shipped attack surface, and any change to the bind address, authentication, or TLS model as high-risk.
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
| `voice_changer` | `voiceChanger` | Not under `state.services` (in-memory / sidecar state). Do not use `patchService` for this tab. |

### Host modules (not `Services` tabs)

| Field | Role |
|-------|------|
| `_state` (`Service_State`) | Load/save `BackendState`, debounced writes |
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
| Speech-to-text | `stt` | `stt` | Yes | `Services.stt` | Sidebar, Sigil nav, action bar | `whisper`, `vosk-stt`, `moonshine-stt`, `audio` (capture), `uwu` (optional) |
| Text-to-speech | `tts` | `tts` | Yes | `Services.tts` | Sidebar, Sigil nav, action bar | `windows-tts`, `uberduck-tts`, `kokoro-tts`, `melo-tts`, `chatterbox-tts`, `fish-speech`, `neural-onnx-tts`, `neural-sidecar-tts`; native uses Web Speech API |
| Translation | `translation` | `translation` | Yes | `Services.translation` | Sidebar, Sigil nav, action bar | Azure (renderer `fetch`); `translate` plugin for LibreTranslate-compatible HTTP (`set_endpoint`, `set_api_key`, `translate`, …) |
| AI transform | `transform` | `transform` | Yes | `Services.transform` | Sidebar, Sigil nav, action bar | Cloud/API per inspector (no single Rust plugin) |
| Twitch | `twitch` | `twitch` | Yes | `Services.twitch` | Sidebar, Sigil nav, dock, action bar | OAuth + IRC/WebSocket (JS); `web` for browser open |
| Kick | `kick` | `kick` | Yes | `Services.kick` | Sidebar, Sigil nav, action bar | OAuth + chat (JS) |
| Discord | `discord` | `discord` | Yes | `Services.discord` | Sidebar, Sigil nav, dock, action bar | Webhook/HTTP (JS) |
| VRChat | `vrc` | `vrc` | Yes | `Services.vrc` | Sidebar, Sigil nav, action bar | `osc` |
| OBS | `obs` | `obs` | Yes | `Services.obs` | Sidebar, Sigil nav, action bar | WebSocket to OBS (JS `obs-websocket-js`) |
| Keyboard / shortcuts | `keyboard` | — (uses root `shortcuts`) | Yes (no-op if `background_input` off) | Settings | — | `keyboard` (Windows low-level hook when `background_input` **Cargo feature** enabled); `global-shortcut` for hotkeys |
| Voice changer | `voiceChanger` | — (in-memory / defaults in service) | Yes | `Services.voice_changer` | Sigil nav; legacy sidebar button added | `voice-changer`: built-in DSP, WebSocket chunk bridge, optional ONNX file (`OnnxVoiceBridge` + `ort`), TTS aux via `voice_changer_push_aux_wav`; tunable **chunk frames**, gains, buffer ms, optional silence truncation, **`get_voice_changer_metrics`**, optional **second (monitor) output** + **monitor gain**. See [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#voice-changer) vs VCClient-style expectations. |
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
| `moonshine` | `STT_MoonshineService` | `plugin:moonshine-stt|*`, `plugin:audio|*` |
| `openai_audio` | `STT_OpenAI_AudioService` | `plugin:audio|*` + HTTP to user server |

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

### Translation backend wiring

| `Translation_Backends` | TS implementation | Runtime |
|------------------------|---------------------|---------|
| `azure` | `Translation_AzureService` | Microsoft Translator REST from webview (`fetch`) |
| `libretranslate` | `Translation_LibreTranslateService` | `plugin:translate|*` → user HTTP server ([scripts/translation-sidecar/README.md](../scripts/translation-sidecar/README.md)) |
| `onDevice` | `Translation_OnDeviceService` | Experimental `globalThis.Translator` / `LanguageDetector` when exposed by the WebView |

### Tauri plugins registered (`main.rs`)

`osc`, `web`, `audio`, `windows-tts`, `uberduck-tts`, `keyboard`, `uwu`, `whisper`, `translate`, `kokoro-tts`, `moonshine-stt`, `melo-tts`, `chatterbox-tts`, `fish-speech`, `neural-onnx-tts`, `neural-sidecar-tts`, `voice-changer`, `vosk-stt`

Core commands: `get_port`, `get_native_features`, `app_close`, `oauth_loopback_start`.

### IPC vs capabilities

Permissions in `src-tauri/capabilities/migrated.json` include defaults for each custom plugin above plus `web`, `osc`, `audio`, `fs`, `shell`, `dialog`, `global-shortcut`, etc.

**Note:** Legacy `plugin:bg_input|start` / `stop` invokes were removed from `Service_Keyboard`; background input uses `plugin:keyboard|start_tracking` / `stop_tracking` via `startBackgroundInput()` / `stopBackgroundInput()`.

### Element types (`ElementType`)

| Type | Left inspector (`Inspector_Project`) | Property panel | Add menus (sidebar / studio / inspector elements) |
|------|--------------------------------------|----------------|-----------------------------------------------------|
| `text` | Yes | Yes | Yes |
| `image` | Yes | Yes | Yes |

Legacy `audioViz` entries in saved documents are **stripped on load** (`stripLegacyAudioVizElements` in [src/client/schema/index.ts](../src/client/schema/index.ts)).

### Voice changer (summary)

Inspector maps to `plugin:voice-changer|set_voice_changer_params` (pitch, formant, devices, mode, sidecar URL, ONNX path, **chunk_frames**, **input/output gain**, **output_prefill_ms** / **output_max_queue_ms**, **latency_preset**, truncation fields) plus the RVC fields (slot and base paths, **`rvc_execution_provider`** (separate from `onnx_execution_provider`), pitch extractor, extra context, crossfade, index rate, protect, GPU device index, CUDA-only VRAM cap). Non-fatal RVC load issues arrive as `voice_changer:error` with code `rvc_warning`. **Socket.IO / REST** servers from the w-okada stack are **not** interchangeable with Sigil’s WebSocket sidecar without an adapter. Details: [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#voice-changer).

### Follow-up (optional)

- Playwright: open each inspector tab against `vite preview` to catch routing regressions without Tauri (PR CI now runs smoke tests on push; see `.github/workflows/ci.yml`).

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
