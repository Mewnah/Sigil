# Service and feature connectivity audit

This document inventories how host features connect across **UI → `ApiServer` → persisted schema → Tauri plugins** and records verification status. Last updated as part of the connectivity audit implementation.

## Feature matrix

| Feature | `ApiServer` property | `BackendSchema.services` key | `ApiServer.init()` | Left inspector | Primary nav | Tauri plugin(s) / notes |
|--------|----------------------|------------------------------|--------------------|----------------|-------------|-------------------------|
| Speech-to-text | `stt` | `stt` | Yes | `Services.stt` | Sidebar, Sigil nav, action bar | `whisper`, `vosk-stt`, `moonshine-stt`, `audio` (capture), `uwu` (optional) |
| Text-to-speech | `tts` | `tts` | Yes | `Services.tts` | Sidebar, Sigil nav, action bar | `windows-tts`, `uberduck-tts`, `kokoro-tts`, `melo-tts`, `chatterbox-tts`, `fish-speech`; native uses Web Speech API |
| Translation | `translation` | `translation` | Yes | `Services.translation` | Sidebar, Sigil nav, action bar | `translate` (LibreTranslate path) |
| AI transform | `transform` | `transform` | Yes | `Services.transform` | Sidebar, Sigil nav, action bar | Cloud/API per inspector (no single Rust plugin) |
| Twitch | `twitch` | `twitch` | Yes | `Services.twitch` | Sidebar, Sigil nav, dock, action bar | OAuth + IRC/WebSocket (JS); `web` for browser open |
| Kick | `kick` | `kick` | Yes | `Services.kick` | Sidebar, Sigil nav, action bar | OAuth + chat (JS) |
| Discord | `discord` | `discord` | Yes | `Services.discord` | Sidebar, Sigil nav, dock, action bar | Webhook/HTTP (JS) |
| VRChat | `vrc` | `vrc` | Yes | `Services.vrc` | Sidebar, Sigil nav, action bar | `osc` |
| OBS | `obs` | `obs` | Yes | `Services.obs` | Sidebar, Sigil nav, action bar | WebSocket to OBS (JS `obs-websocket-js`) |
| Keyboard / shortcuts | `keyboard` | — (uses root `shortcuts`) | Yes (no-op if `background_input` off) | Settings | — | `keyboard` (Windows low-level hook when `background_input` **Cargo feature** enabled); `global-shortcut` for hotkeys |
| Voice changer | `voiceChanger` | — (in-memory / defaults in service) | Yes | `Services.voice_changer` | Sigil nav; legacy sidebar button added | `voice-changer` |
| Sound / SFX | `sound` | — (uses root `muteSoundEffects`, devices) | **Yes** (awaited with other inits) | — | — | `plugin:audio|play_async` (TTS voice clips); no canvas visualizer |
| Document undo/redo | — | — | — | History panel / toolbar | — | **Yjs / client `documentUndoState`** (not `Service_History`) |
| `Service_History` | `history` | — | **Not called** | — | — | **Stub** — reserved; undo/redo is document-layer |
| PubSub / link | `ApiShared.pubsub` | `linkAddress`, etc. | via `ApiShared.init()` | — | — | `web` (`pubsub_broadcast`, WS, config) |
| Peer / client | `ApiShared.peer` | — | `peer.startServer()` in host init | — | — | `web` (PeerJS signaling) |
| System logs | `systemLog` module | — | `initSystemLogListeners()` after service init | `SystemLogsPanel` | — | Subscribes to service proxies (no IPC) |

## STT backend wiring

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

## TTS backend wiring

| `TTS_Backends` | TS implementation | Rust / runtime |
|----------------|-------------------|----------------|
| `native` | `TTS_NativeService` | Web Speech API |
| `windows` | `TTS_WindowsService` | `plugin:windows-tts|speak` |
| `azure` | `TTS_AzureService` | Azure (JS) |
| `uberduck` | `TTS_UberduckService` | `plugin:uberduck-tts|speak` |
| `voicevox` | `TTS_VoicevoxService` | HTTP to VOICEVOX |
| `kokoro` / `melo` / `chatterbox` / `fishSpeech` | matching `TTS_*Service` | respective `plugin:*` |

## Tauri plugins registered (`main.rs`)

`osc`, `web`, `audio`, `windows-tts`, `uberduck-tts`, `keyboard`, `uwu`, `whisper`, `translate`, `kokoro-tts`, `moonshine-stt`, `melo-tts`, `chatterbox-tts`, `fish-speech`, `voice-changer`, `vosk-stt`

Core commands: `get_port`, `get_native_features`, `app_close`, `oauth_loopback_start`.

## IPC vs capabilities

Permissions in `src-tauri/capabilities/migrated.json` include defaults for each custom plugin above plus `web`, `osc`, `audio`, `fs`, `shell`, `dialog`, `global-shortcut`, etc.

**Note:** `Service_Keyboard.start()` / `stop()` invoke `plugin:bg_input|start` / `stop`, but the **keyboard** plugin only registers `start_tracking` / `stop_tracking`. Those `start`/`stop` methods are **not referenced** from the UI; background input uses `start_tracking` via `startBackgroundInput()`. Treat `bg_input` invokes as legacy unless a Rust `bg_input` plugin is reintroduced.

## Remote `/client` mode

`src/client/` contains **no** `invoke(` calls — clients rely on the host for Tauri. STT/TTS configuration is host-only ([docs/BACKENDS.md](BACKENDS.md)).

## Element types (`ElementType`)

| Type | Left inspector (`Inspector_Project`) | Property panel | Add menus (sidebar / studio / inspector elements) |
|------|--------------------------------------|----------------|-----------------------------------------------------|
| `text` | Yes | Yes | Yes |
| `image` | Yes | Yes | Yes |

Legacy `audioViz` entries in saved documents are **stripped on load** (`stripLegacyAudioVizElements` in [src/client/schema/index.ts](src/client/schema/index.ts)).

## Follow-up (optional)

- Playwright: open each inspector tab against `vite preview` to catch routing regressions without Tauri.
- Remove or implement `keyboard.start()` / `stop()` `bg_input` calls to match a real plugin or delete dead API.
