# Changelog

All notable changes to **Sigil** are documented here. Sigil continues from the **Curses / Curses+** lineage; compare **0.3.0** to **Curses+ 0.2.2** for the largest delta.


## [0.3.0] - 2026-03-30

### Highlights

- **Rebrand** to **Sigil** (app name, icons, bundle id `com.sigil.app`, Windows binary **Sigil.exe**).
- **Tauri 2** migration: plugins, capabilities/ACL, `@tauri-apps/api` v2, fs/dialog/shell/process/global-shortcut plugins.
- **pnpm** as the standard package manager for the repo (npm lockfile removed).

### Speech-to-text

- **Whisper (Rust)** — Expanded model options (additional quantized and Large-v3–class models); download and pipeline hardening across the upgrade branch.
- **Moonshine** — HTTP integration (Useful Sensors–style server: health + transcribe).
- **Vosk** — Rust-side catalog and on-disk model zip caching; load via Tauri FS + browser runtime.
- **OpenAI-style local HTTP** — `POST …/v1/audio/transcriptions` for local sidecars (e.g. speaches, LocalAI-style stacks).
- **PubSub** — Binary audio path can feed the Whisper pipeline for linked/distributed setups.
- **Stream alignment** — Options to stop STT/TTS with Twitch stream state; inspector and locale strings updated.

### Text-to-speech

### Translation

### Voice, audio, and VRChat

- **Voice changer** — Real-time pitch (semitones), formant control, presets, input/output device selection (Tauri plugin).
- **Global audio devices** — Shared input/output selection with Rust audio capture for multiple services.
- **OSC** — Remains available from the Tauri host for VRChat-oriented workflows.

### Streaming and integrations

- **Twitch** — OAuth for public and confidential apps; loopback and env via `SIGIL_*` (legacy `CURSES_*` still supported).
- **Kick** — OAuth and chat with the same env conventions.
- **OBS** — Browser source and obs-websocket workflows; fixes for browser source captions and related behavior.
- **Discord** — Webhook outbound paths unchanged in scope; see README.

### Editor and UI

- **Sigil Studio** — Dashboard/header refresh; **Outfit** for key chrome typography.
- **Layout** — Restored modern host layout, bottom panel, themes, property inspector placement.
- **System logs** — In-app diagnostics for services.
- **Legacy templates** — Old audio visualizer elements stripped on load (`legacyDocument` migration).

### Internationalization

- English plus Chinese (`zh`, `zh_cn`, `zh_tw`) updated for new service keys (sidebar, STT/TTS Twitch options, Azure secondary language, transform title, chat delay, etc.).

### Documentation and build

- README, **ARCHITECTURE.md**, **docs/BACKENDS.md**, **docs/MANUAL-TAURI-CHECKLIST.md**, and related upgrade notes.
- GitHub **release** workflow: Node 20, `Sigil.exe` + MSI artifacts, build-time secrets for Twitch, Kick, Google Fonts, optional Twitch client secret.

### Upgrade notes (from Curses+ 0.2.2)

1. Installers and shortcuts use **Sigil**; refresh OBS browser URLs if they still reference Curses.
2. Prefer **`SIGIL_*`** env vars in `.env`; **`CURSES_*`** remains as fallback where documented.
3. Custom automation targeting **Tauri 1** APIs must move to **Tauri 2**.
4. Use **`pnpm install`** / **`pnpm tauri build`** for this repo.

### Credits

Sigil is a fork of **[Curses](https://github.com/mmpneo/curses)** by **mmpneo**. Thanks to the original author and all contributors.

---

## [0.2.2] - (Curses+)

Last pre-Sigil milestone in this fork’s history: **Whisper FFI refactor** for high-performance local STT. For full detail, see git history at tag/commit `Release v0.2.2: Whisper FFI Refactor`.

[0.3.0]: https://github.com/Mewnah/Sigil/releases/tag/v0.3.0
