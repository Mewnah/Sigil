# Changelog

All notable changes to **Sigil** are documented here. Sigil continues from the **Curses / Curses+** lineage; compare **0.3.0** to **Curses+ 0.2.2** for the largest delta.


## [0.3.1] - 2026-04-02

### Internationalization

- **New locale bundles:** Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese (Brazil), Russian, Thai, Turkish, Urdu, and Vietnamese (`public/i18n/<code>/translation.json`), plus refreshed Simplified / Traditional Chinese (`zh`, `zh_cn`, `zh_tw`).
- **Tooling:** `pnpm i18n:check` (key parity vs English + registered codes), `pnpm i18n:sync` (merge missing keys from `en`), `pnpm i18n:report` (keys still identical to English), `pnpm i18n:translate` → `scripts/build-translations.py` (optional Google MT refresh).
- **RTL:** `ar` and `ur` set `dir` / `lang` on `<html>`; shared UI chrome uses logical Tailwind (`ms`/`me`, `ps`/`pe`, `start`/`end`, etc.) where mirroring matters.
- **i18next plural forms** for multi-element duplicate actions and toasts (`elements.duplicate_n_*`, `elements.toast_duplicated_n_*`).
- **Client mirror** respects host **UI language** after peer init.

### Fixes and polish

- **Tauri CSP:** allow `blob:` for release WebView image rendering.
- **0.3.0 follow-ups:** image blob handling, file library modal, and delete confirmation flow.

### CI and quality

- **GitHub Actions:** `ci.yml` (pnpm, Node 20, `i18n:check`, typecheck); Playwright workflow aligned to **pnpm** + **Node 20** with `i18n:check` before tests; release workflow runs locale checks before build.
- **Playwright:** preview server uses `pnpm`; smoke tests verify `ar` / `ur` translation assets and basic RTL `dir` sanity.
- **Docs:** `ARCHITECTURE.md` — translation review checklist, RTL smoke checklist, i18n script summary.

### Native (Rust)

- Pre-webview **MessageBox** strings remain English; inline note defers optional localized OS dialogs to a future bridge.

### Documentation

- README updates from the community (synced on `master` before this release).

---

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
[0.3.1]: https://github.com/Mewnah/Sigil/releases/tag/v0.3.1
