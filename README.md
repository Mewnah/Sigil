# Sigil

**Voice, captions, and stream tooling for VRChat and live streaming.**

Sigil is an open-source desktop app (Tauri + React) that turns speech into text, text into speech, and both into something your viewers can actually read. Wire up STT, TTS, translation, chat, and OBS in one place instead of juggling a handful of half-connected tools.

Configure the heavy lifting once on the **host**, then drive captions from OBS, another machine, or a slim browser **client**—everything stays synced over the network when you want it distributed.

## Features

There is a lot under the hood; mix local models, cloud APIs, and browser fallbacks however your setup prefers.

### Speech-to-text (STT)

- **Whisper** (Rust plugin) — local inference; models downloaded on first use
- **Vosk** — local models via Tauri + browser loader
- **Moonshine** — HTTP server integration (e.g. Useful Sensors container)
- **OpenAI-style HTTP** — `POST …/audio/transcriptions` for local sidecars (Parakeet, speaches, etc.)
- **Azure** / **Deepgram** — cloud APIs where configured
- **Native / Web Speech** — browser speech recognition where available

STT is configured on the **desktop host**. The lightweight **`/client`** page does not run Tauri; it syncs with the host over PeerJS (see [ARCHITECTURE.md](ARCHITECTURE.md)).

### Text-to-speech (TTS)

- **Windows** — system SAPI voices (Rust plugin)
- **Azure** — cloud voices
- **Uberduck** — API-backed voices (Rust plugin)
- **VOICEVOX** — HTTP to a local VOICEVOX engine
- **Native / Web Speech** — browser synthesis where available

### AI transform

Rewrite or style transcribed text with **OpenAI**, **OpenRouter**, or **local** LLM endpoints (configured in the inspector).

### Streaming and chat

- **Twitch** — OAuth (public or confidential app), chat, BTTV / FFZ / 7TV emotes
- **Kick** — OAuth and chat
- **Discord** — outbound webhooks for captions or announcements
- **OBS Studio** — browser source URLs; optional **obs-websocket** automation from the app

### VRChat and audio

- **OSC** — captions and related control from the **Tauri** build (not from the in-browser preview alone)
- **Voice changer** — Tauri plugin where enabled
- **Sound effects** — host-side playback for SFX tied to the canvas

### Editor and collaboration

- **Canvas** — text (and image) elements, scenes, fonts, styling
- **Host / client** — Yjs + PeerJS document sync; optional pubsub / link modes (see architecture doc)
- **System logs** — in-app log panel for service diagnostics
- Legacy **audio visualizer** elements from older projects are stripped on load; see [src/client/migrate/legacyDocument.ts](src/client/migrate/legacyDocument.ts)

## Getting started

### Prerequisites

- [Node.js](https://nodejs.org/) (LTS recommended) and [pnpm](https://pnpm.io/)
- [Rust](https://rustup.rs/) (stable) for Tauri
- Windows is the primary target; other platforms follow Tauri support

### Run and build

```bash
pnpm install
pnpm tauri dev      # desktop app + Vite
pnpm tauri build    # release artifacts under src-tauri/target/
```

For a static web build only (no Tauri): `pnpm build` then `pnpm preview`.

### Environment variables

Copy [.env.example](.env.example) to `.env` and fill in only what you use (Twitch, Kick, Google Fonts API, etc.). **Never commit `.env`.**

```env
# See .env.example for full list and redirect URLs.
SIGIL_TWITCH_CLIENT_ID=your_client_id
# SIGIL_TWITCH_CLIENT_SECRET=your_client_secret   # confidential Twitch apps only
SIGIL_KICK_CLIENT_ID=your_client_id
SIGIL_KICK_CLIENT_SECRET=your_client_secret
```

- **`pnpm build` / `pnpm tauri build`** can embed `SIGIL_*`, legacy `CURSES_*`, and `VITE_*` from your environment into the frontend bundle. Do not upload a local `dist/` zip to public issues if it was built with real secrets.
- **GitHub Actions:** the `release` workflow passes `SIGIL_TWITCH_CLIENT_ID` from the repository secret of the same name (or legacy `CURSES_TWITCH_CLIENT_ID`).

### Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — stack, host vs client, IPC, pubsub
- [docs/BACKENDS.md](docs/BACKENDS.md) — STT/TTS backends and minimum configuration
- [docs/MANUAL-TAURI-CHECKLIST.md](docs/MANUAL-TAURI-CHECKLIST.md) — manual QA checklist for Tauri builds

## Development

Sanity checks before you open a PR (or when something feels off):

```bash
pnpm typecheck
pnpm test:e2e          # Playwright (build + preview smoke)
```

## License

Sigil is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE.md](LICENSE.md).

## Credits

**Sigil** began as a modern evolution of [curses](https://github.com/mmpneo/curses). **Please consider supporting the developers of the original curses application!** None of this would have been possible without their work.

### ☕ Support the development of Sigil
If Sigil has made your workflow smoother, saved you from "cloud-tax" via local models, or you just enjoy the new features, consider supporting my work! Your contributions help me dedicate time to new features, localizations, and keeping the engine running.

[![Ko-fi](https://img.shields.io/badge/Support%20Me%20on%20Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/mewnah)

*Every coffee helps fuel the next update. Thank you for being part of the Sigil community!*
