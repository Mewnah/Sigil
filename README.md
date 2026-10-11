# Sigil

**Voice, captions, and stream tooling for VRChat and live streaming.**

Sigil is an open-source desktop app (Tauri + React) that turns speech into text, text into speech, and both into something your viewers can actually read. Wire up STT, TTS, translation, chat, and OBS in one place instead of juggling a handful of half-connected tools.

Configure the heavy lifting once on the **host**, then drive captions from OBS, another machine, or a slim browser **client**—everything stays synced over the network when you want it distributed.

## Features

There is a lot under the hood; mix local models, cloud APIs, and browser fallbacks however your setup prefers.

### Speech-to-text (STT)

- **On this PC (automatic)** — the default for new desktop installs: Sigil picks local models for your language and PC (live partials from Nemotron streaming, a more accurate final per sentence from Parakeet, Qwen3-ASR, SenseVoice or Whisper), asks before it downloads anything and runs them inside the app, on the processor by default so the game keeps the GPU. Opt-in GPU models (Qwen3-ASR 1.7B, Cohere Transcribe, Voxtral) need a runtime pack.
- **Whisper** (Rust plugin) — local inference; models downloaded on first use through the model store
- **Vosk** — local models via Tauri + browser loader
- **OpenAI-style HTTP** — `POST …/audio/transcriptions` for local servers (speaches and others), plus a **Groq** free-cloud preset that stays inside Groq's limits
- **Moonshine** — an external Moonshine HTTP server (legacy backend); Moonshine base also exists inside Sigil's speech engine
- **Azure** / **Deepgram** / **Gemini Live** — cloud APIs where configured
- **Native / Web Speech** — browser speech recognition where available (not inside the desktop app)

STT is configured on the **desktop host**. The lightweight **`/client`** page does not run Tauri; it syncs with the host over PeerJS (see [ARCHITECTURE.md](ARCHITECTURE.md)). A Silero voice detector keeps silence and keyboard noise away from every engine in Sigil's own speech pipeline (automatic, Whisper, Moonshine, local HTTP, Groq; the cloud engines, Vosk and Web Speech do not use it). [Which engine for which PC](docs/OPERATOR_REFERENCE.md#choosing-engines-for-your-pc).

### Text-to-speech (TTS)

- **Kokoro / Piper / Kitten (local)** — in-process neural voices on the processor, the default for new desktop installs
- **Edge** (keyless) and **Gemini TTS** (your free key) — cloud voices that fall back to the local Kokoro voice within about a second when the service fails
- **Windows** — system SAPI voices, played through Sigil's output like every other voice
- **Azure**, **Uberduck**, **VOICEVOX** / **AivisSpeech**, **Style-Bert-VITS2** (`.aivmx` voices, in process), **GPT-SoVITS**
- **OpenAI-compatible HTTP** — presets for Kokoro-FastAPI, Chatterbox, vLLM-Omni voices and qwentts, plus **Fish Speech** and your own services (example: [scripts/neural-sidecar/](scripts/neural-sidecar/))
- **Cloning and GPU voices (opt-in)** — Pocket TTS and Chatterbox Multilingual in process, Qwen3-TTS, and experimental Magpie, Breeze, OmniVoice and Fish S2 Pro helpers
- **ONNX (local)** — in-process neural speech from a **Hugging Face** repo id or a local **speech package** folder (`speech_package_v1`; see [docs/OPERATOR_REFERENCE.md](docs/OPERATOR_REFERENCE.md#local-speech-package-onnx)); always runs on the CPU

Speech plays sentence by sentence, with Skip and Stop, and can be **routed into the voice changer** (or both outputs).

### Translation

- **Automatic (best available)** — an ordered chain with fallbacks: a local AI model (Hy-MT2 on a `llama-server` helper that Sigil starts and stops itself) → Firefox Translations (Bergamot, no video memory) → NLLB-200
- **DeepL API Free**, **Gemini** (free tier), **Azure**, **LibreTranslate** — cloud or self-hosted
- **On-device** — Chromium **WebView Translator / Language Detector** APIs when the runtime exposes them

Partial captions translate without flicker (LocalAgreement), and final translations are never dropped.

### AI transform

Rewrite or style transcribed text with **OpenAI**, **OpenRouter**, **Google Gemini** (free tier) or **local** LLM endpoints (configured in the inspector).

### Streaming and chat

- **Twitch** — OAuth (public or confidential app), chat, BTTV / FFZ / 7TV emotes
- **Kick** — OAuth and chat
- **Discord** — outbound webhooks for captions or announcements
- **OBS Studio** — browser source URLs; optional **obs-websocket** automation from the app

### VRChat and audio

- **OSC** — captions and related control from the **Tauri** build (not from the in-browser preview alone): paced chatbox messages with numbered pages, mute sync with VRChat (captions stop when you mute in VRChat), and avatar parameters for emotion and sound tags
- **Other players' captions** — transcribe what others say in VRChat (only the game's sound on Windows 11), kept apart from your own captions and optionally translated
- **Caption window and SteamVR overlay** — an always-on-top click-through caption window (pinnable in XSOverlay / OVR Toolkit) and a native SteamVR overlay; both off until you turn them on
- **Voice changer (Sigil Voice Engine)** — import the RVC voices you already have (`.pth` / `.zip`, with their `.index`) and run them on Sigil's own real-time engine on the processor (**Streamer** preset), CUDA or DirectML, with a stale-audio latency budget and an autotuner; zero-shot (Sigil-ZS), singing (DDSP-SVC) and experimental voice families run on the same engine. The older modes remain: **built-in** phase vocoder, **WebSocket** sidecar (binary PCM) and **local ONNX**. See [docs/OPERATOR_REFERENCE.md](docs/OPERATOR_REFERENCE.md#voice-changer) and [scripts/neural-sidecar/README.md](scripts/neural-sidecar/README.md)
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
pnpm tauri build    # installers under src-tauri/target/release/bundle/
pnpm tauri:release  # release build that also stages the GPU (CUDA/DirectML) provider DLLs
pnpm test:unit      # vitest unit tests
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

Consolidated guides (fewer files, same depth):

- [docs/manual/README.md](docs/manual/README.md) — **user manual** (chapters under `docs/manual/`); **`pnpm dev`** / **`pnpm build`** run **`scripts/sync-manual.mjs`** so **`public/manual/`** and **`public/docs/`** stay in sync for the in-app **Help** reader (**book** icon in the header)
- [docs/OPERATOR_REFERENCE.md](docs/OPERATOR_REFERENCE.md) — choosing engines for your PC, STT/TTS backends, STT reliability, voice changer, streamer features, local speech package
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) — architecture, host services, the `--headless` commands, the speech benchmark, connectivity audit, `src/client` rules
- [docs/RELEASE_AND_ENGINEERING.md](docs/RELEASE_AND_ENGINEERING.md) — release smoke tests, optional VR / two-PC checks, exit matrices, threat model, dependency audits, performance, upgrade backlog
- [ARCHITECTURE.md](ARCHITECTURE.md) — short pointer to the developer guide
- [docs/DOCUMENTATION_GAPS_AND_RECOMMENDATIONS.md](docs/DOCUMENTATION_GAPS_AND_RECOMMENDATIONS.md) — resolved follow-ups and future backlog; manual screenshots use `scripts/write-manual-placeholder-pngs.mjs` until real captures replace them
- **i18n:** `pnpm i18n:check` / `pnpm i18n:sync`; new UI strings from parallel work are staged in `scripts/i18n-staging/<WS-ID>.json` and merged with `node scripts/merge-i18n-staging.mjs` (the older `merge-http-tts-i18n.mjs` / `merge-neural-i18n.mjs` scripts are retired). See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#plugin-commands-capabilities-and-i18n-staging).

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
