# Getting started

![Getting started](assets/ui-placeholder.svg)

## Prerequisites

- **Node.js** (LTS) and **pnpm**
- **Rust** (stable) for Tauri desktop builds
- **Windows** is the primary target; other platforms follow Tauri support

See the repository [README.md](../../README.md) for exact install commands.

## Run Sigil (desktop)

```bash
pnpm install
pnpm tauri dev
```

This starts the Vite dev server and the Tauri shell. The app listens on a local port (see the status area or logs if you need the number for clients or OBS).

## First launch and onboarding

On first run you may see the **setup wizard** (welcome, audio, STT, integrations, accessibility). You can complete it step by step or use **Skip setup**; you can configure everything later in **Settings** and each service inspector.

## Projects

Use **Project** in the sidebar to create, open, and save Sigil project files. Your **canvas**, **scenes**, **elements**, and related data live in the project; service **settings** are typically persisted separately (see **Settings**).

**Tip:** Save a project before long sessions so layout and assets are not lost.

## Environment variables (optional)

For Twitch, Kick, Google Fonts, and some cloud APIs, copy `.env.example` to `.env` and set only what you need. Never commit real secrets. Details: [README.md](../../README.md#environment-variables).

## Next steps

- [UI overview](ui-overview.md)
- [Client mirror](client.md) if you use OBS or a second machine
- [Speech to text (STT)](stt.md) to get captions running
