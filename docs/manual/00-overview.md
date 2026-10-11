# Overview

This chapter covers **install and first run**, the **host UI layout**, and the **browser client mirror** (`/client`).

## Table of contents

- [Getting started](#getting-started)
- [UI overview](#ui-overview)
- [Client mirror](#client-mirror)

## Getting started

![Getting started](assets/dashboard.png)

### Prerequisites

- **Node.js** (LTS) and **pnpm**
- **Rust** (stable) for Tauri desktop builds
- **Windows** is the primary target; other platforms follow Tauri support

See the repository [README.md](../../README.md) for exact install commands.

### Run Sigil (desktop)

```bash
pnpm install
pnpm tauri dev
```

This starts the Vite dev server and the Tauri shell. The app listens on a local port (see the status area or logs if you need the number for clients or OBS).

### First launch and onboarding

A **setup wizard** (welcome, audio, STT, integrations, accessibility) is part of the app's code but is not shown in this version yet; you configure everything in **Settings** and each service inspector. (Once it is shown you can complete it step by step or use **Skip setup**.)

### Projects

Use **Project** in the sidebar to create, open, and save Sigil project files. Your **canvas**, **scenes**, **elements**, and related data live in the project; service **settings** are typically persisted separately (see **Settings** in [System](04-system.md)).

**Tip:** Save a project before long sessions so layout and assets are not lost.

### Environment variables (optional)

For Twitch, Kick, Google Fonts, and some cloud APIs, copy `.env.example` to `.env` and set only what you need. Never commit real secrets. Details: [README.md](../../README.md#environment-variables).

### Next steps

- [UI overview](#ui-overview) below
- [Client mirror](#client-mirror) if you use OBS or a second machine
- [Speech to text (STT)](01-audio.md#speech-to-text-stt) in **Audio and AI**

## UI overview

![Studio layout](assets/layout-overview.png)

Sigil **Studio** (host) is organized into a **header**, **left navigation**, optional **left inspector**, **canvas** (center), **right panel**, and **bottom status / controls**.

### Header

- **App menu / branding** — current section title.
- **Sidebar toggle** — collapse or expand the narrow navigation column.
- **Keyboard shortcuts** — opens a cheat sheet (matches [Shortcuts and tips](04-system.md#shortcuts-and-tips)).
- **Help** (book icon) — opens the in-app user manual (same content as `docs/manual`, synced to `/manual` at dev/build time).
- **Theme** — switch light/dark (stored with your settings).

### Left navigation

Groups:

1. **Main** — Dashboard, Project  
2. **Live services** — STT, TTS, Translation, AI Transform, Voice Changer  
3. **Studio** — Canvas and Elements, Files  
4. **Integrations** — Twitch, Kick, Discord, OBS, VRChat  
5. **Settings** — bottom of the list  

Choosing an item opens the matching **inspector** (left panel). Choosing the same item again closes the inspector.

### Canvas (center)

The editable **stage** for text and image elements, organized into **scenes**. Use the **canvas toolbar** for zoom, grid, undo/redo, alignment, history, and **presentation mode** (full-screen preview).

### Right panel

- With **no element selected:** default studio panel (logs / context depending on build).
- With an element selected: **Properties** for that text or image element.

### Bottom area

**Status** and quick links into service inspectors; **STT / AI / TTS** controls mirror start/stop for those pipelines (see [Shortcuts and tips](04-system.md#shortcuts-and-tips)).

### Floating / overlays

Depending on configuration you may see **TTS input**, **overlay typing**, **background input**, or **stats** (see shortcuts). These are documented on their respective service sections in [Audio and AI](01-audio.md) where relevant.

### Host vs client

The **host** runs all inspectors and native features. The **`/client`** page is a **mirror**: it syncs the document with the host and does **not** run Tauri plugins. See [Client mirror](#client-mirror).

## Client mirror

![Client mirror](assets/client-mirror.png)

### What it is

The **client** is a lightweight browser view that syncs with the **host** over **PeerJS** (and Yjs for the document). It is meant for **OBS Browser Source**, a second PC, or a tablet—anywhere you want the canvas without running the full desktop app.

### What it is not

- It does **not** run **Tauri** plugins (Whisper, Windows TTS, OSC from Rust, etc.).
- You **do not** configure STT/TTS backends here; do that on the **host** in the inspectors.

See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) and [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#remote-client).

### How to open it

The host serves the site on `http://127.0.0.1:<port>` (port from your run or Tauri config). The client URL includes query parameters for the host connection (for example host id, port, and peer id). When in doubt, copy the URL from **OBS** or **network** documentation in the host app if provided, or from your deployment notes.

**Tip:** OBS’s embedded browser can be an older Chromium; Sigil sets `data-sigil-client` on `/client` for CSS fallbacks. If something looks wrong, try a normal browser to compare.

### Typical workflow

1. Start **Sigil** on the streaming PC (host).  
2. Configure **STT**, **TTS**, and **canvas** on the host.  
3. Add a **Browser Source** in OBS pointing at the **client** or host-published URL as you prefer for layout.  
4. Keep the host running while streaming so sync and services stay alive.

### Troubleshooting

- **No sync:** confirm firewall allows local traffic; host and client must use matching port and peer settings.  
- **Stale OBS preview:** older CEF caches; the client path may unregister service workers—see code comments in `src/index.tsx` if issues persist after updates.
