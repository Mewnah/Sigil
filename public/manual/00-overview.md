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

The first time Sigil starts on a PC (no settings file yet), a **setup wizard** opens over the app. It has five short steps: **welcome**, **microphone and speakers**, **Speech to Text** (the built-in engine, the language you speak and **Start captions when Sigil opens**, ticked by default), **where captions should go** (VRChat or OBS Studio, or Twitch, Kick or Discord), and **done**. Complete it step by step or use **Skip setup** on any step but the last; everything can be changed later in **Settings** and each inspector. If you quit in the middle it opens again at the next start. A profile that already exists never sees it. Captions do not start on that first launch: press **Start** in Speech to Text once, which asks before it downloads the speech models; with **Start captions when Sigil opens** ticked they start by themselves from the next launch. The window opens sized for your monitor (960 x 720, larger on a 4K monitor that Windows runs at 100%, where **Settings → Interface size** on **Auto** enlarges the menus too), and Sigil remembers the size you leave it at. A new install also starts with **LAN access** on **This PC only**: Sigil listens on this PC alone, so Windows shows no Firewall question and OBS on this PC works at once. A phone or OBS on another PC needs **Settings → LAN access → Other devices** switched to **Allowed with the LAN token**; Windows may ask once then ([how the links work](#how-to-open-it)).

### Projects

Use **Project** in the sidebar to create, open, and save Sigil project files. Your **canvas**, **scenes**, **elements**, and related data live in the project; service **settings** are typically persisted separately (see **Settings** in [System](04-system.md)).

**Tip:** Save a project before long sessions so layout and assets are not lost.

### Environment variables (optional)

For Twitch, Kick, Google Fonts, and some cloud APIs, copy `.env.example` to `.env` and set only what you need. Never commit real secrets. Details: [README.md](../../README.md#environment-variables).

### Next steps

- [UI overview](#ui-overview) below
- [Client mirror](#client-mirror) if you use OBS or a second machine
- [Speech to text (STT)](01-audio.md#speech-to-text-stt) in **Speech and voice**

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
2. **Speech** — Speech to Text (with its **Rewrite** tab), Translation, Text to Speech, Voice Changer, Other players  
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

**Status** and quick links into service inspectors; **STT / Rewrite / TTS** controls mirror start/stop for those pipelines (see [Shortcuts and tips](04-system.md#shortcuts-and-tips)).

### Floating / overlays

Depending on configuration you may see **TTS input**, **overlay typing**, **background input**, or **stats** (see shortcuts). These are documented on their respective service sections in [Speech and voice](01-audio.md) where relevant.

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

The host serves the page at `http://localhost:<port>/client` (port **3030** unless you started Sigil with `--port`). The link has no host, id or peer parameters; the page connects back to the Sigil that served it. Copy it with **OBS Studio → Copy browser source link**: that is the right address for OBS on this PC, and it keeps working after Sigil restarts.

For a phone, a tablet or OBS on **another PC**, copy the **LAN link** instead (OBS Studio inspector, or **Settings → LAN access**). It ends in `?token=…`, the LAN token that other devices need; links for this PC never carry one. A new install starts on **This PC only**, where Sigil listens on this PC alone and the OBS Studio inspector shows no LAN link: set **Settings → LAN access → Other devices** to **Allowed with the LAN token** first (Windows may ask once to allow Sigil on the network; a profile that already existed keeps the setting it had). Change the token or switch back to **This PC only** and the old LAN links are refused until you copy them again.

**Tip:** OBS’s embedded browser can be an older Chromium; Sigil sets `data-sigil-client` on `/client` for CSS fallbacks. If something looks wrong, try a normal browser to compare.

### Typical workflow

1. Start **Sigil** on the streaming PC (host).
2. Configure **STT**, **TTS**, and **canvas** on the host.
3. Add a **Browser Source** in OBS and paste the copied link. Starting Sigil first is good practice, but the order does not matter: a source that loads before Sigil is listening keeps showing "Connecting to Sigil…" and connects as soon as Sigil is up.
4. Keep the host running while streaming. Sigil can stay minimized.

### If the connection drops

The source reconnects **by itself, for as long as it lives**. What you see on stream while it does:

| You see | What it means | What to do |
|---|---|---|
| **Connecting to Sigil…** | The source is loading or has not reached Sigil yet. After three failed tries it also names the address it is waiting for and the last error. | Nothing. It connects as soon as Sigil answers. |
| Yellow bar **Lost connection to Sigil — reconnecting.** | Sigil restarted, slept or the network dropped. The source tries again 1, 2, 4, 8 and 16 s apart, then every 30 s, with no limit. | Nothing. The bar goes away and captions resume by themselves, on the scene Sigil is showing now. |
| Yellow bar **Disconnected from Sigil…** | Sigil refused this source: the LAN token changed, or Sigil accepts only its own PC and the source runs elsewhere. It asks again every minute. | Copy the link again (the LAN link for another PC) and paste it into the source. |
| The dark page **Application failed to start** | The source can never work as set up: Sigil refused the link when it loaded, the browser has no WebRTC, or the address is wrong. This page does not retry on its own. | Fix the link or the setting it names, then use **Refresh cache of current page** (below). |
| Nothing at all | The active scene has no element that shows captions (see [Canvas and elements](02-studio.md#canvas-and-elements)), or the page crashed. A crashed page reloads itself after 4 s (twice as long for each further reload within 10 minutes, at most 2 minutes), once Sigil answers. | Add a Text element to the scene. Otherwise wait a few seconds. |

In an ordinary browser tab (a phone, a second PC) a start-up failure that may pass also shows the dark panel, and it says when the page reloads itself. In OBS the same failure stays transparent and the page reloads itself.

**When "Refresh cache of current page" is still needed.** In the browser source's properties, only in two cases: after the dark **Application failed to start** page, once you have fixed what it names; and once after you update Sigil, so the source loads the new page (an open source keeps running the page it loaded). It is not needed after a Sigil restart or a dropped connection.

**Shutdown source when not visible** (OBS) is optional. It saves CPU while the scene is hidden; the page reloads and reconnects by itself when the scene is shown again, so the first seconds of the scene may be without captions. If captions lag right after a scene switch, turn it off. It is not a workaround for a stuck source: a source that lost Sigil already reconnects or reloads itself.

### Troubleshooting

- **No sync:** check that Sigil is running and that the port in the link is the one Sigil shows (**OBS Studio → Copy browser source link** fills it in). The page for this PC needs no token; a page on another PC (or a phone) needs the LAN link, and a new install only gives one after **Settings → LAN access → Other devices** is set to **Allowed with the LAN token**. See [If the connection drops](#if-the-connection-drops) for what each message means.
- **Stale OBS preview:** Sigil removes any service worker it finds on `/client`, so the only cache is OBS's own: use **Refresh cache of current page** in the source's properties.
