# Client mirror (`/client`)

![Client mirror](assets/ui-placeholder.svg)

## What it is

The **client** is a lightweight browser view that syncs with the **host** over **PeerJS** (and Yjs for the document). It is meant for **OBS Browser Source**, a second PC, or a tablet—anywhere you want the canvas without running the full desktop app.

## What it is not

- It does **not** run **Tauri** plugins (Whisper, Windows TTS, OSC from Rust, etc.).
- You **do not** configure STT/TTS backends here; do that on the **host** in the inspectors.

See [ARCHITECTURE.md](../../ARCHITECTURE.md) and [BACKENDS.md](../BACKENDS.md#remote-client).

## How to open it

The host serves the site on `http://127.0.0.1:<port>` (port from your run or Tauri config). The client URL includes query parameters for the host connection (for example host id, port, and peer id). When in doubt, copy the URL from **OBS** or **network** documentation in the host app if provided, or from your deployment notes.

**Tip:** OBS’s embedded browser can be an older Chromium; Sigil sets `data-sigil-client` on `/client` for CSS fallbacks. If something looks wrong, try a normal browser to compare.

## Typical workflow

1. Start **Sigil** on the streaming PC (host).  
2. Configure **STT**, **TTS**, and **canvas** on the host.  
3. Add a **Browser Source** in OBS pointing at the **client** or host-published URL as you prefer for layout.  
4. Keep the host running while streaming so sync and services stay alive.

## Troubleshooting

- **No sync:** confirm firewall allows local traffic; host and client must use matching port and peer settings.  
- **Stale OBS preview:** older CEF caches; the client path may unregister service workers—see code comments in `src/index.tsx` if issues persist after updates.
