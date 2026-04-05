# OAuth, redirects, and networking ports

This page explains how **desktop loopback**, **dev server**, and **packaged app** URLs differ so Twitch, Kick, and other OAuth setups work in both development and release builds.

## Ports (three different roles)

| Port | Role |
|------|------|
| **3030** (default) | Embedded HTTP server inside Sigil: PeerJS signaling, pubsub WebSocket (e.g. Chrome `mic.html`), mirrored UI assets, `/client` browser entry. Override with the app’s `--port` CLI argument when starting the binary. |
| **17890** (fixed) | Short-lived **OAuth loopback** server for Twitch/Kick when the system browser completes login. Register redirect URLs that use **`http://localhost:17890`** for these flows (not the same as 3030). |
| **1420** | **Vite dev server only** (`pnpm dev`). Not used by the installed app. |

If a second Sigil instance cannot bind its main port, the app shows a warning: free the port or pass a different `--port`.

## Packaged app vs `pnpm dev` (redirect URLs)

During **`pnpm dev`**, in-app or embedded-browser OAuth helpers often use `http://localhost:1420/...` callback paths because that is where Vite serves the UI.

In a **release (Tauri) build**, the main window uses a **secure custom-protocol origin** (`useHttpsScheme` in `tauri.conf.json`). The effective redirect base is `window.location.origin` at runtime (for example an `https://` URL on a `*.localhost` host), **not** `http://localhost:1420`.

**What to register in Twitch / Kick developer consoles**

1. **Desktop OAuth (browser → loopback)** — always include:  
   - `http://localhost:17890/oauth/twitch/implicit` (Twitch public / implicit)  
   - `http://localhost:17890/oauth/twitch/callback` (Twitch confidential / code)  
   - `http://localhost:17890/oauth/kick/callback` (Kick)  

2. **Flows that redirect into the app webview** — also register the **exact** HTTPS origin of the packaged app plus the path (`/oauth/twitch/implicit`, `/oauth/twitch/callback`, `/oauth/kick/callback`, etc.). To read that origin once: open the **installed** app, enable devtools if you use them, and run `window.location.origin` in the console; or set explicit overrides (below).

3. **Local development** — keep `http://localhost:1420/oauth/...` entries if you test OAuth against Vite.

## Environment overrides (build-time)

Optional variables pin redirect URIs when the defaults are wrong for your Twitch/Kick app registration. They are read at **frontend build time** (Vite). Prefer the `SIGIL_*` names; `CURSES_*` still works as a legacy alias.

| Variable | Use |
|----------|-----|
| `SIGIL_TWITCH_IMPLICIT_REDIRECT_LOCAL` | Full URL for Twitch implicit return in the webview (public apps). |
| `SIGIL_TWITCH_CLIENT_REDIRECT_LOCAL` | Full URL for Twitch OAuth code callback in the webview (confidential apps). |
| `SIGIL_KICK_REDIRECT_LOCAL` | Full URL for Kick OAuth callback in the webview. |

See [.env.example](../../.env.example) in the repo root for copy-paste templates.

## Related

- [Twitch](twitch.md)  
- [Kick](kick.md)  
- [Getting started](getting-started.md)  
- [Manual Tauri smoke checklist](../MANUAL-TAURI-CHECKLIST.md)  
