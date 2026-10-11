# Integrations

This chapter covers **OAuth and ports**, **Twitch**, **Kick**, **Discord**, **OBS Studio**, and **VRChat**.

## Table of contents

- [OAuth, redirects, and ports](#oauth-redirects-and-ports)
- [Twitch](#twitch)
- [Kick](#kick)
- [Discord](#discord)
- [OBS Studio](#obs-studio)
- [VRChat](#vrchat)

## OAuth, redirects, and ports

This section explains how **desktop loopback**, **dev server**, and **packaged app** URLs differ so Twitch, Kick, and other OAuth setups work in both development and release builds.

### Ports (three different roles)

| Port | Role |
|------|------|
| **3030** (default) | Embedded HTTP server inside Sigil: PeerJS signaling, pubsub WebSocket (e.g. Chrome `mic.html`), mirrored UI assets, `/client` browser entry. Override with the app’s `--port` CLI argument when starting the binary. |
| **17890** (fixed) | Short-lived **OAuth loopback** server for Twitch/Kick when the system browser completes login. Register redirect URLs that use **`http://localhost:17890`** for these flows (not the same as 3030). |
| **1420** | **Vite dev server only** (`pnpm dev`). Not used by the installed app. |

If a second Sigil instance cannot bind its main port, the app shows a warning: free the port or pass a different `--port`.

### Packaged app vs `pnpm dev` (redirect URLs)

During **`pnpm dev`**, in-app or embedded-browser OAuth helpers often use `http://localhost:1420/...` callback paths because that is where Vite serves the UI.

In a **release (Tauri) build**, the main window uses a **secure custom-protocol origin** (`useHttpsScheme` in `tauri.conf.json`). The effective redirect base is `window.location.origin` at runtime (for example an `https://` URL on a `*.localhost` host), **not** `http://localhost:1420`.

**What to register in Twitch / Kick developer consoles**

1. **Desktop OAuth (browser → loopback)** — always include:  
   - `http://localhost:17890/oauth/twitch/implicit` (Twitch public / implicit)  
   - `http://localhost:17890/oauth/twitch/callback` (Twitch confidential / code)  
   - `http://localhost:17890/oauth/kick/callback` (Kick)  

2. **Flows that redirect into the app webview** — also register the **exact** HTTPS origin of the packaged app plus the path (`/oauth/twitch/implicit`, `/oauth/twitch/callback`, `/oauth/kick/callback`, etc.). To read that origin once: open the **installed** app, enable devtools if you use them, and run `window.location.origin` in the console; or set explicit overrides (below).

3. **Local development** — keep `http://localhost:1420/oauth/...` entries if you test OAuth against Vite.

### Environment overrides (build-time)

Optional variables pin redirect URIs when the defaults are wrong for your Twitch/Kick app registration. They are read at **frontend build time** (Vite). Prefer the `SIGIL_*` names; `CURSES_*` still works as a legacy alias.

| Variable | Use |
|----------|-----|
| `SIGIL_TWITCH_IMPLICIT_REDIRECT_LOCAL` | Full URL for Twitch implicit return in the webview (public apps). |
| `SIGIL_TWITCH_CLIENT_REDIRECT_LOCAL` | Full URL for Twitch OAuth code callback in the webview (confidential apps). |
| `SIGIL_KICK_REDIRECT_LOCAL` | Full URL for Kick OAuth callback in the webview. |

See `.env.example` in the repo root for copy-paste templates.

### Related

- [Twitch](#twitch)  
- [Kick](#kick)  
- [Getting started](00-overview.md#getting-started)  
- Manual Tauri smoke checklist: [RELEASE_AND_ENGINEERING.md](../RELEASE_AND_ENGINEERING.md#manual-tauri-checklist)  

## Twitch

![Twitch inspector](assets/twitch-inspector.png)

### Purpose

Connect **Twitch** chat (and related features) to Sigil: OAuth login, chat display, and **BTTV / FFZ / 7TV** emote support where implemented.

### Open the inspector

Sidebar → **Twitch**.

### Setup (high level)

1. Create a Twitch developer application if you use custom **Client ID** (see [README.md](../../README.md) env vars `SIGIL_TWITCH_CLIENT_ID` and optional secret for confidential apps).  
2. Register redirect URLs for **loopback** (`http://localhost:17890/...`), **Vite dev** (`http://localhost:1420/...` if needed), and the **packaged app** HTTPS origin — see [OAuth, redirects, and ports](#oauth-redirects-and-ports).  
3. In the inspector, run **Connect** / OAuth flow and approve scopes.  
4. Configure **channel** and chat-related options shown in the UI.

### Tips

- Redirect URLs must match what you registered in the Twitch console and optional `SIGIL_TWITCH_*_REDIRECT_LOCAL` build-time overrides in `.env`.  
- Never share **client secret** or committed `.env` with real credentials.  
- If emotes fail to load, check network and third-party emote CDN status.

### Related

- [OAuth, redirects, and ports](#oauth-redirects-and-ports)  
- [OBS Studio](#obs-studio)  
- [Discord](#discord)  
- [Kick](#kick)  

## Kick

![Kick inspector](assets/kick-inspector.png)

### Purpose

Connect **Kick** chat via OAuth for streaming workflows alongside Sigil’s canvas and captions.

### Open the inspector

Sidebar → **Kick**.

### Setup (high level)

1. Register an application with Kick and set `SIGIL_KICK_CLIENT_ID` / `SIGIL_KICK_CLIENT_SECRET` in `.env` as documented in [README.md](../../README.md).  
2. Register redirect URLs: at minimum `http://localhost:17890/oauth/kick/callback`, plus dev/packaged webview URLs as described in [OAuth, redirects, and ports](#oauth-redirects-and-ports).  
3. Complete OAuth in the inspector.  
4. Set channel and options per UI fields.

### Tips

- Keep secrets out of git and public builds.  
- If OAuth fails, verify redirect URLs (loopback vs packaged `https` origin) and that the Kick app is in the correct environment (dev vs prod).

### Related

- [OAuth, redirects, and ports](#oauth-redirects-and-ports)  
- [Twitch](#twitch)  
- [OBS Studio](#obs-studio)  

## Discord

![Discord inspector](assets/discord-inspector.png)

### Purpose

Send **outbound webhooks** to Discord (for example captions, announcements, or bot channels), configured from the host. This is unrelated to **VRChat** (which uses OSC from the desktop app)—separate settings and transport.

### Open the inspector

Sidebar → **Discord**.

### Setup (high level)

1. Create a **Webhook URL** in your Discord server channel settings (Integrations → Webhooks). Use only `https://discord.com/api/webhooks/...` URLs.
2. Paste it into the Sigil inspector (treat it like a password).
3. Use **Send test message** to verify the webhook before relying on STT.
4. For **forum** or **media** channels, set either an existing **thread ID** (query) or a **new thread name** so Discord accepts the message—see Discord’s Execute Webhook docs.

### Tips

- Anyone with the webhook URL can post to that channel—rotate it if leaked.
- Discord limits message `content` length (2000 characters); Sigil splits longer finals into multiple webhook posts with a short delay.
- **Post only when live** uses Twitch or Kick live status (either platform counts as live).
- Rate limits apply per Discord rules; the app retries briefly on HTTP 429.

### Related

- [Twitch](#twitch)
- [UI overview](00-overview.md#ui-overview) — dock may link here

## OBS Studio

![OBS inspector](assets/obs-inspector.png)

### Purpose

Copy **Browser Source** URLs for Sigil’s output and optionally drive OBS via **obs-websocket** (create sources, switch scenes) from the app.

### Open the inspector

Sidebar → **OBS Studio**.

### Browser source (most common)

1. Click **Copy browser source link** (or equivalent).  
2. In OBS, add a **Browser** source and paste the URL.  
3. Set width/height to match your canvas (often 1920×1080 or your base resolution).

### WebSocket automation

1. Enable **obs-websocket** in OBS and set **port** and **password**.  
2. In Sigil, connect with matching credentials.  
3. Use any **setup** or automation controls (for example “Sigil browser source” creation) from the inspector dropdowns.

### Tips

- Browser source **FPS** and **shutdown when not visible** affect CPU; tune per scene.
- If the preview is blank, confirm the **host** is running and the URL uses the correct **port**.
- Use the `http://localhost:<port>/…` URL Sigil gives you. A **Local file** browser source (a saved HTML file) is refused by Sigil's connection check; see **Settings → Local web server**.

### Related

- [Canvas and elements](02-studio.md#canvas-and-elements)  
- [Client mirror](00-overview.md#client-mirror)  

## VRChat

![VRChat inspector](assets/vrc-inspector.png)

### Purpose

Send **OSC** messages for captions and related control into **VRChat** from the **Tauri** host, follow your mute in VRChat, caption other players, and show captions in VR. OSC is **not** available from the in-browser preview alone—use the desktop app. All of these extras are off until you turn them on.

### Open the inspector

Sidebar → **VRChat**.

### Setup (high level)

1. Enable OSC in **VRChat** settings and note the receive port.  
2. In Sigil, match host/port and options in the inspector.  
3. Test with short phrases from STT or manual send paths if exposed.

### Chatbox, mute sync and tags

- **Chatbox pacing:** messages are queued at least **Time between chatbox messages** apart (default 1.5 s) because VRChat drops faster ones. **Split long text into pages** sends text over 144 characters as numbered pages instead of cutting it.
- **Mute captions when I mute in VRChat:** captions stop the moment you mute in VRChat and resume when you unmute. Sigil uses OSCQuery by default, so there is no port to set up. The first start may show a Windows Firewall prompt.
- **Emotion and sound tags** (with the SenseVoice model) can drive two avatar parameters and optional emoji in the chatbox.
- **Wait until I finish speaking** (needs Smart Turn in speech-to-text) holds a sentence you leave unfinished and sends it together with the rest.

Details: [VRChat chatbox, mute sync and tags](../OPERATOR_REFERENCE.md#vrchat-chatbox-mute-sync-and-tags).

### Other players' captions

The **Other players** tab captions what other players say (by default only VRChat's sound, on Windows 11), optionally translated into your language. They stay apart from your own captions: never in the chatbox, text to speech or history. Show them with a text element, the caption window or the VR overlay and the source **Other players: speech** or **Other players: translation**. Keep the hardware tier on **Processor** so the game stays smooth. Details: [Other players' captions](../OPERATOR_REFERENCE.md#other-players-captions).

### Caption window and SteamVR overlay

- **Caption window:** an always-on-top, click-through window that never takes focus from a game; pin it in XSOverlay or OVR Toolkit by its title "Sigil captions". Turn click-through off to move or resize it.
- **SteamVR overlay:** captions on your head, a hand or in the world. It needs the OpenVR runtime (**Models & Runtimes**); Sigil never starts SteamVR.

Details: [Caption window and SteamVR overlay](../OPERATOR_REFERENCE.md#caption-window-and-steamvr-overlay-vrchat-tab). A light setup for all of it: [VRChat streamer quick start](../OPERATOR_REFERENCE.md#vrchat-streamer-quick-start).

### Tips

- Firewall rules must allow **UDP** loopback or LAN as your setup requires.  
- Latency stacks: STT → (optional transform) → OSC → VRChat; keep STT local if possible.
- **Chatbox** mode: you can disable the per-line **notification sound** in the inspector (VRChat OSC third argument on `/chatbox/input`; see VRChat’s OSC docs).

### Related

- [Speech to text (STT)](01-audio.md#speech-to-text-stt)  
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — native IPC  
