# Twitch

![Twitch inspector](assets/ui-placeholder.svg)

## Purpose

Connect **Twitch** chat (and related features) to Sigil: OAuth login, chat display, and **BTTV / FFZ / 7TV** emote support where implemented.

## Open the inspector

Sidebar → **Twitch**.

## Setup (high level)

1. Create a Twitch developer application if you use custom **Client ID** (see [README.md](../../README.md) env vars `SIGIL_TWITCH_CLIENT_ID` and optional secret for confidential apps).  
2. Register redirect URLs for **loopback** (`http://localhost:17890/...`), **Vite dev** (`http://localhost:1420/...` if needed), and the **packaged app** HTTPS origin — see [OAuth, redirects, and ports](oauth-and-ports.md).  
3. In the inspector, run **Connect** / OAuth flow and approve scopes.  
4. Configure **channel** and chat-related options shown in the UI.

## Tips

- Redirect URLs must match what you registered in the Twitch console and optional `SIGIL_TWITCH_*_REDIRECT_LOCAL` build-time overrides in `.env`.  
- Never share **client secret** or committed `.env` with real credentials.  
- If emotes fail to load, check network and third-party emote CDN status.

## Related

- [OAuth, redirects, and ports](oauth-and-ports.md)  
- [OBS](obs.md)  
- [Discord](discord.md)  
- [Kick](kick.md)  
