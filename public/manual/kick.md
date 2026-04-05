# Kick

![Kick inspector](assets/ui-placeholder.svg)

## Purpose

Connect **Kick** chat via OAuth for streaming workflows alongside Sigil’s canvas and captions.

## Open the inspector

Sidebar → **Kick**.

## Setup (high level)

1. Register an application with Kick and set `SIGIL_KICK_CLIENT_ID` / `SIGIL_KICK_CLIENT_SECRET` in `.env` as documented in [README.md](../../README.md).  
2. Register redirect URLs: at minimum `http://localhost:17890/oauth/kick/callback`, plus dev/packaged webview URLs as described in [OAuth, redirects, and ports](oauth-and-ports.md).  
3. Complete OAuth in the inspector.  
4. Set channel and options per UI fields.

## Tips

- Keep secrets out of git and public builds.  
- If OAuth fails, verify redirect URLs (loopback vs packaged `https` origin) and that the Kick app is in the correct environment (dev vs prod).

## Related

- [OAuth, redirects, and ports](oauth-and-ports.md)  
- [Twitch](twitch.md)  
- [OBS](obs.md)  
