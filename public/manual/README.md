# Sigil user manual

This manual describes **Sigil Studio** (the desktop host) and the **client mirror** used in OBS or other browsers. It is written for streamers and VRChat creators using captions, TTS, chat, and canvas layouts.

## When to update this manual

If you change an inspector flow, default behavior, or a user-visible label, update the matching page here (or the screenshot list under `assets/SHOTLIST.md`) in the same pull request when practical.

## Related technical docs

| Document | Contents |
|----------|----------|
| [README.md](../../README.md) (repo root) | Install, build, environment variables, feature list |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | Host vs client, sync, IPC |
| [BACKENDS.md](../BACKENDS.md) | STT/TTS engines, minimum configuration, remote client notes |
| [RELEASE_BUILD_SMOKE.md](../RELEASE_BUILD_SMOKE.md) | Pre-release build checks and manual smoke matrix |

## Manual map

### Start here

- [Getting started](getting-started.md) — install, first run, onboarding, projects
- [UI overview](ui-overview.md) — layout, panels, host vs client
- [Client mirror](client.md) — `/client` limitations and setup

### Live services

- [Speech to text (STT)](stt.md)
- [Text to speech (TTS)](tts.md)
- [Translation](translation.md)
- [AI transform](transform.md)
- [Voice changer](voice-changer.md)

### Studio

- [Canvas and elements](canvas.md)
- [Files](files.md)

### Integrations

- [OAuth, redirects, and ports](oauth-and-ports.md) — loopback vs dev vs packaged app URLs
- [Twitch](twitch.md)
- [Kick](kick.md)
- [Discord](discord.md)
- [OBS Studio](obs.md)
- [VRChat](vrc.md)

### System

- [Project](project.md)
- [Settings](settings.md)
- [Shortcuts and tips](shortcuts-and-tips.md)

## Screenshots

Pages reference images under `assets/`. Until you drop in real captures, a shared placeholder is used. See [assets/SHOTLIST.md](assets/SHOTLIST.md) for the recommended shot list and capture settings.
