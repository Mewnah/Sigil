# Manual Tauri smoke checklist

Automated Playwright tests use Vite preview only; native Tauri behavior requires a manual pass before release.

## Host (desktop app)

- [ ] `npm run tauri dev` — app window opens, no startup error overlay in `#root`.
- [ ] **Config / network** — `plugin:web|config` loads; status/footer shows expected host/port.
- [ ] **Whisper** — download or load model path; start recording; partial transcript event fires.
- [ ] **One STT path** — e.g. native or engine you ship; Start/Stop from bottom bar or inspector.
- [ ] **One TTS path** — voice selected; Start; speak or trigger text event.
- [ ] **Twitch** — `SIGIL_TWITCH_CLIENT_ID` set; console includes the right loopback URL: **Public** → `http://localhost:17890/oauth/twitch/implicit` (and `…/1420/…/implicit` if browser host); **Confidential** → `http://localhost:17890/oauth/twitch/callback`; OAuth completes; optional: IRC chat + post/receive.
- [ ] **Discord** — webhook URL; one post with “post only when live” on and off if you use that gate.
- [ ] **OBS** — WebSocket connect; captions only send when “Enable” is on under native captions.
- [ ] **VRChat OSC** — desktop build only; verify chatbox / OSC target with VRChat running.
- [ ] **Kick** — optional: OAuth only; register `http://localhost:17890/oauth/kick/callback` for desktop (chat/emotes not wired yet).

## Remote client (`/client`)

- [ ] Open client URL with `host`, `port`, `id` query params; canvas loads when host is running.
- [ ] Disconnect host — user sees a clear error (not a silent blank screen).

## Packaging

- [ ] `pnpm tauri build` (or `npm run tauri build`) — exe and installer under `src-tauri/target/release/` match CI expectations in `.github/workflows/release.yml`.
