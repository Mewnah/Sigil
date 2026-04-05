# Release build smoke matrix

Use this list before shipping a **packaged** Tauri build. Items that need a human (OAuth, second instance, mic in a real browser) cannot be fully automated in CI; run them locally after `pnpm build` and `pnpm tauri build`.

## Automated / agent checks (run in repo root)

- `pnpm run build` — TypeScript + Vite production bundle (includes `sync-manual.mjs`).
- `cd src-tauri && cargo check --release` — release-profile Rust compiles (LTO/strip settings apply at link time; full `cargo build --release` is optional for a longer pass).

## Manual checks (installed or `tauri build` output)

1. **Startup** — App opens with default embedded server port (3030 unless `--port` is set); no blank `#root` error overlay.
2. **OAuth** — Twitch/Kick: loopback `http://localhost:17890/...` works; packaged webview redirects match [manual/oauth-and-ports.md](manual/oauth-and-ports.md).
3. **External STT / mic.html** — From STT inspector, open Chrome or Edge (or default browser on Linux/macOS); WebSocket pubsub connects; speech reaches the host.
4. **Client mirror** — Open `/client` with `host`, `port`, `id` query params against the running host; canvas loads; disconnect shows a clear error.
5. **i18n** — Switch languages; translations load from `/i18n/...` without JSON parse errors.
6. **Neural** — ONNX TTS load path and optional sidecar (`http://127.0.0.1:8766`) if you ship those features.
7. **Second instance** — Launch a second Sigil while the first holds the port; expect the native port-unavailable dialog.

See also [MANUAL-TAURI-CHECKLIST.md](MANUAL-TAURI-CHECKLIST.md) for a broader feature pass.
