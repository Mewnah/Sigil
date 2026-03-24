# Major upgrades backlog

Use this list to plan larger follow-on work after the current release push. Order is suggested priority, not fixed.

## Platform and build

- **Tauri v2** — Migrate app shell, IPC, and plugins when dependencies are ready; re-validate all `invoke` paths and bundling.
- **Release CI** — Confirm Windows MSI/exe names from `tauri build` match [`.github/workflows/release.yml`](../.github/workflows/release.yml); add macOS/Linux jobs if you ship those.
- **Rust toolchain** — Periodically `cargo update` / `cargo audit`; keep `whisper-rs` / `whisper-rs-sys` versions aligned.

## Architecture

- **App runtime API** — Narrow use of `window.ApiServer` / `window.ApiClient` behind a small typed facade for tests and clearer init order.
- **Shared layer** — `consumePubSubMessage` in pubsub still reaches into `window.ApiServer` for STT; consider a registered handler from core only.
- **Reactive state** — Finish moving incidental Valtio UI slices to Zustand where it simplifies React subscriptions; keep Yjs/document on existing stack.

## Product and quality

- **E2E** — Extend Playwright beyond Vite preview (e.g. smoke on packaged app or documented manual checklist for Tauri-only flows).
- **TTS/STT matrix** — Scripted or manual pass per backend (native, Windows, Azure, Whisper, Vosk, etc.) with “minimum config” documented.
- **i18n and legacy strings** — Replace remaining “curses” product strings in `public/i18n` and static HTML where Sigil is the product name.
- **Client mode** — Harden `/client` for production: errors when host unavailable, reconnect, and security notes for LAN use.

## Optional performance

- **Inspector** — Already lazy-loaded; consider lazy routes for heavy dashboard-only views if cold start is still an issue.
- **Bundle** — Revisit Rollup/manual chunks for largest editor chunks; keep `stats/` out of git (see `.gitignore`).

## Security

- **Invoke surface review** — Re-check Tauri allowlist and custom commands after any plugin changes.
- **WebSocket / pubsub** — Document threat model for localhost-only server and linked remote pubsub.

See also [ARCHITECTURE.md](../ARCHITECTURE.md) for how host, client, and native pieces fit together.
