# Sigil architecture

## Stack

- **Shell:** Tauri 1 (Rust) — native plugins (audio, STT/TTS, OSC, local HTTP server).
- **UI:** Vite, React 18, TypeScript.
- **State:** Valtio for backend/service and document sync; Zustand (`src/core/ui/store.ts`) for shell UI (sidebar, stats panel); Yjs + PeerJS for collaborative document sync.

## Runtime modes

| Mode | How | UI root |
|------|-----|---------|
| **Host** | Default path in desktop app | `SigilRoot` — full editor, integrations, inspectors |
| **Client** | Path `/client` + query (`host`, `port`, `id`) | `ClientView` — canvas/elements only; talks to host over PeerJS |

`AppConfiguration` in `src/config.ts` decides mode and loads network settings (Tauri `plugin:web|config` on host).

## Layering

- **`src/client/`** — Document, scenes, elements, files, particles, sound. Must not depend on host-only APIs (see `src/client/readme.md`).
- **`src/core/`** — Host services (Twitch, Kick, Discord, OBS, VRC, STT/TTS, translation, transform, etc.), persisted settings (`Service_State`), Sigil chrome.
- **`src/shared/`** — PubSub (PubSub-JS + optional WebSocket + Tauri `pubsub` event), Peer bridge.
- **`src/services-registry.ts`** — `Services` enum (no imports from core/client) to avoid circular types.

## Data flow (host)

1. **Text/events:** `Service_PubSub.publish` → local PubSub-JS topics → `invoke("plugin:web|pubsub_broadcast")` → Rust → `emit_all("pubsub")` → JS listener; plus PeerJS broadcast to clients; optional outbound WS when “linked.”
2. **Document:** Yjs doc in `ApiClient.document`; `Service_Peer` syncs over PeerJS (see `src/shared/services/peer/`).
3. **Local server:** Rust `web` plugin serves `127.0.0.1:<port>` — `ping`, PeerJS path, pubsub WebSocket, static assets (`src-tauri/src/services/web/`).

## Native IPC

- Generic `invoke` commands: `get_native_features`, `get_port`, `app_close`.
- Feature work is mostly `plugin:<name>|<command>` from the frontend.

## UI state

Sidebar tab visibility, expand/collapse, and multi-selection live in **Zustand** (`useAppUIStore`) so React and `ApiServer.changeTab` / `closeSidebar` stay in sync without a Valtio `window.ApiServer.ui` object.

## Release / quality

- `npm run typecheck` — `tsc --noEmit`
- `npm run test:e2e` — Playwright smoke (build + preview; checks `#root` is present)

For deeper diagrams, see the internal architecture audit / roadmap in your planning docs.
