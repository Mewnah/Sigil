# Sigil architecture

## Stack

- **Shell:** Tauri 2 (Rust) — native plugins (audio, STT/TTS, OSC, local HTTP server); capabilities in `src-tauri/capabilities/` replace the v1 allowlist.
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

1. **Text/events:** `Service_PubSub.publish` → local PubSub-JS topics → `invoke("plugin:web|pubsub_broadcast")` → Rust → `emit("pubsub", …)` → JS listener; plus PeerJS broadcast to clients; optional outbound WS when “linked.”
2. **Document:** Yjs doc in `ApiClient.document`; `Service_Peer` syncs over PeerJS (see `src/shared/services/peer/`).
3. **Local server:** Rust `web` plugin serves `127.0.0.1:<port>` — `ping`, PeerJS path, pubsub WebSocket, static assets (`src-tauri/src/services/web/`).

## Native IPC

- Generic `invoke` commands: `get_native_features`, `get_port`, `app_close`.
- Feature work is mostly `plugin:<name>|<command>` from the frontend.

## UI state

Sidebar tab visibility, expand/collapse, and multi-selection live in **Zustand** (`useAppUIStore`) so React and `ApiServer.changeTab` / `closeSidebar` stay in sync without a Valtio `window.ApiServer.ui` object.

## Release / quality

- `npm run typecheck` — `tsc --noEmit`
- `npm run test:e2e` — Playwright smoke (build + preview; `/` and `/client` shells mount `#root`)

For deeper diagrams, see the internal architecture audit / roadmap in your planning docs.

## Security notes (Tauri 2)

- **Capabilities:** IPC and filesystem access are gated by capability JSON (`migrated`, `desktop-capability`). Prefer tightening `fs:scope` and plugin permissions over broad `*:default` when adding features.
- **Threat model (short):** The Rust `web` plugin binds an HTTP/WebSocket stack to **localhost** only (`127.0.0.1`) for PeerJS signaling, pubsub relay, and static assets. Remote clients use the **PeerJS** path over the LAN; they do not execute Tauri commands. Treat any future exposure of that server beyond localhost as a high-risk change requiring authentication and TLS review.
