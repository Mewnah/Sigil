# Performance and connection-timeout audit

This document captures baseline methodology, pass/fail thresholds (tune to your hardware), operational guidance, and implementation notes from the audit.

## Test scenarios

| # | Scenario | What to verify |
|---|----------|----------------|
| 1 | Host idle 1–5 min after boot | CPU %, private bytes, thread count stable |
| 2 | Host + local STT/TTS backends | No hung `invoke`; errors within timeout |
| 3 | `/client` same machine | PeerJS connects; canvas syncs |
| 4 | `/client` over LAN | Same as 3; tolerate brief packet loss |
| 5 | Linked pubsub (`linkConnect`) | WS survives idle; reconnects after drop |
| 6 | External browser vs Tauri webview | Same URL: comparable WS/Peer stability |

## How to measure

- **Process:** Task Manager or Process Explorer — `Sigil.exe` CPU (average over 60s idle), Memory (private working set), Threads.
- **Main thread:** Chrome DevTools → Performance — record 10s idle; note long tasks &gt; 50ms.
- **WebSocket:** Network → WS — frames/sec when idle (should be near zero except keepalive pings).
- **Bundle:** Set `ANALYZE=true` and run `npm run build` (PowerShell: `$env:ANALYZE="true"; npm run build`), then open `stats/rollup-stats.html` (treemap). The visualizer is skipped by default so normal builds stay faster.

## Suggested pass/fail thresholds (adjust per release)

| Metric | Target (guidance) |
|--------|-------------------|
| Host idle CPU (no chat/OBS) | &lt; 2–5% on a typical laptop |
| Private working set growth | No unbounded climb over 30 min idle |
| Linked pubsub idle | Connection alive ≥ 10 min behind a consumer router, or reconnect ≤ 30s |
| Outbound HTTP (local TTS/STT) | Fails or completes within configured timeout (no indefinite hang) |
| `/client` | Connects to host PeerJS within 30s on LAN under normal conditions |

## Top hotspots (from bundle + code review)

Last measured production build (gzip sizes in build log):

| Asset | Size (min) | Notes |
|-------|------------|--------|
| `index-*.js` (main) | ~438 KB (~135 KB gzip) | After `manualChunks` split |
| `react-vendor-*.js` | ~349 KB (~116 KB gzip) | React |
| `twurple-*.js` | ~232 KB (~54 KB gzip) | Twitch integration |
| `collab-*.js` | ~193 KB (~58 KB gzip) | PeerJS / Yjs |
| `vosk-*.js` | ~5.8 MB (~2.3 MB gzip) | Lazy-loaded with Vosk STT path only |
| Inspector chunks | 1–24 KB each | Already `React.lazy` in [`src/core/ui/inspector/index.tsx`](src/core/ui/inspector/index.tsx) |

Inspect `stats/rollup-stats.html` after `ANALYZE=true npm run build`. Large deps: Twurple, Microsoft Speech SDK, OpenAI, framer-motion, PeerJS/Yjs, lodash. Inspectors are lazy; further wins come from `manualChunks` (see `vite.config.ts`) and avoiding new eager imports of heavy SDKs.

## Operational guidance

- **Firewalls:** Allow inbound TCP on the configured Sigil port for `/client` and PeerJS (`/peer`).
- **Reverse proxies:** If terminating TLS in front of Sigil, set WebSocket read/write timeouts above the app keepalive interval (`LINK_PUBSUB_KEEPALIVE_MS` in [`src/shared/services/pubsub/index.ts`](../src/shared/services/pubsub/index.ts), default 25s) or disable proxy idle timeout for that route.
- **Background browser tabs:** Chrome may throttle timers; client uses reconnection logic — prefer keeping the client tab focused during live use.
- **Service worker (`/client`):** Workbox precaches the production build. After a host update, clients may need a refresh so the SW picks up new assets; `index.tsx` already listens for `updatefound` and reloads when appropriate.

## Service lifecycle matrix (host)

| Service | Timers / network idle when “off” | Notes |
|---------|-----------------------------------|--------|
| Twitch | No live interval until logged in | `setInterval` only after successful `connect()` |
| Kick | Same | |
| Others | Unchanged from prior behavior | Further lazy-init can follow the same pattern |

## Rust HTTP timeouts

Shared client: [`src-tauri/src/services/http_client.rs`](../src-tauri/src/services/http_client.rs) — connect 10s, total 300s (per-request `.timeout()` on Whisper/Vosk model downloads: 7200s). Errors surface as plugin `Result::Err` strings.

## Frontend `fetch` inventory

| Area | Pattern |
|------|--------|
| Kick OAuth/API | `fetchWithTimeout` (25–30s) |
| VoiceVox | `fetchWithTimeout` + existing `AbortSignal` for cancel |
| Discord webhooks | `fetchWithTimeout` (25s) |
| Azure translate (languages list + translate API) | `fetchWithTimeout` (20s / 60s) |
| Twitch emote loaders (BTTV/FFZ/7TV) | `fetchWithTimeout` (25s) via [`src/core/services/twitch/emote_loaders.ts`](../src/core/services/twitch/emote_loaders.ts) |
| LLM transform | Existing 15s `Promise.race` timeout |

## Soak checklist (manual)

1. Start host, leave 60+ minutes idle; note memory every 15 min.
2. Open `/client`, disconnect Wi‑Fi 10s, reconnect; confirm recovery without full page reload when possible.
3. Enable linked pubsub, idle 15+ min, confirm state stays connected or shows reconnecting.
