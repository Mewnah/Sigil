# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sigil is a Tauri 2 (Rust shell) + Vite/React 18/TypeScript desktop app for voice, captions, and stream tooling (STT, TTS, translation, AI transform, Twitch/Kick/Discord/OBS, VRChat OSC, voice changer). Package manager is **pnpm**. Windows is the primary target. It began as a fork of [curses](https://github.com/mmpneo/curses), so `CURSES_*` env vars and legacy persisted values still appear for compatibility.

## Commands

```bash
pnpm install
pnpm tauri dev            # full desktop app (Rust + Vite) — the real runtime
pnpm tauri build          # Windows NSIS/MSI under src-tauri/target/ (no staged GPU provider DLLs)
pnpm tauri:release        # release build: overlay scripts/release/tauri.release.conf.json stages the ORT CUDA/DirectML DLLs
pwsh scripts/release/verify-bundle.ps1          # after tauri:release: DLLs in both installers, no new load-time imports
pnpm dev                  # Vite only (web, no Tauri) — runs scripts/sync-manual.mjs first
pnpm build && pnpm preview  # static web build + serve; no native features

pnpm typecheck           # tsc --noEmit — run before any PR
pnpm test:unit           # vitest (pure TS logic; also scripts/__tests__); pnpm exec vitest run <path> for one area
pnpm test:e2e            # Playwright: builds then smokes / and /client (both mount #root)
pnpm exec playwright test tests/smoke.spec.ts   # single test file
PW_PREVIEW_ONLY=1 pnpm test:e2e                 # skip rebuild when dist/ already exists (bash)
# PowerShell (primary platform): $env:PW_PREVIEW_ONLY=1; pnpm test:e2e

# Rust (from src-tauri/; or pass --manifest-path src-tauri/Cargo.toml from the root)
cargo clippy --all-targets   # CI gate on the Rust side (no new warnings in touched files)
cargo test                   # needs dist/ (run pnpm build, or mkdir dist); model tests are #[ignore]:
SIGIL_TEST_MODELS_DIR=<dir> cargo test real_model -- --ignored   # e.g. a folder with ggml-tiny.en.bin

# i18n (CI runs i18n:check and will fail on key drift)
pnpm i18n:check          # locale JSON key parity vs public/i18n/en + codes in src/i18n.ts
pnpm i18n:sync           # copy missing English keys into other locales
pnpm i18n:report         # list keys whose value still equals English
node scripts/merge-i18n-staging.mjs [--dry-run]   # merge scripts/i18n-staging/<WS>.json into en, then i18n:sync
# merge-http-tts-i18n.mjs / merge-neural-i18n.mjs are retired (they overwrote locale sections; they exit 1)

# docs and release checks (CI runs both)
pnpm docs:check          # node docs/check-docs.mjs [--no-todo]: doc links, anchors, tables, facts that must agree with the code
pnpm release:pins        # node scripts/release/check-catalog-pins.mjs [--online]: every published catalog file is hash-pinned
```

`@/` is aliased to `src/` (see `tsconfig.json` + `vite.config.ts`). `pnpm dev`/`pnpm build` run `scripts/sync-manual.mjs`, which copies `docs/manual/` → `public/manual/` and, separately, the consolidated top-level guides (`docs/OPERATOR_REFERENCE.md`, `DEVELOPER_GUIDE.md`, `RELEASE_AND_ENGINEERING.md`, …) → `public/docs/` for the in-app Help reader — edit the sources under `docs/`, not the generated copies.

## Architecture

**Two runtime modes**, decided by `AppConfiguration` in `src/config.ts` from the URL path:
- **Host** (default, desktop): root `SigilRoot` — full editor, integrations, inspectors. Has Tauri.
- **Client** (`/client?host=&port=&id=`): root `ClientView` — canvas/elements only, no Tauri; syncs to the host over PeerJS.

**Global singletons** live on `window` (declared in `src/index.tsx`): `Config`, `ApiShared`, `ApiServer` (host services), `ApiClient` (document). Code reaches services via `window.ApiServer.*`.

**Three source layers** (`src/services-registry.ts` holds the `Services` enum and must not import from core/client, to break circular types):
- `src/client/` — document, scenes, elements, files, particles, sound. **Remote-safe: this code runs on clients that have no Tauri, so it must never *execute* Tauri at client runtime.** There are no `invoke()` calls here; the few `@tauri-apps` APIs some services do import (document/files/sound) must stay guarded behind `window.Config.isServer()`/`isClient()`. Legacy `audioViz` elements are stripped on load (`src/client/schema/`).
- `src/core/` — host services (`src/core/services/<id>/`), persisted settings, Sigil chrome, `ApiServer` (`src/core/index.ts`), Zod schema (`src/core/schema.ts`).
- `src/shared/` — PubSub bridge and Peer bridge used by both.

**State management** is deliberately split: **Valtio** for backend/service state and doc sync, **Zustand** (`src/core/ui/store.ts`, `useAppUIStore`) for shell UI (sidebar tabs, stats panel), **Yjs + PeerJS** for the collaborative canvas document (undo/redo is document-layer via `documentUndoState`, *not* the stubbed `Service_History`).

**Native side** (`src-tauri/`): frontend calls `invoke("plugin:<name>|<command>")`. Custom Rust plugins are in `src-tauri/src/services/` and registered in `src-tauri/src/main.rs` (diagnostics first, then osc, web, audio, whisper, vosk-stt, moonshine-stt, windows-tts, uberduck-tts, kokoro/melo/chatterbox/fish/neural-onnx/neural-sidecar TTS, translate, voice-changer, keyboard, uwu, and the P1 foundations model-store, runtime, sysinfo, resources, process-host, bench; then speech, tts-sink, local-tts, openai-speech (P2), llm-server, mt-local (P4), voice-library (P5), and caption-window, vr-overlay, text-readings, edge-tts, gemini-tts, tts-helpers, gpt-sovits (P6)). Secondary webview windows (the `captions` window, an early `/captions` route in `src/index.tsx` that never boots `ApiServer`) close with the main window. IPC and FS access are gated by capability JSON in `src-tauri/capabilities/` (Tauri 2 model) — prefer tightening `fs:scope`/plugin permissions over broad `*:default`. **A command is registered in three places:** the `#[command]` fn, `generate_handler!`, and a line in `src-tauri/plugin-commands/<plugin>.txt` (build.rs generates the permissions and `gen/schemas` from these files; keep the regenerated schemas). A new plugin also gets its own `src-tauri/capabilities/<plugin>.json`. Every ONNX Runtime session goes through `neural_audio/ort_session.rs` (EP ladder CUDA → DirectML → CPU, outcome recorded per component and shown in Settings → GPU & diagnostics). Audio callbacks never allocate, lock, log or do IPC (rtrb rings only; MMCSS on callback threads only). Heavy entry points run as `Sigil.exe --headless <cmd>` (parsed before the port check): `ep-probe`, `fetch`, `bench`, `mt-bench` (translation benchmark), `speech-bench` (the P7 speech benchmark), `engine-probe`, `vc-import-worker`, `sysinfo`, `stt-worker` (the isolated STT worker), `gpu-load` (the benchmark's out-of-process GPU load, `Sigil.exe --headless gpu-load <spec>`, run in a KILL_ON_JOB_CLOSE job). Downloads go through the model store only (`model_store::installed_path(id)` / `ensure`; catalog fragments in `src-tauri/model-catalog/`); native runtimes load lazily through `native_runtime` or run as `process_host` helpers, so the app starts with no pack installed (runtime policy in `docs/DEVELOPER_GUIDE.md` → Speech stack internals).

**Data flow (host):** `Service_PubSub.publish` → PubSub-JS topics → `invoke("plugin:web|pubsub_broadcast")` → Rust → `emit("pubsub", …)` → JS listeners, plus PeerJS broadcast to clients. The Rust `web` plugin (`src-tauri/src/services/web/mod.rs`) binds **`0.0.0.0:<port>`** — all IPv4 interfaces, so it is reachable across the LAN, not just localhost (deliberate, matching upstream curses; a `127.0.0.1`-only bind rejected some localhost paths). It serves PeerJS signaling, the pubsub WebSocket, and static assets with **no TLS**, so treat that LAN exposure as a real attack surface and any change to the bind/auth model as high-risk. Since stage 1: `/pubsub` and `/peer` handshakes pass an **Origin allow-list** (`web/origin.rs`; kill switch `lanOriginCheck` setting / `SIGIL_LAN_ORIGIN_CHECK` env), binary frames are refused, and the Rust plugin **no longer relays pubsub between WebSocket peers**: peer text goes only to the host (`pubsub` event), where `consumePubSubMessage` applies the inbound allow-list in `src/shared/services/pubsub/inbound.ts` and re-broadcasts. **A new topic that must cross WebSockets needs a route in `inbound.ts`**, or it is dropped. Wire messages carry `via` (host session ids) for Link-apps loop prevention. Stage 2 (P6-D, `web/auth.rs`): `lanAccess {mode, token, legacyNoToken}`. Clients on other machines need the persistent LAN token: `?token=` on /pubsub and pages, the PeerJS `key` on /peer. Same-PC peers (loopback or this PC's own address, without proxy headers) are exempt, and same-PC links never carry the token. `mode=localhost` binds 127.0.0.1 and rebinds live. Refusals after the upgrade close with 4401/4403. Every admitted socket is closed when the token, mode, legacy switch or Origin switch changes so that it would now be refused. Other players' speech and readings use `others.*` / `readings.*`, **never `text.*`** (pubsub-js bubbles `text.*` to the chatbox, TTS and history); they are never accepted inbound from WebSocket peers. Build links to `public/*.html` pages with `localPageUrl()` (`src/utils/localPages.ts`).

## Adding or changing a host service

A service exists across three registration points that must stay in sync — miss one and TypeScript exhaustiveness checks (or runtime routing) will catch it:
1. `Services.<id>` in `src/services-registry.ts` (if it needs a sidebar/inspector tab).
2. `Service_*_Schema` under `BackendSchema.services` in `src/core/schema.ts`, plus `PERSISTED_SERVICE_KEYS` — **only** for persisted services. Use `window.ApiServer.patchService(key, fn)` exclusively for these keys.
3. `ApiServer` field + `init` task in `src/core/index.ts` (`getHostServiceInitTasks`), and a panel entry in `src/core/ui/servicePanels.tsx` (lazy inspector, nav, sidebar, title).

Non-persisted tabs (e.g. `voice_changer`) skip step 2's schema slice — do **not** call `patchService` for them. The voice changer's settings are persisted by **Rust** (`<app_data>/voice-changer/settings.json`, per-voice `<slot>/slot.json`) and the TS service hydrates from `get_voice_changer_params`; the Rust↔TS params contract is `src-tauri/src/services/voice_changer/params_contract.json`, checked by tests on both sides. The full checklists and the STT/TTS/translation backend wiring tables are in `docs/DEVELOPER_GUIDE.md`.

**UI strings:** new English strings for a parallel workstream go to `scripts/i18n-staging/<WS-ID>.json` and are merged by `node scripts/merge-i18n-staging.mjs` + `pnpm i18n:sync`; a single-author change may edit `public/i18n/en/translation.json` directly and run `pnpm i18n:sync`.

## Key docs

- `docs/DEVELOPER_GUIDE.md` — architecture, host-service registration, connectivity audit, `src/client` rules (the deep reference).
- `docs/OPERATOR_REFERENCE.md` — STT/TTS backends, STT reliability, voice changer, local speech package.
- `docs/RELEASE_AND_ENGINEERING.md` — release smoke, GPU EP smoke, P0 exit matrix, threat model, third-party components, dependency/perf audits.
- `docs/SPEECH_STACK_IMPLEMENTATION_PLAN.md` (phased plan, global rules) and `docs/SPEECH_STACK_RESEARCH_2026-09.md`; ADRs in `docs/adr/`. `docs/VOICE_CHANGER_AI_PLAN.md` is superseded.

## Gotchas

- **Secrets are build-time.** `pnpm build`/`pnpm tauri build` can embed `SIGIL_*`, legacy `CURSES_*`, and `VITE_*` from the environment into the frontend bundle. Never publish a `dist/` built with real secrets. Copy `.env.example` → `.env`; never commit `.env`.
- **RTL:** `ar`/`ur` are supported; on shared chrome prefer logical Tailwind utilities (`ms-`/`me-`, `ps-`/`pe-`, `start-`/`end-`) over physical `ml-`/`mr-` where mirroring matters. `src/i18n.ts` sets `<html dir lang>`.
- **Background input** (Windows low-level keyboard hook) is behind the `background_input` Cargo feature; it is a no-op init when off.
- Voice changer chunk sizes, ONNX execution provider (CPU/CUDA/DirectML), and neural TTS all select their backend at runtime via `ort`. ONNX Runtime is 1.28 (ort 2.0.0-rc.13, CUDA 13: driver R580+, Turing or newer). The release installers bundle the CUDA and DirectML provider DLLs but **not** NVIDIA's CUDA runtime: CUDA needs the opt-in `runtime/nvidia-cuda13` pack (Models & Runtimes), otherwise sessions fall back to DirectML/CPU with a visible reason. w-okada Socket.IO/REST servers are **not** drop-in compatible with Sigil's WebSocket sidecar.
- **Never list build-generated files in the base `tauri.conf.json` `bundle.resources`** (tauri-build copies them at compile time and `cargo check` fails on clean checkouts); release-only resources live in `scripts/release/tauri.release.conf.json`.
- **Never cargo-build a scratch copy of the crate in the shared `src-tauri/target`** (cargo hashes the root package by relative path, so the copy clobbers the real build script / `OUT_DIR`; symptom "Permission diagnostics:default not found"). Use a separate `CARGO_TARGET_DIR`; recover with `cargo clean -p sigil`.
- **SteamVR overlay:** `openvr_api.dll` comes only from the `runtime/openvr` pack (loaded lazily as a Background app, never launching SteamVR); `SIGIL_OPENVR_DIR` points at another folder for development.
- **Closing the app must end it.** `src-tauri/src/exit_guard.rs` ends the process 20 s after the first close request if tao's event loop stalls (log line `exit guard:`). A `WM_CLOSE` sent to tao's hidden "Tao Thread Event Target" window causes that stall (`Process.CloseMainWindow()` on a minimized app, `taskkill` without `/F`). Harnesses close only the window titled `Sigil`, and they launch the full app with `SIGIL_START_MINIMIZED=1` (minimized, never takes focus). They also seed the scratch profile so no one-time offer dialog appears: for both `stt` and `tts`, `desktopDefaultMigrated` and `localOfferShown`. The user works on this PC while agents run.
- **Logs:** `%LOCALAPPDATA%\com.sigil.app\logs\sigil.log` (10 MB × 5); verbosity via `SIGIL_LOG` (tracing `Targets` syntax, e.g. `sigil=debug,warn`). Rust code logs with `tracing`, not `eprintln!`.
- **Voice-changer output queue floor.** A template voice whose engine block is longer than the device chunk returns audio in bursts: one model call per block (p50 74-81 ms idle, 88-98 ms under the game stand-in on the CPU at 120 ms blocks) during which the worker pushes nothing. If the queue stands shorter than the call, the device runs dry and `OutputPlayback` recovery trims the catch-up to `ring_target()` (prefill + chunk + capture burst), which lost 37-52 % of the audio at the 35/70 ms default. `output_max_queue_ms` alone is NOT the lever (ring_target follows the prefill; replaying 35/170 and 35/300 equals 35/70). `voice_changer/block_buffer.rs` derives a per-voice floor from (engine block, device chunk, realized EP) and carries it as `CreatedEngine::min_output_buffer_ms`; worker `port_limits` raises the stock buffers (Low latency 35/70, Stable 80/150) to it, keeps a buffer the user typed (preset Custom) unless the run measures that its prefill is shorter than the median model call (`StoredBufferProof`, latched), and the bench's `EngineInfo::output_buffer_ms` always raises; larger stored values win, nothing is rewritten; GPU, short-block, BYO and family voices get none. Never size a prefill above (pushes before the first model call) x chunk - 2 ms: the gate would open after that call and the queue stands 20-40 ms higher all run. Kill switch `SIGIL_VC_BLOCK_BUFFER=0`. The Streamer on the CPU with Block size on Auto runs 120 ms blocks from 7 physical cores, 160 ms with 6 and 200 ms with 5 or fewer (`presets::cpu_streamer_block`: the 20 ms the 512-frame device block derives cannot keep up; `SIGIL_VC_STREAMER_BLOCK=0`), so a bench spec without `block_ms` measures the block of the machine's core count. The CPU probe ignores a process affinity mask: a harness that stands a core slice in for a smaller PC sets `SIGIL_VC_PHYSICAL_CORES=<n>` (the thread policy and this block both read it).
- **Judge CPU voice-changer rows on delivery, not on RTF or latency alone.** `final_scorecard.py` `played_ratio` (>= 98 %) and dry output (`underrun_ms`) come first: latency is measured on the frames that were played, so a row that drops half the audio can still read a good p99.
