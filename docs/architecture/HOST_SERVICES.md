# Host services, persistence, and UI tabs

This document describes how **integrations**, **host-only modules**, and **sidebar/nav tabs** relate in Sigil. It complements [SERVICE_CONNECTIVITY_AUDIT.md](../SERVICE_CONNECTIVITY_AUDIT.md) (Rust ↔ TS wiring).

## Three layers

1. **`Services` enum** ([`src/services-registry.ts`](../../src/services-registry.ts))  
   Stable string ids used for **inspector navigation** (`InspectorTabPath.tab`), action bar ordering where applicable, and UI routing. Kept free of imports from `src/core` to avoid circular dependencies.

2. **`BackendState.services`** ([`src/core/schema.ts`](../../src/core/schema.ts))  
   Zod-backed **persisted** settings per integration. Keys here are **`PersistedHostServiceKey`** (see `PERSISTED_SERVICE_KEYS`, `isPersistedServiceKey`, and `isPersistedServicesTab` in the same file). Use **`window.ApiServer.patchService(key, fn)`** only for these keys.

3. **`ApiServer` fields** ([`src/core/index.ts`](../../src/core/index.ts))  
   Live service instances. Some correspond 1:1 with a persisted key (e.g. `stt` ↔ `state.services.stt`). Others are **host-only** (no row in `BackendState.services`).

## Persisted integrations (`BackendState.services`)

| Key | `ApiServer` field | Notes |
|-----|-------------------|--------|
| `vrc` | `vrc` | |
| `stt` | `stt` | |
| `tts` | `tts` | |
| `translation` | `translation` | |
| `transform` | `transform` | |
| `twitch` | `twitch` | |
| `kick` | `kick` | |
| `discord` | `discord` | |
| `obs` | `obs` | |

## UI tab without persisted slice

| `Services` id | `ApiServer` field | Persistence |
|---------------|-------------------|-------------|
| `voice_changer` | `voiceChanger` | Not under `state.services` (in-memory / sidecar state). Do not use `patchService` for this tab. |

## Host modules (not `Services` tabs)

| Field | Role |
|-------|------|
| `_state` (`Service_State`) | Load/save `BackendState`, debounced writes |
| `keyboard` | Shortcuts / input |
| `sound` | Sound effects |
| `history` | Undo/history style state |

These participate in startup **`init()`** where applicable but are **not** inspector tabs in the `Services` enum.

## Central registration (TypeScript)

- **Startup init order and labels**: single registry in [`src/core/index.ts`](../../src/core/index.ts) (`buildHostServiceInitTasks`).
- **Inspector lazy imports, nav groups, sidebar buttons, header titles**: [`src/core/ui/servicePanels.tsx`](../../src/core/ui/servicePanels.tsx).

## Checklist: add a **persisted** integration

1. Add `Services.<id>` in [`src/services-registry.ts`](../../src/services-registry.ts) if it needs a sidebar/inspector tab.
2. Add `Service_*_Schema` and register under `BackendSchema.services` in [`src/core/schema.ts`](../../src/core/schema.ts). Update **`PERSISTED_SERVICE_KEYS`** (TypeScript will enforce exhaustiveness against `PersistedHostServiceKey`).
3. Implement `src/core/services/<id>/` (`index.ts`, `schema.ts`, optional `services/` backends).
4. Add **`ApiServer`** field and include **`init`** in the registry inside [`src/core/index.ts`](../../src/core/index.ts).
5. Add a panel entry in [`src/core/ui/servicePanels.tsx`](../../src/core/ui/servicePanels.tsx) (lazy inspector, nav, sidebar, title).
6. If the service should appear on the bottom action bar, add the id to **`DEFAULT_ACTION_BAR_SERVICE_ORDER`** in [`src/core/schema.ts`](../../src/core/schema.ts).
7. Optional: Tauri plugin + TS `invoke` (see connectivity audit doc).

## Checklist: add a **non-persisted** inspector tab

1. Add `Services.<id>` if it is a first-class tab.
2. **Do not** add a `BackendSchema.services` slice unless you intend to persist settings.
3. Implement service module and wire **`ApiServer`** + **init registry** as needed.
4. Register the panel in **`servicePanels.tsx`** only (avoid `patchService` for this id).

## Checklist: host-only module (no inspector)

1. Add **`ApiServer`** field and hook **`init`** in the registry if it needs startup work.
2. No changes to `Services`, `servicePanels`, or `BackendSchema.services` unless you later expose UI or persistence.
