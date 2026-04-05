# UI overview

![Studio layout](assets/ui-placeholder.svg)

Sigil **Studio** (host) is organized into a **header**, **left navigation**, optional **left inspector**, **canvas** (center), **right panel**, and **bottom status / controls**.

## Header

- **App menu / branding** — current section title.
- **Sidebar toggle** — collapse or expand the narrow navigation column.
- **Keyboard shortcuts** — opens a cheat sheet (matches [Shortcuts and tips](shortcuts-and-tips.md)).
- **Help** (book icon) — opens the in-app user manual (same content as `docs/manual`, synced to `/manual` at dev/build time).
- **Theme** — switch light/dark (stored with your settings).

## Left navigation

Groups:

1. **Main** — Dashboard, Project  
2. **Live services** — STT, TTS, Translation, AI Transform, Voice Changer  
3. **Studio** — Canvas and Elements, Files  
4. **Integrations** — Twitch, Kick, Discord, OBS, VRChat  
5. **Settings** — bottom of the list  

Choosing an item opens the matching **inspector** (left panel). Choosing the same item again closes the inspector.

## Canvas (center)

The editable **stage** for text and image elements, organized into **scenes**. Use the **canvas toolbar** for zoom, grid, undo/redo, alignment, history, and **presentation mode** (full-screen preview).

## Right panel

- With **no element selected:** default studio panel (logs / context depending on build).
- With an element selected: **Properties** for that text or image element.

## Bottom area

**Status** and quick links into service inspectors; **STT / AI / TTS** controls mirror start/stop for those pipelines (see [Shortcuts and tips](shortcuts-and-tips.md)).

## Floating / overlays

Depending on configuration you may see **TTS input**, **overlay typing**, **background input**, or **stats** (see shortcuts). These are documented on their respective service pages where relevant.

## Host vs client

The **host** runs all inspectors and native features. The **`/client`** page is a **mirror**: it syncs the document with the host and does **not** run Tauri plugins. See [Client mirror](client.md).
