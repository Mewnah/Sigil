# System

This chapter covers **project**, **settings**, and **keyboard shortcuts**.

## Table of contents

- [Project](#project)
- [Settings](#settings)
- [Shortcuts and tips](#shortcuts-and-tips)

## Project

![Project inspector](assets/project-inspector.png)

### Purpose

**Create, open, save, and manage** Sigil project files that contain your document (elements, scenes, canvas size, etc.).

### Open the inspector

Sidebar → **Project**.

### Typical workflow

1. **New project** — start fresh; configure canvas size early if you target a fixed OBS resolution.  
2. **Open** — load an existing `.sigil` / project file (extension per your build).  
3. **Save** often during editing.

### Tips

- After opening a project, jump to **Canvas and Elements** to resume layout work (`Project` may offer shortcuts to scenes).  
- Version control: binary/json project files can be committed; exclude secrets and large caches.

### Related

- [Files](02-studio.md#files)  
- [Getting started](00-overview.md#getting-started)  

## Settings

![Settings inspector](assets/settings-inspector.png)

### Purpose

Global application preferences: **language**, **theme**, **shortcuts**, **background input**, timers, and cross-cutting options that are not limited to one service.

### Open the inspector

Sidebar → **Settings** (or `Ctrl+,` / `Cmd+,`).

### Sections (representative)

- **UI language** — i18n; RTL locales include Arabic and Urdu (see [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) for RTL smoke checklist).  
- **Appearance** — theme selection aligned with header toggle.  
- **Audio** — default input and output devices. On the desktop app the **default input device** is a native device; Whisper, Vosk, Moonshine, and local HTTP STT use it unless their inspector picks a different mic (Azure and Deepgram have their own browser-device pickers).  
- **Shortcuts** — global **background input** combo and other recorded shortcuts (Tauri global shortcut registration).  
- **Timers** — e.g. background input auto-submit duration.

### Background input

When enabled and a shortcut is set, Sigil can capture **typed text** from other applications (feature-flagged via native capabilities). Use the shortcut recorder in Settings; conflicts show a toast if registration fails.

### Tips

- After changing **language**, some panels may need a reopen to pick up every string.  
- **Global shortcuts** must not collide with OS or game bindings.

### Related

- [Shortcuts and tips](#shortcuts-and-tips)  
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — translation maintenance  

## Shortcuts and tips

![Bottom control bar](assets/control-bar.png)

Verified against the host UI code (`src/core/ui/shortcuts.ts` and related). On **macOS**, **Cmd** is used where **Ctrl** appears below for the same binding.

### Global shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` / `Cmd+1` | Toggle **STT** start/stop |
| `Ctrl+2` / `Cmd+2` | Toggle **AI Transform** start/stop |
| `Ctrl+3` / `Cmd+3` | Toggle **TTS** start/stop |
| `Ctrl+,` / `Cmd+,` | Open **Settings** |
| `Ctrl+I` / `Cmd+I` | Toggle **Stats** panel |
| `Ctrl+B` / `Cmd+B` | Toggle **sidebar navigation** collapse (narrow left column) |
| `Ctrl+Z` / `Cmd+Z` | **Undo** document (not when typing in inputs) |
| `Ctrl+Y` / `Cmd+Y` | **Redo** |
| `Ctrl+Shift+Z` / `Cmd+Shift+Z` | **Redo** |
| `F11` | Toggle canvas **presentation mode** |
| `Esc` | Exit **presentation mode** when active |

### Canvas toolbar (see tooltips)

- **Undo** — `Ctrl+Z`  
- **Redo** — `Ctrl+Y` or `Ctrl+Shift+Z`  

### Presentation mode

**F11** hides chrome for a clean preview; **Esc** or the on-screen **Exit presentation** control restores the full UI.

### Service control bar

The bottom **STT / AI / TTS** cards mirror the same start/stop behavior as the inspectors and the `Ctrl+1`–`Ctrl+3` shortcuts above.

### Background input shortcut

Configured in **Settings**; uses **Tauri global shortcuts** when native features allow. Invalid combinations show an error toast and clear the binding.

### Tips

- **Number keys** `Ctrl+1`–`Ctrl+3` control **services**, not sidebar tabs—use the mouse or sidebar to open inspectors.  
- If a shortcut seems dead, click outside text fields; editable targets intentionally skip some canvas shortcuts.

### Related

- [UI overview](00-overview.md#ui-overview)  
- [Settings](#settings)  
