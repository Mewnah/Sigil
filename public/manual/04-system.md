# System

This chapter covers **project**, **settings**, **models and runtimes**, and **keyboard shortcuts**.

## Table of contents

- [Project](#project)
- [Settings](#settings)
- [Models & Runtimes](#models--runtimes)
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
- **Audio** — default input and output devices. On the desktop app the **default input device** is a native device; the built-in engine, Whisper, Vosk and local HTTP STT use it unless their inspector picks a different mic (Azure and Deepgram have their own browser-device pickers).  
- **GPU & diagnostics** — which device (CUDA, DirectML, CPU) each model actually runs on, your graphics adapters, **Test GPU acceleration**, the optional **NVIDIA GPU runtime** download (about 1.9 GB, needed for CUDA on a GeForce 16/20 series or newer with driver R580 or newer; the installer does not include it, and without it Sigil uses DirectML; the same pack is also in [Models & Runtimes](#models--runtimes)) and **Open log folder** (`%LOCALAPPDATA%\com.sigil.app\logs`).
- **Shortcuts** — the global **mute sound effects** shortcut (the STT start/stop and mute shortcuts are in the STT inspector), plus the **background input** combo when that feature is built in. They use Tauri global shortcut registration and work while Sigil is in the background.
- **Timers** — e.g. background input auto-submit duration.
- **Local web server** — whether the server is listening (and on which port) and **Only allow Sigil pages to connect** (on by default): it blocks websites open in your browser from reading or sending captions through Sigil, while the app, `/client`, OBS, the mic page, the captions page and Link apps keep working. Turn it off only if a tool you trust is refused. The `SIGIL_LAN_ORIGIN_CHECK` environment variable overrides it.

### Background input

When enabled and a shortcut is set, Sigil can capture **typed text** from other applications (feature-flagged via native capabilities). Use the shortcut recorder in Settings; conflicts show a toast if registration fails.

### Tips

- After changing **language**, some panels may need a reopen to pick up every string.
- **Global shortcuts** must not collide with OS or game bindings. A shortcut another app already holds is kept with a warning.
- If the settings file is ever damaged, Sigil keeps it as `settings.corrupt-<time>.bak`, restores the last good copy and says so in the System log.

### Related

- [Shortcuts and tips](#shortcuts-and-tips)  
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — translation maintenance  

## Models & Runtimes

### Purpose

One place for everything Sigil downloads: speech models, runtime packs (such as the NVIDIA GPU runtime), voice templates and benchmark assets, plus a **GPU and runtimes** card that shows what your PC can accelerate.

### Open the tab

Sigil navigation → **Models & Runtimes** (just above Settings).

### What you can do

- **Download**, **Cancel** (the partial download is kept, so the next attempt resumes), **Resume**, **Retry**, **Update**, **Check files** and **Delete** any entry. One download runs at a time by default.
- **Import…** a file or folder you already have. Entries marked **Import only** are Sigil-built packs that are not published online yet.
- Whisper and Vosk models you already downloaded are recognised and marked **Found on this PC**; they are not downloaded again.
- **Download source:** Automatic, a preferred mirror, or your own mirror URL (for a LAN copy). Every file is checked against its known checksum whatever the source.
- **Storage** shows how much space models use and where they live.
- **GPU and runtimes** lists your graphics cards and drivers, which acceleration options work (and why not), and suggested model sizes for your PC.

### Tips

- Large downloads can be paused with **Cancel** and resumed later, even after restarting Sigil.
- On a slow or blocked connection, try the **hf-mirror.com** source.

## Shortcuts and tips

![Bottom control bar](assets/control-bar.png)

Verified against the host UI code (`src/core/ui/shortcuts.ts` and related). On **macOS**, **Cmd** is used where **Ctrl** appears below for the same binding.

### Global shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` / `Cmd+1` | Toggle **STT** start/stop |
| `Ctrl+2` / `Cmd+2` | Toggle **Rewrite** start/stop |
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

The bottom **STT / Rewrite / TTS** cards mirror the same start/stop behavior as the inspectors and the `Ctrl+1`–`Ctrl+3` shortcuts above.

### Global shortcuts you record

- **STT start/stop** and **STT mute** — STT inspector → **Shortcuts**.
- **Mute sound effects** — **Settings → Shortcuts**.
- **Background input** — **Settings**, when the feature is built in.

These use **Tauri global shortcuts** and work while Sigil is in the background. Recording finishes on the first full key combination; **Esc** cancels, a plain letter needs Ctrl, Alt or Shift, and a combination already used by another Sigil shortcut is refused. Stored combinations that can never register show an error toast and are cleared once.

### Tips

- **Number keys** `Ctrl+1`–`Ctrl+3` control **services**, not sidebar tabs—use the mouse or sidebar to open inspectors.  
- If a shortcut seems dead, click outside text fields; editable targets intentionally skip some canvas shortcuts.

### Related

- [UI overview](00-overview.md#ui-overview)  
- [Settings](#settings)  
