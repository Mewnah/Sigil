# Shortcuts and tips

Verified against the host UI code (`src/core/ui/shortcuts.ts` and related). On **macOS**, **Cmd** is used where **Ctrl** appears below for the same binding.

## Global shortcuts

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

## Canvas toolbar (see tooltips)

- **Undo** — `Ctrl+Z`  
- **Redo** — `Ctrl+Y` or `Ctrl+Shift+Z`  

## Presentation mode

**F11** hides chrome for a clean preview; **Esc** or the on-screen **Exit presentation** control restores the full UI.

## Service control bar

The bottom **STT / AI / TTS** cards mirror the same start/stop behavior as the inspectors and the `Ctrl+1`–`Ctrl+3` shortcuts above.

## Background input shortcut

Configured in **Settings**; uses **Tauri global shortcuts** when native features allow. Invalid combinations show an error toast and clear the binding.

## Tips

- **Number keys** `Ctrl+1`–`Ctrl+3` control **services**, not sidebar tabs—use the mouse or sidebar to open inspectors.  
- If a shortcut seems dead, click outside text fields; editable targets intentionally skip some canvas shortcuts.

## Related

- [UI overview](ui-overview.md)  
- [Settings](settings.md)  
