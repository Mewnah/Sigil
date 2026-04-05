# Settings

![Settings inspector](assets/ui-placeholder.svg)

## Purpose

Global application preferences: **language**, **theme**, **shortcuts**, **background input**, timers, and cross-cutting options that are not limited to one service.

## Open the inspector

Sidebar → **Settings** (or `Ctrl+,` / `Cmd+,`).

## Sections (representative)

- **UI language** — i18n; RTL locales include Arabic and Urdu (see ARCHITECTURE for smoke checklist).  
- **Appearance** — theme selection aligned with header toggle.  
- **Shortcuts** — global **background input** combo and other recorded shortcuts (Tauri global shortcut registration).  
- **Timers** — e.g. background input auto-submit duration.

## Background input

When enabled and a shortcut is set, Sigil can capture **typed text** from other applications (feature-flagged via native capabilities). Use the shortcut recorder in Settings; conflicts show a toast if registration fails.

## Tips

- After changing **language**, some panels may need a reopen to pick up every string.  
- **Global shortcuts** must not collide with OS or game bindings.

## Related

- [Shortcuts and tips](shortcuts-and-tips.md)  
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — translation maintenance  
