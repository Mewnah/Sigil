# Canvas and elements

![Canvas and scenes](assets/ui-placeholder.svg)

## Purpose

Design **on-stream layouts**: text and images positioned per **scene**, with fonts, colors, and animation suitable for OBS or the **client** mirror.

## Open the inspector

Sidebar → **Canvas and Elements**. Use the **Scenes** vs **Elements** subtabs inside the inspector (your last choice is remembered).

## Canvas toolbar

- **Undo / Redo** — `Ctrl+Z`, `Ctrl+Y` / `Ctrl+Shift+Z` (when not typing in a field).  
- **Zoom, grid, alignment** — see tooltips on buttons.  
- **History** — optional panel for document history.  
- **Presentation mode** — **F11** enters full-screen canvas preview; **Esc** exits (see [Shortcuts](shortcuts-and-tips.md)).

## Working with elements

1. Select an element on the canvas (click or marquee).  
2. The **Properties** panel on the **right** opens for **text** or **image** elements.  
3. Drag handles to move/resize; use inspector fields for precise values.

## Scenes

Switch **active scene** from the canvas inspector to edit different layouts (for example “Just Chatting” vs “BRB”).

## Tips

- **Double-click** the canvas background may focus **Scenes** (per current editor behavior).  
- Duplicate selection from the toolbar when reusing lower-thirds or labels.  
- Legacy **audio visualizer** elements from older projects are stripped on load; see repo `src/client/migrate/legacyDocument.ts` if you migrate old files.

## Related

- [Files](files.md) — assets used on canvas  
- [OBS](obs.md) — browser source URL  
- [Client mirror](client.md)  
