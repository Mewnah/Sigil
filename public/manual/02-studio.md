# Studio

This chapter covers the **canvas**, **elements**, and **project files** asset library.

## Table of contents

- [Canvas and elements](#canvas-and-elements)
- [Files](#files)

## Canvas and elements

![Canvas and scenes](assets/canvas-scenes.png)

### Purpose

Design **on-stream layouts**: text and images positioned per **scene**, with fonts, colors, and animation suitable for OBS or the **client** mirror.

### Open the inspector

Sidebar → **Canvas and Elements**. Use the **Scenes** vs **Elements** subtabs inside the inspector (your last choice is remembered).

### Canvas toolbar

- **Undo / Redo** — `Ctrl+Z`, `Ctrl+Y` / `Ctrl+Shift+Z` (when not typing in a field).  
- **Zoom, grid, alignment** — see tooltips on buttons.  
- **History** — optional panel for document history.  
- **Presentation mode** — **F11** enters full-screen canvas preview; **Esc** exits (see [Shortcuts and tips](04-system.md#shortcuts-and-tips)).

### Working with elements

1. Select an element on the canvas (click or marquee).  
2. The **Properties** panel on the **right** opens for **text** or **image** elements.  
3. Drag handles to move/resize; use inspector fields for precise values.

### Scenes

Switch **active scene** from the canvas inspector to edit different layouts (for example “Just Chatting” vs “BRB”).

### Tips

- **Double-click** the canvas background may focus **Scenes** (per current editor behavior).  
- Duplicate selection from the toolbar when reusing lower-thirds or labels.  
- Legacy **audio visualizer** elements from older projects are stripped on load; see repo `src/client/migrate/legacyDocument.ts` if you migrate old files.

### Related

- [Files](#files) — assets used on canvas  
- [OBS Studio](03-integrations.md#obs-studio) — browser source URL  
- [Client mirror](00-overview.md#client-mirror)  

## Files

![Files inspector](assets/files-inspector.png)

### Purpose

Manage **project media** (audio, images, and related files) used by the canvas and playback features.

### Open the inspector

Sidebar → **Files**.

### Typical actions

- **Import** or add files according to the controls in the inspector.  
- Organize or remove unused assets to keep project size manageable.  
- Some **sound effects** play from host-side playback tied to canvas events (see stream setup).

### Tips

- Prefer **short paths** and avoid special characters in filenames to reduce cross-platform issues.  
- Large audio files increase load time; trim clips where possible.

### Related

- [Canvas and elements](#canvas-and-elements)  
- [Project](04-system.md#project)  
