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

### Text captions

- Captions revise in place: when speech-to-text or translation corrects the end of a line, only the changed end re-animates; the rest stays put.
- Translation partials show their confirmed part in the text colour and the unconfirmed end in the interim colour (CSS classes `.interim.tentative` on text, `interim profanity tentative` on masks, `interim tentative` on emotes). Interims without that information still dim the whole line (`.interim`).
- **Use keyboard input** controls whether typed text and Twitch chat appear in the element. With **Any text source**, each topic gets its own line (speech-to-text and its translation separately; Rewrite's synced echo stays one line).

### Scenes

Switch **active scene** from the canvas inspector to edit different layouts (for example “Just Chatting” vs “BRB”).

### Event text

Wherever a setting pairs an **event** with a line of **text**, the text can use values from the event. Put them between `${` and `}`, or between `<%=` and `%>`:

| You write | Sigil shows |
|-----------|-------------|
| `${event}` or `<%= event %>` | The event's value |
| `${event.user.name}` or `<%= event.user.name %>` | A field inside the value |
| `${event.items[0]}` or `${event["first name"]}` | A list item, or a field whose name is not a single word |
| `<%- event.user.name %>` | The same field with `&`, `<`, `>`, `"` and `'` replaced by `&amp;`, `&lt;`, `&gt;`, `&quot;` and `&#39;`, for text that ends up in HTML |

For example, `Welcome ${event.user.name}, you have <%= event.count %> points` shows `Welcome Ada, you have 3 points` for the value `{user: {name: "Ada"}, count: 3}`.

- A **path** is the word `event` followed by `.name`, `[0]`, `["key"]` or `['key']` (`?.name` means the same as `.name`). Spaces are allowed inside the delimiters and inside the square brackets. A plain name is made of letters, digits, `_` and `$` and does not start with a digit; any other name goes in `["…"]`.
- A field that does not exist shows nothing, and so do `null` and `undefined`. Numbers and true/false show as text, a list as its items joined by commas, and any other object as `[object Object]` (pick one of its fields instead).
- `${…}` and `<%= … %>` insert the value as it is. Use `<%- … %>` where the text is shown as HTML.
- A path only reads the fields an object or list holds itself: not what it inherits, and it never runs a getter. The field names `constructor`, `prototype` and `__proto__` are refused.
- **Nothing in event text is ever run.** Whatever Sigil does not understand is shown exactly as you typed it: `<% … %>` blocks, function calls such as `event.name.toUpperCase()`, operators and conditions, and names other than `event` (such as `window`). If you see your own `${…}` or `<% … %>` on screen, the reader refused it. Blocks are not stripped because that would quietly change what a text like `<% if (event.live) { %>LIVE<% } %>` shows.
- This is deliberate. A scene file from someone else can carry event text, and opening it must never run that person's code on your PC. Earlier versions gave event text to a JavaScript template engine, so `<% … %>` blocks and expressions used to work; they no longer do. Move the logic into the event, or into one text per event.

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
