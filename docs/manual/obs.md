# OBS Studio

![OBS inspector](assets/ui-placeholder.svg)

## Purpose

Copy **Browser Source** URLs for Sigil’s output and optionally drive OBS via **obs-websocket** (create sources, switch scenes) from the app.

## Open the inspector

Sidebar → **OBS Studio**.

## Browser source (most common)

1. Click **Copy browser source link** (or equivalent).  
2. In OBS, add a **Browser** source and paste the URL.  
3. Set width/height to match your canvas (often 1920×1080 or your base resolution).

## WebSocket automation

1. Enable **obs-websocket** in OBS and set **port** and **password**.  
2. In Sigil, connect with matching credentials.  
3. Use any **setup** or automation controls (for example “Sigil browser source” creation) from the inspector dropdowns.

## Tips

- Browser source **FPS** and **shutdown when not visible** affect CPU; tune per scene.  
- If the preview is blank, confirm the **host** is running and the URL uses the correct **port**.

## Related

- [Canvas and elements](canvas.md)  
- [Client mirror](client.md)  
