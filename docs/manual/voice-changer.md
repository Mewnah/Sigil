# Voice changer

![Voice changer inspector](assets/ui-placeholder.svg)

## Purpose

Apply **real-time voice modification** through the Tauri **voice changer** plugin when enabled in your build. Availability depends on **native features** and platform support.

## Open the inspector

Sidebar → **Voice Changer**.

## Usage

1. Open the inspector and review **device** / **model** options exposed by the plugin.  
2. **Start** processing when you are ready; **Stop** when finished.  
3. If options are missing, confirm you are on a **desktop host** build (not the `/client` page).

## Tips

- Voice changer is **host-only**; do not expect it in the browser mirror.  
- CPU/GPU load affects game and stream performance—test with VRChat or your title running.

## Related

- [STT](stt.md) — separate from VC; both use audio paths  
- [Settings](settings.md)  
