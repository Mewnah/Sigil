# VRChat

![VRChat inspector](assets/ui-placeholder.svg)

## Purpose

Send **OSC** messages for captions and related control into **VRChat** from the **Tauri** host. OSC is **not** available from the in-browser preview alone—use the desktop app.

## Open the inspector

Sidebar → **VRChat**.

## Setup (high level)

1. Enable OSC in **VRChat** settings and note the receive port.  
2. In Sigil, match host/port and options in the inspector.  
3. Test with short phrases from STT or manual send paths if exposed.

## Tips

- Firewall rules must allow **UDP** loopback or LAN as your setup requires.  
- Latency stacks: STT → (optional transform) → OSC → VRChat; keep STT local if possible.
- **Chatbox** mode: you can disable the per-line **notification sound** in the inspector (VRChat OSC third argument on `/chatbox/input`; see VRChat’s OSC docs).

## Related

- [STT](stt.md)  
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — native IPC  
