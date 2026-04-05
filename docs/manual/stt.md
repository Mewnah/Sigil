# Speech to text (STT)

![STT inspector](assets/ui-placeholder.svg)

## Purpose

Turn microphone (or other) audio into **text** for captions, chat pipelines, and downstream **TTS** or **translation**. STT runs on the **host** (desktop app), not on the `/client` mirror.

## Open the inspector

Sidebar → **Speech to Text**.

## Start and stop

- Use **Start** / **Stop** in the inspector or the bottom **STT** control on the dashboard strip.
- **Keyboard:** `Ctrl+1` / `Cmd+1` toggles STT start/stop (when the shortcut is not captured by the browser or OS).

## Engines (summary)

Pick a **backend** in the inspector. Minimum requirements per engine are in [BACKENDS.md](../BACKENDS.md#speech-to-text).

| Engine class | Notes |
|--------------|--------|
| **Whisper (Rust)** | Local; model download on first use |
| **Vosk** | Local models via Tauri + browser loader |
| **Moonshine** | HTTP server (e.g. container) |
| **OpenAI-style HTTP** | Local sidecar `POST …/audio/transcriptions` |
| **Azure / Deepgram** | Cloud; keys in settings |
| **Native / Web Speech** | Browser speech APIs where available |
| **Chrome / Edge** | Optional path to open a **mic helper** page (`mic.html`) for browser-based capture |

## Audio input

For native engines, choose the **input device** (including **System Default**). Rust-backed device lists apply to the desktop build.

## Tips

- Prefer **local** engines for latency-sensitive VRChat or game audio.  
- For **OpenAI-compatible** local servers, confirm they accept **non-streaming multipart** uploads; see BACKENDS “Upstream verification.”  
- Use **mute** in the STT UI if you need to pause recognition without stopping the service (behavior depends on build; watch the status indicator).

## Related

- [Translation](translation.md) — can run on STT output  
- [AI transform](transform.md) — optional rewrite of transcribed text  
- [OBS](obs.md) — showing captions in a browser source  
