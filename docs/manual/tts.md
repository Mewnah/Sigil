# Text to speech (TTS)

![TTS inspector](assets/ui-placeholder.svg)

## Purpose

Speak text aloud with **Windows SAPI**, **Azure**, **VOICEVOX**, **Uberduck**, **browser** voices, or other configured backends. Often used with chat, macros, or transformed captions.

## Open the inspector

Sidebar → **Text to Speech**.

## Start and stop

- Inspector **Start** / **Stop** or bottom bar **TTS**.
- **Keyboard:** `Ctrl+3` / `Cmd+3` toggles TTS start/stop.

## Configuration (summary)

Each backend needs different fields (voice, endpoint, API keys). See [BACKENDS.md](../BACKENDS.md#text-to-speech).

## In-app input

A **TTS input** bar can appear above the bottom of the canvas (host) so you can type or send phrases to TTS without leaving the editor.

## Tips

- **VOICEVOX / Kokoro / similar** usually expect a **local HTTP** engine; start the engine before starting Sigil’s TTS service.  
- **Windows** backend uses native output device selection in the inspector.  
- On the **client mirror**, heavy TTS may be limited; prefer driving TTS from the **host**.

## Related

- [AI transform](transform.md)  
- [Translation](translation.md)  
- [Discord](discord.md) — outbound hooks may carry text  
