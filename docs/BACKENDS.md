# STT / TTS backend minimum configuration

Use this when testing or supporting users. Engines are selected in the Sigil host app (inspectors).

## Text-to-speech

| Backend | Minimum required |
|--------|-------------------|
| **Native (browser)** | Select a **voice** in Text to Speech settings. |
| **Windows** | **Output device** and **voice** (native Windows SAPI). |
| **Azure** | Speech **key**, **region/location**, **voice**, optional device. |
| **Uberduck** | **API key**, **secret**, **voice**. |
| **VOICEVOX / Kokoro / Melo / Chatterbox / Fish Speech** | **Endpoint** (and voice/speaker where applicable), often localhost for self-hosted stacks. |

## Speech-to-text

| Backend | Minimum required |
|--------|-------------------|
| **Whisper (Rust)** | Download or place a **model**; first run may fetch GGML weights. |
| **Vosk / Moonshine / etc.** | Per-inspector: model path, language, or API keys as shown in UI. |
| **Azure / Deepgram / native** | Keys or browser permissions as prompted in the STT inspector. |

## Remote client

The `/client` page does not run Tauri plugins; it syncs with the **host** over PeerJS. Configure STT/TTS on the **desktop host**, not in the browser client.
