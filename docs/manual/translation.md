# Translation

![Translation inspector](assets/ui-placeholder.svg)

## Purpose

Translate text (often from **STT** or chat) into another language for captions or downstream TTS.

## Open the inspector

Sidebar → **Translation**.

## Typical flow

1. **Start** the translation service when your pipeline is ready.  
2. Set **source** and **target** languages (options depend on configured provider).  
3. Ensure **STT** (or another text source) is producing text that feeds the same pipeline.

## Configuration

Providers and API keys vary by integration; use the fields in the inspector and any cloud account requirements. For engine-specific minimums, see [BACKENDS.md](../BACKENDS.md) where applicable.

## Tips

- Match **language codes** to what your STT engine outputs.  
- If output looks wrong, verify whether **interim** vs **final** text is being translated in your setup.

## Related

- [STT](stt.md)  
- [TTS](tts.md)  
- [AI transform](transform.md)  
