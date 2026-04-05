# AI transform

![AI transform inspector](assets/ui-placeholder.svg)

## Purpose

Rewrite, summarize, or stylize text with an **LLM** (OpenAI, OpenRouter, or a **local** HTTP endpoint). Often applied to **STT** output before display or TTS.

## Open the inspector

Sidebar → **AI Transform**.

## Start and stop

- Inspector controls or bottom bar **AI**.
- **Keyboard:** `Ctrl+2` / `Cmd+2` toggles the transform service start/stop.

## Configuration

Set the **provider**, **model**, **API key** (if required), and **base URL** for local or proxy setups. Prompts and options are in the inspector; keep them aligned with your stream’s tone and latency budget.

## Tips

- **Local endpoints** reduce cloud cost and privacy risk; latency still depends on GPU/CPU.  
- Short, focused prompts reduce delay versus long system prompts.  
- If the service errors, check the **logs** panel and provider rate limits.

## Related

- [STT](stt.md)  
- [TTS](tts.md)  
- [Settings](settings.md) for related keys  
