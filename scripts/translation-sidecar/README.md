# Local translation (LibreTranslate-compatible API)

Sigil’s **LibreTranslate** translation backend talks to any server that exposes the same HTTP API as [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate):

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/languages` | List `{ code, name }[]` |
| `POST` | `/translate` | JSON `{ q, source, target }` (+ optional `api_key`) → `{ translatedText }` |
| `POST` | `/detect` | JSON `{ q }` (+ optional `api_key`) → `[{ language, confidence }]` |

The Rust plugin (`plugin:translate|*`) runs these calls from the Tauri host. Set **Server base URL** in the Translation inspector (no trailing slash required).

## Quick start: official Docker image

```bash
docker run -d --name libretranslate -p 5000:5000 libretranslate/libretranslate
```

Then in Sigil choose **LibreTranslate**, endpoint `http://localhost:5000`, and start the service.

## API key

If your server requires an API key, enable keys in LibreTranslate (`--api-keys`) and paste the key in **API key (optional)** in Sigil. The plugin sends it only as the `api_key` JSON field on `translate` / `detect`; it is never put in a URL, and `GET /languages` (public) gets no key. Requests time out after 20 s plus 10 ms per character (at most 60 s).

## LTEngine (LLM-backed, same API)

[LTEngine](https://github.com/LibreTranslate/LTEngine) serves the LibreTranslate API from a local LLM (Gemma 3 4B by default). In Sigil pick the **LTEngine** server preset: endpoint port **5050**, no key, source language **auto** (LTEngine detects the language itself). Start it with `ltengine`; `-m gemma3-12b` selects a larger model.

## Language codes

Sigil works with canonical language codes and maps them to the server's own list (read from `GET /languages` when translation starts), for example `zt` or `zh-Hant` for Traditional Chinese. A source language Sigil does not know is sent as `auto`. A server without `/languages` still works with plain ISO 639-1 codes.

## Other stacks

Any proxy or service that implements the same routes and JSON shapes works (e.g. self-hosted gateways, Argos-based deployments, or custom FastAPI wrappers). For heavy models (ctranslate2, etc.), run them behind this contract on a second machine or GPU to avoid contending with VRChat/OBS.

## Built-in local translation (no sidecar needed)

Since the P4 translation stack, Sigil can translate locally without any server you run yourself:

- **AI model (LLM)** on **This PC**: Sigil's own llama.cpp server (`llama-server`, started and stopped by Sigil, listening only on `127.0.0.1` with a random key) runs a GGUF translation model such as Hy-MT2 1.8B, downloaded from **Models & Runtimes**.
- **Firefox Translations (Bergamot)** and **NLLB-200** run on the processor with no network.
- **Your own server (Ollama, LM Studio, llama.cpp, vLLM)** in the AI model panel talks to any OpenAI-compatible `/v1/chat/completions` server; servers on your network are reached through Sigil's Rust proxy.

See [OPERATOR_REFERENCE.md → Translation backends](../../docs/OPERATOR_REFERENCE.md#translation-backends). This sidecar remains the way to plug in any LibreTranslate-compatible service.

## On-device alternative

Sigil also offers **On-device (WebView Translator API)** when the embedded WebView exposes the experimental `Translator` / `LanguageDetector` globals (varies by WebView2 / browser version). If that backend fails to start, use this sidecar or **Azure**.
