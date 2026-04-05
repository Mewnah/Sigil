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

If your server requires an API key, enable keys in LibreTranslate (`--api-keys`) and paste the key in **API key (optional)** in Sigil. The plugin sends it as the `api_key` JSON field on `translate` / `detect` and as the `api_key` query parameter on `GET /languages`.

## Other stacks

Any proxy or service that implements the same routes and JSON shapes works (e.g. self-hosted gateways, Argos-based deployments, or custom FastAPI wrappers). For heavy models (ctranslate2, etc.), run them behind this contract on a second machine or GPU to avoid contending with VRChat/OBS.

## On-device alternative

Sigil also offers **On-device (WebView Translator API)** when the embedded WebView exposes the experimental `Translator` / `LanguageDetector` globals (varies by WebView2 / browser version). If that backend fails to start, use this sidecar or **Azure**.
