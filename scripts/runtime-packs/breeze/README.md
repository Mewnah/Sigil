# Breeze TTS 2 through Breeze-TTS-2.cpp (`runtime/breeze-server-vulkan`, P6-F)

Sigil's `breeze` voice engine (Text to speech → **Experimental GPU voices** → Breeze TTS 2) runs
[Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp)'s `breeze-server` (Apache-2.0; the
BreezeBlue weights are research and non-commercial) as the `tts-breeze` helper process. Engine code:
`src-tauri/src/services/tts_helpers/breeze.rs`, the shared runner and adapter in `tts_helpers/ggml/`.

**Outcome of the P6-F spike: experimental preset.** Nothing is built by Sigil.

| Catalog id | What | Source |
| --- | --- | --- |
| `runtime/breeze-server-vulkan` | `breeze-server.exe` and its libraries: the v0.1.0 Windows Vulkan release zip (74 MB). Also the processor variant (`--cpu`) | GitHub release, pinned by digest, **published** |
| `tts/breeze-tts-2-q8_0` | Q8_0 GGUF (3.6 GB, about 4 GB of video memory; the port's recommended quality) | `HoppouAI/Breeze-TTS-2.cpp@81b22ba`, pinned, **published** |
| `tts/breeze-tts-2-q4_k` | Q4_K GGUF (2.5 GB, about 3 GB of video memory) | same |

The GGUFs carry the vocoder; no other file is needed.

## How Sigil runs it

`breeze-server <model.gguf> --host 127.0.0.1 --port <auto> --ws-port -1 --voices-dir
<app data>\tts-helpers\breeze\voices [--cpu]`: no web UI, no CORS, no WebSocket port. `GET /health`
answers once the model is loaded.

The server takes **no key** and serves **one request at a time** (409 for a second). Sigil therefore
talks to it through its in-process adapter (`tts_helpers/ggml/adapter.rs`): a key-protected loopback
endpoint that refuses web pages (Origin) and foreign host names (DNS rebinding), turns each
OpenAI-style sentence into Breeze's native form request, sends one at a time, and reads an abandoned
answer to its end before the next request (Breeze-TTS-2.cpp issue #20: its lock stays taken when a
streaming client disconnects). Residual risk, recorded in the threat model request: while it runs,
other programs on this PC can reach the server's own loopback port; it never listens on the LAN
(process host's exposure audit stops a helper that does).

Per sentence (native form): `text`, `instruction` (the voice described or directed in words),
`cfg_scale` (1 by default; 2-3 for inline events such as `(laugh)`), `seed` (42 by default, so a
described voice stays the same), and for a cloned voice-library voice `ref_audio` (the clip) plus
`ref_text` (its transcript). The answer streams 24 kHz PCM about every 2 s of audio. English and
Mandarin only.

Not done in the spike: Breeze's saved voices (`POST /v1/voices`), which cut the first audio of a
cloned voice from about 900 ms to 280 ms; the adapter uploads the reference with every sentence.

## Manual speak test (P6-F verification)

```powershell
$env:SIGIL_GGML_ENGINE = "breeze"
$env:SIGIL_GGML_PACK_DIR = "<extracted breeze-tts-2-v0.1.0-windows-x64-vulkan.zip>"
$env:SIGIL_TEST_MODELS_DIR = "<folder with breeze-tts-2-q8_0.gguf>"
$env:SIGIL_GGML_VOICE = "<a voice-library id>"     # optional: clone it
$env:SIGIL_GPU_LOCK = "<scratchpad>\gpu_timing.lock"
cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
```

Research figure to compare with: about 1.2x real time at Q8_0 on an idle RTX 3060 (Vulkan), 280 ms to
the first audio with a saved voice.
