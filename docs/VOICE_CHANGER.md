# Voice changer architecture and VCClient parity

Sigil’s voice changer runs in the **`voice-changer`** Tauri plugin: **cpal** capture → process (built-in phase vocoder, **ONNX** file, or **WebSocket** sidecar) → optional post-DSP (formant tilt, phase vocoder refinement) → playback queue. It is **not** compatible with [w-okada/voice-changer](https://github.com/w-okada/voice-changer)’s Python **Socket.IO / REST** server as a drop-in; Sigil’s sidecar speaks **raw WebSocket binary PCM** as documented in [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md).

## Feature parity (conceptual vs VCClient)

| VCClient-style area | VCClient (typical) | Sigil |
|--------------------|--------------------|--------|
| **Chunk / batch size** | Client `inputChunkNum` × fixed block size | **`chunk_frames`**: allowlisted sizes from **160** up to **48000** (~**1 s** at 48 kHz) frames per channel, with **coarser steps** above 4096 for very slow hosts (see `ALLOWED_CHUNK_FRAMES` / `VOICE_CHANGER_CHUNK_FRAMES`). Changing value **restarts** the engine. **WebSocket / ONNX** should match server/model shape when possible; Sigil adapts when it can. |
| **Pitch** | Server + client | Phase vocoder (**±24** semitones) in **all** modes: on the mic in Built-in, or on converter/model **output** after WebSocket / ONNX returns audio (same post-DSP chain as brightness / vocoder refinement). |
| **Formant / brightness** | Model-dependent | **Formant tilt** [−1, 1] on processed audio (all modes where post chain runs). |
| **Input / output gain** | Client gain nodes | **`input_gain` / `output_gain`** (0–4) in Rust on the capture/playback path. |
| **Output buffer** | Prefill + max queue (worklet) | **`output_prefill_ms`**, **`output_max_queue_ms`** (converted to samples using **output device rate**). Independent of chunk size: they only govern the **playback deque** after processing. **Prefill** = minimum queued audio before cpal starts pulling to speakers (reduces underruns). **Max queue** = hard cap; excess drops oldest samples. Presets **low_latency** / **stable** fill templates; **custom** when edited. |
| **Silence truncation** | `volTrancate*` thresholds | Optional **RMS + consecutive silent chunks** trims playback deque toward safe minimum. |
| **Metrics UI** | buf / res | IPC **`get_voice_changer_metrics`**: **buffer_ms** (queued audio), **process_ms** (exponential moving average of per-chunk processing time so the readout is not jittery). |
| **Monitor output** | Separate monitor device + gain | **Yes** — second **cpal** stream: **`monitor_output_device`** (empty = off, or device name / `"default"`), **`monitor_gain`** (0–4) on **pre-main-gain** PCM; channel layout and sample rate are **adapted** (mix / linear resample) when the monitor device differs from the main output. |
| **Noise / AEC** | Browser constraints, VoiceFocus, etc. | **Not built in**; use **OBS / OS / mic vendor** filters or a virtual cable stack. RNNoise-class DSP remains a possible future plugin. |
| **Wire protocol** | Socket.IO + REST | **WebSocket binary** sidecar only; see sidecar script for framing expectations. |

## Chunk size vs playback buffer (ms)

- **`chunk_frames`** controls the **DSP / inference step** (how much audio is consumed per processing callback). Larger chunks tend to **lower CPU/GPU utilization per real-time second** (fewer invocations) but add **algorithmic/step latency**. They do **not** directly set the values of `output_prefill_ms` / `output_max_queue_ms`.
- **`output_prefill_ms`** and **`output_max_queue_ms`** control the **output ring buffer** in `OutputPlayback`: prefill is the startup cushion before the first non-silent output block; max queue bounds worst-case listening delay when production outpaces consumption. Both are translated to sample counts with the **output sample rate** (`voice_changer.rs`).

## ONNX chunk shape

Models should use **float32** input/output. Sigil reads the graph’s fixed element count when present and **adapts** mic chunks when sizes differ (mono/stereo and length). Prefer a model whose `N` matches **`chunk_frames × channels`** for minimal extra buffering; otherwise adaptation still applies when supported.

## Related

- [SERVICE_CONNECTIVITY_AUDIT.md](SERVICE_CONNECTIVITY_AUDIT.md) — where the voice changer sits in the host matrix.
- [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md) — WebSocket echo server and identity ONNX helper.
