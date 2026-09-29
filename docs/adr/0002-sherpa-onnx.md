# ADR 0002: sherpa-onnx 1.13.8 C API as the in-process CPU speech runtime

- **Status:** accepted (go: STT and TTS smoke tests pass through the C API on the dev rig, next to
  Sigil's own CUDA sessions; the streaming Nemotron 3.5 and Qwen3-ASR packages are supported by the
  pinned runtime but were not run here, see Consequences)
- **Date:** 2026-09-29
- **Workstream:** P1-G1
- **Plan decision:** "STT runtimes: sherpa-onnx vs transcribe.cpp vs own ORT" (role 1, CPU tier),
  "ggml isolation and whisper-rs" (nothing load-time linked), research §1.3 runtime spike

## Context

The CPU tier of the new speech stack (research §1.1, §2.1) needs Nemotron 3.5 streaming partials,
SenseVoice / Qwen3-ASR / Parakeet finals, Moonshine, spoken-language ID and Kokoro / Kitten / Piper
TTS with real C espeak-ng. sherpa-onnx (Apache-2.0) ships official int8 packages for all of them
and prebuilt Windows binaries. Constraints from the plan:

- Sigil.exe already links ONNX Runtime **1.28.0 statically** (`ort` rc.13, ADR 0001) and owns every
  GPU session. A second ORT must not collide with it, must stay CPU-only, and must not be load-time
  linked: the app starts, and every existing feature works, with no pack installed.
- Windows loads DLL imports by base name. `C:\Windows\System32\onnxruntime.dll` exists on current
  Windows 11 (Windows ML; **1.17** on the dev rig, `1.17.260613-0100.1.os-germanium`), so a DLL that
  imports `onnxruntime.dll` can silently bind to it.
- Real users have non-ASCII Windows user names (`José`, `山田`), and `%APPDATA%` contains them.
- Network gentleness: one small download, hash-pinned.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Official prebuilt `win-x64-shared-MT-Release-lib` (8 MB archive: `onnxruntime.dll` 1.28.2 + `sherpa-onnx-c-api.dll`), C API through libloading with vendored bindgen output | Upstream binaries, no build; static CRT (no VC++ redist); imports only `onnxruntime.dll`, KERNEL32, ADVAPI32; C ABI so no CRT / C++ ABI coupling with Sigil; one pinned version | Two ORT copies in the process (~18 MB DLL, separate thread pools); no spin-wait control through the C API; C++ exceptions do not cross the C API (a corrupt model aborts the process) | Chosen; measurements below |
| B. Official `sherpa-onnx` Rust crate, `shared` feature against Sigil's ORT | One ORT | Links sherpa at load time; the crate's build downloads archives during `cargo build` (network rule); couples sherpa's C++ to Sigil's ORT ABI; `/MT` vs `/MD` mix | Rejected |
| C. Static `win-x64-static-MT-Release-lib` (123 MB) linked into Sigil.exe | No DLLs | Duplicate ORT symbols against the static ORT at the MSVC link; exe +100 MB even for users who never use it | Rejected |
| D. sherpa `cuda-13.x` build (478 MB) | GPU decode | GPU belongs to Sigil's ORT / transcribe.cpp (plan roles); huge pack; two CUDA ORTs in one process | Rejected (CPU tier only) |
| E. Fallback: STT CPU on transcribe.cpp CPU, TTS on Sigil's ORT with pure-Rust espeak-ng | No second ORT | Loses sherpa's official packages (Nemotron streaming, SenseVoice tags, Kokoro multi-lang lexicons) | Not needed (A passed) |

## Decision

1. **Pack.** Catalog entry `runtime/sherpa-onnx` (`src-tauri/model-catalog/runtime-sherpa.json`,
   status `published`, upstream asset): `sherpa-onnx-v1.13.8-win-x64-shared-MT-Release-lib.tar.bz2`,
   8,032,957 bytes, sha256 `b8eedf41bd6d3779218887b48367bb7a3ece5aaa7667f01f69ee823a12b0a9e7`
   (GitHub's asset digest, re-hashed locally), extracted with `strip: 2` and
   `include: [onnxruntime.dll, sherpa-onnx-c-api.dll]` into
   `<app_data>/runtimes/sherpa-onnx/1.13.8/cpu/` (22.4 MB). `onnxruntime_providers_shared.dll` and
   the C++ API DLL are not installed: no execution provider can ever load inside sherpa.
2. **Bindings.** `speech_rt/sherpa/bindings.rs` is bindgen 0.72.1 output for v1.13.8 `c-api.h`
   (sha256 `2a1b9508…b7f4ce`), dynamic-loading mode, allow-listed to the 46 functions the wrapper
   calls, with bindgen's compile-time size / alignment / offset checks. `SherpaOnnxCApi::new`
   (default DLL search) is stripped. Reproducible with `scripts/runtime-packs/sherpa/gen-bindings.ps1`.
3. **Loading** (`speech_rt/sherpa/lib.rs`, `SherpaRuntime::load`):
   refuse when another `onnxruntime.dll` module is already loaded → load the pack's
   `onnxruntime.dll` by full path (`native_runtime::load_library_in_dir`) → require ORT C API level
   28 from its `OrtGetApiBase` → load `sherpa-onnx-c-api.dll` the same way and resolve every
   allow-listed symbol → require `SherpaOnnxGetVersionStr() == "1.13.8"` (the config structs grow
   between releases, so any other version is refused) → require that the `onnxruntime.dll` module
   the process now resolves is the pack's file. The runtime is cached per process and never
   unloaded; failed loads are not cached.
4. **Wrappers**: `OnlineRecognizer`/`OnlineStream` (streaming), `OfflineRecognizer::transcribe`
   (SenseVoice, transducer / NeMo Parakeet, Qwen3-ASR, Moonshine v1/v2, Whisper, Canary, Cohere
   Transcribe, FireRed, paraformer, CTC families), `OfflineTts::generate` (Kokoro incl. multi-lang
   lexicons and rule FSTs, Kitten, VITS/Piper, Matcha; per-sentence chunk callback, cancel flag,
   panics never unwind into C), `SpokenLanguageId`, `SpeakerEmbeddingExtractor`. Configs are serde
   types (P2 stores them); `detect.rs` builds them from an official package folder.
5. **Rules.** `provider = "cpu"` in every config. Threads: 2 by default, 1 for streaming recognizers
   (see spin-wait below), at most 4. Every native object serializes its calls (per-object lock), so
   the wrappers are `Send + Sync`; TTS generation is additionally serialized process-wide because
   espeak-ng keeps global state. Every referenced file is checked before a create call. Paths are
   passed as 8.3 short names when they contain non-ASCII characters (sherpa and espeak-ng open files
   through the narrow C runtime). Models that do not come from the hash-pinned catalog are first
   loaded in a child `Sigil.exe --headless engine-probe sherpa -- load_only=1 config=<file>`
   (`probe::validate_in_child`), which loads exactly the serialized config the caller will load
   (never a re-detection, so precision, model type, language and decoding match).
6. **Refused before any native call** (sherpa answers these with `exit(-1)` or a fatal abort, which
   would take Sigil down):
   - modified beam search on anything but a transducer (SenseVoice: "Only greedy_search is
     supported"), and on a NeMo streaming transducer (Nemotron, cache-aware FastConformer). sherpa
     picks its streaming transducer implementation by the decoder's graph output count, so the
     wrapper reads that count from the ONNX file first (`onnx_info.rs`, no initializers read);
   - hotwords (per-stream or `hotwords_file`) unless the model is a transducer decoded with modified
     beam search ("Only transducer models support contextual biasing"); offline NeMo transducers
     (Parakeet) do support both. Qwen3-ASR takes hotwords only in its model config;
   - audio after `input_finished` on a streaming stream: sherpa's `Reset` keeps the feature
     extractor's input-finished flag and kaldi-native-fbank aborts on more audio, so
     `input_finished` is terminal (`reset` only follows an endpoint; the next utterance needs a new
     stream);
   - an offline decode of nothing or under 100 ms (`OfflineStream::decode` skips the model; the
     result is empty).
7. **Float options.** The C API reads a float of exactly 0 as "unset" (`SHERPA_ONNX_OR`). A
   deliberate 0 is passed as `1e-6` (`lib::C_FLOAT_ZERO`: VITS / Matcha noise, hotwords score).
   TTS `silence_scale` is clamped to `[0.01, 10]`, the range sherpa's `ScaleSilence` applies; below
   0.01 it skips scaling and keeps the model's full pauses, so "no pause" is 0.01. Endpoint rules
   must be positive (a large value keeps a rule from firing).
8. **TTS cancel.** A cancel flag already set before `generate` (or while it waited for the
   process-wide TTS lock) returns at once with no audio; otherwise generation stops at the next
   sentence batch.
9. **Probe.** `Sigil.exe --headless engine-probe sherpa [model] [wav] [--runtime-dir] [-- key=value]`
   reports versions, the bound `onnxruntime.dll`, load times, RTF, cores, idle CPU after decoding,
   working set, VRAM and the output; `cuda_load=<duty>` repeats the decodes while Sigil's own ORT
   keeps a GPU session busy (the benchmark's load generator) and checks both runtimes.

## Evidence

Dev rig H2 (Ryzen 7 7800X3D 8C/16T, RTX 4090, Windows 11 26200), debug `sigil.exe` (the sherpa and
ORT DLLs are upstream release builds, so the wrapper's build mode does not matter), sherpa-onnx
1.13.8 (`11afbd00`, 2026-09-10) with ONNX Runtime 1.28.2; Sigil's own ONNX Runtime 1.28.0. Timings
come from `scripts/runtime-packs/sherpa/probe-matrix.ps1` (5 runs each) on an otherwise idle machine
(CPU load 5-10% before the run); reports in `bench-results/sherpa/20260929-020202/`. Earlier runs
taken while a parallel CUDA build held the CPU at 100% were 1.5-5x slower and are not used.

| Check | Result |
| --- | --- |
| Install | `Sigil.exe --headless fetch runtime/sherpa-onnx` (with `SIGIL_LOCAL_ASSETS`) → verified, extracted, 22.4 MB |
| Runtime load | 3.2 ms; working set +3.6 MB; bound `onnxruntime.dll` = the pack's file; ORT 1.28.2 (Sigil's 1.28.0 unaffected) |
| SenseVoice int8 2024-07-17, `en.wav` (7.15 s) | "The tribal chieftain called for the boy and presented him with 50 pieces of code." lang `en`, emotion `neutral`, event `speech`; identical over every run |
| SenseVoice `ja.wav` (7.2 s) | "うちの中学は弁当制で持っていきない場合は50円の学校販売のパンを買う。" lang `ja` |
| SenseVoice RTF at 1 / 2 / 4 threads | 0.05 / 0.03 / 0.02 (en and ja alike; 370 / 205 / 130 ms per 7 s utterance; 0.99 / 1.99 / 3.94 cores while decoding) |
| SenseVoice model load / memory | 690-770 ms; +301 MB working set (237 MB `model.int8.onnx`) |
| Spin-wait after a decode (process CPU in the 2 s idle window) | 1 thread: 0.01 ms/s; 2 threads: 12.3 ms/s; 4 threads: 37.5 ms/s, i.e. about 25 ms of spinning per extra intra-op worker after every run |
| SenseVoice while Sigil's ORT runs a CUDA session | `cuda_load=0.7 cuda_preload=<CUDA 13.2 bind>;<cuDNN 9>`: Sigil's MatMul load ran 1,562 runs at 70.0% duty on **CUDA** (VRAM 338 → 388 MB); sherpa RTF 0.03 → 0.04; transcript identical in all 5 loaded runs; Sigil's CUDA canary passed before and after, its CPU canary after |
| Same, as a test | `real_runtime_coexists_with_sigil_cuda_sessions`: 101 CUDA canary runs on another thread during 3 SenseVoice decodes, transcript unchanged, both canaries pass afterwards |
| Memory with Sigil's CUDA session live | working set 921 MB / private 1.48 GB (CUDA context + cuDNN + SenseVoice); sherpa itself adds 0 MB VRAM |
| Kokoro int8 multi-lang v1.0, English sid 0 (6.6 s of speech) | 54 speakers, 24 kHz, RMS 0.07; load 1.0 s, +240 MB. **RTF 1.10 / 1.11 / 0.97 at 1 / 2 / 4 threads; first sentence after 2.4 / 2.6 / 2.2 s; "Hello there." (1.1 s) took 1.7 s.** Not real-time: the int8 graph barely scales with threads |
| Kokoro Japanese (`lang=ja`, sid 37) | 3.9 s of speech (RMS 0.08), RTF 1.3; sherpa logs skipped IPA diacritics (U+031E, U+0308): Japanese goes through espeak-ng phonemes, not Kokoro's misaki G2P |
| TTS chunking / cancel | one callback per sentence; the cancel flag stops after the current sentence and keeps its audio; a flag set before the call returns in under 200 ms with no audio; a panicking callback is reported as an error, never unwound into C |
| TTS pause scale | Kokoro `silence=0` (clamped to 0.01): 6.18 s of speech vs 6.60 s at the 0.2 default. Passing `1e-6` instead made sherpa log "Silence scale 0.000001 is not in [0.01, 10.00]. Skip scaling." and the output came out *longer* than the default (8.0 s vs 5.8 s in the test text) |
| Degenerate input | < 100 ms returns an empty result without a native call (through `transcribe` and through a stream's `decode`); 300 ms of silence gives SenseVoice's lone `。`, which the wrapper drops (punctuation-only text is empty) |
| Unsupported combinations | `-- hotwords=Sigil` on SenseVoice: the probe exits 1 with "the sense_voice model does not take hotwords" (sherpa itself would `exit(-1)` mid-decode); beam search on SenseVoice / paraformer / a 3-output (NeMo) streaming decoder is refused by `validate` (unit tests) |
| Per-stream `language` option | ignored by SenseVoice (`language=zh` on `en.wav` still returns English tagged `en`); SenseVoice's language lives in the recognizer config |
| Isolated first load | `-- validate_child=1 lang=en`: the child loads the exact final config (with the override) from a temp JSON file in 1.0 s. A corrupt `model.int8.onnx` kills the child with `0xc0000409` ("Rust cannot catch foreign exceptions") and the parent reports it and exits cleanly; loaded in-process, the same file aborts the process |
| Wrong pack | missing DLLs, non-DLL files, a real non-ORT DLL named `onnxruntime.dll` → `Err`, never a panic (unit tests) |

Wrong-package finding: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09` is **not** a newer
SenseVoiceSmall. It is converted from ASLP-lab WSYue-ASR `sensevoice_small_yue` (a Cantonese
fine-tune): on the same `en.wav` it returned spelled-out letters ("THE TRIVBAL CHIEF THIN … COOOD")
tagged `yue`. P2 must use the 2024-07-17 package (163,002,883 bytes, sha256 `7d1efa21…fd347e`) for
zh / en / ja / ko / yue and may offer 2025-09-09 only as a Cantonese option.

**Nemotron 3.5 and Qwen3-ASR are supported by the pinned runtime** (not run here: neither package
was in this workstream's approved downloads):

- Nemotron 3.5 ASR streaming 0.6B: official packages
  `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{80,160,320,560,1120}ms-int8-2026-06-11.tar.bz2`
  (~475 MB each; e.g. 160 ms: 475,273,363 bytes, sha256 `a81909a1…f7a61a`) in the `asr-models`
  release, published 2026-07-09, before 1.13.8. They are streaming transducers (encoder / decoder /
  joiner + tokens), which `OnlineModel::Transducer` covers; `detect` maps a `nemotron` package to it.
- Qwen3-ASR-0.6B: 1.13.8's C API has `SherpaOnnxOfflineQwen3ASRModelConfig` (conv frontend,
  encoder, KV-cache decoder, tokenizer folder, hotwords); official package
  `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2` (878,702,423 bytes, sha256 `393f8a14…78ee96`).
  The wrapper decodes greedily (temperature 1e-6, top_p 1.0, seed 42) so captions are deterministic.
- Also packaged for 1.13.8: Parakeet TDT 0.6B v3 int8 (`nemo_transducer`), Moonshine v2 "quantized"
  2026-02-27 (en/ja/ko/zh/es/ar/vi/uk), SenseVoice + FunASR Nano. "Parakeet Ultra" is a weight swap
  with no sherpa package; it needs a Sigil export through the transducer path.

Commands (repeatable; `probe-matrix.ps1` runs the whole table and saves every report):

```powershell
Sigil.exe --headless fetch runtime/sherpa-onnx
Sigil.exe --headless engine-probe sherpa <sense-voice-2024-07-17> <test_wavs\ja.wav> -- threads=2 idle_ms=3000
Sigil.exe --headless engine-probe sherpa <sense-voice-2024-07-17> <test_wavs\en.wav> -- cuda_load=0.7 repeat=5
Sigil.exe --headless engine-probe sherpa <kokoro-int8-multi-lang-v1_0> -- threads=2 out=kokoro.wav
cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa -- --include-ignored --test-threads=1
```

## Consequences

- **Go.** P2's `speech` plugin builds its CPU engines on `speech_rt::sherpa` and installs the pack
  through `model_store::ensure(["runtime/sherpa-onnx"])` (`lib::ensure_installed`). The app starts
  and every existing feature works with no pack installed (every entry point returns `Err`).
- **Spin-wait.** sherpa's C API cannot set ONNX Runtime's `allow_spinning`; each extra intra-op
  worker spins about 25 ms after every run. Streaming recognizers therefore default to 1 thread (at a
  160 ms chunk, 2 threads would spend ~15% of a core spinning); utterance decoders and TTS default
  to 2. P2 should measure Nemotron RTF at 1 thread before raising it.
- **One `onnxruntime.dll` per process.** Windows binds a DLL import to any loaded module of the
  same base name. If another component already loaded an `onnxruntime.dll` (for example Windows ML's
  System32 copy), `SherpaRuntime::load` refuses rather than let sherpa bind to it; P2 should load the
  runtime when a sherpa engine is configured (speech plugin init), before anything else could. A
  component loading `onnxruntime.dll` by name after sherpa gets the pack's 1.28.2, which serves
  every older ORT API level.
- **Crash surface.** C++ exceptions inside sherpa (e.g. ONNX Runtime rejecting a corrupt model) do
  not cross its C API, and several unsupported option combinations `exit(-1)`. The combinations are
  refused by the wrapper (Decision 6); catalog models are hash-verified; user-supplied folders go
  through `probe::validate_in_child` first, called with the final config P2 will load (after
  detection and every setting layered on top), never just the folder.
- **Upgrades** are deliberate: `pin-pack.ps1` → catalog, `gen-bindings.ps1` → bindings,
  `EXPECTED_VERSION`, size table in `tests.rs`, probe matrix (README in `scripts/runtime-packs/sherpa/`).
- **Kokoro int8 is not real-time through sherpa on CPU** (RTF about 1.1 even on an idle 7800X3D,
  and more threads barely help), so it cannot meet P2's "first sentence within 400 ms". P2 must
  measure the fp32 package (`kokoro-multi-lang-v1_0`, 350 MB) with `probe-matrix.ps1` before
  choosing the default Kokoro build, and should not ship int8 Kokoro as the CPU default. Japanese
  Kokoro through sherpa uses espeak-ng phonemes; compare it with the AivisSpeech / SBV2 paths
  before routing ja to it.
- **Manual checks still open (P2 / P7):** Nemotron 3.5 streaming at 160 ms with
  `chunk_ms=160 threads=1`; Qwen3-ASR-0.6B on `ja.wav`; a multilingual Whisper tiny for language ID
  and a CAM++ model for speaker embeddings (wrappers compile and are unit-tested, not model-tested);
  the probe matrix on H1 / H3 rigs.
- **Optional hardening:** a tiny C++ shim (`try`/`catch` around each sherpa call, returning an error
  code) would turn in-process model exceptions into `Err` instead of an abort; build it only if field
  reports show crashes that the catalog hashes and `validate_in_child` do not prevent.
- Fallback if a later sherpa release breaks this path: keep 1.13.8 pinned (the catalog and the
  version check make that the default); otherwise plan option E.
