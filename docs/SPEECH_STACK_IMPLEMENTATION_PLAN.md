<!-- Working plan generated 2026-09-28 by a multi-agent architecture pass over docs/SPEECH_STACK_RESEARCH_2026-09.md. Not synced to public/docs. P1, P2, P3, P4 and P5 are complete: the sections '## P1 outcomes' to '## P5 outcomes' (binding amendments) after 'Open questions' amend the later workstreams; read them before starting P6+ work. -->
# Sigil Speech Stack 2026-09: merged master implementation plan (every research item; Sigil-owned voice engine built to beat all other engines, proven by its own benchmark)

## Principles
- Quality first, then local, then free. Use the runtime each model runs best in. Ship defaults only from Sigil's own measurements (§8), never from vendor numbers.
- Sigil owns the premium path. The main voice-changer engine is Sigil's in-process Rust/ORT engine. Zero-shot, singing, partial decode, CUDA graphs, RefineGAN and custom embedders are rebuilt inside it where feasible. Engines that cannot be ported run as models inside Sigil's harness, which gives them the same latency budget, metrics and TTS mixing. They are never the recommended tier.
- Prove it with numbers. A benchmark checked into the repo measures latency p50/p99, per-block compute, VRAM, CPU and quality against an offline ceiling, under a GPU/CPU load simulator and in real VRChat+OBS sessions. The autotuner uses the same harness to pick settings per PC.
- Always shippable. Every phase ends green on pnpm typecheck, pnpm i18n:check, pnpm test:unit, pnpm build, cargo clippy --all-targets (no new warnings in touched files) and cargo test. New features default OFF or ship as new backend values. Persisted enum values are never renamed or removed.
- Front-load irreversible decisions. P1 holds the gate and spike workstreams that must produce an ADR plus compiling, setting-gated code before the phases that depend on them: ORT upgrade, sherpa-onnx, transcribe.cpp, RVC templates and process host.
- Native isolation. Sigil.exe links only its own static ORT and whisper-rs's static ggml. Every other runtime is a lazily loaded DLL in its own app_data folder, or a helper process in a kill-on-close Job Object bound to 127.0.0.1. The app starts, and all existing features work, with no pack installed.
- Real-time hygiene. STT always gets the raw mic, and the voice changer gets a conditioned copy. MMCSS goes on audio callback threads only. ORT spinning is always off. Latency budgets drop stale audio instead of queueing it. Epoch discipline applies in every Rust engine. No PCM crosses JSON IPC.
- Safe parallelism. Implementers own disjoint files. Hotspots change only in each phase's SCAFFOLD (before implementers) and INTEGRATE (after), one agent each. Scaffolds pre-split big shared files and create per-workstream fragment files: plugin command lists, capabilities, i18n staging and model catalog.
- Network gentleness. Nothing downloads during the build except the one-time ORT binary dist. Models and runtimes download at runtime through one verified, resumable, mirror-aware store, one file at a time. Agents fetch at most one large file at a time, sequentially, and only files listed in their external_assets.
- Security by construction. LAN WebSockets use an Origin allow-list, plus a token for non-loopback clients. There is no unauthenticated audio injection. Untrusted .pth/.zip files are parsed by an allow-list parser in an isolated worker. Keys go in headers only. LLM prompts fence source text as data.
- AGPL lens. Licenses appear only as catalog metadata, never as UI or doc warnings. Vendored code keeps its attribution headers.

## Decisions
- **How work lands without git history**: All work happens in place in the current working tree. There are no commits, branches, stashes, worktrees, resets or checkouts. Each phase scaffold first takes a read-only checkpoint outside the repo: `git diff --binary HEAD` plus a zip of untracked sources (excluding node_modules, target and dist) in the scratchpad. A failed gate is reverted from that checkpoint.
  - why: The hard constraint forbids commits, but the risky gates (ORT upgrade, voice_changer.rs split) still need a rollback point. This merges plan 1's idea into the in-place rule every plan shares.
  - fallback: If a checkpoint cannot be taken, that phase only runs workstreams that create new files, until one can be taken.
- **Removing hotspots from the registration points**: P0-S0 makes build.rs data-driven.
- Per-plugin command lists move to src-tauri/plugin-commands/<plugin>.txt, migrated verbatim.
- The AppManifest list stays inline.
- Each new plugin gets its own src-tauri/capabilities/<plugin>.json. tauri.conf.json does not enumerate capabilities (checked), so every file loads.
- UI strings go to scripts/i18n-staging/<WS>.json. A new non-destructive scripts/merge-i18n-staging.mjs merges them, then pnpm i18n:sync runs. The destructive merge-neural-i18n.mjs and merge-http-tts-i18n.mjs are retired.
- The model catalog is per-area fragments in src-tauri/model-catalog/*.json, which build.rs validates and embeds.
- Heavy entry points (bench, import worker, fetch, engine probe) go through a clap `--headless <cmd>` branch parsed before the port check.
  - why: This removes build.rs, migrated.json, the 21 locale files and the catalog as merge magnets. All three plans converged on this. Plan 1's build-time catalog embedding keeps adding a fragment a no-Rust-change operation.
  - fallback: If file-driven command lists misbehave, the integrator inlines the lists at INTEGRATE. The regenerated gen/schemas diff is the check.
- **ONNX Runtime version and the kittentts pin**: Step 1 (P1-S0a): vendor kittentts 0.3.0 into src-tauri/src/services/local_tts/kitten/. It is Apache-2.0, 2,614 lines, and already in the cargo registry. Keep its attribution, depend on espeak-ng 0.1 directly, and drop hf-hub; the Kitten hub download moves to the model store.

Step 2 (same gate): bump to ort =2.0.0-rc.13 (ORT 1.28.0) with features cuda, directml, nvrtx, tensorrt, half, preload-dylibs, copy-dylibs and ndarray, and migrate every call site. This was verified locally in ort-2.0.0-rc.13:
- with_cuda_graph, with_compute_stream, preload_dylibs and register_ep_library;
- ep::directml::api() exposing OrtDmlApi;
- IoBinding, with_external_initializer_file_in_memory and with_dimension_override;
- the Windows dist is cuda13-only (`cuda13,tensorrt,nvrtx,directml`).

The gate runs only after the user OKs the one-time pyke dist download (~hundreds of MB).
  - why: The kittentts pin blocks every ORT upgrade and parakeet-rs. Vendoring is zero-behaviour-change, unlike replacing Kitten. Upgrading before the new ORT code is written (VAD, Smart Turn, NLLB, the voice-engine GPU layer) avoids migrating everything twice. ORT 1.28 matches sherpa-onnx 1.13.8's bundled ORT, and CUDA 13 matches the dev toolkit (13.2 installed, driver 616.56).
  - fallback: If rc.13 isn't green within the gate, try rc.12 with ORT_CUDA_VERSION=12. If that fails, keep rc.11 on CUDA 12 with kittentts still vendored, keep CUDA graphs, IOBinding and the CUDA priority stream, and defer the DirectML high-priority queue and TensorRT-RTX.
- **CUDA 13 hardware floor**: Accept that the CUDA execution provider (EP) needs driver R580 or newer and a Turing or later GPU after the upgrade. A startup EP probe detects CUDA availability and falls back automatically to DirectML, with a visible reason.
  - why: Target users run RTX 3060/4060 (Ampere/Ada) and are unaffected. DirectML covers GTX 10xx, AMD and Intel.
  - fallback: A legacy CUDA 12 path only on request, via the rc.12 fallback.
- **STT runtimes: sherpa-onnx vs transcribe.cpp vs own ORT**: Runtimes have explicit roles, all behind one Rust AsrEngine trait in a new `speech` plugin.

(1) sherpa-onnx 1.13.8, CPU tier:
- Covers Nemotron 3.5 streaming, Parakeet Ultra, Qwen3-ASR-0.6B, SenseVoice, Moonshine v2 and spoken-language ID, plus Kokoro/Kitten/Piper TTS with real C espeak-ng.
- Uses its C API through libloading, with bindgen output generated once and vendored.
- The pinned prebuilt Windows x64 shared CPU archive ships as a runtime pack. onnxruntime.dll is loaded by full path first, then sherpa-onnx-c-api.dll.
- It is CPU-only and never registers GPU EPs.

(2) transcribe.cpp (MIT), GPU tier:
- Covers the GGUF finals: Qwen3-ASR-1.7B Q8_0/BF16, Cohere Q4_K, Voxtral Realtime Q4_K_M, Granite TurboCTC and Confucius4-R2T2.
- Built by Sigil as sigil_transcribe.dll packs in cpu, vulkan (default GPU) and cuda13 variants.

(3) Sigil's own ORT runs Silero, Smart Turn, the pitch models, denoisers, NLLB and the whole voice engine.

(4) whisper-rs stays for Whisper and Anime-Whisper.
  - why: Each runtime runs the official builds the research recommends: sherpa has official int8 packages and prebuilts, and transcribe.cpp is the only runtime for the GGUF GPU finals. Loading through libloading means nothing is load-time linked, the /MT-vs-/MD CRT question disappears across the DLL boundary, and there are no duplicate symbols with the static ORT in the exe. This takes plan 1's loading mechanism and plan 3's role split. Plan 2's transcribe-only approach puts the whole CPU tier on a single-maintainer runtime.
  - fallback: sherpa spike fails:
- STT CPU runs on transcribe.cpp CPU.
- TTS runs on Sigil's own ORT runner for Kokoro, Kitten and Piper, using the vendored pure-Rust espeak-ng.

transcribe.cpp spike fails:
- Run it as a helper process (sigil-asr-host.exe over pipes).
- Otherwise the GPU tier is deferred and finals use Qwen3-ASR-0.6B on CPU.
- **ggml isolation and whisper-rs**: whisper-rs remains the only statically linked ggml. Every other ggml or C++ engine is one of two things.
- In-process DLL: a uniquely named sigil_<engine>.dll that links ggml statically inside it, with GGML_BACKEND_DL=OFF, GGML_NATIVE=OFF and separate avx2/avx512 CPU variants picked by CPUID. Each loads from its own folder with LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|DEFAULT_DIRS).
- Helper process: llama-server, qwentts tts-server, sbv2 and the TTS shims. These may use GGML_BACKEND_DL with GGML_CPU_ALL_VARIANTS.

release.yml and src-tauri/.cargo/config.toml set GGML_NATIVE=OFF and GGML_AVX2=ON for whisper-rs-sys. whisper-rs retires in P7 only if transcribe.cpp matches it.
  - why: A second static ggml fails at the MSVC link. Two in-process engines that both depend on a same-named ggml.dll or ggml-cpu-*.dll bind to whichever loaded first, because the Windows loader matches by base name. Static ggml inside uniquely named DLLs avoids that, while shared cudart64_13 and cublas64_13 are the same version and meant to be shared. GGML_NATIVE=OFF prevents AVX-512 builds crashing on friends' CPUs.
  - fallback: Any engine that cannot be wrapped runs its own server binary under the process host.
- **Delivering GPU execution providers (today CUDA silently runs on CPU in releases)**: P0, on rc.11:
- Bundle onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll and DirectML.dll next to Sigil.exe through bundle.resources, staged from target/release.
- NVIDIA runtime libraries (cudart, cublas, cublasLt, cufft, cuDNN 9) come from NVIDIA's redist into app_data/runtimes/nvidia-cuXX. Their sha256 comes from the redistrib JSON. They are preloaded with SetDefaultDllDirectories and AddDllDirectory before the first CUDA session.
- Every GPU EP uses error_on_failure followed by an explicit CPU retry, and the active EP plus the reason is shown in the UI.
- NSIS per-user becomes the recommended installer; both NSIS and MSI are uploaded.

P1, after rc.13: move the 144 MB CUDA provider out of the installer into the opt-in NVIDIA pack, loaded with register_ep_library if the gate verifies it. Otherwise the pack writes it next to the exe in the per-user install.
  - why: This is the smallest change that makes GPU work for users now. The provider DLL must sit beside the ORT module when statically linked, and rc.13's plugin-EP registration removes that constraint.
  - fallback: If providers fail to load from the resource directory, use NSIS per-user plus a pack. Last resort: ort load-dynamic with a full ORT runtime pack in app_data (changing EP family then needs a restart).
- **Model and runtime store**: New `model-store` plugin.

Catalog:
- Entries carry id, family, kind (model|runtime|template|bench), version, license (as a fact) and files[{path, size, sha256 or git_oid, sources pinned to a revision}].
- Also archive{zip|tar.bz2|tar.gz, strip, include}, requires[], min_vram and status (published|unpublished).

Downloads:
- Resume with HTTP Range into a stable <sha>.partial plus metadata.
- 30 s stall timeout and no total timeout.
- Streaming sha256.
- Mirror failover (HF, then hf-mirror, then ModelScope, then Sigil GitHub release, then a custom base URL) on connect errors, 403/404/429/5xx, HTML bodies or hash mismatch.
- One download at a time by default (at most 2, 1 per host).

Install:
- Safe extraction (no traversal or symlinks; entry, size and ratio caps) and a free-disk check.
- Atomic rename into app_data/models/<family>/<id>/ plus installed.json.
- Existing whisper/ and vosk-models/ files are adopted, with no re-download.

Interfaces:
- Events models:progress (≤4 Hz) and models:state.
- Rust API ensure(app, ids) / installed_path.
- TS Models & Runtimes tab plus a reusable <ModelRequirement>.

Dev and pinning:
- SIGIL_MODEL_CATALOG_OVERRIDE / SIGIL_LOCAL_ASSETS for developers.
- scripts/models/pin-catalog.mjs pins hashes from HF tree and LFS metadata, sequentially with ≥2 s spacing.
- Trust-on-first-use only for non-executable files in dev builds; release catalogs must be fully pinned.
  - why: It replaces two copies of the downloader and the ~/.cache hf-hub store, and adds hashes, resume and mirrors for friends. Sequential downloads respect the router's burst blocking. All plans agreed; plan 2's git-oid support for small non-LFS files is kept.
  - fallback: The whole-file download path with sha256 still works if Range or mirrors misbehave.
- **Helper processes and llama.cpp**: A new process_host module runs helpers with:
- CREATE_NO_WINDOW, and a Job Object with KILL_ON_JOB_CLOSE and an optional memory cap;
- a free 127.0.0.1 port and a random --api-key;
- a /health poll, logs piped to tracing, a restart policy, and shutdown_all on RunEvent::Exit.

LLM translation uses the official pinned ggml-org llama-server builds (vulkan default, cuda13 opt-in, cpu fallback) as a runtime pack. There is no in-process llama-cpp-2.
  - why: llama-cpp-2 in-process hits the ggml link clash, the MSVC >4 GB GGUF bug and cross-machine crashes. A helper isolates crashes and VRAM. The CSP already allows 127.0.0.1. All plans agreed.
  - fallback: Point the OpenAI-compatible client at a user-run Ollama, LM Studio or vLLM, through a Rust proxy for LAN hosts.
- **How RVC importer templates are produced and shipped**: A maintainer Python pipeline (scripts/vc-templates) uses the local torch 2.3.1+cu121 and onnxruntime 1.19.2 (verified installed; `pip install onnx` is still needed). It uses vendored MIT RVC/Applio model code (HiFiGAN-NSF, RefineGAN, MRF) with attribution and exports random-initialised split graphs per config: v1/v2 × 32/40/48k × f0/nof0 × vocoder.
- enc.onnx (phone+pitch to m_p/logs_p) over a fixed context T.
- flowdec.onnx (flow reverse + decoder) over a fixed return window R+margin. Its inputs are sliced m/logs, time-anchored noise, speaker vector g, f0, and a phase0 input for phase-continuous NSF. SineGen stays fp32.
- knn.onnx: exact GPU retrieval blend.

Variants and weights:
- Each graph ships in fp32 and in fp16-internal with fp32 I/O. weight_norm is removed before export.
- All weights are external initializers.
- layout.json maps .pth keys to initializer, shape, dtype and transform (identity, weight_norm_fold(dim), speaker-table extraction).

Gating and shipping:
- A parity test against PyTorch gates every graph: fp32 max-abs ≤1e-4, fp16 ≤5e-3. Partial decode must equal the tail of a full decode.
- The graph-only templates (≤30 MB total) are bundled as installer resources; updates arrive through the model store.

At import, Rust writes weights.f32.bin and supplies it with with_external_initializer_file_in_memory, casting to fp16 with `half` for GPU. No protobuf code is needed at runtime.
  - why: This is plan 3's split-graph design, which enables partial decode, fixed shapes for CUDA graphs, identical noise across overlapping ticks and phase-continuous excitation (the quality edge over competitors). It uses plan 1/2's bundling for zero-setup and offline use. Templates need only architecture code, so no bulk downloads.
  - fallback: If in-memory external data misbehaves, copy the graph next to weights.bin using the ONNX external-data file convention. A variant that fails parity falls back to a full-decode fp32 template, or is rejected with a reason code. BYO ONNX keeps working throughout.
- **Parsing untrusted .pth/.zip**: Sigil's own allow-list pickle VM for the torch zip format.
- Allowed GLOBALs: OrderedDict, torch._utils._rebuild_tensor_v2/_rebuild_parameter, float/half/bfloat16/long/int storages, and _codecs.encode. Anything else is rejected.
- Legacy non-zip pickles are rejected with a clear message.
- It runs in a child `Sigil.exe --headless vc-import-worker <job.json>` inside a Job Object with a 2–4 GB memory limit and a 60–120 s time limit.
- Zip guards: ≤64 entries, ≤2 GB, ≤200× ratio, no absolute or `..` paths, no links.
- pthrs is used only as a dev-dependency cross-check in tests.
  - why: Memory-safe Rust plus process isolation makes untrusted community files safe. pthrs is too young to be the production parser.
  - fallback: In-process parsing under the same limits if the worker path causes packaging problems.
- **Voice-changer engine architecture (main engine)**: Build the Sigil Voice Engine (SVE): a fixed-shape, latency-governed streaming pipeline over template slots.
- Shared content-encoder pool: ContentVec, SPIN-v2, zh/ja/ko HuBERT; fp16 on GPU, int8 on VNNI CPUs.
- F0 computed only over the new block plus a margin, with a rolling cache and F0 fusion.
- Exact GPU kNN retrieval.
- Partial decode.
- Time-anchored noise and phase-continuous NSF.
- IOBinding with device-resident buffers, and CUDA graphs per fixed shape.
- A highest-priority CUDA stream, and a DirectML high-priority D3D12 queue via OrtDmlApi (rc.13).
- A post stage: rms_mix, cepstral formant shift, limiter.
- An adaptive quality governor plus a hard latency budget.

EP ladder, chosen by the autotuner:
1. TensorRT-RTX (engine cache) or classic TRT.
2. CUDA with graphs.
3. DirectML-HP.
4. CPU: SwiftF0, int8 content encoder, fp32 generator, 1–2 threads.

The legacy full-decode ONNX path stays and gains fp16-I/O support.
  - why: Each competitor lacks at least one of partial decode, graphs+fp16, priority scheduling, a backlog bound and exact retrieval. SVE combines them all (see vc_best_engine).
  - fallback: Every lever can be switched off on its own and returns to the existing proven path. The benchmark decides what ships by default.
- **Zero-shot and singing inside Sigil's engine**: Singing:
- DDSP-SVC 6.3 as an in-engine VoiceConversionModel, using upstream encoder/velocity ONNX with a Rust Euler loop at a fixed step count, graph-captured, at 44.1 kHz, covering E2–D#7.
- RVC singing mode (wide F0 range, autotune).

Zero-shot:
- Sigil-ZS, an in-engine Seed-VC realtime port: exported content encoder, DiT estimator, vocoder and CAMPPlus, with a Rust CFM/Euler sampler, CFG and fp16 graphs.
- It takes references from the shared voice library.

Other families: MeanVC2 (experimental CPU zero-shot) and so-vits-svc 4.1 BYO. X-VC and StreamVoiceAnon run only through sidecar protocol v2, wrapped as a VoiceConversionModel. The P3 spike decides feasibility with offline parity numbers (ADR 0006).
  - why: The user asked for this directly: best-in-class capabilities inside Sigil's own engine, where they get its latency, VRAM and scheduling advantages.
  - fallback: Any family whose export fails parity ships through sidecar v2 (Seed-VC via seed-vc-realtime), still under Sigil's harness and budget, labelled experimental.
- **Voice-changer settings persistence**: Rust is authoritative. Settings are written atomically (temp file, rename, .bak) to app_data/voice-changer/settings.json for global mode, preset, devices, EP, block and budget, and to models/<slot>/slot.json for per-voice pitch, index, protect, formant, F0 method and rms_mix. The TS service stays non-persisted (no patchService) and hydrates from Rust.
  - why: This fixes settings resetting on every launch without growing src/core/schema.ts for each new parameter. Per-voice settings travel with the voice, and CLAUDE.md's non-persisted rule is kept.
  - fallback: A persisted BackendSchema slice added in an INTEGRATE step if Rust persistence causes problems.
- **Audio capture ownership**: One capture hub (audio/hub.rs) with one cpal stream per source, whether an input device or an output device opened for loopback. It fans out through rtrb consumers:
- raw 16 kHz mono to STT;
- the device-rate conditioned chain to the voice changer;
- the reference recorder and loopback.

The hub:
- timestamps blocks with QPC;
- registers MMCSS 'Pro Audio' once per callback thread and opts out of Win11 EcoQoS;
- drops a lagging consumer's oldest data;
- restarts on stream error under the same epoch;
- runs a device watcher (IMMNotificationClient via the windows 'implement' feature, with polling as fallback);
- has a Rust mute gate.

It stays on cpal 0.15, since rodio 0.18 pins it.
  - why: This keeps STT on the pre-VC signal (§1.2), allows mic and loopback to run concurrently, removes three duplicate captures and the JSON audio:chunk flood, and fixes silent device loss.
  - fallback: Legacy per-consumer streams stay behind an internal flag until soak tests pass. If hub fan-out adds more than 1 ms p99 to the VC, the VC uses an exclusive-consumer mode.
- **TTS audio path**: PCM contract: ITTSService gains synthesize(text, signal) returning AsyncIterable<PcmChunk>, plus a selfPlaying flag.

The TS sentence pipeline splits on . ! ? … 。 ！ ？ 」 and newlines, merges fragments, synthesizes N ahead in order, and supports cancel and skip.

One Rust tts_sink:
- persistent output per device, a queue with cancel and flush;
- binary IPC through tauri::ipc Channel/Response;
- a top-level route (device, voiceChanger or both), feeding the VC aux queue with back-pressure and mic duck or mute.

SAPI moves to SpMemoryStream on a dedicated STA thread in P6. Web Speech stays device-only.
  - why: This replaces six private AudioContexts, JSON byte arrays and per-clip streams, and allows routing every TTS backend through the VC (§5).
  - fallback: Backends not yet migrated use a legacy play() adapter.
- **Translation pipeline placement**: Text-level logic lives in TypeScript, unit-tested with vitest: utterance tracker, LocalAgreement-2, router with fallback chain and circuit breaker, prompt builder, glossary and rolling context, and a multi-instance pipeline factory. Engines follow a Promise/throw contract. DeepL, NLLB, Bergamot and the LLM proxy run in Rust (because of CORS, CSP and native inference). Bergamot runs as a helper exe (JSON lines over stdio) under process_host, so its C++ runtime stays out of the real-time process.
  - why: Consumers already live in TS and the logic is pure. The helper exe keeps the voice-changer process lean and isolates crashes.
  - fallback: If Bergamot won't build on MSVC, NLLB-600M int8 plus Hy-MT2 1.8B on CPU form the CPU fallback chain.
- **Other-player (loopback) topics**: New topics others.stt and others.translation, deliberately outside the text.* hierarchy. They are off by default, never fed to the chatbox or TTS by default, and translated in the reverse direction by a second pipeline instance. Readings are published on readings.stt.
  - why: pubsub-js bubbles text.* up to 'text', which would echo other players into the user's chatbox, TTS and history.
  - fallback: None needed.
- **LAN security rollout**: Stage 1 (P0):
- Origin allow-list on /pubsub and /peer.
- Refuse binary frames; remove the binary-frame-to-Whisper path.
- Allow-list inbound topics.
- Fix the XSS sinks (index.tsx error fallback, inputfield.html).
- Fix the bind-failure spin loop and the port pre-check mismatch.
- A kill switch for the Origin check.

Stage 2 (P6):
- A persistent random token (root key, regenerate button), required only for non-loopback remote addresses, so same-PC OBS keeps working.
- A bind mode (LAN or localhost).
- The token propagates to /client, mic.html, obs-captions, inputfield, OBS URLs and Link apps.
  - why: Cross-site WebSocket hijacking and audio injection close early at low regression risk. Token rollout breaks more clients, so it comes later with a full client test matrix.
  - fallback: A 'legacy LAN without token' toggle, off by default, shown with a caution label.
- **Default STT backend and zero setup**: P0: desktop users on the broken 'native' default migrate once to hardened Whisper (small-q5_1 with 6 or more physical cores, else base-q5_1) plus Silero, after one click to download.

P2: the new 'local' backend (routing by language, two-pass on GPU) becomes the default for new installs once P2's exit checks pass. Existing users get a one-time prompt, never a silent switch. The first start asks before the ~650 MB Nemotron download.

P7: final defaults from measurements.
  - why: 'native' fails in Tauri today. Zero setup is a research goal, but large downloads need the user's consent.
  - fallback: If the user declines the download, stay on Whisper base with Silero. If CPU benchmarks fail, the default becomes Moonshine v2.
- **Qwen3-TTS and ggml TTS engines**: Qwen3-TTS runs as a qwentts.cpp tts-server helper: Sigil-built packs (vulkan default, cuda13 optional) under process_host, called through the P2 OpenAI-compatible streaming client, with a clone registry fed from the voice library. It uses 0.6B automatically while the VC runs on 8 GB cards. Breeze, Magpie (NeMo-Speech.cpp), OmniVoice and s2.cpp each get a time-boxed spike, then either a helper preset or a documented no-go.
  - why: A helper process isolates young C++ ports from the real-time process and reuses existing server code.
  - fallback: Chatterbox Multilingual V3 on Sigil's ORT as the in-process cloning fallback, then Kokoro.
- **SteamVR overlay**: An always-on-top, click-through caption window first. It uses an early route that never runs ApiServer.init and closes when the main window closes. Then a native overlay with hand-written minimal FFI to openvr_api.dll, loaded lazily as a Background app (never launches SteamVR). Text is rendered in Rust with cosmic-text and tiny-skia, with CJK fallback and RTL, updated at 10 Hz or less.
  - why: openvr_sys links an import library; a load-time import would stop Sigil starting on PCs without VR.
  - fallback: The caption window, pinned in XSOverlay or OVR Toolkit.
- **Hosting Sigil-produced assets**: Runtime packs (sherpa, transcribe, bergamot, qwentts, ggml TTS), exported embedder, pitch and denoiser ONNX files, and zero-shot graphs are published as GitHub Release assets on Mewnah/Sigil under assets-vN tags. Agents only prepare scripts/assets/publish-assets.ps1 and the workflow_dispatch workflows; the user runs them. Until then, catalog entries stay 'unpublished' and can be met by local import or dev override.
  - why: Stable, hash-pinned, free and under the user's control. No agent publishes.
  - fallback: A Hugging Face repo under the user's account, or a local or LAN folder mirror via a custom base URL.
- **Testing and benchmark placement**: Add vitest (pnpm test:unit, also run in CI) for pure TS logic. Rust unit tests stay inline. Tests that need real models are #[ignore] unless SIGIL_TEST_MODELS_DIR is set. Benchmarks run through `Sigil.exe --headless bench <spec>` and IPC run_benchmark, with no lib/bin split. Results go to gitignored bench-results/ plus curated docs/benchmarks/*.md.
  - why: There is no TS test runner today. Routing, LocalAgreement, the splitter and migrations need tests, while CI must stay offline.
  - fallback: A separate src-tauri/src/bin/sigil-bench.rs using #[path] includes if the clap branch conflicts with Tauri.
- **Voice-changer worker priority**: MMCSS goes only on audio callback threads. The worker priority setting is auto, protect_game or protect_voice. auto means NORMAL on GPU EPs and BELOW_NORMAL on CPU. protect_voice is NORMAL plus HighQoS; protect_game is BELOW_NORMAL. The process opts out of EcoQoS. Admin rights are never requested.
  - why: This resolves research §4.3#3 without starving VRChat on CPU presets, and the benchmark shows the trade-off.
  - fallback: None needed.

## VC best engine
The Sigil Voice Engine (SVE) is one Rust-owned, in-process, latency-governed real-time conversion engine. It runs every voice users already have:
- RVC v1/v2 as .pth, .zip or ONNX, at 32/40/48 kHz, with or without f0, with HiFiGAN-NSF, RefineGAN or MRF vocoders and ContentVec, SPIN-v2 or JP/KO/ZH HuBERT embedders;
- w-okada and wok000 exports;
- so-vits-svc 4.1;
- DDSP-SVC singing;
- Seed-VC-class zero-shot voices from a clean 10 s clip.

All of them share one harness: fixed shapes, partial decode, CUDA graphs, fp16, priority GPU scheduling, a stale-audio budget and an adaptive governor. The aim is to beat w-okada VCClient, the Deiteris fork, Applio realtime, RVC WebUI realtime/VST, vc-rs, rvc-realtime, Seed-VC and X-VC on end-to-end latency and p99 stability under VRChat+OBS load, on VRAM and CPU cost, and on compatibility, at equal or better quality. Sigil's own benchmark proves this on identical models and clips, and a per-PC autotuner applies the winning configuration.
### Targets
- Rigs:
- H1: RTX 3060 12 GB or RTX 4060 8 GB with a 6-core CPU, running VRChat in a busy world plus OBS NVENC 1080p60.
- H2: RTX 4090 dev rig, with the benchmark's GPU/CPU load simulator at 70% duty.
- H3: a 6-core CPU only.

Unless stated otherwise, the reference config is a v2 40k model, a 40 ms block and RMVPE on the GPU.
- T1 end-to-end latency (mic to speaker through real WASAPI shared mode, measured with the impulse loopback meter):
- Low-latency: p50 ≤ 80 ms and p99 ≤ 110 ms on H1 under load; p50 ≤ 55 ms on H2.
- Quality: p99 ≤ 170 ms on H1 under load.
- Streamer (CPU): p99 ≤ 230 ms on H3 with VRChat running.
- Engine-added latency in the paced harness: Low-latency p99 ≤ 85 ms.
- T2 per-block compute:
- H2: p95 ≤ 6 ms.
- H1 idle: p95 ≤ 12 ms.
- H1 under load: p99 ≤ 20 ms (RTF ≤ 0.5), with ≤ 0.1% of blocks over budget in 10 minutes.
- Growing the extra context from 64 ms to 512 ms raises p95 by ≤ 30%.
- The GPU hot path makes zero heap allocations per block, checked by a counting-allocator test.
- T3 VRAM (process delta, NVML/DXGI):
- Low-latency, one voice: ≤ 650 MB.
- Each extra hot voice: ≤ +150 MB.
- Quality with the index resident on the GPU: ≤ 900 MB.
- Community engines use about 1–1.2 GB per model.
- T4 CPU:
- GPU presets: ≤ 0.25 cores on average.
- Streamer: ≤ 1.5 cores on average at an 80–128 ms block, RTF ≤ 0.6.
- VC enabled and user silent: ≤ 0.02 cores, with no GPU inference.
- ORT never spins.
- T5 fidelity against the offline ceiling (same model converting the whole utterance in-process):
- CAMPPlus speaker cosine ≥ 0.97.
- MCD within 0.5 dB.
- Voiced F0 RMSE ≤ 15 cents; V/UV error ≤ 3%.
- 0 block-boundary clicks in 10 minutes.
- CER of the converted speech (Sigil STT) ≤ offline + 1.0 pt.
- T6 head to head on the same model, corpus and rig:
- DNSMOS OVRL at least 0.05 above each of w-okada, RVC WebUI realtime and Applio realtime.
- Gross pitch error ≤ every competitor's.
- Blind A/B preference ≥ 60% (5+ listeners × 20 pairs).
- End-to-end latency ≤ 0.7× the best competitor at matched quality.
- vc-rs (13.9 ms p95 per 100 ms chunk) and rvc-realtime (15.7 ms per 60 ms block on a 3060 Ti) are reference rows; SVE must beat both on normalized RTF.
- T7 robustness:
- Backlog never exceeds the budget (default 300 ms, range 100–2000) plus one block.
- After an injected 3 s GPU stall, back within budget in ≤ 2 blocks.
- 30-minute soak under load with 0 underruns.
- An EP failure is surfaced within 1 s, and the EP actually in use is always shown.
- A device that returns resumes within 2 s.
- T8 compatibility:
- ≥ 95% of a user-supplied 40-model community corpus imports with fp16 relative L2 ≤ 1e-2 against PyTorch offline. The corpus spans v1/v2, 32/40/48k, f0/nof0, HiFiGAN/RefineGAN/MRF, SPIN and CJK embedders, and WebUI, Applio, Mangio and w-okada exports.
- The rest are rejected with a specific reason.
- A 200-file malformed .pth/.zip fuzz corpus causes 0 crashes or hangs.
- Import takes ≤ 10 s per model.
- Every existing BYO ONNX slot keeps working.
- T9 startup and switching:
- Hot voice switch within one block, with a crossfade.
- Cold slot load ≤ 2.5 s, including graph capture.
- TensorRT-RTX first engine build ≤ 60 s, then ≤ 3 s from cache.
- Zero-shot slot from a 10 s reference in ≤ 5 s.
- T10 in-engine capabilities:
- Sigil-ZS zero-shot end-to-end: p95 ≤ 300 ms on H1 idle and p99 ≤ 450 ms under load (upstream live runs 430–690 ms), with speaker similarity ≥ upstream offline for the same reference.
- DDSP-SVC singing over E2–D#7 with gross pitch error ≤ 2%.
- TTS routed through the voice plays intact at any length, with the mic ducked.
### Components
- Capture hub consumer: MMCSS callback threads, park/unpark wake (no sleep-poll), per-block QPC timestamps, device-follow, and any cpal sample format.
- VC-only input chain: input gain, optional denoiser (GTCRN at 16 kHz for lowest CPU; DeepFilterNet3 or DPDFNet at 48 kHz), and a TTS aux mixer. The aux mixer is a VecDeque with back-pressure, rubato resampling and mic duck/mute. STT keeps the raw feed.
- Latency budget plus adaptive governor. The budget drops the oldest whole blocks with a 5 ms fade and a SOLA reset while keeping exact length accounting; it applies to every family and sidecar. The governor steps quality down under load (index off, then less context, then SwiftF0, then a lighter denoiser) and back up with hysteresis.
- Fixed-shape stream harness (VcStreamHarness v2): exactly block_16k new samples per tick, a fixed context, a pre-accumulator that absorbs resampler floor/ceil jitter, NCC SOLA at model rate, and rubato sinc resamplers. All existing exact-length tests are preserved.
- Model families behind VoiceConversionModel with a manifest.format factory:
- SigilRvc: template slots, the premium path.
- LegacyOnnxRvc: full decode, fp16 I/O accepted; also emulates w-okada exactly for comparisons.
- Sovits.
- DdspSvc.
- SigilZS (Seed-VC realtime).
- MeanVC2.
- SidecarV2.
- BeatriceVst (optional).
- Shared content-encoder pool (ContentVec 768/L12 and 256/L9+final_proj, SPIN-v2, zh/ja/ko HuBERT) chosen per slot from checkpoint metadata. fp16 on GPU, int8 on VNNI CPUs, shared across hot slots.
- Pitch bank: SwiftF0 (CPU default after A/B), FCPE, RMVPE, real CREPE-tiny, streaming PESTO and Dio. Confidence-weighted F0 fusion (voicing from SwiftF0, pitch from RMVPE). Incremental F0 over the new block plus a margin with a rolling cache. Octave gate and median. Singing range up to 2.5 kHz. Auto-key suggests a transpose.
- Retrieval: exact GPU kNN graph (MatMul, TopK k=8, 1/d² blend) with the index resident on the device; parallel CPU IVF reader checked against pthrs.
- Split generator: enc over the context and flowdec over the return window (partial decode). Counter-based time-anchored Gaussian noise (×0.667). A phase0 input for phase-continuous NSF. Speaker vector g computed in Rust, which enables speaker blending.
- GPU layer:
- One ORT env with env allocators and a VRAM ledger.
- IOBinding with device-resident buffers and CUDA graphs captured per shape at warm-up.
- Highest-priority CUDA stream (cudart via libloading).
- DirectML on a high-priority D3D12 queue (DML1 via OrtDmlApi).
- TensorRT-RTX or classic TRT with an engine cache and build progress.
- error_on_failure everywhere.
- Post stage: rms_mix envelope matching, cepstral-envelope formant shift (replacing the one-pole tilt on the RVC path), a dB silence gate with hangover, and an output limiter.
- Importer: zip guard, isolated allow-list unpickler, metadata detection (version, sr, f0, embedder_model, vocoder), weight_norm fold, template selection and shape validation, weights writer, manifest sigil-rvc-1, speakers and index adoption, smoke inference, sha256 cache, and auto-import of the MIT default voice.
- Slot manager: drag-and-drop, per-slot settings, N hot slots within the VRAM budget, voice hotkeys, and weighted merge of same-architecture slots.
- Telemetry: a per-tick timeline (capture, start, end, enqueue, estimated playout); p50/p95/p99; dropped ms; the EP in use per session; VRAM.
- Autotuner: sweeps EP × block (20–100 ms) × F0 method × precision × context on a bundled clip. Picks the lowest-latency configuration with p99 RTF ≤ 0.5 and fidelity ≥ T5, and stores it per GPU and driver.
- Zero-shot and singing in-engine: Sigil-ZS CFM sampler with CFG and a CAMPPlus reference embedding cached per voice-library entry; a DDSP-SVC Euler loop. Sidecar v2 (async, timeouts, reference upload, rate negotiation) for X-VC and StreamVoiceAnon.
### Benchmark harness
**Location and entry points.** Code lives in src-tauri/src/services/bench/ (core in P1) and neural_audio/vc/bench (VC v2 in P3). There are three entry points:
- `Sigil.exe --headless bench <spec.json> --out <dir>`, dispatched before the port check, writing report.json and report.md;
- IPC plugin:bench|run_benchmark, which powers the in-app "Benchmark this PC" and the autotuner;
- cargo tests with stub models.

**Modes.** Every mode drives the exact production VcStreamHarness and engine factory.
- offline: throughput.
- paced: a simulated device clock at wall-clock cadence. It gives exact algorithmic latency from the harness accounting (block, SOLA, resamplers, prefill), plus dropouts.
- e2e-loopback: an impulse or chirp train played to an output and looped to the input over a virtual cable. It measures true latency by cross-correlation (±1 ms) and can time any external app, including competitors.
- contention: runs alongside a GPU hammer (an ORT conv/MatMul gpu_load.onnx on a normal-priority stream at a set duty cycle), CPU burner threads, and 2–3 s stall injection.
- live-record: 10 minutes of the in-app timeline during a real VRChat+OBS session, exportable by friends. This is the authoritative H1 evidence.
- compare: scores competitor output WAVs recorded from the same input, and drives RVC WebUI rtrvc and Applio realtime in the user's own Python environment through scripts/vc-bench. The w-okada algorithm is emulated exactly by LegacyOnnxRvc.

**Metrics, all in Rust.**
- Per-block percentiles, RTF and over-budget ratio.
- CPU from GetProcessTimes; VRAM from NVML (loaded dynamically) or DXGI QueryVideoMemoryInfo; the EP actually used.
- MCD from mel-cepstra (reusing the rmvpe.rs mel code); F0 RMSE, gross pitch error and V/UV error against offline RMVPE.
- CAMPPlus speaker cosine and DNSMOS P.835 ONNX, both fetched by the store as kind 'bench' and never needed by normal users.
- CER of the converted speech using Sigil's STT.
- A click detector: spectral-flux spikes at block edges.

**Offline ceiling.** The same model converts the whole utterance in-process.

**Corpus.** 20 consented user and friend clips, plus VRChat world-music bleed at -20 dB and a singing set. It lives in app_data/bench/corpus and is never committed; the manifest template is scripts/bench/corpus.example.json.

**Outputs and gates.**
- tools/listening-test is a local blind A/B page generator.
- CI runs stub-model plumbing and the allocation-free hot-path test.
- The dev-rig perf gate is scripts/bench/run-perf-gate.ps1, with target thresholds and a rule that a p95 regression over 5% blocks acceptance.
- Scorecards are published to docs/benchmarks/VC_BENCHMARKS.md.
### Beats competitors
- Extra context costs almost nothing. Partial decode (flowdec only over the return window) plus incremental F0 means a larger context barely raises compute. w-okada and RVC WebUI recompute and decode the whole context every chunk, which is why their quality settings stutter.
- No backlog by design. The latency budget drops stale input, the output queue is bounded, and I/O is event-driven on MMCSS threads. The RVC VST's 9 s drift (issue #2859) cannot happen, and p99 is measured rather than the average.
- Near-zero GPU overhead per block. Fixed-shape CUDA graphs, IOBinding, fp16 templates, a shared arena and a highest-priority stream, plus DirectML on a high-priority D3D12 queue, keep VRChat rendering from starving conversion. Deiteris and w-okada run eager with host round trips. TensorRT-RTX gives vc-rs-class speed without its 86–93 s rebuilds.
- Cleaner joins. Time-anchored noise and phase-continuous NSF excitation make overlapping frames identical across ticks, so SOLA seams disappear. Competitors redraw noise and reset sine phase every chunk.
- Better retrieval and pitch. Exact GPU kNN replaces approximate CPU IVF. F0 fusion removes octave errors. The engine also offers true formant shift, rms_mix, protect, a denoiser on the VC path only, and an adaptive governor that degrades gracefully instead of stuttering.
- The widest compatibility. Direct .pth/.zip import (w-okada needs ONNX), including Applio RefineGAN and MRF voices that w-okada and mainline RVC cannot run, per-slot SPIN and CJK embedders, w-okada/wok000 ONNX and so-vits.
- Cheaper to run next to a game. Encoder and pitch sessions are shared across hot voices (+≤150 MB per voice versus about 1–1.2 GB per model). Silence gating means no GPU work while quiet. CPU thread caps, spinning off and the EcoQoS opt-out keep VRChat's frame rate intact.
- One engine for every capability. Zero-shot (Sigil-ZS), singing (DDSP-SVC), CPU zero-shot (MeanVC2) and research sidecars all inherit the same harness, budget, GPU scheduling and TTS mixing, in one process with no Python.
- Tuned per user and proven. The autotuner picks the fastest safe configuration on the user's PC. EP failures are surfaced, never silent. Settings persist and devices are followed. The built-in benchmark publishes p50/p99, VRAM and quality against the offline ceiling, and can time competitors with the loopback meter.

## Phases
### P0 Stabilize: verified bug fixes, hardening and tooling
Fix every §6 bug and the extra bugs found while mapping the subsystems.
- Replace the RMS gate with Silero VAD and harden Whisper.
- Make HTTP STT send one request per utterance, with a Groq-safe limiter.
- Fix the voice changer's real-time path: condvar wake, MMCSS, drop-oldest, latency budget, aux back-pressure, persistence.
- Make GPU EPs actually ship and report failures.
- Land LAN security stage 1.
- Add the AivisSpeech preset.
- Add tooling: data-driven build.rs, i18n staging, vitest, diagnostics log, atomic settings.
Exit: All gates green, and all 80 pre-existing Rust tests still pass. | 10 minutes of silence plus keyboard noise produce 0 Whisper calls. HTTP STT sends ≤1 request per VAD utterance and ≤20/min on the Groq preset. | VC: a 60 s TTS clip through the VC plays intact. Worker idle < 1% of a core. A forced 2 s stall keeps backlog ≤ budget + 1 block. Settings and devices survive a restart. | A packaged NSIS build runs the VC on CUDA after the NVIDIA pack installs. Without the pack it runs on DirectML; without DirectML.dll, on CPU with a visible warning. | A foreign-origin WebSocket is rejected and the binary audio path is gone. The manual matrix passes: host, /client, OBS, mic.html, Link apps. | Every §6 row is verified fixed, with a test or a documented manual check. Translation finals are never dropped. Desktop users no longer default to the broken native STT.
### P1 Platform foundations and decision gates
The ORT rc.13 gate with kittentts vendored. The model store and its UI. The capture hub with device-follow. The native runtime loader and NVIDIA EP pack. System probe and resource governor. Process host. Benchmark harness core. The sherpa-onnx, transcribe.cpp and RVC-template spikes, each producing ADRs and compiling, setting-gated code.
Exit: Gates green on the chosen ORT version (ADR 0001), with kittentts and hf-hub gone from Cargo.lock. | `Sigil.exe --headless fetch <id>` and the Models tab install, resume, verify, cancel and delete entries sequentially. whisper/vosk files are adopted without re-downloading. | The capture hub serves Whisper and Vosk from one stream. Unplugging and replugging a USB mic recovers within 2 s. | An echo DLL loads lazily, and the app starts without it. process_host's children die when Sigil is killed. | `--headless bench` runs a stub model through the real harness in paced mode and emits report.json. | ADRs 0002–0005 record measured go/no-go decisions: sherpa, transcribe.cpp, RVC templates, process host/llama-server.
### P2 In-process speech engine: STT and TTS
Build the speech plugin pipeline: hub, then rubato, then Silero segmenter, then engines.
- sherpa CPU engines.
- Language router, two-pass decoding and Smart Turn.
- transcribe.cpp GPU tier.
- The 'local' STT backend with zero setup.
- Legacy STT engines on the new plumbing.
- TTS: sentence pipeline and unified Rust sink, in-process Kokoro/Kitten/Piper, and the generic OpenAI-compatible streaming client.
Exit: Gates green. The app starts with no runtime packs installed. | CPU-only: en gives Nemotron partials, zh gives SenseVoice finals, ja gives Qwen3-ASR-0.6B finals. The router never sends ja/zh/ko to Cohere. | On the dev GPU, two-pass finals replace partials under the same segmentId, ≤ 800 ms after VAD end for an 8 s utterance. | Kokoro starts playing after the first sentence (≤ 400 ms on CPU). Stop cancels. Every PCM backend honours the device and can route into the VC. | New installs default to 'local'. Native-backend users get a one-time prompt.
### P3 Voice changer: the Sigil Voice Engine core
Template matrix and base assets. The production importer. The engine's algorithmic core: partial decode, anchored noise, phase0, F0 cache, fixed shapes. The GPU layer: IOBinding, CUDA graphs, priority stream, DML high-priority queue, TensorRT-RTX. The pitch, retrieval and post stack. The runtime loop on the capture hub with denoiser, governor and hot slots. Benchmark v2 with autotuner. The VC frontend. The zero-shot and singing export spike.
Exit: Gates green, and every bundled template passes parity. | The benchmark scorecard on H2 with the load simulator meets T2, T3, T5, T7 and T9. Streamer CPU meets T4 with GPU EPs disabled. The H1 run is scheduled with the user or a friend. | The corpus reaches T8 (≥ 95% import), and the fuzz corpus causes 0 crashes. The MIT default voice works with zero setup. | The autotuner returns a working preset on both CPU and GPU. ADR 0006 records the zero-shot and singing export feasibility with parity numbers.
### P4 Translation stack
Shared OpenAI-compatible LLM client and transform upgrade. The llama-server helper with a GGUF catalog. The translation pipeline host: LocalAgreement-2, router with fallback, prompt builder with glossary, multi-instance. NLLB and DeepL. The Bergamot helper. Translation UI. Caption rendering that handles revising text.
Exit: Gates green, and vitest covers LocalAgreement, router, prompt builder and language mapping. | Hy-MT2 1.8B on the helper-launched llama-server translates finals. Interims never flicker (monotonic committed prefix), and finals are never dropped. Killing Sigil leaves no llama-server running. | Stopping the LLM falls back to Bergamot or NLLB within one timeout. The MT chrF table is recorded.
### P5 Voice changer families in-engine and the shared voice library
Family registry. DDSP-SVC singing. Sigil-ZS zero-shot in-engine. MeanVC2. so-vits and w-okada/wok000 compatibility. Sidecar protocol v2 in the harness (X-VC, StreamVoiceAnon). The shared reference-voice recorder and library. Family UI.
Exit: Gates green. | Zero-shot from a recorded 5–10 s clip meets T10 on H2 (H1 scheduled). Singing tracks up to D#7. Both are scored against their offline ceilings by the benchmark. | A stalled sidecar never blocks the loop, and its stale audio is dropped by the budget.
### P6 Streamer features, extended engines and free cloud
Other-player loopback with reverse translation. Caption window and SteamVR overlay. VRChat OSC: MuteSelf, chatbox pacing, Smart-Turn commit, tags. LAN auth stage 2. Qwen3-TTS and experimental ggml TTS. SBV2/AIVMX and GPT-SoVITS. Edge and Gemini TTS with fallback. Pocket and Chatterbox-ort. SAPI to PCM with mic mute during TTS. Gemini Live STT and translate. Readings and the music gate. Optional Beatrice VST3 host.
Exit: Gates green, every feature OFF by default, and the full manual feature matrix passes. | The LAN token matrix passes: host, LAN /client, OBS, mic.html, Link apps. Foreign-origin and unauthenticated LAN clients are rejected. | Loopback captions appear only on others.* topics. Qwen3-TTS clones from a library clip and streams into the sink. An Edge or Gemini outage falls back to Kokoro within 1 s.
### P7 Measure, set defaults, document, release
Run the §8 campaign (STT WER, CPU speed, GPU under load, MT blind A/B, TTS and VC listening tests, competitor head-to-head). Set router tables and presets from the results. Finish the docs. Publish packs with finalized hashes. Decide whether to retire whisper-rs. Run the packaged smoke test on a clean VM.
Exit: docs/benchmarks/VC_BENCHMARKS.md and SPEECH_BENCHMARKS.md hold dev-rig numbers plus at least one H1 or CPU-only friend export. They state which of T1–T10 are met, with remediation for any gaps. | Catalog and preset defaults equal the measured winners, and approved default flips are applied with migrations. | Every release catalog entry is hash-pinned. The packaged smoke passes with no GPU and no packs. The docs are current.

## Global rules
- Work in place in C:/Users/chris/Desktop/Sigil. Never commit, branch, stash, create worktrees, reset, checkout, clean or restore. Only read-only git (status, diff) is allowed, for checkpoints. Never revert another agent's uncommitted change.
- Hotspots are edited only by the phase SCAFFOLD or INTEGRATE agent: Cargo.toml and Cargo.lock, build.rs, main.rs, services/mod.rs, capabilities/*, tauri.conf.json, src/core/schema.ts, services-registry.ts, src/core/index.ts, servicePanels.tsx, SigilNavigation.tsx, inspector/index.tsx, src/types.ts, public/i18n/*, package.json and pnpm-lock.yaml, ci.yml, .gitignore, and the module declarations in neural_audio/mod.rs and vc/mod.rs. Implementers put hotspot needs in their final report and stub around them.
- Edit only your owned_paths and new_paths. Read-only use of other modules' public APIs is fine. Signatures frozen by a scaffold change only through INTEGRATE.
- Register Tauri commands in the #[command] fn, in generate_handler!, and in src-tauri/plugin-commands/<plugin>.txt. A new plugin also gets its own src-tauri/capabilities/<plugin>.json.
- UI strings go only into scripts/i18n-staging/<WS>.json (English). INTEGRATE runs node scripts/merge-i18n-staging.mjs and then pnpm i18n:sync. Never run the retired merge-*-i18n.mjs scripts. Use logical RTL utilities.
- Each phase ends green on: pnpm typecheck, pnpm i18n:check, pnpm test:unit, pnpm build, cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets (no new warnings in touched files), and cargo test --manifest-path src-tauri/Cargo.toml (dist/ must exist). Playwright ($env:PW_PREVIEW_ONLY=1) runs when routes or index.tsx change. Keep your own files compiling after each coherent edit. Use at most 3 CARGO_TARGET_DIRs per phase under %LOCALAPPDATA%/sigil-target/.
- New features default OFF or ship as new enum values. Persisted enum values are never renamed or removed (zSafe resets them); map legacy values in start(). Add new keys rather than retyping old ones.
- Nothing new is load-time linked. DLLs load lazily via libloading with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR from app_data/runtimes/<name>/<ver>/<variant>/. In-process ggml DLLs link ggml statically with unique DLL names, /MD and GGML_NATIVE=OFF. Helpers run in a KILL_ON_JOB_CLOSE Job Object with CREATE_NO_WINDOW, bind 127.0.0.1 with a random key, and log to the file log. The app must start with no pack present.
- ORT: all Sigil sessions are built through neural_audio/ort_session.rs after ensure_ort_environment, with spinning off. Real-time sessions use 1–2 threads; background sessions are capped at 2 threads. GPU EPs use error_on_failure, then an explicit CPU retry, and report the EP actually in use. sherpa's bundled ORT stays CPU-only inside its DLL.
- Real-time: no allocation, locking, logging or IPC in audio callbacks (rtrb only). MMCSS goes on callback threads only, never on inference workers. Never request admin. STT consumes the raw hub feed; the voice changer consumes the conditioned copy.
- Epoch discipline: Rust engines bump the epoch under the stop lock on every start and stop, re-check it before each emit, and emit while the epoch is current. TS drivers honor startAbortSignal and use listenWithAbort. Inference and I/O commands are async with spawn_blocking. No PCM goes over JSON IPC.
- Network: no bulk or parallel downloads during development. Fetch one large file at a time, only files listed in your external_assets. Use pnpm install --network-concurrency=2 and single pip packages. Hash lookups are sequential with ≥2 s spacing. Never clone model repos. Prefer `Sigil.exe --headless fetch <id>` or SIGIL_MODEL_CATALOG_OVERRIDE/SIGIL_LOCAL_ASSETS. Tests never touch the network; model tests are #[ignore] unless SIGIL_TEST_MODELS_DIR is set.
- Voice-changer changes record benchmark results before and after (Sigil.exe --headless bench). A p95 regression over 5% blocks acceptance.
- Security: parse untrusted archives and pickles only with the allow-list parsers in the isolated worker. API keys go in headers, never in URLs or logs, and never use build-time VITE_/SIGIL_ secrets for user keys. LLM prompts fence source text and strip reasoning before publishing. Other players' speech uses others.*/readings.* topics, never text.*. src/client stays free of Tauri execution.
- No license or commercial-use warnings in the UI or docs. Licenses appear as catalog facts only. Vendored code keeps its attribution headers.
- Docs are edited only by INTEGRATE and P7-C, in the docs/ sources (never public/docs or public/manual). ADRs belong to their workstream.
- Every workstream's final report lists: files changed, contracts relied on, hotspot requests, doc deltas, the i18n staging file, external assets fetched with their sizes, and benchmark numbers where relevant.

## Deferred/risky
- **ort rc.13 upgrade (CUDA 13 only on Windows; one-time pyke dist download)**: It needs the user's consent for the build download, carries API churn, and drops the CUDA EP on pre-Turing GPUs or drivers older than R580. → The P1-S0a gate runs with checkpoint rollback. On failure, use rc.12 (CUDA 12/13). If that fails too, stay on rc.11, keeping CUDA graphs, IOBinding and the CUDA priority stream while deferring the DirectML high-priority queue and TensorRT-RTX. DirectML covers old GPUs automatically, with a visible notice.
- **CUDA provider DLL placement with static ORT**: The provider must sit next to the ORT module, and MSI per-machine installs are not writable. → P0 bundles it (installer grows ~165 MB). P1 moves it to an app_data pack loaded with register_ep_library if that is verified. Otherwise NSIS per-user writes it next to the exe. Last resort: ort load-dynamic with a full ORT runtime pack.
- **transcribe.cpp on Windows (undocumented CUDA build, single maintainer, ABI churn, R2T2 support unknown)**: The whole GPU ASR tier depends on it. → Pin a commit, build packs ourselves with vulkan as default and cuda13 built locally, and wrap only the C API in a shim. If it proves unstable, run it in a worker process. If that also fails, defer the GPU tier and use Qwen3-ASR-0.6B CPU finals. The capability degrades but still ships.
- **sherpa-onnx in-process next to Sigil's static ORT (possible System32 onnxruntime.dll conflict; Nemotron 3.5 package availability)**: Two ORT copies run in one process, and some models may not be packaged for sherpa. → Load the pack's onnxruntime.dll by full path first, CPU-only. If the spike fails, STT CPU uses transcribe.cpp CPU and TTS uses Sigil's own ORT runner with pure-Rust espeak-ng. If Nemotron is missing from sherpa, the router takes partials from transcribe.cpp CPU.
- **Seed-VC realtime in-engine export (no public ONNX; DiT/CFG export; ~1.5 GB weights)**: Export feasibility is unproven. → The P3-H spike decides. On no-go, ship zero-shot through sidecar v2 using seed-vc-realtime, still under Sigil's harness, budget and TTS mixing. Retry the in-engine port later.
- **X-VC and StreamVoiceAnon in-engine**: They are PyTorch codec-LM research code (~5 GB) with no realistic ONNX path. → Sidecar v2 only, wrapped as a VoiceConversionModel so they inherit the budget and metrics. Labelled for 12 GB+ GPUs and as a privacy mode.
- **MeanVC2**: Low confidence, Mandarin-trained, 16 kHz. → An experimental family, kept only if it is real-time on CPU and passes the A/B. Otherwise the CPU tier is RVC Streamer with SwiftF0 and the int8 encoder.
- **Beatrice 2**: Its beatrice.lib is closed, it has been at release candidate since August 2025, and it has no ONNX port. → P6-M is optional and time-boxed. If dropped, the CPU tier is RVC Streamer tuned by the autotuner, plus the documented option of running an external voice changer over a virtual cable.
- **Bergamot on MSVC**: marian/intgemm builds are fragile on MSVC. → A helper exe build with ADR. On no-go, the CPU fallback chain is NLLB-600M int8 plus Hy-MT2 1.8B on llama-server CPU.
- **Young ggml TTS runtimes (Magpie/NeMo-Speech.cpp, Breeze, OmniVoice, s2.cpp alpha) and qwentts.cpp maturity**: Immature ports with uncertain servers. → Each is marked experimental, with a per-engine no-go note. Fallbacks: user-run servers via the OpenAI or Fish presets, Chatterbox-ORT for in-process cloning, then Kokoro and Edge.
- **sbv2_core pins its own ORT**: A second static ORT would clash. → Run it as a helper process. The AivisSpeech preset (P0) remains the fallback.
- **Real RTX 3060/4060 numbers under VRChat+OBS load**: The dev rig is a 4090. → Provisional numbers come from the load simulator, then friends run live-record exports. Targets stay marked provisional until at least one H1 run exists.
- **Competitor head-to-head**: Multi-GB installs on a sensitive network. → Optional and manual with user consent. Otherwise the harness scores WAVs the user records from competitors, and published numbers serve as reference rows.
- **Publishing Sigil-produced assets and pinning third-party hashes**: Agents cannot publish, and hashes need metadata API calls. → Entries stay 'unpublished' and are served locally until the user runs the prepared workflows. The pin script is sequential and throttled, with trust-on-first-use allowed only in dev.
- **MiLMMT-46-12B GGUF, IndicTrans2**: No ready GGUF, and IndicTrans2 has a custom architecture. → A make-gguf script for MiLMMT-12B, with 4B as the default challenger. IndicTrans2 goes through the NLLB-style ORT seq2seq path only if requested; NLLB covers Indic languages in the meantime.
- **LAN token rollout**: It could break saved OBS URLs or Link apps. → A persistent token, loopback exempt, a legacy toggle that defaults off, and the full client matrix before exit.
- **Unverified preview cloud APIs (Gemini 3.5 Transcribe Live, 3.8 Flash TTS, Live Translate) and the unofficial Edge protocol**: Model ids, quotas and tokens may change. → Model ids are editable, local fallback is automatic, and each client is a small isolated module.
- **Paid cloud (Appendix A: ElevenLabs, AssemblyAI, Soniox)**: Low priority under quality > local > free. → Not planned. Documented in P7-C. The existing Azure and Deepgram stay.

## Open questions
- Do you approve the one-time ORT 1.28 (rc.13) build download, knowing it makes CUDA 13 (driver R580+, Turing or newer) the CUDA floor, with DirectML as the automatic fallback? Or should we stay on rc.11?
- Is a roughly 165 MB larger installer acceptable in P0 (the bundled CUDA provider) until the P1 pack path is verified? Should per-user NSIS become the primary installer, with MSI kept secondary?
- Can Sigil-built runtime packs, templates and exports be published as GitHub Release assets on Mewnah/Sigil, which you would run from prepared workflows? Or should they go to a Hugging Face repo under your account?
- Do you consent to the one-time dev downloads: `pip install onnx`, embedder checkpoints (~1.8 GB spread over sessions), Seed-VC/DDSP/MeanVC2 weights (~1.5 GB), and test STT/TTS models, all sequential?
- Zero setup: may new installs default to the 'local' STT backend after P2 (asking before the ~650 MB Nemotron download), and may native-backend users get a one-time switch prompt?
- Should the LAN token be persistent (stable OBS URLs, and chosen here) or rotated per session? And should 'Allow LAN clients' default to on (token required for LAN) or to localhost-only?
- Can you supply a folder of the community .zip/.pth voices you actually use (including RefineGAN and JP HuBERT) for the 40-model importer corpus, plus consented VRChat mic clips for the benchmark corpus?
- Is a friend with an RTX 3060/4060 or a CPU-only PC available to run the in-app live-record benchmark? Do you want competitor installs (VCClient/Deiteris, RVC WebUI, Applio) on the dev rig for the head-to-head, or only recorded WAVs?
- Should the voice-changer worker default be 'auto' (NORMAL on GPU, BELOW_NORMAL on CPU), or do you prefer always protecting game FPS?
- Should the optional Beatrice 2 VST3 host (P6-M) be attempted at all, given the closed runtime?

## P1 outcomes (binding amendments for P2–P7)
Recorded by P1-I0 (2026-09-29) from the P1 spike and foundation results. Where this section and a workstream entry disagree, this section wins. ADRs: [0001](adr/0001-ort-upgrade.md), [0002](adr/0002-sherpa-onnx.md), [0003b](adr/0003b-transcribe-cpp.md), [0004](adr/0004-rvc-templates.md), [0005](adr/0005-process-host-llama.md), [0007](adr/0007-native-runtime-packs.md). Developer-facing contracts are in [DEVELOPER_GUIDE.md → Speech stack internals](DEVELOPER_GUIDE.md#speech-stack-internals).

### Every phase
- **ORT is rc.13 (ORT 1.28.0, CUDA 13).** CUDA needs driver R580+ and Turing or newer; DirectML covers the rest. Build every session through `ort_session` (rc.13's bare `SessionBuilder::new()` applies ORT's auto-EP policy); import `TensorElementType` from `ort::value`; builders are `mut` before `commit_from_*`; never call `ort::ep::cuda::preload_dylibs` (its list still names CUDA 12 DLLs). `register_ep_library` works (ADR 0001/0007).
- **Model store is the only downloader.** Locate files with `model_store::installed_path(id)`, install with `model_store::ensure(&[ids]).await`; custom URLs use `Store::fetch_unpinned`. Ids: `whisper/<plugin id>`, `vosk/<id>`, `runtime/<name>`, `runtime/sigil-<engine>-<variant>`. Runtime packs land in `runtimes/<id rest>/<version>/<variant>/`. Pin with `scripts/models/pin-catalog.mjs` (≥ 2 s apart).
- **Asset hosting is local only** (user decision 2026-09-28). Sigil-built packs, templates and exports are built by local scripts and installed by Import, `SIGIL_LOCAL_ASSETS` or a custom mirror; their catalog entries stay `unpublished`. No publishing or runtime-pack release workflows: P3-A's "publish script", P7-D's "publish packs" and ADR 0007's CUDA-provider publication wait for a new user decision.
- **Catalog ids for size recommendations** come from `resources::policy::ids` (`mt/hy-mt2-7b-q4_k_m`, `mt/hy-mt2-1.8b-q4_k_m`, `mt/hy-mt2-1.8b-1.25bit`, `mt/milmmt-46-4b-q4_k_m`, `mt/milmmt-46-12b-q4_k_m`, `mt/hy-mt2-30b-a3b-q4_k_m`, `tts/qwen3-tts-1.7b`, `tts/qwen3-tts-0.6b`, `stt/qwen3-asr-1.7b-bf16`, `stt/qwen3-asr-1.7b-q8_0`, `stt/cohere-transcribe-q4_k`, `stt/qwen3-asr-0.6b-int8`). Use them in the catalog fragments, or change `policy.rs` and `resources/format.ts` together (a vitest drift test checks the pair). GPU-model subsystems hold `resources::claim_vram` (in-process) or `claim_external_vram` (helpers) resized to the measured footprint; policy footprints are provisional until P7.
- **Native code** loads only through `native_runtime` (sigil_engine ABI v1, `open_engine`, hash-checked `sigil-pack.json`) or `process_host` helpers. Engines with special build needs keep their own `native/engines/<name>/build.ps1`; `scripts/native/build-runtime-packs.ps1` delegates to it.
- **The NVIDIA runtime** is `runtime/nvidia-cuda13` (CUDA 13.2, cuBLAS 13.3, cuFFT 12.2, cuDNN 9.26 `cuda13`). cuDNN 9.26 also needs `cudnn_engines_tensor_ir64_9.dll` and `cudnn_ext64_9.dll`; P1-I0 added them to the interim Settings installer, whose earlier installs fail CUDA until reinstalled. Follow-up (any phase that touches diagnostics): point `install/remove/get_nvidia_pack_status` at the store pack and retire the interim downloader.
- **Headless:** `Sigil.exe --headless sysinfo [--out f]` exists (P1-I0); new arms are added by the phase integrator.

### P2 (speech engine)
- **P2-S0:** the Azure speech-SDK chunk has a circular CommonJS init that breaks the production app ("Class extends value undefined", "Application failed to start") when module evaluation order shifts; P1-S0b triggered it with one runtime import of `@/types` in `src/core/schema.ts`. The schema and inspector splits must keep cross-module imports type-only where possible and prove the production bundle boots (unchanged speech-sdk chunk hash, or an IPC-mock host smoke in the browser pane); lazy-loading the SDK in `stt/tts` `azure.ts` removes the hazard. Playwright cannot catch it (web mode fails earlier at `plugin:web|config`).
- **P2-A:** consume the capture hub (`hub::global()?.attach(SourceSpec::parse(device), ConsumerOptions::speech("<label>"))`); `ReadInfo.gap_after` / `restarted` must close the open utterance; the STT mute gate is `MuteGroup::Stt` (the TS audioDevices service already mirrors the hard mute to it). The JSON-PCM whisper feed is already gone. Move `whisper.rs`'s downloader to `model_store::ensure([whisper_entry_id(id)])` (the store already installs `whisper/*` into the same folder and adopts existing files).
- **P2-B:** use SenseVoice **`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`** (163,002,883 B, sha256 `7d1efa21…fd347e`); the 2025-09-09 package is a Cantonese fine-tune. SenseVoice ignores the per-stream language option; punctuation-only output is dropped by the wrapper. Nemotron 3.5 160 ms int8 is 475,273,363 B (sha256 `a81909a1…f7a61a`, not ~650 MB), Qwen3-ASR-0.6B int8 878,702,423 B (sha256 `393f8a14…78ee96`); both are supported by 1.13.8 but not yet run. Streaming recognizers default to 1 thread (ORT's spin-wait costs ~25 ms per extra thread per run and cannot be disabled through the C API); others 2, max 4. Load the sherpa runtime at speech-plugin init whenever a sherpa engine is configured (it refuses to load after another `onnxruntime.dll`). Non-catalog model folders go through `probe::validate_in_child`.
- **P2-C / P2-D:** measured Qwen3-ASR-1.7B Q8_0 on CUDA: about 3.45–3.5 GB process VRAM while loaded and running plus ~0.38 GB CUDA context kept while any `cuda13` engine DLL is loaded (whisper-tiny ~0.46 GB); RTF ≈ 0.018 on the RTX 4090, ≈ 0.25 on 8 CPU threads. Replace `policy.rs`'s provisional `ASR_Q8` 2700 MB with the measured value (and re-check the 16 GB rows) when the stt_final claim lands.
- **P2-D:** `speech_rt::transcribe::TranscribeModel` is the adapter base. Qwen3-ASR in transcribe.cpp 0.2.4 checks cancel only before its single-shot decode: drop stale results by epoch (or use `transcribe_run_batch` with n=1). The `vulkan` variant is unbuilt and unmeasured (no Vulkan SDK on the dev PC): until it is, non-NVIDIA GPUs use the CPU variants. ORT's CUDA EP and ggml-cuda in one process (both on `cublas64_13`): **verified in P2** (P2-D: the ORT CUDA canary passes before and after Qwen3-ASR CUDA runs; both share one CUDA context).
- **P2-E:** mount `<ModelRequirement ids …>` for consent and downloads; deep-link with `changeTab({ tab: "models", value: id })`.
- **P2-F:** `start_audio_capture({ sampleRate: 16000 })` already delivers 16 kHz mono `audio:chunk` from the hub; a binary channel for Vosk is optional.
- **P2-H:** **risk to the acceptance "Kokoro RTF ≤ 0.3 on 4 threads":** Kokoro int8 multi-lang v1.0 through sherpa measured RTF ≈ 1.1 (en) / 1.3 (ja) on an idle 7800X3D at 1–4 threads (first sentence 2.2–2.6 s). Measure the fp32 package (`kokoro-multi-lang-v1_0`, ~350 MB) first; if neither meets the target, take the plan fallback (Sigil's own ORT runner for Kokoro) or ask the user to relax the target. TTS generation through sherpa is serialized process-wide (espeak-ng). Move the Kitten hub downloader (`local_tts/kitten/hub.rs`) onto catalog entries plus `ensure()`.
- **P2-I0:** the Nemotron consent text says about 475 MB, not 650 MB.

### P3 (voice engine)
- **P3-S0:** the vc-templates are already bundled (`tauri.conf.json` `bundle.resources` map form, P1-I0): drop that hotspot. `voice_changer::{budget, playback, fallback}` are `pub(crate)` and used by `bench::vc_rt`; keep them so.
- **P3-A:** the fp16 text encoder misses the 5e-3 gate on every GPU EP (DirectML 2.1e-2, CUDA 6.0e-3): `enc` always runs fp32. The shipped fp16 `flowdec` keeps decoder stage 0 in fp32 (DirectML 2.9e-3 on one voice); the T8 corpus must confirm it, else export with `--fp32-stages 0,1` (1.4e-3). nof0 `flowdec`, RefineGAN and MRF are not exported yet (Applio code must be vendored). `knn` needs ≥ 64 index rows (pad or disable retrieval below that).
- **P3-B:** build on `neural_audio::vc::import` (P1-G3); run it through `process_host::run_worker` (Job Object memory cap, wall-clock limit, bounded output). Still to do: slot manifest `sigil-rvc-1`, `.index` adoption, sha256 import cache, the 200-file fuzz corpus, and the `pthrs` dev-dependency cross-check (needs a Cargo.toml request). Training checkpoints, bare state dicts and multi-`.pth` archives are refused today.
- **P3-C1 / P3-C2:** add a builder hook to `ort_session` (for example `build_session_with(…, configure)`) so template sessions get `with_external_initializer_file_in_memory` and dimension overrides through the full EP ladder (preflight, canary, CPU retry, EpOutcome); today they build on one explicit EP through `builder_for_ep`. Set `use_tf32 = 0` for voice-changer CUDA sessions (ORT's default TF32 puts fp32 RVC graphs at ~1.2e-3, 10× over the 1e-4 gate), benchmarked before and after. Real-time sessions move to 1–2 intra-op threads once measured (bench spec `engine.intra_threads`).
- **P3-E:** move the voice changer onto the hub: `ConsumerOptions::device_rate("vc").with_mute_group(Some(MuteGroup::VoiceChanger))`, using `gap_after` / `restarted` for SOLA resets and `end_qpc_ns` for latency telemetry (this completes one stream per physical mic). Fix the benchmark's post-stall finding: after a stall the output queue refills to `output_max_queue_ms` instead of the prefill and never drains (latency 74.4 → 94.4 ms, persistent); trim back toward the prefill with a crossfade after an underrun or budget drop. `resources::set_vc_running` is wired (P1-I0); add a `claim_vram(Subsystem::Vc, …)` for GPU sessions resized to the measured process-VRAM delta (0 MB on CPU).
- **P3-F:** the bench core exists (`--headless bench`, paced/offline/e2e-loopback, load simulator, perf gate). Not yet run: the live e2e-loopback path (the dev PC's virtual-cable endpoints are disabled) and any real RVC voice; the first real-voice run on CUDA, DirectML and CPU becomes the engine baseline. Optional: make `params.rs`'s normalization helpers `pub(crate)` so `bench/{engine,spec}.rs` stop mirroring them.
- **P3-G:** optional UX from P1: a Remove button for `legacy/*` runtime folders (`plugin:runtime|remove_legacy_runtime`).

### P4–P7
- **P4-B:** `process_host::recipes::llama_server` and `runtime/llama-server-{vulkan,cuda13,cpu}` (llama.cpp b11146, pinned) exist. The live llama-server + Hy-MT2 1.8B smoke has **not** run (running the downloaded exe was not permitted to the P1 agent): run ADR 0005's procedure first; the recipe's flags are untested against b11146. The CUDA entry extracts two zips into one folder: check for duplicate file names on first install. Hold `claim_external_vram` for the server. The exposure audit runs once after readiness; a periodic re-audit is a cheap follow-up.
- **P4-E:** Bergamot uses `StdioMode::Protocol` + `take_stdio`; the client must keep reading stdout or the helper blocks.
- **P6-A:** pass `includeLoopback` to `InputRustAudioInput` only in the others.stt picker; loopback sources are `loopback:default` / `loopback:<output name>` (WASAPI loopback is silent while nothing plays; per-process loopback is P6).
- **P6-E:** Qwen3-TTS ids are `tts/qwen3-tts-1.7b` / `tts/qwen3-tts-0.6b`; the helper claims VRAM with `claim_external_vram`.
- **P7:** replace every provisional footprint in `resources::policy` with measured values; decide on publishing (runtime packs, templates, the CUDA provider pack, then drop the 62 MB provider from the installer per ADR 0007).

### P1 manual checks still open
Real-app Models tab pass (download, cancel/resume, verify, delete, import, custom mirror) in `pnpm tauri dev`; USB microphone unplug/replug; the llama-server + Hy-MT2 smoke; Sigil killed in Task Manager ends llama-server; the NVIDIA runtime pack through the Models tab with a CUDA voice-changer session; a real RVC voice benchmark; the e2e-loopback meter over a virtual cable; the transcribe `vulkan` variant. See [RELEASE_AND_ENGINEERING.md → P1 exit matrix](RELEASE_AND_ENGINEERING.md#p1-exit-matrix-platform-foundations).

## P2 outcomes (binding amendments for P3–P7)
Recorded by P2-I0 (2026-09-29) from the P2 workstream results. Where this section and a workstream entry disagree, this section wins. Contracts: [DEVELOPER_GUIDE.md → Speech pipeline](DEVELOPER_GUIDE.md#speech-pipeline-speech) and the sections after it; measurements: [docs/benchmarks/SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md), [ADR 0002](adr/0002-sherpa-onnx.md) and [ADR 0003b](adr/0003b-transcribe-cpp.md) (P2 results).

### Every phase
- **One STT path.** Every Rust STT engine implements `speech::engine_api::AsrEngine` and is created through `speech::engines::create` (a new `EngineKind` is an INTEGRATE change in `engines/mod.rs` and `engine_api.rs`); pipelines start with `plugin:speech|start_pipeline` (keys `mic`, `loopback`; `captureOnly` for voice detection without an engine). TS drivers follow `stt/services/local.ts` (router plans, consent, epochs) or `stt/rustCapturePolling.ts` (explicit engines). Events carry `pipelineId` + `epoch`; the TS segment id is `<pipelineId>:<epoch>:<segmentId>` and is published on `TextEvent.segmentId` / `revision` / `lang` / `turnEnd`.
- **One TTS output.** Every new TTS backend implements ITTSService v2 (`synthesizeToSink` for Rust engines, `synthesizePcm` for TS-produced PCM) and plays through `tts_sink` (binary `push_pcm`, begin order, the top-level route). No Web Audio clip playback and no JSON sample arrays; the sink applies the backend slice's `volume`. HTTP speech servers go through the `openai-speech` client (`HttpSpeech` / `speak_http` for non-OpenAI request shapes).
- **Catalog ids** of the P2 models: `stt/nemotron-3.5-streaming-{560ms,160ms}-int8`, `stt/parakeet-ultra-int8`, `stt/parakeet-tdt-0.6b-v3-int8`, `stt/qwen3-asr-0.6b-int8`, `stt/sense-voice-small-int8`, `stt/moonshine-base-en-int8`, `stt/lid-whisper-base-int8`, `stt/qwen3-asr-1.7b-{q8_0,bf16}`, `stt/cohere-transcribe-q4_k`, `stt/granite-speech-5.0-turboctc-q8_0`, `stt/voxtral-mini-4b-realtime-q4_k_m`, `vad/smart-turn-v3.2`, `tts/kokoro-v1.0`, `tts/kokoro-v1.1-zh`, `tts/kitten-{mini,micro,nano}-en-v0.8`, `tts/piper-<voice>`.
- **Measured footprints:** `resources::policy` STT finals are measured (Qwen3-ASR 1.7B Q8_0 3500 MB, BF16 6000 MB, Cohere 2000 MB, process VRAM with the CUDA context); keep `stt-routing.json` `vramMb` equal. A GPU runtime's CUDA context (about 0.4 GB) stays claimed for the process lifetime once a model ran on it.
- **CPU arena:** `VcSessionOptions::cpu_arena` enables ONNX Runtime's CPU memory arena (ort rc.13 disables it by default). Dynamic-shape CPU graphs should opt in (Kokoro does: RTF 0.29 vs 0.29–0.40 without it on new sentence lengths).
- **Defaults are flipped:** new desktop installs start on STT `local` and TTS `localNeural`; existing profiles on the old defaults get one-time offers (`stt/migrate.ts`, `tts/defaults.ts`, `ApiServer.runSpeechDefaultOffers`). Later default changes follow the same pattern (pure planner, a `*Migrated` flag, an offer for existing users).
- **Final-gate contract additions (P2 review fixes):** `plugin:speech|stop_pipeline { pipelineId, epoch? }` (with `epoch` it stops only that session while it is the key's current one and never cancels a queued start; drivers that clean up after a start they gave up on pass it, and every `mic` driver calls `noteSttPipelineStart()` before `start_pipeline`); `TtsSinkContext.lang` carries the sentence's `TextEvent.lang` to sink engines (`TtsPipeline.speak(text, lang)`); an auto plan's streaming `options.streamLanguages` is enforced by the pipeline (other detected languages get the final alone).
- **Legacy IPC kept for one release, then removed** (P7-D or the next integrator): the `neural-onnx-tts` plugin, `moonshine-stt` commands, `audio`'s `start_utterance_capture` / `take_utterance` / `stop_utterance_capture` (+ `audio:utterance`), and the JSON `voice_changer_push_aux_pcm` / `_wav` commands (neural sidecar and ONNX paths no longer use them).

### P3 (voice engine)
- **TTS into the voice changer** arrives as tagged blocks through `voice_changer::aux_feed::tts_aux_try_push` (sink route `voiceChanger` / `both`; cancel with a 5 ms fade). P3-E keeps that API (and the mic duck / mute) when it moves the voice changer onto the capture hub.
- **CUDA sharing:** ORT's CUDA EP and ggml-cuda share one CUDA context in the process; the STT context claim double-counts about 380 MB next to a CUDA voice changer (conservative). P3-E's `claim_vram(Subsystem::Vc, …)` is still to do.
- **CPU arena for the voice changer:** decide with before / after benchmarks (the voice changer and `bench::engine` pass `cpu_arena: false` today).

### P4 (translation)
- STT finals from the `local` backend carry `segmentId` / `revision`; the TS `SegmentPublisher` publishes at most one final per segment (the two-pass text), with interims of rising revision before it. P4-C / P4-G may key LocalAgreement and revising captions on `segmentId`. `turnEnd` (Smart Turn, off by default) is a commit hint only.

### P6 (streamer features)
- **P6-A:** the `loopback` pipeline key and `MuteGroup::Others` exist; start it with `source: "loopback:default"` (or `loopback:<output>`).
- **P6-C:** chatbox commits can use `TextEvent.turnEnd`. SenseVoice's `emotion:` / `event:` tags ride on `speech:final` but `TextEvent` has no tags field yet: add one there.
- **P6-E:** the qwentts `tts-server` preset (:8080) already exists in the OpenAI-compatible client; the helper only has to start the server.
- **P6-G (or a new workstream):** Kokoro's Japanese reads kanji with Mandarin readings (sherpa-onnx 1.13.8 frontend). Proposal: a morphological analyzer crate (for example vibrato) with an IPADIC dictionary from the model store, converting kanji to katakana before sherpa (a Cargo.toml hotspot).
- **P6-J:** Windows SAPI and Web Speech are still self-playing.

### P7
- The routing table's per-language rows come from vendor and transcribe.cpp FLEURS tables, not Sigil's runs: replace them from the campaign (the shared fixture and `table_models_match_the_catalogs` guard edits). Voxtral Realtime (GPU streaming) is catalogued but not routed; Granite TurboCTC is never a caption final (no casing or punctuation).
- **Confucius4-R2T2** needs a transcribe.cpp re-pin newer than 0.2.4 plus rebuilt packs; the engine already maps r2t2 to append-only streaming. The `vulkan` transcribe variant is still unbuilt.
- The first Kokoro audio for a long first sentence is 0.65 s (its first clause is about 2 s of audio); P2's "≤ 400 ms" holds only for short opening clauses. P7 decides whether to split the first clause shorter.

### P2 deviations from the workstream entries (recorded)
- The default streaming model is `stt/nemotron-3.5-streaming-560ms-int8` (not 160 ms): the same 475 MB download, 2.7× less CPU and fewer errors; 160 ms is a low-latency option. Nemotron is supported by sherpa 1.13.8, so no transcribe-CPU fallback flag was needed.
- Parakeet Ultra is a pinned, hash-verified community sherpa export (`mldecode/parakeet-ultra-onnx-int8`); Parakeet TDT v3 is its fallback. Moonshine v2 Streaming is not packaged for sherpa; Moonshine base-en is catalogued instead.
- The consent text is built from the plan's `missing` ids and catalog sizes (English two-pass about 1.1 GB, Japanese about 1.35 GB; the smaller single-pass setup is offered when it saves at least a fifth), not a fixed 475 MB.
- Kokoro ships as the fp32 package (350 MB) and runs its graph on Sigil's ORT with the CPU memory arena (RTF 0.29 on 4 threads); int8 is not catalogued. The ITTSService v2 PCM method is `synthesizePcm` (the plan's `synthesize` collided with the old clip hook), plus `synthesizeToSink`.
- Confucius4-R2T2 is not available in transcribe.cpp 0.2.4 (see P7 above); the opt-in append-only streaming exists and was verified on Voxtral Realtime.
- `openai_speech` is a single file (`services/openai_speech.rs`); `voice_changer/aux.rs` is `aux_feed.rs` (AUX is a reserved device name on Windows).

### P2 manual checks still open
Real-app runs in `pnpm tauri dev`: a fresh profile with the `local` backend (consent, downloads, captions in en and ja); Whisper with a real microphone (download row, STT mute drops the open sentence, USB replug continues); Vosk and Groq (real key: captions, rate-limit status, bad-key message, nothing sent while muted); a LAN OpenAI-compatible STT server; Moonshine server off; Kokoro en / ja / zh, Piper ru and Kitten from the panel (the UI never freezes); a long VOICEVOX text starting after its first sentence; TTS routed into a running voice changer with duck / mute; unplugging the output device mid-sentence; the OpenAI-compatible TTS presets against real servers; the one-time STT and TTS offers on an existing profile. See [RELEASE_AND_ENGINEERING.md → P2 exit matrix](RELEASE_AND_ENGINEERING.md#p2-exit-matrix-speech-engine).

## P3 outcomes (binding amendments for P5–P7)
Recorded by P3-I0 (2026-09-30) from the P3 workstream results. Where this section and a workstream entry disagree, this section wins. Contracts: [DEVELOPER_GUIDE.md → Sigil Voice Engine](DEVELOPER_GUIDE.md#sigil-voice-engine-p3) and [`params_contract.json`](../src-tauri/src/services/voice_changer/params_contract.json); decisions: [ADR 0004 → P3 outcomes](adr/0004-rvc-templates.md#p3-outcomes) and [ADR 0006](adr/0006-vc-zero-shot-singing.md); measurements: [docs/benchmarks/VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md); exit evidence: [RELEASE_AND_ENGINEERING.md → P3 exit matrix](RELEASE_AND_ENGINEERING.md#p3-exit-matrix-voice-engine).

### Every phase
- **One engine factory.** Every voice-changer family is built by `voice_changer::factory::create_engine_with`, dispatching on the slot's `manifest.format` (`rvc` legacy ONNX, `sigil-rvc-1` template voices); a new family is a new format arm there plus a `VoiceConversionModel` (with `fixed_block` / `configure_stream` when it runs fixed shapes). The worker and the benchmark both build through it. Hot voices, the switch crossfade and the per-voice `claim_vram` come with `LoadedVoice` / `VoiceSet`.
- **Levers.** A lever's `auto` value is `engine::presets::auto_lever(profile, lever, base_ep)`, resolved with what the build and the rung can run by `voice_changer::capabilities::resolve_levers`; `off` is always the legacy path. The table (P3-I0, from the P3-C2 and P3-F measurements): the four algorithmic levers on everywhere, `io_binding` on for CUDA and DirectML, `cuda_graphs` on for CUDA; `gpu_priority`, `gpu_knn` and `governor` off until a benchmark shows a gain. A change to the table records its before / after numbers in `presets.rs` and VC_BENCHMARKS.
- **GPU sessions of the voice engine** go through `gpu::plan::SessionPlan` (zero host allocations per tick on the bound paths) and `ort_session::build_session_with` / `build_session_ext`. Never use ONNX Runtime's `enable_cuda_graph`: its global capture mode breaks every other CUDA user in the process (the speech engine, ggml); `gpu::cuda_graph` captures in thread-local mode. **TF32 follows the precision request** (on unless `fp32` is requested; this supersedes the P1 amendment "set `use_tf32 = 0`": turning it off costs 2.3× for an error inside the fp16 budget). Test builds have exactly one `#[global_allocator]` (`gpu::alloc_probe`).
- **Pitch** comes from `vc::f0_fusion::create_pitch_extractor_with(&req, GpuLevers::from_effective(&levers))` (the factory does since P3-I0, so RMVPE runs as its CUDA-graph plan with the CUDA levers); every extractor reports its EP outcome.
- **Templates:** 22 bundled (15.5 MiB). The importer and the engine read **layout version 1 only**; the 15 version-2 templates (every f0-less, MRF and RefineGAN cell) are refused at import as `unsupported_model` (P3-I0). Supporting them is the first follow-up and belongs to the importer (`import/layout.rs`, `session.rs`: the `features` list, `streaming.latent_offset` (serde default 0; validate `== dec_margin − main_margin` for RefineGAN, 0 otherwise), smoke feeds sized by each input's own dim name (`Fz` vs `F`) from `graphs.flowdec.variants.*.inputs`, `noise.f32.bin` as a second in-memory external initializer, no pitch for `nof0`) and the engine (`engine/generator.rs`, `plan.rs`: `phase` `[1,F]`, MRF `src_noise` `[1,S,9]` with `phase_ini`, RefineGAN `noise_offsets` and its pool, falling back to the recommended variant when a forced one is marked `fail`). **RefineGAN windows (P3 final gate correction):** the source window `F = flow_margin + dec_margin + R` feeds `f0`, `phase`, `noise_offsets` and `src_noise`; the latent window `Fz = F − latent_offset = flow_margin + main_margin + R` feeds `m_p`, `logs_p` and `z_noise`, so the content-encoder / `enc` context is Fz (about 0.40 s at 32 kHz, R = 4), not F; source frames the graph does not read may be zero-filled; `excitation.f0_lookahead_frames` = 1 means decode R + 1 frames and drop the last (or delay 10 ms); lookahead and crossfade are sized per template from `matrix.json` `seam_frames_fp32`, never from `dec_receptive_right_frames` ([DEVELOPER_GUIDE.md → layout version 2](DEVELOPER_GUIDE.md)). The fixtures are `test-fixtures/vc/tiny/v2`.
- **Asset hosting stays local:** the content encoders, the FCPE adapter and `bench/gpu-load` are Sigil exports (`unpublished`), so the first-run card shows **Import** for them and "zero setup" needs a local import (or `SIGIL_LOCAL_ASSETS`) until they are published. Split each `vc/embedder-*` entry per precision variant before publishing (575–867 MB installed together).
- **Benchmarks:** `pwsh scripts/bench/run-perf-gate.ps1 -Profile p3` is the voice-engine gate (T-target rows; `-TargetsAdvisory` reports without failing). For fixed-block engines per-block compute is `call_ms` and `rtf_*` / `over_budget_ratio` are judged on the device blocks that ran the model; baselines from before P3-F are not comparable on those metrics.
- **UI:** new voice-changer strings follow `options.ts` (choices from `voice_changer_get_capabilities`; a stored value this build cannot run stays visible with what runs instead) and `restart.ts` (every param has a hot / switch / restart / idle rule in every mode; a test enforces it).

### P5 (voice-changer families)
- ADR 0006 decides the families: Sigil-ZS (Seed-VC realtime) in-engine on CUDA and DirectML at fp32 (the CPU is no-go); DDSP-SVC 6.3 on CUDA from Sigil's own export (P5-A restructures the `ddsp` graph for DirectML, which refuses its pinned shapes and takes 1.8–4 s per call); MeanVC2 as an experimental CPU family if P5-C's A/B passes (its speaker encoder is a manual Google Drive download). The H1 figures in ADR 0006 are 4090 × 3 estimates.
- Families register in the factory (above), report `resources::set_vc_gpu` / `claim_vram`, and reuse the harness, budget, timeline, metrics and TTS aux path. Keep DirectML sessions alive for the process where possible (onnxruntime-directml 1.24.4 in Python crashed when a session ran after another DirectML session was released; unverified on ort rc.13) and fall back to unpinned dimensions when DirectML refuses pinned ones.
- **The bench CAM++ asset** (`bench/campplus`, sherpa-onnx's 3D-Speaker export) has 52 `AveragePool` nodes with `ceil_mode = 1` that ONNX Runtime computes differently from PyTorch: it matches upstream CAM++ at only 4.4 dB (cosine 0.925). It ranks speakers correctly (same speaker 0.918, others 0.089), but T5's and T10's absolute speaker-similarity thresholds need a re-exported CAM++ (as `scripts/vc-families/seed_vc/export.py` does) before they are final.

### P6 / P7
- **P7 (VC campaign)** owns what P3 could not measure on the dev rig: an H1 live-record from a friend (T1 / T2 / T7 under real load), the e2e loopback meter over a virtual cable (T1 true latency), the T6 head to head (RVC WebUI, Applio realtime, w-okada through `scripts/vc-bench`, blind A/B with `tools/listening-test`), the 20-clip consented corpus (T5) and the 40-voice community corpus (T8), a TensorRT-RTX live run (T9 cache hit ≤ 3 s), a 30-minute soak, and the governor, `gpu_priority` and `gpu_knn` defaults.
- **Engine gaps the P3 scorecard points to** (owners in brackets; evidence in VC_BENCHMARKS "What closes the gaps"): per-block compute (T2: the content encoder is the largest cost; put it on the plan path with IOBinding and a CUDA graph), the DirectML content encoder re-planning when the crossfade is not 20 ms (34–49 ms per call), RMVPE's VRAM (T3; SwiftF0 / CREPE-tiny keep DirectML under 650 MB), the CPU generator (T4 Streamer: 3.4× slower than real time on one thread), DirectML cold load (T9: 2.7–3.2 s), the streaming pitch track (T5; the final gate's end-aligned RMVPE mel took V/UV to 3.5–3.8 %, F0 RMSE is still 47–53 cents), the Quality voices' double generator build at load (T9: pass `lookahead_ms` in `SigilRvcRequest` so `load_with` plans it from the start), and contexts over 0.5 s muting utterance onsets after silence.
- Open implementation items: the benchmark's `pitch_ab` mode (`f0_fusion::ab::report` exists), the `engine.denoiser` and `engine.cpu_arena` bench options and running the governor in the contention mode, wiring `vc::autokey::AutoKey` into the metrics and the inspector (it needs a target F0 per voice), the `pthrs` dev-dependency cross-check of the pickle reader and the FAISS reader, the maintainer run of `parity_test.py --upstream-applio`, a PESTO export, a per-EP (CUDA vs DirectML) fp16 parity class, and the hub-level device-family match so STT also follows a renumbered USB microphone (`voice_changer::devices::InputFollow` does it for the voice changer).

### P3 deviations from the workstream entries (recorded)
- `lookahead_ms` (0–80 ms, default 0) is a P3 param added at integration: Quality and Studio-TRT use 20 ms (default voice on CUDA with the shipping levers: speaker cosine 0.940 → 0.975, V/UV error 7.4 → 3.5 %, MCD 1.66 → 1.40 dB, DNSMOS 3.12 → 3.29), and the presets keep RMVPE because RMVPE with lookahead beat fusion with lookahead on that voice. The factory skips it while the content encoder runs on DirectML (35–49 ms per call there with lookahead, the same DirectML shape problem as a crossfade other than 20 ms) until that encoder runs a fixed shape on DirectML. Fusion stays offered (it wins on the synthetic octave-trap A/B); `fcpe` stays the stored default.
- SPIN v1 voices run (`vc/embedder-spin`, `EmbedderKind::Spin`); the P3-S0 refusal is lifted.
- SwiftF0 is not the CPU default: it wins on gross pitch error but loses on voicing (P3-D's A/B).
- DeepFilterNet3 is not offered as its own denoiser: a stored `dfn3` runs DPDFNet (its published ONNX cannot stream). PESTO is import-only. The governor's context step is not implemented and is skipped.
- No asset publish script (asset hosting is local only); `scripts/assets/` holds the catalog writer and a local-folder verifier.
- TensorRT-RTX is implemented and unit-tested but not run live (no runtime downloaded; the pyke rc.13 provider imports TensorRT 10 names, so it needs a matching runtime or a provider built for TensorRT-RTX).
- The `ContentVec` fp16 variant stays the GPU default although `embedder.json` marks it `fail` on DirectML: the failing row is a 6 s clip (rel. L2 2.5e-2); the ≤ 1 s windows the engine runs pass (4.6e-3–5.9e-3).
- **Final gate (2026-10-01), from the review fixers' late requests, each benchmarked** ([VC_BENCHMARKS → P3 final gate](benchmarks/VC_BENCHMARKS.md#p3-final-gate-pitch-track-fixes)): RMVPE's log-mel is end-aligned (silence floor before the audio instead of RVC's zero log-mel after it; V/UV against the ceiling 8.0–9.2 % → 3.5–3.8 %, no cost change); a bring-your-own ONNX voice anchors its noise only on an explicit `on` (`EngineRequest::anchored_noise_mode`, `presets::byo_lever`; T8); `LoadedVoice::load` folds the model's `stream_report` into the voice's outcomes and levers (T7); the worker publishes the slot folders it holds (`VoiceChangerState::held_slots`) so imports never replace or delete a hot voice; the requirement card lists FCPE for fusion on the CPU rung; the autotune card shows a stale tune's reasons with **Tune again**; `parity_test.py` accepts bare / DataParallel state dicts. **Not adopted, by measurement:** FCPE's 20-frame lead-in (better pitch track, lower DNSMOS on CUDA) and 10 ms of lookahead for Low latency (+0.02 cosine, −0.11 dB MCD for +10 ms on one clip; the P7 corpus decides). **Not done (optional):** `lookahead_ms` in `SigilRvcRequest` (T9), `post.process_at` in `convert_inner` (post.rs already matches short overlaps), sizing RMVPE's plan for the incremental call.

### P3 manual checks still open
The in-app voice changer in `pnpm tauri dev`: first-run card and default voice, drag-and-drop import (a `.pth` with its `.index`, a `.zip`, a refused file), rename / duplicate / merge / delete, hot switch between two imported voices while running, voice hotkeys while a game has focus, TensorRT-RTX build and cache hit, noise removal on and off, USB microphone and output unplug / replug (same and another port), a 30-minute soak with VRChat and OBS, the live e2e loopback, an H1 live-record, the T6 head to head and the two corpora; the maintainer checks `parity_test.py --upstream-applio` and a PESTO export. See [RELEASE_AND_ENGINEERING.md → P3 exit matrix](RELEASE_AND_ENGINEERING.md#p3-exit-matrix-voice-engine).

## P4 outcomes (binding amendments for P5–P7)
Recorded by P4-I0 (2026-09-30) from the P4 workstream results. Where this section and a workstream entry disagree, this section wins. Contracts: [DEVELOPER_GUIDE.md → Shared LLM client](DEVELOPER_GUIDE.md#shared-llm-client-srccoreservicesllm) and the sections after it; [ADR 0005](adr/0005-process-host-llama.md) (P4 update) and [ADR 0008](adr/0008-bergamot.md); measurements in [SPEECH_BENCHMARKS.md → Machine translation](benchmarks/SPEECH_BENCHMARKS.md#machine-translation); exit evidence in [RELEASE_AND_ENGINEERING.md → P4 exit matrix](RELEASE_AND_ENGINEERING.md#p4-exit-matrix-translation-stack).

### Every phase
- **One LLM client.** Every OpenAI-compatible call from TypeScript goes through `@/core/services/llm` (`resolveLlmEndpoint(settings)` + `createLlmClient(endpoint)`, `chat` / `complete` / `listModels`); LAN and CORS-blocked hosts go through `plugin:llm-server|llm_proxy_chat` automatically. Aborts reject with the signal's reason; errors are `LlmError` (`kind`, `status`, `retryable`, `retryAfterMs`). The `openai` npm dependency is gone (P4-I0 removed it and its dead `manualChunks` line).
- **One llama-server.** Consumers take a lease with `acquireLlmServer(holder, config, signal)` from `@/core/services/llm_server`; there is one server, the last acquire's config wins, so two consumers that want different models restart it back and forth (give them the same model, or add a second helper name first). It stops 10 s after the last release, holds `claim_external_vram`, and obeys the 8 GB rule (models above 4.5 GB are refused while the voice changer runs on the GPU of a card under 10 GB). The key is handed to TypeScript for loopback calls (random per session, `127.0.0.1` only); status surfaces never carry it.
- **Ids are unique per process.** STT segment ids are `<pipelineId>-<PROCESS_NONCE>:<epoch>:<n>`, generated translation line ids `<name>.<stream>.<PROCESS_NONCE>.<epoch>:<n>`, and the AI transform's numeric line `id` starts at `PROCESS_ID_BASE` (`src/utils/processNonce.ts`), so restarted or linked hosts never reuse another process's ids. New id producers use the same helpers; consumers compare ids for equality and order segments only within one stream (the part before the first `:`).
- **Translation instances.** `createTranslationPipeline({ name, settings, engines, publish, pair })` makes independent pipelines; translation events carry the line's `segmentId`, a rising `revision`, the target `lang`, `stableLength` on interims and `turnEnd` on finals. `TextEvent.stableLength` (comment amended in `src/types.ts`): the committed prefix of interims only grows; the final may replace it once.
- **Catalog ids (P4):** `mt/*` GGUF models in `llm-gguf.json` (family `mt-llm`, default `mt/hy-mt2-1.8b-q4_k_m`), `mt/nllb-200-distilled-600m-int8` (published) and `mt/nllb-200-distilled-1.3b-int8` (unpublished), `mt-bergamot/<src>_<tgt>` (106 directions, family `mt-bergamot`, generated by `gen-catalog.mjs`), `runtime/bergamot` (unpublished, local import).
- **Headless:** `Sigil.exe --headless mt-bench <spec.json> [--out <dir>]` (engines `nllb`, `openai`, `hypotheses`; chrF2 equal to sacrebleu).
- **Gemini** defaults to `gemini-3.5-flash-lite` in transform and translation (Google limits the 2.5 models to projects that already used them); the key goes in `x-goog-api-key` (unverified live: if Google rejects it, switch `keyHeader` to `bearer` in `llm/providers.ts`).

### P5 (voice-changer families)
- The llm-server plan reads the voice changer's GPU state from `resources` (`set_vc_running` / `set_vc_gpu`). New GPU families must report through the same calls, or a 7B translation model will be allowed next to them on 8 GB cards.

### P6 (streamer features)
- **P6-A:** other-player translation is a second `createTranslationPipeline` with `pair: () => ({ from, to })` (the reverse pair) and its own `publish` on `others.*`; a topic that crosses WebSockets needs a route in `inbound.ts`.
- **P6-C:** the chatbox should send translation finals (or the committed prefix, `value.slice(0, stableLength)`), never tentative tails; finals carry `turnEnd`.
- **P6-K (Gemini Live translate):** reuse the translation `llm` slice's Gemini key and the shared client's error mapping.

### P7
- **MT campaign:** record Hy-MT2 1.8B / 7B and MiLMMT-46-4B through the mt-bench `openai` engine on the bundled server, and Bergamot (add a `bergamot` mt-bench engine calling `mt_local::bergamot::translate_line`, which exists for it); set the router's default chain and interim engine from the results; replace the smoke corpus (24 author-written lines per direction) with consented real caption logs.
- Replace the provisional llm-server footprints (TranslateGemma 3000 / 8100 MB, Seed-X 5300, LMT-60 3600, CAT-Translate 5300, Shisa 5900, VNTL 6400, GalTransl 3800, MADLAD 2600) with measured values; revisit `resources::policy`'s recommendation of the Hy-MT2 1.25-bit build for low-memory CPU PCs (llm-server substitutes the Q4 build until llama.cpp's STQ1_0 kernel has an x86 path and the runtime is re-pinned).
- Transform can offer the `bundled` provider once the catalog has a general instruct model.

### P4 deviations from the workstream entries (recorded)
- The MT benchmark is `src-tauri/src/services/mt_local/bench.rs` (not `services/bench/mt.rs`) and DeepL is `services/translate/deepl.rs`, because P3 owned `services/bench/**` during the concurrent run.
- The Bergamot helper source is `scripts/runtime-packs/bergamot/helper/` (not `tools/bergamot-helper/`), because P3 owned `tools/**`; the Rust client spans `mt_local/bergamot/`. The real `sigil-bergamot.exe` is not built yet (the upstream source fetch was refused in-session; ADR 0008 has the manual build and its no-go fallback: NLLB-600M int8 plus Hy-MT2 1.8B on the llama-server CPU pack).
- The llm-server hands its key to the webview for loopback calls (ADR 0005's "the webview never holds the key" is superseded; see its P4 update).
- AI transform does not offer the bundled llama-server (the bundled catalog is translation models); local transform uses the custom provider.
- New files outside the entries: `src/client/elements/text/revising.ts` (+ tests), `src/core/services/transform/streaming.ts`, `mt_local/nllb/tests.rs`, `translation/__tests__/{nllb,deepl,bergamot}.engine.test.ts`.
- Santali maps to FLORES `sat_Beng` (the NLLB tokenizer's token) in both `languages.ts` and `flores.rs` (P4-I0).

### P4 manual checks still open
The in-app Hy-MT2 1.8B run (Test load and a translated line in `pnpm tauri dev`, then Sigil ended in Task Manager; `real_model_end_to_end` and the Hy-MT2 chrF row passed at the final gate); the Bergamot pack build and its en ↔ ja p95 < 150 ms self-test; DeepL with a real key (`deepl_live`); AI transform against a real Gemini key and a real Ollama on localhost (automatic proxy fallback on 403); the chrF row for Bergamot; the llm-server VRAM claim does not resize because b11146 logs its buffers only at `-lv 4` (owner decision: pass `-lv 4` and accept about 24 log lines per request, or measure per-process GPU memory another way). See [RELEASE_AND_ENGINEERING.md → P4 exit matrix](RELEASE_AND_ENGINEERING.md#p4-exit-matrix-translation-stack).

## P5 outcomes (binding amendments for P6–P7)
Recorded by P5-I0 (2026-10-01) from the P5 workstream results. Where this section and a workstream entry disagree, this section wins. Contracts: [DEVELOPER_GUIDE.md → Voice families](DEVELOPER_GUIDE.md#voice-families-p5) and [Voice library](DEVELOPER_GUIDE.md#voice-library-voice-library-p5-f), [`params_contract.json`](../src-tauri/src/services/voice_changer/params_contract.json) (P5 fields and commands); decisions: [ADR 0006 → P5 outcomes](adr/0006-vc-zero-shot-singing.md#p5-outcomes); measurements: [VC_BENCHMARKS.md → P5 voice families](benchmarks/VC_BENCHMARKS.md#p5-voice-families); exit evidence: [RELEASE_AND_ENGINEERING.md → P5 exit matrix](RELEASE_AND_ENGINEERING.md#p5-exit-matrix-voice-families).

### Every phase
- **Voice families.** A voice family is a `vc::families::<x>` module plus a `FamilyKind` variant and a `manifest.format`; the registry dispatches `available` / `info` / `create` / `check_slot` / `normalize_options` / `create_slot` / `run_import_job`, the factory's `create_family` arm loads it behind the shared harness, budget, metrics and TTS aux path, and the UI is driven entirely by `VcCapabilities.families` (a new family needs only its i18n keys). Five families ship: `ddsp_svc` (DDSP-SVC 6.3 singing, CUDA + DirectML), `sigil_zs` (Seed-VC real-time zero-shot, CUDA + DirectML), `meanvc` (MeanVC2, experimental, CPU on every rung), `sovits` (so-vits-svc 4.x ONNX, experimental), `sidecar_v2` (X-VC, StreamVoiceAnon, seed-vc-realtime, custom; experimental). Family options live in `manifest.family` and apply when the voice next loads; the per-voice `reference_voice_id` lives in `slot.json` (class "switch").
- **Output buffer for burst families.** A family whose blocks compute in one burst declares `families::min_output_buffer_ms(kind, ep, manifest)` (Sigil-ZS per preset and EP: `sigil_low` 95 / 165 ms on CUDA, 85 / 155 on DirectML, never below 80 / 150; MeanVC2 80 / 150 for its 160 ms tick; DDSP-SVC 70 / 140 as the fallback), and a loaded voice that knows its EP returns its own in `FamilyEngine::min_output_buffer_ms` (DDSP-SVC: 70 / 140 to 150 / 300 by preset and EP); the worker plays the active voice with the larger of it and the user's prefill / queue, and the benchmark's paced runs do the same. A new burst family declares its own minimum from its measured p99 tick (`sigil_zs::tune::output_buffer_ms` is the rule: p99 + 15 ms, rounded up to 5 ms).
- **Reference voices** come from the shared voice library only (`voice_library::reference_clip`; embeddings through `families::reference::cached_embedding`, stored as `voices/<id>/cache/<backend>.bin`, invalidated by the clip's sha256). A new consumer (TTS cloning in P6) uses the same API and its own backend name.
- **Identify foreign ONNX files at load.** Every loader of a user-supplied or store ONNX calls `vc::rvc::foreign_graph_problem` (RVC, the content-encoder pool, RMVPE, SwiftF0, CREPE, FCPE, so-vits do), so a wrong file is named for what it is. Content encoders pick their output by width (`feature_output_index`).
- **Benchmarks of family voices** use the factory build; their offline ceiling is the family's own (`bench::ceiling::family_ceiling`; none for MeanVC2 and sidecars), `vram_engine_mb` is read after one harness build, and the autotuner refuses them until its family stage exists. Singing is scored with CREPE (`f0_method: "crepe"`) or pYIN (`scripts/bench/score_singing.py`): the bench's RMVPE analyzer reads singing above about 1 kHz an octave low. The P5 spec templates and `-Profile p5` are in `scripts/bench/`.
- **Read paced numbers with the GPU clock in mind.** An otherwise idle NVIDIA GPU drops to about 900 MHz for a once-per-block burst, stretching paced ticks 1.4–2.4× (spikes to 80–105 ms for Sigil-ZS); a game keeps the clocks up. P3's idle paced RVC numbers likely carry the same effect. Report the GPU state with paced runs; the user-facing mitigation is NVIDIA's "Prefer maximum performance" for Sigil.exe (documented, never set by Sigil).
- **Asset hosting stays local:** `vc/zs-seed-vc`, `vc/zs-seed-vc-fp16`, `vc/ddsp-svc-6.3`, `vc/vocoder-pc-nsf-hifigan-44k`, `vc/meanvc2` are Sigil exports (`unpublished`, hash-pinned, installed through Import or `SIGIL_LOCAL_ASSETS`); `vc/meanvc2-speaker` is unpinned (manual Google Drive download). The sidecar v2 scripts and `seed_vc_sampler.py` ship as installer resources (`resources/neural-sidecar`).

### P6 (streamer features)
- **TTS cloning (P6-E, P6-G, P6-I)** reads references from the voice library: mount `ReferenceVoicePicker` (`ReferenceVoiceRecorder.tsx`) with `needsTranscript`; transcripts are already made locally with installed STT models (`transcriptError` when none). Qwen3-TTS's clone registry caches its speaker data per voice through `read_cache` / `write_cache` with its own backend name.
- GPU helpers that hold their own memory report it with `resources::claim_external_vram` (the sidecar recipes do, with each recipe's estimate when the helper cannot measure).

### P7
- **H1 evidence** for T10: a friend's live-record of Sigil-ZS (`sigil_low` and `sigil_lean`) and DDSP-SVC on an RTX 3060 / 4060 under VRChat + OBS; the 4090 × 3 estimate says `sigil_lean` on DirectML for H1.
- **Autotuner family stage** for `sigil_zs` (measure each preset of `tune::PREFERENCE` on each GPU rung, `tune::choose`, write `execution_provider`, the output buffer and the preset), and an adaptive output buffer sized from the measured p99 tick instead of the static family minimum (the idle-clock spikes still underran at 80 ms: 75 per 20 s for `sigil_low` on CUDA).
- **Shared content encoder on the session-plan path** (IOBinding plus a CUDA graph per fixed shape): it is 65 % of DDSP-SVC's CUDA tick (two passes at about 9 ms against 2.6 ms graph-captured) and the largest part of RVC's T2 gap. Benchmark before / after on the main path.
- **Listening and real-voice checks:** a real trained DDSP-SVC 6.3 voice (only a seeded random voice was available), a real so-vits-svc 4.1 export A/B against upstream inference (then drop its `experimental` flag), MeanVC2 with the real speaker encoder (the A/B decides whether it can become the CPU default), a blind listen of Sigil-ZS's joins (join correlation 0.53–0.65), X-VC and StreamVoiceAnon with their checkpoints.
- **CPU presets need the Stable buffer:** with 35 / 70 ms the RVC CPU tier underran heavily in P5-C's A/B (the autotuned 200 ms preset about 1000 times per 20 s); P7's preset tuning pairs CPU presets with 80 / 150 ms or sizes the buffer from the tick.
- Move the pYIN-style `pitch_track` from `ddsp_svc/tests.rs` into `bench/quality.rs` as a singing analyzer; the CAM++ re-export for absolute similarity thresholds (carried from P3) applies to T10's similarity row too.
- Seed-VC's fp16 DiT / XLS-R / vocoder stay behind their switches (fp16 fails waveform parity), whatever their speed.

### P5 deviations from the workstream entries (recorded)
- Owned paths grew by the child-module convention: `families/ddsp_svc/**`, `families/meanvc/**`, `families/sovits/**`, `families/sidecar_v2/**`, plus `src-tauri/test-fixtures/vc/ddsp_svc/` (732 KB CI fixture), `scripts/vc-families/ddsp_svc/export_templates.py`, `scripts/vc-families/meanvc/rust_reference.py`, `vc-meanvc.json` (assigned to P5-C by the scaffold). P5-I0 moved maintainer scripts out of the scratch folder into the repo: `scripts/vc-families/seed_vc/make_vectors.py`, `scripts/vc-templates/fcpe_mel_extract.py`, `scripts/bench/{singing_sweep,score_singing}.py`, `scripts/bench/specs/p5/*.template.json`, `scripts/bench/profiles/p5.json`.
- DDSP-SVC uses per-config template graphs without weights plus a Rust importer (no per-voice export, no Python at import), and runs on DirectML too (P5-A rebuilt the `ddsp` graph's inverse STFT).
- Sigil-ZS's preset prompts changed by measurement (`sigil_low` 4 s, `sigil_lean` 3 s) and the prompt is taken from the clip's speech; its streaming configuration is not chosen by the autotuner yet (`tune.rs` decides; the autotuner's family stage is P7).
- MeanVC2's 80 ms tick runs as two 40 ms harness blocks (one DiT call each), and it computes on the CPU on every rung.
- so-vits-svc is `experimental` until a real export is A/B-tested.
- P5-I0 applied: the family output-buffer minimum (worker + paced bench), DDSP-SVC's singing defaults in `slot.json` (CREPE, singing range), the GTCRN rate check (a 22.05 kHz sidecar voice used to go silent with noise removal on), the reference reset on entering a voice without one, family ceilings / family EP outcomes / VRAM after the harness in the benchmark, the autotuner's refusal of family voices, the content-encoder output by width and foreign-graph checks in the shared pool and the pitch loaders, the sidecar scripts as installer resources, `voice_library::reference_clip` off the async runtime in `cmd_family`. Not applied (P7 above): the autotuner's family stage, the shared encoder on the plan path, the bench's pYIN analyzer.
- The P5 final gate applied the review fixers' late requests: per-voice output-buffer minimums (`families::min_output_buffer_ms(kind, ep, manifest)`: Sigil-ZS per preset and EP, `sigil_low` 95 / 165 ms on CUDA at +10.7 ms of latency; MeanVC2's 160 ms tick 80 / 150; DDSP-SVC per preset and landed EP through `FamilyEngine::min_output_buffer_ms`), the load cancel for sidecar voices (`EngineRequest::cancel` → `FamilyRequest::cancel`: a background load stops when another voice is selected, the worker's first load on Stop or a newer start), and the P5-C / P5-F / P5-G i18n merges (P5-C rewords `family_meanvc_tick_info`). Not applied (P7 above): the worker's adaptive output buffer from the running voice's measured p99 tick (`sigil_zs::tune::burst_buffer_ms`).

### P5 manual checks still open
A microphone take in `pnpm tauri dev` (record, save, see it in the zero-shot picker, transcript arrives; `cargo test real_microphone -- --ignored`), a zero-shot voice from a library voice in the app (create, switch reference while running, hot switch), a DDSP-SVC `.pt` / `.zip` import in the app and a listening / T10 check with a real trained voice, a real so-vits-svc export, the MeanVC2 speaker encoder (manual Google Drive download) and its A/B, X-VC / StreamVoiceAnon with their checkpoints and a sidecar voice in the app, a release installer that ships `resources/neural-sidecar`, and the H1 live-record. See [RELEASE_AND_ENGINEERING.md → P5 exit matrix](RELEASE_AND_ENGINEERING.md#p5-exit-matrix-voice-families).

## Workstreams

### P0-S0 [P0, L] P0 scaffold: checkpoint, conventions, splits, tooling, stubs
Goal: Take the phase checkpoint. Make build.rs read src-tauri/plugin-commands/*.txt (lists migrated verbatim). Add the i18n staging merger and retire the destructive merge scripts. Add vitest (test:unit, plus a CI step). Split voice_changer.rs mechanically into voice_changer/{mod,params,state,playback,aux,worker,commands,persist,fallback,metrics,debug}.rs with tests moved and push_aux_wav_bytes re-exported at the same path. Add the headless clap branch before the port check. Align the port pre-check to 0.0.0.0. Set GGML env in .cargo/config.toml. Add tracing dependencies. Create stubs with frozen signatures: speech_core, rt_thread, diagnostics, localPages.ts. Add root schema keys.
Research: Build and registration infrastructure for all P0 items; Pitfalls: registration drift, destructive i18n scripts, port pre-check mismatch, no log sink
Owned: src-tauri/Cargo.toml, src-tauri/Cargo.lock, src-tauri/build.rs, src-tauri/src/main.rs, src-tauri/src/services/mod.rs, src-tauri/src/services/voice_changer.rs (split then removed), src-tauri/capabilities/**, src-tauri/tauri.conf.json, package.json, pnpm-lock.yaml, .github/workflows/ci.yml, .gitignore, src/core/schema.ts, src/types.ts, public/i18n/**, scripts/merge-neural-i18n.mjs, scripts/merge-http-tts-i18n.mjs
New: src-tauri/plugin-commands/<plugin>.txt (all 18 existing plugins, plus diagnostics.txt), src-tauri/.cargo/config.toml ([env] GGML_NATIVE=OFF, GGML_AVX2=ON), src-tauri/model-catalog/{README.md,_schema.json}, src-tauri/src/services/voice_changer/*.rs, src-tauri/src/services/{speech_core/mod.rs,rt_thread.rs,diagnostics/mod.rs,cli/mod.rs} (stubs), src/utils/localPages.ts (stub), vitest.config.ts, scripts/merge-i18n-staging.mjs, scripts/i18n-staging/README.md, docs/adr/0000-template.md
Hotspots: build.rs becomes data-driven. gen/schemas acl-manifests must be byte-identical before and after. | Cargo: tracing, tracing-subscriber, tracing-appender; windows features Win32_System_JobObjects, Win32_Graphics_Dxgi, Win32_System_Console, Win32_System_LibraryLoader; license = AGPL-3.0-only. | package.json: vitest plus `test:unit`. Install with --network-concurrency=2. | Root schema keys: lanOriginCheck (default true), modelMirror (later P1), and STT VAD / Whisper-hardening fields owned by P0-D (added to stt/schema.ts by P0-D, not here). | Stubs register the diagnostics plugin and its capability file.
Depends: 
Verify: pnpm install --network-concurrency=2; pnpm typecheck; pnpm test:unit; pnpm i18n:check; pnpm build | cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets; cargo test --manifest-path src-tauri/Cargo.toml | git diff --stat -- src-tauri/gen/schemas (empty) | src-tauri/target/debug/Sigil.exe --headless help exits without binding the port
Accept: No user-visible behaviour change. | The voice_changer split is purely mechanical and its 16 tests are green. | Every P0 implementer can build against the stubs without touching a hotspot.
Assets: 
Risk: medium: the mechanical split of a 2,670-line file, and the build.rs refactor. Verified by an identical ACL diff and all tests green.
Notes: Frozen stub signatures are listed in each stub's doc comments, including speech_core::{VadConfig, Segmenter, UtteranceEvent}, rt_thread::{register_mmcss_pro_audio_once, opt_out_power_throttling, set_worker_priority}, and diagnostics::EpOutcome.

### P0-A [P0, L] Voice-changer real-time hardening and Rust-side persistence
Goal: Real-time fixes:
- Aux queue becomes a VecDeque with back-pressure (60 s cap), rubato resampling, a lock that tolerates poisoning, and an aux mic gate (none/duck/mute).
- A park/unpark wake from the input callback replaces the 2 ms sleep-poll.
- MMCSS on capture, output and monitor callbacks.
- Worker priority auto/protect_game/protect_voice.
- The input ring drops the oldest audio when full.
- A latency budget (default 300 ms) for every mode, keeping exact length accounting.
- A stateful monitor resampler.
- Models load before streams open. Any capture sample format goes through pcm_convert. The sidecar read timeout equals the budget. The 'crepe' alias is hidden.

Also: metrics for p99, backlog, dropped audio, underruns and the EP in use, and atomic settings.json persistence including devices.
Research: §6 aux queue cap drains the front; §6 2 ms sleep-poll and BELOW_NORMAL; §4.3#3 audio-thread priority (MMCSS on callbacks, never the worker); §4.3#2 latency budget that drops stale audio (all engines and sidecars); Extra bugs: drop-newest ring, stateless monitor resampler, load after streams, F32/I16-only capture, sidecar without timeout, crepe alias, no persistence
Owned: src-tauri/src/services/voice_changer/**, src-tauri/src/services/rt_thread.rs, src-tauri/src/services/neural_audio/{sidecar_bridge.rs,onnx_voice.rs,chunk_processor.rs}, src-tauri/src/services/neural_audio/vc/stream.rs, src-tauri/plugin-commands/voice-changer.txt
New: src-tauri/src/services/voice_changer/budget.rs
Hotspots: 
Depends: P0-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml voice_changer neural_audio. Existing tests plus new ones:
- the budget drops the oldest whole frames and keeps accounting exact;
- a 60 s aux clip stays intact;
- the monitor resampler at 44.1k to 48k is exact over 10 simulated minutes;
- a poisoned aux lock recovers;
- persistence round-trips. | Manual: RVC and classic sessions, idle CPU below 1%, MMCSS log line, settings restored after restart
Accept: No sleep-poll remains. The worker is never registered with MMCSS. | Backlog stays ≤ budget + 1 block under an injected stall. | Long TTS through the VC is never truncated.
Assets: 
Risk: medium: real-time path. Mitigated by the split landing first and the existing harness and ring tests.
Notes: 

### P0-B [P0, M] Voice-changer TS: hydration, budget, EP status, metrics UI
Goal: Hydrate every parameter (including the input device) from Rust persistence. Add controls for the latency budget, aux mic gate and worker priority. Show the active EP, with a warning when it differs from the requested one. Show p99, backlog and dropped metrics. List slots using the existing list_voice_changer_model_slots. Move consent into the Rust settings.
Research: §4.3#10 surface EP failures in the UI; Persistence of VC settings (UI side)
Owned: src/core/services/voice_changer/**, src/core/ui/inspector/inspector_voice_changer.tsx
New: src/core/services/voice_changer/__tests__/params.test.ts, scripts/i18n-staging/P0-B.json
Hotspots: 
Depends: P0-S0
Verify: pnpm typecheck; pnpm test:unit (the TS normalize lists match the Rust allow-lists through a shared JSON fixture)
Accept: No silent option remapping between TS and Rust. | The EP actually in use and the latency metrics are visible.
Assets: 
Risk: low
Notes: 

### P0-C [P0, L] Silero VAD segmenter, rubato resampling, Whisper hardening, utterance capture
Goal: speech_core:
- Silero v6.2 on Sigil's ORT (CPU, 1 thread, spinning off, stateful 512-sample frames).
- Segmenter: hysteresis 0.5/0.35, 400 ms pre-roll, 500 ms end silence, forced flush at 18 s, nothing emitted on silence.
- Streaming rubato to 16 kHz mono.
- Per-language whole-segment hallucination phrases, and repeat removal only at 3 or more (character n-grams for CJK).

whisper.rs:
- Replace the RMS gate and the nearest-neighbour decimation.
- set_suppress_nst, no-speech drop, max_tokens cap, one reused WhisperState, configurable n_threads, initial_prompt hotwords.
- One ordered worker with a backlog cap, replacing a thread per segment.
- Add the missing q5_1, q8_0 and large-v3-turbo ids, an error (not base.en) for unknown ids, and an Anime-Whisper BYO preset.

audio plugin: start_utterance_capture emits VAD-closed utterances for HTTP engines.
Research: §1.2 Silero VAD v6.2 gate; §1.2 rubato; §1.2 Whisper hardening; §6 whisper.rs RMS gate and uncapped buffer; §5 hotwords part 1 (Whisper initial prompt); §1.1 Anime-Whisper BYO preset; §1.4 Groq prerequisite (VAD segmentation, Rust side)
Owned: src-tauri/src/services/speech_core/**, src-tauri/src/services/whisper.rs, src-tauri/src/services/audio/**, src-tauri/src/services/pcm_convert.rs, src-tauri/plugin-commands/{whisper,audio}.txt
New: src-tauri/src/services/speech_core/{silero.rs,segmenter.rs,resample.rs,hallucination.rs,utterance.rs}, src-tauri/src/services/speech_core/phrases/{en,ja,zh,ko,es,fr,de,ru,pt}.txt, src-tauri/resources/models/silero_vad_v6_2.onnx (include_bytes)
Hotspots: 
Depends: P0-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech_core whisper. Cases:
- 10 s of silence gives 0 segments;
- pre-roll is kept;
- the 18 s forced flush splits at the lowest-probability point;
- a 12 kHz tone is down at least 60 dB after resampling;
- 'thanks for watching' is dropped only as a whole segment, and 'let's go' is kept;
- a CJK loop is collapsed;
- the ordered worker drops the oldest past its cap. | Manual: 5 minutes of silence gives 0 transcribe calls in the log
Accept: Whisper never receives silence, and segments come out in order with a bounded backlog. | The Silero I/O contract (state tensor [2,1,128], 64-sample context) is verified against the real model.
Assets: silero_vad.onnx v6.2 (~2 MB, MIT): one download
Risk: medium: the Silero v6 I/O contract, and whether whisper-rs exposes no_speech_prob
Notes: 

### P0-D [P0, L] STT TS: status channel, utterance polling, Groq preset, mute, shortcuts, desktop default
Goal: - Split status and progress into serviceState.statusText and progress; they still re-arm the connecting fallback and never reach publishText.
- rustCapturePolling consumes utterances, sending one request each.
- Add the openai_audio Groq preset (whisper-large-v3-turbo) with a 20 req/min token bucket and merging of utterances below the minimum length.
- Add Service_STT.setMuted(bool) and wire the muteMic, start and muteSound shortcuts.
- Migrate native to Whisper once for desktop users.
- VAD and Whisper-hardening fields in stt/schema.ts, all zSafe.
- Verify and expose the Deepgram model=nova-3 and Azure segmentation-timeout fixes.
- Translate the Whisper/Vosk inspector panels into i18n.
Research: §1.4 Groq whisper-large-v3-turbo; §6 Deepgram model=nova-3; §6 Azure Speech_SegmentationSilenceTimeoutMs; Extra bugs: status strings leaking into captions, dead shortcuts, broken native default on desktop, 500 ms polling finals
Owned: src/core/services/stt/**, src/core/ui/inspector/inspector_stt.tsx, src/core/services/keyboard/**, src/utils/rustAudioInput.ts
New: src/core/services/stt/{rateLimiter.ts,statusChannel.ts,migrate.ts}, src/core/services/stt/__tests__/{rateLimiter,statusChannel,migrate}.test.ts, scripts/i18n-staging/P0-D.json
Hotspots: 
Depends: P0-S0, P0-C (event contract via stub)
Verify: pnpm typecheck; pnpm test:unit -- src/core/services/stt | Manual: a fresh profile starts STT after one-click download; no status text in captions; the mute shortcut works
Accept: No status string is ever published as a caption. | Groq never exceeds 20 requests/min, and finals are queued, never dropped.
Assets: 
Risk: medium: regressions in untouched backends
Notes: 

### P0-E [P0, M] TTS TS fixes and AivisSpeech preset
Goal: Fixes:
- Service_TTS.stop sets status, and every backend calls onStop.
- inputField dedupe.
- Azure SSML: XML escaping, xmlns, xml:lang from the voice.
- `parseFloat ?? 1` replaced with Number.isFinite.
- Native TTS creates a fresh utterance each time.
- azureVoices data fixes; Service_Sound surfaces errors; the onStop toast is i18n'd.
- Remove tiktokVoices and onFilePlayRequest.
- Verify the in-tree pitch step (0.01, plus reset) and the voicevox `??` fix.

Add a VOICEVOX hostPreset (voicevox :50021 / aivisspeech :10101 / custom) that resets pitch to 0 for AivisSpeech. Pass the Fish apiKey and model through.
Research: §2.1 AivisSpeech preset (step 1); §6 pitch slider; §6 voicevox intonation; §2.2 VOICEVOX/AivisSpeech inspector fixes; §2.1 VOICEVOX (keep)
Owned: src/core/services/tts/**, src/core/services/sound/**, src/core/ui/inspector/inspector_tts.tsx, src/core/ui/sigil/StatusFooter.tsx
New: src/core/services/tts/{ssml.ts,num.ts}, src/core/services/tts/__tests__/{ssml,num}.test.ts, scripts/i18n-staging/P0-E.json
Hotspots: The integrator removes the tts.tiktok_title key from all locales.
Depends: P0-S0
Verify: pnpm typecheck; pnpm test:unit -- src/core/services/tts | Manual: Azure speaks text containing '&'; Stop resets every backend; AivisSpeech preset connects
Accept: Every listed bug has a test or a manual check. | No regression in existing backends.
Assets: 
Risk: low
Notes: 

### P0-F [P0, M] Rust TTS, HTTP and translate fixes
Goal: - fish_speech: /v1/health, /v1/references/list, streamed WAV, optional Bearer key and `model` header.
- Uberduck: verify the /v1/text-to-speech migration and the device fix.
- http_client: per-request timeout helper (TTS 30 s, health 10 s, translate 20 s) and UTF-8-safe describe_failed_response.
- SAPI rate clamped to -10..10.
- get_kokoro_voices accepts both response shapes.
- translate.rs takes endpoint and key per call.
- All neural_onnx_tts commands become async with spawn_blocking.
- Timeouts for moonshine.
Research: §6 fish_speech; §6 Uberduck migration; §2.3 Fish S2.1 Pro free API (key and model header); §2.3 Uberduck /v1; Extra bugs: 2 h timeout, UTF-8 truncate panic, SAPI rate range, Kokoro voices shape, global translate endpoint, UI freeze from main-thread kittentts commands
Owned: src-tauri/src/services/{fish_speech.rs,http_client.rs,kokoro_tts.rs,chatterbox_tts.rs,melo_tts.rs,translate.rs,neural_onnx_tts.rs,neural_sidecar_tts.rs,moonshine_stt.rs}, src-tauri/src/services/{uberduck_tts,windows_tts}/**, src-tauri/plugin-commands/{fish-speech,kokoro-tts,chatterbox-tts,melo-tts,translate,neural-onnx-tts,neural-sidecar-tts,moonshine-stt,uberduck-tts,windows-tts}.txt
New: 
Hotspots: 
Depends: P0-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml http_client windows_tts kokoro fish translate | cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets
Accept: No API call can hang for 2 hours. | Japanese HTML error pages don't panic. | Kitten loading doesn't freeze the UI.
Assets: 
Risk: low: rebase on the in-flight edits in the tree
Notes: 

### P0-G [P0, M] Translation and transform correctness
Goal: Translation:
- Per-utterance sequencing replaces the #id gate: latest wins for interims, finals are never dropped.
- Azure fetch wrapped in try/catch, with location optional.
- Start/stop epoch guards for LibreTranslate and onDevice; getLanguages passes the endpoint per call.
- The '[...]' cancel marker is not translated.
- Verify the onDevice typeof fix and the openRouterModel default.

Transform:
- max_tokens configurable (default 1024) and reasoning_effort.
- An AbortController on timeout.
- No '[AI Rate Limited]' literal published.
- Ordered publishing.
- transform_raw gets a final when AI is off.
- stopWithStream works, live config changes apply, openai.ts is deleted, and the inspector uses i18n.
Research: §6 onDevice typeof; §6 stale openRouterModel; §3.3 Gemini Flash-Lite blockers (100-token cap, reasoning_effort); Extra bugs: final-dropping #id gate, Azure unhandled rejection, start/stop race, transform ordering and literal publish
Owned: src/core/services/translation/**, src/core/services/transform/**, src/core/ui/inspector/inspector_translation.tsx, src/core/ui/inspector/inspector_transform.tsx
New: src/core/services/translation/sequencer.ts, src/core/services/translation/__tests__/sequencer.test.ts, src/core/services/transform/__tests__/ordering.test.ts, scripts/i18n-staging/P0-G.json
Hotspots: 
Depends: P0-S0, P0-F (translate per-call signature)
Verify: pnpm typecheck; pnpm test:unit -- src/core/services/translation src/core/services/transform
Accept: A final is still published when an interim from the next utterance is in flight. | Transform output is never truncated at 100 tokens.
Assets: 
Risk: low
Notes: 

### P0-H [P0, L] GPU EP delivery, EP probe, diagnostics log, atomic settings, release hardening
Goal: - ort_session: error_on_failure, then an explicit CPU retry, returning EpOutcome{requested, active, reason}.
- Startup EP probe: providers, NVIDIA driver, DXGI adapters, CUDA/cuDNN DLL versions.
- NVIDIA redist runtime pack for CUDA 12 (current rc.11), preloaded with AddDllDirectory. This is an interim pack until the model store exists; it uses a minimal sequential downloader with sha256.
- Rotating file log (10 MB × 5) forwarding warnings and errors to the system log. EcoQoS opt-out.
- Settings saved atomically (temp file, rename, .bak); a corrupt file is backed up, never overwritten.
- release.yml: GGML_NATIVE=OFF and AVX2, cargo test, cache ort.pyke.io, upload NSIS, and a verify-bundle script.
Research: §4.3#10 surface EP failures and check CUDA/cuDNN at startup; Build pitfalls: EP DLLs not shipped, GGML_NATIVE portability, no cargo test in release; Extra bugs: silent settings loss, no log sink
Owned: src-tauri/src/services/neural_audio/ort_session.rs, src-tauri/src/services/diagnostics/**, src-tauri/plugin-commands/diagnostics.txt, src/core/services/diagnostics/**, src/core/services/systemLog.ts, src/core/services/state/index.ts, .github/workflows/release.yml, scripts/release/**
New: src-tauri/src/services/diagnostics/{logging.rs,ep_probe.rs,qos.rs,gpu_mem.rs,nvidia_pack.rs}, scripts/release/{stage-native-dlls.mjs,verify-bundle.ps1}, docs/adr/0003-gpu-ep-delivery.md, scripts/i18n-staging/P0-H.json
Hotspots: Integrator: tauri.conf.json bundle.resources maps onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll and DirectML.dll to the install root; beforeBuildCommand runs stage-native-dlls.mjs; bundle.windows.nsis.installMode is currentUser. | Integrator: main.rs setup calls diagnostics::logging::init and qos::opt_out_power_throttling.
Depends: P0-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml diagnostics ort_session | pnpm test:unit (state backup logic) | pnpm tauri build; pwsh scripts/release/verify-bundle.ps1; dumpbin /dependents shows no new load-time DLLs | Manual: CUDA after the pack is installed; DirectML without it; CPU plus a warning without DirectML.dll
Accept: The EP in use is reported truthfully in the UI and in the log. | The packaged build runs CUDA on the dev rig. | A truncated settings file is kept as .bak.
Assets: NVIDIA CUDA 12 redist components (~1 GB, one file at a time, manual dev test only)
Risk: high: provider DLL placement assumption. Fallbacks are in the decisions.
Notes: 

### P0-I [P0, M] LAN security stage 1
Goal: - Origin allow-list on /pubsub and /peer (tauri origins, dev :1420, localhost/127.0.0.1/LAN IP on the port; a missing Origin is allowed only from loopback), with a kill switch.
- Binary frames are refused, and the Whisper injection path is removed.
- Inbound topic allow-list in consumePubSubMessage.
- XSS fixes in index.tsx and inputfield.html; inputfield.html no longer hard-codes its port.
- A bind failure emits an error event instead of spinning.
- localPageUrl() helper.
Research: §5 authenticated LAN server, stage 1 (Origin allow-list, remove unauthenticated audio injection); Extra bugs: cross-site WebSocket hijacking, XSS sinks, emitter spin loop
Owned: src-tauri/src/services/web/**, src-tauri/plugin-commands/web.txt, src/shared/services/pubsub/index.ts, src/index.tsx, public/{mic,obs-captions,inputfield}.html, src/utils/localPages.ts
New: src-tauri/src/services/web/origin.rs, scripts/i18n-staging/P0-I.json
Hotspots: Integrator removes whisper::process_audio_chunk if nothing else uses it.
Depends: P0-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml web:: | $env:PW_PREVIEW_ONLY=1; pnpm test:e2e | Manual matrix: host, LAN /client, OBS, mic.html, obs-captions, Link apps, and a foreign page (rejected)
Accept: Every legitimate client connects and every foreign origin is rejected.
Assets: 
Risk: medium: an Origin rule could block a legitimate client (there is a kill switch)
Notes: 

### P0-I0 [P0, M] P0 integrate
Goal: Merge i18n staging and run i18n:sync. Apply the hotspot requests: bundle resources, main.rs setup hooks, the tiktok key, dead-code removal. Update docs: OPERATOR_REFERENCE (STT reliability with VAD and Groq, EP status and logs, VC budget and persistence), RELEASE_AND_ENGINEERING (threat model, GPU EP smoke, §6 verification table), DEVELOPER_GUIDE (plugin-commands convention). Mark VOICE_CHANGER_AI_PLAN.md as superseded. Run the full gates plus the P0 exit matrix.
Research: §7 P0 exit (integration only)
Owned: (all P0-S0 hotspots), docs/**, CLAUDE.md
New: docs/adr/README.md
Hotspots: All P0 integration requests
Depends: P0-A, P0-B, P0-C, P0-D, P0-E, P0-F, P0-G, P0-H, P0-I
Verify: pnpm typecheck; pnpm i18n:check; pnpm test:unit; pnpm build; $env:PW_PREVIEW_ONLY=1; pnpm test:e2e | cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets; cargo test --manifest-path src-tauri/Cargo.toml | pnpm tauri build plus the manual exit matrix
Accept: P0 exit criteria met
Assets: 
Risk: medium
Notes: 

### P1-S0a [P1, L] ORT gate: vendor kittentts, upgrade to ort rc.13
Goal: Take the checkpoint. Vendor kittentts into local_tts/kitten/ (Apache-2.0 headers plus NOTICE), depend on espeak-ng 0.1 directly with bundled-data features, and remove hf-hub; Kitten hub loading temporarily uses a direct http_download_get. Bump ort to =2.0.0-rc.13 with cuda, directml, nvrtx, tensorrt, half, preload-dylibs, copy-dylibs and ndarray, and migrate every call site. Verify register_ep_library for onnxruntime_providers_cuda.dll loaded from an app_data folder. Smoke test the VC CUDA/DML/CPU paths, Silero and Kitten. Write ADR 0001, with the fallback ladder rc.12 then rc.11.
Research: §1.3 kittentts pin blocking ORT upgrade and parakeet-rs; §4.3#6 DirectML high-priority queue prerequisite (rc.13)
Owned: src-tauri/Cargo.toml, src-tauri/Cargo.lock, src-tauri/src/services/neural_audio/**, src-tauri/src/services/neural_onnx_tts.rs, src-tauri/src/services/speech_core/**, src-tauri/src/services/voice_changer/**, src-tauri/src/services/diagnostics/**
New: src-tauri/src/services/local_tts/{mod.rs,kitten/**}, docs/adr/0001-ort-upgrade.md
Hotspots: Cargo: remove kittentts; bump ort; add espeak-ng, half, libloading, zip 2, tar, bzip2, flate2; windows 'implement' feature
Depends: P0-I0
Verify: cargo clippy --all-targets; cargo test (all existing suites) | Manual: VC on CUDA 13 and DirectML; neuralOnnx speaks; Silero gate works
Accept: The tree is green on the chosen ORT version. | kittentts and hf-hub are absent from Cargo.lock. | The ADR records the provider-registration result.
Assets: pyke ORT 1.28.0 dist cuda13+tensorrt+nvrtx+directml (one-time build download, needs user consent)
Risk: high: API churn and the CUDA 13 floor. It runs alone, with checkpoint rollback.
Notes: 

### P1-S0b [P1, M] P1 scaffold: foundation plugin stubs and Models tab plumbing
Goal: Register stub plugins with command lists and capability files: model-store, runtime, bench, process-host. Add the headless arms fetch, bench, engine-probe and vc-import-worker. Add RunEvent::Exit calling process_host::shutdown_all. Add the Models inspector tab (InspectorTabPath, nav, title, ApiServer.models, sysinfo and audioDevices init tasks). Add the root keys modelMirror, modelMirrorCustomUrl and modelMaxConcurrentDownloads. Create empty catalog fragments and the native/common/sigil_engine.h ABI v1.
Research: Phase scaffolding
Owned: src-tauri/build.rs, src-tauri/src/main.rs, src-tauri/src/services/mod.rs, src-tauri/capabilities/**, src-tauri/tauri.conf.json, src/core/schema.ts, src/core/index.ts, src/types.ts, src/core/ui/servicePanels.tsx, src/core/ui/inspector/index.tsx, src/core/ui/sigil/SigilNavigation.tsx, public/i18n/**
New: src-tauri/src/services/{model_store,native_runtime,sysinfo,resources,process_host,bench,speech_rt}/mod.rs (stubs), src-tauri/model-catalog/{runtimes.json,stt-legacy.json,base.json}, native/common/sigil_engine.h, src/core/services/{models,sysinfo,resources}/index.ts (stubs), src/core/services/audioDevices.ts (stub), src/core/ui/inspector/inspector_models.tsx (stub mounting GpuRuntimeCard stub)
Hotspots: Registrations and capabilities for model-store, runtime, bench and process-host | Models tab wiring and root schema keys
Depends: P1-S0a
Verify: All gates
Accept: Stubs compile, and the Models tab renders a placeholder.
Assets: 
Risk: low
Notes: 

### P1-A [P1, L] Model store core
Goal: Implement the model-store design from the decisions: catalog, Range resume, mirrors, sha256/git-oid, safe extraction, atomic install, queue, cancel, import_local, disk usage, legacy adoption of vosk and whisper paths. Move Vosk downloads onto it with a streaming import, fixing the whole-file RAM read. Add the Rust API ensure/installed_path, the pin-catalog script, and the dev override.
Research: §5 model installer with mirror fallback; Infrastructure for the zero-setup auto-download rows (§1.1, §2.1, §3.1, §4.2)
Owned: src-tauri/src/services/model_store/**, src-tauri/src/services/http_client.rs, src-tauri/src/services/vosk_stt.rs, src-tauri/model-catalog/{base.json,stt-legacy.json}, src-tauri/plugin-commands/{model-store,vosk-stt}.txt, scripts/models/**
New: model_store/{catalog.rs,queue.rs,download.rs,mirrors.rs,verify.rs,extract.rs,install.rs,legacy.rs,commands.rs,events.rs,tests.rs}, scripts/models/pin-catalog.mjs
Hotspots: 
Depends: P1-S0b
Verify: cargo test --manifest-path src-tauri/Cargo.toml model_store, against a local warp fixture server:
- resume after abort;
- 404, HTML or hash mismatch moves to the next mirror;
- zip-slip and symlinks rejected, ratio cap enforced;
- one download at a time;
- cancel keeps the .partial. | Sigil.exe --headless fetch base.test-small
Accept: A corrupt or partial file is never installed. | Existing files are adopted without re-downloading.
Assets: 
Risk: medium
Notes: 

### P1-B [P1, M] Models and runtimes UI
Goal: Build the Models & Runtimes tab: entries grouped by kind with size and state, a progress bar showing mirror and speed, cancel, delete and verify, disk usage, mirror selection and custom URL, and an offline file import. Add a reusable <ModelRequirement ids>. Generalize the download progress formatter.
Research: §5 model installer (UI)
Owned: src/core/services/models/**, src/core/ui/inspector/inspector_models.tsx, src/core/ui/inspector/components/ModelRequirement.tsx, src/core/services/stt/downloadProgress.ts
New: src/core/services/models/__tests__/format.test.ts, scripts/i18n-staging/P1-B.json
Hotspots: 
Depends: P1-S0b
Verify: pnpm typecheck; pnpm test:unit -- src/core/services/models
Accept: Every store command is reachable from the UI.
Assets: 
Risk: low
Notes: 

### P1-C [P1, L] Capture hub, loopback sources, stall recovery and device-follow
Goal: Add audio/hub.rs: one cpal stream per source (input, or an output device opened for loopback), fanned out through rtrb with QPC timestamps. Consumers get Raw16kMono or DeviceRate. A lagging consumer drops its own oldest data. MMCSS via rt_thread. The error callback triggers a watchdog restart under the same epoch and emits audio:stream_error / audio:stream_recovered. A device watcher uses IMMNotificationClient, falling back to polling, and emits audio:devices_changed. Sessions are keyed, so mic and loopback coexist. Add a Rust mute gate. Migrate whisper start_recording and the legacy start_audio_capture onto the hub. The TS audioDevices module auto-restarts STT; pickers list loopback sources and refresh on events.
Research: §4.3#7 one owned mic capture, fanned out (hub plus STT raw copy); §1.2 keep transcribing the raw mic before the voice changer; §5 follow audio-device changes and recover stalled streams; §5 loopback phase 1 plumbing (output devices as capture sources)
Owned: src-tauri/src/services/audio/**, src-tauri/src/services/pcm_convert.rs, src-tauri/src/services/rt_thread.rs, src-tauri/src/services/whisper.rs, src-tauri/plugin-commands/audio.txt, src/core/services/audioDevices.ts, src/core/ui/inspector/components/input/index.tsx, src/utils/rustAudioInput.ts
New: src-tauri/src/services/audio/{hub.rs,consumer.rs,watchdog.rs,device_watch.rs,loopback.rs,hub_tests.rs}, scripts/i18n-staging/P1-C.json
Hotspots: 
Depends: P1-S0b
Verify: cargo test --manifest-path src-tauri/Cargo.toml audio, using a fake source: identical fan-out, exact 16 kHz output, drop-oldest on lag, restart with the same epoch, mute gate. | Manual: Whisper and Vosk run on one stream; USB unplug and replug recovers within 2 s; an output device appears as a loopback source
Accept: At most one stream per physical source, and no raw PCM over IPC for Rust engines.
Assets: 
Risk: medium-high: cross-cutting audio refactor. The old path stays behind an internal flag.
Notes: 

### P1-D1 [P1, L] Native runtime loader, sigil_engine ABI, NVIDIA CUDA 13 runtime pack
Goal: native_runtime:
- LoadLibraryExW with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|DEFAULT_DIRS and AddDllDirectory.
- Hash-verified lazy load, ABI version check, and variant selection (cuda13, then vulkan, then cpu-avx512, then cpu-avx2).

Native build infrastructure: the ABI v1 implementation, CMake presets (/MD, static ggml per DLL, GGML_NATIVE=OFF, BACKEND_DL off for in-process DLLs), an echo engine, build and package scripts, and a runtime-packs.yml workflow_dispatch that builds vulkan and cuda13 on runners.

The NVIDIA pack (cudart, cublas, cublasLt, cufft, cuDNN 9 for CUDA 13) moves to the model store. The CUDA provider is registered from app_data via register_ep_library, or placed next to the exe on per-user installs.
Research: §0 each ggml runtime as a separate DLL; §1.3 ggml-as-DLL / ORT alignment policy; GPU EP delivery finalization (post-rc.13)
Owned: src-tauri/src/services/native_runtime/**, src-tauri/src/services/neural_audio/ort_session.rs, native/CMakeLists.txt, native/common/**, native/cmake/**, native/engines/echo/**, scripts/native/**, .github/workflows/runtime-packs.yml, src-tauri/model-catalog/{runtimes.json,runtime-nvidia.json}, src-tauri/plugin-commands/runtime.txt
New: native_runtime/{loader.rs,abi.rs,pack.rs,variant.rs,dll_dirs.rs}, scripts/native/{build-runtime-packs.ps1,versions.json,README.md}
Hotspots: Integrator: move onnxruntime_providers_cuda.dll out of bundle.resources if register_ep_library is verified
Depends: P1-S0b, P1-A (ensure API)
Verify: pwsh scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu | cargo test --manifest-path src-tauri/Cargo.toml native_runtime: a missing DLL returns Err, never panics; an ABI mismatch is refused | Manual: install the NVIDIA pack; the next VC session uses CUDA; after deleting it, DirectML
Accept: The app starts with no packs. | The installer no longer needs the 144 MB CUDA provider, or the ADR explains why it still does.
Assets: NVIDIA CUDA 13 redist components (runtime download, sequential)
Risk: medium
Notes: 

### P1-D2 [P1, M] System probe and resource governor
Goal: sysinfo: DXGI adapters and VRAM, the NVIDIA driver version via nvcuda loaded dynamically, NVML loaded dynamically, and EP eligibility (CUDA needs Turing or newer and R580 or newer; TRT-RTX needs Ampere or newer; DirectML). resources: VRAM budget via QueryVideoMemoryInfo, claims by subsystem (vc, stt_final, llm, tts), a VC-running flag, and a deterministic recommendation API (e.g. Hy-MT2 7B only if RVC is off on 8 GB; Qwen3-TTS 0.6B while the VC runs). Includes a GpuRuntimeCard UI and a TS mirror.
Research: §3.1 VRAM-aware model choice; §2.1 0.6B when the VC runs; §8 GPU contention accounting
Owned: src-tauri/src/services/{sysinfo,resources}/**, src/core/services/{sysinfo,resources}/**, src/core/ui/inspector/components/GpuRuntimeCard.tsx
New: sysinfo/{dxgi.rs,cuda.rs,nvml.rs,eps.rs}, resources/{ledger.rs,policy.rs}, scripts/i18n-staging/P1-D2.json
Hotspots: 
Depends: P1-S0b
Verify: cargo test --manifest-path src-tauri/Cargo.toml sysinfo resources (policy table)
Accept: The 4090 is reported with its budget, and recommendations are deterministic.
Assets: 
Risk: low
Notes: 

### P1-E [P1, M] Process host plus llama-server pin and smoke test
Goal: process_host: a Job Object with KILL_ON_JOB_CLOSE and a memory cap, CREATE_NO_WINDOW, a free port on 127.0.0.1, a random API key, a /health poll, logs piped to tracing, a restart policy with backoff, and shutdown_all. Pin the llama.cpp vulkan, cpu and cuda13 release zips in the runtime catalog. Smoke test Hy-MT2 1.8B. Write ADR 0005.
Research: §3.2 bundled llama-server helper (host foundation); §2.2 sidecars bind 127.0.0.1
Owned: src-tauri/src/services/process_host/**, src-tauri/model-catalog/runtime-llama.json, src-tauri/plugin-commands/process-host.txt
New: process_host/{job.rs,spawn.rs,ports.rs,health.rs,logs.rs,tests.rs}, docs/adr/0005-process-host-llama.md
Hotspots: 
Depends: P1-S0b
Verify: cargo test --manifest-path src-tauri/Cargo.toml process_host: a child dies with the job, health timeout, port collision retry | Manual: kill Sigil in Task Manager and llama-server exits
Accept: No orphaned helpers, and bind is 127.0.0.1 only.
Assets: llama.cpp vulkan zip (~30 MB) | Hy-MT2 1.8B Q4 GGUF (~1.1 GB, optional, with consent)
Risk: medium
Notes: 

### P1-F [P1, M] Benchmark harness core
Goal: bench/: spec parsing, offline and paced drivers over the real VcStreamHarness, the e2e-loopback meter (±1 ms), a GPU load generator (ORT graph at a set duty cycle, CPU fallback), CPU burners, stall injection, CPU and VRAM metrics, JSON and markdown reports, and `--headless bench`. Includes a synthetic clip generator and a perf-gate script.
Research: §8 benchmark program (infrastructure); User adjustment: a harness that proves superiority (core)
Owned: src-tauri/src/services/bench/**, scripts/bench/**, docs/benchmarks/README.md
New: bench/{spec.rs,offline.rs,paced.rs,e2e_loopback.rs,load_sim.rs,metrics.rs,report.rs,cli.rs}, scripts/bench/{make-test-clips.py,corpus.example.json,run-perf-gate.ps1,specs/stub-smoke.json}
Hotspots: 
Depends: P1-S0b
Verify: cargo test --manifest-path src-tauri/Cargo.toml bench (percentiles; a known 1234-sample delay is recovered) | Sigil.exe --headless bench scripts/bench/specs/stub-smoke.json
Accept: Reports are produced, and the CLI exits non-zero on failure.
Assets: 
Risk: low
Notes: 

### P1-G1 [P1, M] sherpa-onnx C API: spike to production wrapper
Goal: Pin sherpa-onnx 1.13.8 with its Windows x64 shared CPU archive as a runtime pack. Vendor bindgen output for c-api.h. Write a safe libloading wrapper covering the online and offline recognizers, OfflineTts, spoken language ID and speaker embedding. Load onnxruntime.dll by full path before the C API DLL. Add an engine probe. Measure RTF and RSS while VC CUDA sessions run. Write ADR 0002, including whether Nemotron 3.5 and Qwen3-ASR are supported.
Research: §1.3 runtime spike (sherpa-onnx side)
Owned: src-tauri/src/services/speech_rt/sherpa/**, src-tauri/model-catalog/runtime-sherpa.json, scripts/runtime-packs/sherpa/**
New: speech_rt/sherpa/{bindings.rs,lib.rs,online.rs,offline.rs,tts.rs,lid.rs,probe.rs}, docs/adr/0002-sherpa-onnx.md
Hotspots: 
Depends: P1-S0b, P1-A
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa (struct layout matches the header; a missing DLL returns Err) | Sigil.exe --headless engine-probe sherpa <model> <wav>
Accept: The app launches without the pack. | STT and TTS smoke tests pass with it. | No ORT conflict with Sigil's CUDA sessions.
Assets: sherpa-onnx shared archive (~40 MB) | SenseVoice int8 (228 MB) and a Kokoro int8 test model, sequentially
Risk: medium
Notes: 

### P1-G2 [P1, M] transcribe.cpp: spike to DLL pack
Goal: Pin a commit and build sigil_transcribe.dll with a sigil_engine shim over transcribe.h, in cpu and cuda13 variants locally (vulkan in CI). Write the Rust wrapper. Test coexistence with whisper-rs in one process. Measure RTF and VRAM for whisper-tiny and Qwen3-ASR-1.7B Q8_0. Write ADR 0003b.
Research: §1.3 runtime spike (transcribe.cpp side)
Owned: native/engines/transcribe/**, src-tauri/src/services/speech_rt/transcribe/**, src-tauri/model-catalog/runtime-transcribe.json
New: native/engines/transcribe/{CMakeLists.txt,shim.cpp}, docs/adr/0003b-transcribe-cpp.md
Hotspots: 
Depends: P1-S0b, P1-D1 (ABI header)
Verify: pwsh scripts/native/build-runtime-packs.ps1 -Engine transcribe -Variant cpu,cuda13 | cargo test --manifest-path src-tauri/Cargo.toml speech_rt::transcribe -- --ignored (with SIGIL_TEST_MODELS_DIR)
Accept: Reproducible packs, and a go or no-go with fallback recorded.
Assets: transcribe.cpp source (one fetch at a pinned commit) | whisper-tiny GGUF (~40 MB) | optional Qwen3-ASR-1.7B Q8_0 (2.19 GB)
Risk: high: single maintainer, undocumented Windows CUDA build
Notes: 

### P1-G3 [P1, M] RVC template and importer spike
Goal: Export the v2-40k-f0 HiFiGAN-NSF split graphs (enc, flowdec with phase0 and anchored noise, knn) in fp32 and fp16-internal with fp32 I/O, plus layout.json and a tiny test config. Prototype in Rust: the allow-list unpickler, weight_norm fold, weights writer, and in-memory external initializers. Check parity against PyTorch on the MIT default voice. Write ADR 0004.
Research: §4.3#1 importer design decision; §4.2 fp16 main engine (feasibility)
Owned: scripts/vc-templates/**, src-tauri/resources/vc-templates/**, src-tauri/src/services/neural_audio/vc/import/**, src-tauri/test-fixtures/vc/tiny/**
New: scripts/vc-templates/{export_rvc.py,parity_test.py,layout_schema.json,requirements.txt,rvc_arch/**}, docs/adr/0004-rvc-templates.md
Hotspots: 
Depends: P1-S0b
Verify: pip install onnx; python scripts/vc-templates/parity_test.py (fp32 ≤1e-4, fp16 ≤5e-3; partial decode equals the full-decode tail) | cargo test --manifest-path src-tauri/Cargo.toml vc::import
Accept: The default voice imported by Rust matches PyTorch within tolerance.
Assets: onnx pip package (small) | PhoenixStormJr RVC-V2-default-voice (~55 MB) | ~8 MIT RVC/Applio source files (sequential)
Risk: high: decides the importer architecture
Notes: 

### P1-I0 [P1, M] P1 integrate
Goal: Merge i18n. Apply the spike outcomes to the P2 and P3 contracts. Handle the resource-bundling change. Docs: DEVELOPER_GUIDE runtime policy, native ABI and catalog authoring; OPERATOR_REFERENCE GPU packs and log location, correcting the claim that 'one installer ships CUDA'. Gates and exit matrix.
Research: §7 P1 runtime spike exit (integration)
Owned: (hotspots), docs/**
New: 
Hotspots: Collected P1 requests
Depends: P1-A, P1-B, P1-C, P1-D1, P1-D2, P1-E, P1-F, P1-G1, P1-G2, P1-G3
Verify: All gates; Sigil.exe --headless bench scripts/bench/specs/stub-smoke.json
Accept: P1 exit criteria met
Assets: 
Risk: medium
Notes: 

### P2-S0 [P2, M] P2 scaffold: speech and TTS plugins, splits, contracts
Goal: Take the checkpoint. Register the speech, tts-sink, openai-speech and local-tts plugins. Mechanically split stt/schema.ts into stt/schema/*.ts, inspector_stt.tsx into inspector/stt/*.tsx, tts/schema.ts into tts/schema/*.ts, and inspector_tts.tsx into inspector/tts/*.tsx. Add enum values: STT 'local', TTS 'localNeural' and 'openaiSpeech', with legacy values kept. Freeze the speech::engine_api contract (AsrEngine trait, events speech:partial/final/status/utterance/turn_end with pipelineId, epoch, segmentId, lang, tags). Freeze the tts_sink API and the ITTSService v2 PCM contract. Add optional TextEvent fields: segmentId, revision, lang, turnEnd.
Research: Phase scaffolding
Owned: (hotspots), src/core/services/stt/schema.ts, src/core/services/tts/schema.ts, src/core/services/tts/types.ts, src/core/ui/inspector/inspector_stt.tsx, src/core/ui/inspector/inspector_tts.tsx
New: src/core/services/stt/schema/*.ts, src/core/ui/inspector/stt/*.tsx, src/core/services/tts/schema/*.ts, src/core/ui/inspector/tts/*.tsx, src-tauri/src/services/speech/{mod.rs,engine_api.rs,events.rs}, src-tauri/src/services/{tts_sink,openai_speech}/mod.rs (stubs), src-tauri/plugin-commands/{speech,tts-sink,openai-speech,local-tts}.txt
Hotspots: Registrations and capabilities; TextEventSchema optional fields
Depends: P1-I0
Verify: All gates; manual inspector smoke (splits identical)
Accept: Implementers can work on disjoint files.
Assets: 
Risk: low
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-A [P2, L] Speech pipeline core
Goal: Keyed pipelines (mic, later loopback): hub Raw16k consumer, then the Silero segmenter, then an ordered engine worker with a latency budget, then events. Epoch discipline and the Rust mute gate. A Whisper engine adapter (whisper.rs keeps model management and delegates legacy start_recording). Whisper downloads move to the model store. The HTTP utterance engine: OpenAI-compatible, Groq and Moonshine HTTP in Rust, with a limiter and keys only in headers.
Research: §1.2 shared VAD gate for every engine; §5 mute gate upstream of engines (Rust)
Owned: src-tauri/src/services/speech/{pipeline.rs,worker.rs,utterance.rs,mute.rs}, src-tauri/src/services/speech/engines/{mod.rs,whisper.rs,http.rs}, src-tauri/src/services/speech_core/**, src-tauri/src/services/whisper.rs, src-tauri/src/services/moonshine_stt.rs, src-tauri/model-catalog/stt-whisper.json, src-tauri/plugin-commands/{speech,whisper,moonshine-stt}.txt
New: 
Hotspots: 
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech::pipeline (fake engine: epoch isolation, no stale segments, utterance framing) | Manual: Whisper works through the pipeline
Accept: No segment leaks across sessions. | Whisper downloads are adopted into the store.
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-B [P2, L] sherpa CPU STT engines
Goal: Engine adapters:
- Nemotron 3.5 streaming (partials, and finals for its 19 locales; always with an explicit language).
- Parakeet Ultra.
- Qwen3-ASR-0.6B with context hotwords.
- SenseVoice, emitting emotion and event tags as metadata.
- Moonshine v2.

Also transducer hotword biasing, spoken language ID, thread caps, and pinned catalog entries. If Nemotron is unsupported, flag the router to use transcribe CPU.
Research: §1.1 Nemotron 3.5 streaming; §1.1 Parakeet Ultra; §1.1 Qwen3-ASR-0.6B; §1.1 SenseVoiceSmall (including tag emission); §1.1 Moonshine v2; §5 hotwords part 2 (engine biasing)
Owned: src-tauri/src/services/speech_rt/sherpa/**, src-tauri/src/services/speech/engines/sherpa_{online,offline}.rs, src-tauri/model-catalog/stt-sherpa.json
New: 
Hotspots: 
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech -- --ignored (with SIGIL_TEST_MODELS_DIR) | Sigil.exe --headless engine-probe on en/ja/zh/ru clips, with RTF recorded
Accept: All five engines decode through the pipeline, and tags ride on finals.
Assets: Nemotron int8 (~650 MB), Parakeet Ultra, Qwen3-ASR-0.6B (0.94 GB), SenseVoice, Moonshine v2: sequential, only when tested
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-C [P2, L] Language router, two-pass orchestration, Smart Turn
Goal: router.rs: a catalog-driven table mapping language, tier, VRAM and VC state to a streaming model, a final model and a runtime. It encodes every never-route rule:
- Nemotron always gets a language;
- zh/yue go to SenseVoice; ja/ko/ru go to Qwen3; European languages get a Parakeet Ultra final;
- Cohere is never used for ja/zh/ko/ru;
- auto mode uses spoken language ID.

Plus:
- two_pass.rs: a final re-decode queue with a budget. The final replaces the partial under the same segmentId, with revision+1; a GPU failure promotes the CPU final.
- smart_turn.rs: Smart Turn v3.2 on Sigil's ORT at each pause, emitting speech:turn_end (for commits only).
- A shared routing fixture JSON used by both the Rust and TS tests.
Research: §1.1 two-pass captioning; §1.1 zero-setup automatic model pick by language; §1.2 Smart Turn v3.2 end-of-turn gate
Owned: src-tauri/src/services/speech/{router.rs,two_pass.rs,smart_turn.rs,lang.rs}, src-tauri/model-catalog/{stt-routing.json,smart-turn.json}, src-tauri/test-fixtures/routing.json
New: 
Hotspots: 
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech::router speech::two_pass speech::smart_turn
Accept: The rules are enforced in code, and turn_end never shortens caption finals.
Assets: Smart Turn v3.2 int8 ONNX (8 MB)
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-D [P2, L] transcribe.cpp GPU tier
Goal: Engine adapters:
- Qwen3-ASR-1.7B Q8_0 final, and BF16 for 16 GB+ cards.
- Cohere Q4_K for en/EU finals.
- Confucius4-R2T2, opt-in append-only streaming.
- Granite 5.0 TurboCTC.
- Voxtral Mini 4B Realtime Q4_K_M partials.

Pack variant selection and VRAM claims go through the resource governor, and the worker-process isolation mode is optional.
Research: §1.1 Qwen3-ASR-1.7B Q8_0; §1.1 Cohere Transcribe Q4_K; §1.1 Confucius4-R2T2; §1.1 Granite Speech 5.0 TurboCTC; §1.1 Voxtral Mini 4B Realtime; §1.1 Qwen3-ASR-1.7B BF16 on 16 GB+ cards
Owned: src-tauri/src/services/speech_rt/transcribe/**, src-tauri/src/services/speech/engines/transcribe.rs, native/engines/transcribe/**, src-tauri/model-catalog/stt-gguf.json
New: src-tauri/src/services/speech/worker_proc.rs
Hotspots: Integrator: cli arm stt-worker
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml speech::engines::transcribe -- --ignored | Dev: two-pass ja with VRAM logged
Accept: The GPU tier is offered only when the governor allows it.
Assets: 
Risk: high
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-E [P2, M] STT TS: local backend and zero setup
Goal: A TS driver for speech events that maps segmentId, revision and lang into TextEvent. Local panel: language (auto or explicit), tier, two-pass, Smart Turn commit, hotwords, route preview and ModelRequirement. A 'Best for my PC' option. Consent before downloading. Auto-restart on stream errors. 'local' counts as heavy for connectingFallbackMs. A TS router mirror tested against the shared fixture.
Research: §5 automatic STT model by language (UI); §1.1 zero-setup row (UI and consent); §5 hotwords (UI)
Owned: src/core/services/stt/{index.ts,types.ts,router.ts}, src/core/services/stt/services/{local.ts,whisper.ts}, src/core/services/stt/schema/{index,local,whisper}.ts, src/core/ui/inspector/stt/{index,Local,Whisper}.tsx
New: src/core/services/stt/__tests__/{router,localDriver}.test.ts, scripts/i18n-staging/P2-E.json
Hotspots: Integrator: new-install default is 'local' plus a one-time prompt for native users
Depends: P2-S0
Verify: pnpm typecheck; pnpm test:unit -- src/core/services/stt | Manual: fresh profile, consent, captions in en and ja
Accept: Zero setup works on a clean profile.
Assets: 
Risk: low
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-F [P2, M] Legacy STT backends on the new plumbing
Goal: rustCapturePolling, openai_audio (Groq) and Moonshine use speech:utterance from the Rust HTTP engine. Vosk is fed by the hub, keeping audio:chunk compatibility. Azure, Deepgram, native and external-mic report status through the status channel. Moonshine HTTP is marked legacy, and dead vosk stubs are removed.
Research: Legacy engine migration (polling path gate)
Owned: src/core/services/stt/{rustCapturePolling.ts,audioConvert.ts,loadVoskModel.ts,encodeWav.ts,rateLimiter.ts}, src/core/services/stt/services/{vosk,moonshine,openaiAudio,azure,deepgram,native,externalMic}.ts, src/core/services/stt/schema/{vosk,moonshine,openaiAudio,azure,deepgram}.ts, src/core/ui/inspector/stt/{Vosk,Moonshine,OpenAIAudio,Azure,Deepgram,ExternalMic}.tsx
New: scripts/i18n-staging/P2-F.json
Hotspots: 
Depends: P2-S0
Verify: pnpm typecheck; pnpm test:unit | Manual: Vosk and Groq
Accept: No backend regresses.
Assets: 
Risk: low
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-G [P2, L] TTS sentence pipeline and unified Rust sink
Goal: pipeline.ts: a splitter covering JP punctuation, merging and capping fragments, synthesizing N ahead in order, with cancel and skip. tts_sink: persistent output per device, a queue with cancel and flush, binary IPC, and routes device|voiceChanger|both through the VC aux API with the P0 back-pressure. A top-level route setting replaces the per-backend routeThroughVoiceChanger flags, which are migrated. Azure, VOICEVOX, Uberduck and SAPI (self-playing for now) use the PCM contract or an adapter. Service_Sound is folded in. Stop and skip in StatusFooter. Device lists come from cpal names, migrating old SAPI names.
Research: §2.2 sentence chunking and pipelining; §5 route every TTS backend through the VC, part 1 (sink plus PCM backends)
Owned: src/core/services/tts/{index.ts,pipeline.ts,sentence.ts}, src/core/services/tts/services/{azure,voicevox,uberduck,native,windows}.ts, src/core/services/tts/schema/{index,common,route,azure,voicevox,uberduck,native,windows}.ts, src/core/ui/inspector/tts/{index,Routing,Azure,VoiceVox,UberDuck,Native,Windows}.tsx, src/core/services/sound/**, src/core/ui/sigil/StatusFooter.tsx, src-tauri/src/services/tts_sink/**, src-tauri/src/services/uberduck_tts/**, src-tauri/src/services/voice_changer/aux.rs, src-tauri/plugin-commands/{tts-sink,uberduck-tts}.txt
New: src/core/services/tts/__tests__/{sentence,pipeline}.test.ts, scripts/i18n-staging/P2-G.json
Hotspots: 
Depends: P2-S0
Verify: pnpm test:unit -- src/core/services/tts; cargo test --manifest-path src-tauri/Cargo.toml tts_sink | Manual: long VOICEVOX text starts after the first sentence; routing into the VC works
Accept: First audio after sentence one. | The device field is honoured. | No JSON byte arrays.
Assets: 
Risk: medium
Notes: 

### P2-H [P2, L] In-process local TTS: Kokoro, Kokoro v1.1-zh, Piper, Kitten
Goal: local_tts via sherpa OfflineTts:
- Kokoro v1.0 (en/ja/zh/es/fr/hi/it/pt-BR) with 2–4 threads.
- Kokoro v1.1-zh.
- Piper voices for ru, ar and the long tail.
- Kitten, with the vendored kitten as fallback.

C espeak-ng G2P is used inside sherpa, serialized. Commands are async with spawn_blocking. Models come from the store, with speech_package_v1 compatibility. Output follows the PCM contract. neuralOnnx maps onto localNeural.
Research: §2.1 Kokoro 82M in-process; §2.1 espeak-ng in-process G2P (serialized); §2.1 Piper voices; §2.1 Kokoro v1.1-zh
Owned: src-tauri/src/services/local_tts/**, src-tauri/src/services/neural_onnx_tts.rs, src/core/services/tts/services/{neuralOnnx,localNeural}.ts, src/core/services/tts/schema/{neuralOnnx,localNeural}.ts, src/core/ui/inspector/tts/{NeuralOnnx,LocalNeural}.tsx, src-tauri/model-catalog/tts-local.json, src-tauri/plugin-commands/{local-tts,neural-onnx-tts}.txt
New: local_tts/{kokoro.rs,piper.rs,voices.rs,tests.rs}, scripts/i18n-staging/P2-H.json
Hotspots: 
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml local_tts -- --ignored | Manual: Kokoro en/ja/zh, Piper ru, Kitten; the UI never freezes
Accept: Kokoro RTF ≤ 0.3 on 4 threads (dev CPU). | No inference on the main thread.
Assets: kokoro-multi-lang int8 (~130 MB) | one Piper voice (~60 MB)
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P2-I [P2, M] Generic OpenAI-compatible streaming TTS client
Goal: openai_speech.rs: POST {model, input, voice, response_format pcm|wav, speed, stream, extra}, optional Bearer, bytes_stream into the sink, per-request timeout. Presets: Kokoro-FastAPI :8880, Chatterbox :5555, Voxtral-4B-TTS on vLLM, IndexTTS-2.5, VoxCPM2, and qwentts tts-server. Legacy kokoro, chatterbox and melo enums map onto it, keeping their settings. neural_sidecar streams. The reference sidecar gets a streaming mode and binds 127.0.0.1.
Research: §2.2 one generic OpenAI-compatible /v1/audio/speech client; §2.1 IndexTTS-2.5 and Voxtral-4B-TTS server presets; §2.1 VoxCPM2 preset
Owned: src-tauri/src/services/{openai_speech.rs,kokoro_tts.rs,chatterbox_tts.rs,melo_tts.rs,neural_sidecar_tts.rs,fish_speech.rs}, src/core/services/tts/services/{openaiSpeech,kokoro,chatterbox,melo,fishspeech,neuralSidecar}.ts, src/core/services/tts/schema/{openaiSpeech,kokoro,chatterbox,melo,fishSpeech,neuralSidecar}.ts, src/core/ui/inspector/tts/{OpenAISpeech,Kokoro,Chatterbox,Melo,FishSpeech,NeuralSidecar}.tsx, src-tauri/plugin-commands/{openai-speech,kokoro-tts,chatterbox-tts,melo-tts,neural-sidecar-tts,fish-speech}.txt, scripts/neural-sidecar/**
New: scripts/i18n-staging/P2-I.json
Hotspots: 
Depends: P2-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml openai_speech (mock streaming server)
Accept: First audio arrives before the full response, and legacy settings keep working.
Assets: 
Risk: low
Notes: 

### P2-I0 [P2, M] P2 integrate
Goal: Merge i18n. Add the stt-worker CLI arm. Flip the new-install STT default to 'local' and TTS to Kokoro local, with prompts for existing users. Docs: replace the OPERATOR_REFERENCE advice that prefers sidecars; update the DEVELOPER_GUIDE STT and TTS wiring tables and the manual audio chapter. Run the exit matrix and record bench-results.
Research: §7 P1 roadmap exit (integration)
Owned: (hotspots), docs/**
New: 
Hotspots: Collected P2 requests and default flips
Depends: P2-A, P2-B, P2-C, P2-D, P2-E, P2-F, P2-G, P2-H, P2-I
Verify: All gates plus the P2 exit matrix
Accept: P2 exit criteria met
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-S0 [P3, M] P3 scaffold: voice-engine contracts
Goal: Take the checkpoint. Add every new VC parameter end to end with no-op serde defaults: slotId, block, context, precision, f0Method (+swiftf0, crepe, pesto, fusion), rmsMix, formant, noiseScale, silenceDb, partialDecode, f0Cache, governor, denoiser, hotSlots, vramBudget, ep (+tensorrt_rtx, dml_hp, which also updates normalize_execution_provider and its test). Add manifest v2 fields. Stub modules: vc/engine/*, neural_audio/{gpu,denoise}/*, vc/import/* commands, pitch files, bench v2. Split inspector_voice_changer.tsx into inspector/voice_changer/*. Add the TS IPC contract. Bundle vc-templates as resources.
Research: Phase scaffolding
Owned: (hotspots), src-tauri/src/services/voice_changer/params.rs, src-tauri/src/services/neural_audio/{mod.rs,vc/mod.rs,vc/manifest.rs}, src/core/services/voice_changer/{schema.ts,ipc.ts}, src/core/ui/inspector/inspector_voice_changer.tsx
New: src/core/ui/inspector/voice_changer/*.tsx, src-tauri/src/services/neural_audio/vc/engine/*.rs (stubs), src-tauri/src/services/neural_audio/{gpu,denoise}/mod.rs (stubs)
Hotspots: Registrations; tauri.conf.json resources for vc-templates
Depends: P2-I0
Verify: All gates; VC bench within 5% of the P0 baseline
Accept: No behaviour change.
Assets: 
Risk: low
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-A [P3, L] Template matrix and base-asset export pipeline
Goal: Full split-graph templates for v1/v2 × 32/40/48k × f0/nof0 × HiFiGAN-NSF/RefineGAN/MRF, in fp32 and fp16-internal. Exports:
- Embedders: ContentVec v1/v2 (fp32, fp16, int8), SPIN-v2, zh/ja/ko HuBERT (400/320 front end verified).
- Pitch: SwiftF0, FCPE, RMVPE, CREPE-tiny, PESTO.
- Denoisers: GTCRN, DeepFilterNet3, DPDFNet.
- CAMPPlus and DNSMOS for the benchmark, and gpu_load.onnx.

Parity for everything. Pinned vc-bases catalog including the MIT default voice. Publish script. Sigil-exported entries stay unpublished until the user uploads them.
Research: §4.3#5 SPIN-v2 and zh/ja/ko HuBERT embedders; §4.3#5 RefineGAN generator template; §4.2 zero-setup base models (ContentVec, RMVPE, FCPE, SwiftF0, MIT default voice); §4.2 High GPU with 48 kHz models (template coverage)
Owned: scripts/vc-templates/**, src-tauri/resources/vc-templates/**, src-tauri/test-fixtures/vc/tiny/**, src-tauri/model-catalog/{vc-bases.json,vc-embedders.json,vc-denoise.json,bench-models.json}, scripts/assets/**
New: scripts/vc-templates/{export_matrix.py,export_embedders.py,export_pitch.py,export_denoise.py,export_bench.py}, scripts/assets/publish-assets.ps1
Hotspots: 
Depends: P3-S0
Verify: python scripts/vc-templates/parity_test.py --all (fp32 ≤1e-4, fp16 ≤5e-3)
Accept: Every matrix cell passes parity or is documented as unsupported. | Bundled templates total ≤ 30 MB.
Assets: embedder checkpoints (5 × ~360 MB, sequential across sessions) | pitch and denoiser checkpoints (small)
Risk: medium: RefineGAN and MRF export correctness
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-B [P3, L] Production .pth/.zip importer and slot store
Goal: Zip guard. The --headless vc-import-worker running in a Job Object. The allow-list unpickler. Metadata detection. weight_norm fold. Template selection and shape validation. The weights.f32.bin writer. Manifest sigil-rvc-1 plus speakers.bin and index adoption. Smoke inference. sha256 cache. Specific rejection codes. Commands for import, slot CRUD and set-active. Auto-import of the default voice. A 200-file fuzz corpus.
Research: §4.3#1 import the checkpoints users actually have; §4.2 drag-and-drop .zip/.pth import (backend)
Owned: src-tauri/src/services/neural_audio/vc/import/**, src-tauri/src/services/neural_audio/vc/bases.rs, src-tauri/src/services/voice_changer/cmd_import.rs, src-tauri/test-fixtures/vc/fuzz/**
New: vc/import/{pickle.rs,pth.rs,zipguard.rs,layout.rs,fold.rs,writer.rs,slot.rs,worker.rs}
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc::import: tiny fixtures; fuzz corpus with no panic or hang; malicious GLOBAL rejected; traversal and zip bomb rejected | Dev: corpus report
Accept: T8 compatibility met on the user-supplied corpus.
Assets: 
Risk: high
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-C1 [P3, L] Engine algorithmic core: partial decode, anchored noise, phase0, F0 cache, fixed shapes
Goal: SigilRvc family over the split templates. Harness fixed-block mode with a pre-accumulator. The enc and flowdec partial decode. Counter-based time-anchored noise. phase0 continuity. Incremental F0 with cache. Shape plans per preset. Speaker vector g with blending. The legacy full-decode path accepts fp16 I/O and falls back when there is no skip support. Per-slot embedder resolution and int8 encoder selection.
Research: §4.2 fp16 main engine on CUDA/DirectML; Core-quality items (Gaussian rnd, whole-context recompute, fixed shapes for §4.3#6)
Owned: src-tauri/src/services/neural_audio/vc/{rvc.rs,stream.rs,traits.rs,int8_gate.rs}, src-tauri/src/services/neural_audio/vc/engine/{mod,content,generator,noise,f0_cache,plan,presets}.rs
New: 
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc: existing harness tests; partial decode equals the full-decode tail; noise is identical across overlaps; no phase jump on a synthetic f0 ramp; F0 cache matches within 5 cents
Accept: Extra-context cost scaling meets T2, and T5 is met in P3-F runs.
Assets: 
Risk: high
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-C2 [P3, L] GPU layer: IOBinding, CUDA graphs, priority stream/queue, TensorRT-RTX, arena
Goal: neural_audio/gpu:
- Session plans with device-resident IOBinding and CUDA graph capture per shape, recaptured when the preset changes.
- A highest-priority CUDA stream via cudart loaded with libloading.
- A DirectML high-priority D3D12 queue through OrtDmlApi DML1.
- TensorRT-RTX (nvrtx, or a plugin EP) with an engine cache and build-progress events; classic TRT as an option.
- Env allocators and VRAM cap, error_on_failure, VRAM ledger reporting to the governor.
- A TRT-RTX runtime pack.
Research: §4.3#6 GPU scheduling (CUDA graphs, IOBinding, priority CUDA stream, high-priority DirectML queue, no admin); §4.2 optional TensorRT on the High GPU tier
Owned: src-tauri/src/services/neural_audio/gpu/**, src-tauri/src/services/neural_audio/ort_session.rs, src-tauri/model-catalog/runtime-trt-rtx.json
New: gpu/{plan.rs,io_binding.rs,cuda_graph.rs,cuda_stream.rs,dml_queue.rs,trt.rs,arena.rs}
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml neural_audio::gpu (plan reuse, zero reallocations) | Dev bench: graphs on and off, TRT on and off, VRAM
Accept: Zero host allocations per block. | T3 met. | A TRT cache hit loads in ≤ 3 s.
Assets: TensorRT-RTX runtime (<200 MB, one download)
Risk: high
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-D [P3, L] Pitch bank, retrieval and post stage
Goal: Pitch: SwiftF0 resampled to the hop-160 grid, a real CREPE-tiny, streaming PESTO, F0 fusion, an improved octave gate, singing range, and auto-key.

Retrieval: an exact GPU kNN graph, plus parallel CPU IVF search checked against pthrs.

Post: rms_mix, cepstral-envelope formant shift (±12 st), a dB silence gate with hangover, and a limiter. It replaces the tilt on the RVC path.
Research: §4.3#4 SwiftF0 (CPU default after a cents A/B); §4.3#9 check FAISS against pthrs and parallelize; §4.2 RVC on the CPU EP with SwiftF0 (CPU tier)
Owned: src-tauri/src/services/neural_audio/vc/{fcpe.rs,rmvpe.rs,world.rs,pitch_onnx.rs,f0_post.rs,faiss.rs}, src-tauri/src/services/neural_audio/vc/engine/{retrieval.rs,post.rs}, src-tauri/src/services/neural_audio/chunk_processor.rs
New: vc/{swiftf0.rs,crepe.rs,pesto.rs,f0_fusion.rs,formant.rs,autokey.rs}
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc: sweeps and octave traps; GPU kNN matches brute force exactly; parallel IVF matches serial; the formant shift preserves F0 within 5 cents | Bench pitch A/B spec
Accept: Fusion lowers gross pitch error against the best single extractor. | SwiftF0 becomes the default only if it passes the A/B.
Assets: uses P3-A exports
Risk: medium
Notes: 

### P3-E [P3, L] Runtime loop: hub consumer, denoiser, governor, hot slots, device-follow, persistence
Goal: The voice changer consumes the hub's conditioned DeviceRate copy (STT stays raw). Denoiser chain: GTCRN at 16 kHz inside the harness; DFN3 or DPDFNet at 48 kHz before the aux mix. Adaptive governor. Hot slots with crossfade. Device-follow restarts. Per-tick timeline metrics. The new parameters are persisted. The factory dispatches on manifest.format.
Research: §4.3#7 VC half of the owned capture (conditioned copy); §4.3#8 optional input denoiser on the VC path only
Owned: src-tauri/src/services/voice_changer/{mod,state,playback,aux,worker,commands,persist,metrics}.rs, src-tauri/src/services/neural_audio/denoise/**, src-tauri/src/services/neural_audio/vc/engine/governor.rs, src-tauri/src/services/neural_audio/{onnx_voice.rs,sidecar_bridge.rs}, src-tauri/plugin-commands/voice-changer.txt
New: voice_changer/{devices.rs,factory.rs}, denoise/{gtcrn.rs,dfn.rs}
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml voice_changer: stall recovers within budget; a test that STT PCM equals device input; accounting invariants hold
Accept: T7 met. | With the denoiser off, e2e latency is unchanged (±2 ms).
Assets: 
Risk: high
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-F [P3, L] Benchmark v2, quality metrics, competitor compare, autotuner
Goal: Quality metrics: MCD, F0 RMSE/GPE/VUV, CAMPPlus cosine, DNSMOS, CER via Sigil STT, click detector. Offline ceiling. Contention and live-record modes. A compare mode for competitor WAVs, with drivers for RVC WebUI and Applio. The autotuner search. A counting-allocator test. A listening-test kit. run-matrix and perf-gate scripts. VC_BENCHMARKS scorecard.
Research: §8 RVC engine p99 latency under load; §8 SwiftF0 vs RMVPE cents; User adjustment: prove the main engine beats the others
Owned: src-tauri/src/services/bench/**, src-tauri/src/services/voice_changer/cmd_bench.rs, scripts/bench/**, scripts/vc-bench/**, tools/listening-test/**, docs/benchmarks/**
New: bench/{quality.rs,contention.rs,live_record.rs,compare.rs,autotune.rs}
Hotspots: 
Depends: P3-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml bench (metrics on analytic signals, allocation-free hot path) | pwsh scripts/bench/run-perf-gate.ps1 -Profile p3
Accept: The scorecard covers T1–T9. | The autotuner returns a valid preset on CPU and GPU.
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-G [P3, L] Voice-changer frontend
Goal: Slot manager with drag-and-drop import, rename, delete and duplicate, voice hotkeys, and model merge. First-run one-click base set plus default voice. Presets: Streamer, Low latency, Quality, Studio-TRT. All new parameters. Embedder and vocoder shown read-only. EP status with its reason. Live e2e/p99/dropped/VRAM readouts. Benchmark and autotune panel. TRT build progress.
Research: §4.2 zero-setup UI and drag-and-drop
Owned: src/core/services/voice_changer/**, src/core/ui/inspector/voice_changer/**
New: scripts/i18n-staging/P3-G.json
Hotspots: 
Depends: P3-S0
Verify: pnpm typecheck; pnpm test:unit (parameter normalization parity with Rust)
Accept: Every engine feature is reachable, with no silent remapping.
Assets: 
Risk: low
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P3-H [P3, M] Zero-shot and singing export spike
Goal: Export Seed-VC realtime (XLSR/whisper content encoder, DiT step, vocoder, CAMPPlus), DDSP-SVC 6.3 (encoder, velocity) and MeanVC2 to ONNX. Write reference Python sampler loops. Check offline parity (SNR ≥ 35 dB). Measure step cost on the 4090 GPU and CPU. Write ADR 0006 with a go or no-go per family.
Research: User adjustment: bring zero-shot and singing into the engine (feasibility)
Owned: scripts/vc-families/**
New: scripts/vc-families/{seed_vc,ddsp_svc,meanvc}/{export.py,parity.py}, docs/adr/0006-vc-zero-shot-singing.md
Hotspots: 
Depends: P3-S0
Verify: python parity scripts per family
Accept: A measured go or no-go for each family.
Assets: Seed-VC realtime checkpoints (~1.5 GB), DDSP-SVC and MeanVC2 weights: sequential, with consent
Risk: high
Notes: 

### P3-I0 [P3, M] P3 integrate
Goal: Merge i18n. Docs: rewrite the OPERATOR_REFERENCE voice-changer section (importer, presets, EP packs, TRT, governor, budget) and update VC_BENCHMARKS. Run the full benchmark and the exit matrix.
Research: §7 P3 roadmap exit (integration)
Owned: (hotspots), docs/**
New: 
Hotspots: Collected P3 requests
Depends: P3-A, P3-B, P3-C1, P3-C2, P3-D, P3-E, P3-F, P3-G, P3-H
Verify: All gates plus pwsh scripts/bench/run-perf-gate.ps1 -Profile p3
Accept: P3 exit criteria met
Assets: 
Risk: medium
Notes: 

### P4-S0 [P4, S] P4 scaffold: translation contracts
Goal: Take the checkpoint. Register the llm-server and mt-local plugins. Split the translation schema and inspector per backend. Add Translation_Backends llm, deepl, bergamot, nllb and auto. Add the LLM client types. Add a Cargo tokenizers dependency (default-features off). Add a process_host recipe type for the Bergamot helper.
Research: Phase scaffolding
Owned: (hotspots), src/core/services/translation/schema.ts, src/core/ui/inspector/inspector_translation.tsx
New: src/core/services/translation/schema/*.ts, src/core/ui/inspector/translation/*.tsx, src/core/services/llm/types.ts, src-tauri/src/services/{llm_server,mt_local}/mod.rs (stubs)
Hotspots: Registrations
Depends: P3-I0
Verify: All gates
Accept: Contracts compile.
Assets: 
Risk: low
Notes: 

### P4-A [P4, M] Shared LLM client and transform upgrade
Goal: src/core/services/llm client:
- providers: bundled llama-server, Gemini OpenAI endpoint (key in the x-goog-api-key header), OpenRouter, custom;
- streaming, reasoning_effort, max_tokens, AbortSignal;
- a raw /completion path for Seed-X;
- think-tag stripping;
- webview or Rust-proxy transport.

Transform moves onto it.
Research: §3.2 generic OpenAI-compatible translation backend (client); §3.3 Gemini Flash-Lite free tier
Owned: src/core/services/llm/**, src/core/services/transform/**, src/core/ui/inspector/inspector_transform.tsx
New: src/core/services/llm/{client.ts,providers.ts,stream.ts,sanitize.ts}, src/core/services/llm/__tests__/*.test.ts, scripts/i18n-staging/P4-A.json
Hotspots: 
Depends: P4-S0
Verify: pnpm test:unit -- src/core/services/llm src/core/services/transform
Accept: Transform streams and aborts correctly.
Assets: 
Risk: low
Notes: 

### P4-B [P4, M] llama-server helper, GGUF catalog, Rust LLM proxy
Goal: llm_server on process_host: start, stop, status and logs. Variant choice via the governor. -ngl, ctx and flash-attn set from the VRAM plan, refusing 7B on 8 GB while the VC runs. Catalog: Hy-MT2 1.8B (Q4 and 1.25-bit), Hy-MT2 7B, Hy-MT2 30B-A3B, MiLMMT-46-4B, MiLMMT-46-12B (self-made GGUF script), TranslateGemma 4B/12B, Seed-X-PPO-7B, NiuTrans LMT-60-4B (A/B), plus the specialists CAT-Translate, Shisa V2.1, VNTL, SakuraLLM/GalTransl and MADLAD-400-3B as opt-in presets. llm_proxy_chat streams through a Channel for LAN hosts. Includes a TS host module and panel.
Research: §3.2 bundled llama-server helper; §3.1 Hy-MT2 1.8B; §3.1 Hy-MT2 7B; §3.1 MiLMMT-46-4B; §3.1 TranslateGemma 4B/12B; §3.1 NiuTrans LMT-60-4B (A/B); §3.1 MiLMMT-46-12B; §3.1 Hy-MT2 30B-A3B and Seed-X-PPO-7B; §3.1 BYO specialists (CAT-Translate, Shisa, VNTL, Sakura/GalTransl, MADLAD-400-3B); §3.1 zero setup (Hy-MT2 1.8B default GGUF)
Owned: src-tauri/src/services/llm_server/**, src-tauri/model-catalog/llm-gguf.json, src-tauri/plugin-commands/llm-server.txt, src/core/services/llm_server/**, src/core/ui/inspector/translation/LlmServer.tsx, scripts/models/make-gguf.md
New: scripts/i18n-staging/P4-B.json
Hotspots: 
Depends: P4-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml llm_server | Manual: Hy-MT2 1.8B end to end; 7B refused on an 8 GB budget while the VC runs
Accept: Helpers are managed with no orphans, and choices obey the governor.
Assets: GGUF models on demand (≥1 GB each, with consent)
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P4-C [P4, L] Translation pipeline host
Goal: Engine contract returns a Promise and throws (Azure, LibreTranslate and onDevice ported). Utterance tracker. LocalAgreement-2: word LCP, or Intl.Segmenter for CJK; commits go out as interims, FINAL only on the STT final; handles interim-after-final and revisions. Throttling and latest-wins. Prompt builder: Hy-MT2 context and terminology blocks, TranslateGemma, MiLMMT, Seed-X raw prompt, generic; rolling final-pair context; glossary filtered to the line; source fenced as data. Router: per-pair chain, circuit breaker, cloud-to-local fallback, interim engine. languages.ts canonicalizes BCP-47 to Azure, ISO-639-1 and FLORES. turnEnd triggers commits. A createTranslationPipeline factory for multiple instances.
Research: §3.2 per-model prompt builder with rolling context and glossary; §3.2 LocalAgreement-2; §3.2 language-pair router with fallback chain; §1.2 Smart Turn consumption (translation commit)
Owned: src/core/services/translation/{index.ts,types.ts,sequencer.ts}, src/core/services/translation/services/{azure,libretranslate,onDevice,llm}.ts, src/core/services/translation/schema/**
New: src/core/services/translation/{pipeline.ts,tracker.ts,localAgreement.ts,prompt.ts,router.ts,languages.ts,glossary.ts}, src/core/services/translation/__tests__/*.test.ts
Hotspots: 
Depends: P4-S0
Verify: pnpm test:unit -- src/core/services/translation (property test: the committed prefix is monotonic; CJK LCP; fallback on throw; prompt fencing)
Accept: No flicker and no dropped finals. | The chain falls through within the timeout.
Assets: 
Risk: medium
Notes: 

### P4-D [P4, M] NLLB-200 on Sigil's ORT, DeepL Free, MT benchmark
Goal: NLLB: encoder plus decoder_merged with KV cache, greedy or beam decoding, the tokenizers crate (parity-checked against Python sentencepiece), FLORES codes, CPU background profile with 2 threads. deepl_translate in Rust: DeepL-Auth-Key header, api-free host, per-request timeout. The --headless mt-bench command with chrF.
Research: §3.1 NLLB-200 distilled 600M/1.3B; §3.3 DeepL API Free
Owned: src-tauri/src/services/mt_local/{nllb.rs,tokenizer.rs,flores.rs}, src-tauri/src/services/translate.rs, src-tauri/src/services/bench/mt.rs, src-tauri/model-catalog/mt-nllb.json, src-tauri/plugin-commands/{translate,mt-local}.txt, src/core/services/translation/services/{nllb,deepl}.ts
New: scripts/mt-bench/**
Hotspots: Integrator: cli arm mt-bench
Depends: P4-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml mt_local::nllb translate::deepl
Accept: NLLB works offline and DeepL works with the user's key.
Assets: NLLB-600M ONNX int8 (~0.9 GB)
Risk: medium
Notes: 

### P4-E [P4, L] Mozilla Bergamot CPU translation helper
Goal: Build sigil-bergamot.exe from bergamot-translator with USE_WASM_COMPATIBLE_SOURCE=ON, using JSON lines over stdio, as a runtime pack under process_host. Per-direction models from Mozilla's registry, with English pivot. A Rust client in mt_local/bergamot.rs. ADR 0008.
Research: §3.1 Mozilla Bergamot (CPU fallback and cheap partial re-translator); §3.1 zero setup (Bergamot on demand)
Owned: tools/bergamot-helper/**, scripts/runtime-packs/bergamot/**, src-tauri/src/services/mt_local/bergamot.rs, src-tauri/model-catalog/{mt-bergamot.json,runtime-bergamot.json}, src/core/services/translation/services/bergamot.ts
New: docs/adr/0008-bergamot.md
Hotspots: 
Depends: P4-S0
Verify: pwsh scripts/runtime-packs/bergamot/build.ps1; manual en↔ja < 150 ms per line
Accept: The helper works, or the ADR records a no-go and names NLLB and Hy-MT2 CPU as the fallback.
Assets: bergamot-translator source (one fetch) | one direction's model (~40 MB)
Risk: high: marian/intgemm on MSVC
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P4-F [P4, M] Translation inspector UI
Goal: Panels for LLM (presets, reasoning, templates), NLLB, Bergamot, DeepL and LibreTranslate (including an LTEngine preset). A router chain editor and interim engine picker. Glossary and context editors outside Deactivatable, so they apply live.
Research: §3.1 LTEngine via the LibreTranslate backend (preset)
Owned: src/core/ui/inspector/translation/** (except LlmServer.tsx)
New: scripts/i18n-staging/P4-F.json
Hotspots: 
Depends: P4-S0
Verify: pnpm typecheck; manual
Accept: Every setting is reachable, and glossary edits apply without a restart.
Assets: 
Risk: low
Notes: 

### P4-G [P4, S] Caption rendering for revising text
Goal: The text element animates from the longest common prefix instead of resetting, pairs updates by id and revision, and handles tentative tails. It stays remote-safe.
Research: §3.2 LocalAgreement display requirement
Owned: src/client/elements/text/index.tsx
New: 
Hotspots: 
Depends: P4-S0
Verify: pnpm typecheck; $env:PW_PREVIEW_ONLY=1; pnpm test:e2e
Accept: No full-sentence reset when only the tail changes.
Assets: 
Risk: low
Notes: 

### P4-I0 [P4, S] P4 integrate
Goal: Merge i18n. Docs: manual translation chapter and the translation-sidecar README. Exit matrix, including the Job Object kill test and the MT bench.
Research: §7 P2 roadmap exit (integration)
Owned: (hotspots), docs/**, scripts/translation-sidecar/README.md
New: 
Hotspots: Collected P4 requests
Depends: P4-A, P4-B, P4-C, P4-D, P4-E, P4-F, P4-G
Verify: All gates plus the P4 exit matrix
Accept: P4 exit criteria met
Assets: 
Risk: low
Notes: 

### P5-S0 [P5, S] P5 scaffold: family registry and voice library
Goal: Take the checkpoint. Add a factory arm per manifest.format (ddsp_svc, sigil_zs, meanvc, sovits, sidecar_v2). Extend ConversionCtx with a reference embedding and singing flags. Stub vc/families/*.rs. Register the voice-library plugin. Add the TS contracts and family UI stubs.
Research: Phase scaffolding
Owned: (hotspots), src-tauri/src/services/neural_audio/vc/{mod.rs,traits.rs,manifest.rs}, src-tauri/src/services/voice_changer/factory.rs
New: vc/families/*.rs (stubs), src-tauri/plugin-commands/voice-library.txt
Hotspots: Registrations
Depends: P4-I0
Verify: All gates
Accept: Families compile as stubs.
Assets: 
Risk: low
Notes: 

### P5-A [P5, M] DDSP-SVC 6.3 singing engine
Goal: A Rust Euler loop over the encoder and velocity ONNX, with a fixed step count and graph capture. 44.1 kHz through the harness. F0 over E2–D#7 with no coarse clamp. A catalog entry and a ceiling benchmark.
Research: §4.2 DDSP-SVC 6.3 for singers
Owned: src-tauri/src/services/neural_audio/vc/families/ddsp_svc.rs, src-tauri/model-catalog/vc-ddsp.json
New: 
Hotspots: 
Depends: P5-S0, P3-H (go)
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc::families::ddsp_svc; bench singing spec
Accept: The T10 singing target is met.
Assets: 
Risk: medium
Notes: 

### P5-B [P5, L] Sigil-ZS: in-engine Seed-VC realtime zero-shot
Goal: A Rust CFM/Euler sampler with CFG over the exported graphs. CAMPPlus reference style cached per library voice. fp16 plans with CUDA graphs. Streaming configuration chosen by the autotuner. The latency budget applies.
Research: §4.2 Seed-VC zero-shot (in-engine)
Owned: src-tauri/src/services/neural_audio/vc/families/sigil_zs/**, src-tauri/model-catalog/vc-zeroshot.json
New: 
Hotspots: 
Depends: P5-S0, P3-H (go), P5-F (reference API)
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc::families::sigil_zs (sampler on an analytic flow; exact length accounting); bench zero-shot spec
Accept: The T10 zero-shot target is met, or the family ships through P5-E as documented.
Assets: 
Risk: high
Notes: 

### P5-C [P5, M] MeanVC2 CPU zero-shot (experimental)
Goal: A 1-step 16 kHz CPU engine, A/B-tested against RVC Streamer.
Research: §4.2 MeanVC2
Owned: src-tauri/src/services/neural_audio/vc/families/meanvc.rs
New: 
Hotspots: 
Depends: P5-S0, P3-H
Verify: cargo test; bench
Accept: Runs in real time on CPU, or has a documented no-go.
Assets: 
Risk: high
Notes: 

### P5-D [P5, M] so-vits-svc 4.1 BYO and w-okada/wok000 compatibility
Goal: A sovits family (c, f0, mel2ph, uv, noise and sid inputs). An audit of the wok000 weights_gpl pack and w-okada ONNX, adding roles in rvc.rs. Mel-input FCPE support.
Research: §4.2 so-vits-svc 4.1 user exports; §4.2 wok000 weights_gpl ONNX pack
Owned: src-tauri/src/services/neural_audio/vc/families/sovits.rs, src-tauri/src/services/neural_audio/vc/{rvc.rs,fcpe.rs}
New: 
Hotspots: 
Depends: P5-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vc
Accept: Pack models load, or each rejection is specific.
Assets: 
Risk: medium
Notes: 

### P5-E [P5, M] Sidecar protocol v2 inside the harness
Goal: sidecar_bridge v2: async pipelined frames, a timeout equal to the budget, rate negotiation, reference upload, and parameters including pitch. Keeps v1 compatibility. Wrapped as a VoiceConversionModel. process_host recipes for X-VC, StreamVoiceAnon and seed-vc-realtime, bound to 127.0.0.1.
Research: §4.2 X-VC sidecar; §4.2 StreamVoiceAnon privacy mode; §4.2 Seed-VC sidecar fallback
Owned: src-tauri/src/services/neural_audio/sidecar_bridge.rs, src-tauri/src/services/neural_audio/vc/families/sidecar_v2.rs, scripts/neural-sidecar/**
New: scripts/neural-sidecar/{PROTOCOL_V2.md,xvc_sidecar.py,streamvoiceanon_sidecar.py,seedvc_sidecar.py}
Hotspots: 
Depends: P5-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml sidecar (echo server; a 700 ms delay stays within budget)
Accept: A stalled sidecar never blocks the loop.
Assets: 
Risk: medium
Notes: 

### P5-F [P5, M] Shared reference-voice recorder and voice library
Goal: voice-library plugin: record_reference_clip through the hub, trimmed by Silero, with SNR and clipping checks and LUFS normalisation. Transcript from local STT. Stored under app_data/voices/<id>/{ref.wav, meta.json, cache/<backend>.bin}. Includes the recorder UI component, shared by VC zero-shot and TTS cloning.
Research: §2.2 shared clean 3–10 s reference voice with a cached embedding per backend
Owned: src-tauri/src/services/voice_library/**, src/core/services/voice_library/**, src/core/ui/inspector/components/ReferenceVoiceRecorder.tsx
New: scripts/i18n-staging/P5-F.json
Hotspots: 
Depends: P5-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml voice_library
Accept: Clean clips are trimmed, stored and listed in both pickers.
Assets: 
Risk: low
Notes: 

### P5-G [P5, M] Voice-changer family UI
Goal: Family-aware slots, a zero-shot reference picker, singing mode, sidecar configuration and experimental labels.
Research: Family UI (no standalone research item)
Owned: src/core/ui/inspector/voice_changer/**, src/core/services/voice_changer/**
New: scripts/i18n-staging/P5-G.json
Hotspots: 
Depends: P5-S0
Verify: pnpm typecheck; pnpm test:unit
Accept: Every family is usable from the UI.
Assets: 
Risk: low
Notes: 

### P5-I0 [P5, S] P5 integrate
Goal: Merge i18n, update the docs (zero-shot, singing, sidecar v2), update the benchmark scorecard, run the exit matrix.
Research: §7 P3 optional engines exit (integration)
Owned: (hotspots), docs/**
New: 
Hotspots: Collected P5 requests
Depends: P5-A, P5-B, P5-C, P5-D, P5-E, P5-F, P5-G
Verify: All gates plus the P5 exit matrix
Accept: P5 exit criteria met
Assets: 
Risk: medium
Notes: 

### P6-S0 [P6, M] P6 scaffold: topics, windows, keys, backend values
Goal: Take the checkpoint. Add others.stt, others.translation and readings.stt, with registerEvent, pickers and the remote-safe text-element schema. Add a captions window capability. Root keys lanAccess{mode, token, legacyNoToken} and captionWindow. VRC schema fields for mute sync, OSC listen port, chatbox pacing, emotion parameters and overlay. New TTS values qwen3, edge, gemini, sbv2, gptSovits, pocket, chatterboxLocal, ggmlTts; STT value geminiLive. Plugin stubs: vr-overlay, text-readings, edge-tts, gemini-tts, tts-helpers, vst3-host. Cargo: wasapi, cosmic-text, tiny-skia, mdns-sd, jpreprocess, jieba-rs, pinyin, wana_kana. on_window_event closes secondary windows.
Research: Phase scaffolding
Owned: (hotspots), src/core/ui/inspector/components/input/index.tsx, src/client/elements/text/schema.ts, src/core/services/tts/index.ts (backend map only), src/core/services/stt/index.ts (backend map only)
New: src-tauri/capabilities/captions.json, stubs for the new modules and panels
Hotspots: Registrations and dependencies
Depends: P5-I0
Verify: All gates
Accept: Contracts compile, and everything is OFF by default.
Assets: 
Risk: medium
Notes: 

### P6-A [P6, L] Other players: loopback transcription and reverse translation
Goal: A loopback speech pipeline on an output device through the hub. Phase 2: per-process loopback of VRChat.exe via wasapi, with device fallback. An Others service publishing others.stt and a reverse-direction translation pipeline instance publishing others.translation. Off by default, never reaching the chatbox or TTS.
Research: §5 transcribe and translate other players (phase 1 device loopback, phase 2 per-process, separate topic, reverse direction)
Owned: src/core/services/others/**, src/core/ui/inspector/inspector_others.tsx, src-tauri/src/services/audio/loopback_process.rs, src-tauri/src/services/speech/session.rs
New: scripts/i18n-staging/P6-A.json
Hotspots: 
Depends: P6-S0
Verify: pnpm test:unit (a test that 'text' subscribers never receive others.*) | cargo test --manifest-path src-tauri/Cargo.toml audio speech::session
Accept: The user's own captions are unaffected.
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P6-B [P6, L] Caption window and SteamVR overlay
Goal: An always-on-top, click-through caption window: an early route with no ApiServer, closing with main. A vr_overlay plugin using lazy openvr_api.dll FFI (Background app type, handles VREvent_Quit), rendered with cosmic-text and tiny-skia (CJK and RTL) at 10 Hz or less. openvr_api.dll is a runtime pack.
Research: §5 SteamVR caption overlay (and the always-on-top window as the first step)
Owned: src-tauri/src/services/{vr_overlay/**,caption_window.rs}, src/overlay/**, src/index.tsx, src/core/services/overlay/**, src/core/ui/inspector/vrc/OverlaySection.tsx, src-tauri/model-catalog/runtime-openvr.json
New: scripts/i18n-staging/P6-B.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vr_overlay (rasterizer) | Manual with SteamVR, and on a PC without SteamVR
Accept: The app is unaffected when VR isn't present, and SteamVR is never launched.
Assets: 
Risk: medium
Notes: 

### P6-C [P6, M] VRChat OSC: MuteSelf, chatbox pacing, Smart-Turn commit, tags
Goal: OSC receiver: OSCQuery via mdns-sd with a fallback on port 9001, tolerating bind failure. MuteSelf drives stt.setMuted and the Rust gate. The send socket binds 0.0.0.0:0 for non-loopback hosts. Chatbox pagination past 144 characters with pacing (≥1.3 s). Optional commit on turn_end for chatbox and TTS. SenseVoice emotion and event tags go to configurable OSC avatar parameters and caption tags.
Research: §5 sync with VRChat's mic mute; §1.2 SenseVoice tags as caption tags or VRChat OSC parameters; §1.2 Smart Turn consumption (chatbox and TTS commit)
Owned: src-tauri/src/services/osc.rs, src-tauri/plugin-commands/osc.txt, src/core/services/vrc/**, src/core/ui/inspector/inspector_vrc.tsx
New: src-tauri/src/services/osc_query.rs, src/core/services/vrc/__tests__/chatboxPager.test.ts, scripts/i18n-staging/P6-C.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml osc; pnpm test:unit -- src/core/services/vrc | Manual in VRChat
Accept: Mute in VRChat stops the engine feed within 200 ms. | No chatbox truncation.
Assets: 
Risk: medium
Notes: 

### P6-D [P6, L] LAN auth stage 2
Goal: A persistent token required for non-loopback clients. Bind mode. The token travels as the PeerJS key and as ?token= on /pubsub and every HTML page. OBS links, mic.html and a Link apps remote-token field. Show and regenerate UI. Threat-model update.
Research: §5 authenticated LAN web/WebSocket server (stage 2: token)
Owned: src-tauri/src/services/web/**, src/config.ts, src/shared/services/peer/**, src/shared/services/pubsub/index.ts, public/{mic,obs-captions,inputfield}.html, src/core/services/obs/browserSource.ts, src/core/ui/inspector/inspector_settings.tsx, src/core/ui/inspector/stt/ExternalMic.tsx, src/utils/localPages.ts
New: src-tauri/src/services/web/auth.rs, scripts/i18n-staging/P6-D.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml web; full manual client matrix
Accept: The LAN token exit criterion is met.
Assets: 
Risk: high
Notes: 

### P6-E [P6, L] Qwen3-TTS GPU tier (qwentts.cpp tts-server helper)
Goal: Pack build (vulkan default, cuda13 optional). Launch through process_host. Streaming via the OpenAI client. A clone registry fed from the voice library. 9 preset voices with emotion instructions. VoiceDesign on 1.7B. 0.6B while the VC runs, via the governor.
Research: §2.1 Qwen3-TTS 0.6B/1.7B via qwentts.cpp; §2.1 Qwen3-TTS 1.7B BF16/Q8_0 including VoiceDesign
Owned: src-tauri/src/services/tts_helpers/qwen.rs, scripts/runtime-packs/qwentts/**, src-tauri/model-catalog/{tts-qwen3.json,runtime-qwentts.json}, src/core/services/tts/services/qwen3.ts, src/core/services/tts/schema/qwen3.ts, src/core/ui/inspector/tts/Qwen3.tsx
New: scripts/i18n-staging/P6-E.json
Hotspots: 
Depends: P6-S0
Verify: Manual: time to first audio and RTF logged; cloning from a library clip
Accept: Streaming cloning works.
Assets: qwentts.cpp source | Qwen3-TTS 0.6B GGUF (sequential)
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P6-F [P6, L] Experimental ggml TTS helpers: Magpie, Breeze, OmniVoice, Fish S2 Pro
Goal: A time-boxed spike per engine: NeMo-Speech.cpp Magpie (sharing the runtime with Nemotron if feasible), Breeze-TTS-2.cpp, omnivoice.cpp and s2.cpp. Each becomes a helper-process preset (server or a thin shim) or gets a no-go note. Fish S2 Pro is also reachable through the fishSpeech backend.
Research: §2.1 NVIDIA Magpie TTS Multilingual; §2.1 Breeze TTS 2 (mid and high GPU); §2.1 OmniVoice; §2.1 Fish Audio S2 Pro
Owned: src-tauri/src/services/tts_helpers/{magpie,breeze,omnivoice,s2}.rs, scripts/runtime-packs/{nemo-speech,breeze,omnivoice,s2}/**, src-tauri/model-catalog/tts-ggml-experimental.json, tools/tts-shim/**, src/core/services/tts/services/ggmlTts.ts, src/core/ui/inspector/tts/GgmlTts.tsx
New: scripts/i18n-staging/P6-F.json
Hotspots: 
Depends: P6-S0
Verify: Per-engine build script plus a manual speak test
Accept: Each engine is either an experimental preset or has a documented no-go.
Assets: 
Risk: high
Notes: 

### P6-G [P6, M] Japanese character voices: SBV2/AIVMX and GPT-SoVITS
Goal: The sbv2 server (sbv2_core, pure-Rust OpenJTalk) as a helper process to avoid a second ORT, with AIVMX file-drop import. A GPT-SoVITS api_v2 client with weight switching, a reference clip (upload or path) and streaming_mode.
Research: §2.1 Style-Bert-VITS2 JP-Extra / AIVMX in-process (step 2); §2.1 GPT-SoVITS v2ProPlus/v4 via api_v2; §2.1 user AIVMX/SBV2 voices file-drop
Owned: src-tauri/src/services/tts_helpers/sbv2.rs, src-tauri/src/services/gpt_sovits.rs, scripts/runtime-packs/sbv2/**, src/core/services/tts/services/{sbv2,gptSovits}.ts, src/core/services/tts/schema/{sbv2,gptSovits}.ts, src/core/ui/inspector/tts/{Sbv2,GptSovits}.tsx
New: scripts/i18n-staging/P6-G.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml gpt_sovits (mock); manual AIVMX drop
Accept: An AIVMX voice speaks, and GPT-SoVITS streams.
Assets: 
Risk: medium
Notes: 

### P6-H [P6, M] Free and keyless cloud TTS with local fallback
Goal: A native Rust Edge Read Aloud client over tungstenite (headers, Sec-MS-GEC token, mp3/webm decode). Gemini 3.8 Flash and Flash-Lite TTS with the key in x-goog-api-key and a streaming s16le decode. A pipeline fallback chain to Kokoro on 429, 503 or a 3 s stall. Verify the Fish S2.1 free-API preset end to end.
Research: §2.1 Edge Read Aloud voices (zero setup, with local fallback); §2.3 Gemini 3.8 Flash TTS / Flash-Lite TTS; §2.3 Fish S2.1 Pro free API (end-to-end preset)
Owned: src-tauri/src/services/{edge_tts.rs,gemini_tts.rs}, src/core/services/tts/pipeline.ts, src/core/services/tts/services/{edge,gemini}.ts, src/core/services/tts/schema/{edge,gemini}.ts, src/core/ui/inspector/tts/{Edge,Gemini}.tsx
New: scripts/i18n-staging/P6-H.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml edge_tts gemini_tts (mocks); pnpm test:unit (fallback policy)
Accept: An outage falls back to Kokoro within 1 s.
Assets: 
Risk: medium
Notes: 

### P6-I [P6, M] In-process cloning extras: Pocket TTS and Chatterbox Multilingual V3
Goal: Kyutai Pocket TTS for CPU cloning, and Chatterbox Multilingual V3 on Sigil's ORT as the cloning fallback. Both consume voice-library references.
Research: §2.1 Kyutai Pocket TTS; §2.1 Chatterbox Multilingual V3 (ORT fallback)
Owned: src-tauri/src/services/local_tts/{pocket.rs,chatterbox_ort.rs}, src-tauri/model-catalog/tts-cloning.json, src/core/services/tts/services/{pocket,chatterboxLocal}.ts, src/core/ui/inspector/tts/{Pocket,ChatterboxLocal}.tsx
New: scripts/i18n-staging/P6-I.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml local_tts -- --ignored
Accept: Both clone from a library entry.
Assets: 
Risk: medium
Notes: 

### P6-J [P6, M] SAPI and Web Speech to PCM; mute the mic while TTS plays
Goal: SAPI renders to an SpMemoryStream on a dedicated STA thread and goes to the sink, so it can be routed to the VC. Web Speech is documented as device-only. A policy ducks or mutes at the hub gate while any TTS plays into the VC.
Research: §5 route every TTS backend through the VC, part 2 (SAPI PCM and mic mute while TTS plays)
Owned: src-tauri/src/services/windows_tts/**, src-tauri/src/services/tts_sink/**, src/core/services/tts/index.ts, src/core/services/tts/services/{windows,native}.ts
New: 
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml windows_tts tts_sink; manual SAPI into the VC
Accept: Every PCM backend can be routed, and the mic mutes during TTS when enabled.
Assets: 
Risk: medium
Notes: 

### P6-K [P6, M] Gemini Live transcribe and translate
Goal: A Gemini 3.5 Transcribe Live STT backend over a webview WebSocket, rotating sessions at 9.5 minutes with 1 s overlap and word dedupe. Gemini Live Translate as a preview combined backend publishing text.stt and text.translation. Model ids are editable.
Research: §1.4 Gemini 3.5 Transcribe Live; §3.3 Gemini 3.5 Live Translate
Owned: src/core/services/stt/services/geminiLive.ts, src/core/services/stt/schema/geminiLive.ts, src/core/ui/inspector/stt/GeminiLive.tsx, src/core/services/translation/services/geminiLive.ts
New: src/core/services/stt/__tests__/geminiRotation.test.ts, scripts/i18n-staging/P6-K.json
Hotspots: 
Depends: P6-S0
Verify: pnpm test:unit (rotation seam dedupe); manual with the user's key
Accept: No caption gap across rotation.
Assets: 
Risk: medium
Notes: 

### P6-L [P6, M] Readings and the music-gate A/B
Goal: text-readings plugin: jpreprocess (dictionary via the store) with wana_kana for hiragana and romaji, and jieba-rs plus pinyin with tone sandhi. Published on readings.stt keyed by utteranceId. FireRedVAD music/singing gate in the speech segmenter behind a default-off A/B flag.
Research: §5 Japanese romaji/hiragana and Chinese pinyin readings; §1.2 FireRedVAD music/singing detector (A/B)
Owned: src-tauri/src/services/text_readings/**, src/core/services/readings/**, src-tauri/src/services/speech/music_gate.rs, src-tauri/model-catalog/{readings.json,firered.json}
New: scripts/i18n-staging/P6-L.json
Hotspots: 
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml text_readings speech::music_gate; pnpm test:unit
Accept: Readings are correct on fixtures, and the gate is off by default.
Assets: 
Risk: low
Notes: 

### P6-M [P6, L] Optional: host a user-installed Beatrice 2 VST3 (time-boxed)
Goal: A minimal VST3 host (vst3 crate) that finds the user's installed Beatrice 2 .vst3 and processes audio through the harness as BeatriceVst, under the budget. The option is hidden when the plugin is absent.
Research: §4.2 Beatrice 2 CPU tier via the user-installed VST3
Owned: src-tauri/src/services/vst3_host/**, src-tauri/src/services/neural_audio/vc/families/beatrice_vst.rs
New: 
Hotspots: Only if go: vst3 dependency and factory arm (integrator)
Depends: P6-S0
Verify: cargo test --manifest-path src-tauri/Cargo.toml vst3_host; manual with the user's install
Accept: Works with the user's install, or can be dropped with no side effects.
Assets: 
Risk: high: lowest priority
Notes: 

### P6-I0 [P6, M] P6 integrate
Goal: Merge i18n, update docs and the threat model, and run the full feature matrix including the LAN token rollout.
Research: §7 P4 roadmap exit (integration)
Owned: (hotspots), docs/**
New: 
Hotspots: Collected P6 requests
Depends: P6-A, P6-B, P6-C, P6-D, P6-E, P6-F, P6-G, P6-H, P6-I, P6-J, P6-K, P6-L, P6-M
Verify: All gates plus the P6 exit matrix
Accept: P6 exit criteria met
Assets: 
Risk: medium
Notes: 

### P7-A [P7, L] Speech benchmark campaign and defaults
Goal: `--headless speech-bench` covers:
- WER on real pre-conversion VRChat clips with bleed;
- CPU speed for Qwen3-ASR-0.6B, Hy-MT2 1.8B and Bergamot CJK with the load simulator or VRChat running;
- the GPU pick matrix under load (Qwen3-1.7B, Cohere, R2T2, Voxtral, qwentts, Breeze, Hy-MT2-7B, MiLMMT-4B);
- an MT blind A/B with chrF (MetricX if available) plus a human-check page;
- a TTS listening test of Qwen3-TTS vs Kokoro vs Magpie.

Update the router tables and presets from the winners.
Research: §8 hardware under load; §8 CPU speed; §8 STT accuracy on real audio; §8 translation quality blind A/B; §8 listening test Qwen3-TTS vs Kokoro vs Magpie
Owned: src-tauri/src/services/bench/speech.rs, scripts/speech-bench/**, src-tauri/model-catalog/stt-routing.json, src/core/services/translation/router.ts (default tables), docs/benchmarks/SPEECH_BENCHMARKS.md
New: 
Hotspots: Integrator: cli arm speech-bench
Depends: P6-I0
Verify: cargo test --manifest-path src-tauri/Cargo.toml bench speech::router; pnpm test:unit -- router
Accept: Reports are published and friends can export theirs.
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P7-B [P7, L] Voice-changer campaign, competitor head-to-head, preset tuning
Goal: H1 live-record runs from friends. Competitor comparison with the user's recordings or installs. Listening tests: X-VC vs Seed-VC vs MeanVC2, and SwiftF0 vs RMVPE in cents. Final T1–T10 scorecard. Tune presets and autotuner priors.
Research: §8 listening tests X-VC vs Seed-VC vs MeanVC2; Final proof of the vc_best_engine targets
Owned: scripts/vc-bench/**, tools/listening-test/**, docs/benchmarks/VC_BENCHMARKS.md, src-tauri/src/services/neural_audio/vc/engine/presets.rs, src/core/services/voice_changer/presets.ts
New: 
Hotspots: 
Depends: P6-I0
Verify: pwsh scripts/bench/run-perf-gate.ps1 -Profile final
Accept: Every T1–T10 row is met, or has a remediation plan.
Assets: 
Risk: medium: depends on friends' H1 hardware
Notes: 

### P7-C [P7, M] Documentation and manual
Goal: Rewrite OPERATOR_REFERENCE, DEVELOPER_GUIDE (runtime policy, ABI, catalog and template authoring), the manual, RELEASE_AND_ENGINEERING and CHANGELOG. Mark VOICE_CHANGER_AI_PLAN.md as superseded by the SVE docs. Document the external-voice-changer-over-virtual-cable option and the paid-cloud appendix. Rejected models are listed as facts. No license warnings.
Research: Docs for everything; §4.2 external VC on a virtual cable (documented BYO path)
Owned: docs/**, README.md, ARCHITECTURE.md, CHANGELOG.md
New: 
Hotspots: 
Depends: P6-I0
Verify: pnpm build (sync-manual); $env:PW_PREVIEW_ONLY=1; pnpm test:e2e
Accept: The docs are current.
Assets: 
Risk: low
Notes: 

### P7-D [P7, M] Release packaging and whisper-rs retirement decision
Goal: The user publishes runtime packs, templates and exports through the prepared workflows. Pin every release catalog hash (no trust-on-first-use). Retire whisper-rs only if transcribe.cpp shows WER and RTF parity. Run the packaged smoke on a clean VM with no GPU and no packs.
Research: §0 single ggml copy (retire whisper-rs on parity)
Owned: .github/workflows/**, scripts/release/**, src-tauri/model-catalog/**, src-tauri/src/services/whisper.rs
New: scripts/release/packaged-smoke-all.ps1
Hotspots: Integrator: remove whisper-rs from Cargo only on parity
Depends: P7-A
Verify: pnpm tauri build; pwsh scripts/release/verify-bundle.ps1; pwsh scripts/release/packaged-smoke-all.ps1
Accept: Hashes are pinned and the smoke passes.
Assets: 
Risk: medium
Notes: amended by '## P1 outcomes (binding amendments for P2–P7)' (P1-I0).

### P7-I0 [P7, S] Final integrate
Goal: Apply the approved default flips and migrations, run the final gates, and prepare the release notes.
Research: Final integration
Owned: (hotspots)
New: 
Hotspots: Approved default flips
Depends: P7-A, P7-B, P7-C, P7-D
Verify: All gates plus pnpm test:e2e
Accept: P7 exit criteria met
Assets: 
Risk: low
Notes: 