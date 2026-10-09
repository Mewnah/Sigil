# Architecture decision records

Decisions that are expensive to reverse (runtimes, native loading, installers, security model) are
recorded here. Each ADR belongs to the workstream that made the decision; the phase integrator links
it from this index. Start a new one from [0000-template.md](0000-template.md) and number it after the
highest existing record (numbers reserved by the [implementation plan](../SPEECH_STACK_IMPLEMENTATION_PLAN.md)
keep their slot even when written out of order).

| ADR | Title | Status | Workstream |
|-----|-------|--------|------------|
| [0000](0000-template.md) | Template | — | — |
| [0001](0001-ort-upgrade.md) | ONNX Runtime 1.28 (ort 2.0.0-rc.13, CUDA 13) with KittenTTS vendored | Accepted (P1 gate passed; CUDA pack install and packaged smoke manual) | P1-S0a |
| [0002](0002-sherpa-onnx.md) | sherpa-onnx 1.13.8 C API as the in-process CPU speech runtime | Accepted (go; Nemotron 3.5 / Qwen3-ASR supported but not model-run in P1) | P1-G1 |
| [0003](0003-gpu-ep-delivery.md) | Delivering GPU execution providers and reporting the one in use | Accepted (P0 interim; its P1 follow-ups are resolved by [0007](0007-native-runtime-packs.md)) | P0-H |
| [0003b](0003b-transcribe-cpp.md) | transcribe.cpp as an in-process sigil_engine DLL (GPU tier of STT) | Accepted (go: cuda13 + CPU variants built and measured; vulkan build path ready, unmeasured) | P1-G2 |
| [0004](0004-rvc-templates.md) | RVC voices run as Sigil templates with imported weights | Accepted (P3 built the matrix, the production importer and the engine on it; layout-v2 voices wait on importer and engine support) | P1-G3 |
| [0005](0005-process-host-llama.md) | Process host for helper processes, with llama-server as the local LLM runtime | Accepted (P4-B built the llm-server on it; the live llama-server + Hy-MT2 smoke is still manual) | P1-E |
| [0006](0006-vc-zero-shot-singing.md) | Zero-shot and singing families run in Sigil's engine (Seed-VC, DDSP-SVC, MeanVC2) | Accepted (P3-H spike: Seed-VC GO on CUDA / DirectML fp32; DDSP-SVC GO on CUDA; MeanVC2 GO as an experimental CPU family pending the P5-C A/B) | P3-H |
| [0007](0007-native-runtime-packs.md) | Native runtime packs: lazily loaded sigil_engine DLLs, and the CUDA 13 runtime and provider as model-store packs | Accepted (provider still bundled until its pack is published) | P1-D1 |
| [0008](0008-bergamot.md) | Mozilla Bergamot as a JSON-lines helper process (zero-VRAM CPU translation) | Accepted (helper build and the en↔ja latency run pending, manual; no-go fallback NLLB-600M + Hy-MT2 1.8B on CPU) | P4-E |
| [0009](0009-vc-scorer-v2.md) | Voice-changer fidelity scorer v2 (T5 v2 definitions: exact CAM++, fed-F0 errors, conditional click tests, envelope MCD) | Accepted (limits pre-registered from the measured floors before the engine workstreams re-scored) | P3B-B |
| [0010](0010-sve-optimization-p3b.md) | Sigil Voice Engine optimization (P3B): lossless and measured-better paths on, fidelity-trading levers as switches | Accepted (defaults flipped from interleaved evidence; the final scorecard of the shipped defaults is P7-B's) | P3B-I0 |
| [0011](0011-vc-content-encoder-precision.md) | Content-encoder precision defaults (GPU fp16, CPU int8 kept as recorded waivers) and the deferred plan's CPU cross-check | Accepted (P3B-D review fix; the fp32 encoder's CPU cost on the Streamer is P7-B's) | P3B-D |

