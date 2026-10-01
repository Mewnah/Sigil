# Documentation hub

| Document | Audience |
|----------|----------|
| [manual/README.md](manual/README.md) | End users — chapters under **manual/** |
| [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md) | STT/TTS backends, speech on this PC (automatic STT, local voices), Models & Runtimes and runtime packs, translation backends and AI transform, reliability, voice changer, GPU packs and logs, speech package |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Architecture, host services, connectivity audit, `src/client` rules |
| [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md) | Release QA, GPU EP smoke, P0, P1, P2, P3 and P4 exit matrices, threat model, audits, third-party components, performance, upgrade backlog |
| [benchmarks/README.md](benchmarks/README.md) | Contributors — the benchmark harness (`--headless bench`, specs, metrics, perf gate) |
| [benchmarks/SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md) | Contributors — Sigil's own STT / TTS / machine-translation speed, quality, latency and VRAM measurements |
| [benchmarks/VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md) | Contributors — the Sigil Voice Engine scorecard against the plan's targets T1–T10 (latency, per-block compute, VRAM, CPU, fidelity, autotuner) |
| [SPEECH_STACK_IMPLEMENTATION_PLAN.md](SPEECH_STACK_IMPLEMENTATION_PLAN.md) | Contributors — phased plan for the speech stack (STT, TTS, translation, voice engine) |
| [SPEECH_STACK_RESEARCH_2026-09.md](SPEECH_STACK_RESEARCH_2026-09.md) | Contributors — the research the plan is based on |
| [adr/README.md](adr/README.md) | Architecture decision records |
| [VOICE_CHANGER_AI_PLAN.md](VOICE_CHANGER_AI_PLAN.md) | Superseded by the implementation plan; kept for history |
| [../README.md](../README.md) | Repo root — install, env vars, dev commands |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Short pointer to the developer guide |

Sidecar scripts: [`../scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md), [`../scripts/translation-sidecar/README.md`](../scripts/translation-sidecar/README.md). Voice-engine tooling: [`../scripts/vc-templates/README.md`](../scripts/vc-templates/README.md) (template and base-asset export with parity), [`../scripts/assets/README.md`](../scripts/assets/README.md) (catalog fragments and local assets), [`../scripts/vc-families/README.md`](../scripts/vc-families/README.md) (zero-shot and singing export spike), [`../scripts/vc-bench/README.md`](../scripts/vc-bench/README.md) and [`../tools/listening-test/README.md`](../tools/listening-test/README.md) (competitor runs and blind listening tests). Translation tooling: [`../scripts/mt-bench/README.md`](../scripts/mt-bench/README.md) (MT benchmark, NLLB export and parity), [`../scripts/runtime-packs/bergamot/README.md`](../scripts/runtime-packs/bergamot/README.md) (Bergamot helper pack), [`../scripts/models/make-gguf.md`](../scripts/models/make-gguf.md) (making a GGUF).
