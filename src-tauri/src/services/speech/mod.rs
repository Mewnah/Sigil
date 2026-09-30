//! The `speech` plugin: Sigil's in-process STT pipeline (P2).
//!
//! capture hub (raw 16 kHz mono consumer) → Silero segmenter (`speech_core`) → ordered engine
//! worker with a latency budget → events. Keyed pipelines: `mic` (Service_STT) and, from P6,
//! `loopback` (other players). Engines sit behind the [`engine_api::AsrEngine`] trait: Whisper,
//! the HTTP utterance engine, sherpa-onnx (CPU tier) and transcribe.cpp (GPU tier). The router
//! picks engines per language, tier, VRAM and voice-changer state; two-pass re-decodes replace
//! partials under the same segment id; Smart Turn emits commit hints.
//!
//! Scaffolded by P2-S0. This file (module list + plugin registration) and [`engine_api`] /
//! [`events`] are scaffold files; every other module belongs to its workstream:
//!
//! | Module | Owner | Role |
//! |---|---|---|
//! | [`pipeline`], [`worker`], [`utterance`], [`mute`] | P2-A | keyed pipelines, hub consumer, segmenter, ordered worker + budget, epochs, Rust mute gate, the pipeline IPC commands |
//! | [`engines`] (`mod`, `whisper`, `http`) | P2-A | engine factory (`engines::create`), Whisper and HTTP utterance engines |
//! | `engines::{sherpa_online, sherpa_offline}` | P2-B | sherpa-onnx engines (Nemotron, Parakeet, Qwen3-ASR-0.6B, SenseVoice, Moonshine v2) |
//! | [`router`], [`two_pass`], [`smart_turn`], [`lang`] | P2-C | language router + `preview_route`, two-pass queue, Smart Turn, language codes |
//! | `engines::transcribe`, [`worker_proc`] | P2-D | transcribe.cpp GPU tier, optional worker-process isolation (`--headless stt-worker`) |
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! - Engine API: [`engine_api`] (`AsrEngine`, `EngineSpec`, `EngineKind`, `RouteRequest`,
//!   `RoutePlan`, `PipelineConfig`, `DecodeOptions`, `AsrResult`, `AsrError`, ...).
//! - Events: [`events`] (`speech:partial` / `final` / `status` / `utterance` / `turn_end`).
//! - Engine factory: `engines::create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>,
//!   AsrError>` dispatching on [`engine_api::EngineKind`] to `<module>::create` with the same
//!   signature in each engine module.
//! - IPC (`plugin-commands/speech.txt`; TS mirror types `RustSpeech*` in `src/types.ts`):
//!   - `plugin:speech|start_pipeline { config: PipelineConfig }` → `PipelineStarted`
//!     (`pipeline::start_pipeline`, P2-A). Replaces a running pipeline with the same id.
//!   - `plugin:speech|stop_pipeline { pipelineId, epoch? }` → `void`; after it resolves no event of
//!     that session is emitted (`pipeline::stop_pipeline`, P2-A). With `epoch` it stops only that
//!     session, while it is still the key's current one, and never cancels a queued start (P2 gate).
//!   - `plugin:speech|get_pipeline_status` → `PipelineStatus[]` (`pipeline::get_pipeline_status`).
//!   - `plugin:speech|take_pipeline_utterance { pipelineId, segmentId }` → `ArrayBuffer` (raw
//!     little-endian f32 mono 16 kHz; only with `retainPcm`; each utterance once)
//!     (`pipeline::take_pipeline_utterance`, P2-A).
//!   - `plugin:speech|preview_route { request: RouteRequest }` → `RoutePlan` (`router::preview_route`,
//!     P2-C; pure, installs nothing).
//! - Setup hooks (called from this plugin's `setup`, must return quickly and spawn threads for
//!   slow work): `pipeline::setup(&AppHandle)` (P2-A) and `engines::setup(&AppHandle)`, which calls
//!   `engines::sherpa_offline::setup` (P2-B: load the sherpa runtime early when a sherpa engine is
//!   configured; it refuses to load after another `onnxruntime.dll`).
//! - Headless: `Sigil.exe --headless stt-worker [<spec.json>]` → `worker_proc::headless_stt_worker`
//!   (P2-D).

pub mod engine_api;
pub mod engines;
pub mod events;
pub mod lang;
pub mod mute;
pub mod pipeline;
pub mod router;
pub mod smart_turn;
pub mod two_pass;
pub mod utterance;
pub mod worker;
pub mod worker_proc;

use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime,
};

/// Plugin name (`plugin:speech|...`, capability `speech:default`).
pub const PLUGIN_NAME: &str = "speech";

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .setup(|app, _api| {
            pipeline::setup(app);
            engines::setup(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            pipeline::start_pipeline,
            pipeline::stop_pipeline,
            pipeline::get_pipeline_status,
            pipeline::take_pipeline_utterance,
            router::preview_route,
        ])
        .build()
}
