//! transcribe.cpp wrapper (GPU tier of the STT stack): `sigil_transcribe.dll`, built by Sigil with a
//! sigil_engine ABI v1 shim over transcribe.h (`native/engines/transcribe/`), loaded lazily through
//! `native_runtime` from its runtime packs (`runtime/sigil-transcribe-{cuda13,vulkan,cpu-avx512,
//! cpu-avx2,cpu}`, `model-catalog/runtime-transcribe.json`). Decision record:
//! `docs/adr/0003b-transcribe-cpp.md`.
//!
//! Owned by P1-G2; scaffolded (empty) by P1-S0b.
//!
//! # Frozen contract (P1-S0b)
//! - [`headless_probe`]`(&EngineProbeArgs) -> Result<String, String>`: body of
//!   `Sigil.exe --headless engine-probe transcribe <model> <wav> [--variant cpu|cuda13]`; returns
//!   the JSON report (RTF, VRAM, transcript).
//!
//! # API for the P2 `speech` plugin (P1-G2)
//! - [`TranscribeModel::open`]`(&ModelOptions)`: best installed variant (or a runtime folder), GPU
//!   failures fall through to the next variant / the CPU backend with a reported reason
//!   ([`TranscribeModel::fallback_reason`]); [`TranscribeModel::info`] says which backend really
//!   runs the model.
//! - [`TranscribeModel::transcribe`] (16 kHz mono utterance) / [`TranscribeModel::transcribe_audio`]
//!   (any rate, resampled with rubato) → [`Transcript`]; [`TranscribeModel::canceller`] interrupts a
//!   running call from another thread; [`TranscribeModel::open_stream`] for streaming models.
//! - Everything is blocking: call it from `spawn_blocking` / a worker thread. A model is `Send`,
//!   calls on one model are serialized by `&mut self`. transcribe.cpp 0.x allows one compute per
//!   loaded model at a time, and a stream counts from its first push until `finish`: while one is
//!   mid-utterance, `transcribe` and other streams of that model fail with
//!   [`ErrorKind::InvalidInput`]. Load one model per worker / caption source for parallel work.
//! - GPU runs are kept away from ggml-cuda's in-process abort paths (a failed VRAM pool growth or
//!   a lost device ends Sigil.exe): loads need the model's estimated VRAM plus
//!   [`ModelOptions::min_free_vram_mb`], GPU warm-ups decode [`GPU_WARMUP_SECONDS`] so the pool is
//!   sized at load time, and every GPU run checks the headroom first (see `engine`'s docs and ADR
//!   0003b).
//! - Nothing is load-time linked; with no pack installed `open` returns
//!   [`ErrorKind::NotInstalled`] and the app is unaffected.

mod engine;
mod probe;
#[cfg(test)]
mod tests;
mod types;

// The API the P2 `speech` plugin consumes; until it lands only the probe and tests use it.
#[allow(unused_imports)]
pub use engine::{
    installed_variants, to_16k_mono, ErrorKind, RuntimeSelection, TranscribeCanceller, TranscribeError, TranscribeModel, TranscribeStream,
    DEV_CUDA_BIN_ENV, ENGINE, RUNTIME_DIR_ENV, SAMPLE_RATE,
};
#[allow(unused_imports)]
pub use types::{
    default_threads, BackendChoice, Capabilities, DeviceInfo, DeviceList, Features, KvType, Limits, ModelInfo, ModelOptions, Segment, StreamFinal,
    StreamUpdate, Timestamps, Timings, Toggle, TranscribeOptions, Transcript, Word, CPU_WARMUP_SECONDS, DEFAULT_MIN_FREE_VRAM_MB, GPU_WARMUP_SECONDS,
};

use crate::services::speech_rt::EngineProbeArgs;

/// Body of `Sigil.exe --headless engine-probe transcribe ...` (options in [`probe`]'s docs).
pub fn headless_probe(args: &EngineProbeArgs) -> Result<String, String> {
    probe::run(args)
}
