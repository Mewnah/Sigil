//! sherpa-onnx C API wrapper (CPU tier): online / offline recognizers, OfflineTts, spoken language
//! ID and speaker embedding, loaded through libloading from the sherpa runtime pack
//! (`onnxruntime.dll` by full path first, then `sherpa-onnx-c-api.dll`, both with
//! `native_runtime::load_library_in_dir`). Never registers GPU EPs.
//!
//! Owned by P1-G1 (decision record: `docs/adr/0002-sherpa-onnx.md`).
//!
//! # Modules
//! - [`bindings`]: vendored bindgen output for sherpa-onnx 1.13.8 `c-api.h` (dynamic loading,
//!   compile-time layout checks). Regenerate with `scripts/runtime-packs/sherpa/gen-bindings.ps1`.
//! - [`lib`]: [`SherpaRuntime`] (load order, version / ORT / module checks, process-wide cache,
//!   never unloaded), pack location ([`RUNTIME_ID`] through `model_store::installed_path`), and the
//!   FFI helpers (C string arena, 8.3 short paths for non-ASCII folders, per-object call lock).
//! - [`online`]: streaming ASR ([`OnlineRecognizer`] → [`OnlineStream`]).
//! - [`offline`]: utterance ASR ([`OfflineRecognizer::transcribe`]).
//! - [`tts`]: [`OfflineTts::generate`] with chunk callback and cancel.
//! - [`lid`]: [`SpokenLanguageId`]; [`speaker`]: [`SpeakerEmbeddingExtractor`].
//! - [`detect`]: official sherpa model package folder → config.
//! - [`probe`]: `Sigil.exe --headless engine-probe sherpa` and [`probe::validate_in_child`].
//! - `onnx_info`: ONNX graph output count (tells a NeMo streaming transducer from a k2 one before
//!   sherpa sees it).
//!
//! # Rules
//! - Every engine is CPU-only (`provider = "cpu"`), 2 threads by default (at most 4).
//! - Every native object serializes its calls (`lib::CallLock`), so all wrappers are
//!   `Send + Sync`; construction and decoding block, so callers run them on `spawn_blocking` /
//!   worker threads, never on async or audio threads.
//! - Referenced files and option combinations are checked before any native call: sherpa-onnx
//!   answers several unsupported combinations with `exit(-1)` (modified beam search or hotwords on a
//!   non-transducer or a NeMo streaming transducer, audio after `InputFinished`), so the wrappers
//!   refuse them with `Err`. sherpa-onnx can still abort the process on a corrupt model file (C++
//!   exceptions do not cross its C API): catalog models are hash-verified by the store; anything
//!   else goes through [`probe::validate_in_child`] (with the exact config that will be loaded)
//!   first.
//! - The C API reads a float option of exactly 0 as "unset" and substitutes its default
//!   (`SHERPA_ONNX_OR`); a deliberate 0 is passed as `lib::C_FLOAT_ZERO`, except TTS
//!   `silence_scale`, which sherpa applies only within `[0.01, 10]` (0 maps to 0.01).
//! - No pack installed → every entry point returns `Err`; the app never needs sherpa to start.
//!
//! # Frozen contract (P1-S0b)
//! - [`headless_probe`]`(&EngineProbeArgs) -> Result<String, String>`: body of
//!   `Sigil.exe --headless engine-probe sherpa <model> <wav>`; returns the JSON report.

// Consumers arrive in P2 (the `speech` plugin); until then only the probe and tests use most of
// the API surface.
#![allow(dead_code)]

pub mod bindings;
pub mod detect;
pub mod lib;
pub mod lid;
pub mod offline;
pub mod online;
mod onnx_info;
pub mod probe;
pub mod speaker;
pub mod tts;

#[cfg(test)]
mod tests;

// The API surface for P2's `speech` plugin (unused inside the crate until then).
#[allow(unused_imports)]
pub use self::{
    detect::{detect, DetectHints, SherpaModel},
    lib::{runtime_dir, RuntimeInfo, SherpaRuntime, EXPECTED_VERSION, RUNTIME_ID},
    lid::{LanguageIdConfig, SpokenLanguageId},
    offline::{DecodeOptions, OfflineModel, OfflineRecognizer, OfflineRecognizerConfig, OfflineResult},
    online::{Decoding, EndpointRules, OnlineModel, OnlineRecognizer, OnlineRecognizerConfig, OnlineResult, OnlineStream},
    speaker::{cosine_similarity, SpeakerEmbeddingConfig, SpeakerEmbeddingExtractor},
    tts::{ChunkCallback, GenerateOptions, GeneratedAudio, OfflineTts, TtsConfig, TtsModel},
};

use crate::services::speech_rt::EngineProbeArgs;

/// Body of `Sigil.exe --headless engine-probe sherpa ...`.
pub fn headless_probe(args: &EngineProbeArgs) -> Result<String, String> {
    probe::headless_probe(args)
}
