// SPDX-License-Identifier: Apache-2.0
// Vendored from kittentts 0.3.0 (https://github.com/eugenehp/kittentts-rs), the Rust port of
// KittenTTS (https://github.com/KittenML/KittenTTS). Copyright 2026 KittenML and the kittentts-rs
// authors (Eugene Hauptmann). Licensed under the Apache License, Version 2.0; see LICENSE in this
// directory. Modified for Sigil; the changes are listed in NOTICE in this directory.

//! KittenTTS: an ultra-lightweight ONNX text-to-speech engine (24 kHz mono).
//!
//! Vendored from the `kittentts` 0.3.0 crate so Sigil controls its ONNX Runtime version (the
//! crate pinned `ort = "=2.0.0-rc.11"`, which blocked every ORT upgrade). Behaviour is the
//! upstream pipeline:
//! 1. **Text preprocessing** ([`preprocess`]): numbers, currencies, abbreviations → words.
//! 2. **Chunking**: long texts split into ≤ 400-char sentence chunks.
//! 3. **Phonemisation** ([`phonemize`]): the pure-Rust `espeak-ng` crate with bundled data.
//! 4. **Tokenisation** ([`tokenize`]): IPA characters → vocabulary ids.
//! 5. **ONNX inference** ([`model`]): `(input_ids, style, speed)` → audio, on Sigil's shared ORT
//!    environment through `neural_audio::ort_session` (CPU EP, spinning off).
//! 6. **Tail trim** and concatenation.
//!
//! Model files come from a local speech-package folder or a Hugging Face repo ([`hub`]); the
//! upstream `hf-hub` downloader and the C FFI (`ffi.rs`, mobile only) are not vendored.

// Vendored library surface: Sigil uses the text → audio path; the IPA entry points, WAV writers and
// helpers stay available for later backends without being called today. The style lints below fire
// on upstream code, which is kept as published rather than rewritten.
#![allow(dead_code)]
#![allow(
    clippy::derivable_impls,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_strip,
    clippy::useless_asref
)]

pub mod hub;
pub mod model;
pub mod npz;
pub mod phonemize;
pub mod preprocess;
pub mod tokenize;

/// The main TTS model handle; load it with [`hub::load_from_hub`] or [`hub::load_from_dir`].
pub use model::KittenTtsOnnx as KittenTTS;

/// Audio sample rate produced by the model.
pub use model::SAMPLE_RATE;
