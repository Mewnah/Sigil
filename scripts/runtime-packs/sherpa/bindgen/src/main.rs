//! Generates `src-tauri/src/services/speech_rt/sherpa/bindings.rs` from sherpa-onnx's `c-api.h`.
//!
//! Usage: `sigil-sherpa-bindgen <c-api.h> <out.rs> <version> <header-sha256>` (run through
//! `../gen-bindings.ps1`, which downloads the pinned header and sets LIBCLANG_PATH).
//!
//! Dynamic-loading mode: the output is a `SherpaOnnxCApi` struct resolved from a
//! `libloading::Library` (Sigil loads the DLL by full path itself; nothing is load-time linked).
//! Only the functions `speech_rt::sherpa` wraps are allow-listed; their types come along.

/// Every C function the wrapper calls. Adding one here and regenerating is the only way to call a
/// new function (all of them must resolve, so a wrong pack is refused at load time).
const FUNCTIONS: &[&str] = &[
    // Version / build info
    "SherpaOnnxGetVersionStr",
    "SherpaOnnxGetGitSha1",
    "SherpaOnnxGetGitDate",
    "SherpaOnnxGetOnnxruntimeVersionStr",
    // Streaming ASR
    "SherpaOnnxCreateOnlineRecognizer",
    "SherpaOnnxDestroyOnlineRecognizer",
    "SherpaOnnxCreateOnlineStream",
    "SherpaOnnxCreateOnlineStreamWithHotwords",
    "SherpaOnnxDestroyOnlineStream",
    "SherpaOnnxOnlineStreamAcceptWaveform",
    "SherpaOnnxIsOnlineStreamReady",
    "SherpaOnnxDecodeOnlineStream",
    "SherpaOnnxGetOnlineStreamResult",
    "SherpaOnnxDestroyOnlineRecognizerResult",
    "SherpaOnnxOnlineStreamReset",
    "SherpaOnnxOnlineStreamInputFinished",
    "SherpaOnnxOnlineStreamSetOption",
    "SherpaOnnxOnlineStreamIsEndpoint",
    // Non-streaming ASR
    "SherpaOnnxCreateOfflineRecognizer",
    "SherpaOnnxDestroyOfflineRecognizer",
    "SherpaOnnxCreateOfflineStream",
    "SherpaOnnxCreateOfflineStreamWithHotwords",
    "SherpaOnnxDestroyOfflineStream",
    "SherpaOnnxAcceptWaveformOffline",
    "SherpaOnnxOfflineStreamSetOption",
    "SherpaOnnxDecodeOfflineStream",
    "SherpaOnnxGetOfflineStreamResult",
    "SherpaOnnxDestroyOfflineRecognizerResult",
    // Offline TTS
    "SherpaOnnxCreateOfflineTts",
    "SherpaOnnxDestroyOfflineTts",
    "SherpaOnnxOfflineTtsSampleRate",
    "SherpaOnnxOfflineTtsNumSpeakers",
    "SherpaOnnxOfflineTtsGenerateWithConfig",
    "SherpaOnnxDestroyOfflineTtsGeneratedAudio",
    // Spoken language identification
    "SherpaOnnxCreateSpokenLanguageIdentification",
    "SherpaOnnxDestroySpokenLanguageIdentification",
    "SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream",
    "SherpaOnnxSpokenLanguageIdentificationCompute",
    "SherpaOnnxDestroySpokenLanguageIdentificationResult",
    // Speaker embedding
    "SherpaOnnxCreateSpeakerEmbeddingExtractor",
    "SherpaOnnxDestroySpeakerEmbeddingExtractor",
    "SherpaOnnxSpeakerEmbeddingExtractorDim",
    "SherpaOnnxSpeakerEmbeddingExtractorCreateStream",
    "SherpaOnnxSpeakerEmbeddingExtractorIsReady",
    "SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding",
    "SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding",
];

fn preamble(version: &str, sha256: &str) -> String {
    format!(
        r#"// SPDX-License-Identifier: Apache-2.0
//! Raw FFI bindings for the sherpa-onnx C API, **vendored bindgen output** (do not edit by hand).
//!
//! - Source: `sherpa-onnx/c-api/c-api.h` of sherpa-onnx **v{version}** (tag `v{version}`, header sha256
//!   `{sha256}`).
//!   Copyright (c) 2023 Xiaomi Corporation; Apache-2.0 (https://github.com/k2-fsa/sherpa-onnx).
//! - Generator: `scripts/runtime-packs/sherpa/gen-bindings.ps1` (rust-bindgen 0.72.1, libclang,
//!   target `x86_64-pc-windows-msvc`), dynamic-loading mode (`dynamic_library_name("SherpaOnnxCApi")`,
//!   `dynamic_link_require_all(true)`), comments off, `derive_default` (zero-initialised configs, the
//!   C API's `memset(&config, 0, ...)` contract), allow-listed to the functions `super` wraps. The
//!   generated `SherpaOnnxCApi::new` (default DLL search) is removed: Sigil only loads by full path.
//!   Regenerate whenever the pinned version changes, and bump `super::lib::EXPECTED_VERSION` with it:
//!   the runtime refuses any other sherpa-onnx version because the config structs grow between
//!   releases.
//! - Every `const _: () = {{ ... }}` block below is a compile-time check that the Rust layout of a
//!   struct (size, alignment, every field offset) equals what clang computed from the header.
//!
//! Nothing here is load-time linked: [`SherpaOnnxCApi::from_library`] resolves every symbol from a
//! `libloading::Library` that `super::lib` loaded by full path from the runtime pack.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    clippy::all,
    clippy::pedantic,
    clippy::missing_safety_doc
)]

"#
    )
}

/// Remove the generated `pub unsafe fn new<P>(path: P)` (it loads through the default DLL search
/// order, which Sigil never uses, and does not compile against libloading 0.9's `AsFilename`).
fn strip_new(code: &str) -> String {
    let start = code
        .find("    pub unsafe fn new<P>(path: P)")
        .expect("generated SherpaOnnxCApi::new not found; bindgen output changed");
    let end_marker = "        Self::from_library(library)\n    }\n";
    let end = code[start..].find(end_marker).expect("end of SherpaOnnxCApi::new not found") + start + end_marker.len();
    format!("{}{}", &code[..start], &code[end..])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: sigil-sherpa-bindgen <c-api.h> <out.rs> <version> <header-sha256>");
        std::process::exit(2);
    }
    let (header, out, version, sha256) = (&args[1], &args[2], &args[3], &args[4]);
    let mut builder = bindgen::Builder::default()
        .header(header)
        .clang_arg("--target=x86_64-pc-windows-msvc")
        .dynamic_library_name("SherpaOnnxCApi")
        .dynamic_link_require_all(true)
        .generate_comments(false)
        .layout_tests(true)
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .prepend_enum_name(false)
        .merge_extern_blocks(true)
        .sort_semantically(true);
    for f in FUNCTIONS {
        builder = builder.allowlist_function(format!("^{f}$"));
    }
    let bindings = builder.generate().expect("bindgen failed");
    // `write` applies the formatter (prettyplease); `to_string` does not.
    let mut formatted = Vec::new();
    bindings.write(Box::new(&mut formatted)).expect("cannot format the bindings");
    let code = strip_new(&String::from_utf8(formatted).expect("bindings are UTF-8").replace("\r\n", "\n"));
    for f in FUNCTIONS {
        assert!(code.contains(&format!("pub {f}:")), "{f} is missing from the generated API struct");
    }
    std::fs::write(out, format!("{}{}", preamble(version, sha256), code)).expect("cannot write the bindings");
    eprintln!("wrote {out} ({} functions)", FUNCTIONS.len());
}
