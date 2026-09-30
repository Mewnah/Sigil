//! Helpers for the ignored real-model tests of the sherpa engines (`speech::engines::sherpa_*`).
//!
//! - Models come from the model store of `SIGIL_APP_DATA_DIR` (catalog ids of [`super::catalog`]
//!   installed with `Sigil.exe --headless fetch <id>` into that folder), so the tests run exactly
//!   what users install. A model that is not installed skips its test with a message.
//! - Clips: `SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips/<name>` (the dev cache keeps copies of the
//!   packages' `test_wavs`, see its README), else the `test_wavs/` folder of any installed sherpa
//!   catalog model. They are converted to 16 kHz mono with the pipeline's own resampler.
//!
//! Run (one test thread: the tests share one native runtime):
//! `SIGIL_APP_DATA_DIR=%LOCALAPPDATA%/sigil-dev-cache/appdata-p2b SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%/sigil-dev-cache
//! cargo test --manifest-path src-tauri/Cargo.toml sherpa -- --include-ignored --test-threads=1`

use std::path::PathBuf;

use crate::services::speech_core::resample::StreamResampler;

/// The install folder of catalog model `id`, or `None` (with a skip message).
pub(crate) fn installed(id: &str) -> Option<PathBuf> {
    let p = crate::services::model_store::installed_path(id);
    if p.is_none() {
        eprintln!("{id} is not installed in the model store of SIGIL_APP_DATA_DIR; skipped");
    }
    p
}

/// Whether the sherpa runtime pack is installed (skip message otherwise).
pub(crate) fn runtime_installed() -> bool {
    let ok = super::lib::runtime_dir().is_some() || super::SherpaRuntime::loaded().is_some();
    if !ok {
        eprintln!("{} is not installed; skipped", super::RUNTIME_ID);
    }
    ok
}

/// Path of test clip `name` (see the module docs).
pub(crate) fn clip_path(name: &str) -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) {
        let p = root.join("sherpa-onnx").join("clips").join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let found = super::catalog::all()
        .iter()
        .filter_map(|m| crate::services::model_store::installed_path(m.id))
        .map(|d| d.join("test_wavs").join(name))
        .find(|p| p.is_file());
    if found.is_none() {
        eprintln!("test clip {name} not found (SIGIL_TEST_MODELS_DIR/sherpa-onnx/clips); skipped");
    }
    found
}

/// Test clip `name` as 16 kHz mono.
pub(crate) fn clip_16k(name: &str) -> Option<Vec<f32>> {
    let path = clip_path(name)?;
    let (mono, rate) = super::probe::read_wav(&path).expect("readable test clip");
    let mut r = StreamResampler::new(rate, 1).expect("resampler");
    let mut out = Vec::new();
    r.push_interleaved(&mono, &mut out).expect("resample");
    r.flush(&mut out).expect("flush");
    Some(out)
}

/// Serializes the tests that load native DLLs (shared with `speech_rt::sherpa`'s own tests).
pub(crate) fn dll_lock() -> std::sync::MutexGuard<'static, ()> {
    super::tests::dll_lock()
}

/// Lower-case text without punctuation or spaces, for loose transcript checks across scripts.
pub(crate) fn squash(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
