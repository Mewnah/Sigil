// SPDX-License-Identifier: Apache-2.0
// Vendored from kittentts 0.3.0 (https://github.com/eugenehp/kittentts-rs), the Rust port of
// KittenTTS (https://github.com/KittenML/KittenTTS). Copyright 2026 KittenML and the kittentts-rs
// authors (Eugene Hauptmann). Licensed under the Apache License, Version 2.0; see LICENSE in this
// directory. Modified for Sigil; the changes are listed in NOTICE in this directory.

//! Model-folder and Hugging Face Hub loading for KittenTTS repos.
//!
//! Replaces upstream's `download.rs`, which used the `hf-hub` crate (blocking `ureq`, a cache under
//! `~/.cache/huggingface/hub`). [`ModelConfig`] is upstream's `config.json` schema, unchanged.
//!
//! [`load_from_hub`] serves the legacy `neuralOnnx` "Hugging Face repo" setting (the catalog Kitten
//! models are official sherpa-onnx packages, `local_tts::catalog`):
//! 1. A repo already in the legacy hf-hub cache (where kittentts 0.3 put it) is used in place, so
//!    existing installs never download again ([`legacy_hf_snapshot`]).
//! 2. Otherwise `config.json`, then the ONNX model, then the voices NPZ are fetched one file at a
//!    time from `https://huggingface.co/<repo>/resolve/main/<file>` into
//!    `<app_data>/models/kitten/<org>--<name>/` ([`repo_dir`]) through the model store
//!    (`Store::fetch_unpinned`: the store's one-download-at-a-time slot, HTTP Range resume, stall
//!    timeout and HTML-page refusal). Each file is sanity-checked (no HTML error pages, not
//!    truncated, NPZ is a ZIP) before it is moved into place.
//!
//! [`load_sherpa_kitten_package`] runs an official sherpa-onnx Kitten package (`model.onnx`,
//! `voices.bin`) on the vendored engine: the fallback when the sherpa runtime cannot load.
//!
//! For Hub repos the file names come from a downloaded `config.json`, so they must be single plain
//! path components ([`ModelConfig::check_plain_names`]) before anything is downloaded or a cached
//! snapshot is used. Local speech-package folders keep upstream's behaviour (any relative name).

use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::model::{KittenTtsOnnx, VoiceMatrix};
use crate::services::diagnostics::app_data_dir;
use crate::services::model_store::{global_store, CancelToken, DownloadError, ExtractLimits};

/// Hugging Face base URL for `resolve` downloads.
pub const HF_BASE_URL: &str = "https://huggingface.co";
/// Organisation assumed for bare repo names (`kitten-tts-mini-0.8` → `KittenML/kitten-tts-mini-0.8`).
pub const DEFAULT_ORG: &str = "KittenML";
/// Folder under the app-data directory holding Hub downloads (`<org>--<name>/` per repo).
pub const MODELS_SUBDIR: &str = "models/kitten";
/// Largest `config.json` accepted.
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
/// Smallest ONNX model / voices file accepted (an error page or an LFS pointer is smaller).
const MIN_BLOB_BYTES: u64 = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// config.json schema (upstream)
// ─────────────────────────────────────────────────────────────────────────────

/// Deserialised `config.json` from a KittenTTS HuggingFace repository.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Must be `"ONNX1"` or `"ONNX2"`.
    #[serde(rename = "type")]
    pub model_type: String,

    /// Filename of the ONNX model inside the repo (e.g. `"model.onnx"`).
    pub model_file: String,

    /// Filename of the voices NPZ file inside the repo (e.g. `"voices.npz"`).
    pub voices: String,

    /// Optional per-voice speed multipliers.
    #[serde(default)]
    pub speed_priors: HashMap<String, f32>,

    /// Optional friendly-name → NPZ-key aliases for voices.
    #[serde(default)]
    pub voice_aliases: HashMap<String, String>,
}

impl ModelConfig {
    /// Parse `config.json` bytes and check the model type. File names are not restricted here, so
    /// local speech packages keep loading as before (e.g. a model in a subfolder).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let config: ModelConfig = serde_json::from_slice(bytes).context("Failed to parse config.json")?;
        if !matches!(config.model_type.as_str(), "ONNX1" | "ONNX2") {
            bail!("Unsupported model type '{}' in config.json (expected ONNX1 or ONNX2)", config.model_type);
        }
        Ok(config)
    }

    /// Both file names as single plain path components. Required before a Hub download, where the
    /// names become URL paths and file names under the app-data folder.
    pub fn check_plain_names(&self) -> Result<()> {
        validate_file_name(&self.model_file).context("config.json model_file")?;
        validate_file_name(&self.voices).context("config.json voices")?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Names and paths
// ─────────────────────────────────────────────────────────────────────────────

fn valid_repo_part(part: &str) -> bool {
    !part.is_empty()
        && part.len() <= 96
        && part != "."
        && part != ".."
        && part.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !part.starts_with('.')
}

/// `org/name` for a Hub repo id; a bare name gets the [`DEFAULT_ORG`] prefix. Only ASCII letters,
/// digits, `-`, `_` and `.` are allowed in each part (the id becomes a URL path and a folder name).
pub fn normalize_repo_id(repo_id: &str) -> Result<String> {
    let id = repo_id.trim();
    let (org, name) = match id.split_once('/') {
        Some((org, name)) => (org, name),
        None => (DEFAULT_ORG, id),
    };
    if !valid_repo_part(org) || !valid_repo_part(name) {
        bail!("Invalid Hugging Face repo id '{id}' (expected 'org/name' with letters, digits, '-', '_' or '.')");
    }
    Ok(format!("{org}/{name}"))
}

/// A file name from `config.json` as a single plain path component: no separators, drive or
/// stream syntax, `..`, control characters or leading dot.
pub fn validate_file_name(name: &str) -> Result<&str> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && !name.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control());
    if !ok {
        bail!("Invalid model file name {name:?} (must be a plain file name)");
    }
    Ok(name)
}

/// `<app_data>/models/kitten/<org>--<name>` for a normalized repo id.
pub fn repo_dir(app_data: &Path, repo: &str) -> PathBuf {
    app_data.join(MODELS_SUBDIR).join(repo.replace('/', "--"))
}

/// The hf-hub cache root: `$HF_HOME/hub`, else `~/.cache/huggingface/hub` (what kittentts 0.3 used).
fn hf_cache_root() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(home).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .filter(|v| !v.is_empty())?;
    Some(PathBuf::from(home).join(".cache").join("huggingface").join("hub"))
}

/// The snapshot folder of `repo` in an hf-hub cache rooted at `cache_root`
/// (`models--<org>--<name>/snapshots/<refs/main>`), when `refs/main` names one that exists.
pub fn hf_snapshot_in(cache_root: &Path, repo: &str) -> Option<PathBuf> {
    let base = cache_root.join(format!("models--{}", repo.replace('/', "--")));
    let rev = fs::read_to_string(base.join("refs").join("main")).ok()?;
    let rev = rev.trim();
    if rev.is_empty() || !rev.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let snap = base.join("snapshots").join(rev);
    snap.is_dir().then_some(snap)
}

/// [`hf_snapshot_in`] for the default hf-hub cache.
pub fn legacy_hf_snapshot(repo: &str) -> Option<PathBuf> {
    hf_snapshot_in(&hf_cache_root()?, repo)
}

/// Whether `dir` has a readable `config.json` with plain file names and both files it names.
pub fn is_complete(dir: &Path) -> bool {
    let Ok(bytes) = fs::read(dir.join("config.json")) else {
        return false;
    };
    match ModelConfig::parse(&bytes).and_then(|c| c.check_plain_names().map(|()| c)) {
        Ok(cfg) => dir.join(&cfg.model_file).is_file() && dir.join(&cfg.voices).is_file(),
        Err(_) => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading
// ─────────────────────────────────────────────────────────────────────────────

/// Load a model folder: `config.json` plus the ONNX model and voices NPZ it names.
pub fn load_from_dir(dir: &Path) -> Result<KittenTtsOnnx> {
    let config_path = dir.join("config.json");
    let bytes = fs::read(&config_path).with_context(|| format!("Cannot read {}", config_path.display()))?;
    let config = ModelConfig::parse(&bytes).with_context(|| format!("Invalid config.json in {}", dir.display()))?;
    KittenTtsOnnx::load(
        &dir.join(&config.model_file),
        &dir.join(&config.voices),
        config.speed_priors,
        config.voice_aliases,
    )
}

/// Load a Hub repo (see the module docs): the legacy hf-hub cache if it already has the repo, else
/// the app-data copy, downloading the missing files first. Blocking; call it off the async runtime.
pub fn load_from_hub(repo_id: &str) -> Result<KittenTtsOnnx> {
    let repo = normalize_repo_id(repo_id)?;
    if let Some(snapshot) = legacy_hf_snapshot(&repo).filter(|d| is_complete(d)) {
        tracing::info!(target: "sigil::tts", "Kitten TTS: using {repo} from the Hugging Face cache ({})", snapshot.display());
        return load_from_dir(&snapshot);
    }
    let app_data = app_data_dir().context("The app data folder is unknown")?;
    let dir = repo_dir(&app_data, &repo);
    ensure_repo_files(&repo, &dir)?;
    load_from_dir(&dir)
}

/// Download whichever of `config.json`, the model and the voices file `dir` still lacks, one file
/// at a time.
fn ensure_repo_files(repo: &str, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("Cannot create {}", dir.display()))?;
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        download_file(repo, "config.json", dir, FileKind::Config)?;
    }
    let bytes = fs::read(&config_path).with_context(|| format!("Cannot read {}", config_path.display()))?;
    let config = match ModelConfig::parse(&bytes).and_then(|c| c.check_plain_names().map(|()| c)) {
        Ok(c) => c,
        Err(e) => {
            // A damaged cached config would otherwise fail forever; fetch it again next time.
            let _ = fs::remove_file(&config_path);
            return Err(e.context(format!("Invalid config.json from {repo}")));
        }
    };
    if !dir.join(&config.model_file).is_file() {
        download_file(repo, &config.model_file, dir, FileKind::Onnx)?;
    }
    if !dir.join(&config.voices).is_file() {
        download_file(repo, &config.voices, dir, FileKind::Npz)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileKind {
    Config,
    Onnx,
    Npz,
}

/// Check a finished download before it is renamed into place.
fn check_download(kind: FileKind, len: u64, head: &[u8]) -> Result<()> {
    let first = head.iter().copied().find(|b| !b.is_ascii_whitespace());
    match kind {
        FileKind::Config => {
            if len > MAX_CONFIG_BYTES {
                bail!("config.json is too large ({len} bytes)");
            }
            if first != Some(b'{') {
                bail!("config.json is not a JSON object (got an error page?)");
            }
        }
        FileKind::Onnx | FileKind::Npz => {
            if len < MIN_BLOB_BYTES {
                bail!("the file is too small to be a model ({len} bytes; got an error page?)");
            }
            if first == Some(b'<') {
                bail!("the download returned HTML instead of a model file");
            }
            if kind == FileKind::Npz && !head.starts_with(b"PK\x03\x04") {
                bail!("the voices file is not an NPZ (ZIP) archive");
            }
        }
    }
    Ok(())
}

/// `https://huggingface.co/<repo>/resolve/main/<file>`.
pub fn resolve_url(repo: &str, file: &str) -> String {
    format!("{HF_BASE_URL}/{repo}/resolve/main/{file}")
}

/// Style-vector width of every Kitten model.
pub const KITTEN_STYLE_DIM: usize = 256;

/// Load an official sherpa-onnx Kitten package on the vendored engine: `voices.bin` holds one
/// row-major `[rows, 256]` f32 style matrix per speaker, in the order of `speakers` (the model's
/// `speaker_names`); `aliases` maps friendly names onto them.
pub fn load_sherpa_kitten_package(
    model: &Path,
    voices_bin: &Path,
    speakers: &[String],
    aliases: HashMap<String, String>,
) -> Result<KittenTtsOnnx> {
    let bytes = fs::read(voices_bin).with_context(|| format!("Cannot read {}", voices_bin.display()))?;
    let voices = split_voices_bin(&bytes, speakers, KITTEN_STYLE_DIM)?;
    KittenTtsOnnx::load_with_voices(model, voices, HashMap::new(), aliases)
}

/// Split a sherpa `voices.bin` into one `(name, rows, dim, data)` matrix per speaker (`dim` = the
/// style width, [`KITTEN_STYLE_DIM`] for Kitten).
pub fn split_voices_bin(bytes: &[u8], speakers: &[String], dim: usize) -> Result<Vec<VoiceMatrix>> {
    if speakers.is_empty() {
        bail!("the voice list is empty");
    }
    if bytes.len() % 4 != 0 {
        bail!("voices.bin is not a whole number of f32 values ({} bytes)", bytes.len());
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let per_voice = floats.len() / speakers.len();
    if dim == 0 || per_voice == 0 || floats.len() % speakers.len() != 0 || per_voice % dim != 0 {
        bail!(
            "voices.bin has {} values, which is not {} voices of N x {dim}",
            floats.len(),
            speakers.len()
        );
    }
    let rows = per_voice / dim;
    Ok(speakers
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), rows, dim, floats[i * per_voice..(i + 1) * per_voice].to_vec()))
        .collect())
}

/// Read a finished download's length and first bytes for [`check_download`].
fn head_of(path: &Path) -> std::result::Result<(u64, Vec<u8>), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    let mut head = [0u8; 16];
    let n = fs::File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|e| e.to_string())?;
    Ok((len, head[..n].to_vec()))
}

fn check_file(kind: FileKind, path: &Path) -> std::result::Result<(), String> {
    let (len, head) = head_of(path)?;
    check_download(kind, len, &head).map_err(|e| format!("{e:#}"))
}

// `Store::fetch_unpinned` takes plain fn pointers.
fn check_config_file(path: &Path, _limits: &ExtractLimits) -> std::result::Result<(), String> {
    check_file(FileKind::Config, path)
}

fn check_onnx_file(path: &Path, _limits: &ExtractLimits) -> std::result::Result<(), String> {
    check_file(FileKind::Onnx, path)
}

fn check_npz_file(path: &Path, _limits: &ExtractLimits) -> std::result::Result<(), String> {
    check_file(FileKind::Npz, path)
}

fn download_file(repo: &str, file: &str, dir: &Path, kind: FileKind) -> Result<()> {
    let file = validate_file_name(file)?;
    let url = resolve_url(repo, file);
    let dest = dir.join(file);
    let validate: fn(&Path, &ExtractLimits) -> std::result::Result<(), String> = match kind {
        FileKind::Config => check_config_file,
        FileKind::Onnx => check_onnx_file,
        FileKind::Npz => check_npz_file,
    };
    tracing::info!(target: "sigil::tts", "Kitten TTS: downloading {file} from {repo}");
    let store = global_store().map_err(anyhow::Error::msg)?;
    let cancel = CancelToken::default();
    let label = format!("kitten/{repo}");
    match tauri::async_runtime::block_on(store.fetch_unpinned(&label, file, &url, &dest, validate, &cancel)) {
        Ok(_) => Ok(()),
        Err(DownloadError::Cancelled) => bail!("Download of '{file}' from '{repo}' was cancelled"),
        Err(DownloadError::Failed(m)) => bail!("Failed to download '{file}' from '{repo}': {m}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_are_normalized_and_validated() {
        assert_eq!(normalize_repo_id("kitten-tts-mini-0.8").unwrap(), "KittenML/kitten-tts-mini-0.8");
        assert_eq!(normalize_repo_id(" KittenML/kitten-tts-nano-0.8-int8 ").unwrap(), "KittenML/kitten-tts-nano-0.8-int8");
        for bad in ["", "a/b/c", "../x", "org/..", "org/.hidden", "org/na me", "org/name?x=1", "org/näme", "/name"] {
            assert!(normalize_repo_id(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn config_file_names_must_be_plain() {
        for ok in ["model.onnx", "voices.npz", "kitten_tts_mini_v0_8.onnx"] {
            assert!(validate_file_name(ok).is_ok(), "{ok}");
        }
        for bad in ["", "../model.onnx", "sub/model.onnx", "sub\\model.onnx", "C:model.onnx", ".env", "a\u{0}b", "voices.npz:stream"] {
            assert!(validate_file_name(bad).is_err(), "{bad:?} must be rejected");
        }
        let cfg = ModelConfig::parse(br#"{"type":"ONNX2","model_file":"m.onnx","voices":"v.npz","speed_priors":{"Jasper":1.1}}"#).unwrap();
        assert_eq!((cfg.model_file.as_str(), cfg.voices.as_str()), ("m.onnx", "v.npz"));
        assert_eq!(cfg.speed_priors.get("Jasper"), Some(&1.1));
        cfg.check_plain_names().unwrap();
        // A Hub config naming a path is refused before anything is downloaded; a local package may
        // still point into a subfolder, as before.
        let traversal = ModelConfig::parse(br#"{"type":"ONNX2","model_file":"../../evil.dll","voices":"v.npz"}"#).unwrap();
        assert!(traversal.check_plain_names().is_err());
        let nested = ModelConfig::parse(br#"{"type":"ONNX2","model_file":"onnx/model.onnx","voices":"v.npz"}"#).unwrap();
        assert!(nested.check_plain_names().is_err());
        assert!(ModelConfig::parse(br#"{"type":"PT","model_file":"m.onnx","voices":"v.npz"}"#).is_err());
    }

    #[test]
    fn paths_and_urls() {
        let dir = repo_dir(Path::new("C:/data"), "KittenML/kitten-tts-mini-0.8");
        assert!(dir.ends_with("models/kitten/KittenML--kitten-tts-mini-0.8"), "{}", dir.display());
        assert_eq!(
            resolve_url("KittenML/kitten-tts-mini-0.8", "config.json"),
            "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/config.json"
        );
    }

    #[test]
    fn downloads_are_sanity_checked() {
        assert!(check_download(FileKind::Config, 40, b"  {\"type\":").is_ok());
        assert!(check_download(FileKind::Config, 40, b"<!DOCTYPE html>").is_err());
        assert!(check_download(FileKind::Onnx, 10, b"\x08\x07").is_err(), "too small");
        assert!(check_download(FileKind::Onnx, 5000, b"<html>").is_err(), "HTML error page");
        assert!(check_download(FileKind::Onnx, 5000, b"\x08\x07\x12\x07pytorch").is_ok());
        assert!(check_download(FileKind::Npz, 5000, b"PK\x03\x04rest").is_ok());
        assert!(check_download(FileKind::Npz, 5000, b"version https://git-lfs").is_err(), "LFS pointer");
    }

    #[test]
    fn legacy_hf_cache_snapshots_are_found_and_checked() {
        let root = std::env::temp_dir().join(format!("sigil-kitten-hub-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let repo = "KittenML/kitten-tts-mini-0.8";
        assert_eq!(hf_snapshot_in(&root, repo), None, "no cache yet");
        let base = root.join("models--KittenML--kitten-tts-mini-0.8");
        fs::create_dir_all(base.join("refs")).unwrap();
        fs::write(base.join("refs").join("main"), "abc123\n").unwrap();
        assert_eq!(hf_snapshot_in(&root, repo), None, "refs/main names a missing snapshot");
        let snap = base.join("snapshots").join("abc123");
        fs::create_dir_all(&snap).unwrap();
        assert_eq!(hf_snapshot_in(&root, repo), Some(snap.clone()));
        assert!(!is_complete(&snap), "no config.json yet");
        fs::write(snap.join("config.json"), br#"{"type":"ONNX1","model_file":"m.onnx","voices":"v.npz"}"#).unwrap();
        fs::write(snap.join("m.onnx"), b"x").unwrap();
        assert!(!is_complete(&snap), "voices missing");
        fs::write(snap.join("v.npz"), b"x").unwrap();
        assert!(is_complete(&snap));
        fs::write(base.join("refs").join("main"), "../../escape").unwrap();
        assert_eq!(hf_snapshot_in(&root, repo), None, "a revision is a plain hash");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn load_from_dir_reports_missing_config() {
        let missing = std::env::temp_dir().join("sigil-kitten-hub-missing-dir");
        let err = format!("{:#}", load_from_dir(&missing).err().unwrap());
        assert!(err.contains("config.json"), "{err}");
    }
}
