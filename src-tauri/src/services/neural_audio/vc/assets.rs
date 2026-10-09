//! Model-store ids and file layout of the voice-changer base assets (P3-S0, frozen).
//!
//! P3-A produces these entries (catalog fragments `vc-bases.json`, `vc-embedders.json`,
//! `vc-denoise.json`, `bench-models.json`; Sigil-exported files stay `unpublished` and are
//! installed by local import / `SIGIL_LOCAL_ASSETS`), and every consumer finds the files the same
//! way: [`model_file`]`(id, variant)` = the entry's install folder
//! (`model_store::installed_path`) joined with [`file_name`]`(variant)`:
//! - `model.onnx` for a single-variant asset (`variant` = `None`);
//! - `model.<variant>.onnx` for precision variants (`fp32`, `fp16`, `int8`).
//!
//! Extra files some entries carry next to the model (P3-A; read by their consumers):
//! - every `vc/embedder-*` entry: `embedder.json` (the export contract: frame rate, channels,
//!   `parity` per variant and EP class, and the `recommended` variant per device class);
//! - `vc/denoise-dfn3`: the DeepFilterNet3 archive's `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` and
//!   `config.ini` (three graphs, no `model.onnx`);
//! - `bench/dnsmos-p835`: `p808.onnx` next to `model.onnx` (the P.808 overall score);
//! - `vc/voice-default`: `default.pth` and its retrieval index
//!   `added_IVF511_Flat_nprobe_1_default_v2.index` (adopted by the importer as the sibling index).
//!
//! Resolution order for the engine's base models: an explicit path from the params, then the
//! model-store entry, then the legacy `app_data/voice-changer/base/` candidates (`bases.rs`), so
//! existing setups keep working with no download.
//!
//! # Frozen (change only through INTEGRATE; P3-A may add ids, never rename)
//! The id constants below, [`file_name`], [`model_file`], [`embedder_id`].

use std::path::PathBuf;

// ---- content encoders (P3-C1's pool; names are `engine::content::EmbedderKind::as_str`) ----
pub const EMBEDDER_CONTENTVEC: &str = "vc/embedder-contentvec";
pub const EMBEDDER_CONTENTVEC_256: &str = "vc/embedder-contentvec-256";
pub const EMBEDDER_SPIN_V2: &str = "vc/embedder-spin-v2";
/// Applio's first SPIN checkpoint (a different network from SPIN-v2; P3-A export, added by P3-I0).
pub const EMBEDDER_SPIN: &str = "vc/embedder-spin";
pub const EMBEDDER_HUBERT_JA: &str = "vc/embedder-hubert-ja";
pub const EMBEDDER_HUBERT_KO: &str = "vc/embedder-hubert-ko";
pub const EMBEDDER_HUBERT_ZH: &str = "vc/embedder-hubert-zh";

// ---- pitch bank (P3-D) ----
pub const PITCH_RMVPE: &str = "vc/pitch-rmvpe";
pub const PITCH_FCPE: &str = "vc/pitch-fcpe";
pub const PITCH_SWIFTF0: &str = "vc/pitch-swiftf0";
pub const PITCH_CREPE_TINY: &str = "vc/pitch-crepe-tiny";
pub const PITCH_PESTO: &str = "vc/pitch-pesto";

// ---- denoisers (P3-E) ----
pub const DENOISE_GTCRN: &str = "vc/denoise-gtcrn";
pub const DENOISE_DFN3: &str = "vc/denoise-dfn3";
pub const DENOISE_DPDFNET: &str = "vc/denoise-dpdfnet";

// ---- voices ----
/// The MIT default voice (`PhoenixStormJr/RVC-V2-default-voice` `default.pth`), auto-imported
/// into a slot on first run (P3-B). Its entry holds the `.pth` as `default.pth`.
pub const VOICE_DEFAULT: &str = "vc/voice-default";

// ---- benchmark-only models (kind `bench`; never needed by normal users, P3-F) ----
/// CAM++ as sherpa-onnx exported it: its segment poolings are `AveragePool(ceil_mode)`, which ONNX
/// Runtime divides by the full kernel on the last, partial segment, so it is not upstream's function
/// (ADR 0006). The legacy speaker scorer (`scoring.speaker_model: legacy`), kept for one release.
pub const BENCH_CAMPPLUS: &str = "bench/campplus";
/// The same CAM++ checkpoint re-exported with exact segment pooling (P3B-B,
/// `scripts/vc-templates/export_bench.py`; >= 100 dB SNR against upstream PyTorch): the speaker
/// scorer of the fidelity scorer v2 (ADR 0009). Same I/O contract as [`BENCH_CAMPPLUS`].
pub const BENCH_CAMPPLUS_EXACT: &str = "bench/campplus-exact";
pub const BENCH_DNSMOS: &str = "bench/dnsmos-p835";
/// The GPU load generator of the contention runs (a small deterministic convolution workload, input
/// `x` `[1, 32, H, W]`; P3-A export, added by P3-I0). The benchmark's load simulator builds the same
/// graph in memory (`bench::load_sim`); the id names the store copy for external tools.
#[allow(dead_code)]
pub const BENCH_GPU_LOAD: &str = "bench/gpu-load";

// ---- runtimes (P3-C2) ----
pub const RUNTIME_TENSORRT_RTX: &str = "runtime/tensorrt-rtx";

/// File of an asset inside its entry folder: `model.onnx`, or `model.<variant>.onnx`.
pub fn file_name(variant: Option<&str>) -> String {
    match variant.map(str::trim).filter(|v| !v.is_empty()) {
        Some(v) => format!("model.{v}.onnx"),
        None => "model.onnx".to_string(),
    }
}

/// The installed file of `id` (and `variant`), when the entry is installed and the file exists.
pub fn model_file(id: &str, variant: Option<&str>) -> Option<PathBuf> {
    let path = crate::services::model_store::installed_path(id)?.join(file_name(variant));
    path.is_file().then_some(path)
}

/// Catalog id of an embedder by its manifest name (`contentvec`, `spin-v2`, ...).
pub fn embedder_id(name: &str) -> Option<&'static str> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "contentvec" => EMBEDDER_CONTENTVEC,
        "contentvec-256" => EMBEDDER_CONTENTVEC_256,
        "spin-v2" => EMBEDDER_SPIN_V2,
        "spin" => EMBEDDER_SPIN,
        "hubert-ja" => EMBEDDER_HUBERT_JA,
        "hubert-ko" => EMBEDDER_HUBERT_KO,
        "hubert-zh" => EMBEDDER_HUBERT_ZH,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_and_embedder_ids() {
        assert_eq!(file_name(None), "model.onnx");
        assert_eq!(file_name(Some("fp16")), "model.fp16.onnx");
        assert_eq!(file_name(Some(" ")), "model.onnx");
        assert_eq!(embedder_id("SPIN-v2"), Some(EMBEDDER_SPIN_V2));
        assert_eq!(embedder_id("spin"), Some(EMBEDDER_SPIN));
        assert_eq!(embedder_id("wavlm"), None);
    }
}
