//! so-vits-svc 4.0 / 4.1 bring-your-own ONNX voices (P5-D).
//!
//! A so-vits-svc voice is the user's ONNX export (so-vits-svc's `onnx_export.py`, the MoeVS format:
//! inputs `c`, `f0`, `mel2ph`, `uv`, `noise`, `sid`, plus `vol` for 4.1 `vol_embedding` voices; the
//! accepted contract is [`graph`]'s) with its training `config.json` and / or the MoeVS JSON
//! ([`config`]). It runs behind the shared harness like a template voice ([`model`]): fixed blocks,
//! the shared content encoder (so-vits-svc 4.1's `vec768l12` and 4.0's `vec256l9` are RVC's
//! ContentVec 768 / 256), the pitch bank and its F0 cache, time-anchored flow noise, the latency
//! budget, the governor (light F0, retrieval) and the post stage. Adding one ([`slot`]) copies the
//! export into the slot, checks its graph, measures its hop on a probe and writes the manifest; an
//! ONNX file is not a pickle, so this runs in process (not in the isolated import worker).
//!
//! The family's options ([`SovitsOptions`], `manifest.family`): `cluster_ratio`, the
//! feature-retrieval mix with the voice's FAISS `.index` (so-vits-svc 4.1's feature retrieval is
//! RVC's k = 8 / `1/d²` search). kmeans cluster models (`kmeans_*.pt`) are pickles and are not read.
//!
//! Hot controls: pitch, formant and rms_mix (the post stage), the speaker (an id, or a one-hot mix
//! for `CharaMix` exports), `noise_scale` (relative: the default 0.667 runs so-vits-svc's 0.4) and
//! `silence_db` (the silence skip and gate).
//!
//! # Frozen by P5-S0 (signatures; see `families` for the contract)
//! [`available`], [`info`], [`create`], [`check_slot`], [`normalize_options`], [`create_slot`],
//! [`run_import_job`], [`SovitsOptions`] (the `manifest.family` object; TS `VcSovitsOptions`).
//!
//! Also P5-D (the audit of w-okada / wok000 files, `vc::rvc`): a so-vits-svc export given as an RVC
//! generator, and the pack's content encoders and pitch models given as a so-vits-svc export, are
//! refused by what they are.

pub mod config;
pub mod graph;
pub mod model;
pub mod slot;

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{CreatedSlot, FamilyEngine, FamilyInfo, FamilyRequest, FamilySlotContext, FamilySlotRequest, CREATION_IMPORT};
use crate::services::neural_audio::vc::assets;
use crate::services::neural_audio::vc::engine::content::EmbedderKind;
use crate::services::neural_audio::vc::import::slot::safe_file_name;
use crate::services::neural_audio::vc::import::ImportJob;
use crate::services::neural_audio::vc::manifest::RvcManifest;

/// The family's options (`manifest.family`, snake_case).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SovitsOptions {
    /// Feature-retrieval (cluster / index) mix 0..=1 when the voice ships one (default 0 = off).
    pub cluster_ratio: f32,
}

/// Whether this build runs so-vits-svc voices.
pub fn available() -> bool {
    true
}

pub fn info() -> FamilyInfo {
    FamilyInfo {
        family: "sovits",
        format: super::SOVITS_FORMAT,
        // Verified through ONNX Runtime and the harness with stand-in exports of the exact
        // onnx_export.py contract; labelled experimental until a real so-vits-svc 4.1 export has
        // been compared against so-vits-svc's own inference (the manual check in the P5 report).
        experimental: true,
        zero_shot: false,
        singing: true,
        execution_providers: &["cuda", "dml", "cpu"],
        needs_pitch: true,
        needs_content_encoder: true,
        creation: CREATION_IMPORT,
        import_extensions: &[".onnx", ".zip"],
        controls: &["pitch", "formant", "rms_mix", "speaker", "noise_scale", "silence_db"],
        requires: &[assets::EMBEDDER_CONTENTVEC],
    }
}

/// Load a so-vits-svc voice.
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    model::SovitsModel::load(request)
}

/// Cheap checks of one slot: the export, the record and the content encoder the voice names.
pub fn check_slot(slot_dir: &Path, manifest: &RvcManifest) -> Result<(), String> {
    let g = manifest.generator.trim();
    if !safe_file_name(g) || !slot_dir.join(g).is_file() {
        return Err(format!("the so-vits-svc export {g:?} is missing; add the voice again"));
    }
    config::SovitsRecord::load(slot_dir)?;
    let kind = EmbedderKind::from_manifest(manifest.embedder.as_deref(), manifest.emb_channels)?;
    if !matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) {
        return Err(format!("so-vits-svc voices run on ContentVec; this one names {}", kind.as_str()));
    }
    if !(8_000..=96_000).contains(&manifest.sample_rate) {
        return Err(format!("the voice's sample rate {} Hz is not supported", manifest.sample_rate));
    }
    Ok(())
}

/// Validate and normalize [`SovitsOptions`].
pub fn normalize_options(options: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut o: SovitsOptions = serde_json::from_value(options.clone()).map_err(|e| format!("invalid so-vits-svc options: {e}"))?;
    o.cluster_ratio = if o.cluster_ratio.is_finite() { o.cluster_ratio.clamp(0.0, 1.0) } else { 0.0 };
    serde_json::to_value(o).map_err(|e| e.to_string())
}

/// Copy the user's export into the new slot and write its manifest ([`slot`]).
pub fn create_slot(request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    slot::create_slot(request, ctx)
}

/// so-vits-svc ONNX exports are not parsed by the isolated worker (they are not pickles).
pub fn run_import_job(_job: &ImportJob) -> Result<String, String> {
    Err("job: so-vits-svc voices are imported in process".into())
}

#[cfg(test)]
mod tests;
