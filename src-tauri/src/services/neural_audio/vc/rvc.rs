//! RVC generator + HuBERT content encoder (multi-graph ONNX): the legacy full-decode path for
//! bring-your-own ONNX voices (manifest format `rvc`; template voices run on the Sigil Voice
//! Engine, `vc::engine`).
//!
//! P3-C1 additions (each off by default or only active for exports that need it, so existing
//! slots behave exactly as before unless the user turns a lever on):
//! - **fp16 I/O**: generators and content encoders exported with fp16 inputs / outputs (w-okada's
//!   half-precision exports) run; values are converted at the boundary.
//! - **Partial decode when the export supports it**: a generator with RVC realtime's `skip_head` /
//!   `return_length` (/ `return_length2`) inputs decodes only the returned frames; every other
//!   export falls back to the full decode.
//! - **Anchored flow noise** ([`RvcModel::set_noise`], opt-in): the `rnd` input gets counter-based
//!   Gaussian noise anchored to the absolute sample position of each frame instead of the legacy
//!   zeros. No benchmark has measured it on a bring-your-own voice, so only an explicit
//!   `anchored_noise: on` enables it here (`engine::presets::byo_lever`; `auto` keeps the zeros).
//!   The legacy feed's frame grid follows the newest sample, which need not sit on the 160-sample
//!   grid (512 frames at 48 kHz add 170 or 171 samples per tick), so a frame between two anchored
//!   frames gets their variance-preserving blend (`cos·n₀ + sin·n₁`): identical noise wherever the
//!   grids of two ticks coincide, and noise that moves continuously with a sub-frame shift instead
//!   of jumping a whole frame.
//! - **Per-slot content encoder**: a manifest naming its own encoder file (`hubert`) or a
//!   non-ContentVec `embedder` uses it instead of the global HuBERT path.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use half::f16;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{DynValue, TensorElementType, TensorRef};

use crate::services::neural_audio::ort_session::{normalize_ep, session_builder_for_vc, VcSessionOptions};

use super::bases::{load_or_infer_manifest, DEFAULT_INFERRED_SAMPLE_RATE};
use super::engine::content::{resolve_encoder_file, EmbedderKind};
use super::engine::noise::{AnchoredNoise, NoiseStream, FRAME_STRIDE};
use super::engine::options::PrecisionMode;
use super::faiss::FaissIndex;
use super::manifest::{RvcManifest, RvcVersion};
use super::traits::{ConversionCtx, PitchExtractor, VoiceConversionModel};

/// Default latent (flow noise) channels of RVC generators.
const LATENT: usize = 192;
const F0_BIN: f32 = 256.0;
const F0_MAX_HZ: f32 = 1100.0;
const F0_MIN_HZ: f32 = 50.0;
/// 10 ms at 16 kHz: the F0 / upsampled-feature grid.
const HOP_16K: usize = 160;
/// HuBERT / ContentVec conv front end: 400-sample receptive field, 320-sample stride.
const HUBERT_RECEPTIVE_FIELD: usize = 400;
const HUBERT_HOP: usize = 320;
/// Fewest 16 kHz samples for which HuBERT yields two feature frames.
pub const MIN_CONTEXT_16K: usize = HUBERT_RECEPTIVE_FIELD + HUBERT_HOP;
/// Frames fed to the generator when probing its output sample rate.
const PROBE_FRAMES: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvcPresetProfile {
    Streamer,
    LowLatency,
    Quality,
    /// Quality shapes on TensorRT-RTX (P3; the accelerator runs on the CUDA rung, and on plain
    /// CUDA until it is available).
    StudioTrt,
}

impl RvcPresetProfile {
    /// Infallible parse of the persisted preset id (unknown ids fall back to Streamer).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "low_latency" => Self::LowLatency,
            "quality" => Self::Quality,
            "studio_trt" => Self::StudioTrt,
            _ => Self::Streamer,
        }
    }

    /// Default EP ladder rung (`cpu` / `cuda`); Studio-TRT's accelerator request is
    /// `voice_changer::params::preset_default_ep`.
    pub fn default_ep(&self) -> &'static str {
        match self {
            Self::Streamer => "cpu",
            Self::LowLatency | Self::Quality | Self::StudioTrt => "cuda",
        }
    }
}

/// Role of one generator input, resolved from its name (or position) at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenRole {
    Phone,
    PhoneLengths,
    Pitch,
    PitchF,
    Speaker,
    Noise,
    /// RVC realtime's partial decode: frames to skip before the returned span.
    SkipHead,
    /// Frames to return.
    ReturnLength,
    /// Decoder output frames (RVC realtime resamples to it; equal to `ReturnLength` here).
    ReturnLength2,
}

impl GenRole {
    /// Float inputs (fp32, or fp16 in half-precision exports); the rest are int64.
    fn is_float(self) -> bool {
        matches!(self, GenRole::Phone | GenRole::PitchF | GenRole::Noise)
    }
}

fn role_for_name(name: &str) -> Option<GenRole> {
    match name.to_ascii_lowercase().as_str() {
        // RVC-WebUI export names / w-okada export names.
        "phone" | "feats" => Some(GenRole::Phone),
        "phone_lengths" | "p_len" | "lengths" => Some(GenRole::PhoneLengths),
        "pitch" => Some(GenRole::Pitch),
        "pitchf" | "nsff0" => Some(GenRole::PitchF),
        "ds" | "sid" | "g" => Some(GenRole::Speaker),
        "rnd" => Some(GenRole::Noise),
        "skip_head" => Some(GenRole::SkipHead),
        "return_length" => Some(GenRole::ReturnLength),
        "return_length2" => Some(GenRole::ReturnLength2),
        _ => None,
    }
}

/// RVC `f0_to_coarse` (256 bins, mel-scale); unvoiced (and NaN) maps to 1 (bins 1..=255). Shared
/// with the Sigil Voice Engine.
pub(crate) fn coarse_pitch(hz: f32) -> i64 {
    if hz.is_nan() || hz <= 0.0 {
        return 1;
    }
    let f0_mel_min = 1127.0 * (1.0 + F0_MIN_HZ / 700.0).ln();
    let f0_mel_max = 1127.0 * (1.0 + F0_MAX_HZ / 700.0).ln();
    let f0_mel = 1127.0 * (1.0 + hz / 700.0).ln();
    let f0_mel = (f0_mel - f0_mel_min) * (F0_BIN - 2.0) / (f0_mel_max - f0_mel_min) + 1.0;
    f0_mel.clamp(1.0, F0_BIN - 1.0).round() as i64
}

/// Flow prior noise fed to a generator's `rnd` input.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LegacyNoise {
    /// Zeros (the proven legacy behaviour: the prior's mean).
    #[default]
    Zero,
    /// `N(0,1) × scale` anchored to absolute frames (see `engine::noise`), RVC's 0.66666 by
    /// default: overlapping ticks get identical noise.
    Anchored { seed: u64, scale: f32 },
}

/// fp16 copy of `data` when `half`, else nothing (the fp32 slice is fed directly).
fn half_copy(data: &[f32], half: bool) -> Vec<f16> {
    if half {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    } else {
        Vec::new()
    }
}

/// A float input tensor: `half_data` (fp16 exports) or `data` (fp32).
fn float_value<'a>(shape: Vec<usize>, data: &'a [f32], half_data: &'a [f16], half: bool, what: &str) -> Result<SessionInputValue<'a>, String> {
    if half {
        Ok(TensorRef::from_array_view((shape, half_data))
            .map_err(|e| format!("{what} tensor: {e}"))?
            .into())
    } else {
        Ok(TensorRef::from_array_view((shape, data))
            .map_err(|e| format!("{what} tensor: {e}"))?
            .into())
    }
}

/// The `rnd` input for `t` frames whose first starts at absolute 16 kHz sample `start_16k` (`[1, C,
/// t]`, channel-major; negative while the stream's first window is still padded): zeros, or
/// anchored noise indexed like the Sigil Voice Engine's flow noise (`channel * FRAME_STRIDE +
/// frame`). A frame starting `r` samples after absolute frame `k` gets `cos(θ)·n[k] + sin(θ)·n[k+1]`
/// with `θ = (r / 160)·π/2`: exactly `n[k]` on the grid, unit variance everywhere, continuous in the
/// offset (module docs).
fn legacy_flow_noise(noise: LegacyNoise, channels: usize, t: usize, start_16k: i64) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * t];
    if let LegacyNoise::Anchored { seed, scale } = noise {
        let gen = AnchoredNoise::new(seed);
        let hop = HOP_16K as i64;
        let first = start_16k.div_euclid(hop) as u64;
        let rem = start_16k.rem_euclid(hop);
        let theta = rem as f64 / hop as f64 * std::f64::consts::FRAC_PI_2;
        let (w0, w1) = (theta.cos() as f32, theta.sin() as f32);
        let mut next = vec![0.0f32; t + 1];
        for (ch, row) in out.chunks_exact_mut(t.max(1)).enumerate() {
            let key = (ch as u64).wrapping_mul(FRAME_STRIDE).wrapping_add(first);
            if rem == 0 {
                gen.fill(NoiseStream::Flow, key, scale, row);
                continue;
            }
            gen.fill(NoiseStream::Flow, key, scale, &mut next);
            for (j, v) in row.iter_mut().enumerate() {
                *v = w0 * next[j] + w1 * next[j + 1];
            }
        }
    }
    out
}

/// An output tensor as `(shape, f32 values)`, converting fp16 outputs.
fn extract_f32(value: &DynValue, what: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok((shape.iter().map(|d| (*d).max(0) as usize).collect(), data.to_vec()));
    }
    let (shape, data) = value
        .try_extract_tensor::<f16>()
        .map_err(|e| format!("{what} out: {e}"))?;
    Ok((shape.iter().map(|d| (*d).max(0) as usize).collect(), data.iter().map(|h| h.to_f32()).collect()))
}

/// Canonical input order by arity, for exports with unrecognised input names.
fn positional_roles(arity: usize) -> Option<&'static [GenRole]> {
    use GenRole::*;
    match arity {
        6 => Some(&[Phone, PhoneLengths, Pitch, PitchF, Speaker, Noise]),
        5 => Some(&[Phone, PhoneLengths, Pitch, PitchF, Speaker]),
        4 => Some(&[Phone, PhoneLengths, Speaker, Noise]),
        3 => Some(&[Phone, PhoneLengths, Speaker]),
        _ => None,
    }
}

struct GeneratorInputs {
    names: Vec<String>,
    roles: Vec<GenRole>,
    /// Per input: the export takes fp16 (float roles only).
    half: Vec<bool>,
    /// Static last dimension of the phone/feats input (= embedding channels), if declared.
    phone_dim: Option<usize>,
    noise_channels: usize,
}

impl GeneratorInputs {
    fn resolve(session: &Session, path: &Path) -> Result<Self, String> {
        let inputs = session.inputs();
        let names: Vec<String> = inputs.iter().map(|i| i.name().to_string()).collect();
        let roles: Vec<GenRole> = match names.iter().map(|n| role_for_name(n)).collect() {
            Some(roles) => roles,
            None => positional_roles(names.len())
                .ok_or_else(|| format!("RVC generator {} has unsupported inputs {:?}", path.display(), names))?
                .to_vec(),
        };
        for (i, role) in roles.iter().enumerate() {
            if roles[..i].contains(role) {
                return Err(format!("RVC generator {} maps two inputs to {:?} ({:?})", path.display(), role, names));
            }
        }
        let has = |r: GenRole| roles.contains(&r);
        if !has(GenRole::Phone) || !has(GenRole::PhoneLengths) {
            return Err(format!("RVC generator {} lacks phone/phone_lengths inputs ({:?})", path.display(), names));
        }
        if has(GenRole::Pitch) != has(GenRole::PitchF) {
            return Err(format!("RVC generator {} has only one of pitch/pitchf ({:?})", path.display(), names));
        }
        if has(GenRole::SkipHead) != has(GenRole::ReturnLength) || (has(GenRole::ReturnLength2) && !has(GenRole::SkipHead)) {
            return Err(format!(
                "RVC generator {} has an incomplete skip_head / return_length set ({:?})",
                path.display(),
                names
            ));
        }
        let mut phone_dim = None;
        let mut noise_channels = LATENT;
        let mut half = Vec::with_capacity(roles.len());
        for (input, role) in inputs.iter().zip(roles.iter()) {
            let actual = input.dtype().tensor_type();
            let ok = if role.is_float() {
                matches!(actual, Some(TensorElementType::Float32) | Some(TensorElementType::Float16))
            } else {
                actual == Some(TensorElementType::Int64)
            };
            if !ok {
                return Err(format!(
                    "RVC generator {} input '{}' is {:?}, expected {}",
                    path.display(),
                    input.name(),
                    actual,
                    if role.is_float() { "Float32 or Float16" } else { "Int64" }
                ));
            }
            half.push(actual == Some(TensorElementType::Float16));
            let dims = input.dtype().tensor_shape().map(|s| s.to_vec());
            match (role, dims.as_deref()) {
                (GenRole::Phone, Some([.., c])) if *c > 0 => phone_dim = Some(*c as usize),
                (GenRole::Noise, Some([_, c, _])) if *c > 0 => noise_channels = *c as usize,
                _ => {}
            }
        }
        Ok(Self {
            names,
            roles,
            half,
            phone_dim,
            noise_channels,
        })
    }

    /// The export decodes only the returned frames (RVC realtime's `skip_head` / `return_length`).
    fn partial_decode(&self) -> bool {
        self.has(GenRole::SkipHead) && self.has(GenRole::ReturnLength)
    }

    fn has(&self, role: GenRole) -> bool {
        self.roles.contains(&role)
    }
}

/// Model facts w-okada's exporter stores as JSON under the ONNX custom metadata key "metadata"
/// (its own "version" field is the exporter format, not RVC v1/v2).
#[derive(Default)]
struct ExportMetadata {
    sample_rate: Option<u32>,
    if_f0: Option<bool>,
    emb_channels: Option<u32>,
    emb_layer: Option<u32>,
    use_final_proj: Option<bool>,
}

fn read_export_metadata(session: &Session) -> Option<ExportMetadata> {
    let raw = session.metadata().ok()?.custom("metadata")?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let num = |key: &str| {
        v.get(key)
            .and_then(|x| x.as_f64())
            .filter(|x| x.is_finite() && *x > 0.0 && *x < f64::from(u32::MAX))
            .map(|x| x.round() as u32)
    };
    let flag = |key: &str| {
        v.get(key)
            .and_then(|x| x.as_bool().or_else(|| x.as_i64().map(|n| n != 0)))
    };
    Some(ExportMetadata {
        sample_rate: num("samplingRate"),
        if_f0: flag("f0"),
        emb_channels: num("embChannels"),
        emb_layer: num("embOutputLayer"),
        use_final_proj: flag("useFinalProj"),
    })
}

pub struct RvcModel {
    hubert: Session,
    /// The content encoder takes fp16 audio.
    hubert_half: bool,
    generator: Session,
    gen_inputs: GeneratorInputs,
    pitch: Box<dyn PitchExtractor>,
    index: Option<FaissIndex>,
    manifest: RvcManifest,
    out_sr: u32,
    /// Whether the generator takes pitch/pitchf (authoritative over the manifest's `if_f0`).
    wants_f0: bool,
    silence_rms: f32,
    silence_skip: bool,
    noise: LegacyNoise,
    warnings: Vec<String>,
}

/// The content encoder of a legacy slot: a file its manifest names (`hubert`, relative to the
/// slot or absolute), else the encoder its `embedder` names when that is not ContentVec (from the
/// model store), else the global HuBERT / ContentVec path. A named file that is missing falls back
/// to the global path with a warning (older manifests carried the field unused).
fn slot_content_encoder(
    manifest: &RvcManifest,
    inferred: bool,
    slot_dir: &Path,
    hubert_path: &str,
    ep_hint: &str,
    warnings: &mut Vec<String>,
) -> Result<PathBuf, String> {
    if !inferred {
        if let Some(name) = manifest.hubert.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let p = Path::new(name);
            let p = if p.is_absolute() { p.to_path_buf() } else { slot_dir.join(p) };
            if p.is_file() {
                return Ok(p);
            }
            warnings.push(format!(
                "manifest.json names the content encoder {name}, which is not in {}; using the HuBERT / ContentVec from the settings.",
                slot_dir.display()
            ));
        } else if let Some(embedder) = manifest.embedder.as_deref().filter(|s| !s.trim().is_empty()) {
            match EmbedderKind::from_manifest(Some(embedder), manifest.emb_channels) {
                Ok(kind) if !matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) => {
                    let gpu = normalize_ep(ep_hint) != "cpu";
                    return resolve_encoder_file(kind, None, Some(slot_dir), gpu, PrecisionMode::Auto).map(|r| r.path);
                }
                Ok(_) => {}
                // A name Sigil does not know (Applio's `custom`): the settings' encoder, as before.
                Err(e) => warnings.push(format!("{e}; using the HuBERT / ContentVec from the settings.")),
            }
        }
    }
    if hubert_path.trim().is_empty() || !Path::new(hubert_path).is_file() {
        return Err(format!("HuBERT / ContentVec missing: {}", hubert_path));
    }
    Ok(PathBuf::from(hubert_path))
}

impl RvcModel {
    pub fn load(
        slot_dir: &Path,
        hubert_path: &str,
        pitch: Box<dyn PitchExtractor>,
        ep_hint: &str,
        opts: &VcSessionOptions,
        silence_skip: bool,
    ) -> Result<Self, String> {
        let (mut manifest, inferred) = load_or_infer_manifest(slot_dir)?;
        let gen_path = manifest.generator_path(slot_dir);
        if !gen_path.is_file() {
            return Err(format!("RVC generator missing: {}", gen_path.display()));
        }
        let mut warnings = Vec::new();
        let hubert_file = slot_content_encoder(&manifest, inferred, slot_dir, hubert_path, ep_hint, &mut warnings)?;

        let hubert = session_builder_for_vc(ep_hint, opts)?
            .component("vc.content")
            .commit_from_file(&hubert_file)
            .map_err(|e| format!("HuBERT load: {}", e))?;
        let hubert_half = hubert
            .inputs()
            .first()
            .is_some_and(|i| i.dtype().tensor_type() == Some(TensorElementType::Float16));
        let generator = session_builder_for_vc(ep_hint, opts)?
            .component("vc")
            .commit_from_file(&gen_path)
            .map_err(|e| format!("RVC generator load: {}", e))?;
        let gen_inputs = GeneratorInputs::resolve(&generator, &gen_path)?;
        let has_pitch = gen_inputs.has(GenRole::Pitch);
        let metadata = read_export_metadata(&generator);

        let mut sample_rate_known = !inferred;
        if inferred {
            // No manifest.json: refine the in-memory defaults from the generator ONNX.
            if let Some(meta) = &metadata {
                if let Some(sr) = meta.sample_rate {
                    manifest.sample_rate = sr;
                    sample_rate_known = true;
                }
                if let Some(c) = meta.emb_channels {
                    manifest.emb_channels = c;
                }
                if let Some(layer) = meta.emb_layer {
                    manifest.emb_layer = layer;
                }
                if let Some(fp) = meta.use_final_proj {
                    manifest.use_final_proj = fp;
                }
                if let Some(meta_f0) = meta.if_f0.filter(|&f0| f0 != has_pitch) {
                    warnings.push(format!(
                        "RVC generator metadata says f0={} but its inputs {} pitch; following the inputs.",
                        meta_f0,
                        if has_pitch { "include" } else { "do not include" }
                    ));
                }
            }
            if let Some(dim) = gen_inputs.phone_dim {
                manifest.emb_channels = dim as u32;
            } else if metadata.as_ref().and_then(|m| m.emb_channels).is_none() {
                warnings.push(format!(
                    "Could not determine the RVC model's feature size; assuming {} (v2). Add a manifest.json with \"version\" / \"emb_channels\" to {} if conversion fails.",
                    manifest.emb_channels,
                    slot_dir.display()
                ));
            }
            // 256-dim content features are RVC v1 (HuBERT layer 9 + final_proj), 768 is v2.
            if manifest.emb_channels == 256 {
                manifest.version = RvcVersion::V1;
                if metadata.as_ref().is_none_or(|m| m.emb_layer.is_none()) {
                    manifest.emb_layer = 9;
                    manifest.use_final_proj = true;
                }
            } else {
                manifest.version = RvcVersion::V2;
            }
            manifest.if_f0 = has_pitch;
        } else {
            // manifest.json wins, except where the graph itself dictates the tensors to feed.
            if manifest.wants_f0() != has_pitch {
                warnings.push(format!(
                    "manifest.json declares {} but the generator {} pitch inputs; following the generator.",
                    if manifest.wants_f0() { "an f0 model" } else { "an f0-less model" },
                    if has_pitch { "has" } else { "has no" }
                ));
            }
            if let Some(dim) = gen_inputs.phone_dim {
                if dim as u32 != manifest.emb_channels {
                    warnings.push(format!(
                        "manifest.json emb_channels={} but the generator expects {}; using {}.",
                        manifest.emb_channels, dim, dim
                    ));
                    manifest.emb_channels = dim as u32;
                }
            }
            if let Some(sr) = metadata.as_ref().and_then(|m| m.sample_rate) {
                if sr != manifest.sample_rate {
                    warnings.push(format!(
                        "manifest.json sample_rate={} differs from the generator metadata ({}); using manifest.json.",
                        manifest.sample_rate, sr
                    ));
                }
            }
        }

        let emb = manifest.emb_channels as usize;
        let index = match manifest.index_path(slot_dir) {
            Some(path) if path.is_file() => match FaissIndex::load(&path) {
                Ok(idx) if idx.dim() == emb => Some(idx),
                Ok(idx) => {
                    warnings.push(format!(
                        "Retrieval index {} has {} dimensions but the model uses {}; index rate has no effect.",
                        path.display(),
                        idx.dim(),
                        emb
                    ));
                    None
                }
                Err(e) => {
                    warnings.push(format!(
                        "Retrieval index {} could not be read ({}); continuing without it (index rate has no effect).",
                        path.display(),
                        e
                    ));
                    None
                }
            },
            Some(path) => {
                warnings.push(format!(
                    "Retrieval index {} listed in manifest.json was not found; index rate has no effect.",
                    path.display()
                ));
                None
            }
            None => None,
        };

        let mut model = Self {
            hubert,
            hubert_half,
            generator,
            gen_inputs,
            pitch,
            index,
            out_sr: manifest.sample_rate,
            wants_f0: has_pitch,
            silence_rms: 0.008,
            silence_skip,
            noise: LegacyNoise::Zero,
            manifest,
            warnings,
        };
        if !sample_rate_known {
            // Measure it: RVC generators emit exactly `hop = sample_rate / 100` samples per frame.
            match model.probe_output_rate() {
                Ok(sr) => {
                    model.out_sr = sr;
                    model.manifest.sample_rate = sr;
                }
                Err(e) => {
                    model.out_sr = DEFAULT_INFERRED_SAMPLE_RATE;
                    model.manifest.sample_rate = DEFAULT_INFERRED_SAMPLE_RATE;
                    model.warnings.push(format!(
                        "Could not determine the RVC model's sample rate ({}); assuming {} Hz. Add a manifest.json with \"sample_rate\" to {} if the voice sounds too fast, slow or off-pitch.",
                        e,
                        DEFAULT_INFERRED_SAMPLE_RATE,
                        slot_dir.display()
                    ));
                }
            }
        }
        Ok(model)
    }

    /// Non-fatal load issues (manifest guessed, index unreadable, graph/manifest mismatch, ...).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The flow prior noise fed to a generator with an `rnd` input (default zeros, the proven
    /// legacy behaviour). The factory sets [`LegacyNoise::Anchored`] when the user turned the
    /// `anchored_noise` lever on for the voice (`engine::presets::byo_lever`: `auto` keeps the
    /// zeros on bring-your-own voices); generators without `rnd` ignore it.
    pub fn set_noise(&mut self, noise: LegacyNoise) {
        self.noise = noise;
    }

    /// Whether the generator decodes only the returned frames (it has RVC realtime's
    /// `skip_head` / `return_length` inputs); otherwise every tick decodes the whole context.
    pub fn partial_decode(&self) -> bool {
        self.gen_inputs.partial_decode()
    }

    /// Run HuBERT, the pitch extractor and the generator once (silence skip bypassed) on a fixed
    /// 200 ms input. The stream harness warms at the live input shape instead
    /// (`VoiceConversionModel::warmup_at`), which is what GPU kernel selection needs; this stays
    /// for callers that convert without the harness (contract C-3), so it may have no caller.
    #[allow(dead_code)]
    pub fn warmup(&mut self) -> Result<(), String> {
        let n = 16_000 / 5; // 200 ms
        let tone: Vec<f32> = (0..n)
            .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 16_000.0).sin())
            .collect();
        let ctx = ConversionCtx {
            mono_16k: &tone,
            new_frames: n,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: self.manifest.speaker_id,
            stream_pos_16k: 0,
        };
        self.convert_inner(&ctx, false).map(|_| ())
    }

    fn probe_output_rate(&mut self) -> Result<u32, String> {
        let emb = self.manifest.emb_channels as usize;
        let phone = vec![0.0f32; PROBE_FRAMES * emb];
        let pitch = vec![1i64; PROBE_FRAMES];
        let pitchf = vec![0.0f32; PROBE_FRAMES];
        // The whole probe is returned (skip 0), so a partial-decode export measures the same rate.
        let audio = self.run_generator(&phone, PROBE_FRAMES, &pitch, &pitchf, self.manifest.speaker_id, 0, PROBE_FRAMES)?;
        let sr = (audio.len() * 100 + PROBE_FRAMES / 2) / PROBE_FRAMES;
        if !(8_000..=96_000).contains(&sr) {
            return Err(format!("{} output samples for {} frames", audio.len(), PROBE_FRAMES));
        }
        Ok(sr as u32)
    }

    /// `round(frames_16k * out_sr / 16000)`: the length both the model and silence paths return.
    fn output_len(&self, frames_16k: usize) -> usize {
        ((frames_16k as u64 * u64::from(self.out_sr) + 8_000) / 16_000) as usize
    }

    /// HuBERT features, time-major `[frames * emb]` (fp16 exports converted at the boundary).
    fn run_hubert(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let n = mono_16k.len();
        let half = half_copy(mono_16k, self.hubert_half);
        let tensor = float_value(vec![1usize, n], mono_16k, &half, self.hubert_half, "HuBERT")?;
        let name = self
            .hubert
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| "HuBERT ONNX declares no inputs".to_string())?;
        let outputs = self
            .hubert
            .run(ort::inputs![name.as_str() => tensor])
            .map_err(|e| format!("HuBERT run: {}", e))?;
        let (dims, data) = extract_f32(&outputs[0], "HuBERT")?;

        // Flatten to [T*C] in time-major order. Accepted shapes: [1,T,C], [T,C], [1,C,T].
        let emb = self.manifest.emb_channels as usize;
        match dims.as_slice() {
            [1, c, t] if *c == emb && *t != emb => {
                let mut out = vec![0.0f32; t * c];
                for ti in 0..*t {
                    for ci in 0..*c {
                        out[ti * c + ci] = data[ci * t + ti];
                    }
                }
                Ok(out)
            }
            [.., c] if *c == emb => Ok(data),
            _ => Err(format!(
                "HuBERT output shape {:?} does not carry the model's {} embedding channels \
                 (v1 models need a 256-dim final_proj ContentVec, v2 a 768-dim one)",
                dims, emb
            )),
        }
    }

    /// Run the generator on `t` frames, feeding each declared input by role. `start_16k` is the
    /// absolute 16 kHz sample the first frame starts at (anchored noise); an export with
    /// `skip_head` / `return_length` decodes only the last `ret` frames, every other export all
    /// `t`.
    #[allow(clippy::too_many_arguments)]
    fn run_generator(
        &mut self,
        phone: &[f32],
        t: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        start_16k: i64,
        ret: usize,
    ) -> Result<Vec<f32>, String> {
        let emb = self.manifest.emb_channels as usize;
        let lengths = [t as i64];
        let speaker = [speaker_id];
        let ret = ret.clamp(1, t.max(1));
        let skip = [(t - ret) as i64];
        let ret_len = [ret as i64];
        let noise_ch = self.gen_inputs.noise_channels;
        let noise = if self.gen_inputs.has(GenRole::Noise) {
            legacy_flow_noise(self.noise, noise_ch, t, start_16k)
        } else {
            Vec::new()
        };
        let tensor_err = |what: &str, e: ort::Error| format!("{what} tensor: {e}");
        let half_role = |r: GenRole| {
            self.gen_inputs
                .roles
                .iter()
                .zip(self.gen_inputs.half.iter())
                .any(|(&x, &h)| x == r && h)
        };
        let h_phone = half_copy(phone, half_role(GenRole::Phone));
        let h_pitchf = half_copy(pitchf, half_role(GenRole::PitchF));
        let h_noise = half_copy(&noise, half_role(GenRole::Noise));
        let mut feed: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(self.gen_inputs.names.len());
        for ((name, role), &half) in self
            .gen_inputs
            .names
            .iter()
            .zip(self.gen_inputs.roles.iter())
            .zip(self.gen_inputs.half.iter())
        {
            let value: SessionInputValue<'_> = match role {
                GenRole::Phone => float_value(vec![1, t, emb], phone, &h_phone, half, "phone")?,
                GenRole::PhoneLengths => TensorRef::from_array_view((vec![1usize], &lengths[..]))
                    .map_err(|e| tensor_err("phone_lengths", e))?
                    .into(),
                GenRole::Pitch => TensorRef::from_array_view((vec![1, t], pitch))
                    .map_err(|e| tensor_err("pitch", e))?
                    .into(),
                GenRole::PitchF => float_value(vec![1, t], pitchf, &h_pitchf, half, "pitchf")?,
                GenRole::Speaker => TensorRef::from_array_view((vec![1usize], &speaker[..]))
                    .map_err(|e| tensor_err("speaker id", e))?
                    .into(),
                GenRole::Noise => float_value(vec![1, noise_ch, t], &noise, &h_noise, half, "rnd")?,
                GenRole::SkipHead => TensorRef::from_array_view((vec![1usize], &skip[..]))
                    .map_err(|e| tensor_err("skip_head", e))?
                    .into(),
                GenRole::ReturnLength | GenRole::ReturnLength2 => TensorRef::from_array_view((vec![1usize], &ret_len[..]))
                    .map_err(|e| tensor_err("return_length", e))?
                    .into(),
            };
            feed.push((name.clone(), value));
        }
        let outputs: SessionOutputs<'_> = self
            .generator
            .run(feed)
            .map_err(|e| format!("RVC generator run: {}", e))?;
        let (_dims, data) = extract_f32(&outputs[0], "RVC audio")?;
        Ok(data)
    }

    /// RVC `f0_to_coarse` (256 bins, mel-scale); unvoiced maps to 1 (bins 1..=255).
    fn f0_to_coarse(hz: f32) -> i64 {
        coarse_pitch(hz)
    }

    fn convert_inner(&mut self, ctx: &ConversionCtx<'_>, allow_silence_skip: bool) -> Result<Vec<f32>, String> {
        let audio = ctx.mono_16k;
        let want = ctx.new_frames.min(audio.len());
        if want == 0 {
            return Ok(Vec::new());
        }
        let out_len = self.output_len(want);

        if allow_silence_skip {
            let new = &audio[audio.len() - want..];
            let energy = new.iter().map(|x| x * x).sum::<f32>() / want as f32;
            if energy.sqrt() < self.silence_rms {
                return Ok(vec![0.0; out_len]);
            }
        }
        if audio.len() < MIN_CONTEXT_16K {
            return Err(format!(
                "RVC needs at least {} samples of 16 kHz context, got {}",
                MIN_CONTEXT_16K,
                audio.len()
            ));
        }

        // Zero-pad the *oldest* end to whole 10 ms hops so the generator output (p_len frames of
        // hop samples) ends exactly at the newest input sample.
        let pad = (HOP_16K - audio.len() % HOP_16K) % HOP_16K;
        let padded: Cow<'_, [f32]> = if pad == 0 {
            Cow::Borrowed(audio)
        } else {
            let mut v = vec![0.0f32; pad];
            v.extend_from_slice(audio);
            Cow::Owned(v)
        };
        let p_len = padded.len() / HOP_16K;
        let emb = self.manifest.emb_channels as usize;

        let mut feats = self.run_hubert(&padded)?;
        let frames_50hz = if emb > 0 { feats.len() / emb } else { 0 };
        if frames_50hz == 0 {
            return Err(format!("HuBERT produced empty features (len={}, emb={})", feats.len(), emb));
        }
        feats.truncate(frames_50hz * emb);

        // RVC `protect`: keep the pre-retrieval features to restore consonants/breaths.
        let protect_on = self.wants_f0 && ctx.protect < 0.5;
        let feats0 = protect_on.then(|| upsample_to_hops(&feats, emb, p_len));
        if let Some(ref index) = self.index {
            if ctx.index_rate > 0.0 {
                // Like RVC realtime (`feats[skip_head // 2:]`): search only the 50 Hz frames that
                // can reach the returned tail, not the whole rolling context, every tick.
                let skip = index_skip_frames(padded.len(), want, frames_50hz);
                index.blend_inplace(&mut feats[skip * emb..], frames_50hz - skip, ctx.index_rate);
            }
        }
        // RVC realtime: duplicate the last 50 Hz frame, upsample 2x onto the 10 ms grid, cut to
        // p_len (so the features cover the whole padded input).
        let mut phone = upsample_to_hops(&feats, emb, p_len);

        let (pitch, pitchf) = if self.wants_f0 {
            let f0 = align_f0(&self.pitch.f0(&padded)?, p_len);
            let pitchf = shift_f0(&f0, ctx.pitch_shift_semitones);
            let pitch: Vec<i64> = pitchf.iter().map(|&hz| Self::f0_to_coarse(hz)).collect();
            (pitch, pitchf)
        } else {
            (Vec::new(), Vec::new())
        };

        if let Some(feats0) = feats0 {
            // pitchff = 1 on voiced frames, `protect` where pitchf < 1 (unvoiced).
            let protect = ctx.protect.max(0.0);
            for ((frame, frame0), &pf) in phone
                .chunks_exact_mut(emb)
                .zip(feats0.chunks_exact(emb))
                .zip(pitchf.iter())
            {
                if pf < 1.0 {
                    for (x, &x0) in frame.iter_mut().zip(frame0.iter()) {
                        *x = *x * protect + x0 * (1.0 - protect);
                    }
                }
            }
        }

        // A partial-decode export decodes the returned frames plus a decoder guard (RVC realtime's
        // graph gives the decoder no left context of its own); every other export decodes all.
        let ret = if self.gen_inputs.partial_decode() {
            (want.div_ceil(HOP_16K) + SKIP_DECODER_GUARD_FRAMES).min(p_len)
        } else {
            p_len
        };
        // The padded window starts `pad` samples before the context (anchored noise follows the
        // true sample offset, which the legacy feed does not keep on the 160-sample grid).
        let start_16k = i64::try_from(ctx.stream_pos_16k).unwrap_or(i64::MAX) - pad as i64;
        let full = self.run_generator(&phone, p_len, &pitch, &pitchf, ctx.speaker_id, start_16k, ret)?;
        let n = out_len.min(full.len());
        Ok(full[full.len() - n..].to_vec())
    }
}

/// Frames decoded before the returned span by an export with `skip_head` / `return_length`: its
/// decoder sees only the frames it returns, so a few extra keep its left edge out of the audio
/// (the measured HiFi-GAN receptive field is 11 frames at 40 kHz, ADR 0004).
const SKIP_DECODER_GUARD_FRAMES: usize = 12;

/// Nearest 2x upsampling of 50 Hz features onto `p_len` 10 ms frames: frame `j` takes 50 Hz frame
/// `min(j / 2, last)`, i.e. RVC's duplicated-last-frame + interpolate(scale 2) + `[:p_len]`.
fn upsample_to_hops(feats: &[f32], emb: usize, p_len: usize) -> Vec<f32> {
    let frames = feats.len() / emb.max(1);
    let mut out = Vec::with_capacity(p_len * emb);
    if frames == 0 {
        out.resize(p_len * emb, 0.0);
        return out;
    }
    for j in 0..p_len {
        let src = (j / 2).min(frames - 1);
        out.extend_from_slice(&feats[src * emb..(src + 1) * emb]);
    }
    out
}

/// First 50 Hz feature frame worth a retrieval search: frames before it cannot reach the returned
/// tail (the last `want` samples of the `padded_len`-sample HuBERT input). A frame is kept when its
/// 400-sample receptive field reaches the tail start, which is at or before the frame the tail's
/// first 10 ms hop is upsampled from. The last frame is always searched: it also feeds every hop
/// past the end of a short HuBERT output.
fn index_skip_frames(padded_len: usize, want: usize, frames_50hz: usize) -> usize {
    let tail_start = padded_len.saturating_sub(want);
    (tail_start.saturating_sub(HUBERT_RECEPTIVE_FIELD) / HUBERT_HOP).min(frames_50hz.saturating_sub(1))
}

/// Pitch-shifted F0 for the generator's `pitchf`; non-finite or non-positive frames become 0.0
/// (unvoiced) so a NaN from a pitch extractor never reaches the NSF source.
fn shift_f0(f0: &[f32], semitones: f32) -> Vec<f32> {
    let shift = (2.0f32).powf(semitones / 12.0);
    f0.iter()
        .map(|&hz| {
            let v = hz * shift;
            if v.is_finite() && v > 0.0 {
                v
            } else {
                0.0
            }
        })
        .collect()
}

/// Fit an F0 contour to `p_len` frames of the hop grid: a longer contour is cut to its first
/// `p_len` frames (RVC `pitch[:, :p_len]`); a shorter one is right-aligned so the newest frames
/// line up with the newest audio (missing oldest frames are unvoiced).
fn align_f0(f0: &[f32], p_len: usize) -> Vec<f32> {
    if f0.len() >= p_len {
        return f0[..p_len].to_vec();
    }
    let mut out = vec![0.0f32; p_len];
    out[p_len - f0.len()..].copy_from_slice(f0);
    out
}

impl VoiceConversionModel for RvcModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let skip = self.silence_skip;
        self.convert_inner(ctx, skip)
    }

    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    fn output_sample_rate(&self) -> u32 {
        self.out_sr
    }

    fn min_context_samples(&self) -> usize {
        MIN_CONTEXT_16K
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        // Silence skip bypassed: HuBERT, the pitch extractor and the generator all run at the
        // live tensor shapes (cuDNN exhaustive search / DirectML compile are per shape).
        self.convert_inner(ctx, false).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unvoiced_coarse_pitch_is_one() {
        assert_eq!(RvcModel::f0_to_coarse(0.0), 1);
        assert_eq!(RvcModel::f0_to_coarse(-5.0), 1);
        assert_eq!(RvcModel::f0_to_coarse(f32::NAN), 1);
        assert_eq!(RvcModel::f0_to_coarse(f32::INFINITY), 255);
        assert_eq!(RvcModel::f0_to_coarse(F0_MIN_HZ), 1);
        assert_eq!(RvcModel::f0_to_coarse(F0_MAX_HZ), 255);
        let mid = RvcModel::f0_to_coarse(220.0);
        assert!((2..255).contains(&mid), "{mid}");
    }

    #[test]
    fn shifted_f0_drops_non_finite_frames() {
        let f0 = [100.0, 0.0, f32::NAN, f32::INFINITY, -3.0, 220.0];
        let up = shift_f0(&f0, 12.0);
        assert_eq!(up, vec![200.0, 0.0, 0.0, 0.0, 0.0, 440.0]);
        assert!(shift_f0(&[f32::MAX], 12.0).iter().all(|&v| v == 0.0), "overflow to inf is unvoiced");
    }

    #[test]
    fn index_search_covers_only_frames_reaching_the_tail() {
        // Whole input returned: every frame is searched.
        assert_eq!(index_skip_frames(1280, 1280, 3), 0);
        // 1280 samples -> 3 HuBERT frames (0..400, 320..720, 640..1040); tail = last 400 samples
        // (880..1280): frame 0 cannot reach it, frame 1 is kept as the conservative margin.
        assert_eq!(index_skip_frames(1280, 400, 3), 1);
        // A tiny tail still searches the last frame (never an empty slice).
        assert_eq!(index_skip_frames(1280, 1, 3), 2);
        assert_eq!(index_skip_frames(1280, 0, 3), 2);
        assert_eq!(index_skip_frames(4000, 10, 1), 0);
        for p_len in (MIN_CONTEXT_16K / HOP_16K)..60 {
            let padded = p_len * HOP_16K;
            let frames = (padded - HUBERT_RECEPTIVE_FIELD) / HUBERT_HOP + 1;
            for want in 1..=padded {
                let skip = index_skip_frames(padded, want, frames);
                assert!(skip < frames, "padded {padded} want {want}");
                // The 50 Hz frame the tail's first 10 ms hop is upsampled from is searched.
                let first_hop = (padded - want) / HOP_16K;
                assert!(skip <= (first_hop / 2).min(frames - 1), "padded {padded} want {want}: skip {skip}");
            }
        }
    }

    #[test]
    fn short_f0_contour_is_right_aligned() {
        assert_eq!(align_f0(&[1.0, 2.0], 4), vec![0.0, 0.0, 1.0, 2.0]);
        assert_eq!(align_f0(&[1.0, 2.0, 3.0, 4.0, 5.0], 4), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn features_upsample_like_rvc_realtime() {
        // 3 frames of dim 2 onto 7 hops: 0,0,1,1,2,2,2 (last frame duplicated).
        let feats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let up = upsample_to_hops(&feats, 2, 7);
        let firsts: Vec<f32> = up.chunks_exact(2).map(|f| f[0]).collect();
        assert_eq!(firsts, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        // HuBERT frame count for a hop-aligned input always covers p_len after duplication.
        for p_len in (MIN_CONTEXT_16K / HOP_16K)..40 {
            let n = p_len * HOP_16K;
            let frames = (n - HUBERT_RECEPTIVE_FIELD) / HUBERT_HOP + 1;
            assert!(2 * (frames + 1) >= p_len, "p_len {p_len}");
        }
    }

    #[test]
    fn generator_roles_resolve_by_name_or_position() {
        assert_eq!(role_for_name("feats"), Some(GenRole::Phone));
        assert_eq!(role_for_name("p_len"), Some(GenRole::PhoneLengths));
        assert_eq!(role_for_name("sid"), Some(GenRole::Speaker));
        assert_eq!(role_for_name("rnd"), Some(GenRole::Noise));
        assert_eq!(role_for_name("skip_head"), Some(GenRole::SkipHead));
        assert_eq!(role_for_name("RETURN_LENGTH"), Some(GenRole::ReturnLength));
        assert_eq!(role_for_name("return_length2"), Some(GenRole::ReturnLength2));
        assert_eq!(role_for_name("x.1"), None);
        assert_eq!(positional_roles(5).map(|r| r.len()), Some(5));
        assert!(positional_roles(2).is_none());
        assert_eq!(coarse_pitch(220.0), RvcModel::f0_to_coarse(220.0));
    }

    /// The legacy `rnd` stays zeros unless anchored noise is chosen; anchored noise is the Sigil
    /// Voice Engine's flow noise for the same absolute frames (overlapping ticks agree on the grid).
    #[test]
    fn legacy_rnd_is_zero_or_anchored_to_absolute_frames() {
        assert!(legacy_flow_noise(LegacyNoise::Zero, 4, 10, 123).iter().all(|&v| v == 0.0));
        let mode = LegacyNoise::Anchored { seed: 9, scale: 0.66666 };
        let a = legacy_flow_noise(mode, 3, 20, 1_000 * 160);
        let b = legacy_flow_noise(mode, 3, 20, 1_005 * 160);
        for ch in 0..3 {
            assert_eq!(&a[ch * 20 + 5..ch * 20 + 20], &b[ch * 20..ch * 20 + 15], "channel {ch}");
        }
        let mut direct = vec![0.0; 20];
        AnchoredNoise::new(9).fill(NoiseStream::Flow, 2 * FRAME_STRIDE + 1_000, 0.66666, &mut direct);
        assert_eq!(&a[40..60], &direct[..]);
    }

    /// Off the 160-sample grid (the legacy feed at 512 frames / 48 kHz advances 170 or 171 samples
    /// per tick) the noise moves continuously with the offset: one sample of shift changes each
    /// frame by about π/320 of its scale, never by a whole frame's jump, and a 160-sample shift is
    /// exactly one frame. The blend keeps the variance (std ≈ scale) at every offset; the first
    /// padded window (negative start) is anchored too.
    #[test]
    fn legacy_rnd_moves_continuously_off_the_grid() {
        let mode = LegacyNoise::Anchored { seed: 5, scale: 0.66666 };
        let (c, t) = (2usize, 400usize);
        let at = |s: i64| legacy_flow_noise(mode, c, t, s);
        let base = 12_345i64 * 160;
        // Walk one frame of offsets sample by sample: no step larger than a small fraction.
        let mut prev = at(base);
        let mut worst = 0.0f32;
        for k in 1..=160 {
            let cur = at(base + k);
            let step = prev
                .iter()
                .zip(&cur)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            worst = worst.max(step);
            prev = cur;
        }
        // |d/dθ (cos θ n0 + sin θ n1)| <= |n0| + |n1|; per sample dθ = π/320; values stay below 5σ.
        assert!(worst < 2.0 * 5.0 * 0.66666 * std::f32::consts::PI / 320.0, "largest per-sample step {worst}");
        // A whole frame of offset is exactly the next anchored frame.
        let a = at(base);
        let b = at(base + 160);
        for ch in 0..c {
            assert_eq!(&a[ch * t + 1..(ch + 1) * t], &b[ch * t..(ch + 1) * t - 1]);
        }
        // The old floor mapping jumped a frame at the wrap: shifting by 159 vs 161 samples now
        // differs by a little (two samples' worth), not by a frame of fresh noise.
        let (x, y) = (at(base + 159), at(base + 161));
        let jump = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(jump < 0.2, "{jump}");
        for s in [base + 1, base + 40, base + 80, base + 159, -37] {
            let v = at(s);
            let var = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / v.len() as f64;
            let std = var.sqrt();
            assert!((std - 0.66666).abs() < 0.06, "offset {s}: std {std}");
        }
    }

    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};
    use crate::services::neural_audio::vc::test_models;
    use crate::services::neural_audio::vc::world::DioPitchExtractor;

    /// `TensorProto.DataType.FLOAT16`.
    const FLOAT16: u64 = 10;

    /// The fake generator of `test_models` (audio = 0.001 × pitchf held for 400 samples), with
    /// fp16 float inputs / output (`half`) and / or RVC realtime's skip inputs (`skip`: the output
    /// is the last `return_length` frames).
    fn fake_generator(half: bool, skip: bool) -> Vec<u8> {
        let ft = if half { FLOAT16 } else { FLOAT };
        let emb = test_models::EMB as u64;
        let mut g = Graph::new();
        g.input("phone", ft, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(emb)])
            .input("phone_lengths", INT64, &[Dim::Fixed(1)])
            .input("pitch", INT64, &[Dim::Fixed(1), Dim::Named("t")])
            .input("pitchf", ft, &[Dim::Fixed(1), Dim::Named("t")])
            .input("ds", INT64, &[Dim::Fixed(1)])
            .input("rnd", ft, &[Dim::Fixed(1), Dim::Fixed(192), Dim::Named("t")]);
        if skip {
            g.input("skip_head", INT64, &[Dim::Fixed(1)]).input("return_length", INT64, &[Dim::Fixed(1)]);
        }
        g.output("audio", ft, &[Dim::Fixed(1), Dim::Fixed(1), Dim::Named("s")])
            .init_i64("axes2", &[1], &[2])
            .init_i64("repeats", &[3], &[1, 1, test_models::HOP as i64])
            .init_i64("flat", &[3], &[1, 1, -1])
            .init_f32("gain", &[1], &[test_models::GAIN])
            .node("Cast", &["pitchf"], &["pf32"], &[Attr::Int("to", FLOAT as i64)])
            .node("Unsqueeze", &["pf32", "axes2"], &["p3"], &[])
            .node("Tile", &["p3", "repeats"], &["tiled"], &[])
            .node("Reshape", &["tiled", "flat"], &["r"], &[])
            .node("Mul", &["r", "gain"], &["full"], &[]);
        let last = if skip {
            g.init_i64("hop", &[1], &[test_models::HOP as i64])
                .init_i64("big", &[1], &[i64::MAX])
                .node("Mul", &["skip_head", "hop"], &["start"], &[])
                .node("Slice", &["full", "start", "big", "axes2"], &["cut"], &[]);
            "cut"
        } else {
            "full"
        };
        g.node("Cast", &[last], &["audio"], &[Attr::Int("to", ft as i64)]);
        g.model_bytes(13)
    }

    fn convert_with_generator(tag: &str, generator: Vec<u8>) -> (Vec<f32>, bool) {
        let (root, slot, hubert) = test_models::write_slot(&format!("rvc-{tag}"));
        for entry in std::fs::read_dir(&slot).unwrap().flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
        std::fs::write(slot.join("gen.onnx"), generator).unwrap();
        let mut model = RvcModel::load(
            &slot,
            hubert.to_str().unwrap(),
            Box::new(DioPitchExtractor::new()),
            "cpu",
            &VcSessionOptions::default(),
            false,
        )
        .unwrap();
        assert_eq!(model.output_sample_rate(), 40_000, "{tag}");
        let partial = model.partial_decode();
        let audio: Vec<f32> = (0..16_000).map(|i| 0.3 * (std::f32::consts::TAU * 150.0 * i as f32 / 16_000.0).sin()).collect();
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: 1_600,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
        };
        let out = model.convert(&ctx).unwrap();
        drop(model);
        let _ = std::fs::remove_dir_all(&root);
        (out, partial)
    }

    /// fp16-I/O exports and exports with RVC realtime's skip inputs run, and give the fp32 full
    /// decode's audio (to fp16 precision).
    #[test]
    fn fp16_io_and_skip_exports_match_the_fp32_full_decode() {
        let (reference, partial) = convert_with_generator("fp32", fake_generator(false, false));
        assert!(!partial);
        assert_eq!(reference.len(), 4_000);
        assert!(reference.iter().filter(|x| (**x - 0.15).abs() < 0.05).count() > 2_000);
        for (tag, half, skip) in [("fp16", true, false), ("skip", false, true), ("fp16-skip", true, true)] {
            let (out, partial) = convert_with_generator(tag, fake_generator(half, skip));
            assert_eq!(partial, skip, "{tag}");
            assert_eq!(out.len(), reference.len(), "{tag}");
            let tol = if half { 2e-3 } else { 1e-7 };
            let err = out.iter().zip(&reference).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err <= tol, "{tag}: max difference {err}");
        }
    }

    /// A manifest naming its own content encoder uses it; a missing one falls back to the settings'
    /// path with a warning; a SPIN voice without its encoder installed says what to install.
    #[test]
    fn legacy_slots_use_their_own_content_encoder() {
        let (root, slot, hubert) = test_models::write_slot("rvc-own-encoder");
        let own = slot.join("my_hubert.onnx");
        std::fs::copy(&hubert, &own).unwrap();
        let mut warnings = Vec::new();
        let mut m = RvcManifest { hubert: Some("my_hubert.onnx".into()), ..RvcManifest::default() };
        let p = slot_content_encoder(&m, false, &slot, "", "cpu", &mut warnings).unwrap();
        assert_eq!(p, own);
        assert!(warnings.is_empty());
        m.hubert = Some("gone.onnx".into());
        let p = slot_content_encoder(&m, false, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap();
        assert_eq!(p, hubert);
        assert_eq!(warnings.len(), 1);
        // An inferred manifest never reads those fields.
        let p = slot_content_encoder(&m, true, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap();
        assert_eq!(p, hubert);
        m.hubert = None;
        m.embedder = Some("spin-v2".into());
        let err = slot_content_encoder(&m, false, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap_err();
        assert!(err.contains("vc/embedder-spin-v2"), "{err}");
        m.embedder = Some("contentvec".into());
        assert_eq!(slot_content_encoder(&m, false, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap(), hubert);
        // An encoder name Sigil does not know keeps the settings' encoder, with a warning.
        m.embedder = Some("custom".into());
        let before = warnings.len();
        assert_eq!(slot_content_encoder(&m, false, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap(), hubert);
        assert_eq!(warnings.len(), before + 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
