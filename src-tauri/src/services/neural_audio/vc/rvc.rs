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
        // A graph of another kind (a so-vits-svc export, a content encoder or pitch model from the
        // wok000 pack, ...) is named before the positional fallback could misread it (P5-D).
        if let Some(msg) = wrong_graph_message(path, "an RVC generator", &graph_inputs(session), &graph_outputs(session)) {
            return Err(msg);
        }
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
    /// w-okada's `embedder` (P5-D): `hubert_base` / `contentvec` (ContentVec), `hubert-base-japanese`
    /// (rinna's Japanese HuBERT), ...
    embedder: Option<String>,
}

fn read_export_metadata(session: &Session) -> Option<ExportMetadata> {
    let raw = session.metadata().ok()?.custom("metadata")?;
    parse_export_metadata(&raw)
}

/// [`ExportMetadata`] from w-okada's JSON (`None` when it is not a JSON object).
fn parse_export_metadata(raw: &str) -> Option<ExportMetadata> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    if !v.is_object() {
        return None;
    }
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
        embedder: v
            .get("embedder")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty() && s.len() <= 64)
            .map(str::to_string),
    })
}

/// The manifest `embedder` name of a w-okada export's `embedder` metadata (P5-D): `Ok(None)` for
/// the ContentVec family (picked by feature width, as before), `Ok(Some(name))` for another encoder
/// Sigil has (`hubert-ja`, ...), and `Err(warning)` for a name Sigil does not know (VCClient's
/// `whisper`, ...): the voice then runs on the settings' encoder and the caller shows the warning.
fn wokada_embedder_name(raw: &str) -> Result<Option<&'static str>, String> {
    let name = raw.trim().to_ascii_lowercase().replace(['_', ' '], "-");
    Ok(match name.as_str() {
        // w-okada's `hubert_base` is RVC's hubert_base.pt, i.e. ContentVec; `contentvec` (its ONNX
        // contentvec-f / -h) gives the same features.
        "hubert-base" | "hubert" | "contentvec" | "contentvec-onnx" | "content-vec" | "hubert-base-l12" => None,
        "hubert-base-japanese" | "hubert-base-jp" | "hubert-jp" | "japanese-hubert-base" | "rinna-hubert-base" => {
            Some(EmbedderKind::HubertJa.as_str())
        }
        "hubert-base-korean" | "korean-hubert-base" | "hubert-ko" => Some(EmbedderKind::HubertKo.as_str()),
        "hubert-base-chinese" | "chinese-hubert-base" | "hubert-zh" => Some(EmbedderKind::HubertZh.as_str()),
        "spin-v2" | "spinv2" => Some(EmbedderKind::SpinV2.as_str()),
        "spin" => Some(EmbedderKind::Spin.as_str()),
        _ => {
            return Err(format!(
                "the voice's w-okada metadata says it was trained with the content encoder {:?}, which Sigil does not have; \
                 using the HuBERT / ContentVec from the settings, so the voice may sound wrong",
                raw.trim()
            ))
        }
    })
}

// ---------------------------------------------------------------------------------------------
// P5-D: what a voice-changer file holds (specific rejections for w-okada / wok000 files)
// ---------------------------------------------------------------------------------------------

/// A graph input or output as declared: name, element type and dims (`-1` = dynamic).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphValue {
    pub name: String,
    pub dtype: Option<TensorElementType>,
    pub dims: Option<Vec<i64>>,
}

impl GraphValue {
    /// A declared value (tests and callers that describe graphs by hand).
    #[cfg(test)]
    pub(crate) fn new(name: &str, dtype: TensorElementType, dims: &[i64]) -> GraphValue {
        GraphValue {
            name: name.to_string(),
            dtype: Some(dtype),
            dims: Some(dims.to_vec()),
        }
    }

    fn rank(&self) -> Option<usize> {
        self.dims.as_ref().map(Vec::len)
    }

    /// The static last dimension, if declared positive.
    pub(crate) fn last_dim(&self) -> Option<usize> {
        self.dims
            .as_ref()
            .and_then(|d| d.last())
            .and_then(|&d| usize::try_from(d).ok())
            .filter(|&d| d > 0)
    }

    fn is_float(&self) -> bool {
        matches!(self.dtype, Some(TensorElementType::Float32) | Some(TensorElementType::Float16))
    }

    fn lower_name(&self) -> String {
        self.name.to_ascii_lowercase()
    }
}

/// The declared inputs of `session`.
pub(crate) fn graph_inputs(session: &Session) -> Vec<GraphValue> {
    session
        .inputs()
        .iter()
        .map(|o| GraphValue {
            name: o.name().to_string(),
            dtype: o.dtype().tensor_type(),
            dims: o.dtype().tensor_shape().map(|s| s.to_vec()),
        })
        .collect()
}

/// The declared outputs of `session`.
pub(crate) fn graph_outputs(session: &Session) -> Vec<GraphValue> {
    session
        .outputs()
        .iter()
        .map(|o| GraphValue {
            name: o.name().to_string(),
            dtype: o.dtype().tensor_type(),
            dims: o.dtype().tensor_shape().map(|s| s.to_vec()),
        })
        .collect()
}

/// What kind of graph a voice-changer ONNX file holds, from its declared inputs and outputs (P5-D:
/// the audit of w-okada's VCClient exports and the wok000 `weights_gpl` pack). A loader that gets
/// the wrong kind says what the file is and where it belongs instead of a bare input mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphKind {
    /// An RVC generator (RVC WebUI / Applio / w-okada / RVC realtime input names).
    RvcGenerator,
    /// A so-vits-svc 4.0 / 4.1 export (`c`, `f0`, `mel2ph`, `uv`, `noise`, `sid`; the `sovits`
    /// family runs it).
    Sovits4,
    /// A so-vits-svc 3.0 export (`hidden_unit`, `lengths`, `pitch`, `sid`: hubert-soft features).
    Sovits3,
    /// A content encoder (HuBERT / ContentVec / w-okada's `contentvec-f`, `hubert-jp-f`).
    ContentEncoder,
    /// A pitch model (RMVPE, FCPE, CREPE, SwiftF0 exports).
    PitchModel,
    /// w-okada VCClient's fused RMVPE + ContentVec pipeline (wok000 `rmvpe2_silence.onnx`).
    WokadaPitchPipeline,
    /// A DDSP-SVC / Diffusion-SVC graph (encoder or reflow / diffusion step).
    DdspSvc,
    /// The sampler step of another diffusion / flow-matching converter (Seed-VC's or MeanVC's DiT:
    /// `x`, `t`, `cond`, ...), one part of a zero-shot family's graph set.
    FlowStep,
    /// A vocoder (mel in, waveform out: NSF-HiFiGAN with F0, HiFT, Vocos).
    Vocoder,
    /// w-okada's LLVC streaming graph.
    Llvc,
    /// An MMVC v1.3 / v1.5 voice from w-okada's MMVC client (`specs`, `lengths`, `sid_src`,
    /// `sid_tgt`, plus `f0` on v1.5: a linear spectrogram in, many-to-many speaker ids).
    Mmvc,
    /// None of the above.
    Unknown,
}

/// Feature widths of the content encoders voice models are trained on (ContentVec 256 / 768,
/// whisper-ppg 1024 / 1280, HuBERT-large 1024).
const CONTENT_WIDTHS: [usize; 4] = [256, 768, 1024, 1280];

/// Identify a graph from its declared inputs and outputs (see [`GraphKind`]).
pub(crate) fn identify_graph(inputs: &[GraphValue], outputs: &[GraphValue]) -> GraphKind {
    let names: Vec<String> = inputs.iter().map(GraphValue::lower_name).collect();
    let has = |n: &str| names.iter().any(|x| x == n);
    if has("prevpitchf") || (has("waveform") && (has("upkey") || has("silencefrontsec") || has("pitchlength"))) {
        return GraphKind::WokadaPitchPipeline;
    }
    if has("enc_buf") || has("dec_buf") || has("out_buf") {
        return GraphKind::Llvc;
    }
    if has("specs") && (has("sid_src") || has("sid_tgt")) {
        return GraphKind::Mmvc;
    }
    if (has("mel2ph") && (has("c") || has("hidden_unit"))) || (has("c") && has("uv") && has("f0")) {
        return GraphKind::Sovits4;
    }
    if has("hidden_unit") && has("pitch") && !has("pitchf") {
        return GraphKind::Sovits3;
    }
    if has("volume") && (has("f0") || has("units") || has("hubert")) {
        return GraphKind::DdspSvc;
    }
    if has("x") && has("t") && has("cond") {
        // DDSP-SVC's reflow / velocity step works on a 128-band mel `[1, 1, 128, T]`; Seed-VC's and
        // MeanVC's DiT steps take `[B, 80, L]` / `[1, 8, 80]`.
        let x = inputs.iter().find(|i| i.lower_name() == "x");
        let ddsp = x.and_then(|x| x.dims.as_deref()).is_some_and(|d| matches!(d, [_, _, 128, _]));
        return if ddsp { GraphKind::DdspSvc } else { GraphKind::FlowStep };
    }
    let roles: Option<Vec<GenRole>> = names.iter().map(|n| role_for_name(n)).collect();
    if let Some(roles) = roles {
        if roles.contains(&GenRole::Phone) && roles.contains(&GenRole::PhoneLengths) {
            return GraphKind::RvcGenerator;
        }
    }
    let audio_out = |o: &GraphValue| {
        let n = o.lower_name();
        n.contains("audio") || n.contains("wav")
    };
    if has("mel") && outputs.iter().any(audio_out) {
        return GraphKind::Vocoder;
    }
    // One audio input (plus optional scalar settings: a voicing threshold, SwiftF0's F0 range).
    let audio: Vec<&GraphValue> = inputs
        .iter()
        .filter(|i| {
            let n = i.lower_name();
            !(n.contains("threshold") || n == "thred" || n == "fmin" || n == "fmax")
        })
        .collect();
    if audio.len() == 1 && audio[0].is_float() {
        let input = audio[0];
        let pitch_name = |o: &GraphValue| {
            let n = o.lower_name();
            ["f0", "pitch", "salience", "probabilit", "uv"].iter().any(|k| n.contains(k))
        };
        let salience = |o: &GraphValue| o.last_dim() == Some(super::pitch_onnx::SALIENCE_BINS);
        if outputs.iter().any(|o| pitch_name(o) || salience(o)) {
            return GraphKind::PitchModel;
        }
        let features = |o: &GraphValue| o.rank() == Some(3) && o.last_dim().is_some_and(|c| CONTENT_WIDTHS.contains(&c));
        // A waveform of any length in (a fixed last axis is a vector, such as a speaker embedding).
        let waveform = matches!(input.rank(), Some(1) | Some(2)) && input.last_dim().is_none();
        if waveform && !outputs.is_empty() && outputs.iter().all(features) {
            return GraphKind::ContentEncoder;
        }
        // CREPE takes 1024-sample frames `[n, 1024]`.
        if input.rank() == Some(2) && input.last_dim() == Some(1024) {
            return GraphKind::PitchModel;
        }
        // A 128-band mel spectrogram (RVC's rmvpe.onnx `[1, 128, T]`, a mel FCPE `[1, T, 128]`).
        let mel = |d: &[i64]| matches!(d, [_, 128, _] | [_, _, 128]);
        if input.dims.as_deref().is_some_and(mel) {
            return GraphKind::PitchModel;
        }
    }
    GraphKind::Unknown
}

/// What a file of `kind` is and where it belongs (completes "<file> is ..."), or `None` for an RVC
/// generator / an unknown graph.
pub(crate) fn graph_kind_advice(kind: GraphKind) -> Option<&'static str> {
    Some(match kind {
        GraphKind::RvcGenerator | GraphKind::Unknown => return None,
        GraphKind::Sovits4 => "a so-vits-svc 4.x ONNX export; add it as a so-vits-svc voice (it runs in the sovits voice family, not as an RVC generator)",
        GraphKind::Sovits3 => "a so-vits-svc 3.0 export (hubert-soft features and coarse pitch), which Sigil does not run; export the voice with so-vits-svc 4.0 or 4.1 and add it as a so-vits-svc voice",
        GraphKind::ContentEncoder => "a content encoder (HuBERT / ContentVec, such as w-okada's contentvec-f.onnx); set it as the HuBERT / ContentVec model, not as a voice",
        GraphKind::PitchModel => "a pitch model (RMVPE / FCPE / CREPE); set it as the RMVPE or FCPE model, not as a voice",
        GraphKind::WokadaPitchPipeline => "w-okada VCClient's fused RMVPE + ContentVec pipeline (wok000 rmvpe2_silence.onnx, inputs waveform / prevPitchf / upKey / threshold / pitchLength / silenceFrontSec), which only VCClient's own pipeline drives; use rmvpe_20231006.onnx or rmvpe_s.onnx from the same pack as the RMVPE model and contentvec-f.onnx as the content encoder",
        GraphKind::DdspSvc => "a DDSP-SVC / Diffusion-SVC graph; add the DDSP-SVC checkpoint as a DDSP-SVC voice instead",
        GraphKind::FlowStep => "the sampler step of a diffusion / flow-matching voice converter (such as Seed-VC's or MeanVC's DiT), one part of a zero-shot voice family's graphs, not a voice on its own",
        GraphKind::Vocoder => "a vocoder (a mel spectrogram in, audio out), one part of another voice converter's graphs, not a voice",
        GraphKind::Llvc => "w-okada VCClient's LLVC streaming graph, which Sigil does not run",
        GraphKind::Mmvc => "an MMVC v1.3 / v1.5 voice (w-okada's MMVC client: a spectrogram in, source and target speaker ids), which Sigil does not run; an RVC or so-vits-svc voice of the same speaker works",
    })
}

/// "`<file>` is not <expected>: it is <advice>" when the file holds a graph of another kind.
pub(crate) fn wrong_graph_message(path: &Path, expected: &str, inputs: &[GraphValue], outputs: &[GraphValue]) -> Option<String> {
    let advice = graph_kind_advice(identify_graph(inputs, outputs))?;
    Some(format!("{} is not {expected}: it is {advice}", file_label(path)))
}

/// Why `session` (of `path`) cannot be `expected` (`"an RMVPE pitch model"`, ...): it holds a
/// graph of another kind than `accepted` (an unknown graph is always left to the caller's own
/// checks). One call at load for every loader of a voice-changer ONNX file.
pub(crate) fn foreign_graph_problem(session: &Session, path: &Path, expected: &str, accepted: &[GraphKind]) -> Option<String> {
    let (inputs, outputs) = (graph_inputs(session), graph_outputs(session));
    let kind = identify_graph(&inputs, &outputs);
    if kind == GraphKind::Unknown || accepted.contains(&kind) {
        return None;
    }
    wrong_graph_message(path, expected, &inputs, &outputs)
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Why `path` cannot be an ONNX graph, when its first bytes say it is another known format (a
/// PyTorch checkpoint, a pickle, a JSON config, a NumPy array, a GGUF model); `None` otherwise
/// (ONNX Runtime then parses it). Unreadable files are left to the loader's own error.
pub(crate) fn non_onnx_file_problem(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path).ok()?.read(&mut head).ok()?;
    non_onnx_bytes_problem(&head[..n]).map(|what| format!("{} is {what}, not an ONNX graph", file_label(path)))
}

/// The format of `head` when it is a known non-ONNX one (see [`non_onnx_file_problem`]).
fn non_onnx_bytes_problem(head: &[u8]) -> Option<&'static str> {
    if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
        return Some("a zip archive (a PyTorch .pt / .pth checkpoint or a zipped voice)");
    }
    if head.len() >= 2 && head[0] == 0x80 && (2..=5).contains(&head[1]) {
        return Some("a Python pickle (a legacy PyTorch checkpoint)");
    }
    if head.starts_with(b"\x93NUMPY") {
        return Some("a NumPy array (.npy)");
    }
    if head.starts_with(b"GGUF") {
        return Some("a GGUF model");
    }
    let text = head.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(head);
    if let Some(&first) = text.iter().find(|b| !b.is_ascii_whitespace()) {
        if first == b'{' || first == b'[' {
            return Some("a JSON file (a model's config)");
        }
    }
    None
}

/// The output of a content encoder that carries `channels`-wide features (P5-D): w-okada's ContentVec
/// / Japanese-HuBERT exports (wok000 `contentvec-f/-h.onnx`, `hubert-jp-f/-h.onnx`) return three
/// (`units9` `[1, T, 256]`, `unit12` `[1, T, 768]`, `unit12s`), one encoder for v1 and v2 voices.
/// The first output whose static last dimension is `channels` wins (its `[1, C, T]` layout too, when
/// the time axis is dynamic); otherwise the first output, as before.
pub(crate) fn feature_output_index(outputs: &[GraphValue], channels: usize) -> usize {
    let wide = |o: &GraphValue| match o.dims.as_deref() {
        Some([.., c]) if *c == channels as i64 => true,
        Some([1, c, t]) => *c == channels as i64 && *t < 0,
        _ => false,
    };
    outputs.iter().position(wide).unwrap_or(0)
}

pub struct RvcModel {
    hubert: Session,
    /// The content encoder takes fp16 audio.
    hubert_half: bool,
    /// The content encoder output carrying the voice's feature width ([`feature_output_index`]).
    hubert_output: usize,
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
/// model store, else w-okada's export of it next to the voice or in `voice-changer/base/`), else
/// the global HuBERT / ContentVec path. A named file that is missing falls back to the global path
/// with a warning (older manifests carried the field unused). An inferred manifest (no
/// `manifest.json`) never has a `hubert` file; its `embedder` comes only from the generator's
/// w-okada metadata (P5-D), and when that encoder is not found an existing settings path is used
/// with a warning, as before P5-D (the user may have set their own export of it there).
fn slot_content_encoder(
    manifest: &RvcManifest,
    inferred: bool,
    slot_dir: &Path,
    hubert_path: &str,
    ep_hint: &str,
    warnings: &mut Vec<String>,
) -> Result<PathBuf, String> {
    let settings_ok = !hubert_path.trim().is_empty() && Path::new(hubert_path).is_file();
    let named = if inferred { None } else { manifest.hubert.as_deref().map(str::trim).filter(|s| !s.is_empty()) };
    if let Some(name) = named {
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
                let found = resolve_encoder_file(kind, None, Some(slot_dir), gpu, PrecisionMode::Auto)
                    .map(|r| r.path)
                    .or_else(|e| wokada_encoder_file(kind, slot_dir, gpu).ok_or(e));
                match found {
                    Ok(p) => return Ok(p),
                    Err(e) if inferred && settings_ok => warnings.push(format!(
                        "The voice's w-okada metadata says it was trained with {} ({e}); using the HuBERT / ContentVec from the settings ({hubert_path}), so the voice sounds wrong unless that is the same encoder.",
                        kind.as_str()
                    )),
                    Err(e) => return Err(e),
                }
            }
            Ok(_) => {}
            // A name Sigil does not know (Applio's `custom`): the settings' encoder, as before.
            Err(e) => warnings.push(format!("{e}; using the HuBERT / ContentVec from the settings.")),
        }
    }
    if !settings_ok {
        return Err(format!("HuBERT / ContentVec missing: {}", hubert_path));
    }
    Ok(PathBuf::from(hubert_path))
}

/// w-okada's export of a non-ContentVec encoder (the wok000 `weights_gpl` pack's
/// `hubert-jp-f.onnx` / `hubert-jp-h.onnx`) next to the voice or in `voice-changer/base/` (the
/// slot's `../../base`), the fp16 one first on GPUs (P5-D). Only rinna's Japanese HuBERT has one.
fn wokada_encoder_file(kind: EmbedderKind, slot_dir: &Path, gpu: bool) -> Option<PathBuf> {
    let names: [&str; 2] = match kind {
        EmbedderKind::HubertJa if gpu => ["hubert-jp-h.onnx", "hubert-jp-f.onnx"],
        EmbedderKind::HubertJa => ["hubert-jp-f.onnx", "hubert-jp-h.onnx"],
        _ => return None,
    };
    let mut dirs = vec![slot_dir.to_path_buf()];
    if let Some(root) = slot_dir.parent().and_then(Path::parent) {
        dirs.push(root.join("base"));
    }
    dirs.iter()
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|p| p.is_file())
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
        if let Some(problem) = non_onnx_file_problem(&gen_path) {
            return Err(format!(
                "RVC generator: {problem}; a bring-your-own voice needs its ONNX export (add a .pth / .zip voice with Add voice instead)"
            ));
        }
        let mut warnings = Vec::new();
        // The generator first (P5-D): its w-okada metadata can name the content encoder.
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
                // w-okada names the content encoder the voice was trained with (P5-D).
                if let Some(raw) = meta.embedder.as_deref() {
                    manifest.embedder = match wokada_embedder_name(raw) {
                        Ok(name) => name.map(str::to_string),
                        Err(warning) => {
                            warnings.push(warning);
                            None
                        }
                    };
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

        let hubert_file = slot_content_encoder(&manifest, inferred, slot_dir, hubert_path, ep_hint, &mut warnings)?;
        if let Some(problem) = non_onnx_file_problem(&hubert_file) {
            return Err(format!(
                "HuBERT / ContentVec: {problem}; set the content encoder's ONNX export (w-okada's contentvec-f.onnx works) or clear the path to use the installed one"
            ));
        }
        let hubert = session_builder_for_vc(ep_hint, opts)?
            .component("vc.content")
            .commit_from_file(&hubert_file)
            .map_err(|e| format!("HuBERT load: {}", e))?;
        let hubert_half = hubert
            .inputs()
            .first()
            .is_some_and(|i| i.dtype().tensor_type() == Some(TensorElementType::Float16));
        if let Some(msg) = foreign_graph_problem(&hubert, &hubert_file, "a content encoder", &[GraphKind::ContentEncoder]) {
            return Err(format!("HuBERT / ContentVec: {msg}"));
        }
        // w-okada's encoders return one output per feature width (P5-D).
        let hubert_output = feature_output_index(&graph_outputs(&hubert), manifest.emb_channels as usize);

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
            hubert_output,
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
            reference: None,
            singing: Default::default(),
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
        let pick = if self.hubert_output < outputs.len() { self.hubert_output } else { 0 };
        let (dims, data) = extract_f32(&outputs[pick], "HuBERT")?;

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
        let frames_50hz = feats.len().checked_div(emb).unwrap_or(0);
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
        let firsts: Vec<f32> = up.as_chunks::<2>().0.iter().map(|f| f[0]).collect();
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
            reference: None,
            singing: Default::default(),
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

    // ---- P5-D: w-okada / wok000 audit ----

    use TensorElementType::{Bool, Float16 as F16, Float32 as F32, Int32, Int64 as I64};

    fn gv(name: &str, t: TensorElementType, dims: &[i64]) -> GraphValue {
        GraphValue::new(name, t, dims)
    }

    /// The wok000 `weights_gpl` pack's ONNX files, as their graphs declare them (read from the
    /// published files' tails during the P5-D audit), are identified by what they are.
    #[test]
    fn the_wok000_pack_is_identified() {
        let wokada_encoder_out = [gv("units9", F16, &[1, -1, 256]), gv("unit12", F16, &[1, -1, 768]), gv("unit12s", F16, &[1, -1, 768])];
        // content-vec/contentvec-h.onnx and hubert-jp/hubert-jp-h.onnx (-f: the same in fp32).
        assert_eq!(identify_graph(&[gv("audio", F16, &[1, -1])], &wokada_encoder_out), GraphKind::ContentEncoder);
        // rmvpe/rmvpe_20231006.onnx.
        let rmvpe = [gv("waveform", F32, &[1, -1]), gv("threshold", F32, &[1])];
        assert_eq!(identify_graph(&rmvpe, &[gv("pitchf", F32, &[-1, -1])]), GraphKind::PitchModel);
        // rmvpe/rmvpe_s.onnx.
        let rmvpe_s = [gv("waveform", F32, &[1, -1]), gv("threshold", F32, &[])];
        let outs = [gv("f0", F32, &[1, -1]), gv("uv", Bool, &[1, -1]), gv("f0_s", F32, &[1, -1])];
        assert_eq!(identify_graph(&rmvpe_s, &outs), GraphKind::PitchModel);
        // rmvpe/rmvpe2_silence.onnx: the fused pipeline.
        let fused = [
            gv("waveform", F32, &[1, -1]),
            gv("prevPitchf", F32, &[1, -1]),
            gv("upKey", Int32, &[1]),
            gv("threshold", F32, &[1]),
            gv("pitchLength", I64, &[1]),
            gv("silenceFrontSec", F32, &[]),
        ];
        let outs = [gv("pitch", Int32, &[-1, -1]), gv("pitchf", F32, &[-1, -1]), gv("feats", F32, &[1, -1, 768])];
        assert_eq!(identify_graph(&fused, &outs), GraphKind::WokadaPitchPipeline);
        let msg = wrong_graph_message(Path::new("rmvpe2_silence.onnx"), "an RVC generator", &fused, &outs).unwrap();
        assert!(msg.contains("rmvpe_20231006.onnx") && msg.contains("VCClient"), "{msg}");
        // The other pitch exports Sigil knows: RVC's mel RMVPE, the community FCPE, CREPE.
        assert_eq!(identify_graph(&[gv("input", F32, &[1, 128, -1])], &[gv("output", F32, &[-1, -1, -1])]), GraphKind::PitchModel);
        assert_eq!(identify_graph(&[gv("audio", F32, &[1, -1, 1])], &[gv("f0_hz", F32, &[-1, -1, 1])]), GraphKind::PitchModel);
        assert_eq!(identify_graph(&[gv("frames", F32, &[-1, 1024])], &[gv("probabilities", F32, &[-1, 360])]), GraphKind::PitchModel);
    }

    /// Voices: RVC exports by every known exporter, so-vits-svc 4.x / 3.0, DDSP-SVC, LLVC.
    #[test]
    fn voice_graphs_are_identified() {
        let audio = [gv("audio", F32, &[1, 1, -1])];
        // w-okada's RVC export (fp16) and RVC WebUI's.
        let wokada = [gv("feats", F16, &[1, -1, 768]), gv("p_len", I64, &[1]), gv("pitch", I64, &[1, -1]), gv("pitchf", F16, &[1, -1]), gv("sid", I64, &[1])];
        assert_eq!(identify_graph(&wokada, &audio), GraphKind::RvcGenerator);
        let nono = [gv("feats", F32, &[1, -1, 256]), gv("p_len", I64, &[1]), gv("sid", I64, &[1])];
        assert_eq!(identify_graph(&nono, &audio), GraphKind::RvcGenerator);
        let webui = [
            gv("phone", F32, &[1, -1, 768]),
            gv("phone_lengths", I64, &[1]),
            gv("pitch", I64, &[1, -1]),
            gv("pitchf", F32, &[1, -1]),
            gv("ds", I64, &[1]),
            gv("rnd", F32, &[1, 192, -1]),
        ];
        assert_eq!(identify_graph(&webui, &audio), GraphKind::RvcGenerator);
        let sovits = [
            gv("c", F32, &[1, -1, 768]),
            gv("f0", F32, &[1, -1]),
            gv("mel2ph", I64, &[1, -1]),
            gv("uv", F32, &[1, -1]),
            gv("noise", F32, &[1, 192, -1]),
            gv("sid", I64, &[1]),
        ];
        assert_eq!(identify_graph(&sovits, &audio), GraphKind::Sovits4);
        let sovits3 = [gv("hidden_unit", F32, &[1, -1, 256]), gv("lengths", I64, &[1]), gv("pitch", I64, &[1, -1]), gv("sid", I64, &[1])];
        assert_eq!(identify_graph(&sovits3, &audio), GraphKind::Sovits3);
        let ddsp = [gv("units", F32, &[1, -1, 768]), gv("f0", F32, &[1, -1, 1]), gv("volume", F32, &[1, -1, 1]), gv("spk_mix", F32, &[1, -1])];
        assert_eq!(identify_graph(&ddsp, &audio), GraphKind::DdspSvc);
        let llvc = [gv("audio", F32, &[1, -1]), gv("enc_buf", F32, &[1, 512, 12]), gv("dec_buf", F32, &[1, 512, 12]), gv("out_buf", F32, &[1, 1, 60])];
        assert_eq!(identify_graph(&llvc, &audio), GraphKind::Llvc);
        let vocoder = [gv("mel", F32, &[1, -1, 128]), gv("f0", F32, &[1, -1])];
        assert_eq!(identify_graph(&vocoder, &audio), GraphKind::Vocoder);
        // w-okada's MMVC v1.3 and v1.5 exports (v1.5 adds f0).
        let mmvc13 = [gv("specs", F32, &[1, 257, -1]), gv("lengths", I64, &[1]), gv("sid_src", I64, &[1]), gv("sid_tgt", I64, &[1])];
        assert_eq!(identify_graph(&mmvc13, &audio), GraphKind::Mmvc);
        let mut mmvc15 = mmvc13.to_vec();
        mmvc15.insert(2, gv("f0", F32, &[1, -1]));
        assert_eq!(identify_graph(&mmvc15, &audio), GraphKind::Mmvc);
        let msg = wrong_graph_message(Path::new("G_mmvc.onnx"), "an RVC generator", &mmvc15, &audio).unwrap();
        assert!(msg.starts_with("G_mmvc.onnx is not an RVC generator") && msg.contains("MMVC"), "{msg}");
        // Unknown names stay unknown (the RVC positional fallback still applies to them).
        let positional: Vec<GraphValue> = webui.iter().enumerate().map(|(i, v)| GraphValue { name: format!("x.{i}"), ..v.clone() }).collect();
        assert_eq!(identify_graph(&positional, &audio), GraphKind::Unknown);
        for kind in [GraphKind::RvcGenerator, GraphKind::Unknown] {
            assert!(graph_kind_advice(kind).is_none());
        }
    }

    /// The other models Sigil ships, exports or benchmarks with, by their real declared I/O
    /// (captured from the files in the dev cache with ONNX Runtime 1.28; `-1` = a named dynamic
    /// axis): pitch models, encoders, vocoders, the zero-shot families' parts, denoisers and the
    /// benchmark's models are each what they are, and never a voice or a content encoder by
    /// accident (a loader refuses every identified kind it does not accept).
    #[test]
    fn real_model_declarations_are_identified() {
        use TensorElementType::Float64 as F64;
        let cases: Vec<(&str, Vec<GraphValue>, Vec<GraphValue>, GraphKind)> = vec![
            (
                "vc/pitch-swiftf0",
                vec![gv("audio", F32, &[1, -1]), gv("fmin", F32, &[]), gv("fmax", F32, &[])],
                vec![gv("pitch", F64, &[1, -1]), gv("confidence", F32, &[1, -1])],
                GraphKind::PitchModel,
            ),
            ("vc/pitch-crepe-tiny", vec![gv("frames", F32, &[-1, 1024])], vec![gv("probabilities", F32, &[-1, 360])], GraphKind::PitchModel),
            ("rmvpe.onnx", vec![gv("input", F32, &[1, 128, -1])], vec![gv("output", F32, &[-1, -1, 360])], GraphKind::PitchModel),
            ("fcpe.onnx (community)", vec![gv("audio", F32, &[1, -1, 1])], vec![gv("f0_hz", F32, &[-1, -1, 1])], GraphKind::PitchModel),
            ("vc/pitch-fcpe", vec![gv("audio", F32, &[1, -1])], vec![gv("f0", F32, &[1, -1])], GraphKind::PitchModel),
            ("vc/embedder-contentvec", vec![gv("audio", F32, &[1, -1])], vec![gv("feats", F32, &[1, -1, 768])], GraphKind::ContentEncoder),
            ("vc/embedder-contentvec-256", vec![gv("audio", F32, &[1, -1])], vec![gv("feats", F32, &[1, -1, 256])], GraphKind::ContentEncoder),
            ("seed_vc/xlsr", vec![gv("wav16k", F32, &[1, -1])], vec![gv("feats", F32, &[1, -1, 1024])], GraphKind::ContentEncoder),
            (
                "vc/vocoder-pc-nsf-hifigan-44k",
                vec![gv("mel", F32, &[1, 128, -1]), gv("f0", F32, &[1, -1]), gv("phase0", F32, &[1])],
                vec![gv("audio", F32, &[-1, -1])],
                GraphKind::Vocoder,
            ),
            (
                "seed_vc/hift",
                vec![gv("mel", F32, &[1, 80, -1]), gv("harm_phase", F32, &[1, 9, 1]), gv("sine_noise", F32, &[1, 9, -1])],
                vec![gv("audio", F32, &[-1, -1]), gv("f0", F32, &[1, -1])],
                GraphKind::Vocoder,
            ),
            ("meanvc/vocos", vec![gv("mel", F32, &[1, 80, -1])], vec![gv("wav", F32, &[1, -1])], GraphKind::Vocoder),
            (
                "ddsp_svc/encoder",
                vec![
                    gv("hubert", F32, &[1, -1, 768]),
                    gv("mel2ph", I64, &[1, -1]),
                    gv("f0", F32, &[1, -1]),
                    gv("volume", F32, &[1, -1]),
                    gv("randn", F32, &[1, -1]),
                ],
                vec![gv("x", F32, &[-1, 1, 128, -1]), gv("cond", F32, &[-1, 128, -1])],
                GraphKind::DdspSvc,
            ),
            (
                "ddsp_svc/velocity",
                vec![gv("x", F32, &[1, 1, 128, -1]), gv("t", I64, &[1]), gv("cond", F32, &[1, 128, -1])],
                vec![gv("o", F32, &[1, 1, 128, -1])],
                GraphKind::DdspSvc,
            ),
            (
                "seed_vc/dit_step",
                vec![
                    gv("x", F32, &[1, 80, -1]),
                    gv("prompt_x", F32, &[1, 80, -1]),
                    gv("cond", F32, &[1, -1, 384]),
                    gv("style", F32, &[1, 192]),
                    gv("t", F32, &[1]),
                    gv("dt", F32, &[1]),
                    gv("cfg_rate", F32, &[1]),
                    gv("prompt_len", I64, &[1]),
                ],
                vec![gv("x_next", F32, &[1, 80, -1])],
                GraphKind::FlowStep,
            ),
            (
                "meanvc/dit",
                vec![
                    gv("x", F32, &[1, 8, 80]),
                    gv("t", F32, &[1]),
                    gv("r", F32, &[1]),
                    gv("cond", F32, &[1, 8, 256]),
                    gv("spks", F32, &[1, 256]),
                    gv("k_mem", F32, &[1, 32, 256]),
                    gv("v_mem", F32, &[1, 32, 256]),
                    gv("offset", I64, &[1]),
                    gv("n_valid", I64, &[1]),
                    gv("cache", F32, &[4, 2, 2, 12, 64]),
                ],
                vec![gv("u", F32, &[1, 8, 80]), gv("cache_out", F32, &[4, 2, 2, 12, 64])],
                GraphKind::FlowStep,
            ),
            // A speaker vector in, not audio: never a content encoder although its outputs are
            // 256-wide.
            ("meanvc/gtm", vec![gv("spks", F32, &[1, 256])], vec![gv("k_mem", F32, &[1, 32, 256]), gv("v_mem", F32, &[1, 32, 256])], GraphKind::Unknown),
            ("meanvc/fbank", vec![gv("wav16k", F32, &[1, -1])], vec![gv("fbank", F32, &[1, -1, 80])], GraphKind::Unknown),
            ("seed_vc/mel", vec![gv("wav22k", F32, &[1, -1])], vec![gv("mel", F32, &[1, 80, -1])], GraphKind::Unknown),
            ("seed_vc/lr", vec![gv("feats", F32, &[1, -1, 1024]), gv("n_frames", I64, &[1])], vec![gv("cond", F32, &[1, -1, 384])], GraphKind::Unknown),
            ("bench/campplus", vec![gv("x", F32, &[-1, -1, 80])], vec![gv("embedding", F32, &[-1, 192])], GraphKind::Unknown),
            ("bench/dnsmos-p835", vec![gv("input_1", F32, &[-1, 144_160])], vec![gv("Identity:0", F32, &[-1, 3])], GraphKind::Unknown),
            ("bench/gpu-load", vec![gv("x", F32, &[1, 32, -1, -1])], vec![gv("y", F32, &[1, 32, -1, -1])], GraphKind::Unknown),
            (
                "vc/denoise-dpdfnet",
                vec![gv("spec", F32, &[1, 1, 481, 2]), gv("state_in", F32, &[56_436])],
                vec![gv("spec_e", F32, &[1, 1, 481, 2]), gv("state_out", F32, &[56_436])],
                GraphKind::Unknown,
            ),
            (
                "vc/denoise-gtcrn",
                vec![
                    gv("mix", F32, &[1, 257, 1, 2]),
                    gv("conv_cache", F32, &[2, 1, 16, 16, 33]),
                    gv("tra_cache", F32, &[2, 3, 1, 1, 16]),
                    gv("inter_cache", F32, &[2, 1, 33, 16]),
                ],
                vec![gv("enh", F32, &[1, 257, 1, 2])],
                GraphKind::Unknown,
            ),
            ("vad/smart-turn", vec![gv("input_features", F32, &[-1, 80, 800])], vec![gv("logits", F32, &[-1, 1])], GraphKind::Unknown),
        ];
        for (file, inputs, outputs, want) in &cases {
            assert_eq!(identify_graph(inputs, outputs), *want, "{file}");
        }
        // A step graph that is not DDSP-SVC's says so.
        let (_, i, o, _) = cases.iter().find(|c| c.0 == "seed_vc/dit_step").unwrap();
        let msg = wrong_graph_message(Path::new("dit_step.onnx"), "an RVC generator", i, o).unwrap();
        assert!(msg.contains("flow-matching") && !msg.contains("DDSP"), "{msg}");
    }

    /// Dev rig: the real files behind [`real_model_declarations_are_identified`] (and every other
    /// voice-changer ONNX in the dev cache) open in ONNX Runtime and identify the same way.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_dev_cache_graphs -- --ignored --nocapture`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR (the dev cache with vc-assets, upstream-assets and vc-families)"]
    fn real_dev_cache_graphs_are_identified() {
        use crate::services::neural_audio::ort_session::{build_session, ModelSource};
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR not set; skipped");
            return;
        };
        let cases: [(&str, GraphKind); 24] = [
            ("vc-assets/vc/pitch-swiftf0/model.onnx", GraphKind::PitchModel),
            ("vc-assets/vc/pitch-crepe-tiny/model.onnx", GraphKind::PitchModel),
            ("vc-assets/vc/pitch-fcpe/model.onnx", GraphKind::PitchModel),
            ("upstream-assets/rmvpe/rmvpe.onnx", GraphKind::PitchModel),
            ("upstream-assets/fcpe/fcpe.onnx", GraphKind::PitchModel),
            ("vc-assets/vc/embedder-contentvec/model.fp32.onnx", GraphKind::ContentEncoder),
            ("vc-assets/vc/embedder-contentvec-256/model.fp32.onnx", GraphKind::ContentEncoder),
            ("vc-assets/vc/embedder-hubert-ja/model.fp16.onnx", GraphKind::ContentEncoder),
            ("vc-assets/vc/embedder-spin-v2/model.fp32.onnx", GraphKind::ContentEncoder),
            ("vc-assets/vc/vocoder-pc-nsf-hifigan-44k/model.onnx", GraphKind::Vocoder),
            ("vc-families/ddsp_svc/encoder.onnx", GraphKind::DdspSvc),
            ("vc-families/ddsp_svc/ddsp.onnx", GraphKind::DdspSvc),
            ("vc-families/ddsp_svc/velocity.onnx", GraphKind::DdspSvc),
            ("vc-families/ddsp_svc/reflow_step.onnx", GraphKind::DdspSvc),
            ("vc-families/ddsp_svc/vocoder.onnx", GraphKind::Vocoder),
            ("vc-families/seed_vc/dit_step.onnx", GraphKind::FlowStep),
            ("vc-families/seed_vc/hift.onnx", GraphKind::Vocoder),
            ("vc-families/meanvc/dit_step.onnx", GraphKind::FlowStep),
            ("vc-families/meanvc/gtm.onnx", GraphKind::Unknown),
            ("vc-families/meanvc/vocos.onnx", GraphKind::Vocoder),
            ("vc-assets/bench/campplus/model.onnx", GraphKind::Unknown),
            ("vc-assets/vc/denoise-dpdfnet/model.onnx", GraphKind::Unknown),
            ("vc-assets/vc/denoise-gtcrn/model.onnx", GraphKind::Unknown),
            ("smart-turn/smart-turn-v3.2-cpu.onnx", GraphKind::Unknown),
        ];
        let mut checked = 0;
        for (rel, want) in cases {
            let path = root.join(rel);
            if !path.is_file() {
                eprintln!("{rel}: not in the dev cache; skipped");
                continue;
            }
            let (session, _) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::File(&path)).unwrap_or_else(|e| panic!("{rel}: {e}"));
            let kind = identify_graph(&graph_inputs(&session), &graph_outputs(&session));
            eprintln!("{rel}: {kind:?}");
            assert_eq!(kind, want, "{rel}");
            checked += 1;
        }
        assert!(checked > 0, "no dev-cache model found under {}", root.display());
    }

    #[test]
    fn content_encoder_outputs_are_chosen_by_width() {
        let outs = [gv("units9", F16, &[1, -1, 256]), gv("unit12", F16, &[1, -1, 768]), gv("unit12s", F16, &[1, -1, 768])];
        assert_eq!(feature_output_index(&outs, 768), 1);
        assert_eq!(feature_output_index(&outs, 256), 0);
        assert_eq!(feature_output_index(&outs, 1024), 0, "no match: the first, as before");
        // Channel-major with a dynamic time axis.
        assert_eq!(feature_output_index(&[gv("a", F32, &[1, -1, 256]), gv("b", F32, &[1, 768, -1])], 768), 1);
        assert_eq!(feature_output_index(&[gv("feats", F32, &[1, -1, -1])], 768), 0);
    }

    #[test]
    fn non_onnx_files_are_named() {
        assert!(non_onnx_bytes_problem(b"PK\x03\x04....").unwrap().contains("PyTorch"));
        assert!(non_onnx_bytes_problem(b"\x80\x02}q\x00").unwrap().contains("pickle"));
        assert!(non_onnx_bytes_problem(b"\x93NUMPY\x01\x00").unwrap().contains("NumPy"));
        assert!(non_onnx_bytes_problem(b"GGUF\x03\x00").unwrap().contains("GGUF"));
        assert!(non_onnx_bytes_problem(b"\xEF\xBB\xBF  {\"a\": 1}").unwrap().contains("JSON"));
        // An ONNX ModelProto starts with its ir_version field.
        assert_eq!(non_onnx_bytes_problem(&[0x08, 0x07, 0x12, 0x0a]), None);
        assert_eq!(non_onnx_bytes_problem(&[]), None);
    }

    #[test]
    fn wokada_metadata_names_the_content_encoder() {
        let meta = parse_export_metadata(
            r#"{"application":"VC_CLIENT","version":"2.1","modelType":"pyTorchRVCv2","samplingRate":40000,"f0":true,
                "embChannels":768,"embedder":"hubert-base-japanese","embOutputLayer":12,"useFinalProj":false}"#,
        )
        .unwrap();
        assert_eq!((meta.sample_rate, meta.emb_channels, meta.if_f0), (Some(40_000), Some(768), Some(true)));
        assert_eq!(meta.embedder.as_deref(), Some("hubert-base-japanese"));
        assert_eq!(wokada_embedder_name("hubert-base-japanese"), Ok(Some("hubert-ja")));
        assert_eq!(wokada_embedder_name("hubert_base"), Ok(None), "RVC's hubert_base is ContentVec");
        assert_eq!(wokada_embedder_name("contentvec"), Ok(None));
        // A name Sigil does not know is a warning that names it (P5-D review).
        let warning = wokada_embedder_name(" whisper ").unwrap_err();
        assert!(warning.contains("\"whisper\"") && warning.contains("settings"), "{warning}");
        assert!(parse_export_metadata("[1]").is_none());
        assert!(parse_export_metadata("not json").is_none());
    }

    /// ModelProto `metadata_props` (field 14) appended to `bytes` (protobuf fields may come in any
    /// order).
    fn with_metadata(mut bytes: Vec<u8>, key: &str, value: &str) -> Vec<u8> {
        fn varint(out: &mut Vec<u8>, mut v: u64) {
            while v >= 0x80 {
                out.push((v as u8) | 0x80);
                v >>= 7;
            }
            out.push(v as u8);
        }
        let mut entry = vec![0x0A];
        varint(&mut entry, key.len() as u64);
        entry.extend_from_slice(key.as_bytes());
        entry.push(0x12);
        varint(&mut entry, value.len() as u64);
        entry.extend_from_slice(value.as_bytes());
        bytes.push(0x72);
        varint(&mut bytes, entry.len() as u64);
        bytes.extend_from_slice(&entry);
        bytes
    }

    /// w-okada's content encoder layout (fp16 audio in; `units9` 256 / `unit12` 768 / `unit12s` out,
    /// all fp16): one conv front end, a 768 → 256 projection.
    fn wokada_encoder() -> Vec<u8> {
        let emb = test_models::EMB as u64;
        let w: Vec<f32> = (0..test_models::EMB * 400).map(|i| ((i % 7) as f32 - 3.0) * 1e-3).collect();
        let proj: Vec<f32> = (0..test_models::EMB * 256).map(|i| ((i % 5) as f32 - 2.0) * 1e-3).collect();
        let mut g = Graph::new();
        g.input("audio", FLOAT16, &[Dim::Fixed(1), Dim::Named("n")])
            .output("units9", FLOAT16, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(256)])
            .output("unit12", FLOAT16, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(emb)])
            .output("unit12s", FLOAT16, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(emb)])
            .init_i64("axes1", &[1], &[1])
            .init_f32("conv_w", &[emb, 1, 400], &w)
            .init_f32("proj", &[emb, 256], &proj)
            .init_f32("half", &[1], &[0.5])
            .node("Cast", &["audio"], &["a32"], &[Attr::Int("to", FLOAT as i64)])
            .node("Unsqueeze", &["a32", "axes1"], &["x3"], &[])
            .node(
                "Conv",
                &["x3", "conv_w"],
                &["c"],
                &[Attr::Ints("strides", vec![320]), Attr::Ints("kernel_shape", vec![400]), Attr::Int("group", 1)],
            )
            .node("Transpose", &["c"], &["f12"], &[Attr::Ints("perm", vec![0, 2, 1])])
            .node("MatMul", &["f12", "proj"], &["f9"], &[])
            .node("Mul", &["f12", "half"], &["f12s"], &[])
            .node("Cast", &["f9"], &["units9"], &[Attr::Int("to", FLOAT16 as i64)])
            .node("Cast", &["f12"], &["unit12"], &[Attr::Int("to", FLOAT16 as i64)])
            .node("Cast", &["f12s"], &["unit12s"], &[Attr::Int("to", FLOAT16 as i64)]);
        g.model_bytes(13)
    }

    /// The fake generator with `emb`-wide features (w-okada names: feats / p_len / pitch / pitchf /
    /// sid) and optional w-okada metadata.
    fn wokada_generator(emb: u64, metadata: Option<&str>) -> Vec<u8> {
        let mut g = Graph::new();
        g.input("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(emb)])
            .input("p_len", INT64, &[Dim::Fixed(1)])
            .input("pitch", INT64, &[Dim::Fixed(1), Dim::Named("t")])
            .input("pitchf", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
            .input("sid", INT64, &[Dim::Fixed(1)])
            .output("audio", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1), Dim::Named("s")])
            .init_i64("axes2", &[1], &[2])
            .init_i64("repeats", &[3], &[1, 1, test_models::HOP as i64])
            .init_i64("flat", &[3], &[1, 1, -1])
            .init_f32("gain", &[1], &[test_models::GAIN])
            .node("Unsqueeze", &["pitchf", "axes2"], &["p3"], &[])
            .node("Tile", &["p3", "repeats"], &["tiled"], &[])
            .node("Reshape", &["tiled", "flat"], &["r"], &[])
            .node("Mul", &["r", "gain"], &["audio"], &[]);
        let bytes = g.model_bytes(13);
        match metadata {
            Some(m) => with_metadata(bytes, "metadata", m),
            None => bytes,
        }
    }

    fn load_slot(slot: &Path, hubert: &str) -> Result<RvcModel, String> {
        RvcModel::load(slot, hubert, Box::new(DioPitchExtractor::new()), "cpu", &VcSessionOptions::default(), false)
    }

    fn convert_tone(model: &mut RvcModel) -> Vec<f32> {
        let audio: Vec<f32> = (0..16_000).map(|i| 0.3 * (std::f32::consts::TAU * 150.0 * i as f32 / 16_000.0).sin()).collect();
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: 1_600,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: Default::default(),
        };
        model.convert(&ctx).unwrap()
    }

    /// Empty the fake slot of `test_models::write_slot`.
    fn empty_slot(slot: &Path) {
        for entry in std::fs::read_dir(slot).unwrap().flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }

    /// w-okada's three-output fp16 ContentVec serves a v2 voice (its `unit12`) and a v1 voice (its
    /// `units9`); before P5-D the legacy path always read output 0 (`units9`) and refused v2 voices.
    #[test]
    fn wokada_content_encoders_serve_v1_and_v2_voices() {
        for (tag, emb) in [("wokada-v2", 768u64), ("wokada-v1", 256)] {
            let (root, slot, _) = test_models::write_slot(tag);
            empty_slot(&slot);
            std::fs::write(slot.join("gen.onnx"), wokada_generator(emb, Some(r#"{"samplingRate": 40000}"#))).unwrap();
            let encoder = root.join("contentvec-h.onnx");
            std::fs::write(&encoder, wokada_encoder()).unwrap();
            let mut model = load_slot(&slot, encoder.to_str().unwrap()).unwrap_or_else(|e| panic!("{tag}: {e}"));
            assert_eq!(model.hubert_output, if emb == 768 { 1 } else { 0 }, "{tag}");
            let out = convert_tone(&mut model);
            assert_eq!(out.len(), 4_000, "{tag}");
            let near = out.iter().filter(|x| (**x - 0.15).abs() < 0.05).count();
            assert!(near > 2_000, "{tag}: {near}");
            drop(model);
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// A w-okada export whose metadata names the Japanese HuBERT runs on w-okada's `hubert-jp`
    /// export next to it (or the installed `vc/embedder-hubert-ja`), not the settings' ContentVec.
    #[test]
    fn wokada_jp_voices_use_the_japanese_encoder() {
        use crate::services::neural_audio::vc::assets::{model_file, EMBEDDER_HUBERT_JA};
        let (root, slot, _) = test_models::write_slot("wokada-jp");
        empty_slot(&slot);
        let meta = r#"{"samplingRate": 40000, "f0": true, "embChannels": 768, "embedder": "hubert-base-japanese", "embOutputLayer": 12}"#;
        std::fs::write(slot.join("gen.onnx"), wokada_generator(768, Some(meta))).unwrap();
        // Without the Japanese encoder anywhere, the error names what to install.
        let installed = [None, Some("fp32"), Some("fp16")].iter().any(|v| model_file(EMBEDDER_HUBERT_JA, *v).is_some());
        if !installed {
            let err = load_slot(&slot, "").err().expect("no Japanese encoder");
            assert!(err.contains("hubert-ja"), "{err}");
            // A settings path that exists (the user's own Japanese HuBERT export) is used, with a
            // warning, as before P5-D (P5-D review).
            let own = root.join("japanese-hubert.onnx");
            std::fs::write(&own, test_models::content_encoder_bytes()).unwrap();
            let mut model = load_slot(&slot, own.to_str().unwrap()).expect("the settings' encoder serves an inferred JP voice");
            assert!(model.warnings.iter().any(|w| w.contains("hubert-ja") && w.contains("settings")), "{:?}", model.warnings);
            assert_eq!(convert_tone(&mut model).len(), 4_000);
        }
        std::fs::write(slot.join("hubert-jp-f.onnx"), test_models::content_encoder_bytes()).unwrap();
        let mut model = load_slot(&slot, "").expect("the slot's hubert-jp export is used, the settings' path is empty");
        assert_eq!(convert_tone(&mut model).len(), 4_000);
        drop(model);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A manifest.json that names the Japanese HuBERT keeps refusing to run on the settings'
    /// encoder when it is not installed (only inferred slots fall back; P5-D review).
    #[test]
    fn manifest_named_encoders_do_not_fall_back() {
        use crate::services::neural_audio::vc::assets::{model_file, EMBEDDER_HUBERT_JA};
        let installed = [None, Some("fp32"), Some("fp16")].iter().any(|v| model_file(EMBEDDER_HUBERT_JA, *v).is_some());
        if installed {
            eprintln!("vc/embedder-hubert-ja is installed on this PC; the missing-encoder path is not reachable");
            return;
        }
        let (root, slot, hubert) = test_models::write_slot("jp-manifest");
        let m = RvcManifest { embedder: Some("hubert-ja".into()), emb_channels: 768, ..RvcManifest::default() };
        let mut warnings = Vec::new();
        let err = slot_content_encoder(&m, false, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap_err();
        assert!(err.contains("hubert-ja"), "{err}");
        // The same embedder from w-okada metadata (an inferred slot) falls back with a warning.
        let p = slot_content_encoder(&m, true, &slot, hubert.to_str().unwrap(), "cpu", &mut warnings).unwrap();
        assert_eq!(p, hubert);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        // ... but never to an empty or missing settings path.
        assert!(slot_content_encoder(&m, true, &slot, "", "cpu", &mut warnings).unwrap_err().contains("hubert-ja"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An inferred w-okada export naming a content encoder Sigil does not know runs on the settings'
    /// encoder and says so (P5-D review: it used to load silently).
    #[test]
    fn unknown_wokada_embedders_are_a_warning() {
        let (root, slot, hubert) = test_models::write_slot("wokada-whisper");
        empty_slot(&slot);
        let meta = r#"{"samplingRate": 40000, "f0": true, "embChannels": 768, "embedder": "whisper"}"#;
        std::fs::write(slot.join("gen.onnx"), wokada_generator(768, Some(meta))).unwrap();
        let model = load_slot(&slot, hubert.to_str().unwrap()).expect("the settings' encoder");
        assert!(model.warnings.iter().any(|w| w.contains("\"whisper\"")), "{:?}", model.warnings);
        assert_eq!(model.manifest.embedder, None);
        drop(model);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A so-vits-svc export, a content encoder and a PyTorch file in an RVC slot are refused by what
    /// they are, and so are a checkpoint or a pitch model set as the HuBERT / ContentVec path.
    #[test]
    fn foreign_files_in_an_rvc_slot_are_refused_specifically() {
        let (root, slot, hubert) = test_models::write_slot("foreign");
        let gen = slot.join("fake_v2_40k.onnx");
        let sovits = {
            let mut g = Graph::new();
            g.input("c", FLOAT, &[Dim::Fixed(1), Dim::Named("s"), Dim::Fixed(768)])
                .input("f0", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
                .input("mel2ph", INT64, &[Dim::Fixed(1), Dim::Named("t")])
                .input("uv", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
                .input("noise", FLOAT, &[Dim::Fixed(1), Dim::Fixed(192), Dim::Named("t")])
                .input("sid", INT64, &[Dim::Fixed(1)])
                .output("audio", FLOAT, &[Dim::Fixed(1), Dim::Named("t")])
                .node("Identity", &["f0"], &["audio"], &[]);
            g.model_bytes(13)
        };
        std::fs::write(&gen, sovits).unwrap();
        let err = load_slot(&slot, hubert.to_str().unwrap()).err().expect("a so-vits-svc export is not an RVC generator");
        assert!(err.contains("so-vits-svc") && err.contains("not an RVC generator"), "{err}");
        std::fs::write(&gen, test_models::content_encoder_bytes()).unwrap();
        let err = load_slot(&slot, hubert.to_str().unwrap()).err().expect("a content encoder is not a voice");
        assert!(err.contains("content encoder"), "{err}");
        std::fs::write(&gen, b"PK\x03\x04 torch").unwrap();
        let err = load_slot(&slot, hubert.to_str().unwrap()).err().expect("a PyTorch file is not ONNX");
        assert!(err.contains("PyTorch") && err.contains("Add voice"), "{err}");
        // The settings' HuBERT path holding a checkpoint or a pitch model.
        std::fs::write(&gen, test_models::generator_bytes()).unwrap();
        let pt = root.join("checkpoint_best_legacy_500.pt");
        std::fs::write(&pt, b"PK\x03\x04 fairseq").unwrap();
        let err = load_slot(&slot, pt.to_str().unwrap()).err().expect("a PyTorch content encoder");
        assert!(err.starts_with("HuBERT / ContentVec") && err.contains("PyTorch"), "{err}");
        let crepe = {
            let mut g = Graph::new();
            g.input("frames", FLOAT, &[Dim::Named("n"), Dim::Fixed(1024)])
                .output("probabilities", FLOAT, &[Dim::Named("n"), Dim::Fixed(360)])
                .init_i64("axes", &[1], &[1])
                .init_i64("starts", &[1], &[0])
                .init_i64("ends", &[1], &[360])
                .node("Slice", &["frames", "starts", "ends", "axes"], &["probabilities"], &[]);
            g.model_bytes(13)
        };
        let p = root.join("tiny.onnx");
        std::fs::write(&p, crepe).unwrap();
        let err = load_slot(&slot, p.to_str().unwrap()).err().expect("a pitch model is not a content encoder");
        assert!(err.contains("pitch model"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
