//! RVC generator + HuBERT content encoder (multi-graph ONNX).

use std::borrow::Cow;
use std::path::Path;

use ort::session::{Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::TensorRef;

use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};

use super::bases::{load_or_infer_manifest, DEFAULT_INFERRED_SAMPLE_RATE};
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
}

impl RvcPresetProfile {
    /// Infallible parse of the persisted preset id (unknown ids fall back to Streamer).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "low_latency" => Self::LowLatency,
            "quality" => Self::Quality,
            _ => Self::Streamer,
        }
    }

    pub fn default_ep(&self) -> &'static str {
        match self {
            Self::Streamer => "cpu",
            Self::LowLatency | Self::Quality => "cuda",
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
        _ => None,
    }
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
        let mut phone_dim = None;
        let mut noise_channels = LATENT;
        for (input, role) in inputs.iter().zip(roles.iter()) {
            let expected = match role {
                GenRole::Phone | GenRole::PitchF | GenRole::Noise => TensorElementType::Float32,
                GenRole::PhoneLengths | GenRole::Pitch | GenRole::Speaker => TensorElementType::Int64,
            };
            let actual = input.dtype().tensor_type();
            if actual != Some(expected) {
                return Err(format!(
                    "RVC generator {} input '{}' is {:?}, expected {:?}{}",
                    path.display(),
                    input.name(),
                    actual,
                    expected,
                    if actual == Some(TensorElementType::Float16) {
                        " (fp16 exports are not supported; export the model in fp32)"
                    } else {
                        ""
                    }
                ));
            }
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
            phone_dim,
            noise_channels,
        })
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
    warnings: Vec<String>,
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
        if hubert_path.trim().is_empty() || !Path::new(hubert_path).is_file() {
            return Err(format!("HuBERT / ContentVec missing: {}", hubert_path));
        }

        let hubert = session_builder_for_vc(ep_hint, opts)?
            .component("vc.content")
            .commit_from_file(hubert_path)
            .map_err(|e| format!("HuBERT load: {}", e))?;
        let generator = session_builder_for_vc(ep_hint, opts)?
            .component("vc")
            .commit_from_file(&gen_path)
            .map_err(|e| format!("RVC generator load: {}", e))?;
        let gen_inputs = GeneratorInputs::resolve(&generator, &gen_path)?;
        let has_pitch = gen_inputs.has(GenRole::Pitch);
        let metadata = read_export_metadata(&generator);
        let mut warnings = Vec::new();

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
            generator,
            gen_inputs,
            pitch,
            index,
            out_sr: manifest.sample_rate,
            wants_f0: has_pitch,
            silence_rms: 0.008,
            silence_skip,
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
        };
        self.convert_inner(&ctx, false).map(|_| ())
    }

    fn probe_output_rate(&mut self) -> Result<u32, String> {
        let emb = self.manifest.emb_channels as usize;
        let phone = vec![0.0f32; PROBE_FRAMES * emb];
        let pitch = vec![1i64; PROBE_FRAMES];
        let pitchf = vec![0.0f32; PROBE_FRAMES];
        let audio = self.run_generator(&phone, PROBE_FRAMES, &pitch, &pitchf, self.manifest.speaker_id)?;
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

    /// HuBERT features, time-major `[frames * emb]`.
    fn run_hubert(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let n = mono_16k.len();
        let tensor = TensorRef::from_array_view((vec![1usize, n], mono_16k)).map_err(|e| format!("HuBERT tensor: {}", e))?;
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
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("HuBERT out: {}", e))?;

        // Flatten to [T*C] in time-major order. Accepted shapes: [1,T,C], [T,C], [1,C,T].
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
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
            [.., c] if *c == emb => Ok(data.to_vec()),
            _ => Err(format!(
                "HuBERT output shape {:?} does not carry the model's {} embedding channels \
                 (v1 models need a 256-dim final_proj ContentVec, v2 a 768-dim one)",
                dims, emb
            )),
        }
    }

    /// Run the generator on `t` frames, feeding each declared input by role.
    fn run_generator(&mut self, phone: &[f32], t: usize, pitch: &[i64], pitchf: &[f32], speaker_id: i64) -> Result<Vec<f32>, String> {
        let emb = self.manifest.emb_channels as usize;
        let lengths = [t as i64];
        let speaker = [speaker_id];
        let noise_ch = self.gen_inputs.noise_channels;
        let noise = if self.gen_inputs.has(GenRole::Noise) {
            vec![0.0f32; noise_ch * t]
        } else {
            Vec::new()
        };
        let tensor_err = |what: &str, e: ort::Error| format!("{what} tensor: {e}");
        let mut feed: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(6);
        for (name, role) in self
            .gen_inputs
            .names
            .iter()
            .zip(self.gen_inputs.roles.iter())
        {
            let value: SessionInputValue<'_> = match role {
                GenRole::Phone => TensorRef::from_array_view((vec![1, t, emb], phone))
                    .map_err(|e| tensor_err("phone", e))?
                    .into(),
                GenRole::PhoneLengths => TensorRef::from_array_view((vec![1usize], &lengths[..]))
                    .map_err(|e| tensor_err("phone_lengths", e))?
                    .into(),
                GenRole::Pitch => TensorRef::from_array_view((vec![1, t], pitch))
                    .map_err(|e| tensor_err("pitch", e))?
                    .into(),
                GenRole::PitchF => TensorRef::from_array_view((vec![1, t], pitchf))
                    .map_err(|e| tensor_err("pitchf", e))?
                    .into(),
                GenRole::Speaker => TensorRef::from_array_view((vec![1usize], &speaker[..]))
                    .map_err(|e| tensor_err("speaker id", e))?
                    .into(),
                GenRole::Noise => TensorRef::from_array_view((vec![1, noise_ch, t], &noise[..]))
                    .map_err(|e| tensor_err("rnd", e))?
                    .into(),
            };
            feed.push((name.clone(), value));
        }
        let outputs = self
            .generator
            .run(feed)
            .map_err(|e| format!("RVC generator run: {}", e))?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RVC audio out: {}", e))?;
        Ok(data.to_vec())
    }

    /// RVC `f0_to_coarse` (256 bins, mel-scale); unvoiced maps to 1 (bins 1..=255).
    fn f0_to_coarse(hz: f32) -> i64 {
        // NaN is unvoiced too (it would otherwise reach the generator as coarse bin 0).
        if hz.is_nan() || hz <= 0.0 {
            return 1;
        }
        let f0_mel_min = 1127.0 * (1.0 + F0_MIN_HZ / 700.0).ln();
        let f0_mel_max = 1127.0 * (1.0 + F0_MAX_HZ / 700.0).ln();
        let f0_mel = 1127.0 * (1.0 + hz / 700.0).ln();
        let f0_mel = (f0_mel - f0_mel_min) * (F0_BIN - 2.0) / (f0_mel_max - f0_mel_min) + 1.0;
        f0_mel.clamp(1.0, F0_BIN - 1.0).round() as i64
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

        let full = self.run_generator(&phone, p_len, &pitch, &pitchf, ctx.speaker_id)?;
        let n = out_len.min(full.len());
        Ok(full[full.len() - n..].to_vec())
    }
}

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
        assert_eq!(role_for_name("x.1"), None);
        assert_eq!(positional_roles(5).map(|r| r.len()), Some(5));
        assert!(positional_roles(2).is_none());
    }
}
