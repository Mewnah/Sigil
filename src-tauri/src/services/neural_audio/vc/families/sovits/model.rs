//! A so-vits-svc voice behind the shared stream harness (P5-D).
//!
//! Every tick (fixed blocks: the harness hands exactly `block` new 16 kHz samples and one fixed
//! context, as for template voices, so every session runs one shape):
//! 1. content features over the whole context from the shared encoder pool (ContentVec 768 / 256,
//!    `engine::content`), with the optional feature-retrieval blend (`cluster_ratio`, a FAISS
//!    `.index` in the slot, RVC's k = 8 / `1/d²` search, which is so-vits-svc 4.1's too);
//! 2. F0 on the 10 ms grid from the voice's pitch extractor through the incremental F0 cache
//!    (`f0_cache` lever; else the whole window);
//! 3. the model's frame grid (`hop / sr`): `T` frames ending exactly at the newest sample (or the
//!    lookahead before it). Per frame, exactly as so-vits-svc's inference prepares them: `f0` by
//!    linear interpolation in time through the voiced 10 ms frames (edges hold, so unvoiced frames
//!    carry the interpolated pitch), `uv` from the 10 ms frame the frame starts in, `mel2ph` = the
//!    content frame the frame starts in (`floor(t · 50) + 1`, so-vits-svc's `repeat_expand_2d`),
//!    `vol` = the input's RMS over one hop around the frame, the pitch shift on `f0`;
//! 4. the flow noise `[1, 192, T]`: time-anchored (`anchored_noise` lever) on the absolute model
//!    frame grid; the grid moves by a fraction of a frame per tick, so a frame between two anchored
//!    frames gets their variance-preserving blend (as the legacy RVC path does on its 10 ms grid),
//!    scaled by so-vits-svc's 0.4 at the default `noise_scale` (the control is relative: 0.667 =
//!    the family default); without the lever, fresh noise every tick (upstream);
//! 5. the generator over the whole window (so-vits-svc exports have no partial decode), the
//!    returned span cut from its end;
//! 6. the post stage at the model rate: rms_mix, the true formant shift, the silence gate (with the
//!    silence skip) and the limiter, as template voices have it.

use std::path::{Path, PathBuf};
use std::sync::PoisonError;

use half::f16;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{DynValue, TensorRef};

use crate::services::neural_audio::ort_session::{build_session, ModelSource};
use crate::services::neural_audio::vc::engine::content::{acquire_for_slot, EmbedderKind, SharedEncoder};
use crate::services::neural_audio::vc::engine::f0_cache::{F0Cache, F0CacheConfig, DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES, HOP_16K};
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream, FRAME_STRIDE};
use crate::services::neural_audio::vc::engine::options::{Accelerator, EffectiveLevers, Lever};
use crate::services::neural_audio::vc::engine::plan::{block_frames, MAX_CONTEXT_FRAMES, MAX_LOOKAHEAD_FRAMES};
use crate::services::neural_audio::vc::engine::post::{PostConfig, PostStage};
use crate::services::neural_audio::vc::engine::presets::engine_preset;
use crate::services::neural_audio::vc::faiss::{BlendScratch, FaissIndex};
use crate::services::neural_audio::vc::import::slot::safe_file_name;
use crate::services::neural_audio::vc::rvc::{non_onnx_file_problem, MIN_CONTEXT_16K};
use crate::services::neural_audio::vc::traits::{ConversionCtx, PitchExtractor, VoiceControls, VoiceConversionModel, DEFAULT_NOISE_SCALE};

use super::super::{FamilyEngine, FamilyRequest};
use super::config::SovitsRecord;
use super::graph::{Role, SovitsGraph, SpeakerInput};
use super::SovitsOptions;

/// so-vits-svc's inference noise scale (`noice_scale` 0.4), what the default `noise_scale` control
/// runs.
pub const SOVITS_NOISE_SCALE: f32 = 0.4;
/// ContentVec's frame hop at 16 kHz (50 Hz).
const CONTENT_HOP_16K: f64 = 320.0;
/// Engine-level silence skip: quiet ticks after this much quiet are not converted (template voices'
/// value).
const SILENCE_HANGOVER_MS: f64 = 80.0;
/// Output the silence gate lets ring out after the input falls quiet (template voices' value).
const POST_GATE_HANGOVER_MS: f32 = 150.0;
/// Longest context (template voices' limit, 4 s).
const MAX_CONTEXT_16K: usize = MAX_CONTEXT_FRAMES * HOP_16K;

fn db_to_rms(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Model frames over `n` 16 kHz samples at `sr` / `hop` (the window never starts before the context).
fn frames_over(n: usize, sr: u32, hop: usize) -> usize {
    ((n as u64 * u64::from(sr)) / (16_000 * hop as u64)) as usize
}

/// The content frame (0-based, of `s`) model frame starting at 16 kHz sample `pos` of the context
/// takes: the 50 Hz frame it starts in (so-vits-svc's `repeat_expand_2d`), clamped to the frames
/// the encoder returned.
fn content_frame(pos: f64, s: usize) -> usize {
    ((pos / CONTENT_HOP_16K).floor().max(0.0) as usize).min(s.saturating_sub(1))
}

/// Flow noise `[c, t]` (channel-major, into `out`) for the model frames at absolute positions
/// `a0 + j` (fractional frames since the stream's start): frame `j` is the variance-preserving blend
/// `cos(θ)·n[k] + sin(θ)·n[k + 1]` of the anchored unit noise of absolute frames `k = floor(a0) + j`
/// and `k + 1`, `θ = frac(a0)·π/2`, times `scale`. Overlapping ticks agree wherever their grids
/// coincide, and a sub-frame shift moves the noise continuously instead of redrawing it. `cache` (a
/// stream-sized [`NoiseFrames`] of the same seed) generates only frames it has not seen; without it
/// (a one-off call longer than any stream window) the values are generated directly.
#[allow(clippy::too_many_arguments)]
fn anchored_window(cache: Option<&mut NoiseFrames>, seed: u64, a0: f64, t: usize, c: usize, scale: f32, tmp: &mut Vec<f32>, out: &mut [f32]) {
    let k0 = a0.floor().max(0.0);
    let frac = (a0 - k0).clamp(0.0, 1.0);
    let theta = frac * std::f64::consts::FRAC_PI_2;
    let (w0, w1) = (theta.cos() as f32, theta.sin() as f32);
    tmp.clear();
    tmp.resize((t + 1) * c, 0.0);
    match cache {
        Some(frames) => frames.copy_window(k0 as u64, t + 1, scale, false, tmp),
        None => {
            let gen = AnchoredNoise::new(seed);
            for (f, row) in tmp.chunks_exact_mut(c.max(1)).enumerate() {
                for (ch, v) in row.iter_mut().enumerate() {
                    let index = (ch as u64).wrapping_mul(FRAME_STRIDE).wrapping_add(k0 as u64 + f as u64);
                    *v = gen.unit(NoiseStream::Flow, index) * scale;
                }
            }
        }
    }
    for j in 0..t {
        let (r0, r1) = (&tmp[j * c..(j + 1) * c], &tmp[(j + 1) * c..(j + 2) * c]);
        for ch in 0..c {
            out[ch * t + j] = w0 * r0[ch] + w1 * r1[ch];
        }
    }
}

/// The flow-noise scale of a `noise_scale` control (relative: the default control is so-vits-svc's
/// 0.4).
pub fn noise_scale_for(control: f32) -> f32 {
    let c = if control.is_finite() { control.max(0.0) } else { DEFAULT_NOISE_SCALE };
    (SOVITS_NOISE_SCALE * c / DEFAULT_NOISE_SCALE).min(1.0)
}

/// so-vits-svc's F0 preparation onto the model grid (`post_process` of its F0 predictors): `f0[j]`
/// is the time-linear interpolation through the voiced 10 ms frames (holding the first / last
/// voiced value beyond them; all 0 without any), `uv[j]` whether the 10 ms frame `floor(p[j])` is
/// voiced, where `p[j]` is model frame `j`'s position in 10 ms frames (ascending).
fn grid_f0(f0_10: &[f32], positions: impl Iterator<Item = f64>, f0: &mut Vec<f32>, uv: &mut Vec<f32>) {
    f0.clear();
    uv.clear();
    let voiced = |k: usize| f0_10[k] > 0.0 && f0_10[k].is_finite();
    let first = (0..f0_10.len()).find(|&k| voiced(k));
    let last = (0..f0_10.len()).rev().find(|&k| voiced(k));
    // `lo` = the last voiced frame at or before p; `hi` = the first voiced after it.
    let mut lo: Option<usize> = None;
    let mut hi = first;
    for p in positions {
        let idx = if p <= 0.0 { 0 } else { (p.floor() as usize).min(f0_10.len().saturating_sub(1)) };
        uv.push(if !f0_10.is_empty() && voiced(idx) { 1.0 } else { 0.0 });
        let (Some(fv), Some(lv)) = (first, last) else {
            f0.push(0.0);
            continue;
        };
        while let Some(h) = hi {
            if (h as f64) <= p {
                lo = Some(h);
                hi = (h + 1..f0_10.len()).find(|&k| voiced(k));
            } else {
                break;
            }
        }
        let v = match (lo, hi) {
            (None, _) => f0_10[fv],
            (Some(_), None) => f0_10[lv],
            (Some(a), Some(b)) => {
                let w = ((p - a as f64) / (b - a) as f64) as f32;
                f0_10[a] + (f0_10[b] - f0_10[a]) * w
            }
        };
        f0.push(v);
    }
}

pub struct SovitsModel {
    generator: Session,
    graph: SovitsGraph,
    /// Declared rank of each input (rank-1 `[T]` exports of f0 / uv / vol exist).
    ranks: Vec<usize>,
    encoder: SharedEncoder,
    width: usize,
    pitch: Box<dyn PitchExtractor>,
    f0_cache: F0Cache,
    f0_cache_on: bool,
    sr: u32,
    hop: usize,
    /// 16 kHz samples per model frame.
    h16: f64,
    /// Speakers the voice has (0 = unknown).
    speakers: usize,
    /// Width of a speaker-mix input.
    mix_width: usize,
    retrieval: Option<FaissIndex>,
    cluster_ratio: f32,
    retrieval_on: bool,
    blend: BlendScratch,
    noise_seed: u64,
    /// Unit flow noise per absolute model frame (stream ticks); `noise_capacity` frames.
    noise_frames: NoiseFrames,
    noise_capacity: usize,
    anchored: bool,
    ticks: u64,
    controls: VoiceControls,
    post: PostStage,
    silence_skip: bool,
    quiet_ticks: u32,
    block_16k: usize,
    lookahead_16k: usize,
    extra_16k: usize,
    // Per-tick buffers (capacity kept across ticks).
    feats: Vec<f32>,
    content: Vec<f32>,
    f0_10: Vec<f32>,
    f0: Vec<f32>,
    uv: Vec<f32>,
    uv_i: Vec<i64>,
    mel2ph: Vec<i64>,
    vol: Vec<f32>,
    noise: Vec<f32>,
    noise_tmp: Vec<f32>,
    mix: Vec<f32>,
    /// A one-off call's front-padded context.
    padded: Vec<f32>,
    halves: [Vec<f16>; 6],
}

/// Indices into `SovitsModel::halves`.
const H_CONTENT: usize = 0;
const H_F0: usize = 1;
const H_UV: usize = 2;
const H_NOISE: usize = 3;
const H_VOL: usize = 4;
const H_MIX: usize = 5;

pub(super) fn to_half(src: &[f32], dst: &mut Vec<f16>) {
    dst.clear();
    dst.extend(src.iter().map(|&x| f16::from_f32(x)));
}

/// A float input: fp16 (`half`) or fp32.
pub(super) fn float_input<'a>(shape: Vec<usize>, data: &'a [f32], half_data: &'a [f16], half: bool, what: &str) -> Result<SessionInputValue<'a>, String> {
    if half {
        Ok(TensorRef::from_array_view((shape, half_data)).map_err(|e| format!("{what} tensor: {e}"))?.into())
    } else {
        Ok(TensorRef::from_array_view((shape, data)).map_err(|e| format!("{what} tensor: {e}"))?.into())
    }
}

/// The audio output as f32 (fp16 exports converted), its last `n + skip_end` samples minus the
/// last `skip_end`.
fn tail_of(value: &DynValue, n: usize, skip_end: usize) -> Result<Vec<f32>, String> {
    let need = n + skip_end;
    if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
        if data.len() < need {
            return Err(format!("the so-vits-svc generator returned {} samples, {need} needed", data.len()));
        }
        return Ok(data[data.len() - need..data.len() - skip_end].to_vec());
    }
    let (_, data) = value.try_extract_tensor::<f16>().map_err(|e| format!("so-vits-svc output: {e}"))?;
    if data.len() < need {
        return Err(format!("the so-vits-svc generator returned {} samples, {need} needed", data.len()));
    }
    Ok(data[data.len() - need..data.len() - skip_end].iter().map(|h| h.to_f32()).collect())
}

/// The generator file of a slot (a plain file name in the slot, present and not a PyTorch file).
pub fn generator_file(slot_dir: &Path, generator: &str) -> Result<PathBuf, String> {
    let g = generator.trim();
    if !safe_file_name(g) {
        return Err(format!("the voice's generator name {g:?} is not a file in its folder; add the voice again"));
    }
    let path = slot_dir.join(g);
    if !path.is_file() {
        return Err(format!("the so-vits-svc generator {g} is missing in {}; add the voice again", slot_dir.display()));
    }
    if let Some(problem) = non_onnx_file_problem(&path) {
        return Err(format!("{problem}; add the voice's ONNX export again"));
    }
    Ok(path)
}

impl SovitsModel {
    /// Load the voice of `req` (see the module docs); the engine the family registry returns.
    pub fn load(mut req: FamilyRequest) -> Result<FamilyEngine, String> {
        let m = req.manifest.clone();
        let record = SovitsRecord::load(&req.slot_dir)?;
        let gen_path = generator_file(&req.slot_dir, &m.generator)?;
        let mut warnings = Vec::new();
        let (generator, gen_outcome) = build_session(Some("vc"), &req.ep, &req.session, ModelSource::File(&gen_path))
            .map_err(|e| format!("so-vits-svc generator load: {e}"))?;
        let graph = SovitsGraph::resolve(&generator, &gen_path)?;
        let ranks = generator
            .inputs()
            .iter()
            .map(|i| i.dtype().tensor_shape().map(|s| s.len()).unwrap_or(2))
            .collect();
        let width = match graph.content_width {
            Some(w) if w as u32 != m.emb_channels => {
                warnings.push(format!("the voice's manifest says {}-wide content but its graph takes {w}; using {w}", m.emb_channels));
                w
            }
            Some(w) => w,
            None => m.emb_channels as usize,
        };
        let kind = EmbedderKind::from_manifest(m.embedder.as_deref(), width as u32)?;
        if !matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) {
            return Err(format!("so-vits-svc voices run on ContentVec; this one names {}", kind.as_str()));
        }
        let sr = m.sample_rate;
        let hop = record.hop as usize;
        if !(8_000..=96_000).contains(&sr) {
            return Err(format!("the voice's sample rate {sr} Hz is not supported"));
        }
        let pitch = req.pitch.take().ok_or("so-vits-svc voices need a pitch extractor")?;

        // The stream shape the user asked for (the harness confirms it in `configure_stream`).
        let preset = engine_preset(req.preset, req.block_ms, req.chunk_frames, req.device_rate, req.extra_context_ms, req.precision);
        let block_16k = block_frames(preset.block_ms)? * HOP_16K;
        let extra_16k = if req.extra_context_ms.is_finite() && req.extra_context_ms > 0.0 {
            (req.extra_context_ms * 16.0).round() as usize
        } else {
            0
        };
        let look_frames = if req.lookahead_ms.is_finite() && req.lookahead_ms > 0.0 {
            ((req.lookahead_ms / 10.0).round() as usize).min(MAX_LOOKAHEAD_FRAMES)
        } else {
            0
        };
        let probe = (extra_16k + block_16k).clamp(MIN_CONTEXT_16K, MAX_CONTEXT_16K);
        let encoder = acquire_for_slot(kind, req.content_encoder.as_deref(), Some(&req.slot_dir), &req.ep, &req.session, req.precision, probe, &mut warnings)?;
        let enc_outcome = encoder.lock().unwrap_or_else(PoisonError::into_inner).outcome().clone();

        let speakers = m.speaker_count.map(|c| c as usize).filter(|&c| c > 0).unwrap_or(record.speakers.len());
        let mix_width = match graph.speaker {
            Some(SpeakerInput::Mix { speakers: Some(n) }) => n,
            Some(SpeakerInput::Mix { speakers: None }) if speakers > 0 => speakers,
            Some(SpeakerInput::Mix { speakers: None }) => {
                return Err("the voice takes speaker weights but its speaker count is unknown; add it again with the MoeVS JSON its export wrote".into())
            }
            _ => 0,
        };

        let options: SovitsOptions = serde_json::from_value(super::super::options_of(&m)).unwrap_or_default();
        let cluster_ratio = if options.cluster_ratio.is_finite() { options.cluster_ratio.clamp(0.0, 1.0) } else { 0.0 };
        let retrieval = match m.index.as_deref() {
            Some(ix) if cluster_ratio > 0.0 => match load_index(&req.slot_dir, ix, width) {
                Ok(i) => Some(i),
                Err(e) => {
                    warnings.push(format!("feature retrieval: {e}; the voice runs without it"));
                    None
                }
            },
            Some(_) => None,
            None if cluster_ratio > 0.0 => {
                warnings.push("feature retrieval: the voice has no feature index (.index), so cluster_ratio has no effect".into());
                None
            }
            None => None,
        };

        // Levers: what this family runs (the rest keep their notes).
        let mut levers: EffectiveLevers = req.levers.clone();
        for lever in Lever::ALL {
            let on = levers.levers.get(lever);
            let used = match lever {
                Lever::AnchoredNoise => on && graph.has(Role::Noise),
                Lever::F0Cache | Lever::Governor => on,
                _ => false,
            };
            if on && !used {
                levers.notes.push(format!("{}: not used by so-vits-svc voices", lever.as_str()));
            }
            levers.levers.set(lever, used);
        }
        if levers.accelerator != Accelerator::None {
            levers
                .notes
                .push(format!("{}: not used by so-vits-svc voices (they run on the plain execution provider)", levers.accelerator.as_str()));
            levers.accelerator = Accelerator::None;
        }
        let anchored = levers.levers.get(Lever::AnchoredNoise);
        let f0_cache_on = levers.levers.get(Lever::F0Cache);

        let (lead, margin) = match pitch.receptive_frames() {
            Some((left, right)) => (left, right.max(1)),
            None => (DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES),
        };
        let f0_cache = F0Cache::new(F0CacheConfig {
            margin_frames: margin,
            capacity_frames: MAX_CONTEXT_FRAMES * 2,
        })
        .with_lead_frames(lead);
        let max_frames = frames_over(MAX_CONTEXT_16K, sr, hop) + 2;
        let noise_seed = req.noise_seed;
        let model = SovitsModel {
            noise_frames: NoiseFrames::new(AnchoredNoise::new(noise_seed), NoiseStream::Flow, graph.noise_channels, max_frames + 1),
            noise_capacity: max_frames + 1,
            generator,
            ranks,
            encoder,
            width,
            pitch,
            f0_cache,
            f0_cache_on,
            sr,
            hop,
            h16: hop as f64 * 16_000.0 / f64::from(sr),
            speakers,
            mix_width,
            retrieval,
            cluster_ratio,
            retrieval_on: true,
            blend: BlendScratch::new(),
            noise_seed,
            anchored,
            ticks: 0,
            controls: VoiceControls::default(),
            post: PostStage::new(PostConfig {
                sample_rate: sr,
                limiter: true,
                silence_gate: req.silence_skip,
                silence_hangover_ms: POST_GATE_HANGOVER_MS,
            }),
            silence_skip: req.silence_skip,
            quiet_ticks: 0,
            block_16k,
            lookahead_16k: look_frames * HOP_16K,
            extra_16k,
            graph,
            feats: Vec::new(),
            content: Vec::new(),
            f0_10: Vec::new(),
            f0: Vec::new(),
            uv: Vec::new(),
            uv_i: Vec::new(),
            mel2ph: Vec::new(),
            vol: Vec::new(),
            noise: Vec::new(),
            noise_tmp: Vec::new(),
            mix: Vec::new(),
            padded: Vec::new(),
            halves: Default::default(),
        };
        let model_delay_16k = model.lookahead_16k;
        Ok(FamilyEngine {
            model: Box::new(model),
            levers,
            outcomes: vec![("vc".to_string(), gen_outcome), ("vc.content".to_string(), enc_outcome)],
            warnings,
            model_delay_16k,
            reference: None,
            min_output_buffer_ms: None,
        })
    }

    /// `round(frames_16k · sr / 16000)`: what every call returns.
    fn output_len(&self, frames_16k: usize) -> usize {
        ((frames_16k as u64 * u64::from(self.sr) + 8_000) / 16_000) as usize
    }

    /// The speaker id fed (clamped to the voice's speakers when known).
    fn speaker(&self, id: i64) -> i64 {
        let id = id.max(0);
        if self.speakers > 0 {
            id.min(self.speakers as i64 - 1)
        } else {
            id
        }
    }

    fn convert_tick(&mut self, ctx: &ConversionCtx<'_>, allow_skip: bool) -> Result<Vec<f32>, String> {
        let audio = ctx.mono_16k;
        let n = audio.len();
        let want = ctx.new_frames.min(n);
        if want == 0 {
            return Ok(Vec::new());
        }
        let out_len = self.output_len(want);
        if allow_skip && self.silence_skip {
            let tail = &audio[n - want..];
            let rms = (tail.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / want as f64).sqrt() as f32;
            if rms < db_to_rms(self.controls.silence_db) {
                self.quiet_ticks = self.quiet_ticks.saturating_add(1);
                let hangover = (SILENCE_HANGOVER_MS / (want.min(self.block_16k.max(1)) as f64 / 16.0)).ceil() as u32;
                if self.quiet_ticks > hangover {
                    return Ok(vec![0.0; out_len]);
                }
            } else {
                self.quiet_ticks = 0;
            }
        }
        if n < MIN_CONTEXT_16K {
            return Err(format!("so-vits-svc needs at least {MIN_CONTEXT_16K} samples of 16 kHz context, got {n}"));
        }
        let look = self.lookahead_16k.min(n - want);
        let look_out = self.output_len(look);
        let need = out_len + look_out;
        // The model window ends at the context's end and never starts before it on a stream tick
        // (`configure_stream` sized the context); a one-off call whose length is not a whole number
        // of model frames (the benchmark's whole-utterance ceiling) gets silence before it instead.
        let fit = frames_over(n, self.sr, self.hop);
        let mut padded = std::mem::take(&mut self.padded);
        let end_abs = ctx.stream_pos_16k + n as u64;
        let result = if fit >= 2 && fit * self.hop >= need {
            self.run_window(ctx, audio, fit, Some(ctx.stream_pos_16k), end_abs, out_len, look_out)
        } else {
            let t = need.div_ceil(self.hop).max(2);
            let pad = (t as f64 * self.h16 - n as f64).ceil().max(0.0) as usize;
            padded.clear();
            padded.resize(pad, 0.0);
            padded.extend_from_slice(audio);
            self.run_window(ctx, &padded, t, None, end_abs, out_len, look_out)
        };
        self.padded = padded;
        let mut out = result?;
        self.ticks = self.ticks.wrapping_add(1);

        // 6. Post stage on the span the returned audio converts.
        let source = &audio[n - look - want..n - look];
        // A post-stage failure keeps the converted audio unchanged (the stage is polish).
        let _ = self.post.process(source, &mut out, &self.controls);
        Ok(out)
    }

    /// Steps 1-5 of a tick (module docs) over `audio` (the context, or a one-off call's padded copy)
    /// with `t` model frames ending at its end: the generator's last `out_len` samples before the
    /// lookahead's `look_out`. `cache_pos` is the context's stream position when the F0 cache may
    /// use it; `end_abs` the absolute 16 kHz position of the window's end (anchored noise).
    #[allow(clippy::too_many_arguments)]
    fn run_window(
        &mut self,
        ctx: &ConversionCtx<'_>,
        audio: &[f32],
        t: usize,
        cache_pos: Option<u64>,
        end_abs: u64,
        out_len: usize,
        look_out: usize,
    ) -> Result<Vec<f32>, String> {
        let n = audio.len();
        // 1. Content features over the context (+ feature retrieval).
        let w = self.width;
        let s = {
            let mut enc = self.encoder.lock().unwrap_or_else(PoisonError::into_inner);
            enc.encode(audio, &mut self.feats)?
        };
        if s == 0 || self.feats.len() < s * w {
            return Err("the content encoder returned no frames".into());
        }
        if let Some(index) = self.retrieval.as_ref() {
            if self.retrieval_on && self.cluster_ratio > 0.0 {
                index.blend_with(&mut self.feats[..s * w], s, self.cluster_ratio, &mut self.blend);
            }
        }

        // 2. F0 on the 10 ms grid.
        let f0_10 = match cache_pos {
            Some(p) if self.f0_cache_on => self.f0_cache.update(self.pitch.as_mut(), audio, p)?,
            _ => self.f0_cache.extract_window(self.pitch.as_mut(), audio)?,
        };
        self.f0_10.clear();
        self.f0_10.extend_from_slice(f0_10);

        // 3. The model grid: frame j starts at `start + j · h16` (16 kHz samples into the context).
        let h16 = self.h16;
        let start = n as f64 - t as f64 * h16;
        let pos = |j: usize| start + j as f64 * h16;
        grid_f0(&self.f0_10, (0..t).map(|j| pos(j) / HOP_16K as f64), &mut self.f0, &mut self.uv);
        let shift = 2f32.powf(ctx.pitch_shift_semitones / 12.0);
        for v in self.f0.iter_mut() {
            let x = *v * shift;
            *v = if x.is_finite() && x > 0.0 { x } else { 0.0 };
        }
        self.mel2ph.clear();
        self.mel2ph.extend((0..t).map(|j| content_frame(pos(j), s) as i64 + 1));
        if self.graph.has(Role::Volume) {
            self.vol.clear();
            for j in 0..t {
                let a = (pos(j) - h16 / 2.0).max(0.0) as usize;
                let b = ((pos(j) + h16 / 2.0).ceil() as usize).clamp(a + 1, n);
                let seg = &audio[a.min(n - 1)..b];
                let e = seg.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / seg.len().max(1) as f64;
                self.vol.push(e.sqrt() as f32);
            }
        }

        // 4. Flow noise [C, T].
        if self.graph.has(Role::Noise) {
            let c = self.graph.noise_channels;
            let scale = noise_scale_for(self.controls.noise_scale);
            self.noise.clear();
            self.noise.resize(c * t, 0.0);
            if self.anchored {
                let a0 = end_abs as f64 / h16 - t as f64;
                let cached = (t < self.noise_capacity).then_some(&mut self.noise_frames);
                anchored_window(cached, self.noise_seed, a0, t, c, scale, &mut self.noise_tmp, &mut self.noise);
            } else {
                let gen = AnchoredNoise::new(self.noise_seed ^ self.ticks.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
                gen.fill(NoiseStream::Flow, 0, scale, &mut self.noise);
            }
        }

        // 5. Content tensor: at the encoder's rate (the export maps it with mel2ph), else upsampled
        // to the model grid here; time- or channel-major as the export declares.
        let (frames_c, rows): (usize, &[f32]) = if self.graph.maps_content() { (s, &self.feats[..s * w]) } else { (t, &[]) };
        self.content.clear();
        if self.graph.maps_content() {
            if self.graph.content_time_major {
                self.content.extend_from_slice(rows);
            } else {
                self.content.resize(w * s, 0.0);
                for (i, row) in rows.chunks_exact(w).enumerate() {
                    for (ch, &v) in row.iter().enumerate() {
                        self.content[ch * s + i] = v;
                    }
                }
            }
        } else {
            self.content.resize(w * t, 0.0);
            for j in 0..t {
                let src = (self.mel2ph[j] - 1) as usize;
                let row = &self.feats[src * w..(src + 1) * w];
                if self.graph.content_time_major {
                    self.content[j * w..(j + 1) * w].copy_from_slice(row);
                } else {
                    for (ch, &v) in row.iter().enumerate() {
                        self.content[ch * t + j] = v;
                    }
                }
            }
        }
        let speaker = self.speaker(ctx.speaker_id);
        if self.mix_width > 0 {
            self.mix.clear();
            self.mix.resize(t * self.mix_width, 0.0);
            let id = (speaker as usize).min(self.mix_width - 1);
            for j in 0..t {
                self.mix[j * self.mix_width + id] = 1.0;
            }
        }
        if self.graph.uv_int {
            self.uv_i.clear();
            self.uv_i.extend(self.uv.iter().map(|&u| u as i64));
        }
        let g = &self.graph;
        let [h_content, h_f0, h_uv, h_noise, h_vol, h_mix] = &mut self.halves;
        if g.half(Role::Content) {
            to_half(&self.content, h_content);
        }
        if g.half(Role::F0) {
            to_half(&self.f0, h_f0);
        }
        if g.half(Role::Uv) {
            to_half(&self.uv, h_uv);
        }
        if g.half(Role::Noise) {
            to_half(&self.noise, h_noise);
        }
        if g.half(Role::Volume) {
            to_half(&self.vol, h_vol);
        }
        if g.half(Role::Speaker) {
            to_half(&self.mix, h_mix);
        }
        let ids = [speaker];
        let lengths = [frames_c as i64];
        let row_shape = |rank: usize| if rank == 1 { vec![t] } else { vec![1, t] };
        let content_shape = if g.content_time_major { vec![1, frames_c, w] } else { vec![1, w, frames_c] };
        let mut feed: Vec<(&str, SessionInputValue<'_>)> = Vec::with_capacity(g.names.len());
        for (i, (name, role)) in g.names.iter().zip(g.roles.iter()).enumerate() {
            let half = g.half(*role);
            let rank = self.ranks[i];
            let value: SessionInputValue<'_> = match role {
                Role::Content => float_input(content_shape.clone(), &self.content, &self.halves[H_CONTENT], half, "c")?,
                Role::F0 => float_input(row_shape(rank), &self.f0, &self.halves[H_F0], half, "f0")?,
                Role::Uv if g.uv_int => TensorRef::from_array_view((row_shape(rank), &self.uv_i[..]))
                    .map_err(|e| format!("uv tensor: {e}"))?
                    .into(),
                Role::Uv => float_input(row_shape(rank), &self.uv, &self.halves[H_UV], half, "uv")?,
                Role::Mel2ph => TensorRef::from_array_view((vec![1, t], &self.mel2ph[..]))
                    .map_err(|e| format!("mel2ph tensor: {e}"))?
                    .into(),
                Role::Noise => float_input(vec![1, g.noise_channels, t], &self.noise, &self.halves[H_NOISE], half, "noise")?,
                Role::Volume => float_input(row_shape(rank), &self.vol, &self.halves[H_VOL], half, "vol")?,
                Role::Speaker => match g.speaker {
                    Some(SpeakerInput::Mix { .. }) => float_input(vec![t, self.mix_width], &self.mix, &self.halves[H_MIX], half, "sid")?,
                    Some(SpeakerInput::Id { rank: 2 }) => TensorRef::from_array_view((vec![1usize, 1], &ids[..]))
                        .map_err(|e| format!("sid tensor: {e}"))?
                        .into(),
                    _ => TensorRef::from_array_view((vec![1usize], &ids[..]))
                        .map_err(|e| format!("sid tensor: {e}"))?
                        .into(),
                },
                Role::Lengths => TensorRef::from_array_view((vec![1usize], &lengths[..]))
                    .map_err(|e| format!("lengths tensor: {e}"))?
                    .into(),
            };
            feed.push((name.as_str(), value));
        }
        let outputs: SessionOutputs<'_> = self.generator.run(feed).map_err(|e| format!("so-vits-svc generator run: {e}"))?;
        tail_of(&outputs[0], out_len, look_out)
    }
}

/// The voice's feature index (`<slot>/<index>`), checked against the content width.
fn load_index(slot_dir: &Path, name: &str, width: usize) -> Result<FaissIndex, String> {
    if !safe_file_name(name) {
        return Err(format!("the index name {name:?} is not a file in the voice's folder"));
    }
    let index = FaissIndex::load(&slot_dir.join(name))?;
    if index.dim() != width {
        return Err(format!("{name} has {} dimensions but the voice uses {width}", index.dim()));
    }
    Ok(index)
}

impl VoiceConversionModel for SovitsModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.convert_tick(ctx, true)
    }

    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    fn output_sample_rate(&self) -> u32 {
        self.sr
    }

    fn min_context_samples(&self) -> usize {
        MIN_CONTEXT_16K
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        // Silence skip bypassed, so the encoder, the pitch extractor and the generator all run at the
        // live shape; the warm-up audio is not the stream, so its F0 frames and post state go.
        let r = self.convert_tick(ctx, false).map(|_| ());
        self.f0_cache.reset();
        self.post.reset();
        self.quiet_ticks = 0;
        r
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.controls = *controls;
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        // Feature retrieval is the step this voice applies itself.
        if self.retrieval.is_some() && self.cluster_ratio > 0.0 && self.retrieval_on != level.retrieval {
            self.retrieval_on = level.retrieval;
            return true;
        }
        false
    }

    fn reset_stream(&mut self) {
        self.f0_cache.reset();
        self.post.reset();
        self.noise_frames.clear();
        self.quiet_ticks = 0;
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.block_16k)
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        if block != self.block_16k {
            return Err(format!("the stream harness asked for {block}-sample blocks, the voice runs {}", self.block_16k));
        }
        let extra = extra.max(self.extra_16k);
        // The returned span plus the lookahead, two model frames of margin, the user's context.
        let need = want + self.lookahead_16k + (2.0 * self.h16).ceil() as usize;
        let context = (extra + block).max(need).max(MIN_CONTEXT_16K);
        // Whole content frames (50 Hz), so ticks frame the content identically in absolute time.
        let context = context.div_ceil(320) * 320;
        if context > MAX_CONTEXT_16K {
            return Err(format!("a context of {} ms is longer than {} ms", context / 16, MAX_CONTEXT_16K / 16));
        }
        Ok(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f0_follows_so_vits_interpolation() {
        // 10 ms frames: unvoiced, 100, unvoiced, 200, unvoiced.
        let f0_10 = [0.0, 100.0, 0.0, 200.0, 0.0];
        let (mut f0, mut uv) = (Vec::new(), Vec::new());
        grid_f0(&f0_10, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 4.9].into_iter(), &mut f0, &mut uv);
        assert_eq!(f0, vec![100.0, 100.0, 150.0, 175.0, 200.0, 200.0, 200.0]);
        assert_eq!(uv, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // No voiced frame: zeros; one: constant.
        grid_f0(&[0.0, 0.0], [0.0, 1.5].into_iter(), &mut f0, &mut uv);
        assert_eq!((f0.clone(), uv.clone()), (vec![0.0, 0.0], vec![0.0, 0.0]));
        grid_f0(&[0.0, 300.0, 0.0], [0.0, 2.0].into_iter(), &mut f0, &mut uv);
        assert_eq!(f0, vec![300.0, 300.0]);
        // Fractional positions between two voiced frames.
        grid_f0(&[100.0, 200.0], [0.25, 0.5].into_iter(), &mut f0, &mut uv);
        assert_eq!(f0, vec![125.0, 150.0]);
        // NaN frames are unvoiced.
        grid_f0(&[f32::NAN, 120.0], [0.0].into_iter(), &mut f0, &mut uv);
        assert_eq!((f0[0], uv[0]), (120.0, 0.0));
    }

    #[test]
    fn noise_scale_is_relative_to_the_family_default() {
        assert!((noise_scale_for(DEFAULT_NOISE_SCALE) - SOVITS_NOISE_SCALE).abs() < 1e-6);
        assert_eq!(noise_scale_for(0.0), 0.0);
        assert!(noise_scale_for(1.0) > SOVITS_NOISE_SCALE && noise_scale_for(1.0) <= 1.0);
        assert!((noise_scale_for(f32::NAN) - SOVITS_NOISE_SCALE).abs() < 1e-6);
    }

    #[test]
    fn the_window_never_starts_before_the_context() {
        for (sr, hop) in [(44_100u32, 512usize), (32_000, 320), (48_000, 480), (22_050, 256)] {
            for n in [720usize, 3_200, 16_000, 16_001, 63_999] {
                let t = frames_over(n, sr, hop);
                let h16 = hop as f64 * 16_000.0 / f64::from(sr);
                assert!(t as f64 * h16 <= n as f64 + 1e-6, "{sr}/{hop} n={n}");
                assert!((t + 1) as f64 * h16 > n as f64, "{sr}/{hop} n={n}: the window covers all but < 1 frame");
            }
        }
    }
}
