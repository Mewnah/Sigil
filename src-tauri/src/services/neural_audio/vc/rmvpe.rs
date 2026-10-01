//! RMVPE ONNX pitch extractor (default for Low latency / Quality GPU presets).
//!
//! Two export contracts are supported, chosen from the graph's declared input at load:
//! - RVC's `rmvpe.onnx`: log-mel `[1, 128, T]` in, salience `[1, T, 360]` out. The log-mel is
//!   computed here exactly like RVC's `MelSpectrogram` (16 kHz, n_fft = win = 1024, hop 160,
//!   periodic Hann, centred reflect padding, 128 HTK/Slaney mels over 30..8000 Hz,
//!   `log(clamp(x, 1e-5))`), with `T` padded to a multiple of 32 as in `mel2hidden`, but
//!   **end-aligned** (P3 final gate, review finding 4 on T5): the audio's frames end the padded mel
//!   and the frames before them hold the log-mel floor `ln(1e-5)` (silence), where `mel2hidden`
//!   appends zeros after the audio. A zero log-mel frame is a loud flat spectrum, and with the
//!   model's BiGRU it pulled the newest frames of every short incremental call (the F0 cache's
//!   24-frame slices carried 7 such frames after their newest frame, 39 on a 64-frame plan) towards
//!   unvoiced. Measured with the F0 cache's 40 ms ticks on 7 speech clips against whole-clip F0
//!   (`engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`, CPU): V/UV error 9.42 % -> 4.99 %
//!   without lookahead and 5.79 % -> 3.66 % with 20 ms, voiced RMSE (within 600 cents) 41.2 -> 40.6
//!   and 35.6 -> 25.5 cents, at the same padded size (the same cost). The last `frames` output rows
//!   are the audio's.
//! - waveform exports (`[1, N]`, optional `threshold`; w-okada's `rmvpe.onnx`), whose output is
//!   either salience or F0 in Hz. On DirectML every call runs at the first call's length
//!   (`pitch_onnx::ShapePin`): a shorter call is front-padded with silence in whole 10 ms frames and
//!   the padding's frames are dropped, as FCPE does.
//!
//! Salience is decoded like RVC `to_local_average_cents`: argmax ± 4 bins, salience-weighted mean
//! of `cents = 20 * bin + 1997.3794084376191`, 0 when the peak is <= 0.03, `f0 = 10 * 2^(c/1200)`.
//!
//! # Fixed-shape plan (P3-D)
//! A log-mel export can run as a fixed-shape session plan (`gpu::plan`), built at the first call
//! with the mel length pinned to that call's (later, shorter calls are front-padded to it with the
//! silence floor, as the DirectML pin of the plain session is; a longer call rebuilds the plan once
//! at its length):
//! - on **DirectML** with the `io_binding` lever (on in the `auto` lever table): a static shape lets
//!   DirectML compile the whole graph (measured on the RTX 4090 at 32 mel frames: 2.0 ms per call
//!   against 5.6 ms for the dynamic-shape session);
//! - on **CUDA** with the `io_binding` and `cuda_graphs` levers ([`RmvpePitchExtractor::load_with_levers`]):
//!   device-resident I/O and a captured CUDA graph (3.1 ms against 7.8 ms), plus the plan's priority
//!   stream and TF32 per the levers.
//!
//! With the levers off RMVPE runs the plain session (the legacy path; on DirectML still at its pinned
//! shape). A plan that cannot be built (or fails later) hands the extractor back to the plain
//! session with a warning. Waveform exports always run the plain session.

use std::path::PathBuf;
use std::sync::Arc;

use ort::session::Session;
use ort::value::ValueType;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::neural_audio::gpu::{GpuLevers, PlanSpec, SessionPlan, TensorSpec};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::f0_post::smooth_f0;
use super::pitch_onnx::{flat_f0, local_average_f0, waveform_shape, MainInput, PitchInputs, ShapePin, SALIENCE_BINS};
use super::traits::PitchExtractor;

const SR: u32 = 16_000;
const HOP: usize = 160;
const N_BINS: usize = SALIENCE_BINS;
const N_FFT: usize = 1024;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 30.0;
const MEL_FMAX: f64 = 8000.0;
const MEL_CLAMP: f32 = 1e-5;
/// RVC `infer_from_audio(audio, thred=0.03)`.
const SALIENCE_THRESHOLD: f32 = 0.03;
/// Value w-okada feeds its waveform-input RMVPE exports' `threshold` input.
const WAVEFORM_EXPORT_THRESHOLD: f32 = 0.3;
#[cfg(test)]
const CENTS_OFFSET: f32 = super::pitch_onnx::SALIENCE_CENTS_OFFSET;

pub struct RmvpePitchExtractor {
    /// The plain session (`None` once a plan has replaced it).
    session: Option<Session>,
    inputs: PitchInputs,
    mel: Option<LogMel>,
    /// Peak salience per frame of the last [`PitchExtractor::f0`] call (P3-D: the confidence F0
    /// fusion weighs; 1 / 0 for exports that output F0 directly).
    confidence: Vec<f32>,
    /// DirectML: every call runs at the first call's mel length (`pitch_onnx::ShapePin`).
    pin: ShapePin,
    /// Fixed-shape plan settings (module docs); `None` = the plain session only.
    planned: Option<PlanRequest>,
    plan: Option<RmvpePlan>,
    /// Mel scratch of the plain-session path.
    mel_buf: Vec<f32>,
    /// Front-padded waveform of a pinned (DirectML) waveform-export call.
    wave_buf: Vec<f32>,
    /// EP outcome of the load (`vc.pitch`; the plan, when one is built later, runs on the same rung).
    outcome: EpOutcome,
}

/// What the fixed-shape plan is built from.
struct PlanRequest {
    path: PathBuf,
    ep: String,
    opts: VcSessionOptions,
    levers: GpuLevers,
    input: String,
    output: String,
    /// The mel input's symbolic time dimension (overridden to the plan's length).
    time_dim: Option<String>,
    /// The plan failed: the plain session runs from now on.
    failed: bool,
}

struct RmvpePlan {
    plan: SessionPlan,
    /// Mel frames of every call.
    frames: usize,
}

impl RmvpePitchExtractor {
    /// Load on the EP ladder of `execution_provider_hint` with no GPU levers (the plain session on
    /// every rung, see the module docs).
    #[allow(dead_code)] // The P0 constructor; the pitch bank passes the voice's levers.
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        Self::load_with_levers(model_path, execution_provider_hint, opts, GpuLevers::default())
    }

    /// [`load`](Self::load) with the voice's GPU levers: on CUDA, `io_binding` plus `cuda_graphs`
    /// run a log-mel export as a CUDA-graph plan; on DirectML, `io_binding` runs it as a fixed-shape
    /// plan (module docs).
    pub fn load_with_levers(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions, levers: GpuLevers) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("RMVPE model path is empty".into());
        }
        let (session, outcome) = build_session(
            Some("vc.pitch"),
            execution_provider_hint,
            opts,
            ModelSource::File(std::path::Path::new(path)),
        )
        .map_err(|e| format!("Failed to load RMVPE ONNX {}: {}", path, e))?;
        let output = session.outputs().first().map(|o| o.name().to_string());
        let mut this = Self::from_session(session, path)?;
        this.outcome = outcome.clone();
        let time_dim = this
            .session
            .as_ref()
            .and_then(|s| mel_time_dim(s, this.inputs.main_name()));
        this.pin = ShapePin::for_ep(&outcome.active);
        let wants_plan = wants_plan(&outcome.active, levers);
        if let (true, Some(_), Some(output)) = (wants_plan, this.mel.as_ref(), output) {
            this.planned = Some(PlanRequest {
                path: PathBuf::from(path),
                ep: outcome.active.clone(),
                opts: opts.clone(),
                levers,
                input: this.inputs.main_name().to_string(),
                output,
                time_dim,
                failed: false,
            });
        }
        Ok(this)
    }

    /// Build the extractor around a session of `path` (shared by the loaders).
    fn from_session(session: Session, path: &str) -> Result<Self, String> {
        let inputs = PitchInputs::inspect(&session, "RMVPE", path, WAVEFORM_EXPORT_THRESHOLD)?;
        let mel = match inputs.main {
            MainInput::Waveform { .. } => None,
            MainInput::LogMel { n_mels } => {
                if n_mels.is_some_and(|m| m != N_MELS) {
                    return Err(format!(
                        "RMVPE ONNX {} expects a {:?} mel input; Sigil computes RVC's {}-band \
                         log-mel [1, {}, T]",
                        path,
                        inputs.main_dims(),
                        N_MELS,
                        N_MELS
                    ));
                }
                Some(LogMel::new())
            }
        };
        Ok(Self {
            session: Some(session),
            inputs,
            mel,
            confidence: Vec::new(),
            pin: ShapePin::default(),
            planned: None,
            plan: None,
            mel_buf: Vec::new(),
            wave_buf: Vec::new(),
            outcome: EpOutcome::granted("cpu"),
        })
    }

    /// EP outcome of the load (`vc.pitch`), for callers that report where F0 runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// Per-frame confidence of the last [`PitchExtractor::f0`] call, aligned with its contour: the
    /// peak salience (0..1) for salience exports, 1 / 0 (voiced / unvoiced) for exports that output
    /// F0 directly.
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }

    /// Whether calls run through the fixed-shape plan (after the first call built it).
    #[allow(dead_code)] // Diagnostics accessor (tests).
    pub fn planned(&self) -> bool {
        self.plan.is_some()
    }

    /// Run a mel-input call through the plan (building or growing it first).
    fn f0_planned(&mut self, audio: &[f32]) -> Result<Vec<f32>, String> {
        let (Some(req), Some(mel)) = (self.planned.as_ref(), self.mel.as_mut()) else {
            return Err("RMVPE plan not configured".into());
        };
        let natural = 32 * ((audio.len() / HOP) / 32 + 1);
        if self.plan.as_ref().is_none_or(|p| natural > p.frames) {
            if let Some(old) = &self.plan {
                tracing::info!(target: "sigil::vc", "RMVPE: a {natural}-frame call outgrew the {}-frame plan; rebuilding it", old.frames);
            }
            // Free the old plan's device memory before building the larger one.
            self.plan = None;
            self.plan = Some(build_plan(req, natural)?);
        }
        let Some(plan) = self.plan.as_mut() else {
            return Err("RMVPE plan missing".into());
        };
        let frames_p = plan.frames;
        let frames = mel.compute_into(audio, frames_p, plan.plan.input_f32(&req.input)?)?;
        plan.plan.run()?;
        let out = plan.plan.output(&req.output)?;
        let mut f0 = decode_tail(&[1, frames_p, N_BINS], out, frames, &mut self.confidence)?;
        smooth_f0(&mut f0, 3);
        // The plan works: its session replaces the plain one (frees that session's memory).
        self.session = None;
        Ok(f0)
    }
}

impl PitchExtractor for RmvpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            return Ok(Vec::new());
        }
        if self.planned.as_ref().is_some_and(|r| !r.failed) {
            match self.f0_planned(mono_16k) {
                Ok(f0) => return Ok(f0),
                Err(e) => {
                    tracing::warn!(target: "sigil::vc", "RMVPE: the fixed-shape plan failed ({e}); running the plain session");
                    self.plan = None;
                    if let Some(req) = self.planned.as_mut() {
                        req.failed = true;
                    }
                }
            }
        }
        let n = mono_16k.len();
        // Log-mel exports: the mel into the reused buffer, end-aligned in a multiple of 32 frames
        // (and in the DirectML pin).
        let mel_frames = match &mut self.mel {
            Some(mel) => {
                let padded = self.pin.size(32 * ((n / HOP) / 32 + 1));
                self.mel_buf.clear();
                self.mel_buf.resize(N_MELS * padded, 0.0);
                Some((mel.compute_into(mono_16k, padded, &mut self.mel_buf)?, padded))
            }
            None => None,
        };
        // Waveform exports on DirectML: the pinned length, front-padded with silence in whole frames
        // (the padding's frames are dropped below; the newest frames keep their real right context).
        let run_n = match mel_frames {
            Some(_) => n,
            None => self.pin.size_aligned(n, HOP),
        };
        let lead_frames = (run_n - n) / HOP;
        let wave: &[f32] = if run_n == n {
            mono_16k
        } else {
            self.wave_buf.clear();
            self.wave_buf.resize(run_n - n, 0.0);
            self.wave_buf.extend_from_slice(mono_16k);
            &self.wave_buf
        };
        let session = plain_session(&mut self.session, self.planned.as_ref())?;
        let outputs = match (mel_frames, self.inputs.main) {
            (Some((_, padded)), _) => self
                .inputs
                .run(session, vec![1, N_MELS, padded], &self.mel_buf, "RMVPE")?,
            (None, MainInput::Waveform { rank }) => self
                .inputs
                .run(session, waveform_shape(rank, run_n), wave, "RMVPE")?,
            (None, MainInput::LogMel { .. }) => return Err("RMVPE mel front end missing".into()),
        };
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("RMVPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = match mel_frames {
            // The audio's frames end the end-aligned mel.
            Some((frames, _)) => decode_tail(&dims, data, frames, &mut self.confidence)?,
            None => decode_output(&dims, data, 1 + run_n / HOP, &mut self.confidence)?,
        };
        if lead_frames > 0 {
            f0.drain(..lead_frames.min(f0.len()));
            self.confidence
                .drain(..lead_frames.min(self.confidence.len()));
        }
        smooth_f0(&mut f0, 3);
        Ok(f0)
    }
}

/// Whether a log-mel export on `active_ep` runs as a fixed-shape plan with `levers` (module docs):
/// DirectML with `io_binding`, CUDA with `io_binding` and `cuda_graphs`; every other case runs the
/// plain session (the legacy path).
fn wants_plan(active_ep: &str, levers: GpuLevers) -> bool {
    match active_ep {
        "dml" => levers.io_binding,
        "cuda" => levers.io_binding && levers.cuda_graphs,
        _ => false,
    }
}

/// The plain session, rebuilt when a plan that had replaced it failed.
fn plain_session<'a>(session: &'a mut Option<Session>, planned: Option<&PlanRequest>) -> Result<&'a mut Session, String> {
    if session.is_none() {
        let req = planned.ok_or("RMVPE session missing")?;
        let (built, _) = build_session(Some("vc.pitch"), &req.ep, &req.opts, ModelSource::File(&req.path))
            .map_err(|e| format!("Failed to reload RMVPE ONNX {}: {}", req.path.display(), e))?;
        *session = Some(built);
    }
    session
        .as_mut()
        .ok_or_else(|| "RMVPE session missing".to_string())
}

/// The symbolic time dimension of the log-mel input `name` (`[1, n_mels, time]`), if it has one.
fn mel_time_dim(session: &Session, name: &str) -> Option<String> {
    let input = session.inputs().iter().find(|i| i.name() == name)?;
    match input.dtype() {
        ValueType::Tensor {
            dimension_symbols, shape, ..
        } if shape.len() == 3 && shape[2] < 0 => dimension_symbols.get(2).filter(|s| !s.is_empty()).cloned(),
        _ => None,
    }
}

/// Build the fixed-shape plan of `req` at `frames` mel frames.
fn build_plan(req: &PlanRequest, frames: usize) -> Result<RmvpePlan, String> {
    let inputs = [TensorSpec::f32(&req.input, &[1, N_MELS, frames])];
    let outputs = [TensorSpec::f32(&req.output, &[1, frames, N_BINS])];
    let dims: Vec<(&str, i64)> = req
        .time_dim
        .iter()
        .map(|d| (d.as_str(), frames as i64))
        .collect();
    let plan = SessionPlan::build(&PlanSpec {
        component: "vc.pitch",
        ep: &req.ep,
        opts: &req.opts,
        model: ModelSource::File(&req.path),
        weights: None,
        dims: &dims,
        inputs: &inputs,
        outputs: &outputs,
        levers: req.levers,
    })?;
    for note in plan.notes() {
        tracing::info!(target: "sigil::vc", "RMVPE plan: {note}");
    }
    Ok(RmvpePlan { plan, frames })
}

/// Decode the output of an end-aligned mel ([`LogMel::compute_into`]): every output row, then the
/// last `frames` (the audio's; the rows before them are the silence padding's).
fn decode_tail(dims: &[usize], data: &[f32], frames: usize, confidence: &mut Vec<f32>) -> Result<Vec<f32>, String> {
    // No cut (`usize::MAX`): the padded mel has at least 32 frames, so a `[T, 360]` output is never
    // mistaken for a one-row F0.
    let mut f0 = decode_output(dims, data, usize::MAX, confidence)?;
    let pad = f0.len().saturating_sub(frames);
    if pad > 0 {
        f0.drain(..pad);
        confidence.drain(..pad.min(confidence.len()));
    }
    Ok(f0)
}

/// Decode an RMVPE output into F0 (Hz). Salience `[1, T, 360]` / `[T, 360]` is decoded and cut to
/// `frames` rows (RVC `hidden[:, :n_frames]`); a direct F0 output `[1, T]` / `[T]` passes through.
/// `confidence` receives one value per returned frame (the peak salience, or 1 / 0).
fn decode_output(dims: &[usize], data: &[f32], frames: usize, confidence: &mut Vec<f32>) -> Result<Vec<f32>, String> {
    confidence.clear();
    let salience_rows = match dims {
        [1, t, b] if *b == N_BINS => Some(*t),
        [t, b] if *b == N_BINS && (*t != 1 || frames <= 1) => Some(*t),
        _ => None,
    };
    if let Some(t) = salience_rows {
        let rows = t.min(frames).min(data.len() / N_BINS);
        return Ok(data[..rows * N_BINS]
            .chunks_exact(N_BINS)
            .map(|row| {
                // A NaN near the peak (or an overflowing weighted mean) is unvoiced, never NaN F0.
                let f = salience_to_f0(row);
                let peak = row
                    .iter()
                    .copied()
                    .fold(0.0f32, |a, v| if v > a { v } else { a });
                confidence.push(peak.min(1.0));
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            })
            .collect());
    }
    let mut f0 = flat_f0(dims, data).ok_or_else(|| {
        format!(
            "RMVPE output shape {dims:?} unsupported; expected salience [1, T, 360] / [T, 360] \
             or F0 [1, T] / [T]"
        )
    })?;
    for v in f0.iter_mut() {
        if !v.is_finite() || *v < 0.0 {
            *v = 0.0;
        }
        confidence.push(if *v > 0.0 { 1.0 } else { 0.0 });
    }
    Ok(f0)
}

/// RVC `to_local_average_cents` + `decode` for one salience row (0 at or below the 0.03 peak
/// threshold).
fn salience_to_f0(row: &[f32]) -> f32 {
    let (hz, peak) = local_average_f0(row);
    if peak <= SALIENCE_THRESHOLD {
        return 0.0;
    }
    hz
}

/// RVC `MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000)`, padded like `mel2hidden`
/// but end-aligned (module docs).
struct LogMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Per mel band: first FFT bin and its non-zero weights.
    bands: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
}

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// librosa `filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True, norm="slaney")`.
fn mel_filterbank() -> Vec<(usize, Vec<f32>)> {
    let n_freqs = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * f64::from(SR) / N_FFT as f64)
        .collect();
    let (mel_lo, mel_hi) = (hz_to_mel_htk(MEL_FMIN), hz_to_mel_htk(MEL_FMAX));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|i| {
            let (lo, center, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (hi - lo);
            let weights: Vec<f64> = fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (center - lo);
                    let upper = (hi - f) / (hi - center);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect();
            let first = weights.iter().position(|&w| w > 0.0).unwrap_or(0);
            let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
            let band: Vec<f32> = if last >= first && weights[first] > 0.0 {
                weights[first..=last].iter().map(|&w| w as f32).collect()
            } else {
                Vec::new()
            };
            (first, band)
        })
        .collect()
}

/// torch `reflect` padding index (no edge repeat) for any offset.
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut j = i.rem_euclid(period);
    if j >= n as isize {
        j = period - j;
    }
    j as usize
}

impl LogMel {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // torch.hann_window(1024) is periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        let frame = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft,
            window,
            bands: mel_filterbank(),
            frame,
            spectrum,
            scratch,
            magnitude: vec![0.0; N_FFT / 2 + 1],
        }
    }

    /// Returns `(frames, padded_frames, data)` with `data` = `[N_MELS, padded_frames]` row-major.
    /// `min_padded`: pad to at least this many frames (a DirectML session's pinned shape).
    #[cfg(test)]
    fn compute(&mut self, audio: &[f32], min_padded: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let frames = 1 + audio.len() / HOP;
        let padded_frames = (32 * ((frames - 1) / 32 + 1)).max(min_padded);
        let mut data = vec![0.0f32; N_MELS * padded_frames];
        self.compute_into(audio, padded_frames, &mut data)?;
        Ok((frames, padded_frames, data))
    }

    /// The log-mel of `audio` into `data` (`[N_MELS, padded_frames]` row-major, every value written),
    /// end-aligned: the audio's frames are the last `frames` columns and the columns before them
    /// hold the log-mel floor `ln(1e-5)` (silence). Returns the audio's frame count.
    fn compute_into(&mut self, audio: &[f32], padded_frames: usize, data: &mut [f32]) -> Result<usize, String> {
        let n = audio.len();
        let frames = 1 + n / HOP;
        if padded_frames < frames || data.len() != N_MELS * padded_frames {
            return Err(format!(
                "RMVPE mel buffer: {} values for {frames} frames padded to {padded_frames}",
                data.len()
            ));
        }
        data.fill(MEL_CLAMP.ln());
        let lead = padded_frames - frames;
        let pad = (N_FFT / 2) as isize;
        for t in 0..frames {
            let origin = (t * HOP) as isize - pad;
            for (k, (dst, &w)) in self.frame.iter_mut().zip(self.window.iter()).enumerate() {
                *dst = audio[reflect_index(origin + k as isize, n)] * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("RMVPE mel FFT: {e}"))?;
            for (m, c) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
                *m = (c.re * c.re + c.im * c.im).sqrt();
            }
            for (band_idx, (first, weights)) in self.bands.iter().enumerate() {
                let energy: f32 = weights
                    .iter()
                    .zip(&self.magnitude[*first..])
                    .map(|(w, m)| w * m)
                    .sum();
                data[band_idx * padded_frames + lead + t] = energy.max(MEL_CLAMP).ln();
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hz(cents: f32) -> f32 {
        10.0 * (cents / 1200.0).exp2()
    }

    #[test]
    fn salience_decode_matches_rvc_local_average() {
        // Symmetric peak at bin 100 → exactly the bin's cents.
        let mut row = vec![0.0f32; N_BINS];
        row[99] = 0.5;
        row[100] = 0.9;
        row[101] = 0.5;
        let expected = hz(20.0 * 100.0 + CENTS_OFFSET);
        assert!((salience_to_f0(&row) - expected).abs() < 1e-3);
        assert!((expected - 100.64).abs() < 0.05, "{expected}");

        // Asymmetric window: weighted mean over argmax ± 4 only (bin 110 is outside).
        let mut row = vec![0.0f32; N_BINS];
        row[200] = 0.8;
        row[203] = 0.4;
        row[210] = 0.7;
        let cents = (0.8 * (20.0 * 200.0 + CENTS_OFFSET) + 0.4 * (20.0 * 203.0 + CENTS_OFFSET)) / 1.2;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-2);

        // Edge bins: the window is clipped (RVC pads zeros).
        let mut row = vec![0.0f32; N_BINS];
        row[0] = 0.6;
        row[2] = 0.2;
        let cents = (0.6 * CENTS_OFFSET + 0.2 * (40.0 + CENTS_OFFSET)) / 0.8;
        assert!((salience_to_f0(&row) - hz(cents)).abs() < 1e-3);

        // Peak at or below the 0.03 threshold → unvoiced.
        let mut row = vec![0.0f32; N_BINS];
        row[150] = 0.03;
        assert_eq!(salience_to_f0(&row), 0.0);
    }

    #[test]
    fn decode_output_handles_salience_and_flat_shapes() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        sal[N_BINS + 100] = 1.0;
        // [1, T, 360] cut to `frames` rows.
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 2, &mut Vec::new()).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(f0[0], 0.0);
        assert!((f0[1] - hz(20.0 * 100.0 + CENTS_OFFSET)).abs() < 1e-3);
        // [T, 360].
        assert_eq!(
            decode_output(&[3, N_BINS], &sal, 3, &mut Vec::new())
                .unwrap()
                .len(),
            3
        );
        // A flat [1, T] F0 output is taken as Hz, not collapsed into one salience row.
        let flat = vec![100.0f32, 0.0, 220.0, 230.0];
        assert_eq!(decode_output(&[1, 4], &flat, 5, &mut Vec::new()).unwrap(), flat);
        assert_eq!(decode_output(&[4], &flat, 5, &mut Vec::new()).unwrap(), flat);
        assert!(decode_output(&[2, 2, 2], &[0.0; 8], 2, &mut Vec::new()).is_err());
    }

    #[test]
    fn decode_output_maps_non_finite_salience_to_unvoiced() {
        let mut sal = vec![0.0f32; 3 * N_BINS];
        // Row 0: NaN next to the peak poisons the weighted mean.
        sal[100] = 0.9;
        sal[101] = f32::NAN;
        // Row 1: all NaN (no peak at all).
        for v in &mut sal[N_BINS..2 * N_BINS] {
            *v = f32::NAN;
        }
        // Row 2: infinite salience.
        sal[2 * N_BINS + 50] = f32::INFINITY;
        let f0 = decode_output(&[1, 3, N_BINS], &sal, 3, &mut Vec::new()).unwrap();
        assert_eq!(f0.len(), 3);
        assert!(f0.iter().all(|v| v.is_finite()), "{f0:?}");
        assert_eq!(f0[0], 0.0);
        assert_eq!(f0[1], 0.0);
        // A flat F0 output with NaN / negative values is sanitized too.
        assert_eq!(
            decode_output(&[3], &[f32::NAN, -1.0, 150.0], 3, &mut Vec::new()).unwrap(),
            vec![0.0, 0.0, 150.0]
        );
    }

    #[test]
    fn mel_filterbank_matches_librosa_htk_slaney() {
        let bands = mel_filterbank();
        assert_eq!(bands.len(), N_MELS);
        // Band 0 spans 30 Hz .. ~58.7 Hz (centre ~44.2 Hz): FFT bins 2 (31.25 Hz) and 3 (46.875 Hz).
        let (first, w) = &bands[0];
        assert_eq!(*first, 2);
        assert_eq!(w.len(), 2);
        assert!((w[0] - 0.006_12).abs() < 2e-4, "{w:?}");
        assert!((w[1] - 0.057_0).abs() < 1e-3, "{w:?}");
        // Last band ends at fmax = 8000 Hz = bin 512, where its weight is (numerically) zero.
        let (first, w) = &bands[N_MELS - 1];
        assert!(first + w.len() <= 513);
        if first + w.len() == 513 {
            assert!(w[w.len() - 1] < 1e-6, "{:?}", &w[w.len() - 3..]);
        }
        assert!(bands.iter().all(|(_, w)| w.iter().all(|&x| x >= 0.0)));
    }

    #[test]
    fn log_mel_frame_count_padding_and_silence_floor() {
        let mut mel = LogMel::new();
        let (frames, padded, data) = mel.compute(&vec![0.0f32; 1600], 0).unwrap();
        assert_eq!(frames, 11); // 1 + 1600 / 160 (centred STFT)
        assert_eq!(padded, 32);
        assert_eq!(data.len(), N_MELS * 32);
        // Silence and the padding are the same floor (end-aligned: the padding comes first).
        assert!(data.iter().all(|&v| (v - MEL_CLAMP.ln()).abs() < 1e-4));
        // A 1 kHz tone peaks in the band containing 1 kHz; its frames end the padded mel, the
        // padding before them is the silence floor.
        let tone: Vec<f32> = (0..3200)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR as f32).sin())
            .collect();
        let (frames, padded, data) = mel.compute(&tone, 0).unwrap();
        assert_eq!((frames, padded), (21, 32));
        let lead = padded - frames;
        for band in 0..N_MELS {
            assert!(data[band * padded..band * padded + lead].iter().all(|&v| v == MEL_CLAMP.ln()), "band {band}");
        }
        let t = lead + frames / 2;
        let best = (0..N_MELS)
            .max_by(|&a, &b| {
                data[a * padded + t]
                    .partial_cmp(&data[b * padded + t])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let mel_lo = hz_to_mel_htk(MEL_FMIN);
        let mel_hi = hz_to_mel_htk(MEL_FMAX);
        let centre = mel_to_hz_htk(mel_lo + (mel_hi - mel_lo) * (best + 1) as f64 / 129.0);
        assert!((centre - 1000.0).abs() < 40.0, "band {best} centre {centre}");
    }

    #[test]
    fn directml_calls_run_at_the_first_shape() {
        let mut pin = ShapePin::for_ep("dml");
        assert_eq!(pin.size(64), 64);
        assert_eq!(pin.size(32), 64);
        assert_eq!(pin.size(64), 64);
        // A larger call cannot fit: it runs at its own size, the pin stays.
        assert_eq!(pin.size(96), 96);
        assert_eq!(pin.size(40), 64);
        // Aligned pinning (FCPE pads whole 10 ms frames).
        let mut pin = ShapePin::for_ep("dml");
        assert_eq!(pin.size_aligned(7_040, 160), 7_040);
        assert_eq!(pin.size_aligned(3_840, 160), 7_040);
        assert_eq!(pin.size_aligned(3_841, 160), 3_841);
        // Other EPs never pad.
        for ep in ["cpu", "cuda"] {
            let mut pin = ShapePin::for_ep(ep);
            assert_eq!(pin.size(64), 64);
            assert_eq!(pin.size(32), 32);
        }
        // The mel pads to the pinned frame count, end-aligned: the silence floor before the audio.
        let mut mel = LogMel::new();
        let tone: Vec<f32> = (0..3_840).map(|i| (i as f32 * 0.3).sin()).collect();
        let (frames, padded, data) = mel.compute(&tone, 64).unwrap();
        assert_eq!((frames, padded, data.len()), (25, 64, N_MELS * 64));
        for band in 0..N_MELS {
            assert!(data[band * 64..band * 64 + 39].iter().all(|&v| v == MEL_CLAMP.ln()), "band {band}");
        }
        assert!(
            (39..64).all(|t| (0..N_MELS).any(|b| data[b * 64 + t] > MEL_CLAMP.ln() + 1.0)),
            "the audio's frames end the mel"
        );
    }

    /// The fixed-shape plan follows the levers: DirectML with `io_binding`, CUDA with `io_binding`
    /// and `cuda_graphs`; `off` (and the CPU) is the plain session, the legacy path.
    #[test]
    fn the_plan_follows_the_levers() {
        let bound = GpuLevers {
            io_binding: true,
            ..GpuLevers::default()
        };
        let graphs = GpuLevers {
            cuda_graphs: true,
            ..bound
        };
        assert!(wants_plan("dml", bound));
        assert!(wants_plan("dml", graphs));
        assert!(!wants_plan("dml", GpuLevers::default()), "io_binding off: the plain DirectML session");
        assert!(!wants_plan("cuda", bound));
        assert!(wants_plan("cuda", graphs));
        assert!(!wants_plan("cuda", GpuLevers::default()));
        assert!(!wants_plan("cpu", graphs));
    }

    /// A stand-in waveform RMVPE (w-okada style, `[1, N]` in, F0 `[1, N / 160]` out): each 10 ms
    /// frame's F0 is `50 + 1000 * |mean|` of its samples, so frames are independent and the zero
    /// front padding's frames read exactly 50 Hz.
    fn fake_waveform_rmvpe_file(tag: &str) -> std::path::PathBuf {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("waveform", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .init_i64("shape", &[3], &[1, -1, HOP as i64])
            .init_f32("scale", &[], &[1000.0])
            .init_f32("offset", &[], &[50.0])
            .node("Reshape", &["waveform", "shape"], &["frames"], &[])
            .node("ReduceMean", &["frames"], &["mean"], &[Attr::Ints("axes", vec![2]), Attr::Int("keepdims", 0)])
            .node("Abs", &["mean"], &["level"], &[])
            .node("Mul", &["level", "scale"], &["scaled"], &[])
            .node("Add", &["scaled", "offset"], &["f0"], &[])
            .output("f0", FLOAT, &[Dim::Fixed(1), Dim::Named("t")]);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("sigil-p3d-rmvpe-wave-{tag}-{}-{nanos}.onnx", std::process::id()));
        std::fs::write(&path, g.model_bytes(17)).unwrap();
        path
    }

    /// Waveform exports pin their shape on DirectML too (the incremental F0 cache's slices are
    /// shorter than its first, whole-window call): a shorter call runs front-padded to the pinned
    /// length and returns exactly the frames of the unpadded call.
    #[test]
    fn waveform_exports_pin_their_shape_on_directml() {
        let path = fake_waveform_rmvpe_file("pin");
        let p = path.to_string_lossy().to_string();
        let opts = VcSessionOptions::default();
        let x: Vec<f32> = (0..16_000)
            .map(|i| 0.2 * ((i as f32) * 0.05).sin() + 0.1 * ((i / 700) as f32 * 0.3).cos())
            .collect();
        let mut plain = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        assert!(matches!(plain.inputs.main, MainInput::Waveform { rank: 2 }));
        let mut pinned = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        pinned.pin = ShapePin::for_ep("dml");
        // The first call (the whole window) pins 7 040 samples; the slices that follow are padded.
        for (k, len) in [7_040usize, 3_840, 3_840, 7_040, 4_480].into_iter().enumerate() {
            let slice = &x[k * 640..k * 640 + len];
            let want = plain.f0(slice).unwrap();
            let got = pinned.f0(slice).unwrap();
            assert_eq!(got.len(), len / HOP, "call {k}");
            assert_eq!(got.len(), want.len(), "call {k}");
            // The same frames (ONNX Runtime's reduction may differ in the last bit between the
            // padded and the unpadded tensor).
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-4 * w.abs(), "call {k} ({len} samples) frame {j}: {g} vs {w}");
            }
            assert_eq!(pinned.last_confidence(), plain.last_confidence(), "call {k}");
            assert_eq!(pinned.last_confidence().len(), got.len(), "call {k}");
            if len < 7_040 {
                assert_eq!(pinned.wave_buf.len(), 7_040, "call {k} ran at the pinned length");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A stand-in log-mel RMVPE: `output = sigmoid(transpose(input) · W)` (`[1, 128, time]` in,
    /// `[1, time, 360]` out; frames are independent, so padding never changes a frame).
    fn fake_rmvpe_file(tag: &str) -> std::path::PathBuf {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let w: Vec<f32> = (0..N_MELS * N_BINS)
            .map(|i| {
                let (m, b) = (i / N_BINS, i % N_BINS);
                if b == (3 * m + 40) % N_BINS {
                    0.05
                } else {
                    -0.002 * ((b % 7) as f32)
                }
            })
            .collect();
        let mut g = Graph::new();
        g.input("input", FLOAT, &[Dim::Fixed(1), Dim::Fixed(N_MELS as u64), Dim::Named("time")])
            .init_f32("w", &[N_MELS as u64, N_BINS as u64], &w)
            .node("Transpose", &["input"], &["t"], &[Attr::Ints("perm", vec![0, 2, 1])])
            .node("MatMul", &["t", "w"], &["logits"], &[])
            .node("Sigmoid", &["logits"], &["output"], &[])
            .output("output", FLOAT, &[Dim::Fixed(1), Dim::Named("time"), Dim::Fixed(N_BINS as u64)]);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("sigil-p3d-rmvpe-{tag}-{}-{nanos}.onnx", std::process::id()));
        std::fs::write(&path, g.model_bytes(17)).unwrap();
        path
    }

    fn plan_request(e: &RmvpePitchExtractor, path: &std::path::Path, output: &str) -> PlanRequest {
        PlanRequest {
            path: path.to_path_buf(),
            ep: "cpu".into(),
            opts: VcSessionOptions::default(),
            levers: GpuLevers {
                io_binding: true,
                ..GpuLevers::default()
            },
            input: e.inputs.main_name().to_string(),
            output: output.to_string(),
            time_dim: e
                .session
                .as_ref()
                .and_then(|s| mel_time_dim(s, e.inputs.main_name())),
            failed: false,
        }
    }

    /// The fixed-shape plan path (forced onto the CPU rung here; it runs on DirectML and on CUDA with
    /// graphs) gives the plain session's contour and confidence at every call length, pins the first
    /// call's length, grows once for a longer call, and falls back to the plain session when the
    /// plan cannot be built.
    #[test]
    fn plan_path_matches_the_plain_session_and_falls_back() {
        let path = fake_rmvpe_file("plan");
        let p = path.to_string_lossy().to_string();
        let opts = VcSessionOptions::default();
        let x: Vec<f32> = (0..20_000)
            .map(|i| (0.3 * (i as f32 * 0.071).sin() + 0.2 * (i as f32 * 0.013).sin()) * (1.0 + (i / 3000) as f32 * 0.2))
            .collect();
        let mut plain = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        assert!(plain.planned.is_none(), "no plan on the CPU rung without asking");
        assert_eq!(
            plain
                .session
                .as_ref()
                .and_then(|s| mel_time_dim(s, "input"))
                .as_deref(),
            Some("time")
        );
        let mut planned = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        planned.planned = Some(plan_request(&planned, &path, "output"));
        for (k, len) in [7_040usize, 3_840, 3_840, 7_040, 12_000, 3_840]
            .into_iter()
            .enumerate()
        {
            let slice = &x[k * 640..k * 640 + len];
            let want = plain.f0(slice).unwrap();
            let got = planned.f0(slice).unwrap();
            assert_eq!(got, want, "call {k} ({len} samples)");
            assert_eq!(planned.last_confidence(), plain.last_confidence(), "call {k}");
            assert!(planned.planned(), "call {k}");
            assert!(planned.session.is_none(), "the plan replaced the session");
        }
        // 12 000 samples (96 mel frames) outgrew the 64-frame plan once.
        assert_eq!(planned.plan.as_ref().map(|p| p.frames), Some(96));
        // A plan that cannot be built: the plain session runs, with the same results.
        let mut broken = RmvpePitchExtractor::load(&p, "cpu", &opts).unwrap();
        broken.planned = Some(plan_request(&broken, &path, "no-such-output"));
        let want = plain.f0(&x[..7_040]).unwrap();
        assert_eq!(broken.f0(&x[..7_040]).unwrap(), want);
        assert!(broken.planned.as_ref().is_some_and(|r| r.failed));
        assert!(!broken.planned());
        assert_eq!(broken.f0(&x[640..640 + 3_840]).unwrap(), plain.f0(&x[640..640 + 3_840]).unwrap());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reflect_padding_matches_torch() {
        // torch reflect: [x2, x1 | x0 x1 x2 x3 | x2 x1]
        let n = 4;
        let idx: Vec<usize> = (-2..6).map(|i| reflect_index(i, n)).collect();
        assert_eq!(idx, vec![2, 1, 0, 1, 2, 3, 2, 1]);
    }

    /// Dev rig: RMVPE per-call cost at a whole 44-frame window, then the incremental F0 cache's
    /// 24-frame slice, then the window again, per EP (DirectML pins the first shape: without the
    /// pin the second size ran at 84-116 ms per call; the fixed-shape plan, `io_binding` on, then
    /// halves it again), and on CUDA with the CUDA-graph plan. `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_tick_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model"]
    fn rmvpe_tick_cost() {
        let Some(path) = super::super::f0_fusion::test_model("pitch-rmvpe") else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        let x = super::super::f0_fusion::ab::harmonic_tone(180.0, 16_000, 0.3);
        let graphs = GpuLevers {
            io_binding: true,
            cuda_graphs: true,
            priority: true,
            tf32: true,
            ..GpuLevers::default()
        };
        let bound = GpuLevers {
            io_binding: true,
            ..GpuLevers::default()
        };
        for (ep, levers) in [
            ("cuda", GpuLevers::default()),
            ("cuda", graphs),
            ("dml", GpuLevers::default()),
            ("dml", bound),
            ("cpu", GpuLevers::default()),
        ] {
            let opts = VcSessionOptions {
                intra_threads: Some(2),
                ..VcSessionOptions::default()
            };
            let Ok(mut r) = RmvpePitchExtractor::load_with_levers(&path.to_string_lossy(), ep, &opts, levers) else {
                continue;
            };
            let ep = match (ep, levers.io_binding, levers.cuda_graphs) {
                (_, _, true) => "cuda+graphs",
                ("dml", true, _) => "dml+plan",
                _ => ep,
            };
            for n in [7_040usize, 3_840, 7_040] {
                let mut times = Vec::new();
                for k in 0..40 {
                    let t0 = std::time::Instant::now();
                    let f0 = r.f0(&x[k * 160..k * 160 + n]).unwrap();
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert!(!f0.is_empty());
                }
                let first = times[0];
                times.sort_by(f64::total_cmp);
                println!(
                    "rmvpe {ep} {n} samples: first {first:.1} ms, p50 {:.2} ms, p95 {:.2} ms",
                    times[20], times[38]
                );
            }
        }
    }

    /// Dev rig: RMVPE through a session plan (bound I/O, CUDA graphs, priority stream, TF32) against
    /// the plain session, at the incremental F0 cache's 32 mel frames and the whole window's 64.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test rmvpe_plan_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "dev rig: needs SIGIL_TEST_MODELS_DIR with the RMVPE model and a GPU"]
    fn rmvpe_plan_cost() {
        use crate::services::neural_audio::gpu::{GpuLevers, PlanSpec, SessionPlan, TensorSpec};
        let Some(path) = super::super::f0_fusion::test_model("pitch-rmvpe") else {
            eprintln!("RMVPE model not found; skipped");
            return;
        };
        let x = super::super::f0_fusion::ab::harmonic_tone(180.0, 16_000, 0.3);
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            ..VcSessionOptions::default()
        };
        let full = GpuLevers {
            io_binding: true,
            cuda_graphs: true,
            priority: true,
            tf32: true,
            ..GpuLevers::default()
        };
        let variants = [
            ("host", GpuLevers::default()),
            (
                "bound",
                GpuLevers {
                    io_binding: true,
                    ..GpuLevers::default()
                },
            ),
            (
                "bound+tf32",
                GpuLevers {
                    io_binding: true,
                    tf32: true,
                    ..GpuLevers::default()
                },
            ),
            ("graphs+priority+tf32", full),
        ];
        for ep in ["cuda", "dml"] {
            for frames in [32usize, 64] {
                let n = (frames - 1) * HOP;
                let mut mel = LogMel::new();
                let (_, padded, data) = mel.compute(&x[..n.min(x.len())], frames).unwrap();
                assert_eq!(padded, frames);
                for (label, levers) in variants {
                    let inputs = [TensorSpec::f32("input", &[1, N_MELS, frames])];
                    let outputs = [TensorSpec::f32("output", &[1, frames, N_BINS])];
                    let spec = PlanSpec {
                        component: "test.rmvpe",
                        ep,
                        opts: &opts,
                        model: ModelSource::File(&path),
                        weights: None,
                        dims: &[("time", frames as i64)],
                        inputs: &inputs,
                        outputs: &outputs,
                        levers,
                    };
                    let mut plan = match SessionPlan::build(&spec) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("{ep} {frames} {label}: {e}");
                            continue;
                        }
                    };
                    plan.input_f32("input").unwrap().copy_from_slice(&data);
                    plan.warm_up(3).unwrap();
                    let mut times = Vec::new();
                    let mut f0 = Vec::new();
                    for _ in 0..60 {
                        let t0 = std::time::Instant::now();
                        plan.input_f32("input").unwrap().copy_from_slice(&data);
                        plan.run().unwrap();
                        let out = plan.output("output").unwrap();
                        f0 = decode_output(&[1, frames, N_BINS], out, frames, &mut Vec::new()).unwrap();
                        times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    times.sort_by(f64::total_cmp);
                    let voiced: Vec<f32> = f0.iter().copied().filter(|&v| v > 0.0).collect();
                    let med = if voiced.is_empty() { 0.0 } else { voiced[voiced.len() / 2] };
                    println!(
                        "rmvpe plan {ep} ({}) {frames} frames {label}: p50 {:.2} ms, p95 {:.2} ms, median F0 {med:.1} Hz, notes {:?}",
                        plan.outcome().active,
                        times[30],
                        times[57],
                        plan.notes()
                    );
                }
            }
        }
    }
}
