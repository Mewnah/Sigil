//! FCPE ONNX pitch extractor (default for Streamer / CPU).
//!
//! Supported contracts, chosen from the graph's declared main input at load:
//! - **Waveform** exports: 16 kHz audio `[N]`, `[1, N]` (Sigil's adapter `vc/pitch-fcpe`) or
//!   `[1, N, 1]` (the community export gzivdo/fcpe-onnx as published, P5-D), with an optional scalar
//!   `threshold` input; F0 out in Hz `[1, T]` / `[T]` / `[1, T, 1]` on the hop-160 grid.
//! - **Mel** exports (P5-D): the graph takes torchfcpe's log-mel, computed here exactly like
//!   torchfcpe / RVC's `Wav2Mel` (16 kHz, n_fft = win = 1024, hop 160, periodic Hann, reflect padding
//!   of 432 samples on both sides with no centring, `sqrt(re² + im² + 1e-9)`, 128 Slaney mels over
//!   0..8000 Hz, `ln(max(x, 1e-5))`, `N / 160 + 1` frames with the last one repeated), as
//!   `[1, T, 128]` (torchfcpe's layout, and the original FCPE's; a channel-major `[1, 128, T]` mel
//!   is RVC's `rmvpe.onnx` layout and is refused as RMVPE). The output, told apart once at load
//!   from its declared shape (the frame count fed decides when the declaration does not), is either
//!   the 360-bin latent `[1, T, 360]`, decoded like torchfcpe's `local_argmax` (cent table
//!   `linspace(cents(32.70 Hz), cents(1975.5 Hz), 360)`, argmax ± 4 bins with edge clamping,
//!   unvoiced when the peak is <= 0.006), or F0 `[1, T]` / `[1, T, 1]`: Hz, or `ln(1 + f0 / 700)`
//!   from an export of FCPE's `infer` without `return_hz_f0` (told apart by value: a voiced F0 in Hz
//!   is at least 32.7). Torchfcpe's frame `t` is centred on sample `160 t + 80`, as upstream.
//!
//! A file that is not an FCPE model gets a specific rejection (a PyTorch checkpoint, a content
//! encoder, RVC's mel RMVPE, w-okada's RMVPE exports or fused pipeline, a voice), so the graph is
//! never fed a mismatched tensor.

use std::sync::Arc;

use ort::session::Session;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::f0_post::smooth_f0;
use super::pitch_onnx::{flat_f0, waveform_shape, MainInput, PitchInputs, ShapePin};
use super::rvc::{foreign_graph_problem, graph_inputs, graph_outputs, non_onnx_file_problem, GraphKind, GraphValue};
use super::traits::PitchExtractor;

/// torchfcpe's default voicing threshold, fed to exports that expose a `threshold` input and used
/// to decode the latent of mel exports.
const FCPE_THRESHOLD: f32 = 0.006;

/// torchfcpe's mel front end (see the module docs).
const SR: f64 = 16_000.0;
const HOP: usize = 160;
const N_FFT: usize = 1024;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 0.0;
const MEL_FMAX: f64 = 8_000.0;
const MEL_CLAMP: f32 = 1e-5;
/// `(win - hop) / 2` samples of reflect padding before the audio (and as many after it).
const PAD_LEFT: usize = (N_FFT - HOP) / 2;
const PAD_RIGHT: usize = (N_FFT - HOP).div_ceil(2);

/// FCPE's latent: 360 bins between these F0s (torchfcpe `CFNaiveMelPE` / the original `FCPE`).
const LATENT_BINS: usize = 360;
const LATENT_F0_MIN: f64 = 32.70;
const LATENT_F0_MAX: f64 = 1975.5;
/// The lowest voiced F0 an FCPE export can output, in Hz: a voiced output below it is not Hz.
const MIN_VOICED_HZ: f32 = 30.0;

/// How the main input is fed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FcpeInput {
    /// Waveform `[N]` (1), `[1, N]` (2) or `[1, N, 1]` (3).
    Waveform { rank: usize },
    /// torchfcpe's log-mel, time-major `[1, T, 128]`.
    Mel,
}

/// The main input kind of a declared FCPE input (`dims`: `-1` = dynamic), or why it is not one.
/// Only torchfcpe's time-major mel is taken: a channel-major `[1, 128, T]` mel is RMVPE's layout
/// (RVC's `rmvpe.onnx`), and no FCPE export takes it (P5-D review: feeding RMVPE torchfcpe's mel and
/// decoding its salience with FCPE's cent table gives detuned F0, or fails on most frame counts).
fn classify_input(main: MainInput, dims: &[i64]) -> Result<FcpeInput, String> {
    match main {
        MainInput::Waveform { rank } => Ok(FcpeInput::Waveform { rank }),
        MainInput::LogMel { .. } => {
            if dims.len() != 3 {
                return Err(format!("takes a rank-{} input {dims:?}; expected a waveform or a [1, T, {N_MELS}] log-mel", dims.len()));
            }
            let fixed = |d: i64| usize::try_from(d).ok().filter(|&v| v > 0);
            let (d1, d2) = (fixed(dims[1]), fixed(dims[2]));
            match (d1, d2) {
                // The community waveform export: `[1, n_samples, 1]`.
                (_, Some(1)) => Ok(FcpeInput::Waveform { rank: 3 }),
                (_, Some(N_MELS)) | (None, None) => Ok(FcpeInput::Mel),
                (Some(N_MELS), _) => Err(format!(
                    "takes a channel-major [1, {N_MELS}, T] mel, RMVPE's layout (RVC's rmvpe.onnx), not torchfcpe's [1, T, {N_MELS}]; \
                     set an RMVPE export as the RMVPE model and choose rmvpe as the F0 method"
                )),
                _ => Err(format!(
                    "takes a {dims:?} spectrogram; Sigil computes torchfcpe's {N_MELS}-band log-mel [1, T, {N_MELS}]"
                )),
            }
        }
    }
}

/// What a mel export returns (output 0), decided once at load from its declared shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MelOutput {
    /// The 360-bin latent: declared `[_, T, 360]`, or `[T, 360]` with a first axis that is not a
    /// fixed 1.
    Latent,
    /// F0 per frame: declared `[T]`, `[1, T, 1]`, or `[1, T]` with `T` not a fixed 360 (Hz or
    /// `ln(1 + f0 / 700)`).
    F0,
    /// The declaration does not say (undeclared, or `[-1, -1]` / `[1, 360]` / `[1, -1, -1]`): each
    /// call decides from its shape and the frame count fed (`[1, 360]` is F0 when 360 frames were
    /// fed, else one latent frame).
    ByShape,
}

/// [`MelOutput`] of a declared output shape (`-1` = dynamic).
fn mel_output_kind(declared: Option<&[i64]>) -> MelOutput {
    const BINS: i64 = LATENT_BINS as i64;
    match declared {
        Some([_, _, BINS]) => MelOutput::Latent,
        Some([d0, BINS]) if *d0 != 1 => MelOutput::Latent,
        Some([_]) | Some([_, _, 1]) => MelOutput::F0,
        Some([1, t]) if *t != BINS => MelOutput::F0,
        _ => MelOutput::ByShape,
    }
}

/// Whether a call's output (`dims`) is the latent, for `kind` and `fed` frames fed.
fn output_is_latent(kind: MelOutput, dims: &[usize], fed: usize) -> bool {
    match kind {
        MelOutput::Latent => true,
        MelOutput::F0 => false,
        MelOutput::ByShape => match dims {
            [_, _, LATENT_BINS] => true,
            // `[1, 360]`: one latent frame when one frame was fed, else F0 (of the 360 frames fed).
            [t, LATENT_BINS] => *t != 1 || fed == 1,
            _ => false,
        },
    }
}

/// RVC's `rmvpe.onnx` layout: one channel-major 128-band mel input `[_, 128, T]` (besides an
/// optional threshold) and a 360-bin salience output `[.., 360]`. torchfcpe's mel exports take
/// `[1, T, 128]`.
fn is_rmvpe_mel_layout(inputs: &[GraphValue], outputs: &[GraphValue]) -> bool {
    let audio: Vec<&GraphValue> = inputs
        .iter()
        .filter(|i| {
            let n = i.name.to_ascii_lowercase();
            !(n.contains("threshold") || n == "thred")
        })
        .collect();
    let [input] = audio.as_slice() else {
        return false;
    };
    let channel_major = matches!(input.dims.as_deref(), Some([_, 128, t]) if *t != 128);
    channel_major && outputs.iter().any(|o| o.last_dim() == Some(LATENT_BINS))
}

pub struct FcpePitchExtractor {
    session: Session,
    inputs: PitchInputs,
    input: FcpeInput,
    /// DirectML: every call runs the first call's length, front-padded with silence in whole
    /// 10 ms frames (`pitch_onnx::ShapePin`); other EPs run each call as it is.
    pin: ShapePin,
    padded: Vec<f32>,
    /// The mel front end of mel exports.
    mel: Option<FcpeMel>,
    /// What a mel export returns (unused for waveform exports).
    mel_output: MelOutput,
    /// EP outcome of the load (`vc.pitch`).
    outcome: EpOutcome,
}

impl FcpePitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("FCPE model path is empty".into());
        }
        let file = std::path::Path::new(path);
        if let Some(problem) = non_onnx_file_problem(file) {
            return Err(format!("FCPE: {problem}; set an FCPE ONNX export (vc/pitch-fcpe), or switch F0 to rmvpe or dio"));
        }
        let (session, outcome) = build_session(Some("vc.pitch"), execution_provider_hint, opts, ModelSource::File(file))
            .map_err(|e| format!("Failed to load FCPE ONNX {}: {}", path, e))?;
        if let Some(why) = not_fcpe(&session, file) {
            return Err(format!("FCPE: {why}"));
        }
        let inputs = PitchInputs::inspect(&session, "FCPE", path, FCPE_THRESHOLD)?;
        let input = classify_input(inputs.main, inputs.main_dims()).map_err(|e| format!("FCPE ONNX {path} {e}"))?;
        let mel = (input == FcpeInput::Mel).then(FcpeMel::new);
        let mel_output = mel_output_kind(graph_outputs(&session).first().and_then(|o| o.dims.as_deref()));
        Ok(Self {
            session,
            inputs,
            input,
            pin: ShapePin::for_ep(&outcome.active),
            padded: Vec::new(),
            mel,
            mel_output,
            outcome,
        })
    }

    /// EP outcome of the load (`vc.pitch`), for callers that report where F0 runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// A waveform export's F0 (`rank` 1 / 2 / 3).
    fn f0_waveform(&mut self, mono_16k: &[f32], rank: usize) -> Result<Vec<f32>, String> {
        let n = mono_16k.len();
        let run_n = self.pin.size_aligned(n, HOP);
        let lead_frames = (run_n - n) / HOP;
        let audio: &[f32] = if run_n == n {
            mono_16k
        } else {
            self.padded.clear();
            self.padded.resize(run_n - n, 0.0);
            self.padded.extend_from_slice(mono_16k);
            &self.padded
        };
        let shape = if rank == 3 { vec![1, run_n, 1] } else { waveform_shape(rank, run_n) };
        let outputs = self.inputs.run(&mut self.session, shape, audio, "FCPE")?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("FCPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = flat_f0(&dims, data).ok_or_else(|| format!("FCPE output shape {dims:?} unsupported; expected F0 [1, T] or [T]"))?;
        if lead_frames > 0 {
            f0.drain(..lead_frames.min(f0.len()));
        }
        for v in f0.iter_mut() {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
            }
        }
        Ok(f0)
    }

    /// A mel export's F0: the log-mel here (end-aligned in the DirectML pin, the frames before the
    /// audio's at the silence floor), the graph, then the output decode.
    fn f0_mel(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let frames = mono_16k.len() / HOP + 1;
        let padded = self.pin.size(frames);
        let lead = padded - frames;
        let mel = self.mel.as_mut().ok_or("FCPE mel front end missing")?;
        self.padded.clear();
        self.padded.resize(N_MELS * padded, MEL_CLAMP.ln());
        mel.compute_into(mono_16k, padded, &mut self.padded)?;
        let outputs = self.inputs.run(&mut self.session, vec![1, padded, N_MELS], &self.padded, "FCPE")?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("FCPE output: {}", e))?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let mut f0 = decode_mel_output(&dims, data, self.mel_output, padded)?;
        if f0.len() > frames {
            f0.drain(..f0.len() - frames);
        } else if lead > 0 && f0.len() == padded {
            f0.drain(..lead);
        }
        Ok(f0)
    }
}

/// Why `session` (of `path`) is not an FCPE model, when it is clearly something else: another kind
/// of graph (a content encoder, w-okada's fused RMVPE pipeline, a voice), RVC's mel RMVPE
/// (`rmvpe.onnx`: a channel-major `[1, 128, T]` mel in, 360-bin salience out, no metadata), or one
/// of w-okada's RMVPE waveform exports (wok000 `rmvpe_20231006.onnx` / `rmvpe_s.onnx`, whose
/// metadata says it was converted from `rmvpe.pt`): they take the same waveform + threshold inputs
/// as FCPE but want RMVPE's threshold.
fn not_fcpe(session: &Session, path: &std::path::Path) -> Option<String> {
    if let Some(problem) = foreign_graph_problem(session, path, "an FCPE pitch model", &[GraphKind::PitchModel]) {
        return Some(problem);
    }
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if is_rmvpe_mel_layout(&graph_inputs(session), &graph_outputs(session)) {
        return Some(format!(
            "{file} is an RMVPE export (a [1, 128, T] mel in and 360-bin salience out, RVC's rmvpe.onnx), not FCPE; set it as the RMVPE model and choose rmvpe as the F0 method"
        ));
    }
    let note = session.metadata().ok().and_then(|m| m.custom("metadata")).unwrap_or_default();
    if note.to_ascii_lowercase().contains("rmvpe") {
        return Some(format!(
            "{file} is an RMVPE export (w-okada's, converted from rmvpe.pt), not FCPE; set it as the RMVPE model and choose rmvpe as the F0 method"
        ));
    }
    None
}

/// F0 in Hz from a mel export's output (`dims`, `fed` frames fed; see the module docs): the 360-bin
/// latent, or a direct F0 (Hz, or `ln(1 + f0 / 700)`), as `kind` (decided at load) says. Unvoiced,
/// non-finite and negative frames are 0.
fn decode_mel_output(dims: &[usize], data: &[f32], kind: MelOutput, fed: usize) -> Result<Vec<f32>, String> {
    if output_is_latent(kind, dims, fed) {
        let t = match dims {
            [1, t, LATENT_BINS] | [t, LATENT_BINS] => *t,
            _ => return Err(format!("FCPE output shape {dims:?} unsupported; the export declares the 360-bin latent [1, T, 360]")),
        };
        if data.len() < t * LATENT_BINS {
            return Err(format!("FCPE output {dims:?} holds {} values", data.len()));
        }
        let table = cent_table();
        return Ok(data[..t * LATENT_BINS]
            .chunks_exact(LATENT_BINS)
            .map(|row| latent_to_f0(row, &table, FCPE_THRESHOLD))
            .collect());
    }
    let mut f0 = flat_f0(dims, data).ok_or_else(|| format!("FCPE output shape {dims:?} unsupported; expected the latent [1, T, 360] or F0 [1, T]"))?;
    for v in f0.iter_mut() {
        if !v.is_finite() || *v < 0.0 {
            *v = 0.0;
        }
    }
    // `ln(1 + f0 / 700)` tops out at ln(1 + 1975.5 / 700) ≈ 1.34; a voiced F0 in Hz is >= 32.7.
    let peak = f0.iter().copied().fold(0.0f32, f32::max);
    if peak > 0.0 && peak < MIN_VOICED_HZ {
        for v in f0.iter_mut() {
            *v = if *v > 0.0 { 700.0 * v.exp_m1() } else { 0.0 };
        }
    }
    Ok(f0)
}

/// torchfcpe's cent table: `linspace(f0_to_cent(32.70), f0_to_cent(1975.5), 360)`,
/// `f0_to_cent(f) = 1200 log2(f / 10)`.
fn cent_table() -> Arc<[f32]> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Arc<[f32]>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let lo = 1200.0 * (LATENT_F0_MIN / 10.0).log2();
            let hi = 1200.0 * (LATENT_F0_MAX / 10.0).log2();
            (0..LATENT_BINS)
                .map(|i| (lo + (hi - lo) * i as f64 / (LATENT_BINS - 1) as f64) as f32)
                .collect()
        })
        .clone()
}

/// torchfcpe's `latent2cents_local_decoder` + `cent_to_f0` for one frame: the salience-weighted mean
/// cents over argmax ± 4 bins (indices clamped to the table, so an edge bin counts more than once,
/// as `torch.gather` with clamped indices does), `10 · 2^(c / 1200)` Hz; 0 (unvoiced) when the peak
/// is at most `threshold` or the mean is not finite.
fn latent_to_f0(row: &[f32], table: &[f32], threshold: f32) -> f32 {
    let (peak_i, peak) = row
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, v)| if v > best.1 { (i, v) } else { best });
    if peak.is_nan() || peak <= threshold {
        return 0.0;
    }
    let last = row.len().min(table.len()) as isize - 1;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for k in -4isize..=4 {
        let i = (peak_i as isize + k).clamp(0, last) as usize;
        num += f64::from(row[i]) * f64::from(table[i]);
        den += f64::from(row[i]);
    }
    let cents = num / den;
    let hz = 10.0 * (cents / 1200.0).exp2();
    if hz.is_finite() && hz > 0.0 {
        hz as f32
    } else {
        0.0
    }
}

impl PitchExtractor for FcpePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            return Ok(Vec::new());
        }
        let mut f0 = match self.input {
            FcpeInput::Waveform { rank } => self.f0_waveform(mono_16k, rank)?,
            FcpeInput::Mel => self.f0_mel(mono_16k)?,
        };
        smooth_f0(&mut f0, 3);
        Ok(f0)
    }

    // No `receptive_frames` (the F0 cache's default lead-in, 12 frames). Measured at the P3 final
    // gate: a 20-frame lead-in improves the pitch track against whole-clip F0 (V/UV 3.31 % -> 2.85 %,
    // `engine::f0_cache::tests::lead_sweep_against_whole_clip_f0`), but the converted audio did not
    // follow on the benchmark (`--headless bench` fidelity, 5 clips, the default voice): on CUDA with
    // 40 ms blocks DNSMOS fell 0.06-0.18 on every clip and the speaker cosine by 0.009 on average;
    // on the CPU Streamer preset (80 ms blocks) it was neutral. Revisit with the P7 corpus.
}

// ---------------------------------------------------------------------------------------------
// torchfcpe's log-mel front end
// ---------------------------------------------------------------------------------------------

/// librosa's Slaney mel scale (`htk=False`): linear below 1 kHz, logarithmic above.
fn hz_to_mel_slaney(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// librosa `filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000)` (Slaney scale and
/// norm), each band as `(first FFT bin, weights)`.
fn slaney_filterbank() -> Vec<(usize, Vec<f32>)> {
    let n_freqs = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs).map(|k| k as f64 * SR / N_FFT as f64).collect();
    let (lo, hi) = (hz_to_mel_slaney(MEL_FMIN), hz_to_mel_slaney(MEL_FMAX));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|i| {
            let (f0, f1, f2) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (f2 - f0);
            let weights: Vec<f64> = fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - f0) / (f1 - f0);
                    let upper = (f2 - f) / (f2 - f1);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect();
            let first = weights.iter().position(|&w| w > 0.0).unwrap_or(0);
            let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
            let band = if last >= first && weights[first] > 0.0 {
                weights[first..=last].iter().map(|&w| w as f32).collect()
            } else {
                Vec::new()
            };
            (first, band)
        })
        .collect()
}

/// torch `reflect` padding index (no edge repeat).
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

/// The log-mel of torchfcpe's `Wav2Mel` at 16 kHz (see the module docs), with reused buffers.
struct FcpeMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    bands: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
}

impl FcpeMel {
    fn new() -> FcpeMel {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // torch.hann_window(1024) is periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        FcpeMel {
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            bands: slaney_filterbank(),
            magnitude: vec![0.0; N_FFT / 2 + 1],
        }
    }

    /// The log-mel of `audio` (`audio.len() / 160 + 1` frames) into the last frames of `out`
    /// (`padded_frames` frames, `[T, 128]`); the frames before them keep what `out` held (the
    /// caller fills the silence floor). Returns the audio's frames.
    fn compute_into(&mut self, audio: &[f32], padded_frames: usize, out: &mut [f32]) -> Result<usize, String> {
        let n = audio.len();
        let frames = n / HOP + 1;
        if padded_frames < frames || out.len() != N_MELS * padded_frames {
            return Err(format!("FCPE mel buffer: {} values for {frames} frames padded to {padded_frames}", out.len()));
        }
        // torch.nn.functional.pad(y, (432, max(432, 1024 - n - 432))), reflect when the right pad is
        // shorter than the audio, else zeros.
        let pad_right = PAD_RIGHT.max(N_FFT.saturating_sub(n + PAD_LEFT));
        let reflect = pad_right < n;
        let total = n + PAD_LEFT + pad_right;
        let stft_frames = if total >= N_FFT { (total - N_FFT) / HOP + 1 } else { 0 };
        let lead = padded_frames - frames;
        let floor = MEL_CLAMP.ln();
        for t in 0..frames {
            // Wav2Mel repeats the last STFT frame when the STFT has fewer frames than `n / 160 + 1`.
            let src = t.min(stft_frames.saturating_sub(1));
            let origin = (src * HOP) as isize - PAD_LEFT as isize;
            for (k, (dst, &w)) in self.frame.iter_mut().zip(self.window.iter()).enumerate() {
                let i = origin + k as isize;
                let x = if (0..n as isize).contains(&i) {
                    audio[i as usize]
                } else if reflect && n > 1 {
                    audio[reflect_index(i, n)]
                } else {
                    0.0
                };
                *dst = x * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("FCPE mel FFT: {e}"))?;
            for (m, c) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
                *m = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
            }
            let row = &mut out[(lead + t) * N_MELS..(lead + t + 1) * N_MELS];
            for (v, (first, weights)) in row.iter_mut().zip(self.bands.iter()) {
                let energy: f32 = weights.iter().zip(&self.magnitude[*first..]).map(|(w, m)| w * m).sum();
                *v = if stft_frames == 0 { floor } else { energy.max(MEL_CLAMP).ln() };
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};

    fn tone(n: usize, hz: f32, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / 16_000.0).sin()).collect()
    }

    fn temp_model(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("sigil-fcpe-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("fcpe.onnx");
        std::fs::write(&p, bytes).unwrap();
        (dir, p)
    }

    /// librosa reference values (`librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=0,
    /// fmax=8000)`, librosa 0.10.2; the last band spans bins 488..=511).
    #[test]
    fn slaney_filterbank_matches_librosa() {
        let bands = slaney_filterbank();
        assert_eq!(bands.len(), N_MELS);
        // (band, first bin, weights) from librosa 0.10.2 (float32).
        let expect: [(usize, usize, &[f32]); 3] = [
            (0, 1, &[0.028_577_78, 0.028_377_54]),
            (1, 2, &[0.014_389_01, 0.042_566_31, 0.013_988_53]),
            (64, 107, &[0.001_910_408, 0.011_216_71, 0.020_523, 0.018_818_28, 0.009_733_714, 0.000_649_144_5]),
        ];
        for (band, first, w) in expect {
            let (f, got) = &bands[band];
            assert_eq!(*f, first, "band {band} starts at bin {f}");
            assert_eq!(got.len(), w.len(), "band {band}: {got:?}");
            for (a, b) in got.iter().zip(w) {
                assert!((a - b).abs() < 1e-6, "band {band}: {got:?} vs {w:?}");
            }
        }
        let (first, w) = &bands[N_MELS - 1];
        assert_eq!((*first, w.len()), (488, 24));
        assert!((w[12] - 0.005_223_189).abs() < 1e-6, "{w:?}");
    }

    /// The log-mel equals torchfcpe's `Wav2Mel` (reference computed with torch 2.3 `torch.stft`
    /// and librosa's filterbank exactly as torchfcpe's `MelModule` does; see the module docs) on a
    /// 1 s two-tone signal.
    #[test]
    fn log_mel_matches_torchfcpe() {
        // Built in f64 and rounded once, like the numpy reference (an f32 phase drifts audibly far
        // below the tones, where the near-floor bands live).
        let audio: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = f64::from(i) / 16_000.0;
                (0.3 * (std::f64::consts::TAU * 220.0 * t).sin() + 0.1 * (std::f64::consts::TAU * 1_830.0 * t).sin()) as f32
            })
            .collect();
        let mut mel = FcpeMel::new();
        let frames = audio.len() / HOP + 1;
        let mut out = vec![MEL_CLAMP.ln(); N_MELS * frames];
        assert_eq!(mel.compute_into(&audio, frames, &mut out).unwrap(), 101);
        // (frame, band, value) from the torch reference (torch 2.3.1 + librosa 0.10.2).
        let expect: [(usize, usize, f32); 10] = [
            (0, 0, -0.936_682),
            (0, 13, -1.002_044),
            (1, 5, -0.742_249),
            (37, 90, -11.512_925),
            (50, 13, -6.565_802),
            (50, 60, -10.259_079),
            (50, 79, -11.512_925),
            (99, 13, -0.992_296),
            (100, 13, -0.992_296),
            (100, 127, -5.996_698),
        ];
        for (t, b, v) in expect {
            let got = out[t * N_MELS + b];
            assert!((got - v).abs() < 5e-4, "frame {t} band {b}: {got} vs {v}");
        }
        // 220 Hz is band 8's.
        let row = &out[50 * N_MELS..51 * N_MELS];
        let best = (0..N_MELS).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        assert_eq!(best, 8);
        // The last frame repeats the last STFT frame (`n / 160 + 1` frames, one more than the STFT).
        assert_eq!(&out[99 * N_MELS..100 * N_MELS], &out[100 * N_MELS..101 * N_MELS]);
        // Padded (the DirectML pin): the same frames end-aligned after the silence floor.
        let mut padded = vec![MEL_CLAMP.ln(); N_MELS * (frames + 3)];
        mel.compute_into(&audio, frames + 3, &mut padded).unwrap();
        assert_eq!(padded[(3 + 50) * N_MELS + 13], out[50 * N_MELS + 13]);
        assert!(padded[..3 * N_MELS].iter().all(|&v| v == MEL_CLAMP.ln()));
        // Short inputs (zero padding) stay finite.
        let mut short = vec![0.0; N_MELS * 2];
        mel.compute_into(&tone(200, 300.0, 0.2), 2, &mut short).unwrap();
        assert!(short.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn latent_decodes_like_torchfcpe() {
        let table = cent_table();
        assert!((table[0] - 2_051.148_8).abs() < 1e-2, "{}", table[0]);
        assert!((table[LATENT_BINS - 1] - 9_151.289).abs() < 1e-2, "{}", table[LATENT_BINS - 1]);
        // A symmetric peak decodes to the bin's cents.
        let mut row = vec![0.0f32; LATENT_BINS];
        row[99] = 0.4;
        row[100] = 0.9;
        row[101] = 0.4;
        let hz = latent_to_f0(&row, &table, FCPE_THRESHOLD);
        let want = 10.0 * (f64::from(table[100]) / 1200.0).exp2();
        assert!((f64::from(hz) - want).abs() < 1e-3, "{hz} vs {want}");
        // Below the threshold: unvoiced.
        let quiet = vec![0.005f32; LATENT_BINS];
        assert_eq!(latent_to_f0(&quiet, &table, FCPE_THRESHOLD), 0.0);
        // Edge clamping counts bin 0 five times (torch.gather with clamped indices).
        let mut edge = vec![0.0f32; LATENT_BINS];
        edge[0] = 0.5;
        edge[4] = 0.5;
        let c = (5.0 * 0.5 * f64::from(table[0]) + 0.5 * f64::from(table[4])) / 3.0;
        let want = 10.0 * (c / 1200.0).exp2();
        assert!((f64::from(latent_to_f0(&edge, &table, FCPE_THRESHOLD)) - want).abs() < 1e-3);
        // NaN rows are unvoiced, never NaN.
        assert_eq!(latent_to_f0(&[f32::NAN; LATENT_BINS], &table, FCPE_THRESHOLD), 0.0);
    }

    #[test]
    fn direct_f0_outputs_in_hz_or_log_f0() {
        assert_eq!(decode_mel_output(&[1, 3], &[0.0, 220.0, -1.0], MelOutput::F0, 3).unwrap(), vec![0.0, 220.0, 0.0]);
        // ln(1 + f0 / 700) of 220 Hz and 0.
        let x = (1.0f32 + 220.0 / 700.0).ln();
        let got = decode_mel_output(&[1, 2, 1], &[x, 0.0], MelOutput::F0, 2).unwrap();
        assert!((got[0] - 220.0).abs() < 1e-2 && got[1] == 0.0, "{got:?}");
        assert!(decode_mel_output(&[2, 2, 2], &[0.0; 8], MelOutput::ByShape, 2).is_err());
        assert!(decode_mel_output(&[2, 2, 2], &[0.0; 8], MelOutput::Latent, 2).is_err());
    }

    /// A direct-F0 export's output of exactly 360 frames (`[1, 360]`, a 57,440-57,599-sample call
    /// or a 360-frame DirectML pin) is 360 F0 values, never one latent row (P5-D review).
    #[test]
    fn a_360_frame_f0_output_is_not_a_latent_row() {
        assert_eq!(mel_output_kind(Some(&[1, -1])), MelOutput::F0);
        assert_eq!(mel_output_kind(Some(&[1, -1, 1])), MelOutput::F0);
        assert_eq!(mel_output_kind(Some(&[-1])), MelOutput::F0);
        assert_eq!(mel_output_kind(Some(&[1, -1, 360])), MelOutput::Latent);
        assert_eq!(mel_output_kind(Some(&[-1, 360])), MelOutput::Latent);
        for undecided in [None, Some(&[-1, -1][..]), Some(&[1, 360][..]), Some(&[1, -1, -1][..])] {
            assert_eq!(mel_output_kind(undecided), MelOutput::ByShape, "{undecided:?}");
        }
        let hz = vec![220.0f32; LATENT_BINS];
        for kind in [MelOutput::F0, MelOutput::ByShape] {
            let f0 = decode_mel_output(&[1, LATENT_BINS], &hz, kind, LATENT_BINS).unwrap();
            assert_eq!(f0.len(), LATENT_BINS, "{kind:?}");
            assert!(f0.iter().all(|&v| v == 220.0), "{kind:?}");
        }
        // Undeclared, one frame fed: `[1, 360]` is that frame's latent row.
        let table = cent_table();
        let mut row = vec![0.0f32; LATENT_BINS];
        row[100] = 0.9;
        let f0 = decode_mel_output(&[1, LATENT_BINS], &row, MelOutput::ByShape, 1).unwrap();
        assert_eq!(f0.len(), 1);
        assert!((f64::from(f0[0]) - 10.0 * (f64::from(table[100]) / 1200.0).exp2()).abs() < 1e-2, "{f0:?}");
        // A declared latent keeps decoding `[t, 360]` rows.
        assert_eq!(decode_mel_output(&[1, LATENT_BINS], &row, MelOutput::Latent, LATENT_BINS).unwrap().len(), 1);
    }

    #[test]
    fn input_layouts_are_told_apart() {
        let mel = MainInput::LogMel { n_mels: None };
        assert_eq!(classify_input(mel, &[1, -1, 1]).unwrap(), FcpeInput::Waveform { rank: 3 });
        assert_eq!(classify_input(mel, &[1, -1, 128]).unwrap(), FcpeInput::Mel);
        assert_eq!(classify_input(mel, &[1, -1, -1]).unwrap(), FcpeInput::Mel);
        // Channel-major is RMVPE's layout (RVC's rmvpe.onnx), never fed torchfcpe's mel.
        let err = classify_input(mel, &[1, 128, -1]).unwrap_err();
        assert!(err.contains("RMVPE") && err.contains("[1, T, 128]"), "{err}");
        assert!(classify_input(mel, &[1, 80, -1]).unwrap_err().contains("128"));
        assert!(classify_input(mel, &[1, 128]).unwrap_err().contains("rank-2"), "never indexes past the declared dims");
        assert_eq!(
            classify_input(MainInput::Waveform { rank: 2 }, &[1, -1]).unwrap(),
            FcpeInput::Waveform { rank: 2 }
        );
    }

    /// A stand-in mel FCPE: mel `[1, T, 128]` → latent `[1, T, 360]` that peaks at bin 100 on every
    /// frame (sigmoid of a constant row), or F0 `[1, T]` (`hz`).
    fn fake_mel_fcpe(hz: bool) -> Vec<u8> {
        let mut row = vec![-8.0f32; LATENT_BINS];
        row[99] = 0.0;
        row[100] = 3.0;
        row[101] = 0.0;
        let mut g = Graph::new();
        g.input("mel", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(N_MELS as u64)]);
        g.init_i64("axes2", &[1], &[2]).init_f32("zero", &[1], &[0.0]);
        // Reduce the mel to [1, T, 1] times 0 (keeps T), then add the constant row.
        g.node("ReduceSum", &["mel", "axes2"], &["s"], &[Attr::Int("keepdims", 1)])
            .node("Mul", &["s", "zero"], &["z"], &[]);
        if hz {
            g.init_f32("f0", &[1], &[220.0])
                .node("Add", &["z", "f0"], &["f3"], &[])
                .node("Squeeze", &["f3", "axes2"], &["f0_hz"], &[])
                .output("f0_hz", FLOAT, &[Dim::Fixed(1), Dim::Named("t")]);
        } else {
            g.init_f32("row", &[1, 1, LATENT_BINS as u64], &row)
                .node("Add", &["z", "row"], &["logits"], &[])
                .node("Sigmoid", &["logits"], &["latent"], &[])
                .output("latent", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(LATENT_BINS as u64)]);
        }
        g.model_bytes(13)
    }

    #[test]
    fn mel_exports_run_and_decode() {
        let table = cent_table();
        let want = (10.0 * (f64::from(table[100]) / 1200.0).exp2()) as f32;
        for (tag, hz) in [("latent", false), ("hz", true)] {
            let (dir, p) = temp_model(tag, &fake_mel_fcpe(hz));
            let mut e = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default()).unwrap();
            assert_eq!(e.input, FcpeInput::Mel);
            assert_eq!(e.mel_output, if hz { MelOutput::F0 } else { MelOutput::Latent }, "{tag}");
            let f0 = e.f0(&tone(3_200, 150.0, 0.2)).unwrap();
            assert_eq!(f0.len(), 21, "{tag}: 3200 / 160 + 1 frames");
            let target = if hz { 220.0 } else { want };
            assert!(f0.iter().all(|&v| (v - target).abs() < 0.05), "{tag}: {f0:?}");
            drop(e);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The community export's `[1, N, 1]` waveform input runs (it used to be refused as a mel input).
    #[test]
    fn rank3_waveform_exports_run() {
        let mut g = Graph::new();
        g.input("audio", FLOAT, &[Dim::Fixed(1), Dim::Named("n"), Dim::Fixed(1)])
            .output("f0_hz", FLOAT, &[Dim::Fixed(1), Dim::Named("f"), Dim::Fixed(1)])
            .init_i64("axes", &[1], &[1])
            .init_f32("k", &[1], &[0.0])
            .init_f32("f0", &[1], &[180.0])
            .init_i64("starts", &[1], &[0])
            .init_i64("ends", &[1], &[i64::MAX])
            .init_i64("steps", &[1], &[160])
            .node("Slice", &["audio", "starts", "ends", "axes", "steps"], &["frames"], &[])
            .node("Mul", &["frames", "k"], &["z"], &[])
            .node("Add", &["z", "f0"], &["f0_hz"], &[]);
        let (dir, p) = temp_model("rank3", &g.model_bytes(13));
        let mut e = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default()).unwrap();
        assert_eq!(e.input, FcpeInput::Waveform { rank: 3 });
        let f0 = e.f0(&tone(1_600, 180.0, 0.2)).unwrap();
        assert_eq!(f0.len(), 10);
        assert!(f0.iter().all(|&v| (v - 180.0).abs() < 1e-3), "{f0:?}");
        drop(e);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Files that are not FCPE models are refused by what they are (P5-D audit).
    #[test]
    fn other_files_are_refused_specifically() {
        let (dir, p) = temp_model("pt", b"PK\x03\x04 a torch checkpoint");
        let err = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default())
            .err()
            .expect("a zip is refused");
        assert!(err.contains("PyTorch") && err.contains("not an ONNX graph"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
        // A content encoder (the test stand-in: audio [1, n] -> feats [1, T, 768]).
        let (dir, p) = temp_model("hubert", &crate::services::neural_audio::vc::test_models::content_encoder_bytes());
        let err = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default())
            .err()
            .expect("a content encoder is refused");
        assert!(err.contains("content encoder"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// RVC's mel RMVPE as its file declares it (`rmvpe.onnx` in the dev cache, ORT 1.28: input
    /// `input` `[1, 128, time]`, output `[?, ?, 360]`, no metadata) set as the FCPE model is refused as
    /// RMVPE at load (P5-D review: it used to load as a channel-major mel FCPE and give detuned F0, or
    /// fail on every frame count that is not a multiple of 32).
    #[test]
    fn rvc_mel_rmvpe_is_not_fcpe() {
        use ort::value::TensorElementType::Float32 as F32;
        let gv = GraphValue::new;
        let rmvpe_in = [gv("input", F32, &[1, 128, -1])];
        let rmvpe_out = [gv("output", F32, &[-1, -1, 360])];
        assert!(is_rmvpe_mel_layout(&rmvpe_in, &rmvpe_out));
        assert!(is_rmvpe_mel_layout(&[gv("mel", F32, &[1, 128, -1]), gv("threshold", F32, &[1])], &rmvpe_out));
        // torchfcpe's time-major mel FCPE (latent out) and a waveform pitch model are not.
        assert!(!is_rmvpe_mel_layout(&[gv("mel", F32, &[1, -1, 128])], &[gv("latent", F32, &[1, -1, 360])]));
        assert!(!is_rmvpe_mel_layout(&[gv("audio", F32, &[1, -1])], &[gv("f0", F32, &[1, -1])]));
        // The real ONNX Runtime path with the same declarations.
        let mut row = vec![-8.0f32; LATENT_BINS];
        row[100] = 3.0;
        let mut g = Graph::new();
        g.input("input", FLOAT, &[Dim::Fixed(1), Dim::Fixed(N_MELS as u64), Dim::Named("time")])
            .output("output", FLOAT, &[Dim::Named("b"), Dim::Named("t"), Dim::Fixed(LATENT_BINS as u64)])
            .init_i64("axes2", &[1], &[2])
            .init_f32("zero", &[1], &[0.0])
            .init_f32("row", &[1, 1, LATENT_BINS as u64], &row)
            .node("Transpose", &["input"], &["tm"], &[Attr::Ints("perm", vec![0, 2, 1])])
            .node("ReduceSum", &["tm", "axes2"], &["s"], &[Attr::Int("keepdims", 1)])
            .node("Mul", &["s", "zero"], &["z"], &[])
            .node("Add", &["z", "row"], &["output"], &[]);
        let (dir, p) = temp_model("rvc-rmvpe", &g.model_bytes(13));
        let p = {
            let renamed = dir.join("rmvpe.onnx");
            std::fs::rename(&p, &renamed).unwrap();
            renamed
        };
        let err = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default())
            .err()
            .expect("RVC's rmvpe.onnx is not FCPE");
        assert!(err.contains("rmvpe.onnx is an RMVPE export") && err.contains("set it as the RMVPE model"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Dev rig: the real `rmvpe.onnx` (RVC's, in the dev cache) set as the FCPE model is refused as
    /// RMVPE. `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_rmvpe_is_not_fcpe -- --ignored --nocapture`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with upstream-assets/rmvpe/rmvpe.onnx"]
    fn real_rmvpe_is_not_fcpe() {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR not set; skipped");
            return;
        };
        let p = root.join("upstream-assets").join("rmvpe").join("rmvpe.onnx");
        if !p.is_file() {
            eprintln!("{} not found; skipped", p.display());
            return;
        }
        let err = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default())
            .err()
            .expect("RVC's rmvpe.onnx is not FCPE");
        eprintln!("{err}");
        assert!(err.contains("is an RMVPE export"), "{err}");
    }

    /// w-okada's RMVPE waveform exports (wok000 `rmvpe_20231006.onnx` / `rmvpe_s.onnx`: waveform +
    /// threshold in, like FCPE) are refused as FCPE by the note their metadata carries.
    #[test]
    fn wokada_rmvpe_exports_are_not_fcpe() {
        let mut g = Graph::new();
        g.input("waveform", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .input("threshold", FLOAT, &[Dim::Fixed(1)])
            .output("pitchf", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .node("Mul", &["waveform", "threshold"], &["pitchf"], &[]);
        let mut bytes = g.model_bytes(13);
        // ModelProto.metadata_props (field 14): {"metadata": "{\"note\": \"converted from ...rmvpe.pt\"}"}.
        let (key, value) = ("metadata", r#"{"note": "converted from https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"}"#);
        let mut entry = vec![0x0A, key.len() as u8];
        entry.extend_from_slice(key.as_bytes());
        entry.push(0x12);
        entry.push(value.len() as u8);
        entry.extend_from_slice(value.as_bytes());
        bytes.push(0x72);
        bytes.push(entry.len() as u8);
        bytes.extend_from_slice(&entry);
        let (dir, p) = temp_model("wokada-rmvpe", &bytes);
        let err = FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &VcSessionOptions::default())
            .err()
            .expect("an RMVPE export is not FCPE");
        assert!(err.contains("RMVPE export"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A glide from 140 to 280 Hz over 2 s with a 250 ms silent gap (the dev-rig comparisons).
    fn glide() -> Vec<f32> {
        let mut audio = Vec::new();
        let mut phase = 0.0f64;
        for i in 0..32_000 {
            let hz = 140.0 + 140.0 * f64::from(i) / 32_000.0;
            phase += std::f64::consts::TAU * hz / 16_000.0;
            let gate = if (12_000..16_000).contains(&i) { 0.0 } else { 0.3 };
            audio.push((gate * phase.sin()) as f32);
        }
        audio
    }

    /// (frames voiced in both, voicing disagreements, worst |Δ| in Hz over the frames voiced in both).
    fn compare_f0(a: &[f32], b: &[f32]) -> (usize, usize, f32) {
        let both = a.iter().zip(b).filter(|(x, y)| **x > 0.0 && **y > 0.0).count();
        let disagree = a.iter().zip(b).filter(|(x, y)| (**x > 0.0) != (**y > 0.0)).count();
        let worst = a
            .iter()
            .zip(b)
            .filter(|(x, y)| **x > 0.0 && **y > 0.0)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        (both, disagree, worst)
    }

    /// Dev rig: real mel-input FCPE graphs (the community export cut at its log-mel tensor, so the
    /// same torchfcpe 0.0.4 weights: `mel [1, T, 128]` → the 360-bin latent, or → F0 by the graph's
    /// own decode) driven by Sigil's log-mel front end and latent decode give the F0 of the whole
    /// waveform export, which computes torchfcpe's mel inside the graph.
    /// Make the two files once (no download; `onnx` 1.17):
    /// `python scripts/vc-templates/fcpe_mel_extract.py %LOCALAPPDATA%\sigil-dev-cache\upstream-assets\fcpe\fcpe.onnx`
    /// (`onnx.utils.Extractor(model).extract_model(["/model/Concat_1_output_0"], ["/model/Sigmoid_output_0"])`,
    /// resp. `["f0_hz"]`, with the input renamed `mel` and the output `latent` / `f0`), then
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_mel_fcpe -- --ignored --nocapture`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with upstream-assets/fcpe/{fcpe,fcpe-mel-latent,fcpe-mel-f0}.onnx"]
    fn real_mel_fcpe_matches_the_waveform_export() {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR not set; skipped");
            return;
        };
        let dir = root.join("upstream-assets").join("fcpe");
        let files = ["fcpe.onnx", "fcpe-mel-latent.onnx", "fcpe-mel-f0.onnx"].map(|f| dir.join(f));
        if let Some(missing) = files.iter().find(|p| !p.is_file()) {
            eprintln!("{} not found; skipped", missing.display());
            return;
        }
        let opts = VcSessionOptions::default();
        let load = |p: &std::path::Path| FcpePitchExtractor::load(p.to_str().unwrap(), "cpu", &opts).unwrap();
        let mut wave = load(&files[0]);
        let mut latent = load(&files[1]);
        let mut hz = load(&files[2]);
        assert_eq!((latent.input, latent.mel_output), (FcpeInput::Mel, MelOutput::Latent));
        assert_eq!(hz.input, FcpeInput::Mel);
        let audio = glide();
        let reference = wave.f0(&audio).unwrap();
        for (tag, e) in [("latent", &mut latent), ("f0", &mut hz)] {
            let f0 = e.f0(&audio).unwrap();
            assert_eq!(f0.len(), reference.len(), "{tag}");
            let (both, disagree, worst) = compare_f0(&f0, &reference);
            eprintln!("mel {tag} vs waveform export: {} frames, {both} voiced in both, {disagree} voicing disagreements, worst {worst} Hz", f0.len());
            assert!(both > f0.len() / 2, "{tag}: {both} of {}", f0.len());
            assert!(disagree <= f0.len() / 100, "{tag}: {disagree} voicing disagreements");
            // Measured 0.0024 Hz (Rust log-mel vs the graph's torch-exported mel, CPU).
            assert!(worst < 0.05, "{tag}: worst {worst} Hz");
        }
    }

    /// Dev rig: the community FCPE export as published (gzivdo/fcpe-onnx `fcpe.onnx`, a `[1, N, 1]`
    /// waveform, accepted since P5-D) gives the F0 of Sigil's adapter of the same weights
    /// (`vc/pitch-fcpe`, `[1, N]`).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test real_community_fcpe -- --ignored --nocapture`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with upstream-assets/fcpe/fcpe.onnx and the vc/pitch-fcpe adapter"]
    fn real_community_fcpe_matches_the_adapter() {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR not set; skipped");
            return;
        };
        let raw = root.join("upstream-assets").join("fcpe").join("fcpe.onnx");
        let Some(adapter) = super::super::f0_fusion::test_model("pitch-fcpe") else {
            eprintln!("vc/pitch-fcpe not found; skipped");
            return;
        };
        if !raw.is_file() {
            eprintln!("{} not found; skipped", raw.display());
            return;
        }
        let opts = VcSessionOptions::default();
        let mut a = FcpePitchExtractor::load(raw.to_str().unwrap(), "cpu", &opts).unwrap();
        assert_eq!(a.input, FcpeInput::Waveform { rank: 3 });
        let mut b = FcpePitchExtractor::load(adapter.to_str().unwrap(), "cpu", &opts).unwrap();
        let audio = glide();
        let (fa, fb) = (a.f0(&audio).unwrap(), b.f0(&audio).unwrap());
        assert_eq!(fa.len(), fb.len());
        let (voiced_both, disagree, worst) = compare_f0(&fa, &fb);
        eprintln!("community vs adapter: {} frames, {voiced_both} voiced in both, {disagree} voicing disagreements, worst {worst} Hz", fa.len());
        assert!(voiced_both > fa.len() / 2, "{voiced_both} of {}", fa.len());
        assert!(disagree <= fa.len() / 100, "{disagree} voicing disagreements");
        assert!(worst < 0.5, "worst {worst} Hz");
    }
}
