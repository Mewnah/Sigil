//! Smart Turn v3.2 end-of-turn classifier on Sigil's ONNX Runtime (P2-C; research §1.2).
//!
//! Model: pipecat-ai Smart Turn v3.2 (BSD-2-Clause, <https://github.com/pipecat-ai/smart-turn>),
//! the 8 MB int8 CPU export `smart-turn-v3.2-cpu.onnx` from the Hugging Face repo
//! `pipecat-ai/smart-turn-v3`, installed by the model store as [`MODEL_ID`]
//! (`model-catalog/smart-turn.json`). A Whisper Tiny encoder with a linear head decides from the
//! audio alone whether the speaker has finished their turn.
//!
//! # What it is for (and what it never does)
//! The pipeline (P2-A) calls [`SmartTurn::predict`] when the VAD closes an utterance at a pause
//! (never for a forced max-length split) and emits `speech:turn_end` at or above [`THRESHOLD`].
//! Translation, TTS and the VRChat chatbox may commit on that hint instead of waiting a fixed
//! 1.5–2.5 s. It is a **commit hint only**: it never closes, shortens, re-times or cuts a
//! segment, and caption finals are always decoded from the whole closed utterance (an independent
//! test found 35% false cut-offs at a 300 ms budget, so it must never cut speech). The classifier
//! only reads the audio it is given.
//!
//! It is not free, though: [`SmartTurn::predict`] blocks its caller for the whole inference (see
//! Runtime). The pipeline calls it on the capture thread when a pause closes an utterance, before
//! it hands the utterance to its worker (`pipeline::FrontEnd::closed`), so that the verdict (and
//! `speech:turn_end`) comes before the segment's finals. Each pause-closed caption final therefore
//! reaches its worker about one prediction later with Smart Turn on (~32 ms on the dev PC, more on
//! a slow 4-core CPU, plus the first prediction's warm-up), and the capture thread drains the hub
//! ring that much later (its ring absorbs it). Forced max-length splits never wait.
//!
//! # Input contract (upstream `inference.py` + `audio_utils.py`, verified in the tests below against
//! transformers' `WhisperFeatureExtractor(chunk_length=8)` and onnxruntime)
//! 1. The last 8 s of the turn's 16 kHz mono audio, zero-padded **at the start** to exactly 8 s.
//! 2. Zero-mean / unit-variance normalization over the whole 8 s window (`do_normalize=True`:
//!    `(x - mean) / sqrt(var + 1e-7)`).
//! 3. Whisper log-mel: periodic Hann window of 400, hop 160, centred with reflect padding, power
//!    spectrum, 80 Slaney mel filters over 0–8 kHz (Slaney normalization), `log10(max(x, 1e-10))`,
//!    the last frame dropped (800 frames), clamped to `max - 8`, then `(x + 4) / 4`.
//! 4. `input_features` f32 `[1, 80, 800]` → one f32 output (named `logits`, but it already is the
//!    sigmoid probability that the turn is complete).
//!
//! # Turn context
//! Upstream feeds the model everything since the start of the user's turn. The pipeline hands over
//! one closed utterance at a time, so [`SmartTurn`] keeps the audio of a turn it judged unfinished
//! and puts it (plus the silence of the pause, measured on the wall clock) in front of the next
//! utterance when that one closes within [`MAX_TURN_GAP`] of the previous call. A complete verdict,
//! a longer pause or [`SmartTurn::reset`] starts a new turn.
//!
//! # Runtime
//! One CPU session through `neural_audio::ort_session` ([`INTRA_THREADS`] intra-op threads,
//! sequential, spinning off; EP outcome recorded as [`EP_COMPONENT`]). Measured on the dev PC
//! (Ryzen 7 7800X3D, ORT 1.28): the model takes ~30 ms per prediction on 2 threads (~50 ms on
//! 1); window + log-mel features take ~1.6 ms in a release build (~48 ms in an unoptimized test
//! build), so a whole warm prediction is ~32 ms in release (`real_model_matches_the_python_reference`
//! prints it). That is the latency it adds to each pause-closed caption (see above). Call it on a
//! worker or capture thread, never on an audio callback.
//!
//! # Frozen signatures (P2-S0)
//! - [`MODEL_ID`]: catalog id of the model.
//! - [`SmartTurn::load`]`() -> Result<SmartTurn, String>`: `Err` (naming [`MODEL_ID`]) when the
//!   model is not installed or fails to load.
//! - [`SmartTurn::predict`]`(&mut self, audio_16k: &[f32]) -> Result<f32, String>`: turn-end
//!   probability 0..1 for audio that ends now (blocking; worker thread only).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::{Tensor, TensorElementType};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::model_store;
use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};
use crate::services::speech_core::SAMPLE_RATE;

/// Catalog id of the Smart Turn model (`model-catalog/smart-turn.json`).
pub const MODEL_ID: &str = "vad/smart-turn-v3.2";
/// The model file inside the installed entry folder.
pub const MODEL_FILE: &str = "smart-turn-v3.2-cpu.onnx";
/// Default turn-end probability threshold (upstream's `probability > 0.5`).
pub const THRESHOLD: f32 = 0.5;
/// Diagnostics component of the ORT session (Settings → GPU & diagnostics).
pub const EP_COMPONENT: &str = "stt.turn";
/// Audio the model looks at.
pub const WINDOW_MS: u32 = 8_000;
/// Longest pause that still joins two utterances into one turn (upstream's default stop timeout).
pub const MAX_TURN_GAP: Duration = Duration::from_secs(3);
/// Intra-op threads of the session: the verdict gates commits, so two threads (the cap for
/// background sessions) cut its latency from ~50 to ~30 ms for a short burst on a second core.
pub const INTRA_THREADS: usize = 2;

const WINDOW_SAMPLES: usize = (SAMPLE_RATE as usize) * (WINDOW_MS as usize) / 1000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = N_FFT / 2 + 1;
const N_MELS: usize = 80;
/// `1 + WINDOW_SAMPLES / HOP` centred frames, minus the dropped last one.
const N_FRAMES: usize = WINDOW_SAMPLES / HOP;
const FEATURES: usize = N_MELS * N_FRAMES;
const MEL_FLOOR: f64 = 1e-10;
const NORM_EPS: f64 = 1e-7;

// ---------------------------------------------------------------------------------------------
// Whisper log-mel features
// ---------------------------------------------------------------------------------------------

fn hz_to_mel_slaney(hz: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() * (27.0 / 6.4f64.ln())
    } else {
        3.0 * hz / 200.0
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((6.4f64.ln() / 27.0) * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// transformers `mel_filter_bank(201, 80, 0, 8000, 16000, norm="slaney", mel_scale="slaney")` as
/// `(first bin, weights)` per mel band (the non-zero run of each triangle).
fn mel_filterbank() -> Vec<(usize, Vec<f64>)> {
    let (lo, hi) = (hz_to_mel_slaney(0.0), hz_to_mel_slaney(f64::from(SAMPLE_RATE) / 2.0));
    let edges: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let nyquist = f64::from(SAMPLE_RATE) / 2.0;
    let fft_hz = |k: usize| nyquist * k as f64 / (N_BINS - 1) as f64;
    (0..N_MELS)
        .map(|m| {
            let (left, centre, right) = (edges[m], edges[m + 1], edges[m + 2]);
            let enorm = 2.0 / (right - left);
            let weights: Vec<(usize, f64)> = (0..N_BINS)
                .map(|k| {
                    let f = fft_hz(k);
                    let down = (f - left) / (centre - left);
                    let up = (right - f) / (right - centre);
                    (k, down.min(up).max(0.0) * enorm)
                })
                .filter(|&(_, w)| w > 0.0)
                .collect();
            let first = weights.first().map_or(0, |&(k, _)| k);
            (first, weights.into_iter().map(|(_, w)| w).collect())
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

/// Whisper's 80-bin log-mel over one 8 s window (reused buffers: no allocation per call).
struct LogMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    bands: Vec<(usize, Vec<f64>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f64>,
}

impl LogMel {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // `torch.hann_window(400)` / `window_function(400, "hann")` are periodic.
        let window = (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect();
        Self {
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            bands: mel_filterbank(),
            power: vec![0.0; N_BINS],
        }
    }

    /// `audio`: exactly [`WINDOW_SAMPLES`] normalized samples. Writes `[N_MELS, N_FRAMES]`
    /// row-major into `out`.
    fn compute(&mut self, audio: &[f32], out: &mut [f32]) -> Result<(), String> {
        if audio.len() != WINDOW_SAMPLES || out.len() != FEATURES {
            return Err(format!(
                "Smart Turn features: {} samples / {} outputs, expected {WINDOW_SAMPLES} / {FEATURES}",
                audio.len(),
                out.len()
            ));
        }
        let n = audio.len();
        let pad = (N_FFT / 2) as isize;
        let mut max = f64::NEG_INFINITY;
        for t in 0..N_FRAMES {
            let origin = (t * HOP) as isize - pad;
            for (k, (dst, &w)) in self.frame.iter_mut().zip(&self.window).enumerate() {
                *dst = audio[reflect_index(origin + k as isize, n)] * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("Smart Turn features: FFT: {e}"))?;
            for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
                *p = f64::from(c.re) * f64::from(c.re) + f64::from(c.im) * f64::from(c.im);
            }
            for (m, (first, weights)) in self.bands.iter().enumerate() {
                let energy: f64 = weights
                    .iter()
                    .zip(&self.power[*first..])
                    .map(|(w, p)| w * p)
                    .sum();
                let v = energy.max(MEL_FLOOR).log10();
                max = max.max(v);
                out[m * N_FRAMES + t] = v as f32;
            }
        }
        let floor = (max - 8.0) as f32;
        for v in out.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        Ok(())
    }
}

/// Fill `window` (exactly [`WINDOW_SAMPLES`]) with the last 8 s of `parts` (concatenated, oldest
/// first), zero-padded at the start, then normalized to zero mean / unit variance over the whole
/// window (upstream `truncate_audio_to_last_n_seconds` + `do_normalize`). Non-finite samples count
/// as silence.
fn assemble_window(parts: &[Part<'_>], window: &mut [f32]) {
    debug_assert_eq!(window.len(), WINDOW_SAMPLES);
    window.fill(0.0);
    // Walk the parts from the newest backwards, writing from the end of the window.
    let mut end = window.len();
    for part in parts.iter().rev() {
        if end == 0 {
            break;
        }
        match *part {
            Part::Silence(len) => end = end.saturating_sub(len),
            Part::Audio(samples) => {
                let take = samples.len().min(end);
                let src = &samples[samples.len() - take..];
                for (dst, &s) in window[end - take..end].iter_mut().zip(src) {
                    *dst = if s.is_finite() { s } else { 0.0 };
                }
                end -= take;
            }
        }
    }
    let len = window.len() as f64;
    let mean = window.iter().map(|&s| f64::from(s)).sum::<f64>() / len;
    let var = window
        .iter()
        .map(|&s| (f64::from(s) - mean).powi(2))
        .sum::<f64>()
        / len;
    let scale = 1.0 / (var + NORM_EPS).sqrt();
    for s in window.iter_mut() {
        *s = ((f64::from(*s) - mean) * scale) as f32;
    }
}

/// A piece of the turn audio handed to [`assemble_window`].
#[derive(Clone, Copy, Debug)]
enum Part<'a> {
    Audio(&'a [f32]),
    Silence(usize),
}

// ---------------------------------------------------------------------------------------------
// Turn context
// ---------------------------------------------------------------------------------------------

/// Audio of a turn the model judged unfinished, joined to the next utterance of the same turn.
#[derive(Debug, Default)]
struct TurnContext {
    /// The unfinished turn so far (at most one window, oldest first).
    audio: Vec<f32>,
    /// When the last prediction was made.
    last: Option<Instant>,
}

impl TurnContext {
    /// Samples of silence between the stored turn and an utterance of `len` samples that closes
    /// `now`; `None` when the utterance starts a new turn.
    fn gap_before(&self, len: usize, now: Instant) -> Option<usize> {
        let last = self.last?;
        if self.audio.is_empty() {
            return None;
        }
        let elapsed = now.checked_duration_since(last)?;
        let utterance = Duration::from_secs_f64(len as f64 / f64::from(SAMPLE_RATE));
        // The utterance was being captured during most of `elapsed`; the rest was the pause.
        let pause = elapsed.saturating_sub(utterance);
        (pause <= MAX_TURN_GAP).then(|| (pause.as_secs_f64() * f64::from(SAMPLE_RATE)).round() as usize)
    }

    /// Record the verdict for the audio just classified (`gap`/`audio` as passed to the model).
    fn update(&mut self, gap: Option<usize>, audio: &[f32], complete: bool, now: Instant) {
        if complete {
            self.clear();
            return;
        }
        if gap.is_none() {
            self.audio.clear();
        }
        if let Some(g) = gap {
            self.audio
                .extend(std::iter::repeat_n(0.0, g.min(WINDOW_SAMPLES)));
        }
        self.audio.extend_from_slice(audio);
        if self.audio.len() > WINDOW_SAMPLES {
            let drop = self.audio.len() - WINDOW_SAMPLES;
            self.audio.drain(..drop);
        }
        self.last = Some(now);
    }

    fn clear(&mut self) {
        self.audio.clear();
        self.last = None;
    }
}

// ---------------------------------------------------------------------------------------------
// The classifier
// ---------------------------------------------------------------------------------------------

/// Resolved graph names.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TurnIo {
    input: String,
    output: String,
}

impl TurnIo {
    /// One f32 `[N, 80, 800]` input and one f32 output (the probability).
    fn inspect(session: &Session) -> Result<Self, String> {
        let [input] = session.inputs() else {
            return Err(format!("Smart Turn: expected one input, found {}", session.inputs().len()));
        };
        let dims = input.dtype().tensor_shape().map(|s| s.to_vec());
        let dims_ok = dims
            .as_deref()
            .is_some_and(|d| d.len() == 3 && d[0] <= 1 && d[1] == N_MELS as i64 && d[2] == N_FRAMES as i64);
        if input.dtype().tensor_type() != Some(TensorElementType::Float32) || !dims_ok {
            return Err(format!(
                "Smart Turn: input '{}' is {:?} {:?}, expected f32 [1, {N_MELS}, {N_FRAMES}]",
                input.name(),
                input.dtype().tensor_type(),
                dims
            ));
        }
        let output = session
            .outputs()
            .iter()
            .find(|o| o.dtype().tensor_type() == Some(TensorElementType::Float32))
            .ok_or_else(|| "Smart Turn: the model has no f32 output".to_string())?;
        Ok(Self {
            input: input.name().to_string(),
            output: output.name().to_string(),
        })
    }
}

/// A loaded Smart Turn model with the turn context of one pipeline session. Not `Sync`; one per
/// capture thread.
pub struct SmartTurn {
    session: Session,
    io: TurnIo,
    mel: LogMel,
    /// The assembled, normalized 8 s window.
    window: Vec<f32>,
    /// `[1, 80, 800]`.
    features: Tensor<f32>,
    turn: TurnContext,
}

impl SmartTurn {
    /// Load the installed model ([`MODEL_ID`]); never downloads.
    pub fn load() -> Result<Self, String> {
        let dir = model_store::installed_path(MODEL_ID).ok_or_else(|| format!("{MODEL_ID} is not installed"))?;
        Self::load_from(&dir.join(MODEL_FILE))
    }

    /// Load a Smart Turn ONNX file (tests, development).
    pub fn load_from(path: &Path) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("{MODEL_ID}: the model file {MODEL_FILE} is missing"));
        }
        let opts = VcSessionOptions {
            intra_threads: Some(INTRA_THREADS),
            ..VcSessionOptions::default()
        };
        let session = session_builder_for_vc("cpu", &opts)?
            .component(EP_COMPONENT)
            .commit_from_file(path)
            .map_err(|e| format!("{MODEL_ID}: the model failed to load: {e}"))?;
        let io = TurnIo::inspect(&session).map_err(|e| format!("{MODEL_ID}: {e}"))?;
        let features = Tensor::from_array(([1usize, N_MELS, N_FRAMES], vec![0f32; FEATURES])).map_err(|e| format!("{MODEL_ID}: tensor: {e}"))?;
        Ok(Self {
            session,
            io,
            mel: LogMel::new(),
            window: vec![0.0; WINDOW_SAMPLES],
            features,
            turn: TurnContext::default(),
        })
    }

    /// Turn-end probability (0..1) for 16 kHz mono audio that ends now: the utterance the VAD just
    /// closed. The audio of an unfinished previous utterance of the same turn is prepended (see the
    /// module docs). Blocking, ~30 ms (see the module docs).
    pub fn predict(&mut self, audio_16k: &[f32]) -> Result<f32, String> {
        self.predict_at(audio_16k, Instant::now())
    }

    /// Forget the turn context (the next utterance starts a new turn).
    #[allow(dead_code)] // For pipelines that know a turn ended some other way (restart, mute).
    pub fn reset(&mut self) {
        self.turn.clear();
    }

    fn predict_at(&mut self, audio: &[f32], now: Instant) -> Result<f32, String> {
        let gap = self.turn.gap_before(audio.len(), now);
        let probability = {
            let parts: Vec<Part<'_>> = match gap {
                Some(g) => vec![Part::Audio(&self.turn.audio), Part::Silence(g), Part::Audio(audio)],
                None => vec![Part::Audio(audio)],
            };
            assemble_window(&parts, &mut self.window);
            self.classify_window()?
        };
        self.turn.update(gap, audio, probability >= THRESHOLD, now);
        Ok(probability)
    }

    /// Run the model on the assembled window.
    fn classify_window(&mut self) -> Result<f32, String> {
        {
            let (_, buf) = self.features.extract_tensor_mut();
            self.mel.compute(&self.window, buf)?;
        }
        let outputs = self
            .session
            .run(ort::inputs![self.io.input.as_str() => &self.features])
            .map_err(|e| format!("Smart Turn: inference failed: {e}"))?;
        let (_, p) = outputs[self.io.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Smart Turn: output: {e}"))?;
        let p = p
            .first()
            .copied()
            .ok_or_else(|| "Smart Turn: empty output".to_string())?;
        if !p.is_finite() {
            return Err("Smart Turn: the model returned a non-finite probability".into());
        }
        Ok(p.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deterministic test signal of the reference script (`scratchpad/p2c/st_ref.py`): f64
    /// arithmetic, then f32.
    fn synth(n: usize) -> Vec<f32> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let env = 0.5 + 0.5 * (2.0 * PI * 3.0 * t).sin();
                let v =
                    0.3 * (2.0 * PI * 440.0 * t).sin() + 0.2 * env * (2.0 * PI * 1234.5 * t).sin() + 0.05 * (2.0 * PI * 6000.0 * t * t / 3.0).sin();
                v as f32
            })
            .collect()
    }

    fn features_of(audio: &[f32]) -> Vec<f32> {
        let mut window = vec![0.0; WINDOW_SAMPLES];
        assemble_window(&[Part::Audio(audio)], &mut window);
        let mut out = vec![0.0; FEATURES];
        LogMel::new().compute(&window, &mut out).unwrap();
        out
    }

    const SAMPLE_MELS: [usize; 9] = [0, 1, 7, 20, 39, 40, 55, 70, 79];
    const SAMPLE_FRAMES: [usize; 10] = [0, 1, 199, 400, 500, 640, 700, 750, 798, 799];

    /// Reference values from transformers 4.39.3 `WhisperFeatureExtractor(chunk_length=8)` called
    /// exactly like upstream `inference.py` (after `truncate_audio_to_last_n_seconds`), for
    /// `synth(3 s)`: rows are the mels of [`SAMPLE_MELS`], columns the frames of [`SAMPLE_FRAMES`].
    #[rustfmt::skip]
    const REF_SYNTH_3S: [[f32; 10]; 9] = [
        [-0.185765, -0.185765, -0.185765, -0.185765, 1.189932, -0.195209, -0.188433, -0.190985, -0.183576, 0.764454],
        [-0.229058, -0.229058, -0.229058, -0.229058, 1.107739, -0.229058, -0.229058, -0.229058, -0.229058, 0.772229],
        [-0.229058, -0.229058, -0.229058, -0.229058, 1.298945, -0.229058, -0.229058, -0.229058, -0.229058, 0.820322],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.865215, -0.229058, -0.229058, -0.229058, -0.229058, 0.702994],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.818120, -0.229058, -0.229058, -0.229058, -0.229058, 0.193704],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.802725, -0.229058, -0.229058, -0.229058, -0.229058, 0.129619],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.489513, -0.229058, -0.229058, -0.229058, -0.229058, -0.229058],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.284420, 1.103575, -0.229058, -0.229058, -0.229058, -0.229058],
        [-0.229058, -0.229058, -0.229058, -0.229058, 0.225874, -0.229058, 0.699785, -0.229058, -0.229058, -0.229058],
    ];
    const REF_SYNTH_3S_MEAN: f32 = -0.129584;
    const REF_SYNTH_3S_MIN: f32 = -0.229058;
    const REF_SYNTH_3S_MAX: f32 = 1.770942;
    #[rustfmt::skip]
    const REF_SYNTH_10S: [[f32; 10]; 9] = [
        [1.212617, 0.708311, 1.253028, -0.011775, -0.010320, -0.008503, -0.010323, -0.009630, -0.011647, 0.698921],
        [1.216371, 0.705880, 1.140279, -0.285887, -0.284432, -0.282454, -0.284435, -0.283703, -0.285764, 0.709677],
        [1.343075, 0.732735, -0.200276, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, 0.742826],
        [0.909332, 0.474738, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, 0.486282],
        [0.862236, 0.361365, -0.335454, -0.335454, -0.335454, 1.083378, -0.335454, -0.335454, -0.335454, 0.349965],
        [0.846842, 0.340633, -0.335454, -0.335454, -0.335454, 0.656060, -0.335454, -0.335454, -0.335454, 0.330085],
        [0.533632, 0.022995, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, 0.018312],
        [0.328540, -0.181613, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.335454, -0.185817],
        [0.606524, 0.871773, -0.335454, 0.593389, -0.335454, -0.335454, -0.335454, -0.335454, 0.948705, 0.871772],
    ];
    const REF_SYNTH_10S_MEAN: f32 = -0.065383;
    #[rustfmt::skip]
    const REF_SPEECH: [[f32; 10]; 9] = [
        [1.076196, 1.076196, 1.076196, 1.076196, 1.47604, 1.337646, 1.312113, 1.076196, 1.076196, 1.076196],
        [0.802068, 0.802068, 0.802068, 0.802068, 1.564138, 1.449927, 1.419597, 0.802068, 0.802068, 0.802068],
        [-0.166917, -0.166917, -0.166917, -0.166917, 1.543447, 1.421253, 1.217664, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, 1.509652, 0.645523, 0.502052, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, 0.904490, 0.998568, 0.653483, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, 0.910219, 0.926363, 0.524384, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, 0.980485, 0.638195, 0.257103, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, 0.206200, 0.998370, -0.011994, -0.166917, -0.166917, -0.166917],
        [-0.166917, -0.166917, -0.166917, -0.166917, -0.083093, 0.720295, -0.034711, -0.166917, -0.166917, -0.166917],
    ];
    const REF_SPEECH_MEAN: f32 = 0.115403;
    const REF_PROB_SPEECH: f32 = 0.9875;
    const REF_PROB_SPEECH_CUT55: f32 = 0.0717;
    const REF_PROB_SYNTH_3S: f32 = 0.0315;

    fn assert_close(what: &str, got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{what}: got {got}, want {want} (±{tol})");
    }

    #[test]
    fn filterbank_matches_the_slaney_definition() {
        let bands = mel_filterbank();
        assert_eq!(bands.len(), N_MELS);
        // Slaney scale: linear below 1 kHz (200/3 Hz per mel), log above; round trip is exact.
        for hz in [0.0, 440.0, 999.0, 1000.0, 4321.0, 8000.0] {
            assert!((mel_to_hz_slaney(hz_to_mel_slaney(hz)) - hz).abs() < 1e-9, "{hz}");
        }
        assert!((hz_to_mel_slaney(1000.0) - 15.0).abs() < 1e-12);
        // Every band has weight, bands start in ascending order, and the Slaney area normalization
        // makes each triangle's weights sum to about 2 / width-in-bins * (width / 2) = 1 / 25 Hz.
        let mut last_first = 0;
        for (m, (first, w)) in bands.iter().enumerate() {
            assert!(!w.is_empty(), "band {m} is empty");
            assert!(*first >= last_first, "band {m}");
            last_first = *first;
            assert!(w.iter().all(|&x| x > 0.0 && x.is_finite()));
        }
    }

    #[test]
    fn reflect_padding_matches_torch() {
        let n = 5;
        let idx: Vec<usize> = (-3..8).map(|i| reflect_index(i, n)).collect();
        assert_eq!(idx, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn window_is_start_padded_truncated_to_the_tail_and_normalized() {
        let mut w = vec![0.0; WINDOW_SAMPLES];
        // Short audio lands at the end; everything before it is the (normalized) zero padding.
        let audio: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        assemble_window(&[Part::Audio(&audio)], &mut w);
        let mean: f64 = w.iter().map(|&s| f64::from(s)).sum::<f64>() / w.len() as f64;
        let var: f64 = w
            .iter()
            .map(|&s| (f64::from(s) - mean).powi(2))
            .sum::<f64>()
            / w.len() as f64;
        assert!(mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-3, "mean {mean} var {var}");
        let pad = w[0];
        assert!(w[..WINDOW_SAMPLES - 1000].iter().all(|&s| s == pad));
        assert!(w[WINDOW_SAMPLES - 1000] > 1.0 && w[WINDOW_SAMPLES - 999] < -1.0);
        // Long audio keeps only its last 8 s.
        let long: Vec<f32> = (0..WINDOW_SAMPLES + 5)
            .map(|i| if i < 5 { 1.0 } else { 0.0 })
            .collect();
        assemble_window(&[Part::Audio(&long)], &mut w);
        assert!(w.iter().all(|&s| s == 0.0), "the first 5 samples must have been cut");
        // Parts join oldest first with the silence between them; non-finite samples are silence.
        let a = [1.0f32; 3];
        let b = [f32::NAN, -1.0, -1.0];
        assemble_window(&[Part::Audio(&a), Part::Silence(2), Part::Audio(&b)], &mut w);
        let tail = &w[WINDOW_SAMPLES - 8..];
        assert!(tail[0] > 0.0 && tail[1] > 0.0 && tail[2] > 0.0);
        assert_eq!(tail[3], tail[4]);
        assert_eq!(tail[3], tail[5]);
        assert!(tail[6] < 0.0 && tail[7] < 0.0);
        // Silence stays silence (no division by zero).
        assemble_window(&[Part::Audio(&[0.0; 10])], &mut w);
        assert!(w.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn features_match_transformers_whisper_feature_extractor() {
        let f = features_of(&synth(3 * 16_000));
        for (r, &m) in SAMPLE_MELS.iter().enumerate() {
            for (c, &t) in SAMPLE_FRAMES.iter().enumerate() {
                assert_close(&format!("synth3s mel {m} frame {t}"), f[m * N_FRAMES + t], REF_SYNTH_3S[r][c], 2e-3);
            }
        }
        let mean = f.iter().map(|&v| f64::from(v)).sum::<f64>() / f.len() as f64;
        assert_close("synth3s mean", mean as f32, REF_SYNTH_3S_MEAN, 1e-4);
        assert_close("synth3s min", f.iter().copied().fold(f32::MAX, f32::min), REF_SYNTH_3S_MIN, 1e-4);
        assert_close("synth3s max", f.iter().copied().fold(f32::MIN, f32::max), REF_SYNTH_3S_MAX, 1e-4);

        // 10 s of audio: only the last 8 s count.
        let f = features_of(&synth(10 * 16_000));
        for (r, &m) in SAMPLE_MELS.iter().enumerate() {
            for (c, &t) in SAMPLE_FRAMES.iter().enumerate() {
                assert_close(&format!("synth10s mel {m} frame {t}"), f[m * N_FRAMES + t], REF_SYNTH_10S[r][c], 2e-3);
            }
        }
        let mean = f.iter().map(|&v| f64::from(v)).sum::<f64>() / f.len() as f64;
        assert_close("synth10s mean", mean as f32, REF_SYNTH_10S_MEAN, 1e-4);

        // Real speech (the Silero test fixture).
        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let f = features_of(&speech);
        for (r, &m) in SAMPLE_MELS.iter().enumerate() {
            for (c, &t) in SAMPLE_FRAMES.iter().enumerate() {
                assert_close(&format!("speech mel {m} frame {t}"), f[m * N_FRAMES + t], REF_SPEECH[r][c], 2e-3);
            }
        }
        let mean = f.iter().map(|&v| f64::from(v)).sum::<f64>() / f.len() as f64;
        assert_close("speech mean", mean as f32, REF_SPEECH_MEAN, 1e-4);
    }

    #[test]
    fn turn_context_joins_unfinished_utterances_of_one_turn() {
        let t0 = Instant::now();
        let sec = |s: f64| t0 + Duration::from_secs_f64(s);
        let mut turn = TurnContext::default();
        let one_s = vec![0.25f32; 16_000];
        // The first utterance of a session starts a turn.
        assert_eq!(turn.gap_before(one_s.len(), sec(1.0)), None);
        turn.update(None, &one_s, false, sec(1.0));
        assert_eq!(turn.audio.len(), 16_000);
        // The next one (1 s long) closes 2.5 s later: 1.5 s of pause in between.
        let gap = turn.gap_before(one_s.len(), sec(3.5));
        assert_eq!(gap, Some(24_000));
        turn.update(gap, &one_s, false, sec(3.5));
        assert_eq!(turn.audio.len(), 16_000 + 24_000 + 16_000);
        // A pause longer than MAX_TURN_GAP starts a new turn.
        assert_eq!(turn.gap_before(one_s.len(), sec(3.5 + 1.0 + 3.2)), None);
        // Timing jitter never makes the pause negative.
        assert_eq!(turn.gap_before(one_s.len(), sec(4.2)), Some(0));
        // The context is capped at one window.
        for i in 0..10 {
            let now = sec(5.0 + i as f64 * 1.5);
            let g = turn.gap_before(one_s.len(), now);
            turn.update(g, &one_s, false, now);
            assert!(turn.audio.len() <= WINDOW_SAMPLES);
        }
        // A complete verdict ends the turn.
        turn.update(Some(0), &one_s, true, sec(30.0));
        assert!(turn.audio.is_empty() && turn.last.is_none());
        assert_eq!(turn.gap_before(one_s.len(), sec(30.5)), None);
        // A new turn after a long pause drops the old audio.
        turn.update(None, &one_s, false, sec(40.0));
        turn.update(None, &[0.5; 10], false, sec(50.0));
        assert_eq!(turn.audio, vec![0.5; 10]);
    }

    #[test]
    fn a_missing_model_names_the_catalog_id() {
        let e = SmartTurn::load_from(Path::new("C:/definitely/missing/smart-turn.onnx"))
            .err()
            .unwrap();
        assert!(e.contains(MODEL_ID), "{e}");
    }

    fn real_model() -> Option<SmartTurn> {
        let dir = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from)?;
        let path = dir.join("smart-turn").join(MODEL_FILE);
        if !path.is_file() {
            eprintln!("{} not found; skipped", path.display());
            return None;
        }
        Some(SmartTurn::load_from(&path).expect("load Smart Turn"))
    }

    /// Real-model check: `SIGIL_TEST_MODELS_DIR/smart-turn/smart-turn-v3.2-cpu.onnx` (the dev cache
    /// layout). Probabilities match onnxruntime 1.19.2 + transformers on the same input.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with smart-turn/smart-turn-v3.2-cpu.onnx"]
    fn real_model_matches_the_python_reference() {
        let Some(mut st) = real_model() else { return };
        assert_eq!(st.io.input, "input_features");
        assert!(crate::services::diagnostics::ep_outcome(EP_COMPONENT).is_some_and(|o| o.active == "cpu"));
        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let cut = &speech[..(speech.len() as f64 * 0.55) as usize];
        let mut classify = |audio: &[f32]| {
            st.reset();
            assemble_window(&[Part::Audio(audio)], &mut st.window);
            st.classify_window().unwrap()
        };
        let started = Instant::now();
        let complete = classify(&speech);
        let first_call = started.elapsed();
        assert_close("complete sentence", complete, REF_PROB_SPEECH, 0.02);
        assert_close("cut sentence", classify(cut), REF_PROB_SPEECH_CUT55, 0.02);
        assert_close("synthetic tones", classify(&synth(3 * 16_000)), REF_PROB_SYNTH_3S, 0.02);
        assert!(complete >= THRESHOLD && classify(cut) < THRESHOLD);
        // Timing (warm): whole predictions, and the features alone (Rust code: much faster in a
        // release build; the model runs in the prebuilt ONNX Runtime either way).
        const RUNS: u32 = 5;
        let started = Instant::now();
        for _ in 0..RUNS {
            classify(&speech);
        }
        let per_call = started.elapsed() / RUNS;
        let mut buf = vec![0.0; FEATURES];
        let started = Instant::now();
        for _ in 0..RUNS {
            st.mel.compute(&st.window, &mut buf).unwrap();
        }
        let features = started.elapsed() / RUNS;
        eprintln!(
            "Smart Turn: first prediction {first_call:?}; warm prediction {per_call:?} = features {features:?} + model {:?} (features at this build's optimization level)",
            per_call.saturating_sub(features)
        );
    }

    /// The turn context changes the verdict the way upstream's accumulated turn audio does: the
    /// second half of a sentence closed right after its unfinished first half is judged together
    /// with it.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with smart-turn/smart-turn-v3.2-cpu.onnx"]
    fn real_model_joins_a_paused_sentence() {
        let Some(mut st) = real_model() else { return };
        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let split = (speech.len() as f64 * 0.55) as usize;
        let (first, second) = speech.split_at(split);
        let t0 = Instant::now();
        let p1 = st.predict_at(first, t0).unwrap();
        assert!(p1 < THRESHOLD, "an unfinished sentence: {p1}");
        assert!(!st.turn.audio.is_empty());
        // The rest follows after a 0.4 s pause.
        let second_len = Duration::from_secs_f64(second.len() as f64 / 16_000.0);
        let p2 = st
            .predict_at(second, t0 + second_len + Duration::from_millis(400))
            .unwrap();
        assert!(p2 >= THRESHOLD, "the finished sentence: {p2}");
        assert!(st.turn.audio.is_empty(), "a complete verdict ends the turn");
    }
}
