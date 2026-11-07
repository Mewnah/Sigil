//! FireRedVAD music / singing gate for the speech segmenter (P6-L; research §1.2, an A/B
//! experiment, off by default).
//!
//! When `PipelineConfig::music_gate` (TS `musicGate` of the `local` STT settings) is on, the
//! capture thread classifies every closed segment and drops music and singing before any engine
//! sees it (`speech:utterance` kind `dropped`, reason `music`; not counted as lost speech), so
//! VRChat world music and karaoke stop turning into captions. Speech over background music is
//! kept (see [`decide`]).
//!
//! # Model
//! FireRedVAD's AED (audio event detection) model from FireRedTeam/FireRedVAD (Apache-2.0): a
//! DFSMN with about 588 k parameters and three sigmoid outputs per 10 ms frame, `speech`,
//! `singing` and `music` (multi-label: vocals in a song are also "speech"). The official ONNX export
//! `pretrained_models/onnx_models/fireredvad_aed.onnx` and its Kaldi `cmvn.ark` are installed by the
//! model store as [`MODEL_ID`] (`model-catalog/firered.json`; only ever installed from the Models
//! tab or the STT panel's requirement card, never downloaded here). It runs on Sigil's ONNX Runtime
//! through `neural_audio::ort_session` (CPU, one intra-op thread, spinning off, CPU arena for the
//! dynamic length; EP outcome recorded as [`EP_COMPONENT`]).
//!
//! # Input contract (upstream `fireredvad/core/audio_feat.py`, checked against torchaudio below)
//! 1. 16 kHz mono audio at int16 scale (`x * 32768`).
//! 2. Kaldi fbank (`kaldi_native_fbank` defaults): 25 ms Povey window, 10 ms shift, `snip_edges`,
//!    DC removal, pre-emphasis 0.97, a 512-point FFT power spectrum, 80 mel bins over 20 Hz–8 kHz,
//!    natural log floored at `f32::EPSILON`, no dither.
//! 3. Global CMVN from `cmvn.ark` (Kaldi binary double matrix of accumulated stats).
//! 4. `feat` f32 `[1, T, 80]` → `probs` f32 `[1, T, 3]` (already sigmoid probabilities).
//!
//! # Verdict
//! [`decide`] smooths each class with upstream's 5-frame window and thresholds (speech 0.4,
//! singing 0.5, music 0.5), then:
//! - **singing** when at least half the frames sing and singing is not clearly behind speech
//!   (the AED also hears sung vocals as speech, so "speech present" cannot keep a song);
//! - **music** when at least half the frames are music and under a fifth are speech
//!   (instrumental music; speech over background music stays speech);
//! - **speech** otherwise, and for segments too short to judge or when inference fails (the gate
//!   fails open: it never drops speech because of an error).
//!
//! Calibration (onnxruntime 1.19.2 + torchaudio on the FireRedVAD sample clips, fractions of
//! frames over threshold as `speech / singing / music`): `hello_en.wav` 0.58 / 0 / 0 → speech;
//! the same clip over synthetic music at −20 / −10 / −5 dB 0.60 / 0.06 / 1.0, 0.55 / 0 / 1.0,
//! 0.34 / 0 / 1.0 → speech; at 0 dB 0 / 0 / 1.0 → music (the music is as loud as the voice);
//! music alone 0 / 0 / 1.0 → music; 3 s windows of the `event.wav` song 0.17–0.98 / 0.99–1.0 /
//! 1.0 → singing (its 3 s intro, 0.81 / 0.39 / 0.96, stays speech). Live VRChat world music is the
//! A/B that decides whether the gate graduates (P7).
//!
//! # Runtime
//! About 1.2 MFLOP per frame. [`MusicGate::classify`] runs on the capture thread when a segment
//! closes (not an audio callback), so it delays that segment's hand-off to its engine by the
//! inference time (the hub ring absorbs it); `real_model_matches_the_python_reference` prints the
//! measured cost per second of audio.
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE)
//! - [`MusicGate::load`]`() -> Result<MusicGate, String>`: on the speech capture thread when a
//!   session starts with the gate on; `Err` leaves the gate off with a warning status naming
//!   [`MODEL_ID`].
//! - [`MusicGate::classify`]`(&mut self, audio_16k) -> MusicVerdict`: one closed segment, 16 kHz
//!   mono; runs on the capture thread (not an audio callback).

use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::{Tensor, TensorElementType};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::services::model_store;
use crate::services::neural_audio::ort_session::{session_builder_for_vc, VcSessionOptions};

/// Catalog id of the FireRedVAD AED model (`model-catalog/firered.json`).
pub const MODEL_ID: &str = "vad/firered-aed";
/// The ONNX model inside the installed entry folder.
pub const MODEL_FILE: &str = "fireredvad_aed.onnx";
/// Kaldi global CMVN statistics inside the installed entry folder.
pub const CMVN_FILE: &str = "cmvn.ark";
/// Diagnostics component of the ORT session (Settings → GPU & diagnostics).
pub const EP_COMPONENT: &str = "stt.music";
/// Intra-op threads: the gate is small and must not compete with the engines.
const INTRA_THREADS: usize = 1;

const LOG: &str = "sigil::speech::music_gate";

/// Classes of the model output, in order.
const N_CLASSES: usize = 3;
const SPEECH: usize = 0;
const SINGING: usize = 1;
const MUSIC: usize = 2;

/// Upstream per-class thresholds (`FireRedAedConfig`).
const SPEECH_THRESHOLD: f32 = 0.4;
const SINGING_THRESHOLD: f32 = 0.5;
const MUSIC_THRESHOLD: f32 = 0.5;
/// Upstream smoothing window (frames, centred moving average).
const SMOOTH_FRAMES: usize = 5;
/// Shorter segments (frames) are never judged: too little context to call music.
const MIN_FRAMES: usize = 20;
/// Singing: at least this fraction of the frames sing ...
const SINGING_MIN_FRACTION: f32 = 0.5;
/// ... and the singing fraction is at most this far below the speech fraction.
const SINGING_SPEECH_MARGIN: f32 = 0.1;
/// Music: at least this fraction of the frames are music ...
const MUSIC_MIN_FRACTION: f32 = 0.5;
/// ... and fewer than this fraction are speech.
const MUSIC_MAX_SPEECH_FRACTION: f32 = 0.2;

// Kaldi fbank (kaldi_native_fbank defaults used by FireRedVAD).
const SAMPLE_RATE: f64 = 16_000.0;
const FRAME_LEN: usize = 400;
const FRAME_SHIFT: usize = 160;
const N_FFT: usize = 512;
const N_MELS: usize = 80;
const LOW_FREQ: f64 = 20.0;
const PREEMPH: f32 = 0.97;
/// int16 scale of the training features.
const PCM_SCALE: f32 = 32_768.0;

/// What a closed segment sounds like.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MusicVerdict {
    #[default]
    Speech,
    Music,
    Singing,
}

impl MusicVerdict {
    /// Only speech goes on to the engines.
    pub fn is_speech(self) -> bool {
        matches!(self, MusicVerdict::Speech)
    }
}

// ---------------------------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------------------------

/// Fractions of frames over each class threshold after smoothing (`[speech, singing, music]`).
fn active_fractions(probs: &[[f32; N_CLASSES]]) -> [f32; N_CLASSES] {
    let n = probs.len();
    if n == 0 {
        return [0.0; N_CLASSES];
    }
    let thresholds = [SPEECH_THRESHOLD, SINGING_THRESHOLD, MUSIC_THRESHOLD];
    let half = SMOOTH_FRAMES / 2;
    let mut counts = [0usize; N_CLASSES];
    for t in 0..n {
        let (lo, hi) = (t.saturating_sub(half), (t + half + 1).min(n));
        for (class, count) in counts.iter_mut().enumerate() {
            let mean = probs[lo..hi].iter().map(|p| p[class]).sum::<f32>() / (hi - lo) as f32;
            if mean >= thresholds[class] {
                *count += 1;
            }
        }
    }
    counts.map(|c| c as f32 / n as f32)
}

/// The verdict for one segment's per-frame probabilities (see the module docs).
pub(crate) fn decide(probs: &[[f32; N_CLASSES]]) -> MusicVerdict {
    if probs.len() < MIN_FRAMES {
        return MusicVerdict::Speech;
    }
    let [speech, singing, music] = active_fractions(probs);
    if singing >= SINGING_MIN_FRACTION && singing + SINGING_SPEECH_MARGIN >= speech {
        MusicVerdict::Singing
    } else if music >= MUSIC_MIN_FRACTION && speech < MUSIC_MAX_SPEECH_FRACTION {
        MusicVerdict::Music
    } else {
        MusicVerdict::Speech
    }
}

// ---------------------------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------------------------

/// Kaldi's mel scale.
fn mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Kaldi `MelBanks` for a 512-point FFT at 16 kHz, 80 bins over 20 Hz–8 kHz, as `(first FFT bin,
/// weights)` per bin. Like Kaldi, only FFT bins `0..N_FFT / 2` contribute (never the Nyquist bin).
fn mel_banks() -> Vec<(usize, Vec<f32>)> {
    let fft_bins = N_FFT / 2;
    let bin_width = SAMPLE_RATE / N_FFT as f64;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(SAMPLE_RATE / 2.0));
    let delta = (mel_high - mel_low) / (N_MELS + 1) as f64;
    (0..N_MELS)
        .map(|b| {
            let left = mel_low + b as f64 * delta;
            let centre = left + delta;
            let right = centre + delta;
            let weights: Vec<(usize, f32)> = (0..fft_bins)
                .filter_map(|i| {
                    let m = mel(bin_width * i as f64);
                    if m > left && m < right {
                        let w = if m <= centre { (m - left) / (centre - left) } else { (right - m) / (right - centre) };
                        Some((i, w as f32))
                    } else {
                        None
                    }
                })
                .collect();
            let first = weights.first().map_or(0, |&(i, _)| i);
            (first, weights.into_iter().map(|(_, w)| w).collect())
        })
        .collect()
}

/// Kaldi log-mel filterbank (reused buffers).
struct Fbank {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    banks: Vec<(usize, Vec<f32>)>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl Fbank {
    fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        // Kaldi's "povey" window: a Hann window (over N - 1) to the power 0.85.
        let window = (0..FRAME_LEN)
            .map(|i| {
                let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LEN - 1) as f64).cos();
                hann.powf(0.85) as f32
            })
            .collect();
        Self {
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            banks: mel_banks(),
            power: vec![0.0; N_FFT / 2 + 1],
        }
    }

    /// Frames of `n` samples (`snip_edges`).
    fn frames(n: usize) -> usize {
        if n < FRAME_LEN {
            0
        } else {
            1 + (n - FRAME_LEN) / FRAME_SHIFT
        }
    }

    /// Log-mel features of `audio` (16 kHz, [-1, 1]) appended to `out` as `[frames, 80]` row-major.
    /// Returns the frame count.
    fn compute(&mut self, audio: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        let frames = Self::frames(audio.len());
        out.reserve(frames * N_MELS);
        for t in 0..frames {
            let start = t * FRAME_SHIFT;
            let src = &audio[start..start + FRAME_LEN];
            let (head, tail) = self.frame.split_at_mut(FRAME_LEN);
            let mut sum = 0.0f64;
            for (dst, &x) in head.iter_mut().zip(src) {
                // Non-finite input would poison the whole segment; treat it as silence.
                let v = if x.is_finite() { x * PCM_SCALE } else { 0.0 };
                *dst = v;
                sum += f64::from(v);
            }
            tail.fill(0.0);
            let mean = (sum / FRAME_LEN as f64) as f32;
            for v in head.iter_mut() {
                *v -= mean;
            }
            for i in (1..FRAME_LEN).rev() {
                head[i] -= PREEMPH * head[i - 1];
            }
            head[0] -= PREEMPH * head[0];
            for (v, w) in head.iter_mut().zip(&self.window) {
                *v *= w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .map_err(|e| format!("music gate features: FFT: {e}"))?;
            for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
                *p = c.re * c.re + c.im * c.im;
            }
            for (first, weights) in &self.banks {
                let energy: f32 = weights.iter().zip(&self.power[*first..]).map(|(w, p)| w * p).sum();
                out.push(energy.max(f32::EPSILON).ln());
            }
        }
        Ok(frames)
    }
}

/// Global CMVN (`(x - mean) * inverse_std`) from Kaldi accumulated statistics.
#[derive(Clone, Debug, PartialEq)]
struct Cmvn {
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
}

impl Cmvn {
    /// Parse a Kaldi binary matrix of CMVN stats (`\0B` then `DM ` or `FM `, two rows of `dim + 1`
    /// columns: sums with the frame count last, then sums of squares), as upstream `CMVN` does.
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let rest = bytes.strip_prefix(b"\0B").ok_or("not a Kaldi binary matrix")?;
        let (double, rest) = if let Some(r) = rest.strip_prefix(b"DM ") {
            (true, r)
        } else if let Some(r) = rest.strip_prefix(b"FM ") {
            (false, r)
        } else {
            return Err("not a Kaldi float or double matrix".into());
        };
        let read_int = |b: &[u8]| -> Result<(usize, usize), String> {
            match b {
                [4, a, b, c, d, ..] => {
                    let v = i32::from_le_bytes([*a, *b, *c, *d]);
                    usize::try_from(v).map(|v| (v, 5)).map_err(|_| format!("negative matrix size {v}"))
                }
                _ => Err("truncated matrix header".into()),
            }
        };
        let (rows, used) = read_int(rest)?;
        let rest = &rest[used..];
        let (cols, used) = read_int(rest)?;
        let data = &rest[used..];
        if rows != 2 || !(2..=1024).contains(&cols) {
            return Err(format!("unexpected CMVN stats shape {rows}x{cols}"));
        }
        let width = if double { 8 } else { 4 };
        if data.len() < rows * cols * width {
            return Err("truncated CMVN stats".into());
        }
        let value = |i: usize| -> f64 {
            let b = &data[i * width..(i + 1) * width];
            if double {
                f64::from_le_bytes(b.try_into().expect("8 bytes"))
            } else {
                f64::from(f32::from_le_bytes(b.try_into().expect("4 bytes")))
            }
        };
        let dim = cols - 1;
        let count = value(dim);
        if count.is_nan() || count < 1.0 {
            return Err(format!("CMVN frame count {count} is below 1"));
        }
        let mut mean = Vec::with_capacity(dim);
        let mut inverse_std = Vec::with_capacity(dim);
        for d in 0..dim {
            let m = value(d) / count;
            let var = (value(cols + d) / count - m * m).max(1e-20);
            mean.push(m as f32);
            inverse_std.push((1.0 / var.sqrt()) as f32);
        }
        if mean.iter().chain(&inverse_std).any(|v| !v.is_finite()) {
            return Err("CMVN stats are not finite".into());
        }
        Ok(Self { mean, inverse_std })
    }

    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn apply(&self, features: &mut [f32]) {
        for row in features.chunks_exact_mut(self.dim()) {
            for ((v, m), s) in row.iter_mut().zip(&self.mean).zip(&self.inverse_std) {
                *v = (*v - m) * s;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------------------------

/// Input / output names of the session.
#[derive(Debug)]
struct GateIo {
    input: String,
    output: String,
}

impl GateIo {
    /// One f32 `[N, T, 80]` input and one f32 output.
    fn inspect(session: &Session) -> Result<Self, String> {
        let [input] = session.inputs() else {
            return Err(format!("expected one model input, found {}", session.inputs().len()));
        };
        let dims = input.dtype().tensor_shape().map(|s| s.to_vec());
        let dims_ok = dims
            .as_deref()
            .is_some_and(|d| d.len() == 3 && d[0] <= 1 && d[2] == N_MELS as i64);
        if input.dtype().tensor_type() != Some(TensorElementType::Float32) || !dims_ok {
            return Err(format!(
                "input '{}' is {:?} {:?}, expected f32 [1, T, {N_MELS}]: not the FireRedVAD AED model",
                input.name(),
                input.dtype().tensor_type(),
                dims
            ));
        }
        let output = session
            .outputs()
            .iter()
            .find(|o| o.dtype().tensor_type() == Some(TensorElementType::Float32))
            .ok_or_else(|| "the model has no f32 output".to_string())?;
        Ok(Self {
            input: input.name().to_string(),
            output: output.name().to_string(),
        })
    }
}

/// A loaded gate (one per speech session).
pub struct MusicGate {
    session: Session,
    io: GateIo,
    fbank: Fbank,
    cmvn: Cmvn,
    features: Vec<f32>,
}

impl std::fmt::Debug for MusicGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicGate").field("io", &self.io).finish_non_exhaustive()
    }
}

impl MusicGate {
    /// Load the installed model ([`MODEL_ID`]); never downloads.
    pub fn load() -> Result<Self, String> {
        let dir = model_store::installed_path(MODEL_ID).ok_or_else(|| format!("{MODEL_ID} is not installed"))?;
        Self::load_from(&dir)
    }

    /// Load the model and CMVN stats from a folder holding [`MODEL_FILE`] and [`CMVN_FILE`].
    pub fn load_from(dir: &Path) -> Result<Self, String> {
        let model = dir.join(MODEL_FILE);
        let cmvn_path = dir.join(CMVN_FILE);
        if !model.is_file() || !cmvn_path.is_file() {
            return Err(format!("{MODEL_ID}: {MODEL_FILE} or {CMVN_FILE} is missing in {}", dir.display()));
        }
        let cmvn = std::fs::read(&cmvn_path)
            .map_err(|e| format!("{MODEL_ID}: cannot read {CMVN_FILE}: {e}"))
            .and_then(|b| Cmvn::parse(&b).map_err(|e| format!("{MODEL_ID}: {CMVN_FILE}: {e}")))?;
        if cmvn.dim() != N_MELS {
            return Err(format!("{MODEL_ID}: {CMVN_FILE} has {} dimensions, expected {N_MELS}", cmvn.dim()));
        }
        let opts = VcSessionOptions {
            intra_threads: Some(INTRA_THREADS),
            cpu_arena: true,
            ..VcSessionOptions::default()
        };
        let session = session_builder_for_vc("cpu", &opts)?
            .component(EP_COMPONENT)
            .commit_from_file(&model)
            .map_err(|e| format!("{MODEL_ID}: the model failed to load: {e}"))?;
        let io = GateIo::inspect(&session).map_err(|e| format!("{MODEL_ID}: {e}"))?;
        Ok(Self {
            session,
            io,
            fbank: Fbank::new(),
            cmvn,
            features: Vec::new(),
        })
    }

    /// Classify one closed segment (16 kHz mono). Fails open: an inference error is logged and the
    /// segment counts as speech.
    pub fn classify(&mut self, audio_16k: &[f32]) -> MusicVerdict {
        if Fbank::frames(audio_16k.len()) < MIN_FRAMES {
            return MusicVerdict::Speech;
        }
        match self.probabilities(audio_16k) {
            Ok(probs) => decide(&probs),
            Err(e) => {
                tracing::warn!(target: LOG, "music gate: {e}; the segment is kept as speech");
                MusicVerdict::Speech
            }
        }
    }

    /// Per-frame `[speech, singing, music]` probabilities of `audio_16k`.
    pub(crate) fn probabilities(&mut self, audio_16k: &[f32]) -> Result<Vec<[f32; N_CLASSES]>, String> {
        self.features.clear();
        let frames = self.fbank.compute(audio_16k, &mut self.features)?;
        if frames == 0 {
            return Ok(Vec::new());
        }
        self.cmvn.apply(&mut self.features);
        let input = Tensor::from_array(([1usize, frames, N_MELS], std::mem::take(&mut self.features)))
            .map_err(|e| format!("input tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.io.input.as_str() => input])
            .map_err(|e| format!("inference failed: {e}"))?;
        let (shape, values) = outputs[self.io.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output: {e}"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[1] != frames as i64 || dims[2] != N_CLASSES as i64 || values.len() != frames * N_CLASSES {
            return Err(format!("unexpected output shape {dims:?} for {frames} frames"));
        }
        let probs = values
            .as_chunks::<N_CLASSES>().0.iter()
            .map(|c| [c[SPEECH], c[SINGING], c[MUSIC]].map(|p| if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 }))
            .collect();
        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deterministic test signal of `speech::smart_turn`'s tests (f64 arithmetic, then f32).
    fn synth(n: usize) -> Vec<f32> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let env = 0.5 + 0.5 * (2.0 * PI * 3.0 * t).sin();
                let v = 0.3 * (2.0 * PI * 440.0 * t).sin()
                    + 0.2 * env * (2.0 * PI * 1234.5 * t).sin()
                    + 0.05 * (2.0 * PI * 6000.0 * t * t / 3.0).sin();
                v as f32
            })
            .collect()
    }

    const REF_FRAMES: [usize; 6] = [0, 1, 17, 48, 96, 97];
    const REF_BINS: [usize; 6] = [0, 1, 9, 30, 55, 79];
    /// `torchaudio.compliance.kaldi.fbank(synth(1 s) * 32768, num_mel_bins=80, frame_length=25,
    /// frame_shift=10, dither=0, energy_floor=0)` (torchaudio 2.3.1) at [`REF_FRAMES`] x [`REF_BINS`].
    #[rustfmt::skip]
    const REF_FBANK: [[f32; 6]; 6] = [
        [16.53999, 17.20587, 12.06947, 22.75145, 1.76659, -2.86692],
        [15.28421, 16.87425, 12.18601, 23.01887, 1.56212, -6.43387],
        [9.28659, 10.20674, 12.11261, 21.65477, 0.68263, -5.50137],
        [9.43696, 9.72006, 12.08933, 22.6004, 2.62914, -3.32593],
        [8.84232, 9.77126, 12.09027, 20.99098, 8.98545, -0.99496],
        [9.13148, 10.01249, 12.06737, 21.56451, 8.57372, -0.28246],
    ];
    const REF_FBANK_MEAN: f32 = 8.66609;

    #[test]
    fn fbank_matches_kaldi() {
        let audio = synth(16_000);
        let mut out = Vec::new();
        let frames = Fbank::new().compute(&audio, &mut out).unwrap();
        assert_eq!(frames, 98);
        assert_eq!(out.len(), frames * N_MELS);
        for (i, &f) in REF_FRAMES.iter().enumerate() {
            for (j, &b) in REF_BINS.iter().enumerate() {
                let got = out[f * N_MELS + b];
                let want = REF_FBANK[i][j];
                // Bins more than ~15 nats below the frame's loudest one sit near the f32 FFT
                // round-off floor of both implementations (pocketfft vs rustfft): they agree only
                // roughly, and after CMVN they are far below anything the model distinguishes.
                let tol = if want < 5.0 { 0.5 } else { 2e-3 };
                assert!((got - want).abs() <= tol, "frame {f} bin {b}: {got} vs {want}");
            }
        }
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - REF_FBANK_MEAN).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn fbank_frame_count_follows_snip_edges() {
        assert_eq!(Fbank::frames(0), 0);
        assert_eq!(Fbank::frames(399), 0);
        assert_eq!(Fbank::frames(400), 1);
        assert_eq!(Fbank::frames(559), 1);
        assert_eq!(Fbank::frames(560), 2);
        let mut out = Vec::new();
        assert_eq!(Fbank::new().compute(&[0.0; 300], &mut out).unwrap(), 0);
        assert!(out.is_empty());
        // Silence floors at log(f32::EPSILON) instead of -inf; NaN input is treated as silence.
        let frames = Fbank::new().compute(&[f32::NAN; 400], &mut out).unwrap();
        assert_eq!(frames, 1);
        assert!(out.iter().all(|v| (*v - f32::EPSILON.ln()).abs() < 1e-6), "{out:?}");
    }

    /// A Kaldi binary CMVN matrix (`\0B`, `DM `, rows, cols, doubles).
    fn kaldi_matrix(double: bool, rows: i32, cols: i32, values: &[f64]) -> Vec<u8> {
        let mut b = b"\0B".to_vec();
        b.extend_from_slice(if double { b"DM " } else { b"FM " });
        b.push(4);
        b.extend_from_slice(&rows.to_le_bytes());
        b.push(4);
        b.extend_from_slice(&cols.to_le_bytes());
        for v in values {
            if double {
                b.extend_from_slice(&v.to_le_bytes());
            } else {
                b.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn cmvn_stats_parse_like_upstream() {
        // dim 2: sums (4, 8, count 2), sums of squares (10, 40, 0).
        for double in [true, false] {
            let bytes = kaldi_matrix(double, 2, 3, &[4.0, 8.0, 2.0, 10.0, 40.0, 0.0]);
            let cmvn = Cmvn::parse(&bytes).unwrap();
            assert_eq!(cmvn.mean, vec![2.0, 4.0]);
            // var = 10/2 - 4 = 1, 40/2 - 16 = 4.
            assert_eq!(cmvn.inverse_std, vec![1.0, 0.5]);
            let mut x = vec![3.0, 6.0, 2.0, 4.0];
            cmvn.apply(&mut x);
            assert_eq!(x, vec![1.0, 1.0, 0.0, 0.0]);
        }
        // A zero variance is floored, not a division by zero.
        let flat = Cmvn::parse(&kaldi_matrix(true, 2, 2, &[2.0, 2.0, 2.0, 0.0])).unwrap();
        assert!(flat.inverse_std[0].is_finite() && flat.inverse_std[0] > 1e9);
        for bad in [
            b"not kaldi".to_vec(),
            kaldi_matrix(true, 3, 3, &[0.0; 9]),
            kaldi_matrix(true, 2, 3, &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            kaldi_matrix(true, 2, 3, &[1.0, 1.0]),
            b"\0BDM \x04".to_vec(),
        ] {
            assert!(Cmvn::parse(&bad).is_err(), "{bad:?}");
        }
    }

    fn track(n: usize, f: impl Fn(usize) -> [f32; 3]) -> Vec<[f32; 3]> {
        (0..n).map(f).collect()
    }

    #[test]
    fn speech_with_background_music_is_kept() {
        // Speech over music: the AED hears both.
        let probs = track(300, |t| [if t % 100 < 70 { 0.9 } else { 0.1 }, 0.05, 0.99]);
        assert_eq!(decide(&probs), MusicVerdict::Speech);
        // Plain speech.
        assert_eq!(decide(&track(300, |_| [0.95, 0.0, 0.0])), MusicVerdict::Speech);
        // A little speech (a fifth of the frames) is enough to keep music-heavy audio.
        let probs = track(300, |t| [if t < 75 { 0.9 } else { 0.0 }, 0.0, 1.0]);
        assert_eq!(decide(&probs), MusicVerdict::Speech);
    }

    #[test]
    fn instrumental_music_and_singing_are_dropped() {
        assert_eq!(decide(&track(300, |_| [0.02, 0.02, 0.95])), MusicVerdict::Music);
        // A few stray speech frames do not keep instrumental music.
        let probs = track(300, |t| [if t < 30 { 0.9 } else { 0.05 }, 0.02, 0.95]);
        assert_eq!(decide(&probs), MusicVerdict::Music);
        // Sung vocals: the AED also calls them speech; singing wins.
        assert_eq!(decide(&track(300, |_| [0.9, 0.95, 1.0])), MusicVerdict::Singing);
        // A cappella singing.
        assert_eq!(decide(&track(300, |_| [0.3, 0.9, 0.2])), MusicVerdict::Singing);
    }

    #[test]
    fn singing_clearly_behind_speech_is_kept() {
        // Speech in every frame, singing in just over half: someone talks over a song.
        let probs = track(300, |t| [0.95, if t < 160 { 0.9 } else { 0.1 }, 1.0]);
        assert_eq!(decide(&probs), MusicVerdict::Speech);
    }

    #[test]
    fn short_or_empty_segments_are_speech() {
        assert_eq!(decide(&[]), MusicVerdict::Speech);
        assert_eq!(decide(&track(MIN_FRAMES - 1, |_| [0.0, 0.0, 1.0])), MusicVerdict::Speech);
        assert_eq!(decide(&track(MIN_FRAMES, |_| [0.0, 0.0, 1.0])), MusicVerdict::Music);
    }

    #[test]
    fn smoothing_ignores_single_frame_spikes() {
        // One-frame music spikes every 10 frames never reach the threshold after smoothing.
        let probs = track(300, |t| [0.0, 0.0, if t % 10 == 0 { 1.0 } else { 0.0 }]);
        assert_eq!(active_fractions(&probs)[MUSIC], 0.0);
        assert_eq!(decide(&probs), MusicVerdict::Speech);
        // Edges average over the frames that exist (no zero padding bias).
        let probs = track(10, |_| [0.0, 0.0, 0.6]);
        assert_eq!(active_fractions(&probs), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn verdicts_only_pass_speech() {
        assert!(MusicVerdict::Speech.is_speech());
        assert!(!MusicVerdict::Music.is_speech());
        assert!(!MusicVerdict::Singing.is_speech());
        assert_eq!(MusicVerdict::default(), MusicVerdict::Speech);
    }

    #[test]
    fn a_missing_model_is_an_error_naming_the_catalog_id() {
        // `load()` reads this PC's real model store, so its "not installed" message can only be checked
        // where the model is not installed (a PC that fetched it passes through to the hermetic half).
        if model_store::installed_path(MODEL_ID).is_none() {
            let e = MusicGate::load().unwrap_err();
            assert!(e.contains(MODEL_ID), "{e}");
        }
        let dir = std::env::temp_dir().join(format!("sigil-music-gate-missing-{}", std::process::id()));
        let e = MusicGate::load_from(&dir).unwrap_err();
        assert!(e.contains(MODEL_ID) && e.contains(MODEL_FILE), "{e}");
    }

    /// The model folder for the real-model tests: `$SIGIL_TEST_MODELS_DIR/firered-aed` holding
    /// `fireredvad_aed.onnx` and `cmvn.ark` (FireRedVAD `pretrained_models/onnx_models` at commit
    /// c30ec49e8cc69642b0ee65362eba11b9d11c6e54).
    fn real_gate() -> Option<MusicGate> {
        let dir = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from)?.join("firered-aed");
        if !dir.join(MODEL_FILE).is_file() {
            eprintln!("{} not found; skipped", dir.display());
            return None;
        }
        Some(MusicGate::load_from(&dir).expect("load the FireRedVAD AED model"))
    }

    fn mean_probs(probs: &[[f32; 3]]) -> [f32; 3] {
        let mut m = [0.0f32; 3];
        for p in probs {
            for c in 0..3 {
                m[c] += p[c] / probs.len() as f32;
            }
        }
        m
    }

    /// Real model: probabilities match onnxruntime 1.19.2 on the same input (features from
    /// `torchaudio.compliance.kaldi.fbank` as in [`REF_FBANK`], then the CMVN of `cmvn.ark`), speech
    /// passes and the synthetic tones count as music.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with firered-aed/{fireredvad_aed.onnx,cmvn.ark}"]
    fn real_model_matches_the_python_reference() {
        let Some(mut gate) = real_gate() else { return };
        assert_eq!((gate.io.input.as_str(), gate.io.output.as_str()), ("feat", "probs"));
        assert!(crate::services::diagnostics::ep_outcome(EP_COMPONENT).is_some_and(|o| o.active == "cpu"));

        let tones = synth(16_000);
        let probs = gate.probabilities(&tones).unwrap();
        assert_eq!(probs.len(), 98);
        let close = |name: &str, got: [f32; 3], want: [f32; 3], tol: f32| {
            for c in 0..3 {
                assert!((got[c] - want[c]).abs() <= tol, "{name} class {c}: {got:?} vs {want:?}");
            }
        };
        close("tones frame 50", probs[50], [0.015_515_6, 0.019_705_1, 0.949_934_4], 0.01);
        close("tones mean", mean_probs(&probs), [0.017_764_3, 0.027_636_4, 0.900_846_4], 0.01);
        assert_eq!(gate.classify(&tones), MusicVerdict::Music);

        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let probs = gate.probabilities(&speech).unwrap();
        assert_eq!(probs.len(), 332);
        close("speech mean", mean_probs(&probs), [0.623_482_4, 0.000_084_4, 0.000_272_5], 0.01);
        assert_eq!(gate.classify(&speech), MusicVerdict::Speech);

        // Cost per second of audio (warm).
        const RUNS: u32 = 5;
        let started = std::time::Instant::now();
        for _ in 0..RUNS {
            gate.classify(&speech);
        }
        let per_call = started.elapsed() / RUNS;
        let audio_s = speech.len() as f64 / 16_000.0;
        eprintln!(
            "music gate: {:.1} ms per {audio_s:.2} s segment ({:.1} ms per second of audio)",
            per_call.as_secs_f64() * 1e3,
            per_call.as_secs_f64() * 1e3 / audio_s
        );
    }

    /// Real model on FireRedVAD's own sample clips when present next to the model: `event.wav` (a
    /// song) is singing in every 3 s window after its intro, `hello_en.wav` is speech.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with firered-aed/{fireredvad_aed.onnx,cmvn.ark,event.wav,hello_en.wav}"]
    fn real_model_drops_a_song_and_keeps_speech() {
        let Some(mut gate) = real_gate() else { return };
        let dir = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from).unwrap().join("firered-aed");
        let read = |name: &str| -> Option<Vec<f32>> {
            let reader = hound::WavReader::open(dir.join(name)).ok()?;
            assert_eq!(reader.spec().sample_rate, 16_000);
            Some(reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect())
        };
        let (Some(song), Some(hello)) = (read("event.wav"), read("hello_en.wav")) else {
            eprintln!("sample clips not found; skipped");
            return;
        };
        assert_eq!(gate.classify(&hello), MusicVerdict::Speech);
        for start in (3..21).step_by(3) {
            let window = &song[start * 16_000..(start + 3) * 16_000];
            assert_eq!(gate.classify(window), MusicVerdict::Singing, "song window at {start} s");
        }
    }
}
