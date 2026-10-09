//! SwiftF0 pitch extractor (P3-D). The CPU default candidate after a cents A/B against RMVPE (MIT,
//! ~135 KB ONNX, about 160x real time on one CPU core); its contour is resampled onto the hop-160
//! grid.
//!
//! Model: model-store entry `vc/pitch-swiftf0` (`assets::model_file`), else an explicit path (the
//! pitch bank factory in `f0_fusion` resolves it).
//!
//! # Graph contract (lars76/swift-f0 `model.onnx`, checked by `scripts/vc-templates/export_pitch.py`)
//! - inputs: `audio` f32 `[1, N]` (16 kHz mono, raw samples), `fmin` / `fmax` f32 scalars (Hz; the
//!   model covers 46.875-2093.75 Hz);
//! - outputs: `pitch` f64 (or f32) `[1, F]` Hz and `confidence` f32 `[1, F]`, `F = N / 256`.
//! - Frame `i` is centred on sample `256 * i` (measured with pitch-step sweeps on the real model:
//!   mean offset -9 samples over 20 steps), and the network sees about ±10 frames (±2560 samples;
//!   beyond ±3 frames the effect is below 1 cent).
//!
//! # What this extractor adds
//! - **Grid resampling.** Frame `j` of the returned contour is sample `160 * j`; its F0 is
//!   interpolated in log frequency between the two SwiftF0 frames around it when both are voiced,
//!   else it takes the voicing and pitch of the nearer one. Grid frames after the last SwiftF0
//!   centre hold it.
//! - **Edge voicing.** The newest SwiftF0 frame of a slice sees at most 511 samples after its centre,
//!   and its confidence is depressed while its pitch stays accurate (measured on the real model:
//!   0.02-0.20 on formant-filtered voices and tones, against 0.9-0.99 two frames earlier; below
//!   0.005 on noise), and the two frames before it dip under 0.9 too (0.68-0.83 for a 350 Hz voice).
//!   With the plain 0.9 threshold most of the newest ~30 ms of every streaming tick was unvoiced
//!   (57 % missed voicing in the streaming A/B), and those are exactly the frames a tick returns.
//!   So the three outermost frames use [`EDGE_VOICING_THRESHOLDS`], each only when its inner
//!   neighbour is voiced and within [`EDGE_AGREE_CENTS`] (the same at the oldest end).
//! - **End alignment.** SwiftF0 returns `N / 256` frames, so its newest frame sits `256 + N % 256`
//!   samples before the end of the input, and the newest grid frames hold it. Every call is
//!   front-padded to a whole number of SwiftF0 frames, which puts the newest frame 256 samples
//!   (16 ms) from the end whatever the slice length. Streaming A/B (the F0 cache's 3520-sample
//!   slices, `N % 256 = 192` before): gross pitch error 0.72 -> 0.45 %, fine error 10.4 -> 7.0
//!   cents, voicing error 25.0 -> 22.0 % ([`SwiftF0PitchExtractor::with_end_alignment`] turns it
//!   off).
//! - **Range and octave gate.** `fmin` / `fmax` come from the voice's F0 range (`F0Range`), and the
//!   contour runs through the pitch bank's post-processing (`f0_post::F0Post`).
//!
//! # Frozen by P3-S0 (signatures)
//! - [`available`]`() -> bool`, [`SwiftF0PitchExtractor::load`]`(model_path, execution_provider_hint,
//!   opts)` and its `PitchExtractor` impl (F0 on the 16 kHz / hop-160 grid, Hz, 0 = unvoiced).
//!
//! # Added by P3-D
//! [`SwiftF0PitchExtractor::with_range`], [`SwiftF0PitchExtractor::with_voicing_threshold`],
//! [`SwiftF0PitchExtractor::with_end_alignment`], [`SwiftF0PitchExtractor::last_confidence`],
//! [`SwiftF0PitchExtractor::last_candidates`], [`SwiftF0PitchExtractor::outcome`] and the constants.

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, TensorRef};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::engine::options::F0Range;
use super::f0_post::{cents, F0Limits, F0Post, F0PostConfig};
use super::pitch_onnx::ShapePin;
use super::traits::PitchExtractor;

/// Samples per SwiftF0 frame (16 ms).
pub const SWIFTF0_HOP: usize = 256;
/// Lowest / highest F0 the model covers.
pub const SWIFTF0_FMIN: f32 = 46.875;
pub const SWIFTF0_FMAX: f32 = 2093.75;
/// SwiftF0's own voicing threshold.
pub const DEFAULT_VOICING_THRESHOLD: f32 = 0.9;
/// Voicing thresholds of the SwiftF0 frames at either end of a slice, outermost first. Each counts
/// as voiced at its threshold when its inner neighbour is voiced and within [`EDGE_AGREE_CENTS`].
/// Measured on formant-filtered voices (110-350 Hz, clean and 10 dB SNR) in 3520-sample slices: the
/// outermost frame's confidence has a median of 0.03-0.16 (noise stays at 0.002-0.005, and its
/// neighbours never pass), the next 0.68-0.99, the third 0.90-0.99 (5th percentile down to 0.87).
pub const EDGE_VOICING_THRESHOLDS: [f32; 3] = [0.008, 0.4, 0.7];
/// An edge frame counts as voiced at the lower threshold only this close to its voiced neighbour.
pub const EDGE_AGREE_CENTS: f32 = 150.0;
/// Shorter slices are zero-padded at the front to this many samples (the model needs a few frames
/// of context to be confident).
const MIN_SAMPLES: usize = 1024;
/// Whether calls are front-padded to a whole number of SwiftF0 frames by default (the streaming
/// A/B decides; see [`SwiftF0PitchExtractor::f0`]).
const ALIGN_END_DEFAULT: bool = true;
const GRID_HOP: usize = 160;

/// Whether SwiftF0 runs in this build (the model is installed separately; loading reports a
/// missing file).
pub fn available() -> bool {
    true
}

/// Where each input and output of the graph is.
#[derive(Clone, Debug)]
struct SwiftIo {
    audio: String,
    audio_rank: usize,
    fmin: String,
    fmin_rank: usize,
    fmax: String,
    fmax_rank: usize,
    pitch: String,
    confidence: String,
}

impl SwiftIo {
    fn inspect(session: &Session, path: &str) -> Result<SwiftIo, String> {
        let mut audio = None;
        let mut fmin = None;
        let mut fmax = None;
        for input in session.inputs() {
            let name = input.name().to_string();
            let dims = input
                .dtype()
                .tensor_shape()
                .map(|s| s.len())
                .ok_or_else(|| format!("SwiftF0 ONNX {path}: input '{name}' is not a tensor"))?;
            if input.dtype().tensor_type() != Some(TensorElementType::Float32) {
                return Err(format!("SwiftF0 ONNX {path}: input '{name}' is not float32"));
            }
            match name.to_ascii_lowercase().as_str() {
                "fmin" => fmin = Some((name, dims)),
                "fmax" => fmax = Some((name, dims)),
                _ if audio.is_none() && (1..=2).contains(&dims) => audio = Some((name, dims)),
                _ => return Err(format!("SwiftF0 ONNX {path}: unexpected input '{name}'")),
            }
        }
        let (audio, audio_rank) = audio.ok_or_else(|| format!("SwiftF0 ONNX {path}: no audio input [1, N]"))?;
        let (fmin, fmin_rank) = fmin.ok_or_else(|| format!("SwiftF0 ONNX {path}: no 'fmin' input"))?;
        let (fmax, fmax_rank) = fmax.ok_or_else(|| format!("SwiftF0 ONNX {path}: no 'fmax' input"))?;
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let find = |key: &str, fallback: usize| -> Option<String> {
            outputs
                .iter()
                .find(|n| n.eq_ignore_ascii_case(key))
                .or_else(|| outputs.get(fallback))
                .cloned()
        };
        let pitch = find("pitch", 0).ok_or_else(|| format!("SwiftF0 ONNX {path}: no pitch output"))?;
        let confidence = find("confidence", 1).ok_or_else(|| format!("SwiftF0 ONNX {path}: no confidence output"))?;
        if pitch == confidence {
            return Err(format!(
                "SwiftF0 ONNX {path}: expected 'pitch' and 'confidence' outputs, found {outputs:?}"
            ));
        }
        Ok(SwiftIo {
            audio,
            audio_rank,
            fmin,
            fmin_rank,
            fmax,
            fmax_rank,
            pitch,
            confidence,
        })
    }
}

fn scalar(rank: usize, value: f32) -> Result<Tensor<f32>, String> {
    if rank == 0 {
        Tensor::from_array(((), vec![value]))
    } else {
        Tensor::from_array((vec![1usize; rank], vec![value]))
    }
    .map_err(|e| format!("SwiftF0 scalar input: {e}"))
}

pub struct SwiftF0PitchExtractor {
    session: Session,
    io: SwiftIo,
    outcome: EpOutcome,
    /// DirectML: every call runs the first call's length (front zero padding; see `pitch_onnx::ShapePin`).
    pin: ShapePin,
    range: F0Range,
    voicing_threshold: f32,
    /// Front-pad every call to a whole number of SwiftF0 frames (see [`SwiftF0PitchExtractor::f0`]).
    align_end: bool,
    post: F0Post,
    post_cfg: F0PostConfig,
    // ---- scratch (reused every call) ----
    padded: Vec<f32>,
    raw_pitch: Vec<f32>,
    raw_conf: Vec<f32>,
    raw_voiced: Vec<bool>,
    confidence: Vec<f32>,
    /// The grid pitch of every frame before the voicing decision (F0 fusion's candidates).
    candidates: Vec<f32>,
    all_frames: Vec<bool>,
    scratch_conf: Vec<f32>,
}

impl SwiftF0PitchExtractor {
    /// Build the session on the EP ladder of `execution_provider_hint` (the outcome is recorded as
    /// `vc.pitch`). SwiftF0 is small enough that the pitch bank runs it on the CPU by default.
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("SwiftF0 model path is empty (install the vc/pitch-swiftf0 model)".into());
        }
        let (session, outcome) = build_session(
            Some("vc.pitch"),
            execution_provider_hint,
            opts,
            ModelSource::File(std::path::Path::new(path)),
        )
        .map_err(|e| format!("Failed to load SwiftF0 ONNX {path}: {e}"))?;
        // Another kind of graph set as the SwiftF0 model is named for what it is (P5-D, load only).
        if let Some(why) = super::rvc::foreign_graph_problem(&session, std::path::Path::new(path), "a SwiftF0 pitch model", &[super::rvc::GraphKind::PitchModel]) {
            return Err(format!("SwiftF0: {why}"));
        }
        let io = SwiftIo::inspect(&session, path)?;
        let range = F0Range::Speech;
        let pin = ShapePin::for_ep(&outcome.active);
        Ok(SwiftF0PitchExtractor {
            session,
            io,
            outcome,
            range,
            pin,
            voicing_threshold: DEFAULT_VOICING_THRESHOLD,
            align_end: ALIGN_END_DEFAULT,
            post: F0Post::new(),
            post_cfg: F0PostConfig::for_range(range),
            padded: Vec::new(),
            raw_pitch: Vec::new(),
            raw_conf: Vec::new(),
            raw_voiced: Vec::new(),
            confidence: Vec::new(),
            candidates: Vec::new(),
            all_frames: Vec::new(),
            scratch_conf: Vec::new(),
        })
    }

    /// The voice's F0 range (default speech): the model's `fmin` / `fmax` and the octave gate.
    pub fn with_range(mut self, range: F0Range) -> Self {
        self.range = range;
        self.post_cfg = F0PostConfig::for_range(range);
        self
    }

    /// Voicing threshold on SwiftF0's confidence (default 0.9).
    #[allow(dead_code)] // Tuning knob of the pitch A/B (dev-rig tests).
    pub fn with_voicing_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() {
            self.voicing_threshold = threshold.clamp(0.0, 1.0);
        }
        self
    }

    /// Front-pad each call to a whole number of SwiftF0 frames (default on; off runs the input as
    /// it is, the pre-A/B behaviour).
    #[allow(dead_code)] // Tuning knob of the pitch A/B (dev-rig tests).
    pub fn with_end_alignment(mut self, align: bool) -> Self {
        self.align_end = align;
        self
    }

    /// Per-frame confidence (0..1) of the last [`PitchExtractor::f0`] call, on its grid.
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }

    /// The last call's grid pitch before the voicing decision (Hz for every frame the model gave a
    /// pitch for, voiced or not): what F0 fusion confirms against a second extractor.
    pub fn last_candidates(&self) -> &[f32] {
        &self.candidates
    }

    /// Where the session runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    fn model_range(&self) -> (f32, f32) {
        let limits = F0Limits::for_range(self.range);
        (limits.min_hz.max(SWIFTF0_FMIN), limits.max_hz.min(SWIFTF0_FMAX))
    }

    /// Run the graph over `audio`, leaving `F` frames in `raw_pitch` / `raw_conf`.
    fn run(&mut self, audio_len: usize) -> Result<(), String> {
        let (lo, hi) = self.model_range();
        let shape = if self.io.audio_rank == 1 { vec![audio_len] } else { vec![1, audio_len] };
        let audio = TensorRef::from_array_view((shape, &self.padded[..audio_len])).map_err(|e| format!("SwiftF0 input: {e}"))?;
        let feed: Vec<(String, SessionInputValue<'_>)> = vec![
            (self.io.audio.clone(), audio.into()),
            (self.io.fmin.clone(), scalar(self.io.fmin_rank, lo)?.into()),
            (self.io.fmax.clone(), scalar(self.io.fmax_rank, hi)?.into()),
        ];
        let outputs = self
            .session
            .run(feed)
            .map_err(|e| format!("SwiftF0 run: {e}"))?;
        self.raw_pitch.clear();
        let pitch = &outputs[self.io.pitch.as_str()];
        if let Ok((_, data)) = pitch.try_extract_tensor::<f64>() {
            self.raw_pitch.extend(data.iter().map(|&v| v as f32));
        } else {
            let (_, data) = pitch
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("SwiftF0 pitch output: {e}"))?;
            self.raw_pitch.extend_from_slice(data);
        }
        let (_, conf) = outputs[self.io.confidence.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("SwiftF0 confidence output: {e}"))?;
        self.raw_conf.clear();
        self.raw_conf.extend_from_slice(conf);
        if self.raw_conf.len() != self.raw_pitch.len() {
            return Err(format!(
                "SwiftF0 returned {} pitch and {} confidence frames",
                self.raw_pitch.len(),
                self.raw_conf.len()
            ));
        }
        Ok(())
    }
}

/// Voicing of SwiftF0 frames: `conf >= threshold`, except that the three frames at each end of the
/// slice (they lack context on one side) use [`EDGE_VOICING_THRESHOLDS`] when their inner neighbour
/// is voiced and within [`EDGE_AGREE_CENTS`]; the inner ones are decided first, so the outermost
/// can lean on them.
fn frame_voicing(pitch: &[f32], conf: &[f32], threshold: f32, voiced: &mut Vec<bool>) {
    voiced.clear();
    voiced.extend(
        pitch
            .iter()
            .zip(conf)
            .map(|(&p, &c)| p.is_finite() && p > 0.0 && c >= threshold),
    );
    let n = voiced.len();
    let relax = |edge: usize, inner: usize, min_conf: f32, voiced: &mut Vec<bool>| {
        let (p, q) = (pitch[edge], pitch[inner]);
        if !voiced[edge] && voiced[inner] && p.is_finite() && p > 0.0 && conf[edge] >= min_conf && cents(p, q).abs() <= EDGE_AGREE_CENTS {
            voiced[edge] = true;
        }
    };
    for depth in (0..EDGE_VOICING_THRESHOLDS.len()).rev() {
        if n < depth + 2 {
            continue;
        }
        let min_conf = EDGE_VOICING_THRESHOLDS[depth].min(threshold);
        relax(n - 1 - depth, n - 2 - depth, min_conf, voiced);
        relax(depth, depth + 1, min_conf, voiced);
    }
}

/// Resample SwiftF0 frames (centre `256 * i` of a buffer whose first `offset` samples are padding)
/// onto the hop-160 grid of the original `n` samples: `n / 160 + 1` frames into `f0` / `conf`.
fn to_grid(pitch: &[f32], conf: &[f32], voiced: &[bool], n: usize, offset: usize, f0: &mut Vec<f32>, grid_conf: &mut Vec<f32>) {
    let frames = n / GRID_HOP + 1;
    f0.clear();
    grid_conf.clear();
    let last = pitch.len().saturating_sub(1);
    for j in 0..frames {
        if pitch.is_empty() {
            f0.push(0.0);
            grid_conf.push(0.0);
            continue;
        }
        let u = (GRID_HOP * j + offset) as f64 / SWIFTF0_HOP as f64;
        let (i0, i1, w) = if u >= last as f64 {
            (last, last, 0.0)
        } else {
            let i0 = u.floor() as usize;
            (i0, i0 + 1, (u - i0 as f64) as f32)
        };
        let c = conf[i0] + (conf[i1] - conf[i0]) * w;
        let v = match (voiced[i0], voiced[i1]) {
            (true, true) if w == 0.0 || i0 == i1 => pitch[i0],
            (true, true) => (pitch[i0].log2() + (pitch[i1].log2() - pitch[i0].log2()) * w).exp2(),
            (a, b) => {
                let near = if w < 0.5 { i0 } else { i1 };
                let voiced_near = if w < 0.5 { a } else { b };
                if voiced_near {
                    pitch[near]
                } else {
                    0.0
                }
            }
        };
        f0.push(if v.is_finite() && v > 0.0 { v } else { 0.0 });
        grid_conf.push(c.clamp(0.0, 1.0));
    }
}

impl PitchExtractor for SwiftF0PitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let mut f0 = Vec::with_capacity(mono_16k.len() / GRID_HOP + 1);
        self.f0_into(mono_16k, &mut f0)?;
        Ok(f0)
    }

    /// P3B-E: straight into the caller's buffer (no contour allocation; the ONNX Runtime call
    /// itself still allocates its inputs and outputs).
    fn f0_into(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        out.clear();
        let n = mono_16k.len();
        if n == 0 {
            self.confidence.clear();
            return Ok(());
        }
        // SwiftF0 returns `N / 256` frames centred on `256 i`, so the newest frame sits `256 + N % 256`
        // samples before the end of the input. Front padding to a multiple of 256 puts it 256
        // samples (16 ms) before the newest sample: the newest grid frames then hold a frame at most
        // 6 ms older than themselves instead of up to 22 ms (`to_grid`).
        let want = n.max(MIN_SAMPLES);
        let total = if self.align_end {
            self.pin
                .size_aligned(want.next_multiple_of(SWIFTF0_HOP), SWIFTF0_HOP)
        } else {
            self.pin.size(want)
        };
        let offset = total - n;
        self.padded.clear();
        self.padded.resize(offset, 0.0);
        self.padded.extend_from_slice(mono_16k);
        for v in self.padded.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        self.run(offset + n)?;
        frame_voicing(&self.raw_pitch, &self.raw_conf, self.voicing_threshold, &mut self.raw_voiced);
        to_grid(&self.raw_pitch, &self.raw_conf, &self.raw_voiced, n, offset, out, &mut self.confidence);
        self.all_frames.clear();
        self.all_frames
            .extend(self.raw_pitch.iter().map(|&p| p.is_finite() && p > 0.0));
        to_grid(
            &self.raw_pitch,
            &self.raw_conf,
            &self.all_frames,
            n,
            offset,
            &mut self.candidates,
            &mut self.scratch_conf,
        );
        self.post.process(out, &self.post_cfg);
        Ok(())
    }

    /// Network context ±10 SwiftF0 frames, of which ±3-4 matter above 1 cent; plus the grid
    /// interpolation and the 3-tap median: 10 grid frames before, 8 after.
    fn receptive_frames(&self) -> Option<(usize, usize)> {
        Some((10, 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_frames_use_the_lower_thresholds_only_next_to_a_voiced_neighbour() {
        let pitch = [200.0, 201.0, 202.0, 203.0, 204.0, 205.0];
        let mut v = Vec::new();
        // Measured edge confidences of a voiced 350 Hz slice: 0.899, 0.679, 0.016.
        frame_voicing(&pitch, &[0.95, 0.99, 0.99, 0.899, 0.679, 0.016], 0.9, &mut v);
        assert_eq!(v, vec![true; 6]);
        // Noise-level confidence (0.004) at the edge stays unvoiced.
        frame_voicing(&pitch, &[0.95, 0.99, 0.99, 0.99, 0.99, 0.004], 0.9, &mut v);
        assert_eq!(v, vec![true, true, true, true, true, false]);
        // Each edge frame needs its own threshold, and a voiced inner neighbour.
        frame_voicing(&pitch, &[0.95, 0.99, 0.99, 0.99, 0.3, 0.3], 0.9, &mut v);
        assert_eq!(v, vec![true, true, true, true, false, false]);
        frame_voicing(&pitch, &[0.95, 0.99, 0.99, 0.6, 0.9, 0.3], 0.9, &mut v);
        assert_eq!(v, vec![true, true, true, false, true, true]);
        // An edge frame that jumps away from its neighbour is not voiced.
        frame_voicing(
            &[200.0, 201.0, 202.0, 203.0, 204.0, 410.0],
            &[0.95, 0.99, 0.99, 0.99, 0.99, 0.3],
            0.9,
            &mut v,
        );
        assert_eq!(v, vec![true, true, true, true, true, false]);
        // The oldest frames get the same rules.
        frame_voicing(&pitch, &[0.05, 0.6, 0.8, 0.99, 0.99, 0.99], 0.9, &mut v);
        assert_eq!(v, vec![true; 6]);
        // Non-finite pitch is never voiced; short slices are fine.
        frame_voicing(&[f32::NAN, 200.0], &[0.99, 0.99], 0.9, &mut v);
        assert_eq!(v, vec![false, true]);
        frame_voicing(&[200.0], &[0.2], 0.9, &mut v);
        assert_eq!(v, vec![false]);
    }

    #[test]
    fn grid_interpolates_in_log_frequency_and_holds_the_end() {
        // SwiftF0 frames at 0, 256, 512, 768: 100, 200, 200, 400 Hz, all voiced.
        let pitch = [100.0, 200.0, 200.0, 400.0];
        let conf = [1.0, 1.0, 1.0, 1.0];
        let voiced = [true; 4];
        let (mut f0, mut c) = (Vec::new(), Vec::new());
        to_grid(&pitch, &conf, &voiced, 1200, 0, &mut f0, &mut c);
        assert_eq!(f0.len(), 1200 / 160 + 1);
        // j=0 -> u=0 -> 100; j=1 -> u=0.625 -> 100 * 2^0.625.
        assert!((f0[0] - 100.0).abs() < 1e-3);
        assert!((f0[1] - 100.0 * 0.625f32.exp2()).abs() < 1e-2, "{f0:?}");
        // j=4 -> u=2.5 -> between 200 and 400: 200 * 2^0.5.
        assert!((f0[4] - 200.0 * 0.5f32.exp2()).abs() < 1e-2, "{f0:?}");
        // j=5 (800) and after: past the last centre (768) -> hold 400.
        assert!(f0[5..].iter().all(|&v| (v - 400.0).abs() < 1e-3), "{f0:?}");
    }

    #[test]
    fn grid_takes_voicing_from_the_nearer_frame_and_respects_padding() {
        let pitch = [150.0, 150.0, 300.0];
        let conf = [0.95, 0.1, 0.95];
        let voiced = [true, false, true];
        let (mut f0, mut c) = (Vec::new(), Vec::new());
        to_grid(&pitch, &conf, &voiced, 512, 0, &mut f0, &mut c);
        // j=0 u=0 -> 150; j=1 u=0.625 -> nearer frame 1 (unvoiced); j=2 u=1.25 -> frame 1;
        // j=3 u=1.875 -> frame 2 (voiced 300).
        assert_eq!(f0, vec![150.0, 0.0, 0.0, 300.0]);
        assert!((c[1] - (0.95 + (0.1 - 0.95) * 0.625)).abs() < 1e-5);
        // With 512 samples of front padding, grid frame 0 is SwiftF0 frame 2.
        to_grid(&pitch, &conf, &voiced, 160, 512, &mut f0, &mut c);
        assert_eq!(f0, vec![300.0, 300.0]);
        // No frames at all: unvoiced grid.
        to_grid(&[], &[], &[], 320, 0, &mut f0, &mut c);
        assert_eq!(f0, vec![0.0; 3]);
    }

    /// Real SwiftF0: harmonic tones read within 10 cents on the grid, silence is unvoiced, the
    /// newest frames of a slice stay voiced, and short slices work.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test swiftf0 -- --ignored`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with vc-assets/vc/pitch-swiftf0/model.onnx"]
    fn real_swiftf0_tracks_tones_on_the_grid() {
        let Some(path) = super::super::f0_fusion::test_model("pitch-swiftf0") else {
            eprintln!("SwiftF0 model not found; skipped");
            return;
        };
        let mut sw = SwiftF0PitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        for hz in [98.0f32, 220.0, 440.0, 880.0] {
            let x = super::super::f0_fusion::ab::harmonic_tone(hz, 16_000, 0.3);
            let f0 = sw.f0(&x).unwrap();
            assert_eq!(f0.len(), 16_000 / 160 + 1);
            let interior = &f0[10..f0.len() - 1];
            let voiced = interior.iter().filter(|&&v| v > 0.0).count();
            assert!(voiced as f32 >= 0.95 * interior.len() as f32, "{hz} Hz: {voiced}/{}", interior.len());
            for &v in interior.iter().filter(|&&v| v > 0.0) {
                assert!(cents(v, hz).abs() < 10.0, "{hz} Hz read as {v}");
            }
            // The newest frames the F0 cache keeps (all but the one at the very end) are voiced.
            assert!(
                f0[f0.len() - 4..f0.len() - 1].iter().all(|&v| v > 0.0),
                "{hz} Hz edge: {:?}",
                &f0[f0.len() - 6..]
            );
            let short = sw.f0(&x[..700]).unwrap();
            assert_eq!(short.len(), 700 / 160 + 1);
        }
        let silence = sw.f0(&vec![0.0; 8000]).unwrap();
        assert!(silence.iter().all(|&v| v == 0.0));
        assert_eq!(sw.last_confidence().len(), silence.len());
    }
}
