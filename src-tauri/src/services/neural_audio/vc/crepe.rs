//! CREPE-tiny pitch extractor (P3-D). A real CREPE-tiny (the legacy `crepe` value was an alias of
//! RMVPE until P3).
//!
//! Model: model-store entry `vc/pitch-crepe-tiny` (`assets::model_file`), else an explicit path (the
//! pitch bank factory in `f0_fusion` resolves it).
//!
//! # Graph contract (yqzhishen/onnxcrepe v1.1.0 `tiny.onnx`, checked by
//! `scripts/vc-templates/export_pitch.py`)
//! `frames` f32 `[n, 1024]` -> `probabilities` f32 `[n, 360]` (sigmoid salience on CREPE's 20-cent
//! grid, bin 0 = 1997.38 cents ≈ 31.7 Hz). Frame `j` is the 1024-sample window centred on sample
//! `160 * j` of the signal zero-padded by 512 on both sides (torchcrepe's framing), each window minus
//! its mean and divided by `max(std, 1e-10)`. So the frames are exactly the hop-160 grid:
//! `n = N / 160 + 1`.
//!
//! # Decoding (RVC's `crepe` F0 method)
//! Local weighted average of argmax ± 4 bins (`to_local_average_cents`), periodicity = the peak
//! probability; then, like RVC: a 3-tap median on the periodicity, a 3-tap mean on the F0, and
//! frames with periodicity below [`PERIODICITY_THRESHOLD`] are unvoiced. Windows quieter than
//! [`SILENCE_DBFS`] are unvoiced as well (CREPE normalises every window, so it would otherwise read a
//! pitch from the noise floor). Then the voice's F0 range and the pitch bank's octave gate
//! (`f0_post::F0Post`).
//!
//! Streaming: zero padding at the newest edge measured better than reflection (median error of the
//! newest frame 15.7 vs 39.7 cents on a gliding voice), so the torchcrepe framing is kept.
//!
//! # Frozen by P3-S0 (signatures)
//! - [`available`]`() -> bool`, [`CrepePitchExtractor::load`]`(model_path, execution_provider_hint,
//!   opts)` and its `PitchExtractor` impl (F0 on the 16 kHz / hop-160 grid, Hz, 0 = unvoiced).
//!
//! # Added by P3-D
//! [`CrepePitchExtractor::with_range`], [`CrepePitchExtractor::with_periodicity_threshold`],
//! [`CrepePitchExtractor::last_confidence`], [`CrepePitchExtractor::outcome`].

use ort::session::{Session, SessionInputValue};
use ort::value::{TensorElementType, TensorRef};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::engine::options::F0Range;
use super::f0_post::{F0Post, F0PostConfig};
use super::pitch_onnx::{local_average_f0, ShapePin, SALIENCE_BINS};
use super::traits::PitchExtractor;

/// Samples per CREPE window.
pub const CREPE_WINDOW: usize = 1024;
const HOP: usize = 160;
/// RVC's voicing threshold on CREPE's periodicity (`f0[pd < 0.1] = 0`).
pub const PERIODICITY_THRESHOLD: f32 = 0.1;
/// Windows below this RMS level are unvoiced.
pub const SILENCE_DBFS: f32 = -60.0;

/// Whether CREPE-tiny runs in this build (the model is installed separately; loading reports a
/// missing file).
pub fn available() -> bool {
    true
}

pub struct CrepePitchExtractor {
    session: Session,
    input: String,
    output: String,
    outcome: EpOutcome,
    /// DirectML: every call runs the first call's window count (extra windows are zeros; CREPE's
    /// windows are independent, so this is exact).
    pin: ShapePin,
    periodicity_threshold: f32,
    post: F0Post,
    post_cfg: F0PostConfig,
    // ---- scratch ----
    frames: Vec<f32>,
    levels: Vec<f32>,
    raw_f0: Vec<f32>,
    periodicity: Vec<f32>,
    tmp: Vec<f32>,
    confidence: Vec<f32>,
}

impl CrepePitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("CREPE-tiny model path is empty (install the vc/pitch-crepe-tiny model)".into());
        }
        let (session, outcome) = build_session(
            Some("vc.pitch"),
            execution_provider_hint,
            opts,
            ModelSource::File(std::path::Path::new(path)),
        )
        .map_err(|e| format!("Failed to load CREPE-tiny ONNX {path}: {e}"))?;
        // Another kind of graph set as the CREPE model is named for what it is (P5-D, load only).
        if let Some(why) = super::rvc::foreign_graph_problem(&session, std::path::Path::new(path), "a CREPE pitch model", &[super::rvc::GraphKind::PitchModel]) {
            return Err(format!("CREPE: {why}"));
        }
        let inputs = session.inputs();
        let [input] = inputs else {
            return Err(format!(
                "CREPE ONNX {path}: expected one input (frames [n, 1024]), found {}",
                inputs.len()
            ));
        };
        let dims: Vec<i64> = input
            .dtype()
            .tensor_shape()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        if input.dtype().tensor_type() != Some(TensorElementType::Float32) || dims.len() != 2 || (dims[1] > 0 && dims[1] as usize != CREPE_WINDOW) {
            return Err(format!(
                "CREPE ONNX {path}: input '{}' is {dims:?}; expected float32 frames [n, 1024]",
                input.name()
            ));
        }
        let input = input.name().to_string();
        let output = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| format!("CREPE ONNX {path}: no output"))?;
        let range = F0Range::Speech;
        let pin = ShapePin::for_ep(&outcome.active);
        Ok(CrepePitchExtractor {
            session,
            input,
            output,
            outcome,
            pin,
            periodicity_threshold: PERIODICITY_THRESHOLD,
            post: F0Post::new(),
            post_cfg: F0PostConfig::for_range(range),
            frames: Vec::new(),
            levels: Vec::new(),
            raw_f0: Vec::new(),
            periodicity: Vec::new(),
            tmp: Vec::new(),
            confidence: Vec::new(),
        })
    }

    /// The voice's F0 range (default speech) for the range limits and the octave gate.
    pub fn with_range(mut self, range: F0Range) -> Self {
        self.post_cfg = F0PostConfig::for_range(range);
        self
    }

    /// Voicing threshold on the periodicity (default RVC's 0.1).
    #[allow(dead_code)] // Tuning knob (A/B experiments).
    pub fn with_periodicity_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() {
            self.periodicity_threshold = threshold.clamp(0.0, 1.0);
        }
        self
    }

    /// Per-frame periodicity (0..1, after the 3-tap median) of the last [`PitchExtractor::f0`] call.
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }

    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }
}

/// torchcrepe framing: `n / 160 + 1` windows of 1024 samples centred on `160 * j` of the signal
/// zero-padded by 512 on both sides, each normalised to zero mean and unit variance. `levels`
/// receives each window's RMS in dBFS (before normalisation).
fn frame_windows(audio: &[f32], frames: &mut Vec<f32>, levels: &mut Vec<f32>) -> usize {
    let n = audio.len() / HOP + 1;
    frames.clear();
    frames.resize(n * CREPE_WINDOW, 0.0);
    levels.clear();
    let half = CREPE_WINDOW / 2;
    for (j, window) in frames.chunks_exact_mut(CREPE_WINDOW).enumerate() {
        let origin = (j * HOP) as isize - half as isize;
        for (k, dst) in window.iter_mut().enumerate() {
            let i = origin + k as isize;
            *dst = if i >= 0 && (i as usize) < audio.len() {
                let v = audio[i as usize];
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            } else {
                0.0
            };
        }
        let mean = window.iter().map(|&v| f64::from(v)).sum::<f64>() / CREPE_WINDOW as f64;
        let energy = window
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / CREPE_WINDOW as f64;
        levels.push((10.0 * energy.max(1e-20).log10()) as f32);
        let var = window
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / CREPE_WINDOW as f64;
        let std = var.sqrt().max(1e-10);
        for v in window.iter_mut() {
            *v = ((f64::from(*v) - mean) / std) as f32;
        }
    }
    n
}

/// 3-tap median with reflected edges (torchcrepe `filter.median(pd, 3)`: the first and last
/// windows are `(x1, x0, x1)` and `(x[n-2], x[n-1], x[n-2])`).
fn median3(src: &[f32], out: &mut Vec<f32>) {
    out.clear();
    let n = src.len();
    for i in 0..n {
        let b = src[i];
        let (a, c) = if n == 1 {
            (b, b)
        } else if i == 0 {
            (src[1], src[1])
        } else if i == n - 1 {
            (src[n - 2], src[n - 2])
        } else {
            (src[i - 1], src[i + 1])
        };
        out.push(a.max(b).min(a.min(b).max(c)));
    }
}

/// 3-tap mean (torchcrepe `filter.mean(f0, 3)`): the average of the positive values of the window,
/// positions past either end not counted. A non-positive frame stays as it is.
fn mean3(src: &[f32], out: &mut Vec<f32>) {
    out.clear();
    let n = src.len();
    for i in 0..n {
        if src[i] <= 0.0 {
            out.push(src[i]);
            continue;
        }
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for &v in &src[i.saturating_sub(1)..(i + 2).min(n)] {
            if v > 0.0 {
                sum += v;
                count += 1.0;
            }
        }
        out.push(sum / count);
    }
}

impl PitchExtractor for CrepePitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if mono_16k.is_empty() {
            self.confidence.clear();
            return Ok(Vec::new());
        }
        let n = frame_windows(mono_16k, &mut self.frames, &mut self.levels);
        let run_n = self.pin.size(n);
        self.frames.resize(run_n * CREPE_WINDOW, 0.0);
        let input = TensorRef::from_array_view((vec![run_n, CREPE_WINDOW], self.frames.as_slice())).map_err(|e| format!("CREPE input: {e}"))?;
        let feed: Vec<(String, SessionInputValue<'_>)> = vec![(self.input.clone(), input.into())];
        let outputs = self
            .session
            .run(feed)
            .map_err(|e| format!("CREPE run: {e}"))?;
        let (shape, prob) = outputs[self.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("CREPE output: {e}"))?;
        let rows = match shape
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [r, b] if *b == SALIENCE_BINS => *r,
            [1, r, b] if *b == SALIENCE_BINS => *r,
            other => return Err(format!("CREPE output shape {other:?}; expected probabilities [n, 360]")),
        };
        if rows != run_n || prob.len() < rows * SALIENCE_BINS {
            return Err(format!("CREPE returned {rows} frames for {run_n} windows"));
        }
        let rows = n;
        self.raw_f0.clear();
        self.periodicity.clear();
        for row in prob[..rows * SALIENCE_BINS].chunks_exact(SALIENCE_BINS) {
            let (hz, peak) = local_average_f0(row);
            self.raw_f0.push(hz);
            self.periodicity
                .push(if peak.is_finite() { peak.clamp(0.0, 1.0) } else { 0.0 });
        }
        drop(outputs);
        median3(&self.periodicity, &mut self.confidence);
        mean3(&self.raw_f0, &mut self.tmp);
        let mut f0 = Vec::with_capacity(n);
        for ((&hz, &pd), &level) in self.tmp.iter().zip(&self.confidence).zip(&self.levels) {
            f0.push(if pd >= self.periodicity_threshold && level >= SILENCE_DBFS {
                hz
            } else {
                0.0
            });
        }
        self.post.process(&mut f0, &self.post_cfg);
        Ok(f0)
    }

    /// ±512-sample windows (±3.2 frames), then the 3-tap filters and the median of the octave gate.
    fn receptive_frames(&self) -> Option<(usize, usize)> {
        Some((6, 6))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_follow_torchcrepe_framing() {
        let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin() + 0.25).collect();
        let (mut frames, mut levels) = (Vec::new(), Vec::new());
        let n = frame_windows(&audio, &mut frames, &mut levels);
        assert_eq!(n, 1000 / 160 + 1);
        assert_eq!(frames.len(), n * CREPE_WINDOW);
        assert_eq!(levels.len(), n);
        // Window j sample k is audio[160 j + k - 512] (zero outside), normalised.
        for j in [0usize, 3, n - 1] {
            let raw: Vec<f64> = (0..CREPE_WINDOW)
                .map(|k| {
                    let i = (j * HOP + k) as isize - 512;
                    if i >= 0 && (i as usize) < audio.len() {
                        f64::from(audio[i as usize])
                    } else {
                        0.0
                    }
                })
                .collect();
            let mean = raw.iter().sum::<f64>() / 1024.0;
            let std = (raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 1024.0).sqrt();
            for k in [0usize, 100, 511, 512, 700, 1023] {
                let want = ((raw[k] - mean) / std) as f32;
                assert!((frames[j * CREPE_WINDOW + k] - want).abs() < 1e-4, "j {j} k {k}");
            }
        }
        // A silent input stays finite (std floor) and reads as far below the silence level.
        let n = frame_windows(&[0.0; 320], &mut frames, &mut levels);
        assert_eq!(n, 3);
        assert!(frames.iter().all(|v| *v == 0.0));
        assert!(levels.iter().all(|&l| l < SILENCE_DBFS));
    }

    #[test]
    fn rvc_filters() {
        let mut out = Vec::new();
        median3(&[0.0, 0.9, 0.05, 0.8, 0.7], &mut out);
        assert_eq!(out, vec![0.9, 0.05, 0.8, 0.7, 0.8]);
        mean3(&[100.0, 110.0, 0.0, 120.0], &mut out);
        assert_eq!(out, vec![105.0, 105.0, 0.0, 120.0]);
    }

    /// Real CREPE-tiny: harmonic tones within 15 cents, silence unvoiced.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test crepe -- --ignored`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with vc-assets/vc/pitch-crepe-tiny/model.onnx"]
    fn real_crepe_tracks_tones() {
        let Some(path) = super::super::f0_fusion::test_model("pitch-crepe-tiny") else {
            eprintln!("CREPE-tiny model not found; skipped");
            return;
        };
        let mut crepe = CrepePitchExtractor::load(&path.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        for hz in [98.0f32, 220.0, 440.0, 880.0] {
            let x = super::super::f0_fusion::ab::harmonic_tone(hz, 8_000, 0.3);
            let f0 = crepe.f0(&x).unwrap();
            assert_eq!(f0.len(), 8_000 / 160 + 1);
            let interior = &f0[4..f0.len() - 4];
            assert!(interior.iter().all(|&v| v > 0.0), "{hz} Hz: {interior:?}");
            for &v in interior {
                assert!(super::super::f0_post::cents(v, hz).abs() < 15.0, "{hz} Hz read as {v}");
            }
        }
        let silence = crepe.f0(&vec![0.0; 4000]).unwrap();
        assert!(silence.iter().all(|&v| v == 0.0));
    }
}
