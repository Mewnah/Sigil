//! Input/output contract shared by the ONNX pitch extractors (FCPE, RMVPE).
//!
//! The declared inputs are inspected once at load so a graph is never fed a tensor of the wrong
//! kind: a rank-1/2 main input is a 16 kHz waveform (`[N]` / `[1, N]`), a rank-3 one is a log-mel
//! spectrogram (`[1, n_mels, T]`). An optional scalar `threshold` input (w-okada style exports) is
//! fed a fixed value; any other extra input is rejected at load with a clear message.

use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{Tensor, TensorElementType, TensorRef};

/// Kind of the extractor's main (audio) input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MainInput {
    /// 16 kHz mono waveform, `[N]` (rank 1) or `[1, N]` (rank 2).
    Waveform { rank: usize },
    /// Log-mel spectrogram `[1, n_mels, T]`; `n_mels` is `None` when the dimension is dynamic.
    LogMel { n_mels: Option<usize> },
}

pub(super) struct PitchInputs {
    main_name: String,
    pub(super) main: MainInput,
    main_dims: Vec<i64>,
    /// (name, rank, value) of a declared threshold input.
    threshold: Option<(String, usize, f32)>,
}

fn is_threshold_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("threshold") || lower == "thred"
}

impl PitchInputs {
    /// Validate `session`'s declared inputs. `label` names the extractor in messages.
    pub(super) fn inspect(session: &Session, label: &str, path: &str, threshold_value: f32) -> Result<Self, String> {
        let mut main: Option<(String, MainInput, Vec<i64>)> = None;
        let mut threshold = None;
        for input in session.inputs() {
            let name = input.name().to_string();
            let dims: Vec<i64> = input
                .dtype()
                .tensor_shape()
                .map(|s| s.to_vec())
                .ok_or_else(|| format!("{label} ONNX {path}: input '{name}' is not a tensor"))?;
            match input.dtype().tensor_type() {
                Some(TensorElementType::Float32) => {}
                Some(other) => {
                    return Err(format!(
                        "{label} ONNX {path}: input '{name}' is {other:?}; Sigil feeds float32 \
                         (export the model in fp32)"
                    ))
                }
                None => return Err(format!("{label} ONNX {path}: input '{name}' has no type")),
            }
            if is_threshold_name(&name) {
                threshold = Some((name, dims.len(), threshold_value));
                continue;
            }
            if let Some((first, _, _)) = &main {
                return Err(format!(
                    "{label} ONNX {path}: unexpected extra input '{name}' (besides '{first}'); \
                     expected one audio input plus an optional threshold"
                ));
            }
            let kind = match dims.len() {
                1 | 2 => MainInput::Waveform { rank: dims.len() },
                3 => MainInput::LogMel {
                    n_mels: usize::try_from(dims[1]).ok().filter(|&m| m > 0),
                },
                _ => {
                    return Err(format!(
                        "{label} ONNX {path}: input '{name}' has unsupported shape {dims:?}; \
                         expected waveform [1, N] or log-mel [1, n_mels, T]"
                    ))
                }
            };
            main = Some((name, kind, dims));
        }
        let (main_name, main, main_dims) = main.ok_or_else(|| format!("{label} ONNX {path}: no audio input found"))?;
        Ok(Self {
            main_name,
            main,
            main_dims,
            threshold,
        })
    }

    /// Name of the main (audio) input.
    pub(super) fn main_name(&self) -> &str {
        &self.main_name
    }

    /// Declared shape of the main input (for error messages).
    pub(super) fn main_dims(&self) -> &[i64] {
        &self.main_dims
    }

    /// Run `session` with the main input (`shape`, `data`) plus the threshold input if declared.
    pub(super) fn run<'s>(&self, session: &'s mut Session, shape: Vec<usize>, data: &[f32], label: &str) -> Result<SessionOutputs<'s>, String> {
        let main = TensorRef::from_array_view((shape, data)).map_err(|e| format!("{label} input tensor: {e}"))?;
        let mut feed: Vec<(String, SessionInputValue<'_>)> = vec![(self.main_name.clone(), main.into())];
        if let Some((name, rank, value)) = &self.threshold {
            let t = if *rank == 0 {
                Tensor::from_array(((), vec![*value]))
            } else {
                Tensor::from_array((vec![1usize; *rank], vec![*value]))
            }
            .map_err(|e| format!("{label} threshold tensor: {e}"))?;
            feed.push((name.clone(), t.into()));
        }
        session.run(feed).map_err(|e| format!("{label} run: {e}"))
    }
}

/// Waveform tensor shape for `n` samples and the declared rank.
pub(super) fn waveform_shape(rank: usize, n: usize) -> Vec<usize> {
    if rank == 1 {
        vec![n]
    } else {
        vec![1, n]
    }
}

/// DirectML compiles a session for the first input shape it runs; every other shape then runs about
/// 20x slower on every call (measured on RMVPE: 4.5 ms at its first shape, 84-116 ms at the others;
/// CUDA and the CPU are unaffected). Extractors on DirectML therefore run every call at the size of
/// their first call (padding: RMVPE mel frames, CREPE windows, SwiftF0 front samples), which is the
/// largest in the voice engine (the first F0-cache call is the whole window). P3-D.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ShapePin {
    enabled: bool,
    pinned: Option<usize>,
    warned: bool,
}

impl ShapePin {
    /// Pinning for a session running on `active_ep`.
    pub(super) fn for_ep(active_ep: &str) -> ShapePin {
        ShapePin {
            enabled: active_ep == "dml",
            pinned: None,
            warned: false,
        }
    }

    /// The size to run a call of size `n` at (`n` itself when not pinning, or when `n` exceeds the
    /// pinned size: that call runs at its own, slower shape). The first call pins.
    pub(super) fn size(&mut self, n: usize) -> usize {
        self.size_aligned(n, 1)
    }

    /// [`size`](Self::size), pinning only when the padding `pinned - n` is a multiple of `align`.
    pub(super) fn size_aligned(&mut self, n: usize, align: usize) -> usize {
        if !self.enabled {
            return n;
        }
        match self.pinned {
            None => {
                self.pinned = Some(n);
                n
            }
            Some(p) if n <= p && (p - n).is_multiple_of(align.max(1)) => p,
            Some(p) => {
                if !self.warned {
                    self.warned = true;
                    tracing::debug!(target: "sigil::vc", "pitch extractor on DirectML: a call of {n} does not fit its pinned shape {p}; it runs slower");
                }
                n
            }
        }
    }
}

/// Pitch bins of the 20-cent salience grids of RMVPE and CREPE.
pub(super) const SALIENCE_BINS: usize = 360;
/// Cents of bin 0 (`cents = 20 * bin + 1997.3794084376191`, shared by RMVPE and CREPE).
pub(super) const SALIENCE_CENTS_OFFSET: f32 = 1_997.379_4;

/// RVC / CREPE `to_local_average_cents` for one salience row: the salience-weighted mean of
/// `cents = 20 * bin + offset` over argmax ± 4 bins, as Hz (`10 * 2^(c / 1200)`), plus the peak
/// salience. The Hz value is 0 when the window has no positive weight or the result is not
/// finite (a NaN near the peak is unvoiced, never NaN F0).
pub(super) fn local_average_f0(row: &[f32]) -> (f32, f32) {
    let (center, peak) = row
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, v)| if v > best.1 { (i, v) } else { best });
    let start = center.saturating_sub(4);
    let end = (center + 5).min(row.len());
    let mut weighted = 0.0f32;
    let mut weight = 0.0f32;
    for (bin, &s) in row.iter().enumerate().take(end).skip(start) {
        weighted += s * (20.0 * bin as f32 + SALIENCE_CENTS_OFFSET);
        weight += s;
    }
    if weight.is_nan() || weight <= 0.0 {
        return (0.0, peak);
    }
    let hz = 10.0 * (weighted / weight / 1200.0).exp2();
    (if hz.is_finite() { hz } else { 0.0 }, peak)
}

/// Flatten a direct F0 output (`[T]`, `[1, T]` or `[1, T, 1]`, Hz) into `T` values.
pub(super) fn flat_f0(dims: &[usize], data: &[f32]) -> Option<Vec<f32>> {
    let t = match dims {
        [t] => *t,
        [1, t] => *t,
        [1, t, 1] => *t,
        _ => return None,
    };
    (data.len() >= t).then(|| data[..t].to_vec())
}
