//! PESTO pitch extractor (P3-D). Streaming PESTO (self-supervised, tiny, causal): the model keeps
//! its own state, so each tick feeds only the audio it has not seen.
//!
//! Model: model-store entry `vc/pitch-pesto` (`assets::model_file`), else an explicit path. The entry
//! is import-only: a maintainer exports it with PESTO's own `realtime/export_onnx.py` (pesto commit
//! 62bc0c97, `mir-1k_g7.ckpt`) **at 16 kHz with a 10 ms step** (`--sampling_rate 16000
//! --chunk_size 160`) and imports the file (`scripts/vc-templates/export_pitch.py` notes). So
//! [`available`] is true only while that model is installed.
//!
//! # Graph contract (PESTO `StatelessPESTO` export)
//! - inputs: `audio` f32 `[B, chunk]` and `cache` f32 `[B, C]` (C fixed: the flattened state of
//!   PESTO's cached convolutions, zeros at the start);
//! - outputs: `prediction` `[B, T]`, `confidence` `[B, T]` (plus `volume` and `activations`, unused)
//!   and `cache_out` `[B, C]`, the state for the next chunk.
//!
//! What the export does not record is checked once at load ([`PestoPitchExtractor::load`], with a
//! synthetic probe): the frames per chunk (the hop must be 10 ms or divide it), whether
//! `prediction` is in Hz or in semitones (MIDI), and the model's delay (a causal CQT reports a pitch
//! change some samples after it happens), measured on a 220 -> 330 Hz step. A model the probe cannot
//! read is refused with the export command.
//!
//! # Streaming inside a stateless trait
//! `PitchExtractor::f0` receives overlapping slices of one rolling ring (the incremental F0 cache
//! re-extracts its margin every tick). A [`SliceTracker`] finds how many samples at the end of the
//! slice are new, only those are fed through the model (whole chunks; the remainder waits), and the
//! produced frames are kept in a ring keyed by stream time. The slice's grid (`160 * j`) is then read
//! from that ring, shifted by the measured delay; grid frames newer than the last frame hold it (the
//! F0 cache re-extracts them next tick, [`PitchExtractor::receptive_frames`]). A slice that does not
//! continue the last one (the first tick, a gap, a restart) resets the model state.
//!
//! # Frozen by P3-S0 (signatures)
//! - [`available`]`() -> bool`, [`PestoPitchExtractor::load`]`(model_path, execution_provider_hint,
//!   opts)` and its `PitchExtractor` impl (F0 on the 16 kHz / hop-160 grid, Hz, 0 = unvoiced).
//!
//! # Added by P3-D
//! [`PestoConfig`], [`PestoUnits`], [`PestoPitchExtractor::load_with`],
//! [`PestoPitchExtractor::with_range`], [`PestoPitchExtractor::last_confidence`],
//! [`PestoPitchExtractor::delay_samples`], [`PestoPitchExtractor::units`].

use std::collections::VecDeque;

use ort::session::{Session, SessionInputValue};
use ort::value::{TensorElementType, TensorRef};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::assets;
use super::engine::options::F0Range;
use super::engine::post::SliceTracker;
use super::f0_post::{cents, F0Post, F0PostConfig};
use super::traits::PitchExtractor;

const GRID_HOP: usize = 160;
/// Frames kept in the stream ring (10 s at 10 ms).
const RING_FRAMES: usize = 1_000;
/// Default voicing threshold on PESTO's confidence.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;
/// Longest delay the probe accepts (samples).
const MAX_DELAY: usize = 8_000;
/// Largest fixed chunk accepted (100 ms at 16 kHz): a longer one adds its whole length to the
/// latency of every frame, and the load probe needs several chunks.
const MAX_CHUNK: usize = 1_600;

/// Whether PESTO runs: the code does in every build, the model is import-only, so this is whether
/// `vc/pitch-pesto` is installed.
pub fn available() -> bool {
    assets::model_file(assets::PITCH_PESTO, None).is_some()
}

/// Unit of the `prediction` output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PestoUnits {
    Hz,
    /// MIDI semitones (69 = 440 Hz).
    Semitones,
}

/// Load options; `None` fields are measured by the load-time probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PestoConfig {
    pub confidence_threshold: f32,
    pub units: Option<PestoUnits>,
    /// Samples between the audio a frame describes and the end of the audio fed before it.
    pub delay_samples: Option<usize>,
}

impl Default for PestoConfig {
    fn default() -> Self {
        PestoConfig {
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            units: None,
            delay_samples: None,
        }
    }
}

#[derive(Clone, Debug)]
struct PestoIo {
    audio: String,
    cache: String,
    prediction: String,
    confidence: String,
    cache_out: String,
    /// Samples per call (the declared chunk, or 160 when it is dynamic).
    chunk: usize,
    cache_size: usize,
}

impl PestoIo {
    fn inspect(session: &Session, path: &str) -> Result<PestoIo, String> {
        let inputs: Vec<(String, Vec<i64>)> = session
            .inputs()
            .iter()
            .map(|i| {
                if i.dtype().tensor_type() != Some(TensorElementType::Float32) {
                    return Err(format!("PESTO ONNX {path}: input '{}' is not float32", i.name()));
                }
                Ok((
                    i.name().to_string(),
                    i.dtype()
                        .tensor_shape()
                        .map(|s| s.to_vec())
                        .unwrap_or_default(),
                ))
            })
            .collect::<Result<_, _>>()?;
        if inputs.len() != 2 {
            return Err(format!(
                "PESTO ONNX {path}: expected the streaming export's two inputs (audio, cache), found {}",
                inputs.len()
            ));
        }
        let pick = |key: &str, fallback: usize| {
            inputs
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(key))
                .unwrap_or(&inputs[fallback])
                .clone()
        };
        let (audio, audio_dims) = pick("audio", 0);
        let (cache, cache_dims) = pick("cache", 1);
        if audio == cache || audio_dims.len() != 2 || cache_dims.len() != 2 {
            return Err(format!("PESTO ONNX {path}: expected audio [B, chunk] and cache [B, C], found {inputs:?}"));
        }
        let cache_size = usize::try_from(cache_dims[1])
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| format!("PESTO ONNX {path}: the cache size is not fixed ({cache_dims:?}); re-export with realtime/export_onnx.py"))?;
        let chunk = usize::try_from(audio_dims[1])
            .ok()
            .filter(|&c| c > 0)
            .unwrap_or(GRID_HOP);
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let find = |key: &str| {
            outputs
                .iter()
                .find(|n| n.eq_ignore_ascii_case(key))
                .cloned()
        };
        let prediction = find("prediction").or_else(|| outputs.first().cloned());
        let confidence = find("confidence").or_else(|| outputs.get(1).cloned());
        let cache_out = find("cache_out").or_else(|| outputs.last().cloned());
        match (prediction, confidence, cache_out) {
            (Some(prediction), Some(confidence), Some(cache_out))
                if prediction != confidence && cache_out != prediction && cache_out != confidence =>
            {
                Ok(PestoIo {
                    audio,
                    cache,
                    prediction,
                    confidence,
                    cache_out,
                    chunk,
                    cache_size,
                })
            }
            _ => Err(format!(
                "PESTO ONNX {path}: expected prediction, confidence and cache_out outputs, found {outputs:?}"
            )),
        }
    }
}

/// Streaming state: the model cache and the frames produced so far.
struct Stream {
    cache: Vec<f32>,
    /// Samples received (fed or waiting in `pending`).
    received: u64,
    pending: Vec<f32>,
    /// Frames (Hz, 0 = unvoiced; confidence); `frames[0]` is stream frame `first`.
    frames: VecDeque<(f32, f32)>,
    first: u64,
}

impl Stream {
    fn new(cache_size: usize) -> Stream {
        Stream {
            cache: vec![0.0; cache_size],
            received: 0,
            pending: Vec::with_capacity(4 * GRID_HOP),
            frames: VecDeque::with_capacity(RING_FRAMES + 16),
            first: 0,
        }
    }

    fn reset(&mut self) {
        self.cache.fill(0.0);
        self.received = 0;
        self.pending.clear();
        self.frames.clear();
        self.first = 0;
    }
}

pub struct PestoPitchExtractor {
    session: Session,
    io: PestoIo,
    outcome: EpOutcome,
    /// Samples per produced frame.
    hop: usize,
    units: PestoUnits,
    delay: usize,
    threshold: f32,
    tracker: SliceTracker,
    stream: Stream,
    cache_next: Vec<f32>,
    post: F0Post,
    post_cfg: F0PostConfig,
    confidence: Vec<f32>,
    raw: Vec<(f32, f32)>,
    chunk_buf: Vec<f32>,
}

impl PestoPitchExtractor {
    pub fn load(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<Self, String> {
        Self::load_with(model_path, execution_provider_hint, opts, PestoConfig::default())
    }

    /// [`load`](Self::load) with explicit options (units and delay skip their probe).
    pub fn load_with(model_path: &str, execution_provider_hint: &str, opts: &VcSessionOptions, config: PestoConfig) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("PESTO model path is empty (import the vc/pitch-pesto model)".into());
        }
        let (session, outcome) = build_session(
            Some("vc.pitch"),
            execution_provider_hint,
            opts,
            ModelSource::File(std::path::Path::new(path)),
        )
        .map_err(|e| format!("Failed to load PESTO ONNX {path}: {e}"))?;
        Self::from_session(session, outcome, path, config)
    }

    fn from_session(session: Session, outcome: EpOutcome, path: &str, config: PestoConfig) -> Result<Self, String> {
        let io = PestoIo::inspect(&session, path)?;
        if io.chunk > MAX_CHUNK {
            return Err(format!(
                "PESTO ONNX {path}: a fixed {}-sample chunk is too long for real time (at most {MAX_CHUNK}); \
                 export with --sampling_rate 16000 --chunk_size 160",
                io.chunk
            ));
        }
        let cache_size = io.cache_size;
        let mut this = PestoPitchExtractor {
            session,
            outcome,
            hop: io.chunk,
            units: config.units.unwrap_or(PestoUnits::Hz),
            delay: config.delay_samples.unwrap_or(0),
            threshold: if config.confidence_threshold.is_finite() {
                config.confidence_threshold.clamp(0.0, 1.0)
            } else {
                DEFAULT_CONFIDENCE_THRESHOLD
            },
            tracker: SliceTracker::new(),
            stream: Stream::new(cache_size),
            cache_next: vec![0.0; cache_size],
            post: F0Post::new(),
            post_cfg: F0PostConfig::for_range(F0Range::Speech),
            confidence: Vec::new(),
            raw: Vec::new(),
            chunk_buf: Vec::new(),
            io,
        };
        // Frames per chunk: one chunk of silence.
        let silence = vec![0.0f32; this.io.chunk];
        let frames = this.run_chunk(&silence)?;
        if frames == 0 || !this.io.chunk.is_multiple_of(frames) {
            return Err(format!(
                "PESTO ONNX {path}: {frames} frame(s) for a {}-sample chunk; export with --chunk_size 160 at 16 kHz",
                this.io.chunk
            ));
        }
        this.hop = this.io.chunk / frames;
        if this.hop > GRID_HOP || !GRID_HOP.is_multiple_of(this.hop) {
            return Err(format!(
                "PESTO ONNX {path}: {} samples per frame; Sigil needs a 10 ms step at 16 kHz (export with --sampling_rate 16000 --chunk_size 160)",
                this.hop
            ));
        }
        if config.units.is_none() || config.delay_samples.is_none() {
            let (units, delay) = this.probe(path)?;
            this.units = config.units.unwrap_or(units);
            this.delay = config.delay_samples.unwrap_or(delay);
        }
        this.stream.reset();
        this.tracker.reset();
        Ok(this)
    }

    /// The voice's F0 range (default speech) for the range limits and the octave gate.
    pub fn with_range(mut self, range: F0Range) -> Self {
        self.post_cfg = F0PostConfig::for_range(range);
        self
    }

    /// Per-frame confidence (0..1) of the last [`PitchExtractor::f0`] call.
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }

    /// The model's measured (or configured) delay in samples.
    #[allow(dead_code)] // Diagnostics accessor (tests).
    pub fn delay_samples(&self) -> usize {
        self.delay
    }

    #[allow(dead_code)] // Diagnostics accessor (tests).
    pub fn units(&self) -> PestoUnits {
        self.units
    }

    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// Run one chunk through the model with the stream's cache; the produced frames go to
    /// `self.raw` (prediction as output, confidence). Returns the frame count.
    fn run_chunk(&mut self, chunk: &[f32]) -> Result<usize, String> {
        let audio = TensorRef::from_array_view((vec![1usize, chunk.len()], chunk)).map_err(|e| format!("PESTO input: {e}"))?;
        let cache =
            TensorRef::from_array_view((vec![1usize, self.io.cache_size], self.stream.cache.as_slice())).map_err(|e| format!("PESTO cache: {e}"))?;
        let feed: Vec<(String, SessionInputValue<'_>)> = vec![(self.io.audio.clone(), audio.into()), (self.io.cache.clone(), cache.into())];
        let outputs = self
            .session
            .run(feed)
            .map_err(|e| format!("PESTO run: {e}"))?;
        let (_, pred) = outputs[self.io.prediction.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("PESTO prediction: {e}"))?;
        let (_, conf) = outputs[self.io.confidence.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("PESTO confidence: {e}"))?;
        let (_, next) = outputs[self.io.cache_out.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("PESTO cache_out: {e}"))?;
        if next.len() != self.io.cache_size || pred.len() != conf.len() {
            return Err(format!(
                "PESTO returned {} cache values (expected {}), {} predictions and {} confidences",
                next.len(),
                self.io.cache_size,
                pred.len(),
                conf.len()
            ));
        }
        self.raw.clear();
        self.raw
            .extend(pred.iter().copied().zip(conf.iter().copied()));
        self.cache_next.copy_from_slice(next);
        drop(outputs);
        std::mem::swap(&mut self.stream.cache, &mut self.cache_next);
        Ok(self.raw.len())
    }

    /// A raw prediction in Hz (0 = unvoiced).
    fn to_hz(&self, prediction: f32, confidence: f32) -> f32 {
        if !(prediction.is_finite() && confidence.is_finite()) || confidence < self.threshold {
            return 0.0;
        }
        let hz = match self.units {
            PestoUnits::Hz => prediction,
            PestoUnits::Semitones => 440.0 * ((prediction - 69.0) / 12.0).exp2(),
        };
        if hz.is_finite() && hz > 0.0 {
            hz
        } else {
            0.0
        }
    }

    /// Feed `samples` (new stream audio) through the model, whole chunks at a time.
    fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
        self.stream.received += samples.len() as u64;
        self.stream
            .pending
            .extend(samples.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        let chunk = self.io.chunk;
        let mut offset = 0;
        let mut block = std::mem::take(&mut self.chunk_buf);
        while self.stream.pending.len() - offset >= chunk {
            block.clear();
            block.extend_from_slice(&self.stream.pending[offset..offset + chunk]);
            if let Err(e) = self.run_chunk(&block) {
                self.chunk_buf = block;
                return Err(e);
            }
            offset += chunk;
            for i in 0..self.raw.len() {
                let (p, c) = self.raw[i];
                let hz = self.to_hz(p, c);
                self.stream.frames.push_back((hz, c.clamp(0.0, 1.0)));
            }
            while self.stream.frames.len() > RING_FRAMES {
                self.stream.frames.pop_front();
                self.stream.first += 1;
            }
        }
        self.chunk_buf = block;
        self.stream.pending.drain(..offset);
        Ok(())
    }

    /// The probe: a 220 Hz harmonic tone, then 330 Hz from `step`; returns the units and the delay.
    fn probe(&mut self, path: &str) -> Result<(PestoUnits, usize), String> {
        self.stream.reset();
        let step = 12_800usize;
        // Whole chunks only (`chunks_exact` below), covering the step and 0.5 s after it.
        let total = (step + 8_000).div_ceil(self.io.chunk) * self.io.chunk;
        let mut phase = 0.0f64;
        let tone: Vec<f32> = (0..total)
            .map(|i| {
                let hz = if i < step { 220.0 } else { 330.0 };
                phase += hz / 16_000.0;
                (1..=6)
                    .map(|k| 0.25 / k as f64 * (std::f64::consts::TAU * phase * k as f64).sin())
                    .sum::<f64>() as f32
            })
            .collect();
        let chunk = self.io.chunk;
        let mut preds: Vec<(f32, f32)> = Vec::with_capacity(total / self.hop + 1);
        for c in tone.chunks_exact(chunk) {
            self.run_chunk(c)?;
            preds.extend_from_slice(&self.raw);
        }
        // Units: the confident predictions of 0.4-0.75 s (well after warm-up, before the step).
        let (lo, hi) = (6_400 / self.hop, 12_000 / self.hop);
        if preds.len() <= hi {
            return Err(format!(
                "PESTO ONNX {path}: {} frames for a {total}-sample probe; export with --sampling_rate 16000 --chunk_size 160",
                preds.len()
            ));
        }
        let mut steady: Vec<f32> = preds[lo..hi]
            .iter()
            .filter(|(p, c)| p.is_finite() && *c >= 0.3)
            .map(|(p, _)| *p)
            .collect();
        if steady.len() < 5 {
            return Err(format!(
                "PESTO ONNX {path}: the model is not confident on a 220 Hz probe tone; export it with realtime/export_onnx.py at 16 kHz"
            ));
        }
        steady.sort_by(f32::total_cmp);
        let median = steady[steady.len() / 2];
        let units = if median > 0.0 && cents(median, 220.0).abs() < 100.0 {
            PestoUnits::Hz
        } else if (median - 57.0).abs() < 1.0 {
            PestoUnits::Semitones
        } else {
            return Err(format!(
                "PESTO ONNX {path}: a 220 Hz probe reads {median} (expected about 220 Hz or 57 semitones); export at 16 kHz"
            ));
        };
        self.units = units;
        // Delay: the first frame after the step that reads closer to 330 Hz than to 220 Hz.
        let first = (step / self.hop..preds.len()).find(|&k| {
            let hz = match units {
                PestoUnits::Hz => preds[k].0,
                PestoUnits::Semitones => 440.0 * ((preds[k].0 - 69.0) / 12.0).exp2(),
            };
            hz > 0.0 && hz.is_finite() && cents(hz, 330.0).abs() < cents(hz, 220.0).abs()
        });
        let Some(k) = first else {
            return Err(format!("PESTO ONNX {path}: the model did not follow a 220 -> 330 Hz probe step"));
        };
        let delay = ((k + 1) * self.hop).saturating_sub(step);
        if delay > MAX_DELAY {
            return Err(format!("PESTO ONNX {path}: {delay} samples of delay is too much for real time"));
        }
        Ok((units, delay))
    }

    /// Stream frame (fractional) describing absolute sample `a`.
    fn frame_for(&self, a: u64) -> f64 {
        (a + self.delay as u64) as f64 / self.hop as f64 - 1.0
    }
}

impl PitchExtractor for PestoPitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        let n = mono_16k.len();
        if n == 0 {
            self.confidence.clear();
            return Ok(Vec::new());
        }
        let new = match self.tracker.advance(mono_16k) {
            Some(new) => new,
            None => {
                self.stream.reset();
                n
            }
        };
        self.feed(&mono_16k[n - new..])?;
        // The slice covers absolute samples [received - n, received).
        let start = self.stream.received - n as u64;
        let frames = n / GRID_HOP + 1;
        let mut f0 = Vec::with_capacity(frames);
        self.confidence.clear();
        let count = self.stream.frames.len();
        for j in 0..frames {
            let (hz, conf) = if count == 0 {
                (0.0, 0.0)
            } else {
                let k = self.frame_for(start + (j * GRID_HOP) as u64) - self.stream.first as f64;
                if k < 0.0 {
                    (0.0, 0.0)
                } else if k >= (count - 1) as f64 {
                    self.stream.frames[count - 1]
                } else {
                    let i0 = k.floor() as usize;
                    let w = (k - i0 as f64) as f32;
                    let (a, b) = (self.stream.frames[i0], self.stream.frames[i0 + 1]);
                    let c = a.1 + (b.1 - a.1) * w;
                    let v = match (a.0 > 0.0, b.0 > 0.0) {
                        (true, true) => (a.0.log2() + (b.0.log2() - a.0.log2()) * w).exp2(),
                        _ => {
                            if w < 0.5 {
                                a.0
                            } else {
                                b.0
                            }
                        }
                    };
                    (v, c)
                }
            };
            f0.push(hz);
            self.confidence.push(conf);
        }
        self.post.process(&mut f0, &self.post_cfg);
        Ok(f0)
    }

    /// The model's state carries the past; the newest frames wait for the delay plus a chunk.
    fn receptive_frames(&self) -> Option<(usize, usize)> {
        Some((2, (self.delay + self.io.chunk).div_ceil(GRID_HOP) + 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};

    /// A stand-in streaming model with PESTO's I/O: `cache` is a 4-slot delay line of chunk means,
    /// `prediction = 200 + 1000 * cache[0]` (the mean of the chunk fed four chunks ago), confidence 1.
    /// So a frame describes the chunk fed four chunks before it: delay = 5 * 160 - 80 = 720 samples.
    fn fake_model() -> Vec<u8> {
        let mut g = Graph::new();
        g.input("audio", FLOAT, &[Dim::Fixed(1), Dim::Fixed(160)])
            .input("cache", FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)])
            .init_i64("s0", &[1], &[0])
            .init_i64("s1", &[1], &[1])
            .init_i64("s4", &[1], &[4])
            .init_i64("ax", &[1], &[1])
            .init_f32("k1000", &[1], &[1000.0])
            .init_f32("k200", &[1], &[200.0])
            .init_f32("k0", &[1], &[0.0])
            .init_f32("k1", &[1], &[1.0])
            .node(
                "ReduceMean",
                &["audio"],
                &["mean"],
                &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)],
            )
            .node("Slice", &["cache", "s1", "s4", "ax"], &["rest"], &[])
            .node("Concat", &["rest", "mean"], &["cache_out"], &[Attr::Int("axis", 1)])
            .node("Slice", &["cache", "s0", "s1", "ax"], &["oldest"], &[])
            .node("Mul", &["oldest", "k1000"], &["scaled"], &[])
            .node("Add", &["scaled", "k200"], &["prediction"], &[])
            .node("Mul", &["oldest", "k0"], &["zero"], &[])
            .node("Add", &["zero", "k1"], &["confidence"], &[])
            .output("prediction", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1)])
            .output("confidence", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1)])
            .output("cache_out", FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)]);
        g.model_bytes(17)
    }

    /// A stand-in with a fixed `chunk` of `chunk / 160` frames (10 ms each): prediction = 1000 +
    /// 1000 * |frame mean| Hz (never near a 220 Hz probe), confidence 1, the cache passed through.
    fn fake_chunked_model(chunk: usize) -> Vec<u8> {
        let mut g = Graph::new();
        g.input("audio", FLOAT, &[Dim::Fixed(1), Dim::Fixed(chunk as u64)])
            .input("cache", FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)])
            .init_i64("shape", &[3], &[1, -1, 160])
            .init_f32("k1000", &[1], &[1000.0])
            .init_f32("base", &[1], &[1000.0])
            .init_f32("k0", &[1], &[0.0])
            .init_f32("k1", &[1], &[1.0])
            .node("Reshape", &["audio", "shape"], &["frames"], &[])
            .node(
                "ReduceMean",
                &["frames"],
                &["mean"],
                &[Attr::Ints("axes", vec![2]), Attr::Int("keepdims", 0)],
            )
            .node("Abs", &["mean"], &["level"], &[])
            .node("Mul", &["level", "k1000"], &["scaled"], &[])
            .node("Add", &["scaled", "base"], &["prediction"], &[])
            .node("Mul", &["level", "k0"], &["zero"], &[])
            .node("Add", &["zero", "k1"], &["confidence"], &[])
            .node("Identity", &["cache"], &["cache_out"], &[])
            .output("prediction", FLOAT, &[Dim::Fixed(1), Dim::Fixed((chunk / 160) as u64)])
            .output("confidence", FLOAT, &[Dim::Fixed(1), Dim::Fixed((chunk / 160) as u64)])
            .output("cache_out", FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)]);
        g.model_bytes(17)
    }

    /// An export with a long fixed chunk is refused with the export hint (it used to pass the hop
    /// checks and panic in the load probe: 32 000 samples, 200 frames, no probe chunk at all); a
    /// chunk the probe can run but that is not PESTO is an error, never a panic.
    #[test]
    fn long_chunks_are_refused_and_the_probe_never_panics() {
        let load = |chunk: usize| {
            let bytes = fake_chunked_model(chunk);
            let (session, outcome) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).unwrap();
            PestoPitchExtractor::from_session(session, outcome, "fake", PestoConfig::default())
        };
        let err = load(32_000).err().expect("a 32 000-sample chunk is refused");
        assert!(err.contains("--chunk_size 160"), "{err}");
        assert!(err.contains("32000"), "{err}");
        // 1 600 samples (10 frames per chunk): the probe runs; the stand-in reads about 1 kHz on the
        // 220 Hz tone, so the load is refused with the probe's message.
        let err = load(1_600).err().expect("the stand-in is not PESTO");
        assert!(err.contains("probe"), "{err}");
    }

    fn fake(delay: usize) -> PestoPitchExtractor {
        let bytes = fake_model();
        let (session, outcome) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).unwrap();
        let cfg = PestoConfig {
            confidence_threshold: 0.5,
            units: Some(PestoUnits::Hz),
            delay_samples: Some(delay),
        };
        let mut p = PestoPitchExtractor::from_session(session, outcome, "fake", cfg).unwrap();
        // The stand-in is not a pitch tracker: keep its contour as it is.
        p.post_cfg = F0PostConfig::limits_only(super::super::f0_post::F0Limits {
            min_hz: 1.0,
            max_hz: 100_000.0,
        });
        p
    }

    /// Chunk `c` of the test signal averages `c * 0.001` (so the fake reads 200 + c Hz for it), with
    /// a zero-mean ramp inside each chunk so consecutive slices are recognisable.
    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i / 160) as f32 * 0.001 + 1e-5 * ((i % 160) as f32 - 79.5))
            .collect()
    }

    #[test]
    fn io_and_hop_are_read_from_the_graph() {
        let p = fake(720);
        assert_eq!(p.io.chunk, 160);
        assert_eq!(p.io.cache_size, 4);
        assert_eq!(p.hop, 160);
        assert_eq!(p.delay_samples(), 720);
        assert_eq!(p.units(), PestoUnits::Hz);
        assert_eq!(p.receptive_frames(), Some((2, (720 + 160usize).div_ceil(160) + 2)));
    }

    /// Overlapping slices (as the F0 cache hands them out) are fed only once, and each grid frame
    /// reads the stream frame describing its time. With the delay set to 800 samples (the chunk
    /// start four chunks back), grid sample `160 j` reads stream frame `j + 4` exactly: 200 + j.
    #[test]
    fn streams_overlapping_slices_with_the_delay_applied() {
        let x = signal(16_000);
        let mut p = fake(800);
        let mut fed_before = 0u64;
        let (lead, block) = (1_600usize, 640usize);
        let mut end = lead + block;
        while end <= 16_000 {
            let slice = &x[end - lead - block..end];
            let f0 = p.f0(slice).unwrap();
            assert_eq!(f0.len(), slice.len() / 160 + 1);
            // Only the new block went through the model (after the first call).
            if fed_before > 0 {
                assert_eq!(p.stream.received - fed_before, block as u64);
            }
            fed_before = p.stream.received;
            let start = end - lead - block;
            // Grid frames old enough to have their stream frame: chunk (start + 160 j) / 160.
            let newest = (p.stream.first + p.stream.frames.len() as u64 - 1) as usize;
            for (j, &v) in f0.iter().enumerate() {
                let a = start + 160 * j;
                let k = (a + 800) / 160 - 1; // stream frame describing `a`
                if k <= newest {
                    assert!((v - (200.0 + (a / 160) as f32)).abs() < 1e-3, "end {end} j {j}: {v}");
                }
            }
            // The newest grid frames hold the newest stream frame.
            let last_known = 200.0 + (newest - 4) as f32;
            assert!((f0[f0.len() - 1] - last_known).abs() < 1e-3, "{} vs {last_known}", f0[f0.len() - 1]);
            end += block;
        }
    }

    #[test]
    fn a_gap_resets_the_stream() {
        let x = signal(20_000);
        let mut p = fake(720);
        p.f0(&x[0..3_200]).unwrap();
        p.f0(&x[640..3_840]).unwrap();
        assert_eq!(p.stream.received, 3_840);
        // A slice that does not overlap: the state restarts from it.
        p.f0(&x[12_000..15_200]).unwrap();
        assert_eq!(p.stream.received, 3_200);
        assert_eq!(p.stream.first, 0);
    }

    #[test]
    fn semitone_predictions_convert_to_hz() {
        let mut p = fake(0);
        p.units = PestoUnits::Semitones;
        assert!((p.to_hz(69.0, 0.9) - 440.0).abs() < 1e-3);
        assert!((p.to_hz(57.0, 0.9) - 220.0).abs() < 1e-2);
        assert_eq!(p.to_hz(57.0, 0.1), 0.0);
        assert_eq!(p.to_hz(f32::NAN, 0.9), 0.0);
    }
}
