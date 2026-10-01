//! The offline ceiling (plan "Benchmark harness": the same model converting the whole utterance
//! in-process), against which streaming fidelity is scored (T5).
//!
//! The streaming run records the exact 16 kHz stream its model received ([`StreamRecorder`]: every
//! `ConversionCtx` at its absolute `stream_pos_16k`, so the priming silence and the resampler's
//! delay are included), and the ceiling converts that same stream at the same absolute positions:
//! - **Dynamic-shape models** (bring-your-own ONNX `RvcModel`, the stub): the whole stream in one
//!   `convert` call (segments of 20 s with 2 s of context beyond 30 s).
//! - **Fixed-shape models** (the Sigil Voice Engine): a second instance of the voice with long
//!   shapes (200 ms blocks, 80 ms lookahead, 500 ms of extra context by default: `CeilingSpec`
//!   explains why not the longest context) converting the stream block after block with no
//!   crossfade. Its time-anchored noise is keyed to the same absolute frames as the stream's, so
//!   the two renditions differ only by what streaming changes (block, context, joins). The first
//!   block starts where its whole context lies inside the stream; the quality run leads the clip
//!   with silence (`lead_s`) so every sample of the clip is covered.
//!
//! Stream and ceiling are placed against each other from this timeline (`fidelity.rs`): the
//! ceiling's first absolute position, where the input entered the recorded stream, and the
//! probed harness latency.

use std::time::Instant;

use serde::Serialize;

use crate::services::neural_audio::vc::traits::ConversionCtx;
use crate::services::neural_audio::VoiceConversionModel;

use super::engine::Engine;
use super::rvc_build;
use super::spec::{CeilingSpec, StreamSpec};

/// One model call as the stream recorder saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CallSpan {
    /// Absolute 16 kHz position one past the call's newest sample.
    pub end_16k: u64,
    /// Samples the call returned (block + SOLA tail).
    pub returned_16k: usize,
}

/// The 16 kHz stream a model received, by absolute position (warm-ups are not recorded: the
/// harness warms through `warmup_at`).
#[derive(Debug, Default)]
pub struct StreamRecorder {
    start: Option<u64>,
    data: Vec<f32>,
    calls: Vec<CallSpan>,
}

impl StreamRecorder {
    pub fn record(&mut self, ctx: &ConversionCtx<'_>) {
        let pos = ctx.stream_pos_16k;
        let len = ctx.mono_16k.len();
        let start = match self.start {
            Some(s) if pos < s => {
                // Earlier than anything recorded (not expected from the harness): grow the front.
                let grow = (s - pos) as usize;
                self.data.splice(0..0, std::iter::repeat_n(0.0, grow));
                self.start = Some(pos);
                pos
            }
            Some(s) => s,
            None => {
                self.start = Some(pos);
                pos
            }
        };
        let off = (pos - start) as usize;
        if self.data.len() < off + len {
            self.data.resize(off + len, 0.0);
        }
        self.data[off..off + len].copy_from_slice(ctx.mono_16k);
        self.calls.push(CallSpan {
            end_16k: pos + len as u64,
            returned_16k: ctx.new_frames,
        });
    }

    /// `(absolute position of data[0], data)`.
    pub fn stream(&self) -> (u64, &[f32]) {
        (self.start.unwrap_or(0), &self.data)
    }

    pub fn calls(&self) -> &[CallSpan] {
        &self.calls
    }
}

/// The ceiling's audio.
#[derive(Debug)]
pub struct CeilingOutput {
    /// At the model's output rate; sample `i` is the conversion of absolute 16 kHz position
    /// `start_16k + i · 16000 / rate`.
    pub audio: Vec<f32>,
    pub rate: u32,
    pub start_16k: u64,
    /// How it was made.
    pub method: String,
    pub compute_ms: f64,
    pub notes: Vec<String>,
}

/// Samples of `data` (starting at absolute `start`) over absolute `[from, from + len)`, zero
/// outside.
fn gather(data: &[f32], start: u64, from: u64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    for (i, o) in out.iter_mut().enumerate() {
        let p = from + i as u64;
        if p >= start {
            if let Some(&v) = data.get((p - start) as usize) {
                *o = v;
            }
        }
    }
    out
}

fn ctx<'a>(mono: &'a [f32], new_frames: usize, pos: u64, stream: &StreamSpec) -> ConversionCtx<'a> {
    ConversionCtx {
        mono_16k: mono,
        new_frames,
        pitch_shift_semitones: stream.pitch_semitones,
        index_rate: stream.index_rate,
        protect: stream.protect,
        speaker_id: stream.speaker_id,
        stream_pos_16k: pos,
    }
}

/// Whole-utterance segments: one call up to 30 s, else 20 s pieces with 2 s of context.
const WHOLE_MAX_16K: usize = 30 * 16_000;
const SEGMENT_16K: usize = 20 * 16_000;
const SEGMENT_CONTEXT_16K: usize = 2 * 16_000;

/// Convert the recorded stream offline (see the module docs).
pub fn ceiling(engine: &Engine, stream: &StreamSpec, rec: &StreamRecorder, cfg: &CeilingSpec) -> Result<CeilingOutput, String> {
    let (start, data) = rec.stream();
    if data.is_empty() {
        return Err("the ceiling has no input: the stream recorder saw no model call".into());
    }
    let started = Instant::now();
    let fixed = engine.fixed_block().is_some();
    match (&engine.rvc, fixed) {
        (Some(plan), true) => {
            let mut ce = rvc_build::ceiling_engine(plan, cfg)?;
            ce.set_voice_controls(&stream.voice_controls());
            let p = *ce.plan();
            let block = p.block_16k;
            let la = p.lookahead_frames * 160;
            let context = p.context_16k;
            let rate = ce.output_sample_rate();
            let per_block = (block as u64 * u64::from(rate) / 16_000) as usize;
            let end = start + data.len() as u64;
            // First block whose whole context is inside the stream's absolute timeline.
            let min_k0 = (context as u64).saturating_sub((block + la) as u64);
            let first = start.max(min_k0);
            let first = first.div_ceil(160) * 160;
            let mut out = Vec::with_capacity(((end.saturating_sub(first)) as usize / block + 1) * per_block);
            let mut k0 = first;
            while k0 < end {
                let ctx_start = k0 + (block + la) as u64 - context as u64;
                let window = gather(data, start, ctx_start, context);
                let got = ce.convert(&ctx(&window, block, ctx_start, stream))?;
                if got.len() != per_block {
                    return Err(format!("ceiling: the engine returned {} samples for a {per_block}-sample block", got.len()));
                }
                out.extend_from_slice(&got);
                k0 += block as u64;
            }
            let mut notes = vec![format!(
                "ceiling: {} ms blocks with {} ms of context and {} ms of lookahead, back to back (no crossfade){}",
                block / 16,
                context / 16,
                la / 16,
                if cfg.exact { ", full decode" } else { ", the run's levers" }
            )];
            if first > start {
                notes.push(format!(
                    "ceiling: the first {:.2} s of the stream precede the ceiling's first block",
                    (first - start) as f64 / 16_000.0
                ));
            }
            Ok(CeilingOutput {
                audio: out,
                rate,
                start_16k: first,
                method: "fixed-shape".into(),
                compute_ms: started.elapsed().as_secs_f64() * 1000.0,
                notes,
            })
        }
        _ => {
            let rate = engine.info.output_rate;
            let out = engine.with_model(|m| -> Result<Vec<f32>, String> {
                if data.len() <= WHOLE_MAX_16K {
                    return m.convert(&ctx(data, data.len(), start, stream));
                }
                let mut out = Vec::new();
                let mut s = 0usize;
                while s < data.len() {
                    let e = (s + SEGMENT_16K).min(data.len());
                    let c0 = s.saturating_sub(SEGMENT_CONTEXT_16K);
                    out.extend(m.convert(&ctx(&data[c0..e], e - s, start + c0 as u64, stream))?);
                    s = e;
                }
                Ok(out)
            })?;
            let method = if data.len() <= WHOLE_MAX_16K { "whole utterance" } else { "20 s segments" };
            Ok(CeilingOutput {
                audio: out,
                rate,
                start_16k: start,
                method: method.into(),
                compute_ms: started.elapsed().as_secs_f64() * 1000.0,
                notes: vec![format!("ceiling: {method}, one call per piece")],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_rebuilds_the_stream_by_absolute_position() {
        let mut r = StreamRecorder::default();
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let c = |data: &[f32], pos: u64| -> Vec<f32> {
            // An overlapping window of the same signal: positions pos..pos+len carry value = pos.
            let _ = data;
            (pos..pos + 60).map(|p| p as f32).collect()
        };
        let w1 = c(&a, 1_000);
        let w2 = c(&a, 1_040);
        for (w, pos) in [(&w1, 1_000u64), (&w2, 1_040)] {
            r.record(&ConversionCtx {
                mono_16k: w,
                new_frames: 40,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos,
            });
        }
        let (start, data) = r.stream();
        assert_eq!(start, 1_000);
        assert_eq!(data.len(), 100);
        assert!(data.iter().enumerate().all(|(i, &v)| v == (1_000 + i) as f32));
        assert_eq!(r.calls(), &[CallSpan { end_16k: 1_060, returned_16k: 40 }, CallSpan { end_16k: 1_100, returned_16k: 40 }]);
        // gather: zero outside the recorded span.
        let g = gather(data, start, 990, 20);
        assert_eq!(&g[..10], &[0.0; 10]);
        assert_eq!(g[10], 1_000.0);
    }
}
