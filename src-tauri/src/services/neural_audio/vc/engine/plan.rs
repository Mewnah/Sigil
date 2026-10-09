//! Fixed-shape plans (P3-C1): every tick feeds exactly `block_16k` new samples and a fixed context,
//! so every session runs one shape (CUDA graphs and IOBinding need that, P3-C2). A preset change
//! (block, context) builds a new plan and the GPU layer recaptures.
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`ShapePlan`] fields and [`ShapePlan::new`]`(streaming, preset, model_out_sr, partial_decode)`.
//!
//! # Geometry (P3-C1)
//! All lengths are whole 10 ms frames (160 samples at 16 kHz):
//! - **block**: the preset's `block_ms` rounded up to a whole number of **20 ms** (an even frame
//!   count), so the context always starts on the content encoder's 320-sample grid in absolute
//!   time: two ticks that share audio compute its content features with the same framing, which
//!   keeps their overlap identical (a 10 ms-odd block would shift the framing every other tick).
//! - **return**: the frames each tick returns, `ceil(want / 160)` with `want` = block + the
//!   harness's SOLA tail (see [`ShapePlan::for_stream`]).
//! - **flowdec window** `F`: partial decode = return + `flow_margin` + `dec_margin` (the template's
//!   measured margins, ADR 0004); legacy (partial decode off) = the whole context.
//! - **context** `T` (the content encoder / `enc` length): at least the user's extra context plus
//!   the block, and at least the flowdec window (the window's margin frames need real `m_p` /
//!   `logs_p`), rounded up to even.
//!
//! # Added by P3-C1
//! [`ShapePlan::for_stream`], [`DEFAULT_SOLA_TAIL_16K`], and the fields after `f0_margin_frames`.
//!
//! # Frozen by P3B-S0
//! [`sola_tail_16k`]`(crossfade_ms, out_sr)`: the harness's SOLA tail at a crossfade, delegating to
//! `stream::sola_tail_in_16k` (the formula lives there; [`DEFAULT_SOLA_TAIL_16K`] equals it at 20 ms).
//!
//! # Content window (P3B-D)
//! The content encoder's window is separate from the `enc` window: by default it *is* the context
//! (`context_16k`), and with a content context (`SigilRvcRequest::content_context_16k`, the
//! `content_context_ms` param) it reaches further back, to [`ShapePlan::content_window_16k`]
//! samples ending at the newest input, so the encoder sees real left context (RVC trains on
//! ~3.7 s slices; the `enc` window is ~0.44 s) while `enc`, the flowdec window, F0 and the stream
//! harness keep their shapes. The extra samples are a whole number of the encoder's 320-sample
//! frames, so its frames stay on the stream's absolute grid: 50 Hz frame `k` of the window is 50 Hz
//! frame `k - content_offset_frames50` of the context ([`ShapePlan::content_offset_frames50`]).

use crate::services::neural_audio::vc::import::layout::Streaming;

use super::f0_cache::{DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES};
use super::presets::EnginePreset;

/// 16 kHz samples per 10 ms frame.
pub const HOP_16K: usize = 160;
/// Longest block (plan "VC best engine": up to 200 ms per tick).
pub const MAX_BLOCK_FRAMES: usize = 20;
/// Longest context (4 s of audio, as far as competitors' largest "extra" settings go).
pub const MAX_CONTEXT_FRAMES: usize = 400;
/// SOLA tail assumed before the stream harness reports the real one (its default 20 ms crossfade,
/// the search window and slack: 481 samples at 40 kHz, 480 at 48 kHz, rounded up to 3 frames).
pub const DEFAULT_SOLA_TAIL_16K: usize = 481;
/// Longest lookahead (latency the engine may trade for cleaner block joins).
pub const MAX_LOOKAHEAD_FRAMES: usize = 8;
/// The content encoder's frame hop at 16 kHz (`EmbedderKind::frontend`).
pub const CONTENT_HOP_16K: usize = 320;
/// Longest content window (the `content_context_ms` param goes to 3 s; the window is that or the
/// context, whichever is longer, rounded to the encoder's frames).
pub const MAX_CONTENT_WINDOW_16K: usize = MAX_CONTEXT_FRAMES * HOP_16K;

/// The stream harness's SOLA tail (16 kHz samples) at a `crossfade_ms` crossfade for a template
/// that outputs `out_sr` Hz (P3B-S0; `stream::sola_tail_in_16k`).
pub fn sola_tail_16k(crossfade_ms: f64, out_sr: u32) -> usize {
    crate::services::neural_audio::vc::stream::sola_tail_in_16k(crossfade_ms, out_sr)
}

/// Shapes of one tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapePlan {
    /// New 16 kHz samples per tick (a multiple of the 160-sample F0 hop).
    pub block_16k: usize,
    /// 16 kHz samples fed to the content encoder (context + block, aligned to its front end).
    pub context_16k: usize,
    /// `T` of `enc` (10 ms frames over the context).
    pub enc_frames: usize,
    /// Frames returned per tick (`block_16k / 160`).
    pub return_frames: usize,
    /// `F` of `flowdec`: `return_frames + flow_margin + dec_margin` (the whole context window when
    /// partial decode is off).
    pub flow_frames: usize,
    /// Output samples per frame (`sr / 100`).
    pub upp: usize,
    /// F0 frames re-extracted before the new block (the F0 cache margin).
    pub f0_margin_frames: usize,
    // ---- P3-C1 ----
    /// Samples each tick returns (`ConversionCtx::new_frames`: block + SOLA tail).
    pub want_16k: usize,
    /// First context frame of the flowdec window (`enc_frames - flow_frames`).
    pub window_start: usize,
    /// Frames `flowdec` decodes and returns (`flow_frames - flow_margin - dec_margin`; the return
    /// frames with partial decode, more without).
    pub decode_frames: usize,
    /// `flow_margin + dec_margin` of the template.
    pub margin_frames: usize,
    /// Lead-in frames of the F0 cache (see `f0_cache`).
    pub f0_lead_frames: usize,
    /// Whether this plan decodes the return window only.
    pub partial_decode: bool,
    /// Frames of lookahead: the returned audio ends this many frames before the newest input, so
    /// every returned sample was decoded with that much future context (less right-edge error in
    /// the SOLA overlap, at `lookahead × 10 ms` of latency). 0 = time-aligned to the newest input.
    pub lookahead_frames: usize,
}

impl ShapePlan {
    /// The plan of `preset` for a template with `streaming` margins; `partial_decode` selects the
    /// flowdec window.
    pub fn new(streaming: &Streaming, preset: &EnginePreset, model_out_sr: u32, partial_decode: bool) -> Result<ShapePlan, String> {
        let block_16k = block_frames(preset.block_ms)? * HOP_16K;
        let extra_16k = ms_to_16k(preset.extra_context_ms);
        Self::for_stream(
            streaming,
            model_out_sr,
            partial_decode,
            block_16k,
            block_16k + DEFAULT_SOLA_TAIL_16K,
            extra_16k,
        )
    }

    /// The plan for the stream harness's real shape: `block_16k` new samples per tick (a multiple of
    /// 320), `want_16k` returned (block + SOLA tail) and `extra_16k` of requested extra context.
    pub fn for_stream(
        streaming: &Streaming,
        model_out_sr: u32,
        partial_decode: bool,
        block_16k: usize,
        want_16k: usize,
        extra_16k: usize,
    ) -> Result<ShapePlan, String> {
        Self::for_stream_with_lookahead(streaming, model_out_sr, partial_decode, block_16k, want_16k, extra_16k, 0)
    }

    /// [`for_stream`](Self::for_stream) with `lookahead_frames` of lookahead (see the field; at
    /// most [`MAX_LOOKAHEAD_FRAMES`]): the return window grows by the lookahead, whose newest frames
    /// are decoded but not returned.
    pub fn for_stream_with_lookahead(
        streaming: &Streaming,
        model_out_sr: u32,
        partial_decode: bool,
        block_16k: usize,
        want_16k: usize,
        extra_16k: usize,
        lookahead_frames: usize,
    ) -> Result<ShapePlan, String> {
        if lookahead_frames > MAX_LOOKAHEAD_FRAMES {
            return Err(format!(
                "lookahead of {} ms is longer than {} ms",
                lookahead_frames * 10,
                MAX_LOOKAHEAD_FRAMES * 10
            ));
        }
        if block_16k == 0 || !block_16k.is_multiple_of(2 * HOP_16K) {
            return Err(format!("block of {block_16k} samples is not a whole number of 20 ms"));
        }
        let block = block_16k / HOP_16K;
        if block > MAX_BLOCK_FRAMES {
            return Err(format!("block of {} ms is longer than {} ms", block * 10, MAX_BLOCK_FRAMES * 10));
        }
        if want_16k < block_16k {
            return Err(format!("the returned span ({want_16k} samples) is shorter than the block ({block_16k})"));
        }
        if model_out_sr == 0 || !model_out_sr.is_multiple_of(100) {
            return Err(format!("output rate {model_out_sr} Hz is not a whole number of samples per 10 ms"));
        }
        let upp = (model_out_sr / 100) as usize;
        let margin = (streaming.flow_margin + streaming.dec_margin) as usize;
        let ret = want_16k.div_ceil(HOP_16K) + lookahead_frames;
        let extra_frames = extra_16k.div_ceil(HOP_16K);
        // The context covers the user's extra context and the block, and the flowdec window.
        let mut t = (extra_frames + block).max(ret + margin).max(ret);
        t = t.div_ceil(2) * 2;
        if t > MAX_CONTEXT_FRAMES {
            return Err(format!("context of {} ms is longer than {} ms", t * 10, MAX_CONTEXT_FRAMES * 10));
        }
        let flow_frames = if partial_decode { ret + margin } else { t };
        let decode_frames = flow_frames - margin;
        Ok(ShapePlan {
            block_16k,
            context_16k: t * HOP_16K,
            enc_frames: t,
            return_frames: ret,
            flow_frames,
            upp,
            f0_margin_frames: DEFAULT_MARGIN_FRAMES.min(t),
            want_16k,
            window_start: t - flow_frames,
            decode_frames,
            margin_frames: margin,
            f0_lead_frames: DEFAULT_LEAD_FRAMES,
            partial_decode,
            lookahead_frames,
        })
    }

    /// Model-rate samples a tick returns (`round(want · upp / 160)`, the harness's rounding).
    pub fn out_samples(&self) -> usize {
        (self.want_16k * self.upp * 100 + 8_000) / 16_000
    }

    /// Samples the content encoder sees per tick for a content context of `content_context_16k`
    /// (`None` or anything up to the context: the context itself, as before P3B-D). A longer
    /// context extends the window to the left by whole 320-sample encoder frames, capped at
    /// [`MAX_CONTENT_WINDOW_16K`].
    pub fn content_window_16k(&self, content_context_16k: Option<usize>) -> usize {
        let c = self.context_16k;
        match content_context_16k {
            Some(want) if want > c => {
                let extra = (want.min(MAX_CONTENT_WINDOW_16K).saturating_sub(c)).div_ceil(CONTENT_HOP_16K) * CONTENT_HOP_16K;
                c + extra
            }
            _ => c,
        }
    }

    /// 50 Hz content frames the window of `content_window_16k` adds before the context (the offset
    /// of the context's first content frame in the window's features).
    pub fn content_offset_frames50(&self, content_window_16k: usize) -> usize {
        content_window_16k.saturating_sub(self.context_16k) / CONTENT_HOP_16K
    }
}

fn ms_to_16k(ms: f64) -> usize {
    if ms.is_finite() && ms > 0.0 {
        (ms * 16.0).round() as usize
    } else {
        0
    }
}

/// Block frames for `block_ms`: rounded up to 20 ms (at least 20 ms).
pub fn block_frames(block_ms: f64) -> Result<usize, String> {
    if !block_ms.is_finite() || block_ms < 0.0 {
        return Err(format!("block of {block_ms} ms"));
    }
    let frames = (block_ms / 10.0 - 1e-9).ceil().max(1.0) as usize;
    let frames = frames.div_ceil(2) * 2;
    if frames > MAX_BLOCK_FRAMES {
        return Err(format!("block of {block_ms} ms is longer than {} ms", MAX_BLOCK_FRAMES * 10));
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::engine::options::PrecisionMode;
    use crate::services::neural_audio::vc::engine::presets::engine_preset;
    use crate::services::neural_audio::vc::rvc::RvcPresetProfile;

    /// The default tail is the harness's formula at its default 20 ms crossfade (P3B-S0).
    #[test]
    fn default_sola_tail_is_the_harness_formula() {
        for sr in [32_000, 40_000, 48_000] {
            assert_eq!(sola_tail_16k(20.0, sr), DEFAULT_SOLA_TAIL_16K, "{sr}");
        }
    }

    fn streaming(flow: u32, dec: u32) -> Streaming {
        Streaming {
            flow_margin: flow,
            dec_margin: dec,
            noise_scale: 0.66666,
            knn_k: 8,
            knn_coarse_k: 64,
            min_index_rows: 64,
        }
    }

    #[test]
    fn blocks_round_up_to_20_ms() {
        assert_eq!(block_frames(0.0).unwrap(), 2);
        assert_eq!(block_frames(10.0).unwrap(), 2);
        assert_eq!(block_frames(10.67).unwrap(), 2);
        assert_eq!(block_frames(20.0).unwrap(), 2);
        assert_eq!(block_frames(30.0).unwrap(), 4);
        assert_eq!(block_frames(40.0).unwrap(), 4);
        assert_eq!(block_frames(200.0).unwrap(), 20);
        assert!(block_frames(210.0).is_err());
        assert!(block_frames(f64::NAN).is_err());
    }

    /// Reference config (v2 40k, 40 ms block, 20 ms crossfade): partial decode returns the block
    /// plus the SOLA tail and decodes only that; the context covers the flowdec window.
    #[test]
    fn partial_decode_window_is_the_return_plus_margins() {
        let s = streaming(24, 11);
        let preset = engine_preset(RvcPresetProfile::LowLatency, 40.0, 512, 48_000, 96.0, PrecisionMode::Auto);
        let p = ShapePlan::new(&s, &preset, 40_000, true).unwrap();
        assert_eq!(p.block_16k, 640);
        assert_eq!(p.upp, 400);
        assert_eq!(p.return_frames, 8); // 640 + 481 samples
        assert_eq!(p.flow_frames, 8 + 35);
        assert_eq!(p.decode_frames, 8);
        assert_eq!(p.enc_frames % 2, 0);
        assert!(p.enc_frames >= p.flow_frames);
        assert_eq!(p.enc_frames, 44); // max(ceil(1536/160)=10 + 4, 43) rounded to even
        assert_eq!(p.window_start, p.enc_frames - p.flow_frames);
        assert_eq!(p.context_16k, p.enc_frames * 160);
        assert_eq!(p.out_samples(), 2_803); // round(1121 * 40000 / 16000)
                                            // Legacy: flowdec over the whole context, returning the same tail.
        let full = ShapePlan::new(&s, &preset, 40_000, false).unwrap();
        assert_eq!(full.enc_frames, p.enc_frames);
        assert_eq!(full.flow_frames, full.enc_frames);
        assert_eq!(full.window_start, 0);
        assert_eq!(full.decode_frames, full.enc_frames - 35);
        assert!(full.decode_frames >= full.return_frames);
    }

    /// The content window is the context unless a longer content context asks for more; the extra
    /// is whole encoder frames, so the window stays on the stream's 320-sample grid, and the `enc`
    /// window, the flowdec window and the context the harness feeds are unchanged.
    #[test]
    fn content_window_extends_left_in_whole_encoder_frames() {
        let s = streaming(24, 11);
        let preset = engine_preset(RvcPresetProfile::LowLatency, 40.0, 512, 48_000, 64.0, PrecisionMode::Auto);
        let p = ShapePlan::new(&s, &preset, 40_000, true).unwrap();
        assert_eq!(p.context_16k, 7_040);
        for none in [None, Some(0), Some(1_000), Some(7_040)] {
            assert_eq!(p.content_window_16k(none), p.context_16k, "{none:?}");
            assert_eq!(p.content_offset_frames50(p.content_window_16k(none)), 0);
        }
        // 1 s: 8960 more samples = 28 frames of 320.
        let w = p.content_window_16k(Some(16_000));
        assert_eq!(w, 16_000);
        assert_eq!((w - p.context_16k) % CONTENT_HOP_16K, 0);
        assert_eq!(p.content_offset_frames50(w), 28);
        // Not a whole frame more: rounded up to one.
        let w = p.content_window_16k(Some(7_041));
        assert_eq!(w, 7_040 + 320);
        assert_eq!(p.content_offset_frames50(w), 1);
        // Capped.
        assert!(p.content_window_16k(Some(10_000_000)) <= MAX_CONTENT_WINDOW_16K + CONTENT_HOP_16K);
    }

    /// Extra context grows only the encoders' shapes; the flowdec window stays the same.
    #[test]
    fn extra_context_does_not_grow_the_decode_window() {
        let s = streaming(24, 11);
        let mut windows = Vec::new();
        for extra in [64.0, 192.0, 512.0, 1000.0] {
            let preset = engine_preset(RvcPresetProfile::Quality, 40.0, 0, 48_000, extra, PrecisionMode::Auto);
            let p = ShapePlan::new(&s, &preset, 48_000, true).unwrap();
            assert!(p.context_16k >= (extra * 16.0) as usize + 640);
            windows.push((p.flow_frames, p.decode_frames));
        }
        assert!(windows.windows(2).all(|w| w[0] == w[1]), "{windows:?}");
        // The harness's real SOLA tail replaces the default.
        let p = ShapePlan::for_stream(&s, 40_000, true, 640, 640 + 1_201, 1_536).unwrap();
        assert_eq!(p.return_frames, 12);
        assert_eq!(p.flow_frames, 47);
        assert!(
            ShapePlan::for_stream(&s, 40_000, true, 480, 900, 0).is_err(),
            "30 ms is not a 20 ms multiple"
        );
        assert!(
            ShapePlan::for_stream(&s, 44_050, true, 640, 1_000, 0).is_err(),
            "not a whole number of samples per frame"
        );
        assert!(ShapePlan::for_stream(&s, 40_000, true, 640, 100, 0).is_err());
        assert!(ShapePlan::for_stream(&s, 40_000, true, 640, 1_000, 16_000 * 5).is_err());
    }
}
