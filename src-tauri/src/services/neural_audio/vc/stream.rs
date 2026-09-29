//! Rolling-context + SOLA stream harness for neural VC.
//!
//! Per `process_frames` device block: band-limited resample to the model input rate (16 kHz),
//! append to the rolling context ring, convert the newest block plus a SOLA tail, stitch exactly
//! `block_out` model-rate samples, then band-limited resample to the playback rate through a FIFO.
//!
//! Length accounting (exact integer carries): each block contributes `new_16k * out_sr / 16k`
//! model-rate samples, the output resampler is rate-exact, and every call returns exactly the
//! device frames owed for its input (`in_frames * device_out_sr / device_in_sr`), so after any
//! number of calls the harness has emitted `round(total_in * device_out_sr / device_in_sr)`
//! frames. A small cushion (`prime_margin`) absorbs the per-tick jitter of the FIFO/resamplers;
//! the resamplers' start-up delay is absorbed once by the priming silence.
//!
//! A model error does not disturb that accounting: the failed block contributes silence of its
//! accounted length and the tick's owed frames are still taken from the backlog, so the caller's
//! fallback audio (`last_owed_frames()` frames) slots into exactly the same timeline an error-free
//! run would produce.

use rubato::{calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::traits::{ConversionCtx, VoiceConversionModel};

/// Frames per output-side resampler call, at the model output rate.
const OUT_RESAMPLE_CHUNK: usize = 64;
/// Output backlog cap, in expected ticks; older samples beyond it are dropped.
const MAX_PENDING_TICKS: usize = 4;
/// Score penalty (per unit of distance from the window centre, relative to half the window) that
/// keeps SOLA offsets from drifting to an edge of the search window.
const SOLA_CENTER_BIAS: f32 = 0.05;
const RESAMPLER_SINC_LEN: usize = 128;

/// Crossfade join with normalized-cross-correlation alignment (SOLA).
///
/// Per tick the model is asked for `block + overlap + search` trailing samples. The previous
/// tick's held-back tail nominally lines up with the middle of the `search` window, so the
/// alignment can move either way. The first `overlap` samples of the aligned chunk are crossfaded
/// with that tail, exactly `block` samples are emitted, the next `overlap` samples are held back
/// for the following join, and the remaining margin is discarded (it is re-synthesised next tick).
struct SolaStitcher {
    overlap: usize,
    search: usize,
    prev_tail: Vec<f32>,
}

impl SolaStitcher {
    fn new(overlap: usize) -> Self {
        let overlap = overlap.max(1);
        Self {
            overlap,
            search: overlap / 2,
            prev_tail: Vec::new(),
        }
    }

    /// Alignment search window requested from the model beyond `block + overlap`.
    fn search(&self) -> usize {
        self.search
    }

    /// The last emitted audio was silence: the next chunk fades in from zero.
    fn hold_silence(&mut self) {
        self.prev_tail.clear();
        self.prev_tail.resize(self.overlap, 0.0);
    }

    /// Offset in `0..=window` whose `n` samples best match the held-back tail (normalized
    /// cross-correlation with a mild preference for the nominal rejoin point `center`).
    fn best_offset(&self, chunk: &[f32], n: usize, window: usize, center: usize) -> usize {
        let center = center.min(window);
        let tail = &self.prev_tail[..n];
        let tail_energy: f64 = tail.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if tail_energy <= 1e-10 {
            return center;
        }
        let mut cand_energy: f64 = chunk[..n]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        let half = (window / 2).max(1) as f32;
        let mut best = center;
        let mut best_score = f32::NEG_INFINITY;
        for off in 0..=window {
            if off > 0 {
                let leaving = f64::from(chunk[off - 1]);
                let entering = f64::from(chunk[off + n - 1]);
                cand_energy = (cand_energy - leaving * leaving + entering * entering).max(0.0);
            }
            let ncc = if cand_energy > 1e-12 {
                (f64::from(dot(tail, &chunk[off..off + n])) / (tail_energy * cand_energy).sqrt()) as f32
            } else {
                0.0
            };
            let score = ncc - SOLA_CENTER_BIAS * (off.abs_diff(center) as f32 / half);
            if score > best_score {
                best_score = score;
                best = off;
            }
        }
        best
    }

    /// Stitch one model chunk; always returns exactly `block` samples.
    fn process(&mut self, chunk: &[f32], block: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(block);
        if block == 0 {
            return out;
        }
        if chunk.len() < block {
            // The model produced less than a block: emit it, pad with silence, fade in next time.
            out.extend_from_slice(chunk);
            out.resize(block, 0.0);
            self.hold_silence();
            return out;
        }
        let ov = self.overlap.min(block).min(chunk.len() - block);
        let window = chunk.len() - block - ov;
        // Nominal rejoin point from the full overlap, not the (possibly clamped) `ov`: a block
        // shorter than the overlap widens `window`, and `window / 2` would move the join (and the
        // stream's lag) by a sample or so on such ticks.
        let center = ((chunk.len() - block).saturating_sub(self.overlap) / 2).min(window);
        let n = ov.min(self.prev_tail.len());
        let off = if n >= 8 && window > 0 {
            self.best_offset(chunk, n, window, center)
        } else {
            center
        };
        // `off <= window`, so `w` always holds at least `block + ov` samples.
        let w = &chunk[off..];
        for (i, (&old, &new)) in self.prev_tail[..n].iter().zip(w.iter()).enumerate() {
            let t = (i as f32 + 1.0) / (n as f32 + 1.0);
            out.push(old * (1.0 - t) + new * t);
        }
        out.extend_from_slice(&w[n..block]);
        self.prev_tail.clear();
        self.prev_tail.extend_from_slice(&w[block..block + ov]);
        out
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for ((acc_j, &xj), &yj) in acc.iter_mut().zip(x).zip(y) {
            *acc_j += xj * yj;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        sum += x * y;
    }
    sum
}

/// Stateful band-limited mono resampler fed through a FIFO in fixed `chunk`-frame calls.
///
/// rubato's `SincFixedIn` output is time-aligned with its input; its `output_delay()` frames only
/// arrive later (they are the look-ahead of the filter), which the harness absorbs once at start.
struct StreamResampler {
    inner: Option<SincFixedIn<f32>>,
    chunk: usize,
    fifo: Vec<f32>,
    out_buf: Vec<Vec<f32>>,
}

impl StreamResampler {
    fn new(from_sr: u32, to_sr: u32, chunk: usize) -> Result<Self, String> {
        let chunk = chunk.max(1);
        if from_sr == to_sr || from_sr == 0 || to_sr == 0 {
            return Ok(Self {
                inner: None,
                chunk,
                fifo: Vec::new(),
                out_buf: Vec::new(),
            });
        }
        let params = SincInterpolationParameters {
            sinc_len: RESAMPLER_SINC_LEN,
            f_cutoff: calculate_cutoff(RESAMPLER_SINC_LEN, WindowFunction::BlackmanHarris2),
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = SincFixedIn::<f32>::new(f64::from(to_sr) / f64::from(from_sr), 1.0, params, chunk, 1)
            .map_err(|e| format!("VC resampler {from_sr}->{to_sr}: {e}"))?;
        let out_buf = inner.output_buffer_allocate(true);
        Ok(Self {
            inner: Some(inner),
            chunk,
            fifo: Vec::with_capacity(chunk * 2),
            out_buf,
        })
    }

    /// Queue `input` and append everything produced by complete chunks to `out`.
    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let Some(inner) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return Ok(());
        };
        self.fifo.extend_from_slice(input);
        let mut start = 0;
        while self.fifo.len() - start >= self.chunk {
            let frame = [&self.fifo[start..start + self.chunk]];
            let result = inner.process_into_buffer(&frame, &mut self.out_buf, None);
            start += self.chunk;
            match result {
                Ok((_, produced)) => out.extend_from_slice(&self.out_buf[0][..produced]),
                Err(e) => {
                    self.fifo.drain(..start);
                    return Err(format!("VC resampler: {e}"));
                }
            }
        }
        self.fifo.drain(..start);
        Ok(())
    }
}

fn downmix_to_mono(interleaved: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn mono_to_interleaved(mono: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return mono.to_vec();
    }
    let mut out = Vec::with_capacity(mono.len() * ch);
    for &s in mono {
        for _ in 0..ch {
            out.push(s);
        }
    }
    out
}

pub struct VcStreamHarness {
    model: Box<dyn VoiceConversionModel>,
    model_in_sr: u32,
    model_out_sr: u32,
    ring_16k: Vec<f32>,
    extra_samples: usize,
    /// Fewest ring samples before the model is called (HuBERT receptive field + hop). The ring
    /// starts zero-filled to its steady length, which is never below this, so the check is only
    /// a safety net (normal runs never hit it).
    min_context: usize,
    sola: SolaStitcher,
    /// Model-input samples requested beyond each new block for SOLA (overlap + search + slack).
    sola_tail_in: usize,
    device_in_sr: u32,
    device_out_sr: u32,
    ch_in: usize,
    ch_out: usize,
    pitch_shift: f32,
    index_rate: f32,
    protect: f32,
    speaker_id: i64,
    in_buf_acc: Vec<f32>,
    in_resampler: StreamResampler,
    out_resampler: StreamResampler,
    out_pending: Vec<f32>,
    process_frames: usize,
    /// Exact integer remainders (numerator mod rate) so every stage feeds the next at the exact
    /// rate ratio with no floating-point drift.
    block_out_carry: u64,
    need_frames_carry: u64,
    /// False at start-up and after an underrun: output silence until the cushion is rebuilt.
    primed: bool,
    prime_margin: usize,
    pending_cap: usize,
    /// Device frames (per channel) owed by the most recent `push_and_convert`, set even when it
    /// returned `Err`.
    last_owed_frames: usize,
}

/// Quiet 150 Hz tone (voiced for the pitch extractor, above any silence gate) used for warmup.
fn warmup_tone(len: usize, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate.max(1) as f32;
    (0..len)
        .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 150.0 * i as f32 / sr).sin())
        .collect()
}

impl VcStreamHarness {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut model: Box<dyn VoiceConversionModel>,
        device_in_sr: u32,
        device_out_sr: u32,
        ch_in: usize,
        ch_out: usize,
        extra_samples: usize,
        crossfade_samples: usize,
        process_frames: usize,
    ) -> Result<Self, String> {
        let device_in_sr = device_in_sr.max(1);
        let device_out_sr = device_out_sr.max(1);
        let model_in_sr = model.input_sample_rate().max(1);
        let model_out_sr = model.output_sample_rate().max(1);
        let process_frames = process_frames.max(160);
        let min_context = model.min_context_samples();

        // The crossfade can never span more than one emitted block, so clamp it to the expected
        // block length; a larger overlap would only add latency and ring size.
        let block_expected = ((process_frames as u64 * u64::from(model_out_sr)) / u64::from(device_in_sr)).max(1) as usize;
        let overlap = crossfade_samples.max(64).min(block_expected);
        let sola = SolaStitcher::new(overlap);
        // +2 output samples of slack so rounding never shrinks the search window below `search`.
        let sola_tail_in = div_ceil_u64(
            (sola.overlap + sola.search() + 2) as u64 * u64::from(model_in_sr),
            u64::from(model_out_sr),
        ) as usize;

        // Warm the model at the steady-state input shape (same ring trim and `want` as
        // `convert_block`) so GPU execution providers pick kernels for the shapes live ticks use,
        // before the first spoken tick. The input resampler yields the floor or the ceiling of
        // the exact per-block count, so both ring lengths are warmed when they differ (the model
        // may pad them to different tensor shapes; an identical shape costs one quick run).
        let exact_new = process_frames as u64 * u64::from(model_in_sr);
        let new_hi = div_ceil_u64(exact_new, u64::from(device_in_sr)) as usize;
        let new_lo = (exact_new / u64::from(device_in_sr)) as usize;
        let steady_news: &[usize] = if new_lo == new_hi || new_lo == 0 {
            &[new_hi]
        } else {
            &[new_hi, new_lo]
        };
        for &steady_new in steady_news {
            let steady_want = steady_new + sola_tail_in;
            let steady_ring = extra_samples.saturating_add(steady_new).max(steady_want).max(min_context);
            let tone = warmup_tone(steady_ring, model_in_sr);
            let warm_ctx = ConversionCtx {
                mono_16k: &tone,
                new_frames: steady_want,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
            };
            if let Err(e) = model.warmup_at(&warm_ctx) {
                eprintln!("[VC] warmup at live shape ({steady_ring} samples): {e}");
            }
        }

        // Start the ring as if silence had already been flowing: after the first block is appended
        // and trimmed it holds exactly the steady ring length, so every live call (the first one
        // included) uses a shape warmed above, with full context from the first tick. The zeros
        // are what the model would have heard before capture started. The shortest steady block
        // needs the most history; `ring_prefill` covers it, and longer blocks are trimmed.
        let shortest_new = steady_news.iter().copied().min().unwrap_or(new_hi);
        let ring_prefill = extra_samples
            .max(sola_tail_in)
            .max(min_context.saturating_sub(shortest_new));
        let mut ring_16k = Vec::with_capacity(ring_prefill + new_hi + sola_tail_in + min_context + 8192);
        ring_16k.resize(ring_prefill, 0.0);

        let mut in_resampler = StreamResampler::new(device_in_sr, model_in_sr, process_frames)?;
        // The resampler's first call returns `output_delay()` fewer samples (its look-ahead), which
        // would give the first live block a cold, shorter shape. Run one block of the same silence
        // through it now and discard the output: live blocks then get the steady floor/ceil counts
        // from the first tick. Latency is unchanged (the delay moves from a short first block into
        // the zero history already in the ring).
        let mut discarded = Vec::new();
        in_resampler.push(&vec![0.0; process_frames], &mut discarded)?;
        let out_resampler = StreamResampler::new(model_out_sr, device_out_sr, OUT_RESAMPLE_CHUNK)?;

        let out_per_model = f64::from(device_out_sr) / f64::from(model_out_sr);
        let out_per_16k = f64::from(device_out_sr) / f64::from(model_in_sr);
        // Worst-case per-tick shortfall: FIFO residue (< one resampler chunk) plus +-1 sample of
        // rounding in each counting stage.
        let prime_margin = ((OUT_RESAMPLE_CHUNK + 1) as f64 * out_per_model + 3.0 * out_per_16k).ceil() as usize + 4;
        let need_expected = div_ceil_u64(process_frames as u64 * u64::from(device_out_sr), u64::from(device_in_sr)) as usize;
        let pending_cap = MAX_PENDING_TICKS * need_expected + prime_margin;

        Ok(Self {
            model,
            model_in_sr,
            model_out_sr,
            ring_16k,
            extra_samples,
            min_context,
            sola,
            sola_tail_in,
            device_in_sr,
            device_out_sr,
            ch_in: ch_in.max(1),
            ch_out: ch_out.max(1),
            pitch_shift: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            in_buf_acc: Vec::new(),
            in_resampler,
            out_resampler,
            out_pending: Vec::new(),
            process_frames,
            block_out_carry: 0,
            // Start at half a unit so the running total rounds instead of truncating.
            need_frames_carry: u64::from(device_in_sr / 2),
            primed: false,
            prime_margin,
            pending_cap,
            last_owed_frames: 0,
        })
    }

    /// Device frames per channel owed by the most recent `push_and_convert` call, valid whether it
    /// returned `Ok` (the output is exactly `last_owed_frames() * ch_out` samples) or `Err` (the
    /// caller's fallback for that tick should be exactly this long, truncated or zero-padded).
    pub fn last_owed_frames(&self) -> usize {
        self.last_owed_frames
    }

    pub fn set_controls(&mut self, pitch_shift: f32, index_rate: f32, protect: f32, speaker_id: i64) {
        self.pitch_shift = pitch_shift;
        self.index_rate = index_rate;
        self.protect = protect;
        self.speaker_id = speaker_id;
    }

    /// Convert one tick of capture audio; returns exactly the device frames owed for it
    /// (`last_owed_frames() * ch_out` interleaved samples).
    ///
    /// On `Err` the tick is accounted exactly like a success: the input is consumed, a block the
    /// model failed on contributes silence of its accounted length (and the next block fades in
    /// from silence), and the owed frames are taken from the backlog and dropped. The caller plays
    /// its own audio for the tick, `last_owed_frames()` frames long, so the backlog and the output
    /// timeline stay exactly where an error-free run would have them (no drift, no re-prime).
    pub fn push_and_convert(&mut self, interleaved_in: &[f32]) -> Result<Vec<f32>, String> {
        let in_frames = interleaved_in.len() / self.ch_in;
        let mono = downmix_to_mono(&interleaved_in[..in_frames * self.ch_in], self.ch_in);
        self.in_buf_acc.extend_from_slice(&mono);

        // Device frames owed for this tick (the exact carry keeps the feed rate equal to the sink
        // rate when capture and playback devices differ).
        let (need_frames, carry) = scale_with_carry(in_frames, self.device_out_sr, self.device_in_sr, self.need_frames_carry);
        self.need_frames_carry = carry;
        self.last_owed_frames = need_frames;

        let mut first_err: Option<String> = None;
        let mut model_out: Vec<f32> = Vec::new();
        let mut mono_16k: Vec<f32> = Vec::new();
        while self.in_buf_acc.len() >= self.process_frames {
            let block: Vec<f32> = self.in_buf_acc.drain(..self.process_frames).collect();
            mono_16k.clear();
            if let Err(e) = self.in_resampler.push(&block, &mut mono_16k) {
                // rubato only fails on buffer-size mismatches, which the fixed chunk rules out;
                // the block's share cannot be accounted without its output length.
                first_err.get_or_insert(e);
                continue;
            }
            // On a model error `convert_block` still appends the block's accounted length (as
            // silence), so the stream downstream advances exactly as on success.
            if let Err(e) = self.convert_block(&mono_16k, &mut model_out) {
                first_err.get_or_insert(e);
            }
        }
        if let Err(e) = self.out_resampler.push(&model_out, &mut self.out_pending) {
            first_err.get_or_insert(e);
        }
        // Drain the owed frames on success and failure alike: the caller's fallback replaces this
        // tick's output one-for-one, so the backlog must move as if the tick had played.
        let out = self.emit(need_frames);
        match first_err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    /// Add one resampled block to the ring and append exactly its model-rate share to `out`.
    fn convert_block(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let new_frames = mono_16k.len();
        let (block_out, carry) = scale_with_carry(new_frames, self.model_out_sr, self.model_in_sr, self.block_out_carry);
        self.block_out_carry = carry;

        self.ring_16k.extend_from_slice(mono_16k);
        let want = new_frames.saturating_add(self.sola_tail_in);
        let max_ring = self
            .extra_samples
            .saturating_add(new_frames)
            .max(want)
            .max(self.min_context);
        if self.ring_16k.len() > max_ring {
            let drop = self.ring_16k.len() - max_ring;
            self.ring_16k.drain(..drop);
        }
        if block_out == 0 {
            return Ok(());
        }
        if self.ring_16k.len() < want.max(self.min_context) {
            // Not enough context for the content encoder yet: silence of the accounted length.
            out.resize(out.len() + block_out, 0.0);
            self.sola.hold_silence();
            return Ok(());
        }
        let ctx = ConversionCtx {
            mono_16k: &self.ring_16k,
            new_frames: want,
            pitch_shift_semitones: self.pitch_shift,
            index_rate: self.index_rate,
            protect: self.protect,
            speaker_id: self.speaker_id,
        };
        let result = self.model.convert(&ctx).and_then(|chunk| {
            // NaN / inf would poison the SOLA tail, the output resampler's filter state and the
            // device output; treat it like any other model error.
            if chunk.iter().all(|x| x.is_finite()) {
                Ok(chunk)
            } else {
                Err("VC model produced non-finite output (NaN/inf)".to_string())
            }
        });
        let chunk = match result {
            Ok(chunk) => chunk,
            Err(e) => {
                // Keep the length accounting of a success (silence of the block's share), and
                // rejoin from silence at the window centre instead of crossfading the next chunk
                // against a stale pre-error tail.
                out.resize(out.len() + block_out, 0.0);
                self.sola.hold_silence();
                return Err(e);
            }
        };
        out.extend(self.sola.process(&chunk, block_out));
        Ok(())
    }

    /// Take `need` device frames from the backlog (silence while priming) as interleaved output.
    fn emit(&mut self, need: usize) -> Vec<f32> {
        if self.out_pending.len() > self.pending_cap {
            let excess = self.out_pending.len() - self.pending_cap;
            self.out_pending.drain(..excess);
        }
        let mut mono_out: Vec<f32> = Vec::with_capacity(need);
        if !self.primed {
            // Keep `prime_margin` samples back as the jitter cushion; lead with silence.
            if self.out_pending.len() > self.prime_margin {
                let take = (self.out_pending.len() - self.prime_margin).min(need);
                mono_out.resize(need - take, 0.0);
                mono_out.extend(self.out_pending.drain(..take));
                self.primed = true;
            } else {
                mono_out.resize(need, 0.0);
            }
        } else if self.out_pending.len() >= need {
            mono_out.extend(self.out_pending.drain(..need));
        } else {
            // Underrun: play what is left, pad with silence and rebuild the cushion.
            mono_out.append(&mut self.out_pending);
            mono_out.resize(need, 0.0);
            self.primed = false;
        }
        mono_to_interleaved(&mono_out, self.ch_out)
    }
}

/// `floor((frames * to + carry) / from)` and the new remainder: an exact integer (Bresenham)
/// rate conversion whose running total is always `floor(total_frames * to / from)`.
fn scale_with_carry(frames: usize, to: u32, from: u32, carry: u64) -> (usize, u64) {
    let from = u64::from(from.max(1));
    let total = frames as u64 * u64::from(to) + carry;
    ((total / from) as usize, total % from)
}

/// Ceiling division with a zero divisor treated as 1.
fn div_ceil_u64(a: u64, b: u64) -> u64 {
    a.div_ceil(b.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const MIN_CTX: usize = 720;

    /// `(mono_16k.len(), new_frames)` of every warmup and every conversion call.
    #[derive(Default)]
    struct ShapeLog {
        warm: Vec<(usize, usize)>,
        convert: Vec<(usize, usize)>,
    }

    /// Stub model: returns a constant (or the trailing input itself when `out_sr == 16k` and
    /// `passthrough`), and asserts the harness never calls it without enough context.
    struct StubModel {
        out_sr: u32,
        value: f32,
        passthrough: bool,
        calls: Arc<AtomicUsize>,
        fail_calls: Vec<usize>,
        /// Failing calls return NaN-filled audio instead of `Err`.
        fail_with_nan: bool,
        log: Option<Arc<Mutex<ShapeLog>>>,
    }

    /// Catmull-Rom resampling of the trailing `n` samples of `x` from 16 kHz to `out_sr` (edges
    /// clamped; the last few samples land in the discarded SOLA margin).
    fn upsample_tail(x: &[f32], n: usize, out_sr: u32) -> Vec<f32> {
        let start = x.len() - n;
        let len = ((n as u64 * u64::from(out_sr) + 8_000) / 16_000) as usize;
        let at = |i: isize| x[i.clamp(0, x.len() as isize - 1) as usize];
        (0..len)
            .map(|k| {
                let pos = start as f64 + k as f64 * 16_000.0 / f64::from(out_sr);
                let i = pos.floor() as isize;
                let t = (pos - i as f64) as f32;
                let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
                p1 + 0.5
                    * t
                    * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
            })
            .collect()
    }

    impl VoiceConversionModel for StubModel {
        fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            assert!(ctx.mono_16k.len() >= MIN_CTX, "called with {} samples", ctx.mono_16k.len());
            assert!(ctx.mono_16k.len() >= ctx.new_frames);
            if let Some(log) = &self.log {
                log.lock().unwrap().convert.push((ctx.mono_16k.len(), ctx.new_frames));
            }
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let len = (ctx.new_frames as u64 * u64::from(self.out_sr) + 8_000) / 16_000;
            if self.fail_calls.contains(&call) {
                if self.fail_with_nan {
                    let mut bad = vec![self.value; len as usize];
                    bad[len as usize / 2] = f32::NAN;
                    return Ok(bad);
                }
                return Err("stub failure".into());
            }
            if self.passthrough {
                let n = ctx.new_frames;
                if self.out_sr == 16_000 {
                    return Ok(ctx.mono_16k[ctx.mono_16k.len() - n..].to_vec());
                }
                return Ok(upsample_tail(ctx.mono_16k, n, self.out_sr));
            }
            Ok(vec![self.value; len as usize])
        }

        fn input_sample_rate(&self) -> u32 {
            16_000
        }

        fn output_sample_rate(&self) -> u32 {
            self.out_sr
        }

        fn min_context_samples(&self) -> usize {
            MIN_CTX
        }

        fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
            assert!(ctx.mono_16k.len() >= MIN_CTX, "warmed with {} samples", ctx.mono_16k.len());
            assert!(ctx.mono_16k.len() >= ctx.new_frames);
            if let Some(log) = &self.log {
                log.lock().unwrap().warm.push((ctx.mono_16k.len(), ctx.new_frames));
            }
            Ok(())
        }
    }

    fn stub(out_sr: u32, fail_calls: Vec<usize>) -> (Box<StubModel>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Box::new(StubModel {
                out_sr,
                value: 0.5,
                passthrough: false,
                calls: calls.clone(),
                fail_calls,
                fail_with_nan: false,
                log: None,
            }),
            calls,
        )
    }

    #[test]
    fn sola_emits_exactly_block() {
        let mut sola = SolaStitcher::new(100);
        let cases = [
            (200usize, 60usize),
            (60, 60),
            (30, 60),
            (170, 100),
            (500, 100),
            (100, 100),
            (0, 10),
            (1000, 1),
            (260, 100),
        ];
        for (i, &(len, block)) in cases.iter().enumerate() {
            let chunk: Vec<f32> = (0..len)
                .map(|k| ((k + i * 7) as f32 * 0.37).sin())
                .collect();
            assert_eq!(sola.process(&chunk, block).len(), block, "case {i}");
        }
        assert!(sola.process(&[1.0; 50], 0).is_empty());
    }

    #[test]
    fn sola_search_is_centred_and_normalized() {
        // Deterministic noise (no periodic aliases).
        let mut seed = 0x9e37_79b9u32;
        let x: Vec<f32> = (0..4000)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let (ov, block) = (64usize, 200usize);
        let mut sola = SolaStitcher::new(ov);
        let search = sola.search();
        assert_eq!(search, 32);
        let len = block + ov + search;
        let center = search / 2;
        // Tick 1 (no history): nominal centre offset.
        let out1 = sola.process(&x[..len], block);
        assert_eq!(out1, x[center..center + block].to_vec());
        // Tick 2: the matching content sits 6 samples *before* the centre; a one-sided search
        // (offsets only forward of the nominal) could not reach it.
        let s = block + 6;
        let out2 = sola.process(&x[s..s + len], block);
        let off = center - 6;
        assert_eq!(out2.len(), block);
        // Crossfade of identical content is seamless: output continues x exactly.
        for (i, (&o, &e)) in out2.iter().zip(x[s + off..].iter()).enumerate() {
            assert!((o - e).abs() < 1e-5, "sample {i}: {o} vs {e}");
        }
    }

    #[test]
    fn sola_alignment_is_amplitude_independent() {
        let mut seed = 0x0bad_f00du32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let n = 64usize;
        let window = 32usize;
        let tail: Vec<f32> = (0..n).map(|_| noise()).collect();
        // A quiet exact copy at offset 12, loud unrelated audio everywhere else: raw dot
        // products favour the loud offsets, normalized cross-correlation finds the copy.
        let mut chunk: Vec<f32> = (0..n + window).map(|_| 10.0 * noise()).collect();
        for (dst, &src) in chunk[12..12 + n].iter_mut().zip(tail.iter()) {
            *dst = 0.1 * src;
        }
        let mut sola = SolaStitcher::new(n);
        sola.prev_tail = tail;
        assert_eq!(sola.best_offset(&chunk, n, window, window / 2), 12);
    }

    fn run_dc(r_in: u32, r_out: u32, m_out: u32, ch: usize, frames: usize, ticks: usize, fail_calls: Vec<usize>) {
        let (model, calls) = stub(m_out, fail_calls.clone());
        let mut h = VcStreamHarness::new(model, r_in, r_out, ch, ch, 1024, 320, frames).unwrap();
        let input = vec![0.1f32; frames * ch];
        // round(total_in * r_out / r_in), in exact integers.
        let owed = |frames_in: usize| ((frames_in as u64 * u64::from(r_out) + u64::from(r_in / 2)) / u64::from(r_in)) as usize;
        let mut total_out = 0usize;
        let mut errors = 0usize;
        let mut last_err_tick: Option<usize> = None;
        let mut primed_since: Option<usize> = None;
        for tick in 1..=ticks {
            let before = owed((tick - 1) * frames);
            let after = owed(tick * frames);
            let need = after - before;
            match h.push_and_convert(&input) {
                Ok(out) => {
                    assert_eq!(out.len(), need * ch, "tick {tick}");
                    assert_eq!(h.last_owed_frames(), need, "tick {tick}");
                    total_out += out.len() / ch;
                    // A failed block plays as one block of silence a few ticks later (after the
                    // pipeline latency): skip the value check around it.
                    let near_error = last_err_tick.is_some_and(|e| tick <= e + 8);
                    if primed_since.is_some_and(|t| tick > t + 2) && !near_error {
                        for (i, v) in out.iter().enumerate() {
                            assert!((v - 0.5).abs() < 0.02, "{r_in}->{r_out} (model {m_out}) tick {tick} sample {i}: {v}");
                        }
                    }
                    if primed_since.is_none() && out.iter().all(|v| (v - 0.5).abs() < 0.02) {
                        primed_since = Some(tick);
                    }
                }
                Err(_) => {
                    // The engine plays its fallback for this tick's owed frames.
                    assert_eq!(h.last_owed_frames(), need, "tick {tick}");
                    errors += 1;
                    last_err_tick = Some(tick);
                    total_out += h.last_owed_frames();
                }
            }
            if primed_since.is_some() {
                assert!(h.primed, "underrun at tick {tick}");
                assert!(
                    h.out_pending.len() <= 2 * h.prime_margin + 8,
                    "backlog grew to {} at tick {tick}",
                    h.out_pending.len()
                );
            }
        }
        let expected = owed(ticks * frames);
        assert_eq!(total_out, expected);
        assert_eq!(errors, fail_calls.len());
        assert!(primed_since.is_some_and(|t| t < 40), "never settled: {primed_since:?}");
        assert!(calls.load(Ordering::SeqCst) > ticks - 40);
    }

    #[test]
    fn harness_long_run_length_is_exact_48k_to_48k() {
        run_dc(48_000, 48_000, 40_000, 1, 512, 3000, Vec::new());
        run_dc(48_000, 48_000, 48_000, 2, 160, 3000, Vec::new());
    }

    #[test]
    fn harness_long_run_length_is_exact_44k1_to_48k() {
        run_dc(44_100, 48_000, 40_000, 1, 512, 3000, Vec::new());
        run_dc(44_100, 48_000, 32_000, 2, 441, 3000, Vec::new());
    }

    #[test]
    fn harness_long_run_length_is_exact_48k_to_44k1() {
        run_dc(48_000, 44_100, 40_000, 1, 512, 3000, Vec::new());
        run_dc(48_000, 44_100, 48_000, 1, 1024, 3000, Vec::new());
    }

    #[test]
    fn harness_model_errors_keep_accounting_consistent() {
        run_dc(44_100, 48_000, 40_000, 1, 512, 3000, vec![200, 201, 1500]);
    }

    #[test]
    fn harness_model_error_keeps_backlog_of_error_free_run() {
        // Two harnesses in lockstep, one whose model fails now and then: every length, carry and
        // priming state must match tick for tick (no random walk, no re-prime after an error).
        for &(r_in, r_out, m_out, ch, frames) in &[
            (44_100u32, 48_000u32, 40_000u32, 1usize, 512usize),
            (48_000, 44_100, 32_000, 2, 480),
            (48_000, 48_000, 48_000, 1, 160),
        ] {
            let fails = vec![30usize, 31, 90, 150];
            let (ok_model, _) = stub(m_out, Vec::new());
            let (bad_model, _) = stub(m_out, fails.clone());
            let mut ok = VcStreamHarness::new(ok_model, r_in, r_out, ch, ch, 1024, 320, frames).unwrap();
            let mut bad = VcStreamHarness::new(bad_model, r_in, r_out, ch, ch, 1024, 320, frames).unwrap();
            let input = vec![0.1f32; frames * ch];
            let mut errors = 0;
            for tick in 1..=250 {
                let a = ok.push_and_convert(&input).unwrap();
                match bad.push_and_convert(&input) {
                    Ok(b) => assert_eq!(b.len(), a.len(), "tick {tick}"),
                    Err(_) => {
                        errors += 1;
                        // The failed block left a silent tail: the next join fades in from zero
                        // at the window centre instead of crossfading a stale pre-error tail.
                        assert_eq!(bad.sola.prev_tail.len(), bad.sola.overlap);
                        assert!(bad.sola.prev_tail.iter().all(|&x| x == 0.0), "tick {tick}");
                    }
                }
                let label = format!("{r_in}->{r_out} ch{ch} tick {tick}");
                assert_eq!(bad.last_owed_frames() * ch, a.len(), "{label}");
                assert_eq!(bad.last_owed_frames(), ok.last_owed_frames(), "{label}");
                assert_eq!(bad.out_pending.len(), ok.out_pending.len(), "backlog diverged: {label}");
                assert_eq!(bad.primed, ok.primed, "{label}");
                assert_eq!(bad.block_out_carry, ok.block_out_carry, "{label}");
                assert_eq!(bad.need_frames_carry, ok.need_frames_carry, "{label}");
                assert_eq!(bad.ring_16k.len(), ok.ring_16k.len(), "{label}");
            }
            assert_eq!(errors, fails.len());
            assert!(ok.primed);
        }
    }

    #[test]
    fn harness_warms_model_at_live_shapes() {
        for &(r_in, frames) in &[(48_000u32, 512usize), (44_100, 512), (48_000, 480), (16_000, 320), (22_050, 1024)] {
            let log = Arc::new(Mutex::new(ShapeLog::default()));
            let model = Box::new(StubModel {
                out_sr: 40_000,
                value: 0.5,
                passthrough: false,
                calls: Arc::new(AtomicUsize::new(0)),
                fail_calls: Vec::new(),
                fail_with_nan: false,
                log: Some(log.clone()),
            });
            let mut h = VcStreamHarness::new(model, r_in, 48_000, 1, 1, 1024, 320, frames).unwrap();
            // Warmed inside `new`, before any audio, and never again.
            let warm = log.lock().unwrap().warm.clone();
            assert!(!warm.is_empty() && warm.len() <= 2, "{r_in}/{frames}: {warm:?}");
            assert!(log.lock().unwrap().convert.is_empty());
            let input = vec![0.1f32; frames];
            for _ in 0..300 {
                h.push_and_convert(&input).unwrap();
            }
            let log = log.lock().unwrap();
            assert_eq!(log.warm, warm, "warmup runs only in new()");
            // The ring starts zero-filled to the steady length, so every live call, the very
            // first one included, uses a warmed input shape (no cold start-up shapes).
            assert!(log.convert.len() >= 290, "{}", log.convert.len());
            for (i, shape) in log.convert.iter().enumerate() {
                assert!(warm.contains(shape), "{r_in}/{frames}: call {i} shape {shape:?} not warmed ({warm:?})");
            }
            // Non-integer resampling ratios alternate between the floor and ceiling shapes; both
            // must show up live, or the second warmup is wasted.
            let live: Vec<&(usize, usize)> = warm.iter().filter(|w| log.convert.contains(w)).collect();
            assert_eq!(live.len(), warm.len(), "{r_in}/{frames}: warmed {warm:?}");
        }
    }

    #[test]
    fn harness_converts_from_the_first_tick() {
        // Small extra context too, so the prefill is set by the model's minimum context.
        for &(r_in, frames, extra) in &[(48_000u32, 512usize, 1024usize), (48_000, 512, 0), (44_100, 441, 64), (16_000, 160, 0)] {
            let (model, calls) = stub(40_000, Vec::new());
            let mut h = VcStreamHarness::new(model, r_in, 48_000, 1, 1, extra, 320, frames).unwrap();
            assert!(h.ring_16k.iter().all(|&v| v == 0.0));
            let input = vec![0.1f32; frames];
            let mut ring_lens = Vec::new();
            for tick in 1..=40usize {
                let out = h.push_and_convert(&input).unwrap();
                // Exact owed length on every tick, the first included.
                assert_eq!(out.len(), h.last_owed_frames(), "{r_in}/{frames} tick {tick}");
                // One model call per tick from the very first tick (no silent start-up ticks
                // waiting for context).
                assert_eq!(calls.load(Ordering::SeqCst), tick, "{r_in}/{frames} tick {tick}");
                assert!(h.ring_16k.len() >= MIN_CTX);
                ring_lens.push(h.ring_16k.len());
            }
            // The ring is at its steady length from tick 1 (floor/ceil block sizes at most).
            ring_lens.sort_unstable();
            ring_lens.dedup();
            assert!(ring_lens.len() <= 2, "{r_in}/{frames}: ring lengths {ring_lens:?}");
            assert!(ring_lens.last().unwrap() - ring_lens[0] <= 1, "{r_in}/{frames}: {ring_lens:?}");
        }
    }

    #[test]
    fn harness_min_context_guard_still_holds() {
        // Safety net: if the ring is ever shorter than the model's minimum context (normal runs
        // never are, see `harness_converts_from_the_first_tick`), the model is not called and
        // the block is accounted silence.
        let (model, calls) = stub(40_000, Vec::new());
        let mut h = VcStreamHarness::new(model, 48_000, 48_000, 1, 1, 1024, 320, 512).unwrap();
        h.ring_16k.clear();
        let input = vec![0.1f32; 512];
        let mut calls_before = 0;
        let mut guarded_ticks = 0;
        for tick in 0..10 {
            let out = h.push_and_convert(&input).unwrap();
            assert_eq!(out.len(), 512);
            let calls_now = calls.load(Ordering::SeqCst);
            if h.ring_16k.len() < MIN_CTX {
                // Short ring: the model must not run (the stub also asserts on every call).
                assert_eq!(calls_now, calls_before, "model called at tick {tick}");
                guarded_ticks += 1;
            }
            calls_before = calls_now;
        }
        // 512 frames @ 48 kHz = 170.67 samples @ 16 kHz per tick: ticks 1-4 stay below 720.
        assert!(guarded_ticks >= 4, "{guarded_ticks}");
        assert!(calls_before >= 5, "{calls_before}");
    }

    #[test]
    fn harness_rejects_non_finite_model_output() {
        // A model returning NaN is handled exactly like one returning `Err`: same accounting as an
        // error-free run, silent tail, and no NaN ever reaches the output.
        for &(r_in, r_out, m_out, ch, frames) in &[(44_100u32, 48_000u32, 40_000u32, 1usize, 512usize), (48_000, 48_000, 48_000, 2, 480)] {
            let fails = vec![5usize, 30, 31, 90];
            let (ok_model, _) = stub(m_out, Vec::new());
            let (mut nan_model, _) = stub(m_out, fails.clone());
            nan_model.fail_with_nan = true;
            let mut ok = VcStreamHarness::new(ok_model, r_in, r_out, ch, ch, 1024, 320, frames).unwrap();
            let mut bad = VcStreamHarness::new(nan_model, r_in, r_out, ch, ch, 1024, 320, frames).unwrap();
            let input = vec![0.1f32; frames * ch];
            let mut errors = 0;
            for tick in 1..=150 {
                let a = ok.push_and_convert(&input).unwrap();
                match bad.push_and_convert(&input) {
                    Ok(b) => {
                        assert_eq!(b.len(), a.len(), "tick {tick}");
                        assert!(b.iter().all(|v| v.is_finite()), "non-finite output at tick {tick}");
                    }
                    Err(e) => {
                        assert!(e.contains("non-finite"), "{e}");
                        errors += 1;
                        assert!(bad.sola.prev_tail.iter().all(|&x| x == 0.0), "tick {tick}");
                    }
                }
                assert_eq!(bad.last_owed_frames(), ok.last_owed_frames(), "tick {tick}");
                assert_eq!(bad.out_pending.len(), ok.out_pending.len(), "tick {tick}");
                assert!(bad.out_pending.iter().all(|v| v.is_finite()), "tick {tick}");
                assert_eq!(bad.primed, ok.primed, "tick {tick}");
            }
            assert_eq!(errors, fails.len());
        }
    }

    /// Three incommensurate tones (unique best lag, no single-sine period ambiguity).
    fn multitone(t: f64) -> f64 {
        [311.0f64, 523.0, 797.0]
            .iter()
            .map(|&f| 0.2 * (2.0 * std::f64::consts::PI * f * t).sin())
            .sum()
    }

    /// Identity model at 16 kHz fed the multi-tone. Returns the output timeline, with each failed
    /// tick filled by `last_owed_frames()` zeros (the engine's fallback), and the output index at
    /// which each failed tick starts.
    fn run_passthrough(r_in: u32, r_out: u32, frames: usize, ticks: usize, fail_calls: Vec<usize>) -> (Vec<f32>, Vec<usize>) {
        run_passthrough_model(16_000, r_in, r_out, frames, ticks, fail_calls)
    }

    /// `run_passthrough` with an identity model whose output rate is `m_out` (upsampled input).
    fn run_passthrough_model(
        m_out: u32,
        r_in: u32,
        r_out: u32,
        frames: usize,
        ticks: usize,
        fail_calls: Vec<usize>,
    ) -> (Vec<f32>, Vec<usize>) {
        let model = Box::new(StubModel {
            out_sr: m_out,
            value: 0.0,
            passthrough: true,
            calls: Arc::new(AtomicUsize::new(0)),
            fail_calls,
            fail_with_nan: false,
            log: None,
        });
        let mut h = VcStreamHarness::new(model, r_in, r_out, 1, 1, 1024, 320, frames).unwrap();
        let mut out = Vec::new();
        let mut err_at = Vec::new();
        for tick in 0..ticks {
            let input: Vec<f32> = (0..frames)
                .map(|i| multitone((tick * frames + i) as f64 / f64::from(r_in)) as f32)
                .collect();
            match h.push_and_convert(&input) {
                Ok(v) => {
                    assert_eq!(v.len(), h.last_owed_frames());
                    out.extend(v);
                }
                Err(_) => {
                    err_at.push(out.len());
                    out.resize(out.len() + h.last_owed_frames(), 0.0);
                }
            }
        }
        (out, err_at)
    }

    const MAX_LAG: usize = 6_000;

    /// Best integer lag of `out` against the multi-tone over `[start, start + 2000)`.
    fn fit_lag(out: &[f32], reference: &[f64], start: usize) -> usize {
        assert!(start >= MAX_LAG && start + 2_000 <= out.len());
        let (mut best_lag, mut best_err) = (0usize, f64::MAX);
        for lag in 0..MAX_LAG {
            let err: f64 = (start..start + 2_000)
                .map(|j| (f64::from(out[j]) - reference[j - lag]).powi(2))
                .sum();
            if err < best_err {
                best_err = err;
                best_lag = lag;
            }
        }
        best_lag
    }

    /// Largest deviation from the multi-tone delayed by `lag`, over `start..` minus `skip`
    /// ranges, and the number of samples checked.
    fn max_error_at_lag(out: &[f32], reference: &[f64], lag: usize, start: usize, skip: &[(usize, usize)]) -> (f64, usize) {
        let mut max_err = 0.0f64;
        let mut checked = 0usize;
        for j in start..out.len() {
            if skip.iter().any(|&(a, b)| (a..b).contains(&j)) {
                continue;
            }
            max_err = max_err.max((f64::from(out[j]) - reference[j - lag]).abs());
            checked += 1;
        }
        (max_err, checked)
    }

    fn reference_at(len: usize, r_out: u32) -> Vec<f64> {
        (0..len).map(|k| multitone(k as f64 / f64::from(r_out))).collect()
    }

    #[test]
    fn harness_passthrough_is_a_constant_delay() {
        // Identity model at 16 kHz: the output must be the input, band-limited and delayed by a
        // constant latency (no drift, no chunk-boundary glitches, no SOLA jumps).
        for &(r_in, r_out, frames) in &[(48_000u32, 48_000u32, 512usize), (44_100, 48_000, 512)] {
            let (out, err_at) = run_passthrough(r_in, r_out, frames, 600, Vec::new());
            assert!(err_at.is_empty());
            let reference = reference_at(out.len(), r_out);
            let start = out.len() / 4;
            let lag = fit_lag(&out, &reference, start);
            let (max_err, _) = max_error_at_lag(&out, &reference, lag, start, &[]);
            assert!(max_err < 0.03, "{r_in}->{r_out}: max error {max_err} at constant lag {lag}");
            // The ring starts at its steady length, so the lag is already the steady one right
            // after the priming cushion (no start-up drift while context accumulates).
            let early = fit_lag(&out, &reference, MAX_LAG);
            assert_eq!(early, lag, "{r_in}->{r_out}: early lag {early} vs steady {lag}");
            let (early_err, _) = max_error_at_lag(&out, &reference, lag, lag + 4_096, &[]);
            assert!(early_err < 0.03, "{r_in}->{r_out}: early max error {early_err} at lag {lag}");
        }
    }

    #[test]
    fn harness_passthrough_delay_survives_model_errors() {
        // Same identity model, but calls 200, 201 and 1500 fail. Away from the failed ticks and
        // the silent blocks they leave, the output must follow the input at the *same* lag
        // before and after every error (the old truncate-on-error path shifted it each time).
        // The 48 kHz model variants run SOLA at the model's own rate (blocks alternating between
        // floor and ceiling lengths at 44.1 kHz input; overlap clamped to the block at 160 frames).
        for &(m_out, r_in, r_out, frames) in &[
            (16_000u32, 48_000u32, 48_000u32, 512usize),
            (16_000, 44_100, 48_000, 512),
            (48_000, 48_000, 48_000, 512),
            (48_000, 44_100, 48_000, 512),
            (48_000, 48_000, 48_000, 160),
        ] {
            // Enough ticks after the last failure for 50k compared samples at small chunk sizes.
            let ticks = 1_500 + (60_000 / frames).max(200);
            let (out, err_at) = run_passthrough_model(m_out, r_in, r_out, frames, ticks, vec![200, 201, 1500]);
            assert_eq!(err_at.len(), 3, "{r_in}->{r_out} (model {m_out})");
            let reference = reference_at(out.len(), r_out);
            let start = err_at[0] / 2;
            let lag = fit_lag(&out, &reference, start);
            // The failed tick itself (fallback), then the failed block's silence arriving one
            // pipeline latency later, plus the fade-in and resampler ringing around it.
            let owed_per_tick = div_ceil_u64(frames as u64 * u64::from(r_out), u64::from(r_in)) as usize;
            let guard = lag + 4 * owed_per_tick + 1_024;
            let skip: Vec<(usize, usize)> = err_at.iter().map(|&p| (p, p + guard)).collect();
            let (max_err, _) = max_error_at_lag(&out, &reference, lag, start, &skip);
            assert!(max_err < 0.03, "{r_in}->{r_out}: max error {max_err} at constant lag {lag}");
            // Plenty of audio after the last error was actually compared at that lag.
            let tail_start = err_at[2] + guard;
            let (tail_err, tail_checked) = max_error_at_lag(&out, &reference, lag, tail_start, &[]);
            assert!(tail_checked > 50_000, "{tail_checked}");
            assert!(tail_err < 0.03, "{r_in}->{r_out}: after the last error {tail_err}");
            // And the errors really did disturb the guarded windows (the test is not vacuous).
            let (guard_err, _) = max_error_at_lag(&out[..err_at[0] + guard], &reference, lag, err_at[0], &[]);
            assert!(guard_err > 0.1, "{r_in}->{r_out}: error window looks clean ({guard_err})");
        }
    }
}
