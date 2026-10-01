//! The MeanVC2 streaming loop: upstream `runtime/run_rt.py` `VCRunner` (40 ms model), as
//! `scripts/vc-families/meanvc/sampler.py` reproduces it on Sigil's graphs (ADR 0006), state for
//! state, with three Sigil changes:
//!
//! - **4k + 1 BN frames.** Upstream interpolates each 160 ms audio chunk's BN frames (with the
//!   previous chunk's last one in front) to 17 frames and drops the first. The port interpolates
//!   `k` new BN frames to `4k + 1` frames, which is the same positions (step 1/4 of a BN frame) in
//!   the steady state, so the loop can be fed 80 ms at a time.
//! - **Phase lead.** A cold stream starts with [`PHASE_LEAD_16K`] samples of silence in its
//!   sample cache. Every ASR window (11 fbank frames, stride 8) then ends exactly on a multiple of
//!   80 ms of input, so a tick's new audio is encoded in the same tick (without the lead every
//!   window waits for 3 more fbank frames, one tick later: 35 ms of extra delay).
//! - **Anchored noise.** Upstream draws `z ~ N(0, 1)` `[1, 8, 80]` per DiT call and overwrites its
//!   first 4 frames with the previous call's last 4 (the future frames' noise is reused once they
//!   are current). The port draws every mel frame's noise once from Sigil's counter-based generator
//!   (`engine::noise`, stream `Flow`, index `channel × 2^40 + frame`), which is the same scheme
//!   with a reproducible generator.
//!
//! # Per call ([`MeanVcStream::process`])
//! 1. Fbank over the sample cache plus the new samples; the samples no frame covered yet are kept.
//! 2. ASR every 8 fbank frames over an 11-frame window (offset += 2, wrapped at 4000).
//! 3. The new BN frames, with the last BN frame of the previous call in front, upsampled 4×
//!    (linear, align_corners, PyTorch's float32 arithmetic), first frame dropped.
//! 4. While 8 upsampled frames are buffered: the mean-flow steps (t = 1 − i/N, r = max(0, t − 1/N),
//!    dt = 1/N, every step from the same KV cache, the last step's cache kept; offset += 4, wrapped
//!    at 4000 with the cache cleared), the first 4 mel frames out, 4 frames dropped.
//! 5. Vocos over the previous 2 mel frames and the 4 new ones (800 samples), crossfaded over 160
//!    samples with the previous call's tail; 640 new samples out (the last 160 are held back).
//!
//! In the steady state every 1280 input samples give exactly 1280 output samples. A cold stream's
//! first call gives 320 (the vocoder's first call has no overlap), and from then on every call
//! gives exactly its input length. [`StreamState`] is plain data, so the model can keep a copy of a
//! known state (the steady silence it starts from) and return to it.
//!
//! # Spreading the work ([`MeanVcStream::set_dit_limit`])
//! An 80 ms tick holds one ASR call and two DiT calls (each DiT call is 40 ms of output,
//! [`DIT_HOP_16K`]). Fed 40 ms at a time with a limit of one DiT call per call, the loop runs the
//! ASR and the first DiT call when a window completes and the second DiT call (its frames already
//! buffered) in the next call: every 40 ms call then gives exactly 640 samples for about half the
//! compute, and the deferred half lags 40 ms more, which the smaller block gives back (block plus
//! delay is unchanged). The deferred state is ordinary buffered state, so snapshots and restores
//! keep working.

use super::graphs::{
    DitInputs, GraphShapes, MeanVcGraphs, SpeakerState, ASR_OUT_FRAMES, ASR_STRIDE, ASR_WINDOW, BN_DIM, DIT_FRAMES, HOP, MEL_BINS, SPK_DIM,
};
use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream};

/// Silence a cold stream starts with (see the module docs).
pub const PHASE_LEAD_16K: usize = 720;
/// ASR positional offset at the start, its step per call, and where both offsets wrap.
pub const ASR_OFFSET_INIT: i64 = 4;
pub const ASR_OFFSET_STEP: i64 = 2;
pub const OFFSET_WRAP: i64 = 4000;
/// Current (returned) frames per DiT call; the other 4 of its 8 are the future block.
pub const DIT_CHUNK: usize = 4;
/// Mel frames of the previous Vocos call fed again, and the crossfade in samples.
pub const VOC_OVERLAP: usize = 2;
pub const WAV_OVERLAP: usize = 160;
/// Upsampled (10 ms) frames per BN (40 ms) frame.
pub const UPSAMPLE: usize = 4;
/// Output samples per DiT call in the steady state (4 mel frames of 10 ms).
pub const DIT_HOP_16K: usize = DIT_CHUNK * HOP;

/// `torch.linspace(start, end, steps)` in float32, bit for bit (the first half counts up from
/// `start`, the second half down from `end`), as `sampler.linspace_f32`.
pub fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// `F.interpolate(x.transpose(1, 2), size=out_len, mode='linear', align_corners=True)` of
/// `t_in` frames of `width` values (frame-major), first output frame dropped, appended to `dst`
/// (PyTorch's CPU kernel: `scale = (t_in − 1) / (out_len − 1)` and the weights in float32).
pub fn upsample_align_corners_drop_first(src: &[f32], t_in: usize, width: usize, out_len: usize, dst: &mut Vec<f32>) {
    debug_assert_eq!(src.len(), t_in * width);
    if t_in == 0 || out_len < 2 {
        return;
    }
    let scale = if out_len > 1 { (t_in - 1) as f32 / (out_len - 1) as f32 } else { 0.0 };
    for j in 1..out_len {
        let pos = scale * j as f32;
        let i0 = (pos.floor() as usize).min(t_in - 1);
        let i1 = (i0 + 1).min(t_in - 1);
        let l1 = pos - i0 as f32;
        let l0 = 1.0 - l1;
        let (a, b) = (&src[i0 * width..(i0 + 1) * width], &src[i1 * width..(i1 + 1) * width]);
        dst.extend(a.iter().zip(b).map(|(&x0, &x1)| l0 * x0 + l1 * x1));
    }
}

/// Everything the loop carries between calls (plain data: cloned to snapshot it).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamState {
    /// Input samples no fbank frame has covered yet.
    samples_cache: Vec<f32>,
    /// Fbank frames not yet consumed by an ASR window (`frames × 80`).
    frame_cache: Vec<f32>,
    asr_offset: i64,
    att: Vec<f32>,
    cnn: Vec<f32>,
    /// The last BN frame of the previous call (empty before the first one).
    enc_last: Vec<f32>,
    /// Upsampled BN frames waiting for a DiT call (`frames × 256`).
    bn_buffer: Vec<f32>,
    kv: Vec<f32>,
    n_valid: i64,
    vc_offset: i64,
    /// Absolute mel frame of the next DiT call's first frame (the noise index; never rewinds).
    frame: u64,
    /// The previous Vocos call's last 2 mel frames, raw, frame-major (empty before the first).
    voc_cache: Vec<f32>,
    /// The previous Vocos call's last 160 samples (empty before the first).
    last_wav: Vec<f32>,
}

impl StreamState {
    /// A cold stream (with the phase lead).
    pub fn cold(shapes: &GraphShapes) -> StreamState {
        StreamState {
            samples_cache: vec![0.0; PHASE_LEAD_16K],
            frame_cache: Vec::new(),
            asr_offset: ASR_OFFSET_INIT,
            att: vec![0.0; GraphShapes::len(&shapes.att_cache)],
            cnn: vec![0.0; GraphShapes::len(&shapes.cnn_cache)],
            enc_last: Vec::new(),
            bn_buffer: Vec::new(),
            kv: vec![0.0; GraphShapes::len(&shapes.dit_cache)],
            n_valid: 0,
            vc_offset: 0,
            frame: 0,
            voc_cache: Vec::new(),
            last_wav: Vec::new(),
        }
    }

    /// The noise index of the next DiT call.
    #[cfg(test)]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// The state with the noise index cleared (what a restore brings back).
    #[cfg(test)]
    pub fn without_frame(&self) -> StreamState {
        StreamState { frame: 0, ..self.clone() }
    }
}

/// The loop over a graph set (see the module docs).
pub struct MeanVcStream {
    graphs: Box<dyn MeanVcGraphs>,
    state: StreamState,
    speaker: SpeakerState,
    steps: u32,
    /// Most DiT calls per `process` call (`None` = every one the buffer allows).
    dit_limit: Option<usize>,
    noise: NoiseFrames,
    down: Vec<f32>,
    up: Vec<f32>,
    /// KV cache frames (`n_valid` stops there) and the ASR offset floor after a wrap.
    kv_frames: i64,
    att_floor: i64,
    // ---- scratch (reused every call) ----
    padded: Vec<f32>,
    fb: Vec<f32>,
    bn_call: Vec<f32>,
    bn_new: Vec<f32>,
    bn_ctx: Vec<f32>,
    x: Vec<f32>,
    x_next: Vec<f32>,
    cache_next: Vec<f32>,
    mel_frames: Vec<f32>,
    mel_in: Vec<f32>,
    voc_next: Vec<f32>,
    wav: Vec<f32>,
}

impl MeanVcStream {
    /// A cold stream converting toward `spks` (runs the GTM once).
    pub fn new(graphs: Box<dyn MeanVcGraphs>, spks: &[f32], steps: u32, noise_seed: u64) -> Result<MeanVcStream, String> {
        let shapes = graphs.shapes().clone();
        // Ranks and sizes before anything is allocated from them (`dit_cache[3]` and
        // `att_cache[2]` are read below).
        shapes.validate()?;
        let mem = GraphShapes::len(&shapes.memory);
        let mut stream = MeanVcStream {
            graphs,
            state: StreamState::cold(&shapes),
            speaker: SpeakerState {
                spks: vec![0.0; SPK_DIM],
                k_mem: vec![0.0; mem],
                v_mem: vec![0.0; mem],
            },
            steps: steps.clamp(1, 8),
            dit_limit: None,
            noise: NoiseFrames::new(AnchoredNoise::new(noise_seed), NoiseStream::Flow, MEL_BINS, DIT_FRAMES),
            down: linspace_f32(1.0, 0.0, WAV_OVERLAP),
            up: linspace_f32(0.0, 1.0, WAV_OVERLAP),
            kv_frames: shapes.dit_cache[3] as i64,
            att_floor: shapes.att_cache[2] as i64,
            padded: Vec::new(),
            fb: Vec::new(),
            bn_call: vec![0.0; ASR_OUT_FRAMES * BN_DIM],
            bn_new: Vec::new(),
            bn_ctx: Vec::new(),
            x: vec![0.0; DIT_FRAMES * MEL_BINS],
            x_next: vec![0.0; DIT_FRAMES * MEL_BINS],
            cache_next: vec![0.0; GraphShapes::len(&shapes.dit_cache)],
            mel_frames: Vec::new(),
            mel_in: Vec::new(),
            voc_next: Vec::new(),
            wav: Vec::new(),
        };
        stream.set_speaker(spks)?;
        Ok(stream)
    }

    /// Convert toward another speaker embedding from the next DiT call on (the GTM runs once).
    pub fn set_speaker(&mut self, spks: &[f32]) -> Result<(), String> {
        if spks.len() != SPK_DIM {
            return Err(format!("a MeanVC2 speaker embedding has {SPK_DIM} values, not {}", spks.len()));
        }
        if spks.iter().any(|v| !v.is_finite()) {
            return Err("the MeanVC2 speaker embedding holds NaN or infinite values".into());
        }
        let mem = self.speaker.k_mem.len();
        let mut k = vec![0.0; mem];
        let mut v = vec![0.0; mem];
        self.graphs.gtm(spks, &mut k, &mut v)?;
        if k.iter().chain(&v).any(|x| !x.is_finite()) {
            return Err("MeanVC2 gtm produced non-finite memories".into());
        }
        self.speaker = SpeakerState {
            spks: spks.to_vec(),
            k_mem: k,
            v_mem: v,
        };
        Ok(())
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Mean-flow steps from the next DiT call on (1..=8; the export's 1 or 2 in practice).
    pub fn set_steps(&mut self, steps: u32) {
        self.steps = steps.clamp(1, 8);
    }

    /// At most `limit` DiT calls per [`process`](Self::process) call (`None` or 0: no limit); the
    /// rest wait, buffered, for the next call (module docs).
    pub fn set_dit_limit(&mut self, limit: Option<usize>) {
        self.dit_limit = limit.filter(|&n| n > 0);
    }

    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// A cold stream (the phase lead included); the noise index keeps counting.
    pub fn reset_cold(&mut self) {
        let frame = self.state.frame;
        self.state = StreamState::cold(self.graphs.shapes());
        self.state.frame = frame;
    }

    /// Return to `snapshot` (a state of this graph set); the noise index keeps counting.
    pub fn restore(&mut self, snapshot: &StreamState) {
        let frame = self.state.frame;
        self.state.clone_from(snapshot);
        self.state.frame = frame;
    }

    /// Feed `samples` (16 kHz) and append the new output samples to `out`; returns how many. On an
    /// error the state is partway through the call: restore a snapshot before the next one.
    pub fn process(&mut self, samples: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        let start = out.len();
        self.encode(samples)?;
        let mut calls = 0usize;
        while self.state.bn_buffer.len() >= DIT_FRAMES * BN_DIM && self.dit_limit.is_none_or(|n| calls < n) {
            self.dit_call()?;
            self.decode(out)?;
            calls += 1;
        }
        Ok(out.len() - start)
    }

    /// Steps 1-3 of the module docs (new BN frames, if any, join the buffer).
    fn encode(&mut self, samples: &[f32]) -> Result<(), String> {
        self.padded.clear();
        self.padded.extend_from_slice(&self.state.samples_cache);
        self.padded.extend_from_slice(samples);
        if self.padded.len() < super::graphs::FBANK_WINDOW {
            std::mem::swap(&mut self.state.samples_cache, &mut self.padded);
            return Ok(());
        }
        self.fb.clear();
        self.fb.extend_from_slice(&self.state.frame_cache);
        let frames = self.graphs.fbank(&self.padded, &mut self.fb)?;
        self.state.samples_cache.clear();
        self.state
            .samples_cache
            .extend_from_slice(&self.padded[(HOP * frames).min(self.padded.len())..]);
        if self.state.asr_offset >= OFFSET_WRAP {
            self.state.asr_offset = self.att_floor.max(self.state.asr_offset - OFFSET_WRAP);
        }
        let total = self.fb.len() / MEL_BINS;
        self.bn_new.clear();
        let mut i = 0;
        while i + ASR_WINDOW <= total {
            let window = &self.fb[i * MEL_BINS..(i + ASR_WINDOW) * MEL_BINS];
            self.graphs
                .asr(window, self.state.asr_offset, &mut self.state.att, &mut self.state.cnn, &mut self.bn_call)?;
            self.bn_new.extend_from_slice(&self.bn_call);
            self.state.asr_offset += ASR_OFFSET_STEP;
            i += ASR_STRIDE;
        }
        self.state.frame_cache.clear();
        self.state
            .frame_cache
            .extend_from_slice(&self.fb[i * MEL_BINS..]);
        if self.bn_new.is_empty() {
            return Ok(());
        }
        let k = self.bn_new.len() / BN_DIM;
        self.bn_ctx.clear();
        self.bn_ctx.extend_from_slice(&self.state.enc_last);
        self.bn_ctx.extend_from_slice(&self.bn_new);
        let t_in = self.bn_ctx.len() / BN_DIM;
        self.state.enc_last.clear();
        self.state
            .enc_last
            .extend_from_slice(&self.bn_ctx[(t_in - 1) * BN_DIM..]);
        if t_in < 2 {
            return Ok(());
        }
        upsample_align_corners_drop_first(&self.bn_ctx, t_in, BN_DIM, UPSAMPLE * k + 1, &mut self.state.bn_buffer);
        Ok(())
    }

    /// Step 4: one DiT call over the first 8 buffered frames; its 4 mel frames go to `mel_frames`.
    fn dit_call(&mut self) -> Result<(), String> {
        self.noise
            .copy_window(self.state.frame, DIT_FRAMES, 1.0, false, &mut self.x);
        if self.state.vc_offset >= OFFSET_WRAP {
            self.state.vc_offset = 0;
            self.state.kv.fill(0.0);
            self.state.n_valid = 0;
        }
        let n = self.steps.max(1);
        let dt = (1.0 / f64::from(n)) as f32;
        for i in 0..n {
            let t = 1.0 - f64::from(i) * (1.0 / f64::from(n));
            let r = (t - 1.0 / f64::from(n)).max(0.0);
            let inputs = DitInputs {
                x: &self.x,
                t: t as f32,
                r: r as f32,
                dt,
                cond: &self.state.bn_buffer[..DIT_FRAMES * BN_DIM],
                offset: self.state.vc_offset,
                n_valid: self.state.n_valid,
                cache: &self.state.kv,
            };
            self.graphs
                .dit_step(&inputs, &self.speaker, &mut self.x_next, &mut self.cache_next)?;
            std::mem::swap(&mut self.x, &mut self.x_next);
        }
        // Every step read the call's starting cache; the last step's cache is the new one.
        std::mem::swap(&mut self.state.kv, &mut self.cache_next);
        self.state.n_valid = (self.state.n_valid + DIT_CHUNK as i64).min(self.kv_frames);
        self.state.vc_offset += DIT_CHUNK as i64;
        self.state.frame += DIT_CHUNK as u64;
        self.state.bn_buffer.drain(..DIT_CHUNK * BN_DIM);
        if self
            .x
            .iter()
            .take(DIT_CHUNK * MEL_BINS)
            .any(|v| !v.is_finite())
        {
            return Err("MeanVC2 dit_step produced non-finite mel frames".into());
        }
        self.mel_frames.clear();
        self.mel_frames
            .extend_from_slice(&self.x[..DIT_CHUNK * MEL_BINS]);
        Ok(())
    }

    /// Step 5: Vocos over the cached and new mel frames, crossfaded into `out`.
    fn decode(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        let cached = self.state.voc_cache.len() / MEL_BINS;
        let frames = cached + DIT_CHUNK;
        // Channel-major `(mel + 1) / 2` over [cache | new].
        self.mel_in.clear();
        self.mel_in.resize(MEL_BINS * frames, 0.0);
        for f in 0..frames {
            let src = if f < cached {
                &self.state.voc_cache[f * MEL_BINS..(f + 1) * MEL_BINS]
            } else {
                &self.mel_frames[(f - cached) * MEL_BINS..(f - cached + 1) * MEL_BINS]
            };
            for (c, &v) in src.iter().enumerate() {
                self.mel_in[c * frames + f] = (v + 1.0) / 2.0;
            }
        }
        // The new cache: the last 2 raw frames of [cache | new].
        let keep = VOC_OVERLAP.min(frames);
        self.voc_next.clear();
        for f in frames - keep..frames {
            let src = if f < cached {
                &self.state.voc_cache[f * MEL_BINS..(f + 1) * MEL_BINS]
            } else {
                &self.mel_frames[(f - cached) * MEL_BINS..(f - cached + 1) * MEL_BINS]
            };
            self.voc_next.extend_from_slice(src);
        }
        std::mem::swap(&mut self.state.voc_cache, &mut self.voc_next);
        self.graphs.vocos(&self.mel_in, frames, &mut self.wav)?;
        let len = self.wav.len();
        if len < 2 * WAV_OVERLAP {
            return Err(format!("MeanVC2 vocos returned {len} samples for {frames} frames"));
        }
        if self.wav.iter().any(|v| !v.is_finite()) {
            return Err("MeanVC2 vocos produced non-finite audio".into());
        }
        if self.state.last_wav.len() == WAV_OVERLAP {
            for i in 0..WAV_OVERLAP {
                out.push(self.state.last_wav[i] * self.down[i] + self.wav[i] * self.up[i]);
            }
            out.extend_from_slice(&self.wav[WAV_OVERLAP..len - WAV_OVERLAP]);
        } else {
            out.extend_from_slice(&self.wav[..len - WAV_OVERLAP]);
        }
        self.state.last_wav.clear();
        self.state
            .last_wav
            .extend_from_slice(&self.wav[len - WAV_OVERLAP..]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_torch_float32() {
        let down = linspace_f32(1.0, 0.0, 160);
        let up = linspace_f32(0.0, 1.0, 160);
        assert_eq!((down[0], down[159], up[0], up[159]), (1.0, 0.0, 0.0, 1.0));
        // torch.linspace(0, 1, 160)[1] and [158] in float32.
        assert_eq!(up[1], 1.0f32 / 159.0);
        assert_eq!(up[158], 1.0 - 1.0f32 / 159.0);
        for i in 0..160 {
            assert!((down[i] + up[i] - 1.0).abs() < 1e-6, "{i}");
        }
        assert!(linspace_f32(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace_f32(2.0, 5.0, 1), vec![2.0]);
    }

    #[test]
    fn upsampling_lands_on_quarter_frames_in_the_steady_state() {
        // [prev, b1, b2] (3 frames of width 1) -> 9 frames, first dropped: positions 0.25 .. 2.
        let src = [0.0f32, 4.0, 8.0];
        let mut dst = Vec::new();
        upsample_align_corners_drop_first(&src, 3, 1, 4 * 2 + 1, &mut dst);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Upstream's 160 ms chunk ([prev, b1..b4] -> 17) gives the same positions.
        let src = [0.0f32, 4.0, 8.0, 12.0, 16.0];
        let mut dst = Vec::new();
        upsample_align_corners_drop_first(&src, 5, 1, 17, &mut dst);
        assert_eq!(dst, (1..=16).map(|v| v as f32).collect::<Vec<_>>());
        // A cold stream's first call stretches its k frames over 4k (no previous frame).
        let mut dst = Vec::new();
        upsample_align_corners_drop_first(&[0.0, 8.0], 2, 1, 9, &mut dst);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Width > 1 keeps channels apart.
        let mut dst = Vec::new();
        upsample_align_corners_drop_first(&[0.0, 10.0, 4.0, 30.0], 2, 2, 3, &mut dst);
        assert_eq!(dst, vec![2.0, 20.0, 4.0, 30.0]);
    }
}
