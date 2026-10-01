//! Incremental F0 with a rolling cache (P3-C1): each tick runs the pitch extractor only over the
//! new block plus a margin (the extractor's own context), splices the result into a cache on the
//! 16 kHz / hop-160 grid keyed by absolute frame, and serves the whole context window from it.
//! Plan acceptance: the cached contour matches a whole-context extraction within 5 cents.
//! Legacy lever path (`f0_cache` off): the extractor runs over the whole context every tick.
//!
//! How a tick works (window = the context's `T` frames, absolute frames `[ws, we)`):
//! - frames the cache has never seen (the new block), plus the newest `margin_frames` it already
//!   has (they were extracted without the audio after them, so they are provisional), are
//!   re-extracted from a slice that also carries `lead_frames` of audio before them (the
//!   extractor's left context; those outputs are discarded);
//! - every older frame keeps the value it got when it had enough audio on both sides.
//!
//! With `margin_frames >= right` and `lead_frames >= left` of the extractor's receptive field
//! ([`PitchExtractor::receptive_frames`]) the cached contour is exactly what a whole-context
//! extraction gives on every frame at least `left` frames from the window's start (and there the
//! cache is *better*: it saw the audio before the window). Recurrent extractors (RMVPE's BiGRU)
//! see their whole input; their difference decays with distance and is measured by the benchmark.
//!
//! A gap, a jump back, a misaligned window or more new frames than the window holds falls back to a
//! whole-window extraction (the legacy path) for that tick.
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`F0CacheConfig`], [`F0Cache::new`], [`F0Cache::update`] (returns the F0 of the context
//!   window, one value per 10 ms frame, in a buffer the cache owns), [`F0Cache::reset`].
//!
//! # Added by P3-C1
//! [`F0Cache::with_lead_frames`], [`F0Cache::extract_window`] (the whole-window legacy path
//! aligned like the cache), [`F0Cache::last_extracted_frames`], [`DEFAULT_MARGIN_FRAMES`],
//! [`DEFAULT_LEAD_FRAMES`], [`HOP_16K`].

use super::super::traits::PitchExtractor;

/// 10 ms at 16 kHz: the F0 grid.
pub const HOP_16K: usize = 160;
/// Provisional frames re-extracted when the extractor does not say (covers dio / FCPE / RMVPE's
/// analysis windows and median smoothing with room to spare).
pub const DEFAULT_MARGIN_FRAMES: usize = 8;
/// Lead-in frames fed before the kept frames when the extractor does not say.
pub const DEFAULT_LEAD_FRAMES: usize = 12;

/// Cache geometry.
#[derive(Clone, Copy, Debug)]
pub struct F0CacheConfig {
    /// Frames re-extracted before the new block (the extractor's warm-up context).
    pub margin_frames: usize,
    /// Frames kept (at least the context window).
    pub capacity_frames: usize,
}

pub struct F0Cache {
    config: F0CacheConfig,
    lead_frames: usize,
    /// Cached F0, `frames[i]` = absolute frame `start + i`.
    frames: Vec<f32>,
    start: u64,
    /// Whole-window extraction scratch (the aligned contour of the last full pass).
    window: Vec<f32>,
    /// Frames the extractor analysed on the last update (for telemetry and tests).
    last_extracted: usize,
}

/// Fit a contour to `n` frames of the hop grid like RVC: a longer contour is cut to its first `n`
/// frames (`pitch[:, :p_len]`), a shorter one is right-aligned (missing oldest frames unvoiced).
/// Non-finite and negative values become unvoiced.
fn align_into(f0: &[f32], n: usize, out: &mut Vec<f32>) {
    out.clear();
    if f0.len() >= n {
        out.extend_from_slice(&f0[..n]);
    } else {
        out.resize(n - f0.len(), 0.0);
        out.extend_from_slice(f0);
    }
    for v in out.iter_mut() {
        if !(v.is_finite() && *v > 0.0) {
            *v = 0.0;
        }
    }
}

impl F0Cache {
    pub fn new(config: F0CacheConfig) -> F0Cache {
        let capacity = config.capacity_frames.max(1);
        F0Cache {
            config: F0CacheConfig {
                margin_frames: config.margin_frames,
                capacity_frames: capacity,
            },
            lead_frames: DEFAULT_LEAD_FRAMES,
            frames: Vec::with_capacity(capacity + 64),
            start: 0,
            window: Vec::with_capacity(capacity),
            last_extracted: 0,
        }
    }

    /// Audio before the re-extracted frames that the extractor sees but whose F0 is discarded.
    pub fn with_lead_frames(mut self, lead_frames: usize) -> F0Cache {
        self.lead_frames = lead_frames;
        self
    }

    pub fn config(&self) -> F0CacheConfig {
        self.config
    }

    pub fn lead_frames(&self) -> usize {
        self.lead_frames
    }

    /// Frames the extractor analysed on the last [`update`](Self::update) (the whole window on a
    /// fallback tick).
    pub fn last_extracted_frames(&self) -> usize {
        self.last_extracted
    }

    fn end(&self) -> u64 {
        self.start + self.frames.len() as u64
    }

    /// Whole-window extraction (the legacy path): the F0 of `mono_16k`'s `len / 160` frames,
    /// aligned like the cache, in the cache's own buffer. Does not touch the cache.
    pub fn extract_window(&mut self, extractor: &mut dyn PitchExtractor, mono_16k: &[f32]) -> Result<&[f32], String> {
        let n = mono_16k.len() / HOP_16K;
        let f0 = extractor.f0(mono_16k)?;
        align_into(&f0, n, &mut self.window);
        self.last_extracted = n;
        Ok(&self.window)
    }

    /// Extract F0 for the new part of `mono_16k` (whose first sample is absolute 16 kHz index
    /// `stream_pos_16k`) and return the F0 of the whole window.
    pub fn update(&mut self, extractor: &mut dyn PitchExtractor, mono_16k: &[f32], stream_pos_16k: u64) -> Result<&[f32], String> {
        let t = mono_16k.len() / HOP_16K;
        let aligned = stream_pos_16k.is_multiple_of(HOP_16K as u64) && mono_16k.len().is_multiple_of(HOP_16K);
        let ws = stream_pos_16k / HOP_16K as u64;
        let we = ws + t as u64;
        let incremental = aligned
            && t > 0
            && !self.frames.is_empty()
            && ws >= self.start
            && we >= self.end()
            && self.end() >= ws
            && (we - self.end()) as usize + self.config.margin_frames < t;
        if !incremental {
            // First tick, a gap, a jump or a misaligned call: the whole window (and it seeds the
            // cache when it is on the grid).
            let f0 = extractor.f0(mono_16k)?;
            align_into(&f0, t, &mut self.window);
            self.last_extracted = t;
            if aligned {
                self.frames.clear();
                self.frames.extend_from_slice(&self.window);
                self.start = ws;
                return Ok(&self.frames);
            }
            self.frames.clear();
            return Ok(&self.window);
        }
        // Frames to (re)compute: the new ones plus the provisional tail of the cache.
        let new = (we - self.end()) as usize;
        let redo = (new + self.config.margin_frames).min(t);
        let lead = self.lead_frames.min(t - redo);
        let n = redo + lead;
        let slice = &mono_16k[(t - n) * HOP_16K..];
        let f0 = extractor.f0(slice)?;
        align_into(&f0, n, &mut self.window);
        self.last_extracted = n;
        // Splice: drop the provisional frames, append the fresh ones.
        let keep = (we - redo as u64 - self.start) as usize;
        self.frames.truncate(keep);
        self.frames.extend_from_slice(&self.window[lead..]);
        // Trim to capacity (never below the window).
        let cap = self.config.capacity_frames.max(t);
        if self.frames.len() > cap {
            let drop = self.frames.len() - cap;
            self.frames.drain(..drop);
            self.start += drop as u64;
        }
        let from = (ws - self.start) as usize;
        Ok(&self.frames[from..from + t])
    }

    /// Forget the cached contour (stream gap, restart, extractor change).
    pub fn reset(&mut self) {
        self.frames.clear();
        self.start = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::world::DioPitchExtractor;

    /// Dio with its receptive field declared (±512-sample window, then a ±1-frame octave gate and a
    /// 3-tap median: at most 6 frames each side).
    struct Dio(DioPitchExtractor);

    impl PitchExtractor for Dio {
        fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
            self.0.f0(mono_16k)
        }
        fn receptive_frames(&self) -> Option<(usize, usize)> {
            Some((6, 6))
        }
    }

    /// Counts extracted samples; output frame j = mean of the input over a ±`w`-frame neighbourhood
    /// (a synthetic extractor with a known receptive field), so exactness is checkable bit for bit.
    struct Local {
        w: usize,
        samples: usize,
    }

    impl PitchExtractor for Local {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            self.samples += x.len();
            let n = x.len() / HOP_16K;
            Ok((0..n)
                .map(|j| {
                    let lo = j.saturating_sub(self.w) * HOP_16K;
                    let hi = ((j + self.w + 1) * HOP_16K).min(x.len());
                    let s: f64 = x[lo..hi].iter().map(|&v| f64::from(v.abs())).sum();
                    100.0 + (s / (hi - lo) as f64) as f32 * 100.0
                })
                .collect())
        }
    }

    /// A voiced signal with a gliding pitch, vibrato and unvoiced gaps (so V/UV matters).
    fn voice(len: usize) -> Vec<f32> {
        let mut phase = 0.0f64;
        (0..len)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let f = 110.0 + 90.0 * (t * 0.7).sin() + 6.0 * (t * 31.0).sin();
                phase += f / 16_000.0;
                let gate = if (t % 1.3) > 1.1 { 0.0 } else { 1.0 };
                (gate * (0.3 * (std::f64::consts::TAU * phase).sin() + 0.1 * (std::f64::consts::TAU * 2.0 * phase).sin())) as f32
            })
            .collect()
    }

    fn cents(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).log2().abs()
    }

    /// Plan acceptance: over a long stream, every tick's cached window matches a whole-context
    /// extraction of that tick's context within 5 cents (identical V/UV) away from the window's
    /// left edge, while the extractor analyses only the new block plus the margins.
    #[test]
    fn cached_f0_matches_whole_context_extraction_within_5_cents() {
        let (ctx_frames, block) = (60usize, 4usize);
        let audio = voice(16_000 * 8);
        let dio = Dio(DioPitchExtractor::new());
        let (lead, margin) = dio.receptive_frames().unwrap();
        let mut ex = dio;
        let mut reference = Dio(DioPitchExtractor::new());
        let mut cache = F0Cache::new(F0CacheConfig {
            margin_frames: margin,
            capacity_frames: ctx_frames * 2,
        })
        .with_lead_frames(lead);
        let mut worst = 0.0f32;
        let mut voiced = 0usize;
        let mut extracted = 0usize;
        let mut ticks = 0usize;
        let mut pos = 0usize;
        while pos + ctx_frames * HOP_16K <= audio.len() {
            let ctx = &audio[pos..pos + ctx_frames * HOP_16K];
            let cached = cache.update(&mut ex, ctx, pos as u64).unwrap().to_vec();
            if ticks > 0 {
                extracted += cache.last_extracted_frames();
            }
            let mut full = reference.f0(ctx).unwrap();
            full.truncate(ctx_frames);
            assert_eq!(cached.len(), ctx_frames);
            for j in lead..ctx_frames {
                let (a, b) = (cached[j], full[j]);
                assert_eq!(a > 0.0, b > 0.0, "tick {ticks} frame {j}: V/UV {a} vs {b}");
                if a > 0.0 {
                    voiced += 1;
                    worst = worst.max(cents(a, b));
                }
            }
            pos += block * HOP_16K;
            ticks += 1;
        }
        assert!(voiced > 5_000, "{voiced} voiced frames compared");
        assert!(worst <= 5.0, "worst difference {worst} cents");
        // The extractor saw block + margin + lead frames per tick, not the whole context.
        assert_eq!(extracted, (ticks - 1) * (block + margin + lead));
        assert!(((block + margin + lead) as f32) < 0.5 * ctx_frames as f32);
    }

    /// With a declared receptive field the cache is bit-exact against a whole-context extraction
    /// (outside the left edge), for any block and margin that covers the field; a gap or a jump
    /// falls back to a whole-window extraction.
    #[test]
    fn local_extractor_is_exact_and_gaps_fall_back() {
        let audio = voice(16_000 * 3);
        for &(ctx_frames, block, w) in &[(40usize, 2usize, 3usize), (64, 8, 5), (30, 6, 2)] {
            let mut ex = Local { w, samples: 0 };
            let mut reference = Local { w, samples: 0 };
            let mut cache = F0Cache::new(F0CacheConfig {
                margin_frames: w,
                capacity_frames: ctx_frames,
            })
            .with_lead_frames(w);
            let mut pos = 0usize;
            let mut tick = 0;
            while pos + ctx_frames * HOP_16K <= audio.len() {
                let ctx = &audio[pos..pos + ctx_frames * HOP_16K];
                let cached = cache.update(&mut ex, ctx, pos as u64).unwrap().to_vec();
                let full = reference.f0(ctx).unwrap();
                assert_eq!(&cached[w..], &full[w..], "ctx {ctx_frames} block {block} tick {tick}");
                pos += block * HOP_16K;
                tick += 1;
            }
            // A jump forward larger than the window: whole-window extraction.
            let before = ex.samples;
            let ctx = &audio[..ctx_frames * HOP_16K];
            cache
                .update(&mut ex, ctx, (pos + 100 * HOP_16K) as u64)
                .unwrap();
            assert_eq!(ex.samples - before, ctx.len());
            assert_eq!(cache.last_extracted_frames(), ctx_frames);
            // Back in time: whole window again. Misaligned: whole window, cache not seeded.
            cache.update(&mut ex, ctx, 0).unwrap();
            assert_eq!(cache.last_extracted_frames(), ctx_frames);
            let got = cache.update(&mut ex, ctx, 7).unwrap().to_vec();
            assert_eq!(got.len(), ctx_frames);
            assert_eq!(cache.last_extracted_frames(), ctx_frames);
            cache.reset();
            cache.update(&mut ex, ctx, 0).unwrap();
            assert_eq!(cache.last_extracted_frames(), ctx_frames);
        }
    }

    /// An extractor fed silence before its slice so the slice ends a whole `frames`-frame block of
    /// mel frames; the padded frames are dropped from the result. `frames` 64 with slices longer
    /// than 63 frames passes them as is. (RMVPE end-aligns its own mel since the P3 final gate, with
    /// the log-mel floor before the audio; these rows pad with silent audio instead, at the plain
    /// path's 32-frame and the fixed-shape plan's 64-frame sizes.)
    struct EndAligned {
        inner: Box<dyn PitchExtractor>,
        frames: usize,
    }

    impl PitchExtractor for EndAligned {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            let n = x.len() / HOP_16K;
            // `n` hops give `n + 1` mel frames; pad to the next multiple of `frames`.
            let target = (n + 1).div_ceil(self.frames) * self.frames - 1;
            let pad = target - n;
            let mut padded = vec![0.0f32; pad * HOP_16K];
            padded.extend_from_slice(x);
            let mut f0 = self.inner.f0(&padded)?;
            f0.drain(..pad.min(f0.len()));
            Ok(f0)
        }
    }

    /// Dev-rig sweep (P3-C1 fixer; review finding on T5): how the cache's lead-in (audio before the
    /// re-extracted frames) and lookahead change the streamed F0 of extractors that declare no
    /// receptive field (RMVPE's BiGRU, FCPE), against each clip's whole-clip F0. 40 ms blocks,
    /// each frame taken from the tick that decodes it (the engine's rule), contexts of 44 (the
    /// Low-latency / Quality plans) and 64 frames. Prints voiced-frame RMSE in cents (all frames and
    /// those within 600 cents), the V/UV disagreement and the median call time. Also scores RMVPE
    /// fed silent audio before the slice (the measurement behind RMVPE's end-aligned mel; before
    /// the P3 final gate RMVPE zero-padded its log-mel after the audio).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache` (the 16 kHz `sherpa-onnx/clips/*.wav`
    /// and the pitch models), optional `SIGIL_AB_EP` (default `cpu`):
    /// `cargo test --manifest-path src-tauri/Cargo.toml lead_sweep_against_whole_clip_f0 -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig sweep: needs SIGIL_TEST_MODELS_DIR with the pitch models and the speech clips"]
    fn lead_sweep_against_whole_clip_f0() {
        use crate::services::neural_audio::ort_session::VcSessionOptions;
        use crate::services::neural_audio::vc::f0_fusion::test_model;
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return;
        };
        let ep = std::env::var("SIGIL_AB_EP").unwrap_or_else(|_| "cpu".into());
        let mut clips: Vec<(String, Vec<f32>)> = Vec::new();
        for name in ["en", "ja", "ko", "ru", "uk", "yue", "zh"] {
            let path = root.join("sherpa-onnx").join("clips").join(format!("{name}.wav"));
            let Ok(mut r) = hound::WavReader::open(&path) else { continue };
            let spec = r.spec();
            if (spec.sample_rate, spec.channels) != (16_000, 1) {
                continue;
            }
            let x: Vec<f32> = r
                .samples::<i32>()
                .map(|v| v.unwrap() as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
                .collect();
            clips.push((name.to_string(), x));
        }
        assert!(!clips.is_empty(), "no 16 kHz clips under {}", root.display());
        let opts = VcSessionOptions::default();
        let mut extractors: Vec<(String, Box<dyn PitchExtractor>)> = Vec::new();
        if let Some(p) = test_model("pitch-rmvpe") {
            let e = crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor::load(&p.to_string_lossy(), &ep, &opts).unwrap();
            extractors.push((format!("rmvpe ({ep})"), Box::new(e)));
        }
        if let Some(p) = test_model("pitch-rmvpe") {
            // What end-aligning RMVPE's mel would give (the padding before the audio, as silence,
            // instead of zero log-mel after it): to whole 32-frame blocks, and to a fixed 64-frame
            // plan (the CUDA-graph plan / DirectML pin built at the whole window's length).
            for frames in [32usize, 64] {
                let e = crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor::load(&p.to_string_lossy(), &ep, &opts).unwrap();
                extractors.push((format!("rmvpe ({ep}) end-aligned {frames}"), Box::new(EndAligned { inner: Box::new(e), frames })));
            }
        }
        if let Some(p) = test_model("pitch-fcpe") {
            let e = crate::services::neural_audio::vc::fcpe::FcpePitchExtractor::load(&p.to_string_lossy(), &ep, &opts).unwrap();
            extractors.push((format!("fcpe ({ep})"), Box::new(e)));
        }
        let block = 4usize;
        eprintln!("| extractor | ctx | lead | lookahead | slice frames | RMSE cents | RMSE (<600 c) | V/UV % | call ms p50 |");
        eprintln!("|---|---|---|---|---|---|---|---|---|");
        // `SIGIL_SWEEP_ONLY=<substring>` limits the run to matching extractors.
        let only = std::env::var("SIGIL_SWEEP_ONLY").unwrap_or_default();
        for (name, ex) in extractors.iter_mut().filter(|(n, _)| n.contains(only.as_str())) {
            let offline: Vec<Vec<f32>> = clips
                .iter()
                .map(|(_, x)| {
                    let mut f = ex.f0(x).unwrap();
                    f.resize(x.len() / HOP_16K, 0.0);
                    f
                })
                .collect();
            for ctx in [44usize, 64] {
                let max_lead = ctx - block - DEFAULT_MARGIN_FRAMES;
                let mut leads: Vec<usize> = if name.contains("end-aligned") {
                    vec![DEFAULT_LEAD_FRAMES]
                } else {
                    [DEFAULT_LEAD_FRAMES, 19, 20, 28, max_lead - 1, max_lead].into_iter().filter(|&l| l <= max_lead).collect()
                };
                leads.dedup();
                for &lead in &leads {
                    for la in [0usize, 2] {
                        let (mut se, mut se_ok, mut n_v, mut n_ok, mut vuv, mut n_all) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize, 0usize);
                        let mut times = Vec::new();
                        for ((_, x), reference) in clips.iter().zip(&offline) {
                            let total = x.len() / HOP_16K;
                            let mut cache = F0Cache::new(F0CacheConfig {
                                margin_frames: DEFAULT_MARGIN_FRAMES,
                                capacity_frames: ctx * 2,
                            })
                            .with_lead_frames(lead);
                            let mut est = vec![f32::NAN; total];
                            let mut end = ctx * HOP_16K;
                            while end <= x.len() {
                                let t0 = std::time::Instant::now();
                                let w = cache.update(ex.as_mut(), &x[end - ctx * HOP_16K..end], (end - ctx * HOP_16K) as u64).unwrap();
                                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                                let w0 = end / HOP_16K - ctx;
                                let last = (end / HOP_16K - la).min(total);
                                let first = last - block;
                                est[first..last].copy_from_slice(&w[first - w0..last - w0]);
                                end += block * HOP_16K;
                            }
                            for f in ctx..total {
                                let (e, r) = (est[f], reference[f]);
                                if e.is_nan() {
                                    continue;
                                }
                                n_all += 1;
                                if (e > 0.0) != (r > 0.0) {
                                    vuv += 1;
                                } else if e > 0.0 {
                                    let c = f64::from(1200.0 * (e / r).log2());
                                    se += c * c;
                                    n_v += 1;
                                    if c.abs() < 600.0 {
                                        se_ok += c * c;
                                        n_ok += 1;
                                    }
                                }
                            }
                        }
                        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        eprintln!(
                            "| {name} | {ctx} | {lead} | {} ms | {} | {:.1} | {:.1} | {:.2} | {:.2} |",
                            la * 10,
                            lead + block + DEFAULT_MARGIN_FRAMES,
                            (se / n_v.max(1) as f64).sqrt(),
                            (se_ok / n_ok.max(1) as f64).sqrt(),
                            100.0 * vuv as f64 / n_all.max(1) as f64,
                            times.get(times.len() / 2).copied().unwrap_or(0.0),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn contours_are_aligned_like_rvc() {
        let mut out = Vec::new();
        align_into(&[1.0, 2.0], 4, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, 2.0]);
        align_into(&[1.0, 2.0, 3.0, 4.0, 5.0], 4, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        align_into(&[f32::NAN, -1.0, 3.0], 3, &mut out);
        assert_eq!(out, [0.0, 0.0, 3.0]);
    }
}
