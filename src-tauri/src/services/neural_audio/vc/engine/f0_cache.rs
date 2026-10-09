//! Incremental F0 with a rolling cache (P3-C1): each tick runs the pitch extractor only over the
//! new block plus a margin (the extractor's own context), splices the result into a cache on the
//! 16 kHz / hop-160 grid keyed by absolute frame, and serves the whole context window from it.
//! Plan acceptance: the cached contour matches a whole-context extraction within 5 cents.
//! Legacy lever path (`f0_cache` off): the extractor runs over the whole context every tick.
//!
//! How a tick works (window = the context's `T` frames, absolute frames `[ws, we)`):
//! - frames the cache has never seen (the new block), plus the newest `margin_frames` it already
//!   has (they were extracted without the audio after them, so they are provisional), are
//!   re-extracted from a slice that also carries a **lead** of audio before them (the extractor's
//!   left context; those outputs are discarded);
//! - every older frame keeps the value it got when it had enough audio on both sides (it is
//!   committed: the generator was fed it).
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
//! # Streaming F0 v2 (P3B-F)
//! Each part is a switch (one executable A/Bs it, rule R5); [`F0Geometry`] holds them, read from
//! the environment when the cache is built. All three ship OFF ([`F0Geometry::DEFAULT`] is P3's
//! cache; rule R6: priming and continuity change a few fed frames and have no per-frame or FCPE
//! evidence that quality is not lower, see the constant; filled calls lower the speaker cosine).
//! [`F0Geometry::PRIMED`] and [`F0Geometry::FILLED`] are the opt-in presets:
//! - **Filled calls** (opt-in: [`LeadMode::Fill`], `SIGIL_VC_F0_LEAD=fill|legacy|<frames>` or
//!   `SIGIL_VC_F0_GEOMETRY=filled`, which is the measured combination with priming and continuity;
//!   `SIGIL_VC_F0_LEAD=fill` alone changes only the lead): an extractor with a fixed input shape
//!   ([`PitchExtractor::preferred_input_frames`]: RMVPE's plan, or the 32-frame padding of its plain
//!   session) pays for the whole shape on every call, so the lead can be whatever fills it:
//!   `preferred - redo` frames of real audio (19 on a 32-frame plan, 51 on the 64-frame default plan
//!   at 40 ms blocks) instead of 12 frames plus padding. On a plain session (no
//!   [`PitchExtractor::max_call_frames`]) the call is filled to the padded size it already pays for,
//!   never past it. An **audio history** of the stream reaches past the context window when the
//!   lead needs it (the window is 44 frames at Low latency). Off by default (rule R6; P3B-I0 kept it
//!   off: speaker cosine -0.021 at Low latency). Why it helps: on the 64-frame plan the 24-frame incremental call sits behind 40 frames
//!   of the extractor's silence floor (its end-aligned padding), and fill replaces the floor with real
//!   audio. In the engine (DirectML, 8 real speech clips x 3 offsets, scorer v2, fed F0 against
//!   whole-clip RMVPE of the input) it improves every pitch row at Low latency: fine error 23.6 ->
//!   17.8 cents (CI -9.8 .. -1.8), V/UV 3.99 -> 2.33 % (better on 8 of 8 clips), GPE 0.71 -> 0.51 %
//!   (within noise); at the Quality preset 21.5 -> 15.5 cents and 3.83 -> 2.15 % (8 of 8). MCD and
//!   DNSMOS are unchanged within noise; the advisory speaker cosine against the ceilings is lower
//!   (-0.021 at Low latency, CI -0.044 .. +0.002, -0.006 at Quality, both within noise), mostly on
//!   the tonal clips (yue, zh). P3B-E's 32-frame plan gets most of the same gain without it (23.6 ->
//!   18.7 cents, V/UV 2.97 %: 24 of its 32 frames are real audio) and fill adds nothing measurable on
//!   top of it, on clean clips or on copies at 10 dB SNR. On the CPU sweep
//!   (`tests::lead_sweep_against_whole_clip_f0`) fill moves the stream away from the exact truth of
//!   the synthetic corpora on the 64-frame plan (held-out: fine error 13.3 -> 16.8 cents, GPE 6.2 ->
//!   6.7 %, V/UV 4.65 -> 5.17 %; worse under noise, creak and missing fundamentals, neutral on clean
//!   speech), and on the 32-frame plan it trades gross errors (creak 21 -> 15 %, SNR 5 dB 16 -> 4.5 %)
//!   for voicing errors (SNR 10 dB 4.1 -> 6.3 %). It costs no VRAM and no GPU time on a fixed-shape
//!   plan. The pitch fusion does not forward `preferred_input_frames` (its SwiftF0 would get the same
//!   long calls), so fill applies to the plain RMVPE method.
//! - **Primed cold starts** (opt-in: `SIGIL_VC_F0_PRIME=1`): a first tick, gap or jump on an
//!   extractor whose calls are bounded ([`PitchExtractor::max_call_frames`]; P3B-E's 32-frame plan
//!   holds 31 audio frames) is extracted as overlapping slices of at most that length laid out like
//!   steady-state ticks (each slice after the first carries the lead, each before the last leaves
//!   its newest `margin_frames` to the next), so no call is ever longer than the extractor's shape.
//! - **Silence continuity** (opt-in: `SIGIL_VC_F0_CONTINUITY=1`): a tick skipped as silence
//!   appends its frames to the cache as unvoiced at their absolute positions
//!   ([`F0Cache::advance_unvoiced`]), so the first voiced tick after a pause is a normal incremental
//!   call (its margin frames re-extracted from the real audio) instead of a whole-window call.
//!   Neutral on the CPU sweeps (the same fine error and GPE; V/UV within 0.03 pt on both corpora),
//!   with fewer whole-window calls at onsets (on the 32-frame plan the calls longer than the plan
//!   fall from 16 to 7 over the 7 clips without priming, to 0 with it). On clips with 1.2 s pauses
//!   (`tests::pauses_do_not_change_the_fed_f0`, CPU RMVPE) 3 to 22 of about 5,000 fed frames differ
//!   from P3's cache (one by an octave; `multi-3` at the 64-frame plan: GPE 2.012 -> 2.117 %), and
//!   the engine's per-clip medians over offsets, which hide isolated frames, are bit-identical.
//!   That is why both parts stay off until P3B-I0 has per-frame evidence for RMVPE and FCPE.
//! - **Cross-tick decoding** (the `f0_decode` lever, [`F0Cache::set_decode`]; off by default;
//!   `SIGIL_VC_F0_DECODE=1|0` overrides it): the span each tick re-extracts is decoded again from
//!   the extractor's salience, continuing the committed frame before it (`vc::f0_decode`). Only the
//!   re-extracted span changes; committed frames never do. Never for singing.
//!
//! `SIGIL_VC_F0_GEOMETRY=legacy|primed|filled` sets the first three at once to P3's cache (the
//! default), priming and continuity, or those plus filled calls (the individual variables override
//! it). Extractor calls go through [`PitchExtractor::f0_into`], so the cache itself allocates
//! nothing per tick once its buffers have grown (the load's warm-up ticks).
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`F0CacheConfig`], [`F0Cache::new`], [`F0Cache::update`] (returns the F0 of the context
//!   window, one value per 10 ms frame, in a buffer the cache owns), [`F0Cache::reset`].
//!
//! # Added by P3-C1
//! [`F0Cache::with_lead_frames`], [`F0Cache::extract_window`] (the whole-window legacy path
//! aligned like the cache), [`F0Cache::last_extracted_frames`], [`DEFAULT_MARGIN_FRAMES`],
//! [`DEFAULT_LEAD_FRAMES`], [`HOP_16K`].
//!
//! # Frozen by P3B-S0 (signature; P3B-F implements)
//! [`F0Cache::advance_unvoiced`]`(abs_start_frame, frames)`: called by the tick's silence-skip path.
//!
//! # Added by P3B-F
//! [`F0Geometry`] (with [`F0Geometry::from_env_over`]: a preset's base plus the switches),
//! [`LeadMode`], [`F0Cache::with_geometry`], [`F0Cache::geometry`],
//! [`F0Cache::set_decode`], [`F0Cache::with_decode_params`], [`F0Cache::decoding`],
//! [`F0Cache::decode_note`], [`F0Cache::last_lead_frames`], [`F0Cache::last_call_frames`],
//! [`F0Cache::calls`], [`F0Cache::long_calls`], [`process_call_counts`] and the environment
//! switches.

use std::sync::atomic::{AtomicU64, Ordering};

use super::super::f0_decode::{self, DecodeParams, F0Decoder};
use super::super::pitch_onnx::SALIENCE_BINS;
use super::super::traits::PitchExtractor;
use super::options::F0Range;

/// 10 ms at 16 kHz: the F0 grid.
pub const HOP_16K: usize = 160;
/// Provisional frames re-extracted when the extractor does not say (covers dio / FCPE / RMVPE's
/// analysis windows and median smoothing with room to spare).
pub const DEFAULT_MARGIN_FRAMES: usize = 8;
/// Lead-in frames fed before the kept frames when the extractor does not say.
pub const DEFAULT_LEAD_FRAMES: usize = 12;
/// The shortest lead a filled call keeps; a block so long that a bounded extractor's call would
/// carry less is extracted with the cache's own lead (the extractor splits that call).
pub const MIN_FILL_LEAD_FRAMES: usize = 8;
/// The longest audio history the cache keeps (frames): RMVPE's longest plan (256 mel frames).
const MAX_HISTORY_FRAMES: usize = 256;

/// `SIGIL_VC_F0_LEAD=fill|legacy|<frames>`: how incremental calls are sized ([`LeadMode`]).
pub const F0_LEAD_ENV: &str = "SIGIL_VC_F0_LEAD";
/// `SIGIL_VC_F0_PRIME=0|1`: primed cold starts ([`F0Geometry::prime`]; off unless `1`).
pub const F0_PRIME_ENV: &str = "SIGIL_VC_F0_PRIME";
/// `SIGIL_VC_F0_CONTINUITY=0|1`: silence-skipped ticks append unvoiced frames
/// ([`F0Geometry::continuity`]; off unless `1`).
pub const F0_CONTINUITY_ENV: &str = "SIGIL_VC_F0_CONTINUITY";
/// `SIGIL_VC_F0_GEOMETRY=legacy|primed|filled`: P3's cache (all three above off, the default),
/// priming and continuity on, or those plus filled calls; the individual variables override it.
pub const F0_GEOMETRY_ENV: &str = "SIGIL_VC_F0_GEOMETRY";

/// Cache geometry.
#[derive(Clone, Copy, Debug)]
pub struct F0CacheConfig {
    /// Frames re-extracted before the new block (the extractor's warm-up context).
    pub margin_frames: usize,
    /// Frames kept (at least the context window).
    pub capacity_frames: usize,
}

/// How an incremental call's lead (audio before the re-extracted frames) is sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeadMode {
    /// The cache's own lead (`with_lead_frames`; P3: 12 frames, or the extractor's receptive field).
    Legacy,
    /// Fill the extractor's fixed input shape with real audio (module docs).
    Fill,
    /// Exactly this many frames, whatever the extractor (sweeps; the audio history still bounds it).
    Frames(usize),
}

/// The P3B-F parts of the cache and their switches (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F0Geometry {
    pub lead: LeadMode,
    /// Prime cold starts, gaps and jumps of bounded extractors with steady-state slices.
    pub prime: bool,
    /// Silence-skipped ticks append unvoiced frames ([`F0Cache::advance_unvoiced`]).
    pub continuity: bool,
}

impl F0Geometry {
    /// P3's cache (before P3B-F).
    pub const LEGACY: F0Geometry = F0Geometry {
        lead: LeadMode::Legacy,
        prime: false,
        continuity: false,
    };
    /// The default: P3's cache. Primed cold starts and silence continuity ship OFF (rule R6: a
    /// change that can lower fidelity is a switch that defaults to off until P3B-I0 flips it on
    /// interleaved, per-frame evidence). The evidence so far is not "no regression":
    /// - on the pause clips (`tests::pauses_do_not_change_the_fed_f0`, CPU RMVPE) 3 to 22 of about
    ///   5,000 fed frames differ from P3's cache, one of them by an octave (1216 cents), and on
    ///   `multi-3` at the 64-frame plan the gross-error rate rises 2.012 -> 2.117 % and V/UV
    ///   5.02 -> 5.11 % (the other rows move by 0.01 cents or 0.002 pt at most);
    /// - the engine A/Bs that read "bit-identical" compare per-clip medians over three offsets,
    ///   which hide isolated frames ("no regression at the scorer's granularity", not "no
    ///   change");
    /// - every measurement used RMVPE; none used FCPE (the app's default pitch extractor), which
    ///   the silence-skip path ([`F0Cache::advance_unvoiced`]) reaches too.
    ///
    /// What they buy is structural, not quality: no extractor call longer than its plan (priming)
    /// and no whole-window call at an onset (continuity). Priming is a no-op on the default
    /// 64-frame plan at the 44-frame Low latency window (44 <= 63); it matters on the opt-in
    /// 32-frame plan (`SIGIL_RMVPE_PLAN_FRAMES=32`), where a user or preset that wants the saver's
    /// call shape takes [`PRIMED`](Self::PRIMED) (`SIGIL_VC_F0_PRIME=1`, `SIGIL_VC_F0_CONTINUITY=1`).
    pub const DEFAULT: F0Geometry = F0Geometry::LEGACY;
    /// P3B-F's primed cold starts and silence continuity with P3's lead (opt-in, see
    /// [`DEFAULT`](Self::DEFAULT)).
    pub const PRIMED: F0Geometry = F0Geometry {
        lead: LeadMode::Legacy,
        prime: true,
        continuity: true,
    };
    /// [`PRIMED`](Self::PRIMED) with filled calls (`SIGIL_VC_F0_LEAD=fill`; opt-in, module docs).
    pub const FILLED: F0Geometry = F0Geometry {
        lead: LeadMode::Fill,
        prime: true,
        continuity: true,
    };

    /// The geometry of this process: [`F0Geometry::DEFAULT`] unless the environment switches say
    /// otherwise (module docs). Reads the environment: call it at load time, not per tick.
    pub fn from_env() -> F0Geometry {
        F0Geometry::from_env_over(F0Geometry::DEFAULT)
    }

    /// `base` (a preset's choice, e.g. [`F0Geometry::FILLED`] for a profile that wants filled calls)
    /// with the environment switches applied over it: `SIGIL_VC_F0_GEOMETRY` replaces the base
    /// wholesale, the individual variables override single parts, so the kill switches keep working
    /// whatever the base is. Load time only.
    pub fn from_env_over(base: F0Geometry) -> F0Geometry {
        F0Geometry::resolve(base, |name| std::env::var(name).ok())
    }

    /// [`from_env_over`](Self::from_env_over) with the variables read through `get`.
    fn resolve(base: F0Geometry, get: impl Fn(&str) -> Option<String>) -> F0Geometry {
        let var = |name: &str| get(name).map(|v| v.trim().to_ascii_lowercase());
        let flag = |name: &str| match var(name).as_deref() {
            Some("0" | "off" | "false" | "no") => Some(false),
            Some("1" | "on" | "true" | "yes") => Some(true),
            _ => None,
        };
        let mut g = match var(F0_GEOMETRY_ENV).as_deref() {
            Some("legacy" | "p3" | "off" | "0") => F0Geometry::LEGACY,
            Some("primed" | "v2") => F0Geometry::PRIMED,
            Some("filled" | "fill") => F0Geometry::FILLED,
            _ => base,
        };
        match var(F0_LEAD_ENV).as_deref() {
            Some("legacy" | "p3" | "off" | "0") => g.lead = LeadMode::Legacy,
            Some("fill" | "on" | "1") => g.lead = LeadMode::Fill,
            Some(v) => {
                if let Ok(n) = v.parse::<usize>() {
                    g.lead = LeadMode::Frames(n.min(MAX_HISTORY_FRAMES));
                }
            }
            None => {}
        }
        if let Some(on) = flag(F0_PRIME_ENV) {
            g.prime = on;
        }
        if let Some(on) = flag(F0_CONTINUITY_ENV) {
            g.continuity = on;
        }
        g
    }

    /// Whether incremental calls may reach past the context window (the audio history is kept).
    fn needs_history(&self) -> bool {
        self.lead != LeadMode::Legacy
    }
}

/// The cross-tick decoder's state (module docs; `vc::f0_decode`).
#[derive(Debug, Default)]
struct DecodeState {
    /// The `f0_decode` lever and the voice's range ([`F0Cache::set_decode`]).
    lever: bool,
    range: F0Range,
    /// `SIGIL_VC_F0_DECODE` (read when the cache is built).
    env: Option<bool>,
    /// Explicit parameters (sweeps and tests), over the range's.
    explicit: Option<Option<DecodeParams>>,
    /// The parameters in effect (`None` = not decoding).
    params: Option<DecodeParams>,
    decoder: F0Decoder,
    /// Spans decoded, and spans left as extracted because the extractor had no usable salience.
    decoded: u64,
    no_salience: u64,
}

impl DecodeState {
    fn resolve(&mut self) {
        self.params = match self.explicit {
            Some(p) => p,
            None if self.env.unwrap_or(self.lever) => DecodeParams::for_range(self.range),
            None => None,
        };
    }
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
    // ---- P3B-F ----
    geometry: F0Geometry,
    /// Stream audio of absolute frames `[audio_start, audio_start + audio.len() / HOP_16K)`,
    /// contiguous and ending at the last window's end (empty: none); kept when the lead may reach
    /// past the window.
    audio: Vec<f32>,
    audio_start: u64,
    /// One extractor call's contour (`f0_into`; capacity kept).
    call: Vec<f32>,
    /// The lead of the last incremental call, its longest call, and call counts.
    last_lead: usize,
    last_call: usize,
    calls: u64,
    long_calls: u64,
    decode: DecodeState,
}

/// Extractor calls every F0 cache of the process has made, and those longer than the extractor's
/// [`PitchExtractor::max_call_frames`] at the time (calls it had to split). Relaxed counters for the
/// benchmark: it reads them before and after a run, and a run on a bounded plan with priming on
/// (P3B-E's 32-frame plan) must not move the second one. The engine's own caches are swapped
/// between threads, so per-cache counts ([`F0Cache::long_calls`]) cannot be read from outside.
static CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
static LONG_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// `(calls, long calls)` of every F0 cache of this process since it started (see [`CALLS_TOTAL`]).
pub fn process_call_counts() -> (u64, u64) {
    (CALLS_TOTAL.load(Ordering::Relaxed), LONG_CALLS_TOTAL.load(Ordering::Relaxed))
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

/// Decode the cached frames `[from, from + len)` from the extractor's salience rows
/// `[row0, row0 + len)` of its last call (when the decoder is on and the salience fits; module
/// docs). The committed frame before `from` (if any) anchors the path.
fn decode_span(decode: &mut DecodeState, extractor: &dyn PitchExtractor, frames: &mut [f32], from: usize, row0: usize, len: usize) {
    let Some(params) = decode.params else { return };
    if len == 0 {
        return;
    }
    // A local extractor (a declared receptive field) is already exact in the cache, and its voicing
    // is not a salience threshold (CREPE's periodicity filters): only whole-input extractors decode.
    let salience = if extractor.receptive_frames().is_none() {
        extractor
            .last_salience()
            .filter(|(s, bins)| *bins == SALIENCE_BINS && s.len() >= (row0 + len) * bins)
    } else {
        None
    };
    let Some((salience, bins)) = salience else {
        decode.no_salience += 1;
        return;
    };
    let anchor = (from > 0).then(|| frames[from - 1]);
    if decode.decoder.decode(
        anchor,
        &mut frames[from..from + len],
        &salience[row0 * bins..(row0 + len) * bins],
        &params,
    ) {
        decode.decoded += 1;
    }
}

impl F0Cache {
    pub fn new(config: F0CacheConfig) -> F0Cache {
        let capacity = config.capacity_frames.max(1);
        let mut decode = DecodeState {
            env: f0_decode::decode_env_override(),
            ..DecodeState::default()
        };
        decode.resolve();
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
            geometry: F0Geometry::from_env(),
            audio: Vec::new(),
            audio_start: 0,
            call: Vec::with_capacity(capacity + 1),
            last_lead: 0,
            last_call: 0,
            calls: 0,
            long_calls: 0,
            decode,
        }
    }

    /// Audio before the re-extracted frames that the extractor sees but whose F0 is discarded (the
    /// lead of [`LeadMode::Legacy`], and the least a filled call carries).
    pub fn with_lead_frames(mut self, lead_frames: usize) -> F0Cache {
        self.lead_frames = lead_frames;
        self
    }

    /// The P3B-F parts in effect (default: [`F0Geometry::from_env`]).
    pub fn with_geometry(mut self, geometry: F0Geometry) -> F0Cache {
        self.geometry = geometry;
        self
    }

    pub fn geometry(&self) -> F0Geometry {
        self.geometry
    }

    /// The `f0_decode` lever of the voice and its F0 range (load time; the engine calls it when it
    /// builds the cache). Singing never decodes; `SIGIL_VC_F0_DECODE` overrides the lever.
    pub fn set_decode(&mut self, lever: bool, range: F0Range) {
        self.decode.lever = lever;
        self.decode.range = range;
        self.decode.resolve();
    }

    /// Decode with exactly these parameters (`None`: never), whatever the lever, range and
    /// environment (sweeps and tests).
    pub fn with_decode_params(mut self, params: Option<DecodeParams>) -> F0Cache {
        self.decode.explicit = Some(params);
        self.decode.resolve();
        self
    }

    /// Whether ticks decode their span (the lever, or its override, for a speech voice).
    pub fn decoding(&self) -> bool {
        self.decode.params.is_some()
    }

    /// Why decoding is asked for but does nothing (after a few updates): the extractor has no
    /// salience to decode (only RMVPE-like extractors do). `None` while it decodes or is off.
    pub fn decode_note(&self) -> Option<&'static str> {
        (self.decoding() && self.decode.decoded == 0 && self.decode.no_salience > 0)
            .then_some("f0_decode: the F0 method has no salience to decode (RMVPE does); each call's own contour is used")
    }

    pub fn config(&self) -> F0CacheConfig {
        self.config
    }

    pub fn lead_frames(&self) -> usize {
        self.lead_frames
    }

    /// Frames the extractor analysed on the last [`update`](Self::update) (the whole window on a
    /// fallback tick; the sum of the slices on a primed one).
    pub fn last_extracted_frames(&self) -> usize {
        self.last_extracted
    }

    /// The lead of the last incremental call (frames of audio before the re-extracted ones).
    pub fn last_lead_frames(&self) -> usize {
        self.last_lead
    }

    /// The longest single extractor call of the last update (frames).
    pub fn last_call_frames(&self) -> usize {
        self.last_call
    }

    /// Extractor calls since the cache was built, and those longer than the extractor's
    /// [`PitchExtractor::max_call_frames`] at the time (calls the extractor must split).
    pub fn calls(&self) -> u64 {
        self.calls
    }

    pub fn long_calls(&self) -> u64 {
        self.long_calls
    }

    fn end(&self) -> u64 {
        self.start + self.frames.len() as u64
    }

    fn count_call(&mut self, frames: usize, max: Option<usize>) {
        self.calls += 1;
        self.last_call = self.last_call.max(frames);
        CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
        if max.is_some_and(|m| frames > m) {
            self.long_calls += 1;
            LONG_CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whole-window extraction (the legacy path): the F0 of `mono_16k`'s `len / 160` frames,
    /// aligned like the cache, in the cache's own buffer. Does not touch the cache.
    pub fn extract_window(&mut self, extractor: &mut dyn PitchExtractor, mono_16k: &[f32]) -> Result<&[f32], String> {
        let n = mono_16k.len() / HOP_16K;
        self.last_call = 0;
        extractor.f0_into(mono_16k, &mut self.call)?;
        self.count_call(n, extractor.max_call_frames());
        align_into(&self.call, n, &mut self.window);
        self.last_extracted = n;
        Ok(&self.window)
    }

    /// The lead of an incremental call that re-extracts `redo` frames, before the audio history
    /// bounds it (module docs, [`LeadMode`]).
    fn lead_for(&self, extractor: &dyn PitchExtractor, redo: usize) -> usize {
        let base = self.lead_frames;
        match self.geometry.lead {
            LeadMode::Legacy => base,
            LeadMode::Frames(n) => n,
            LeadMode::Fill => {
                // A local extractor's lead is its receptive field already: more audio changes nothing.
                if extractor.receptive_frames().is_some() {
                    return base;
                }
                let Some(p) = extractor.preferred_input_frames().filter(|&p| p > 0) else {
                    return match extractor.max_call_frames() {
                        // Keep the call inside a bound the extractor would otherwise split at.
                        Some(m) if redo + base > m && m >= redo + MIN_FILL_LEAD_FRAMES => m - redo,
                        _ => base,
                    };
                };
                match extractor.max_call_frames() {
                    Some(m) => {
                        let fill = p.min(m);
                        if fill >= redo + base {
                            fill - redo
                        } else if m >= redo + MIN_FILL_LEAD_FRAMES {
                            m - redo
                        } else {
                            base
                        }
                    }
                    None => {
                        // A padded plain session: fill the padded size today's call already pays.
                        let unit = p + 1;
                        (1 + redo + base).div_ceil(unit) * unit - 1 - redo
                    }
                }
            }
        }
    }

    /// Keep the stream audio contiguous up to the window's end (module docs): append what the
    /// window adds after the history, or restart the history from the window.
    fn sync_history(&mut self, mono_16k: &[f32], ws: u64, we: u64, keep_frames: usize) {
        let end = self.audio_start + (self.audio.len() / HOP_16K) as u64;
        if !self.audio.is_empty() && end >= ws && end <= we {
            let from = (end - ws) as usize * HOP_16K;
            self.audio.extend_from_slice(&mono_16k[from..]);
        } else {
            self.audio.clear();
            self.audio.extend_from_slice(mono_16k);
            self.audio_start = ws;
        }
        // Trim in bulk (at most once every `keep` frames): the history holds `keep..2 keep` frames.
        let frames = self.audio.len() / HOP_16K;
        if frames > 2 * keep_frames {
            let drop = frames - keep_frames;
            self.audio.drain(..drop * HOP_16K);
            self.audio_start += drop as u64;
        }
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
        self.last_call = 0;
        // Frames to (re)compute on an incremental tick: the new ones plus the provisional tail of
        // the cache; and the lead the geometry wants before them.
        let redo = if incremental {
            ((we - self.end()) as usize + self.config.margin_frames).min(t)
        } else {
            0
        };
        let want_lead = if incremental { self.lead_for(extractor, redo) } else { 0 };
        // The audio history is kept only while calls can reach past the window (an extractor whose
        // shape is longer than the window, or a lead that is): otherwise every call is a tail of
        // the window. A history that lapsed restarts from the next window it is needed at.
        let preferred = extractor.preferred_input_frames().unwrap_or(0);
        let history = aligned
            && t > 0
            && self.geometry.needs_history()
            && (preferred >= t || redo + want_lead > t || matches!(self.geometry.lead, LeadMode::Frames(_)));
        if history {
            // The history holds the longest call this geometry makes (and the extractor's shape).
            let keep = t
                .max(redo + want_lead)
                .max(preferred + 1)
                .min(MAX_HISTORY_FRAMES.max(t));
            self.sync_history(mono_16k, ws, we, keep);
        }
        if !incremental {
            return self.update_whole(extractor, mono_16k, ws, aligned, t);
        }
        let available = if history { ((we - self.audio_start) as usize).max(t) } else { t };
        let lead = want_lead.min(available - redo);
        let n = redo + lead;
        let slice: &[f32] = if n <= t {
            &mono_16k[(t - n) * HOP_16K..]
        } else {
            &self.audio[self.audio.len() - n * HOP_16K..]
        };
        extractor.f0_into(slice, &mut self.call)?;
        self.count_call(n, extractor.max_call_frames());
        align_into(&self.call, n, &mut self.window);
        self.last_extracted = n;
        self.last_lead = lead;
        // Splice: drop the provisional frames, append the fresh ones.
        let keep = (we - redo as u64 - self.start) as usize;
        self.frames.truncate(keep);
        self.frames.extend_from_slice(&self.window[lead..]);
        decode_span(&mut self.decode, extractor, &mut self.frames, keep, lead, redo);
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

    /// A first tick, a gap, a jump or a misaligned call: the whole window, primed in slices when the
    /// extractor's calls are bounded (module docs), seeding the cache when it is on the grid.
    fn update_whole(&mut self, extractor: &mut dyn PitchExtractor, mono_16k: &[f32], ws: u64, aligned: bool, t: usize) -> Result<&[f32], String> {
        let max = extractor.max_call_frames();
        let (lead, margin) = (self.lead_frames, self.config.margin_frames);
        if let Some(m) = max.filter(|&m| aligned && self.geometry.prime && t > m && m > lead + margin) {
            self.prime(extractor, mono_16k, ws, t, m)?;
            return Ok(&self.frames);
        }
        extractor.f0_into(mono_16k, &mut self.call)?;
        self.count_call(t, max);
        align_into(&self.call, t, &mut self.window);
        self.last_extracted = t;
        if aligned {
            self.frames.clear();
            self.frames.extend_from_slice(&self.window);
            self.start = ws;
            decode_span(&mut self.decode, extractor, &mut self.frames, 0, 0, t);
            return Ok(&self.frames);
        }
        self.frames.clear();
        Ok(&self.window)
    }

    /// Seed the cache with the window `[ws, ws + t)` from slices of at most `max` frames laid out
    /// like steady-state ticks: the first starts at the window, each later one carries the cache's
    /// lead before its kept frames, and each one but the last leaves its newest `margin` frames to
    /// the next (so every kept frame but the window's provisional tail had that much audio after
    /// it, as in a stream that had been running).
    fn prime(&mut self, extractor: &mut dyn PitchExtractor, mono_16k: &[f32], ws: u64, t: usize, max: usize) -> Result<(), String> {
        let (lead, margin) = (self.lead_frames, self.config.margin_frames);
        self.frames.clear();
        self.start = ws;
        let (mut kept, mut total) = (0usize, 0usize);
        loop {
            let s = if kept == 0 { 0 } else { kept - lead };
            let e = (s + max).min(t);
            extractor.f0_into(&mono_16k[s * HOP_16K..e * HOP_16K], &mut self.call)?;
            self.count_call(e - s, Some(max));
            total += e - s;
            align_into(&self.call, e - s, &mut self.window);
            let to = if e == t { t } else { e - margin };
            self.frames
                .extend_from_slice(&self.window[kept - s..to - s]);
            decode_span(&mut self.decode, extractor, &mut self.frames, kept, kept - s, to - kept);
            kept = to;
            if e == t {
                break;
            }
        }
        self.last_extracted = total;
        Ok(())
    }

    /// Forget the cached contour (stream gap, restart, extractor change).
    pub fn reset(&mut self) {
        self.frames.clear();
        self.start = 0;
        self.audio.clear();
        self.audio_start = 0;
    }

    /// A tick skipped as silence (P3B-S0 call site; P3B-F): its `frames` frames from absolute
    /// frame `abs_start_frame` are appended to the cache as unvoiced, so the first voiced tick
    /// after the silence is a normal incremental call whose margin frames are extracted again from
    /// the real audio. A block that does not continue the cache (a gap) is left alone: the next
    /// voiced tick then extracts its whole window. `SIGIL_VC_F0_CONTINUITY=0` (P3) leaves every
    /// skipped block as a gap.
    pub fn advance_unvoiced(&mut self, abs_start_frame: u64, frames: usize) {
        if !self.geometry.continuity || frames == 0 || self.frames.is_empty() {
            return;
        }
        let end = self.end();
        let new_end = abs_start_frame + frames as u64;
        if abs_start_frame > end || new_end <= end {
            return;
        }
        // Frames of the quiet block the cache already holds (provisional ones of the last tick).
        let from = (abs_start_frame.max(self.start) - self.start) as usize;
        for v in &mut self.frames[from..] {
            *v = 0.0;
        }
        self.frames
            .resize(self.frames.len() + (new_end - end) as usize, 0.0);
        let cap = self.config.capacity_frames;
        if self.frames.len() > cap {
            let drop = self.frames.len() - cap;
            self.frames.drain(..drop);
            self.start += drop as u64;
        }
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

    // ---- P3B-F ----

    /// The local mean extractor of [`Local`] with a declared fixed shape like RMVPE's
    /// (`preferred` / `max` audio frames, no receptive field), recording every call's slice.
    struct Shaped {
        w: usize,
        preferred: Option<usize>,
        max: Option<usize>,
        calls: Vec<usize>,
        last: Vec<f32>,
    }

    impl Shaped {
        fn new(w: usize, preferred: Option<usize>, max: Option<usize>) -> Shaped {
            Shaped {
                w,
                preferred,
                max,
                calls: Vec::new(),
                last: Vec::new(),
            }
        }
    }

    impl PitchExtractor for Shaped {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            self.calls.push(x.len() / HOP_16K);
            self.last.clear();
            self.last.extend_from_slice(x);
            Local { w: self.w, samples: 0 }.f0(x)
        }
        fn preferred_input_frames(&self) -> Option<usize> {
            self.preferred
        }
        fn max_call_frames(&self) -> Option<usize> {
            self.max
        }
    }

    fn cache_for(ctx_frames: usize, geometry: F0Geometry) -> F0Cache {
        F0Cache::new(F0CacheConfig {
            margin_frames: DEFAULT_MARGIN_FRAMES,
            capacity_frames: ctx_frames * 2,
        })
        .with_geometry(geometry)
        .with_decode_params(None)
    }

    /// Filled calls: a fixed-shape extractor's incremental calls carry real audio up to its shape
    /// (51 frames of lead on a 64-frame plan, from the audio history past the 44-frame window; 19
    /// on a 32-frame plan), a padded plain session is filled to the padded size it pays anyway and
    /// never past it, a bounded extractor never gets a call over its bound, and the legacy geometry
    /// keeps P3's 12-frame lead. The contour stays exact for a local extractor.
    #[test]
    fn filled_calls_fill_the_shape_from_the_stream_history() {
        let audio = voice(16_000 * 4);
        let ctx = 44usize;
        // (preferred, max, block frames) -> steady call frames.
        for &(preferred, max, block, want) in &[
            (Some(63usize), Some(63usize), 4usize, 63usize),
            (Some(31), Some(31), 4, 31),
            (Some(31), None, 4, 31),
            (Some(31), None, 12, 63),
            (Some(31), Some(31), 12, 31),
            (None, None, 4, 24),
        ] {
            let mut ex = Shaped::new(3, preferred, max);
            let mut reference = Local { w: 3, samples: 0 };
            let mut cache = cache_for(ctx, F0Geometry::FILLED);
            let mut pos = 0usize;
            let mut tick = 0usize;
            while pos + ctx * HOP_16K <= audio.len() {
                let window = &audio[pos..pos + ctx * HOP_16K];
                let cached = cache.update(&mut ex, window, pos as u64).unwrap().to_vec();
                let full = reference.f0(window).unwrap();
                assert_eq!(&cached[3..], &full[3..], "{preferred:?}/{max:?} block {block} tick {tick}");
                if tick > 0 {
                    let n = cache.last_extracted_frames();
                    assert_eq!(n, block + DEFAULT_MARGIN_FRAMES + cache.last_lead_frames());
                    // The slice is the stream's audio ending at the window's end.
                    let end = pos + ctx * HOP_16K;
                    assert_eq!(ex.last.as_slice(), &audio[end - n * HOP_16K..end], "tick {tick}");
                    if tick > 20 {
                        assert_eq!(n, want, "{preferred:?}/{max:?} block {block}: steady call");
                    }
                    if let Some(m) = max {
                        assert!(n <= m, "call of {n} frames over the bound {m}");
                    }
                }
                pos += block * HOP_16K;
                tick += 1;
            }
            // The cold start of a bounded extractor is primed in slices: no call is ever long.
            assert_eq!(cache.long_calls(), 0, "{preferred:?}/{max:?} block {block}");
        }
        // The legacy geometry: P3's 24-frame calls whatever the extractor declares.
        let mut ex = Shaped::new(3, Some(63), Some(63));
        let mut cache = cache_for(ctx, F0Geometry::LEGACY);
        for tick in 0..30usize {
            let pos = tick * 4 * HOP_16K;
            cache
                .update(&mut ex, &audio[pos..pos + ctx * HOP_16K], pos as u64)
                .unwrap();
            if tick > 0 {
                assert_eq!(cache.last_extracted_frames(), 24);
            }
        }
    }

    /// Priming: a cold start, gap or jump on an extractor whose calls are bounded (P3B-E's 32-frame
    /// plan holds 31 audio frames) is extracted in slices of at most that, laid out like
    /// steady-state ticks, so no call ever exceeds the bound; for a local extractor the primed
    /// window equals the whole-window extraction exactly. Off, the cold start is one long call.
    #[test]
    fn priming_never_exceeds_the_extractors_bound() {
        let audio = voice(16_000 * 3);
        let ctx = 44usize;
        let mut ex = Shaped::new(3, Some(31), Some(31));
        let mut reference = Local { w: 3, samples: 0 };
        let mut cache = cache_for(ctx, F0Geometry::FILLED);
        let mut pos = 0usize;
        for tick in 0..40usize {
            let window = &audio[pos..pos + ctx * HOP_16K];
            let cached = cache.update(&mut ex, window, pos as u64).unwrap().to_vec();
            let full = reference.f0(window).unwrap();
            // The cache saw audio before later windows: their first `w` frames differ from a
            // window-only extraction by design.
            let from = if tick == 0 { 0 } else { 3 };
            assert_eq!(
                &cached[from..],
                &full[from..],
                "tick {tick}: primed / cached window against the whole window"
            );
            if tick == 0 {
                // Three slices: [0, 31) keeps 23, [11, 42) keeps 11, [22, 44) keeps the rest.
                assert_eq!(ex.calls, vec![31, 31, 22]);
                assert_eq!(cache.last_extracted_frames(), 84);
            }
            pos += 4 * HOP_16K;
        }
        // A jump re-primes.
        let n_calls = ex.calls.len();
        cache
            .update(&mut ex, &audio[..ctx * HOP_16K], 1_000 * HOP_16K as u64)
            .unwrap();
        assert_eq!(ex.calls.len() - n_calls, 3);
        assert!(ex.calls.iter().all(|&n| n <= 31), "{:?}", ex.calls);
        assert_eq!(cache.long_calls(), 0);
        // Priming off: one whole-window call over the bound.
        let mut ex = Shaped::new(3, Some(31), Some(31));
        let mut cache = cache_for(
            ctx,
            F0Geometry {
                prime: false,
                ..F0Geometry::PRIMED
            },
        );
        let before = process_call_counts();
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        assert_eq!(ex.calls, vec![44]);
        assert_eq!(cache.long_calls(), 1);
        // The process-wide counters (other tests move them too: at least this call).
        let after = process_call_counts();
        assert!(after.0 > before.0 && after.1 > before.1, "{before:?} -> {after:?}");
        // A bound too small to lay slices out (lead + margin) keeps the whole-window call.
        let mut ex = Shaped::new(3, Some(15), Some(15));
        let mut cache = cache_for(ctx, F0Geometry::FILLED);
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        assert_eq!(ex.calls, vec![44]);
    }

    /// Silence continuity: ticks skipped as silence append their frames unvoiced, so the first
    /// voiced tick after the pause is a normal incremental call (its margin re-extracted from the
    /// real audio) and the frames before the pause are untouched; a gap is left to the next tick's
    /// whole-window call; off, every pause ends in a whole-window call.
    #[test]
    fn silence_skips_keep_the_cache_incremental() {
        let audio = voice(16_000 * 3);
        let (ctx, block) = (44usize, 4usize);
        for continuity in [true, false] {
            let mut ex = Shaped::new(3, None, None);
            let mut cache = cache_for(
                ctx,
                F0Geometry {
                    continuity,
                    ..F0Geometry::PRIMED
                },
            );
            let mut pos = 0usize;
            let mut before = Vec::new();
            for tick in 0..30usize {
                let window = &audio[pos..pos + ctx * HOP_16K];
                if (20..25).contains(&tick) {
                    // Skipped: the block's frames at the window's end.
                    let first = (pos / HOP_16K + ctx - block) as u64;
                    cache.advance_unvoiced(first, block);
                } else {
                    let got = cache.update(&mut ex, window, pos as u64).unwrap().to_vec();
                    if tick == 19 {
                        before = got.clone();
                    }
                    if tick == 25 {
                        if continuity {
                            assert_eq!(cache.last_extracted_frames(), block + DEFAULT_MARGIN_FRAMES + DEFAULT_LEAD_FRAMES);
                            // Window of tick 25 starts 6 blocks after tick 19's: the frames that were
                            // committed before the pause are unchanged ...
                            let shift = 6 * block;
                            let committed = ctx - DEFAULT_MARGIN_FRAMES;
                            assert_eq!(&got[..committed - shift], &before[shift..committed]);
                            // ... and the skipped frames (tick 20's block on) that were not
                            // re-extracted are unvoiced (`Local` never returns 0 itself).
                            let skipped = ctx - shift;
                            let redone = ctx - block - DEFAULT_MARGIN_FRAMES;
                            assert!(got[skipped..redone].iter().all(|&v| v == 0.0), "{got:?}");
                            assert!(got[redone..].iter().all(|&v| v > 0.0));
                        } else {
                            assert_eq!(cache.last_extracted_frames(), ctx);
                        }
                    }
                }
                pos += block * HOP_16K;
            }
        }
        // A skipped block that does not continue the cache (a gap) is left alone.
        let mut ex = Shaped::new(3, None, None);
        let mut cache = cache_for(ctx, F0Geometry::FILLED);
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        cache.advance_unvoiced(ctx as u64 + 100, block);
        cache
            .update(&mut ex, &audio[100 * HOP_16K..(100 + ctx) * HOP_16K], (100 * HOP_16K) as u64)
            .unwrap();
        assert_eq!(cache.last_extracted_frames(), ctx);
        // An empty cache (a reset) stays empty: the next voiced tick extracts its window.
        cache.reset();
        cache.advance_unvoiced(0, block);
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        assert_eq!(cache.last_extracted_frames(), ctx);
    }

    /// A salience extractor for the decoder tests: every 10 ms frame of the audio holds a code,
    /// `bin / 1000` (one peak at `bin`, salience 0.9) or `-bin / 1000` (an octave error: the argmax
    /// an octave up at 0.8, the true pitch a peak at 0.5); 0 is silence. Like RMVPE it returns
    /// `n + 1` frames (and salience rows) for `n` frames of audio. Allocation-free once warm.
    struct Coded {
        salience: Vec<f32>,
    }

    impl Coded {
        fn new() -> Coded {
            Coded {
                salience: Vec::with_capacity(512 * 360),
            }
        }

        fn row(code: f32, row: &mut [f32]) {
            row.fill(0.0);
            let put = |row: &mut [f32], b: usize, h: f32| {
                for (j, v) in row.iter_mut().enumerate() {
                    let d = j as f32 - b as f32;
                    *v = v.max(h * (-d * d / 4.0).exp());
                }
            };
            if code > 0.0 {
                put(row, (code * 1000.0).round() as usize, 0.9);
            } else if code < 0.0 {
                let b = (-code * 1000.0).round() as usize;
                put(row, b + 60, 0.8);
                put(row, b, 0.5);
            }
        }
    }

    impl PitchExtractor for Coded {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            let mut out = Vec::new();
            self.f0_into(x, &mut out)?;
            Ok(out)
        }
        fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
            let n = x.len() / HOP_16K;
            out.clear();
            self.salience.clear();
            for j in 0..=n {
                let code = x[(j * HOP_16K).min(x.len() - 1)];
                let at = self.salience.len();
                self.salience.resize(at + SALIENCE_BINS, 0.0);
                let row = &mut self.salience[at..];
                Coded::row(code, row);
                let (hz, peak) = crate::services::neural_audio::vc::pitch_onnx::local_average_f0(row);
                out.push(if peak <= 0.03 { 0.0 } else { hz });
            }
            Ok(())
        }
        fn last_salience(&self) -> Option<(&[f32], usize)> {
            Some((&self.salience, SALIENCE_BINS))
        }
        fn preferred_input_frames(&self) -> Option<usize> {
            Some(63)
        }
        fn max_call_frames(&self) -> Option<usize> {
            Some(63)
        }
    }

    /// A coded stream: 140 Hz with octave errors on frames `errors`, silence on `quiet`.
    fn coded_stream(frames: usize, errors: std::ops::Range<usize>, quiet: std::ops::Range<usize>) -> Vec<f32> {
        let bin = ((1200.0 * (140.0f32 / 10.0).log2() - 1_997.379_4) / 20.0).round();
        (0..frames * HOP_16K)
            .map(|i| {
                let f = i / HOP_16K;
                if quiet.contains(&f) {
                    0.0
                } else if errors.contains(&f) {
                    -bin / 1000.0
                } else {
                    bin / 1000.0
                }
            })
            .collect()
    }

    /// The decoder changes only the span each tick re-extracts (committed frames never move), takes
    /// the true pitch over a short octave error the salience still supports, is off unless asked
    /// (and never for singing), and the cache with it allocates nothing per tick once warm.
    #[test]
    fn decoding_changes_only_the_reextracted_span() {
        let ctx = 44usize;
        let audio = coded_stream(400, 200..202, 300..330);
        let run = |params: Option<DecodeParams>| -> (Vec<f32>, F0Cache) {
            let mut ex = Coded::new();
            let mut cache = cache_for(ctx, F0Geometry::FILLED).with_decode_params(params);
            let mut fed = vec![0.0f32; 400];
            let mut prev: Option<(usize, Vec<f32>)> = None;
            let mut pos = 0usize;
            while pos + ctx * HOP_16K <= audio.len() {
                let window = cache
                    .update(&mut ex, &audio[pos..pos + ctx * HOP_16K], pos as u64)
                    .unwrap()
                    .to_vec();
                let ws = pos / HOP_16K;
                if let Some((pws, p)) = &prev {
                    // The frames the last window had committed (older than its margin) are
                    // unchanged now: only the re-extracted span is decoded.
                    for f in ws..(pws + ctx - DEFAULT_MARGIN_FRAMES) {
                        assert_eq!(window[f - ws], p[f - pws], "frame {f} moved");
                    }
                }
                let we = ws + ctx;
                fed[we - 4..we].copy_from_slice(&window[ctx - 4..]);
                prev = Some((ws, window));
                pos += 4 * HOP_16K;
            }
            (fed, cache)
        };
        let (plain, cache) = run(None);
        assert!(!cache.decoding());
        assert!(cents(plain[200], 140.0) > 1100.0, "the octave error reaches the cache undecoded");
        let (decoded, cache) = run(Some(DecodeParams::speech()));
        assert!(cache.decoding() && cache.decode_note().is_none());
        for (f, v) in decoded.iter().enumerate().take(206).skip(196) {
            assert!(cents(*v, 140.0) < 30.0, "frame {f}: {v}");
        }
        // Silence stays silent; frames away from the error are the extractor's own.
        assert!(decoded[305..325].iter().all(|&v| v == 0.0));
        assert_eq!(&decoded[100..190], &plain[100..190]);
        // The lever, the range and the override (`env` set by hand: tests never touch the process
        // environment).
        let mut c = F0Cache::new(F0CacheConfig {
            margin_frames: 8,
            capacity_frames: 88,
        });
        c.decode.env = None;
        c.set_decode(true, F0Range::Singing);
        assert!(!c.decoding(), "singing never decodes");
        c.set_decode(true, F0Range::Speech);
        assert!(c.decoding());
        c.set_decode(false, F0Range::Speech);
        assert!(!c.decoding());
        c.decode.env = Some(true);
        c.set_decode(false, F0Range::Speech);
        assert!(c.decoding(), "the override decodes whatever the lever says");
        c.set_decode(false, F0Range::Singing);
        assert!(!c.decoding(), "... but never a singing voice");
        // An extractor without salience: the note says why nothing is decoded.
        let mut c = cache_for(ctx, F0Geometry::FILLED).with_decode_params(Some(DecodeParams::speech()));
        let mut ex = Shaped::new(3, None, None);
        let a = voice(16_000);
        c.update(&mut ex, &a[..ctx * HOP_16K], 0).unwrap();
        c.update(&mut ex, &a[4 * HOP_16K..(4 + ctx) * HOP_16K], 4 * HOP_16K as u64)
            .unwrap();
        assert!(c.decode_note().is_some());
    }

    /// The tick path of the cache (filled calls through the history, the decoder, silence
    /// continuity) makes no heap allocation once warm.
    #[test]
    fn the_cache_is_allocation_free_once_warm() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let ctx = 44usize;
        let audio = coded_stream(1_200, 500..502, 700..720);
        let mut ex = Coded::new();
        let mut cache = cache_for(ctx, F0Geometry::FILLED).with_decode_params(Some(DecodeParams::speech()));
        let tick = |k: usize, cache: &mut F0Cache, ex: &mut Coded| {
            let pos = k * 4 * HOP_16K;
            if (180..185).contains(&k) {
                cache.advance_unvoiced((pos / HOP_16K + ctx - 4) as u64, 4);
            } else {
                std::hint::black_box(
                    cache
                        .update(ex, &audio[pos..pos + ctx * HOP_16K], pos as u64)
                        .unwrap()[ctx - 1],
                );
            }
        };
        for k in 0..60 {
            tick(k, &mut cache, &mut ex);
        }
        let ((), allocs) = allocations_during(|| {
            for k in 60..280 {
                tick(k, &mut cache, &mut ex);
            }
        });
        assert_eq!(allocs, 0, "the F0 cache allocated {allocs} times in 220 ticks");
        assert_eq!(cache.last_lead_frames(), 51);
    }

    // ---------------------------------------------------------------------------------------------
    // Dev-rig sweep (P3B-F measure-first, rule R1; extends P3-C1's lead sweep)
    // ---------------------------------------------------------------------------------------------

    /// The 16 kHz speech clips of the dev cache (`sherpa-onnx/clips/*.wav`).
    fn sweep_clips(root: &std::path::Path) -> Vec<(String, Vec<f32>)> {
        let mut clips = Vec::new();
        for name in ["en", "ja", "ko", "ru", "uk", "yue", "zh"] {
            let path = root
                .join("sherpa-onnx")
                .join("clips")
                .join(format!("{name}.wav"));
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
        clips
    }

    /// The pooled scores of one sweep row: scorer v2's fed-F0 definitions
    /// (`bench::quality::fed_f0_errors`: fine pitch error = RMSE over the frames voiced in both and
    /// within 20 %, gross pitch error = voiced in both and off by more than 20 %, strict V/UV, onset
    /// lag) pooled over the clips, plus the newest read frame's missed voicing.
    #[derive(Default)]
    struct SweepScore {
        fine_sq: f64,
        fine_n: usize,
        gross: usize,
        both: usize,
        vuv: usize,
        frames: usize,
        newest_voiced: usize,
        newest_missed: usize,
        onsets: usize,
        onsets_missed: usize,
        lags: Vec<i64>,
        times: Vec<f64>,
        call_frames: usize,
        /// Replaces the call-frames column when set.
        call_label: Option<String>,
    }

    impl SweepScore {
        /// Add one clip: `est` (NaN = not read) against `reference` (NaN = not scored) from frame
        /// `from` on; `newest` lists the newest read frame of every tick.
        fn add(&mut self, reference: &[f32], est: &[f32], from: usize, newest: &[usize]) {
            use crate::services::bench::quality::fed_f0_errors;
            let mask: Vec<bool> = (0..reference.len())
                .map(|f| f >= from && f < est.len() && !est[f].is_nan() && !reference[f].is_nan())
                .collect();
            let clean: Vec<f32> = est
                .iter()
                .map(|v| if v.is_nan() { 0.0 } else { *v })
                .collect();
            let Ok(e) = fed_f0_errors(reference, &clean, Some(&mask)) else { return };
            let fine_n = e.errors.v1.voiced_frames - e.errors.gross_frames;
            self.fine_sq += e
                .errors
                .fine_rmse_cents
                .map_or(0.0, |r| r * r * fine_n as f64);
            self.fine_n += fine_n;
            self.gross += e.errors.gross_frames;
            self.both += e.errors.v1.voiced_frames;
            self.vuv += (e.errors.v1.vuv_error * e.errors.v1.frames as f64).round() as usize;
            self.frames += e.errors.v1.frames;
            self.onsets += e.onsets;
            self.onsets_missed += e.onsets_missed;
            self.lags.extend(e.onset_lags);
            for &f in newest.iter().filter(|&&f| f < reference.len() && mask[f]) {
                if reference[f] > 0.0 {
                    self.newest_voiced += 1;
                    if clean[f] <= 0.0 {
                        self.newest_missed += 1;
                    }
                }
            }
        }

        fn row(&mut self, cols: &str) -> (String, serde_json::Value) {
            let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
            self.times.sort_by(f64::total_cmp);
            let p50 = self.times.get(self.times.len() / 2).copied().unwrap_or(0.0);
            let lag = if self.lags.is_empty() {
                0.0
            } else {
                self.lags.iter().sum::<i64>() as f64 / self.lags.len() as f64
            };
            let fpe = (self.fine_sq / self.fine_n.max(1) as f64).sqrt();
            let (gpe, vuv, newest) = (
                pct(self.gross, self.both),
                pct(self.vuv, self.frames),
                pct(self.newest_missed, self.newest_voiced),
            );
            let calls = self
                .call_label
                .clone()
                .unwrap_or_else(|| self.call_frames.to_string());
            let md = format!(
                "| {cols} | {calls} | {fpe:.2} | {gpe:.3} | {vuv:.2} | {newest:.2} | {lag:+.2} | {}/{} | {p50:.2} |",
                self.onsets_missed, self.onsets
            );
            let json = serde_json::json!({
                "row": cols, "call_frames": self.call_frames, "fpe_cents": fpe, "gpe_pct": gpe, "vuv_pct": vuv,
                "missed_newest_pct": newest, "onset_lag_mean": lag, "onsets": self.onsets, "onsets_missed": self.onsets_missed,
                "frames": self.frames, "voiced_both": self.both, "call_ms_p50": p50,
            });
            (md, json)
        }
    }

    /// A 3-tap (or 5-tap) median whose edges replicate the end frames, with `smooth_f0`'s octave-jump
    /// repair before it: the measure-first stand-in for the streaming smoother (`f0_post`), written
    /// here so the sweep runs before the production change.
    fn sweep_median_replicate(f0: &mut [f32], window: usize) {
        let n = f0.len();
        if n == 0 {
            return;
        }
        for v in f0.iter_mut() {
            if !(v.is_finite() && *v > 0.0) {
                *v = 0.0;
            }
        }
        let raw = f0.to_vec();
        let jumps = |a: f32, b: f32| a / b > 1.9 || b / a > 1.9;
        for i in 0..n {
            let cur = raw[i];
            if cur <= 0.0 {
                continue;
            }
            let (mut sum, mut count, mut all_jump) = (0.0f32, 0usize, true);
            for nb in [if i > 0 { raw[i - 1] } else { 0.0 }, raw.get(i + 1).copied().unwrap_or(0.0)] {
                if nb > 0.0 {
                    sum += nb;
                    count += 1;
                    all_jump &= jumps(cur, nb);
                }
            }
            if count > 0 && all_jump {
                f0[i] = sum / count as f32;
            }
        }
        let half = window.clamp(1, 5) / 2;
        let src = f0.to_vec();
        for (i, out) in f0.iter_mut().enumerate() {
            let mut win: Vec<f32> = (0..=2 * half)
                .map(|k| src[(i + k).saturating_sub(half).min(n - 1)])
                .collect();
            win.sort_by(f32::total_cmp);
            *out = win[half];
        }
    }

    /// Dev-rig sweep, P3B-F measure-first (rule R1), table **geometry**: streamed RMVPE against each
    /// clip's whole-clip RMVPE on the 7 speech clips, 40 ms blocks (4 new frames plus the 8-frame
    /// margin re-extracted every tick), each frame read from the tick whose newest block holds it
    /// (`readout lag` = frames before the newest, the engine's F0 lookahead), at:
    /// - the 32-frame geometry (the CPU plain path's padded size and P3B-E's VRAM saver plan) with
    ///   real-audio leads 12 (P3) and 19 (filled: 31 frames);
    /// - the 64-frame geometry (P3B-E's default plan) with leads 12, 19, 32 and 51 (filled: 63
    ///   frames). Calls shorter than 32 frames are front-padded with digital silence to 32 frames, so
    ///   RMVPE's own end-aligned padding fills the rest of the 64 mel frames with its floor;
    /// - each with the streaming smoother's median zero-padded (`smooth_f0`, P3) and edge-replicated.
    ///
    /// The contour is decoded again from the call's salience (RMVPE's own decode: argmax ± 4 bins,
    /// 0.03 threshold), smoothed over the audio's `n + 1` frames as the plan path does, and checked
    /// against RMVPE's own output where no padding was added.
    ///
    /// Table **cache** (`SIGIL_SWEEP_TABLE=cache`): the production F0 cache and decoder (P3B-F),
    /// see [`cache_sweep`].
    ///
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache` (the 16 kHz `sherpa-onnx/clips/*.wav`
    /// and the pitch models), `SIGIL_SWEEP_TABLE=geometry|cache|all` (default `all`),
    /// `SIGIL_SWEEP_OUT=<file.json>` (rows as JSON):
    /// `cargo test --manifest-path src-tauri/Cargo.toml lead_sweep_against_whole_clip_f0 -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig sweep: needs SIGIL_TEST_MODELS_DIR with the pitch models and the speech clips"]
    fn lead_sweep_against_whole_clip_f0() {
        use crate::services::neural_audio::ort_session::VcSessionOptions;
        use crate::services::neural_audio::vc::f0_fusion::test_model;
        use crate::services::neural_audio::vc::pitch_onnx::local_average_f0;
        use crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor;
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return;
        };
        let table = std::env::var("SIGIL_SWEEP_TABLE").unwrap_or_else(|_| "all".into());
        let clips = sweep_clips(&root);
        assert!(!clips.is_empty(), "no 16 kHz clips under {}", root.display());
        let Some(model) = test_model("pitch-rmvpe") else {
            eprintln!("pitch-rmvpe not found; skipped");
            return;
        };
        let mut json_rows = Vec::new();
        let mut ex = RmvpePitchExtractor::load(&model.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        // Whole-clip references (the scorer v2 input analyzer: RMVPE over the whole clip).
        let references: Vec<Vec<f32>> = clips
            .iter()
            .map(|(_, x)| {
                let mut f = ex.f0(x).unwrap();
                f.resize(x.len() / HOP_16K, 0.0);
                f
            })
            .collect();
        if table == "all" || table == "geometry" {
            const CTX: usize = 64;
            const BLOCK: usize = 4;
            const REDO: usize = BLOCK + DEFAULT_MARGIN_FRAMES;
            const LAGS: [usize; 3] = [0, 2, 4];
            println!(
                "\nP3B-F sweep, table geometry (CPU RMVPE, {} clips, 40 ms blocks; scored from frame {CTX})",
                clips.len()
            );
            println!("| geometry | lead | median edges | readout lag | call frames | FPE c | GPE % | V/UV % | missed newest % | onset lag | onsets missed | call ms p50 |");
            println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
            for (geometry, lead) in [(32usize, 12usize), (32, 19), (64, 12), (64, 19), (64, 32), (64, 51)] {
                let n = lead + REDO;
                assert!(n < geometry);
                // [edge][lag] -> per-clip estimates; timing is shared.
                let mut scores: Vec<Vec<SweepScore>> = (0..2)
                    .map(|_| LAGS.iter().map(|_| SweepScore::default()).collect())
                    .collect();
                let mut times = Vec::new();
                let mut checked = 0usize;
                for ((_, x), reference) in clips.iter().zip(&references) {
                    let total = x.len() / HOP_16K;
                    let mut est: Vec<Vec<Vec<f32>>> = (0..2)
                        .map(|_| LAGS.iter().map(|_| vec![f32::NAN; total]).collect())
                        .collect();
                    let mut newest: Vec<Vec<usize>> = LAGS.iter().map(|_| Vec::new()).collect();
                    let mut end = CTX * HOP_16K;
                    let pad = if geometry == 64 && n < 32 { 32 - n } else { 0 };
                    let mut buf = Vec::with_capacity((pad + n) * HOP_16K);
                    while end <= x.len() {
                        buf.clear();
                        buf.resize(pad * HOP_16K, 0.0);
                        buf.extend_from_slice(&x[end - n * HOP_16K..end]);
                        let t0 = std::time::Instant::now();
                        let own = ex.f0(&buf).unwrap();
                        times.push(t0.elapsed().as_secs_f64() * 1000.0);
                        let (sal, bins) = ex.last_salience().expect("RMVPE salience");
                        let rows = sal.len() / bins;
                        assert_eq!(rows, pad + n + 1, "rows of a {}-frame call", pad + n);
                        // The audio's frames and the extra one, decoded as RMVPE does.
                        let raw: Vec<f32> = (rows - (n + 1)..rows)
                            .map(|r| {
                                let (hz, peak) = local_average_f0(&sal[r * bins..(r + 1) * bins]);
                                if peak <= 0.03 || !hz.is_finite() {
                                    0.0
                                } else {
                                    hz
                                }
                            })
                            .collect();
                        let mut zero = raw.clone();
                        super::super::super::f0_post::smooth_f0(&mut zero, 3);
                        let mut rep = raw;
                        sweep_median_replicate(&mut rep, 3);
                        if pad == 0 {
                            // The re-decode is RMVPE's own contour (smoothed by `smooth_f0_stream`,
                            // whose edges this process uses).
                            let mine = if super::super::super::f0_post::stream_edges_zero() {
                                &zero
                            } else {
                                &rep
                            };
                            assert_eq!(
                                &mine[..n],
                                &own[own.len() - (n + 1)..own.len() - 1],
                                "re-decode differs from RMVPE's own contour"
                            );
                            checked += 1;
                        }
                        let first_frame = end / HOP_16K - n;
                        for (li, &la) in LAGS.iter().enumerate() {
                            let last = end / HOP_16K - la;
                            let (from, to) = (last - BLOCK - first_frame, last - first_frame);
                            est[0][li][last - BLOCK..last].copy_from_slice(&zero[from..to]);
                            est[1][li][last - BLOCK..last].copy_from_slice(&rep[from..to]);
                            newest[li].push(last - 1);
                        }
                        end += BLOCK * HOP_16K;
                    }
                    for edge in 0..2 {
                        for li in 0..LAGS.len() {
                            scores[edge][li].add(reference, &est[edge][li], CTX, &newest[li]);
                        }
                    }
                }
                for (edge, name) in [(0usize, "zero (P3)"), (1, "replicated")] {
                    for (li, &la) in LAGS.iter().enumerate() {
                        let s = &mut scores[edge][li];
                        s.times = times.clone();
                        s.call_frames = n;
                        let (md, json) = s.row(&format!("{geometry} | {lead} | {name} | {la}"));
                        println!("{md}");
                        json_rows.push(json);
                    }
                }
                eprintln!("geometry {geometry}, lead {lead}: re-decode checked against RMVPE on {checked} calls");
            }
        }
        if table == "all" || table == "cache" || table == "cost" {
            json_rows.extend(cache_sweep(&clips, &references, &model));
        }
        if let Ok(path) = std::env::var("SIGIL_SWEEP_OUT") {
            std::fs::write(&path, serde_json::to_string_pretty(&json_rows).unwrap()).unwrap();
            eprintln!("rows written to {path}");
        }
    }

    /// RMVPE on the CPU at a plan's geometry: the plan's bound and shape declared (`frames - 1`
    /// audio frames, like `RmvpePitchExtractor` on a GPU plan), calls shorter than 32 frames on a
    /// 64-frame geometry front-padded with digital silence to 32 frames (RMVPE's end-aligned padding
    /// then fills the 64 mel frames with its floor), the padding's frames and salience rows dropped.
    struct Geo<'a> {
        inner: &'a mut dyn PitchExtractor,
        frames: usize,
        buf: Vec<f32>,
        sal: Vec<f32>,
    }

    impl PitchExtractor for Geo<'_> {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            let mut out = Vec::new();
            self.f0_into(x, &mut out)?;
            Ok(out)
        }
        fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
            let n = x.len() / HOP_16K;
            let pad = if self.frames == 64 && n < 32 { 32 - n } else { 0 };
            let slice: &[f32] = if pad > 0 {
                self.buf.clear();
                self.buf.resize(pad * HOP_16K, 0.0);
                self.buf.extend_from_slice(x);
                &self.buf
            } else {
                x
            };
            self.inner.f0_into(slice, out)?;
            out.drain(..pad.min(out.len()));
            self.sal.clear();
            if let Some((s, bins)) = self.inner.last_salience() {
                self.sal.extend_from_slice(&s[(pad * bins).min(s.len())..]);
            }
            Ok(())
        }
        fn preferred_input_frames(&self) -> Option<usize> {
            Some(self.frames - 1)
        }
        fn max_call_frames(&self) -> Option<usize> {
            Some(self.frames - 1)
        }
        fn last_salience(&self) -> Option<(&[f32], usize)> {
            (!self.sal.is_empty()).then_some((self.sal.as_slice(), SALIENCE_BINS))
        }
    }

    /// One recorded extractor call (its input length, contour and salience).
    struct Recorded {
        frames: usize,
        f0: Vec<f32>,
        sal: Vec<f32>,
    }

    /// Records the calls of `inner` (one model pass per geometry; the decoder variants replay it).
    struct Recorder<'a> {
        inner: Geo<'a>,
        calls: Vec<Recorded>,
    }

    impl PitchExtractor for Recorder<'_> {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            let mut out = Vec::new();
            self.f0_into(x, &mut out)?;
            Ok(out)
        }
        fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
            self.inner.f0_into(x, out)?;
            self.calls.push(Recorded {
                frames: x.len() / HOP_16K,
                f0: out.clone(),
                sal: self.inner.sal.clone(),
            });
            Ok(())
        }
        fn preferred_input_frames(&self) -> Option<usize> {
            self.inner.preferred_input_frames()
        }
        fn max_call_frames(&self) -> Option<usize> {
            self.inner.max_call_frames()
        }
        fn last_salience(&self) -> Option<(&[f32], usize)> {
            self.inner.last_salience()
        }
    }

    /// Replays recorded calls in order (the call sequence depends on the geometry and the silence
    /// skips, never on the decoder), so one model pass scores every decoder variant.
    struct Replay<'a> {
        calls: &'a [Recorded],
        k: usize,
        frames: usize,
    }

    impl PitchExtractor for Replay<'_> {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            let mut out = Vec::new();
            self.f0_into(x, &mut out)?;
            Ok(out)
        }
        fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
            let c = self
                .calls
                .get(self.k)
                .ok_or("the replay ran out of calls")?;
            if c.frames != x.len() / HOP_16K {
                return Err(format!(
                    "replay call {}: {} frames recorded, {} asked",
                    self.k,
                    c.frames,
                    x.len() / HOP_16K
                ));
            }
            self.k += 1;
            out.clear();
            out.extend_from_slice(&c.f0);
            Ok(())
        }
        fn preferred_input_frames(&self) -> Option<usize> {
            Some(self.frames - 1)
        }
        fn max_call_frames(&self) -> Option<usize> {
            Some(self.frames - 1)
        }
        fn last_salience(&self) -> Option<(&[f32], usize)> {
            self.k
                .checked_sub(1)
                .and_then(|k| self.calls.get(k))
                .map(|c| (c.sal.as_slice(), SALIENCE_BINS))
        }
    }

    /// The fed F0 of one clip streamed through `cache` the engine's way: a 44-frame window (the
    /// Low-latency plan), 40 ms blocks, the silence skip (RMS of the returned span — block plus
    /// the default SOLA tail — under -42 dBFS for more than the 80 ms hangover: the block's frames
    /// go to `advance_unvoiced` and are fed unvoiced). Per readout lag: the estimate (NaN = never
    /// read) and the newest read frame of every converted tick. Also the extractor's call lengths.
    fn stream_clip(ex: &mut dyn PitchExtractor, cache: &mut F0Cache, x: &[f32], lags: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<usize>) {
        const CTX: usize = 44;
        const BLOCK: usize = 4;
        const WANT: usize = BLOCK * HOP_16K + 481;
        let total = x.len() / HOP_16K;
        let mut est: Vec<Vec<f32>> = lags.iter().map(|_| vec![f32::NAN; total]).collect();
        let mut newest: Vec<Vec<usize>> = lags.iter().map(|_| Vec::new()).collect();
        let mut calls = Vec::new();
        let floor = 10f32.powf(-42.0 / 20.0);
        let mut quiet = 0u32;
        let mut end = CTX * HOP_16K;
        while end <= x.len() {
            let pos = end - CTX * HOP_16K;
            let tail = &x[end - WANT..end];
            let rms = (tail
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / WANT as f64)
                .sqrt() as f32;
            quiet = if rms < floor { quiet + 1 } else { 0 };
            let we = end / HOP_16K;
            if quiet > 2 {
                cache.advance_unvoiced((we - BLOCK) as u64, BLOCK);
                for (li, &la) in lags.iter().enumerate() {
                    est[li][we - BLOCK - la..we - la].fill(0.0);
                }
            } else {
                let w = cache.update(ex, &x[pos..end], pos as u64).unwrap();
                for (li, &la) in lags.iter().enumerate() {
                    est[li][we - BLOCK - la..we - la].copy_from_slice(&w[CTX - BLOCK - la..CTX - la]);
                    newest[li].push(we - 1 - la);
                }
                calls.push(cache.last_call_frames());
            }
            end += BLOCK * HOP_16K;
        }
        (est, newest, calls)
    }

    /// Table **cache** of the sweep (P3B-F): the production F0 cache (`F0Cache` with its geometry
    /// and decoder) over the real clips (against whole-clip RMVPE) and the synthetic speech corpus
    /// of the pitch A/B (against its exact truth; `SIGIL_AB_SEED`, default the held-out set), the
    /// engine's silence skip emulated, at RMVPE's 64- and 32-frame plan geometries, P3's cache
    /// against v2, and the decoder's variants (one model pass per geometry, replayed).
    fn cache_sweep(clips: &[(String, Vec<f32>)], references: &[Vec<f32>], model: &std::path::Path) -> Vec<serde_json::Value> {
        use crate::services::neural_audio::ort_session::VcSessionOptions;
        use crate::services::neural_audio::vc::f0_decode::DecodeParams;
        use crate::services::neural_audio::vc::f0_fusion::ab;
        use crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor;
        const LAGS: [usize; 3] = [0, 2, 4];
        const FROM: usize = 64;
        let seconds: f32 = std::env::var("SIGIL_AB_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0);
        // `SIGIL_AB_SEED=<seed>[,<seed>...]`: synthetic corpora (default the held-out set).
        let seeds: Vec<u64> = std::env::var("SIGIL_AB_SEED")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect::<Vec<u64>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![ab::HELD_OUT_SEED]);
        let synthetic: Vec<(u64, Vec<ab::Clip>)> = seeds
            .iter()
            .map(|&seed| {
                let c = ab::corpus_seeded(seconds, seed)
                    .into_iter()
                    .filter(|c| !c.name.starts_with("sing/"))
                    .collect();
                (seed, c)
            })
            .collect();
        // (corpus, [(audio, reference, clip or condition)])
        // (name, [(audio, reference, clip or condition)])
        type Corpus<'a> = (String, Vec<(&'a [f32], &'a [f32], String)>);
        let mut corpora: Vec<Corpus<'_>> = vec![(
            "speech clips vs whole-clip RMVPE".to_string(),
            clips
                .iter()
                .zip(references)
                .map(|((name, x), r)| (x.as_slice(), r.as_slice(), name.clone()))
                .collect(),
        )];
        for (seed, set) in &synthetic {
            corpora.push((
                format!("synthetic speech vs truth (seeds from {seed})"),
                set.iter()
                    .map(|c| {
                        (
                            c.audio.as_slice(),
                            c.truth.as_slice(),
                            c.name.rsplit('/').next().unwrap_or(&c.name).to_string(),
                        )
                    })
                    .collect(),
            ));
        }
        let base = DecodeParams::speech();
        let pitch = |cost: f32| DecodeParams {
            voice_off: 0.0,
            cost_per_semitone: cost,
            ..base
        };
        let hyst = |off: f32| DecodeParams {
            voice_off: off,
            cost_per_semitone: 0.0,
            ..base
        };
        let decoders: Vec<(&str, Option<DecodeParams>)> = vec![
            ("off", None),
            ("default", Some(base)),
            ("pitch 0.25", Some(pitch(0.25))),
            ("pitch 0.5", Some(pitch(0.5))),
            (
                "pitch 0.25, peaks >= 0.5 best",
                Some(DecodeParams {
                    candidate_ratio: 0.5,
                    ..pitch(0.25)
                }),
            ),
            (
                "pitch 0.5, peaks >= 0.5 best",
                Some(DecodeParams {
                    candidate_ratio: 0.5,
                    ..pitch(0.5)
                }),
            ),
            ("hysteresis 0.02", Some(hyst(0.02))),
            ("hysteresis 0.025", Some(hyst(0.025))),
            (
                "hysteresis 0.025 + pitch 0.25, peaks >= 0.5",
                Some(DecodeParams {
                    voice_off: 0.025,
                    candidate_ratio: 0.5,
                    ..pitch(0.25)
                }),
            ),
            (
                "onset 0.05 + pitch 0.25",
                Some(DecodeParams {
                    voice_on: 0.05,
                    ..pitch(0.25)
                }),
            ),
            (
                "onset 0.05 + pitch 0.25, peaks >= 0.5",
                Some(DecodeParams {
                    voice_on: 0.05,
                    candidate_ratio: 0.5,
                    ..pitch(0.25)
                }),
            ),
        ];
        let configs: Vec<(usize, F0Geometry, &str, bool)> = vec![
            (64, F0Geometry::LEGACY, "P3 (lead 12)", false),
            (64, F0Geometry::PRIMED, "primed (continuity, priming)", true),
            (64, F0Geometry::FILLED, "filled (lead 51)", false),
            (32, F0Geometry::LEGACY, "P3 (lead 12)", false),
            (32, F0Geometry::PRIMED, "primed (continuity, priming)", true),
            (32, F0Geometry::FILLED, "filled (lead 19)", false),
        ];
        let mut ex = RmvpePitchExtractor::load(&model.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let mut out = Vec::new();
        // `SIGIL_SWEEP_TABLE=cost`: only the decoder's CPU cost (below).
        let cost_only = std::env::var("SIGIL_SWEEP_TABLE").as_deref() == Ok("cost");
        for (corpus, items) in corpora.iter().filter(|_| !cost_only) {
            println!(
                "\nP3B-F sweep, table cache: {corpus} ({} clips; 44-frame window, 40 ms blocks, silence skip; scored from frame {FROM})",
                items.len()
            );
            println!("| geometry | cache | decoder | readout lag | call frames (mean / max) | FPE c | GPE % | V/UV % | missed newest % | onset lag | onsets missed | call ms | long calls |");
            println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
            // Per clip / condition (lag 0, decoder off): (config, condition) -> score.
            let mut per_condition: Vec<(String, String, SweepScore)> = Vec::new();
            for &(frames, geometry, gname, with_decoders) in &configs {
                // The model pass, recorded.
                let mut recs: Vec<Vec<Recorded>> = Vec::new();
                let mut base_scores: Vec<SweepScore> = LAGS.iter().map(|_| SweepScore::default()).collect();
                let (mut call_sum, mut call_n, mut call_max, mut long) = (0usize, 0usize, 0usize, 0u64);
                for (x, reference, condition) in items {
                    let mut rec = Recorder {
                        inner: Geo {
                            inner: &mut ex,
                            frames,
                            buf: Vec::new(),
                            sal: Vec::new(),
                        },
                        calls: Vec::new(),
                    };
                    let mut cache = F0Cache::new(F0CacheConfig {
                        margin_frames: DEFAULT_MARGIN_FRAMES,
                        capacity_frames: 88,
                    })
                    .with_geometry(geometry)
                    .with_decode_params(None);
                    let (est, newest, calls) = stream_clip(&mut rec, &mut cache, x, &LAGS);
                    for (li, s) in base_scores.iter_mut().enumerate() {
                        s.add(reference, &est[li], FROM, &newest[li]);
                    }
                    let key = format!("{frames} | {gname}");
                    match per_condition
                        .iter_mut()
                        .find(|(k, c, _)| *k == key && c == condition)
                    {
                        Some((_, _, s)) => s.add(reference, &est[0], FROM, &newest[0]),
                        None => {
                            let mut s = SweepScore::default();
                            s.add(reference, &est[0], FROM, &newest[0]);
                            per_condition.push((key, condition.clone(), s));
                        }
                    }
                    call_sum += calls.iter().sum::<usize>();
                    call_n += calls.len();
                    call_max = call_max.max(calls.iter().copied().max().unwrap_or(0));
                    long += cache.long_calls();
                    recs.push(rec.calls);
                }
                let call_cols = format!("{:.1} / {call_max}", call_sum as f64 / call_n.max(1) as f64);
                let variants: &[(&str, Option<DecodeParams>)] = if with_decoders { &decoders } else { &decoders[..1] };
                for (dname, params) in variants {
                    let mut scores: Vec<SweepScore> = LAGS.iter().map(|_| SweepScore::default()).collect();
                    if params.is_none() {
                        scores = std::mem::take(&mut base_scores);
                    } else {
                        for ((x, reference, _), calls) in items.iter().zip(&recs) {
                            let mut replay = Replay { calls, k: 0, frames };
                            let mut cache = F0Cache::new(F0CacheConfig {
                                margin_frames: DEFAULT_MARGIN_FRAMES,
                                capacity_frames: 88,
                            })
                            .with_geometry(geometry)
                            .with_decode_params(*params);
                            let (est, newest, _) = stream_clip(&mut replay, &mut cache, x, &LAGS);
                            assert_eq!(replay.k, calls.len(), "replay call count");
                            for (li, s) in scores.iter_mut().enumerate() {
                                s.add(reference, &est[li], FROM, &newest[li]);
                            }
                        }
                    }
                    for (li, &la) in LAGS.iter().enumerate() {
                        let s = &mut scores[li];
                        s.call_label = Some(call_cols.clone());
                        let (md, mut json) = s.row(&format!("{frames} | {gname} | {dname} | {la}"));
                        println!("{md} {long} |");
                        json["corpus"] = serde_json::json!(corpus);
                        json["call_frames_mean"] = serde_json::json!(call_sum as f64 / call_n.max(1) as f64);
                        json["call_frames_max"] = serde_json::json!(call_max);
                        json["long_calls"] = serde_json::json!(long);
                        out.push(json);
                    }
                }
            }
            println!("\nper clip / condition, readout lag 0, decoder off ({corpus})");
            println!("| geometry | cache | clip / condition | call frames | FPE c | GPE % | V/UV % | missed newest % | onset lag | onsets missed | call ms |");
            println!("|---|---|---|---|---|---|---|---|---|---|---|");
            per_condition.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
            for (key, condition, s) in &mut per_condition {
                s.call_label = Some(String::new());
                let (md, mut json) = s.row(&format!("{key} | {condition}"));
                println!("{md}");
                json["corpus"] = serde_json::json!(corpus);
                json["per_condition"] = serde_json::json!(true);
                out.push(json);
            }
        }
        // The decoder's CPU cost per tick (this debug test build: an upper bound of the release cost):
        // every tick of the first clip (no silence skip), recorded once, replayed with the decoder
        // off and on.
        {
            fn ticks(x: &[f32], ex: &mut dyn PitchExtractor, cache: &mut F0Cache, times: &mut Vec<f64>) {
                let mut end = 44 * HOP_16K;
                while end <= x.len() {
                    let t0 = std::time::Instant::now();
                    std::hint::black_box(
                        cache
                            .update(ex, &x[end - 44 * HOP_16K..end], (end - 44 * HOP_16K) as u64)
                            .unwrap()[43],
                    );
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    end += 4 * HOP_16K;
                }
            }
            let x = clips[0].1.as_slice();
            let fresh = |params: Option<DecodeParams>| {
                F0Cache::new(F0CacheConfig {
                    margin_frames: DEFAULT_MARGIN_FRAMES,
                    capacity_frames: 88,
                })
                .with_geometry(F0Geometry::PRIMED)
                .with_decode_params(params)
            };
            let calls = {
                let mut rec = Recorder {
                    inner: Geo {
                        inner: &mut ex,
                        frames: 64,
                        buf: Vec::new(),
                        sal: Vec::new(),
                    },
                    calls: Vec::new(),
                };
                ticks(x, &mut rec, &mut fresh(None), &mut Vec::new());
                rec.calls
            };
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..6 {
                // Alternate the order (ABBA) so neither side always runs warm.
                let order = if round % 2 == 0 { [0usize, 1] } else { [1, 0] };
                for i in order {
                    let params = if i == 0 { None } else { Some(base) };
                    let mut replay = Replay {
                        calls: &calls,
                        k: 0,
                        frames: 64,
                    };
                    ticks(x, &mut replay, &mut fresh(params), &mut times[i]);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let q = |t: &[f64], p: f64| t[((t.len() - 1) as f64 * p).round() as usize];
            println!(
                "\ncache update without the model ({} build), per tick: decoder off p50 {:.4} ms p99 {:.4} ms; on p50 {:.4} ms p99 {:.4} ms",
                if cfg!(debug_assertions) { "debug" } else { "release" },
                q(&times[0], 0.5),
                q(&times[0], 0.99),
                q(&times[1], 0.5),
                q(&times[1], 0.99)
            );
            out.push(serde_json::json!({
                "row": "decoder cost (cache update without the model)", "build": if cfg!(debug_assertions) { "debug" } else { "release" },
                "off_p50_ms": q(&times[0], 0.5), "off_p99_ms": q(&times[0], 0.99),
                "on_p50_ms": q(&times[1], 0.5), "on_p99_ms": q(&times[1], 0.99),
            }));
        }
        out
    }

    /// Dev rig: P3's cache (`F0Geometry::LEGACY`, the default) against the opt-in
    /// `F0Geometry::PRIMED` over clips with pauses (`SIGIL_PAUSE_CLIPS=<wav>|<wav>`, 16 kHz mono),
    /// real RMVPE on the CPU at the 64- and the 32-frame geometries, the engine's silence skip
    /// emulated (`stream_clip`): how many
    /// fed frames differ, by how much, and the extractor's call lengths. Equal contours mean the parts
    /// the opt-in turns on do not change what the generator is fed on that material.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache SIGIL_PAUSE_CLIPS=a.wav|b.wav
    /// cargo test pauses_do_not_change_the_fed_f0 -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig: needs SIGIL_TEST_MODELS_DIR with RMVPE and SIGIL_PAUSE_CLIPS"]
    fn pauses_do_not_change_the_fed_f0() {
        use crate::services::neural_audio::ort_session::VcSessionOptions;
        use crate::services::neural_audio::vc::f0_fusion::test_model;
        use crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor;
        let (Some(root), Some(list)) = (std::env::var_os("SIGIL_TEST_MODELS_DIR"), std::env::var("SIGIL_PAUSE_CLIPS").ok()) else {
            eprintln!("SIGIL_TEST_MODELS_DIR or SIGIL_PAUSE_CLIPS is not set; skipped");
            return;
        };
        let _ = root;
        let Some(model) = test_model("pitch-rmvpe") else {
            eprintln!("pitch-rmvpe not found; skipped");
            return;
        };
        let mut ex = RmvpePitchExtractor::load(&model.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        for path in list.split('|').filter(|p| !p.trim().is_empty()) {
            let mut r = hound::WavReader::open(path.trim()).unwrap();
            let spec = r.spec();
            assert_eq!((spec.sample_rate, spec.channels), (16_000, 1), "{path}");
            let x: Vec<f32> = r
                .samples::<i32>()
                .map(|v| v.unwrap() as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
                .collect();
            // The scorer's reference: whole-clip RMVPE of the input.
            let reference = {
                let mut f = ex.f0(&x).unwrap();
                f.resize(x.len() / HOP_16K, 0.0);
                f
            };
            for frames in [64usize, 32] {
                let mut run = |geometry: F0Geometry| {
                    let mut geo = Geo {
                        inner: &mut ex,
                        frames,
                        buf: Vec::new(),
                        sal: Vec::new(),
                    };
                    let mut cache = F0Cache::new(F0CacheConfig {
                        margin_frames: DEFAULT_MARGIN_FRAMES,
                        capacity_frames: 88,
                    })
                    .with_geometry(geometry)
                    .with_decode_params(None);
                    let (est, newest, calls) = stream_clip(&mut geo, &mut cache, &x, &[0]);
                    let whole = calls.iter().filter(|&&c| c >= 44).count();
                    let est = est.into_iter().next().unwrap();
                    let mut score = SweepScore::default();
                    score.add(&reference, &est, 64, &newest[0]);
                    let (_, json) = score.row("");
                    let scored = format!(
                        "FPE {:.2} c, GPE {:.3} %, V/UV {:.2} %",
                        json["fpe_cents"].as_f64().unwrap_or(0.0),
                        json["gpe_pct"].as_f64().unwrap_or(0.0),
                        json["vuv_pct"].as_f64().unwrap_or(0.0)
                    );
                    (est, cache.long_calls(), calls.len(), whole, scored)
                };
                let (a, a_long, ticks, a_whole, a_score) = run(F0Geometry::LEGACY);
                let (b, b_long, _, b_whole, b_score) = run(F0Geometry::PRIMED);
                let mut differ = 0usize;
                let mut worst = 0.0f32;
                for (u, v) in a.iter().zip(&b) {
                    let (u, v) = (if u.is_nan() { 0.0 } else { *u }, if v.is_nan() { 0.0 } else { *v });
                    if u != v {
                        differ += 1;
                        if u > 0.0 && v > 0.0 {
                            worst = worst.max(cents(u, v));
                        }
                    }
                }
                eprintln!(
                    "{path} plan {frames}: {ticks} converted ticks; P3 vs primed: {differ} fed frames differ (worst voiced-in-both difference {worst:.2} cents); \
                     calls of 44+ frames {a_whole} vs {b_whole}; calls over the plan {a_long} vs {b_long}
    against whole-clip RMVPE: P3 {a_score} | primed {b_score}"
                );
            }
        }
    }

    /// Dev rig: the decoder decodes a real RMVPE stream (CPU) and changes its contour, and
    /// `SIGIL_VC_F0_DECODE` reaches a cache built the engine's way (`F0Cache::new`).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache [SIGIL_VC_F0_DECODE=1]
    /// cargo test decoder_runs_on_a_real_rmvpe_stream -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig: needs SIGIL_TEST_MODELS_DIR with RMVPE and the speech clips"]
    fn decoder_runs_on_a_real_rmvpe_stream() {
        use crate::services::neural_audio::ort_session::VcSessionOptions;
        use crate::services::neural_audio::vc::f0_fusion::test_model;
        use crate::services::neural_audio::vc::rmvpe::RmvpePitchExtractor;
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return;
        };
        let (Some(model), Some((_, x))) = (test_model("pitch-rmvpe"), sweep_clips(&root).into_iter().next()) else {
            eprintln!("RMVPE or the clips not found; skipped");
            return;
        };
        let mut ex = RmvpePitchExtractor::load(&model.to_string_lossy(), "cpu", &VcSessionOptions::default()).unwrap();
        let config = F0CacheConfig {
            margin_frames: DEFAULT_MARGIN_FRAMES,
            capacity_frames: 88,
        };
        let mut stream = |cache: &mut F0Cache| -> Vec<f32> {
            let mut fed = Vec::new();
            let mut end = 44 * HOP_16K;
            while end <= x.len() {
                let w = cache
                    .update(&mut ex, &x[end - 44 * HOP_16K..end], (end - 44 * HOP_16K) as u64)
                    .unwrap();
                fed.extend_from_slice(&w[40..44]);
                end += 4 * HOP_16K;
            }
            fed
        };
        let plain = stream(&mut F0Cache::new(config).with_decode_params(None));
        let mut decoded_cache = F0Cache::new(config).with_decode_params(Some(DecodeParams::speech()));
        let decoded = stream(&mut decoded_cache);
        let changed = plain.iter().zip(&decoded).filter(|(a, b)| a != b).count();
        eprintln!(
            "decoded spans {}, spans without salience {}, frames changed {changed} of {}",
            decoded_cache.decode.decoded,
            decoded_cache.decode.no_salience,
            plain.len()
        );
        assert!(decoded_cache.decode.decoded > 0 && decoded_cache.decode.no_salience == 0);
        assert!(changed > 0, "the decoder changed no frame of a real stream");
        let mut env_cache = F0Cache::new(config);
        eprintln!(
            "SIGIL_VC_F0_DECODE={:?}: a cache built with F0Cache::new decodes: {}",
            std::env::var(f0_decode::DECODE_ENV).ok(),
            env_cache.decoding()
        );
        let via_env = stream(&mut env_cache);
        if f0_decode::decode_env_override() == Some(true) {
            assert_eq!(via_env, decoded);
        }
    }

    /// The environment switches over a base geometry: the base stands when nothing is set, the
    /// geometry variable replaces it, and the single-part variables override either (the kill
    /// switches work whatever a preset chose).
    #[test]
    fn env_switches_apply_over_the_base_geometry() {
        let with = |base: F0Geometry, vars: &[(&str, &str)]| {
            F0Geometry::resolve(base, |name| {
                vars.iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(with(F0Geometry::DEFAULT, &[]), F0Geometry::DEFAULT);
        assert_eq!(with(F0Geometry::FILLED, &[]), F0Geometry::FILLED);
        // The kill switch of the lead turns a preset's filled calls back to P3's lead.
        let g = with(F0Geometry::FILLED, &[(F0_LEAD_ENV, "legacy")]);
        assert_eq!(g, F0Geometry::PRIMED);
        // `SIGIL_VC_F0_LEAD` changes only the lead: over the default it leaves priming and continuity
        // off, over the primed preset it is the filled one (`SIGIL_VC_F0_GEOMETRY=filled` is the
        // measured combination in one switch).
        assert_eq!(
            with(F0Geometry::DEFAULT, &[(F0_LEAD_ENV, " FILL ")]),
            F0Geometry {
                lead: LeadMode::Fill,
                ..F0Geometry::DEFAULT
            }
        );
        assert_eq!(with(F0Geometry::PRIMED, &[(F0_LEAD_ENV, " FILL ")]), F0Geometry::FILLED);
        assert_eq!(with(F0Geometry::DEFAULT, &[(F0_LEAD_ENV, "30")]).lead, LeadMode::Frames(30));
        assert_eq!(
            with(F0Geometry::DEFAULT, &[(F0_LEAD_ENV, "9999")]).lead,
            LeadMode::Frames(MAX_HISTORY_FRAMES)
        );
        assert_eq!(with(F0Geometry::FILLED, &[(F0_LEAD_ENV, "nonsense")]), F0Geometry::FILLED);
        // The geometry variable replaces the base; single parts override it.
        assert_eq!(with(F0Geometry::FILLED, &[(F0_GEOMETRY_ENV, "legacy")]), F0Geometry::LEGACY);
        assert_eq!(with(F0Geometry::DEFAULT, &[(F0_GEOMETRY_ENV, "filled")]), F0Geometry::FILLED);
        assert_eq!(with(F0Geometry::DEFAULT, &[(F0_GEOMETRY_ENV, "primed")]), F0Geometry::PRIMED);
        assert_eq!(with(F0Geometry::FILLED, &[(F0_GEOMETRY_ENV, " V2 ")]), F0Geometry::PRIMED);
        let g = with(
            F0Geometry::FILLED,
            &[(F0_GEOMETRY_ENV, "legacy"), (F0_PRIME_ENV, "1"), (F0_CONTINUITY_ENV, "on")],
        );
        assert_eq!(
            g,
            F0Geometry {
                lead: LeadMode::Legacy,
                prime: true,
                continuity: true
            }
        );
        // The kill switches still turn the opt-in parts off ...
        let g = with(F0Geometry::PRIMED, &[(F0_PRIME_ENV, "0"), (F0_CONTINUITY_ENV, "off")]);
        assert!(!g.prime && !g.continuity && g.lead == LeadMode::Legacy);
        // ... and the switches turn them on over the default, one part at a time.
        let g = with(F0Geometry::DEFAULT, &[(F0_PRIME_ENV, "1")]);
        assert!(g.prime && !g.continuity && g.lead == LeadMode::Legacy);
        let g = with(F0Geometry::DEFAULT, &[(F0_CONTINUITY_ENV, "true")]);
        assert!(!g.prime && g.continuity && g.lead == LeadMode::Legacy);
        assert_eq!(
            with(F0Geometry::DEFAULT, &[(F0_PRIME_ENV, "on"), (F0_CONTINUITY_ENV, "yes")]),
            F0Geometry::PRIMED
        );
    }

    /// Rule R6: priming and silence continuity can change the fed F0 (3 to 22 of about 5,000 frames
    /// on the pause clips, one by an octave; no FCPE measurement), so a cache built without any
    /// switch is P3's. Pins the default, the environment wiring, and that the default really runs
    /// P3's calls: a whole-window cold start over a bounded extractor's bound, and a silence skip
    /// that leaves the gap to the next tick's whole-window call.
    #[test]
    fn the_default_geometry_is_p3s_cache_and_the_parts_are_opt_in() {
        assert_eq!(F0Geometry::DEFAULT, F0Geometry::LEGACY);
        assert_eq!(
            F0Geometry::DEFAULT,
            F0Geometry {
                lead: LeadMode::Legacy,
                prime: false,
                continuity: false
            }
        );
        assert_ne!(F0Geometry::DEFAULT, F0Geometry::PRIMED);
        assert_eq!(
            F0Geometry::PRIMED,
            F0Geometry {
                lead: LeadMode::Legacy,
                prime: true,
                continuity: true
            }
        );
        // No switch set: the resolved geometry is the default.
        assert_eq!(F0Geometry::resolve(F0Geometry::DEFAULT, |_| None), F0Geometry::DEFAULT);
        // `F0Cache::new` (what the engine uses) reads the process environment into `DEFAULT`
        // (skipped when the developer's shell sets one of the switches).
        let set = [F0_PRIME_ENV, F0_CONTINUITY_ENV, F0_GEOMETRY_ENV, F0_LEAD_ENV];
        if set.iter().all(|v| std::env::var_os(v).is_none()) {
            let cache = F0Cache::new(F0CacheConfig {
                margin_frames: DEFAULT_MARGIN_FRAMES,
                capacity_frames: 88,
            });
            assert_eq!(cache.geometry(), F0Geometry::DEFAULT);
            assert_eq!(F0Geometry::from_env(), F0Geometry::DEFAULT);
        }
        // Behaviour: a bounded extractor's cold start is one whole-window call (not primed) ...
        let audio = voice(16_000 * 3);
        let (ctx, block) = (44usize, 4usize);
        let mut ex = Shaped::new(3, Some(31), Some(31));
        let mut cache = cache_for(ctx, F0Geometry::DEFAULT);
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        assert_eq!(ex.calls, vec![44], "the default cold start is P3's single whole-window call");
        assert_eq!(cache.long_calls(), 1);
        // ... and a skipped block leaves the cache alone, so the first voiced tick after the pause
        // extracts its whole window again.
        let mut ex = Shaped::new(3, None, None);
        let mut cache = cache_for(ctx, F0Geometry::DEFAULT);
        let mut pos = 0usize;
        for tick in 0..30usize {
            let window = &audio[pos..pos + ctx * HOP_16K];
            if (20..25).contains(&tick) {
                cache.advance_unvoiced((pos / HOP_16K + ctx - block) as u64, block);
            } else {
                cache.update(&mut ex, window, pos as u64).unwrap();
                if tick == 25 {
                    assert_eq!(cache.last_extracted_frames(), ctx, "no continuity: a whole-window call after the pause");
                }
            }
            pos += block * HOP_16K;
        }
        // Opting in (the same switches the benchmarks flip) gives the P3B-F behaviour.
        let opted = F0Geometry::resolve(F0Geometry::DEFAULT, |name| match name {
            F0_PRIME_ENV | F0_CONTINUITY_ENV => Some("1".to_string()),
            _ => None,
        });
        assert_eq!(opted, F0Geometry::PRIMED);
        let mut ex = Shaped::new(3, Some(31), Some(31));
        let mut cache = cache_for(ctx, opted);
        cache.update(&mut ex, &audio[..ctx * HOP_16K], 0).unwrap();
        assert_eq!(ex.calls, vec![31, 31, 22], "primed: slices within the bound");
        assert_eq!(cache.long_calls(), 0);
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
