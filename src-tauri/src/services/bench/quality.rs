//! Quality metrics against the offline ceiling (P3-F; plan T5 / T6).
//!
//! Pure metric functions (no models, no files) used by the `quality` and `compare` run modes
//! (`fidelity.rs`, `compare.rs`) and the autotuner:
//! - [`mcd_db`]: mel-cepstral distortion of two time-aligned clips (natural-log amplitude
//!   mel-cepstra `c1..c24` from 40 mel bands, 25 ms / 10 ms frames, speech frames only; the usual
//!   `10 / ln 10 · √(2 Σ Δc²)` per frame, averaged). Sigil's own mel code (`dsp.rs`): the RMVPE mel
//!   front end in `vc/rmvpe.rs` is private to the pitch bank.
//! - [`f0_errors`]: voiced F0 RMSE in cents, gross pitch error (both voiced, off by more than
//!   20 %) and V/UV error (voicing decisions that differ, over every frame given) of two contours on
//!   the 10 ms grid. The contours come from one [`F0Analyzer`] (offline RMVPE when its model is
//!   installed, else DIO), so both clips are judged by the same extractor.
//! - Clicks: [`click_events`] finds impulsive events (a spike of the second difference far above
//!   its local level: discontinuities and block-edge pops, not fricatives or voicing);
//!   [`count_block_clicks`] counts those at block edges, [`click_report`] also locates the edge
//!   phase when it is unknown, and [`unmatched`] keeps the events a reference rendition does not
//!   have (content transients such as plosives are in both). [`seam_flux_ratio`] is the spectral-flux
//!   seam statistic of P3-C1's render test (max over the 16 phases of the block period / median;
//!   about 1 when joins are inaudible).
//! - [`cer`]: character error rate of a transcript (Unicode letters and digits, case-folded; the
//!   transcripts come from Sigil's own STT, `scorers.rs`).
//!
//! CAM++ speaker cosine and DNSMOS P.835 are model-based and live in `scorers.rs`; the metric
//! record of one clip is [`QualityMetrics`].
//!
//! # Frozen by P3-S0 (signatures; P3-F implements)
//! [`QualityMetrics`], [`F0Errors`], [`mcd_db`], [`f0_errors`], [`count_block_clicks`]. Both
//! structs gained fields (P3-F); nothing was renamed.

use std::path::Path;

use serde::Serialize;

use super::dsp::{mel_cepstra, MelCepConfig};
use super::metrics::Metrics;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::engine::options::F0Range;
use crate::services::neural_audio::vc::traits::PitchExtractor;

/// Scores of one converted clip against its ceiling (absent = not measured).
#[derive(Clone, Debug, Default, Serialize)]
pub struct QualityMetrics {
    /// Mel-cepstral distortion against the ceiling (dB).
    pub mcd_db: Option<f64>,
    /// F0 RMSE over frames voiced in both (cents).
    pub f0_rmse_cents: Option<f64>,
    /// Frames voiced in both whose F0 differs by more than 20 % (0..1).
    pub gross_pitch_error: Option<f64>,
    /// Frames whose voicing decision differs (0..1).
    pub vuv_error: Option<f64>,
    /// CAM++ cosine against the reference voice / ceiling (-1..1).
    pub speaker_cosine: Option<f64>,
    /// DNSMOS P.835 overall (1..5).
    pub dnsmos_ovrl: Option<f64>,
    /// Character error rate of the transcript (0..1, can exceed 1).
    pub cer: Option<f64>,
    /// Clicks at block edges (artifact events, see the module docs).
    pub clicks: Option<u64>,
    // ---- P3-F ----
    pub dnsmos_sig: Option<f64>,
    pub dnsmos_bak: Option<f64>,
    /// Spectral-flux seam statistic (1 = no seam energy).
    pub seam_flux_ratio: Option<f64>,
    /// Impulsive events the reference rendition does not have, anywhere in the clip.
    pub artifact_events: Option<u64>,
    /// Frames the F0 errors were computed over.
    pub f0_frames: Option<u64>,
    /// Time offset of the clip against its reference (ms; alignment).
    pub offset_ms: Option<f64>,
    /// Audio scored (s).
    pub seconds: Option<f64>,
}

impl QualityMetrics {
    /// `(name, value)` of every measured field (the report's metric names).
    pub fn pairs(&self) -> Vec<(&'static str, f64)> {
        let mut v = Vec::new();
        let mut put = |k: &'static str, x: Option<f64>| {
            if let Some(x) = x.filter(|x| x.is_finite()) {
                v.push((k, x));
            }
        };
        put("mcd_db", self.mcd_db);
        put("f0_rmse_cents", self.f0_rmse_cents);
        put("gross_pitch_error", self.gross_pitch_error);
        put("vuv_error", self.vuv_error);
        put("speaker_cosine", self.speaker_cosine);
        put("dnsmos_ovrl", self.dnsmos_ovrl);
        put("dnsmos_sig", self.dnsmos_sig);
        put("dnsmos_bak", self.dnsmos_bak);
        put("cer", self.cer);
        put("clicks", self.clicks.map(|c| c as f64));
        put("artifact_events", self.artifact_events.map(|c| c as f64));
        put("seam_flux_ratio", self.seam_flux_ratio);
        put("f0_frames", self.f0_frames.map(|c| c as f64));
        put("offset_ms", self.offset_ms);
        put("quality_seconds", self.seconds);
        v
    }

    /// Insert every measured field as `<prefix><name>` (prefix `""` for the plain names).
    pub fn insert_into(&self, prefix: &str, metrics: &mut Metrics) {
        for (k, v) in self.pairs() {
            metrics.insert(format!("{prefix}{k}"), v);
        }
    }
}

/// F0 errors of a contour against a reference (same 10 ms grid).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct F0Errors {
    pub rmse_cents: f64,
    pub gross_pitch_error: f64,
    pub vuv_error: f64,
    // ---- P3-F ----
    /// Frames voiced in both (the RMSE / GPE population).
    pub voiced_frames: usize,
    /// Frames compared (the V/UV population).
    pub frames: usize,
}

/// Relative F0 deviation above which a frame is a gross pitch error (the usual 20 %).
pub const GROSS_PITCH_RATIO: f64 = 0.2;

/// `10 / ln 10`: natural-log cepstral distance to dB.
const MCD_K: f64 = 10.0 / std::f64::consts::LN_10;
/// Frames more than this far (dB) below the reference's loudest frame are not speech for MCD.
const MCD_SPEECH_RANGE_DB: f64 = 40.0;

/// Mel-cepstral distortion (dB) between two time-aligned clips at `sample_rate` (the common
/// length is compared; frames where the reference is more than 40 dB below its loudest frame are
/// skipped as silence).
pub fn mcd_db(reference: &[f32], test: &[f32], sample_rate: u32) -> Result<f64, String> {
    let n = reference.len().min(test.len());
    let cfg = MelCepConfig::default();
    let (cr, er) = mel_cepstra(&reference[..n], sample_rate, &cfg);
    let (ct, _) = mel_cepstra(&test[..n], sample_rate, &cfg);
    if cr.is_empty() {
        return Err(format!("MCD: the clips are shorter than one {} ms frame", cfg.frame_ms));
    }
    let peak = er.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut count = 0usize;
    for ((a, b), &e) in cr.iter().zip(&ct).zip(&er) {
        if e < peak - MCD_SPEECH_RANGE_DB {
            continue;
        }
        let d2: f64 = (1..=cfg.order).map(|d| (a[d] - b[d]) * (a[d] - b[d])).sum();
        sum += MCD_K * (2.0 * d2).sqrt();
        count += 1;
    }
    if count == 0 {
        return Err("MCD: the reference has no speech frames".into());
    }
    Ok(sum / count as f64)
}

/// F0 errors of `test` against `reference` (Hz per frame, 0 = unvoiced; the common length is
/// compared). RMSE and GPE are over frames voiced in both; V/UV over every frame. An error when no
/// frame is voiced in both.
pub fn f0_errors(reference: &[f32], test: &[f32]) -> Result<F0Errors, String> {
    let n = reference.len().min(test.len());
    if n == 0 {
        return Err("F0 errors: empty contours".into());
    }
    let (mut sq, mut gross, mut both, mut vuv) = (0.0f64, 0usize, 0usize, 0usize);
    for (&r, &t) in reference[..n].iter().zip(&test[..n]) {
        let (rv, tv) = (r.is_finite() && r > 0.0, t.is_finite() && t > 0.0);
        if rv != tv {
            vuv += 1;
        }
        if rv && tv {
            let ratio = f64::from(t) / f64::from(r);
            let cents = 1200.0 * ratio.log2();
            sq += cents * cents;
            if (ratio - 1.0).abs() > GROSS_PITCH_RATIO {
                gross += 1;
            }
            both += 1;
        }
    }
    if both == 0 {
        return Err("F0 errors: no frame is voiced in both contours".into());
    }
    Ok(F0Errors {
        rmse_cents: (sq / both as f64).sqrt(),
        gross_pitch_error: gross as f64 / both as f64,
        vuv_error: vuv as f64 / n as f64,
        voiced_frames: both,
        frames: n,
    })
}

// ---------------------------------------------------------------------------------------------
// Clicks
// ---------------------------------------------------------------------------------------------

/// Click detector settings.
#[derive(Debug, Clone, Copy)]
pub struct ClickConfig {
    /// A cell is an event when its second-difference peak exceeds this many times the local level.
    pub threshold: f64,
    /// Analysis cell (ms).
    pub cell_ms: f64,
    /// Cells on each side that set the local level (the two nearest are skipped).
    pub context_cells: usize,
    /// An event this close to an edge (ms) is at the edge.
    pub tolerance_ms: f64,
}

impl Default for ClickConfig {
    fn default() -> Self {
        Self {
            threshold: 6.0,
            cell_ms: 1.0,
            context_cells: 10,
            tolerance_ms: 1.5,
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Impulsive events of `audio`: sample positions where the second difference
/// `x[n] − 2x[n−1] + x[n−2]` (a strong high-pass: a step or a pop stands out, smooth tones do not)
/// peaks at more than `threshold` times its level in the surrounding cells. Adjacent event cells
/// merge into one event at their largest peak. Voiced speech has such events too (glottal pulses
/// of a pulse-like voice), so clicks are judged against a reference rendition ([`unmatched`]) or
/// against the rate away from the block edges ([`count_block_clicks`]).
pub fn click_events(audio: &[f32], sample_rate: u32, cfg: &ClickConfig) -> Vec<usize> {
    click_events_scored(audio, sample_rate, cfg).into_iter().map(|e| e.0).collect()
}

/// [`click_events`] with each event's strength (peak over the local level).
pub fn click_events_scored(audio: &[f32], sample_rate: u32, cfg: &ClickConfig) -> Vec<(usize, f64)> {
    let n = audio.len();
    if n < 8 {
        return Vec::new();
    }
    let cell = ((cfg.cell_ms / 1000.0) * f64::from(sample_rate)).round().max(4.0) as usize;
    let mut h = vec![0.0f64; n];
    for i in 2..n {
        h[i] = f64::from(audio[i]) - 2.0 * f64::from(audio[i - 1]) + f64::from(audio[i - 2]);
    }
    let cells: Vec<(f64, f64, usize)> = h
        .chunks(cell)
        .enumerate()
        .map(|(c, chunk)| {
            let rms = (chunk.iter().map(|v| v * v).sum::<f64>() / chunk.len() as f64).sqrt();
            let (k, peak) = chunk
                .iter()
                .enumerate()
                .map(|(k, v)| (k, v.abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            (rms, peak, c * cell + k)
        })
        .collect();
    // Level floor: quiet passages must not turn dither into events.
    let mut rms_sorted: Vec<f64> = cells.iter().map(|c| c.0).collect();
    rms_sorted.sort_by(f64::total_cmp);
    let p95 = rms_sorted[((rms_sorted.len() - 1) as f64 * 0.95) as usize];
    let floor = (0.02 * p95).max(2e-4);
    // The local level is the louder of the medians before and after the cell (the two nearest
    // cells skipped): a click in steady audio stands out from both, while the first cell of a
    // noise burst (a fricative after a pause) does not stand out from what follows it.
    let k = cfg.context_cells.max(2);
    let mut strength = vec![0.0f64; cells.len()];
    let (mut pre, mut post) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for (c, &(_, peak, _)) in cells.iter().enumerate() {
        pre.clear();
        post.clear();
        pre.extend(cells[c.saturating_sub(k)..c.saturating_sub(1)].iter().map(|x| x.0));
        post.extend(cells[(c + 2).min(cells.len())..(c + k + 1).min(cells.len())].iter().map(|x| x.0));
        let level = median(&mut pre).max(median(&mut post)).max(floor);
        strength[c] = peak / level;
    }
    let mut events = Vec::new();
    let mut c = 0;
    while c < cells.len() {
        if strength[c] <= cfg.threshold {
            c += 1;
            continue;
        }
        let mut best = c;
        while c < cells.len() && strength[c] > cfg.threshold {
            if cells[c].1 > cells[best].1 {
                best = c;
            }
            c += 1;
        }
        events.push((cells[best].2, strength[best]));
    }
    events
}

/// Distance of `pos` to the nearest multiple of `period` shifted by `phase`.
fn edge_distance(pos: usize, period: usize, phase: usize) -> usize {
    let r = (pos + period - phase % period) % period;
    r.min(period - r)
}

/// Clicks at block edges (every `block_frames` samples from sample 0): impulsive events
/// ([`click_events`], default settings) within 1.5 ms of an edge, in excess of what the event
/// rate away from the edges predicts there (plus two standard deviations), so the pitch pulses of
/// voiced speech, which fall on the edges by chance, do not count.
#[allow(dead_code)] // P3-S0 frozen API (audio without a reference rendition, e.g. recordings).
pub fn count_block_clicks(audio: &[f32], sample_rate: u32, block_frames: usize) -> u64 {
    if block_frames == 0 {
        return 0;
    }
    let cfg = ClickConfig::default();
    let tol = ((cfg.tolerance_ms / 1000.0) * f64::from(sample_rate)).round() as usize;
    let width = (2 * tol + 1).min(block_frames);
    let events = click_events(audio, sample_rate, &cfg);
    let at_edges = events.iter().filter(|&&p| edge_distance(p, block_frames, 0) <= tol).count();
    if width >= block_frames {
        return at_edges as u64;
    }
    let away = (events.len() - at_edges) as f64;
    let expected = away * width as f64 / (block_frames - width) as f64;
    let allowance = (expected + 2.0 * expected.sqrt()).ceil();
    (at_edges as f64 - allowance).max(0.0) as u64
}

/// Events of `a` (sorted positions) with no event of `b` within `tol` samples (`b` sorted). With
/// `b` detected at a lower threshold than `a` (hysteresis), an impulse just above the threshold in
/// one rendition and just below it in the other is not an artifact.
pub fn unmatched(a: &[usize], b: &[usize], tol: usize) -> Vec<usize> {
    a.iter()
        .copied()
        .filter(|&p| {
            let i = b.partition_point(|&q| q + tol < p);
            b.get(i).is_none_or(|&q| q > p + tol)
        })
        .collect()
}

/// Where a periodic stream's clicks are.
#[derive(Debug, Clone, Serialize)]
pub struct ClickReport {
    pub events: usize,
    /// Block period (samples) and the edge phase used (given, or the busiest one found).
    pub period: usize,
    pub phase: usize,
    pub phase_given: bool,
    /// Events within the tolerance of an edge.
    pub edge_events: usize,
    /// Edge events expected by chance if the events were spread evenly.
    pub expected_by_chance: f64,
    /// `edge_events` when the phase was given; with a searched phase, the excess over chance when
    /// it is clearly phase-locked (at least 3 events, and 3x chance plus two standard deviations
    /// and one), else 0.
    pub clicks: u64,
    pub events_per_min: f64,
}

/// Clicks of `events` (sample positions in a clip of `len` samples at `sample_rate`) on a block
/// grid of `period` samples. With `phase = None` the busiest phase is searched: seam clicks line
/// up on one phase, content transients do not.
pub fn click_report(events: &[usize], len: usize, sample_rate: u32, period: usize, phase: Option<usize>, tolerance_ms: f64) -> ClickReport {
    let minutes = len as f64 / f64::from(sample_rate.max(1)) / 60.0;
    let per_min = if minutes > 0.0 { events.len() as f64 / minutes } else { 0.0 };
    let tol = ((tolerance_ms / 1000.0) * f64::from(sample_rate)).round() as usize;
    if period == 0 {
        return ClickReport {
            events: events.len(),
            period,
            phase: 0,
            phase_given: phase.is_some(),
            edge_events: 0,
            expected_by_chance: 0.0,
            clicks: 0,
            events_per_min: per_min,
        };
    }
    let width = (2 * tol + 1).min(period);
    let expected = events.len() as f64 * width as f64 / period as f64;
    if width == period {
        // Every position is within the tolerance of an edge: nothing to tell apart.
        return ClickReport {
            events: events.len(),
            period,
            phase: phase.unwrap_or(0) % period,
            phase_given: phase.is_some(),
            edge_events: events.len(),
            expected_by_chance: expected,
            clicks: if phase.is_some() { events.len() as u64 } else { 0 },
            events_per_min: per_min,
        };
    }
    let count_at = |ph: usize| events.iter().filter(|&&p| edge_distance(p, period, ph) <= tol).count();
    let (phase_used, edge) = match phase {
        Some(ph) => (ph % period, count_at(ph)),
        None => {
            // Histogram by phase, then the circular window of `width` with the most events.
            let mut hist = vec![0usize; period];
            for &p in events {
                hist[p % period] += 1;
            }
            let mut window: usize = (0..width).map(|i| hist[(period - tol + i) % period]).sum();
            let (mut best, mut best_ph) = (window, 0usize);
            for ph in 1..period {
                window -= hist[(ph + period - tol - 1) % period];
                window += hist[(ph + tol) % period];
                if window > best {
                    best = window;
                    best_ph = ph;
                }
            }
            (best_ph, best)
        }
    };
    let clicks = match phase {
        Some(_) => edge as u64,
        None if edge >= 3 && edge as f64 >= 3.0 * expected + 2.0 * expected.sqrt() + 1.0 => (edge as f64 - expected).round().max(0.0) as u64,
        None => 0,
    };
    ClickReport {
        events: events.len(),
        period,
        phase: phase_used,
        phase_given: phase.is_some(),
        edge_events: edge,
        expected_by_chance: expected,
        clicks,
        events_per_min: per_min,
    }
}

/// Spectral-flux seam statistic of a stream whose blocks repeat every `period` samples: positive
/// spectral flux on a hop of `period / 16` (4-hop Hann frames), averaged per phase of the period;
/// the loudest phase over the median phase. About 1 when the joins are inaudible; seams pile
/// flux onto one phase. `None` for periods under 64 samples or clips under four periods.
pub fn seam_flux_ratio(audio: &[f32], sample_rate: u32, period: usize) -> Option<f64> {
    let _ = sample_rate;
    if period < 64 || audio.len() < 4 * period {
        return None;
    }
    const PHASES: usize = 16;
    let hop = period / PHASES;
    let win = 4 * hop;
    let n_fft = win.next_power_of_two();
    let window = super::dsp::hann_symmetric(win);
    let spectra = super::dsp::power_spectra(audio, win, hop, n_fft, &window);
    let mut sums = [0.0f64; PHASES];
    let mut counts = [0usize; PHASES];
    for (i, pair) in spectra.windows(2).enumerate() {
        let flux: f64 = pair[1]
            .iter()
            .zip(&pair[0])
            .map(|(b, a)| (b.sqrt() - a.sqrt()).max(0.0))
            .sum();
        // Phase of the frame centre within the period.
        let centre = (i + 1) * hop + win / 2;
        let ph = (centre % period) * PHASES / period;
        sums[ph] += flux;
        counts[ph] += 1;
    }
    let mut means: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .filter(|(_, &c)| c > 0)
        .map(|(s, &c)| s / c as f64)
        .collect();
    let top = means.iter().cloned().fold(0.0, f64::max);
    let med = median(&mut means);
    (med > 1e-12).then(|| top / med)
}

// ---------------------------------------------------------------------------------------------
// Character error rate
// ---------------------------------------------------------------------------------------------

/// Characters CER compares: Unicode letters and digits, lowercased (punctuation, spaces and
/// symbols are dropped, so every script, CJK included, is compared character by character).
pub fn cer_chars(s: &str) -> Vec<char> {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Levenshtein distance of two character sequences.
pub fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Character error rate of `hypothesis` against `reference` (edits / reference characters);
/// `None` when the reference has no characters.
pub fn cer(reference: &str, hypothesis: &str) -> Option<f64> {
    let r = cer_chars(reference);
    if r.is_empty() {
        return None;
    }
    let h = cer_chars(hypothesis);
    Some(edit_distance(&r, &h) as f64 / r.len() as f64)
}

// ---------------------------------------------------------------------------------------------
// F0 analysis
// ---------------------------------------------------------------------------------------------

/// The F0 extractor both renditions of a comparison are analysed with (16 kHz input, 10 ms grid).
pub struct F0Analyzer {
    extractor: Box<dyn PitchExtractor>,
    /// The method that runs (`rmvpe`, `dio`, ...).
    pub method: &'static str,
    /// Why another method runs than the one asked for.
    pub notes: Vec<String>,
}

impl F0Analyzer {
    /// `method`: `auto` (offline RMVPE when its model is found, else DIO), or any pitch-bank
    /// method (`rmvpe`, `fcpe`, `swiftf0`, `crepe`, `pesto`, `fusion`, `dio`). `model` is an explicit
    /// RMVPE / FCPE file. Never fails: a method whose model is missing falls back to DIO (pure Rust,
    /// always available) with a note.
    pub fn new(method: &str, model: Option<&Path>, ep: &str, range: F0Range) -> F0Analyzer {
        use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor, resolve_pitch_model, PitchBankRequest};
        let app_data = crate::services::diagnostics::app_data_dir();
        let wanted = match method.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => {
                let found = model.is_some_and(Path::is_file) || resolve_pitch_model("rmvpe", app_data.as_deref()).is_some();
                if found {
                    "rmvpe".to_string()
                } else {
                    "dio".to_string()
                }
            }
            m => m.to_string(),
        };
        // Offline analysis: two threads, background priority rules (no worker priority).
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            component_prefix: Some("bench.analysis"),
            ..VcSessionOptions::default()
        };
        let model = model.filter(|p| !p.as_os_str().is_empty());
        let req = PitchBankRequest {
            method: &wanted,
            ep,
            opts: &opts,
            range,
            fcpe_path: model.filter(|_| wanted == "fcpe"),
            rmvpe_path: model.filter(|_| wanted == "rmvpe" || wanted == "fusion"),
            app_data: app_data.as_deref(),
            small_models_on_cpu: true,
        };
        let mut notes = Vec::new();
        if method.trim().eq_ignore_ascii_case("auto") && wanted == "dio" {
            notes.push("F0 analysis: no RMVPE model installed (vc/pitch-rmvpe); using DIO".to_string());
        }
        match create_pitch_extractor(&req) {
            Ok(bank) => {
                notes.extend(bank.notes);
                F0Analyzer {
                    extractor: bank.extractor,
                    method: bank.method,
                    notes,
                }
            }
            Err(e) => {
                notes.push(format!("F0 analysis: {wanted} could not load ({e}); using DIO"));
                F0Analyzer::dio(range, notes)
            }
        }
    }

    /// DIO (pure Rust; no model).
    pub fn dio(range: F0Range, notes: Vec<String>) -> F0Analyzer {
        F0Analyzer {
            extractor: Box::new(crate::services::neural_audio::vc::world::DioPitchExtractor::new().with_range(range)),
            method: "dio",
            notes,
        }
    }

    /// F0 (Hz, 0 = unvoiced) per 10 ms frame of a 16 kHz clip.
    pub fn contour(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>, String> {
        self.extractor.f0(audio_16k)
    }
}

/// Frames (10 ms at 16 kHz) of `audio_16k` within `range_db` of its loudest frame: the speech
/// region V/UV errors are counted over (silence would dilute them).
pub fn speech_frames(audio_16k: &[f32], range_db: f64) -> Vec<bool> {
    let env = super::dsp::log_energy_envelope(audio_16k, 160);
    let peak = env.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    env.into_iter().map(|e| e >= peak - range_db).collect()
}

/// [`f0_errors`] over the frames where `mask` is set (frames beyond the mask are skipped).
pub fn f0_errors_masked(reference: &[f32], test: &[f32], mask: &[bool]) -> Result<F0Errors, String> {
    let (r, t): (Vec<f32>, Vec<f32>) = reference
        .iter()
        .zip(test)
        .zip(mask)
        .filter(|(_, &m)| m)
        .map(|((&a, &b), _)| (a, b))
        .unzip();
    f0_errors(&r, &t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::signals::{synth, SplitMix64};
    use crate::services::bench::spec::SignalKind;
    use std::f64::consts::PI;

    fn sine(freq: f64, sr: u32, secs: f64, amp: f64) -> Vec<f32> {
        (0..(secs * f64::from(sr)) as usize)
            .map(|i| (amp * (2.0 * PI * freq * i as f64 / f64::from(sr)).sin()) as f32)
            .collect()
    }

    #[test]
    fn mcd_is_zero_for_identical_clips_and_grows_with_distortion() {
        let v = synth(SignalKind::Voice, 16_000, 3.0, 0.5, 5);
        assert!(mcd_db(&v, &v, 16_000).unwrap() < 1e-9);
        // A gain change moves only c0, which MCD excludes.
        let loud: Vec<f32> = v.iter().map(|x| x * 1.5).collect();
        assert!(mcd_db(&v, &loud, 16_000).unwrap() < 1e-3, "f32 rounding only");
        // Added noise and a spectral tilt raise it, more for more distortion.
        let mut rng = SplitMix64::new(9);
        let noisy = |amp: f64, rng: &mut SplitMix64| -> Vec<f32> { v.iter().map(|x| x + (amp * rng.range(-1.0, 1.0)) as f32).collect() };
        let a = mcd_db(&v, &noisy(0.003, &mut rng), 16_000).unwrap();
        let b = mcd_db(&v, &noisy(0.03, &mut rng), 16_000).unwrap();
        assert!(a > 0.05 && b > a + 1.0, "{a} {b}");
        let mut tilted = v.clone();
        for i in (1..tilted.len()).rev() {
            tilted[i] -= 0.9 * tilted[i - 1];
        }
        let t = mcd_db(&v, &tilted, 16_000).unwrap();
        // A first-order tilt 1 - 0.9 z^-1 has c_n = -0.9^n / (2n): 3.2 dB on a linear axis, a little
        // more on the mel axis. (MCD on an orthonormal DCT would read about 9x that.)
        assert!(t > 2.5 && t < 4.5, "{t}");
        assert!(mcd_db(&v[..100], &v[..100], 16_000).is_err(), "shorter than a frame");
        assert!(mcd_db(&vec![0.0; 16_000], &v, 16_000).unwrap() >= 0.0, "silence counts as the loudest frame");
    }

    #[test]
    fn f0_errors_on_known_contours() {
        // 100 frames at 200 Hz; the test is 10 cents sharp, with an octave error on 5 frames and
        // a voicing flip on 3.
        let r = vec![200.0f32; 100];
        let sharp = 200.0 * 2f32.powf(10.0 / 1200.0);
        let mut t = vec![sharp; 100];
        for f in t.iter_mut().take(5) {
            *f = 400.0;
        }
        for f in t.iter_mut().skip(97) {
            *f = 0.0;
        }
        let e = f0_errors(&r, &t).unwrap();
        assert_eq!((e.voiced_frames, e.frames), (97, 100));
        assert!((e.gross_pitch_error - 5.0 / 97.0).abs() < 1e-9);
        assert!((e.vuv_error - 0.03).abs() < 1e-9);
        let want = ((5.0 * 1200.0f64 * 1200.0 + 92.0 * 100.0) / 97.0).sqrt();
        assert!((e.rmse_cents - want).abs() < 0.01, "{} vs {want}", e.rmse_cents);
        let same = f0_errors(&r, &r).unwrap();
        assert_eq!((same.rmse_cents, same.gross_pitch_error, same.vuv_error), (0.0, 0.0, 0.0));
        assert!(f0_errors(&[0.0; 4], &[0.0; 4]).is_err());
        assert!(f0_errors(&[], &[]).is_err());
        // The mask drops frames (here the voicing flips).
        let mask: Vec<bool> = (0..100).map(|i| i < 97).collect();
        assert_eq!(f0_errors_masked(&r, &t, &mask).unwrap().vuv_error, 0.0);
    }

    /// DIO on analytic signals: a steady tone and its shifted copy give the right cents.
    #[test]
    fn f0_analyzer_measures_a_known_shift() {
        let mut a = F0Analyzer::dio(F0Range::Speech, Vec::new());
        let x = sine(150.0, 16_000, 1.5, 0.4);
        let up = sine(150.0 * 2f64.powf(50.0 / 1200.0), 16_000, 1.5, 0.4);
        let (fx, fu) = (a.contour(&x).unwrap(), a.contour(&up).unwrap());
        let e = f0_errors(&fx[10..fx.len() - 10], &fu[10..fu.len() - 10]).unwrap();
        assert!((e.rmse_cents - 50.0).abs() < 10.0, "{e:?}");
        assert_eq!(e.gross_pitch_error, 0.0);
        assert_eq!(a.method, "dio");
    }

    #[test]
    fn clicks_at_block_edges_are_counted_and_nothing_else() {
        let sr = 48_000;
        let block = 1_920;
        let tone = sine(220.0, sr, 2.0, 0.4);
        // A clean tone: no events at all. The synthetic voice (a band-limited sawtooth source) has
        // pitch pulses, but they are not phase-locked to the blocks: no block clicks.
        assert!(click_events(&tone, sr, &ClickConfig::default()).is_empty());
        assert_eq!(count_block_clicks(&tone, sr, block), 0);
        let voice = synth(SignalKind::Voice, sr, 4.0, 0.5, 2);
        let pulses = click_events(&voice, sr, &ClickConfig::default());
        assert!(!pulses.is_empty(), "pitch pulses are impulsive");
        assert_eq!(count_block_clicks(&voice, sr, block), 0);
        assert_eq!(click_report(&pulses, voice.len(), sr, block, None, 1.5).clicks, 0);
        assert!(unmatched(&pulses, &pulses, 0).is_empty());
        // Alternating +-0.05 steps at every block edge: every edge counts.
        let edges = (1..tone.len() / block).map(|k| k * block);
        let clicky = {
            let mut y = tone.clone();
            for (i, p) in edges.clone().enumerate() {
                let s = if i % 2 == 0 { 0.05 } else { -0.05 };
                for v in y.iter_mut().skip(p) {
                    *v += s;
                }
            }
            y
        };
        let n_edges = edges.count() as u64;
        assert_eq!(count_block_clicks(&clicky, sr, block), n_edges);
        // Single-sample pops off the grid (phase 700): not block clicks, but the phase search
        // finds them and they are artifact events against the clean tone.
        let mut off = tone.clone();
        for k in 1..tone.len() / block {
            off[k * block + 700] += 0.08;
        }
        assert_eq!(count_block_clicks(&off, sr, block), 0);
        let ev = click_events(&off, sr, &ClickConfig::default());
        let r = click_report(&ev, off.len(), sr, block, None, 1.5);
        assert_eq!(r.edge_events as u64, n_edges, "{r:?}");
        assert!((r.phase as isize - 700).abs() <= 72, "{r:?}");
        assert!(r.clicks as f64 >= n_edges as f64 - r.expected_by_chance - 1.0, "{r:?}");
        let clean = click_events(&tone, sr, &ClickConfig::default());
        assert_eq!(unmatched(&ev, &clean, 72).len(), ev.len());
        assert!(unmatched(&ev, &ev, 0).is_empty());
        // Random single clicks: found as events, but not phase-locked.
        let mut rng = SplitMix64::new(3);
        let mut sparse = voice.clone();
        let mut spots = Vec::new();
        for _ in 0..12 {
            let p = (rng.unit() * (sparse.len() - 10) as f64) as usize;
            sparse[p] += 0.2;
            spots.push(p);
        }
        let ev = click_events(&sparse, sr, &ClickConfig::default());
        let added = unmatched(&ev, &click_events(&voice, sr, &ClickConfig { threshold: 3.0, ..ClickConfig::default() }), 24);
        assert!(added.len() >= 10, "{} of 12 pops found against the clean voice", added.len());
        let r = click_report(&added, sparse.len(), sr, block, None, 1.5);
        assert!(r.clicks <= 1, "{r:?}");
        // The voice with a pop at every edge: the edges stand out over the pitch pulses.
        let mut popped = voice.clone();
        let n_voice_edges = popped.len() / block - 1;
        for k in 1..=n_voice_edges {
            popped[k * block] += 0.2;
        }
        let c = count_block_clicks(&popped, sr, block);
        assert!(c as usize >= n_voice_edges * 3 / 4 && c as usize <= n_voice_edges, "{c} of {n_voice_edges}");
    }

    #[test]
    fn seam_flux_ratio_sees_periodic_seams() {
        let sr = 48_000;
        let period = 1_920;
        let voice = synth(SignalKind::Voice, sr, 6.0, 0.5, 8);
        let clean = seam_flux_ratio(&voice, sr, period).unwrap();
        // Misaligned joins: every block continues the voice 1 ms (48 samples) further on, a
        // waveform jump at each seam (what a join without SOLA alignment sounds like).
        let seamy: Vec<f32> = (0..voice.len() - 48 * (voice.len() / period + 1))
            .map(|i| voice[i + 48 * (i / period)])
            .collect();
        let jumps = seam_flux_ratio(&seamy, sr, period).unwrap();
        eprintln!("seam flux ratio: clean {clean:.3}, misaligned joins {jumps:.3}");
        assert!(clean < 1.35, "{clean}");
        assert!(jumps > clean + 0.25, "{clean} -> {jumps}");
        assert!(seam_flux_ratio(&voice[..1_000], sr, period).is_none());
    }

    #[test]
    fn cer_counts_character_edits_in_any_script() {
        assert_eq!(cer("Hello, world!", "hello world"), Some(0.0));
        assert_eq!(cer("abcd", "abxd"), Some(0.25));
        assert_eq!(cer("abcd", ""), Some(1.0));
        assert_eq!(cer("", "x"), None);
        assert_eq!(cer("こんにちは。", "こんばんは"), Some(0.4));
        assert_eq!(edit_distance(&cer_chars("kitten"), &cer_chars("sitting")), 3);
        assert_eq!(cer_chars("ÄÖ ü-1"), vec!['ä', 'ö', 'ü', '1']);
    }

    #[test]
    fn quality_metrics_flatten_to_report_names() {
        let q = QualityMetrics {
            mcd_db: Some(1.5),
            clicks: Some(0),
            speaker_cosine: Some(f64::NAN),
            ..QualityMetrics::default()
        };
        let mut m = Metrics::new();
        q.insert_into("", &mut m);
        assert_eq!(m.len(), 2, "NaN and absent fields are not reported: {m:?}");
        assert_eq!(m["mcd_db"], 1.5);
        let mut m = Metrics::new();
        q.insert_into("ceiling_", &mut m);
        assert!(m.contains_key("ceiling_clicks"));
    }

    #[test]
    fn speech_frames_skip_silence() {
        let mut x = vec![0.0f32; 16_000];
        x.extend(sine(200.0, 16_000, 1.0, 0.3));
        let m = speech_frames(&x, 50.0);
        assert_eq!(m.len(), 200);
        assert!(m[..100].iter().all(|&v| !v) && m[100..].iter().all(|&v| v));
    }
}
