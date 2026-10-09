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
//! # Scorer v2 (P3B-B, ADR 0009; `scoring.analyzer: v2`)
//! Added next to the v1 functions, which keep their definitions (v1 metrics are reported next to
//! the v2 ones):
//! - [`f0_errors_v2`]: the v1 numbers plus the fine pitch error (RMSE over voiced frames that are
//!   not gross errors) and every gross frame with its position and size, so a few octave frames no
//!   longer read as "the tracker is 50 cents off"; V/UV stays strict.
//! - [`fed_f0_errors`]: the same split for the F0 an engine fed its generator (captured on the
//!   absolute frame grid by the engine's probe), with no output analyzer in between, plus the voicing
//!   onset lag in frames ([`voicing_onsets`]).
//! - [`envelope_mcd_db`]: MCD of the F0-adaptive spectral envelope (CheapTrick-style,
//!   `dsp::envelope_mel_cepstra`) instead of the STFT, both renditions analysed with the reference's
//!   F0 per frame.
//! - [`binomial_clicks`]: seam clicks as the excess of artifact events in the join windows over
//!   the 97.5 % quantile of Binomial(events, window coverage), conditional on the events found,
//!   instead of v1's rate estimate from the few events away from the joins.
//! - [`F0Analyzer::version`]: the analyzer's version stamp (method, model sha256 and a
//!   fingerprint of its output on a fixed probe signal), written into every v2 report so F0 metrics
//!   from different analyzers are never compared.
//! - [`QUALITY_V2_SCALARS`] / [`QUALITY_V2_HIGHER_IS_BETTER`]: the v2 metric names.
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
// Scorer v2: F0 split into fine and gross error, fed F0, onsets
// ---------------------------------------------------------------------------------------------

/// One gross pitch error: its frame (index into the contours given) and its size in cents.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GrossFrame {
    pub frame: usize,
    pub cents: f64,
}

/// F0 errors of a contour against a reference, v2: the v1 numbers ([`F0Errors`], RMSE over every
/// frame voiced in both) plus the fine pitch error over the voiced frames that are not gross errors
/// and every gross frame. `RMSE² = (1 − g)·FPE² + g·mean(gross²)` with `g` the gross share.
#[derive(Clone, Debug, Default, Serialize)]
pub struct F0ErrorsV2 {
    pub v1: F0Errors,
    /// RMSE (cents) over voiced-in-both frames within 20 % of the reference; `None` when every such
    /// frame is a gross error.
    pub fine_rmse_cents: Option<f64>,
    pub gross_frames: usize,
    pub gross: Vec<GrossFrame>,
}

/// [`f0_errors`] split into fine and gross error (see [`F0ErrorsV2`]); frames where `mask` is
/// `false` (or beyond it) are skipped, gross positions index the unmasked contours.
pub fn f0_errors_v2(reference: &[f32], test: &[f32], mask: Option<&[bool]>) -> Result<F0ErrorsV2, String> {
    let n = reference.len().min(test.len());
    let keep = |i: usize| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let (mut sq, mut fine_sq, mut fine, mut both, mut vuv, mut frames) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize, 0usize);
    let mut gross = Vec::new();
    for i in (0..n).filter(|&i| keep(i)) {
        frames += 1;
        let (r, t) = (reference[i], test[i]);
        let (rv, tv) = (r.is_finite() && r > 0.0, t.is_finite() && t > 0.0);
        if rv != tv {
            vuv += 1;
        }
        if rv && tv {
            let ratio = f64::from(t) / f64::from(r);
            let cents = 1200.0 * ratio.log2();
            sq += cents * cents;
            both += 1;
            if (ratio - 1.0).abs() > GROSS_PITCH_RATIO {
                gross.push(GrossFrame { frame: i, cents });
            } else {
                fine_sq += cents * cents;
                fine += 1;
            }
        }
    }
    if frames == 0 {
        return Err("F0 errors: empty contours".into());
    }
    if both == 0 {
        return Err("F0 errors: no frame is voiced in both contours".into());
    }
    Ok(F0ErrorsV2 {
        v1: F0Errors {
            rmse_cents: (sq / both as f64).sqrt(),
            gross_pitch_error: gross.len() as f64 / both as f64,
            vuv_error: vuv as f64 / frames as f64,
            voiced_frames: both,
            frames,
        },
        fine_rmse_cents: (fine > 0).then(|| (fine_sq / fine as f64).sqrt()),
        gross_frames: gross.len(),
        gross,
    })
}

/// V/UV error with a tolerance: a frame (where `mask` keeps it) counts only when no test frame
/// within `tol` frames has the reference's voicing. A diagnostic of SOLA's join jitter (a voicing
/// edge moved by a frame), never a target: the strict V/UV error is the T5 row.
pub fn vuv_error_tolerant(reference: &[f32], test: &[f32], mask: Option<&[bool]>, tol: usize) -> Option<f64> {
    let n = reference.len().min(test.len());
    let voiced = |v: f32| v.is_finite() && v > 0.0;
    let keep = |i: usize| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let (mut frames, mut errors) = (0usize, 0usize);
    for i in (0..n).filter(|&i| keep(i)) {
        frames += 1;
        let want = voiced(reference[i]);
        if !(i.saturating_sub(tol)..=(i + tol).min(n - 1)).any(|j| voiced(test[j]) == want) {
            errors += 1;
        }
    }
    (frames > 0).then(|| errors as f64 / frames as f64)
}

/// Frames voiced after at least one unvoiced frame and voiced for [`ONSET_MIN_RUN`] frames
/// (stable voicing onsets of a contour; frames where `mask` is `false` are not onsets).
pub fn voicing_onsets(contour: &[f32], mask: Option<&[bool]>) -> Vec<usize> {
    let voiced = |i: usize| contour.get(i).is_some_and(|&v| v.is_finite() && v > 0.0);
    (1..contour.len())
        .filter(|&i| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false)))
        .filter(|&i| !voiced(i - 1) && (i..i + ONSET_MIN_RUN).all(voiced))
        .collect()
}

/// Voiced frames a reference onset needs after it.
pub const ONSET_MIN_RUN: usize = 3;
/// A test onset counts for a reference onset from this many frames before it ...
pub const ONSET_EARLY_FRAMES: usize = 5;
/// ... to this many frames after it; later is a missed onset.
pub const ONSET_LATE_FRAMES: usize = 20;

/// Errors of an engine's fed F0 (no output analyzer between) against a reference contour on the
/// same absolute grid: fine pitch error, gross pitch error, strict V/UV and the voicing onset lag.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FedF0Errors {
    pub errors: F0ErrorsV2,
    /// Reference onsets in the mask, and those the test reached within the window.
    pub onsets: usize,
    pub onsets_missed: usize,
    /// Frames the test's voicing started after (positive) or before the reference's, per onset.
    pub onset_lags: Vec<i64>,
}

impl FedF0Errors {
    pub fn onset_lag_mean(&self) -> Option<f64> {
        (!self.onset_lags.is_empty()).then(|| self.onset_lags.iter().sum::<i64>() as f64 / self.onset_lags.len() as f64)
    }

    pub fn onset_lag_median(&self) -> Option<f64> {
        let mut v: Vec<f64> = self.onset_lags.iter().map(|&x| x as f64).collect();
        (!v.is_empty()).then(|| median(&mut v))
    }

    /// The median of the lags' magnitudes (frames): the gateable form of the signed lag, where
    /// early and late voicing both count against the engine and cannot cancel.
    pub fn onset_lag_abs_median(&self) -> Option<f64> {
        let mut v: Vec<f64> = self.onset_lags.iter().map(|&x| (x as f64).abs()).collect();
        (!v.is_empty()).then(|| median(&mut v))
    }
}

/// [`fed_f0_errors`] of `test` against `reference` over the frames `mask` keeps.
pub fn fed_f0_errors(reference: &[f32], test: &[f32], mask: Option<&[bool]>) -> Result<FedF0Errors, String> {
    let errors = f0_errors_v2(reference, test, mask)?;
    let n = reference.len().min(test.len());
    let voiced = |i: usize| test.get(i).is_some_and(|&v| v.is_finite() && v > 0.0);
    let onsets = voicing_onsets(&reference[..n], mask);
    let mut lags = Vec::new();
    for &o in &onsets {
        let from = o.saturating_sub(ONSET_EARLY_FRAMES);
        let to = (o + ONSET_LATE_FRAMES).min(n.saturating_sub(1));
        if let Some(j) = (from..=to).find(|&j| voiced(j)) {
            lags.push(j as i64 - o as i64);
        }
    }
    Ok(FedF0Errors {
        errors,
        onsets: onsets.len(),
        onsets_missed: onsets.len() - lags.len(),
        onset_lags: lags,
    })
}

// ---------------------------------------------------------------------------------------------
// Scorer v2: envelope MCD
// ---------------------------------------------------------------------------------------------

/// Envelope MCD (dB): the MCD of [`mcd_db`] computed on the F0-adaptive spectral envelope
/// (`dsp::envelope_mel_cepstra`, CheapTrick-style) instead of the STFT. Both renditions are analysed
/// with `f0_ref` (Hz per 10 ms frame of `reference`, 0 = unvoiced), so the distance is the
/// signals', not the voicing decisions of an analyzer on the test. Speech frames as [`mcd_db`].
pub fn envelope_mcd_db(reference: &[f32], test: &[f32], f0_ref: &[f32], sample_rate: u32) -> Result<f64, String> {
    if sample_rate != 16_000 {
        return Err(format!("envelope MCD: {sample_rate} Hz audio (expected 16 kHz)"));
    }
    let n = reference.len().min(test.len());
    let cfg = MelCepConfig::default();
    let (cr, er) = super::dsp::envelope_mel_cepstra(&reference[..n], f0_ref, &cfg);
    let (ct, _) = super::dsp::envelope_mel_cepstra(&test[..n], f0_ref, &cfg);
    mcd_of(&cr, &er, &ct, cfg.order).ok_or_else(|| "envelope MCD: the reference has no speech frames (or is shorter than a frame)".into())
}

/// Offset-corrected MCD (P3B-B): frame `i` of `reference` (the [`mcd_db`] grid at 16 kHz: samples
/// `[160 i, 160 i + 400)`) against the window of `test` read `shift[i]` samples later, where test
/// sample `j + test_origin` is the time of reference sample `j` (zeros outside `test`). Each window
/// is contiguous test audio, so the shifts move the analysis, not the signal (splicing blocks back
/// into place would add discontinuities of its own). Returns the STFT MCD and, with `f0_ref` (Hz per
/// 10 ms frame of `reference`), the envelope MCD; speech frames as [`mcd_db`].
pub fn mcd_db_shifted(reference: &[f32], test: &[f32], test_origin: i64, shift: &[i64], f0_ref: Option<&[f32]>) -> Result<(f64, Option<f64>), String> {
    let cfg = MelCepConfig::default();
    let (cr, er) = mel_cepstra(reference, 16_000, &cfg);
    if cr.is_empty() {
        return Err(format!("MCD: the clips are shorter than one {} ms frame", cfg.frame_ms));
    }
    let at = |i: usize| shift.get(i).copied().unwrap_or(0) + test_origin;
    let starts: Vec<i64> = (0..cr.len()).map(|i| 160 * i as i64 + at(i)).collect();
    let ct = super::dsp::mel_cepstra_at(test, 16_000, &starts, &cfg);
    let mcd = mcd_of(&cr, &er, &ct, cfg.order).ok_or("MCD: the reference has no speech frames")?;
    let env = match f0_ref {
        Some(f0) => {
            let (cre, ere) = super::dsp::envelope_mel_cepstra(reference, f0, &cfg);
            let centres: Vec<i64> = (0..cre.len()).map(|i| 160 * i as i64 + 200 + at(i)).collect();
            let f0s: Vec<f64> = (0..cre.len()).map(|i| f0.get((160 * i + 280) / 160).map_or(0.0, |&v| f64::from(v))).collect();
            let cte = super::dsp::envelope_mel_cepstra_at(test, &centres, &f0s, &cfg);
            mcd_of(&cre, &ere, &cte, cfg.order)
        }
        None => None,
    };
    Ok((mcd, env))
}

/// Mean MCD over the frames within [`MCD_SPEECH_RANGE_DB`] of the reference's loudest frame.
fn mcd_of(cr: &[Vec<f64>], er: &[f64], ct: &[Vec<f64>], order: usize) -> Option<f64> {
    let per = mcd_frames(cr, er, ct, order);
    let kept: Vec<f64> = per.into_iter().flatten().collect();
    (!kept.is_empty()).then(|| kept.iter().sum::<f64>() / kept.len() as f64)
}

/// Per-frame MCD (`None` for frames outside the speech range).
pub(crate) fn mcd_frames(cr: &[Vec<f64>], er: &[f64], ct: &[Vec<f64>], order: usize) -> Vec<Option<f64>> {
    let n = cr.len().min(ct.len()).min(er.len());
    let peak = er[..n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (0..n)
        .map(|i| {
            (er[i] >= peak - MCD_SPEECH_RANGE_DB).then(|| {
                let d2: f64 = (1..=order).map(|d| (cr[i][d] - ct[i][d]) * (cr[i][d] - ct[i][d])).sum();
                MCD_K * (2.0 * d2).sqrt()
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Scorer v2: conditional binomial clicks
// ---------------------------------------------------------------------------------------------

/// `ln P(X = k)` for every `k` of `X ~ Binomial(n, p)` (log space: no underflow at any `n`).
fn binomial_log_pmf(n: u64, p: f64) -> Vec<f64> {
    let p = p.clamp(0.0, 1.0);
    let n_us = n as usize;
    if p <= 0.0 || p >= 1.0 {
        let at = if p <= 0.0 { 0 } else { n_us };
        return (0..=n_us).map(|k| if k == at { 0.0 } else { f64::NEG_INFINITY }).collect();
    }
    let (lp, lq) = (p.ln(), (-p).ln_1p());
    let mut out = Vec::with_capacity(n_us + 1);
    let mut lc = 0.0f64; // ln C(n, k)
    for k in 0..=n_us {
        if k > 0 {
            lc += ((n_us - k + 1) as f64).ln() - (k as f64).ln();
        }
        out.push(lc + k as f64 * lp + (n_us - k) as f64 * lq);
    }
    out
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`, and `P(X >= k)`.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    let lp = binomial_log_pmf(n, p);
    let m = lp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = lp.iter().map(|v| (v - m).exp()).sum();
    let below: f64 = lp[..=k as usize].iter().map(|v| (v - m).exp()).sum();
    (below / total).clamp(0.0, 1.0)
}

/// `P(X >= k)` for `X ~ Binomial(n, p)`.
pub fn binomial_at_least(k: u64, n: u64, p: f64) -> f64 {
    if k == 0 {
        1.0
    } else if k > n {
        0.0
    } else {
        (1.0 - binomial_cdf(k - 1, n, p)).max(0.0)
    }
}

/// The smallest `k` with `P(X <= k) >= q` for `X ~ Binomial(n, p)` (scipy's `binom.ppf`).
pub fn binomial_quantile(q: f64, n: u64, p: f64) -> u64 {
    let lp = binomial_log_pmf(n, p);
    let m = lp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = lp.iter().map(|v| (v - m).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    for (k, x) in w.iter().enumerate() {
        acc += x;
        // A relative tolerance so a quantile on an exact atom is not missed by rounding.
        if acc / total >= q - 1e-12 {
            return k as u64;
        }
    }
    n
}

/// Seam clicks, v2: given `events` artifact events in the scored region of which `at_join` fall in
/// join windows covering `coverage` of it, the events beyond the 97.5 % quantile of
/// Binomial(`events`, `coverage`). Conditioning on the events found removes v1's estimate of the
/// background rate from the few events away from the joins (its comment's "a third": the windows
/// cover about 70 % of a 40 ms block).
pub fn binomial_clicks(at_join: u64, events: u64, coverage: f64) -> u64 {
    if events == 0 {
        return 0;
    }
    at_join.saturating_sub(binomial_quantile(BINOMIAL_CLICKS_Q, events, coverage.clamp(0.0, 1.0)))
}

/// Quantile of the click test ([`binomial_clicks`]).
pub const BINOMIAL_CLICKS_Q: f64 = 0.975;

// ---------------------------------------------------------------------------------------------
// Scorer v2: metric names
// ---------------------------------------------------------------------------------------------

/// Metric names a v2 quality run (and a v2 rescore) adds next to the v1 [`QualityMetrics`] names
/// (`report::QUALITY_SCALARS`). Headline values are medians over the run's offsets; the names in
/// [`QUALITY_V2_SPREAD`] also report `<name>_spread` (max − min over the offsets).
pub const QUALITY_V2_SCALARS: &[&str] = &[
    "speaker_cosine_v2",
    "speaker_set_sim",
    "ceiling_speaker_set_sim",
    "whole_speaker_set_sim",
    "speaker_set_sim_delta",
    "f0_fine_rmse_cents",
    "f0_gross_frames",
    "vuv_error_tolerant",
    "fed_fpe_cents",
    "fed_gpe",
    "fed_vuv_error",
    // Signed (negative = the voicing starts early): a diagnostic with no better direction, so a
    // before/after gate reads the `_abs` form (lower is better).
    "fed_onset_lag_frames",
    "fed_onset_lag_abs_frames",
    "fed_input_fpe_cents",
    "fed_input_gpe",
    "fed_input_vuv_error",
    "fed_input_onset_lag_frames",
    "fed_input_onset_lag_abs_frames",
    "fed_input_onsets_missed",
    "ceiling_fed_input_fpe_cents",
    "ceiling_fed_input_gpe",
    "ceiling_fed_input_vuv_error",
    "whole_fed_input_fpe_cents",
    "whole_fed_input_gpe",
    "whole_fed_input_vuv_error",
    "clicks_binomial",
    "clicks_narrow",
    "env_mcd_db",
    "env_mcd_over_floor_db",
    "mcd_offset_corrected_db",
    "env_mcd_offset_corrected_db",
    "mcd_floor_db",
    "env_mcd_floor_db",
    "mcd_floor_seed_db",
    "env_mcd_floor_seed_db",
    "mcd_offline_floor_db",
    "env_mcd_offline_floor_db",
    "mcd_db_vs_b",
    "env_mcd_db_vs_b",
    "mcd_db_vs_whole",
    "env_mcd_db_vs_whole",
    "speaker_cosine_v2_vs_whole",
    "ceiling_b_dnsmos_ovrl",
    "whole_dnsmos_ovrl",
    "whole_fp32_dnsmos_ovrl",
    "mcd_db_vs_whole_fp32",
    "env_mcd_db_vs_whole_fp32",
    "mcd_db_vs_long_shape",
    "env_mcd_db_vs_long_shape",
    "long_shape_speaker_set_sim",
    "onset_rel_db",
    "onset_rel_db_long_shape",
    "onset_rel_db_b",
    "onset_rel_db_whole",
    "fidelity_offsets",
    // Ablation rows whose oracle did not reach the generator (0 = all applied).
    "ablation_rows_unapplied",
    // spreads over the offsets
    "speaker_cosine_v2_spread",
    "speaker_set_sim_spread",
    "env_mcd_db_spread",
    "mcd_db_spread",
    "f0_fine_rmse_cents_spread",
    "fed_input_fpe_cents_spread",
    "fed_input_vuv_error_spread",
    "clicks_binomial_spread",
];

/// The v2 headline metrics whose spread over the offsets is reported (`<name>_spread`).
pub const QUALITY_V2_SPREAD: &[&str] = &[
    "speaker_cosine_v2",
    "speaker_set_sim",
    "env_mcd_db",
    "mcd_db",
    "f0_fine_rmse_cents",
    "fed_input_fpe_cents",
    "fed_input_vuv_error",
    "clicks_binomial",
];

/// v2 metrics where a larger value is better (`report::higher_is_better` adds these).
pub const QUALITY_V2_HIGHER_IS_BETTER: &[&str] = &[
    "speaker_cosine_v2",
    "speaker_set_sim",
    "ceiling_speaker_set_sim",
    "whole_speaker_set_sim",
    "long_shape_speaker_set_sim",
    "speaker_set_sim_delta",
    "speaker_cosine_v2_vs_whole",
    "ceiling_b_dnsmos_ovrl",
    "whole_dnsmos_ovrl",
    "whole_fp32_dnsmos_ovrl",
    // Onset levels relative to the speech level (dB, negative): closer to 0 = less muted onset.
    "onset_rel_db",
    "onset_rel_db_long_shape",
    "onset_rel_db_b",
    "onset_rel_db_whole",
];

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
    /// The model file the method loads, when it has one (P3B-B: the version stamp).
    model_path: Option<std::path::PathBuf>,
    /// [`F0Analyzer::version`], computed once.
    version: Option<String>,
}

/// Seconds of the analyzer's fingerprint signal ([`F0Analyzer::version`]).
const ANALYZER_PROBE_SECONDS: f64 = 1.5;

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
            precision: crate::services::neural_audio::vc::engine::PrecisionMode::Auto,
            expected_call_frames: None,
            max_context_16k: None,
        };
        let mut notes = Vec::new();
        if method.trim().eq_ignore_ascii_case("auto") && wanted == "dio" {
            notes.push("F0 analysis: no RMVPE model installed (vc/pitch-rmvpe); using DIO".to_string());
        }
        match create_pitch_extractor(&req) {
            Ok(bank) => {
                notes.extend(bank.notes);
                // The model file the bank loads (explicit, else the store / base-folder rule).
                let model_path = model
                    .filter(|p| p.is_file())
                    .map(Path::to_path_buf)
                    .or_else(|| resolve_pitch_model(bank.method, app_data.as_deref()));
                F0Analyzer {
                    extractor: bank.extractor,
                    method: bank.method,
                    notes,
                    model_path,
                    version: None,
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
            model_path: None,
            version: None,
        }
    }

    /// F0 (Hz, 0 = unvoiced) per 10 ms frame of a 16 kHz clip.
    pub fn contour(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>, String> {
        self.extractor.f0(audio_16k)
    }

    /// The analyzer's version stamp (P3B-B): `<method>-<16 hex>`, the sha256 of the method, its
    /// model file's sha256 and the exact output (f32 bits) on a fixed 1.5 s probe signal. It changes
    /// whenever the model, the mel front end, the decoder or the post-filter change what the
    /// analyzer reports (the constants live in the pitch bank, e.g. `vc/rmvpe.rs`), and only then;
    /// F0 metrics of reports with different stamps are not comparable. Computed once.
    pub fn version(&mut self) -> String {
        if let Some(v) = &self.version {
            return v.clone();
        }
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.method.as_bytes());
        h.update([0u8]);
        match self.model_path.as_deref().map(super::scorers::file_sha256) {
            Some(Ok(sha)) => h.update(sha.as_bytes()),
            Some(Err(e)) => h.update(format!("unreadable model: {e}").as_bytes()),
            None => h.update(b"no model"),
        }
        h.update([0u8]);
        let probe = super::signals::synth(super::spec::SignalKind::Voice, 16_000, ANALYZER_PROBE_SECONDS, 0.5, 0xF0F0);
        match self.contour(&probe) {
            Ok(f0) => f0.iter().for_each(|v| h.update(v.to_bits().to_le_bytes())),
            Err(e) => h.update(format!("probe failed: {e}").as_bytes()),
        }
        let digest = hex::encode(h.finalize());
        let v = format!("{}-{}", self.method, &digest[..16]);
        self.version = Some(v.clone());
        v
    }

    /// The model file the analyzer loads, if any.
    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
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

    /// The v2 split on known contours: the v1 numbers are unchanged, the fine error excludes the
    /// gross frames, and `RMSE² = (1 − g)·FPE² + g·mean(gross²)`.
    #[test]
    fn f0_errors_v2_split_fine_and_gross() {
        let r = vec![200.0f32; 100];
        let sharp = 200.0 * 2f32.powf(10.0 / 1200.0);
        let mut t = vec![sharp; 100];
        t[3] = 400.0;
        t[40] = 100.0;
        for f in t.iter_mut().skip(97) {
            *f = 0.0;
        }
        let v1 = f0_errors(&r, &t).unwrap();
        let v2 = f0_errors_v2(&r, &t, None).unwrap();
        assert_eq!((v2.v1.rmse_cents, v2.v1.gross_pitch_error, v2.v1.vuv_error), (v1.rmse_cents, v1.gross_pitch_error, v1.vuv_error));
        assert_eq!(v2.gross_frames, 2);
        assert_eq!(v2.gross.iter().map(|g| g.frame).collect::<Vec<_>>(), vec![3, 40]);
        assert!((v2.gross[0].cents - 1200.0).abs() < 1e-3 && (v2.gross[1].cents + 1200.0).abs() < 1e-3);
        assert!((v2.fine_rmse_cents.unwrap() - 10.0).abs() < 1e-3, "{v2:?}");
        let g = 2.0 / 97.0;
        let fine = v2.fine_rmse_cents.unwrap();
        let recomposed = ((1.0 - g) * fine * fine + g * 0.5 * (v2.gross[0].cents.powi(2) + v2.gross[1].cents.powi(2))).sqrt();
        assert!((v2.v1.rmse_cents - recomposed).abs() < 1e-9, "{} vs {recomposed}", v2.v1.rmse_cents);
        // A mask drops frames; positions stay those of the full contours.
        let mask: Vec<bool> = (0..100).map(|i| i != 3).collect();
        let m = f0_errors_v2(&r, &t, Some(&mask)).unwrap();
        assert_eq!(m.gross.iter().map(|g| g.frame).collect::<Vec<_>>(), vec![40]);
        assert_eq!(m.v1.frames, 99);
        // Every voiced-in-both frame gross: no fine error.
        assert_eq!(f0_errors_v2(&[100.0, 100.0], &[300.0, 300.0], None).unwrap().fine_rmse_cents, None);
        assert!(f0_errors_v2(&[0.0; 3], &[0.0; 3], None).is_err());
    }

    /// The tolerant V/UV error forgives a voicing edge moved by a frame, not a wrong decision.
    #[test]
    fn tolerant_vuv_forgives_edges_only() {
        let r = [0.0f32, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0];
        let late = [0.0f32, 0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0];
        assert_eq!(vuv_error_tolerant(&r, &late, None, 1), Some(0.0));
        assert_eq!(f0_errors(&r, &late).unwrap().vuv_error, 0.25);
        let wrong = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        assert_eq!(vuv_error_tolerant(&r, &wrong, None, 1), Some(3.0 / 8.0));
        assert_eq!(vuv_error_tolerant(&r, &late, Some(&[false; 8]), 1), None);
    }

    /// Onsets of the reference and the test's lag to them.
    #[test]
    fn fed_f0_errors_report_onset_lags() {
        // Reference: voiced 10..30 and 50..80; test: 2 frames late at the first, 1 early at the
        // second; nothing voiced near a third onset at 90.
        let mut r = vec![0.0f32; 100];
        let mut t = vec![0.0f32; 100];
        for i in (10..30).chain(50..80).chain(90..96) {
            r[i] = 150.0;
        }
        for i in (12..30).chain(49..80) {
            t[i] = 151.0;
        }
        assert_eq!(voicing_onsets(&r, None), vec![10, 50, 90]);
        let e = fed_f0_errors(&r, &t, None).unwrap();
        assert_eq!((e.onsets, e.onsets_missed), (3, 1));
        assert_eq!(e.onset_lags, vec![2, -1]);
        assert_eq!(e.onset_lag_median(), Some(0.5));
        assert_eq!(e.onset_lag_mean(), Some(0.5));
        assert_eq!(e.errors.gross_frames, 0);
        assert!((e.errors.fine_rmse_cents.unwrap() - 1200.0 * (151.0f64 / 150.0).log2()).abs() < 1e-3);
        // Masked onsets do not count.
        let mask: Vec<bool> = (0..100).map(|i| i < 40).collect();
        assert_eq!(voicing_onsets(&r, Some(&mask)), vec![10]);
    }

    /// The absolute lag cannot cancel early against late voicing; the signed one can.
    #[test]
    fn the_onset_lag_has_a_gateable_absolute_form() {
        let e = FedF0Errors {
            onset_lags: vec![-3, 3, -1, 1, 2],
            ..FedF0Errors::default()
        };
        assert_eq!(e.onset_lag_median(), Some(1.0));
        assert_eq!(e.onset_lag_abs_median(), Some(2.0));
        assert_eq!(FedF0Errors::default().onset_lag_abs_median(), None);
    }

    /// Direction of the v2 metrics the perf gate compares: a level closer to 0 dB is a less muted
    /// onset (higher is better); every direction-listed name is a metric a v2 run publishes; the
    /// signed lags have an `_abs` form, and no signed name is in the higher-is-better list.
    #[test]
    fn v2_metric_directions_cover_the_onsets_and_the_signed_lags() {
        for name in ["onset_rel_db", "onset_rel_db_long_shape", "onset_rel_db_b", "onset_rel_db_whole"] {
            assert!(QUALITY_V2_HIGHER_IS_BETTER.contains(&name), "{name}");
            assert!(QUALITY_V2_SCALARS.contains(&name), "{name}");
        }
        for name in QUALITY_V2_HIGHER_IS_BETTER {
            assert!(QUALITY_V2_SCALARS.contains(name), "{name} is not a published v2 metric");
        }
        for signed in ["fed_onset_lag_frames", "fed_input_onset_lag_frames"] {
            assert!(!QUALITY_V2_HIGHER_IS_BETTER.contains(&signed));
            let abs = signed.replace("_lag_", "_lag_abs_");
            assert!(QUALITY_V2_SCALARS.contains(&abs.as_str()), "{abs}");
            assert!(!QUALITY_V2_HIGHER_IS_BETTER.contains(&abs.as_str()), "the absolute lag is lower-is-better");
        }
        assert!(QUALITY_V2_SCALARS.contains(&"ablation_rows_unapplied"));
    }

    /// The conditional binomial against scipy (`binom.ppf`, `binom.cdf`, `binom.sf`), and the saved
    /// P3 cases: (30 events, 26 in the windows) and (36, 28) read 0, the lead-1 lookahead runs
    /// (32, 29) and (42, 37) read 2.
    #[test]
    fn binomial_clicks_match_scipy_and_the_saved_cases() {
        for (n, p, q975) in [(30u64, 0.7, 26u64), (36, 0.7, 30), (32, 0.7, 27), (42, 0.7, 35), (20, 0.15, 6), (40, 0.15, 11), (1_000, 0.7, 728), (5, 0.0, 0), (5, 1.0, 5)] {
            assert_eq!(binomial_quantile(0.975, n, p), q975, "ppf(0.975, {n}, {p})");
        }
        assert!((binomial_cdf(20, 30, 0.7) - 0.411_191_314_759_277_84).abs() < 1e-9);
        assert!((binomial_cdf(20, 42, 0.7) - 0.001_997_494_004_850_438_5).abs() < 1e-12);
        assert!((binomial_at_least(26, 30, 0.7) - 0.030_154_943_102_089_3).abs() < 1e-9);
        assert_eq!(binomial_clicks(26, 30, 0.7), 0);
        assert_eq!(binomial_clicks(28, 36, 0.7), 0);
        assert_eq!(binomial_clicks(29, 32, 0.7), 2);
        assert_eq!(binomial_clicks(37, 42, 0.7), 2);
        assert_eq!(binomial_clicks(0, 0, 0.7), 0);
        // Every event at a join with a tiny coverage: all but the chance quantile are clicks.
        assert_eq!(binomial_clicks(40, 40, 0.15), 40 - 11);
    }

    /// The envelope MCD against the NumPy replica (`scripts/bench/score_v2/common.py`,
    /// `envelope_mcd_db`; values printed by it for these signals) within 0.01 dB (here to 1e-6),
    /// and the STFT MCD against its replica.
    #[test]
    fn envelope_mcd_matches_the_numpy_replica() {
        let n: Vec<f64> = (0..16_000).map(f64::from).collect();
        let ph = |i: f64| 2.0 * PI * 150.0 * i / 16_000.0 + 0.8 * (2.0 * PI * 2.5 * i / 16_000.0).sin();
        let reference: Vec<f32> = n
            .iter()
            .map(|&i| (0.3 * ph(i).sin() + 0.12 * (2.0 * PI * 450.0 * i / 16_000.0 + 0.3).sin() + 0.05 * (2.0 * PI * 2_100.0 * i / 16_000.0).sin()) as f32)
            .collect();
        let test: Vec<f32> = n
            .iter()
            .map(|&i| (0.3 * ph(i).sin() + 0.08 * (2.0 * PI * 450.0 * i / 16_000.0 + 0.3).sin() + 0.07 * (2.0 * PI * 2_300.0 * i / 16_000.0).sin()) as f32)
            .collect();
        let f0: Vec<f32> = (0..101).map(|m| if m < 60 { 150.0 } else { 0.0 }).collect();
        let env = envelope_mcd_db(&reference, &test, &f0, 16_000).unwrap();
        assert!((env - 4.747_933_147_454_606).abs() < 1e-6, "envelope MCD {env} vs the replica's 4.7479331");
        let stft = mcd_db(&reference, &test, 16_000).unwrap();
        assert!((stft - 4.647_140_913_431_759).abs() < 1e-6, "STFT MCD {stft} vs the replica's 4.6471409");
        assert_eq!(envelope_mcd_db(&reference, &reference, &f0, 16_000).unwrap(), 0.0);
        // A gain change moves only c0.
        let loud: Vec<f32> = reference.iter().map(|x| x * 1.5).collect();
        assert!(envelope_mcd_db(&reference, &loud, &f0, 16_000).unwrap() < 1e-3);
        assert!(envelope_mcd_db(&reference, &test, &f0, 48_000).is_err());
        // The envelope itself (the replica's `cheaptrick_power`).
        let mut a = super::super::dsp::EnvelopeAnalyzer::default();
        let mut p = Vec::new();
        a.power(&reference, 4_000, 150.0, &mut p);
        for (bin, want) in [(0usize, 0.014_534_816_071_221_12), (10, 7.573_810_729_255_779), (100, 2.053_005_417_675_389_2e-9)] {
            // Bins 100 dB down carry the two FFTs' rounding (pocketfft vs realfft) through log / exp.
            assert!((p[bin] / want - 1.0).abs() < 1e-4, "voiced bin {bin}: {} vs {want}", p[bin]);
        }
        a.power(&reference, 12_000, 0.0, &mut p);
        for (bin, want) in [(3usize, 0.004_704_767_178_572_399), (200, 5.659_350_789_653_181e-11)] {
            assert!((p[bin] / want - 1.0).abs() < 1e-4, "unvoiced bin {bin}: {} vs {want}", p[bin]);
        }
    }

    /// The offset-corrected MCD reads each frame where the shift says: with no shift it is
    /// [`mcd_db`] / [`envelope_mcd_db`] bit for bit, and a delayed copy read at its delay is at 0.
    #[test]
    fn shifted_mcd_reads_the_test_where_its_content_is() {
        let x: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = f64::from(i) / 16_000.0;
                (0.3 * (2.0 * PI * 140.0 * t + 0.5 * (2.0 * PI * 3.0 * t).sin()).sin() + 0.1 * (2.0 * PI * 900.0 * t).sin()) as f32
            })
            .collect();
        let y: Vec<f32> = x.iter().enumerate().map(|(i, v)| v * (1.0 + 0.3 * (i as f32 * 0.003).sin())).collect();
        let f0 = vec![140.0f32; 101];
        let (m, e) = mcd_db_shifted(&x, &y, 0, &[], Some(&f0)).unwrap();
        assert_eq!(m.to_bits(), mcd_db(&x, &y, 16_000).unwrap().to_bits());
        assert_eq!(e.unwrap().to_bits(), envelope_mcd_db(&x, &y, &f0, 16_000).unwrap().to_bits());
        // A copy 37 samples late: read 37 samples later it is the reference.
        let mut late = vec![0.0f32; 37];
        late.extend_from_slice(&x);
        let shift = vec![37i64; 200];
        let (m0, _) = mcd_db_shifted(&x, &late, 0, &[], None).unwrap();
        let (m1, e1) = mcd_db_shifted(&x, &late, 0, &shift, Some(&f0)).unwrap();
        assert!(m0 > 0.0 && m1 == 0.0 && e1 == Some(0.0), "{m0} {m1} {e1:?}");
        // The same through the origin.
        assert_eq!(mcd_db_shifted(&x, &late, 37, &[], None).unwrap().0, 0.0);
    }

    /// The analyzer's version stamp is stable for one analyzer and names its method.
    #[test]
    fn analyzer_version_is_stable() {
        let mut a = F0Analyzer::dio(F0Range::Speech, Vec::new());
        let v = a.version();
        assert!(v.starts_with("dio-") && v.len() == "dio-".len() + 16, "{v}");
        assert_eq!(a.version(), v);
        assert_eq!(F0Analyzer::dio(F0Range::Speech, Vec::new()).version(), v, "the same analyzer, the same stamp");
        let mut sing = F0Analyzer::dio(F0Range::Singing, Vec::new());
        assert!(sing.version().starts_with("dio-"));
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
