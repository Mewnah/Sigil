//! End-to-end latency meter: an irregular train of chirp bursts is played to an output device and
//! recorded from an input device (a virtual cable through Sigil's voice changer, through another
//! app, or a direct cable for calibration), and every burst is located in the recording.
//!
//! # Precision
//! - The locator is a matched filter (FFT cross-correlation against the emitted chirp) with
//!   parabolic peak interpolation: sample-accurate or better (tests recover a 1234-sample and a
//!   fractional delay to within 0.1 sample). For paths that change the timbre (a voice changer),
//!   the `onset` detector finds the envelope rise instead (about 1-2 ms).
//! - Both streams are put on one time base from the drivers' own stream timestamps (WASAPI: the
//!   QPC position of each buffer), fitted with a line per stream so callback jitter averages out;
//!   the fit residual is reported. What remains is the drivers' constant timestamp bias (WASAPI's
//!   playback instant is an estimate), which `calibration_ms` removes: measure a direct cable once
//!   with the same devices and pass its median; results are then within +-1 ms.
//!
//! The same cross-correlation core ([`estimate_delay`]) measures the stream harness's algorithmic
//! latency offline (see `engine::probe_algorithmic_latency`).
//!
//! # Play and record (P3B-C; plan T6 live tier)
//! [`run_play_record`] plays one clip (a corpus clip) into the system under test and records its
//! response on the same two devices and the same time base as the burst meter: the response WAV
//! starts at the instant the clip's first sample played (minus the calibrated bare loop), so the
//! system's delay is inside it. Every system is captured the same way, Sigil's live voice changer
//! included (the compare mode's corpus `captures`). The run reports:
//! - the delay of the response's energy envelope against the clip's ([`envelope_delay`], about
//!   1 ms on 1 ms hops; the burst meter stays the latency reference, this is its cross-check);
//! - driver glitches: steps of either stream's time base between consecutive callbacks
//!   ([`timebase_glitches`]: audio the driver dropped or inserted), lost samples and device errors
//!   (errors end the run);
//! - dropouts inside speech ([`speech_dropouts`]): digital silence or a 30 dB drop of the response
//!   inside the clip's loud speech, in runs of at least 20 ms.
//!
//! The devices sit behind [`DuplexBackend`] ([`CpalDuplex`] on real devices; the tests drive a
//! loopback stub with a known delay, drift, glitch and dropout).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};

use super::metrics::{insert_summary, Metrics};
use super::signals::{db_to_amp, BurstTrain, SplitMix64};
use super::spec::{Detector, LoopbackRun};

// ---------------------------------------------------------------------------------------------
// Cross-correlation core
// ---------------------------------------------------------------------------------------------

/// Where `reference` sits in `recorded`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DelayEstimate {
    /// `recorded[n + lag] ≈ g * reference[n]`, in samples (sub-sample interpolated).
    pub lag: f64,
    /// Normalized correlation at the peak (-1..1; negative = inverted polarity).
    pub correlation: f64,
    /// Peak magnitude over the median magnitude of all searched lags (detection strength).
    pub snr: f64,
}

/// `r[k] = sum_n recorded[n + k] * reference[n]` for `k` in `0..=max_lag` (FFT, zero-padded, no
/// circular wrap). Shorter than `max_lag + 1` when `recorded` is.
pub fn xcorr_lags(reference: &[f32], recorded: &[f32], max_lag: usize) -> Vec<f64> {
    if reference.is_empty() || recorded.is_empty() {
        return Vec::new();
    }
    let lags = max_lag.min(recorded.len() - 1) + 1;
    // Only recorded[..lags + reference.len()] can contribute to the lags searched.
    let rec = &recorded[..recorded.len().min(lags + reference.len())];
    let n = (rec.len() + reference.len()).next_power_of_two();
    let mut planner = RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut a = fwd.make_input_vec();
    let mut b = fwd.make_input_vec();
    for (d, &s) in a.iter_mut().zip(rec) {
        *d = f64::from(s);
    }
    for (d, &s) in b.iter_mut().zip(reference) {
        *d = f64::from(s);
    }
    let mut fa = fwd.make_output_vec();
    let mut fb = fwd.make_output_vec();
    // The buffers have exactly the planned lengths, so the transforms cannot fail.
    fwd.process(&mut a, &mut fa).expect("fft sizes");
    fwd.process(&mut b, &mut fb).expect("fft sizes");
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x *= y.conj();
    }
    // DC and Nyquist of a real signal's spectrum are real; clear rounding residue for the c2r.
    if let Some(first) = fa.first_mut() {
        first.im = 0.0;
    }
    if let Some(last) = fa.last_mut() {
        last.im = 0.0;
    }
    let mut r = inv.make_output_vec();
    inv.process(&mut fa, &mut r).expect("fft sizes");
    let scale = 1.0 / n as f64;
    r.truncate(lags);
    r.iter_mut().for_each(|v| *v *= scale);
    r
}

/// Sub-sample offset of a peak from three neighbours (parabolic fit), in `-0.5..=0.5`.
fn parabolic(ym: f64, y0: f64, yp: f64) -> f64 {
    let den = ym - 2.0 * y0 + yp;
    if den.abs() < 1e-30 {
        0.0
    } else {
        (0.5 * (ym - yp) / den).clamp(-0.5, 0.5)
    }
}

fn median_abs(r: &[f64]) -> f64 {
    let mut m: Vec<f64> = r.iter().map(|v| v.abs()).collect();
    if m.is_empty() {
        return 0.0;
    }
    let mid = m.len() / 2;
    m.select_nth_unstable_by(mid, f64::total_cmp);
    m[mid]
}

/// Locate `reference` in `recorded` at a lag in `0..=max_lag`. `None` when either is silent or
/// `recorded` is shorter than `reference`.
pub fn estimate_delay(reference: &[f32], recorded: &[f32], max_lag: usize) -> Option<DelayEstimate> {
    if recorded.len() < reference.len() || reference.is_empty() {
        return None;
    }
    let ref_energy: f64 = reference.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    if ref_energy <= 1e-20 {
        return None;
    }
    let r = xcorr_lags(reference, recorded, max_lag.min(recorded.len() - reference.len()));
    let (k, peak) = r
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(k, &v)| (k, v))?;
    if peak == 0.0 {
        return None;
    }
    let sign = peak.signum();
    let frac = if k > 0 && k + 1 < r.len() {
        parabolic(sign * r[k - 1], sign * r[k], sign * r[k + 1])
    } else {
        0.0
    };
    let seg = &recorded[k..(k + reference.len()).min(recorded.len())];
    let rec_energy: f64 = seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let correlation = if rec_energy > 0.0 {
        peak / (ref_energy * rec_energy).sqrt()
    } else {
        0.0
    };
    let med = median_abs(&r);
    Some(DelayEstimate {
        lag: k as f64 + frac,
        correlation,
        snr: if med > 0.0 { peak.abs() / med } else { f64::INFINITY },
    })
}

/// Envelope onset: the first crossing of `floor + 0.2 (peak - floor)` of a 1 ms moving RMS, with
/// linear interpolation between samples. `None` for a flat window.
fn onset_index(x: &[f32], rate: u32) -> Option<(f64, f64)> {
    let w = ((f64::from(rate) / 1000.0).round() as usize).max(1);
    if x.len() <= w {
        return None;
    }
    // Moving RMS aligned to the window's end.
    let mut env = Vec::with_capacity(x.len());
    let mut acc = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        acc += f64::from(v) * f64::from(v);
        if i >= w {
            acc -= f64::from(x[i - w]) * f64::from(x[i - w]);
        }
        env.push((acc.max(0.0) / w as f64).sqrt());
    }
    let mut sorted = env.clone();
    sorted.sort_by(f64::total_cmp);
    let floor = sorted[sorted.len() / 5];
    let peak = sorted[sorted.len() - 1];
    if peak <= floor * 1.5 || peak <= 1e-9 {
        return None;
    }
    let thr = floor + 0.2 * (peak - floor);
    let i = env.iter().position(|&e| e >= thr)?;
    let pos = if i == 0 {
        0.0
    } else {
        let (a, b) = (env[i - 1], env[i]);
        (i - 1) as f64 + (thr - a) / (b - a).max(1e-30)
    };
    Some((pos, peak / floor.max(1e-12)))
}

// ---------------------------------------------------------------------------------------------
// Time base
// ---------------------------------------------------------------------------------------------

/// `t = a + b * frame` (seconds on the shared stream clock).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Line {
    pub a: f64,
    pub b: f64,
}

impl Line {
    pub fn at(&self, frame: f64) -> f64 {
        self.a + self.b * frame
    }

    pub fn frame_at(&self, t: f64) -> f64 {
        (t - self.a) / self.b
    }

    /// Least-squares line through `(frame, t)` points, refitted once without outliers (beyond
    /// 3 sigma and 0.5 ms); returns the line and the RMS residual (s) of the kept points. The slope
    /// falls back to `1 / nominal_rate` with fewer than two distinct frames.
    pub fn fit(points: &[(f64, f64)], nominal_rate: u32) -> Option<(Line, f64)> {
        fn ls(p: &[(f64, f64)], nominal_b: f64) -> Option<Line> {
            if p.is_empty() {
                return None;
            }
            let n = p.len() as f64;
            let mx = p.iter().map(|q| q.0).sum::<f64>() / n;
            let my = p.iter().map(|q| q.1).sum::<f64>() / n;
            let sxx: f64 = p.iter().map(|q| (q.0 - mx) * (q.0 - mx)).sum();
            let sxy: f64 = p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
            let b = if sxx > 0.0 { sxy / sxx } else { nominal_b };
            Some(Line { a: my - b * mx, b })
        }
        let nominal_b = 1.0 / f64::from(nominal_rate.max(1));
        let first = ls(points, nominal_b)?;
        let res = |l: &Line, q: &(f64, f64)| q.1 - l.at(q.0);
        let sd = (points.iter().map(|q| res(&first, q).powi(2)).sum::<f64>() / points.len() as f64).sqrt();
        let cut = (3.0 * sd).max(0.000_5);
        let kept: Vec<(f64, f64)> = points.iter().copied().filter(|q| res(&first, q).abs() <= cut).collect();
        let line = if kept.len() >= 2 { ls(&kept, nominal_b)? } else { first };
        let pts: &[(f64, f64)] = if kept.len() >= 2 { &kept } else { points };
        let rms = (pts.iter().map(|q| res(&line, q).powi(2)).sum::<f64>() / pts.len() as f64).sqrt();
        Some((line, rms))
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis (device independent)
// ---------------------------------------------------------------------------------------------

/// One burst's measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BurstResult {
    pub index: usize,
    /// Latency (ms) after calibration; `None` = missed.
    pub latency_ms: Option<f64>,
    pub snr: f64,
}

/// Everything [`analyze`] needs besides the recording.
pub struct Analysis<'a> {
    /// The burst (matched-filter template) sampled at the input rate.
    pub template: &'a [f32],
    pub in_rate: u32,
    /// Burst onsets in output frames.
    pub onsets_out: &'a [u64],
    pub out_map: Line,
    pub in_map: Line,
    pub max_latency_s: f64,
    pub calibration_s: f64,
    pub detector: Detector,
    pub min_snr: f64,
}

/// Find every burst in `recorded` (mono, input frames from the input stream's frame 0).
pub fn analyze(recorded: &[f32], a: &Analysis<'_>) -> Vec<BurstResult> {
    // With the onset detector, compare like with like: the template's own onset, found the same
    // way in a mostly silent window (as a real search window is), so both see the same threshold.
    let template_onset = match a.detector {
        Detector::Onset => {
            let pad = a.template.len() * 3;
            let mut padded = vec![0.0f32; pad];
            padded.extend_from_slice(a.template);
            padded.resize(padded.len() + pad, 0.0);
            onset_index(&padded, a.in_rate).map(|(p, _)| p - pad as f64).unwrap_or(0.0)
        }
        Detector::Xcorr => 0.0,
    };
    let pre_s = 0.005;
    a.onsets_out
        .iter()
        .enumerate()
        .map(|(index, &onset)| {
            let t_play = a.out_map.at(onset as f64);
            let j0 = a.in_map.frame_at(t_play - pre_s).floor().max(0.0) as usize;
            let j1 = (a.in_map.frame_at(t_play + a.max_latency_s).ceil().max(0.0) as usize)
                .saturating_add(a.template.len())
                .min(recorded.len());
            let missed = BurstResult {
                index,
                latency_ms: None,
                snr: 0.0,
            };
            if j0 >= j1 || j1 - j0 < a.template.len() {
                return missed;
            }
            let window = &recorded[j0..j1];
            let found = match a.detector {
                Detector::Xcorr => {
                    estimate_delay(a.template, window, window.len() - a.template.len()).map(|e| (e.lag, e.snr))
                }
                Detector::Onset => onset_index(window, a.in_rate).map(|(p, s)| (p - template_onset, s)),
            };
            match found {
                Some((pos, snr)) if snr >= a.min_snr => {
                    let t_cap = a.in_map.at(j0 as f64 + pos);
                    BurstResult {
                        index,
                        latency_ms: Some((t_cap - t_play - a.calibration_s) * 1000.0),
                        snr,
                    }
                }
                Some((_, snr)) => BurstResult { snr, ..missed },
                None => missed,
            }
        })
        .collect()
}

/// `e2e_ms_*`, `e2e_detected`, `e2e_missed` from burst results.
pub fn burst_metrics(results: &[BurstResult], metrics: &mut Metrics) {
    let lat: Vec<f64> = results.iter().filter_map(|r| r.latency_ms).collect();
    insert_summary(metrics, "e2e_ms", &lat);
    metrics.insert("e2e_detected".into(), lat.len() as f64);
    metrics.insert("e2e_missed".into(), (results.len() - lat.len()) as f64);
}

// ---------------------------------------------------------------------------------------------
// Live measurement on real devices
// ---------------------------------------------------------------------------------------------

/// Result of a live loopback run.
#[derive(Debug, Clone, Serialize)]
pub struct LoopbackOutcome {
    pub output_device: String,
    pub input_device: String,
    pub output_rate: u32,
    pub input_rate: u32,
    pub bursts: Vec<BurstResult>,
    /// RMS residual of the output / input time-base fits (ms).
    pub timebase_jitter_ms: (f64, f64),
    pub recorded_seconds: f64,
    pub metrics: Metrics,
}

/// Burst schedule in output frames: `warmup`, then one burst per jittered interval.
pub fn schedule(run: &LoopbackRun, out_rate: u32, seed: u64) -> Vec<u64> {
    let mut rng = SplitMix64::new(seed ^ 0x0e2e_100b);
    let sr = f64::from(out_rate);
    let mut t = run.warmup_ms / 1000.0;
    (0..run.bursts)
        .map(|_| {
            let f = (t * sr).round() as u64;
            t += run.interval_ms / 1000.0 * rng.range(0.8, 1.2);
            f
        })
        .collect()
}

fn burst_shape(run: &LoopbackRun) -> BurstTrain {
    BurstTrain {
        onsets_s: Vec::new(),
        burst_s: run.burst_ms / 1000.0,
        f_lo: 200.0,
        f_hi: 4_000.0,
        amp: db_to_amp(run.level_dbfs),
    }
}

#[cfg(test)]
fn render_schedule(shape: &BurstTrain, onsets: &[u64], rate: u32, len: usize) -> Vec<f32> {
    let train = BurstTrain {
        onsets_s: onsets.iter().map(|&f| f as f64 / f64::from(rate)).collect(),
        ..shape.clone()
    };
    train.render(rate, len)
}

/// Which device a spec's name selects: `Ok(None)` is the system default (the literal `default`),
/// `Ok(Some(name))` a named one. The meter plays loud chirps, so an unnamed device is an error and
/// never read as "the default".
fn device_choice(wanted: &str, input: bool) -> Result<Option<&str>, String> {
    let wanted = wanted.trim();
    let kind = if input { "input" } else { "output" };
    if wanted.is_empty() {
        Err(format!("the {kind} device is not named: write its name, or \"default\" for the system default"))
    } else if wanted.eq_ignore_ascii_case("default") {
        Ok(None)
    } else {
        Ok(Some(wanted))
    }
}

fn find_device(host: &cpal::Host, choice: Option<&str>, input: bool) -> Result<cpal::Device, String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let kind = if input { "input" } else { "output" };
    let Some(wanted) = choice else {
        let d = if input { host.default_input_device() } else { host.default_output_device() };
        return d.ok_or_else(|| format!("there is no default {kind} device"));
    };
    let devices: Vec<cpal::Device> = if input {
        host.input_devices().map_err(|e| e.to_string())?.collect()
    } else {
        host.output_devices().map_err(|e| e.to_string())?.collect()
    };
    let names: Vec<String> = devices.iter().map(|d| d.name().unwrap_or_default()).collect();
    if let Some(i) = names.iter().position(|n| n == wanted) {
        return Ok(devices.into_iter().nth(i).expect("index from position"));
    }
    let lower = wanted.to_lowercase();
    let partial: Vec<usize> = (0..names.len()).filter(|&i| names[i].to_lowercase().contains(&lower)).collect();
    match partial.as_slice() {
        [i] => Ok(devices.into_iter().nth(*i).expect("index from filter")),
        [] => Err(format!("no {kind} device named '{wanted}' (have: {})", names.join(", "))),
        many => Err(format!(
            "'{wanted}' matches several {kind} devices: {}",
            many.iter().map(|&i| names[i].as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Signed seconds from `origin` to `t`.
fn secs_between(t: &cpal::StreamInstant, origin: &cpal::StreamInstant) -> f64 {
    match t.duration_since(origin) {
        Some(d) => d.as_secs_f64(),
        None => -origin.duration_since(t).map(|d| d.as_secs_f64()).unwrap_or(0.0),
    }
}

// ---------------------------------------------------------------------------------------------
// Duplex devices (the burst meter and play-and-record share them)
// ---------------------------------------------------------------------------------------------

/// What a duplex run recorded.
#[derive(Debug, Clone, Default)]
pub struct RawCapture {
    pub output_device: String,
    pub input_device: String,
    /// Mono input (the picked channel, or the channel mix), input frames from the input stream's
    /// frame 0.
    pub recorded: Vec<f32>,
    /// `(frame, seconds)` of every output / input callback on one clock (seconds from the first
    /// output timestamp): the drivers' stream timestamps (WASAPI: the QPC position of each buffer).
    pub out_ts: Vec<(f64, f64)>,
    pub in_ts: Vec<(f64, f64)>,
}

/// Two devices that play one signal and record the other side of the path under test: real
/// devices ([`CpalDuplex`]) or, in tests, a loopback stub with a known latency and known faults.
pub trait DuplexBackend {
    fn output_rate(&self) -> u32;
    fn input_rate(&self) -> u32;
    /// Play `signal` (mono at [`output_rate`](Self::output_rate), on every output channel) while
    /// recording the input; returns once the whole signal has played. `progress` gets the share
    /// played.
    fn run(&mut self, signal: Arc<Vec<f32>>, cancel: &AtomicBool, progress: &dyn Fn(f64)) -> Result<RawCapture, String>;
}

/// The two cpal devices of a measurement, opened (named, configured) but not yet streaming.
pub struct CpalDuplex {
    out_dev: cpal::Device,
    in_dev: cpal::Device,
    out_cfg: cpal::SupportedStreamConfig,
    in_cfg: cpal::SupportedStreamConfig,
    out_name: String,
    in_name: String,
    pick: Option<usize>,
}

impl CpalDuplex {
    /// Find and check both devices. Both names are checked before any device is touched (an empty
    /// name never means the default: the meter plays loud signals).
    pub fn open(output_device: &str, input_device: &str, input_channel: Option<u16>) -> Result<CpalDuplex, String> {
        use cpal::traits::DeviceTrait;
        use cpal::SampleFormat;
        let out_choice = device_choice(output_device, false)?;
        let in_choice = device_choice(input_device, true)?;
        let host = cpal::default_host();
        let out_dev = find_device(&host, out_choice, false)?;
        let in_dev = find_device(&host, in_choice, true)?;
        let out_name = out_dev.name().unwrap_or_default();
        let in_name = in_dev.name().unwrap_or_default();
        let out_cfg = out_dev
            .default_output_config()
            .map_err(|e| format!("output '{out_name}': {e}"))?;
        let in_cfg = in_dev
            .default_input_config()
            .map_err(|e| format!("input '{in_name}': {e}"))?;
        if out_cfg.sample_format() != SampleFormat::F32 {
            return Err(format!(
                "output '{out_name}' uses {:?} samples; the meter plays 32-bit float",
                out_cfg.sample_format()
            ));
        }
        let in_ch = usize::from(in_cfg.channels().max(1));
        let pick = input_channel.map(usize::from);
        if pick.is_some_and(|c| c >= in_ch) {
            return Err(format!("input_channel {} but '{in_name}' has {in_ch} channel(s)", pick.unwrap_or(0)));
        }
        Ok(CpalDuplex {
            out_dev,
            in_dev,
            out_cfg,
            in_cfg,
            out_name,
            in_name,
            pick,
        })
    }
}

impl DuplexBackend for CpalDuplex {
    fn output_rate(&self) -> u32 {
        self.out_cfg.sample_rate().0
    }

    fn input_rate(&self) -> u32 {
        self.in_cfg.sample_rate().0
    }

    fn run(&mut self, signal: Arc<Vec<f32>>, cancel: &AtomicBool, progress: &dyn Fn(f64)) -> Result<RawCapture, String> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        use cpal::SampleFormat;

        let (out_name, in_name) = (self.out_name.clone(), self.in_name.clone());
        let out_rate = self.output_rate();
        let in_rate = self.input_rate();
        let out_ch = usize::from(self.out_cfg.channels().max(1));
        let in_ch = usize::from(self.in_cfg.channels().max(1));
        let pick = self.pick;
        let total_out = signal.len();

        // Rings: timestamps (frame index, stream instant) of both streams, and the recorded audio.
        let (mut out_ts_tx, mut out_ts_rx) = rtrb::RingBuffer::<(u64, cpal::StreamInstant)>::new(1 << 16);
        let (mut in_ts_tx, mut in_ts_rx) = rtrb::RingBuffer::<(u64, cpal::StreamInstant)>::new(1 << 16);
        let ring_len = (in_rate as usize * 4).max(1 << 16);
        let (mut rec_tx, mut rec_rx) = rtrb::RingBuffer::<f32>::new(ring_len);
        let written = Arc::new(AtomicU64::new(0));
        let lost = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicBool::new(false));

        let out_stream = {
            let signal = signal.clone();
            let written = written.clone();
            let failed = failed.clone();
            let mut pos = 0usize;
            self.out_dev
                .build_output_stream(
                    &self.out_cfg.config(),
                    move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                        let _ = out_ts_tx.push((pos as u64, info.timestamp().playback));
                        for frame in data.chunks_exact_mut(out_ch) {
                            let v = signal.get(pos).copied().unwrap_or(0.0);
                            frame.fill(v);
                            pos += 1;
                        }
                        written.store(pos as u64, Ordering::Relaxed);
                    },
                    move |_| failed.store(true, Ordering::Relaxed),
                    None,
                )
                .map_err(|e| format!("cannot open output '{out_name}': {e}"))?
        };

        let in_stream = {
            let lost = lost.clone();
            let failed = failed.clone();
            let mut frames = 0u64;
            // One conversion per sample format (allocation-free: each sample goes straight to the
            // ring).
            macro_rules! input_stream {
                ($t:ty, $conv:expr) => {
                    self.in_dev.build_input_stream(
                        &self.in_cfg.config(),
                        move |data: &[$t], info: &cpal::InputCallbackInfo| {
                            let _ = in_ts_tx.push((frames, info.timestamp().capture));
                            for f in data.chunks_exact(in_ch) {
                                let v: f32 = match pick {
                                    Some(c) => $conv(f[c]),
                                    None => f.iter().map(|&s| $conv(s)).sum::<f32>() / in_ch as f32,
                                };
                                if rec_tx.push(v).is_err() {
                                    lost.fetch_add(1, Ordering::Relaxed);
                                }
                                frames += 1;
                            }
                        },
                        move |_| failed.store(true, Ordering::Relaxed),
                        None,
                    )
                };
            }
            use crate::services::pcm_convert as pc;
            match self.in_cfg.sample_format() {
                SampleFormat::F32 => input_stream!(f32, |s: f32| s),
                SampleFormat::I16 => input_stream!(i16, pc::i16_to_f32),
                SampleFormat::I32 => input_stream!(i32, pc::i32_to_f32),
                SampleFormat::U16 => input_stream!(u16, pc::u16_to_f32),
                SampleFormat::U8 => input_stream!(u8, pc::u8_to_f32),
                SampleFormat::I8 => input_stream!(i8, pc::i8_to_f32),
                other => return Err(format!("input '{in_name}' uses unsupported {other:?} samples")),
            }
            .map_err(|e| format!("cannot open input '{in_name}': {e}"))?
        };

        in_stream.play().map_err(|e| format!("cannot start input '{in_name}': {e}"))?;
        out_stream.play().map_err(|e| format!("cannot start output '{out_name}': {e}"))?;

        let mut recorded: Vec<f32> = Vec::with_capacity(((total_out as f64 / f64::from(out_rate) + 2.0) * f64::from(in_rate)) as usize);
        let mut out_ts = Vec::new();
        let mut in_ts = Vec::new();
        let started = Instant::now();
        let expected = Duration::from_secs_f64(total_out as f64 / f64::from(out_rate));
        let deadline = expected + Duration::from_secs(5);
        let cancelled = loop {
            std::thread::sleep(Duration::from_millis(20));
            while let Ok(v) = rec_rx.pop() {
                recorded.push(v);
            }
            while let Ok(p) = out_ts_rx.pop() {
                out_ts.push(p);
            }
            while let Ok(p) = in_ts_rx.pop() {
                in_ts.push(p);
            }
            let done = written.load(Ordering::Relaxed) as usize >= total_out;
            progress((written.load(Ordering::Relaxed) as f64 / total_out.max(1) as f64).min(1.0));
            if done || cancel.load(Ordering::Relaxed) || started.elapsed() > deadline || failed.load(Ordering::Relaxed) {
                break cancel.load(Ordering::Relaxed);
            }
        };
        drop(out_stream);
        drop(in_stream);
        while let Ok(v) = rec_rx.pop() {
            recorded.push(v);
        }
        if cancelled {
            return Err("cancelled".into());
        }
        if failed.load(Ordering::Relaxed) {
            return Err("an audio device reported an error during the measurement (disconnected?)".into());
        }
        if (written.load(Ordering::Relaxed) as usize) < total_out {
            return Err(format!("the output device '{out_name}' did not play the signal in time (is it running?)"));
        }
        if lost.load(Ordering::Relaxed) > 0 {
            return Err(format!("{} recorded samples were lost (the meter fell behind)", lost.load(Ordering::Relaxed)));
        }
        // One time base: seconds from the first output timestamp.
        let origin = out_ts.first().map(|p| p.1).ok_or("the output stream reported no timestamps")?;
        Ok(RawCapture {
            output_device: out_name,
            input_device: in_name,
            recorded,
            out_ts: out_ts.iter().map(|(f, t)| (*f as f64, secs_between(t, &origin))).collect(),
            in_ts: in_ts.iter().map(|(f, t)| (*f as f64, secs_between(t, &origin))).collect(),
        })
    }
}

/// Play the burst train on the output device, record the input device, and measure every burst.
/// Plays audio: only run it on devices the user chose (a virtual cable is inaudible).
pub fn run_live(
    run: &LoopbackRun,
    seed: u64,
    cancel: &AtomicBool,
    progress: &dyn Fn(f64, &str),
) -> Result<LoopbackOutcome, String> {
    // Both names are checked before any device is touched.
    let mut dev = CpalDuplex::open(&run.output_device, &run.input_device, run.input_channel)?;
    run_bursts(&mut dev, run, seed, cancel, progress)
}

/// The burst meter on any duplex backend.
fn run_bursts(
    dev: &mut dyn DuplexBackend,
    run: &LoopbackRun,
    seed: u64,
    cancel: &AtomicBool,
    progress: &dyn Fn(f64, &str),
) -> Result<LoopbackOutcome, String> {
    let out_rate = dev.output_rate();
    let in_rate = dev.input_rate();
    let shape = burst_shape(run);
    let onsets = schedule(run, out_rate, seed);
    let tail_s = run.max_latency_ms / 1000.0 + shape.burst_s + 0.3;
    let last = *onsets.last().unwrap_or(&0);
    let total_out = last as usize + ((tail_s + shape.burst_s) * f64::from(out_rate)).ceil() as usize;
    let signal: Arc<Vec<f32>> = Arc::new({
        let train = BurstTrain {
            onsets_s: onsets.iter().map(|&f| f as f64 / f64::from(out_rate)).collect(),
            ..shape.clone()
        };
        train.render(out_rate, total_out)
    });
    let raw = dev.run(signal, cancel, &|f| progress(f, "playing and recording the burst train"))?;
    let (out_map, out_jit) = Line::fit(&raw.out_ts, out_rate).ok_or("no output timestamps")?;
    let (in_map, in_jit) = Line::fit(&raw.in_ts, in_rate).ok_or("the input stream reported no timestamps")?;

    let template = shape.template(in_rate);
    let bursts = analyze(
        &raw.recorded,
        &Analysis {
            template: &template,
            in_rate,
            onsets_out: &onsets,
            out_map,
            in_map,
            max_latency_s: run.max_latency_ms / 1000.0,
            calibration_s: run.calibration_ms / 1000.0,
            detector: run.detector,
            min_snr: run.min_snr,
        },
    );
    let mut metrics = Metrics::new();
    burst_metrics(&bursts, &mut metrics);
    metrics.insert("e2e_timebase_jitter_ms".into(), out_jit.max(in_jit) * 1000.0);
    Ok(LoopbackOutcome {
        output_device: raw.output_device,
        input_device: raw.input_device,
        output_rate: out_rate,
        input_rate: in_rate,
        bursts,
        timebase_jitter_ms: (out_jit * 1000.0, in_jit * 1000.0),
        recorded_seconds: raw.recorded.len() as f64 / f64::from(in_rate),
        metrics,
    })
}

// ---------------------------------------------------------------------------------------------
// Play and record (P3B-C; plan T6 live tier)
// ---------------------------------------------------------------------------------------------

/// The play-and-record mode: play one clip into the system under test (any voice changer:
/// Sigil's own live voice changer, w-okada, RVC WebUI, Applio, ...) and record its response, on
/// the same devices and the same time base as the burst meter, so every system's live rendition
/// is captured the same way.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayRecordSpec {
    /// Output device (the cable into the system under test); required, `default` = the system
    /// default (your speakers).
    #[serde(default)]
    pub output_device: String,
    /// Input device (the cable out of the system under test); required, `default` = the system
    /// default.
    #[serde(default)]
    pub input_device: String,
    /// Recorded input channel (0-based); `None` = the mix of all channels.
    #[serde(default)]
    pub input_channel: Option<u16>,
    /// Gain applied to the clip (dB).
    #[serde(default)]
    pub gain_db: f64,
    /// Silence played (and recorded) before the clip: the system's start-up, and the noise floor.
    #[serde(default = "default_pr_lead_ms")]
    pub lead_ms: f64,
    /// Silence played after the clip, on top of `max_latency_ms` (the system's tail).
    #[serde(default = "default_pr_tail_ms")]
    pub tail_ms: f64,
    /// The bare loop's latency with the same devices (a direct cable): subtracted, so the
    /// response's timeline is the system's own.
    #[serde(default)]
    pub calibration_ms: f64,
    /// Longest response delay searched (and recorded after the clip).
    #[serde(default = "default_pr_max_latency_ms")]
    pub max_latency_ms: f64,
    /// Shortest dropout reported inside speech (ms).
    #[serde(default = "default_pr_dropout_ms")]
    pub dropout_min_ms: f64,
    /// A speech frame of the response this far (dB) below the response's level around it is a
    /// dropout.
    #[serde(default = "default_pr_dropout_db")]
    pub dropout_db: f64,
}

fn default_pr_lead_ms() -> f64 {
    1_000.0
}
fn default_pr_tail_ms() -> f64 {
    1_000.0
}
fn default_pr_max_latency_ms() -> f64 {
    1_500.0
}
fn default_pr_dropout_ms() -> f64 {
    20.0
}
fn default_pr_dropout_db() -> f64 {
    30.0
}

impl PlayRecordSpec {
    /// Check the values (the device names are checked when the devices open).
    pub fn validate(&self) -> Result<(), String> {
        let check = |what: &str, v: f64, lo: f64, hi: f64| {
            if v.is_finite() && v >= lo && v <= hi {
                Ok(())
            } else {
                Err(format!("{what} = {v} is outside {lo}..={hi}"))
            }
        };
        check("gain_db", self.gain_db, -60.0, 24.0)?;
        check("lead_ms", self.lead_ms, 0.0, 10_000.0)?;
        check("tail_ms", self.tail_ms, 0.0, 10_000.0)?;
        check("calibration_ms", self.calibration_ms, -1_000.0, 5_000.0)?;
        check("max_latency_ms", self.max_latency_ms, 10.0, 5_000.0)?;
        check("dropout_min_ms", self.dropout_min_ms, 5.0, 1_000.0)?;
        check("dropout_db", self.dropout_db, 6.0, 120.0)
    }
}

/// A discontinuity of a stream's time base: consecutive callbacks whose timestamps disagree with
/// their frame counts (the driver dropped or inserted audio: a WASAPI glitch).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Glitch {
    /// `output` or `input`.
    pub stream: &'static str,
    /// When (s on the shared clock).
    pub at_s: f64,
    /// How far the time base stepped (ms; positive = audio missing).
    pub step_ms: f64,
}

/// A gap in the response inside the clip's speech.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Dropout {
    /// Start on the clip's timeline (s).
    pub at_s: f64,
    pub ms: f64,
    /// `zeros` (digital silence: an underrun or a gate) or `drop` (far below the response's
    /// level around it).
    pub kind: &'static str,
}

/// Result of a play-and-record run.
#[derive(Debug, Clone, Serialize)]
pub struct PlayRecordOutcome {
    pub output_device: String,
    pub input_device: String,
    pub output_rate: u32,
    pub input_rate: u32,
    /// The response at the input rate on the clip's timeline: sample 0 was captured when the
    /// clip's first sample played (plus `calibration_ms`), so the system's delay is inside it.
    #[serde(skip)]
    pub response: Vec<f32>,
    /// The system's delay: the response's energy envelope against the clip's (ms; about 1 ms
    /// accurate when the system keeps the timing of the speech); `None` when no match.
    pub latency_ms: Option<f64>,
    /// Correlation of the envelopes at that delay.
    pub latency_correlation: Option<f64>,
    /// RMS residual of the output / input time-base fits (ms).
    pub timebase_jitter_ms: (f64, f64),
    pub glitches: Vec<Glitch>,
    pub dropouts: Vec<Dropout>,
    pub metrics: Metrics,
}

/// Fine envelope alignment: where `reference`'s energy envelope sits in `response`'s (both at
/// `rate`), searching delays `0..=max_lag_s`, on 1 ms hops of a 10 ms RMS window in dB (floored
/// 60 dB under each peak), mean-removed normalized correlation, parabolic interpolation. Robust to
/// a change of timbre (a voice changer), about 1 ms accurate. `(delay_s, correlation)`.
pub fn envelope_delay(reference: &[f32], response: &[f32], rate: u32, max_lag_s: f64) -> Option<(f64, f64)> {
    let hop = (rate as usize / 1000).max(1);
    let win = 10 * hop;
    let env = |x: &[f32]| -> Vec<f64> {
        if x.len() < win {
            return Vec::new();
        }
        let n = (x.len() - win) / hop + 1;
        let mut acc: f64 = x[..win].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let mut e = Vec::with_capacity(n);
        for k in 0..n {
            if k > 0 {
                let (out, inn) = ((k - 1) * hop, (k - 1) * hop + win);
                acc -= x[out..out + hop].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
                acc += x[inn..(inn + hop).min(x.len())].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
            }
            e.push(10.0 * (acc.max(0.0) / win as f64 + 1e-12).log10());
        }
        let peak = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        e.into_iter().map(|v| v.max(peak - 60.0)).collect()
    };
    let (er, et) = (env(reference), env(response));
    if er.len() < 20 || et.len() < 20 {
        return None;
    }
    let max_k = ((max_lag_s * 1000.0).round() as usize).min(et.len().saturating_sub(1));
    let corr_at = |k: usize| -> Option<f64> {
        let n = er.len().min(et.len().saturating_sub(k));
        if n < er.len() / 2 || n < 20 {
            return None;
        }
        let a = &er[..n];
        let b = &et[k..k + n];
        let ma = a.iter().sum::<f64>() / n as f64;
        let mb = b.iter().sum::<f64>() / n as f64;
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (x - ma, y - mb);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        (aa > 0.0 && bb > 0.0).then(|| ab / (aa * bb).sqrt())
    };
    let corr: Vec<Option<f64>> = (0..=max_k).map(corr_at).collect();
    let (k, c) = corr
        .iter()
        .enumerate()
        .filter_map(|(k, c)| c.map(|c| (k, c)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    let frac = match (k.checked_sub(1).and_then(|j| corr[j]), corr.get(k + 1).copied().flatten()) {
        (Some(ym), Some(yp)) => parabolic(ym, c, yp),
        _ => 0.0,
    };
    Some(((k as f64 + frac) * hop as f64 / f64::from(rate), c))
}

/// Steps of a stream's time base: the residuals of the line fit jump between consecutive
/// callbacks by more than `tolerance_s`, or 6x the ordinary callback jitter (the robust spread,
/// 1.4826 x MAD, of consecutive residual differences, so a step does not hide itself by inflating
/// the jitter): audio the driver dropped (positive) or inserted.
pub fn timebase_glitches(ts: &[(f64, f64)], line: &Line, tolerance_s: f64, stream: &'static str) -> Vec<Glitch> {
    if ts.len() < 3 {
        return Vec::new();
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let d: Vec<f64> = ts.windows(2).map(|w| (w[1].1 - line.at(w[1].0)) - (w[0].1 - line.at(w[0].0))).collect();
    let centre = median(&mut d.clone());
    let spread = 1.4826 * median(&mut d.iter().map(|x| (x - centre).abs()).collect());
    let tol = tolerance_s.max(6.0 * spread);
    d.iter()
        .enumerate()
        .filter(|(_, &x)| (x - centre).abs() > tol)
        .map(|(i, &x)| Glitch {
            stream,
            at_s: ts[i + 1].1,
            step_ms: (x - centre) * 1000.0,
        })
        .collect()
}

/// The time-base line of the glitch-free stretch of `ts` that contains time `t` (`whole` when that
/// stretch has fewer than two callbacks): after a step the whole-run fit is off by part of it.
fn local_line(ts: &[(f64, f64)], glitches: &[Glitch], t: f64, rate: u32, whole: Line) -> Line {
    let lo = glitches.iter().map(|g| g.at_s).filter(|&g| g <= t).fold(f64::NEG_INFINITY, f64::max);
    let hi = glitches.iter().map(|g| g.at_s).filter(|&g| g > t).fold(f64::INFINITY, f64::min);
    let pts: Vec<(f64, f64)> = ts.iter().copied().filter(|p| p.1 >= lo && p.1 < hi).collect();
    if pts.len() >= 2 {
        Line::fit(&pts, rate).map_or(whole, |l| l.0)
    } else {
        whole
    }
}

/// Dropouts of `response` inside the loud speech of `clip` (both mono at `rate`, `response`
/// already shifted by the system's delay so frame `i` of both is the same instant): 10 ms frames
/// within 25 dB of the clip's loud level (its 95th-percentile frame) where the response is digital
/// silence (`zeros`) or `drop_db` under its median over the loud frames within 300 ms (`drop`),
/// merged into runs of at least `min_ms`.
pub fn speech_dropouts(clip: &[f32], response: &[f32], rate: u32, min_ms: f64, drop_db: f64) -> Vec<Dropout> {
    let hop = (rate as usize / 100).max(1);
    let n = clip.len().min(response.len()) / hop;
    if n == 0 {
        return Vec::new();
    }
    let db = |x: &[f32]| 10.0 * (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64 + 1e-20).log10();
    let ec: Vec<f64> = (0..n).map(|i| db(&clip[i * hop..(i + 1) * hop])).collect();
    let er: Vec<f64> = (0..n).map(|i| db(&response[i * hop..(i + 1) * hop])).collect();
    let zero: Vec<bool> = (0..n).map(|i| response[i * hop..(i + 1) * hop].iter().all(|&v| v == 0.0)).collect();
    let mut sorted = ec.clone();
    sorted.sort_by(f64::total_cmp);
    let loud = sorted[((n - 1) as f64 * 0.95) as usize];
    let speech: Vec<bool> = ec.iter().map(|&e| e >= loud - 25.0 && e > -80.0).collect();
    let reach = 30usize; // 300 ms of 10 ms frames
    let flags: Vec<Option<&'static str>> = (0..n)
        .map(|i| {
            if !speech[i] {
                return None;
            }
            if zero[i] {
                return Some("zeros");
            }
            let mut around: Vec<f64> = (i.saturating_sub(reach)..(i + reach + 1).min(n))
                .filter(|&j| speech[j] && !zero[j])
                .map(|j| er[j])
                .collect();
            if around.len() < 5 {
                return None;
            }
            around.sort_by(f64::total_cmp);
            let median = around[around.len() / 2];
            (er[i] < median - drop_db).then_some("drop")
        })
        .collect();
    let min_frames = (min_ms / 10.0).ceil().max(1.0) as usize;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let Some(kind) = flags[i] else {
            i += 1;
            continue;
        };
        let start = i;
        let mut zeros = 0usize;
        while i < n && flags[i].is_some() {
            if flags[i] == Some("zeros") {
                zeros += 1;
            }
            i += 1;
        }
        if i - start >= min_frames {
            out.push(Dropout {
                at_s: start as f64 * 0.01,
                ms: (i - start) as f64 * 10.0,
                kind: if zeros * 2 >= i - start { "zeros" } else { kind },
            });
        }
    }
    out
}

/// Play `clip` (mono at `clip_rate`) through `dev` and analyse the response (see
/// [`PlayRecordOutcome`]).
pub fn play_record_with(
    dev: &mut dyn DuplexBackend,
    spec: &PlayRecordSpec,
    clip: &[f32],
    clip_rate: u32,
    cancel: &AtomicBool,
    progress: &dyn Fn(f64, &str),
) -> Result<PlayRecordOutcome, String> {
    spec.validate()?;
    if clip.is_empty() {
        return Err("play and record: the clip is empty".into());
    }
    let out_rate = dev.output_rate();
    let in_rate = dev.input_rate();
    let gain = db_to_amp(spec.gain_db) as f32;
    let clip_out: Vec<f32> = super::signals::resample(clip, clip_rate, out_rate)?.into_iter().map(|v| v * gain).collect();
    let peak = clip_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 1.0 {
        return Err(format!("play and record: the clip peaks at {:.1} dBFS after gain_db; lower gain_db", 20.0 * f64::from(peak).log10()));
    }
    let ms = |v: f64, rate: u32| (v / 1000.0 * f64::from(rate)).round().max(0.0) as usize;
    let lead = ms(spec.lead_ms, out_rate);
    let after = ms(spec.max_latency_ms + spec.tail_ms, out_rate);
    let mut signal = vec![0.0f32; lead];
    signal.extend_from_slice(&clip_out);
    signal.resize(lead + clip_out.len() + after, 0.0);
    let raw = dev.run(Arc::new(signal), cancel, &|f| progress(f, "playing the clip and recording the response"))?;
    let (out_map, out_jit) = Line::fit(&raw.out_ts, out_rate).ok_or("the output stream reported no timestamps")?;
    let (in_map, in_jit) = Line::fit(&raw.in_ts, in_rate).ok_or("the input stream reported no timestamps")?;
    // Glitches: time-base steps of either stream (at least 2 ms).
    let out_glitches = timebase_glitches(&raw.out_ts, &out_map, 0.002, "output");
    let in_glitches = timebase_glitches(&raw.in_ts, &in_map, 0.002, "input");
    // The input frame captured when the clip's first sample played (plus the bare loop), on the
    // glitch-free stretches of both time bases around that instant.
    let t_out = out_map.at(lead as f64);
    let t_play = local_line(&raw.out_ts, &out_glitches, t_out, out_rate, out_map).at(lead as f64);
    let t_cap = t_play + spec.calibration_ms / 1000.0;
    let first = local_line(&raw.in_ts, &in_glitches, t_cap, in_rate, in_map).frame_at(t_cap).round();
    if first < 0.0 {
        return Err("play and record: the input started after the clip (raise lead_ms)".into());
    }
    let first = first as usize;
    let clip_in = super::signals::resample(clip, clip_rate, in_rate)?;
    let want = clip_in.len() + ms(spec.max_latency_ms + spec.tail_ms, in_rate);
    let mut response = raw.recorded.get(first..).unwrap_or(&[]).to_vec();
    response.truncate(want);
    if response.len() < clip_in.len() {
        return Err(format!(
            "play and record: {:.2} s recorded after the clip started, the clip is {:.2} s",
            response.len() as f64 / f64::from(in_rate),
            clip_in.len() as f64 / f64::from(in_rate)
        ));
    }
    let found = envelope_delay(&clip_in, &response, in_rate, spec.max_latency_ms / 1000.0);
    let mut glitches = out_glitches;
    glitches.extend(in_glitches);
    glitches.sort_by(|a, b| a.at_s.total_cmp(&b.at_s));
    let dropouts = match found {
        Some((delay_s, _)) => {
            let shift = (delay_s * f64::from(in_rate)).round() as usize;
            speech_dropouts(&clip_in, response.get(shift..).unwrap_or(&[]), in_rate, spec.dropout_min_ms, spec.dropout_db)
        }
        None => Vec::new(),
    };
    let mut metrics = Metrics::new();
    if let Some((d, c)) = found {
        metrics.insert("play_record_latency_ms".into(), d * 1000.0);
        metrics.insert("play_record_latency_correlation".into(), c);
    }
    metrics.insert("play_record_glitches".into(), glitches.len() as f64);
    metrics.insert("play_record_dropouts".into(), dropouts.len() as f64);
    metrics.insert("play_record_dropout_ms".into(), dropouts.iter().map(|d| d.ms).sum());
    metrics.insert("play_record_timebase_jitter_ms".into(), out_jit.max(in_jit) * 1000.0);
    metrics.insert("play_record_seconds".into(), response.len() as f64 / f64::from(in_rate));
    Ok(PlayRecordOutcome {
        output_device: raw.output_device,
        input_device: raw.input_device,
        output_rate: out_rate,
        input_rate: in_rate,
        response,
        latency_ms: found.map(|f| f.0 * 1000.0),
        latency_correlation: found.map(|f| f.1),
        timebase_jitter_ms: (out_jit * 1000.0, in_jit * 1000.0),
        glitches,
        dropouts,
        metrics,
    })
}

/// [`play_record_with`] on the named real devices. Plays audio: only on devices the user named.
pub fn run_play_record(
    spec: &PlayRecordSpec,
    clip: &[f32],
    clip_rate: u32,
    cancel: &AtomicBool,
    progress: &dyn Fn(f64, &str),
) -> Result<PlayRecordOutcome, String> {
    spec.validate()?;
    let mut dev = CpalDuplex::open(&spec.output_device, &spec.input_device, spec.input_channel)?;
    play_record_with(&mut dev, spec, clip, clip_rate, cancel, progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::signals::synth;
    use crate::services::bench::spec::SignalKind;

    fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| amp * rng.range(-1.0, 1.0) as f32).collect()
    }

    /// The acceptance check: a known 1234-sample delay is recovered exactly, for a chirp train
    /// and for speech-like audio, with noise and an attenuated, inverted copy.
    #[test]
    fn a_known_1234_sample_delay_is_recovered() {
        for (label, reference) in [
            ("chirps", synth(SignalKind::ChirpTrain, 48_000, 2.0, 0.5, 5)),
            ("voice", synth(SignalKind::Voice, 48_000, 2.0, 0.5, 6)),
            ("noise", synth(SignalKind::Noise, 48_000, 1.0, 0.5, 7)),
        ] {
            for gain in [0.3f32, -0.8] {
                let mut recorded = vec![0.0f32; 1234];
                recorded.extend(reference.iter().map(|&v| gain * v));
                recorded.extend(vec![0.0; 3_000]);
                let len = recorded.len();
                for (r, n) in recorded.iter_mut().zip(noise(len, 0.005, 9)) {
                    *r += n;
                }
                let e = estimate_delay(&reference, &recorded, 48_000).expect("found");
                assert!((e.lag - 1234.0).abs() < 0.05, "{label} gain {gain}: lag {}", e.lag);
                assert_eq!(e.correlation.signum(), f64::from(gain.signum()), "{label}");
                assert!(e.correlation.abs() > 0.9, "{label}: {}", e.correlation);
                // Voiced audio correlates with itself at every pitch period, so its peak stands
                // out less than a chirp's; still far above a no-match (< 8).
                let min_snr = if label == "voice" { 10.0 } else { 20.0 };
                assert!(e.snr > min_snr, "{label}: snr {}", e.snr);
            }
        }
    }

    #[test]
    fn fractional_delays_are_interpolated() {
        let mut rng = SplitMix64::new(3);
        let train = BurstTrain::irregular(1.5, 0.04, 0.3, &mut rng);
        let reference = train.render(48_000, 72_000);
        for shift in [1234.25f64, 1234.5, 777.8] {
            let delayed = BurstTrain {
                onsets_s: train.onsets_s.iter().map(|t| t + shift / 48_000.0).collect(),
                ..train.clone()
            };
            let recorded = delayed.render(48_000, 72_000 + 2_000);
            let e = estimate_delay(&reference, &recorded, 4_000).unwrap();
            assert!((e.lag - shift).abs() < 0.1, "shift {shift}: {}", e.lag);
        }
    }

    #[test]
    fn silence_and_short_inputs_are_not_detections() {
        assert!(estimate_delay(&[0.0; 100], &[0.1; 1000], 500).is_none());
        assert!(estimate_delay(&[0.1; 100], &[0.1; 50], 500).is_none());
        assert!(estimate_delay(&[], &[0.1; 50], 500).is_none());
        // Uncorrelated noise: a weak, low-SNR peak.
        let e = estimate_delay(&noise(2_000, 0.5, 1), &noise(20_000, 0.5, 2), 10_000).unwrap();
        assert!(e.snr < 8.0 && e.correlation.abs() < 0.2, "{e:?}");
    }

    #[test]
    fn xcorr_matches_the_direct_sum() {
        let r = noise(37, 1.0, 4);
        let x = noise(300, 1.0, 5);
        let fast = xcorr_lags(&r, &x, 200);
        assert_eq!(fast.len(), 201);
        for (k, &v) in fast.iter().enumerate() {
            let direct: f64 = r
                .iter()
                .enumerate()
                .filter(|(n, _)| n + k < x.len())
                .map(|(n, &a)| f64::from(a) * f64::from(x[n + k]))
                .sum();
            assert!((v - direct).abs() < 1e-9, "lag {k}: {v} vs {direct}");
        }
    }

    #[test]
    fn line_fit_averages_jitter_and_drops_outliers() {
        let rate = 48_000u32;
        let mut rng = SplitMix64::new(8);
        let mut pts: Vec<(f64, f64)> = (0..500)
            .map(|k| {
                let f = (k * 480) as f64;
                (f, 0.123 + f / 48_010.0 + rng.range(-0.000_2, 0.000_2))
            })
            .collect();
        pts[250].1 += 0.05; // one glitched callback
        let (line, rms) = Line::fit(&pts, rate).unwrap();
        assert!((line.b - 1.0 / 48_010.0).abs() < 1e-9, "{}", line.b);
        assert!((line.a - 0.123).abs() < 1e-4, "{}", line.a);
        assert!(rms < 0.000_2, "{rms}");
        let (single, _) = Line::fit(&[(100.0, 1.0)], rate).unwrap();
        assert!((single.at(148.0) - 1.001).abs() < 1e-12);
        assert!(Line::fit(&[], rate).is_none());
    }

    /// The whole device-free pipeline: bursts scheduled at 48 kHz, "recorded" at 44.1 kHz on
    /// another clock origin with a known path latency and noise, found by both detectors.
    #[test]
    fn analysis_recovers_the_path_latency_across_rates() {
        let run: LoopbackRun = serde_json::from_value(serde_json::json!({
            "name": "t", "bursts": 12, "interval_ms": 500, "max_latency_ms": 300
        }))
        .unwrap();
        let (out_rate, in_rate) = (48_000u32, 44_100u32);
        let onsets = schedule(&run, out_rate, 1);
        assert_eq!(onsets.len(), 12);
        let out_map = Line { a: 0.0, b: 1.0 / f64::from(out_rate) };
        // The input stream started 37 ms later on the shared clock and runs 20 ppm fast.
        let in_map = Line {
            a: 0.037,
            b: 1.0 / (f64::from(in_rate) * 1.000_02),
        };
        let latency = 0.023_456;
        let shape = burst_shape(&run);
        let len = (in_map.frame_at(out_map.at(*onsets.last().unwrap() as f64) + 1.0)) as usize;
        let delayed = BurstTrain {
            onsets_s: Vec::new(),
            ..shape.clone()
        };
        let mut recorded: Vec<f32> = (0..len)
            .map(|j| {
                let t = in_map.at(j as f64);
                onsets
                    .iter()
                    .map(|&f| delayed.burst_at(t - out_map.at(f as f64) - latency))
                    .sum::<f64>() as f32
            })
            .collect();
        for (r, n) in recorded.iter_mut().zip(noise(len, 0.003, 12)) {
            *r += n;
        }
        let template = shape.template(in_rate);
        for (detector, tol_ms) in [(Detector::Xcorr, 0.01), (Detector::Onset, 1.5)] {
            let results = analyze(
                &recorded,
                &Analysis {
                    template: &template,
                    in_rate,
                    onsets_out: &onsets,
                    out_map,
                    in_map,
                    max_latency_s: 0.3,
                    calibration_s: 0.003,
                    detector,
                    min_snr: 4.0,
                },
            );
            assert_eq!(results.len(), 12);
            for r in &results {
                let got = r.latency_ms.unwrap_or_else(|| panic!("{detector:?}: burst {} missed ({})", r.index, r.snr));
                assert!((got - 20.456).abs() < tol_ms, "{detector:?} burst {}: {got} ms", r.index);
            }
            let mut m = Metrics::new();
            burst_metrics(&results, &mut m);
            assert_eq!(m["e2e_detected"], 12.0);
            assert_eq!(m["e2e_missed"], 0.0);
            assert!((m["e2e_ms_p50"] - 20.456).abs() < tol_ms);
        }
        // A silent recording: every burst is missed, none invented.
        let silent = vec![0.0f32; len];
        let results = analyze(
            &silent,
            &Analysis {
                template: &template,
                in_rate,
                onsets_out: &onsets,
                out_map,
                in_map,
                max_latency_s: 0.3,
                calibration_s: 0.0,
                detector: Detector::Xcorr,
                min_snr: 4.0,
            },
        );
        assert!(results.iter().all(|r| r.latency_ms.is_none()));
    }

    /// The real device path (open both devices, play, record, stream timestamps, time-base fits)
    /// with a *silent* burst train (-200 dBFS: nothing audible reaches whoever listens to the
    /// devices). Opt-in: set `SIGIL_TEST_LOOPBACK_OUT` / `SIGIL_TEST_LOOPBACK_IN` to device names,
    /// e.g. `CABLE Input` / `CABLE Output`, then
    /// `cargo test -- --ignored live_loopback_device_path --nocapture`.
    #[test]
    #[ignore = "opens real audio devices (SIGIL_TEST_LOOPBACK_OUT / SIGIL_TEST_LOOPBACK_IN)"]
    fn live_loopback_device_path_is_silent_and_timed() {
        let (Ok(out), Ok(inp)) = (std::env::var("SIGIL_TEST_LOOPBACK_OUT"), std::env::var("SIGIL_TEST_LOOPBACK_IN")) else {
            eprintln!("SIGIL_TEST_LOOPBACK_OUT / _IN not set; skipped");
            return;
        };
        let mut run: LoopbackRun = serde_json::from_value(serde_json::json!({
            "name": "live", "output_device": out, "input_device": inp,
            "bursts": 3, "interval_ms": 500, "max_latency_ms": 300, "warmup_ms": 300
        }))
        .unwrap();
        run.level_dbfs = -200.0; // below validation on purpose: silence
        let cancel = AtomicBool::new(false);
        let o = run_live(&run, 1, &cancel, &|_, _| {}).expect("devices open and run");
        eprintln!(
            "live loopback: {} @ {} Hz -> {} @ {} Hz, {:.2} s recorded, time-base jitter out {:.3} ms / in {:.3} ms",
            o.output_device, o.output_rate, o.input_device, o.input_rate, o.recorded_seconds, o.timebase_jitter_ms.0, o.timebase_jitter_ms.1
        );
        assert_eq!(o.bursts.len(), 3);
        assert!(o.bursts.iter().all(|b| b.latency_ms.is_none()), "a silent train must not be detected");
        assert!(o.recorded_seconds > 1.3, "{}", o.recorded_seconds);
        assert!(o.timebase_jitter_ms.0 < 2.0 && o.timebase_jitter_ms.1 < 2.0, "{:?}", o.timebase_jitter_ms);
    }

    /// An empty name never falls back to the default device (the user's real speakers or
    /// headset); only the literal "default" does. A run that skipped validation fails before any
    /// device is touched. (This test must not reach cpal: its WASAPI enumerator is process-global
    /// and breaks when the test thread that created it exits while another test uses it.)
    #[test]
    fn unnamed_devices_never_mean_the_default() {
        assert!(device_choice("", false).unwrap_err().contains("output device is not named"));
        assert!(device_choice("  ", true).unwrap_err().contains("input device is not named"));
        assert_eq!(device_choice("default", false), Ok(None));
        assert_eq!(device_choice(" Default ", true), Ok(None));
        assert_eq!(device_choice(" CABLE Output ", true), Ok(Some("CABLE Output")));
        let cancel = AtomicBool::new(false);
        for (out, inp, needle) in [
            ("", "", "output device is not named"),
            ("default", " ", "input device is not named"),
            ("CABLE Input", "", "input device is not named"),
        ] {
            let run: LoopbackRun =
                serde_json::from_value(serde_json::json!({"name": "t", "output_device": out, "input_device": inp})).unwrap();
            let e = run_live(&run, 1, &cancel, &|_, _| {}).unwrap_err();
            assert!(e.contains(needle), "{out:?}/{inp:?}: {e}");
        }
    }

    #[test]
    fn schedule_spacing_is_jittered_and_rendered_where_scheduled() {
        let run: LoopbackRun = serde_json::from_value(serde_json::json!({"name": "t", "bursts": 30})).unwrap();
        let onsets = schedule(&run, 48_000, 2);
        assert_eq!(onsets[0], 24_000, "500 ms warm-up");
        let gaps: Vec<u64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| (23_040..=34_560).contains(&g)), "{gaps:?}");
        assert!(gaps.windows(2).any(|w| w[0] != w[1]), "spacing must not be periodic");
        let shape = burst_shape(&run);
        let x = render_schedule(&shape, &onsets[..2], 48_000, onsets[1] as usize + 4_000);
        let e = estimate_delay(&shape.template(48_000), &x, x.len()).unwrap();
        assert!((e.lag - onsets[0] as f64).abs() < 0.05, "{}", e.lag);
    }

    // ---- Play and record (P3B-C) ----

    /// A duplex "device pair" around a system under test that is a pure delay with gain: the
    /// output stream starts the shared clock; the input stream starts later and runs fast; the
    /// path delays the signal by `latency_s`. Optional faults: the capture loses `gap` seconds at a
    /// shared time (a driver glitch: later frames were captured that much later), and the
    /// system's output is muted over a span (a dropout).
    struct LoopbackStub {
        out_rate: u32,
        in_rate: u32,
        latency_s: f64,
        gain: f32,
        in_origin_s: f64,
        in_ppm: f64,
        period_s: f64,
        capture_gap: Option<(f64, f64)>,
        mute: Option<(f64, f64)>,
        noise: f32,
    }

    impl LoopbackStub {
        fn new(latency_s: f64) -> Self {
            Self {
                out_rate: 48_000,
                in_rate: 44_100,
                latency_s,
                gain: 0.5,
                in_origin_s: 0.037,
                in_ppm: 20.0,
                period_s: 0.01,
                capture_gap: None,
                mute: None,
                noise: 0.000_5,
            }
        }
    }

    impl DuplexBackend for LoopbackStub {
        fn output_rate(&self) -> u32 {
            self.out_rate
        }

        fn input_rate(&self) -> u32 {
            self.in_rate
        }

        fn run(&mut self, signal: Arc<Vec<f32>>, _cancel: &AtomicBool, progress: &dyn Fn(f64)) -> Result<RawCapture, String> {
            let out_sr = f64::from(self.out_rate);
            let dt = 1.0 / (f64::from(self.in_rate) * (1.0 + self.in_ppm * 1e-6));
            let played_s = signal.len() as f64 / out_sr;
            // Capture time of input frame j (the gap shifts every later frame).
            let tau = |j: usize| {
                let t = self.in_origin_s + j as f64 * dt;
                match self.capture_gap {
                    Some((at, gap)) if t >= at => t + gap,
                    _ => t,
                }
            };
            let sig = |t: f64| -> f32 {
                // The system's output at shared time t: the signal played at t - latency.
                if let Some((at, len)) = self.mute {
                    if (at..at + len).contains(&t) {
                        return 0.0;
                    }
                }
                let x = (t - self.latency_s) * out_sr;
                if x < 0.0 {
                    return 0.0;
                }
                let i = x.floor() as usize;
                let f = (x - i as f64) as f32;
                let a = signal.get(i).copied().unwrap_or(0.0);
                let b = signal.get(i + 1).copied().unwrap_or(0.0);
                self.gain * (a + f * (b - a))
            };
            let mut rng = SplitMix64::new(77);
            let mut recorded = Vec::new();
            let mut in_ts = Vec::new();
            let per_in = ((self.period_s / dt).round() as usize).max(1);
            let mut j = 0usize;
            while tau(j) < played_s + 0.05 {
                if j.is_multiple_of(per_in) {
                    in_ts.push((j as f64, tau(j)));
                }
                recorded.push(sig(tau(j)) + self.noise * rng.range(-1.0, 1.0) as f32);
                j += 1;
            }
            let per_out = (self.period_s * out_sr).round() as usize;
            let out_ts: Vec<(f64, f64)> = (0..signal.len()).step_by(per_out).map(|f| (f as f64, f as f64 / out_sr)).collect();
            progress(1.0);
            Ok(RawCapture {
                output_device: "stub out".into(),
                input_device: "stub in".into(),
                recorded,
                out_ts,
                in_ts,
            })
        }
    }

    fn pr_spec(json: serde_json::Value) -> PlayRecordSpec {
        let mut v = serde_json::json!({"output_device": "stub out", "input_device": "stub in", "lead_ms": 500, "tail_ms": 300, "max_latency_ms": 400});
        if let (Some(o), Some(extra)) = (v.as_object_mut(), json.as_object()) {
            for (k, x) in extra {
                o.insert(k.clone(), x.clone());
            }
        }
        serde_json::from_value(v).unwrap()
    }

    /// The loudest `win`-sample stretch of `x` (searched on a 10 ms grid at 16 kHz).
    fn loudest(x: &[f32], win: usize) -> usize {
        let energy = |s: usize| x[s..s + win].iter().map(|v| v * v).sum::<f32>();
        (0..x.len() - win).step_by(160).max_by(|&a, &b| energy(a).total_cmp(&energy(b))).unwrap()
    }

    /// The acceptance check of the live tier: the response comes back on the clip's timeline with
    /// the path's delay inside it (about 1 ms), across rates and a drifting input clock, with no
    /// glitch or dropout invented.
    #[test]
    fn play_record_recovers_the_delay_on_a_loopback_stub() {
        let clip = synth(SignalKind::Voice, 16_000, 4.0, 0.5, 31);
        let cancel = AtomicBool::new(false);
        for (latency_s, calibration_ms) in [(0.1234, 0.0), (0.2101, 11.8)] {
            let mut dev = LoopbackStub::new(latency_s);
            let spec = pr_spec(serde_json::json!({"calibration_ms": calibration_ms}));
            let o = play_record_with(&mut dev, &spec, &clip, 16_000, &cancel, &|_, _| {}).unwrap();
            let want = latency_s * 1000.0 - calibration_ms;
            let got = o.latency_ms.expect("delay found");
            assert!((got - want).abs() < 1.5, "latency {got} ms, want {want} ms");
            assert!(o.latency_correlation.unwrap() > 0.9, "{:?}", o.latency_correlation);
            assert!(o.glitches.is_empty(), "{:?}", o.glitches);
            assert!(o.dropouts.is_empty(), "{:?}", o.dropouts);
            assert_eq!(o.input_rate, 44_100);
            // The response is the clip, delayed: its waveform matches the clip at the delay.
            let clip_in = crate::services::bench::signals::resample(&clip, 16_000, 44_100).unwrap();
            let e = estimate_delay(&clip_in[..44_100], &o.response, 30_000).unwrap();
            assert!((e.lag / 44.1 - want).abs() < 0.2, "waveform delay {} ms", e.lag / 44.1);
            assert_eq!(o.metrics["play_record_glitches"], 0.0);
        }
    }

    /// A capture discontinuity is flagged as an input glitch of about its size, and a muted span of
    /// the system's output inside speech as a dropout where it happened; the delay before the
    /// glitch is still recovered.
    #[test]
    fn glitches_and_dropouts_in_speech_are_flagged() {
        let clip = synth(SignalKind::Voice, 16_000, 4.0, 0.5, 32);
        // The loudest 150 ms of the clip (a syllable), muted in the system's output.
        let loud = loudest(&clip, 2_400);
        let latency_s = 0.15;
        let lead_s = 0.5;
        let mut dev = LoopbackStub::new(latency_s);
        let mute_at = lead_s + loud as f64 / 16_000.0 + latency_s + 0.02;
        dev.mute = Some((mute_at, 0.08));
        dev.capture_gap = Some((lead_s + 3.2, 0.02));
        let cancel = AtomicBool::new(false);
        let o = play_record_with(&mut dev, &pr_spec(serde_json::json!({})), &clip, 16_000, &cancel, &|_, _| {}).unwrap();
        assert!((o.latency_ms.unwrap() - 150.0).abs() < 2.0, "{:?}", o.latency_ms);
        let g: Vec<&Glitch> = o.glitches.iter().filter(|g| g.stream == "input").collect();
        assert_eq!(g.len(), 1, "{:?}", o.glitches);
        // Reported at the first callback after the step: captured 20 ms late, within one period.
        assert!((g[0].step_ms - 20.0).abs() < 1.0 && (g[0].at_s - (lead_s + 3.2 + 0.02)).abs() <= 0.011, "{:?}", g[0]);
        assert!(!o.dropouts.is_empty(), "the muted syllable is a dropout");
        let d = o.dropouts[0];
        let expect_at = loud as f64 / 16_000.0 + 0.02;
        assert!((d.at_s - expect_at).abs() < 0.03 && (40.0..=90.0).contains(&d.ms), "{d:?} (expected near {expect_at} s)");
        // The capture adds the line's noise to the muted span: a drop, not digital silence.
        assert_eq!(d.kind, "drop");
        assert!(o.metrics["play_record_dropout_ms"] >= 40.0);
    }

    /// The burst meter runs unchanged on the refactored device layer (same analysis, same metrics).
    #[test]
    fn the_burst_meter_runs_on_any_duplex_backend() {
        let run: LoopbackRun = serde_json::from_value(serde_json::json!({
            "name": "t", "output_device": "stub out", "input_device": "stub in",
            "bursts": 6, "interval_ms": 500, "max_latency_ms": 300, "warmup_ms": 300
        }))
        .unwrap();
        let mut dev = LoopbackStub::new(0.0456);
        dev.noise = 0.001;
        let cancel = AtomicBool::new(false);
        let o = run_bursts(&mut dev, &run, 3, &cancel, &|_, _| {}).unwrap();
        assert_eq!(o.metrics["e2e_detected"], 6.0, "{:?}", o.bursts);
        assert!((o.metrics["e2e_ms_p50"] - 45.6).abs() < 0.1, "{}", o.metrics["e2e_ms_p50"]);
        assert!(o.timebase_jitter_ms.1 < 0.01);
    }

    #[test]
    fn envelope_delay_finds_a_timbre_changed_copy() {
        // The "converted" copy has another fine structure (noise instead of the voice) but the
        // same syllable envelope, 87.5 ms later.
        let x = synth(SignalKind::Voice, 16_000, 3.0, 0.5, 33);
        let mut rng = SplitMix64::new(5);
        let w = 160usize;
        let env: Vec<f32> = (0..x.len())
            .map(|i| {
                let s = i.saturating_sub(w / 2);
                let e = (s..(s + w).min(x.len())).map(|k| x[k] * x[k]).sum::<f32>() / w as f32;
                e.sqrt()
            })
            .collect();
        let mut y = vec![0.0f32; 1_400];
        y.extend(env.iter().map(|&e| e * rng.range(-1.7, 1.7) as f32));
        y.extend(vec![0.0; 4_000]);
        let (d, c) = envelope_delay(&x, &y, 16_000, 0.3).unwrap();
        assert!((d * 1000.0 - 87.5).abs() < 2.0, "{} ms (corr {c})", d * 1000.0);
        assert!(c > 0.8, "{c}");
        assert!(envelope_delay(&x[..100], &y, 16_000, 0.3).is_none(), "too short to align");
    }

    #[test]
    fn speech_dropouts_skip_pauses_and_find_holes() {
        let clip = synth(SignalKind::Voice, 16_000, 3.0, 0.5, 34);
        assert!(speech_dropouts(&clip, &clip, 16_000, 20.0, 30.0).is_empty(), "an intact response has no dropouts");
        // Silence in the clip's pauses is not a dropout: gate the response where the clip is quiet.
        let gated: Vec<f32> = clip
            .chunks(160)
            .flat_map(|c| {
                let rms = (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt();
                let keep = rms > 0.01;
                c.iter().map(move |&v| if keep { v } else { 0.0 })
            })
            .collect();
        assert!(speech_dropouts(&clip, &gated, 16_000, 20.0, 30.0).is_empty(), "a gate in pauses is not a dropout");
        // A 60 dB dip over 50 ms inside the loudest syllable.
        let loud = loudest(&clip, 800) / 160 * 160;
        let mut dipped = clip.clone();
        dipped[loud..loud + 800].iter_mut().for_each(|v| *v *= 0.001);
        let d = speech_dropouts(&clip, &dipped, 16_000, 20.0, 30.0);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].kind, "drop");
        assert!((d[0].at_s - loud as f64 / 16_000.0).abs() <= 0.011 && (40.0..=60.0).contains(&d[0].ms), "{d:?}");
        // Digital silence there instead (an underrun): a `zeros` dropout.
        let mut zeroed = clip.clone();
        zeroed[loud..loud + 800].iter_mut().for_each(|v| *v = 0.0);
        let d = speech_dropouts(&clip, &zeroed, 16_000, 20.0, 30.0);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].kind, "zeros");
    }

    #[test]
    fn play_record_specs_are_checked() {
        assert!(pr_spec(serde_json::json!({})).validate().is_ok());
        assert!(pr_spec(serde_json::json!({"gain_db": 40})).validate().is_err());
        assert!(pr_spec(serde_json::json!({"max_latency_ms": 1})).validate().is_err());
        assert!(serde_json::from_value::<PlayRecordSpec>(serde_json::json!({"output": "x"})).is_err(), "unknown fields are errors");
        // Unnamed devices fail before any device is touched (never the default speakers).
        let cancel = AtomicBool::new(false);
        let spec: PlayRecordSpec = serde_json::from_value(serde_json::json!({"input_device": "x"})).unwrap();
        let e = run_play_record(&spec, &[0.1; 1_000], 16_000, &cancel, &|_, _| {}).unwrap_err();
        assert!(e.contains("output device is not named"), "{e}");
        // A clip that would clip after the gain is refused.
        let mut dev = LoopbackStub::new(0.1);
        let e = play_record_with(&mut dev, &pr_spec(serde_json::json!({"gain_db": 12})), &[0.9; 16_000], 16_000, &cancel, &|_, _| {}).unwrap_err();
        assert!(e.contains("peaks at"), "{e}");
    }
}
