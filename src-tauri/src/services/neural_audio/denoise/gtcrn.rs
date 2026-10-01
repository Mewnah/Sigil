//! GTCRN (P3-E): an ultra-light speech enhancer at 16 kHz, run inside the harness on the model
//! input (lowest CPU). Model: `vc::assets::DENOISE_GTCRN` (Xiaobin-Rong/gtcrn's streaming
//! `gtcrn_simple.onnx`, MIT, 0.5 MB).
//!
//! Contract (P3-A `export_denoise.py`, checked at load): one STFT frame `mix` `[1, 257, 1, 2]`
//! (512-point FFT of a periodic sqrt-Hann-windowed frame, hop 256, unnormalized) plus the recurrent
//! caches `conv_cache` `[2, 1, 16, 16, 33]`, `tra_cache` `[2, 3, 1, 1, 16]`, `inter_cache`
//! `[2, 1, 33, 16]` (zeros at the start) → `enh` `[1, 257, 1, 2]` and `*_cache_out`. Synthesis is
//! the normalized inverse FFT times the same window, overlap-added. Measured on the dev rig
//! (jfk.wav + white noise at 5 dB SNR, streaming exactly as below): SI-SDR 5.0 → 14.4 dB at a lag of
//! 256 samples (no model lookahead). Latency: 16 ms of overlap-add plus the hop still buffering
//! (8-16 ms depending on the block).

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::stft::{sqrt_hann, FrameModel, StftConfig, StftStream};
use super::{check_io, Denoiser};

pub const SAMPLE_RATE: u32 = 16_000;
const FFT: usize = 512;
const HOP: usize = 256;
const BINS: usize = FFT / 2 + 1;
const CONV_SHAPE: [usize; 5] = [2, 1, 16, 16, 33];
const TRA_SHAPE: [usize; 5] = [2, 3, 1, 1, 16];
const INTER_SHAPE: [usize; 4] = [2, 1, 33, 16];

pub fn available() -> bool {
    true
}

/// The framing of GTCRN's streaming export.
pub(crate) fn stft_config() -> StftConfig {
    StftConfig {
        fft: FFT,
        hop: HOP,
        window: sqrt_hann(FFT),
        analysis_scale: 1.0,
        synthesis_scale: 1.0 / FFT as f32,
        model_delay_frames: 0,
    }
}

/// One GTCRN session and its caches (the EP it runs on is recorded as `vc.denoise`).
pub struct GtcrnFrames {
    session: Session,
    conv: Vec<f32>,
    tra: Vec<f32>,
    inter: Vec<f32>,
}

impl GtcrnFrames {
    pub fn load(path: &Path, ep: &str, opts: &VcSessionOptions) -> Result<GtcrnFrames, String> {
        let (session, _) = build_session(Some("vc.denoise"), ep, opts, ModelSource::File(path))
            .map_err(|e| format!("GTCRN denoiser {}: {e}", path.display()))?;
        let shape = |s: &[usize]| s.iter().map(|&d| d as i64).collect::<Vec<i64>>();
        check_io(
            &session,
            "GTCRN",
            &[
                ("mix", vec![1, BINS as i64, 1, 2]),
                ("conv_cache", shape(&CONV_SHAPE)),
                ("tra_cache", shape(&TRA_SHAPE)),
                ("inter_cache", shape(&INTER_SHAPE)),
            ],
            &["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
        )?;
        Ok(GtcrnFrames {
            session,
            conv: vec![0.0; CONV_SHAPE.iter().product()],
            tra: vec![0.0; TRA_SHAPE.iter().product()],
            inter: vec![0.0; INTER_SHAPE.iter().product()],
        })
    }
}

fn copy_out(outputs: &ort::session::SessionOutputs<'_>, name: &str, dst: &mut [f32]) -> Result<(), String> {
    let (_, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("GTCRN output {name}: {e}"))?;
    if data.len() != dst.len() {
        return Err(format!("GTCRN output {name} has {} values, expected {}", data.len(), dst.len()));
    }
    dst.copy_from_slice(data);
    Ok(())
}

impl FrameModel for GtcrnFrames {
    fn process(&mut self, spec: &mut [f32]) -> Result<(), String> {
        let mix = TensorRef::from_array_view(([1usize, BINS, 1, 2], &*spec)).map_err(|e| format!("GTCRN input: {e}"))?;
        let conv = TensorRef::from_array_view((CONV_SHAPE, self.conv.as_slice())).map_err(|e| format!("GTCRN cache: {e}"))?;
        let tra = TensorRef::from_array_view((TRA_SHAPE, self.tra.as_slice())).map_err(|e| format!("GTCRN cache: {e}"))?;
        let inter = TensorRef::from_array_view((INTER_SHAPE, self.inter.as_slice())).map_err(|e| format!("GTCRN cache: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["mix" => mix, "conv_cache" => conv, "tra_cache" => tra, "inter_cache" => inter])
            .map_err(|e| format!("GTCRN run: {e}"))?;
        copy_out(&outputs, "enh", spec)?;
        copy_out(&outputs, "conv_cache_out", &mut self.conv)?;
        copy_out(&outputs, "tra_cache_out", &mut self.tra)?;
        copy_out(&outputs, "inter_cache_out", &mut self.inter)?;
        Ok(())
    }

    fn reset(&mut self) {
        self.conv.iter_mut().for_each(|x| *x = 0.0);
        self.tra.iter_mut().for_each(|x| *x = 0.0);
        self.inter.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// The GTCRN denoiser (16 kHz mono).
pub struct Gtcrn {
    stream: StftStream<GtcrnFrames>,
}

/// Load from the model store (see [`super::model_path`]).
#[allow(dead_code)] // P3-S0 frozen entry point; callers use `denoise::create_with` / `load_from`.
pub fn load(ep: &str, opts: &VcSessionOptions) -> Result<Gtcrn, String> {
    let path = super::model_path(super::DenoiserKind::Gtcrn)?;
    load_from(&path, ep, opts, None)
}

/// Load `path`; `chunk`: every `process` call is a multiple of it (lower latency), `None` = any.
pub fn load_from(path: &Path, ep: &str, opts: &VcSessionOptions, chunk: Option<usize>) -> Result<Gtcrn, String> {
    let frames = GtcrnFrames::load(path, ep, opts)?;
    Ok(Gtcrn {
        stream: StftStream::new(frames, stft_config(), chunk)?,
    })
}

impl Denoiser for Gtcrn {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn latency_samples(&self) -> usize {
        self.stream.latency_samples()
    }

    fn process(&mut self, audio: &mut [f32]) -> Result<(), String> {
        let before = self.stream.errors().0;
        self.stream.process(audio);
        match self.stream.errors() {
            (n, Some(e)) if n > before => Err(e.to_string()),
            _ => Ok(()),
        }
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    fn granularity(&self) -> usize {
        self.stream.granularity()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::services::neural_audio::denoise::{test_model, DenoiserKind};

    /// Scale-invariant SDR (dB) of `est` against `clean`.
    pub(crate) fn si_sdr(est: &[f32], clean: &[f32]) -> f64 {
        let dot: f64 = est.iter().zip(clean).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let cc: f64 = clean.iter().map(|&b| f64::from(b) * f64::from(b)).sum();
        let alpha = dot / cc.max(1e-20);
        let (mut target, mut noise) = (0.0f64, 0.0f64);
        for (&e, &c) in est.iter().zip(clean) {
            let t = alpha * f64::from(c);
            target += t * t;
            noise += (f64::from(e) - t).powi(2);
        }
        10.0 * (target / noise.max(1e-20)).log10()
    }

    /// A voiced "speech-like" signal: harmonics of a gliding 120-220 Hz fundamental with a syllable
    /// envelope, plus white noise at `snr_db`.
    pub(crate) fn noisy_voice(rate: u32, seconds: f32, snr_db: f64) -> (Vec<f32>, Vec<f32>) {
        let n = (rate as f32 * seconds) as usize;
        let mut phase = 0.0f64;
        let clean: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let f0 = 170.0 + 50.0 * (2.0 * std::f64::consts::PI * 0.7 * t).sin();
                phase += 2.0 * std::f64::consts::PI * f0 / f64::from(rate);
                let env = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * 3.0 * t).cos()).powf(0.7);
                let v: f64 = (1..=12).map(|h| (phase * h as f64).sin() / h as f64).sum();
                (0.25 * env * v) as f32
            })
            .collect();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            // Box-Muller over xorshift.
            let mut u = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let (a, b) = (u().max(1e-12), u());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        let p_clean: f64 = clean.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
        let scale = (p_clean / 10f64.powf(snr_db / 10.0)).sqrt();
        let noisy = clean.iter().map(|&c| c + (rnd() * scale) as f32).collect();
        (clean, noisy)
    }

    /// The real GTCRN export streams through Sigil's STFT: noisy speech gets clearly cleaner, at
    /// exactly the reported latency, and a frame is cheap. Needs the model (`SIGIL_TEST_MODELS_DIR`
    /// or the dev cache's staged `vc/denoise-gtcrn`).
    #[test]
    #[ignore = "needs the GTCRN model"]
    fn real_model_gtcrn_denoises_at_the_reported_latency() {
        let Some(path) = test_model(DenoiserKind::Gtcrn) else {
            eprintln!("GTCRN model not found; skipped");
            return;
        };
        let mut den = load_from(&path, "cpu", &VcSessionOptions { intra_threads: Some(1), ..VcSessionOptions::default() }, Some(640)).unwrap();
        let (clean, noisy) = noisy_voice(SAMPLE_RATE, 6.0, 5.0);
        let d = den.latency_samples();
        assert_eq!(d, 256 + 128, "640-sample calls on a 256 hop");
        let mut out = Vec::with_capacity(noisy.len());
        let t0 = std::time::Instant::now();
        for chunk in noisy.chunks(640) {
            let mut c = chunk.to_vec();
            den.process(&mut c).unwrap();
            out.extend_from_slice(&c);
        }
        let per_frame_ms = t0.elapsed().as_secs_f64() * 1000.0 / (noisy.len() / HOP) as f64;
        let n = noisy.len() - d;
        let (before, after) = (si_sdr(&noisy[..n], &clean[..n]), si_sdr(&out[d..], &clean[..n]));
        eprintln!("GTCRN: SI-SDR {before:.1} -> {after:.1} dB, {per_frame_ms:.3} ms per 16 ms frame");
        assert!(after - before >= 4.0, "{before:.1} -> {after:.1} dB");
        // The alignment is the reported one: one sample off is clearly worse.
        let shifted = si_sdr(&out[d + 1..], &clean[..n - 1]);
        assert!(after > shifted);
    }
}
