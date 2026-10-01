//! DeepFilterNet3 / DPDFNet (P3-E): full-band 48 kHz denoisers for quality, run on the
//! device-rate conditioned copy before the aux mix. Models: `vc::assets::DENOISE_DFN3` /
//! `DENOISE_DPDFNET`.
//!
//! **DPDFNet-2 48 kHz HR** (Ceva, Apache-2.0; DeepFilterNet2 plus dual-path RNN blocks) is a true
//! streaming export: one 960-point STFT frame `spec` `[1, 1, 481, 2]` (vorbis window, hop 480,
//! unnormalized forward FFT) and the flat recurrent `state_in` (56 436 values, zeros at the start) →
//! `spec_e` and `state_out`; synthesis is the normalized inverse FFT times the window, overlap-added.
//! Measured on the dev rig (jfk.wav upsampled to 48 kHz + white noise at 5 dB SNR, streaming exactly
//! as here): SI-SDR 5.0 → 16.3 dB, with the output lagging the input by 2400 samples: 480 of
//! overlap-add plus a **4-frame (40 ms) model lookahead** (DeepFilterNet's conv and deep-filter
//! lookahead). Its latency is therefore about 50 ms plus the buffered part of a hop: the quality
//! option, not the low-latency one (that is GTCRN).
//!
//! **DeepFilterNet3**'s published ONNX export (`enc` / `erb_dec` / `df_dec`) runs whole sequences:
//! its GRU states are internal, so it cannot stream frame by frame through ONNX Runtime (upstream
//! streams it with tract's pulsed graphs). Until a stateful export exists it is not offered
//! ([`available`] is false for it); a stored `dfn3` runs DPDFNet, its successor on the same
//! architecture, with a note (`super::effective_kind`).

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

use super::stft::{vorbis, FrameModel, StftConfig, StftStream};
use super::{check_io, Denoiser, DenoiserKind};

pub const SAMPLE_RATE: u32 = 48_000;
const FFT: usize = 960;
const HOP: usize = 480;
const BINS: usize = FFT / 2 + 1;
/// Frames by which DPDFNet's output lags its input (measured, see the module docs).
pub const DPDFNET_LOOKAHEAD_FRAMES: usize = 4;

/// Whether `kind` runs in this build: DPDFNet yes; DeepFilterNet3 needs a streaming export.
pub fn available(kind: DenoiserKind) -> bool {
    matches!(kind, DenoiserKind::Dpdfnet)
}

/// Why DeepFilterNet3 does not run (shown when a stored `dfn3` runs its stand-in).
pub const DFN3_UNAVAILABLE: &str =
    "DeepFilterNet3's ONNX export cannot stream frame by frame (its recurrent state is internal)";

pub(crate) fn stft_config() -> StftConfig {
    StftConfig {
        fft: FFT,
        hop: HOP,
        window: vorbis(FFT),
        analysis_scale: 1.0,
        synthesis_scale: 1.0 / FFT as f32,
        model_delay_frames: DPDFNET_LOOKAHEAD_FRAMES,
    }
}

/// One DPDFNet session and its recurrent state (the EP it runs on is recorded as `vc.denoise`).
pub struct DpdfnetFrames {
    session: Session,
    state: Vec<f32>,
}

impl DpdfnetFrames {
    pub fn load(path: &Path, ep: &str, opts: &VcSessionOptions) -> Result<DpdfnetFrames, String> {
        let (session, _) = build_session(Some("vc.denoise"), ep, opts, ModelSource::File(path))
            .map_err(|e| format!("DPDFNet denoiser {}: {e}", path.display()))?;
        let state_len = session
            .inputs()
            .iter()
            .find(|i| i.name() == "state_in")
            .and_then(|i| i.dtype().tensor_shape().map(|s| s.to_vec()))
            .and_then(|s| (s.len() == 1 && s[0] > 0).then_some(s[0] as usize))
            .ok_or_else(|| format!("DPDFNet {}: no fixed-size `state_in` input", path.display()))?;
        check_io(
            &session,
            "DPDFNet",
            &[("spec", vec![1, 1, BINS as i64, 2]), ("state_in", vec![state_len as i64])],
            &["spec_e", "state_out"],
        )?;
        Ok(DpdfnetFrames {
            session,
            state: vec![0.0; state_len],
        })
    }
}

impl FrameModel for DpdfnetFrames {
    fn process(&mut self, spec: &mut [f32]) -> Result<(), String> {
        let input = TensorRef::from_array_view(([1usize, 1, BINS, 2], &*spec)).map_err(|e| format!("DPDFNet input: {e}"))?;
        let state = TensorRef::from_array_view(([self.state.len()], self.state.as_slice())).map_err(|e| format!("DPDFNet state: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["spec" => input, "state_in" => state])
            .map_err(|e| format!("DPDFNet run: {e}"))?;
        let (_, enh) = outputs["spec_e"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("DPDFNet output: {e}"))?;
        if enh.len() != spec.len() {
            return Err(format!("DPDFNet returned {} values, expected {}", enh.len(), spec.len()));
        }
        spec.copy_from_slice(enh);
        let (_, next) = outputs["state_out"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("DPDFNet state output: {e}"))?;
        if next.len() != self.state.len() {
            return Err(format!("DPDFNet state has {} values, expected {}", next.len(), self.state.len()));
        }
        self.state.copy_from_slice(next);
        Ok(())
    }

    fn reset(&mut self) {
        self.state.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// A 48 kHz DeepFilterNet-family denoiser (DPDFNet).
pub struct DeepFilter {
    stream: StftStream<DpdfnetFrames>,
}

/// Load `kind` from the model store (see [`super::model_path`]).
#[allow(dead_code)] // P3-S0 frozen entry point; callers use `denoise::create_with` / `load_from`.
pub fn load(kind: DenoiserKind, ep: &str, opts: &VcSessionOptions) -> Result<DeepFilter, String> {
    if !available(kind) {
        return Err(format!("{}: {DFN3_UNAVAILABLE}", kind.as_str()));
    }
    let path = super::model_path(kind)?;
    load_from(&path, ep, opts, None)
}

/// Load a DPDFNet export from `path`; `chunk`: every `process` call is a multiple of it.
pub fn load_from(path: &Path, ep: &str, opts: &VcSessionOptions, chunk: Option<usize>) -> Result<DeepFilter, String> {
    let frames = DpdfnetFrames::load(path, ep, opts)?;
    Ok(DeepFilter {
        stream: StftStream::new(frames, stft_config(), chunk)?,
    })
}

impl Denoiser for DeepFilter {
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
mod tests {
    use super::*;
    use crate::services::neural_audio::denoise::gtcrn::tests::si_sdr;
    use crate::services::neural_audio::denoise::test_model;

    #[test]
    fn deepfilternet3_is_not_offered_until_it_can_stream() {
        assert!(available(DenoiserKind::Dpdfnet));
        assert!(!available(DenoiserKind::Dfn3));
        assert!(load(DenoiserKind::Dfn3, "cpu", &VcSessionOptions::default()).err().unwrap().contains("dfn3"));
    }

    /// Real speech at 48 kHz: the dev cache's `jfk.wav` (16 kHz) linearly upsampled, plus white noise
    /// at `snr_db`. DPDFNet was trained on speech: a synthetic harmonic buzz is not what it keeps
    /// (it scores worse than its input on `noisy_voice`, in Python too).
    fn noisy_speech_48k(snr_db: f64) -> Option<(Vec<f32>, Vec<f32>)> {
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        if let Some(r) = std::env::var_os("SIGIL_TEST_MODELS_DIR") {
            roots.push(r.into());
        }
        if let Some(l) = std::env::var_os("LOCALAPPDATA") {
            roots.push(std::path::PathBuf::from(l).join("sigil-dev-cache"));
        }
        let wav = roots
            .into_iter()
            .map(|r| r.join("models").join("transcribe").join("samples").join("jfk.wav"))
            .find(|p| p.is_file())?;
        let mut reader = hound::WavReader::open(&wav).ok()?;
        let spec = reader.spec();
        let src: Vec<f32> = reader.samples::<i16>().filter_map(Result::ok).map(|v| f32::from(v) / 32768.0).collect();
        let ratio = f64::from(SAMPLE_RATE) / f64::from(spec.sample_rate);
        let n = ((src.len() as f64 * ratio) as usize).min(SAMPLE_RATE as usize * 8);
        let clean: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f64 / ratio;
                let j = x.floor() as usize;
                let t = (x - j as f64) as f32;
                let a = src[j.min(src.len() - 1)];
                let b = src[(j + 1).min(src.len() - 1)];
                a + (b - a) * t
            })
            .collect();
        // White noise (xorshift + Box-Muller, fixed seed).
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                let (a, b) = (uniform().max(1e-12), uniform());
                ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
            })
            .collect();
        let p_clean: f64 = clean.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
        let p_noise: f64 = noise.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
        let scale = (p_clean / p_noise.max(1e-20) / 10f64.powf(snr_db / 10.0)).sqrt() as f32;
        let noisy = clean.iter().zip(noise.iter()).map(|(&c, &e)| c + e * scale).collect();
        Some((clean, noisy))
    }

    /// The real DPDFNet export streams through Sigil's STFT: noisy speech gets clearly cleaner at the
    /// reported latency (overlap-add plus the 4-frame lookahead), and a frame fits the real-time
    /// budget. Needs the model (`SIGIL_TEST_MODELS_DIR` or the dev cache's `vc/denoise-dpdfnet`) and
    /// a speech clip (`jfk.wav`).
    #[test]
    #[ignore = "needs the DPDFNet model"]
    fn real_model_dpdfnet_denoises_at_the_reported_latency() {
        let Some(path) = test_model(DenoiserKind::Dpdfnet) else {
            eprintln!("DPDFNet model not found; skipped");
            return;
        };
        let Some((clean, noisy)) = noisy_speech_48k(5.0) else {
            eprintln!("jfk.wav not found; skipped");
            return;
        };
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            ..VcSessionOptions::default()
        };
        let mut den = load_from(&path, "cpu", &opts, Some(480)).unwrap();
        let d = den.latency_samples();
        assert_eq!(d, 480 + 4 * 480);
        let mut out = Vec::with_capacity(noisy.len());
        let t0 = std::time::Instant::now();
        for chunk in noisy.chunks(480) {
            let mut c = chunk.to_vec();
            den.process(&mut c).unwrap();
            out.extend_from_slice(&c);
        }
        let per_frame_ms = t0.elapsed().as_secs_f64() * 1000.0 / (noisy.len() / HOP) as f64;
        let n = noisy.len() - d;
        let (before, after) = (si_sdr(&noisy[..n], &clean[..n]), si_sdr(&out[d..], &clean[..n]));
        // The lag with the best score is the reported one.
        let best = (d.saturating_sub(960)..d + 960)
            .step_by(480)
            .max_by(|&a, &b| si_sdr(&out[a..a + n - 960], &clean[..n - 960]).total_cmp(&si_sdr(&out[b..b + n - 960], &clean[..n - 960])))
            .unwrap();
        eprintln!("DPDFNet: SI-SDR {before:.1} -> {after:.1} dB, {per_frame_ms:.3} ms per 10 ms frame");
        assert_eq!(best, d);
        assert!(after - before >= 4.0, "{before:.1} -> {after:.1} dB");
        assert!(per_frame_ms < 10.0, "{per_frame_ms} ms per 10 ms frame");
    }
}
