//! Streaming STFT around a frame-by-frame spectral model (the denoisers' shared DSP): sliding
//! analysis window, one model call per hop, inverse FFT and overlap-add, with a constant delay.
//!
//! - Input is buffered into hops; every complete hop runs one frame (`fft` samples ending at the
//!   newest one, windowed, forward FFT, the model, inverse FFT, windowed, overlap-added). The
//!   stream starts as if silence had been flowing, so the first frame is a full one.
//! - Output is the input delayed by exactly [`StftStream::latency_samples`] samples, for any call
//!   length: a FIFO primed with `prime` zeros covers the part of a hop that is still buffered when a
//!   call returns (`hop - 1` for arbitrary call lengths, `hop - gcd(chunk, hop)` when every call is a
//!   multiple of a known `chunk`). The model's own lookahead (frames by which its output lags its
//!   input) is part of the reported latency.
//! - A model error keeps that frame's spectrum unprocessed (the noisy input passes; the stream and
//!   its timing continue) and is counted.
//! - Allocation-free per call after construction (the FIFO and scratch are sized up front).

use std::collections::VecDeque;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// One spectral model: enhances one frame of `bins` complex bins in place.
pub(crate) trait FrameModel: Send {
    /// `spec` is `[bins][2]` (real, imaginary), the layout of the denoisers' ONNX inputs.
    fn process(&mut self, spec: &mut [f32]) -> Result<(), String>;
    /// Drop recurrent state (stream gap / restart).
    fn reset(&mut self);
}

/// Framing and scaling of one model family.
#[derive(Clone, Debug)]
pub(crate) struct StftConfig {
    pub fft: usize,
    pub hop: usize,
    /// Analysis and synthesis window (`fft` samples); `window[n]^2 + window[n + hop]^2 == 1` for
    /// perfect reconstruction with `hop == fft / 2`.
    pub window: Vec<f32>,
    /// Applied to the forward FFT output.
    pub analysis_scale: f32,
    /// Applied to the (unnormalized) inverse FFT output.
    pub synthesis_scale: f32,
    /// Frames by which the model's output lags its input (lookahead).
    pub model_delay_frames: usize,
}

impl StftConfig {
    pub fn bins(&self) -> usize {
        self.fft / 2 + 1
    }
}

/// Periodic Hann window, square-rooted (GTCRN's `torch.hann_window(n).pow(0.5)`).
pub(crate) fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).sqrt() as f32)
        .collect()
}

/// Vorbis window (DeepFilterNet / DPDFNet): `sin(pi/2 * sin^2(pi (n + 0.5) / N))`.
pub(crate) fn vorbis(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let s = (std::f64::consts::PI * (i as f64 + 0.5) / n as f64).sin();
            (0.5 * std::f64::consts::PI * s * s).sin() as f32
        })
        .collect()
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// The streaming engine.
pub(crate) struct StftStream<M: FrameModel> {
    model: M,
    cfg: StftConfig,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    /// The analysis window's samples: the previous `fft - hop` plus the hop being filled.
    frame: Vec<f32>,
    /// Samples of the current hop received so far.
    hop_fill: usize,
    time: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    spec: Vec<f32>,
    scratch_fwd: Vec<Complex<f32>>,
    scratch_inv: Vec<Complex<f32>>,
    ola: Vec<f32>,
    out: VecDeque<f32>,
    /// Zeros the output FIFO starts with.
    prime: usize,
    /// Call lengths since the last reset must be multiples of this for the FIFO never to run dry
    /// (`gcd(chunk, hop)`; 1 without a chunk promise).
    granularity: usize,
    errors: u64,
    last_error: Option<String>,
    underflows: u64,
}

impl<M: FrameModel> StftStream<M> {
    /// `chunk`: every `process` call's length is a multiple of it (lowers the latency); `None` =
    /// any length.
    pub fn new(model: M, cfg: StftConfig, chunk: Option<usize>) -> Result<Self, String> {
        let (n, h) = (cfg.fft, cfg.hop);
        if n == 0 || h == 0 || h > n || cfg.window.len() != n || n % 2 != 0 {
            return Err(format!("invalid STFT framing: fft {n}, hop {h}, window {}", cfg.window.len()));
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let (prime, granularity) = match chunk.filter(|&c| c > 0) {
            Some(c) => (h - gcd(c, h), gcd(c, h)),
            None => (h - 1, 1),
        };
        let bins = cfg.bins();
        let mut s = Self {
            scratch_fwd: fwd.make_scratch_vec(),
            scratch_inv: inv.make_scratch_vec(),
            time: fwd.make_input_vec(),
            spectrum: fwd.make_output_vec(),
            spec: vec![0.0; bins * 2],
            frame: vec![0.0; n],
            hop_fill: 0,
            ola: vec![0.0; n],
            out: VecDeque::with_capacity(n + prime + 16_384),
            prime,
            granularity,
            errors: 0,
            last_error: None,
            underflows: 0,
            model,
            cfg,
            fwd,
            inv,
        };
        s.out.extend(std::iter::repeat_n(0.0, prime));
        Ok(s)
    }

    /// Constant delay of the output (samples): the overlap-add (`fft - hop`), the hop still being
    /// buffered, and the model's lookahead.
    pub fn latency_samples(&self) -> usize {
        self.cfg.fft - self.cfg.hop + self.prime + self.cfg.model_delay_frames * self.cfg.hop
    }

    /// Call lengths that keep the latency exact (see `Denoiser::granularity`).
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Frames whose model call failed (their noisy spectrum passed) and the last error.
    pub fn errors(&self) -> (u64, Option<&str>) {
        (self.errors, self.last_error.as_deref())
    }

    /// Output samples that had to be zero-filled (a call length that broke the `chunk` promise).
    #[cfg(test)]
    pub fn underflows(&self) -> u64 {
        self.underflows
    }

    /// Process `audio` in place (the output is the input delayed by [`Self::latency_samples`]).
    pub fn process(&mut self, audio: &mut [f32]) {
        let (n, h) = (self.cfg.fft, self.cfg.hop);
        let mut rest: &[f32] = audio;
        // Feed first (every complete hop runs a frame), then read the same count back. The input
        // is copied out of `audio` before any output is written into it.
        let mut consumed = 0usize;
        while consumed < audio.len() {
            let take = (h - self.hop_fill).min(audio.len() - consumed);
            let start = n - h + self.hop_fill;
            self.frame[start..start + take].copy_from_slice(&rest[..take]);
            rest = &rest[take..];
            consumed += take;
            self.hop_fill += take;
            if self.hop_fill == h {
                self.run_frame();
                self.frame.copy_within(h.., 0);
                self.hop_fill = 0;
            }
        }
        for s in audio.iter_mut() {
            *s = match self.out.pop_front() {
                Some(v) => v,
                None => {
                    self.underflows += 1;
                    0.0
                }
            };
        }
    }

    fn run_frame(&mut self) {
        let (n, h) = (self.cfg.fft, self.cfg.hop);
        for ((t, &x), &w) in self.time.iter_mut().zip(self.frame.iter()).zip(self.cfg.window.iter()) {
            *t = x * w;
        }
        if self
            .fwd
            .process_with_scratch(&mut self.time, &mut self.spectrum, &mut self.scratch_fwd)
            .is_err()
        {
            self.spectrum.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
        }
        let a = self.cfg.analysis_scale;
        for (i, c) in self.spectrum.iter().enumerate() {
            self.spec[2 * i] = c.re * a;
            self.spec[2 * i + 1] = c.im * a;
        }
        if let Err(e) = self.model.process(&mut self.spec) {
            self.errors += 1;
            self.last_error = Some(e);
            for (i, c) in self.spectrum.iter().enumerate() {
                self.spec[2 * i] = c.re * a;
                self.spec[2 * i + 1] = c.im * a;
            }
        }
        let bins = self.spectrum.len();
        for (i, c) in self.spectrum.iter_mut().enumerate() {
            let (re, im) = (self.spec[2 * i], self.spec[2 * i + 1]);
            *c = if re.is_finite() && im.is_finite() {
                Complex::new(re, im)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        // A real signal's DC and Nyquist bins are real.
        self.spectrum[0].im = 0.0;
        self.spectrum[bins - 1].im = 0.0;
        if self
            .inv
            .process_with_scratch(&mut self.spectrum, &mut self.time, &mut self.scratch_inv)
            .is_err()
        {
            self.time.iter_mut().for_each(|t| *t = 0.0);
        }
        let s = self.cfg.synthesis_scale;
        for ((o, &t), &w) in self.ola.iter_mut().zip(self.time.iter()).zip(self.cfg.window.iter()) {
            *o += t * w * s;
        }
        self.out.extend(self.ola[..h].iter().copied());
        self.ola.copy_within(h.., 0);
        self.ola[n - h..].iter_mut().for_each(|o| *o = 0.0);
    }

    /// Back to silence (stream gap / restart): model state, buffers and the output delay.
    pub fn reset(&mut self) {
        self.model.reset();
        self.frame.iter_mut().for_each(|x| *x = 0.0);
        self.ola.iter_mut().for_each(|x| *x = 0.0);
        self.hop_fill = 0;
        self.out.clear();
        self.out.extend(std::iter::repeat_n(0.0, self.prime));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A model that changes nothing (checks the DSP alone).
    pub(crate) struct Identity;

    impl FrameModel for Identity {
        fn process(&mut self, _spec: &mut [f32]) -> Result<(), String> {
            Ok(())
        }
        fn reset(&mut self) {}
    }

    /// A model that fails on every frame.
    struct Broken;

    impl FrameModel for Broken {
        fn process(&mut self, spec: &mut [f32]) -> Result<(), String> {
            spec.iter_mut().for_each(|x| *x = f32::NAN);
            Err("boom".into())
        }
        fn reset(&mut self) {}
    }

    pub(crate) fn gtcrn_cfg() -> StftConfig {
        StftConfig {
            fft: 512,
            hop: 256,
            window: sqrt_hann(512),
            analysis_scale: 1.0,
            synthesis_scale: 1.0 / 512.0,
            model_delay_frames: 0,
        }
    }

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 180.0 * t).sin() + 0.2 * (2.0 * std::f32::consts::PI * 1_234.5 * t).sin()
            })
            .collect()
    }

    /// Run `input` through `stream` in calls of the given lengths (cycled).
    fn run<M: FrameModel>(stream: &mut StftStream<M>, input: &[f32], calls: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut pos = 0;
        let mut k = 0;
        while pos < input.len() {
            let n = calls[k % calls.len()].min(input.len() - pos);
            let mut block = input[pos..pos + n].to_vec();
            stream.process(&mut block);
            out.extend_from_slice(&block);
            pos += n;
            k += 1;
        }
        out
    }

    /// With an identity model the stream is a pure delay of exactly `latency_samples`, for any call
    /// lengths, and the same `N in -> N out` per call.
    #[test]
    fn identity_is_an_exact_constant_delay_for_any_chunking() {
        // A length every call pattern below divides (the promise holds to the last call).
        let input = signal(15_360);
        for (chunk, calls) in [
            (None, vec![1usize, 7, 300, 256, 511, 1000]),
            (Some(640), vec![640]),
            (Some(256), vec![256, 512]),
            (Some(160), vec![160, 320]),
        ] {
            let mut s = StftStream::new(Identity, gtcrn_cfg(), chunk).unwrap();
            let d = s.latency_samples();
            let out = run(&mut s, &input, &calls);
            assert_eq!(out.len(), input.len());
            assert_eq!(s.underflows(), 0, "{chunk:?}");
            for i in d..input.len() {
                assert!((out[i] - input[i - d]).abs() < 1e-4, "{chunk:?} sample {i}: {} vs {}", out[i], input[i - d]);
            }
            assert!(out[..d].iter().all(|v| v.abs() < 1e-6), "silence before the delay");
        }
        // The latency formula: fft - hop, plus hop - gcd(chunk, hop).
        let lat = |chunk| StftStream::new(Identity, gtcrn_cfg(), chunk).unwrap().latency_samples();
        assert_eq!(lat(None), 256 + 255);
        assert_eq!(lat(Some(640)), 256 + 128);
        assert_eq!(lat(Some(512)), 256);
    }

    /// DPDFNet's framing (vorbis window, 48 kHz hop 480) also reconstructs exactly, and the model's
    /// lookahead is part of the latency.
    #[test]
    fn vorbis_framing_reconstructs() {
        let cfg = StftConfig {
            fft: 960,
            hop: 480,
            window: vorbis(960),
            analysis_scale: 1.0,
            synthesis_scale: 1.0 / 960.0,
            model_delay_frames: 4,
        };
        let w = &cfg.window;
        for i in 0..480 {
            assert!((w[i] * w[i] + w[i + 480] * w[i + 480] - 1.0).abs() < 1e-5);
        }
        let mut s = StftStream::new(Identity, cfg, Some(480)).unwrap();
        assert_eq!(s.latency_samples(), 480 + 4 * 480);
        let input = signal(9_600);
        let out = run(&mut s, &input, &[480]);
        // The identity model has no lookahead: its output lags by the framing alone.
        for i in 480..input.len() {
            assert!((out[i] - input[i - 480]).abs() < 1e-4);
        }
    }

    /// A failing model passes the frame unprocessed (the stream keeps its timing), and a reset
    /// returns to silence with the same delay.
    #[test]
    fn model_errors_pass_audio_and_reset_restarts_cleanly() {
        let input = signal(4_096);
        let mut s = StftStream::new(Broken, gtcrn_cfg(), Some(256)).unwrap();
        let d = s.latency_samples();
        let out = run(&mut s, &input, &[256]);
        assert!(s.errors().0 > 0);
        assert_eq!(s.errors().1, Some("boom"));
        for i in d..input.len() {
            assert!((out[i] - input[i - d]).abs() < 1e-4);
        }
        s.reset();
        let out = run(&mut s, &input, &[256]);
        assert!(out[..d].iter().all(|v| v.abs() < 1e-6));
        assert!((out[d + 100] - input[100]).abs() < 1e-4);
    }

    /// A call that breaks the `chunk` promise is zero-filled and counted, never a panic.
    #[test]
    fn broken_chunk_promise_is_counted() {
        // Whole hops promised (no start-up FIFO): a 100-sample call cannot be covered.
        let mut s = StftStream::new(Identity, gtcrn_cfg(), Some(512)).unwrap();
        let mut a = vec![0.1f32; 100];
        s.process(&mut a);
        assert_eq!(s.underflows(), 100);
        // Any length promised: never.
        let mut s = StftStream::new(Identity, gtcrn_cfg(), None).unwrap();
        let mut b = vec![0.1f32; 1];
        for _ in 0..600 {
            s.process(&mut b);
        }
        s.process(&mut a);
        assert_eq!(s.underflows(), 0);
    }
}
