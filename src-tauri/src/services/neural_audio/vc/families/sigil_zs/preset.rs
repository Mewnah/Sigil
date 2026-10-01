//! Sigil-ZS presets and tick geometry (P5-B; ADR 0006 "Presets to start P5-B from").
//!
//! Every length comes from upstream `real-time-gui.py` `custom_infer` as `scripts/vc-families/
//! seed_vc/sampler.py` (`Preset.units` / `Preset.shapes`) reproduces it, generalised from upstream's
//! 20 ms units to the stream harness's block and SOLA tail:
//!
//! - the window (`n16` samples at 16 kHz, a whole number of 20 ms units `W`) is the content
//!   encoder's left context (`extra_ce`), the returned span (`want`: the block plus the harness's
//!   SOLA tail, rounded up to 20 ms) and the right context (`right`, 20 ms);
//! - XLS-R gives `T = W − 1` frames; the length regulator drops the first `ce = extra_ce −
//!   extra_dit` of them (the DiT sees `extra_dit` of left context) and stretches the rest to
//!   `n_mel = ⌊(W − ce) · 441 / 256⌋` mel frames (upstream's `target_lengths`);
//! - the DiT runs over `L = P + n_mel` frames (`P` = the reference prompt's mel frames);
//! - the vocoder's `n_mel · 256` samples at 22.05 kHz end at the window's end (upstream slices from
//!   the end); the call returns `out_samples` (the harness's `want` at 22.05 kHz) ending `right`
//!   before it, so the voice runs `right` (20 ms) behind its input (`FamilyEngine::model_delay_16k`);
//! - partial decode synthesises only from `dec_start` = the first returned frame minus
//!   [`HIFT_MARGIN`] (the decoder's receptive field: 12 frames reproduce the whole-tick decode).
//!
//! A tick turns `lr_in` content frames into `n_mel` mel frames, so one block of input moves the
//! content by [`Geometry::frames_per_block`] mel frames (8.91 for `sigil_low`), not by the block's
//! 8.61 frames of output: the anchored noise grid advances [`Geometry::grid_frames`] (its rounding)
//! per tick (ADR 0006 "Coherent ticks").

use crate::services::neural_audio::vc::engine::options::PrecisionMode;

use super::SigilZsOptions;

/// Output sample rate (HiFT, CosyVoice-300M).
pub const SR_OUT: u32 = 22_050;
/// Samples per mel frame at [`SR_OUT`].
pub const HOP: usize = 256;
/// Upstream's time unit (20 ms) at 16 kHz: one XLS-R frame.
pub const UNIT_16K: usize = 320;
/// The same unit at [`SR_OUT`].
pub const UNIT_OUT: usize = 441;
/// HiFT's NSF source: the fundamental and 8 harmonics.
pub const N_HARM: usize = 9;
/// Frames of left context `hift_nsf` needs to reproduce the whole-tick decode (ADR 0006: > 90 dB).
pub const HIFT_MARGIN: usize = 12;
/// Mel bins of the DiT and HiFT.
pub const MEL_BINS: usize = 80;
/// XLS-R layer-12 feature width.
pub const CONTENT_DIM: usize = 1024;
/// Length-regulator output width (the DiT's condition).
pub const COND_DIM: usize = 384;
/// CAM++ style width.
pub const STYLE_DIM: usize = 192;
/// Longest DiT sequence (the exported estimator's RoPE table holds 8192 positions, minus the style
/// and time tokens; kept with margin).
pub const MAX_SEQ: usize = 8_000;
/// XLS-R's receptive field (samples at 16 kHz): the shortest input that gives one frame.
pub const XLSR_FIELD: usize = 400;
/// Shortest reference prompt (seconds).
pub const MIN_PROMPT_S: f32 = 0.5;

/// One preset (ADR 0006). Lengths are in upstream's 20 ms units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub name: &'static str,
    /// Euler steps of the flow.
    pub steps: u32,
    /// Classifier-free guidance rate (0 runs the batch-1 step graph).
    pub cfg_rate: f32,
    /// Longest reference prompt (seconds).
    pub prompt_s: f32,
    /// New audio per tick.
    pub block_units: usize,
    /// Left context of the content encoder.
    pub extra_ce_units: usize,
    /// Left context the DiT keeps (upstream `extra_time`).
    pub extra_dit_units: usize,
    /// Right context (upstream `extra_time_right`): the voice's own delay.
    pub right_units: usize,
}

/// 100 ms blocks, 1 s content context (0.3 s for the DiT), 6 steps, CFG 0.7, 4 s prompt (the
/// default). P5-B's benchmark chose the prompt (ADR 0006 started from 3 s): over three source
/// clips a 4 s speech prompt took the exact-CAM++ similarity to the reference from 0.548 to 0.582
/// (upstream's offline conversion: 0.559; 5 s: 0.587) for +2.7 ms per tick on the RTX 4090 (CUDA
/// 22.0 -> 24.7 ms, DirectML 27.0 -> 30.2 ms).
pub const SIGIL_LOW: Preset = Preset {
    name: "sigil_low",
    steps: 6,
    cfg_rate: 0.7,
    prompt_s: 4.0,
    block_units: 5,
    extra_ce_units: 50,
    extra_dit_units: 15,
    right_units: 1,
};

/// `sigil_low` with 4 steps, no CFG and a 3 s prompt (the lightest). ADR 0006 started from 1.5 s;
/// 3 s took the similarity from 0.447 to 0.485 for +1 ms per tick (CUDA 15.1 -> 16.0 ms, DirectML
/// 18.2 -> 20.0 ms).
pub const SIGIL_LEAN: Preset = Preset {
    name: "sigil_lean",
    steps: 4,
    cfg_rate: 0.0,
    prompt_s: 3.0,
    ..SIGIL_LOW
};

/// Upstream's README recommendation for an RTX 3060 Laptop: 180 ms blocks, 2.5 s / 0.5 s context,
/// 10 steps, CFG 0.7, 3 s prompt.
pub const UPSTREAM_README: Preset = Preset {
    name: "upstream_readme",
    steps: 10,
    cfg_rate: 0.7,
    prompt_s: 3.0,
    block_units: 9,
    extra_ce_units: 125,
    extra_dit_units: 25,
    right_units: 1,
};

/// Every preset, in [`super::PRESETS`] order.
pub const ALL: [Preset; 3] = [SIGIL_LOW, SIGIL_LEAN, UPSTREAM_README];

/// The preset called `name` (the default for an unknown name).
pub fn preset(name: &str) -> Preset {
    ALL.into_iter().find(|p| p.name == name.trim()).unwrap_or(SIGIL_LOW)
}

/// Whether the HiFT decoder runs in fp16 when the options leave it to the build (`None`): no.
/// Measured on the dev rig (P5-B): the fp16 decoder saves about 1 ms per tick on the RTX 4090
/// (ADR 0006: 1.0-2.4 ms), passes its stage gate at 47 dB but moves the waveform of every tick
/// (HiFT predicts its own F0), so the fp32 decoder stays the default and fp16 is a switch.
pub const DEFAULT_FP16_VOCODER: bool = false;

/// What a voice runs: its preset with the options' overrides, and the precision switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub preset: Preset,
    pub fp16_xlsr: bool,
    pub fp16_dit: bool,
    pub fp16_vocoder: bool,
}

impl Settings {
    /// The options (already normalized) applied to their preset. `precision` `fp32` keeps every graph
    /// fp32; `fp16` asks for the fp16 decoder (the one reduced-precision graph that passes its
    /// gate); the XLS-R and DiT fp16 graphs run only on their own switches (ADR 0006: their
    /// waveforms fail parity).
    pub fn new(options: &SigilZsOptions, precision: PrecisionMode) -> Settings {
        let mut p = preset(&options.preset);
        if let Some(s) = options.steps {
            p.steps = s.clamp(1, 30);
        }
        if let Some(c) = options.cfg_rate.filter(|c| c.is_finite()) {
            p.cfg_rate = c.clamp(0.0, 1.0);
        }
        if let Some(s) = options.prompt_s.filter(|s| s.is_finite()) {
            p.prompt_s = s.clamp(MIN_PROMPT_S, 10.0);
        }
        let fp16_vocoder = match precision {
            PrecisionMode::Fp32 => false,
            PrecisionMode::Fp16 => true,
            PrecisionMode::Auto => options.fp16_vocoder.unwrap_or(DEFAULT_FP16_VOCODER),
        };
        Settings {
            preset: p,
            fp16_xlsr: options.fp16_xlsr && precision != PrecisionMode::Fp32,
            fp16_dit: options.fp16_dit && precision != PrecisionMode::Fp32,
            fp16_vocoder,
        }
    }

    /// New 16 kHz samples per tick.
    pub fn block_16k(&self) -> usize {
        self.preset.block_units * UNIT_16K
    }
}

/// Mel frames the `mel` graph gives for `samples` at 22.05 kHz (upstream `mel_spectrogram`, n_fft
/// 1024, hop 256, a (1024 − 256) / 2 reflect pad each side, no centre): `(S + 768 − 1024) / 256 + 1`.
pub fn mel_frames(samples: usize) -> usize {
    if samples < HOP {
        0
    } else {
        (samples - HOP) / HOP + 1
    }
}

/// XLS-R frames for `samples` at 16 kHz: `(N − 400) / 320 + 1`.
pub fn xlsr_frames(samples: usize) -> usize {
    if samples < XLSR_FIELD {
        0
    } else {
        (samples - XLSR_FIELD) / UNIT_16K + 1
    }
}

/// `round(samples_16k · 22050 / 16000)` (half up, as the RVC engine counts).
pub fn out_len(samples_16k: usize) -> usize {
    ((samples_16k as u64 * u64::from(SR_OUT) + 8_000) / 16_000) as usize
}

/// The shapes of one call (see the module docs). Every field is fixed for a stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Geometry {
    /// Window (context) samples at 16 kHz.
    pub n16: usize,
    /// XLS-R frames of the window.
    pub xlsr_frames: usize,
    /// Content frames dropped before the length regulator.
    pub ce: usize,
    /// Content frames the length regulator stretches (`xlsr_frames − ce`).
    pub lr_in: usize,
    /// Mel frames of the tick.
    pub n_mel: usize,
    /// Mel frames of the reference prompt.
    pub prompt: usize,
    /// DiT length (`prompt + n_mel`).
    pub big_l: usize,
    /// 16 kHz samples the call returns (`ConversionCtx::new_frames`).
    pub want_16k: usize,
    /// 22.05 kHz samples the call returns.
    pub out_samples: usize,
    /// First returned sample in the tick's `n_mel · 256` vocoder timeline.
    pub out_start: usize,
    /// Zero samples in front of the returned audio when the vocoded window is shorter than the
    /// returned span (one-off conversions only).
    pub pad_front: usize,
    /// First decoded frame (partial decode; 0 decodes the whole tick).
    pub dec_start: usize,
    /// New 16 kHz samples per tick (0 for a one-off conversion).
    pub block_16k: usize,
    /// Right context the returned audio ends before the window end (16 kHz samples).
    pub right_16k: usize,
    /// Anchored-noise grid step per tick (mel frames; [`Geometry::frames_per_block`] rounded).
    pub grid_frames: usize,
}

impl Geometry {
    /// The stream harness's shape: `block_16k` new samples per call (the preset's block) and
    /// `want_16k` returned (the block plus the harness's SOLA tail), `prompt` mel frames of
    /// reference. `partial` selects partial decode.
    pub fn stream(p: &Preset, block_16k: usize, want_16k: usize, prompt: usize, partial: bool) -> Result<Geometry, String> {
        if block_16k != p.block_units * UNIT_16K {
            return Err(format!(
                "Sigil-ZS: the stream harness asked for {block_16k}-sample blocks, the {} preset runs {}",
                p.name,
                p.block_units * UNIT_16K
            ));
        }
        if want_16k < block_16k {
            return Err(format!("Sigil-ZS: {want_16k} returned samples is less than a {block_16k}-sample block"));
        }
        if p.extra_dit_units > p.extra_ce_units {
            return Err("Sigil-ZS: the DiT context exceeds the content encoder's".into());
        }
        let want_units = want_16k.div_ceil(UNIT_16K);
        let w = p.extra_ce_units + want_units + p.right_units;
        let n16 = w * UNIT_16K;
        let xlsr = xlsr_frames(n16);
        let ce = p.extra_ce_units - p.extra_dit_units;
        if xlsr <= ce + 1 {
            return Err("Sigil-ZS: the window is too short for the content encoder".into());
        }
        let n_mel = (w - ce) * UNIT_OUT / HOP;
        let out_samples = out_len(want_16k);
        let tail = p.right_units * UNIT_OUT;
        let out_start = (n_mel * HOP)
            .checked_sub(out_samples + tail)
            .ok_or("Sigil-ZS: the returned span does not fit the tick")?;
        let lr_in = xlsr - ce;
        let fpb = p.block_units as f64 * n_mel as f64 / lr_in as f64;
        let g = Geometry {
            n16,
            xlsr_frames: xlsr,
            ce,
            lr_in,
            n_mel,
            prompt,
            big_l: prompt + n_mel,
            want_16k,
            out_samples,
            out_start,
            pad_front: 0,
            dec_start: if partial { (out_start / HOP).saturating_sub(HIFT_MARGIN) } else { 0 },
            block_16k,
            right_16k: p.right_units * UNIT_16K,
            grid_frames: (fpb.round() as usize).max(1),
        };
        g.check()?;
        Ok(g)
    }

    /// A one-off conversion of a whole window (`len` samples at 16 kHz, `new_frames` of them
    /// returned, end-aligned): the same model offline (upstream `inference.py`'s length rule,
    /// `⌊len / 16000 · 22050 / 256⌋` mel frames, no right context), decoded whole.
    pub fn offline(len: usize, new_frames: usize, prompt: usize) -> Result<Geometry, String> {
        let xlsr = xlsr_frames(len);
        if xlsr < 2 {
            return Err(format!("Sigil-ZS: {len} samples is too short to convert"));
        }
        let n_mel = (len as u64 * u64::from(SR_OUT) / (16_000 * HOP as u64)) as usize;
        if n_mel == 0 {
            return Err(format!("Sigil-ZS: {len} samples is too short to convert"));
        }
        let new_frames = new_frames.min(len);
        let out_samples = out_len(new_frames);
        let (out_start, pad_front) = match (n_mel * HOP).checked_sub(out_samples) {
            Some(s) => (s, 0),
            None => (0, out_samples - n_mel * HOP),
        };
        let g = Geometry {
            n16: len,
            xlsr_frames: xlsr,
            ce: 0,
            lr_in: xlsr,
            n_mel,
            prompt,
            big_l: prompt + n_mel,
            want_16k: new_frames,
            out_samples,
            out_start,
            pad_front,
            dec_start: 0,
            block_16k: 0,
            right_16k: 0,
            grid_frames: 1,
        };
        g.check()?;
        Ok(g)
    }

    fn check(&self) -> Result<(), String> {
        if self.prompt == 0 {
            return Err("Sigil-ZS: the reference prompt is empty".into());
        }
        if self.big_l > MAX_SEQ {
            return Err(format!(
                "Sigil-ZS: {} DiT frames is more than the model's {MAX_SEQ} (shorten the reference prompt or the conversion)",
                self.big_l
            ));
        }
        Ok(())
    }

    /// Frames the vocoder synthesises (`n_mel − dec_start`).
    pub fn dec_frames(&self) -> usize {
        self.n_mel - self.dec_start
    }

    /// Samples the vocoder synthesises.
    pub fn dec_samples(&self) -> usize {
        self.dec_frames() * HOP
    }

    /// The returned audio inside the decoded slice: `(first sample, samples)` (the call pads
    /// [`Geometry::pad_front`] zeros in front of it).
    pub fn returned(&self) -> (usize, usize) {
        (self.out_start - self.dec_start * HOP, self.out_samples - self.pad_front)
    }

    /// How far one block of input moves the content on the tick's mel-frame grid (mel frames; see
    /// the module docs). 0 for a one-off conversion.
    pub fn frames_per_block(&self) -> f64 {
        if self.block_16k == 0 || self.lr_in == 0 {
            return 0.0;
        }
        (self.block_16k / UNIT_16K) as f64 * self.n_mel as f64 / self.lr_in as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sampler.py` `Preset.shapes()` for the ADR presets with a 40 ms crossfade (upstream's), as
    /// `make_vectors.py` recorded them: the harness geometry with `want` = upstream's
    /// `return_length` reproduces upstream exactly.
    #[test]
    fn stream_geometry_matches_the_reference_sampler() {
        // (preset, return units, P) -> (n16, xlsr, lr_in, n_mel, L, out_samples, out_start, dec_start)
        let cases = [
            (SIGIL_LOW, 8, 258, (18_880, 58, 23, 41, 299, 3_528, 6_527, 13)),
            (SIGIL_LEAN, 8, 129, (18_880, 58, 23, 41, 170, 3_528, 6_527, 13)),
            // upstream_readme: W = 125 + 2 + 1 + 9 + 1 = 138, ce = 100, n_mel = 38 * 441 / 256 = 65.
            (UPSTREAM_README, 12, 258, (44_160, 137, 37, 65, 323, 5_292, 10_907, 30)),
        ];
        for (p, ret_units, prompt, want) in cases {
            let g = Geometry::stream(&p, p.block_units * UNIT_16K, ret_units * UNIT_16K, prompt, true).unwrap();
            assert_eq!(
                (g.n16, g.xlsr_frames, g.lr_in, g.n_mel, g.big_l, g.out_samples, g.out_start, g.dec_start),
                want,
                "{}",
                p.name
            );
            assert_eq!(g.right_16k, 320);
            assert_eq!(g.dec_frames() * HOP, g.dec_samples());
            let (first, len) = g.returned();
            assert_eq!(first + len + p.right_units * UNIT_OUT, g.dec_samples(), "{}: end-aligned before the right context", p.name);
            let full = Geometry::stream(&p, p.block_units * UNIT_16K, ret_units * UNIT_16K, prompt, false).unwrap();
            assert_eq!((full.dec_start, full.returned().0), (0, full.out_start));
        }
        // ADR 0006: 15.81 frames per block for upstream_readme, 8.91 for sigil_low.
        let g = Geometry::stream(&UPSTREAM_README, 2_880, 3_840, 258, true).unwrap();
        assert!((g.frames_per_block() - 15.81).abs() < 0.01, "{}", g.frames_per_block());
        assert_eq!(g.grid_frames, 16);
        let g = Geometry::stream(&SIGIL_LOW, 1_600, 2_560, 258, true).unwrap();
        assert!((g.frames_per_block() - 8.913).abs() < 0.001);
        assert_eq!(g.grid_frames, 9);
    }

    /// The harness's real SOLA tails (not whole 20 ms units) round the window up to the content
    /// encoder's 320-sample grid; the returned length is the harness's exact count.
    #[test]
    fn harness_tails_round_up_to_whole_units() {
        for want in [1_600 + 1, 1_600 + 481, 1_600 + 961, 2_560] {
            let g = Geometry::stream(&SIGIL_LOW, 1_600, want, 258, true).unwrap();
            assert_eq!(g.n16 % UNIT_16K, 0);
            assert_eq!(g.n16, (50 + want.div_ceil(320) + 1) * 320);
            assert_eq!(g.out_samples, out_len(want));
            assert!(g.out_start >= g.dec_start * HOP + HIFT_MARGIN * HOP || g.dec_start == 0);
        }
        assert!(Geometry::stream(&SIGIL_LOW, 1_280, 2_000, 258, true).is_err(), "the block is the preset's");
        assert!(Geometry::stream(&SIGIL_LOW, 1_600, 1_000, 258, true).is_err(), "want < block");
        assert!(Geometry::stream(&SIGIL_LOW, 1_600, 2_000, 0, true).is_err(), "no prompt");
        assert!(Geometry::stream(&SIGIL_LOW, 1_600, 2_000, MAX_SEQ, true).is_err(), "too long for the DiT");
    }

    #[test]
    fn offline_geometry_returns_exactly_the_new_frames() {
        for (len, new) in [(16_000usize, 16_000usize), (176_000, 176_000), (48_000, 32_000), (1_000, 1_000)] {
            let g = Geometry::offline(len, new, 258).unwrap();
            assert_eq!(g.out_samples, out_len(new));
            let (first, n) = g.returned();
            assert_eq!(n + g.pad_front, g.out_samples);
            assert!(first + n <= g.dec_samples());
            assert_eq!(g.n_mel, len * 22_050 / 16_000 / 256);
        }
        assert!(Geometry::offline(300, 300, 258).is_err());
    }

    #[test]
    fn helpers_and_settings() {
        assert_eq!(mel_frames(66_150), 258, "3 s prompt");
        assert_eq!(mel_frames(33_075), 129, "1.5 s prompt");
        assert_eq!(mel_frames(100), 0);
        assert_eq!(xlsr_frames(18_880), 58);
        assert_eq!(xlsr_frames(399), 0);
        assert_eq!(out_len(2_560), 3_528);
        assert_eq!(preset("sigil_lean"), SIGIL_LEAN);
        assert_eq!(preset("nope"), SIGIL_LOW);
        let o = SigilZsOptions { steps: Some(40), cfg_rate: Some(0.2), prompt_s: Some(0.1), fp16_dit: true, fp16_xlsr: true, ..Default::default() };
        let s = Settings::new(&o, PrecisionMode::Auto);
        assert_eq!((s.preset.steps, s.preset.cfg_rate, s.preset.prompt_s), (30, 0.2, MIN_PROMPT_S));
        assert!(s.fp16_dit && s.fp16_xlsr && !s.fp16_vocoder);
        let s = Settings::new(&o, PrecisionMode::Fp32);
        assert!(!s.fp16_dit && !s.fp16_xlsr && !s.fp16_vocoder, "fp32 wins over every switch");
        let s = Settings::new(&SigilZsOptions::default(), PrecisionMode::Fp16);
        assert!(s.fp16_vocoder && !s.fp16_dit && !s.fp16_xlsr, "fp16 asks only for the gated decoder");
        let s = Settings::new(&SigilZsOptions { fp16_vocoder: Some(true), ..Default::default() }, PrecisionMode::Auto);
        assert!(s.fp16_vocoder);
        assert_eq!(s.block_16k(), 1_600);
    }
}
