//! Weights writer: checkpoint tensors -> the template's `weights.f32.bin` image (and its f16
//! twin), plus the speaker table.
//!
//! Byte-identical to the exporter's Python reference writer (`sigil_layout.write_weights`) for the
//! same checkpoint; `parity_test.py --rust-weights` and the tiny-fixture test check that.

use std::io::{Read, Seek};

use super::fold::weight_norm_fold;
use super::layout::{Layout, TensorEntry};
use super::pth::TorchCheckpoint;
use super::{ErrorCode, ImportError};

/// Everything the engine needs from a checkpoint besides the template.
pub struct ImportedWeights {
    /// `weights.f32.bin`: every template weight at its layout offset (little-endian f32).
    pub f32_image: Vec<u8>,
    /// Speaker table `[speakers, speaker_dim]` row-major (`emb_g.weight`): the engine computes the
    /// speaker vector `g` (and speaker blends) from it.
    pub speakers: Vec<f32>,
    pub n_speakers: usize,
    pub speaker_dim: usize,
}

fn shape_mismatch(e: &TensorEntry, key: &str, got: &[usize], want: &[usize]) -> ImportError {
    ImportError::new(
        ErrorCode::ShapeMismatch,
        format!("{key} is {got:?}, the template's {} needs {want:?}", e.name),
    )
}

/// The f32 value of one layout tensor.
fn tensor_value<R: Read + Seek>(e: &TensorEntry, ckpt: &mut TorchCheckpoint<R>) -> Result<Vec<f32>, ImportError> {
    let check = |ckpt: &TorchCheckpoint<R>, key: &str, want: &[usize]| -> Result<(), ImportError> {
        match ckpt.shape(key) {
            Some(got) if got == want => Ok(()),
            Some(got) => Err(shape_mismatch(e, key, &got, want)),
            None => Err(ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {key}"))),
        }
    };
    match (&e.source.key, &e.source.weight_norm) {
        (Some(key), _) => {
            check(ckpt, key, &e.shape)?;
            ckpt.read_f32(key)
        }
        (_, Some(wn)) => {
            if ckpt.has(&wn.g) && ckpt.has(&wn.v) {
                check(ckpt, &wn.v, &e.shape)?;
                check(ckpt, &wn.g, &wn.g_shape)?;
                let g = ckpt.read_f32(&wn.g)?;
                let v = ckpt.read_f32(&wn.v)?;
                weight_norm_fold(&g, &v, &e.shape, wn.dim)
                    .map_err(|m| ImportError::new(ErrorCode::NonFinite, format!("{}: {m}", e.name)))
            } else if ckpt.has(&wn.folded) {
                check(ckpt, &wn.folded, &e.shape)?;
                ckpt.read_f32(&wn.folded)
            } else {
                Err(ImportError::new(
                    ErrorCode::MissingTensor,
                    format!("the checkpoint has neither {} / {} nor {}", wn.g, wn.v, wn.folded),
                ))
            }
        }
        (None, None) => Err(ImportError::new(ErrorCode::LayoutInvalid, format!("{} has no source", e.name))),
    }
}

/// Build the weights image and the speaker table for `layout` from `ckpt`. Every checkpoint tensor
/// must be used by the layout (or be on its ignore list): a leftover means an architecture the
/// template does not model, which would otherwise play back as a broken voice.
pub fn build<R: Read + Seek>(layout: &Layout, ckpt: &mut TorchCheckpoint<R>) -> Result<ImportedWeights, ImportError> {
    let leftovers: Vec<String> = ckpt.tensor_names().filter(|k| !layout.uses_key(k)).map(str::to_string).collect();
    if !leftovers.is_empty() {
        let shown: Vec<&str> = leftovers.iter().take(6).map(String::as_str).collect();
        return Err(ImportError::new(
            ErrorCode::UnsupportedModel,
            format!(
                "{} checkpoint tensors are not part of the {} architecture ({}{})",
                leftovers.len(),
                layout.template,
                shown.join(", "),
                if leftovers.len() > shown.len() { ", ..." } else { "" }
            ),
        ));
    }
    let mut image = vec![0u8; layout.weights.bytes_f32 as usize];
    for e in &layout.weights.tensors {
        let values = tensor_value(e, ckpt)?;
        if let Some(bad) = values.iter().position(|x| !x.is_finite()) {
            return Err(ImportError::new(ErrorCode::NonFinite, format!("{} holds a non-finite value at {bad}", e.name)));
        }
        let dst = &mut image[e.offset as usize..(e.offset + e.bytes) as usize];
        for (d, v) in dst.chunks_exact_mut(4).zip(&values) {
            d.copy_from_slice(&v.to_le_bytes());
        }
    }
    let spk = &layout.speakers;
    let (n, dim) = match ckpt.shape(&spk.key).as_deref() {
        Some([n, d]) if *n > 0 && *d == spk.dim => (*n, *d),
        Some(other) => {
            return Err(ImportError::new(
                ErrorCode::ShapeMismatch,
                format!("{} is {other:?}, expected [speakers, {}]", spk.key, spk.dim),
            ))
        }
        None => return Err(ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {}", spk.key))),
    };
    let speakers = ckpt.read_f32(&spk.key)?;
    if speakers.iter().any(|x| !x.is_finite()) {
        return Err(ImportError::new(ErrorCode::NonFinite, "the speaker table holds non-finite values"));
    }
    Ok(ImportedWeights { f32_image: image, speakers, n_speakers: n, speaker_dim: dim })
}

/// `weights.f16.bin`: every f32 word cast to f16 (so every layout offset halves). `Err` when a
/// finite weight overflows f16, the voice then runs only with the fp32 graphs.
///
/// P3B-H: an overflow scan first (a finite weight at or past 65520 rounds to infinity), then the
/// conversion 8 words at a time with the F16C instructions where the CPU has them (round to nearest
/// even, the same bits as `half`'s scalar conversion), else `half`'s scalar loop. The 220 MB image
/// of a 40 kHz voice converts in a fraction of the 70 ms the per-word loop took.
pub fn f16_image(f32_image: &[u8]) -> Result<Vec<u8>, String> {
    let overflow = count_f16_overflows(f32_image);
    if overflow > 0 {
        return Err(format!("{overflow} weights exceed the fp16 range"));
    }
    let words = f32_image.len() / 4;
    // A zeroed allocation: fresh pages, no memset.
    let mut out = vec![0u8; words * 2];
    convert_f32_to_f16(&f32_image[..words * 4], &mut out);
    Ok(out)
}

/// Finite f32 words (little-endian) whose f16 rounding is infinity: magnitude from 65520 up to the
/// largest finite value.
fn count_f16_overflows(f32_image: &[u8]) -> usize {
    const LIMIT: u32 = 0x477F_F000; // 65520.0
    const INF: u32 = 0x7F80_0000;
    f32_image
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0x7FFF_FFFF)
        .filter(|&b| (LIMIT..INF).contains(&b))
        .count()
}

/// `half`'s conversion of every word, in little-endian bytes.
fn convert_f16_scalar(f32_image: &[u8], out: &mut [u8]) {
    for (c, o) in f32_image.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        o.copy_from_slice(&half::f16::from_f32(x).to_bits().to_le_bytes());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f16_f16c(f32_image: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::{__m128i, _mm256_cvtps_ph, _mm256_loadu_ps, _mm_storeu_si128, _MM_FROUND_TO_NEAREST_INT};
    let words = f32_image.len() / 4;
    let blocks = words / 8;
    let src = f32_image.as_ptr();
    let dst = out.as_mut_ptr();
    for b in 0..blocks {
        // SAFETY: block `b` reads 32 bytes at 32 * b < f32_image.len() and writes 16 bytes at 16 * b
        // < out.len() (`out` holds 2 bytes per word); the loads and stores are unaligned.
        let v = _mm256_loadu_ps(src.add(b * 32) as *const f32);
        let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(v);
        _mm_storeu_si128(dst.add(b * 16) as *mut __m128i, h);
    }
    // The tail (fewer than 8 words) in scalar.
    convert_f16_scalar(&f32_image[blocks * 32..], &mut out[blocks * 16..]);
}

fn convert_f32_to_f16(f32_image: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: the CPU features were just detected; `out` holds 2 bytes per f32 word.
            unsafe { convert_f16_f16c(f32_image, out) };
            return;
        }
    }
    convert_f16_scalar(f32_image, out);
}

/// Little-endian bytes of an f32 slice.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_image_halves_and_rounds() {
        let img = f32_bytes(&[1.0, -2.5, 0.1, 0.0]);
        let h = f16_image(&img).unwrap();
        assert_eq!(h.len(), 8);
        let back: Vec<f32> = h.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
        assert_eq!(back[..2], [1.0, -2.5]);
        assert!((back[2] - 0.1).abs() < 1e-4);
        assert!(f16_image(&f32_bytes(&[1.0e6])).is_err());
    }

    /// A tiny deterministic generator of f32 bit patterns (no dependency).
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 32) as u32
    }

    /// The SIMD conversion gives `half`'s bits for every finite value (all exponents, subnormals,
    /// both zeros, the largest f16 and its rounding edge, odd lengths for the scalar tail), and the
    /// overflow scan matches the scalar rule exactly.
    #[test]
    fn f16_image_matches_the_scalar_conversion() {
        let mut seed = 0x5eed_u64;
        let mut values: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 65519.99, 6.0e-8, 5.9e-8, 3.0e-8, 1.0e-30, f32::MIN_POSITIVE, 0.1, 1.0 / 3.0];
        while values.len() < 100_003 {
            let bits = lcg(&mut seed);
            let x = f32::from_bits(bits);
            // Finite and inside f16's range, plus the whole exponent spread.
            if x.is_finite() && x.abs() < 65_520.0 {
                values.push(x);
            }
        }
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 1_000, 100_003] {
            let img = f32_bytes(&values[..n]);
            let got = f16_image(&img).expect("in range");
            let mut want = vec![0u8; n * 2];
            convert_f16_scalar(&img, &mut want);
            assert_eq!(got.len(), n * 2);
            assert_eq!(got, want, "{n} words");
        }
        // The overflow rule: 65520 and up (finite) overflow, infinities and NaN are not counted.
        for (x, over) in [(65_504.0f32, 0usize), (65_519.0, 0), (65_520.0, 1), (-65_520.0, 1), (3.0e38, 1), (f32::INFINITY, 0), (f32::NAN, 0)] {
            let img = f32_bytes(&[0.5, x, 0.25]);
            assert_eq!(count_f16_overflows(&img), over, "{x}");
            let scalar = [0.5f32, x, 0.25].iter().filter(|v| v.is_finite() && !half::f16::from_f32(**v).is_finite()).count();
            assert_eq!(over, scalar, "{x}: the scan equals the scalar overflow test");
        }
        // Non-finite weights convert like `half` (the importer rejects them earlier).
        let img = f32_bytes(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        let h = f16_image(&img).expect("not an overflow");
        let back: Vec<f32> = h.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
        assert_eq!((back[0], back[1]), (f32::INFINITY, f32::NEG_INFINITY));
        assert!(back[2].is_nan());
    }
}
