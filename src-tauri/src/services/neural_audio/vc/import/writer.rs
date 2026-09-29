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
pub fn f16_image(f32_image: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(f32_image.len() / 2);
    let mut overflow = 0usize;
    for c in f32_image.chunks_exact(4) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let h = half::f16::from_f32(x);
        if x.is_finite() && !h.is_finite() {
            overflow += 1;
        }
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    if overflow > 0 {
        return Err(format!("{overflow} weights exceed the fp16 range"));
    }
    Ok(out)
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
}
