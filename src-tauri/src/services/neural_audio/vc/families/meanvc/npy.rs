//! A strict reader for small NumPy `.npy` vectors: MeanVC2's pre-extracted speaker embeddings
//! (upstream `preprocess/extract_spk_emb.py` saves the raw 256-value ECAPA-TDNN output with
//! `np.save`).
//!
//! Only what such a file holds is accepted: format version 1, 2 or 3, a little-endian `f4` or `f8`
//! array (`<f4`, `<f8`; `|`-prefixed is not a float), any `fortran_order` (a vector reads the same
//! either way), and a shape whose dimensions multiply to the expected length with at most one
//! dimension above 1. Every size is checked before anything is allocated; the file is user
//! supplied (it sits in a voice folder) and is never trusted.

use std::path::Path;

/// Largest `.npy` file read (a 256-value f8 vector is 2 KiB plus a header).
pub const MAX_NPY_BYTES: u64 = 1 << 20;
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Read `path` as a float vector of exactly `len` values (as f32).
pub fn read_vector(path: &Path, len: usize) -> Result<Vec<f32>, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .len();
    if size > MAX_NPY_BYTES {
        return Err(format!("{} is {size} bytes, too large for a speaker embedding", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_vector(&bytes, len).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse `.npy` bytes as a float vector of exactly `len` values (as f32).
pub fn parse_vector(bytes: &[u8], len: usize) -> Result<Vec<f32>, String> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a NumPy .npy file".into());
    }
    let major = bytes[6];
    let (header_len, start): (usize, usize) = match major {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated .npy header".into());
            }
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => return Err(format!(".npy format version {v} is not supported")),
    };
    let end = start
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("truncated .npy header")?;
    let header = std::str::from_utf8(&bytes[start..end]).map_err(|_| ".npy header is not text")?;
    let descr = dict_value(header, "descr").ok_or(".npy header has no 'descr'")?;
    let width = match descr.trim().trim_matches(|c| c == '\'' || c == '"') {
        "<f4" => 4,
        "<f8" => 8,
        other => return Err(format!(".npy dtype {other} is not a little-endian float32 / float64")),
    };
    let shape_text = dict_value(header, "shape").ok_or(".npy header has no 'shape'")?;
    let shape = parse_shape(shape_text)?;
    let count = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(".npy shape overflows")?;
    if count != len || shape.iter().filter(|&&d| d > 1).count() > 1 {
        return Err(format!("expected a vector of {len} values, the file holds shape {shape:?}"));
    }
    let data = &bytes[end..];
    if data.len() != count * width {
        return Err(format!("the .npy data is {} bytes, the header says {}", data.len(), count * width));
    }
    let values: Vec<f32> = if width == 4 {
        data.as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        data.as_chunks::<8>().0.iter()
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err("the .npy vector holds NaN or infinite values".into());
    }
    Ok(values)
}

/// The raw text of `key`'s value in the header's Python dict literal (up to the next top-level
/// comma or the closing brace; a tuple value keeps its parentheses).
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let at = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))?;
    let rest = &header[at + key.len() + 2..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' | '}' if depth == 0 => return Some(rest[..i].trim()),
            _ => {}
        }
    }
    None
}

/// A Python tuple of non-negative integers: `(256,)`, `(1, 256)`, `()`.
fn parse_shape(text: &str) -> Result<Vec<usize>, String> {
    let inner = text
        .trim()
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .ok_or_else(|| format!(".npy shape {text} is not a tuple"))?;
    let mut dims = Vec::new();
    for part in inner.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let d: usize = part
            .trim_end_matches('L')
            .parse()
            .map_err(|_| format!(".npy shape {text} is not a tuple of integers"))?;
        dims.push(d);
    }
    if dims.len() > 4 {
        return Err(format!(".npy shape {text} has too many dimensions"));
    }
    Ok(dims)
}

#[cfg(test)]
pub(crate) fn encode(values: &[f32], shape: &str, descr: &str, major: u8) -> Vec<u8> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let lead = if major == 1 { 10 } else { 12 };
    while (lead + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&[major, 0]);
    if major == 1 {
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(header.as_bytes());
    for v in values {
        if descr == "<f8" {
            out.extend_from_slice(&f64::from(*v).to_le_bytes());
        } else {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.25 - 3.0).collect()
    }

    #[test]
    fn numpy_vectors_of_every_accepted_layout_read_back_exactly() {
        let v = ramp(256);
        for (shape, descr, major) in [
            ("(256,)", "<f4", 1),
            ("(1, 256)", "<f4", 1),
            ("(256, 1)", "<f8", 2),
            ("(1, 1, 256)", "<f4", 3),
        ] {
            let back = parse_vector(&encode(&v, shape, descr, major), 256).unwrap_or_else(|e| panic!("{shape} {descr} v{major}: {e}"));
            assert_eq!(back, v, "{shape} {descr} v{major}");
        }
    }

    #[test]
    fn the_header_written_by_numpy_parses() {
        // np.save of a (256,) float32 vector, header as NumPy 1.26 writes it.
        let v = ramp(256);
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&[1, 0]);
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (256,), }                                                          \n";
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        assert_eq!(parse_vector(&bytes, 256).unwrap(), v);
    }

    #[test]
    fn wrong_files_are_refused_before_allocating() {
        let v = ramp(256);
        let good = encode(&v, "(256,)", "<f4", 1);
        assert!(parse_vector(&good, 192).unwrap_err().contains("256"), "wrong length");
        assert!(
            parse_vector(&encode(&ramp(512), "(2, 256)", "<f4", 1), 512).is_err(),
            "a matrix, not a vector"
        );
        assert!(parse_vector(&encode(&v, "(256,)", ">f4", 1), 256).is_err(), "big-endian");
        assert!(parse_vector(&encode(&v, "(256,)", "<i4", 1), 256).is_err(), "integers");
        assert!(parse_vector(&good[..good.len() - 1], 256).is_err(), "truncated data");
        let mut extra = good.clone();
        extra.push(0);
        assert!(parse_vector(&extra, 256).is_err(), "trailing bytes");
        assert!(parse_vector(b"\x93NUMPY\x09\x00\x00\x00", 256).is_err(), "unknown version");
        assert!(parse_vector(b"PK\x03\x04 not numpy", 256).is_err());
        // A header that claims a huge shape is refused by arithmetic, not by allocation.
        let huge = encode(&[], "(100000000000, 100000000000)", "<f4", 1);
        assert!(parse_vector(&huge, 256).is_err());
        let mut nan = v.clone();
        nan[3] = f32::NAN;
        assert!(parse_vector(&encode(&nan, "(256,)", "<f4", 1), 256)
            .unwrap_err()
            .contains("NaN"));
        // Header length beyond the file.
        let mut lying = good.clone();
        lying[8] = 0xff;
        lying[9] = 0xff;
        assert!(parse_vector(&lying, 256).is_err());
    }
}
