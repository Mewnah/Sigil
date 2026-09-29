//! weight_norm folding: `w = g * v / ||v||` (PyTorch `torch._weight_norm(v, g, dim)`).
//!
//! RVC keeps `weight_g` / `weight_v` in its checkpoints and recomputes the weight on every forward;
//! the templates hold plain weights, so the importer folds once, in f64, rounding to f32 at the end
//! (the result matches PyTorch's own fp32 fold to the last ulp or so).

/// Fold one weight. `dim` is the axis `g` keeps (its size in `v`), or -1 for a whole-tensor norm;
/// `g` has one value per index of `dim` (a single value for -1).
pub fn weight_norm_fold(g: &[f32], v: &[f32], shape: &[usize], dim: i64) -> Result<Vec<f32>, String> {
    let numel: usize = shape.iter().product();
    if v.len() != numel || numel == 0 {
        return Err(format!("weight_v has {} values for shape {shape:?}", v.len()));
    }
    if dim == -1 {
        if g.len() != 1 {
            return Err(format!("weight_g has {} values, expected 1", g.len()));
        }
        let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let scale = g[0] as f64 / norm;
        return finite(v.iter().map(|&x| (x as f64 * scale) as f32).collect());
    }
    let dim = usize::try_from(dim).ok().filter(|&d| d < shape.len()).ok_or_else(|| format!("bad dim {dim}"))?;
    let n = shape[dim];
    if g.len() != n {
        return Err(format!("weight_g has {} values, expected {n}", g.len()));
    }
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    // Sequential sum within each contiguous block, then over the leading axes (the Python
    // reference, sigil_layout.fold_weight_norm, uses the same order: identical f32 results).
    let mut sq = vec![0f64; n];
    for o in 0..outer {
        for (i, acc) in sq.iter_mut().enumerate() {
            let base = (o * n + i) * inner;
            *acc += v[base..base + inner].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        }
    }
    let scale: Vec<f64> = sq.iter().zip(g).map(|(&s, &gi)| gi as f64 / s.sqrt()).collect();
    let mut w = vec![0f32; numel];
    for (block, (dst, src)) in w.chunks_exact_mut(inner).zip(v.chunks_exact(inner)).enumerate() {
        let s = scale[block % n];
        for (d, &x) in dst.iter_mut().zip(src) {
            *d = (x as f64 * s) as f32;
        }
    }
    finite(w)
}

fn finite(w: Vec<f32>) -> Result<Vec<f32>, String> {
    if w.iter().all(|x| x.is_finite()) {
        Ok(w)
    } else {
        Err("folding produced a non-finite weight (an all-zero weight_v?)".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim0_matches_hand_computation() {
        // v: [2, 2, 1]; rows [3, 4] and [0, 2]; g = [10, -1]
        let w = weight_norm_fold(&[10.0, -1.0], &[3.0, 4.0, 0.0, 2.0], &[2, 2, 1], 0).unwrap();
        assert_eq!(w, vec![6.0, 8.0, 0.0, -1.0]);
    }

    #[test]
    fn middle_dim_and_whole_tensor() {
        // shape [2, 2]: norms over axis 0 for each column.
        let v = [3.0, 1.0, 4.0, 1.0];
        let w = weight_norm_fold(&[5.0, 2.0f32.sqrt()], &v, &[2, 2], 1).unwrap();
        let expect = [3.0f32, 1.0, 4.0, 1.0];
        for (a, b) in w.iter().zip(expect) {
            assert!((a - b).abs() < 1e-6, "{w:?}");
        }
        let w = weight_norm_fold(&[2.0], &[3.0, 4.0], &[2], -1).unwrap();
        assert_eq!(w, vec![1.2, 1.6]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(weight_norm_fold(&[1.0], &[0.0, 0.0], &[1, 2], 0).is_err(), "zero norm");
        assert!(weight_norm_fold(&[1.0, 1.0], &[1.0, 2.0], &[1, 2], 0).is_err(), "g size");
        assert!(weight_norm_fold(&[1.0], &[1.0, 2.0], &[1, 2], 2).is_err(), "dim");
        assert!(weight_norm_fold(&[1.0], &[1.0], &[1, 2], 0).is_err(), "v size");
    }
}
