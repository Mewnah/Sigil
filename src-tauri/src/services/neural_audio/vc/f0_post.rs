//! Median smoother + confidence gate for F0 contours (hop-160 grid).

/// Replace octave-jump frames with their neighbours' mean, then median-smooth.
///
/// The median runs over the full window *including unvoiced (0.0) frames* and zero-pads both
/// ends, exactly like `scipy.signal.medfilt` in reference RVC: voiced runs keep their length, an
/// isolated spurious voiced frame is removed instead of spread to its neighbours, and the edge
/// frames (the newest one included) are smoothed like any other instead of passed through.
/// Non-finite or negative input frames are treated as unvoiced (0.0).
pub fn smooth_f0(f0: &mut [f32], window: usize) {
    if f0.is_empty() {
        return;
    }
    for v in f0.iter_mut() {
        if !(v.is_finite() && *v > 0.0) {
            *v = 0.0;
        }
    }
    let w = window.clamp(1, 5);
    // Confidence: a voiced frame more than ~an octave away from *every* voiced neighbour is an
    // octave error; replace it with the neighbours' mean. (Requiring disagreement with all
    // neighbours keeps a correct frame next to a spike from being pulled toward the spike.)
    let raw = f0.to_vec();
    let jumps = |a: f32, b: f32| a / b > 1.9 || b / a > 1.9;
    for i in 0..f0.len() {
        let cur = raw[i];
        if cur <= 0.0 {
            continue;
        }
        let prev = if i > 0 { raw[i - 1] } else { 0.0 };
        let next = raw.get(i + 1).copied().unwrap_or(0.0);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut all_jump = true;
        for nb in [prev, next] {
            if nb > 0.0 {
                sum += nb;
                count += 1;
                all_jump &= jumps(cur, nb);
            }
        }
        if count > 0 && all_jump {
            f0[i] = sum / count as f32;
        }
    }
    // Median over an odd window (zeros included), zero-padded past both ends (scipy medfilt).
    let half = w / 2;
    if half == 0 {
        return;
    }
    let src = f0.to_vec();
    let n = src.len();
    let mut win: Vec<f32> = Vec::with_capacity(2 * half + 1);
    for (i, out) in f0.iter_mut().enumerate() {
        win.clear();
        // Padded index `i + k` maps to source index `i + k - half`.
        for k in 0..=2 * half {
            let j = i + k;
            win.push(if j < half || j - half >= n { 0.0 } else { src[j - half] });
        }
        win.sort_by(f32::total_cmp);
        *out = win[half];
    }
}

#[cfg(test)]
mod tests {
    use super::smooth_f0;

    #[test]
    fn median_removes_isolated_voiced_frame() {
        let mut f0 = vec![0.0, 0.0, 200.0, 0.0, 0.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 5]);
    }

    #[test]
    fn median_keeps_voiced_run_length() {
        let mut f0 = vec![0.0, 0.0, 120.0, 121.0, 122.0, 0.0, 0.0];
        smooth_f0(&mut f0, 3);
        let voiced: Vec<bool> = f0.iter().map(|&v| v > 0.0).collect();
        assert_eq!(voiced, vec![false, false, true, true, true, false, false], "{f0:?}");
    }

    #[test]
    fn median_fills_single_dropout_inside_voiced_run() {
        let mut f0 = vec![110.0, 111.0, 0.0, 112.0, 113.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0[2], 111.0);
        // Edges are zero-padded like scipy medfilt: medfilt([110, 111, 0, 112, 113], 3) =
        // [110, 110, 111, 112, 112].
        assert_eq!(f0, vec![110.0, 110.0, 111.0, 112.0, 112.0]);
    }

    #[test]
    fn edge_frames_are_smoothed_like_scipy_medfilt() {
        // The newest frame is filtered too: an isolated voiced frame at the end is removed ...
        let mut f0 = vec![0.0, 0.0, 0.0, 0.0, 180.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 5]);
        // ... and a voiced end frame becomes the lower of itself and its inner neighbour
        // (median of the two and the zero pad).
        let mut f0 = vec![0.0, 150.0, 152.0, 151.0, 153.0];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0, 150.0, 151.0, 152.0, 151.0]);
        // Window 5: two zeros of padding at each end; a steady run still survives at the edges
        // (at most two of the five window values are pad zeros).
        let mut f0 = vec![100.0; 6];
        smooth_f0(&mut f0, 5);
        assert_eq!(f0, vec![100.0; 6]);
        let mut f0 = vec![100.0, 101.0, 102.0, 103.0];
        smooth_f0(&mut f0, 5);
        // [0,0,100,101,102] -> 100; [0,100,101,102,103] -> 101; [100,101,102,103,0] -> 101;
        // [101,102,103,0,0] -> 101.
        assert_eq!(f0, vec![100.0, 101.0, 101.0, 101.0]);
    }

    #[test]
    fn non_finite_frames_are_unvoiced() {
        let mut f0 = vec![120.0, f32::NAN, 121.0, f32::INFINITY, 122.0, -5.0, 123.0];
        smooth_f0(&mut f0, 3);
        assert!(f0.iter().all(|v| v.is_finite() && *v >= 0.0), "{f0:?}");
        let mut f0 = vec![f32::NAN; 4];
        smooth_f0(&mut f0, 3);
        assert_eq!(f0, vec![0.0; 4]);
    }

    #[test]
    fn octave_jump_is_replaced_before_median() {
        let mut f0 = vec![100.0, 100.0, 400.0, 100.0, 100.0];
        smooth_f0(&mut f0, 3);
        assert!(f0.iter().all(|&v| (v - 100.0).abs() < 1e-3), "{f0:?}");
    }
}
