//! Picking the next speech token from the language model's logits, as the reference
//! implementation's `T3.inference` does (Hugging Face logits processors, in this order):
//!
//! 1. Classifier-free guidance: `cond + cfg · (cond − uncond)` (row 0 sees the text, row 1 does not).
//! 2. Repetition penalty over every token generated so far (the speech-start token included):
//!    positive logits are divided by it, negative ones multiplied.
//! 3. Temperature.
//! 4. Min-p: drop tokens whose probability is below `min_p` × the top probability.
//! 5. Top-p (nucleus), when below 1.
//! 6. Sample from the softmax (a deterministic generator, seedable).

/// Sampling settings (Chatterbox Multilingual's defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub cfg_weight: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            cfg_weight: 0.5,
            temperature: 0.8,
            repetition_penalty: 2.0,
            min_p: 0.05,
            top_p: 1.0,
        }
    }
}

/// A small, fast, seedable generator (SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// A seed from the clock and a process counter (a new one per call).
    pub fn fresh() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self::new(t ^ COUNTER.fetch_add(0x632B_E59B_D9B4_E019, Ordering::Relaxed))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Steps 1-6 of the module docs. `logits` holds `rows` rows of `vocab` values (row 0 conditional,
/// row 1 unconditional when there are two); `scratch` is reused between calls.
pub fn next_token(
    logits: &[f32],
    rows: usize,
    vocab: usize,
    generated: &[i64],
    params: &SamplingParams,
    rng: &mut Rng,
    scratch: &mut Vec<f32>,
) -> Result<i64, String> {
    if vocab == 0 || rows == 0 || logits.len() < rows * vocab {
        return Err(format!("the language model returned {} logits for {rows} x {vocab}", logits.len()));
    }
    scratch.clear();
    scratch.extend_from_slice(&logits[..vocab]);
    if rows >= 2 && params.cfg_weight != 0.0 {
        let uncond = &logits[vocab..2 * vocab];
        for (c, u) in scratch.iter_mut().zip(uncond) {
            *c += params.cfg_weight * (*c - u);
        }
    }
    let l = scratch.as_mut_slice();
    for v in l.iter_mut() {
        if !v.is_finite() {
            *v = if *v == f32::INFINITY { f32::MAX } else { f32::NEG_INFINITY };
        }
    }
    // 2. Repetition penalty (each distinct token once).
    let penalty = params.repetition_penalty;
    if penalty.is_finite() && penalty > 0.0 && penalty != 1.0 {
        let mut seen: Vec<usize> = generated
            .iter()
            .filter_map(|&t| usize::try_from(t).ok())
            .filter(|&t| t < vocab)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            let v = &mut l[t];
            *v = if *v < 0.0 { *v * penalty } else { *v / penalty };
        }
    }
    // Greedy for a non-positive temperature.
    if !(params.temperature.is_finite() && params.temperature > 0.0) {
        return Ok(argmax(l) as i64);
    }
    // 3. Temperature.
    if params.temperature != 1.0 {
        let inv = 1.0 / params.temperature;
        for v in l.iter_mut() {
            *v *= inv;
        }
    }
    // Probabilities (softmax, max-subtracted).
    let top = argmax(l);
    let max = l[top];
    if max == f32::NEG_INFINITY {
        return Ok(top as i64);
    }
    let mut probs: Vec<f64> = l.iter().map(|&v| f64::from(v - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // 4. Min-p (the top token always stays).
    if params.min_p > 0.0 && params.min_p.is_finite() {
        let threshold = f64::from(params.min_p) * probs[top];
        for (i, p) in probs.iter_mut().enumerate() {
            if i != top && *p < threshold {
                *p = 0.0;
            }
        }
    }
    // 5. Top-p: keep the smallest set of most likely tokens reaching `top_p` (at least one).
    if params.top_p < 1.0 && params.top_p.is_finite() {
        let mut order: Vec<usize> = (0..vocab).filter(|&i| probs[i] > 0.0).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total: f64 = order.iter().map(|&i| probs[i]).sum();
        let target = f64::from(params.top_p.max(0.0)) * total;
        let mut acc = 0.0;
        let mut keep = 0;
        for &i in &order {
            keep += 1;
            acc += probs[i];
            if acc >= target {
                break;
            }
        }
        for &i in order.iter().skip(keep.max(1)) {
            probs[i] = 0.0;
        }
    }
    // 6. Sample.
    let total: f64 = probs.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return Ok(top as i64);
    }
    let mut x = rng.next_f64() * total;
    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        if x < p {
            return Ok(i as i64);
        }
        x -= p;
    }
    // Rounding at the very end: the last token with any probability.
    Ok(probs.iter().rposition(|&p| p > 0.0).unwrap_or(top) as i64)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
