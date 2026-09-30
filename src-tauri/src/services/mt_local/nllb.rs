//! NLLB-200 distilled (600M / 1.3B, 8-bit) on Sigil's ONNX Runtime: Optimum's encoder plus merged decoder
//! (`decoder_model_merged*.onnx`: one graph with and without the KV cache, switched by `use_cache_branch`), greedy or
//! beam search, the `tokenizers` crate over `tokenizer.json` ([`super::tokenizer`], parity-checked against Python
//! sentencepiece), FLORES-200 language tokens ([`super::flores`]), CPU background profile.
//!
//! # Decoding
//! The source is `[src_lang] + pieces + [</s>]`; the decoder starts from `[</s>, tgt_lang]` (NLLB's
//! `decoder_start_token_id` plus the forced target-language token) in one first step without cache, which also
//! yields the encoder's cross-attention keys and values. Later steps feed one token per row with the growing
//! decoder cache and the first step's encoder cache (the merged graph's encoder outputs are placeholders in the
//! cached branch, as in Optimum and transformers.js). Special tokens (padding, `<s>`, language tokens, `<mask>`)
//! are never generated.
//! - Greedy (`beam` 1, always for interims) or beam search up to 4 beams (finals; length penalty 1.0, Hugging
//!   Face's stopping rule: done when `beam` hypotheses finished and no running beam can beat the worst of them).
//! - Length cap `3 × source pieces + 10` (at most 384 tokens), plus a runaway guard: once the output is longer than
//!   the source and ends in the same 1–8 token n-gram three times, decoding stops and keeps one copy.
//! - Multi-sentence lines are translated sentence by sentence ([`split_sentences`]; NLLB is trained on single
//!   sentences and tends to drop the second one otherwise) and joined with a space (none for Japanese, Chinese,
//!   Thai, ...). The stray spaces NLLB writes between Chinese or Japanese characters are removed ([`tidy_output`]).
//! - Unknown source language (`source: ""`): a script-based guess ([`flores::guess_flores`]); text without letters
//!   is returned unchanged, and so is a line whose explicit source equals the target.
//! - The Rust loop reproduces `scripts/mt-bench/nllb_reference.py` (Python onnxruntime) token for token on the
//!   fixture `scripts/mt-bench/fixtures/nllb_parity.json` (ignored test `real_model_matches_the_python_reference`).
//!
//! # Runtime profile
//! Sessions are built only through [`ort_session::build_session`] on the CPU EP (component `mt.nllb`, spinning
//! off, the CPU arena on for the growing shapes) with `threads` intra-op threads (default 2, max 4). All inference
//! runs on one dedicated worker thread at below-normal priority (its ONNX Runtime pool threads too), so a game keeps
//! the CPU; requests are served in order, and a cancelled request stops at the next decoding step.
//!
//! # Commands (frozen by P4-S0, see `mt_local/mod.rs`)
//! `nllb_load`, `nllb_translate`, `nllb_cancel`, `nllb_unload`, `nllb_status`; status changes are emitted as
//! [`super::STATUS_EVENT`]. Models come only from the model store (`mt/nllb-*` catalog ids,
//! `model-catalog/mt-nllb.json`); a missing model fails with `not_installed` naming the id, and nothing downloads.
//!
//! Owned by P4-D.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use tauri::{command, AppHandle, Emitter, Runtime};

use super::flores;
use super::tokenizer::NllbTokenizer;
use super::{
    thread_count, MtEngineState, MtEngineStatus, MtError, MtErrorKind, MtKind, MtRequest, MtResponse, ENGINE_NLLB,
    NLLB_DEFAULT_MODEL, STATUS_EVENT,
};
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions, EP_CPU};
use crate::services::rt_thread::{self, WorkerPriority};

const LOG_TARGET: &str = "sigil::mt";
/// Diagnostics component of the NLLB sessions (Settings → GPU & diagnostics).
const EP_COMPONENT: &str = "mt.nllb";
/// Catalog ids this engine accepts (`mt/nllb-200-distilled-600m-int8`, `mt/nllb-200-distilled-1.3b-int8`, ...).
const MODEL_ID_PREFIX: &str = "mt/nllb-";
/// Longest accepted request (characters); captions are far shorter.
const MAX_TEXT_CHARS: usize = 5_000;
/// Longest source segment in tokens (NLLB was trained on sentences up to 512 tokens).
const MAX_SOURCE_TOKENS: usize = 400;
/// Upper bound of generated tokens per segment.
const MAX_NEW_TOKENS: usize = 384;
/// Largest beam.
pub const MAX_BEAM: usize = 4;

/// Encoder file names, best first (Optimum / transformers.js exports).
const ENCODER_NAMES: &[&str] = &[
    "encoder_model_quantized.onnx",
    "encoder_model_int8.onnx",
    "encoder_model_uint8.onnx",
    "encoder_model.onnx",
];
/// Merged decoder file names, best first.
const DECODER_NAMES: &[&str] = &[
    "decoder_model_merged_quantized.onnx",
    "decoder_model_merged_int8.onnx",
    "decoder_model_merged_uint8.onnx",
    "decoder_model_merged.onnx",
];

fn rt_err(context: &str, e: impl std::fmt::Display) -> MtError {
    MtError::runtime(format!("{context}: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Model files and config
// ---------------------------------------------------------------------------------------------

/// The files of one NLLB folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFiles {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokenizer: PathBuf,
    pub config: Option<PathBuf>,
}

/// Find the encoder, merged decoder, `tokenizer.json` and `config.json` in `dir` (or its `onnx/` subfolder).
pub fn find_model_files(dir: &Path) -> Result<ModelFiles, String> {
    let find = |names: &[&str]| -> Option<PathBuf> {
        for base in [dir.to_path_buf(), dir.join("onnx")] {
            for n in names {
                let p = base.join(n);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        None
    };
    let encoder = find(ENCODER_NAMES).ok_or("the NLLB folder has no encoder (encoder_model*.onnx)")?;
    let decoder = find(DECODER_NAMES).ok_or("the NLLB folder has no merged decoder (decoder_model_merged*.onnx)")?;
    let tokenizer = find(&["tokenizer.json"]).ok_or("the NLLB folder has no tokenizer.json")?;
    Ok(ModelFiles {
        encoder,
        decoder,
        tokenizer,
        config: find(&["config.json"]),
    })
}

/// The `config.json` values the decoder needs (NLLB defaults when the file is missing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub decoder_start: i64,
    pub max_positions: usize,
    pub heads: Option<usize>,
    pub head_dim: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            decoder_start: 2,
            max_positions: 1024,
            heads: None,
            head_dim: None,
        }
    }
}

impl ModelConfig {
    fn parse(text: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("config.json: {e}"))?;
        let int = |k: &str| v.get(k).and_then(serde_json::Value::as_i64);
        let heads = int("decoder_attention_heads").filter(|&h| h > 0).map(|h| h as usize);
        let head_dim = match (int("d_model"), heads) {
            (Some(d), Some(h)) if d > 0 && (d as usize).is_multiple_of(h) => Some(d as usize / h),
            _ => None,
        };
        let d = Self::default();
        Ok(Self {
            decoder_start: int("decoder_start_token_id").unwrap_or(d.decoder_start),
            max_positions: int("max_position_embeddings")
                .filter(|&p| p > 8)
                .map(|p| p as usize)
                .unwrap_or(d.max_positions),
            heads,
            head_dim,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Session layouts
// ---------------------------------------------------------------------------------------------

/// Names of the encoder's inputs and output.
#[derive(Clone, Debug)]
struct EncoderLayout {
    input_ids: String,
    attention_mask: Option<String>,
    output: String,
}

fn parse_encoder_layout(session: &Session) -> Result<EncoderLayout, String> {
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if !inputs.contains(&"input_ids") {
        return Err(format!("unexpected NLLB encoder inputs {inputs:?} (expected input_ids, attention_mask)"));
    }
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    let output = if outputs.contains(&"last_hidden_state") {
        "last_hidden_state"
    } else {
        outputs.first().copied().ok_or("the NLLB encoder has no output")?
    };
    Ok(EncoderLayout {
        input_ids: "input_ids".into(),
        attention_mask: inputs.contains(&"attention_mask").then(|| "attention_mask".to_string()),
        output: output.to_string(),
    })
}

/// Input and output names of one decoder layer's cache: decoder key, decoder value, encoder key, encoder value.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LayerIo {
    inputs: [String; 4],
    outputs: [String; 4],
}

/// Names and cache geometry of the merged decoder.
#[derive(Clone, Debug)]
struct DecoderLayout {
    input_ids: String,
    hidden: String,
    mask: Option<String>,
    use_cache: String,
    logits: String,
    layers: Vec<LayerIo>,
    heads: usize,
    head_dim: usize,
}

const KV_PARTS: [&str; 4] = ["decoder.key", "decoder.value", "encoder.key", "encoder.value"];

fn parse_decoder_layout(session: &Session, config: &ModelConfig) -> Result<DecoderLayout, String> {
    let inputs: HashMap<&str, &ValueType> = session.inputs().iter().map(|i| (i.name(), i.dtype())).collect();
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    for required in ["input_ids", "encoder_hidden_states"] {
        if !inputs.contains_key(required) {
            return Err(format!("the NLLB decoder has no {required} input"));
        }
    }
    if !inputs.contains_key("use_cache_branch") {
        return Err("the NLLB decoder is not a merged decoder (no use_cache_branch input); use decoder_model_merged*.onnx".into());
    }
    if !outputs.contains(&"logits") {
        return Err("the NLLB decoder has no logits output".into());
    }
    let mut layers = Vec::new();
    loop {
        let i = layers.len();
        let names = KV_PARTS.map(|p| format!("past_key_values.{i}.{p}"));
        if !inputs.contains_key(names[0].as_str()) {
            break;
        }
        let outs = KV_PARTS.map(|p| format!("present.{i}.{p}"));
        for (input, output) in names.iter().zip(outs.iter()) {
            if !inputs.contains_key(input.as_str()) {
                return Err(format!("the NLLB decoder has no {input} input"));
            }
            if !outputs.contains(&output.as_str()) {
                return Err(format!("the NLLB decoder has no {output} output"));
            }
        }
        layers.push(LayerIo {
            inputs: names,
            outputs: outs,
        });
    }
    if layers.is_empty() {
        return Err("the NLLB decoder has no KV cache inputs (past_key_values.*)".into());
    }
    let (heads, head_dim) = match inputs.get(layers[0].inputs[0].as_str()) {
        Some(ValueType::Tensor { ty, shape, .. }) => {
            if *ty != TensorElementType::Float32 {
                return Err(format!(
                    "the NLLB decoder's cache is {ty:?}; only float32 caches are supported (use the quantized or fp32 export)"
                ));
            }
            let dim = |i: usize| shape.get(i).copied().filter(|&d| d > 0).map(|d| d as usize);
            (dim(1).or(config.heads), dim(3).or(config.head_dim))
        }
        _ => (config.heads, config.head_dim),
    };
    let (Some(heads), Some(head_dim)) = (heads, head_dim) else {
        return Err("cannot tell the NLLB decoder's attention heads (no fixed cache dims and no config.json)".into());
    };
    Ok(DecoderLayout {
        input_ids: "input_ids".into(),
        hidden: "encoder_hidden_states".into(),
        mask: inputs.contains_key("encoder_attention_mask").then(|| "encoder_attention_mask".to_string()),
        use_cache: "use_cache_branch".into(),
        logits: "logits".into(),
        layers,
        heads,
        head_dim,
    })
}

// ---------------------------------------------------------------------------------------------
// Decoding (model-independent)
// ---------------------------------------------------------------------------------------------

/// Last-position logits of every row of one decoder step (`rows × vocab`, row-major).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Logits {
    pub rows: usize,
    pub vocab: usize,
    pub data: Vec<f32>,
}

impl Logits {
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.vocab..(r + 1) * self.vocab]
    }

    fn suppress(&mut self, ids: &[u32]) {
        for r in 0..self.rows {
            let row = &mut self.data[r * self.vocab..(r + 1) * self.vocab];
            for &id in ids {
                if let Some(x) = row.get_mut(id as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
            for x in row.iter_mut() {
                if x.is_nan() {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// One autoregressive decoder over an already-encoded source.
pub(crate) trait StepDecoder {
    /// First step, one row: the decoder prompt (`[</s>, tgt_lang]`).
    fn first(&mut self, prompt: &[i64]) -> Result<Logits, MtError>;
    /// Next step: row `i` continues row `parents[i]` of the previous step with `tokens[i]` (`parents.len()` rows).
    fn next(&mut self, tokens: &[i64], parents: &[usize]) -> Result<Logits, MtError>;
}

/// Decoding settings of one segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GenParams {
    pub beam: usize,
    pub max_new: usize,
    /// Source pieces (without the language token and `</s>`): the runaway guard's baseline.
    pub src_len: usize,
    pub length_penalty: f32,
}

impl GenParams {
    pub fn new(beam: usize, src_len: usize, max_positions: usize) -> Self {
        let cap = MAX_NEW_TOKENS.min(max_positions.saturating_sub(4)).max(4);
        Self {
            beam: beam.clamp(1, MAX_BEAM),
            max_new: (3 * src_len + 10).min(cap),
            src_len,
            length_penalty: 1.0,
        }
    }
}

/// Index of the largest finite value (lowest index on ties).
fn argmax(row: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in row.iter().enumerate() {
        if x.is_finite() && best.is_none_or(|(_, b)| x > b) {
            best = Some((i, x));
        }
    }
    best.map(|(i, _)| i)
}

/// `(index, value)` of the `n` largest finite values, largest first (lowest index on ties).
fn top_n(row: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, &x) in row.iter().enumerate() {
        if !x.is_finite() || (top.len() == n && x <= top[n - 1].1) {
            continue;
        }
        let pos = top.partition_point(|&(_, v)| v >= x);
        top.insert(pos, (i, x));
        top.truncate(n);
    }
    top
}

/// `log(sum(exp(row)))` over the finite values; `None` when there are none.
fn log_sum_exp(row: &[f32]) -> Option<f32> {
    let max = row.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let sum: f64 = row
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| f64::from(x - max).exp())
        .sum();
    Some(max + sum.ln() as f32)
}

/// When `out` (already longer than the source by 6 tokens) ends in the same 1–8 token n-gram three or more times,
/// the length that keeps one copy of it.
pub(crate) fn runaway_repetition(out: &[i64], src_len: usize) -> Option<usize> {
    if out.len() < src_len + 6 {
        return None;
    }
    for n in 1..=8usize {
        if out.len() < 3 * n {
            break;
        }
        let tail = &out[out.len() - n..];
        let mut reps = 1;
        while out.len() >= n * (reps + 1) && &out[out.len() - n * (reps + 1)..out.len() - n * reps] == tail {
            reps += 1;
        }
        if reps >= 3 {
            return Some(out.len() - n * (reps - 1));
        }
    }
    None
}

/// Generate target ids (without `</s>`) for one encoded source.
pub(crate) fn generate(
    dec: &mut dyn StepDecoder,
    prompt: &[i64],
    eos: i64,
    suppressed: &[u32],
    params: &GenParams,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<i64>, MtError> {
    if cancelled() {
        return Err(MtError::cancelled());
    }
    let mut logits = dec.first(prompt)?;
    logits.suppress(suppressed);
    if params.beam <= 1 {
        greedy(dec, logits, eos, suppressed, params, cancelled)
    } else {
        beam_search(dec, logits, eos, suppressed, params, cancelled)
    }
}

fn greedy(
    dec: &mut dyn StepDecoder,
    mut logits: Logits,
    eos: i64,
    suppressed: &[u32],
    params: &GenParams,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<i64>, MtError> {
    let mut out: Vec<i64> = Vec::new();
    loop {
        let token = argmax(logits.row(0)).ok_or_else(|| MtError::runtime("the NLLB model produced no valid token"))? as i64;
        if token == eos {
            break;
        }
        out.push(token);
        if let Some(keep) = runaway_repetition(&out, params.src_len) {
            out.truncate(keep);
            break;
        }
        if out.len() >= params.max_new {
            break;
        }
        if cancelled() {
            return Err(MtError::cancelled());
        }
        logits = dec.next(&[token], &[0])?;
        logits.suppress(suppressed);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct Hypothesis {
    tokens: Vec<i64>,
    /// Sum of log-probabilities.
    score: f32,
}

fn normalized(score: f32, len: usize, length_penalty: f32) -> f32 {
    score / (len.max(1) as f32).powf(length_penalty)
}

fn beam_search(
    dec: &mut dyn StepDecoder,
    mut logits: Logits,
    eos: i64,
    suppressed: &[u32],
    params: &GenParams,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<i64>, MtError> {
    let k = params.beam;
    let lp = params.length_penalty;
    let mut running = vec![Hypothesis {
        tokens: Vec::new(),
        score: 0.0,
    }];
    // (tokens, normalized score)
    let mut finished: Vec<(Vec<i64>, f32)> = Vec::new();
    for step in 0..params.max_new {
        // Candidates: the 2k best continuations of every running row.
        let mut cands: Vec<(f32, usize, i64)> = Vec::with_capacity(running.len() * 2 * k);
        for (r, hyp) in running.iter().enumerate() {
            let row = logits.row(r);
            let Some(lse) = log_sum_exp(row) else { continue };
            for (token, logit) in top_n(row, 2 * k) {
                cands.push((hyp.score + (logit - lse), r, token as i64));
            }
        }
        if cands.is_empty() {
            return Err(MtError::runtime("the NLLB model produced no valid token"));
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut next: Vec<Hypothesis> = Vec::with_capacity(k);
        let mut parents = Vec::with_capacity(k);
        for (rank, &(score, r, token)) in cands.iter().enumerate() {
            if token == eos {
                // An end among the k best candidates finishes a hypothesis (Hugging Face's rule).
                if rank < k {
                    let len = running[r].tokens.len() + 1;
                    finished.push((running[r].tokens.clone(), normalized(score, len, lp)));
                }
            } else {
                let mut tokens = running[r].tokens.clone();
                tokens.push(token);
                next.push(Hypothesis { tokens, score });
                parents.push(r);
            }
            if next.len() == k {
                break;
            }
        }
        finished.sort_by(|a, b| b.1.total_cmp(&a.1));
        finished.truncate(k);
        if next.is_empty() {
            break;
        }
        // Done when k hypotheses finished and the best running one cannot beat the worst finished one.
        if finished.len() == k {
            let worst = finished[k - 1].1;
            let best_possible = normalized(next[0].score, step + 2, lp);
            if best_possible <= worst {
                running = next;
                break;
            }
        }
        if let Some(keep) = runaway_repetition(&next[0].tokens, params.src_len) {
            let mut best = next.swap_remove(0);
            best.tokens.truncate(keep);
            let len = best.tokens.len() + 1;
            finished.push((best.tokens, normalized(best.score, len, lp)));
            running.clear();
            break;
        }
        running = next;
        if step + 1 == params.max_new {
            break;
        }
        if cancelled() {
            return Err(MtError::cancelled());
        }
        let tokens: Vec<i64> = running.iter().map(|h| *h.tokens.last().unwrap_or(&eos)).collect();
        logits = dec.next(&tokens, &parents)?;
        logits.suppress(suppressed);
    }
    // Unfinished beams compete at the length they reached.
    for h in running {
        let len = h.tokens.len() + 1;
        finished.push((h.tokens, normalized(h.score, len, lp)));
    }
    finished
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(tokens, _)| tokens)
        .ok_or_else(|| MtError::runtime("beam search produced nothing"))
}

// ---------------------------------------------------------------------------------------------
// The ONNX Runtime decoder
// ---------------------------------------------------------------------------------------------

fn tensor_err(e: ort::Error) -> MtError {
    rt_err("NLLB tensor", e)
}

fn empty_cache(heads: usize, head_dim: usize) -> Result<DynValue, MtError> {
    Tensor::<f32>::from_array(([1usize, heads, 0, head_dim], Vec::<f32>::new()))
        .map(|t| t.into_dyn())
        .map_err(tensor_err)
}

/// Rows `parents` (in that order) of a float tensor whose first dimension is the batch.
fn gather_rows(value: &DynValue, parents: &[usize]) -> Result<DynValue, MtError> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(tensor_err)?;
    let batch = shape.first().copied().unwrap_or(0).max(0) as usize;
    if batch == 0 {
        return Err(MtError::runtime("NLLB cache tensor without rows"));
    }
    let row = data.len() / batch;
    let mut out = Vec::with_capacity(row * parents.len());
    for &p in parents {
        if p >= batch {
            return Err(MtError::runtime(format!("beam parent {p} out of {batch} rows")));
        }
        out.extend_from_slice(&data[p * row..(p + 1) * row]);
    }
    let mut dims: Vec<i64> = shape.to_vec();
    dims[0] = parents.len() as i64;
    Tensor::<f32>::from_array((dims, out))
        .map(|t| t.into_dyn())
        .map_err(tensor_err)
}

fn ones_mask(rows: usize, len: usize) -> Result<DynValue, MtError> {
    Tensor::<i64>::from_array(([rows, len], vec![1i64; rows * len]))
        .map(|t| t.into_dyn())
        .map_err(tensor_err)
}

/// Copy the last position of every row out of `[rows, T, V]` logits.
fn last_position_logits(value: &DynValue, rows: usize) -> Result<Logits, MtError> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(tensor_err)?;
    if shape.len() != 3 || shape[0] as usize != rows || shape[1] < 1 || shape[2] < 1 {
        return Err(MtError::runtime(format!("unexpected NLLB logits shape {shape:?} for {rows} rows")));
    }
    let (t, v) = (shape[1] as usize, shape[2] as usize);
    let mut out = Vec::with_capacity(rows * v);
    for r in 0..rows {
        let start = (r * t + t - 1) * v;
        out.extend_from_slice(&data[start..start + v]);
    }
    Ok(Logits {
        rows,
        vocab: v,
        data: out,
    })
}

/// [`StepDecoder`] over the merged decoder session for one encoded source.
struct OrtStepDecoder<'m> {
    session: &'m mut Session,
    layout: &'m DecoderLayout,
    /// Encoder output, one row.
    hidden_row: DynValue,
    src_len: usize,
    rows: usize,
    /// `hidden_row` repeated for `rows > 1`.
    hidden_rows: Option<DynValue>,
    mask: DynValue,
    /// Cross-attention cache of the first step, one row (per layer: key, value).
    enc_row: Vec<DynValue>,
    /// `enc_row` repeated for `rows > 1`.
    enc_rows: Option<Vec<DynValue>>,
    /// Self-attention cache (per layer: key, value), `rows` rows.
    dec_cache: Vec<DynValue>,
}

impl<'m> OrtStepDecoder<'m> {
    fn new(session: &'m mut Session, layout: &'m DecoderLayout, hidden_row: DynValue, src_len: usize) -> Result<Self, MtError> {
        Ok(Self {
            session,
            layout,
            hidden_row,
            src_len,
            rows: 1,
            hidden_rows: None,
            mask: ones_mask(1, src_len)?,
            enc_row: Vec::new(),
            enc_rows: None,
            dec_cache: Vec::new(),
        })
    }

    /// Run one step with `ids` (`rows × len`), returning the logits and the present caches in layout order.
    fn run(&mut self, ids: Vec<i64>, len: usize, use_cache: bool) -> Result<(Logits, Vec<DynValue>), MtError> {
        let rows = self.rows;
        let layout = self.layout;
        let ids = Tensor::<i64>::from_array(([rows, len], ids)).map_err(tensor_err)?;
        let flag = Tensor::<bool>::from_array(([1usize], vec![use_cache])).map_err(tensor_err)?;
        let empty = if use_cache {
            None
        } else {
            Some(empty_cache(layout.heads, layout.head_dim)?)
        };
        let hidden = self.hidden_rows.as_ref().unwrap_or(&self.hidden_row);
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(4 + 4 * layout.layers.len());
        inputs.push((Cow::Borrowed(layout.input_ids.as_str()), ids.into()));
        inputs.push((Cow::Borrowed(layout.hidden.as_str()), hidden.into()));
        if let Some(mask) = &layout.mask {
            inputs.push((Cow::Borrowed(mask.as_str()), (&self.mask).into()));
        }
        inputs.push((Cow::Borrowed(layout.use_cache.as_str()), flag.into()));
        let enc = self.enc_rows.as_ref().unwrap_or(&self.enc_row);
        for (l, layer) in layout.layers.iter().enumerate() {
            for (part, name) in layer.inputs.iter().enumerate() {
                let value: &DynValue = match (&empty, part) {
                    (Some(e), _) => e,
                    (None, 0 | 1) => &self.dec_cache[2 * l + part],
                    (None, _) => &enc[2 * l + part - 2],
                };
                inputs.push((Cow::Borrowed(name.as_str()), value.into()));
            }
        }
        let mut outputs = self.session.run(inputs).map_err(|e| rt_err("NLLB decoder", e))?;
        let logits_value = outputs
            .remove(layout.logits.as_str())
            .ok_or_else(|| MtError::runtime("the NLLB decoder returned no logits"))?;
        let mut presents = Vec::with_capacity(4 * layout.layers.len());
        for layer in &layout.layers {
            for name in &layer.outputs {
                presents.push(
                    outputs
                        .remove(name.as_str())
                        .ok_or_else(|| MtError::runtime(format!("the NLLB decoder returned no {name}")))?,
                );
            }
        }
        drop(outputs);
        Ok((last_position_logits(&logits_value, rows)?, presents))
    }

    /// Split presents (per layer: dec key, dec value, enc key, enc value) into the decoder cache and, on the first
    /// step, the encoder cache.
    fn keep_presents(&mut self, presents: Vec<DynValue>, first: bool) {
        self.dec_cache.clear();
        if first {
            self.enc_row.clear();
        }
        for (i, value) in presents.into_iter().enumerate() {
            match i % 4 {
                0 | 1 => self.dec_cache.push(value),
                _ if first => self.enc_row.push(value),
                _ => {}
            }
        }
    }

    /// Reshape every per-row input for `parents` (the rows of the next step).
    fn reorder(&mut self, parents: &[usize]) -> Result<(), MtError> {
        let n = parents.len();
        let identity = n == self.rows && parents.iter().enumerate().all(|(i, &p)| i == p);
        if !identity {
            self.dec_cache = self
                .dec_cache
                .iter()
                .map(|v| gather_rows(v, parents))
                .collect::<Result<_, _>>()?;
        }
        if n != self.rows {
            self.rows = n;
            let repeat = vec![0usize; n];
            if n == 1 {
                self.hidden_rows = None;
                self.enc_rows = None;
            } else {
                self.hidden_rows = Some(gather_rows(&self.hidden_row, &repeat)?);
                self.enc_rows = Some(self.enc_row.iter().map(|v| gather_rows(v, &repeat)).collect::<Result<_, _>>()?);
            }
            self.mask = ones_mask(n, self.src_len)?;
        }
        Ok(())
    }
}

impl StepDecoder for OrtStepDecoder<'_> {
    fn first(&mut self, prompt: &[i64]) -> Result<Logits, MtError> {
        self.rows = 1;
        self.hidden_rows = None;
        self.enc_rows = None;
        self.mask = ones_mask(1, self.src_len)?;
        let (logits, presents) = self.run(prompt.to_vec(), prompt.len(), false)?;
        self.keep_presents(presents, true);
        Ok(logits)
    }

    fn next(&mut self, tokens: &[i64], parents: &[usize]) -> Result<Logits, MtError> {
        if tokens.len() != parents.len() || tokens.is_empty() {
            return Err(MtError::runtime("NLLB step: tokens and parents differ"));
        }
        self.reorder(parents)?;
        let (logits, presents) = self.run(tokens.to_vec(), 1, true)?;
        self.keep_presents(presents, false);
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

/// Split a line into sentences for sentence-level translation: at line breaks, after `。！？` (and closing
/// quotes), and after `.!?…` followed by a space and a character that does not continue a sentence (not a
/// lower-case letter). A period after a short word or a common abbreviation (`Mr.`, `e.g.`) does not split.
pub fn split_sentences(text: &str) -> Vec<String> {
    const ABBREVIATIONS: &[&str] = &[
        "mr", "mrs", "ms", "dr", "st", "vs", "etc", "jr", "sr", "prof", "inc", "ltd", "co", "no", "e.g", "i.e", "approx",
    ];
    let mut out = Vec::new();
    for line in text.split(['\n', '\r']) {
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let cjk = matches!(c, '。' | '！' | '？' | '｡');
            let latin = matches!(c, '.' | '!' | '?' | '…');
            if !cjk && !latin {
                i += 1;
                continue;
            }
            // The whole punctuation run plus closing quotes and brackets.
            let mut end = i + 1;
            while end < chars.len()
                && matches!(chars[end], '.' | '!' | '?' | '…' | '。' | '！' | '？' | '｡' | '"' | '\'' | '”' | '’' | '」' | '』' | ')' | '）' | ']')
            {
                end += 1;
            }
            let split = if cjk {
                true
            } else {
                let next = chars[end..].iter().position(|c| !c.is_whitespace()).map(|p| end + p);
                let spaced = end < chars.len() && chars[end].is_whitespace();
                let continues = next.is_some_and(|n| chars[n].is_lowercase());
                let word: String = chars[start..i]
                    .iter()
                    .rev()
                    .take_while(|c| c.is_alphanumeric() || **c == '.')
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                let abbreviation = c == '.'
                    && (word.chars().filter(|c| c.is_alphanumeric()).count() <= 1
                        || ABBREVIATIONS.contains(&word.to_lowercase().as_str()));
                spaced && next.is_some() && !continues && !abbreviation
            };
            if split {
                let s: String = chars[start..end].iter().collect::<String>().trim().to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                start = end;
            }
            i = end;
        }
        let rest: String = chars[start..].iter().collect::<String>().trim().to_string();
        if !rest.is_empty() {
            out.push(rest);
        }
    }
    out
}

fn is_cjk_char(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F // CJK symbols and punctuation
        | 0x3040..=0x30FF // kana
        | 0x31F0..=0x31FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xFF00..=0xFFEF // full-width forms
        | 0x20000..=0x2FFFF)
}

/// Clean decoded text: whitespace runs become one space, and for Chinese and Japanese targets the spaces NLLB puts
/// between two CJK characters (`今夜 流れに`) are removed; spaces next to Latin words stay.
pub fn tidy_output(text: &str, tgt: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !matches!(tgt, "jpn_Jpan" | "zho_Hans" | "zho_Hant" | "yue_Hant") {
        return collapsed;
    }
    let chars: Vec<char> = collapsed.chars().collect();
    let mut out = String::with_capacity(collapsed.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' && i > 0 && i + 1 < chars.len() && is_cjk_char(chars[i - 1]) && is_cjk_char(chars[i + 1]) {
            continue;
        }
        out.push(c);
    }
    out
}

/// A loaded NLLB model (both sessions and the tokenizer).
pub struct NllbModel {
    id: String,
    threads: u32,
    encoder: Session,
    decoder: Session,
    enc_layout: EncoderLayout,
    dec_layout: DecoderLayout,
    tokenizer: NllbTokenizer,
    config: ModelConfig,
}

impl std::fmt::Debug for NllbModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NllbModel")
            .field("id", &self.id)
            .field("threads", &self.threads)
            .field("layers", &self.dec_layout.layers.len())
            .finish()
    }
}

impl NllbModel {
    /// Load the model in `dir` (see [`find_model_files`]) with `threads` intra-op threads per session. `id` names it
    /// in responses and status.
    pub fn load(dir: &Path, id: &str, threads: u32) -> Result<Self, String> {
        let started = Instant::now();
        let files = find_model_files(dir)?;
        let config = match &files.config {
            Some(p) => ModelConfig::parse(&std::fs::read_to_string(p).map_err(|e| format!("config.json: {e}"))?)?,
            None => ModelConfig::default(),
        };
        let tokenizer = NllbTokenizer::from_file(&files.tokenizer)?;
        let opts = VcSessionOptions {
            intra_threads: Some(threads.clamp(1, super::MAX_THREADS) as usize),
            // Every step has a new cache length: without the arena each shape pays for fresh allocations.
            cpu_arena: true,
            thread_priority: Some(WorkerPriority::ProtectGame),
            ..VcSessionOptions::default()
        };
        let (encoder, _) = build_session(Some(EP_COMPONENT), EP_CPU, &opts, ModelSource::File(&files.encoder))
            .map_err(|e| format!("cannot load the NLLB encoder: {e}"))?;
        let (decoder, _) = build_session(Some(EP_COMPONENT), EP_CPU, &opts, ModelSource::File(&files.decoder))
            .map_err(|e| format!("cannot load the NLLB decoder: {e}"))?;
        let enc_layout = parse_encoder_layout(&encoder)?;
        let dec_layout = parse_decoder_layout(&decoder, &config)?;
        tracing::info!(
            target: LOG_TARGET,
            "NLLB {id} loaded in {} ms ({} decoder layers, {} threads)",
            started.elapsed().as_millis(),
            dec_layout.layers.len(),
            threads
        );
        Ok(Self {
            id: id.to_string(),
            threads,
            encoder,
            decoder,
            enc_layout,
            dec_layout,
            tokenizer,
            config,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn threads(&self) -> u32 {
        self.threads
    }

    #[cfg(test)]
    pub fn tokenizer(&self) -> &NllbTokenizer {
        &self.tokenizer
    }

    fn encode(&mut self, ids: &[i64]) -> Result<DynValue, MtError> {
        let len = ids.len();
        let input = Tensor::<i64>::from_array(([1usize, len], ids.to_vec())).map_err(tensor_err)?;
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            vec![(Cow::Borrowed(self.enc_layout.input_ids.as_str()), input.into())];
        if let Some(mask) = &self.enc_layout.attention_mask {
            inputs.push((Cow::Borrowed(mask.as_str()), ones_mask(1, len)?.into()));
        }
        let mut outputs = self.encoder.run(inputs).map_err(|e| rt_err("NLLB encoder", e))?;
        outputs
            .remove(self.enc_layout.output.as_str())
            .ok_or_else(|| MtError::runtime("the NLLB encoder returned nothing"))
    }

    /// Target ids for source ids `[src_lang] + pieces + [</s>]` into `tgt_id`.
    pub(crate) fn generate_ids(
        &mut self,
        src: &[i64],
        tgt_id: u32,
        beam: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<i64>, MtError> {
        let hidden = self.encode(src)?;
        let params = GenParams::new(beam, src.len().saturating_sub(2), self.config.max_positions);
        let eos = i64::from(self.tokenizer.eos_id());
        let prompt = [self.config.decoder_start, i64::from(tgt_id)];
        let suppressed = self.tokenizer.suppressed_ids().to_vec();
        let mut dec = OrtStepDecoder::new(&mut self.decoder, &self.dec_layout, hidden, src.len())?;
        generate(&mut dec, &prompt, eos, &suppressed, &params, cancelled)
    }

    /// Translate `text` from `src` (FLORES; `None` = guess from the script) into `tgt` (FLORES).
    pub fn translate(
        &mut self,
        text: &str,
        src: Option<&str>,
        tgt: &str,
        beam: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String, MtError> {
        let tgt_id = self
            .tokenizer
            .lang_id(tgt)
            .ok_or_else(|| MtError::unsupported_pair(src.unwrap_or(""), tgt))?;
        if !text.chars().any(char::is_alphabetic) {
            return Ok(text.trim().to_string());
        }
        let src = match src {
            Some(s) => s,
            None => flores::guess_flores(text).unwrap_or("eng_Latn"),
        };
        if self.tokenizer.lang_id(src).is_none() {
            return Err(MtError::unsupported_pair(src, tgt));
        }
        let mut parts = Vec::new();
        for segment in split_sentences(text) {
            if cancelled() {
                return Err(MtError::cancelled());
            }
            if !segment.chars().any(char::is_alphabetic) {
                parts.push(segment);
                continue;
            }
            let mut ids = self.tokenizer.encode(&segment, src).map_err(MtError::runtime)?;
            if ids.len() > MAX_SOURCE_TOKENS {
                let eos = ids[ids.len() - 1];
                ids.truncate(MAX_SOURCE_TOKENS - 1);
                ids.push(eos);
            }
            let out = self.generate_ids(&ids, tgt_id, beam, cancelled)?;
            let piece = tidy_output(&self.tokenizer.decode(&out).map_err(MtError::runtime)?, tgt);
            if !piece.is_empty() {
                parts.push(piece);
            }
        }
        let sep = if flores::joins_without_spaces(tgt) { "" } else { " " };
        Ok(parts.join(sep))
    }
}

// ---------------------------------------------------------------------------------------------
// The engine: one worker thread, status, cancellation
// ---------------------------------------------------------------------------------------------

/// A validated request, ready for the worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prepared {
    pub text: String,
    /// FLORES source; `None` = guess.
    pub source: Option<&'static str>,
    pub target: &'static str,
    pub model: String,
    pub threads: u32,
    pub beam: usize,
}

/// What `nllb_translate` answers without touching the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Shortcut {
    /// Return the text unchanged (no letters, or the explicit source equals the target).
    Unchanged(String),
}

/// Validate a request: languages, text, model id. `loaded` is the model already loaded (the default when the
/// request names none).
pub(crate) fn prepare(request: &MtRequest, loaded: Option<&str>) -> Result<Result<Prepared, Shortcut>, MtError> {
    let text = request.text.trim();
    if text.is_empty() {
        return Err(MtError::new(MtErrorKind::InvalidInput, "nothing to translate"));
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(MtError::new(
            MtErrorKind::InvalidInput,
            format!("the text is too long for NLLB (more than {MAX_TEXT_CHARS} characters)"),
        ));
    }
    let target = flores::to_flores(&request.target).ok_or_else(|| MtError::unsupported_pair(&request.source, &request.target))?;
    let source = if request.source.trim().is_empty() {
        None
    } else {
        Some(flores::to_flores(&request.source).ok_or_else(|| MtError::unsupported_pair(&request.source, &request.target))?)
    };
    if source == Some(target) || !text.chars().any(char::is_alphabetic) {
        return Ok(Err(Shortcut::Unchanged(text.to_string())));
    }
    let model = resolve_model_id(request.model.as_deref(), loaded)?;
    let beam = match request.kind {
        MtKind::Interim => 1,
        MtKind::Final => request.beam.unwrap_or(1).clamp(1, MAX_BEAM as u32) as usize,
    };
    Ok(Ok(Prepared {
        text: text.to_string(),
        source,
        target,
        model,
        threads: thread_count(request.threads),
        beam,
    }))
}

/// The catalog id to use: the requested one, else the loaded one, else the default.
pub(crate) fn resolve_model_id(requested: Option<&str>, loaded: Option<&str>) -> Result<String, MtError> {
    let id = requested
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(loaded)
        .unwrap_or(NLLB_DEFAULT_MODEL);
    if !id.starts_with(MODEL_ID_PREFIX) || id.contains("..") || id.contains('\\') {
        return Err(MtError::new(MtErrorKind::InvalidInput, format!("{id} is not an NLLB model")));
    }
    Ok(id.to_string())
}

/// Install folder of a catalog id, or `not_installed`.
fn model_dir(id: &str) -> Result<PathBuf, MtError> {
    crate::services::model_store::installed_path(id).ok_or_else(|| MtError::not_installed(vec![id.to_string()]))
}

/// Remove local paths (the model folder, the user profile) from a message shown in the UI.
fn redact_paths(message: &str, dir: Option<&Path>) -> String {
    let mut out = message.to_string();
    if let Some(dir) = dir {
        for form in [dir.display().to_string(), dir.to_string_lossy().replace('\\', "/")] {
            if !form.is_empty() {
                out = out.replace(&form, "<model folder>");
            }
        }
    }
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let home = home.to_string_lossy().to_string();
        if home.len() > 3 {
            out = out.replace(&home, "%USERPROFILE%");
            out = out.replace(&home.replace('\\', "/"), "%USERPROFILE%");
        }
    }
    out
}

type StatusSink = Box<dyn Fn(&MtEngineStatus) + Send + Sync>;

enum Job {
    Load {
        id: String,
        dir: PathBuf,
        threads: u32,
        reply: tokio::sync::oneshot::Sender<Result<MtEngineStatus, MtError>>,
    },
    Translate {
        request: Prepared,
        dir: PathBuf,
        cancel: Arc<AtomicBool>,
        reply: tokio::sync::oneshot::Sender<Result<MtResponse, MtError>>,
    },
    Unload {
        reply: tokio::sync::oneshot::Sender<()>,
    },
}

struct Pending {
    serial: u64,
    request_id: Option<String>,
    cancel: Arc<AtomicBool>,
}

struct Engine {
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
    status: Mutex<MtEngineStatus>,
    loaded: Mutex<Option<(String, u32)>>,
    pending: Mutex<Vec<Pending>>,
    serial: AtomicU64,
    emitter: OnceLock<StatusSink>,
}

fn idle_status() -> MtEngineStatus {
    MtEngineStatus {
        engine: ENGINE_NLLB.into(),
        state: MtEngineState::Idle,
        model: None,
        device: None,
        threads: None,
        detail: None,
    }
}

fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| Engine {
        jobs: Mutex::new(None),
        status: Mutex::new(idle_status()),
        loaded: Mutex::new(None),
        pending: Mutex::new(Vec::new()),
        serial: AtomicU64::new(0),
        emitter: OnceLock::new(),
    })
}

impl Engine {
    fn status(&self) -> MtEngineStatus {
        self.status.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn set_status(&self, status: MtEngineStatus) {
        *self.status.lock().unwrap_or_else(PoisonError::into_inner) = status.clone();
        if let Some(emit) = self.emitter.get() {
            emit(&status);
        }
    }

    fn loaded_id(&self) -> Option<String> {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|(id, _)| id.clone())
    }

    fn send(&self, job: Job) -> Result<(), MtError> {
        let mut jobs = self.jobs.lock().unwrap_or_else(PoisonError::into_inner);
        let job = match jobs.as_ref() {
            Some(tx) => match tx.send(job) {
                Ok(()) => return Ok(()),
                // The worker is gone (it never exits on its own; only a failed spawn gets here): start a new one.
                Err(mpsc::SendError(job)) => job,
            },
            None => job,
        };
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("sigil-nllb".into())
            .spawn(move || worker(rx))
            .map_err(|e| rt_err("cannot start the NLLB worker", e))?;
        tx.send(job).map_err(|_| MtError::runtime("the NLLB worker stopped"))?;
        *jobs = Some(tx);
        Ok(())
    }

    fn register(&self, request_id: Option<String>) -> (u64, Arc<AtomicBool>) {
        let serial = self.serial.fetch_add(1, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        self.pending.lock().unwrap_or_else(PoisonError::into_inner).push(Pending {
            serial,
            request_id,
            cancel: cancel.clone(),
        });
        (serial, cancel)
    }

    fn unregister(&self, serial: u64) {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|p| p.serial != serial);
    }

    /// Cancel `request_id`'s requests, or every pending one.
    fn cancel(&self, request_id: Option<&str>) -> usize {
        let pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        let mut n = 0;
        for p in pending.iter() {
            if request_id.is_none() || p.request_id.as_deref() == request_id {
                p.cancel.store(true, Ordering::Relaxed);
                n += 1;
            }
        }
        n
    }
}

fn status_for(state: MtEngineState, model: Option<&str>, threads: Option<u32>, detail: Option<String>) -> MtEngineStatus {
    MtEngineStatus {
        engine: ENGINE_NLLB.into(),
        state,
        model: model.map(str::to_string),
        device: model.map(|_| "cpu".to_string()),
        threads,
        detail,
    }
}

/// The worker thread: owns the model; serves jobs in order.
fn worker(rx: mpsc::Receiver<Job>) {
    // A background workload: below-normal priority, so a game keeps the CPU.
    let _ = rt_thread::set_worker_priority(WorkerPriority::ProtectGame, false);
    let mut model: Option<NllbModel> = None;
    while let Ok(job) = rx.recv() {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&mut model, job)));
        if outcome.is_err() {
            tracing::error!(target: LOG_TARGET, "the NLLB worker panicked; the model is unloaded");
            model = None;
            *engine().loaded.lock().unwrap_or_else(PoisonError::into_inner) = None;
            engine().set_status(status_for(
                MtEngineState::Error,
                None,
                None,
                Some("NLLB stopped after an internal error; the next request reloads it".into()),
            ));
        }
    }
}

/// Load `id` with `threads` unless it is already loaded (with the same threads, when `exact_threads`: an explicit
/// `nllb_load` rebuilds the sessions for a new thread count, a translation keeps whatever is loaded).
fn ensure_model<'a>(
    model: &'a mut Option<NllbModel>,
    id: &str,
    dir: &Path,
    threads: u32,
    exact_threads: bool,
) -> Result<&'a mut NllbModel, MtError> {
    let keep = model
        .as_ref()
        .is_some_and(|m| m.id == id && (!exact_threads || m.threads == threads));
    if !keep {
        let e = engine();
        *model = None;
        *e.loaded.lock().unwrap_or_else(PoisonError::into_inner) = None;
        e.set_status(status_for(MtEngineState::Loading, Some(id), Some(threads), None));
        match NllbModel::load(dir, id, threads) {
            Ok(m) => {
                *model = Some(m);
                *e.loaded.lock().unwrap_or_else(PoisonError::into_inner) = Some((id.to_string(), threads));
                e.set_status(status_for(MtEngineState::Ready, Some(id), Some(threads), None));
            }
            Err(err) => {
                let message = redact_paths(&err, Some(dir));
                tracing::warn!(target: LOG_TARGET, "NLLB {id}: {message}");
                e.set_status(status_for(MtEngineState::Error, Some(id), Some(threads), Some(message.clone())));
                return Err(MtError::runtime(message));
            }
        }
    }
    model.as_mut().ok_or_else(|| MtError::runtime("NLLB is not loaded"))
}

fn handle(model: &mut Option<NllbModel>, job: Job) {
    match job {
        Job::Load { id, dir, threads, reply } => {
            let result = ensure_model(model, &id, &dir, threads, true).map(|_| engine().status());
            let _ = reply.send(result);
        }
        Job::Translate {
            request,
            dir,
            cancel,
            reply,
        } => {
            let cancelled = || cancel.load(Ordering::Relaxed);
            let result = if cancelled() {
                Err(MtError::cancelled())
            } else {
                ensure_model(model, &request.model, &dir, request.threads, false).and_then(|m| {
                    let started = Instant::now();
                    let text = m.translate(&request.text, request.source, request.target, request.beam, &cancelled)?;
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    tracing::debug!(
                        target: LOG_TARGET,
                        "NLLB {} -> {} ({} chars, beam {}) in {elapsed_ms} ms",
                        request.source.unwrap_or("?"),
                        request.target,
                        request.text.chars().count(),
                        request.beam
                    );
                    Ok(MtResponse {
                        text,
                        engine: ENGINE_NLLB.into(),
                        model: m.id.clone(),
                        elapsed_ms,
                        pivot: false,
                    })
                })
            };
            let _ = reply.send(result.map_err(|e| MtError {
                message: redact_paths(&e.message, Some(&dir)),
                ..e
            }));
        }
        Job::Unload { reply } => {
            let was = model.take();
            *engine().loaded.lock().unwrap_or_else(PoisonError::into_inner) = None;
            if let Some(m) = was {
                tracing::info!(target: LOG_TARGET, "NLLB {} unloaded", m.id);
            }
            engine().set_status(idle_status());
            let _ = reply.send(());
        }
    }
}

fn install_emitter<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    let _ = engine().emitter.set(Box::new(move |status: &MtEngineStatus| {
        if let Err(e) = app.emit(STATUS_EVENT, status.clone()) {
            tracing::debug!(target: LOG_TARGET, "cannot emit {STATUS_EVENT}: {e}");
        }
    }));
}

async fn await_reply<T>(rx: tokio::sync::oneshot::Receiver<Result<T, MtError>>) -> Result<T, MtError> {
    rx.await.unwrap_or_else(|_| Err(MtError::runtime("the NLLB worker stopped")))
}

/// Translate one prepared request on the worker (shared by the command and tests).
async fn translate_prepared(prepared: Prepared, request_id: Option<String>) -> Result<MtResponse, MtError> {
    let e = engine();
    let dir = model_dir(&prepared.model)?;
    let (serial, cancel) = e.register(request_id);
    let (reply, rx) = tokio::sync::oneshot::channel();
    let sent = e.send(Job::Translate {
        request: prepared,
        dir,
        cancel,
        reply,
    });
    let result = match sent {
        Ok(()) => await_reply(rx).await,
        Err(err) => Err(err),
    };
    e.unregister(serial);
    result
}

/// Load (or keep) an NLLB model: `model` = catalog id (default [`super::NLLB_DEFAULT_MODEL`]).
#[command]
pub async fn nllb_load<R: Runtime>(app: AppHandle<R>, model: Option<String>, threads: Option<u32>) -> Result<MtEngineStatus, MtError> {
    install_emitter(&app);
    let e = engine();
    let id = resolve_model_id(model.as_deref(), None)?;
    let dir = model_dir(&id)?;
    let (reply, rx) = tokio::sync::oneshot::channel();
    e.send(Job::Load {
        id,
        dir,
        threads: thread_count(threads),
        reply,
    })?;
    await_reply(rx).await
}

/// Translate one line (loads the model on demand).
#[command]
pub async fn nllb_translate<R: Runtime>(app: AppHandle<R>, request: MtRequest) -> Result<MtResponse, MtError> {
    install_emitter(&app);
    let started = Instant::now();
    let loaded = engine().loaded_id();
    match prepare(&request, loaded.as_deref())? {
        Err(Shortcut::Unchanged(text)) => Ok(MtResponse {
            text,
            engine: ENGINE_NLLB.into(),
            model: loaded.unwrap_or_default(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            pivot: false,
        }),
        Ok(prepared) => translate_prepared(prepared, request.request_id).await,
    }
}

/// Cancel one request (`request_id`) or every pending one (`None`).
#[command]
pub async fn nllb_cancel(request_id: Option<String>) -> Result<(), MtError> {
    engine().cancel(request_id.as_deref());
    Ok(())
}

/// Free the loaded model (pending requests are cancelled first).
#[command]
pub async fn nllb_unload<R: Runtime>(app: AppHandle<R>) -> Result<(), MtError> {
    install_emitter(&app);
    let e = engine();
    e.cancel(None);
    if e.jobs.lock().unwrap_or_else(PoisonError::into_inner).is_none() {
        return Ok(());
    }
    let (reply, rx) = tokio::sync::oneshot::channel();
    e.send(Job::Unload { reply })?;
    let _ = rx.await;
    Ok(())
}

#[command]
pub async fn nllb_status() -> Result<MtEngineStatus, MtError> {
    Ok(engine().status())
}

#[cfg(test)]
mod tests;
