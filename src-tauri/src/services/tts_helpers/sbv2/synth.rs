//! Style-Bert-VITS2 inference on Sigil's ONNX Runtime: the Japanese DeBERTa feature model, the
//! character tokenizer, and one voice model (JP-Extra, or the older multilingual layout read for
//! Japanese text).
//!
//! Every session is built through `neural_audio::ort_session` (EP ladder, no spinning, the EP
//! actually used recorded under `tts.sbv2.bert` / `tts.sbv2`). The graphs' inputs are read from the
//! models instead of being assumed, so the exports of the different converters all run:
//! - BERT: `input_ids` plus `attention_mask` / `token_type_ids` when the graph has them; output
//!   `[T, D]` or `[1, T, D]` (the third-to-last hidden state, as Style-Bert-VITS2 uses it).
//! - Voice: `x_tst`, `x_tst_lengths`, `sid`, `tones`, `language`, `bert` (JP-Extra) or `bert` /
//!   `ja_bert` / `en_bert` (older models: Japanese features in `ja_bert`, zeros elsewhere),
//!   `style_vec`, and the scalars `length_scale`, `sdp_ratio`, `noise_scale`, `noise_scale_w`,
//!   whose rank (0-d or `[1]`) differs between converters. Integer inputs may be int64 or int32,
//!   float inputs float32 or float16; the audio output may be either float type.

use std::borrow::Cow;
use std::path::Path;

use half::f16;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType};

use super::format::StyleVectors;
use super::g2p::Sequence;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

/// Diagnostics component of the voice session.
pub const EP_COMPONENT_VOICE: &str = "tts.sbv2";
/// Diagnostics component of the BERT session.
pub const EP_COMPONENT_BERT: &str = "tts.sbv2.bert";
/// Width of the Japanese DeBERTa features when a graph does not say.
pub const BERT_DIM: usize = 1024;

/// Style-Bert-VITS2's defaults (AivisSpeech's too).
pub const DEFAULT_NOISE: f32 = 0.6;
pub const DEFAULT_NOISE_W: f32 = 0.8;
pub const DEFAULT_SDP_RATIO: f32 = 0.2;

fn ort_err(what: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// Declared element type and rank of a session input.
fn input_type(session: &Session, name: &str) -> Option<(TensorElementType, usize, Vec<i64>)> {
    let input = session.inputs().iter().find(|i| i.name() == name)?;
    let ty = input.dtype().tensor_type()?;
    let shape = input.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
    Some((ty, shape.len(), shape))
}

fn int_tensor(ty: TensorElementType, shape: Vec<usize>, data: Vec<i64>) -> Result<DynValue, String> {
    match ty {
        TensorElementType::Int64 => Tensor::<i64>::from_array((shape, data)).map(|t| t.into_dyn()),
        TensorElementType::Int32 => Tensor::<i32>::from_array((shape, data.into_iter().map(|v| v as i32).collect::<Vec<_>>())).map(|t| t.into_dyn()),
        other => return Err(format!("unsupported integer input type {other:?}")),
    }
    .map_err(|e| e.to_string())
}

fn float_tensor(ty: TensorElementType, shape: Vec<usize>, data: Vec<f32>) -> Result<DynValue, String> {
    match ty {
        TensorElementType::Float32 => Tensor::<f32>::from_array((shape, data)).map(|t| t.into_dyn()),
        TensorElementType::Float16 => Tensor::<f16>::from_array((shape, data.into_iter().map(f16::from_f32).collect::<Vec<_>>())).map(|t| t.into_dyn()),
        other => return Err(format!("unsupported float input type {other:?}")),
    }
    .map_err(|e| e.to_string())
}

/// A float output as f32 values (float32 or float16).
fn float_output(value: &DynValue) -> Result<(Vec<i64>, Vec<f32>), String> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok((shape.to_vec(), data.to_vec()));
    }
    let (shape, data) = value.try_extract_tensor::<f16>().map_err(|e| format!("unexpected output type: {e}"))?;
    Ok((shape.to_vec(), data.iter().map(|v| v.to_f32()).collect()))
}

fn session_options(threads: usize) -> VcSessionOptions {
    VcSessionOptions {
        intra_threads: Some(threads.clamp(1, 8)),
        // Every sentence length is a new input shape: the arena avoids paying for each one.
        cpu_arena: true,
        ..VcSessionOptions::default()
    }
}

// -------------------------------------------------------------------------------------------------
// Tokenizer
// -------------------------------------------------------------------------------------------------

/// The DeBERTa character tokenizer (`tokenizer.json`): exactly one id per character, so the
/// features line up with `word2ph` (a character the vocabulary lacks, or one that normalizes to
/// several or none, is `[UNK]`).
pub struct CharTokenizer {
    inner: tokenizers::Tokenizer,
    cls: i64,
    sep: i64,
    unk: i64,
}

impl CharTokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| format!("cannot load the BERT tokenizer {}: {e}", path.display()))?;
        let id = |t: &str, fallback: i64| inner.token_to_id(t).map_or(fallback, i64::from);
        let (cls, sep, unk) = (id("[CLS]", 1), id("[SEP]", 2), id("[UNK]", 3));
        Ok(Self { inner, cls, sep, unk })
    }

    /// `[CLS]`, one id per character of `text`, `[SEP]`.
    pub fn ids(&self, text: &str) -> Vec<i64> {
        let mut out = Vec::with_capacity(text.chars().count() + 2);
        out.push(self.cls);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let id = match self.inner.encode(&*c.encode_utf8(&mut buf), false) {
                Ok(enc) if enc.get_ids().len() == 1 => i64::from(enc.get_ids()[0]),
                _ => self.unk,
            };
            out.push(id);
        }
        out.push(self.sep);
        out
    }
}

// -------------------------------------------------------------------------------------------------
// BERT
// -------------------------------------------------------------------------------------------------

/// The Japanese DeBERTa feature model.
pub struct Bert {
    session: Session,
    ids: (String, TensorElementType),
    mask: Option<(String, TensorElementType)>,
    types: Option<(String, TensorElementType)>,
    output: String,
    pub ep: String,
}

impl Bert {
    pub fn load(path: &Path, ep: &str, threads: usize) -> Result<Self, String> {
        let (session, outcome) = build_session(Some(EP_COMPONENT_BERT), ep, &session_options(threads), ModelSource::File(path))
            .map_err(|e| format!("cannot load the Japanese BERT model: {e}"))?;
        Self::from_session(session, outcome.active.clone())
    }

    pub(crate) fn from_session(session: Session, ep: String) -> Result<Self, String> {
        let find = |name: &str| input_type(&session, name).map(|(t, _, _)| (name.to_string(), t));
        let ids = find("input_ids").ok_or("the BERT model has no input_ids input")?;
        let mask = find("attention_mask");
        let types = find("token_type_ids");
        let known = ["input_ids", "attention_mask", "token_type_ids"];
        if let Some(other) = session.inputs().iter().map(|i| i.name()).find(|n| !known.contains(n)) {
            return Err(format!("the BERT model has an unexpected input {other:?}"));
        }
        let output = session
            .outputs()
            .iter()
            .find(|o| o.name() == "output")
            .or_else(|| session.outputs().first())
            .map(|o| o.name().to_string())
            .ok_or("the BERT model has no output")?;
        Ok(Self { session, ids, mask, types, output, ep })
    }

    /// Features `[T, D]` (row-major) for `ids`; returns `(values, D)`.
    pub fn features(&mut self, ids: &[i64]) -> Result<(Vec<f32>, usize), String> {
        let t = ids.len();
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(3);
        inputs.push((Cow::Owned(self.ids.0.clone()), int_tensor(self.ids.1, vec![1, t], ids.to_vec())?.into()));
        if let Some((name, ty)) = &self.mask {
            inputs.push((Cow::Owned(name.clone()), int_tensor(*ty, vec![1, t], vec![1; t])?.into()));
        }
        if let Some((name, ty)) = &self.types {
            inputs.push((Cow::Owned(name.clone()), int_tensor(*ty, vec![1, t], vec![0; t])?.into()));
        }
        let outputs = self.session.run(inputs).map_err(ort_err("BERT inference failed"))?;
        let value = outputs.get(self.output.as_str()).ok_or("the BERT model returned no features")?;
        let (shape, data) = float_output(value)?;
        let dims: Vec<usize> = shape.iter().map(|d| (*d).max(0) as usize).collect();
        let (rows, width) = match dims[..] {
            [r, w] => (r, w),
            [1, r, w] => (r, w),
            _ => return Err(format!("unexpected BERT output shape {shape:?}")),
        };
        if rows != t || width == 0 || data.len() != rows * width {
            return Err(format!("the BERT model returned {rows} rows for {t} tokens"));
        }
        Ok((data, width))
    }
}

/// Phone-level features `[D, sum(word2ph)]` (row-major): row `i` of `features` (`[T, D]`)
/// repeated `word2ph[i]` times, transposed.
pub fn phone_level_features(features: &[f32], dim: usize, word2ph: &[usize]) -> Result<Vec<f32>, String> {
    if dim == 0 || features.len() != word2ph.len() * dim {
        return Err(format!("{} BERT rows for {} characters", features.len() / dim.max(1), word2ph.len()));
    }
    let frames: usize = word2ph.iter().sum();
    let mut out = vec![0f32; dim * frames];
    let mut t = 0usize;
    for (row, &reps) in word2ph.iter().enumerate() {
        let src = &features[row * dim..(row + 1) * dim];
        for _ in 0..reps {
            for (d, v) in src.iter().enumerate() {
                out[d * frames + t] = *v;
            }
            t += 1;
        }
    }
    Ok(out)
}

// -------------------------------------------------------------------------------------------------
// Voice
// -------------------------------------------------------------------------------------------------

/// What feeds one voice-model input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feed {
    Phones,
    Lengths,
    Sid,
    Tones,
    Language,
    /// The Japanese features (JP-Extra `bert`, or `ja_bert`).
    JaBert,
    /// Zeros of the declared width (`bert` = Chinese and `en_bert` of older models).
    ZeroBert,
    Style,
    LengthScale,
    SdpRatio,
    NoiseScale,
    NoiseScaleW,
}

#[derive(Clone, Debug)]
struct Input {
    name: String,
    feed: Feed,
    ty: TensorElementType,
    rank: usize,
    /// Declared dim 1 (feature width) for BERT inputs; `None` when dynamic.
    width: Option<usize>,
}

/// Synthesis settings of one sentence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthParams {
    pub sid: u32,
    pub style: u32,
    pub style_weight: f32,
    pub sdp_ratio: f32,
    pub noise: f32,
    pub noise_w: f32,
    /// `1 / speed`.
    pub length_scale: f32,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            sid: 0,
            style: 0,
            style_weight: 1.0,
            sdp_ratio: DEFAULT_SDP_RATIO,
            noise: DEFAULT_NOISE,
            noise_w: DEFAULT_NOISE_W,
            length_scale: 1.0,
        }
    }
}

/// One loaded voice model.
pub struct Voice {
    session: Session,
    inputs: Vec<Input>,
    output: String,
    styles: StyleVectors,
    jp_extra: bool,
    pub sample_rate: u32,
    pub ep: String,
}

impl Voice {
    pub fn load(model: &Path, styles: StyleVectors, sample_rate: u32, ep: &str, threads: usize) -> Result<Self, String> {
        let (session, outcome) = build_session(Some(EP_COMPONENT_VOICE), ep, &session_options(threads), ModelSource::File(model))
            .map_err(|e| format!("cannot load the voice model: {e}"))?;
        Self::from_session(session, styles, sample_rate, outcome.active.clone())
    }

    pub(crate) fn from_session(session: Session, styles: StyleVectors, sample_rate: u32, ep: String) -> Result<Self, String> {
        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let jp_extra = !names.iter().any(|n| n == "ja_bert");
        let mut inputs = Vec::with_capacity(names.len());
        for name in &names {
            let feed = match name.as_str() {
                "x_tst" | "x" | "phones" => Feed::Phones,
                "x_tst_lengths" | "x_lengths" => Feed::Lengths,
                "sid" => Feed::Sid,
                "tones" | "tone" => Feed::Tones,
                "language" => Feed::Language,
                "bert" if jp_extra => Feed::JaBert,
                "bert" | "en_bert" => Feed::ZeroBert,
                "ja_bert" => Feed::JaBert,
                "style_vec" => Feed::Style,
                "length_scale" => Feed::LengthScale,
                "sdp_ratio" => Feed::SdpRatio,
                "noise_scale" => Feed::NoiseScale,
                "noise_scale_w" => Feed::NoiseScaleW,
                other => return Err(format!("this voice model has an input Sigil does not know ({other:?}); it is not a Style-Bert-VITS2 model")),
            };
            let (ty, rank, shape) = input_type(&session, name).ok_or_else(|| format!("voice input {name:?} is not a tensor"))?;
            let width = shape.get(1).copied().filter(|d| *d > 0).map(|d| d as usize);
            inputs.push(Input { name: name.clone(), feed, ty, rank, width });
        }
        for needed in [Feed::Phones, Feed::Tones, Feed::JaBert, Feed::Style] {
            if !inputs.iter().any(|i| i.feed == needed) {
                return Err(format!("this voice model lacks a Style-Bert-VITS2 input ({needed:?}); it is not a Style-Bert-VITS2 model"));
            }
        }
        if let Some(i) = inputs.iter().find(|i| i.feed == Feed::Style) {
            if let Some(w) = i.width.filter(|w| *w != styles.width) {
                return Err(format!("the style vectors are {} wide but the model expects {w}", styles.width));
            }
        }
        let output = session
            .outputs()
            .iter()
            .find(|o| o.name() == "output")
            .or_else(|| session.outputs().first())
            .map(|o| o.name().to_string())
            .ok_or("the voice model has no output")?;
        Ok(Self { session, inputs, output, styles, jp_extra, sample_rate, ep })
    }

    /// Load `model` once on the CPU (one thread, no diagnostics row) to check that it is a
    /// Style-Bert-VITS2 graph that fits `styles`; the session is dropped.
    pub fn probe(model: &Path, styles: StyleVectors) -> Result<(), String> {
        let opts = VcSessionOptions { intra_threads: Some(1), ..VcSessionOptions::default() };
        let (session, _) = build_session(None, crate::services::neural_audio::ort_session::EP_CPU, &opts, ModelSource::File(model))
            .map_err(|e| format!("the voice model does not load: {e}"))?;
        Self::from_session(session, styles, 44_100, "cpu".into()).map(|_| ())
    }

    /// JP-Extra layout (one `bert` input).
    pub fn jp_extra(&self) -> bool {
        self.jp_extra
    }

    /// Number of styles.
    pub fn style_count(&self) -> usize {
        self.styles.rows
    }

    /// Width the Japanese BERT features must have (the model's declared width, else [`BERT_DIM`]).
    #[cfg(test)]
    pub fn bert_dim(&self) -> usize {
        self.inputs.iter().find(|i| i.feed == Feed::JaBert).and_then(|i| i.width).unwrap_or(BERT_DIM)
    }

    /// Synthesize one analysed sentence. `bert` is the phone-level features (`[D, T]`).
    pub fn synthesize(&mut self, seq: &Sequence, bert: &[f32], bert_dim: usize, p: &SynthParams) -> Result<Vec<f32>, String> {
        let t = seq.phones.len();
        if seq.tones.len() != t || seq.languages.len() != t || bert.len() != bert_dim * t {
            return Err("internal error: the sentence's inputs do not line up".into());
        }
        if let Some(w) = self.inputs.iter().find(|i| i.feed == Feed::JaBert).and_then(|i| i.width).filter(|w| *w != bert_dim) {
            return Err(format!("the voice expects {w}-wide BERT features, the BERT model gives {bert_dim}"));
        }
        let style = self.styles.vector(p.style as usize, p.style_weight)?;
        let scalar = |input: &Input, v: f32| -> Result<DynValue, String> {
            let shape = if input.rank == 0 { vec![] } else { vec![1; input.rank] };
            float_tensor(input.ty, shape, vec![v])
        };
        let mut feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let value = match input.feed {
                Feed::Phones => int_tensor(input.ty, vec![1, t], seq.phones.clone())?,
                Feed::Tones => int_tensor(input.ty, vec![1, t], seq.tones.clone())?,
                Feed::Language => int_tensor(input.ty, vec![1, t], seq.languages.clone())?,
                Feed::Lengths => int_tensor(input.ty, vec![1], vec![t as i64])?,
                Feed::Sid => int_tensor(input.ty, vec![1], vec![i64::from(p.sid)])?,
                Feed::JaBert => float_tensor(input.ty, vec![1, bert_dim, t], bert.to_vec())?,
                Feed::ZeroBert => {
                    let w = input.width.unwrap_or(BERT_DIM);
                    float_tensor(input.ty, vec![1, w, t], vec![0.0; w * t])?
                }
                Feed::Style => float_tensor(input.ty, vec![1, style.len()], style.clone())?,
                Feed::LengthScale => scalar(input, p.length_scale)?,
                Feed::SdpRatio => scalar(input, p.sdp_ratio)?,
                Feed::NoiseScale => scalar(input, p.noise)?,
                Feed::NoiseScaleW => scalar(input, p.noise_w)?,
            };
            feeds.push((Cow::Owned(input.name.clone()), value.into()));
        }
        let outputs = self.session.run(feeds).map_err(ort_err("voice inference failed"))?;
        let value = outputs.get(self.output.as_str()).ok_or("the voice model returned no audio")?;
        let (_, audio) = float_output(value)?;
        // Not clamped: the caller peak-normalizes the whole sentence (as Style-Bert-VITS2 does), so a
        // clause that peaks over full scale is scaled down rather than clipped.
        Ok(audio.into_iter().map(|s| if s.is_finite() { s } else { 0.0 }).collect())
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! Stand-in graphs with the real models' inputs (no downloads).
    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};

    /// BERT: `input_ids` / `attention_mask` int64 `[1, T]` → `output` `[T, dim]` where every value
    /// of row `t` is `input_ids[t] * 0.001` (so tests can tell rows apart).
    pub fn bert(dim: usize, with_types: bool) -> Vec<u8> {
        let mut g = Graph::new();
        g.input("input_ids", INT64, &[Dim::Fixed(1), Dim::Named("T")])
            .input("attention_mask", INT64, &[Dim::Fixed(1), Dim::Named("T")]);
        if with_types {
            g.input("token_type_ids", INT64, &[Dim::Fixed(1), Dim::Named("T")]);
        }
        g.output("output", FLOAT, &[Dim::Named("T"), Dim::Fixed(dim as u64)])
            .init_f32("scale", &[], &[0.001])
            .init_f32("ones", &[1, dim as u64], &vec![1.0; dim])
            .init_i64("flat", &[2], &[-1, 1])
            .node("Cast", &["input_ids"], &["idsf"], &[Attr::Int("to", 1)])
            .node("Cast", &["attention_mask"], &["maskf"], &[Attr::Int("to", 1)])
            .node("Mul", &["idsf", "maskf"], &["masked"], &[])
            .node("Mul", &["masked", "scale"], &["scaled"], &[])
            .node("Reshape", &["scaled", "flat"], &["col"], &[])
            .node("MatMul", &["col", "ones"], &["output_pre"], &[]);
        if with_types {
            g.node("Cast", &["token_type_ids"], &["typesf"], &[Attr::Int("to", 1)])
                .node("ReduceSum", &["typesf"], &["tsum"], &[Attr::Int("keepdims", 0)])
                .node("Mul", &["tsum", "scale"], &["tzero"], &[])
                .node("Add", &["output_pre", "tzero"], &["output"], &[]);
        } else {
            g.node("Identity", &["output_pre"], &["output"], &[]);
        }
        g.model_bytes(13)
    }

    /// Voice: JP-Extra inputs (or the older layout with `ja_bert` / `en_bert`); output `[1, 1, N]`
    /// with `N = 512 * T` samples of `length_scale * 0.1 + mean(style) * 0.01` (scalars as 0-d
    /// tensors when `scalar_rank0`, else `[1]`).
    pub fn voice(style_dim: usize, multilingual: bool, scalar_rank0: bool) -> Vec<u8> {
        let mut g = Graph::new();
        let scalar: &[Dim] = if scalar_rank0 { &[] } else { &[Dim::Fixed(1)] };
        g.input("x_tst", INT64, &[Dim::Fixed(1), Dim::Named("T")])
            .input("x_tst_lengths", INT64, &[Dim::Fixed(1)])
            .input("sid", INT64, &[Dim::Fixed(1)])
            .input("tones", INT64, &[Dim::Fixed(1), Dim::Named("T")])
            .input("language", INT64, &[Dim::Fixed(1), Dim::Named("T")])
            .input("bert", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1024), Dim::Named("T")]);
        if multilingual {
            g.input("ja_bert", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1024), Dim::Named("T")])
                .input("en_bert", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1024), Dim::Named("T")]);
        }
        g.input("style_vec", FLOAT, &[Dim::Fixed(1), Dim::Fixed(style_dim as u64)])
            .input("length_scale", FLOAT, scalar)
            .input("sdp_ratio", FLOAT, scalar)
            .input("noise_scale", FLOAT, scalar)
            .input("noise_scale_w", FLOAT, scalar)
            .output("output", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1), Dim::Named("N")])
            .init_f32("tenth", &[], &[0.1])
            .init_f32("hundredth", &[], &[0.01])
            .init_f32("zero", &[], &[0.0])
            .init_i64("hop", &[1], &[512])
            .init_i64("one", &[1], &[1])
            // N = T * 512 from the phone count.
            .node("Shape", &["x_tst"], &["xshape"], &[])
            .node("Gather", &["xshape", "one"], &["tlen"], &[Attr::Int("axis", 0)])
            .node("Mul", &["tlen", "hop"], &["n"], &[])
            .node("Concat", &["one", "one", "n"], &["oshape"], &[Attr::Int("axis", 0)])
            // value = length_scale * 0.1 + mean(style) * 0.01 + 0 * (everything else)
            .node("ReduceMean", &["style_vec"], &["smean"], &[Attr::Int("keepdims", 0)])
            .node("Mul", &["smean", "hundredth"], &["sv"], &[])
            .node("Mul", &["length_scale", "tenth"], &["lv"], &[])
            .node("Add", &["sv", "lv"], &["v0"], &[])
            .node("Add", &["sdp_ratio", "noise_scale"], &["s1"], &[])
            .node("Add", &["s1", "noise_scale_w"], &["s2"], &[])
            .node("Mul", &["s2", "zero"], &["s3"], &[])
            .node("Add", &["v0", "s3"], &["v1"], &[])
            .node("ReduceSum", &["bert"], &["bsum"], &[Attr::Int("keepdims", 0)])
            .node("Mul", &["bsum", "zero"], &["bz"], &[])
            .node("Add", &["v1", "bz"], &["v2"], &[])
            .node("Cast", &["sid"], &["sidf"], &[Attr::Int("to", 1)])
            .node("ReduceSum", &["sidf"], &["sids"], &[Attr::Int("keepdims", 0)])
            .node("Mul", &["sids", "zero"], &["sz"], &[])
            .node("Add", &["v2", "sz"], &["v3"], &[])
            .node("Cast", &["tones"], &["tf"], &[Attr::Int("to", 1)])
            .node("ReduceSum", &["tf"], &["ts"], &[Attr::Int("keepdims", 0)])
            .node("Cast", &["language"], &["lf"], &[Attr::Int("to", 1)])
            .node("ReduceSum", &["lf"], &["ls"], &[Attr::Int("keepdims", 0)])
            .node("Cast", &["x_tst_lengths"], &["xlf"], &[Attr::Int("to", 1)])
            .node("ReduceSum", &["xlf"], &["xls"], &[Attr::Int("keepdims", 0)])
            .node("Add", &["ts", "ls"], &["tl"], &[])
            .node("Add", &["tl", "xls"], &["tlx"], &[])
            .node("Mul", &["tlx", "zero"], &["tz"], &[])
            .node("Add", &["v3", "tz"], &["v4"], &[]);
        let mut last = "v4";
        if multilingual {
            g.node("ReduceSum", &["ja_bert"], &["jsum"], &[Attr::Int("keepdims", 0)])
                .node("ReduceSum", &["en_bert"], &["esum"], &[Attr::Int("keepdims", 0)])
                .node("Add", &["jsum", "esum"], &["je"], &[])
                .node("Mul", &["je", "zero"], &["jz"], &[])
                .node("Add", &["v4", "jz"], &["v5"], &[]);
            last = "v5";
        }
        // Broadcast the scalar to [1, 1, N].
        g.node("Reshape", &[last, "one"], &["v1d"], &[]).node("Expand", &["v1d", "oshape"], &["output"], &[]);
        g.model_bytes(13)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::ort_session::EP_CPU;

    fn session(bytes: &[u8]) -> Session {
        build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::Memory(bytes)).expect("the stand-in loads").0
    }

    #[test]
    fn phone_level_features_repeat_and_transpose() {
        // 3 rows of width 2: row r = [r, 10r].
        let f = vec![0.0, 0.0, 1.0, 10.0, 2.0, 20.0];
        let out = phone_level_features(&f, 2, &[1, 2, 1]).unwrap();
        // [D=2, T=4]: d0 = [0,1,1,2], d1 = [0,10,10,20].
        assert_eq!(out, [0.0, 1.0, 1.0, 2.0, 0.0, 10.0, 10.0, 20.0]);
        assert!(phone_level_features(&f, 2, &[1, 1]).is_err());
    }

    #[test]
    fn bert_features_come_from_the_graphs_inputs() {
        for with_types in [false, true] {
            let mut bert = Bert::from_session(session(&fake::bert(8, with_types)), "cpu".into()).unwrap();
            let (f, dim) = bert.features(&[1, 50, 2]).unwrap();
            assert_eq!(dim, 8);
            assert_eq!(f.len(), 24);
            assert!((f[8] - 0.05).abs() < 1e-6, "row 1 is the second id");
        }
    }

    #[test]
    fn voices_of_every_layout_synthesize() {
        let seq = Sequence { phones: vec![0, 57, 0, 8, 0], tones: vec![0, 6, 0, 7, 0], languages: vec![0, 1, 0, 1, 0], word2ph: vec![3, 2] };
        let bert = vec![0.0f32; 1024 * 5];
        let styles = StyleVectors { rows: 2, width: 4, data: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] };
        for (multi, rank0) in [(false, false), (false, true), (true, false)] {
            let mut v = Voice::from_session(session(&fake::voice(4, multi, rank0)), styles.clone(), 44_100, "cpu".into()).unwrap();
            assert_eq!(v.jp_extra(), !multi);
            assert_eq!(v.bert_dim(), 1024);
            let p = SynthParams { style: 1, style_weight: 1.0, length_scale: 2.0, ..SynthParams::default() };
            let audio = v.synthesize(&seq, &bert, 1024, &p).unwrap();
            assert_eq!(audio.len(), 5 * 512);
            // length_scale 2 → 0.2; style 1 at weight 1 → mean 1 → +0.01.
            assert!(audio.iter().all(|s| (s - 0.21).abs() < 1e-5), "{multi} {rank0}: {}", audio[0]);
            // Style weight blends towards the mean (row 0 = zeros).
            let half = v.synthesize(&seq, &bert, 1024, &SynthParams { style: 1, style_weight: 0.5, length_scale: 1.0, ..p }).unwrap();
            assert!((half[0] - 0.105).abs() < 1e-5);
            assert!(v.synthesize(&seq, &bert, 1024, &SynthParams { style: 5, ..p }).is_err(), "no style 5");
            // Output over full scale is passed on unclipped (the sentence is peak-normalized later).
            let loud = v.synthesize(&seq, &bert, 1024, &SynthParams { style: 0, length_scale: 20.0, ..p }).unwrap();
            assert!(loud.iter().all(|s| (s - 2.0).abs() < 1e-4), "{}", loud[0]);
        }
    }

    #[test]
    fn foreign_graphs_are_named_for_what_they_are() {
        let styles = StyleVectors { rows: 1, width: 4, data: vec![0.0; 4] };
        // A BERT is not a voice, and a voice with the wrong style width is refused.
        let e = Voice::from_session(session(&fake::bert(8, false)), styles.clone(), 44_100, "cpu".into()).err().unwrap();
        assert!(e.contains("not a Style-Bert-VITS2 model"), "{e}");
        let e = Voice::from_session(session(&fake::voice(8, false, false)), styles, 44_100, "cpu".into()).err().unwrap();
        assert!(e.contains("4 wide"), "{e}");
        assert!(Bert::from_session(session(&fake::voice(4, false, false)), "cpu".into()).is_err());
    }
}
