//! The MeanVC2 graphs (Sigil's export, `scripts/vc-families/meanvc/export.py`, ADR 0006) behind
//! [`MeanVcGraphs`]: ONNX Runtime sessions in the engine ([`OrtGraphs`]), a pure-Rust stand-in in
//! the tests. The streaming loop (`stream.rs`) only ever talks to the trait.
//!
//! # Graph contract (checked at load; a file that does not match is refused by name)
//! | graph | inputs | outputs |
//! |---|---|---|
//! | `fbank` | `wav16k` f32 `[1,N]` (N ≥ 400) | `fbank` f32 `[1,F,80]`, F = (N − 400) / 160 + 1 |
//! | `asr` | `fbank` f32 `[1,11,80]`, `offset` i64 `[]`, `rcs` i64 `[]` (4), `att_cache` f32 `[6,4,4,128]`, `cnn_cache` f32 `[6,1,256,8]` | `bn` f32 `[1,2,256]`, `att_cache_out`, `cnn_cache_out` (the input shapes) |
//! | `gtm` | `spks` f32 `[1,256]` | `k_mem`, `v_mem` f32 `[1,32,256]` |
//! | `dit_step` | `x` f32 `[1,8,80]`, `t` / `r` / `dt` f32 `[1]`, `cond` f32 `[1,8,256]`, `spks` `[1,256]`, `k_mem` / `v_mem` `[1,32,256]`, `offset` / `n_valid` i64 `[1]`, `cache` f32 `[4,2,2,12,64]` | `x_next` f32 `[1,8,80]`, `cache_out` (the cache shape) |
//! | `vocos` | `mel` f32 `[1,80,M]` (already `(mel + 1) / 2`) | `wav` f32 `[1,160(M − 1)]` |
//!
//! The cache and memory shapes are read from the graphs (every dimension fixed), so an export with
//! another layer count still loads; the streaming constants (11-frame ASR window, 2 BN frames per
//! call, 8 DiT frames, 80 mel bins, 256-value BN and speaker vectors) are checked exactly. The state
//! shapes the loop allocates are bounded before anything is allocated ([`GraphShapes::validate`]:
//! their ranks, and at most [`MAX_STATE_VALUES`] values each), so a voice folder's graph that
//! declares an absurd cache is refused by name instead of aborting the process. A graph carrying
//! Sigil's `sigil.graph` metadata must name itself (`meanvc/<graph>`).

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::{Tensor, TensorElementType, TensorRef};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};

/// Mel bins of the DiT output and the vocoder input.
pub const MEL_BINS: usize = 80;
/// Values per BN (content) frame.
pub const BN_DIM: usize = 256;
/// Values of the speaker embedding.
pub const SPK_DIM: usize = 256;
/// Fbank frames per ASR call, and the stride between calls.
pub const ASR_WINDOW: usize = 11;
pub const ASR_STRIDE: usize = 8;
/// BN frames per ASR call (40 ms each).
pub const ASR_OUT_FRAMES: usize = 2;
/// `rcs` (required cache size) of the 80 ms Fast-U2++ encoder.
pub const ASR_RCS: i64 = 4;
/// DiT frames per call: 4 current + 4 future (10 ms each).
pub const DIT_FRAMES: usize = 8;
/// Samples per fbank frame (the Kaldi window) and per hop.
pub const FBANK_WINDOW: usize = 400;
pub const HOP: usize = 160;
/// Most values one state tensor (a cache or a GTM memory) may hold: 4 Mi floats (16 MB), over 300
/// times the 40 ms export's largest (12288).
pub const MAX_STATE_VALUES: usize = 1 << 22;

/// One graph of the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphName {
    Fbank,
    Asr,
    Gtm,
    DitStep,
    Vocos,
}

impl GraphName {
    pub const ALL: [GraphName; 5] = [GraphName::Fbank, GraphName::Asr, GraphName::Gtm, GraphName::DitStep, GraphName::Vocos];

    /// File name in the model-store entry (or a voice folder holding its own export).
    pub fn file(self) -> &'static str {
        match self {
            GraphName::Fbank => "fbank.onnx",
            GraphName::Asr => "asr.onnx",
            GraphName::Gtm => "gtm.onnx",
            GraphName::DitStep => "dit_step.onnx",
            GraphName::Vocos => "vocos.onnx",
        }
    }

    /// The export's `sigil.graph` metadata.
    fn tag(self) -> &'static str {
        match self {
            GraphName::Fbank => "meanvc/fbank",
            GraphName::Asr => "meanvc/asr",
            GraphName::Gtm => "meanvc/gtm",
            GraphName::DitStep => "meanvc/dit_step",
            GraphName::Vocos => "meanvc/vocos",
        }
    }

    /// EP outcome component (`vc` is the generator, `vc.content` the content encoder, as for RVC).
    pub fn component(self) -> &'static str {
        match self {
            GraphName::Fbank => "vc.fbank",
            GraphName::Asr => "vc.content",
            GraphName::Gtm => "vc.timbre",
            GraphName::DitStep => "vc",
            GraphName::Vocos => "vc.vocoder",
        }
    }
}

/// The fixed state shapes a graph set runs with (read from the graphs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphShapes {
    /// `asr` attention cache, e.g. `[6, 4, 4, 128]`.
    pub att_cache: Vec<usize>,
    /// `asr` convolution cache, e.g. `[6, 1, 256, 8]`.
    pub cnn_cache: Vec<usize>,
    /// `dit_step` KV cache `[layers, 2, heads, frames, dim]`, e.g. `[4, 2, 2, 12, 64]`.
    pub dit_cache: Vec<usize>,
    /// `gtm` memories (`k_mem`, `v_mem` each), e.g. `[1, 32, 256]`.
    pub memory: Vec<usize>,
}

impl GraphShapes {
    /// The shapes of the 40 ms MeanVC2 export (ADR 0006).
    #[cfg(test)]
    pub fn meanvc2_40ms() -> GraphShapes {
        GraphShapes {
            att_cache: vec![6, 4, 4, 128],
            cnn_cache: vec![6, 1, 256, 8],
            dit_cache: vec![4, 2, 2, 12, 64],
            memory: vec![1, 32, 256],
        }
    }

    /// Values of a tensor of shape `dims` (`usize::MAX` when the product overflows; a validated
    /// shape never does).
    pub fn len(dims: &[usize]) -> usize {
        checked_len(dims).unwrap_or(usize::MAX)
    }

    /// The ranks the loop indexes (`att_cache` 4, `cnn_cache` 4, `dit_cache` 5, `memory` 3), no zero
    /// dimension, and at most [`MAX_STATE_VALUES`] values per tensor. Checked before any state is
    /// allocated (`OrtGraphs::load`, `MeanVcStream::new`).
    pub fn validate(&self) -> Result<(), String> {
        for (name, dims, rank) in [
            ("asr att_cache", &self.att_cache, STATE_RANKS[0]),
            ("asr cnn_cache", &self.cnn_cache, STATE_RANKS[1]),
            ("dit_step cache", &self.dit_cache, STATE_RANKS[2]),
            ("dit_step k_mem / v_mem", &self.memory, STATE_RANKS[3]),
        ] {
            check_state_dims(name, dims, rank)?;
        }
        Ok(())
    }
}

/// Ranks of `att_cache`, `cnn_cache`, the DiT `cache` and the GTM memories.
const STATE_RANKS: [usize; 4] = [4, 4, 5, 3];

fn checked_len(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
}

/// One state tensor's shape against its rank and [`MAX_STATE_VALUES`].
fn check_state_dims(name: &str, dims: &[usize], rank: usize) -> Result<(), String> {
    if dims.len() != rank {
        return Err(format!("MeanVC2 {name}: rank {} ({dims:?}), the export's is {rank}", dims.len()));
    }
    match checked_len(dims) {
        Some(n) if n > 0 && n <= MAX_STATE_VALUES => Ok(()),
        _ => Err(format!(
            "MeanVC2 {name}: the shape {dims:?} is not a usable state (1 to {MAX_STATE_VALUES} values); the graph is not a MeanVC2 export"
        )),
    }
}

/// Inputs of one mean-flow step besides the speaker (`stream.rs` owns the buffers).
pub struct DitInputs<'a> {
    /// `[8, 80]` frame-major.
    pub x: &'a [f32],
    pub t: f32,
    pub r: f32,
    pub dt: f32,
    /// `[8, 256]` frame-major.
    pub cond: &'a [f32],
    pub offset: i64,
    pub n_valid: i64,
    /// The KV cache before this call (every step of one call reads the same one).
    pub cache: &'a [f32],
}

/// The speaker conditioning of a voice: its embedding and the GTM memories computed from it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerState {
    pub spks: Vec<f32>,
    pub k_mem: Vec<f32>,
    pub v_mem: Vec<f32>,
}

/// The numeric core the streaming loop drives.
pub trait MeanVcGraphs: Send {
    fn shapes(&self) -> &GraphShapes;
    /// Kaldi fbank of `wav` (16 kHz, −1..1, at least [`FBANK_WINDOW`] samples): appends
    /// `F × 80` values to `out` and returns `F`.
    fn fbank(&mut self, wav: &[f32], out: &mut Vec<f32>) -> Result<usize, String>;
    /// One ASR call over `fbank` (`11 × 80`): writes `2 × 256` BN values to `bn` and replaces both
    /// caches with the graph's.
    fn asr(&mut self, fbank: &[f32], offset: i64, att: &mut [f32], cnn: &mut [f32], bn: &mut [f32]) -> Result<(), String>;
    /// The GTM memories of a speaker embedding.
    fn gtm(&mut self, spks: &[f32], k_mem: &mut [f32], v_mem: &mut [f32]) -> Result<(), String>;
    /// One mean-flow step: `x_next` (`8 × 80`) and the KV cache after the call.
    fn dit_step(&mut self, inputs: &DitInputs<'_>, speaker: &SpeakerState, x_next: &mut [f32], cache_out: &mut [f32]) -> Result<(), String>;
    /// Vocos over `frames` mel frames (`mel` channel-major `80 × frames`, already `(mel + 1) / 2`):
    /// replaces `out` with the `160 (frames − 1)` samples.
    fn vocos(&mut self, mel: &[f32], frames: usize, out: &mut Vec<f32>) -> Result<(), String>;
}

/// Where a graph set comes from.
pub enum GraphSource<'a> {
    /// A folder holding the five files ([`GraphName::file`]).
    Dir(&'a Path),
    /// In-memory models in [`GraphName::ALL`] order (tests).
    #[cfg(test)]
    Memory(&'a [Vec<u8>; 5]),
}

/// The five graphs of a folder that are missing (empty when the set is complete).
pub fn missing_files(dir: &Path) -> Vec<&'static str> {
    GraphName::ALL
        .iter()
        .map(|g| g.file())
        .filter(|f| !dir.join(f).is_file())
        .collect()
}

/// The graph set on ONNX Runtime (CPU EP; see `meanvc.rs` for why).
pub struct OrtGraphs {
    fbank: Session,
    asr: Session,
    gtm: Session,
    dit: Session,
    vocos: Session,
    shapes: GraphShapes,
}

/// The fixed shape of a state input (rank `rank`, bounded by [`MAX_STATE_VALUES`]; nothing is
/// allocated from a shape this refuses).
fn state_dims(session: &Session, name: &str, label: &str, rank: usize) -> Result<Vec<usize>, String> {
    let dims = fixed_dims(session, name, label, true)?;
    check_state_dims(&format!("{label} {name}"), &dims, rank)?;
    Ok(dims)
}

fn fixed_dims(session: &Session, name: &str, label: &str, input: bool) -> Result<Vec<usize>, String> {
    let dims = if input {
        session
            .inputs()
            .iter()
            .find(|i| i.name() == name)
            .and_then(|i| i.dtype().tensor_shape().map(|s| s.to_vec()))
    } else {
        session
            .outputs()
            .iter()
            .find(|o| o.name() == name)
            .and_then(|o| o.dtype().tensor_shape().map(|s| s.to_vec()))
    }
    .ok_or_else(|| format!("MeanVC2 {label}: no {} `{name}`", if input { "input" } else { "output" }))?;
    if dims.is_empty() || dims.iter().any(|&d| d <= 0) {
        return Err(format!("MeanVC2 {label}: `{name}` must have a fixed shape, the graph declares {dims:?}"));
    }
    Ok(dims.iter().map(|&d| d as usize).collect())
}

/// Every declared input with its element type and declared shape (−1 = symbolic).
fn check_inputs(session: &Session, label: &str, want: &[(&str, TensorElementType, &[i64])]) -> Result<(), String> {
    if session.inputs().len() != want.len() {
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        return Err(format!(
            "MeanVC2 {label}: inputs {names:?} are not the export's (is this the right file?)"
        ));
    }
    for (name, ty, shape) in want {
        let input = session
            .inputs()
            .iter()
            .find(|i| i.name() == *name)
            .ok_or_else(|| format!("MeanVC2 {label}: the graph has no input `{name}` (is this the right file?)"))?;
        if input.dtype().tensor_type() != Some(*ty) {
            return Err(format!("MeanVC2 {label}: input `{name}` is not {ty:?}"));
        }
        let declared = input
            .dtype()
            .tensor_shape()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let fits = declared.len() == shape.len()
            && declared
                .iter()
                .zip(shape.iter())
                .all(|(&d, &s)| s < 0 || d < 0 || d == s);
        if !fits {
            return Err(format!("MeanVC2 {label}: input `{name}` has shape {declared:?}, expected {shape:?}"));
        }
    }
    Ok(())
}

fn check_outputs(session: &Session, label: &str, names: &[&str]) -> Result<(), String> {
    for name in names {
        if !session.outputs().iter().any(|o| o.name() == *name) {
            return Err(format!("MeanVC2 {label}: the graph has no output `{name}`"));
        }
    }
    Ok(())
}

fn check_tag(session: &Session, name: GraphName) -> Result<(), String> {
    let Ok(meta) = session.metadata() else { return Ok(()) };
    match meta.custom("sigil.graph") {
        Some(tag) if !tag.is_empty() && tag != name.tag() => Err(format!("MeanVC2 {}: the file is {tag}, not {}", name.file(), name.tag())),
        _ => Ok(()),
    }
}

const F32: TensorElementType = TensorElementType::Float32;
const I64: TensorElementType = TensorElementType::Int64;

impl OrtGraphs {
    /// Build the five sessions (`opts` as `meanvc.rs` sets them: CPU, one thread) and check every
    /// graph against the contract. Returns the EP outcome of each.
    pub fn load(source: &GraphSource<'_>, opts: &VcSessionOptions) -> Result<(OrtGraphs, Vec<(String, EpOutcome)>), String> {
        let mut outcomes = Vec::with_capacity(5);
        let mut open = |name: GraphName| -> Result<Session, String> {
            let path: PathBuf;
            let model = match source {
                GraphSource::Dir(dir) => {
                    path = dir.join(name.file());
                    ModelSource::File(&path)
                }
                #[cfg(test)]
                GraphSource::Memory(models) => {
                    let i = GraphName::ALL.iter().position(|&g| g == name).unwrap_or(0);
                    ModelSource::Memory(&models[i])
                }
            };
            let (session, outcome) =
                build_session(Some(name.component()), "cpu", opts, model).map_err(|e| format!("MeanVC2 {}: {e}", name.file()))?;
            check_tag(&session, name)?;
            outcomes.push((name.component().to_string(), outcome));
            Ok(session)
        };
        let fbank = open(GraphName::Fbank)?;
        let asr = open(GraphName::Asr)?;
        let gtm = open(GraphName::Gtm)?;
        let dit = open(GraphName::DitStep)?;
        let vocos = open(GraphName::Vocos)?;

        check_inputs(&fbank, "fbank", &[("wav16k", F32, &[1, -1])])?;
        check_outputs(&fbank, "fbank", &["fbank"])?;
        let att = state_dims(&asr, "att_cache", "asr", STATE_RANKS[0])?;
        let cnn = state_dims(&asr, "cnn_cache", "asr", STATE_RANKS[1])?;
        check_inputs(
            &asr,
            "asr",
            &[
                ("fbank", F32, &[1, ASR_WINDOW as i64, MEL_BINS as i64]),
                ("offset", I64, &[]),
                ("rcs", I64, &[]),
                ("att_cache", F32, &[-1, -1, -1, -1]),
                ("cnn_cache", F32, &[-1, -1, -1, -1]),
            ],
        )?;
        check_outputs(&asr, "asr", &["bn", "att_cache_out", "cnn_cache_out"])?;
        let memory = state_dims(&dit, "k_mem", "dit_step", STATE_RANKS[3])?;
        let dit_cache = state_dims(&dit, "cache", "dit_step", STATE_RANKS[2])?;
        check_inputs(&gtm, "gtm", &[("spks", F32, &[1, SPK_DIM as i64])])?;
        check_outputs(&gtm, "gtm", &["k_mem", "v_mem"])?;
        let mem: Vec<i64> = memory.iter().map(|&d| d as i64).collect();
        let cache: Vec<i64> = dit_cache.iter().map(|&d| d as i64).collect();
        let (n, m, b, s) = (DIT_FRAMES as i64, MEL_BINS as i64, BN_DIM as i64, SPK_DIM as i64);
        check_inputs(
            &dit,
            "dit_step",
            &[
                ("x", F32, &[1, n, m]),
                ("t", F32, &[1]),
                ("r", F32, &[1]),
                ("dt", F32, &[1]),
                ("cond", F32, &[1, n, b]),
                ("spks", F32, &[1, s]),
                ("k_mem", F32, &mem),
                ("v_mem", F32, &mem),
                ("offset", I64, &[1]),
                ("n_valid", I64, &[1]),
                ("cache", F32, &cache),
            ],
        )?;
        check_outputs(&dit, "dit_step", &["x_next", "cache_out"])?;
        check_inputs(&vocos, "vocos", &[("mel", F32, &[1, m, -1])])?;
        check_outputs(&vocos, "vocos", &["wav"])?;
        let shapes = GraphShapes {
            att_cache: att,
            cnn_cache: cnn,
            dit_cache,
            memory,
        };
        shapes.validate()?;
        Ok((
            OrtGraphs {
                fbank,
                asr,
                gtm,
                dit,
                vocos,
                shapes,
            },
            outcomes,
        ))
    }
}

fn copy_out(dst: &mut [f32], src: &[f32], what: &str) -> Result<(), String> {
    if src.len() != dst.len() {
        return Err(format!("MeanVC2 {what}: {} values, expected {}", src.len(), dst.len()));
    }
    dst.copy_from_slice(src);
    Ok(())
}

fn err(graph: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("MeanVC2 {graph}: {e}")
}

fn shape_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&d| d as i64).collect()
}

impl MeanVcGraphs for OrtGraphs {
    fn shapes(&self) -> &GraphShapes {
        &self.shapes
    }

    fn fbank(&mut self, wav: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        if wav.len() < FBANK_WINDOW {
            return Ok(0);
        }
        let input = TensorRef::from_array_view(([1usize, wav.len()], wav)).map_err(err("fbank"))?;
        let outputs = self
            .fbank
            .run(ort::inputs!["wav16k" => input])
            .map_err(err("fbank"))?;
        let (shape, data) = outputs["fbank"]
            .try_extract_tensor::<f32>()
            .map_err(err("fbank"))?;
        let frames = (wav.len() - FBANK_WINDOW) / HOP + 1;
        if shape.len() != 3 || shape[2] != MEL_BINS as i64 || data.len() != frames * MEL_BINS {
            return Err(format!(
                "MeanVC2 fbank: output shape {:?} for {} samples, expected [1, {frames}, {MEL_BINS}]",
                shape.to_vec(),
                wav.len()
            ));
        }
        out.extend_from_slice(data);
        Ok(frames)
    }

    fn asr(&mut self, fbank: &[f32], offset: i64, att: &mut [f32], cnn: &mut [f32], bn: &mut [f32]) -> Result<(), String> {
        let fb = TensorRef::from_array_view(([1usize, ASR_WINDOW, MEL_BINS], fbank)).map_err(err("asr"))?;
        let off = Tensor::<i64>::from_array(((), vec![offset])).map_err(err("asr"))?;
        let rcs = Tensor::<i64>::from_array(((), vec![ASR_RCS])).map_err(err("asr"))?;
        let att_in = TensorRef::from_array_view((shape_i64(&self.shapes.att_cache), &*att)).map_err(err("asr"))?;
        let cnn_in = TensorRef::from_array_view((shape_i64(&self.shapes.cnn_cache), &*cnn)).map_err(err("asr"))?;
        let outputs = self
            .asr
            .run(ort::inputs!["fbank" => fb, "offset" => off, "rcs" => rcs, "att_cache" => att_in, "cnn_cache" => cnn_in])
            .map_err(err("asr"))?;
        let (_, b) = outputs["bn"]
            .try_extract_tensor::<f32>()
            .map_err(err("asr"))?;
        copy_out(bn, b, "asr bn")?;
        let (_, a) = outputs["att_cache_out"]
            .try_extract_tensor::<f32>()
            .map_err(err("asr"))?;
        copy_out(att, a, "asr att_cache_out")?;
        let (_, c) = outputs["cnn_cache_out"]
            .try_extract_tensor::<f32>()
            .map_err(err("asr"))?;
        copy_out(cnn, c, "asr cnn_cache_out")
    }

    fn gtm(&mut self, spks: &[f32], k_mem: &mut [f32], v_mem: &mut [f32]) -> Result<(), String> {
        let s = TensorRef::from_array_view(([1usize, SPK_DIM], spks)).map_err(err("gtm"))?;
        let outputs = self
            .gtm
            .run(ort::inputs!["spks" => s])
            .map_err(err("gtm"))?;
        let (_, k) = outputs["k_mem"]
            .try_extract_tensor::<f32>()
            .map_err(err("gtm"))?;
        copy_out(k_mem, k, "gtm k_mem")?;
        let (_, v) = outputs["v_mem"]
            .try_extract_tensor::<f32>()
            .map_err(err("gtm"))?;
        copy_out(v_mem, v, "gtm v_mem")
    }

    fn dit_step(&mut self, inputs: &DitInputs<'_>, speaker: &SpeakerState, x_next: &mut [f32], cache_out: &mut [f32]) -> Result<(), String> {
        let mem = shape_i64(&self.shapes.memory);
        let x = TensorRef::from_array_view(([1usize, DIT_FRAMES, MEL_BINS], inputs.x)).map_err(err("dit_step"))?;
        let t = Tensor::<f32>::from_array(([1usize], vec![inputs.t])).map_err(err("dit_step"))?;
        let r = Tensor::<f32>::from_array(([1usize], vec![inputs.r])).map_err(err("dit_step"))?;
        let dt = Tensor::<f32>::from_array(([1usize], vec![inputs.dt])).map_err(err("dit_step"))?;
        let cond = TensorRef::from_array_view(([1usize, DIT_FRAMES, BN_DIM], inputs.cond)).map_err(err("dit_step"))?;
        let spks = TensorRef::from_array_view(([1usize, SPK_DIM], speaker.spks.as_slice())).map_err(err("dit_step"))?;
        let k_mem = TensorRef::from_array_view((mem.clone(), speaker.k_mem.as_slice())).map_err(err("dit_step"))?;
        let v_mem = TensorRef::from_array_view((mem, speaker.v_mem.as_slice())).map_err(err("dit_step"))?;
        let offset = Tensor::<i64>::from_array(([1usize], vec![inputs.offset])).map_err(err("dit_step"))?;
        let n_valid = Tensor::<i64>::from_array(([1usize], vec![inputs.n_valid])).map_err(err("dit_step"))?;
        let cache = TensorRef::from_array_view((shape_i64(&self.shapes.dit_cache), inputs.cache)).map_err(err("dit_step"))?;
        let outputs = self
            .dit
            .run(ort::inputs![
                "x" => x, "t" => t, "r" => r, "dt" => dt, "cond" => cond, "spks" => spks,
                "k_mem" => k_mem, "v_mem" => v_mem, "offset" => offset, "n_valid" => n_valid, "cache" => cache
            ])
            .map_err(err("dit_step"))?;
        let (_, xn) = outputs["x_next"]
            .try_extract_tensor::<f32>()
            .map_err(err("dit_step"))?;
        copy_out(x_next, xn, "dit_step x_next")?;
        let (_, c) = outputs["cache_out"]
            .try_extract_tensor::<f32>()
            .map_err(err("dit_step"))?;
        copy_out(cache_out, c, "dit_step cache_out")
    }

    fn vocos(&mut self, mel: &[f32], frames: usize, out: &mut Vec<f32>) -> Result<(), String> {
        let input = TensorRef::from_array_view(([1usize, MEL_BINS, frames], mel)).map_err(err("vocos"))?;
        let outputs = self
            .vocos
            .run(ort::inputs!["mel" => input])
            .map_err(err("vocos"))?;
        let (_, wav) = outputs["wav"]
            .try_extract_tensor::<f32>()
            .map_err(err("vocos"))?;
        let want = HOP * frames.saturating_sub(1);
        if wav.len() != want {
            return Err(format!("MeanVC2 vocos: {} samples for {frames} frames, expected {want}", wav.len()));
        }
        out.clear();
        out.extend_from_slice(wav);
        Ok(())
    }
}
