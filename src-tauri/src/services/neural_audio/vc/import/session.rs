//! ONNX Runtime sessions for template graphs with the weights supplied from memory.
//!
//! The graphs reference their weights as ONNX external data (`location = weights.f32.bin` /
//! `weights.f16.bin`, offset, length). ONNX Runtime's `AddExternalInitializersFromFilesInMemory`
//! (ort `with_external_initializer_file_in_memory`) resolves those references against a buffer
//! instead of a file and copies each tensor out during session creation, so the buffer is dropped
//! with the builder. Only the prefix of the image a graph uses is handed over (the layout orders
//! `enc` tensors first). Free dimensions (`T`, `F`, `S`, `N`) can be pinned per session, which lets
//! ONNX Runtime fold the index math and plan fixed-shape memory (and CUDA graphs, P3).
//!
//! Sessions are built by `ort_session::builder_for_ep` (explicit EP, spinning off, the shared
//! options); the EP ladder with in-memory weights is P3-C2's (see the P1-G3 report). A variant
//! whose `layout.json` parity is `fail` on the target's device class is refused
//! ([`ErrorCode::PrecisionRefused`]) unless the target is marked `experimental`.

use std::borrow::Cow;

use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;

use crate::services::neural_audio::ort_session::{builder_for_ep, VcSessionOptions};

use super::layout::{device_class, Parity, Template};
use super::{ErrorCode, ImportError};

/// Graph precision variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    /// fp16 inside, fp32 inputs and outputs; weights from `weights.f16.bin`.
    Fp16,
}

impl Precision {
    pub fn variant(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
        }
    }
}

/// Weights images of one voice (the f16 one is derived on demand).
pub struct WeightImages<'a> {
    pub f32: &'a [u8],
    pub f16: Option<&'a [u8]>,
}

/// Where a session runs: an explicit EP (no ladder) on a device, with the shared VC options.
#[derive(Clone, Copy, Debug)]
pub struct EpTarget<'a> {
    pub ep: &'a str,
    pub device: i32,
    pub opts: &'a VcSessionOptions,
    /// Also run variants that fail the parity gate on this device class (parity measurements and
    /// benchmarks only; never for a voice the user hears).
    pub experimental: bool,
}

impl<'a> EpTarget<'a> {
    pub fn new(ep: &'a str, device: i32, opts: &'a VcSessionOptions) -> EpTarget<'a> {
        EpTarget { ep, device, opts, experimental: false }
    }
}

/// Build a session for `graph` (`enc`, `flowdec`, `knn`) of `template` on `target`, with free
/// dimensions pinned by `dims` (e.g. `[("T", 100)]`).
pub fn template_session(
    template: &Template,
    graph: &str,
    precision: Precision,
    weights: Option<&WeightImages<'_>>,
    target: EpTarget<'_>,
    dims: &[(&str, i64)],
) -> Result<Session, ImportError> {
    let layout = &template.layout;
    let variant = layout.graph_variant(graph, precision.variant())?;
    if layout.parity(graph, precision.variant(), target.ep) == Parity::Fail && !target.experimental {
        let class = device_class(target.ep);
        return Err(ImportError::new(
            ErrorCode::PrecisionRefused,
            format!(
                "{graph}.{} of {} fails the parity gate on {class} execution providers (docs/adr/0004-rvc-templates.md); use the fp32 variant",
                precision.variant(),
                layout.template
            ),
        ));
    }
    let model = template.graph_bytes(graph, precision.variant())?;
    let fail = |what: &str, e: String| ImportError::new(ErrorCode::Session, format!("{graph}.{}: {what}: {e}", precision.variant()));
    let mut builder = builder_for_ep(target.ep, target.device, target.opts).map_err(|e| fail("execution provider", e))?;
    if let Some(file) = &variant.weights_file {
        let w = weights.ok_or_else(|| fail("weights", "the graph needs weights".into()))?;
        let extent = layout.graph_extent_f32(graph);
        let (image, need) = match precision {
            Precision::Fp32 => (w.f32, extent),
            Precision::Fp16 => (w.f16.ok_or_else(|| fail("weights", "no fp16 image".into()))?, extent / 2),
        };
        let need = need as usize;
        if image.len() < need {
            return Err(fail("weights", format!("image of {} bytes, the graph reads {need}", image.len())));
        }
        builder = builder
            .with_external_initializer_file_in_memory(file.as_str(), Cow::Owned(image[..need].to_vec()))
            .map_err(|e| fail("external weights", e.to_string()))?;
    }
    for (name, size) in dims {
        builder = builder
            .with_dimension_override(*name, *size)
            .map_err(|e| fail("dimension override", e.to_string()))?;
    }
    let session = builder.commit_from_memory(&model).map_err(|e| fail("load", e.to_string()))?;
    let meta = session.metadata().map_err(|e| fail("metadata", e.to_string()))?;
    let tag = |k: &str| meta.custom(k).unwrap_or_default();
    if tag("sigil.template") != layout.template || tag("sigil.graph") != graph {
        return Err(fail("identity", format!("the file is {}/{}", tag("sigil.template"), tag("sigil.graph"))));
    }
    if variant.weights_file.is_some() && tag("sigil.weights_id") != layout.weights.weights_id {
        return Err(fail("identity", "graph and layout.json come from different exports".into()));
    }
    drop(meta);
    Ok(session)
}

/// Result of [`smoke_inference`].
#[derive(Clone, Debug, serde::Serialize)]
pub struct SmokeReport {
    pub precision: &'static str,
    /// Time to build both sessions and run them once on the CPU.
    pub ms: u64,
    pub frames: usize,
    /// Output level of the smoke audio (finite and not all zero is the pass condition).
    pub rms: f32,
    pub peak: f32,
}

fn tensor<'a, T: ort::value::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static>(
    name: &str,
    shape: Vec<usize>,
    data: &'a [T],
) -> Result<(String, SessionInputValue<'a>), ImportError> {
    let t = TensorRef::from_array_view((shape, data))
        .map_err(|e| ImportError::new(ErrorCode::Session, format!("smoke test input {name}: {e}")))?;
    Ok((name.to_string(), t.into()))
}

/// Build the `enc` and `flowdec` sessions of an imported voice on the CPU and decode a short
/// deterministic input: catches weights that load but produce NaN or silence before the voice is
/// offered to the user (fp16: activations that overflow half precision, on ONNX Runtime's CPU
/// fp16 path). Background work: two intra-op threads.
pub fn smoke_inference(voice: &super::ImportedVoice, precision: Precision) -> Result<SmokeReport, ImportError> {
    let started = std::time::Instant::now();
    let layout = &voice.template.layout;
    let arch = &layout.arch;
    let st = &layout.streaming;
    let (emb, c, upp) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize);
    let dim = voice.weights.speaker_dim;
    let margin = (st.flow_margin + st.dec_margin) as usize;
    let frames = margin + 8;
    let opts = VcSessionOptions { intra_threads: Some(2), ..VcSessionOptions::default() };
    let target = EpTarget::new("cpu", 0, &opts);
    let f16 = match precision {
        Precision::Fp32 => None,
        Precision::Fp16 => Some(super::writer::f16_image(&voice.weights.f32_image).map_err(|e| ImportError::new(ErrorCode::NonFinite, e))?),
    };
    let images = WeightImages { f32: &voice.weights.f32_image, f16: f16.as_deref() };
    let fail = |what: &str, e: String| ImportError::new(ErrorCode::Session, format!("smoke test ({what}): {e}"));

    let phone: Vec<f32> = (0..frames * emb)
        .map(|i| 0.5 * (0.37 * (i / emb) as f32 + 0.11 * (i % emb) as f32).sin())
        .collect();
    let pitch = vec![60i64; frames];
    let mut enc = template_session(&voice.template, "enc", precision, Some(&images), target, &[("T", frames as i64)])?;
    let mut feed = vec![tensor("phone", vec![1, frames, emb], &phone)?];
    if arch.f0 {
        feed.push(tensor("pitch", vec![1, frames], &pitch)?);
    }
    let (m_p, logs_p) = {
        let out = enc.run(feed).map_err(|e| fail("enc", e.to_string()))?;
        let m = out[0].try_extract_tensor::<f32>().map_err(|e| fail("enc", e.to_string()))?.1.to_vec();
        let l = out[1].try_extract_tensor::<f32>().map_err(|e| fail("enc", e.to_string()))?.1.to_vec();
        (m, l)
    };
    let z_noise: Vec<f32> = (0..c * frames)
        .map(|i| st.noise_scale * (1.3 * (i / frames) as f32 + 0.7 * (i % frames) as f32).sin())
        .collect();
    let g = voice.weights.speakers[..dim].to_vec();
    let f0 = vec![150.0f32; frames];
    let phase0 = [0.0f32];
    let src_noise = vec![0.0f32; frames * upp];
    let dims = [("F", frames as i64), ("S", (frames * upp) as i64)];
    let mut fd = template_session(&voice.template, "flowdec", precision, Some(&images), target, &dims)?;
    let feed = vec![
        tensor("m_p", vec![1, c, frames], &m_p)?,
        tensor("logs_p", vec![1, c, frames], &logs_p)?,
        tensor("z_noise", vec![1, c, frames], &z_noise)?,
        tensor("g", vec![1, dim, 1], &g)?,
        tensor("f0", vec![1, frames], &f0)?,
        tensor("phase0", vec![1, 1], &phase0)?,
        tensor("src_noise", vec![1, frames * upp], &src_noise)?,
    ];
    let out = fd.run(feed).map_err(|e| fail("flowdec", e.to_string()))?;
    let audio = out[0].try_extract_tensor::<f32>().map_err(|e| fail("flowdec", e.to_string()))?.1;
    let expect = (frames - margin) * upp;
    if audio.len() != expect {
        return Err(fail("flowdec", format!("{} samples, expected {expect}", audio.len())));
    }
    if audio.iter().any(|x| !x.is_finite()) {
        return Err(fail("flowdec", "the voice produces non-finite audio".into()));
    }
    let rms = (audio.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / audio.len() as f64).sqrt() as f32;
    let peak = audio.iter().fold(0f32, |a, &x| a.max(x.abs()));
    if rms <= 1e-7 {
        return Err(fail("flowdec", "the voice produces silence".into()));
    }
    Ok(SmokeReport { precision: precision.variant(), ms: started.elapsed().as_millis() as u64, frames, rms, peak })
}
