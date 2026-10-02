//! Chatterbox Multilingual V3's language model (the "T3" Llama backbone, 30 layers) as KitsuMate's
//! split-speed export ships it: `static_language_model_int8.onnx`, ONNX Runtime's fused
//! `GroupQueryAttention` with a fixed-size KV cache the caller owns.
//!
//! - **Inputs:** `inputs_embeds [B, S, H]`, `seqlens_k [B]` (int32, total length − 1),
//!   `total_sequence_length` (int32 scalar), `past_key_values.<l>.key/value [B, kv_heads,
//!   capacity, head_dim]`. **Outputs:** `logits [B, vocab]` (last position only),
//!   `present.<l>.key/value` (same shape as the past).
//! - **In-place cache.** Every layer's past and present are bound (IOBinding) to the same tensor, so
//!   ONNX Runtime's attention kernel sees one buffer and appends the new keys / values in place
//!   (`past_present_share_buffer`): no copy of the cache per token. The tensors live on the device
//!   the session runs on (CUDA / DirectML memory from the session's allocator, host memory on the
//!   CPU), are allocated for the batch and length one sentence needs and reused while they fit.
//! - **Batch.** Two rows when classifier-free guidance is on (row 1 is the text-free row), else one.
//! - The layer count, head layout, hidden width and vocabulary are read from the session, so the
//!   driver also runs the small stand-in graphs of the tests.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::{DynValue, Tensor, ValueType};
use ort::AsPointer;

use super::embed::SpeechEmbedding;
use super::sampling::{next_token, Rng, SamplingParams};
use super::text::{START_SPEECH, STOP_SPEECH};
use crate::services::neural_audio::ort_session::{EP_CPU, EP_CUDA, EP_DML};

/// Cache lengths are allocated in steps of this many positions.
const CAPACITY_STEP: usize = 128;

fn status(ptr: ort::sys::OrtStatusPtr) -> Result<(), String> {
    // SAFETY: a status just returned by an ORT API call (null = success); ownership moves into
    // `result_from_status`, which releases it.
    unsafe { ort::Error::result_from_status(ptr) }.map_err(|e| e.to_string())
}

fn c_name(name: &str) -> Result<CString, String> {
    CString::new(name).map_err(|_| format!("tensor name {name:?} contains a NUL byte"))
}

/// Bind `value` to the session input `name` (ORT copies a host value to the device the input is
/// read on now; a value already there is referenced).
fn bind_input<T: ort::value::ValueTypeMarker + ?Sized>(binding: &mut IoBinding, name: &CString, value: &ort::value::Value<T>) -> Result<(), String> {
    // SAFETY: a live binding, a NUL-terminated name and a live OrtValue (ORT keeps its own reference).
    status(unsafe { (ort::api().BindInput)(binding.ptr_mut(), name.as_ptr(), value.ptr()) })
}

/// Bind the session output `name` to the pre-allocated `value` (ORT writes the result into it).
fn bind_output<T: ort::value::ValueTypeMarker + ?Sized>(binding: &mut IoBinding, name: &CString, value: &ort::value::Value<T>) -> Result<(), String> {
    // SAFETY: as in `bind_input`.
    status(unsafe { (ort::api().BindOutput)(binding.ptr_mut(), name.as_ptr(), value.ptr()) })
}

fn run_binding(session: &mut Session, binding: &IoBinding) -> Result<(), String> {
    // SAFETY: a live session and a binding created from it; `&mut Session` makes the call exclusive.
    status(unsafe { (ort::api().RunWithBinding)(session.ptr_mut(), std::ptr::null(), binding.ptr()) })
}

/// The dimensions of a session input (`-1` for symbolic ones).
fn dims_of(session: &Session, name: &str) -> Option<Vec<i64>> {
    session
        .inputs()
        .iter()
        .find(|i| i.name() == name)
        .and_then(|i| match i.dtype() {
            ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        })
}

fn output_dims_of(session: &Session, name: &str) -> Option<Vec<i64>> {
    session
        .outputs()
        .iter()
        .find(|o| o.name() == name)
        .and_then(|o| match o.dtype() {
            ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        })
}

/// Whether the session reads input `name` from GPU memory: where ONNX Runtime placed the node
/// that consumes it (ONNX Runtime's CUDA attention kernel takes fp16 / bf16 only, so on CUDA this
/// fp32 graph's attention runs on the CPU while its matrix products run on the GPU).
fn input_on_gpu(session: &Session, name: &str) -> Result<bool, String> {
    use ort::sys::{OrtMemoryInfo, OrtMemoryInfoDeviceType};
    let inputs = session.inputs();
    let index = inputs
        .iter()
        .position(|i| i.name() == name)
        .ok_or_else(|| format!("no input {name}"))?;
    let mut infos: Vec<*const OrtMemoryInfo> = vec![std::ptr::null(); inputs.len()];
    // SAFETY: a live session and room for its inputs' memory infos (owned by the session, never
    // released here).
    status(unsafe { (ort::api().SessionGetMemoryInfoForInputs)(session.ptr(), infos.as_mut_ptr(), inputs.len()) })?;
    let info = infos[index];
    if info.is_null() {
        return Ok(false);
    }
    let mut ty = OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
    // SAFETY: a valid memory info of the session; writes one enum value.
    unsafe { (ort::api().MemoryInfoGetDeviceType)(info, &mut ty) };
    Ok(ty == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU)
}

/// The cache of one sentence's batch and length. Fields drop in order: the tensors, then the
/// allocator they came from (ONNX Runtime frees a tensor through it).
struct KvCache {
    batch: usize,
    capacity: usize,
    keys: Vec<DynValue>,
    values: Vec<DynValue>,
    _allocator: Allocator,
}

/// The language model and its cache (see the module docs). The cache is declared first so it is
/// freed before the session whose allocator made it.
pub struct StaticLm {
    kv: Option<KvCache>,
    session: Session,
    /// EP the session runs on (`cuda` / `dml` / `cpu`).
    ep: &'static str,
    device_id: i32,
    /// The attention reads the cache from GPU memory (else host memory).
    kv_on_gpu: bool,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    vocab: usize,
    names: Names,
}

struct Names {
    inputs_embeds: CString,
    seqlens_k: CString,
    total: CString,
    logits: CString,
    past_k: Vec<CString>,
    past_v: Vec<CString>,
    present_k: Vec<CString>,
    present_v: Vec<CString>,
}

/// What [`StaticLm::generate`] produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generated {
    /// Sampled speech tokens, the stop token excluded.
    pub tokens: Vec<i64>,
    /// The model emitted its stop token (else it hit the length limit).
    pub stopped: bool,
}

impl StaticLm {
    /// Wrap a loaded session running on `ep` (device `device_id` for GPU EPs).
    pub fn new(session: Session, ep: &str, device_id: i32) -> Result<Self, String> {
        let ep = match ep {
            EP_CUDA => EP_CUDA,
            EP_DML => EP_DML,
            _ => EP_CPU,
        };
        let embeds = dims_of(&session, "inputs_embeds").ok_or("the language model has no inputs_embeds input")?;
        let hidden = embeds
            .get(2)
            .copied()
            .filter(|&h| h > 0)
            .ok_or("the language model's hidden width is not fixed")? as usize;
        for name in ["seqlens_k", "total_sequence_length"] {
            dims_of(&session, name).ok_or_else(|| format!("the language model has no {name} input (not a static-cache export)"))?;
        }
        let logits = output_dims_of(&session, "logits").ok_or("the language model has no logits output")?;
        let vocab = logits
            .last()
            .copied()
            .filter(|&v| v > 0)
            .ok_or("the language model's vocabulary is not fixed")? as usize;
        if vocab <= STOP_SPEECH as usize {
            return Err(format!("the language model's vocabulary ({vocab}) has no speech stop token"));
        }
        let mut layers = 0;
        while dims_of(&session, &format!("past_key_values.{layers}.key")).is_some() {
            layers += 1;
        }
        if layers == 0 {
            return Err("the language model has no KV-cache inputs".into());
        }
        let kv = dims_of(&session, "past_key_values.0.key").unwrap_or_default();
        let (kv_heads, head_dim) = match kv.as_slice() {
            [_, h, _, d] if *h > 0 && *d > 0 => (*h as usize, *d as usize),
            _ => return Err(format!("unexpected KV-cache shape {kv:?}")),
        };
        let mut names = Names {
            inputs_embeds: c_name("inputs_embeds")?,
            seqlens_k: c_name("seqlens_k")?,
            total: c_name("total_sequence_length")?,
            logits: c_name("logits")?,
            past_k: Vec::with_capacity(layers),
            past_v: Vec::with_capacity(layers),
            present_k: Vec::with_capacity(layers),
            present_v: Vec::with_capacity(layers),
        };
        for l in 0..layers {
            names
                .past_k
                .push(c_name(&format!("past_key_values.{l}.key"))?);
            names
                .past_v
                .push(c_name(&format!("past_key_values.{l}.value"))?);
            for (kind, list) in [("key", &mut names.present_k), ("value", &mut names.present_v)] {
                let out = format!("present.{l}.{kind}");
                if output_dims_of(&session, &out).is_none() {
                    return Err(format!("the language model has no {out} output"));
                }
                list.push(c_name(&out)?);
            }
        }
        let kv_on_gpu = ep != EP_CPU
            && input_on_gpu(&session, "past_key_values.0.key").unwrap_or_else(|e| {
                tracing::debug!(target: "sigil::tts", "Chatterbox: cache placement unknown ({e}); using the EP's memory");
                true
            });
        Ok(Self {
            kv: None,
            session,
            ep,
            device_id,
            kv_on_gpu,
            layers,
            kv_heads,
            head_dim,
            hidden,
            vocab,
            names,
        })
    }

    pub fn ep(&self) -> &'static str {
        self.ep
    }

    /// `(layers, kv_heads, head_dim, hidden, vocab)` read from the session.
    #[cfg(test)]
    pub fn dims(&self) -> (usize, usize, usize, usize, usize) {
        (self.layers, self.kv_heads, self.head_dim, self.hidden, self.vocab)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Bytes of a cache for `batch` rows of `capacity` positions.
    pub fn cache_bytes(&self, batch: usize, capacity: usize) -> u64 {
        (2 * self.layers * batch * self.kv_heads * capacity * self.head_dim * 4) as u64
    }

    /// Bytes of the cache allocated now.
    pub fn allocated_cache_bytes(&self) -> u64 {
        self.kv
            .as_ref()
            .map_or(0, |kv| self.cache_bytes(kv.batch, kv.capacity))
    }

    /// Bytes of the cache allocated in graphics memory now.
    pub fn gpu_cache_bytes(&self) -> u64 {
        if self.kv_on_gpu {
            self.allocated_cache_bytes()
        } else {
            0
        }
    }

    /// The cache of `layer` (`key` or `value`) as host floats, with its `[batch, heads, capacity,
    /// head_dim]` shape (CPU sessions only).
    #[cfg(test)]
    pub fn cache(&self, layer: usize, key: bool) -> Option<(Vec<usize>, Vec<f32>)> {
        let kv = self.kv.as_ref()?;
        let v = if key { &kv.keys[layer] } else { &kv.values[layer] };
        let (shape, data) = v.try_extract_tensor::<f32>().ok()?;
        Some((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }

    /// Address of the cache tensor of `layer` (`key` or `value`): unchanged while it is reused.
    #[cfg(test)]
    pub fn cache_ptr(&self, layer: usize, key: bool) -> Option<usize> {
        let kv = self.kv.as_ref()?;
        let v = if key { &kv.keys[layer] } else { &kv.values[layer] };
        Some(v.ptr() as usize)
    }

    fn allocator(&self) -> Result<Allocator, String> {
        if !self.kv_on_gpu {
            return Ok(Allocator::default());
        }
        let device = match self.ep {
            EP_CUDA => AllocationDevice::CUDA,
            EP_DML => AllocationDevice::DIRECTML,
            _ => return Ok(Allocator::default()),
        };
        let info = MemoryInfo::new(device, self.device_id, AllocatorType::Device, MemoryType::Default)
            .map_err(|e| format!("graphics memory description: {e}"))?;
        Allocator::new(&self.session, info).map_err(|e| format!("the language model's graphics memory allocator: {e}"))
    }

    /// A cache for `batch` rows and at least `needed` positions (the current one when it fits).
    fn ensure_cache(&mut self, batch: usize, needed: usize) -> Result<(), String> {
        if let Some(kv) = &self.kv {
            if kv.batch == batch && kv.capacity >= needed {
                return Ok(());
            }
        }
        self.kv = None;
        let capacity = needed.div_ceil(CAPACITY_STEP) * CAPACITY_STEP;
        let allocator = self.allocator()?;
        let shape = [batch, self.kv_heads, capacity, self.head_dim];
        let mut keys = Vec::with_capacity(self.layers);
        let mut values = Vec::with_capacity(self.layers);
        for _ in 0..self.layers {
            for list in [&mut keys, &mut values] {
                let t = Tensor::<f32>::new(&allocator, shape)
                    .map_err(|e| format!("cannot allocate the language model's cache ({batch} x {capacity}): {e}"))?;
                list.push(t.into_dyn());
            }
        }
        self.kv = Some(KvCache {
            batch,
            capacity,
            keys,
            values,
            _allocator: allocator,
        });
        Ok(())
    }

    /// Generate speech tokens after the prompt `prefill` (`[batch, prompt_len, hidden]` input
    /// embeddings: conditioning, text and the two speech-start positions), at most `max_new`.
    /// Decode steps embed each token with `embed` at positions 1, 2, ... `Ok(None)` = cancelled.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        prefill: &DynValue,
        batch: usize,
        prompt_len: usize,
        max_new: usize,
        embed: &SpeechEmbedding,
        params: &SamplingParams,
        rng: &mut Rng,
        cancel: &AtomicBool,
    ) -> Result<Option<Generated>, String> {
        if !(1..=2).contains(&batch) || prompt_len == 0 {
            return Err(format!("invalid prompt ({batch} rows of {prompt_len})"));
        }
        if embed.width() != self.hidden {
            return Err(format!(
                "the token embedding ({}) and the language model ({}) disagree",
                embed.width(),
                self.hidden
            ));
        }
        let max_new = max_new.min(embed.max_position());
        self.ensure_cache(batch, prompt_len + max_new + 1)?;
        let kv = self.kv.as_ref().ok_or("no cache")?;
        let mut binding = self
            .session
            .create_binding()
            .map_err(|e| format!("IO binding: {e}"))?;
        for l in 0..self.layers {
            bind_input(&mut binding, &self.names.past_k[l], &kv.keys[l])?;
            bind_input(&mut binding, &self.names.past_v[l], &kv.values[l])?;
            bind_output(&mut binding, &self.names.present_k[l], &kv.keys[l])?;
            bind_output(&mut binding, &self.names.present_v[l], &kv.values[l])?;
        }
        let logits = Tensor::<f32>::new(&Allocator::default(), [batch, self.vocab]).map_err(|e| format!("logits buffer: {e}"))?;
        bind_output(&mut binding, &self.names.logits, &logits)?;

        // Prompt.
        let mut total = prompt_len;
        self.step(&mut binding, prefill, batch, total)?;
        let mut generated: Vec<i64> = vec![START_SPEECH];
        let mut out = Generated::default();
        let mut scratch = Vec::with_capacity(self.vocab);
        let mut row = vec![0f32; self.hidden];
        let mut embeds = vec![0f32; batch * self.hidden];
        for i in 0..max_new {
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            let (_, l) = logits.extract_tensor();
            let token = next_token(l, batch, self.vocab, &generated, params, rng, &mut scratch)?;
            if token == STOP_SPEECH {
                out.stopped = true;
                break;
            }
            generated.push(token);
            out.tokens.push(token);
            if i + 1 == max_new {
                break;
            }
            embed.embed(token, i + 1, &mut row)?;
            for r in 0..batch {
                embeds[r * self.hidden..(r + 1) * self.hidden].copy_from_slice(&row);
            }
            let input = Tensor::<f32>::from_array(([batch, 1, self.hidden], embeds.clone()))
                .map_err(|e| format!("decode input: {e}"))?
                .into_dyn();
            total += 1;
            self.step(&mut binding, &input, batch, total)?;
        }
        Ok(Some(out))
    }

    /// One run: `input` holds the new positions; `total` is the length after them.
    fn step(&mut self, binding: &mut IoBinding, input: &DynValue, batch: usize, total: usize) -> Result<(), String> {
        let total_i32 = i32::try_from(total).map_err(|_| "sequence too long")?;
        let seqlens = Tensor::<i32>::from_array(([batch], vec![total_i32 - 1; batch])).map_err(|e| e.to_string())?;
        let total_t = Tensor::<i32>::from_array(([0usize; 0], vec![total_i32])).map_err(|e| e.to_string())?;
        bind_input(binding, &self.names.inputs_embeds, input)?;
        bind_input(binding, &self.names.seqlens_k, &seqlens)?;
        bind_input(binding, &self.names.total, &total_t)?;
        run_binding(&mut self.session, binding).map_err(|e| format!("the language model failed: {e}"))
    }
}
