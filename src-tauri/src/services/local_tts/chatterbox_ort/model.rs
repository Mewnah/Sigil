//! The loaded Chatterbox Multilingual V3 model: KitsuMate's split-speed ONNX layout on Sigil's ONNX
//! Runtime, built through `ort_session` (EP ladder CUDA → DirectML → CPU, every outcome recorded).
//!
//! | Graph (`onnx/`) | Runs on | Role |
//! |---|---|---|
//! | `speech_encoder_slim` | CPU, only while a new voice is prepared | reference clip → [`Conditionals`] |
//! | `token_embedding` | CPU | prompt embeddings (conditioning, emotion slot, text, speech start) |
//! | `static_language_model_int8` | the requested EP | speech tokens ([`StaticLm`]) |
//! | `flow_prepare_slim` | the requested EP | tokens + prompt mel → flow inputs and noise |
//! | `flow_step_slim` (CPU) / `flow_step_fp16` (GPU, optional entry) | the requested EP | one Euler step with guidance |
//! | `vocoder_prefix_sum` | the requested EP | mel → 24 kHz audio (drops the prompt part, fades in) |
//!
//! The flow integrates from noise along the reference implementation's cosine time schedule with
//! decoder guidance 0.7: 10 steps on a GPU (the reference), 6 on the CPU (KitsuMate's tested
//! setting). The noise is drawn by `flow_prepare_slim` (unseeded), or by [`seeded_noise`] when the
//! request has a seed. A GPU EP that fails while running (an operator its driver lacks) drops the
//! model to the CPU once, with the reason recorded. GPU memory is claimed in `resources` (an
//! estimate from the graphs on the device plus the language model's cache).
//!
//! Every graphics session (language model, its cache, flow, vocoder) runs on one device: the
//! [`Placement`] of the request. `auto` picks CUDA with an NVIDIA card, else DirectML on the
//! discrete card with the most memory (never integrated graphics), else the CPU.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType};

use super::embed::SpeechEmbedding;
use super::lm::StaticLm;
use super::sampling::{Rng, SamplingParams};
use super::text::{self, Frontend, SPEECH_VOCAB};
use super::voice::Conditionals;
use crate::services::diagnostics::gpu_mem::{self, GpuAdapter, VENDOR_MICROSOFT};
use crate::services::diagnostics::{record_ep_outcome, EpOutcome};
use crate::services::neural_audio::ort_session::{
    build_session, device_for, ep_label, normalize_ep, ModelSource, VcSessionOptions, EP_CPU, EP_CUDA, EP_DML,
};
use crate::services::resources::{self, Subsystem, VramClaim};
use crate::services::sysinfo::arch;

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Diagnostics components (Settings → GPU & diagnostics).
pub const COMPONENT_LM: &str = "tts.chatterbox";
pub const COMPONENT_FLOW: &str = "tts.chatterbox.flow";
pub const COMPONENT_VOCODER: &str = "tts.chatterbox.vocoder";
/// Default CPU threads (ONNX Runtime intra-op).
pub const DEFAULT_THREADS: u32 = 4;
/// Decoder (flow) guidance of the reference implementation.
pub const FLOW_GUIDANCE: f32 = 0.7;
/// Flow steps on a GPU (the reference implementation's) and on the CPU (KitsuMate's tested).
pub const GPU_FLOW_STEPS: usize = 10;
pub const CPU_FLOW_STEPS: usize = 6;
/// Language-model positions the speech position table must cover (1000 tokens = 40 s).
pub const MAX_POSITIONS: usize = 1024;
/// Graphics memory beyond the weights: the flow's activations and ONNX Runtime's workspace (MB).
const GPU_WORKSPACE_MB: u64 = 512;
/// Voices kept prepared at once.
const VOICE_SLOTS: usize = 4;

/// Where the files of an installed model are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFiles {
    pub dir: PathBuf,
    pub speech_encoder: PathBuf,
    pub token_embedding: PathBuf,
    pub language_model: PathBuf,
    pub flow_prepare: PathBuf,
    pub flow_step: PathBuf,
    /// `flow_step_fp16.onnx` from the optional GPU entry.
    pub flow_step_fp16: Option<PathBuf>,
    pub vocoder: PathBuf,
    pub tokenizer: PathBuf,
    pub cangjie: Option<PathBuf>,
    pub japanese: Option<PathBuf>,
    pub russian: Option<PathBuf>,
}

impl ModelFiles {
    /// The files of the model installed in `dir` (and the fp16 flow step in `gpu_flow_dir`).
    pub fn discover(dir: &Path, gpu_flow_dir: Option<&Path>) -> Result<Self, String> {
        let need = |rel: &str| {
            let p = dir.join(rel);
            if p.is_file() {
                Ok(p)
            } else {
                Err(format!(
                    "the Chatterbox model in {} is incomplete: {rel} is missing (reinstall it)",
                    dir.display()
                ))
            }
        };
        let optional = |rel: &str| Some(dir.join(rel)).filter(|p| p.is_file());
        Ok(Self {
            dir: dir.to_path_buf(),
            speech_encoder: need("onnx/speech_encoder_slim.onnx")?,
            token_embedding: need("onnx/token_embedding.onnx")?,
            language_model: need("onnx/static_language_model_int8.onnx")?,
            flow_prepare: need("onnx/flow_prepare_slim.onnx")?,
            flow_step: need("onnx/flow_step_slim.onnx")?,
            flow_step_fp16: gpu_flow_dir
                .map(|d| d.join("onnx/flow_step_fp16.onnx"))
                .filter(|p| p.is_file()),
            vocoder: need("onnx/vocoder_prefix_sum.onnx")?,
            tokenizer: need("tokenizer.json")?,
            cangjie: optional("Cangjie5_TC.json"),
            japanese: optional("japanese_readings.tsv"),
            russian: optional("russian_stress.tsv"),
        })
    }
}

/// Size of a graph and its external data sidecar (MB, rounded up).
fn graph_mb(path: &Path) -> u64 {
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let mut data = path.as_os_str().to_owned();
    data.push(".data");
    (size(path) + size(Path::new(&data))).div_ceil(1 << 20)
}

/// Where the model runs: an EP and its device (the CUDA ordinal for CUDA, the DXGI adapter index
/// for DirectML, which is DirectML's device id; 0 on the CPU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub ep: &'static str,
    pub device: i32,
}

impl Placement {
    pub const CPU: Placement = Placement { ep: EP_CPU, device: 0 };

    /// `cuda@0`, `dml@1`, `cpu` (logs and the worker key).
    pub fn label(&self) -> String {
        if self.ep == EP_CPU {
            EP_CPU.to_string()
        } else {
            format!("{}@{}", self.ep, self.device)
        }
    }
}

/// Dedicated graphics memory `auto` wants before it picks DirectML (MB).
pub const AUTO_DML_MIN_VRAM_MB: u64 = 2048;

/// A real (not software, not Microsoft's) adapter.
fn hardware(a: &GpuAdapter) -> bool {
    !a.software && a.vendor_id != VENDOR_MICROSOFT
}

/// A discrete card: a hardware adapter that is not integrated graphics. An APU's UMA carve-out
/// reports as dedicated memory, so the size alone cannot tell (`sysinfo::arch::is_integrated`
/// knows the integrated device ids).
fn discrete(a: &GpuAdapter) -> bool {
    hardware(a) && !arch::is_integrated(a.vendor_id, a.device_id, a.dedicated_vram_mb)
}

/// The adapter with the most dedicated memory among those `keep` accepts (the first on a tie).
fn largest(adapters: &[GpuAdapter], keep: impl Fn(&GpuAdapter) -> bool) -> Option<&GpuAdapter> {
    adapters
        .iter()
        .filter(|a| keep(a))
        .fold(None, |best: Option<&GpuAdapter>, a| match best {
            Some(b) if b.dedicated_vram_mb >= a.dedicated_vram_mb => Some(b),
            _ => Some(a),
        })
}

/// Where `auto` runs: CUDA with an NVIDIA card; else DirectML on the discrete card with the most
/// memory, when it has at least [`AUTO_DML_MIN_VRAM_MB`] (integrated graphics, even with a large
/// carve-out, are slower than the CPU for this model); else the CPU.
pub fn auto_placement(adapters: &[GpuAdapter]) -> Placement {
    if !gpu_mem::nvidia_adapter_indices(adapters).is_empty() {
        return Placement { ep: EP_CUDA, device: 0 };
    }
    match largest(adapters, |a| discrete(a) && a.dedicated_vram_mb >= AUTO_DML_MIN_VRAM_MB) {
        Some(a) => Placement {
            ep: EP_DML,
            device: a.index as i32,
        },
        None => Placement::CPU,
    }
}

/// The DirectML adapter of an explicit `dml` request: the discrete card with the most memory, else
/// the hardware adapter with the most (the user asked for the graphics card), else adapter 0.
pub fn dml_device(adapters: &[GpuAdapter]) -> i32 {
    largest(adapters, discrete)
        .or_else(|| largest(adapters, hardware))
        .map_or(0, |a| a.index as i32)
}

/// Where a request's `ep` option runs on a PC with `adapters` (`auto` / missing → [`auto_placement`]).
pub fn placement_for(option: Option<&str>, adapters: &[GpuAdapter]) -> Placement {
    let option = option
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());
    match option.as_deref() {
        None | Some("auto") => auto_placement(adapters),
        Some(other) => match normalize_ep(other) {
            EP_DML => Placement {
                ep: EP_DML,
                device: dml_device(adapters),
            },
            // CUDA ordinal 0: the ladder maps a DirectML fallback to that card's adapter.
            EP_CUDA => Placement { ep: EP_CUDA, device: 0 },
            _ => Placement::CPU,
        },
    }
}

/// How long an adapter list is reused (every sentence resolves its placement for the worker key).
const ADAPTERS_FRESH: std::time::Duration = std::time::Duration::from_secs(30);

/// This PC's adapters, enumerated at most every [`ADAPTERS_FRESH`]. An empty enumeration (DXGI
/// failed: Windows always lists its software adapter) never replaces a list that had adapters, so
/// a hiccup cannot move a loaded model to the CPU.
fn adapters_cached() -> Vec<GpuAdapter> {
    static CACHE: std::sync::Mutex<Option<(std::time::Instant, Vec<GpuAdapter>)>> = std::sync::Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, list)) = cache.as_ref() {
        if at.elapsed() < ADAPTERS_FRESH {
            return list.clone();
        }
    }
    let fresh = gpu_mem::adapters();
    let list = match cache.take() {
        Some((_, old)) if fresh.is_empty() && !old.is_empty() => old,
        _ => fresh,
    };
    *cache = Some((std::time::Instant::now(), list.clone()));
    list
}

/// Where a request's `ep` option runs on this PC.
pub fn requested_placement(option: Option<&str>) -> Placement {
    let option = option.map(str::trim).filter(|s| !s.is_empty());
    if option.is_some_and(|o| o.eq_ignore_ascii_case(EP_CPU)) {
        // No need to look at the graphics cards.
        return Placement::CPU;
    }
    placement_for(option, &adapters_cached())
}

/// The reference implementation's cosine time schedule: `steps + 1` times from 0 to 1.
pub fn cosine_schedule(steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    (0..=n)
        .map(|i| {
            let x = i as f64 / n as f64;
            (1.0 - (x * std::f64::consts::FRAC_PI_2).cos()) as f32
        })
        .collect()
}

/// Session options for `threads` CPU threads on graphics `device` (see [`Placement::device`]).
pub(super) fn session_options(threads: u32, device: i32) -> VcSessionOptions {
    VcSessionOptions {
        cuda_device_index: device.max(0),
        intra_threads: Some(threads.clamp(1, 16) as usize),
        // Every sentence length is a new input shape.
        cpu_arena: true,
        ..VcSessionOptions::default()
    }
}

fn first_f32(outputs: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<(Vec<i64>, Vec<f32>), String> {
    let v = outputs
        .remove(name)
        .ok_or_else(|| format!("the graph has no {name} output"))?;
    let (shape, data) = v
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("{name}: {e}"))?;
    Ok((shape.to_vec(), data.to_vec()))
}

/// Run the speech encoder on a 24 kHz mono clip (a session of its own, freed afterwards).
pub fn encode_reference(encoder: &Path, audio_24k: &[f32], threads: u32) -> Result<Conditionals, String> {
    if audio_24k.len() < SAMPLE_RATE as usize {
        return Err("the reference voice is shorter than one second".into());
    }
    let (mut session, _) = build_session(None, EP_CPU, &session_options(threads.max(DEFAULT_THREADS), 0), ModelSource::File(encoder))
        .map_err(|e| format!("cannot load Chatterbox's speech encoder: {e}"))?;
    let input = Tensor::<f32>::from_array(([1usize, audio_24k.len()], audio_24k.to_vec())).map_err(|e| e.to_string())?;
    let mut out = session
        .run(ort::inputs!["audio_values" => input])
        .map_err(|e| format!("the speech encoder failed: {e}"))?;
    let (fshape, features) = first_f32(&mut out, "audio_features")?;
    let (sshape, speaker) = first_f32(&mut out, "speaker_embeddings")?;
    let (pshape, prompt) = first_f32(&mut out, "speaker_features")?;
    let tokens_v = out
        .remove("audio_tokens")
        .ok_or("the speech encoder has no audio_tokens output")?;
    let (_, tokens) = tokens_v
        .try_extract_tensor::<i64>()
        .map_err(|e| format!("audio_tokens: {e}"))?;
    let (feature_len, hidden) = match fshape.as_slice() {
        [1, n, h] => (*n as usize, *h as usize),
        s => return Err(format!("unexpected audio_features shape {s:?}")),
    };
    let (prompt_frames, mel) = match pshape.as_slice() {
        [1, n, m] => (*n as usize, *m as usize),
        s => return Err(format!("unexpected speaker_features shape {s:?}")),
    };
    if sshape.first().copied().unwrap_or(0) != 1 {
        return Err(format!("unexpected speaker_embeddings shape {sshape:?}"));
    }
    let c = Conditionals {
        features,
        feature_len,
        hidden,
        tokens: tokens.to_vec(),
        speaker,
        prompt,
        prompt_frames,
        mel,
    };
    c.validate()?;
    Ok(c)
}

/// The flow step's inputs in the order [`Flow::render`] binds them, with their element types. Both
/// exports (`flow_step_slim`, fp32 weights inline, and `flow_step_fp16`) take exactly these
/// (`testdata/flow_step_*.io.pb` hold the real graphs' input and output declarations).
pub const FLOW_STEP_INPUTS: [(&str, TensorElementType); 8] = [
    ("x", TensorElementType::Float32),
    ("mask", TensorElementType::Bool),
    ("mu", TensorElementType::Float32),
    ("speakers", TensorElementType::Float32),
    ("cond", TensorElementType::Float32),
    ("time", TensorElementType::Float32),
    ("delta", TensorElementType::Float32),
    ("guidance", TensorElementType::Float32),
];
/// The flow step's output (the next `x`).
pub const FLOW_STEP_OUTPUT: (&str, TensorElementType) = ("next_x", TensorElementType::Float32);

/// Why a loaded flow step is not the export [`Flow::render`] drives (`None` = it is): checked at
/// load, so a different export fails Start with the reason instead of every sentence.
pub fn flow_step_problem(session: &Session) -> Option<String> {
    let find = |outlets: &[ort::value::Outlet], name: &str| {
        outlets
            .iter()
            .find(|o| o.name() == name)
            .map(|o| o.dtype().tensor_type())
    };
    for (name, ty) in FLOW_STEP_INPUTS {
        match find(session.inputs(), name) {
            None => return Some(format!("it has no {name} input")),
            Some(t) if t != Some(ty) => return Some(format!("its {name} input is {t:?}, not {ty:?}")),
            Some(_) => {}
        }
    }
    if session.inputs().len() != FLOW_STEP_INPUTS.len() {
        return Some(format!("it takes {} inputs, not {}", session.inputs().len(), FLOW_STEP_INPUTS.len()));
    }
    let (name, ty) = FLOW_STEP_OUTPUT;
    match find(session.outputs(), name) {
        None => Some(format!("it has no {name} output")),
        Some(t) if t != Some(ty) => Some(format!("its {name} output is {t:?}, not {ty:?}")),
        Some(_) => None,
    }
}

/// Standard normal noise shaped like `like` (a float tensor), drawn from `seed`: the flow's
/// starting point when a request has a seed. The export draws it inside `flow_prepare_slim` with an
/// unseeded `RandomNormalLike` (mean 0, scale 1), which this replaces so a seed reproduces the
/// waveform too, not only the speech tokens.
pub fn seeded_noise(like: &DynValue, seed: u64) -> Result<DynValue, String> {
    let (shape, data) = like
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("x: {e}"))?;
    let shape: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    // Its own stream: the token sampler draws from `seed` itself.
    let mut rng = Rng::new(seed.rotate_left(29) ^ 0xD1B5_4A32_D192_ED03);
    let mut noise = Vec::with_capacity(data.len() + 1);
    while noise.len() < data.len() {
        // Box-Muller: two independent normals from two uniforms in (0, 1].
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        noise.push((r * a.cos()) as f32);
        noise.push((r * a.sin()) as f32);
    }
    noise.truncate(data.len());
    Ok(Tensor::<f32>::from_array((shape, noise))
        .map_err(|e| e.to_string())?
        .into_dyn())
}

/// The flow and the vocoder.
pub struct Flow {
    prepare: Session,
    step: Session,
    vocoder: Session,
    /// EP of the flow step (decides the default step count).
    step_ep: &'static str,
    /// Graphs on a GPU and their size (MB).
    gpu_mb: u64,
}

fn landed(ep: &str) -> &'static str {
    match normalize_ep(ep) {
        EP_CUDA => EP_CUDA,
        EP_DML => EP_DML,
        _ => EP_CPU,
    }
}

/// Load the fp32 flow step (`flow_step_slim`) on `ep`, checked against what the driver binds.
fn load_fp32_step(files: &ModelFiles, ep: &str, opts: &VcSessionOptions) -> Result<(Session, EpOutcome), String> {
    let (s, o) = build_session(None, ep, opts, ModelSource::File(&files.flow_step))
        .map_err(|e| format!("cannot load Chatterbox's flow step: {e}"))?;
    if let Some(why) = flow_step_problem(&s) {
        return Err(format!(
            "Chatterbox's flow step ({}) is not the expected export: {why}; reinstall the model",
            files.flow_step.display()
        ));
    }
    Ok((s, o))
}

impl Flow {
    /// Load the flow graphs on `ep` and graphics `device` (each falls back on its own).
    pub fn load(files: &ModelFiles, ep: &str, device: i32, threads: u32, record: bool) -> Result<Self, String> {
        let opts = session_options(threads, device);
        let component = |c: &'static str| record.then_some(c);
        let (prepare, o_prepare) = build_session(component(COMPONENT_FLOW), ep, &opts, ModelSource::File(&files.flow_prepare))
            .map_err(|e| format!("cannot load Chatterbox's flow: {e}"))?;
        // fp16 compute is faster on a GPU and slower on the CPU: only for a GPU rung.
        let (step, o_step, fp16) = match files
            .flow_step_fp16
            .as_deref()
            .filter(|_| normalize_ep(ep) != EP_CPU)
        {
            Some(fp16) => match build_session(None, ep, &opts, ModelSource::File(fp16)) {
                Ok((s, o)) if landed(&o.active) != EP_CPU && flow_step_problem(&s).is_none() => (s, o, true),
                other => {
                    match other {
                        Err(e) => tracing::warn!(target: "sigil::tts", "Chatterbox: the fp16 flow step did not load ({e}); using fp32"),
                        Ok((s, _)) => {
                            if let Some(why) = flow_step_problem(&s) {
                                tracing::warn!(target: "sigil::tts", "Chatterbox: the fp16 flow step is not the expected export ({why}); using fp32");
                            }
                        }
                    }
                    let (s, o) = load_fp32_step(files, ep, &opts)?;
                    (s, o, false)
                }
            },
            None => {
                let (s, o) = load_fp32_step(files, ep, &opts)?;
                (s, o, false)
            }
        };
        let (vocoder, o_vocoder) = build_session(component(COMPONENT_VOCODER), ep, &opts, ModelSource::File(&files.vocoder))
            .map_err(|e| format!("cannot load Chatterbox's vocoder: {e}"))?;
        let mut gpu_mb = 0;
        if landed(&o_prepare.active) != EP_CPU {
            gpu_mb += graph_mb(&files.flow_prepare);
        }
        if landed(&o_step.active) != EP_CPU {
            gpu_mb += graph_mb(if fp16 {
                files.flow_step_fp16.as_deref().unwrap_or(&files.flow_step)
            } else {
                &files.flow_step
            });
        }
        if landed(&o_vocoder.active) != EP_CPU {
            gpu_mb += graph_mb(&files.vocoder);
        }
        Ok(Self {
            prepare,
            step,
            vocoder,
            step_ep: landed(&o_step.active),
            gpu_mb,
        })
    }

    pub fn default_steps(&self) -> usize {
        if self.step_ep == EP_CPU {
            CPU_FLOW_STEPS
        } else {
            GPU_FLOW_STEPS
        }
    }

    /// Mel and audio for the speech `tokens` (prompt tokens excluded) in the voice of `cond`, from
    /// [`seeded_noise`] when `noise_seed` is set (else the graph's own noise). `Ok(None)` =
    /// cancelled.
    pub fn render(
        &mut self,
        tokens: &[i64],
        cond: &Conditionals,
        steps: usize,
        guidance: f32,
        noise_seed: Option<u64>,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>, String> {
        let mut all = Vec::with_capacity(cond.tokens.len() + tokens.len());
        all.extend_from_slice(&cond.tokens);
        all.extend_from_slice(tokens);
        let t_tokens = Tensor::<i64>::from_array(([1usize, all.len()], all)).map_err(|e| e.to_string())?;
        let t_speaker = Tensor::<f32>::from_array(([1usize, cond.speaker.len()], cond.speaker.clone())).map_err(|e| e.to_string())?;
        let t_prompt = Tensor::<f32>::from_array(([1usize, cond.prompt_frames, cond.mel], cond.prompt.clone())).map_err(|e| e.to_string())?;
        let mut prepared = self
            .prepare
            .run(ort::inputs![
                "speech_tokens" => t_tokens,
                "speaker_embeddings" => t_speaker,
                "speaker_features" => t_prompt
            ])
            .map_err(|e| format!("the flow failed: {e}"))?;
        let take = |o: &mut ort::session::SessionOutputs<'_>, name: &str| -> Result<DynValue, String> {
            o.remove(name)
                .ok_or_else(|| format!("the flow has no {name} output"))
        };
        let mut x = take(&mut prepared, "x")?;
        let mask = take(&mut prepared, "mask")?;
        let mu = take(&mut prepared, "mu")?;
        let speakers = take(&mut prepared, "speakers")?;
        let cond_mel = take(&mut prepared, "cond")?;
        let prefix_v = take(&mut prepared, "prefix")?;
        let (_, prefix) = prefix_v
            .try_extract_tensor::<i64>()
            .map_err(|e| format!("prefix: {e}"))?;
        let prefix = *prefix.first().ok_or("the flow returned no prefix")?;
        drop(prepared);
        if let Some(seed) = noise_seed {
            x = seeded_noise(&x, seed)?;
        }
        let schedule = cosine_schedule(steps);
        let [i_x, i_mask, i_mu, i_speakers, i_cond, i_time, i_delta, i_guidance] = FLOW_STEP_INPUTS.map(|(name, _)| name);
        for k in 1..schedule.len() {
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            let t = Tensor::<f32>::from_array(([1usize], vec![schedule[k - 1]])).map_err(|e| e.to_string())?;
            let dt = Tensor::<f32>::from_array(([1usize], vec![schedule[k] - schedule[k - 1]])).map_err(|e| e.to_string())?;
            let g = Tensor::<f32>::from_array(([1usize], vec![guidance])).map_err(|e| e.to_string())?;
            let mut out = self
                .step
                .run(ort::inputs![
                    i_x => &x,
                    i_mask => &mask,
                    i_mu => &mu,
                    i_speakers => &speakers,
                    i_cond => &cond_mel,
                    i_time => t,
                    i_delta => dt,
                    i_guidance => g
                ])
                .map_err(|e| format!("the flow step failed: {e}"))?;
            x = out
                .remove(FLOW_STEP_OUTPUT.0)
                .ok_or("the flow step has no next_x output")?;
        }
        if cancel.load(Ordering::Acquire) {
            return Ok(None);
        }
        let t_prefix = Tensor::<i64>::from_array(([1usize], vec![prefix])).map_err(|e| e.to_string())?;
        let mut out = self
            .vocoder
            .run(ort::inputs!["x" => &x, "prefix" => t_prefix])
            .map_err(|e| format!("the vocoder failed: {e}"))?;
        let (_, wav) = first_f32(&mut out, "waveform")?;
        Ok(Some(wav))
    }
}

/// Per-sentence generation settings.
#[derive(Clone, Debug, PartialEq)]
pub struct GenOptions {
    pub exaggeration: f32,
    pub sampling: SamplingParams,
    /// Flow steps; `None` = the default of the EP the flow runs on.
    pub flow_steps: Option<usize>,
    pub flow_guidance: f32,
    /// Seeds the speech-token sampling and the flow's starting noise ([`seeded_noise`]), so a
    /// sentence comes out the same on the same device; `None` = random.
    pub seed: Option<u64>,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            exaggeration: 0.5,
            sampling: SamplingParams::default(),
            flow_steps: None,
            flow_guidance: FLOW_GUIDANCE,
            seed: None,
        }
    }
}

/// A prepared voice.
struct VoiceSlot {
    id: String,
    sha256: String,
    cond: Arc<Conditionals>,
}

/// The loaded model (see the module docs).
pub struct ChatterboxModel {
    files: ModelFiles,
    /// Identifies the speech encoder (voice caches made by another one are recomputed).
    fingerprint: String,
    frontend: Frontend,
    token_embedding: Session,
    speech: SpeechEmbedding,
    lm: StaticLm,
    flow: Flow,
    voices: Vec<VoiceSlot>,
    threads: u32,
    requested: Placement,
    vram: Option<VramClaim>,
    /// A GPU EP failed while running and the model moved to the CPU.
    gpu_failed: bool,
}

/// sha256 of a file (the speech encoder graph, about 1 MB), shortened.
fn fingerprint(path: &Path) -> Result<String, String> {
    let full = crate::services::neural_audio::vc::families::reference::sha256_file(path)?;
    Ok(full.chars().take(16).collect())
}

impl ChatterboxModel {
    /// Load the model in `files` on `requested` (EP `cuda` / `dml` / `cpu` and its device) with
    /// `threads` CPU threads.
    pub fn load(files: ModelFiles, requested: Placement, threads: u32) -> Result<Self, String> {
        let fingerprint = fingerprint(&files.speech_encoder)?;
        let frontend = Frontend::load(&files.tokenizer, files.cangjie.clone(), files.japanese.clone(), files.russian.clone())?;
        let (token_embedding, _) = build_session(None, EP_CPU, &session_options(threads, 0), ModelSource::File(&files.token_embedding))
            .map_err(|e| format!("cannot load Chatterbox's token embedding: {e}"))?;
        let speech = SpeechEmbedding::read(&files.token_embedding, MAX_POSITIONS + 1)?;
        let mut place = requested;
        if place.ep != EP_CPU {
            if let Some(why) = gpu_budget_problem(&files) {
                for c in [COMPONENT_LM, COMPONENT_FLOW, COMPONENT_VOCODER] {
                    record_ep_outcome(c, EpOutcome::fallback(requested.ep, EP_CPU, why.clone()));
                }
                tracing::warn!(target: "sigil::tts", "Chatterbox runs on the CPU: {why}");
                place = Placement::CPU;
            }
        }
        tracing::debug!(target: "sigil::tts", "Chatterbox: loading on {}", place.label());
        let (lm, flow) = load_parts(&files, place, threads, true)?;
        let mut model = Self {
            files,
            fingerprint,
            frontend,
            token_embedding,
            speech,
            lm,
            flow,
            voices: Vec::new(),
            threads,
            requested,
            vram: None,
            gpu_failed: false,
        };
        model.check_embedding()?;
        model.update_vram();
        Ok(model)
    }

    /// The speech table read from the graph's weights matches what the graph computes (a token
    /// other than 0 at position 1, where the graph embeds speech correctly).
    fn check_embedding(&mut self) -> Result<(), String> {
        let hidden = self.speech.width();
        if hidden != self.lm.hidden() {
            return Err(format!(
                "the token embedding ({hidden}) and the language model ({}) disagree",
                self.lm.hidden()
            ));
        }
        let probe_token = 1i64;
        let session = &mut self.token_embedding;
        let graph = (|| -> Result<Vec<f32>, String> {
            let e = |e: ort::Error| e.to_string();
            let mut out = session
                .run(ort::inputs![
                    "input_ids" => Tensor::<i64>::from_array(([1usize, 1], vec![probe_token])).map_err(e)?,
                    "token_position_ids" => Tensor::<i64>::from_array(([1usize, 1], vec![1i64])).map_err(e)?,
                    "exaggeration" => Tensor::<f32>::from_array(([1usize], vec![0.5f32])).map_err(e)?,
                    "text_conditioning" => Tensor::<f32>::from_array(([1usize], vec![1.0f32])).map_err(e)?,
                    "conditioning" => Tensor::<f32>::from_array(([1usize, 0, hidden], Vec::<f32>::new())).map_err(e)?
                ])
                .map_err(e)?;
            Ok(first_f32(&mut out, "inputs_embeds")?.1)
        })();
        let graph = match graph {
            Ok(g) => g,
            Err(e) => {
                // Not fatal: the tables were found and sized from the graph; only the cross-check
                // is skipped.
                tracing::warn!(target: "sigil::tts", "Chatterbox: the token embedding check did not run: {e}");
                return Ok(());
            }
        };
        let mut mine = vec![0f32; hidden];
        self.speech.embed(probe_token, 1, &mut mine)?;
        let worst = graph
            .iter()
            .zip(&mine)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if graph.len() != hidden || worst > 1e-4 {
            return Err(format!(
                "the token embedding weights do not match the graph (difference {worst}); reinstall the model"
            ));
        }
        Ok(())
    }

    pub fn on_gpu(&self) -> bool {
        self.lm.ep() != EP_CPU || self.flow.gpu_mb > 0
    }

    pub fn lm_ep(&self) -> &'static str {
        self.lm.ep()
    }

    /// Claim (or resize) graphics memory for what runs on the GPU.
    fn update_vram(&mut self) {
        let mut mb = self.flow.gpu_mb;
        if self.lm.ep() != EP_CPU {
            mb += graph_mb(&self.files.language_model) + self.lm.gpu_cache_bytes().div_ceil(1 << 20);
        }
        if mb == 0 {
            self.vram = None;
            return;
        }
        let total = mb + GPU_WORKSPACE_MB;
        match &self.vram {
            Some(c) => c.resize(total),
            None => self.vram = Some(resources::claim_vram(Subsystem::Tts, "chatterbox-ort", total)),
        }
    }

    /// Move everything to the CPU after a GPU failure (once; the reason is recorded).
    fn fall_back_to_cpu(&mut self, why: &str) -> Result<(), String> {
        self.gpu_failed = true;
        let gpu = if self.lm.ep() != EP_CPU { self.lm.ep() } else { self.flow.step_ep };
        let reason = format!("{} failed while running: {why}", ep_label(gpu));
        let (lm, flow) = load_parts(&self.files, Placement::CPU, self.threads, false)?;
        self.lm = lm;
        self.flow = flow;
        for c in [COMPONENT_LM, COMPONENT_FLOW, COMPONENT_VOCODER] {
            record_ep_outcome(c, EpOutcome::fallback(self.requested.ep, EP_CPU, reason.clone()));
        }
        self.update_vram();
        Ok(())
    }

    /// The conditioning of library voice `voice_id` (prepared voices in memory, then the library
    /// cache, else the speech encoder, whose result is cached).
    pub fn conditionals(&mut self, app_data: &Path, voice_id: &str) -> Result<Arc<Conditionals>, String> {
        let id = voice_id.trim();
        if id.is_empty() {
            return Err("Chatterbox clones a voice: choose one from your voice library".into());
        }
        if !crate::services::voice_library::valid_voice_id(id) {
            return Err(format!("invalid voice id {id:?}"));
        }
        let clip = crate::services::voice_library::reference_clip(app_data, id)?;
        if let Some(pos) = self
            .voices
            .iter()
            .position(|v| v.id == id && v.sha256 == clip.sha256)
        {
            let slot = self.voices.remove(pos);
            let cond = slot.cond.clone();
            self.voices.push(slot);
            return Ok(cond);
        }
        let cond = Arc::new(self.load_or_encode(app_data, &clip)?);
        if cond.hidden != self.lm.hidden() {
            return Err("the voice conditioning does not fit this model".into());
        }
        self.voices.retain(|v| v.id != id);
        if self.voices.len() >= VOICE_SLOTS {
            self.voices.remove(0);
        }
        self.voices.push(VoiceSlot {
            id: id.to_string(),
            sha256: clip.sha256.clone(),
            cond: cond.clone(),
        });
        Ok(cond)
    }

    fn load_or_encode(
        &self,
        app_data: &Path,
        clip: &crate::services::neural_audio::vc::families::reference::ReferenceClip,
    ) -> Result<Conditionals, String> {
        use crate::services::neural_audio::vc::families::reference::ReferenceEmbedding;
        let backend = super::CACHE_BACKEND;
        if let Some(bytes) = crate::services::voice_library::read_cache(app_data, &clip.voice_id, backend) {
            match ReferenceEmbedding::from_bytes(&bytes) {
                Ok(e) if e.backend == backend && e.clip_sha256 == clip.sha256 && e.voice_id == clip.voice_id => {
                    if let Some(c) = Conditionals::from_embedding(&e, &self.fingerprint) {
                        return Ok(c);
                    }
                    tracing::debug!(target: "sigil::tts", "Chatterbox: the cached voice {} is from another model; recomputing", clip.voice_id);
                }
                Ok(_) => tracing::debug!(target: "sigil::tts", "Chatterbox: the cached voice {} is stale; recomputing", clip.voice_id),
                Err(e) => tracing::debug!(target: "sigil::tts", "Chatterbox: voice cache {}: {e}", clip.voice_id),
            }
        }
        let started = std::time::Instant::now();
        let audio = clip.load_mono(SAMPLE_RATE)?;
        let cond = encode_reference(&self.files.speech_encoder, &audio, self.threads)?;
        tracing::info!(
            target: "sigil::tts",
            "Chatterbox: prepared voice {} in {} ms",
            clip.voice_id,
            started.elapsed().as_millis()
        );
        let mut e = cond.to_embedding(&self.fingerprint);
        e.voice_id = clip.voice_id.clone();
        e.backend = backend.to_string();
        e.clip_sha256 = clip.sha256.clone();
        match e.to_bytes() {
            Ok(bytes) => {
                if let Err(err) = crate::services::voice_library::write_cache(app_data, &clip.voice_id, backend, &bytes) {
                    tracing::debug!(target: "sigil::tts", "Chatterbox: voice {} not cached: {err}", clip.voice_id);
                }
            }
            Err(err) => tracing::debug!(target: "sigil::tts", "Chatterbox: voice {} not cached: {err}", clip.voice_id),
        }
        Ok(cond)
    }

    /// The prompt embeddings `[batch, S + L, hidden]` for the framed text `ids`.
    fn prompt(&mut self, ids: &[i64], cond: &Conditionals, batch: usize, exaggeration: f32) -> Result<(DynValue, usize), String> {
        let len = ids.len();
        let positions = text::positions(ids);
        let tile = |v: &[i64]| -> Vec<i64> { (0..batch).flat_map(|_| v.iter().copied()).collect() };
        let features: Vec<f32> = (0..batch)
            .flat_map(|_| cond.features.iter().copied())
            .collect();
        // Row 0 reads the text; row 1 (classifier-free guidance) gets the text positions only.
        let text_conditioning: Vec<f32> = (0..batch).map(|r| if r == 0 { 1.0 } else { 0.0 }).collect();
        let mut out = self
            .token_embedding
            .run(ort::inputs![
                "input_ids" => Tensor::<i64>::from_array(([batch, len], tile(ids))).map_err(|e| e.to_string())?,
                "token_position_ids" => Tensor::<i64>::from_array(([batch, len], tile(&positions))).map_err(|e| e.to_string())?,
                "exaggeration" => Tensor::<f32>::from_array(([batch], vec![exaggeration; batch])).map_err(|e| e.to_string())?,
                "text_conditioning" => Tensor::<f32>::from_array(([batch], text_conditioning)).map_err(|e| e.to_string())?,
                "conditioning" => Tensor::<f32>::from_array(([batch, cond.feature_len, cond.hidden], features)).map_err(|e| e.to_string())?
            ])
            .map_err(|e| format!("the token embedding failed: {e}"))?;
        let embeds = out
            .remove("inputs_embeds")
            .ok_or("the token embedding has no inputs_embeds output")?;
        Ok((embeds, cond.feature_len + len))
    }

    /// Speak `text` in `lang` with the voice `cond`: 24 kHz audio, empty when there is nothing to
    /// say, `None` when cancelled. A GPU failure moves the model to the CPU and retries once.
    pub fn synthesize(
        &mut self,
        text: &str,
        lang: &str,
        cond: &Conditionals,
        opts: &GenOptions,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>, String> {
        // Text errors are the sentence's, not the device's: they never move the model.
        let Some(ids) = self.frontend.encode(text, lang)? else {
            return Ok(Some(Vec::new()));
        };
        let batch = if opts.sampling.cfg_weight != 0.0 { 2 } else { 1 };
        let (prefill, prompt_len) = self.prompt(&ids, cond, batch, opts.exaggeration)?;
        let max_new = text::max_speech_tokens(text).min(MAX_POSITIONS - 1);
        match self.generate_and_render(&prefill, batch, prompt_len, max_new, cond, opts, cancel) {
            Err(e) if self.on_gpu() && !self.gpu_failed && !cancel.load(Ordering::Acquire) => {
                tracing::warn!(target: "sigil::tts", "Chatterbox failed on the graphics card: {e}; moving to the CPU");
                self.fall_back_to_cpu(&e)?;
                self.generate_and_render(&prefill, batch, prompt_len, max_new, cond, opts, cancel)
            }
            other => other,
        }
    }

    /// Speech tokens after the prompt, then the flow and the vocoder (`Ok(None)` = cancelled).
    #[allow(clippy::too_many_arguments)]
    fn generate_and_render(
        &mut self,
        prefill: &DynValue,
        batch: usize,
        prompt_len: usize,
        max_new: usize,
        cond: &Conditionals,
        opts: &GenOptions,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>, String> {
        let mut rng = opts.seed.map_or_else(Rng::fresh, Rng::new);
        let started = std::time::Instant::now();
        let cache_before = self.lm.allocated_cache_bytes();
        let generated = self
            .lm
            .generate(prefill, batch, prompt_len, max_new, &self.speech, &opts.sampling, &mut rng, cancel)?;
        if self.lm.allocated_cache_bytes() != cache_before {
            self.update_vram();
        }
        let Some(generated) = generated else {
            return Ok(None);
        };
        let tokens: Vec<i64> = generated
            .tokens
            .into_iter()
            .filter(|t| (0..SPEECH_VOCAB).contains(t))
            .collect();
        tracing::debug!(
            target: "sigil::tts",
            "Chatterbox: {} speech tokens in {} ms{}",
            tokens.len(),
            started.elapsed().as_millis(),
            if generated.stopped { "" } else { " (length limit)" }
        );
        if tokens.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let steps = opts
            .flow_steps
            .unwrap_or_else(|| self.flow.default_steps())
            .clamp(1, 32);
        self.flow
            .render(&tokens, cond, steps, opts.flow_guidance, opts.seed, cancel)
    }
}

/// The language model and flow at `place` (each session falls back on its own; outcomes recorded
/// when `record`).
fn load_parts(files: &ModelFiles, place: Placement, threads: u32, record: bool) -> Result<(StaticLm, Flow), String> {
    let ep = place.ep;
    let opts = session_options(threads, place.device);
    let (session, outcome) = build_session(record.then_some(COMPONENT_LM), ep, &opts, ModelSource::File(&files.language_model))
        .map_err(|e| format!("cannot load Chatterbox's language model: {e}"))?;
    let active = landed(&outcome.active);
    let lm = match StaticLm::new(session, active, device_for(active, ep, &opts)) {
        Ok(lm) => lm,
        Err(e) => return Err(format!("Chatterbox's language model is not the expected export: {e}")),
    };
    let flow = Flow::load(files, ep, place.device, threads, record)?;
    Ok((lm, flow))
}

/// Why the GPU should not be used: the free graphics memory is below what the model needs
/// (weights on the device, a typical cache and the workspace). `None` when it fits or is unknown.
fn gpu_budget_problem(files: &ModelFiles) -> Option<String> {
    let need = graph_mb(&files.language_model)
        + graph_mb(&files.flow_prepare)
        + graph_mb(files.flow_step_fp16.as_deref().unwrap_or(&files.flow_step))
        + graph_mb(&files.vocoder)
        + 256
        + GPU_WORKSPACE_MB;
    let budget = resources::budget();
    (budget.total_mb > 0 && budget.free_for_claims_mb < need).then(|| {
        format!(
            "not enough free graphics memory ({} MB free, about {need} MB needed)",
            budget.free_for_claims_mb
        )
    })
}
