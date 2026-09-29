//! `Sigil.exe --headless engine-probe transcribe <model> [wav] [--variant v] [--runtime-dir d]
//! [-- <options>]`: loads the runtime and the model the way the app does, runs the WAV a few times
//! and prints a JSON report with load time, RTF per run, the backend actually used, the process's
//! VRAM (DXGI) before / after loading / after the runs, and the transcript.
//!
//! Options after `--`:
//! - `--repeat <n>` timed runs after the warm-up (default 3, 1..=50)
//! - `--backend auto|gpu|cpu`, `--threads <n>`, `--device <i>`, `--device-id <pci id or name>`,
//!   `--n-ctx <tokens>`, `--kv auto|f16|f32`, `--no-cpu-fallback`, `--no-warmup`,
//!   `--warmup-seconds <s>`, `--min-free-vram-mb <mb>` (0 disables the VRAM headroom checks)
//! - `--language <code>`, `--translate`, `--timestamps none|segment|word|token|auto`, `--words`,
//!   `--prompt <text>`, `--no-temperature-fallback`
//! - `--stream` also feed the WAV in 200 ms chunks through a stream (streaming models)
//! - `--cancel-after-ms <n>` one more run, cancelled after n ms (reports the abort latency)
//! - `--ort-canary cuda|dml|cpu` run Sigil's ONNX Runtime canary before loading the engine and
//!   after the runs (ORT and the engine's ggml / CUDA state in one process)
//! - `--whisper-rs <ggml .bin>` also run whisper-rs (Sigil's statically linked ggml) on the same
//!   audio before and after, in this process: the coexistence check of ADR 0003b.
//!
//! VRAM: `vram.start` is the process at probe start; `vram.before_engine` is taken after the ORT
//! canary and whisper-rs loaded, right before the engine, so `load_delta` / `total_delta` are the
//! engine's own (`process_delta` still includes the canary's and whisper-rs's). `after_warmup` /
//! `run_growth` show whether the runs grew GPU memory past what the warm-up allocated.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::engine::{to_16k_mono, TranscribeModel, SAMPLE_RATE};
use super::types::{BackendChoice, KvType, ModelOptions, Timestamps, TranscribeOptions};
use crate::services::diagnostics::gpu_mem;
use crate::services::native_runtime::RuntimeVariant;
use crate::services::speech_rt::EngineProbeArgs;

/// Parsed probe options.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProbeOptions {
    pub model: ModelOptions,
    pub transcribe: TranscribeOptions,
    pub repeat: u32,
    pub stream: bool,
    /// Cancel one extra run after this many ms and report the abort latency.
    pub cancel_after_ms: Option<u64>,
    /// Run Sigil's ONNX Runtime canary on this EP before loading and after the runs.
    pub ort_canary: Option<&'static str>,
    pub whisper_rs: Option<PathBuf>,
}

fn value<'a>(it: &mut std::slice::Iter<'a, String>, flag: &str) -> Result<&'a str, String> {
    it.next()
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn number<T: std::str::FromStr>(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<T, String> {
    let v = value(it, flag)?;
    v.parse()
        .map_err(|_| format!("{flag}: '{v}' is not a valid number"))
}

/// Parse `EngineProbeArgs` (+ the extra arguments after `--`).
pub(crate) fn parse(args: &EngineProbeArgs) -> Result<ProbeOptions, String> {
    let model = args
        .model
        .clone()
        .ok_or("engine-probe transcribe: give the model file (GGUF or whisper.cpp .bin)")?;
    let mut m = ModelOptions::new(model);
    m.runtime_dir = args.runtime_dir.clone();
    if let Some(v) = args.variant.as_deref() {
        m.variant = Some(RuntimeVariant::parse(v).ok_or_else(|| format!("unknown variant '{v}' (cuda13, vulkan, cpu-avx512, cpu-avx2, cpu)"))?);
    }
    let mut t = TranscribeOptions::default();
    let mut repeat = 3u32;
    let mut stream = false;
    let mut cancel_after_ms = None;
    let mut ort_canary = None;
    let mut whisper_rs = None;
    let mut it = args.extra.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--repeat" => repeat = number(&mut it, flag)?,
            "--backend" => {
                m.backend = match value(&mut it, flag)? {
                    "auto" => BackendChoice::Auto,
                    "gpu" => BackendChoice::Gpu,
                    "cpu" => BackendChoice::Cpu,
                    other => return Err(format!("--backend: '{other}' (auto, gpu or cpu)")),
                }
            }
            "--threads" => m.threads = Some(number(&mut it, flag)?),
            "--device" => m.device = Some(number(&mut it, flag)?),
            "--device-id" => m.device_id = Some(value(&mut it, flag)?.to_string()),
            "--warmup-seconds" => m.warmup_seconds = Some(number(&mut it, flag)?),
            "--min-free-vram-mb" => m.min_free_vram_mb = number(&mut it, flag)?,
            "--n-ctx" => m.n_ctx = Some(number(&mut it, flag)?),
            "--kv" => {
                m.kv_type = match value(&mut it, flag)? {
                    "auto" => KvType::Auto,
                    "f16" => KvType::F16,
                    "f32" => KvType::F32,
                    other => return Err(format!("--kv: '{other}' (auto, f16 or f32)")),
                }
            }
            "--no-cpu-fallback" => m.cpu_fallback = false,
            "--no-warmup" => m.warmup = false,
            "--language" => t.language = Some(value(&mut it, flag)?.to_string()),
            "--translate" => t.translate = true,
            "--timestamps" => {
                t.timestamps = match value(&mut it, flag)? {
                    "none" => Timestamps::None,
                    "segment" => Timestamps::Segment,
                    "word" => Timestamps::Word,
                    "token" => Timestamps::Token,
                    "auto" => Timestamps::Auto,
                    other => return Err(format!("--timestamps: '{other}'")),
                }
            }
            "--words" => t.words = true,
            "--prompt" => t.prompt = Some(value(&mut it, flag)?.to_string()),
            "--no-temperature-fallback" => t.temperature_fallback = Some(false),
            "--stream" => stream = true,
            "--cancel-after-ms" => cancel_after_ms = Some(number(&mut it, flag)?),
            "--ort-canary" => {
                ort_canary = Some(match value(&mut it, flag)? {
                    "cuda" => crate::services::neural_audio::ort_session::EP_CUDA,
                    "dml" | "directml" => crate::services::neural_audio::ort_session::EP_DML,
                    "cpu" => crate::services::neural_audio::ort_session::EP_CPU,
                    other => return Err(format!("--ort-canary: '{other}' (cuda, dml or cpu)")),
                })
            }
            "--whisper-rs" => whisper_rs = Some(PathBuf::from(value(&mut it, flag)?)),
            other => return Err(format!("engine-probe transcribe: unknown option '{other}'")),
        }
    }
    if !(1..=50).contains(&repeat) {
        return Err("--repeat must be 1..=50".into());
    }
    Ok(ProbeOptions {
        model: m,
        transcribe: t,
        repeat,
        stream,
        cancel_after_ms,
        ort_canary,
        whisper_rs,
    })
}

/// Read a WAV (PCM 8/16/24/32-bit or float) as interleaved f32.
pub(crate) fn read_wav(path: &Path) -> Result<(Vec<f32>, u32, u16), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
        }
    }
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((samples, spec.sample_rate, spec.channels))
}

/// This process's local VRAM use per hardware adapter (MiB), keyed by adapter name.
fn vram_snapshot() -> Vec<Value> {
    gpu_mem::adapters()
        .into_iter()
        .filter(|a| !a.software)
        .map(|a| {
            let usage = gpu_mem::process_vram_usage(a.index);
            json!({
                "adapter": a.index,
                "name": a.name,
                "process_mb": usage.map(|u| u.process_usage_mb),
                "budget_mb": usage.map(|u| u.budget_mb),
            })
        })
        .collect()
}

fn vram_delta(before: &[Value], after: &[Value]) -> Vec<Value> {
    before
        .iter()
        .zip(after)
        .map(|(b, a)| {
            let d = match (b["process_mb"].as_i64(), a["process_mb"].as_i64()) {
                (Some(x), Some(y)) => json!(y - x),
                _ => Value::Null,
            };
            json!({ "adapter": b["adapter"], "name": b["name"], "delta_mb": d })
        })
        .collect()
}

fn stats(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let median = if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    json!({ "min": v[0], "median": median, "max": v[v.len() - 1] })
}

/// whisper-rs (Sigil's own statically linked ggml) on the same audio: `(text, wall ms)`.
fn run_whisper_rs(ctx: &whisper_rs::WhisperContext, pcm: &[f32], language: Option<&str>) -> Result<(String, f64), String> {
    use whisper_rs::{FullParams, SamplingStrategy};
    let started = Instant::now();
    let mut state = ctx
        .create_state()
        .map_err(|e| format!("whisper-rs state: {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(super::types::default_threads(false) as i32);
    params.set_language(Some(language.unwrap_or("auto")));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    state
        .full(params, pcm)
        .map_err(|e| format!("whisper-rs: {e}"))?;
    let mut bytes = Vec::new();
    for seg in state.as_iter() {
        if let Ok(b) = seg.to_bytes() {
            bytes.extend_from_slice(b);
        }
    }
    Ok((
        String::from_utf8_lossy(&bytes).trim().to_string(),
        started.elapsed().as_secs_f64() * 1000.0,
    ))
}

/// Sigil's own ONNX Runtime canary (a tiny convolution) on `ep`: proves ORT's sessions keep
/// working next to the engine DLL's ggml / CUDA state in one process.
fn ort_canary(ep: &str) -> Value {
    match crate::services::diagnostics::canary::run(ep, 0) {
        Ok(ms) => json!({ "ok": true, "ms": ms }),
        Err(e) => json!({ "ok": false, "error": e }),
    }
}

fn loaded_ggml_modules() -> Vec<String> {
    #[cfg(windows)]
    {
        use windows::core::PCWSTR;
        use windows::Win32::System::LibraryLoader::GetModuleHandleW;
        [
            "ggml.dll",
            "ggml-base.dll",
            "ggml-cpu.dll",
            "ggml-cuda.dll",
            "ggml-vulkan.dll",
            "whisper.dll",
        ]
        .iter()
        .filter(|name| {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: a NUL-terminated wide string; GetModuleHandleW does not take a reference.
            unsafe { GetModuleHandleW(PCWSTR(wide.as_ptr())) }.is_ok()
        })
        .map(|n| (*n).to_string())
        .collect()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// Body of `Sigil.exe --headless engine-probe transcribe ...`.
pub fn run(args: &EngineProbeArgs) -> Result<String, String> {
    let opts = parse(args)?;
    let pcm = match &args.wav {
        Some(path) => {
            let (samples, rate, channels) = read_wav(path)?;
            Some((to_16k_mono(&samples, rate, channels).map_err(String::from)?, rate, channels, path.clone()))
        }
        None => None,
    };

    let vram_start = vram_snapshot();
    let mut report = serde_json::Map::new();
    report.insert("engine".into(), json!("transcribe"));

    // whisper-rs first, so its static ggml is initialised before the engine DLL's own copy.
    let whisper_ctx = match (&opts.whisper_rs, &pcm) {
        (Some(model), Some(_)) => Some(
            whisper_rs::WhisperContext::new_with_params(model, whisper_rs::WhisperContextParameters::default())
                .map_err(|e| format!("whisper-rs could not load {}: {e}", model.display()))?,
        ),
        _ => None,
    };
    let whisper_before = match (&whisper_ctx, &pcm) {
        (Some(ctx), Some((audio, ..))) => Some(run_whisper_rs(ctx, audio, opts.transcribe.language.as_deref())?),
        _ => None,
    };

    if opts.ort_canary.is_some() {
        // Development CUDA / cuDNN folders (SIGIL_TRANSCRIBE_CUDA_BIN) for ORT's provider too.
        super::engine::register_dev_cuda_dirs();
    }
    let canary_before = opts.ort_canary.map(ort_canary);

    // Baseline for the engine's own VRAM: after the ORT canary's device / session and whisper-rs.
    let vram_before_engine = vram_snapshot();
    let started = Instant::now();
    let mut model_opts = opts.model.clone();
    let warmup_requested = model_opts.warmup;
    model_opts.warmup = false;
    let mut model = TranscribeModel::open(&model_opts).map_err(String::from)?;
    let open_ms = started.elapsed().as_secs_f64() * 1000.0;
    let vram_loaded = vram_snapshot();
    let warmup_ms = if warmup_requested {
        Some(model.warmup().map_err(String::from)?)
    } else {
        None
    };
    // After the warm-up: on a GPU, runs no longer than the warm-up should not grow past this.
    let vram_warm = vram_snapshot();

    report.insert("runtime".into(), serde_json::to_value(model.runtime()).unwrap_or(Value::Null));
    report.insert("model".into(), serde_json::to_value(model.info()).unwrap_or(Value::Null));
    report.insert("languages".into(), json!(model.info().capabilities.language_codes().len()));
    report.insert("fallback".into(), json!(model.fallback_reason()));
    report.insert("open_ms".into(), json!(open_ms));
    report.insert("warmup_ms".into(), json!(warmup_ms));

    if let Some((audio, rate, channels, path)) = &pcm {
        let audio_ms = audio.len() as f64 * 1000.0 / f64::from(SAMPLE_RATE);
        report.insert(
            "audio".into(),
            json!({ "path": path.display().to_string(), "ms": audio_ms, "source_rate": rate, "source_channels": channels }),
        );
        let mut runs = Vec::new();
        let mut rtfs = Vec::new();
        let mut last = None;
        for _ in 0..opts.repeat {
            let t = model
                .transcribe(audio, &opts.transcribe)
                .map_err(String::from)?;
            rtfs.push(t.rtf);
            runs.push(json!({
                "wall_ms": t.wall_ms,
                "rtf": t.rtf,
                "mel_ms": t.timings.mel_ms,
                "encode_ms": t.timings.encode_ms,
                "decode_ms": t.timings.decode_ms,
            }));
            last = Some(t);
        }
        report.insert("runs".into(), Value::Array(runs));
        report.insert("rtf".into(), stats(&rtfs));
        report.insert("transcript".into(), serde_json::to_value(&last).unwrap_or(Value::Null));

        if let Some(after) = opts.cancel_after_ms {
            // Abort latency: transcribe.cpp polls between decode steps (not inside an encoder pass).
            let canceller = model.canceller();
            let timer = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(after));
                canceller.cancel();
            });
            let started = Instant::now();
            let result = model.transcribe(audio, &opts.transcribe);
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let _ = timer.join();
            let outcome = match &result {
                Ok(_) => "completed",
                Err(e) if e.kind == super::engine::ErrorKind::Cancelled => "cancelled",
                Err(_) => "failed",
            };
            report.insert(
                "cancel".into(),
                json!({
                    "after_ms": after,
                    "outcome": outcome,
                    "wall_ms": wall_ms,
                    "abort_latency_ms": (outcome == "cancelled").then(|| (wall_ms - after as f64).max(0.0)),
                    "error": result.err().map(|e| e.message),
                }),
            );
        }

        if opts.stream {
            let stream_report = match model.open_stream(&opts.transcribe) {
                Ok(mut s) => {
                    let started = Instant::now();
                    let mut updates = 0u32;
                    let mut first_partial_ms = None;
                    for chunk in audio.chunks(SAMPLE_RATE as usize / 5) {
                        if let Some(u) = s.push(chunk).map_err(String::from)? {
                            updates += 1;
                            if first_partial_ms.is_none() && !u.partial.is_empty() {
                                first_partial_ms = Some(u.input_ms);
                            }
                        }
                    }
                    let fin = s.finish().map_err(String::from)?;
                    json!({
                        "updates": updates,
                        "first_partial_audio_ms": first_partial_ms,
                        "wall_ms": started.elapsed().as_secs_f64() * 1000.0,
                        "final": fin,
                    })
                }
                Err(e) => json!({ "error": e.message }),
            };
            report.insert("stream".into(), stream_report);
        }
    }
    let vram_after = vram_snapshot();
    if let Some(ep) = opts.ort_canary {
        report.insert(
            "ort_canary".into(),
            json!({ "ep": ep, "before_engine": canary_before, "after_runs": ort_canary(ep) }),
        );
    }
    report.insert(
        "vram".into(),
        json!({
            "start": vram_start,
            "before_engine": vram_before_engine,
            "after_load": vram_loaded,
            "after_warmup": vram_warm,
            "after_runs": vram_after,
            "load_delta": vram_delta(&vram_before_engine, &vram_loaded),
            "run_growth": vram_delta(&vram_warm, &vram_after),
            "total_delta": vram_delta(&vram_before_engine, &vram_after),
            "process_delta": vram_delta(&vram_start, &vram_after),
            "device": model.device_memory().ok().flatten(),
        }),
    );

    if let (Some(ctx), Some((audio, ..)), Some((text_before, ms_before))) = (&whisper_ctx, &pcm, whisper_before) {
        let (text_after, ms_after) = run_whisper_rs(ctx, audio, opts.transcribe.language.as_deref())?;
        report.insert(
            "whisper_rs".into(),
            json!({
                "model": opts.whisper_rs.as_ref().map(|p| p.display().to_string()),
                "before": { "text": text_before, "wall_ms": ms_before },
                "after": { "text": text_after, "wall_ms": ms_after },
                "stable": text_before == text_after,
            }),
        );
    }
    report.insert("ggml_dlls_loaded".into(), json!(loaded_ggml_modules()));
    report.insert("ok".into(), json!(true));
    serde_json::to_string_pretty(&Value::Object(report)).map_err(|e| e.to_string())
}
