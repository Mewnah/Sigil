//! `Sigil.exe --headless engine-probe sherpa [model] [wav] [--runtime-dir <dir>] [-- key=value ...]`
//!
//! Loads the runtime pack (and a model), runs it, and returns a JSON report: versions, which
//! `onnxruntime.dll` the process bound, load times, RTF, CPU cores, idle CPU after decoding (ORT
//! spinning check), working set, VRAM, and the output (transcript / language / audio stats).
//!
//! Extra arguments after `--` (all optional):
//! - `kind=<family>`, `mode=online|offline`, `fp32=1`, `vocoder=<file>`: model detection hints
//!   ([`super::detect::DetectHints`]).
//! - `threads=N` (default 2; 1 for streaming models), `lang=<code>`, `model_type=<t>`, `hotwords=<text>`.
//! - `repeat=N` (default 3) decode runs; `chunk_ms=N` (streaming feed, default 100);
//!   `idle_ms=N` (idle window after decoding, default 1000; 0 skips it).
//! - `text=<sentence>`, `sid=N`, `speed=F`, `silence=F` (pause scale, 0.01-10), `out=<file.wav>`: TTS.
//! - `config=<file.json>`: load exactly this serialized [`SherpaModel`] instead of detecting the
//!   model folder (the positional model is then optional, and detection hints and `threads` /
//!   `lang` / `model_type` do not apply). [`validate_in_child`] passes its config this way.
//! - `load_only=1`: load the model and stop (used by [`validate_in_child`]).
//! - `validate_child=1`: first load the final config (detection plus every override) in a child
//!   process through [`validate_in_child`] (reported as `child_check`).
//! - `cuda_load=<duty>` (0-1], `load_ep=cuda|dml|cpu`: run the decodes again while Sigil's own ONNX
//!   Runtime keeps a GPU session busy (the benchmark's MatMul load generator) and check both
//!   runtimes: sherpa's output must not change, and Sigil's EP canary must still pass afterwards.
//! - `cuda_preload=<dir>[;<dir>...]` (development): preload the CUDA 13 cuBLAS and cuDNN 9 DLLs
//!   found in these folders by full path, like the NVIDIA runtime pack does, so the contention run
//!   can use CUDA on a PC without the pack installed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use super::detect::{detect, DetectHints, SherpaModel};
use super::lib::{runtime_dir, SherpaRuntime, RUNTIME_ID};
use super::lid::SpokenLanguageId;
use super::offline::{DecodeOptions, OfflineModel, OfflineRecognizer};
use super::online::OnlineRecognizer;
use super::speaker::{cosine_similarity, SpeakerEmbeddingExtractor};
use super::tts::{GenerateOptions, OfflineTts, TtsModel};
use crate::services::bench::metrics::{prepare_cpu_meter, process_cpu_time, process_vram_mb, Summary};
use crate::services::speech_rt::EngineProbeArgs;

const DEFAULT_TEXT: &str = "Sigil speech engine check. The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.";

/// Parsed `key=value` extras.
#[derive(Debug, Default)]
pub(super) struct Extra(BTreeMap<String, String>);

impl Extra {
    pub(super) fn parse(args: &[String]) -> Result<Extra, String> {
        let mut map = BTreeMap::new();
        for a in args {
            let (k, v) = a
                .split_once('=')
                .ok_or_else(|| format!("engine-probe sherpa: expected key=value after `--`, got {a:?}"))?;
            map.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
        Ok(Extra(map))
    }

    pub(super) fn str(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str).filter(|v| !v.is_empty())
    }

    pub(super) fn flag(&self, k: &str) -> bool {
        matches!(self.str(k), Some("1" | "true" | "yes" | "on"))
    }

    pub(super) fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.str(k) {
            None => Ok(None),
            Some(v) => v
                .parse()
                .map(Some)
                .map_err(|_| format!("engine-probe sherpa: {k}={v} is not a number")),
        }
    }
}

/// Process memory in MiB (working set, private commit, peak working set).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MemorySample {
    pub rss_mb: f64,
    pub private_mb: f64,
    pub peak_rss_mb: f64,
}

/// Working set / private bytes of this process (`K32GetProcessMemoryInfo`, kernel32; `None`
/// elsewhere).
pub fn memory_sample() -> Option<MemorySample> {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
        }
        let mut c = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..ProcessMemoryCounters::default()
        };
        // SAFETY: the pseudo handle of this process and a correctly sized, writable struct.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0;
        let mb = |b: usize| (b as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0;
        ok.then(|| MemorySample {
            rss_mb: mb(c.working_set_size),
            private_mb: mb(c.pagefile_usage),
            peak_rss_mb: mb(c.peak_working_set_size),
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Mono samples of a WAV file (any PCM / float format, channels averaged).
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{}: {e}", path.display()))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("{}: {e}", path.display()))?
        }
    };
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|f| f.iter().sum::<f32>() / f.len() as f32)
            .collect()
    };
    if mono.is_empty() {
        return Err(format!("{} holds no audio", path.display()));
    }
    Ok((mono, spec.sample_rate))
}

pub(super) fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    for &s in samples {
        w.write_sample(s)
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    w.finalize().map_err(|e| format!("{}: {e}", path.display()))
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Wall time and process CPU time of `f`.
fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration, Option<Duration>) {
    let cpu0 = process_cpu_time();
    let t0 = Instant::now();
    let out = f();
    let wall = t0.elapsed();
    let cpu = match (cpu0, process_cpu_time()) {
        (Some(a), Some(b)) => Some(b.saturating_sub(a)),
        _ => None,
    };
    (out, wall, cpu)
}

/// Process CPU used while this thread sleeps `window` (ONNX Runtime spin-wait check).
fn idle_cpu_ms_per_s(window: Duration) -> Option<f64> {
    if window.is_zero() {
        return None;
    }
    let c0 = process_cpu_time()?;
    let t0 = Instant::now();
    std::thread::sleep(window);
    let c1 = process_cpu_time()?;
    Some(round3(c1.saturating_sub(c0).as_secs_f64() * 1000.0 / t0.elapsed().as_secs_f64()))
}

/// What one probe workload produces per run.
struct RunOutput {
    /// Seconds of audio processed (ASR / LID / speaker) or produced (TTS).
    audio_s: f64,
    /// Comparable output (transcript, language, ...) for the determinism / isolation checks.
    key: String,
    detail: Value,
}

/// A loaded model plus the workload the probe runs on it.
enum Workload {
    Offline(OfflineRecognizer, Vec<f32>, u32, DecodeOptions),
    Online(OnlineRecognizer, Vec<f32>, u32, usize),
    Tts(Box<OfflineTts>, String, GenerateOptions),
    Lid(SpokenLanguageId, Vec<f32>, u32),
    Speaker(SpeakerEmbeddingExtractor, Vec<f32>, u32),
}

impl Workload {
    fn run(&self) -> Result<RunOutput, String> {
        match self {
            Workload::Offline(rec, samples, rate, opts) => {
                let r = rec.transcribe(*rate, samples, opts)?;
                Ok(RunOutput {
                    audio_s: samples.len() as f64 / f64::from(*rate),
                    key: r.text.clone(),
                    detail: serde_json::to_value(&r).unwrap_or(Value::Null),
                })
            }
            Workload::Online(rec, samples, rate, chunk) => {
                let mut stream = rec.create_stream()?;
                let mut chunk_ms = Vec::new();
                let mut first_partial_s = None;
                let mut fed = 0usize;
                for c in samples.chunks(*chunk) {
                    let t0 = Instant::now();
                    stream.accept_waveform(*rate, c)?;
                    stream.decode();
                    chunk_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    fed += c.len();
                    if first_partial_s.is_none() && !stream.result()?.text.is_empty() {
                        first_partial_s = Some(round3(fed as f64 / f64::from(*rate)));
                    }
                }
                // Right context of the last chunk (NeMo cache-aware models need a tail).
                stream.accept_waveform(*rate, &vec![0.0; (*rate as usize) * 3 / 10])?;
                stream.input_finished();
                stream.decode();
                let r = stream.result()?;
                Ok(RunOutput {
                    audio_s: samples.len() as f64 / f64::from(*rate),
                    key: r.text.clone(),
                    detail: json!({
                        "text": r.text,
                        "tokens": r.tokens.len(),
                        "first_partial_at_s": first_partial_s,
                        "chunk_ms": Summary::of(&chunk_ms),
                        "chunk_audio_ms": round3(*chunk as f64 * 1000.0 / f64::from(*rate)),
                    }),
                })
            }
            Workload::Tts(tts, text, opts) => {
                let mut first_chunk: Option<Duration> = None;
                let t0 = Instant::now();
                let mut cb = |_: &[f32]| {
                    first_chunk.get_or_insert_with(|| t0.elapsed());
                    true
                };
                let audio = tts.generate(text, opts, None, Some(&mut cb))?;
                let n = audio.samples.len().max(1) as f64;
                let rms = (audio
                    .samples
                    .iter()
                    .map(|s| f64::from(*s) * f64::from(*s))
                    .sum::<f64>()
                    / n)
                    .sqrt();
                let peak = audio.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
                let audio_s = audio.samples.len() as f64 / f64::from(audio.sample_rate.max(1));
                Ok(RunOutput {
                    audio_s,
                    key: format!("{}", audio.samples.len()),
                    detail: json!({
                        "samples": audio.samples.len(),
                        "sample_rate": audio.sample_rate,
                        "audio_s": round3(audio_s),
                        "first_chunk_ms": first_chunk.map(ms),
                        "rms": round3(rms),
                        "peak": round3(f64::from(peak)),
                        "num_speakers": tts.num_speakers(),
                    }),
                })
            }
            Workload::Lid(lid, samples, rate) => {
                let lang = lid.identify(*rate, samples)?;
                Ok(RunOutput {
                    audio_s: samples.len() as f64 / f64::from(*rate),
                    key: lang.clone(),
                    detail: json!({ "language": lang }),
                })
            }
            Workload::Speaker(ex, samples, rate) => {
                let e = ex.embed(*rate, samples)?;
                let half = samples.len() / 2;
                let halves = if half as f64 / f64::from(*rate) >= 1.5 {
                    match (ex.embed(*rate, &samples[..half]), ex.embed(*rate, &samples[half..])) {
                        (Ok(a), Ok(b)) => cosine_similarity(&a, &b),
                        _ => None,
                    }
                } else {
                    None
                };
                let norm = e
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>()
                    .sqrt();
                Ok(RunOutput {
                    audio_s: samples.len() as f64 / f64::from(*rate),
                    key: format!("{}", e.len()),
                    detail: json!({
                        "dim": e.len(),
                        "norm": round3(norm),
                        "halves_cosine": halves.map(|c| round3(f64::from(c))),
                    }),
                })
            }
        }
    }

    /// Save TTS output for listening.
    fn save_tts(&self, out: &Path) -> Result<(), String> {
        if let Workload::Tts(tts, text, opts) = self {
            let audio = tts.generate(text, opts, None, None)?;
            write_wav(out, &audio.samples, audio.sample_rate)?;
        }
        Ok(())
    }
}

fn apply_overrides(model: &mut SherpaModel, extra: &Extra) -> Result<(), String> {
    let threads: Option<u32> = extra.num("threads")?;
    let lang = extra.str("lang").map(str::to_string);
    let model_type = extra.str("model_type").map(str::to_string);
    match model {
        SherpaModel::Offline(c) => {
            if threads.is_some() {
                c.num_threads = threads;
            }
            if model_type.is_some() {
                c.model_type = model_type;
            }
            match &mut c.model {
                OfflineModel::SenseVoice { language, .. }
                | OfflineModel::Whisper { language, .. }
                | OfflineModel::CohereTranscribe { language, .. } => {
                    if lang.is_some() {
                        *language = lang;
                    }
                }
                OfflineModel::Canary { src_lang, tgt_lang, .. } => {
                    if lang.is_some() {
                        src_lang.clone_from(&lang);
                        *tgt_lang = lang;
                    }
                }
                _ => {}
            }
        }
        SherpaModel::Online(c) => {
            if threads.is_some() {
                c.num_threads = threads;
            }
            if model_type.is_some() {
                c.model_type = model_type;
            }
        }
        SherpaModel::Tts(c) => {
            if threads.is_some() {
                c.num_threads = threads;
            }
            if let TtsModel::Kokoro { lang: l, .. } = &mut c.model {
                if lang.is_some() {
                    *l = lang;
                }
            }
        }
        SherpaModel::LanguageId(c) => {
            if threads.is_some() {
                c.num_threads = threads;
            }
        }
        SherpaModel::SpeakerEmbedding(c) => {
            if threads.is_some() {
                c.num_threads = threads;
            }
        }
    }
    Ok(())
}

fn build_workload(rt: &Arc<SherpaRuntime>, model: &SherpaModel, wav: Option<&(Vec<f32>, u32)>, extra: &Extra) -> Result<Workload, String> {
    // `load_only` checks that the model loads; it needs no input audio.
    let load_only = extra.flag("load_only");
    let need_wav = || match wav {
        Some(w) => Ok(w.clone()),
        None if load_only => Ok((Vec::new(), 16_000)),
        None => Err("engine-probe sherpa: this model needs a WAV file".to_string()),
    };
    Ok(match model {
        SherpaModel::Offline(c) => {
            let (s, r) = need_wav()?;
            let opts = DecodeOptions {
                language: None,
                hotwords: extra.str("hotwords").map(str::to_string),
            };
            Workload::Offline(OfflineRecognizer::new(rt, c)?, s, r, opts)
        }
        SherpaModel::Online(c) => {
            let (s, r) = need_wav()?;
            let chunk_ms: u32 = extra.num("chunk_ms")?.unwrap_or(100).clamp(10, 2000);
            let chunk = (r as usize * chunk_ms as usize / 1000).max(1);
            Workload::Online(OnlineRecognizer::new(rt, c)?, s, r, chunk)
        }
        SherpaModel::Tts(c) => {
            let text = extra.str("text").unwrap_or(DEFAULT_TEXT).to_string();
            let opts = GenerateOptions {
                sid: extra.num("sid")?.unwrap_or(0),
                speed: extra.num("speed")?.unwrap_or(1.0),
                silence_scale: extra.num("silence")?,
            };
            Workload::Tts(Box::new(OfflineTts::new(rt, c)?), text, opts)
        }
        SherpaModel::LanguageId(c) => {
            let (s, r) = need_wav()?;
            Workload::Lid(SpokenLanguageId::new(rt, c)?, s, r)
        }
        SherpaModel::SpeakerEmbedding(c) => {
            let (s, r) = need_wav()?;
            Workload::Speaker(SpeakerEmbeddingExtractor::new(rt, c)?, s, r)
        }
    })
}

/// Run `workload` `repeat` times; returns the per-run JSON and the outputs' keys.
fn run_series(workload: &Workload, repeat: usize) -> Result<(Value, Vec<String>, Value), String> {
    let mut rtf = Vec::new();
    let mut runs = Vec::new();
    let mut keys = Vec::new();
    let mut last = Value::Null;
    for _ in 0..repeat {
        let (out, wall, cpu) = timed(|| workload.run());
        let out = out?;
        let r = if out.audio_s > 0.0 { wall.as_secs_f64() / out.audio_s } else { f64::NAN };
        rtf.push(r);
        runs.push(json!({
            "wall_ms": ms(wall),
            "rtf": round3(r),
            "cpu_cores": cpu.map(|c| round3(c.as_secs_f64() / wall.as_secs_f64().max(1e-9))),
        }));
        keys.push(out.key);
        last = out.detail;
    }
    let s = Summary::of(&rtf);
    Ok((
        json!({
            "runs": runs,
            "rtf_min": s.map(|s| round3(s.min)),
            "rtf_median": s.map(|s| round3(s.p50)),
            "rtf_max": s.map(|s| round3(s.max)),
            "deterministic": keys.windows(2).all(|w| w[0] == w[1]),
        }),
        keys,
        last,
    ))
}

/// Development helper for `cuda_preload`: preload the CUDA 13 cuBLAS and cuDNN 9 DLLs present in
/// `dirs` (`;`-separated) by full path; returns the files that loaded.
fn preload_cuda(dirs: &str) -> Vec<String> {
    let names = ["cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll"]
        .into_iter()
        .chain(ort::ep::cuda::CUDNN_DYLIBS.iter().copied());
    let names: Vec<&str> = names.collect();
    let mut loaded = Vec::new();
    for dir in dirs.split(';').map(str::trim).filter(|d| !d.is_empty()) {
        for name in &names {
            let path = Path::new(dir).join(name);
            if path.is_file() && ort::util::preload_dylib(&path).is_ok() {
                loaded.push(path.display().to_string());
            }
        }
    }
    loaded
}

fn canary(ep: &str) -> Value {
    match crate::services::diagnostics::canary::run(ep, 0) {
        Ok(ms) => json!({ "ep": ep, "ok": true, "ms": ms }),
        Err(e) => json!({ "ep": ep, "ok": false, "error": e }),
    }
}

/// The Sigil-side ORT contention run (see the module docs).
fn cuda_contention(workload: &Workload, repeat: usize, duty: f64, ep: &str, reference: &[String]) -> Result<Value, String> {
    use crate::services::bench::load_sim::LoadHandle;
    use crate::services::bench::spec::LoadSpec;
    if !(duty > 0.0 && duty <= 1.0) {
        return Err(format!("engine-probe sherpa: cuda_load={duty} must be in (0, 1]"));
    }
    let canary_before = canary(ep);
    let spec: LoadSpec =
        serde_json::from_value(json!({ "gpu": { "duty": duty, "execution_provider": ep } })).map_err(|e| format!("load spec: {e}"))?;
    let vram_before = process_vram_mb();
    let handle = LoadHandle::start(&spec)?;
    let vram_loaded = process_vram_mb();
    let series = run_series(workload, repeat);
    let load = handle.stop();
    let (series, keys, detail) = series?;
    let canary_after = canary(ep);
    let canary_cpu_after = canary("cpu");
    Ok(json!({
        "duty": duty,
        "gpu_load": load.gpu,
        "vram_mb_before_load": vram_before,
        "vram_mb_with_load": vram_loaded,
        "decode": series,
        "output": detail,
        "output_matches_unloaded": keys.iter().all(|k| reference.first() == Some(k)),
        "sigil_canary_before": canary_before,
        "sigil_canary_after": canary_after,
        "sigil_cpu_canary_after": canary_cpu_after,
    }))
}

/// Body of `Sigil.exe --headless engine-probe sherpa ...`.
pub fn headless_probe(args: &EngineProbeArgs) -> Result<String, String> {
    let extra = Extra::parse(&args.extra)?;
    if let Some(v) = args.variant.as_deref() {
        if v != "cpu" {
            return Err(format!("the sherpa-onnx runtime pack is CPU-only; variant {v:?} does not exist"));
        }
    }
    prepare_cpu_meter();
    let mem_start = memory_sample();
    let dir = match &args.runtime_dir {
        Some(d) => d.clone(),
        None => runtime_dir().ok_or_else(|| {
            format!(
                "the sherpa-onnx runtime pack is not installed: run `Sigil.exe --headless fetch {RUNTIME_ID}` or pass \
                 --runtime-dir <folder with onnxruntime.dll and sherpa-onnx-c-api.dll>"
            )
        })?,
    };
    let (rt, load_wall, _) = timed(|| SherpaRuntime::load(&dir));
    let rt = rt?;
    let mem_runtime = memory_sample();
    let mut report = json!({
        "engine": "sherpa-onnx",
        "runtime": rt.info(),
        "runtime_load_ms": ms(load_wall),
        "sigil_onnxruntime_version": crate::services::neural_audio::ort_session::runtime_version(),
        "memory_mb": { "start": mem_start, "after_runtime": mem_runtime },
    });

    let (model, source) = if let Some(file) = extra.str("config") {
        // The exact config (from `validate_in_child`): no detection, no overrides.
        (read_model_config(Path::new(file))?, file.to_string())
    } else {
        let Some(model_path) = args.model.as_deref() else {
            report["note"] = json!("runtime only (pass a model folder and a WAV to run a model)");
            return serde_json::to_string_pretty(&report).map_err(|e| e.to_string());
        };
        let hints = DetectHints {
            kind: extra.str("kind").map(str::to_string),
            mode: extra.str("mode").map(str::to_lowercase),
            prefer_fp32: extra.flag("fp32"),
            vocoder: extra.str("vocoder").map(PathBuf::from),
        };
        let mut model = detect(model_path, &hints)?;
        apply_overrides(&mut model, &extra)?;
        (model, model_path.display().to_string())
    };
    report["model"] = json!({ "path": source, "task": model.label(), "config": &model });
    if extra.flag("validate_child") {
        // Exercise the isolated first-load check P2 uses for models from outside the catalog, on
        // exactly the config loaded below.
        let started = Instant::now();
        match validate_in_child(&model, Duration::from_secs(120)) {
            Ok(child) => {
                report["child_check"] = json!({ "ok": true, "ms": ms(started.elapsed()), "task": child["model"]["task"] });
            }
            // Never load in-process what failed in isolation (it may abort this process too).
            Err(e) => return Err(format!("the model failed the isolated load check: {e}")),
        }
    }

    let wav = match args.wav.as_deref() {
        Some(p) => {
            let (s, r) = read_wav(p)?;
            report["input"] = json!({
                "wav": p.display().to_string(),
                "sample_rate": r,
                "duration_s": round3(s.len() as f64 / f64::from(r)),
            });
            Some((s, r))
        }
        None => None,
    };

    let vram_before_model = process_vram_mb();
    let (workload, model_wall, model_cpu) = timed(|| build_workload(&rt, &model, wav.as_ref(), &extra));
    let workload = workload?;
    report["model_load_ms"] = json!(ms(model_wall));
    report["model_load_cpu_ms"] = json!(model_cpu.map(ms));
    report["memory_mb"]["after_model"] = json!(memory_sample());
    report["vram_mb"] = json!({ "before_model": vram_before_model, "after_model": process_vram_mb() });
    if extra.flag("load_only") {
        report["note"] = json!("model loaded (load_only)");
        return serde_json::to_string_pretty(&report).map_err(|e| e.to_string());
    }

    let repeat: usize = extra.num("repeat")?.unwrap_or(3).clamp(1, 100);
    let (series, keys, detail) = run_series(&workload, repeat)?;
    report["decode"] = series;
    report["output"] = detail;
    report["memory_mb"]["after_decode"] = json!(memory_sample());
    let idle_ms: u64 = extra.num("idle_ms")?.unwrap_or(1000).min(60_000);
    report["idle_cpu_ms_per_s_after_decode"] = json!(idle_cpu_ms_per_s(Duration::from_millis(idle_ms)));

    if let Some(out) = extra.str("out") {
        workload.save_tts(Path::new(out))?;
        report["saved"] = json!(out);
    }
    if let Some(duty) = extra.num::<f64>("cuda_load")? {
        let ep = extra.str("load_ep").unwrap_or("cuda");
        if let Some(dirs) = extra.str("cuda_preload") {
            report["cuda_preloaded"] = json!(preload_cuda(dirs));
        }
        report["contention"] = cuda_contention(&workload, repeat, duty, ep, &keys)?;
        report["memory_mb"]["after_contention"] = json!(memory_sample());
    }
    report["onnxruntime_modules"] = json!({
        "onnxruntime.dll": super::lib::loaded_module_path("onnxruntime.dll").map(|p| p.display().to_string()),
    });
    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}

/// A serialized [`SherpaModel`] (the `config=<file>` extra).
pub(super) fn read_model_config(path: &Path) -> Result<SherpaModel, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("engine-probe sherpa: cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("engine-probe sherpa: {} is not a sherpa model config: {e}", path.display()))
}

/// Removes the child's config file however the check ends.
struct TempConfig(PathBuf);

impl TempConfig {
    fn write(model: &SherpaModel) -> Result<TempConfig, String> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let json = serde_json::to_string(model).map_err(|e| format!("cannot serialize the model config: {e}"))?;
        let path = std::env::temp_dir().join(format!(
            "sigil-sherpa-check-{}-{}.json",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, json).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(TempConfig(path))
    }
}

impl Drop for TempConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Load `model` once in a child
/// `Sigil.exe --headless engine-probe sherpa -- load_only=1 config=<file>` and return its report.
/// sherpa-onnx aborts the process on some malformed model files (C++ exceptions cannot cross the C
/// API), so a model that did not come from the hash-pinned catalog (a user's own folder) is
/// loaded here first; only a model that loads cleanly is then loaded in-process.
///
/// The child loads exactly `model` (serialized, never re-detected), so pass the final config the
/// caller will load: after detection and every setting layered on top (precision, model type,
/// language, decoding, threads). A different config can pick different files.
pub fn validate_in_child(model: &SherpaModel, timeout: Duration) -> Result<Value, String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate Sigil.exe: {e}"))?;
    // Refuse what the in-process load would refuse before spawning anything.
    model.validate()?;
    let config = TempConfig::write(model)?;
    let mut cmd = Command::new(exe);
    cmd.arg(crate::services::cli::HEADLESS_FLAG)
        .arg("engine-probe")
        .arg("sherpa");
    if let Some(dir) = SherpaRuntime::loaded()
        .map(|rt| PathBuf::from(&rt.info().dir))
        .or_else(runtime_dir)
    {
        cmd.arg("--runtime-dir").arg(dir);
    }
    let mut config_arg = std::ffi::OsString::from("config=");
    config_arg.push(config.0.as_os_str());
    cmd.arg("--").arg("load_only=1").arg(config_arg);
    if let Some(app_data) = crate::services::diagnostics::app_data_dir() {
        cmd.env(crate::services::diagnostics::APP_DATA_ENV, app_data);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot start the model check: {e}"))?;
    // Drain the pipes on threads so a chatty child never blocks on a full pipe.
    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();
    let out_t = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = out_pipe.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let err_t = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = err_pipe.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("the model check did not finish within {} s", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(format!("the model check failed: {e}")),
        }
    };
    let stdout = out_t.join().unwrap_or_default();
    let stderr = err_t.join().unwrap_or_default();
    if !status.success() {
        let last = stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("no output");
        return Err(match status.code() {
            Some(1) => format!("the model does not load: {last}"),
            Some(code) => format!("the model crashed sherpa-onnx (exit code {code:#x}): {last}"),
            None => format!("the model check was terminated: {last}"),
        });
    }
    serde_json::from_str(stdout.trim()).map_err(|e| format!("unreadable model check report: {e}"))
}
