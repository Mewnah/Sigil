//! `Sigil.exe --headless speech-bench <spec.json> [--out <dir>]`: the P7 speech benchmark runner.
//!
//! Measures the app's own STT, MT and TTS backends (never re-implementing inference) on the P7
//! corpus, text sets and prompts, and writes what a scorer, a listener or a friend's export needs:
//!
//! ```text
//! <out>/report.json                 sigil-speech-bench-report/1   the measurement (no scores)
//! <out>/report.md                   one table per run
//! <out>/hyps/<run>.<engine>.jsonl   one report item per line
//! <out>/audio/<engine>/<id>.wav     TTS renderings (16-bit PCM at the engine's rate)
//! <out>/export.json                 sigil-speech-bench-export/1   what a friend sends back
//! ```
//!
//! The formats are frozen by P7-S0 in `scripts/speech-bench/` (schemas, examples, README); the Python
//! scorers there turn a report into WER / CER, chrF with intervals, MetricX and listening pages, so one
//! normalizer implementation exists. Default output folder: `bench-results/speech/<name>-<timestamp>/`.
//!
//! # Modes ([`spec::Mode`])
//! | mode | tasks | what the runner does |
//! |---|---|---|
//! | `accuracy` | stt | hypotheses of every selected corpus item, per engine |
//! | `cpu_speed` | stt, mt, tts | compute, RTF, latency, process CPU; optionally under the load simulator |
//! | `gpu_under_load` | stt, mt, tts | the same with the GPU / CPU load profile, plus VRAM |
//! | `mt_quality` | mt | translations and latency over an MT set (a chrF2 quick look is included) |
//! | `tts_listening` | tts | renderings of every prompt with TTFA / RTF and the optional pre-screens |
//!
//! # Rules
//! - **Engines**: [`stt`] (`whisper`, `http`, `sherpa-online`, `sherpa-offline`, `transcribe`,
//!   `pipeline`, `crispasr`, `hypotheses`), [`mt`] (`nllb`, `bergamot`, `llama-server`, `openai`,
//!   `hypotheses`), [`tts`] (`local-tts`, `tts-helper`, `openai-speech`, `rendered`). An engine whose
//!   model or runtime pack is not installed fails that engine's rows with its message and the run goes
//!   on; nothing is downloaded.
//! - **Timing runs**: `warmup` untimed items first, then `repeat` timed passes (per-item timings are
//!   the median of the passes, text and audio from the first). The load simulator is the bench's
//!   ([`load`]); its CPU is subtracted from the process CPU of the items. `gpu_lock.require` (default
//!   on for the speed modes) refuses a run unless the GPU timing lock file names the tag: the campaign
//!   takes the lock with `gpu-lock.ps1 -Run P7-A -Command ...`, the runner only checks it.
//! - **A fallback is never hidden**: the report records the execution provider requested and the one
//!   that ran, with the engine's reason.
//! - **Privacy**: `export.json` is derived from the report by [`export::make_export`]; the runner
//!   verifies it with [`export::privacy_errors`] before it writes it, and writes nothing when a leak is
//!   found.
//!
//! The contract (`headless_speech_bench` and the file formats) is frozen; fields may be added, never
//! renamed.

pub mod corpus;
pub mod export;
pub mod load;
pub mod mt;
pub mod report;
pub mod spec;
pub mod stt;
pub mod stt_pipeline;
#[cfg(test)]
mod tests;
pub mod tts;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use corpus::{CorpusItem, Manifest, MtLine, TtsPrompt};

/// MT lines grouped by direction, in first-seen order.
type MtGroups = Vec<(String, Vec<MtLine>)>;
use report::{
    evaluate_thresholds, markdown, median, mt_aggregate, round_to, stt_aggregate, tts_aggregate, CorpusStamp, EngineInfo, EngineResult,
    EngineTotals, EnvironmentBlock, Group, MtItem, Report, ReportItem, ResultStatus, RunReport, RunStatus, RuntimePack, SttItem, TtsItem,
    REPORT_SCHEMA,
};
use spec::{EngineDef, ExecutionProvider, Feed, FeedKind, LanguageMode, Mode, Paths, RunDef, SpeechSpec, Task};
use stt::{DriverInfo, SttDriver, SttRequest};
use crate::services::bench::load_sim::{LoadHandle, LoadReport};
use crate::services::bench::{gpu_state, report::BenchReport};
use crate::services::mt_local::bench::{chrf_score, chrf_stats_multi, ChrfStats};

/// Items per pipeline session (a session carries consecutive items of one language).
const GROUP_SIZE: usize = 40;

fn unix_ms_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// `YYYYMMDD-HHMMSS` (UTC) of a Unix time: the default output folder's suffix.
pub fn utc_stamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

// ---------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------

/// `Windows 11 Pro 10.0.26200`-style text is not available without the registry; the real version
/// (`RtlGetVersion`, immune to manifest lies) is `Windows <major>.<minor>.<build>`.
fn os_description() -> String {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct OsVersionInfoW {
            size: u32,
            major: u32,
            minor: u32,
            build: u32,
            platform: u32,
            csd: [u16; 128],
        }
        #[link(name = "ntdll")]
        extern "system" {
            fn RtlGetVersion(info: *mut OsVersionInfoW) -> i32;
        }
        let mut info = OsVersionInfoW {
            size: std::mem::size_of::<OsVersionInfoW>() as u32,
            major: 0,
            minor: 0,
            build: 0,
            platform: 0,
            csd: [0; 128],
        };
        // SAFETY: `RtlGetVersion(PRTL_OSVERSIONINFOW)` with a correctly sized, writable struct.
        if unsafe { RtlGetVersion(&mut info) } == 0 {
            return format!("Windows {}.{}.{}", info.major, info.minor, info.build);
        }
    }
    std::env::consts::OS.to_string()
}

/// The environment block: the bench's `Environment` plus the speech fields (host and user names
/// never appear).
pub fn environment_block() -> EnvironmentBlock {
    let base = super::environment();
    let sys = crate::services::sysinfo::cached();
    let plan = gpu_state::power_plan();
    EnvironmentBlock {
        sigil_version: base.sigil_version,
        build: base.build,
        os: os_description(),
        arch: base.arch,
        cpu: base.cpu,
        logical_cores: base.logical_cores.max(1) as u64,
        physical_cores: Some(u64::from(sys.cpu.physical_cores)).filter(|c| *c > 0),
        ram_mb: Some(sys.total_ram_mb).filter(|r| *r > 0),
        onnx_runtime: base.onnx_runtime,
        gpus: base.gpus,
        power_plan: plan.name,
        power_mode: plan.mode,
        voice_changer_running: base.voice_changer_running,
        runtime_packs: Vec::new(),
        sigil_rig: std::env::var("SIGIL_BENCH_RIG").ok().map(|r| r.trim().to_string()).filter(|r| !r.is_empty()),
    }
}

/// Version recorded in the `installed.json` of a catalog entry (model or runtime pack). The record sits
/// at `<app data>/models/<family>/<id below the family>/installed.json`.
pub fn model_version(id: &str) -> Option<String> {
    static CATALOG: std::sync::OnceLock<Option<crate::services::model_store::Catalog>> = std::sync::OnceLock::new();
    let catalog = CATALOG.get_or_init(|| crate::services::model_store::Catalog::load().ok()).as_ref()?;
    let entry = catalog.entries.iter().find(|e| e.id == id)?;
    let rest = id.strip_prefix(&format!("{}/", entry.family)).unwrap_or(id);
    let mut path = crate::services::model_store::models_root()?.join(&entry.family);
    for seg in rest.split('/') {
        path.push(seg);
    }
    path.push("installed.json");
    let record: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    (record.get("id").and_then(Value::as_str) == Some(id)).then(|| record.get("version").and_then(Value::as_str).map(str::to_string))?
}

/// The runtime packs an engine of this backend and variant ran on (catalog ids).
pub fn runtime_pack_ids(backend: &str, variant: Option<&str>, helper: Option<&str>) -> Vec<String> {
    let v = variant.unwrap_or("");
    match backend {
        "sherpa-online" | "sherpa-offline" => vec!["runtime/sherpa-onnx".into()],
        "transcribe" if !v.is_empty() => vec![format!("runtime/sigil-transcribe-{v}")],
        "llama-server" if !v.is_empty() => vec![format!("runtime/llama-server-{v}")],
        "bergamot" => vec!["runtime/bergamot".into()],
        "crispasr" if !v.is_empty() => vec![format!("runtime/crispasr-{v}")],
        "tts-helper" => match (helper, v) {
            (Some("qwen3"), v) if !v.is_empty() => vec![format!("runtime/qwentts-server-{v}")],
            (Some("omnivoice"), v) if !v.is_empty() => vec![format!("runtime/crispasr-{v}")],
            (Some("breeze"), _) => vec!["runtime/breeze-server-vulkan".into()],
            (Some("magpie"), v) if !v.is_empty() => vec![format!("runtime/nemo-speech-{v}")],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------------------------

/// Creates the engines of a run. The production factory loads the app's backends; tests script it.
pub trait Drivers {
    fn stt(&self, def: &EngineDef, feed: &Feed) -> Result<Box<dyn SttDriver>, String>;
    fn mt(&self, def: &EngineDef) -> Result<Box<dyn mt::MtDriver>, String>;
    fn tts(&self, def: &EngineDef, first_language: &str) -> Result<Box<dyn tts::TtsDriver>, String>;

    /// Installed version of a catalog model (`installed.json`).
    fn model_version(&self, id: &str) -> Option<String> {
        model_version(id)
    }

    /// An installed runtime pack, for the environment block.
    fn runtime_pack(&self, id: &str) -> Option<RuntimePack> {
        crate::services::model_store::installed_path(id).map(|_| RuntimePack {
            id: id.to_string(),
            version: model_version(id),
            variant: None,
        })
    }

    /// Whether the runner may query GPU APIs (DXGI for VRAM, NVML for the machine's state).
    fn meter_gpu(&self) -> bool {
        true
    }
}

/// The app's own engines.
pub struct Production {
    paths: Paths,
}

impl Drivers for Production {
    fn stt(&self, def: &EngineDef, feed: &Feed) -> Result<Box<dyn SttDriver>, String> {
        match def.backend.as_str() {
            "pipeline" => stt_pipeline::PipelineDriver::production(def, &self.paths, false).map(|d| Box::new(d) as Box<dyn SttDriver>),
            "hypotheses" => stt::HypothesesDriver::open(def, &self.paths).map(|d| Box::new(d) as Box<dyn SttDriver>),
            "crispasr" => stt::create_crispasr_driver(def, &self.paths),
            _ if feed.kind == FeedKind::Pipeline => stt_pipeline::PipelineDriver::production(def, &self.paths, true).map(|d| Box::new(d) as Box<dyn SttDriver>),
            _ => stt::create_engine_driver(def, &self.paths),
        }
    }

    fn mt(&self, def: &EngineDef) -> Result<Box<dyn mt::MtDriver>, String> {
        mt::create_driver(def, &self.paths)
    }

    fn tts(&self, def: &EngineDef, first_language: &str) -> Result<Box<dyn tts::TtsDriver>, String> {
        tts::create_driver(def, &self.paths, first_language)
    }
}

// ---------------------------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------------------------

/// Progress lines to stderr, at most one every two seconds per engine.
struct Ticker {
    last: Instant,
}

impl Ticker {
    fn new() -> Ticker {
        Ticker { last: Instant::now() }
    }

    fn tick(&mut self, what: impl FnOnce() -> String) {
        if self.last.elapsed() >= Duration::from_secs(2) {
            eprintln!("[speech-bench] {}", what());
            self.last = Instant::now();
        }
    }
}

/// One selected STT item with its audio loaded lazily.
struct SttInput<'a> {
    item: &'a CorpusItem,
}

/// Everything a run needs from the spec and the disk.
pub struct Runner<'a> {
    pub spec: &'a SpeechSpec,
    pub raw_spec: &'a Value,
    pub paths: Paths,
    pub out_dir: PathBuf,
    pub drivers: &'a dyn Drivers,
    spec_dir: PathBuf,
    manifest: Option<Arc<Manifest>>,
    tally: Option<crate::services::bench::load_sim::LoadCpu>,
    selected_ids: HashSet<String>,
    engine_infos: Vec<EngineInfo>,
    failures: Vec<String>,
    mt_sets: BTreeMap<PathBuf, Vec<MtLine>>,
}

impl<'a> Runner<'a> {
    pub fn new(spec: &'a SpeechSpec, raw_spec: &'a Value, spec_file: &Path, out_dir: PathBuf, drivers: &'a dyn Drivers) -> Runner<'a> {
        Runner {
            spec,
            raw_spec,
            paths: Paths::new(spec_file),
            out_dir,
            drivers,
            spec_dir: spec_file.parent().map(Path::to_path_buf).unwrap_or_default(),
            manifest: None,
            tally: None,
            selected_ids: HashSet::new(),
            engine_infos: Vec::new(),
            failures: Vec::new(),
            mt_sets: BTreeMap::new(),
        }
    }

    fn manifest(&mut self) -> Result<Arc<Manifest>, String> {
        if self.manifest.is_none() {
            let def = self.spec.corpus.as_ref().ok_or("the spec has no corpus")?;
            let path = self.paths.resolve(&def.manifest)?;
            self.manifest = Some(Arc::new(Manifest::load(&path)?));
        }
        self.manifest.clone().ok_or_else(|| "the corpus did not load".to_string())
    }

    fn record_engine(&mut self, def: &EngineDef, info: EngineInfo) {
        match self.engine_infos.iter_mut().find(|e| e.id == def.id) {
            // Keep the first record; a later success replaces an earlier failure.
            Some(existing) if existing.error.is_some() && info.error.is_none() => *existing = info,
            Some(_) => {}
            None => self.engine_infos.push(info),
        }
    }

    fn engine_row(&self, def: &EngineDef, driver: Option<&DriverInfo>, load_ms: Option<f64>, vram_load_mb: Option<f64>, error: Option<String>) -> EngineInfo {
        let model = if def.model.is_empty() || spec::is_catalog_id(&def.model) {
            def.model.clone()
        } else {
            // A bring-your-own model is reported as given (the export turns it into `byo`).
            self.paths.resolve(&def.model).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| def.model.clone())
        };
        let used = driver.map(|d| d.device.clone());
        EngineInfo {
            id: def.id.clone(),
            task: def.task,
            backend: def.backend.clone(),
            model_version: spec::is_catalog_id(&def.model).then(|| self.drivers.model_version(&def.model)).flatten(),
            model,
            execution_provider_requested: def.execution_provider.as_str().to_string(),
            execution_provider_used: used,
            ep_reason: driver.and_then(|d| d.ep_reason.clone()),
            variant_used: driver.and_then(|d| d.variant.clone()),
            threads: driver.and_then(|d| d.threads).or(def.threads.map(i64::from)),
            load_ms: load_ms.map(|v| round_to(v, 1)),
            vram_load_mb,
            error,
            extra: driver.and_then(|d| {
                let mut extra = serde_json::Map::new();
                if let Some(r) = &d.runtime {
                    extra.insert("runtime".into(), json!(r));
                }
                if let Some(v) = d.vram_mb.filter(|v| *v > 0) {
                    extra.insert("engine_vram_mb".into(), json!(v));
                }
                if d.helper_process {
                    extra.insert("helper_process".into(), json!(true));
                }
                if let Some(Value::Object(more)) = &d.extra {
                    extra.extend(more.clone());
                }
                (!extra.is_empty()).then_some(Value::Object(extra))
            }),
        }
    }

    // ---- STT ------------------------------------------------------------------------------

    fn select_stt<'m>(&self, manifest: &'m Manifest, run: &RunDef) -> Result<Vec<&'m CorpusItem>, String> {
        let default = self.spec.corpus.as_ref().map(|c| c.select.clone()).unwrap_or_default();
        let sel = run.select.clone().unwrap_or(default);
        manifest.select(&sel)
    }

    fn run_stt(&mut self, run: &RunDef, defs: &[&EngineDef], rr: &mut RunReport, vram_always: bool) -> Result<(), String> {
        let manifest = self.manifest()?;
        let items = self.select_stt(&manifest, run)?;
        if items.is_empty() {
            return Err("the run's selection matches no corpus items".into());
        }
        for it in &items {
            self.selected_ids.insert(it.id.clone());
        }
        let mut order: Vec<usize> = (0..items.len()).collect();
        if run.feed.pace == spec::Pace::Realtime {
            let mut rng = crate::services::bench::signals::SplitMix64::new(self.spec.seed);
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
        }
        let inputs: Vec<SttInput<'_>> = order.into_iter().map(|i| SttInput { item: items[i] }).collect();
        for def in defs.iter().filter(|d| d.task == Task::Stt) {
            let result = self.stt_engine(run, def, &manifest, &inputs, rr, vram_always);
            rr.results.push(result);
        }
        Ok(())
    }

    fn stt_engine(&mut self, run: &RunDef, def: &EngineDef, manifest: &Manifest, inputs: &[SttInput<'_>], rr: &mut RunReport, vram_always: bool) -> EngineResult {
        let mine: Vec<&SttInput<'_>> = inputs
            .iter()
            .filter(|i| def.languages.is_empty() || def.languages.iter().any(|l| l.eq_ignore_ascii_case(&i.item.language)))
            .collect();
        if mine.is_empty() {
            let mut r = EngineResult::failed_to_load(&def.id, "no selected item is in the engine's languages");
            r.status = ResultStatus::Skipped;
            r.error = Some("no selected item is in the engine's languages".into());
            return r;
        }
        eprintln!("[speech-bench] {} / {}: loading", run.name, def.id);
        let mut vram = load::Vram::new(self.drivers.meter_gpu() && (vram_always || def.execution_provider != ExecutionProvider::Cpu), is_helper_backend(def));
        vram.mark_baseline();
        let load_started = Instant::now();
        let mut driver = match self.drivers.stt(def, &run.feed) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[speech-bench] {} / {}: {e}", run.name, def.id);
                let row = self.engine_row(def, None, None, None, Some(e.clone()));
                self.record_engine(def, row);
                return EngineResult::failed_to_load(&def.id, e);
            }
        };
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        vram.sample();
        let vram_load = vram.delta_now();
        let feed = run.feed.clone();
        let timeout = Duration::from_secs_f64(run.timeout_s);
        let given = def.language_mode == LanguageMode::Given;
        let language_of = |i: &CorpusItem| -> Option<String> {
            given.then(|| crate::services::speech::lang::normalize(&i.language)).flatten()
        };
        let speed = run.mode.is_speed();
        let passes = if speed { run.repeat.max(1) } else { 1 };

        // Warm-up (not in the report).
        if !driver.grouped() {
            for input in mine.iter().take(run.warmup as usize) {
                if let Ok(audio) = manifest.load_audio(input.item) {
                    let _ = driver.transcribe(&SttRequest {
                        item: &input.item.id,
                        audio: &audio,
                        language: language_of(input.item),
                        feed: &feed,
                        timeout,
                    });
                }
            }
        }

        // Timed passes. `samples[i]` collects the per-pass measurements of item i.
        struct Pass {
            compute_ms: Option<f64>,
            final_latency_ms: Option<f64>,
            first_partial_ms: Option<f64>,
            cpu_ms: Option<f64>,
        }
        struct Slot {
            out: Option<stt::SttOutput>,
            error: Option<String>,
            audio_s: f64,
            passes: Vec<Pass>,
        }
        let mut slots: Vec<Slot> = mine
            .iter()
            .map(|i| Slot {
                out: None,
                error: None,
                audio_s: i.item.seconds,
                passes: Vec::new(),
            })
            .collect();
        let mut ticker = Ticker::new();
        let engine_window = load::Window::start(self.load_tally().as_ref());
        for pass in 0..passes {
            if driver.grouped() {
                let mut start = 0usize;
                while start < mine.len() {
                    // A group: consecutive items of the same language, at most GROUP_SIZE.
                    let lang = language_of(mine[start].item);
                    let mut end = start + 1;
                    while end < mine.len() && end - start < GROUP_SIZE && language_of(mine[end].item) == lang {
                        end += 1;
                    }
                    let mut audios: Vec<Result<Vec<f32>, String>> = Vec::new();
                    for input in &mine[start..end] {
                        audios.push(manifest.load_audio(input.item));
                    }
                    let ok_idx: Vec<usize> = (0..audios.len()).filter(|&k| audios[k].is_ok()).collect();
                    let reqs: Vec<SttRequest<'_>> = ok_idx
                        .iter()
                        .map(|&k| SttRequest {
                            item: &mine[start + k].item.id,
                            audio: audios[k].as_ref().map(Vec::as_slice).unwrap_or(&[]),
                            language: lang.clone(),
                            feed: &feed,
                            timeout,
                        })
                        .collect();
                    let results = driver.transcribe_group(&reqs);
                    for (k, a) in audios.iter().enumerate() {
                        if let Err(e) = a {
                            if pass == 0 {
                                slots[start + k].error = Some(e.clone());
                            }
                        }
                    }
                    for (&k, res) in ok_idx.iter().zip(results) {
                        let slot = &mut slots[start + k];
                        match res {
                            Ok(out) => {
                                slot.passes.push(Pass {
                                    compute_ms: out.compute_ms,
                                    final_latency_ms: out.final_latency_ms,
                                    first_partial_ms: out.first_partial_ms,
                                    cpu_ms: None,
                                });
                                if pass == 0 {
                                    slot.out = Some(out);
                                }
                            }
                            Err(e) if pass == 0 => slot.error = Some(e),
                            Err(_) => {}
                        }
                    }
                    vram.sample();
                    ticker.tick(|| format!("{} / {}: {}/{} items", run.name, def.id, end, mine.len()));
                    start = end;
                }
                continue;
            }
            for (idx, input) in mine.iter().enumerate() {
                if pass > 0 && slots[idx].error.is_some() {
                    continue;
                }
                let audio = match manifest.load_audio(input.item) {
                    Ok(a) => a,
                    Err(e) => {
                        if pass == 0 {
                            slots[idx].error = Some(e);
                        }
                        continue;
                    }
                };
                let req = SttRequest {
                    item: &input.item.id,
                    audio: &audio,
                    language: language_of(input.item),
                    feed: &feed,
                    timeout,
                };
                let window = load::Window::start(self.load_tally().as_ref());
                let result = driver.transcribe(&req);
                let wall_ms = window.wall().as_secs_f64() * 1000.0;
                let cpu_ms = window.cpu_ms();
                vram.sample();
                match result {
                    Ok(out) => {
                        slots[idx].passes.push(Pass {
                            compute_ms: out.compute_ms.or(Some(wall_ms)),
                            final_latency_ms: out.final_latency_ms,
                            first_partial_ms: out.first_partial_ms,
                            cpu_ms,
                        });
                        if pass == 0 {
                            slots[idx].out = Some(out);
                        }
                    }
                    Err(e) if pass == 0 => slots[idx].error = Some(e),
                    Err(_) => {}
                }
                ticker.tick(|| format!("{} / {}: {}/{} items (pass {})", run.name, def.id, idx + 1, mine.len(), pass + 1));
            }
        }
        let wall_ms = engine_window.wall().as_secs_f64() * 1000.0;
        let cores = engine_window.cores();
        let (vram_peak, vram_delta) = vram.peaks();
        let info = driver.info().clone();
        for note in driver.notes() {
            if !rr.warnings.contains(&format!("{}: {note}", def.id)) {
                rr.warnings.push(format!("{}: {note}", def.id));
            }
        }
        drop(driver);

        // Items.
        let med = |passes: &[Pass], f: fn(&Pass) -> Option<f64>| median(&passes.iter().filter_map(f).collect::<Vec<_>>());
        let mut items_out: Vec<SttItem> = Vec::with_capacity(mine.len());
        for (input, slot) in mine.iter().zip(&slots) {
            let it = input.item;
            let mut item = SttItem {
                item: it.id.clone(),
                language: it.language.clone(),
                condition: it.condition.clone(),
                audio_s: round_to(it.seconds, 3),
                ..SttItem::default()
            };
            match (&slot.out, &slot.error) {
                (Some(out), None) => {
                    item.text = Some(out.text.clone());
                    item.language_detected = out.language.clone();
                    item.partials = out.partials;
                    item.segments = (!out.segments.is_empty()).then(|| out.segments.clone());
                    item.tags = (!out.tags.is_empty()).then(|| out.tags.clone());
                    let compute = med(&slot.passes, |p| p.compute_ms);
                    item.compute_ms = compute.map(|v| round_to(v, 3));
                    item.rtf = compute.filter(|_| slot.audio_s > 0.0).map(|c| round_to(c / 1000.0 / slot.audio_s, 5));
                    item.final_latency_ms = med(&slot.passes, |p| p.final_latency_ms).map(|v| round_to(v, 3));
                    item.first_partial_ms = med(&slot.passes, |p| p.first_partial_ms).map(|v| round_to(v, 3));
                    item.cpu_ms = med(&slot.passes, |p| p.cpu_ms).map(|v| round_to(v, 3));
                }
                (_, Some(e)) => item.error = Some(e.clone()),
                (None, None) => item.error = Some("no result".into()),
            }
            items_out.push(item);
        }
        let empty = items_out
            .iter()
            .filter(|i| i.error.is_none() && i.audio_s >= 1.0 && i.text.as_deref().is_some_and(|t| t.trim().is_empty()))
            .count();
        if empty > 0 {
            rr.warnings.push(format!(
                "{}: {empty} of {} item(s) returned empty text (the clips hold speech, so this is a model or runtime problem; the scorer counts them as all deletions)",
                def.id,
                items_out.len()
            ));
        }
        let totals = EngineTotals {
            wall_ms: Some(wall_ms),
            cpu_cores_mean: cores.map(|c| round_to(c, 3)),
            vram_peak_mb: vram_peak,
            vram_delta_mb: vram_delta,
            ram_peak_mb: vram.ram_peak_mb,
        };
        let refs: Vec<&SttItem> = items_out.iter().collect();
        let aggregate = stt_aggregate(&refs, Some(&totals));
        let mut group_keys: Vec<(String, String)> = items_out.iter().map(|i| (i.language.clone(), i.condition.clone())).collect();
        group_keys.sort();
        group_keys.dedup();
        let groups = group_keys
            .into_iter()
            .map(|(language, condition)| {
                let subset: Vec<&SttItem> = items_out.iter().filter(|i| i.language == language && i.condition == condition).collect();
                Group {
                    aggregate: stt_aggregate(&subset, None),
                    language: Some(language),
                    condition: Some(condition),
                    direction: None,
                }
            })
            .collect();
        let row = self.engine_row(def, Some(&info), Some(load_ms), vram_load, None);
        let used = row.execution_provider_used.clone();
        self.record_engine(def, row);
        self.finish_result(run, def, used, aggregate, groups, items_out.into_iter().map(ReportItem::Stt).collect(), rr)
    }

    // ---- MT -------------------------------------------------------------------------------

    fn mt_lines(&mut self, run: &RunDef) -> Result<(Vec<MtLine>, MtGroups), String> {
        let opts = run.mt.as_ref().ok_or("the run has no mt block")?;
        let path = self.paths.resolve(&opts.set)?;
        if !self.mt_sets.contains_key(&path) {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read the MT set {}: {e}", mt::set_name(&path)))?;
            let lines = corpus::parse_mt_lines(&text)?;
            self.mt_sets.insert(path.clone(), lines);
        }
        let all = self.mt_sets.get(&path).cloned().unwrap_or_default();
        let groups = corpus::select_mt(&all, &opts.directions, opts.limit);
        if groups.is_empty() {
            return Err("the run's directions match no line of the MT set".into());
        }
        Ok((all, groups))
    }

    fn run_mt(&mut self, run: &RunDef, defs: &[&EngineDef], rr: &mut RunReport, vram_always: bool) -> Result<(), String> {
        let (all, groups) = self.mt_lines(run)?;
        for def in defs.iter().filter(|d| d.task == Task::Mt) {
            let result = self.mt_engine(run, def, &all, &groups, rr, vram_always);
            rr.results.push(result);
        }
        Ok(())
    }

    fn mt_engine(&mut self, run: &RunDef, def: &EngineDef, all: &[MtLine], groups: &[(String, Vec<MtLine>)], rr: &mut RunReport, vram_always: bool) -> EngineResult {
        let mine: Vec<(String, Vec<MtLine>)> = groups
            .iter()
            .filter(|(d, _)| def.directions.is_empty() || def.directions.contains(d))
            .cloned()
            .collect();
        if mine.is_empty() {
            let mut r = EngineResult::failed_to_load(&def.id, "no selected direction is in the engine's directions");
            r.status = ResultStatus::Skipped;
            return r;
        }
        eprintln!("[speech-bench] {} / {}: loading", run.name, def.id);
        let mut vram = load::Vram::new(self.drivers.meter_gpu() && (vram_always || def.execution_provider != ExecutionProvider::Cpu), is_helper_backend(def));
        vram.mark_baseline();
        let load_started = Instant::now();
        let mut driver = match self.drivers.mt(def) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[speech-bench] {} / {}: {e}", run.name, def.id);
                let row = self.engine_row(def, None, None, None, Some(e.clone()));
                self.record_engine(def, row);
                return EngineResult::failed_to_load(&def.id, e);
            }
        };
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        vram.sample();
        let vram_load = vram.delta_now();
        let timeout = Duration::from_secs_f64(run.timeout_s);
        let context = run.mt.as_ref().map(|m| m.context).unwrap_or_default();
        let use_context = mt::takes_context(def, context);
        let passes = if run.mode.is_speed() && driver.timed() { run.repeat.max(1) } else { 1 };
        // Warm-up: the first lines of each direction, untimed.
        if driver.timed() {
            for (_, lines) in &mine {
                for line in lines.iter().take(run.warmup as usize) {
                    let ctx = use_context.then(|| corpus::previous_in_doc(all, line)).flatten();
                    let _ = driver.translate(line, ctx, timeout);
                }
            }
        }
        struct Slot {
            text: Option<String>,
            error: Option<String>,
            out_tokens: Option<u64>,
            times: Vec<f64>,
        }
        let mut slots: Vec<Vec<Slot>> = mine.iter().map(|(_, l)| l.iter().map(|_| Slot { text: None, error: None, out_tokens: None, times: Vec::new() }).collect()).collect();
        let mut ticker = Ticker::new();
        let window = load::Window::start(self.load_tally().as_ref());
        let total: usize = mine.iter().map(|(_, l)| l.len()).sum();
        let mut done = 0usize;
        for pass in 0..passes {
            for (g, (_, lines)) in mine.iter().enumerate() {
                for (n, line) in lines.iter().enumerate() {
                    if pass > 0 && slots[g][n].error.is_some() {
                        continue;
                    }
                    let ctx = use_context.then(|| corpus::previous_in_doc(all, line)).flatten();
                    let t0 = Instant::now();
                    let res = driver.translate(line, ctx, timeout);
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    vram.sample();
                    let slot = &mut slots[g][n];
                    match res {
                        Ok(out) => {
                            slot.times.push(ms);
                            if pass == 0 {
                                slot.text = Some(out.text);
                                slot.out_tokens = out.out_tokens;
                            }
                        }
                        Err(e) if pass == 0 => slot.error = Some(e),
                        Err(_) => {}
                    }
                    done += 1;
                    ticker.tick(|| format!("{} / {}: {}/{} lines", run.name, def.id, done, total * passes as usize));
                }
            }
        }
        let wall_ms = window.wall().as_secs_f64() * 1000.0;
        let cores = window.cores();
        let (vram_peak, vram_delta) = vram.peaks();
        let timed = driver.timed();
        let info = driver.info().clone();
        for note in driver.notes() {
            rr.warnings.push(format!("{}: {note}", def.id));
        }
        drop(driver);

        let mut items: Vec<MtItem> = Vec::new();
        let mut per_dir_stats: Vec<(String, ChrfStats, usize)> = Vec::new();
        let mut overall = ChrfStats::default();
        let mut scored = 0usize;
        for (g, (dir, lines)) in mine.iter().enumerate() {
            let mut stats = ChrfStats::default();
            let mut count = 0usize;
            for (n, line) in lines.iter().enumerate() {
                let slot = &slots[g][n];
                let hyp = slot.text.clone().unwrap_or_default();
                if !line.refs.is_empty() {
                    let s = chrf_stats_multi(&hyp, &line.refs);
                    stats.add(&s);
                    overall.add(&s);
                    count += 1;
                    scored += 1;
                }
                items.push(MtItem {
                    item: line.id.clone(),
                    direction: dir.clone(),
                    text: slot.error.is_none().then_some(hyp),
                    latency_ms: (timed && slot.error.is_none()).then(|| median(&slot.times)).flatten().map(|v| round_to(v, 3)),
                    src_chars: Some(line.src.chars().count() as u64),
                    out_tokens: slot.out_tokens,
                    error: slot.error.clone(),
                    extra: None,
                });
            }
            per_dir_stats.push((dir.clone(), stats, count));
        }
        let chrf_of = |s: &ChrfStats, n: usize| (n > 0).then(|| chrf_score(s));
        let totals = EngineTotals {
            wall_ms: Some(wall_ms),
            cpu_cores_mean: cores.map(|c| round_to(c, 3)),
            vram_peak_mb: vram_peak,
            vram_delta_mb: vram_delta,
            ram_peak_mb: vram.ram_peak_mb,
        };
        let refs: Vec<&MtItem> = items.iter().collect();
        let aggregate = mt_aggregate(&refs, chrf_of(&overall, scored), Some(&totals));
        let groups = per_dir_stats
            .iter()
            .map(|(dir, stats, n)| {
                let subset: Vec<&MtItem> = items.iter().filter(|i| &i.direction == dir).collect();
                Group {
                    aggregate: mt_aggregate(&subset, chrf_of(stats, *n), None),
                    language: None,
                    condition: None,
                    direction: Some(dir.clone()),
                }
            })
            .collect();
        let row = self.engine_row(def, Some(&info), Some(load_ms), vram_load, None);
        let used = row.execution_provider_used.clone();
        self.record_engine(def, row);
        self.finish_result(run, def, used, aggregate, groups, items.into_iter().map(ReportItem::Mt).collect(), rr)
    }

    // ---- TTS ------------------------------------------------------------------------------

    fn tts_prompts(&self, run: &RunDef) -> Result<Vec<TtsPrompt>, String> {
        let opts = run.tts.as_ref().ok_or("the run has no tts block")?;
        let path = self.paths.resolve(&opts.prompts)?;
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read the TTS prompts {}: {e}", mt::set_name(&path)))?;
        corpus::parse_prompts(&text)
    }

    fn run_tts(&mut self, run: &RunDef, defs: &[&EngineDef], rr: &mut RunReport, vram_always: bool) -> Result<(), String> {
        let prompts = self.tts_prompts(run)?;
        let opts = run.tts.clone().ok_or("the run has no tts block")?;
        // The pre-screens are shared by the run's engines.
        let mut screens = Screens::new(&opts, self, run);
        for def in defs.iter().filter(|d| d.task == Task::Tts) {
            let mine = corpus::select_prompts(&prompts, &def.languages, opts.limit_per_language);
            let result = self.tts_engine(run, def, &mine, &opts, &mut screens, rr, vram_always);
            rr.results.push(result);
        }
        screens.finish(rr);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tts_engine(&mut self, run: &RunDef, def: &EngineDef, prompts: &[TtsPrompt], opts: &spec::TtsOptions, screens: &mut Screens, rr: &mut RunReport, vram_always: bool) -> EngineResult {
        if prompts.is_empty() {
            let mut r = EngineResult::failed_to_load(&def.id, "no prompt is in the engine's languages");
            r.status = ResultStatus::Skipped;
            return r;
        }
        eprintln!("[speech-bench] {} / {}: loading", run.name, def.id);
        let mut vram = load::Vram::new(self.drivers.meter_gpu() && (vram_always || def.execution_provider != ExecutionProvider::Cpu), is_helper_backend(def));
        vram.mark_baseline();
        let load_started = Instant::now();
        let mut driver = match self.drivers.tts(def, &prompts[0].lang) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[speech-bench] {} / {}: {e}", run.name, def.id);
                let row = self.engine_row(def, None, None, None, Some(e.clone()));
                self.record_engine(def, row);
                return EngineResult::failed_to_load(&def.id, e);
            }
        };
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        vram.sample();
        let vram_load = vram.delta_now();
        let timeout = Duration::from_secs_f64(run.timeout_s);
        let write_audio = run.mode == Mode::TtsListening || opts.write_audio;
        let passes = if run.mode.is_speed() && driver.timed() { run.repeat.max(1) } else { 1 };
        if driver.timed() {
            for p in prompts.iter().take(run.warmup as usize) {
                let _ = driver.render(p, timeout);
            }
        }
        let audio_dir = self.out_dir.join("audio").join(&def.id);
        let mut items: Vec<TtsItem> = Vec::with_capacity(prompts.len());
        let mut ticker = Ticker::new();
        let window = load::Window::start(self.load_tally().as_ref());
        let timed = driver.timed();
        struct Times {
            ttfa: Vec<f64>,
            total: Vec<f64>,
        }
        let mut times: Vec<Times> = prompts.iter().map(|_| Times { ttfa: Vec::new(), total: Vec::new() }).collect();
        for pass in 0..passes {
            for (n, prompt) in prompts.iter().enumerate() {
                if pass > 0 && items[n].error.is_some() {
                    continue;
                }
                let res = driver.render(prompt, timeout);
                vram.sample();
                match res {
                    Ok(out) => {
                        if let Some(t) = out.ttfa_ms {
                            times[n].ttfa.push(t);
                        }
                        if let Some(t) = out.total_ms {
                            times[n].total.push(t);
                        }
                        if pass == 0 {
                            let audio_s = out.samples.len() as f64 / f64::from(out.sample_rate.max(1));
                            let mut item = TtsItem {
                                item: prompt.id.clone(),
                                language: prompt.lang.clone(),
                                voice: out.voice.clone(),
                                audio: None,
                                audio_s: Some(round_to(audio_s, 3)),
                                sample_rate: Some(out.sample_rate),
                                ..TtsItem::default()
                            };
                            if write_audio {
                                let file = audio_dir.join(format!("{}.wav", prompt.id));
                                let written = std::fs::create_dir_all(&audio_dir)
                                    .map_err(|e| format!("cannot create {}: {e}", audio_dir.display()))
                                    .and_then(|_| tts::write_pcm16_wav(&file, &out.samples, out.sample_rate));
                                match written {
                                    Ok(()) => {
                                        let rel = format!("audio/{}/{}.wav", def.id, prompt.id);
                                        rr.files.push(rel.clone());
                                        item.audio = Some(rel);
                                    }
                                    Err(e) => item.error = Some(e),
                                }
                            }
                            if item.error.is_none() {
                                screens.apply(prompt, &out, &mut item);
                            }
                            items.push(item);
                        }
                    }
                    Err(e) if pass == 0 => items.push(TtsItem {
                        item: prompt.id.clone(),
                        language: prompt.lang.clone(),
                        error: Some(e),
                        ..TtsItem::default()
                    }),
                    Err(_) => {}
                }
                ticker.tick(|| format!("{} / {}: {}/{} prompts (pass {})", run.name, def.id, n + 1, prompts.len(), pass + 1));
            }
        }
        for (item, t) in items.iter_mut().zip(&times) {
            if item.error.is_some() || !timed {
                continue;
            }
            item.ttfa_ms = median(&t.ttfa).map(|v| round_to(v, 3));
            item.total_ms = median(&t.total).map(|v| round_to(v, 3));
            item.rtf = item
                .total_ms
                .zip(item.audio_s)
                .filter(|(_, a)| *a > 0.0)
                .map(|(t, a)| round_to(t / 1000.0 / a, 5));
        }
        let wall_ms = window.wall().as_secs_f64() * 1000.0;
        let cores = window.cores();
        let (vram_peak, vram_delta) = vram.peaks();
        let info = driver.info().clone();
        drop(driver);
        let totals = EngineTotals {
            wall_ms: Some(wall_ms),
            cpu_cores_mean: cores.map(|c| round_to(c, 3)),
            vram_peak_mb: vram_peak,
            vram_delta_mb: vram_delta,
            ram_peak_mb: vram.ram_peak_mb,
        };
        let refs: Vec<&TtsItem> = items.iter().collect();
        let aggregate = tts_aggregate(&refs, Some(&totals));
        let mut langs: Vec<String> = items.iter().map(|i| i.language.clone()).collect();
        langs.sort();
        langs.dedup();
        let groups = langs
            .into_iter()
            .map(|l| {
                let subset: Vec<&TtsItem> = items.iter().filter(|i| i.language == l).collect();
                Group {
                    aggregate: tts_aggregate(&subset, None),
                    language: Some(l),
                    condition: None,
                    direction: None,
                }
            })
            .collect();
        let row = self.engine_row(def, Some(&info), Some(load_ms), vram_load, None);
        let used = row.execution_provider_used.clone();
        self.record_engine(def, row);
        self.finish_result(run, def, used, aggregate, groups, items.into_iter().map(ReportItem::Tts).collect(), rr)
    }

    // ---- Shared ---------------------------------------------------------------------------

    /// The running load's CPU tally, if a load runs (set by the run).
    fn load_tally(&self) -> Option<crate::services::bench::load_sim::LoadCpu> {
        self.tally.clone()
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_result(&mut self, run: &RunDef, def: &EngineDef, used: Option<String>, aggregate: report::Aggregate, groups: Vec<Group>, items: Vec<ReportItem>, rr: &mut RunReport) -> EngineResult {
        let thresholds = evaluate_thresholds(&aggregate, &run.thresholds);
        let failed_items = items.iter().filter(|i| i.error().is_some()).count();
        let first_error = items.iter().find_map(|i| i.error()).map(str::to_string);
        let mut status = ResultStatus::Passed;
        if failed_items > 0 {
            status = ResultStatus::Failed;
            self.failures.push(format!(
                "{} / {}: {failed_items} item{} failed ({})",
                run.name,
                def.id,
                if failed_items == 1 { "" } else { "s" },
                first_error.unwrap_or_default()
            ));
        }
        for t in thresholds.iter().filter(|t| !t.passed) {
            status = ResultStatus::Failed;
            self.failures.push(format!(
                "{} / {}: {} {} outside {}{}",
                run.name,
                def.id,
                t.metric,
                t.value.map_or_else(|| "not measured".to_string(), |v| format!("= {v}")),
                t.min.map(|m| format!("min {m} ")).unwrap_or_default(),
                t.max.map(|m| format!("max {m}")).unwrap_or_default(),
            ));
        }
        if self.spec.output.write_hypotheses {
            if let Err(e) = self.write_hypotheses(run, def, &items, rr) {
                rr.warnings.push(e);
            }
        }
        EngineResult {
            engine: def.id.clone(),
            status,
            error: None,
            execution_provider_used: used,
            aggregate,
            groups,
            thresholds,
            items,
        }
    }

    fn write_hypotheses(&self, run: &RunDef, def: &EngineDef, items: &[ReportItem], rr: &mut RunReport) -> Result<(), String> {
        let dir = self.out_dir.join("hyps");
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create the hyps folder: {e}"))?;
        let rel = format!("hyps/{}.{}.jsonl", run.name, def.id);
        let mut text = String::new();
        for item in items {
            text.push_str(&serde_json::to_string(item).map_err(|e| e.to_string())?);
            text.push('\n');
        }
        std::fs::write(self.out_dir.join(&rel), text).map_err(|e| format!("cannot write {rel}: {e}"))?;
        rr.files.push(rel);
        Ok(())
    }

    /// Run one run of the spec.
    pub fn execute_run(&mut self, run: &RunDef) -> RunReport {
        let started = Instant::now();
        let sp = self.spec;
        let defs: Vec<&EngineDef> = sp.run_engines(run);
        let mut rr = RunReport {
            name: run.name.clone(),
            mode: run.mode,
            status: RunStatus::Passed,
            error: None,
            duration_ms: 0.0,
            feed: defs.iter().any(|d| d.task == Task::Stt).then(|| {
                json!({
                    "kind": run.feed.kind.as_str(),
                    "chunk_ms": run.feed.chunk_ms,
                    "pace": match run.feed.pace { spec::Pace::Asap => "asap", spec::Pace::Realtime => "realtime" },
                    "lead_silence_ms": run.feed.lead_silence_ms,
                    "tail_silence_ms": run.feed.tail_silence_ms,
                })
            }),
            load: None,
            co_load: None,
            gpu_lock: None,
            state: None,
            warnings: Vec::new(),
            results: Vec::new(),
            files: Vec::new(),
        };
        let fail = |rr: &mut RunReport, started: Instant, msg: String| {
            rr.status = RunStatus::Error;
            rr.error = Some(msg);
            rr.duration_ms = round_to(started.elapsed().as_secs_f64() * 1000.0, 1);
        };

        // The GPU timing lock.
        let lock_path = run.lock_path().and_then(|p| self.paths.resolve(&p).ok());
        let (block, lock) = load::check_lock(run, lock_path, &self.spec_dir);
        rr.gpu_lock = block;
        if let Err(e) = lock {
            self.failures.push(format!("{}: {e}", run.name));
            fail(&mut rr, started, e.clone());
            return rr;
        }

        // The load and the machine's state.
        let gpu_ep = if defs.iter().any(|d| wants_cuda(d)) { "cuda" } else { "dml" };
        let resolved = match load::resolve_load(run, gpu_ep) {
            Ok(l) => l,
            Err(e) => {
                self.failures.push(format!("{}: {e}", run.name));
                fail(&mut rr, started, e.clone());
                return rr;
            }
        };
        let handle = if resolved.is_idle() {
            None
        } else {
            match LoadHandle::start_kind(&resolved.spec, run.gpu_load_kind) {
                Ok(h) => Some(h),
                Err(e) => {
                    let msg = format!("the load simulator could not start: {e}");
                    self.failures.push(format!("{}: {msg}", run.name));
                    fail(&mut rr, started, msg);
                    return rr;
                }
            }
        };
        self.tally = handle.as_ref().map(LoadHandle::cpu_tally);
        let any_gpu = defs.iter().any(|d| d.execution_provider.wants_gpu()) || resolved.spec.gpu.is_some();
        let probe = (self.drivers.meter_gpu() && (run.mode.is_speed() || any_gpu)).then(|| {
            gpu_state::RunStateProbe::start(gpu_state::StateContext {
                engine_ep: Some(if defs.iter().any(|d| wants_cuda(d)) { "cuda" } else if any_gpu { "dml" } else { "cpu" }.to_string()),
                device: 0,
                period_ms: run.state_period_ms,
                speed_index: false,
            })
        });
        let co_load = run.co_load.as_ref().map(|c| self.start_co_load(c));

        // The engines.
        let vram_always = run.mode == Mode::GpuUnderLoad;
        let work = match run.mode {
            Mode::Accuracy => self.run_stt(run, &defs, &mut rr, vram_always),
            Mode::MtQuality => self.run_mt(run, &defs, &mut rr, vram_always),
            Mode::TtsListening => self.run_tts(run, &defs, &mut rr, vram_always),
            Mode::CpuSpeed | Mode::GpuUnderLoad => {
                let mut r = Ok(());
                if defs.iter().any(|d| d.task == Task::Stt) {
                    r = r.and(self.run_stt(run, &defs, &mut rr, vram_always));
                }
                if defs.iter().any(|d| d.task == Task::Mt) {
                    r = r.and(self.run_mt(run, &defs, &mut rr, vram_always));
                }
                if defs.iter().any(|d| d.task == Task::Tts) {
                    r = r.and(self.run_tts(run, &defs, &mut rr, vram_always));
                }
                r
            }
        };

        // Wind down: the co-load, the state, the load.
        if let Some(co) = co_load {
            rr.co_load = Some(co.finish());
        }
        if let Some(p) = probe {
            let state = p.finish();
            rr.warnings.extend(state.warnings.iter().cloned());
            rr.state = serde_json::to_value(&state).ok();
        }
        let achieved: Option<LoadReport> = handle.map(LoadHandle::stop);
        self.tally = None;
        rr.load = Some(json!({
            "profile": resolved.profile.as_str(),
            "spec": serde_json::to_value(&resolved.spec).unwrap_or(Value::Null),
            "external": run.external_load,
            "achieved": achieved.and_then(|a| serde_json::to_value(a).ok()).unwrap_or_else(|| json!({})),
        }));
        if let Err(e) = work {
            self.failures.push(format!("{}: {e}", run.name));
            rr.status = RunStatus::Error;
            rr.error = Some(e);
        } else if rr.results.iter().any(|r| matches!(r.status, ResultStatus::Failed | ResultStatus::Error)) {
            rr.status = RunStatus::Failed;
            for r in &rr.results {
                if r.status == ResultStatus::Error {
                    self.failures.push(format!("{} / {}: {}", run.name, r.engine, r.error.clone().unwrap_or_default()));
                }
            }
        }
        rr.duration_ms = round_to(started.elapsed().as_secs_f64() * 1000.0, 1);
        rr
    }

    fn start_co_load(&self, co: &spec::CoLoad) -> CoLoadHandle {
        CoLoadHandle::start(self.paths.resolve(&co.voice_changer_spec))
    }
}

fn wants_cuda(def: &EngineDef) -> bool {
    def.execution_provider == ExecutionProvider::Cuda || def.variant.as_deref().is_some_and(|v| v.starts_with("cuda"))
}

/// Engines whose work runs in another process (their VRAM is read device-wide).
fn is_helper_backend(def: &EngineDef) -> bool {
    matches!(def.backend.as_str(), "llama-server" | "tts-helper" | "crispasr" | "bergamot")
}

// ---------------------------------------------------------------------------------------------
// The voice changer beside the run
// ---------------------------------------------------------------------------------------------

/// The paced voice-changer run of `co_load` (the bench's own harness) beside the engines.
struct CoLoadHandle {
    cancel: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<Value>>,
    error: Option<String>,
}

impl CoLoadHandle {
    fn start(path: Result<PathBuf, String>) -> CoLoadHandle {
        let cancel = Arc::new(AtomicBool::new(false));
        let path = match path {
            Ok(p) => p,
            Err(e) => return CoLoadHandle { cancel, thread: None, error: Some(e) },
        };
        let spec = match crate::services::bench::spec::BenchSpec::load(&path) {
            Ok(s) => s,
            Err(e) => return CoLoadHandle { cancel, thread: None, error: Some(e) },
        };
        let flag = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("sigil-speech-bench-vc".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn(move || {
                let progress = |_: &super::Progress| {};
                let report: BenchReport = super::run_spec(&spec, None, &super::BenchContext { cancel: &flag, progress: &progress });
                serde_json::to_value(&report).unwrap_or(Value::Null)
            });
        match thread {
            Ok(t) => CoLoadHandle { cancel, thread: Some(t), error: None },
            Err(e) => CoLoadHandle { cancel, thread: None, error: Some(format!("cannot start the voice changer run: {e}")) },
        }
    }

    /// Stop it (the paced run ends early) and return its report.
    fn finish(mut self) -> Value {
        self.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        match (self.thread.take().map(|t| t.join()), self.error.take()) {
            (Some(Ok(v)), _) => v,
            (Some(Err(_)), _) => json!({ "error": "the voice changer run panicked" }),
            (None, Some(e)) => json!({ "error": e }),
            (None, None) => Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// TTS pre-screens
// ---------------------------------------------------------------------------------------------

/// Intelligibility (back-transcription), DNSMOS and speaker similarity of a rendering.
struct Screens {
    stt: Option<Box<dyn SttDriver>>,
    stt_error: Option<String>,
    dnsmos: Option<crate::services::bench::scorers::Dnsmos>,
    dnsmos_error: Option<String>,
    speaker: Option<crate::services::bench::scorers::SpeakerEmbedder>,
    speaker_error: Option<String>,
    want_dnsmos: bool,
    want_speaker: bool,
    notes: Vec<String>,
}

impl Screens {
    fn new(opts: &spec::TtsOptions, runner: &Runner<'_>, run: &RunDef) -> Screens {
        let mut s = Screens {
            stt: None,
            stt_error: None,
            dnsmos: None,
            dnsmos_error: None,
            speaker: None,
            speaker_error: None,
            want_dnsmos: opts.dnsmos,
            want_speaker: opts.speaker_similarity,
            notes: Vec::new(),
        };
        if let Some(i) = &opts.intelligibility {
            match runner.spec.engine(&i.engine) {
                Some(def) => match runner.drivers.stt(def, &Feed::default()) {
                    Ok(d) => s.stt = Some(d),
                    Err(e) => s.stt_error = Some(format!("intelligibility engine {}: {e}", i.engine)),
                },
                None => s.stt_error = Some(format!("intelligibility engine {} is not in the spec", i.engine)),
            }
        }
        let _ = run;
        s
    }

    fn apply(&mut self, prompt: &TtsPrompt, out: &tts::TtsOutput, item: &mut TtsItem) {
        let mut errors: Vec<String> = Vec::new();
        let audio16 = if out.sample_rate == 16_000 {
            Some(out.samples.clone())
        } else {
            crate::services::bench::signals::resample(&out.samples, out.sample_rate.max(1), 16_000).ok()
        };
        if let (Some(driver), Some(a16)) = (self.stt.as_mut(), audio16.as_ref()) {
            let feed = Feed::default();
            let lang = crate::services::speech::lang::normalize(&prompt.lang);
            match driver.transcribe(&SttRequest {
                item: &prompt.id,
                audio: a16,
                language: lang,
                feed: &feed,
                timeout: Duration::from_secs(120),
            }) {
                Ok(o) => item.asr_text = Some(o.text),
                Err(e) => errors.push(format!("intelligibility: {e}")),
            }
        } else if let Some(e) = &self.stt_error {
            errors.push(e.clone());
        }
        if self.want_dnsmos {
            if self.dnsmos.is_none() && self.dnsmos_error.is_none() {
                match crate::services::bench::scorers::Dnsmos::resolve("", "cpu") {
                    Ok(Some(d)) => self.dnsmos = Some(d),
                    Ok(None) => self.dnsmos_error = Some("DNSMOS: bench/dnsmos-p835 is not installed".into()),
                    Err(e) => self.dnsmos_error = Some(format!("DNSMOS: {e}")),
                }
            }
            match (self.dnsmos.as_mut(), audio16.as_ref()) {
                (Some(d), Some(a16)) => match d.score(a16) {
                    Ok(s) => {
                        item.dnsmos_sig = Some(round_to(s.sig, 3));
                        item.dnsmos_bak = Some(round_to(s.bak, 3));
                        item.dnsmos_ovrl = Some(round_to(s.ovrl, 3));
                    }
                    Err(e) => errors.push(e),
                },
                _ => {
                    if let Some(e) = &self.dnsmos_error {
                        errors.push(e.clone());
                    }
                }
            }
        }
        if self.want_speaker {
            if let Some(reference) = prompt.reference_voice.as_deref().filter(|r| !r.is_empty()) {
                if self.speaker.is_none() && self.speaker_error.is_none() {
                    match crate::services::bench::scorers::SpeakerEmbedder::resolve_v2("", "cpu") {
                        Ok(Some(s)) => self.speaker = Some(s),
                        Ok(None) => self.speaker_error = Some("CAM++: bench/campplus-exact is not installed".into()),
                        Err(e) => self.speaker_error = Some(format!("CAM++: {e}")),
                    }
                }
                match (self.speaker.as_mut(), audio16.as_ref()) {
                    (Some(sp), Some(a16)) => {
                        let result = crate::services::diagnostics::app_data_dir()
                            .ok_or_else(|| "the app-data folder is unknown".to_string())
                            .and_then(|d| crate::services::voice_library::reference_clip(&d, reference))
                            .and_then(|clip| corpus::read_audio_16k(&clip.path))
                            .and_then(|r16| sp.similarity(a16, &r16));
                        match result {
                            Ok(c) => item.speaker_cosine = Some(round_to(c, 4)),
                            Err(e) => errors.push(format!("speaker similarity: {e}")),
                        }
                    }
                    _ => {
                        if let Some(e) = &self.speaker_error {
                            errors.push(e.clone());
                        }
                    }
                }
            }
        }
        if !errors.is_empty() {
            for e in &errors {
                if !self.notes.contains(e) {
                    self.notes.push(e.clone());
                }
            }
            item.extra = Some(json!({ "prescreen_errors": errors }));
        }
    }

    fn finish(self, rr: &mut RunReport) {
        for n in self.notes {
            rr.warnings.push(format!("pre-screen: {n}"));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The whole spec
// ---------------------------------------------------------------------------------------------

/// Everything the finished runs add up to.
pub fn assemble_report(runner: &Runner<'_>, runs: Vec<RunReport>, started_unix_ms: u64, started: Instant) -> Report {
    let spec = runner.spec;
    let mut environment = environment_block();
    // The runtime packs the engines ran on.
    let mut packs: Vec<RuntimePack> = Vec::new();
    for info in &runner.engine_infos {
        let helper = spec.engine(&info.id).and_then(|d| d.options.get("helper").and_then(Value::as_str));
        let variant = info.variant_used.as_deref().or_else(|| if info.execution_provider_used.as_deref() == Some("cpu") { Some("cpu") } else { None });
        for id in runtime_pack_ids(&info.backend, variant, helper) {
            if packs.iter().any(|p| p.id == id) {
                continue;
            }
            if let Some(mut pack) = runner.drivers.runtime_pack(&id) {
                pack.variant = variant.map(str::to_string);
                packs.push(pack);
            }
        }
    }
    environment.runtime_packs = packs;
    // Engines in the spec's order (those that never ran have no row).
    let mut engines: Vec<EngineInfo> = Vec::new();
    for def in &spec.engines {
        if let Some(i) = runner.engine_infos.iter().find(|i| i.id == def.id) {
            engines.push(i.clone());
        }
    }
    let corpus = runner.manifest.as_ref().map(|m| CorpusStamp {
        name: m.name.clone(),
        content_sha256: m.content_sha256.clone(),
        recipe_sha256: m.recipe_sha256.clone(),
        manifest_sha256: Some(m.manifest_sha256.clone()),
        items_selected: runner.selected_ids.len() as u64,
    });
    let mut failures = runner.failures.clone();
    for run in &runs {
        if run.status == RunStatus::Cancelled {
            failures.push(format!("{}: cancelled", run.name));
        }
    }
    let passed = runs.iter().all(|r| r.status == RunStatus::Passed) && failures.is_empty();
    Report {
        schema: REPORT_SCHEMA.to_string(),
        name: spec.name.clone(),
        description: spec.description.clone(),
        spec_sha256: spec::spec_sha256(runner.raw_spec),
        spec: runner.raw_spec.clone(),
        started_unix_ms,
        duration_ms: round_to(started.elapsed().as_secs_f64() * 1000.0, 1),
        passed,
        failures,
        environment,
        corpus,
        engines,
        runs,
        files: Vec::new(),
    }
}

/// The text printed to stdout.
pub fn summary(report: &Report, out_dir: &Path, wrote_export: bool) -> String {
    let mut s = format!(
        "Sigil speech bench '{}': {} ({} run(s), {:.1} s)\n",
        report.name,
        if report.passed { "PASSED" } else { "FAILED" },
        report.runs.len(),
        report.duration_ms / 1000.0
    );
    for run in &report.runs {
        s.push_str(&format!("  {} ({}): {}", run.name, run.mode.as_str(), run.status.as_str()));
        if let Some(e) = &run.error {
            s.push_str(&format!(" - {e}"));
        }
        s.push('\n');
        for r in &run.results {
            s.push_str(&format!(
                "    {:<24} {:<8} {} item(s), {} error(s){}\n",
                r.engine,
                r.status.as_str(),
                r.aggregate.items,
                r.aggregate.errors,
                r.error.as_ref().map(|e| format!(": {e}")).unwrap_or_default()
            ));
        }
    }
    for f in &report.failures {
        s.push_str(&format!("  failure: {f}\n"));
    }
    s.push_str(&format!("  report: {}{}", out_dir.join("report.json").display(), if wrote_export { " (+ report.md, export.json)" } else { " (+ report.md)" }));
    s
}

/// Write the report, its Markdown and the export; returns whether `export.json` was written.
pub fn write_outputs(spec: &SpeechSpec, report: &mut Report, out_dir: &Path) -> Result<bool, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    // The export first: a leak is a failure of the run, recorded in the report.
    let report_value = serde_json::to_value(&*report).map_err(|e| e.to_string())?;
    let export = export::make_export(&report_value, spec.output.export.include_text, spec.output.export.include_items, None, unix_ms_now());
    let leaks = export::privacy_errors(&export);
    let wrote_export = if leaks.is_empty() {
        let text = serde_json::to_string_pretty(&export).map_err(|e| e.to_string())?;
        std::fs::write(out_dir.join("export.json"), text).map_err(|e| format!("cannot write export.json: {e}"))?;
        true
    } else {
        report.failures.push(format!("export.json not written: {} privacy problem(s), first: {}", leaks.len(), leaks[0]));
        report.passed = false;
        false
    };
    report.files = vec!["report.md".to_string()];
    if wrote_export {
        report.files.push("export.json".to_string());
    }
    let json = serde_json::to_string_pretty(&*report).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("report.json"), json).map_err(|e| format!("cannot write report.json: {e}"))?;
    if spec.output.write_markdown {
        std::fs::write(out_dir.join("report.md"), markdown(report)).map_err(|e| format!("cannot write report.md: {e}"))?;
    }
    Ok(wrote_export)
}

/// Run a parsed spec with `drivers` and write its outputs; the testable body of the CLI.
pub fn run_spec_with(spec: &SpeechSpec, raw: &Value, spec_file: &Path, out_dir: &Path, drivers: &dyn Drivers) -> Result<String, String> {
    let started = Instant::now();
    let started_unix_ms = unix_ms_now();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let mut runner = Runner::new(spec, raw, spec_file, out_dir.to_path_buf(), drivers);
    let mut runs = Vec::new();
    for (n, run) in spec.runs.iter().enumerate() {
        eprintln!("[speech-bench] run {}/{}: {} ({})", n + 1, spec.runs.len(), run.name, run.mode.as_str());
        let rr = runner.execute_run(run);
        eprintln!("[speech-bench] run {}: {}", run.name, rr.status.as_str());
        runs.push(rr);
    }
    let mut report = assemble_report(&runner, runs, started_unix_ms, started);
    let wrote_export = write_outputs(spec, &mut report, out_dir)?;
    let text = summary(&report, out_dir, wrote_export);
    if report.passed {
        Ok(text)
    } else {
        Err(text)
    }
}

fn run_on_this_thread(spec_path: &Path, out: Option<&Path>) -> Result<String, String> {
    let text = std::fs::read_to_string(spec_path).map_err(|e| format!("cannot read {}: {e}", spec_path.display()))?;
    let (spec, raw) = SpeechSpec::parse(&text).map_err(|e| format!("{}: {e}", spec_path.display()))?;
    let out_dir = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("bench-results").join("speech").join(format!("{}-{}", spec.name, utc_stamp(unix_ms_now() / 1000))));
    let drivers = Production {
        paths: Paths::new(spec_path),
    };
    run_spec_with(&spec, &raw, spec_path, &out_dir, &drivers)
}

/// Body of `Sigil.exe --headless speech-bench <spec.json> [--out <dir>]` (frozen P7-S0 contract).
///
/// Writes `report.json`, `report.md`, `hyps/`, `audio/` and `export.json` into `out` (default
/// `bench-results/speech/<name>-<timestamp>/`) and returns the summary printed to stdout. `Err`
/// (a bad spec, a failed run, item, engine or threshold) exits with code 1; the files are written
/// either way.
pub fn headless_speech_bench(spec: &Path, out: Option<&Path>) -> Result<String, String> {
    // On a thread with the benchmark stack, not Windows' 1 MB main thread.
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("sigil-speech-bench".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn_scoped(scope, || run_on_this_thread(spec, out))
            .map_err(|e| format!("cannot start the benchmark thread: {e}"))?
            .join()
            .map_err(|_| "the speech benchmark failed unexpectedly".to_string())?
    })
}
