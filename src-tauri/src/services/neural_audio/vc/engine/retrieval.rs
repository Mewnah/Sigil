//! Feature retrieval (P3-D): the exact GPU kNN graph (`knn.onnx`: MatMul, top-k = 8, 1/d² blend,
//! index resident on the device; the template's `knn` graph, ADR 0004) and the CPU IVF reader
//! (`faiss.rs`, parallel search checked against an independent f64 reference). An index with fewer
//! rows than the template's `min_index_rows` (64) is padded or retrieval is turned off.
//!
//! # Backends
//! - [`RetrievalBackend::CpuIvf`] (legacy, RVC's own search): `FaissIndex` (IVF with the file's
//!   `nprobe`, or flat). Frames are searched on a small persistent pool of threads when a tick's
//!   search is big enough to pay for it ([`PARALLEL_MIN_MACS`]); the result is bit-identical to the
//!   serial search. The Streamer preset (one thread) stays serial.
//! - [`RetrievalBackend::GpuKnn`]: the template's `knn` graph on the GPU rung. The index (fp16 in
//!   the fp16 precision, fp32 otherwise) and its squared row norms are bound once through
//!   IOBinding, so ONNX Runtime keeps them on the device; a tick uploads only its features. The
//!   search is **exact** over every row (a coarse top-64 on `|x|² - 2 q·x`, then an exact fp32
//!   re-rank), where RVC's IVF with `nprobe = 1` looks in one list. The graph is found in the
//!   bundled templates by the index's width (768 or 256) and checked against the layout's sha256.
//!   A GPU kNN that would end up on the CPU rung is refused (the engine then keeps the CPU index).
//!
//! # Frozen by P3-S0 (signatures)
//! [`gpu_knn_available`], [`RetrievalBackend`], [`Retrieval::load`], [`Retrieval::backend`],
//! [`Retrieval::dim`], [`Retrieval::rows`], [`Retrieval::blend`] (in place, `frames × dim`
//! features).
//!
//! # Added by P3-D
//! [`Retrieval::prepare`] (warm the GPU graph at the plan's frame count), [`Retrieval::outcome`],
//! [`Retrieval::threads`], [`Retrieval::vram_mb`], [`knn_graph`], [`MIN_INDEX_ROWS`],
//! [`MIN_REAL_ROWS`], [`PARALLEL_MIN_MACS`].
//!
//! # GPU footprint (P3B-G, T3-W6)
//! The GPU kNN's VRAM is **measured** when it can be: the process's VRAM on the session's adapter
//! (DXGI) from before the session is built to after the index is bound, held in the
//! `gpu::arena` ledger under `vc.knn` while the index lives and reported to the benchmark's ledger
//! (`telemetry::vram_mark`). The delta is process-wide, so it is trusted only when no other probe's
//! window overlapped it (`gpu::arena::VramDelta::exclusive`; the engine's default parallel cold
//! start builds the pitch bank, the content encoder and the generator beside the kNN, whose
//! allocations would otherwise be booked as `vc.knn`). An overlapped load holds the computed size
//! instead: the index and norms plus the session's measured fixed cost ([`knn_footprint`]);
//! [`Retrieval::vram_measured`] says which. `SIGIL_VC_PARALLEL_LOAD=0` (the benchmark's kNN probe
//! rows) loads the branches one after another, so the measurement is exclusive. [`Retrieval::vram_mb`]
//! returns the held value. On CUDA the session's arena grows only by what is requested
//! (`arena_extend_strategy = kSameAsRequested`, [`KNN_EXACT_ARENA_DEFAULT`];
//! `SIGIL_VC_KNN_ARENA=pow2` restores ONNX Runtime's power-of-two growth), so the index copy the
//! binding makes on the device does not land in a power-of-two region. The session is built under
//! the `vc.knn` component, so `SIGIL_ORT_PROFILE` / `SIGIL_ORT_OPTIMIZED_DIR` reach it. ONNX
//! Runtime fuses the graph's `Transpose(index) -> MatMul` into one `FusedMatMul` on every EP at the
//! Level3 optimisation Sigil uses, so no transposed copy of the index exists (checked with
//! `SIGIL_ORT_OPTIMIZED_DIR`; see `scripts/bench/specs/p3b/g/README.md`).

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;

use ort::session::{IoBinding, Session};
use ort::value::{DynValue, Tensor};
use sha2::{Digest, Sha256};

use crate::services::diagnostics::{record_ep_outcome, EpOutcome};
use crate::services::neural_audio::gpu::arena::{self, VramDelta, VramHold, VramProbe};
use crate::services::neural_audio::gpu::io_binding;
use crate::services::neural_audio::ort_session::{build_session_ext, device_for, normalize_ep, GpuExtras, ModelSource, VcSessionOptions};
use crate::services::neural_audio::telemetry::{self, LoadStage, SyncKind};
use crate::services::neural_audio::vc::faiss::{BlendScratch, FaissIndex};
use crate::services::neural_audio::vc::import;
use crate::services::rt_thread;

use super::options::PrecisionMode;

/// The kNN graph's coarse top-k: an index needs at least this many rows (padded when smaller).
pub const MIN_INDEX_ROWS: usize = 64;
/// Real rows below which retrieval is refused (fewer than the 8 neighbours RVC blends).
pub const MIN_REAL_ROWS: usize = 8;
/// A tick's CPU search runs on the pool only above this many multiply-adds (about 0.3 ms serial).
pub const PARALLEL_MIN_MACS: usize = 1_500_000;
/// Most CPU search threads (the caller's thread included).
const MAX_SEARCH_THREADS: usize = 4;
/// Value of the padding rows: far from any content feature, so they never rank among the 8.
const PAD_VALUE: f32 = 1.0e4;

/// `SIGIL_VC_KNN_ARENA=exact|pow2`: the GPU kNN session's CUDA arena growth (P3B-G's switch; see the
/// module docs).
pub const KNN_ARENA_ENV: &str = "SIGIL_VC_KNN_ARENA";

/// Whether the GPU kNN's CUDA arena grows only by what is requested when [`KNN_ARENA_ENV`] is unset
/// (exact, zero numeric change; the plans use the same strategy).
pub const KNN_EXACT_ARENA_DEFAULT: bool = true;

fn knn_exact_arena() -> bool {
    match std::env::var(KNN_ARENA_ENV).as_deref().map(str::trim) {
        Ok("pow2" | "default" | "0") => false,
        Ok("exact" | "1") => true,
        _ => KNN_EXACT_ARENA_DEFAULT,
    }
}

/// Whether the exact GPU kNN backend runs in this build (the `gpu_knn` lever): the bundled
/// templates carry the graph.
pub fn gpu_knn_available() -> bool {
    import::templates_dir().is_some_and(|dir| any_knn_graph(&dir))
}

fn any_knn_graph(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().join("knn.fp32.onnx").is_file())
        })
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalBackend {
    /// `faiss.rs` IVF / Flat reader on the CPU (legacy).
    CpuIvf,
    /// Exact kNN on the GPU through the template's `knn` graph.
    GpuKnn,
}

/// Found graphs, by (templates folder, width, fp16).
type GraphCache = Mutex<HashMap<(PathBuf, usize, bool), PathBuf>>;

fn graph_cache() -> &'static GraphCache {
    static CACHE: OnceLock<GraphCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The bundled `knn` graph for features of width `dim` (fp16 index storage or fp32): the first
/// template (by name) whose layout declares that width and whose graph file matches the layout's
/// sha256.
pub fn knn_graph(dim: usize, fp16: bool) -> Result<PathBuf, String> {
    let dir = import::templates_dir().ok_or("the bundled vc-templates folder was not found")?;
    knn_graph_in(&dir, dim, fp16)
}

fn knn_graph_in(dir: &Path, dim: usize, fp16: bool) -> Result<PathBuf, String> {
    let key = (dir.to_path_buf(), dim, fp16);
    if let Some(p) = graph_cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(p.clone());
    }
    let mut ids: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join(import::layout::LAYOUT_FILE).is_file())
        .collect();
    ids.sort();
    let variant = if fp16 { "fp16" } else { "fp32" };
    for folder in ids {
        let Ok(bytes) = std::fs::read(folder.join(import::layout::LAYOUT_FILE)) else {
            continue;
        };
        let Ok(layout) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if layout["arch"]["emb_channels"].as_u64() != Some(dim as u64) {
            continue;
        }
        let v = &layout["graphs"]["knn"]["variants"][variant];
        let (Some(file), Some(sha)) = (v["file"].as_str(), v["sha256"].as_str()) else {
            continue;
        };
        if file.contains(['/', '\\']) {
            continue;
        }
        let path = folder.join(file);
        let Ok(graph) = std::fs::read(&path) else { continue };
        let digest = Sha256::digest(&graph);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        if !hex.eq_ignore_ascii_case(sha) {
            tracing::warn!(target: "sigil::vc", "knn graph {} does not match its layout's sha256; skipped", path.display());
            continue;
        }
        graph_cache()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, path.clone());
        return Ok(path);
    }
    Err(format!(
        "no bundled kNN graph ({variant}) for {dim}-dimension features in {}",
        dir.display()
    ))
}

// ---------------------------------------------------------------------------------------------
// CPU search pool
// ---------------------------------------------------------------------------------------------

/// One tick's search, shared with the pool's workers.
#[derive(Clone, Copy)]
struct Job {
    /// The caller's features (`frames × dim`), valid until every worker reported back.
    ptr: *mut f32,
    frames: usize,
    rate: f32,
    /// Chunks the frames are split into (the caller takes chunk 0).
    chunks: usize,
}

// SAFETY: the pointer is only dereferenced by workers between the job's publication and their
// completion report, while the caller blocks in `CpuPool::blend` holding the `&mut` it came from;
// every worker touches a disjoint range of it.
unsafe impl Send for Job {}

#[derive(Default)]
struct PoolState {
    generation: u64,
    job: Option<Job>,
    remaining: usize,
    shutdown: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    wake: Condvar,
    done: Condvar,
}

/// Persistent search threads (spawned at load, parked between ticks).
struct CpuPool {
    shared: Arc<PoolShared>,
    workers: Vec<JoinHandle<()>>,
}

fn chunk_range(frames: usize, chunks: usize, k: usize) -> (usize, usize) {
    (k * frames / chunks, (k + 1) * frames / chunks)
}

impl CpuPool {
    fn new(index: Arc<FaissIndex>, helpers: usize, priority: Option<rt_thread::WorkerPriority>) -> Result<CpuPool, String> {
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(helpers);
        for w in 0..helpers {
            let shared = Arc::clone(&shared);
            let index = Arc::clone(&index);
            let handle = std::thread::Builder::new()
                .name(format!("sigil-vc-knn-{w}"))
                .spawn(move || {
                    if let Some(p) = priority {
                        let _ = rt_thread::set_worker_priority(p, false);
                    }
                    let mut scratch = BlendScratch::new();
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut st = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
                            while st.generation == seen && !st.shutdown {
                                st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
                            }
                            if st.shutdown {
                                return;
                            }
                            seen = st.generation;
                            st.job
                        };
                        if let Some(job) = job {
                            let (from, to) = chunk_range(job.frames, job.chunks, w + 1);
                            if to > from {
                                let dim = index.dim();
                                // SAFETY: see `Job`: disjoint, live for the job's duration.
                                let slice = unsafe { std::slice::from_raw_parts_mut(job.ptr.add(from * dim), (to - from) * dim) };
                                // A panic must still report back, or the voice worker would wait for
                                // this share forever; the share is then left unblended.
                                let blended =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| index.blend_range(slice, job.rate, &mut scratch)));
                                if blended.is_err() {
                                    scratch = BlendScratch::new();
                                }
                            }
                        }
                        let mut st = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
                        st.remaining = st.remaining.saturating_sub(1);
                        if st.remaining == 0 {
                            shared.done.notify_all();
                        }
                    }
                })
                .map_err(|e| format!("retrieval search thread: {e}"))?;
            workers.push(handle);
        }
        Ok(CpuPool { shared, workers })
    }

    /// Blend `frames` rows of `features` on the caller plus every worker.
    fn blend(&self, index: &FaissIndex, features: &mut [f32], frames: usize, rate: f32, scratch: &mut BlendScratch) {
        let chunks = self.workers.len() + 1;
        let dim = index.dim();
        let frames = frames.min(features.len() / dim.max(1));
        // Every share, the caller's included, is derived from this one pointer; `features` is not
        // touched again until the workers are done.
        let ptr = features.as_mut_ptr();
        {
            let mut st = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            st.job = Some(Job { ptr, frames, rate, chunks });
            st.remaining = self.workers.len();
            st.generation = st.generation.wrapping_add(1);
        }
        self.shared.wake.notify_all();
        let (from, to) = chunk_range(frames, chunks, 0);
        // SAFETY: frames `[from, to)` are inside `features` and no worker touches them.
        let own = unsafe { std::slice::from_raw_parts_mut(ptr.add(from * dim), (to - from) * dim) };
        index.blend_range(own, rate, scratch);
        let mut st = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while st.remaining > 0 {
            st = self
                .shared
                .done
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
        st.job = None;
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        {
            let mut st = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            st.shutdown = true;
        }
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GPU kNN
// ---------------------------------------------------------------------------------------------

/// A host tensor bound to the kNN session.
struct Bound {
    name: CString,
    value: DynValue,
}

impl Bound {
    fn f32(name: &str, shape: Vec<usize>, data: Vec<f32>) -> Result<Bound, String> {
        let value = Tensor::from_array((shape, data))
            .map_err(|e| format!("kNN '{name}': {e}"))?
            .into_dyn();
        Ok(Bound {
            name: io_binding::c_name(name)?,
            value,
        })
    }

    fn f16(name: &str, shape: Vec<usize>, data: Vec<half::f16>) -> Result<Bound, String> {
        let value = Tensor::from_array((shape, data))
            .map_err(|e| format!("kNN '{name}': {e}"))?
            .into_dyn();
        Ok(Bound {
            name: io_binding::c_name(name)?,
            value,
        })
    }

    fn data_mut(&mut self) -> Result<&mut [f32], String> {
        self.value
            .try_extract_tensor_mut::<f32>()
            .map(|(_, d)| d)
            .map_err(|e| format!("kNN tensor: {e}"))
    }

    fn data(&self) -> Result<&[f32], String> {
        self.value
            .try_extract_tensor::<f32>()
            .map(|(_, d)| d)
            .map_err(|e| format!("kNN tensor: {e}"))
    }
}

struct GpuKnn {
    // Declared first: the binding's references go before the values they reference.
    binding: IoBinding,
    session: Session,
    outcome: EpOutcome,
    dim: usize,
    rows: usize,
    /// Index and norms stay bound (and alive) for the session's lifetime.
    _index: Bound,
    _index_sq: Bound,
    feats: Option<Bound>,
    out: Option<Bound>,
    frames: usize,
    rate: Bound,
    rate_value: f32,
    /// The index and norms' bytes (MiB), the fallback when the footprint cannot be measured.
    index_mb: f64,
    /// The footprint held in the VRAM ledger while the index lives (P3B-G): measured, or computed
    /// when the measurement was not exclusive.
    measured: Option<VramHold>,
    /// Whether `measured` is a DXGI measurement (an exclusive one) rather than the computed size.
    vram_measured: bool,
}

/// The diagnostics and VRAM-ledger key of the kNN session: `[<prefix>.]vc.knn`.
fn knn_component(opts: &VcSessionOptions) -> String {
    match opts.component_prefix {
        Some(p) if !p.is_empty() => format!("{p}.vc.knn"),
        _ => "vc.knn".to_string(),
    }
}

/// VRAM the kNN session takes beyond its index and norms (MiB), measured with the kNN loading alone
/// (`scripts/bench/specs/p3b/g/README.md`: a 30 000-row 768-wide index took 106 MB on CUDA fp32
/// (88 MiB of index), 62 MB on CUDA fp16 (44 MiB) and 68 MB on DirectML fp16 (44 MiB); an 8 000-row
/// index 42 MB against 23.5 MiB, fp16 30 MB against 11.7 MiB): the same ~18 MB on CUDA at any size
/// and storage (context buffers and the session's arena granularity), ~24 MB on DirectML.
fn knn_session_overhead_mb(active_ep: &str) -> f64 {
    match active_ep {
        "cuda" => 18.0,
        "dml" => 24.0,
        _ => 0.0,
    }
}

/// What the kNN holds in the VRAM ledger.
#[derive(Clone, Copy, Debug, PartialEq)]
struct KnnFootprint {
    mb: f64,
    /// A DXGI measurement nothing else contributed to (else the computed size).
    measured: bool,
}

/// The kNN's footprint: the DXGI delta when its measurement window was exclusive, else (another
/// load's allocations are in the delta, or DXGI could not tell) the index and norms plus the
/// session's fixed cost on the active EP.
fn knn_footprint(index_mb: f64, active_ep: &str, delta: Option<VramDelta>) -> KnnFootprint {
    match delta {
        Some(d) if d.exclusive => KnnFootprint { mb: d.mb, measured: true },
        _ => KnnFootprint { mb: index_mb + knn_session_overhead_mb(active_ep), measured: false },
    }
}

impl GpuKnn {
    #[allow(clippy::too_many_arguments)]
    fn load(index: &FaissIndex, graph: &Path, ep: &str, opts: &VcSessionOptions, fp16: bool, allow_cpu: bool) -> Result<GpuKnn, String> {
        if !index.is_l2() {
            return Err("the index ranks by inner product; the kNN graph is L2".into());
        }
        let real = index.rows();
        if real < MIN_REAL_ROWS {
            return Err(format!("the index has {real} rows (at least {MIN_REAL_ROWS} are needed)"));
        }
        let dim = index.dim();
        // The footprint is measured on the requested EP's adapter from before the session exists.
        let base = normalize_ep(ep);
        let probe = arena::dxgi_adapter_for(base, device_for(base, base, opts)).and_then(VramProbe::start);
        // The outcome is recorded below, once the session is accepted (a GPU kNN that landed on the
        // CPU rung is refused and must leave no `vc.knn` row).
        let extras = GpuExtras { cuda_same_as_requested: knn_exact_arena(), skip_outcome_record: true, ..GpuExtras::default() };
        let (session, outcome) = {
            let _span = telemetry::load_span(LoadStage::KnnSessionInit);
            let built = build_session_ext(Some("vc.knn"), ep, opts, ModelSource::File(graph), &extras, &|b, _| Ok(b))?;
            (built.session, built.outcome)
        };
        if outcome.active == "cpu" && !allow_cpu {
            let _ = probe.and_then(|p| p.delta_mb()); // ends the measurement window
            return Err(match &outcome.reason {
                Some(r) => format!("the kNN graph would run on the CPU ({r})"),
                None => "the kNN graph would run on the CPU".to_string(),
            });
        }
        let rows = real.max(MIN_INDEX_ROWS);
        let mut data = Vec::new();
        index.copy_vectors(&mut data);
        data.resize(rows * dim, PAD_VALUE);
        let (index_value, norms) = if fp16 {
            let h: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let norms: Vec<f32> = h
                .chunks_exact(dim)
                .map(|r| r.iter().map(|v| v.to_f32() * v.to_f32()).sum())
                .collect();
            (Bound::f16("index", vec![rows, dim], h)?, norms)
        } else {
            let norms: Vec<f32> = data
                .chunks_exact(dim)
                .map(|r| r.iter().map(|v| v * v).sum())
                .collect();
            (Bound::f32("index", vec![rows, dim], data)?, norms)
        };
        let index_sq = Bound::f32("index_sq", vec![rows], norms)?;
        let rate = Bound::f32("index_rate", vec![1], vec![0.0])?;
        let mut binding = session
            .create_binding()
            .map_err(|e| format!("kNN IOBinding: {e}"))?;
        for b in [&index_value, &index_sq, &rate] {
            io_binding::bind_input(&mut binding, &b.name, &b.value).map_err(|e| format!("kNN bind: {e}"))?;
        }
        let bytes = rows * dim * if fp16 { 2 } else { 4 } + rows * 4;
        let index_mb = bytes as f64 / (1024.0 * 1024.0);
        // The binding copied the index to the device: what the session and its index hold now (on
        // the adapter it was measured on: a CUDA request that fell back to DirectML is not
        // measured, and neither is a load another probe's window overlapped).
        // (Read in every case, which also ends the probe's window for the others' overlap check.)
        let delta = probe.and_then(|p| p.delta()).filter(|_| outcome.active == base);
        let footprint = knn_footprint(index_mb, &outcome.active, delta);
        let measured = (outcome.active != "cpu").then(|| {
            if footprint.measured {
                telemetry::vram_mark("vc.knn", footprint.mb);
            }
            tracing::info!(
                target: "sigil::vc",
                "GPU kNN on {}: {:.0} MB VRAM {} ({index_mb:.0} MB of index and norms, {rows} rows x {dim}, {})",
                outcome.active,
                footprint.mb,
                match delta {
                    Some(d) if !d.exclusive => "computed (another load measured at the same time)",
                    _ if footprint.measured => "measured",
                    _ => "computed",
                },
                if fp16 { "fp16" } else { "fp32" }
            );
            arena::ledger().hold(&knn_component(opts), footprint.mb)
        });
        let vram_measured = footprint.measured;
        // The session is accepted: its outcome is the `vc.knn` diagnostics row (`Retrieval::load`
        // recorded it at this point before the session carried a component).
        record_ep_outcome(&knn_component(opts), outcome.clone());
        Ok(GpuKnn {
            binding,
            session,
            outcome,
            dim,
            rows: real,
            _index: index_value,
            _index_sq: index_sq,
            feats: None,
            out: None,
            frames: 0,
            rate,
            rate_value: 0.0,
            index_mb,
            measured,
            vram_measured,
        })
    }

    /// Size the feature input and output for `frames` (only when it changes).
    fn shape_for(&mut self, frames: usize) -> Result<(), String> {
        if self.frames == frames && self.feats.is_some() {
            return Ok(());
        }
        let feats = Bound::f32("feats", vec![frames, self.dim], vec![0.0; frames * self.dim])?;
        let out = Bound::f32("feats_out", vec![frames, self.dim], vec![0.0; frames * self.dim])?;
        io_binding::bind_output(&mut self.binding, &out.name, &out.value).map_err(|e| format!("kNN bind output: {e}"))?;
        self.feats = Some(feats);
        self.out = Some(out);
        self.frames = frames;
        Ok(())
    }

    fn blend(&mut self, features: &mut [f32], frames: usize, rate: f32) -> Result<(), String> {
        let n = frames * self.dim;
        self.shape_for(frames)?;
        if rate != self.rate_value {
            self.rate.data_mut()?[0] = rate;
            io_binding::bind_input(&mut self.binding, &self.rate.name, &self.rate.value).map_err(|e| format!("kNN bind: {e}"))?;
            self.rate_value = rate;
        }
        let feats = self.feats.as_mut().ok_or("kNN input missing")?;
        feats.data_mut()?.copy_from_slice(&features[..n]);
        io_binding::bind_input(&mut self.binding, &feats.name, &feats.value).map_err(|e| format!("kNN bind: {e}"))?;
        io_binding::run(&mut self.session, &self.binding).map_err(|e| format!("kNN run: {e}"))?;
        // A bound run without a plan (P3B-S0 telemetry: one GPU wait).
        telemetry::record_sync(SyncKind::EagerRun);
        let out = self.out.as_ref().ok_or("kNN output missing")?;
        features[..n].copy_from_slice(out.data()?);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------------------------

pub struct Retrieval {
    backend: RetrievalBackend,
    cpu: Option<Arc<FaissIndex>>,
    pool: Option<CpuPool>,
    scratch: BlendScratch,
    gpu: Option<GpuKnn>,
    dim: usize,
    rows: usize,
}

/// fp32 index storage above this size goes to fp16 under `auto` precision (keeps a large index
/// inside the plan's VRAM budget, T3).
pub const KNN_FP16_ABOVE_BYTES: usize = 256 << 20;

/// Index storage of the GPU kNN: an explicit precision request wins; under `auto`, fp16 on
/// DirectML (measured 5.3 vs 6.4 ms per tick at 30 000 rows) and, on CUDA, fp32 (P3: 1.1 vs 5.0 ms;
/// P3B-G measured fp16 no slower on ORT 1.28 and 44 MB smaller at 30 000 rows, with a blend rounding of
/// 2.7e-4; P3B-I0 kept [`KNN_FP16_ON_CUDA_DEFAULT`] off so the blend stays exact) unless the fp32 index would exceed [`KNN_FP16_ABOVE_BYTES`]. Either way the re-rank is
/// fp32. `SIGIL_VC_KNN_STORAGE=fp16|fp32` ([`KNN_STORAGE_ENV`]) overrides `auto` (P3B-G's switch).
fn knn_fp16(precision: PrecisionMode, ep: &str, fp32_bytes: usize) -> bool {
    knn_fp16_with(precision, ep, fp32_bytes, std::env::var(KNN_STORAGE_ENV).ok().as_deref())
}

fn knn_fp16_with(precision: PrecisionMode, ep: &str, fp32_bytes: usize, storage: Option<&str>) -> bool {
    match precision {
        PrecisionMode::Fp32 => false,
        PrecisionMode::Fp16 => true,
        PrecisionMode::Auto => match storage.map(str::trim) {
            Some("fp16") => true,
            Some("fp32") => false,
            _ => {
                let ep = crate::services::neural_audio::ort_session::normalize_ep(ep);
                ep == "dml" || (ep == "cuda" && KNN_FP16_ON_CUDA_DEFAULT) || fp32_bytes > KNN_FP16_ABOVE_BYTES
            }
        },
    }
}

/// `SIGIL_VC_KNN_STORAGE=fp16|fp32`: the GPU kNN index storage under `auto` precision (P3B-G).
pub const KNN_STORAGE_ENV: &str = "SIGIL_VC_KNN_STORAGE";

/// Whether `auto` stores a CUDA kNN index in fp16 below [`KNN_FP16_ABOVE_BYTES`] (half the index's
/// VRAM; the fp32 re-rank keeps the neighbours exact, the blend carries the rows' fp16 rounding).
pub const KNN_FP16_ON_CUDA_DEFAULT: bool = false;

/// CPU search threads for a session's options: 1 for the Streamer preset, else the intra-op
/// thread count (default half the cores), at most [`MAX_SEARCH_THREADS`].
fn search_threads(opts: &VcSessionOptions) -> usize {
    if opts.streamer_profile {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    opts.intra_threads
        .unwrap_or(cores / 2)
        .clamp(1, MAX_SEARCH_THREADS)
}

impl Retrieval {
    /// Load `index` for `backend`; the GPU backend uploads the index once (fp16 storage in the
    /// fp16 variant) on `ep`.
    pub fn load(index: &Path, backend: RetrievalBackend, ep: &str, opts: &VcSessionOptions, precision: PrecisionMode) -> Result<Retrieval, String> {
        Self::load_ext(index, backend, ep, opts, precision, None, false)
    }

    /// [`load`](Self::load) with an explicit kNN graph and, for tests, the CPU rung allowed.
    fn load_ext(
        index: &Path,
        backend: RetrievalBackend,
        ep: &str,
        opts: &VcSessionOptions,
        precision: PrecisionMode,
        graph: Option<&Path>,
        allow_cpu: bool,
    ) -> Result<Retrieval, String> {
        let faiss = FaissIndex::load(index)?;
        let (dim, rows) = (faiss.dim(), faiss.rows());
        match backend {
            RetrievalBackend::CpuIvf => {
                let index = Arc::new(faiss);
                let threads = search_threads(opts);
                let pool = if threads > 1 {
                    Some(CpuPool::new(Arc::clone(&index), threads - 1, opts.thread_priority)?)
                } else {
                    None
                };
                Ok(Retrieval {
                    backend,
                    cpu: Some(index),
                    pool,
                    scratch: BlendScratch::new(),
                    gpu: None,
                    dim,
                    rows,
                })
            }
            RetrievalBackend::GpuKnn => {
                let fp16 = knn_fp16(precision, ep, rows * dim * 4);
                let graph = match graph {
                    Some(g) => g.to_path_buf(),
                    None => knn_graph(dim, fp16)?,
                };
                // `GpuKnn::load` records the accepted session's outcome under `[<prefix>.]vc.knn`.
                let gpu = GpuKnn::load(&faiss, &graph, ep, opts, fp16, allow_cpu)?;
                Ok(Retrieval {
                    backend,
                    cpu: None,
                    pool: None,
                    scratch: BlendScratch::new(),
                    gpu: Some(gpu),
                    dim,
                    rows,
                })
            }
        }
    }

    pub fn backend(&self) -> RetrievalBackend {
        self.backend
    }

    /// Feature dimension of the index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rows in the index (without padding).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Where the GPU kNN runs (`None` on the CPU backend).
    pub fn outcome(&self) -> Option<&EpOutcome> {
        self.gpu.as_ref().map(|g| &g.outcome)
    }

    /// CPU search threads (1 = serial; the GPU backend reports 0).
    pub fn threads(&self) -> usize {
        match (&self.gpu, &self.pool) {
            (Some(_), _) => 0,
            (None, Some(p)) => p.workers.len() + 1,
            (None, None) => 1,
        }
    }

    /// Device memory the GPU kNN takes (MiB; 0 on the CPU): measured at load (session, index and
    /// norms; P3B-G), else the index and norms' bytes.
    pub fn vram_mb(&self) -> f64 {
        self.gpu
            .as_ref()
            .map_or(0.0, |g| g.measured.as_ref().map_or(g.index_mb, VramHold::mb))
    }

    /// The index and norms' bytes on the device (MiB; 0 on the CPU).
    pub fn index_mb(&self) -> f64 {
        self.gpu.as_ref().map_or(0.0, |g| g.index_mb)
    }

    /// Whether [`vram_mb`](Self::vram_mb) was measured (DXGI, with no other load measuring at the
    /// same time) rather than computed.
    pub fn vram_measured(&self) -> bool {
        self.gpu.as_ref().is_some_and(|g| g.vram_measured)
    }

    /// Warm the GPU graph at the plan's frame count (sizes its bound buffers, picks the kernels), so
    /// the first retrieving tick does not pay for it. A no-op on the CPU.
    pub fn prepare(&mut self, frames: usize) -> Result<(), String> {
        if let Some(gpu) = self.gpu.as_mut() {
            if frames > 0 {
                let mut probe = vec![0.0f32; frames * gpu.dim];
                gpu.blend(&mut probe, frames, 0.5)?;
            }
        }
        Ok(())
    }

    /// Blend `features` (`frames × dim`) toward their nearest index rows by `index_rate`.
    pub fn blend(&mut self, features: &mut [f32], frames: usize, index_rate: f32) -> Result<(), String> {
        let rate = if index_rate.is_finite() { index_rate.clamp(0.0, 1.0) } else { 0.0 };
        let frames = frames.min(if self.dim == 0 { 0 } else { features.len() / self.dim });
        if rate <= 0.0 || frames == 0 {
            return Ok(());
        }
        if let Some(gpu) = self.gpu.as_mut() {
            return gpu.blend(features, frames, rate);
        }
        let index = self.cpu.as_ref().ok_or("retrieval has no index")?;
        match &self.pool {
            Some(pool) if frames >= 2 && frames.saturating_mul(index.query_cost()) >= PARALLEL_MIN_MACS => {
                pool.blend(index, features, frames, rate, &mut self.scratch);
            }
            _ => index.blend_with(features, frames, rate, &mut self.scratch),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::faiss::tests::{clustered, ivf_of};

    fn temp_index(bytes: &[u8], tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("sigil-p3d-{tag}-{}-{nanos}.index", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn templates() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("vc-templates")
    }

    /// Exact brute force in f64: the 8 nearest rows by squared L2, `(1 / d)^2` weights.
    fn brute_force(data: &[Vec<f32>], q: &[f32], rate: f32) -> (Vec<f64>, Vec<usize>) {
        let d2 = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum::<f64>()
        };
        let mut all: Vec<(f64, usize)> = data
            .iter()
            .enumerate()
            .map(|(i, r)| (d2(q, r), i))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.truncate(8);
        let w: Vec<f64> = all
            .iter()
            .map(|(d, _)| (1.0 / d.max(1e-8)).powi(2))
            .collect();
        let sum: f64 = w.iter().sum();
        let out = (0..q.len())
            .map(|c| {
                all.iter()
                    .zip(&w)
                    .map(|((_, i), w)| w / sum * f64::from(data[*i][c]))
                    .sum::<f64>()
                    * f64::from(rate)
                    + f64::from(q[c]) * (1.0 - f64::from(rate))
            })
            .collect();
        (out, all.iter().map(|x| x.1).collect())
    }

    #[test]
    fn knn_index_precision_follows_the_request_rung_and_size() {
        let f = |p, ep, bytes| knn_fp16_with(p, ep, bytes, None);
        assert!(!f(PrecisionMode::Fp32, "dml", 1 << 30));
        assert!(f(PrecisionMode::Fp16, "cuda", 1));
        assert!(f(PrecisionMode::Auto, "dml", 1));
        assert_eq!(f(PrecisionMode::Auto, "cuda", 100 << 20), KNN_FP16_ON_CUDA_DEFAULT);
        assert!(f(PrecisionMode::Auto, "cuda", 300 << 20));
        // P3B-G's switch overrides `auto` only.
        assert!(knn_fp16_with(PrecisionMode::Auto, "cuda", 1, Some("fp16")));
        assert!(!knn_fp16_with(PrecisionMode::Auto, "dml", 1, Some("fp32")));
        assert!(!knn_fp16_with(PrecisionMode::Fp32, "cuda", 1, Some("fp16")), "an explicit request wins");
        assert!(knn_fp16_with(PrecisionMode::Auto, "dml", 1, Some("bogus")));
    }

    #[test]
    fn finds_the_bundled_knn_graph_by_width() {
        let dir = templates();
        let g768 = knn_graph_in(&dir, 768, false).unwrap();
        assert!(g768.ends_with("knn.fp32.onnx"));
        assert!(g768
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("v2-"));
        let g256 = knn_graph_in(&dir, 256, true).unwrap();
        assert!(g256.ends_with("knn.fp16.onnx"));
        assert!(g256
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("v1-"));
        assert!(knn_graph_in(&dir, 123, false).is_err());
        assert!(any_knn_graph(&dir));
    }

    /// Plan verification: the kNN graph matches exact brute force (the fp32 graph run on the CPU
    /// EP here; GPU numerics are the dev-rig test below), including a padded small index.
    #[test]
    fn knn_graph_matches_brute_force_exactly() {
        let dim = 768;
        for (rows, clusters) in [(2_000usize, 24usize), (40, 4)] {
            let (centres, data) = clustered(dim, rows, clusters, 5 + rows as u64);
            let path = temp_index(&ivf_of(dim, &centres, &data, 1), "knn");
            let graph = knn_graph_in(&templates(), dim, false).unwrap();
            let mut r = Retrieval::load_ext(
                &path,
                RetrievalBackend::GpuKnn,
                "cpu",
                &VcSessionOptions::default(),
                PrecisionMode::Fp32,
                Some(&graph),
                true,
            )
            .unwrap();
            assert_eq!((r.dim(), r.rows(), r.backend()), (dim, rows, RetrievalBackend::GpuKnn));
            let (_, queries) = clustered(dim, 12, clusters, 77);
            let mut feats: Vec<f32> = queries.concat();
            r.prepare(12).unwrap();
            r.blend(&mut feats, 12, 0.8).unwrap();
            for (t, q) in queries.iter().enumerate() {
                let (want, _) = brute_force(&data, q, 0.8);
                for (c, w) in want.iter().enumerate() {
                    let g = f64::from(feats[t * dim + c]);
                    assert!((g - w).abs() < 2e-4, "rows {rows} frame {t} ch {c}: {g} vs {w}");
                }
            }
            // Rate 0: untouched.
            let mut same: Vec<f32> = queries.concat();
            r.blend(&mut same, 12, 0.0).unwrap();
            assert_eq!(same, queries.concat());
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn tiny_or_inner_product_indexes_are_refused_for_the_gpu() {
        let dim = 768;
        let (centres, data) = clustered(dim, 5, 1, 9);
        let path = temp_index(&ivf_of(dim, &centres, &data, 1), "tiny");
        let graph = knn_graph_in(&templates(), dim, false).unwrap();
        let err = Retrieval::load_ext(
            &path,
            RetrievalBackend::GpuKnn,
            "cpu",
            &VcSessionOptions::default(),
            PrecisionMode::Fp32,
            Some(&graph),
            true,
        )
        .err()
        .unwrap();
        assert!(err.contains("at least 8"), "{err}");
        // Without the test override, a GPU kNN on the CPU rung is refused.
        let (centres, data) = clustered(dim, 200, 4, 9);
        let path2 = temp_index(&ivf_of(dim, &centres, &data, 1), "cpu-rung");
        let err = Retrieval::load_ext(
            &path2,
            RetrievalBackend::GpuKnn,
            "cpu",
            &VcSessionOptions::default(),
            PrecisionMode::Fp32,
            Some(&graph),
            false,
        )
        .err()
        .unwrap();
        assert!(err.contains("CPU"), "{err}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }

    /// P3B-G (review fix): a GPU kNN whose session landed on the CPU rung is refused, and a refused
    /// session leaves no `vc.knn` diagnostics row (the build used to record it before the check, so
    /// Settings would list a kNN session that does not exist). An accepted one records its outcome
    /// under the prefixed key, as `Retrieval::load` did before the session carried a component.
    #[test]
    fn a_refused_cpu_rung_leaves_no_knn_diagnostics_row() {
        use crate::services::diagnostics::ep_outcome;
        let dim = 768;
        let (centres, data) = clustered(dim, 200, 4, 9);
        let path = temp_index(&ivf_of(dim, &centres, &data, 1), "cpu-row");
        let graph = knn_graph_in(&templates(), dim, false).unwrap();
        let opts = VcSessionOptions { component_prefix: Some("test.retrieval.cpu_rung"), ..VcSessionOptions::default() };
        let key = "test.retrieval.cpu_rung.vc.knn";
        assert_eq!(knn_component(&opts), key);
        assert_eq!(knn_component(&VcSessionOptions::default()), "vc.knn");
        assert_eq!(knn_component(&VcSessionOptions { component_prefix: Some(""), ..VcSessionOptions::default() }), "vc.knn");
        let refused = Retrieval::load_ext(&path, RetrievalBackend::GpuKnn, "cpu", &opts, PrecisionMode::Fp32, Some(&graph), false).err();
        assert!(refused.is_some_and(|e| e.contains("CPU")));
        assert_eq!(ep_outcome(key), None, "a refused session has no diagnostics row");
        let r = Retrieval::load_ext(&path, RetrievalBackend::GpuKnn, "cpu", &opts, PrecisionMode::Fp32, Some(&graph), true).unwrap();
        assert_eq!(ep_outcome(key).map(|o| o.active), Some("cpu".to_string()), "an accepted session records its outcome");
        assert_eq!(r.outcome().map(|o| o.active.clone()), Some("cpu".to_string()));
        assert!(!r.vram_measured(), "nothing is measured on the CPU rung");
        let _ = std::fs::remove_file(&path);
    }

    /// P3B-G (review fix): the kNN books the DXGI delta only when its measurement was exclusive;
    /// an overlapped one (the parallel cold start builds the other branches beside it) or an
    /// unmeasurable one holds the computed size, index and norms plus the session's fixed cost.
    #[test]
    fn knn_footprint_is_measured_only_when_exclusive() {
        let exclusive = Some(VramDelta { mb: 106.0, exclusive: true });
        let overlapped = Some(VramDelta { mb: 640.0, exclusive: false });
        assert_eq!(knn_footprint(88.0, "cuda", exclusive), KnnFootprint { mb: 106.0, measured: true });
        // 640 MB of a neighbour's allocations are not the kNN's.
        assert_eq!(knn_footprint(88.0, "cuda", overlapped), KnnFootprint { mb: 106.0, measured: false });
        assert_eq!(knn_footprint(44.0, "dml", overlapped), KnnFootprint { mb: 68.0, measured: false });
        assert_eq!(knn_footprint(44.0, "cuda", None), KnnFootprint { mb: 62.0, measured: false });
        assert_eq!(knn_footprint(23.5, "cpu", None), KnnFootprint { mb: 23.5, measured: false });
        // The computed size reproduces the lone measurements it was calibrated on (README: 106 / 62 / 68 MB).
        for (index_mb, ep, measured_mb) in [(88.0, "cuda", 106.0), (44.0, "cuda", 62.0), (44.0, "dml", 68.0), (23.5, "cuda", 42.0), (11.7, "cuda", 30.0)] {
            let computed = knn_footprint(index_mb, ep, None).mb;
            assert!((computed - measured_mb).abs() < 1.0, "{ep} {index_mb}: {computed} vs {measured_mb}");
        }
    }

    /// Plan verification: the pooled parallel CPU search equals the serial one, bit for bit.
    #[test]
    fn cpu_pool_matches_serial_search() {
        let dim = 768;
        let (centres, data) = clustered(dim, 4_000, 64, 21);
        let bytes = ivf_of(dim, &centres, &data, 1);
        let path = temp_index(&bytes, "pool");
        let serial_index = FaissIndex::from_bytes(&bytes).unwrap();
        for threads in [1usize, 2, 4] {
            let opts = VcSessionOptions {
                intra_threads: Some(threads),
                ..VcSessionOptions::default()
            };
            let mut r = Retrieval::load(&path, RetrievalBackend::CpuIvf, "cpu", &opts, PrecisionMode::Auto).unwrap();
            assert_eq!(r.threads(), threads);
            for frames in [1usize, 3, 7, 16] {
                let (_, q) = clustered(dim, frames, 64, 1_000 + frames as u64);
                let mut want: Vec<f32> = q.concat();
                serial_index.blend_inplace(&mut want, frames, 0.7);
                let mut got: Vec<f32> = q.concat();
                r.blend(&mut got, frames, 0.7).unwrap();
                assert_eq!(got, want, "{threads} threads, {frames} frames");
            }
        }
        // The Streamer preset searches serially.
        let opts = VcSessionOptions {
            streamer_profile: true,
            intra_threads: Some(4),
            ..VcSessionOptions::default()
        };
        let r = Retrieval::load(&path, RetrievalBackend::CpuIvf, "cpu", &opts, PrecisionMode::Auto).unwrap();
        assert_eq!(r.threads(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// Dev rig: the kNN graph on the GPU (fp32 and fp16 index storage) against brute force, and
    /// CPU IVF (serial vs pooled) vs GPU timings per tick.
    /// `cargo test --release gpu_knn_on_the_gpu -- --ignored --nocapture` (SIGIL_AB_EP=cuda|dml)
    #[test]
    #[ignore = "dev rig: needs a GPU"]
    fn gpu_knn_on_the_gpu() {
        let ep = std::env::var("SIGIL_AB_EP").unwrap_or_else(|_| "cuda".to_string());
        let dim = 768;
        let (centres, data) = clustered(dim, 30_000, 769, 3);
        let path = temp_index(&ivf_of(dim, &centres, &data, 1), "gpu");
        let (_, queries) = clustered(dim, 6, 769, 44);
        // The exact blends, once (a debug-build brute force over 30k rows is slow).
        let wants: Vec<Vec<f64>> = queries.iter().map(|q| brute_force(&data, q, 0.75).0).collect();
        for precision in [PrecisionMode::Fp32, PrecisionMode::Fp16] {
            let graph = knn_graph_in(&templates(), dim, precision == PrecisionMode::Fp16).unwrap();
            let mut r = match Retrieval::load_ext(
                &path,
                RetrievalBackend::GpuKnn,
                &ep,
                &VcSessionOptions::default(),
                precision,
                Some(&graph),
                false,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{ep} {precision:?}: {e}");
                    continue;
                }
            };
            r.prepare(6).unwrap();
            let mut worst = 0.0f64;
            let mut times = Vec::new();
            for _ in 0..50 {
                let mut feats: Vec<f32> = queries.concat();
                let t0 = std::time::Instant::now();
                r.blend(&mut feats, 6, 0.75).unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                for (t, want) in wants.iter().enumerate() {
                    for (c, w) in want.iter().enumerate() {
                        worst = worst.max((f64::from(feats[t * dim + c]) - w).abs());
                    }
                }
            }
            times.sort_by(f64::total_cmp);
            println!(
                "gpu kNN {ep} {precision:?} ({:?}): N 30000, T 6, max |err| {worst:.2e}, p50 {:.3} ms, p95 {:.3} ms, VRAM {:.1} MB ({}), index {:.1} MB",
                r.outcome().map(|o| o.active.clone()),
                times[25],
                times[47],
                r.vram_mb(),
                if r.vram_measured() { "measured" } else { "computed" },
                r.index_mb()
            );
        }
        for threads in [1usize, 2, 4] {
            let opts = VcSessionOptions {
                intra_threads: Some(threads),
                ..VcSessionOptions::default()
            };
            let mut r = Retrieval::load(&path, RetrievalBackend::CpuIvf, "cpu", &opts, PrecisionMode::Auto).unwrap();
            let mut times = Vec::new();
            for _ in 0..50 {
                let mut feats: Vec<f32> = queries.concat();
                let t0 = std::time::Instant::now();
                r.blend(&mut feats, 6, 0.75).unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("cpu IVF nprobe 1, {threads} thread(s): p50 {:.3} ms, p95 {:.3} ms", times[25], times[47]);
        }
        let _ = std::fs::remove_file(&path);
    }

    /// P3B-G (dev rig, a GPU): fp16 index storage keeps the exact blend. On 8 000 clustered rows the
    /// fp16 coarse top-64 holds every true top-8 neighbour (re-ranked in fp32), so the fp16 blend
    /// differs from the exact f64 one only by the stored rows' fp16 rounding (well under 1e-2 of the
    /// features' scale), never by a neighbour swap (an error of the order of the features
    /// themselves); fp32 storage is exact to float rounding.
    /// `SIGIL_AB_EP=dml` checks DirectML instead of CUDA.
    #[test]
    #[ignore = "dev rig: needs a GPU"]
    fn gpu_knn_fp16_storage_keeps_the_exact_blend() {
        let ep = std::env::var("SIGIL_AB_EP").unwrap_or_else(|_| "cuda".to_string());
        let dim = 768;
        let (centres, data) = clustered(dim, 8_000, 769, 5);
        let path = temp_index(&ivf_of(dim, &centres, &data, 1), "gpu-fp16");
        let (_, queries) = clustered(dim, 6, 769, 46);
        let wants: Vec<Vec<f64>> = queries.iter().map(|q| brute_force(&data, q, 0.75).0).collect();
        let scale = data.iter().flatten().fold(0f64, |m, v| m.max(f64::from(v.abs())));
        let mut worst = Vec::new();
        for precision in [PrecisionMode::Fp32, PrecisionMode::Fp16] {
            let graph = knn_graph_in(&templates(), dim, precision == PrecisionMode::Fp16).unwrap();
            let mut r = match Retrieval::load_ext(&path, RetrievalBackend::GpuKnn, &ep, &VcSessionOptions::default(), precision, Some(&graph), false) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{ep} {precision:?}: {e}; skipped");
                    let _ = std::fs::remove_file(&path);
                    return;
                }
            };
            let mut feats: Vec<f32> = queries.concat();
            r.blend(&mut feats, 6, 0.75).unwrap();
            let w = wants
                .iter()
                .enumerate()
                .flat_map(|(t, want)| want.iter().enumerate().map(move |(c, w)| (t, c, *w)))
                .fold(0f64, |m, (t, c, w)| m.max((f64::from(feats[t * dim + c]) - w).abs()));
            eprintln!("{ep} {precision:?}: max |blend - exact| {w:.3e} (feature scale {scale:.2}, VRAM {:.1} MB, index {:.1} MB)", r.vram_mb(), r.index_mb());
            worst.push(w);
        }
        let _ = std::fs::remove_file(&path);
        assert!(worst[0] <= 1e-4 * scale.max(1.0), "fp32 storage is not exact: {:.3e}", worst[0]);
        assert!(worst[1] <= 1e-2 * scale.max(1.0), "fp16 storage changed the blend beyond its rounding: {:.3e}", worst[1]);
    }

    /// The CPU search allocates nothing per tick on the voice worker's thread, serial or pooled.
    #[test]
    fn cpu_search_is_allocation_free_per_tick() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let dim = 768;
        let (centres, data) = clustered(dim, 3_000, 48, 2);
        let path = temp_index(&ivf_of(dim, &centres, &data, 1), "alloc");
        for threads in [1usize, 4] {
            let opts = VcSessionOptions {
                intra_threads: Some(threads),
                ..VcSessionOptions::default()
            };
            let mut r = Retrieval::load(&path, RetrievalBackend::CpuIvf, "cpu", &opts, PrecisionMode::Auto).unwrap();
            let (_, q) = clustered(dim, 8, 48, 5);
            let mut feats: Vec<f32> = q.concat();
            r.blend(&mut feats, 8, 0.5).unwrap(); // warm the scratch
            let mut feats: Vec<f32> = q.concat();
            let (res, n) = allocations_during(|| r.blend(&mut feats, 8, 0.5));
            res.unwrap();
            assert_eq!(n, 0, "{threads} thread(s)");
        }
        let _ = std::fs::remove_file(&path);
    }
}
