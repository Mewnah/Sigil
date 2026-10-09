//! Voice-engine telemetry: per-tick stage spans, load spans, CPU-GPU synchronisation counts, GPU
//! time per session plan and VRAM marks (P3B; the attribution and VRAM probes of the benchmark).
//!
//! # Frozen by P3B-S0 (signatures; P3B-A implements, P3B-G reports GPU time)
//! The API is allocation-free on the tick path and costs one relaxed atomic load while recording is
//! off (the default: only a benchmark switches it on, [`RecordingGuard`]):
//! - [`stage_span`]`(Stage) -> StageSpan`: a guard around one tick stage (content, retrieval, F0,
//!   `enc`, `flowdec`, post). Call sites: `vc::engine::tick` (and `convert_inner` for the post stage).
//! - [`load_span`]`(LoadStage) -> LoadSpan`: a guard around one load stage (NVIDIA preload and
//!   canary, pitch plain session and plan, weights read, fp16 image, `enc` / `flowdec` / content
//!   session init, first runs and graph capture, `warm_once`, re-plan with its reason, harness
//!   warm-up). The load spans of the non-plan GPU sessions (ContentVec, plain RMVPE, GPU kNN) are
//!   where the benchmark attributes their VRAM ([`LoadStage::component`]; with
//!   [`set_vram_probe`] on, each such span records the process VRAM it added as a [`VramMark`]).
//! - [`record_sync`]`(SyncKind)`: one host wait on the GPU (every `DeviceIo` synchronisation in
//!   `gpu::plan`, and every eager GPU session run of the non-plan sessions).
//! - [`record_gpu_time`]`(component, ms)`: device time of one plan run (CUDA events, P3B-G), only
//!   when [`gpu_timing_enabled`].
//! - [`vram_mark`]`(component, mb)`: a measured VRAM hold of a component (the ledger of the
//!   benchmark's VRAM report).
//! - [`gpu_timing_enabled`]`() -> bool`.
//!
//! # The recorder (P3B-A)
//! One process-wide [`Recorder`] (lock-free accumulators):
//! - Tick side (allocation-, lock- and log-free): per [`Stage`] the summed span time and span count,
//!   per [`SyncKind`] a counter, and a fixed table of GPU times keyed by component. The benchmark
//!   reads and clears them between ticks ([`take_tick`] → [`TickTelemetry`]), on the worker thread
//!   right after each model call (outside the model's timed region).
//! - Load side (not on the tick path): every [`LoadSpan`] appends a [`LoadEvent`] (stage, thread,
//!   start / end on the process clock [`now_ns`], optional VRAM before / after) to a bounded log
//!   under a mutex; [`load_mark`] / [`load_events_since`] read a window of it. VRAM marks are kept
//!   the same way ([`vram_marks_since`]).
//!
//! - [`arena_mark`] / [`arena_mark_session`] (P3B-A additions): a session's ONNX Runtime arena
//!   after warm-up (in use / reserved), for the owners of sessions that have a BFC arena (CUDA):
//!   `arena_mark_session(component, &session, device)` reads it through `CreateAllocator` +
//!   `AllocatorGetStats` ([`session_arena_stats`]) while a benchmark records. DirectML sessions have
//!   no arena (the benchmark reports their DXGI deltas only).
//!
//! Recording is reference counted ([`RecordingGuard`]): benchmarks that overlap (tests, an in-app
//! run beside the autotuner) keep it on until the last one ends. Everything recorded is
//! process-wide: a voice changer running in the same process during an in-app benchmark adds its
//! ticks (the benchmark's report already flags `voice_changer_running`).

// Frozen API: `record_gpu_time` / `gpu_timing_enabled` are P3B-G's (CUDA events), and a few
// accessors exist for the benchmark's tests and reports only.
#![allow(dead_code)]

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

use serde::Serialize;

/// One stage of a template-engine tick (`vc::engine::tick`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Content,
    Retrieval,
    F0,
    Enc,
    Flowdec,
    Post,
}

impl Stage {
    pub const ALL: [Stage; 6] = [Stage::Content, Stage::Retrieval, Stage::F0, Stage::Enc, Stage::Flowdec, Stage::Post];

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Content => "content",
            Stage::Retrieval => "retrieval",
            Stage::F0 => "f0",
            Stage::Enc => "enc",
            Stage::Flowdec => "flowdec",
            Stage::Post => "post",
        }
    }

    /// Index into [`TickTelemetry::stage_ms`].
    pub fn index(self) -> usize {
        match self {
            Stage::Content => 0,
            Stage::Retrieval => 1,
            Stage::F0 => 2,
            Stage::Enc => 3,
            Stage::Flowdec => 4,
            Stage::Post => 5,
        }
    }
}

/// Why the engine re-planned its sessions (`SigilRvcEngine::replan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReplanReason {
    /// `set_lookahead_ms` changed the shapes.
    Lookahead,
    /// The stream harness's `configure_stream` (its SOLA tail and extra context).
    ConfigureStream,
    /// Anything else (tests, the benchmark).
    Other,
}

impl ReplanReason {
    pub fn as_str(self) -> &'static str {
        match self {
            ReplanReason::Lookahead => "lookahead",
            ReplanReason::ConfigureStream => "configure_stream",
            ReplanReason::Other => "other",
        }
    }
}

/// One stage of loading a voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadStage {
    /// The NVIDIA runtime preload and the EP canary (`ort_session` GPU preflight).
    PreloadCanary,
    /// The pitch extractor's plain session (RMVPE).
    PitchPlain,
    /// RMVPE's fixed-shape session plan.
    PitchPlan,
    /// Reading the slot's `weights.f32.bin`.
    WeightsRead,
    /// Converting the weights image to fp16.
    F16Image,
    /// The `enc` session plan.
    EncSessionInit,
    /// The `flowdec` session plan.
    FlowdecSessionInit,
    /// The content encoder's session (ContentVec / SPIN / HuBERT).
    ContentSessionInit,
    /// The GPU kNN retrieval session.
    KnnSessionInit,
    /// A session plan's warm-up runs and CUDA graph capture.
    FirstRunsCapture,
    /// The engine's load-time test conversion (`warm_once`).
    WarmOnce,
    /// A re-plan of the generator's shapes.
    Replan(ReplanReason),
    /// The stream harness's warm-up at the live shape (`VcStreamHarness::with_context`).
    HarnessWarmup,
    /// Creating (retaining) the device's CUDA primary context (P3B-A: the benchmark retains it
    /// before the engine load to measure its VRAM once, and times it as part of the load).
    CudaContext,
}

impl LoadStage {
    /// Every stage, re-plans by reason (report order).
    pub const ALL: [LoadStage; 16] = [
        LoadStage::CudaContext,
        LoadStage::PreloadCanary,
        LoadStage::PitchPlain,
        LoadStage::PitchPlan,
        LoadStage::WeightsRead,
        LoadStage::F16Image,
        LoadStage::EncSessionInit,
        LoadStage::FlowdecSessionInit,
        LoadStage::ContentSessionInit,
        LoadStage::KnnSessionInit,
        LoadStage::FirstRunsCapture,
        LoadStage::WarmOnce,
        LoadStage::Replan(ReplanReason::Lookahead),
        LoadStage::Replan(ReplanReason::ConfigureStream),
        LoadStage::Replan(ReplanReason::Other),
        LoadStage::HarnessWarmup,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LoadStage::PreloadCanary => "preload_canary",
            LoadStage::PitchPlain => "pitch_plain",
            LoadStage::PitchPlan => "pitch_plan",
            LoadStage::WeightsRead => "weights_read",
            LoadStage::F16Image => "f16_image",
            LoadStage::EncSessionInit => "enc_session_init",
            LoadStage::FlowdecSessionInit => "flowdec_session_init",
            LoadStage::ContentSessionInit => "content_session_init",
            LoadStage::KnnSessionInit => "knn_session_init",
            LoadStage::FirstRunsCapture => "first_runs_capture",
            LoadStage::WarmOnce => "warm_once",
            LoadStage::Replan(_) => "replan",
            LoadStage::HarnessWarmup => "harness_warmup",
            LoadStage::CudaContext => "cuda_context",
        }
    }

    /// The stage's report key: [`as_str`](Self::as_str), with a re-plan's reason
    /// (`replan_lookahead`, `replan_configure_stream`, `replan_other`).
    pub fn key(self) -> &'static str {
        match self {
            LoadStage::Replan(ReplanReason::Lookahead) => "replan_lookahead",
            LoadStage::Replan(ReplanReason::ConfigureStream) => "replan_configure_stream",
            LoadStage::Replan(ReplanReason::Other) => "replan_other",
            other => other.as_str(),
        }
    }

    /// The session component whose VRAM this stage allocates (`None` for stages that hold none).
    pub fn component(self) -> Option<&'static str> {
        match self {
            LoadStage::PitchPlain | LoadStage::PitchPlan => Some("vc.pitch"),
            LoadStage::EncSessionInit => Some("vc.enc"),
            LoadStage::FlowdecSessionInit => Some("vc.flowdec"),
            LoadStage::ContentSessionInit => Some("vc.content"),
            LoadStage::KnnSessionInit => Some("vc.knn"),
            LoadStage::CudaContext => Some("cuda.context"),
            _ => None,
        }
    }
}

/// One host wait on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncKind {
    /// `DeviceIo`: waiting for the input uploads before an eager (ungraphed) ORT run.
    UploadWait,
    /// `DeviceIo`: the end-of-run downloads and stream synchronisation.
    RunEnd,
    /// `DeviceIo`: a host mirror refreshed after a device row copy.
    MirrorRefresh,
    /// `DeviceIo`: a strided device-to-host read of rows.
    RowRead,
    /// An eager GPU session run with host tensors (ContentVec, plain RMVPE, GPU kNN): ONNX Runtime
    /// copies and synchronises inside the run. Also recorded when such a session fell back to the
    /// CPU rung; the benchmark attributes it by the component's EP.
    EagerRun,
}

impl SyncKind {
    pub const ALL: [SyncKind; 5] = [SyncKind::UploadWait, SyncKind::RunEnd, SyncKind::MirrorRefresh, SyncKind::RowRead, SyncKind::EagerRun];

    pub fn as_str(self) -> &'static str {
        match self {
            SyncKind::UploadWait => "upload_wait",
            SyncKind::RunEnd => "run_end",
            SyncKind::MirrorRefresh => "mirror_refresh",
            SyncKind::RowRead => "row_read",
            SyncKind::EagerRun => "eager_run",
        }
    }

    /// Index into [`TickTelemetry::syncs`].
    pub fn index(self) -> usize {
        match self {
            SyncKind::UploadWait => 0,
            SyncKind::RunEnd => 1,
            SyncKind::MirrorRefresh => 2,
            SyncKind::RowRead => 3,
            SyncKind::EagerRun => 4,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------------------------

/// Components with a GPU-time slot (more distinct components than this are dropped and counted).
pub const GPU_SLOTS: usize = 8;
/// Load events kept (a voice load records about 20; the log is cleared by nobody, so a long-lived
/// process that benchmarks many voices stops logging after this many and counts the rest).
pub const LOAD_LOG_CAPACITY: usize = 16_384;

/// Device time of one component's plan runs within one tick.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GpuTime {
    pub component: &'static str,
    pub ms: f64,
    pub runs: u32,
}

/// What one tick recorded ([`take_tick`]): stage times (ms) and span counts by [`Stage::index`],
/// each stage's interval on the process clock (the first span's start to the last span's end; `0`
/// when it did not run), host waits by [`SyncKind::index`], GPU time per component.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickTelemetry {
    pub stage_ms: [f64; 6],
    pub stage_spans: [u32; 6],
    pub stage_start_ns: [u64; 6],
    pub stage_end_ns: [u64; 6],
    pub syncs: [u64; 5],
    pub gpu: [Option<GpuTime>; GPU_SLOTS],
}

impl TickTelemetry {
    /// Time of one stage (ms).
    pub fn stage(&self, s: Stage) -> f64 {
        self.stage_ms[s.index()]
    }

    /// Whether the stage ran (at least one span).
    pub fn ran(&self, s: Stage) -> bool {
        self.stage_spans[s.index()] > 0
    }

    /// Sum of every stage (ms).
    pub fn stages_total_ms(&self) -> f64 {
        self.stage_ms.iter().sum()
    }

    /// Wall time covered by at least one stage (ms): the union of the stages' intervals, so stages
    /// that ran in parallel (F0 beside the content encoder) count once. A stage with several spans
    /// counts from its first start to its last end (any gap between its spans included).
    pub fn stages_union_ms(&self) -> f64 {
        let mut iv = [(0u64, 0u64); 6];
        let mut n = 0;
        for i in 0..6 {
            if self.stage_spans[i] > 0 && self.stage_end_ns[i] > self.stage_start_ns[i] {
                iv[n] = (self.stage_start_ns[i], self.stage_end_ns[i]);
                n += 1;
            }
        }
        let iv = &mut iv[..n];
        iv.sort_unstable();
        let mut total = 0u64;
        let mut cur: Option<(u64, u64)> = None;
        for &(a, b) in iv.iter() {
            cur = match cur {
                Some((s, e)) if a <= e => Some((s, e.max(b))),
                Some((s, e)) => {
                    total += e - s;
                    Some((a, b))
                }
                None => Some((a, b)),
            };
        }
        if let Some((s, e)) = cur {
            total += e - s;
        }
        total as f64 / 1e6
    }

    /// Host waits on the GPU of every kind.
    pub fn syncs_total(&self) -> u64 {
        self.syncs.iter().sum()
    }

    /// Whether anything was recorded.
    pub fn is_empty(&self) -> bool {
        self.stage_spans.iter().all(|&n| n == 0) && self.syncs_total() == 0 && self.gpu.iter().all(Option::is_none)
    }
}

/// One finished load span.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LoadEvent {
    #[serde(serialize_with = "ser_stage_key")]
    pub stage: LoadStage,
    /// Small process-local id of the thread that ran it (1, 2, ...).
    pub thread: u64,
    /// On the process clock ([`now_ns`]).
    pub start_ns: u64,
    pub end_ns: u64,
    /// This process's VRAM (MiB, every hardware adapter) at the start and end, when
    /// [`set_vram_probe`] was on.
    pub vram_before_mb: Option<u64>,
    pub vram_after_mb: Option<u64>,
}

fn ser_stage_key<S: serde::Serializer>(stage: &LoadStage, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(stage.key())
}

impl LoadEvent {
    pub fn duration_ms(&self) -> f64 {
        self.end_ns.saturating_sub(self.start_ns) as f64 / 1e6
    }
}

/// A measured VRAM hold of a component.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct VramMark {
    pub component: &'static str,
    /// MiB; a load span's mark is the signed process delta over the span.
    pub mb: f64,
    pub at_ns: u64,
    /// `span` (measured by a load span with [`set_vram_probe`] on) or `explicit` ([`vram_mark`]).
    pub source: &'static str,
}

/// The lock-free accumulators behind the free functions (one process-wide instance, [`global`];
/// tests build their own).
pub struct Recorder {
    recording: AtomicUsize,
    gpu_timing: AtomicBool,
    vram_probe: AtomicBool,
    stage_ns: [AtomicU64; 6],
    stage_n: [AtomicU32; 6],
    /// Earliest span start / latest span end per stage since the last read (process clock ns;
    /// `u64::MAX` / `0` when none).
    stage_first: [AtomicU64; 6],
    stage_last: [AtomicU64; 6],
    syncs: [AtomicU64; 5],
    /// GPU slots: key = the component string's data pointer (literals are unique per call site;
    /// slots with equal text are merged when read), its length, summed ns and runs.
    gpu_key: [AtomicUsize; GPU_SLOTS],
    gpu_len: [AtomicUsize; GPU_SLOTS],
    gpu_ns: [AtomicU64; GPU_SLOTS],
    gpu_n: [AtomicU32; GPU_SLOTS],
    gpu_dropped: AtomicU64,
    load_log: Mutex<Vec<LoadEvent>>,
    load_dropped: AtomicU64,
    vram_marks: Mutex<Vec<VramMark>>,
}

impl Recorder {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Recorder {
        Recorder {
            recording: AtomicUsize::new(0),
            gpu_timing: AtomicBool::new(false),
            vram_probe: AtomicBool::new(false),
            stage_ns: [const { AtomicU64::new(0) }; 6],
            stage_n: [const { AtomicU32::new(0) }; 6],
            stage_first: [const { AtomicU64::new(u64::MAX) }; 6],
            stage_last: [const { AtomicU64::new(0) }; 6],
            syncs: [const { AtomicU64::new(0) }; 5],
            gpu_key: [const { AtomicUsize::new(0) }; GPU_SLOTS],
            gpu_len: [const { AtomicUsize::new(0) }; GPU_SLOTS],
            gpu_ns: [const { AtomicU64::new(0) }; GPU_SLOTS],
            gpu_n: [const { AtomicU32::new(0) }; GPU_SLOTS],
            gpu_dropped: AtomicU64::new(0),
            load_log: Mutex::new(Vec::new()),
            load_dropped: AtomicU64::new(0),
            vram_marks: Mutex::new(Vec::new()),
        }
    }

    /// Whether anything records.
    #[inline]
    pub fn recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed) > 0
    }

    /// Start recording until the guard drops (nested guards keep it on until the last drops).
    pub fn record(&'static self) -> RecordingGuard {
        // Fix the process clock's origin before any span can start (a span that started before
        // the origin would read as time 0).
        let _ = epoch();
        self.recording.fetch_add(1, Ordering::SeqCst);
        RecordingGuard { recorder: self }
    }

    #[inline]
    pub fn stage_span(&'static self, stage: Stage) -> StageSpan {
        StageSpan {
            stage,
            start: self.recording().then(Instant::now),
            recorder: self,
        }
    }

    pub fn load_span(&'static self, stage: LoadStage) -> LoadSpan {
        if !self.recording() {
            return LoadSpan {
                stage,
                start: None,
                vram_before: None,
                recorder: self,
            };
        }
        let vram_before = if self.vram_probe.load(Ordering::Relaxed) { process_vram_now() } else { None };
        LoadSpan {
            stage,
            start: Some(Instant::now()),
            vram_before,
            recorder: self,
        }
    }

    #[inline]
    pub fn record_sync(&self, kind: SyncKind) {
        if self.recording() {
            self.syncs[kind.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_gpu_time(&self, component: &'static str, ms: f64) {
        if !self.recording() || !ms.is_finite() || ms < 0.0 {
            return;
        }
        let key = component.as_ptr() as usize;
        let ns = (ms * 1e6) as u64;
        for i in 0..GPU_SLOTS {
            let k = self.gpu_key[i].load(Ordering::Acquire);
            let mine = if k == key {
                true
            } else if k == 0 {
                match self.gpu_key[i].compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        self.gpu_len[i].store(component.len(), Ordering::Release);
                        true
                    }
                    Err(other) => other == key,
                }
            } else {
                false
            };
            if mine {
                self.gpu_ns[i].fetch_add(ns, Ordering::Relaxed);
                self.gpu_n[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.gpu_dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn vram_mark(&self, component: &'static str, mb: f64, source: &'static str) {
        if !self.recording() || !mb.is_finite() {
            return;
        }
        let mut marks = self.vram_marks.lock().unwrap_or_else(PoisonError::into_inner);
        if marks.len() < LOAD_LOG_CAPACITY {
            marks.push(VramMark {
                component,
                mb,
                at_ns: now_ns(),
                source,
            });
        }
    }

    /// Read and clear the tick accumulators. Allocation-free (call it between ticks).
    pub fn take_tick(&self) -> TickTelemetry {
        let mut t = TickTelemetry::default();
        for i in 0..6 {
            t.stage_ms[i] = self.stage_ns[i].swap(0, Ordering::Relaxed) as f64 / 1e6;
            t.stage_spans[i] = self.stage_n[i].swap(0, Ordering::Relaxed);
            let first = self.stage_first[i].swap(u64::MAX, Ordering::Relaxed);
            let last = self.stage_last[i].swap(0, Ordering::Relaxed);
            if first != u64::MAX && last >= first {
                t.stage_start_ns[i] = first;
                t.stage_end_ns[i] = last;
            }
        }
        for i in 0..5 {
            t.syncs[i] = self.syncs[i].swap(0, Ordering::Relaxed);
        }
        let mut out = 0;
        for i in 0..GPU_SLOTS {
            let key = self.gpu_key[i].load(Ordering::Acquire);
            let len = self.gpu_len[i].load(Ordering::Acquire);
            let runs = self.gpu_n[i].swap(0, Ordering::Relaxed);
            let ns = self.gpu_ns[i].swap(0, Ordering::Relaxed);
            if key == 0 || len == 0 || runs == 0 {
                continue;
            }
            // SAFETY: `key` / `len` are the data pointer and length of a `&'static str` handed to
            // `record_gpu_time` (stored in that order: a slot is read only once its length is set),
            // so they describe valid, immutable UTF-8 for the life of the process.
            let component: &'static str = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(key as *const u8, len)) };
            let ms = ns as f64 / 1e6;
            // Equal text from two call sites (two literals) merges into one entry.
            if let Some(e) = t.gpu[..out].iter_mut().flatten().find(|e| e.component == component) {
                e.ms += ms;
                e.runs += runs;
            } else {
                t.gpu[out] = Some(GpuTime { component, ms, runs });
                out += 1;
            }
        }
        t
    }

    pub fn load_mark(&self) -> usize {
        self.load_log.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn load_events_since(&self, mark: usize) -> Vec<LoadEvent> {
        let log = self.load_log.lock().unwrap_or_else(PoisonError::into_inner);
        log.get(mark..).map(<[LoadEvent]>::to_vec).unwrap_or_default()
    }

    pub fn vram_mark_count(&self) -> usize {
        self.vram_marks.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn vram_marks_since(&self, mark: usize) -> Vec<VramMark> {
        let marks = self.vram_marks.lock().unwrap_or_else(PoisonError::into_inner);
        marks.get(mark..).map(<[VramMark]>::to_vec).unwrap_or_default()
    }

    /// Load events and GPU-time entries that did not fit.
    pub fn dropped(&self) -> (u64, u64) {
        (self.load_dropped.load(Ordering::Relaxed), self.gpu_dropped.load(Ordering::Relaxed))
    }

    fn finish_load(&self, span: &LoadSpan) {
        let Some(start) = span.start else {
            return;
        };
        let end = Instant::now();
        let vram_after = if span.vram_before.is_some() { process_vram_now() } else { None };
        let event = LoadEvent {
            stage: span.stage,
            thread: thread_token(),
            start_ns: ns_since_epoch(start),
            end_ns: ns_since_epoch(end),
            vram_before_mb: span.vram_before,
            vram_after_mb: vram_after,
        };
        if let (Some(component), Some(a), Some(b)) = (span.stage.component(), span.vram_before, vram_after) {
            self.vram_mark(component, b as f64 - a as f64, "span");
        }
        let mut log = self.load_log.lock().unwrap_or_else(PoisonError::into_inner);
        if log.len() < LOAD_LOG_CAPACITY {
            log.push(event);
        } else {
            self.load_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The process-wide recorder.
pub fn global() -> &'static Recorder {
    static GLOBAL: Recorder = Recorder::new();
    &GLOBAL
}

/// Keeps recording on until dropped ([`Recorder::record`]).
#[must_use = "recording stops when the guard drops"]
pub struct RecordingGuard {
    recorder: &'static Recorder,
}

impl Drop for RecordingGuard {
    fn drop(&mut self) {
        self.recorder.recording.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The process clock origin (the first use).
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn ns_since_epoch(t: Instant) -> u64 {
    t.checked_duration_since(epoch()).map_or(0, |d| d.as_nanos() as u64)
}

/// Nanoseconds on the process clock of [`LoadEvent`]s (monotonic, from the first telemetry use).
pub fn now_ns() -> u64 {
    ns_since_epoch(Instant::now())
}

/// A small id of the calling thread (1, 2, ... in first-use order).
fn thread_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static TOKEN: Cell<u64> = const { Cell::new(0) };
    }
    TOKEN
        .try_with(|t| {
            if t.get() == 0 {
                t.set(NEXT.fetch_add(1, Ordering::Relaxed));
            }
            t.get()
        })
        .unwrap_or(0)
}

/// This process's dedicated VRAM over every hardware DXGI adapter (MiB), `None` when DXGI reports
/// nothing.
fn process_vram_now() -> Option<u64> {
    let mut any = false;
    let mut total = 0u64;
    for a in crate::services::diagnostics::gpu_mem::adapters().iter().filter(|a| !a.software) {
        if let Some(v) = a.vram {
            any = true;
            total += v.process_usage_mb;
        }
    }
    any.then_some(total)
}

/// Guard of one tick stage ([`stage_span`]); the stage ends when it drops.
#[must_use = "the span ends when the guard drops"]
pub struct StageSpan {
    stage: Stage,
    start: Option<Instant>,
    recorder: &'static Recorder,
}

impl StageSpan {
    pub fn stage(&self) -> Stage {
        self.stage
    }
}

impl Drop for StageSpan {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let i = self.stage.index();
            let end = Instant::now();
            let r = self.recorder;
            r.stage_ns[i].fetch_add(end.saturating_duration_since(start).as_nanos() as u64, Ordering::Relaxed);
            r.stage_n[i].fetch_add(1, Ordering::Relaxed);
            r.stage_first[i].fetch_min(ns_since_epoch(start), Ordering::Relaxed);
            r.stage_last[i].fetch_max(ns_since_epoch(end), Ordering::Relaxed);
        }
    }
}

/// Guard of one load stage ([`load_span`]); the stage ends when it drops.
#[must_use = "the span ends when the guard drops"]
pub struct LoadSpan {
    stage: LoadStage,
    start: Option<Instant>,
    vram_before: Option<u64>,
    recorder: &'static Recorder,
}

impl LoadSpan {
    pub fn stage(&self) -> LoadStage {
        self.stage
    }
}

impl Drop for LoadSpan {
    #[inline]
    fn drop(&mut self) {
        if self.start.is_some() {
            self.recorder.finish_load(self);
        }
    }
}

/// Start a tick stage.
#[inline]
pub fn stage_span(stage: Stage) -> StageSpan {
    global().stage_span(stage)
}

/// Start a load stage.
#[inline]
pub fn load_span(stage: LoadStage) -> LoadSpan {
    global().load_span(stage)
}

/// Count one host wait on the GPU.
#[inline]
pub fn record_sync(kind: SyncKind) {
    global().record_sync(kind);
}

/// Device time of one run of `component`'s session plan (ms; only measured while
/// [`gpu_timing_enabled`]).
#[inline]
pub fn record_gpu_time(component: &'static str, ms: f64) {
    global().record_gpu_time(component, ms);
}

/// A measured VRAM hold of `component` (MiB).
#[inline]
pub fn vram_mark(component: &'static str, mb: f64) {
    global().vram_mark(component, mb, "explicit");
}

/// A session's ONNX Runtime arena after its warm-up (MiB in use and reserved, from
/// `AllocatorGetStats`: `InUse` / `TotalAllocated`), as VRAM marks of sources `arena_in_use` and
/// `arena_reserved` (P3B-A API; the session owners call it where they hold the session).
#[inline]
pub fn arena_mark(component: &'static str, in_use_mb: f64, reserved_mb: f64) {
    let g = global();
    g.vram_mark(component, in_use_mb, "arena_in_use");
    g.vram_mark(component, reserved_mb, "arena_reserved");
}

/// Statistics of an ONNX Runtime allocator (`AllocatorGetStats`: `Limit`, `InUse`,
/// `TotalAllocated`, `MaxInUse`, `NumAllocs`, `NumReserves`, `NumArenaExtensions`,
/// `NumArenaShrinkages`, `MaxAllocSize`; values in bytes or counts, as text). Only a BFC arena
/// reports them (CUDA; DirectML has no arena): `None` otherwise.
///
/// # Safety
/// `allocator` must be a live ONNX Runtime allocator for the duration of the call.
pub unsafe fn allocator_stats_raw(allocator: *const ort::sys::OrtAllocator) -> Option<std::collections::BTreeMap<String, String>> {
    use std::ffi::{c_char, CStr};
    if allocator.is_null() {
        return None;
    }
    let api = ort::api();
    let mut kvps: *mut ort::sys::OrtKeyValuePairs = std::ptr::null_mut();
    // SAFETY: a live allocator (the caller's contract); ORT creates `kvps`, released below.
    let status = unsafe { (api.AllocatorGetStats)(allocator, &mut kvps) };
    // SAFETY: a status just returned by ORT.
    unsafe { ort::Error::result_from_status(status) }.ok()?;
    if kvps.is_null() {
        return None;
    }
    let mut keys: *const *const c_char = std::ptr::null();
    let mut values: *const *const c_char = std::ptr::null();
    let mut n: usize = 0;
    // SAFETY: `kvps` is live; the arrays it returns stay valid until it is released.
    unsafe { (api.GetKeyValuePairs)(kvps, &mut keys, &mut values, &mut n) };
    let mut out = std::collections::BTreeMap::new();
    for i in 0..n {
        // SAFETY: `n` entries of NUL-terminated strings owned by `kvps`.
        unsafe {
            let k = *keys.add(i);
            let v = *values.add(i);
            if !k.is_null() && !v.is_null() {
                out.insert(CStr::from_ptr(k).to_string_lossy().into_owned(), CStr::from_ptr(v).to_string_lossy().into_owned());
            }
        }
    }
    // SAFETY: created by AllocatorGetStats above, released once.
    unsafe { (api.ReleaseKeyValuePairs)(kvps) };
    (!out.is_empty()).then_some(out)
}

/// The BFC arena statistics of `session`'s CUDA allocator on `device` ([`allocator_stats_raw`]);
/// `None` for sessions without a CUDA arena (CPU, DirectML) or when ONNX Runtime refuses.
pub fn session_arena_stats(session: &ort::session::Session, device: i32) -> Option<std::collections::BTreeMap<String, String>> {
    use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
    use ort::AsPointer;
    let info = MemoryInfo::new(AllocationDevice::CUDA, device.max(0), AllocatorType::Device, MemoryType::Default).ok()?;
    let allocator = Allocator::new(session, info).ok()?;
    // SAFETY: `allocator` lives until the end of this function.
    unsafe { allocator_stats_raw(allocator.ptr()) }
}

/// Record `session`'s CUDA arena after its warm-up as [`arena_mark`]s of `component` (MiB in use
/// and reserved), while a benchmark records (P3B-A API: one call where a session owner holds a
/// warmed-up CUDA session, e.g. after a session plan's first runs:
/// `if on_cuda { telemetry::arena_mark_session(spec.component, &session, cuda_device) }`; a no-op
/// while nothing records and for sessions without a BFC arena). `component` need not be static
/// ([`intern_component`]).
pub fn arena_mark_session(component: &str, session: &ort::session::Session, device: i32) {
    if !recording() {
        return;
    }
    let Some(stats) = session_arena_stats(session, device) else {
        return;
    };
    const MIB: f64 = 1024.0 * 1024.0;
    let mb = |k: &str| stats.get(k).and_then(|v| v.trim().parse::<f64>().ok()).map(|b| b / MIB);
    if let (Some(in_use), Some(reserved)) = (mb("InUse"), mb("TotalAllocated")) {
        arena_mark(intern_component(component), in_use, reserved);
    }
}

/// Components that the recorder keeps as `&'static str` without allocating.
static KNOWN_COMPONENTS: [&str; 8] = ["vc.content", "vc.pitch", "vc.enc", "vc.flowdec", "vc.knn", "vc.f0", "vc.generator", "cuda.context"];
/// Most distinct non-standard component names kept (load-time only; beyond it they read `other`).
const INTERNED_CAPACITY: usize = 64;

/// A `&'static str` for a component name (the load-time marks keep static names): the standard
/// voice-engine components map to their literals, other names are leaked once each (at most
/// [`INTERNED_CAPACITY`], then `other`). Not for the tick path.
pub fn intern_component(component: &str) -> &'static str {
    if let Some(k) = KNOWN_COMPONENTS.iter().find(|k| **k == component) {
        return k;
    }
    static INTERNED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut v = INTERNED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(k) = v.iter().find(|k| **k == component) {
        return k;
    }
    if v.len() >= INTERNED_CAPACITY {
        return "other";
    }
    let leaked: &'static str = Box::leak(component.to_string().into_boxed_str());
    v.push(leaked);
    leaked
}

/// Whether GPU timing (CUDA events per plan run) is on: only during a benchmark that asked for it
/// ([`set_gpu_timing`], while recording).
#[inline]
pub fn gpu_timing_enabled() -> bool {
    let g = global();
    g.gpu_timing.load(Ordering::Relaxed) && g.recording()
}

// ---- control and read side (the benchmark) ----------------------------------------------------

/// Record until the guard drops (see [`RecordingGuard`]).
pub fn start_recording() -> RecordingGuard {
    global().record()
}

/// Whether the process-wide recorder records.
pub fn recording() -> bool {
    global().recording()
}

/// Ask session plans for GPU time per run (CUDA events; P3B-G) while recording.
pub fn set_gpu_timing(on: bool) {
    global().gpu_timing.store(on, Ordering::SeqCst);
}

/// Sample this process's VRAM at the start and end of every load span (about 1 ms per sample
/// through DXGI, so load timings carry it: switch it on for VRAM runs only).
pub fn set_vram_probe(on: bool) {
    global().vram_probe.store(on, Ordering::SeqCst);
}

/// Read and clear the tick accumulators ([`Recorder::take_tick`]).
pub fn take_tick() -> TickTelemetry {
    global().take_tick()
}

/// Position in the load log (pass it to [`load_events_since`]).
pub fn load_mark() -> usize {
    global().load_mark()
}

/// Load events logged since `mark`.
pub fn load_events_since(mark: usize) -> Vec<LoadEvent> {
    global().load_events_since(mark)
}

/// Position in the VRAM marks (pass it to [`vram_marks_since`]).
pub fn vram_mark_count() -> usize {
    global().vram_mark_count()
}

/// VRAM marks recorded since `mark`.
pub fn vram_marks_since(mark: usize) -> Vec<VramMark> {
    global().vram_marks_since(mark)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A private recorder (the global one is shared with every test that benchmarks).
    fn recorder() -> &'static Recorder {
        Box::leak(Box::new(Recorder::new()))
    }

    /// Off (the default): spans, syncs, GPU times and load spans record nothing.
    #[test]
    fn an_idle_recorder_records_nothing() {
        let r = recorder();
        assert!(!r.recording());
        for s in Stage::ALL {
            let span = r.stage_span(s);
            assert_eq!(span.stage(), s);
            assert!(!s.as_str().is_empty());
        }
        let span = r.load_span(LoadStage::Replan(ReplanReason::ConfigureStream));
        assert_eq!(span.stage().as_str(), "replan");
        assert_eq!(span.stage().key(), "replan_configure_stream");
        drop(span);
        r.record_sync(SyncKind::RunEnd);
        r.record_gpu_time("vc.enc", 1.0);
        r.vram_mark("vc.content", 172.0, "explicit");
        assert!(r.take_tick().is_empty());
        assert_eq!(r.load_mark(), 0);
        assert_eq!(r.vram_mark_count(), 0);
        assert_eq!(LoadStage::ContentSessionInit.component(), Some("vc.content"));
        assert!(!gpu_timing_enabled() || recording(), "GPU timing needs recording");
    }

    /// Recording: stage time and counts, sync counts and GPU time per component accumulate and are
    /// cleared by each read; nested guards keep recording on until the last drops.
    #[test]
    fn a_recording_recorder_accumulates_and_clears_per_tick() {
        let r = recorder();
        let outer = r.record();
        let inner = r.record();
        drop(inner);
        assert!(r.recording(), "the outer guard still records");
        {
            let _s = r.stage_span(Stage::Content);
            std::thread::sleep(Duration::from_millis(3));
        }
        {
            let _s = r.stage_span(Stage::Flowdec);
        }
        r.record_sync(SyncKind::RunEnd);
        r.record_sync(SyncKind::RunEnd);
        r.record_sync(SyncKind::EagerRun);
        r.record_gpu_time("vc.enc", 1.5);
        r.record_gpu_time("vc.enc", 0.5);
        r.record_gpu_time("vc.flowdec", 2.0);
        r.record_gpu_time("vc.bad", f64::NAN);
        let t = r.take_tick();
        assert!(t.stage(Stage::Content) >= 2.5, "{t:?}");
        assert!(t.ran(Stage::Flowdec) && !t.ran(Stage::F0));
        assert_eq!(t.stage_spans[Stage::Content.index()], 1);
        assert_eq!(t.syncs[SyncKind::RunEnd.index()], 2);
        assert_eq!(t.syncs_total(), 3);
        let gpu: Vec<GpuTime> = t.gpu.iter().flatten().copied().collect();
        assert_eq!(gpu.len(), 2, "{gpu:?}");
        let enc = gpu.iter().find(|g| g.component == "vc.enc").unwrap();
        assert!((enc.ms - 2.0).abs() < 1e-9 && enc.runs == 2, "{enc:?}");
        assert!(r.take_tick().is_empty(), "a read clears the tick");
        // Slots are reused across ticks.
        r.record_gpu_time("vc.enc", 1.0);
        assert_eq!(r.take_tick().gpu.iter().flatten().count(), 1);
        drop(outer);
        assert!(!r.recording());
    }

    /// Each stage's interval runs from its first span's start to its last span's end, on the
    /// process clock, and a read clears it.
    #[test]
    fn stage_intervals_cover_their_spans() {
        let r = recorder();
        let _g = r.record();
        let before = now_ns();
        for _ in 0..2 {
            let _f0 = r.stage_span(Stage::F0);
            std::thread::sleep(Duration::from_millis(3));
        }
        let after = now_ns();
        let t = r.take_tick();
        let (a, b) = (t.stage_start_ns[Stage::F0.index()], t.stage_end_ns[Stage::F0.index()]);
        assert!(before <= a && a < b && b <= after, "{before} {a} {b} {after}");
        assert!((b - a) as f64 / 1e6 >= t.stage(Stage::F0) - 1e-6, "the interval holds both spans");
        assert_eq!((t.stage_start_ns[Stage::Enc.index()], t.stage_end_ns[Stage::Enc.index()]), (0, 0), "a stage that did not run");
        assert_eq!(r.take_tick().stage_end_ns, [0; 6], "a read clears the intervals");
    }

    /// Stages that ran in parallel count once in the union; a sequential one adds its own time.
    #[test]
    fn parallel_stages_count_once_in_the_union() {
        let ms = 1_000_000u64;
        let mut t = TickTelemetry::default();
        // Content 0..4 ms, F0 1..5 ms (parallel), enc 6..7 ms (after a 1 ms gap).
        for (stage, a, b) in [(Stage::Content, 0, 4), (Stage::F0, 1, 5), (Stage::Enc, 6, 7)] {
            let i = stage.index();
            t.stage_spans[i] = 1;
            t.stage_ms[i] = (b - a) as f64;
            t.stage_start_ns[i] = 1_000 * ms + a * ms;
            t.stage_end_ns[i] = 1_000 * ms + b * ms;
        }
        assert_eq!(t.stages_total_ms(), 9.0);
        assert!((t.stages_union_ms() - 6.0).abs() < 1e-9, "{}", t.stages_union_ms());
        assert_eq!(TickTelemetry::default().stages_union_ms(), 0.0);
    }

    /// More components than slots: the extra ones are counted as dropped, never mixed in.
    #[test]
    fn gpu_slots_overflow_is_counted() {
        let r = recorder();
        let _g = r.record();
        let names: [&'static str; 10] = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"];
        for n in names {
            r.record_gpu_time(n, 1.0);
        }
        let t = r.take_tick();
        assert_eq!(t.gpu.iter().flatten().count(), GPU_SLOTS);
        assert_eq!(r.dropped().1, 2);
    }

    /// Load spans log their stage, thread and interval on the process clock, nested ones inside
    /// their parent; another thread gets another token.
    #[test]
    fn load_spans_log_nested_intervals_per_thread() {
        let r = recorder();
        let _g = r.record();
        let mark = r.load_mark();
        {
            let _outer = r.load_span(LoadStage::EncSessionInit);
            std::thread::sleep(Duration::from_millis(2));
            {
                let _inner = r.load_span(LoadStage::FirstRunsCapture);
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        std::thread::scope(|s| {
            s.spawn(|| {
                let _x = r.load_span(LoadStage::ContentSessionInit);
            });
        });
        let ev = r.load_events_since(mark);
        assert_eq!(ev.len(), 3, "{ev:?}");
        // The inner span finishes first.
        assert_eq!(ev[0].stage, LoadStage::FirstRunsCapture);
        assert_eq!(ev[1].stage, LoadStage::EncSessionInit);
        assert!(ev[1].start_ns <= ev[0].start_ns && ev[0].end_ns <= ev[1].end_ns, "{ev:?}");
        assert!(ev[1].duration_ms() >= 3.5, "{ev:?}");
        assert_eq!(ev[0].thread, ev[1].thread);
        assert_ne!(ev[2].thread, ev[1].thread);
        assert!(ev[0].vram_before_mb.is_none(), "VRAM probing is off by default");
        assert_eq!(r.load_events_since(mark + 3).len(), 0);
        assert_eq!(r.load_events_since(usize::MAX).len(), 0);
        let json = serde_json::to_value(ev[1]).unwrap();
        assert_eq!(json["stage"], "enc_session_init");
    }

    /// With the VRAM probe on, a span of a session component leaves a measured mark.
    #[test]
    fn vram_probing_marks_session_components() {
        let r = recorder();
        let _g = r.record();
        r.vram_probe.store(true, Ordering::SeqCst);
        let marks = r.vram_mark_count();
        drop(r.load_span(LoadStage::KnnSessionInit));
        drop(r.load_span(LoadStage::WarmOnce));
        r.vram_mark("vc.index", 12.0, "explicit");
        let m = r.vram_marks_since(marks);
        let has_dxgi = process_vram_now().is_some();
        // A span mark only where DXGI reports this process's VRAM; the explicit one always.
        assert_eq!(m.len(), if has_dxgi { 2 } else { 1 }, "{m:?}");
        if has_dxgi {
            assert_eq!((m[0].component, m[0].source), ("vc.knn", "span"));
        }
        assert_eq!(m.last().map(|x| (x.component, x.mb)), Some(("vc.index", 12.0)));
    }

    /// The tick side costs nothing measurable and never allocates, off or on (the counting
    /// allocator of `gpu::alloc_probe` sees every Rust heap allocation on this thread).
    #[test]
    fn the_tick_path_is_allocation_free_and_cheap() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let off = recorder();
        let on = recorder();
        let _g = on.record();
        for r in [off, on] {
            let ((), n) = allocations_during(|| {
                for _ in 0..1_000 {
                    for s in Stage::ALL {
                        let _span = r.stage_span(s);
                    }
                    r.record_sync(SyncKind::UploadWait);
                    r.record_gpu_time("vc.enc", 0.1);
                    std::hint::black_box(r.take_tick());
                }
            });
            assert_eq!(n, 0, "allocations on the tick path (recording {})", r.recording());
        }
        // Off: one relaxed load per call. 200k disabled spans in well under a second even in a
        // debug build on a busy runner (about 10 ns each in release).
        let t0 = Instant::now();
        for _ in 0..200_000 {
            let _span = off.stage_span(Stage::Enc);
            off.record_sync(SyncKind::RunEnd);
        }
        let per_call_ns = t0.elapsed().as_nanos() as f64 / 200_000.0;
        assert!(per_call_ns < 2_000.0, "{per_call_ns} ns per disabled span + sync");
    }

    /// A session without a CUDA arena (the CPU EP) has no arena statistics, and marking it records
    /// nothing; a null allocator reads as none.
    #[test]
    fn sessions_without_a_cuda_arena_have_no_arena_stats() {
        use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
        let bytes = crate::services::bench::load_sim::conv_chain_model(16, 1, 7);
        let (session, outcome) = build_session(Some("test.telemetry.arena"), "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).expect("a CPU session");
        assert_eq!(outcome.active, "cpu");
        assert!(session_arena_stats(&session, 0).is_none());
        let _g = start_recording();
        let marks = vram_mark_count();
        arena_mark_session("test.telemetry.arena", &session, 0);
        assert!(!vram_marks_since(marks).iter().any(|m| m.component == "test.telemetry.arena"));
        // SAFETY: a null pointer is checked before any use.
        assert!(unsafe { allocator_stats_raw(std::ptr::null()) }.is_none());
    }

    /// Standard components keep their literals; another name is leaked once and then reused.
    #[test]
    fn component_names_are_interned() {
        assert_eq!(intern_component("vc.enc"), "vc.enc");
        let known = intern_component(&String::from("vc.flowdec"));
        assert!(KNOWN_COMPONENTS.iter().any(|k| std::ptr::eq(*k, known)), "a standard name is not leaked");
        let a = intern_component(&format!("test.telemetry.{}", "interned"));
        let b = intern_component("test.telemetry.interned");
        assert_eq!(a, "test.telemetry.interned");
        assert!(std::ptr::eq(a, b), "one leak per name");
    }

    /// The free functions reach the global recorder (shared with concurrent benchmark tests, so
    /// only lower bounds are checked).
    #[test]
    fn the_free_functions_use_the_global_recorder() {
        let _g = start_recording();
        assert!(recording());
        set_gpu_timing(true);
        assert!(gpu_timing_enabled());
        set_gpu_timing(false);
        let mark = load_mark();
        drop(load_span(LoadStage::WeightsRead));
        assert!(load_events_since(mark).iter().any(|e| e.stage == LoadStage::WeightsRead));
        record_sync(SyncKind::RowRead);
        let marks = vram_mark_count();
        vram_mark("test.telemetry.global", 1.0);
        assert!(vram_marks_since(marks).iter().any(|m| m.component == "test.telemetry.global"));
        assert!(now_ns() > 0);
        let _ = take_tick();
    }
}
