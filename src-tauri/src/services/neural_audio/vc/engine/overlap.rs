//! F0 alongside the content encoder (P3B-D): a persistent, pre-spawned helper thread at the
//! worker's priority (park / unpark, an allocation-free job slot) runs the F0 cache's update,
//! RMVPE's CPU log-mel included, while the worker runs content and retrieval; the tick joins it
//! before the coarse pitch and protect. GPU rungs only (the CPU Streamer stays serial: there the
//! two stages would compete for the same cores). Kill switch `SIGIL_VC_F0_OVERLAP=0`
//! ([`F0_OVERLAP_ENV`]).
//!
//! # How a tick hands F0 over
//! The tick swaps its pitch extractor and its F0 cache into the helper's job slot (a swap with the
//! slot's placeholders: no allocation), copies the context window into the slot's own buffer (a few
//! microseconds; its capacity is kept, so no allocation after the first tick) and wakes the helper.
//! It runs the content encoder (and retrieval) meanwhile, then waits for the job and swaps the
//! extractor and cache back. The join always happens before the tick returns (a [`F0Job`] that is
//! dropped without [`F0Job::join`] joins in its `Drop`); because the helper reads only buffers the
//! slot owns, even a job that is leaked (`mem::forget`) cannot make it read freed memory: the
//! voice then runs F0 serially on the slot's placeholder extractor, which reports an error. Each
//! extractor sees exactly the input of the serial path, so the output is bit-identical
//! (`tick_tests::f0_overlap_matches_serial`).
//!
//! The helper takes the posting thread's priority (and its EcoQoS opt-out) before each job, so a
//! live worker-priority switch reaches it; it never registers MMCSS (an inference thread). A job
//! that panics is reported as an error of that tick; the helper keeps running.
//!
//! # The hand-over protocol (one atomic state word)
//! `IDLE -> READY` (the tick posts) `-> DONE` (the helper finished) `-> IDLE` (the tick joined).
//! The joiner spins first and then announces that it parks by moving the state word itself,
//! `READY -> PARKED` (a compare-exchange); the helper hands the slot over with one compare-exchange
//! `READY | PARKED -> DONE` and unparks the joiner only when that exchange found `PARKED`. Both
//! sides modify the one atomic, so exactly one of them wins: either the helper's exchange came
//! first (the joiner's fails with `DONE`, it does not park) or the joiner's did (the helper sees
//! `PARKED` and unparks it). The announcement is part of the call's own state, so it cannot
//! outlive the call: a helper thread that is descheduled between its hand-over of tick N and its
//! wake-up can never take or clear the announcement of tick N + 1. (An earlier design kept a second
//! flag, `waiting`, that the helper cleared with `swap(false)` after storing `DONE`; the delayed
//! swap consumed the next tick's announcement and that tick's joiner parked for ever.) A wake-up
//! that arrives late is a spurious one for a loop that re-checks the state. The joiner must be the
//! thread that posted (the helper unparks the poster): [`F0Job`] is `!Send` to enforce it.
//!
//! On CUDA the content plan and the pitch plan run on separate streams, so their kernels overlap; on
//! DirectML each session submits to the adapter's queue, where the overlap is the CPU side (RMVPE's
//! log-mel, the submission, the fence waits) unless the queues run concurrently
//! (`tick_tests::content_and_f0_concurrency_bench` measures it).
//!
//! # Frozen by P3B-S0 (signatures; P3B-D implements)
//! - [`F0Helper`] (`Default`: inert, no thread), the engine's `f0_helper` field;
//!   [`F0Helper::active`] says whether ticks overlap F0.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::{JoinHandle, Thread};
use std::time::Instant;

use crate::services::neural_audio::telemetry::{self, Stage};

use super::super::traits::PitchExtractor;
use super::f0_cache::{F0Cache, F0CacheConfig};

/// Kill switch of the F0 overlap: `0` / `off` / `false` / `no` runs F0 after the content encoder on
/// the worker thread (the P3 path), for A/B runs from one executable.
pub const F0_OVERLAP_ENV: &str = "SIGIL_VC_F0_OVERLAP";

/// Whether the overlap is switched on in this process ([`F0_OVERLAP_ENV`]; on unless set to `0` /
/// `off` / `false` / `no`). Reads the environment: call it at load time, not per tick.
pub fn f0_overlap_switch() -> bool {
    match std::env::var(F0_OVERLAP_ENV) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
        Err(_) => true,
    }
}

/// Spins of the joining thread before it parks (the F0 job often ends within microseconds of the
/// content encoder; a park and wake costs tens of microseconds).
const JOIN_SPINS: u32 = 2_000;

const IDLE: u8 = 0;
/// A job is posted; the joining thread has not announced a park (it is spinning, or has not
/// started to wait).
const READY: u8 = 1;
const DONE: u8 = 2;
const SHUTDOWN: u8 = 3;
/// A job is posted and the joining thread announced that it parks (or is about to) until `DONE`.
/// Moved to by the joiner from `READY` and left by the helper's hand-over to `DONE` (module docs,
/// "The hand-over protocol").
const PARKED: u8 = 4;

/// The extractor a slot holds while the engine owns the real one (a zero-sized type: boxing it
/// does not allocate).
struct NoPitch;

impl PitchExtractor for NoPitch {
    fn f0(&mut self, _mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        Err("no pitch extractor in the F0 helper's slot".into())
    }
}

/// One F0 job and its result. Owned by the posting thread while the state is `IDLE` or `DONE`, by
/// the helper while it is `READY` or `PARKED` (until the helper hands the slot over with `DONE`).
struct Slot {
    pitch: Box<dyn PitchExtractor>,
    cache: F0Cache,
    /// A copy of the context window (capacity kept from job to job).
    mono: Vec<f32>,
    pos: u64,
    /// The `f0_cache` lever is off: the whole window, every tick (`F0Cache::extract_window`).
    whole_window: bool,
    /// Frames of the window the tick needs.
    frames: usize,
    /// The window's F0 (`frames` values) when the job succeeded.
    out: Vec<f32>,
    /// Frames the extractor analysed, the job's wall time and its error.
    extracted: usize,
    ms: f64,
    error: Option<String>,
    /// Too few frames came back (the tick reports it).
    short: Option<usize>,
    /// Woken when the job is done.
    waker: Option<Thread>,
    /// The posting thread's scheduling policy, applied by the helper before the job.
    policy: ThreadPolicy,
}

/// A test hook of the helper thread ([`Shared::after_done`]).
#[cfg(test)]
type HelperHook = Arc<dyn Fn(u64) + Send + Sync>;

struct Shared {
    /// The hand-over state word, including the joiner's park announcement (`PARKED`): see the
    /// module docs. The helper unparks the joiner only when its hand-over found `PARKED`, so a
    /// join that saw `DONE` while spinning leaves no stale park token on the worker thread (whose
    /// own waits use `park_timeout`).
    state: AtomicU8,
    slot: UnsafeCell<Slot>,
    /// Tests: runs on the helper thread right after a job's hand-over to `DONE` and before the
    /// wake-up that hand-over may owe the joiner (argument: the jobs handed over before it), to
    /// replay the interleavings of a helper thread that is descheduled there.
    #[cfg(test)]
    after_done: std::sync::Mutex<Option<HelperHook>>,
    #[cfg(test)]
    handed_over: std::sync::atomic::AtomicU64,
}

// SAFETY: the slot is only touched by the side that owns it in the state protocol above
// (`IDLE`/`DONE`: the posting thread; `READY`/`PARKED`: the helper), with Release stores and
// Acquire loads of `state` ordering every hand-over (the extractor is `Send`; everything else in
// the slot is owned data).
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

#[cfg(test)]
impl Shared {
    /// Run the test hook for the job just handed over (outside the lock: the hook may block).
    fn run_after_done_hook(&self) {
        let n = self.handed_over.fetch_add(1, Ordering::SeqCst);
        let hook = self.after_done.lock().ok().and_then(|g| g.clone());
        if let Some(hook) = hook {
            hook(n);
        }
    }
}

struct Worker {
    shared: Arc<Shared>,
    thread: Thread,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // No job is in flight between ticks (every post is joined within its tick); should one be
        // (a leaked job), the helper sees SHUTDOWN instead of storing DONE and exits.
        self.shared.state.store(SHUTDOWN, Ordering::SeqCst);
        self.thread.unpark();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The overlapped-F0 helper (see the module docs). `Default` is inert: no thread, ticks run F0
/// serially until [`F0Helper::configure`] turns it on.
#[derive(Default)]
pub struct F0Helper {
    /// Ticks overlap F0 with the content encoder.
    on: bool,
    /// Why the helper is off (`None` while on, or before it is configured).
    off_reason: Option<&'static str>,
    worker: Option<Worker>,
    /// The thread could not be started (or died): ticks run F0 serially from then on.
    failed: bool,
}

impl std::fmt::Debug for F0Helper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F0Helper")
            .field("on", &self.on)
            .field("off_reason", &self.off_reason)
            .field("thread", &self.worker.is_some())
            .finish()
    }
}

impl F0Helper {
    /// Whether ticks run F0 on the helper (concurrently with content).
    #[inline]
    pub fn active(&self) -> bool {
        self.on && !self.failed
    }

    /// Why ticks run F0 serially (`None` while the overlap is active).
    pub fn off_reason(&self) -> Option<&'static str> {
        if self.active() {
            None
        } else if self.failed {
            Some("the F0 helper thread could not run")
        } else {
            self.off_reason
        }
    }

    /// Switch the overlap on or off (load time; starting the thread allocates). `off_reason` says
    /// why it stays off. Turning it off keeps a started thread parked for a later switch back.
    pub fn configure(&mut self, on: bool, off_reason: Option<&'static str>) {
        self.on = on;
        self.off_reason = if on { None } else { off_reason };
        if on && self.worker.is_none() && !self.failed {
            match spawn() {
                Ok(w) => self.worker = Some(w),
                Err(e) => {
                    tracing::warn!(target: "sigil::vc", "F0 overlap: the helper thread could not start ({e}); F0 runs after the content encoder");
                    self.failed = true;
                }
            }
        }
    }

    /// Hand the tick's F0 to the helper: `pitch` and `cache` move into the job until
    /// [`F0Job::join`] puts them back; `mono` (whose first sample is absolute 16 kHz index `pos`)
    /// is copied into the job. `frames` is how many frames of F0 the tick needs. Allocation-free
    /// once a window of this length has been posted.
    pub fn post<'a>(
        &'a mut self,
        pitch: &'a mut Box<dyn PitchExtractor>,
        cache: &'a mut F0Cache,
        mono: &[f32],
        pos: u64,
        whole_window: bool,
        frames: usize,
    ) -> Option<F0Job<'a>> {
        if !self.active() {
            return None;
        }
        let worker = self.worker.as_ref()?;
        if worker.shared.state.load(Ordering::Acquire) != IDLE {
            // A previous job was never joined (cannot happen through `F0Job`): run serially.
            return None;
        }
        // SAFETY: state IDLE: this thread owns the slot.
        let slot = unsafe { &mut *worker.shared.slot.get() };
        std::mem::swap(&mut slot.pitch, pitch);
        std::mem::swap(&mut slot.cache, cache);
        slot.mono.clear();
        slot.mono.extend_from_slice(mono);
        slot.pos = pos;
        slot.whole_window = whole_window;
        slot.frames = frames;
        slot.error = None;
        slot.short = None;
        slot.extracted = 0;
        slot.ms = 0.0;
        slot.waker = Some(std::thread::current());
        slot.policy = ThreadPolicy::of_current();
        worker.shared.state.store(READY, Ordering::Release);
        worker.thread.unpark();
        Some(F0Job {
            worker,
            pitch,
            cache,
            joined: false,
            _same_thread: PhantomData,
        })
    }
}

/// What a finished F0 job reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct F0JobStats {
    /// Wall time of the job on the helper (ms).
    pub ms: f64,
    /// Frames the extractor analysed.
    pub extracted: usize,
}

/// An F0 job in flight (see [`F0Helper::post`]).
pub struct F0Job<'a> {
    worker: &'a Worker,
    pitch: &'a mut Box<dyn PitchExtractor>,
    cache: &'a mut F0Cache,
    joined: bool,
    /// `!Send`: the helper wakes the thread that posted, so the job must be joined there.
    _same_thread: PhantomData<*const ()>,
}

impl F0Job<'_> {
    /// Wait for the job, put the extractor and the cache back, and copy the window's F0 (`frames`
    /// values) into `f0` (resized to `frames`; no allocation once it has the capacity).
    pub fn join(mut self, f0: &mut Vec<f32>) -> Result<F0JobStats, String> {
        let slot = self.wait_and_restore();
        let stats = F0JobStats { ms: slot.ms, extracted: slot.extracted };
        let result = match (slot.error.take(), slot.short) {
            (Some(e), _) => Err(e),
            (None, Some(n)) => Err(format!("the F0 contour has {n} frames, the context {}", slot.frames)),
            (None, None) => {
                f0.clear();
                f0.extend_from_slice(&slot.out[..slot.frames]);
                Ok(stats)
            }
        };
        self.worker.shared.state.store(IDLE, Ordering::Release);
        result
    }

    /// Wait for `DONE` and swap the extractor and cache back; returns the slot (owned by this
    /// thread until the state goes back to `IDLE`). Spins first (the job often ends within
    /// microseconds of the content encoder), then announces the park by moving this job's own
    /// state word `READY -> PARKED` and parks until the helper's hand-over leaves `PARKED` (module
    /// docs, "The hand-over protocol").
    #[allow(clippy::mut_from_ref)]
    fn wait_and_restore(&mut self) -> &mut Slot {
        let state = &self.worker.shared.state;
        let mut spins = 0u32;
        while state.load(Ordering::Acquire) != DONE {
            if spins < JOIN_SPINS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            match state.compare_exchange(READY, PARKED, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    while state.load(Ordering::Acquire) == PARKED {
                        std::thread::park();
                    }
                }
                // The helper finished first: the loop condition sees DONE.
                Err(now) => debug_assert_eq!(now, DONE, "a join without a posted job"),
            }
        }
        let shared = &self.worker.shared;
        // SAFETY: state DONE: the helper has finished with the slot; this thread owns it.
        let slot = unsafe { &mut *shared.slot.get() };
        if !self.joined {
            std::mem::swap(&mut slot.pitch, self.pitch);
            std::mem::swap(&mut slot.cache, self.cache);
            self.joined = true;
        }
        slot
    }
}

impl Drop for F0Job<'_> {
    fn drop(&mut self) {
        if !self.joined {
            // The tick returned early (a content-stage error): the extractor and cache must go
            // back before the tick's borrows end, so wait for the helper to finish with them.
            self.wait_and_restore();
            self.worker.shared.state.store(IDLE, Ordering::Release);
        }
    }
}

fn spawn() -> Result<Worker, String> {
    let shared = Arc::new(Shared {
        state: AtomicU8::new(IDLE),
        slot: UnsafeCell::new(Slot {
            pitch: Box::new(NoPitch),
            // The placeholder cache the engine's real one is swapped with (allocated once here).
            cache: F0Cache::new(F0CacheConfig {
                margin_frames: 0,
                capacity_frames: 1,
            }),
            mono: Vec::new(),
            pos: 0,
            whole_window: false,
            frames: 0,
            out: Vec::with_capacity(512),
            extracted: 0,
            ms: 0.0,
            error: None,
            short: None,
            waker: None,
            policy: ThreadPolicy::default(),
        }),
        #[cfg(test)]
        after_done: std::sync::Mutex::new(None),
        #[cfg(test)]
        handed_over: std::sync::atomic::AtomicU64::new(0),
    });
    let theirs = Arc::clone(&shared);
    let handle = std::thread::Builder::new()
        .name("sigil-vc-f0".into())
        .spawn(move || helper_loop(&theirs))
        .map_err(|e| e.to_string())?;
    Ok(Worker {
        thread: handle.thread().clone(),
        shared,
        handle: Some(handle),
    })
}

fn helper_loop(shared: &Shared) {
    let mut applied = ThreadPolicy::default();
    loop {
        match shared.state.load(Ordering::Acquire) {
            // PARKED: the joiner announced its park before this thread even woke; the job is the
            // same.
            READY | PARKED => {
                // SAFETY: state READY or PARKED: the posting thread handed the slot over and waits
                // for DONE.
                let slot = unsafe { &mut *shared.slot.get() };
                if slot.policy != applied {
                    slot.policy.apply_to_current();
                    applied = slot.policy;
                }
                run_job(slot);
                let waker = slot.waker.take();
                // Hand the slot back. The exchange decides, atomically with the hand-over, whether
                // the joiner announced a park (`wait_and_restore`); nothing of this job is looked
                // at again afterwards, so a delay between here and the unpark only delays this
                // job's own joiner.
                let mut seen = shared.state.load(Ordering::Acquire);
                let joiner_parked = loop {
                    match seen {
                        READY | PARKED => match shared.state.compare_exchange(seen, DONE, Ordering::SeqCst, Ordering::SeqCst) {
                            Ok(_) => break seen == PARKED,
                            Err(now) => seen = now,
                        },
                        // Shut down while the job ran (only a leaked job is in flight at a drop).
                        _ => return,
                    }
                };
                #[cfg(test)]
                shared.run_after_done_hook();
                if joiner_parked {
                    if let Some(w) = waker {
                        w.unpark();
                    }
                }
            }
            SHUTDOWN => return,
            _ => std::thread::park(),
        }
    }
}

fn run_job(slot: &mut Slot) {
    let started = Instant::now();
    let mono: &[f32] = &slot.mono;
    let (pitch, cache, pos, whole, frames) = (&mut slot.pitch, &mut slot.cache, slot.pos, slot.whole_window, slot.frames);
    let (out, error, short) = (&mut slot.out, &mut slot.error, &mut slot.short);
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _span = telemetry::stage_span(Stage::F0);
        let f0 = if whole {
            cache.extract_window(pitch.as_mut(), mono)?
        } else {
            cache.update(pitch.as_mut(), mono, pos)?
        };
        if f0.len() < frames {
            *short = Some(f0.len());
        } else {
            out.clear();
            out.extend_from_slice(&f0[..frames]);
        }
        Ok::<(), String>(())
    }));
    match run {
        Ok(Ok(())) => {}
        Ok(Err(e)) => *error = Some(e),
        Err(_) => *error = Some("the F0 helper's job panicked".into()),
    }
    slot.extracted = slot.cache.last_extracted_frames();
    slot.ms = started.elapsed().as_secs_f64() * 1000.0;
}

/// A thread's scheduling policy: its priority and whether it opted out of EcoQoS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ThreadPolicy {
    /// `GetThreadPriority` of the thread (`None` where unknown).
    priority: Option<i32>,
    /// The thread opted out of EcoQoS (HighQoS: `protect_voice`).
    high_qos: bool,
}

#[cfg(windows)]
impl ThreadPolicy {
    fn of_current() -> ThreadPolicy {
        use windows::Win32::System::Threading::{
            GetCurrentThread, GetThreadInformation, GetThreadPriority, ThreadPowerThrottling, THREAD_POWER_THROTTLING_CURRENT_VERSION,
            THREAD_POWER_THROTTLING_EXECUTION_SPEED, THREAD_POWER_THROTTLING_STATE,
        };
        // SAFETY: the pseudo-handle of the calling thread; pointer and size describe `state`.
        unsafe {
            let priority = GetThreadPriority(GetCurrentThread());
            let mut state = THREAD_POWER_THROTTLING_STATE {
                Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
                ControlMask: 0,
                StateMask: 0,
            };
            let high_qos = GetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &mut state as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .is_ok()
                && state.ControlMask & THREAD_POWER_THROTTLING_EXECUTION_SPEED != 0
                && state.StateMask & THREAD_POWER_THROTTLING_EXECUTION_SPEED == 0;
            // THREAD_PRIORITY_ERROR_RETURN (MAXLONG) when the call failed.
            ThreadPolicy {
                priority: (priority != i32::MAX).then_some(priority),
                high_qos,
            }
        }
    }

    fn apply_to_current(&self) {
        use windows::Win32::System::Threading::{SetThreadPriority, GetCurrentThread, THREAD_PRIORITY};
        if let Some(p) = self.priority {
            // SAFETY: the pseudo-handle of the calling thread.
            let _ = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY(p)) };
        }
        if self.high_qos {
            let _ = crate::services::rt_thread::opt_out_power_throttling();
        } else {
            let _ = reset_power_throttling();
        }
    }
}

/// Hand the calling thread's EcoQoS state back to the system (best effort).
#[cfg(windows)]
fn reset_power_throttling() -> bool {
    use windows::Win32::System::Threading::{
        GetCurrentThread, SetThreadInformation, ThreadPowerThrottling, THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_STATE,
    };
    let state = THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: 0,
        StateMask: 0,
    };
    // SAFETY: the pseudo-handle of the calling thread; pointer and size describe `state`.
    unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            &state as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
        )
        .is_ok()
    }
}

#[cfg(not(windows))]
impl ThreadPolicy {
    fn of_current() -> ThreadPolicy {
        ThreadPolicy::default()
    }

    fn apply_to_current(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::engine::f0_cache::{F0Cache, F0CacheConfig};

    /// A deterministic extractor: frame j = 100 + mean |x| over the frame (allocates like `f0`).
    struct Mean;

    impl PitchExtractor for Mean {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            Ok(x.chunks(160).map(|c| 100.0 + c.iter().map(|v| v.abs()).sum::<f32>() / c.len() as f32).collect())
        }
    }

    struct Panics;

    impl PitchExtractor for Panics {
        fn f0(&mut self, _x: &[f32]) -> Result<Vec<f32>, String> {
            panic!("boom")
        }
    }

    fn cache() -> F0Cache {
        F0Cache::new(F0CacheConfig { margin_frames: 4, capacity_frames: 128 })
    }

    /// The helper computes exactly what the serial path computes, tick after tick, and hands the
    /// extractor and the cache back each time.
    #[test]
    fn overlapped_f0_equals_serial_f0() {
        let audio: Vec<f32> = (0..16_000 * 2).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let (ctx, block) = (7_040usize, 640usize);
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        assert!(helper.active());
        let mut serial_pitch: Box<dyn PitchExtractor> = Box::new(Mean);
        let mut serial_cache = cache();
        let mut pitch: Box<dyn PitchExtractor> = Box::new(Mean);
        let mut f0_cache = cache();
        let mut got = Vec::with_capacity(64);
        for tick in 0..20usize {
            let pos = tick * block;
            let mono = &audio[pos..pos + ctx];
            let want = serial_cache.update(serial_pitch.as_mut(), mono, pos as u64).unwrap()[..44].to_vec();
            let job = helper.post(&mut pitch, &mut f0_cache, mono, pos as u64, false, 44).expect("posted");
            let stats = job.join(&mut got).unwrap();
            assert_eq!(got, want, "tick {tick}");
            assert_eq!(stats.extracted, serial_cache.last_extracted_frames());
            assert_eq!(f0_cache.last_extracted_frames(), serial_cache.last_extracted_frames());
        }
        // The whole-window path (the `f0_cache` lever off).
        let mono = &audio[..ctx];
        let want = serial_cache.extract_window(serial_pitch.as_mut(), mono).unwrap()[..44].to_vec();
        let job = helper.post(&mut pitch, &mut f0_cache, mono, 0, true, 44).unwrap();
        job.join(&mut got).unwrap();
        assert_eq!(got, want);
    }

    /// The steady state makes no heap allocation on the posting thread (post, join, swap back).
    #[test]
    fn post_and_join_are_allocation_free() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        struct Quiet(Vec<f32>);
        impl PitchExtractor for Quiet {
            fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
                Ok(vec![120.0; x.len() / 160])
            }
            fn f0_into(&mut self, x: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
                out.clear();
                out.extend(self.0.iter().take(x.len() / 160));
                Ok(())
            }
        }
        let audio = vec![0.1f32; 7_040 + 640 * 40];
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        let mut pitch: Box<dyn PitchExtractor> = Box::new(Quiet(vec![120.0; 64]));
        let mut f0_cache = cache();
        let mut got = Vec::with_capacity(64);
        let tick = |t: usize, helper: &mut F0Helper, pitch: &mut Box<dyn PitchExtractor>, f0_cache: &mut F0Cache, got: &mut Vec<f32>| {
            let pos = t * 640;
            let job = helper.post(pitch, f0_cache, &audio[pos..pos + 7_040], pos as u64, false, 44).unwrap();
            job.join(got).unwrap();
        };
        for t in 0..10 {
            tick(t, &mut helper, &mut pitch, &mut f0_cache, &mut got);
        }
        let ((), allocs) = allocations_during(|| {
            for t in 10..30 {
                tick(t, &mut helper, &mut pitch, &mut f0_cache, &mut got);
            }
        });
        // The extractor's own `f0` allocates on the helper thread (not counted here; P3B-E's
        // `f0_into` removes it); the posting thread's hand-over does not allocate.
        assert_eq!(allocs, 0, "post / join allocated {allocs} times in 20 ticks");
    }

    /// A job dropped without a join (a tick that returned early) still waits for the helper and
    /// hands the extractor and cache back; a panicking extractor is an error of its tick and the
    /// helper keeps serving.
    #[test]
    fn dropped_jobs_and_panics_are_contained() {
        let audio = vec![0.2f32; 7_040];
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        let mut pitch: Box<dyn PitchExtractor> = Box::new(Mean);
        let mut f0_cache = cache();
        {
            let _job = helper.post(&mut pitch, &mut f0_cache, &audio, 0, false, 44).unwrap();
        }
        let mut got = Vec::new();
        assert!(pitch.f0(&audio).is_ok(), "the real extractor is back after a dropped job");
        let mut bad: Box<dyn PitchExtractor> = Box::new(Panics);
        let err = helper.post(&mut bad, &mut f0_cache, &audio, 0, false, 44).unwrap().join(&mut got).unwrap_err();
        assert!(err.contains("panicked"), "{err}");
        let ok = helper.post(&mut pitch, &mut f0_cache, &audio, 0, false, 44).unwrap().join(&mut got);
        assert!(ok.is_ok(), "{ok:?}");
        assert_eq!(got.len(), 44);
        // Too few frames for the tick: an error naming both counts.
        let err = helper.post(&mut pitch, &mut f0_cache, &audio, 0, false, 100).unwrap().join(&mut got).unwrap_err();
        assert!(err.contains("44 frames") || err.contains("frames, the context 100"), "{err}");
    }

    /// A leaked job (`mem::forget`, which safe code may do) leaves the helper reading only its own
    /// copy of the window: the window can go at once, later posts decline (the voice then runs F0
    /// serially), and dropping the helper still stops its thread.
    #[test]
    fn a_leaked_job_is_contained() {
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        let mut pitch: Box<dyn PitchExtractor> = Box::new(Mean);
        let mut f0_cache = cache();
        {
            let audio = vec![0.25f32; 7_040];
            let job = helper.post(&mut pitch, &mut f0_cache, &audio, 0, false, 44).unwrap();
            std::mem::forget(job);
            drop(audio);
        }
        assert!(helper.post(&mut pitch, &mut f0_cache, &[0.0; 7_040], 0, false, 44).is_none(), "the slot is still taken");
        // The engine kept the placeholder extractor: its F0 is an error, not a crash.
        assert!(pitch.f0(&[0.0; 1_600]).is_err());
        drop(helper);
    }

    /// Off (the kill switch or a CPU rung): no thread, `post` declines, and the reason is kept.
    #[test]
    fn an_inert_helper_declines() {
        let mut helper = F0Helper::default();
        assert!(!helper.active());
        helper.configure(false, Some("the CPU Streamer runs F0 serially"));
        assert_eq!(helper.off_reason(), Some("the CPU Streamer runs F0 serially"));
        let mut pitch: Box<dyn PitchExtractor> = Box::new(Mean);
        let mut f0_cache = cache();
        assert!(helper.post(&mut pitch, &mut f0_cache, &[0.0; 1_600], 0, false, 10).is_none());
        assert!(helper.worker.is_none(), "no thread while off");
    }

    /// Busy-wait for `us` microseconds (a delay that keeps the thread on its core, as a descheduled
    /// helper does not).
    fn spin_for(us: u64) {
        if us > 0 {
            let t0 = Instant::now();
            while t0.elapsed() < std::time::Duration::from_micros(us) {
                std::hint::spin_loop();
            }
        }
    }

    /// A cheap, well-spread hash of a job number (deterministic jitter for the stress test).
    fn mix(n: u64) -> u64 {
        let x = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 17
    }

    /// An extractor that takes `slow_ms` when the input is `slow_len` samples long (the job a test
    /// wants still running at a given moment) and returns at once otherwise. Its contour is the
    /// input's length in frames, in every frame, so a join that returned another job's result shows.
    struct Paced {
        slow_len: usize,
        slow_ms: u64,
    }

    impl PitchExtractor for Paced {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            if x.len() == self.slow_len {
                std::thread::sleep(std::time::Duration::from_millis(self.slow_ms));
            }
            Ok(vec![(x.len() / 160) as f32; x.len() / 160])
        }
    }

    /// Like [`Paced`] with a job time that jitters from job to job (the joiner sometimes ends its
    /// spin first and sometimes parks).
    struct Jittery {
        calls: u64,
    }

    impl PitchExtractor for Jittery {
        fn f0(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
            self.calls += 1;
            spin_for([0, 0, 40, 150, 400, 0, 80, 250][(mix(self.calls) % 8) as usize]);
            Ok(vec![(x.len() / 160) as f32; x.len() / 160])
        }
    }

    /// The helper thread's shared state, for the tests' hook and state checks.
    fn shared_of(helper: &F0Helper) -> Arc<Shared> {
        Arc::clone(&helper.worker.as_ref().expect("a configured helper has its thread").shared)
    }

    /// The lost wake-up (found in the sibling voicing lane of `f0_fusion.rs`, which had the same
    /// protocol): the helper thread is descheduled between handing job 1 over (state `DONE`) and
    /// its wake-up code. The tick, which saw `DONE` while spinning and never announced a park, runs
    /// on, posts job 2, spins its 2,000 rounds and parks. When the helper resumes, its stale
    /// wake-up of job 1 must not eat job 2's announcement: with a shared `waiting` flag cleared by
    /// `swap(false)` after the hand-over it did (the swap returned true, the spurious unpark woke
    /// the joiner, which re-parked without announcing again, and job 2's hand-over then saw no
    /// announcement) and job 2's joiner parked for ever. The test replays exactly that order with
    /// the helper held in the `after_done` hook, and fails (instead of hanging) after 5 s.
    #[test]
    fn a_late_wake_up_of_one_job_cannot_strand_the_next_joiner() {
        use std::sync::{mpsc, Mutex};
        use std::time::Duration;
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        let shared = shared_of(&helper);
        let (reached_tx, reached_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (reached_tx, release_rx) = (Mutex::new(reached_tx), Mutex::new(release_rx));
        *shared.after_done.lock().unwrap() = Some(Arc::new(move |n| {
            if n == 0 {
                reached_tx.lock().unwrap().send(()).unwrap();
                let _ = release_rx.lock().unwrap().recv_timeout(Duration::from_secs(20));
            }
        }));
        let (result_tx, result_rx) = mpsc::channel::<Result<Vec<f32>, String>>();
        let theirs = Arc::clone(&shared);
        // Detached (not scoped), so that a stranded joiner cannot hang the test run.
        let caller = std::thread::spawn(move || {
            let mut pitch: Box<dyn PitchExtractor> = Box::new(Paced { slow_len: 3_200, slow_ms: 30 });
            let mut f0_cache = cache();
            let mut got = Vec::new();
            // Job 1: join only once the helper has handed it over, so the join never parks.
            let job = helper.post(&mut pitch, &mut f0_cache, &vec![0.1; 1_600], 0, true, 10).unwrap();
            while theirs.state.load(Ordering::Acquire) != DONE {
                std::hint::spin_loop();
            }
            job.join(&mut got).unwrap();
            assert_eq!(got, vec![10.0; 10]);
            // Job 2, posted while the helper thread is still inside job 1's hand-over tail.
            let r = helper
                .post(&mut pitch, &mut f0_cache, &vec![0.2; 3_200], 0, true, 20)
                .unwrap()
                .join(&mut got)
                .map(|_| got.clone());
            let _ = result_tx.send(r);
        });
        reached_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the helper never handed job 1 over");
        // The tick is now posting job 2 and parking in its join (a few tens of microseconds of
        // spinning); give it far longer than that, then let the helper thread resume.
        std::thread::sleep(Duration::from_millis(150));
        release_tx.send(()).unwrap();
        match result_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => assert_eq!(r.unwrap(), vec![20.0; 20]),
            Err(_) => {
                // Free the stuck caller (the state is DONE by now) before failing.
                for _ in 0..50 {
                    caller.thread().unpark();
                    std::thread::sleep(Duration::from_millis(10));
                }
                panic!("job 2's join was never woken: the late wake-up of job 1 took its park announcement");
            }
        }
        caller.join().unwrap();
        assert_eq!(shared.handed_over.load(Ordering::SeqCst), 2);
    }

    /// Stress version of the above: thousands of jobs with a jittering helper (job time, and a
    /// random stall between the hand-over and the wake-up) against a tick with a jittering gap
    /// between jobs, joins that sometimes spin out and park, and every seventh job dropped unjoined
    /// (a tick that returned early). Every join must return its own job's contour, and the whole
    /// run must finish (about 3 s; a watchdog fails it after 30 s instead of hanging).
    #[test]
    fn the_hand_over_survives_random_stalls_and_dropped_jobs() {
        use std::sync::atomic::AtomicBool;
        use std::sync::mpsc;
        use std::time::Duration;
        const JOBS: u64 = 3_000;
        let mut helper = F0Helper::default();
        helper.configure(true, None);
        let shared = shared_of(&helper);
        *shared.after_done.lock().unwrap() = Some(Arc::new(|n| spin_for([0, 0, 20, 100, 300, 1_000, 50, 0][(mix(n + 17) % 8) as usize])));
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let abort = Arc::new(AtomicBool::new(false));
        let (theirs, stop) = (Arc::clone(&shared), Arc::clone(&abort));
        let caller = std::thread::spawn(move || {
            let mut pitch: Box<dyn PitchExtractor> = Box::new(Jittery { calls: 0 });
            let mut f0_cache = cache();
            let mut got = Vec::new();
            for job in 0..JOBS {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let frames = 8 + (mix(job + 3) % 40) as usize;
                let x = vec![0.1f32; frames * 160];
                let posted = helper.post(&mut pitch, &mut f0_cache, &x, 0, true, frames).expect("posted");
                if job % 7 == 6 {
                    // Dropped unjoined, as by a tick that returns early.
                    drop(posted);
                    assert_eq!(theirs.state.load(Ordering::Acquire), IDLE);
                } else {
                    spin_for([0, 0, 5, 30, 120, 0, 600, 60][(mix(job + 9) % 8) as usize]);
                    posted.join(&mut got).unwrap();
                    assert_eq!(got, vec![frames as f32; frames], "job {job}: another job's contour");
                }
                // The gap before the next job: shorter and longer than the helper's stall.
                for _ in 0..[0u32, 1, 40, 400, 3_000, 0, 150, 12][(mix(job + 5) % 8) as usize] {
                    std::hint::spin_loop();
                }
            }
            assert_eq!(theirs.handed_over.load(Ordering::SeqCst), JOBS);
            assert_eq!(theirs.state.load(Ordering::Acquire), IDLE);
            let _ = done_tx.send(());
        });
        match done_rx.recv_timeout(Duration::from_secs(30)) {
            Ok(()) => caller.join().unwrap(),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The tick thread panicked (an assertion above): report that panic.
                if let Err(p) = caller.join() {
                    std::panic::resume_unwind(p);
                }
                unreachable!("the tick thread ended without finishing");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Look before freeing the tick: once it ends it drops the helper (state SHUTDOWN).
                let (state, handed) = (shared.state.load(Ordering::SeqCst), shared.handed_over.load(Ordering::SeqCst));
                // The tick is parked in a join whose wake-up was lost; stop it and free it.
                abort.store(true, Ordering::Relaxed);
                for _ in 0..50 {
                    caller.thread().unpark();
                    std::thread::sleep(Duration::from_millis(10));
                }
                panic!("a join was never woken (lost wake-up): the state word was {state} after {handed} of {JOBS} jobs");
            }
        }
    }
}
