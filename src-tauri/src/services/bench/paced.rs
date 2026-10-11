//! Paced driver: the voice-changer worker loop against a simulated device clock.
//!
//! Capture callbacks deliver `capture_callback_ms` of the input at their period, the playback
//! device pulls `playback_callback_ms` at its period (half a period out of phase), and a worker
//! runs exactly the production sequence on every wake: capture ring → backlog → latency budget
//! (oldest audio dropped) → one block through the harness ([`VcTick`]) → output queue → ring →
//! device. The budget, queue, ring playout and concealment are the voice changer's own code
//! (see [`super::vc_rt`]).
//!
//! # Clocks
//! - `wall`: real time. Callbacks are due at wall-clock instants, the worker sleeps until the next
//!   capture callback when it has no block, injected stalls really sleep, and load generators
//!   contend for real. The authoritative mode.
//! - `virtual`: simulated time advanced by each block's measured compute (plus a stub's declared
//!   cost and injected stalls, which are accounted, not slept). Fast and free of sleep and
//!   scheduling effects; compute is still real.
//! - `simulated`: only a stub's declared compute and injected stalls advance time. Every timeline
//!   metric is reproducible under any load (tests of the latency / budget / queue logic).
//!
//! # Latency
//! Every sample the device plays is traced back to the captured sample it came from: output ring
//! frame → emitted frame (output queue trims accounted) → harness input frame (minus the probe's
//! algorithmic delay) → worker block → capture ring frame (budget drops accounted) → capture frame
//! (ring overflow accounted). `latency_ms` is then *play time − capture time* of the first frame of
//! every playback callback: capture period, queueing, compute, the harness's algorithmic delay and
//! the output cushion, exactly as a listener hears it (device-internal buffers excluded).
//!
//! Callbacks between two worker events are evaluated lazily but in order before the worker's next
//! push, which yields the same ring states as running them on time (only the worker adds to the
//! output ring and only the device takes from it).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::services::neural_audio::VcStreamHarness;
use crate::services::rt_thread::{self, WorkerPriority};

use super::attribution::{insert_call_metrics, insert_t1, p95_shares, JoinStats, T1Parts};
use super::engine::{AlgorithmicLatency, Engine, ModelTimes};
use super::load_sim::{LoadCpu, LoadHandle, LoadReport};
use super::metrics::{cores, insert_components, insert_summary, process_cpu_time, process_cycles, thread_cpu_time, thread_cycles, Metrics, Summary, VramPeak};
use super::report::{RunOutcome, TickRecord};
use super::spec::{ClockKind, PacedRun, StreamSpec};
use super::vc_rt::{InputSide, OutputSide, Pushed, VcTick};

pub(super) struct PacedInput<'a> {
    pub stream: &'a StreamSpec,
    pub run: &'a PacedRun,
    pub engine: &'a Engine,
    /// Interleaved at the stream's input rate / channels; looped.
    pub input: &'a [f32],
    pub probe: AlgorithmicLatency,
    pub record_audio: bool,
    pub cancel: &'a AtomicBool,
    pub progress: &'a (dyn Fn(f64, &str) + Sync),
}

/// The run's clock (seconds since the run started).
enum Clock {
    Wall(Instant),
    Virtual(f64),
}

impl Clock {
    fn now(&self) -> f64 {
        match self {
            Clock::Wall(start) => start.elapsed().as_secs_f64(),
            Clock::Virtual(t) => *t,
        }
    }

    fn wait_until(&mut self, t: f64) {
        match self {
            Clock::Wall(start) => {
                let now = start.elapsed().as_secs_f64();
                if t > now {
                    std::thread::sleep(Duration::from_secs_f64(t - now));
                }
            }
            Clock::Virtual(now) => *now = now.max(t),
        }
    }

    /// Time spent working (virtual only; wall time passes by itself).
    fn charge(&mut self, d: Duration) {
        if let Clock::Virtual(now) = self {
            *now += d.as_secs_f64();
        }
    }

    /// An injected stall.
    fn stall(&mut self, d: Duration) {
        match self {
            Clock::Wall(_) => std::thread::sleep(d),
            Clock::Virtual(now) => *now += d.as_secs_f64(),
        }
    }
}

/// Monotone piecewise map between two frame counters (e.g. capture-ring frame → capture frame).
/// Keys are contiguous: every key below `end()` falls into exactly one segment.
#[derive(Default)]
struct SegMap {
    /// `(key_start, value_start, len)`, ascending keys.
    segs: Vec<(u64, u64, u64)>,
}

impl SegMap {
    fn push(&mut self, key: u64, value: u64, len: u64) {
        if len == 0 {
            return;
        }
        if let Some(last) = self.segs.last_mut() {
            // Merge a continuation (the common case: no drop, no trim between two pushes).
            if last.0 + last.2 == key && last.1 + last.2 == value {
                last.2 += len;
                return;
            }
        }
        self.segs.push((key, value, len));
    }

    fn get(&self, key: f64) -> Option<f64> {
        if key < 0.0 {
            return None;
        }
        let k = key.floor() as u64;
        let i = self.segs.partition_point(|s| s.0 <= k).checked_sub(1)?;
        let (ks, vs, len) = self.segs[i];
        (k < ks + len).then_some(vs as f64 + (key - ks as f64))
    }
}

/// One injected stall and how the run recovered from it.
#[derive(Debug, Clone, Serialize)]
struct StallRecord {
    at_s: f64,
    ms: f64,
    /// Tick whose processing the stall delayed.
    tick: u64,
    /// Blocks processed after the stalled one until the waiting capture was within the budget
    /// again (0 = the very next wake).
    recovery_blocks: Option<u64>,
    /// Waiting capture at the first wake after the stall, before and after the budget dropped the
    /// excess.
    waiting_before_ms: Option<f64>,
    waiting_after_ms: Option<f64>,
    /// Time from the stall's end until the output latency was back near its pre-stall median.
    latency_recovery_ms: Option<f64>,
    /// Median output latency before the first stall and over the last second of the run.
    latency_before_p50_ms: Option<f64>,
    latency_end_p50_ms: Option<f64>,
}

/// Everything the loop tracks besides the production components.
struct Sim {
    r_in: f64,
    r_out: f64,
    ch_in: usize,
    block: usize,
    cap_cb: usize,
    cap_period: f64,
    out_cb: usize,
    out_period: f64,
    out_phase: f64,
    /// Next capture / playback callback index.
    cap_next: u64,
    out_next: u64,
    cap_buf: Vec<f32>,
    /// Capture-ring frame → capture frame.
    cap_map: SegMap,
    cap_ring_total: u64,
    /// Capture-ring frame of the oldest backlog frame.
    backlog_front: u64,
    /// Capture-ring frame of each block's first frame, by block index (= harness input / block).
    block_ring_start: Vec<u64>,
    /// Output-ring frame → emitted frame.
    out_map: SegMap,
    out_ring_in: u64,
    out_ring_out: u64,
    /// Emitted frame at the front of the output queue, and total emitted.
    emitted_front: u64,
    emitted_total: u64,
    /// Algorithmic delay in output frames (probe).
    delay_out: f64,
    warmup_s: f64,
    latencies: Vec<(f64, f64)>,
    played_any: bool,
    underrun_frames: u64,
    trimmed_frames: u64,
    audio: Option<Vec<f32>>,
    // ---- P3B-A: the T1 decomposition of every latency sample ----
    /// Start / end (s, run clock) of every tick, the emitted frames after it and the model calls
    /// made by then (both cumulative).
    tick_start: Vec<f64>,
    tick_end: Vec<f64>,
    emit_end: Vec<u64>,
    calls_end: Vec<u64>,
    /// `(playback time, emitted frame)` of every latency sample (decomposed after the run, when
    /// every tick is on record).
    t1_samples: Vec<(f64, f64)>,
}

impl Sim {
    fn capture_time(&self, capture_frame: f64) -> f64 {
        capture_frame / self.r_in
    }

    /// Deliver every capture callback due by `now`.
    fn deliver_capture(&mut self, now: f64, input: &mut InputSide, clip: &[f32]) {
        let clip_frames = clip.len() / self.ch_in;
        while (self.cap_next + 1) as f64 * self.cap_period <= now {
            let first = self.cap_next * self.cap_cb as u64;
            for (i, frame) in self.cap_buf.chunks_exact_mut(self.ch_in).enumerate() {
                let f = ((first + i as u64) % clip_frames as u64) as usize;
                frame.copy_from_slice(&clip[f * self.ch_in..(f + 1) * self.ch_in]);
            }
            let written = input.capture(&self.cap_buf) as u64;
            self.cap_map.push(self.cap_ring_total, first, written);
            self.cap_ring_total += written;
            self.cap_next += 1;
        }
    }

    /// Time of the next capture callback.
    fn next_capture_at(&self) -> f64 {
        (self.cap_next + 1) as f64 * self.cap_period
    }

    /// Capture frame of harness input frame `s` (fractional), if it was captured.
    fn capture_of_input(&self, s: f64) -> Option<f64> {
        if s < 0.0 {
            return None;
        }
        let k = (s / self.block as f64).floor() as usize;
        let start = *self.block_ring_start.get(k)?;
        self.cap_map.get(start as f64 + (s - (k * self.block) as f64))
    }

    /// Capture frame that emitted frame `e` carries (through the harness's algorithmic delay).
    fn capture_of_emitted(&self, e: f64) -> Option<f64> {
        self.capture_of_input((e - self.delay_out) * self.r_in / self.r_out)
    }

    /// Run every playback callback due by `until`.
    fn run_output(&mut self, until: f64, output: &mut OutputSide) {
        loop {
            let t = self.out_phase + self.out_next as f64 * self.out_period;
            if t > until {
                break;
            }
            let played = output.callback(self.out_cb, self.audio.as_mut());
            if played.frames > 0 {
                self.played_any = true;
                if t >= self.warmup_s {
                    let first = self.out_map.get(self.out_ring_out as f64);
                    if let Some((e, cap)) = first.and_then(|e| self.capture_of_emitted(e).map(|c| (e, c))) {
                        self.latencies.push((t, (t - self.capture_time(cap)) * 1000.0));
                        self.t1_samples.push((t, e));
                    }
                }
                self.out_ring_out += played.frames as u64;
            }
            if self.played_any && played.frames < played.requested {
                self.underrun_frames += (played.requested - played.frames) as u64;
            }
            self.out_next += 1;
        }
    }

    /// Where a fixed-block engine's blocks start in the harness's input (device frames): the
    /// offset `o` with engine block `j` = harness input `[j * b - o, (j + 1) * b - o)`, from the
    /// ticks that made each model call (call `j` runs at the first tick whose input completes block
    /// `j`, its lookahead included). `None` when the calls do not fit one offset (a harness that
    /// skipped a call) or there are none.
    fn engine_block_offset(&self, b: f64) -> Option<f64> {
        let blk = self.block as f64;
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut prev = 0u64;
        for (k, &c) in self.calls_end.iter().enumerate() {
            for j in prev..c {
                let end = (j + 1) as f64 * b;
                lo = lo.max(end - (k + 1) as f64 * blk);
                hi = hi.min(end - k as f64 * blk);
            }
            prev = c;
        }
        (prev > 0 && lo <= hi).then_some(0.5 * (lo + hi))
    }

    /// The T1 decomposition of the sample played at `t` that carries emitted frame `e` (see
    /// [`T1Parts`]; the parts sum to its latency exactly). `engine` is a fixed-block engine's block
    /// (device frames) and its offset ([`engine_block_offset`](Self::engine_block_offset)): the
    /// model call that converted the sample is the call of its engine block (otherwise the model
    /// runs on every tick). `None` when part of the path is not on record.
    fn t1_parts(&self, t: f64, e: f64, engine: Option<(f64, f64)>) -> Option<T1Parts> {
        let s = (e - self.delay_out) * self.r_in / self.r_out;
        if s < 0.0 {
            return None;
        }
        let kb = (s / self.block as f64).floor() as usize;
        let cap = self.capture_of_input(s)?;
        // Capture callback k delivers capture frames [k * cb, (k + 1) * cb) at (k + 1) periods.
        let delivered = |frame: f64| ((frame.max(0.0).floor() / self.cap_cb as f64).floor() + 1.0) * self.cap_period;
        let tc = self.capture_time(cap);
        let td = delivered(cap);
        let last_ring = *self.block_ring_start.get(kb)? + self.block as u64 - 1;
        let td_last = delivered(self.cap_map.get(last_ring as f64)?);
        let (ts, te) = (*self.tick_start.get(kb)?, *self.tick_end.get(kb)?);
        let k_emit = self.emit_end.partition_point(|&x| (x as f64) <= e.floor());
        let te_emit = *self.tick_end.get(k_emit)?;
        let k_model = match engine {
            Some((b, o)) => {
                let j = ((s + o) / b).floor().max(0.0) as u64;
                self.calls_end.partition_point(|&c| c <= j).max(kb)
            }
            None => kb,
        };
        let ms = 1000.0;
        let (engine_wait, model, after) = if k_model > kb {
            let (ts_m, te_m) = (*self.tick_start.get(k_model)?, *self.tick_end.get(k_model)?);
            ((ts_m - te) * ms, (te_m - ts_m) * ms, te_m)
        } else {
            (0.0, 0.0, te)
        };
        Some(T1Parts {
            capture: (td - tc) * ms,
            block_wait: (td_last - td) * ms,
            queue: (ts - td_last) * ms,
            compute: (te - ts) * ms,
            engine_wait,
            model,
            hold: (te_emit - after) * ms,
            output: (t - te_emit) * ms,
        })
    }

    /// Account one worker push.
    fn note_push(&mut self, pushed: Pushed, emitted: usize) {
        self.emitted_total += emitted as u64;
        self.emitted_front += pushed.trimmed_frames as u64;
        self.trimmed_frames += pushed.trimmed_frames as u64;
        self.out_map.push(self.out_ring_in, self.emitted_front, pushed.flushed_frames as u64);
        self.out_ring_in += pushed.flushed_frames as u64;
        self.emitted_front += pushed.flushed_frames as u64;
    }
}

/// Latency recovery after each stall: back within `tol` ms of the pre-stall median for five
/// consecutive playback callbacks. Fills every record and returns `stall_latency_recovery_ms`: the
/// worst stall's, where an unrecovered stall counts the rest of the run (a lower bound). `None`
/// (no stalls, or no pre-stall baseline: no latency measured between the warm-up and the first
/// stall) when recovery cannot be judged; it is then never reported as the worst case.
fn latency_recovery(latencies: &[(f64, f64)], records: &mut [StallRecord], clock_s: f64, tol: f64) -> Option<f64> {
    let first_stall_t = records.first()?.at_s;
    let steady: Vec<f64> = latencies.iter().filter(|(t, _)| *t < first_stall_t).map(|x| x.1).collect();
    let end_window: Vec<f64> = latencies.iter().filter(|(t, _)| *t >= clock_s - 1.0).map(|x| x.1).collect();
    let end_p50 = Summary::of(&end_window).map(|s| s.p50);
    let steady = Summary::of(&steady);
    for rec in records.iter_mut() {
        rec.latency_before_p50_ms = steady.map(|s| s.p50);
        rec.latency_end_p50_ms = end_p50;
        if let Some(steady) = steady {
            let end = rec.at_s + rec.ms / 1000.0;
            let after: Vec<&(f64, f64)> = latencies.iter().filter(|(t, _)| *t >= end).collect();
            rec.latency_recovery_ms = after
                .windows(5)
                .find(|w| w.iter().all(|(_, l)| *l <= steady.p50 + tol))
                .map(|w| (w[0].0 - end) * 1000.0);
        }
    }
    // Only with a baseline: without one, "unrecovered" would just mean "never judged".
    steady.map(|_| {
        records
            .iter()
            .map(|r| {
                r.latency_recovery_ms
                    .unwrap_or_else(|| ((clock_s - r.at_s - r.ms / 1000.0) * 1000.0).max(0.0))
            })
            .fold(0.0, f64::max)
    })
}

/// Run on a dedicated thread (the worker priority policy applies to it alone); load generators
/// run beside it for the whole run.
pub(super) fn run_paced(p: &PacedInput<'_>) -> Result<RunOutcome, String> {
    let load = LoadHandle::start_kind(&p.run.load, p.run.gpu_load_kind)?;
    let tally = load.cpu_tally();
    let result = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("sigil-bench-paced".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn_scoped(scope, || paced_worker(p, &tally))
            .map_err(|e| format!("cannot start the benchmark thread: {e}"))?
            .join()
            .map_err(|_| "the paced benchmark panicked".to_string())?
    });
    let load_report = load.stop();
    let mut outcome = result?;
    finish_load_metrics(&mut outcome, &load_report);
    Ok(outcome)
}

/// Record what the load did.
fn finish_load_metrics(outcome: &mut RunOutcome, load: &LoadReport) {
    if let Some(g) = &load.gpu {
        outcome.metrics.insert("gpu_load_duty".into(), g.duty);
    }
    if let Some(c) = &load.cpu {
        outcome.metrics.insert("cpu_load_duty".into(), c.duty);
    }
    if let Some(obj) = outcome.details.as_object_mut() {
        obj.insert("load".into(), serde_json::to_value(load).unwrap_or_default());
    }
}

fn paced_worker(p: &PacedInput<'_>, load_cpu: &LoadCpu) -> Result<RunOutcome, String> {
    let s = p.stream;
    let (r_in, r_out) = (s.input_rate, s.output_rate);
    let ch_in = usize::from(s.input_channels);
    let ch_out = usize::from(s.output_channels);
    let block = s.block_frames as usize;
    let block_ms = s.block_ms();
    let duration = p.run.seconds;
    if p.input.len() < ch_in {
        return Err("the input clip is empty".into());
    }
    let priority = WorkerPriority::from_str_lossy(&s.worker_priority);
    rt_thread::set_worker_priority(priority, p.engine.info.gpu());
    // P3B-A: the worker's priority and EcoQoS state as the run measures it.
    let worker_state = super::gpu_state::thread_state();
    let clock_kind = p.run.clock;
    let virtual_clock = clock_kind != ClockKind::Wall;
    p.engine.set_simulated(virtual_clock);

    let times = Arc::new(Mutex::new(ModelTimes::default()));
    let init0 = Instant::now();
    let harness = VcStreamHarness::new(
        p.engine.handle(times.clone()),
        r_in,
        r_out,
        ch_in,
        ch_out,
        s.extra_samples_16k(),
        s.crossfade_samples(p.engine.info.output_rate),
        block,
    )?;
    let init_ms = init0.elapsed().as_secs_f64() * 1000.0;
    // Warm-up cost is not part of the run's clock.
    let _ = p.engine.take_simulated_cost();
    let mut tick = VcTick::new(harness, s);
    let cap_cb = s.capture_callback_frames();
    let out_cb = s.playback_callback_frames();
    let mut input = InputSide::new(r_in, ch_in, block, s.latency_budget_ms);
    let (prefill_ms, max_queue_ms) = p.engine.info.output_buffer_ms(s);
    // The voice's floor is also the adaptive cushion's, as the worker sets it (`OutputPort::set_limits`).
    let voice_floor_ms = p.engine.info.min_output_buffer_ms.map_or(0.0, |m| m[0]);
    let mut output = OutputSide::new(r_out, ch_out, prefill_ms, max_queue_ms, r_in, ch_in, cap_cb * ch_in)
        .with_cushion(s.output_cushion())
        .with_voice_floor(voice_floor_ms);
    let budget_frames = input.budget_frames();

    let mut sim = Sim {
        r_in: f64::from(r_in),
        r_out: f64::from(r_out),
        ch_in,
        block,
        cap_cb,
        cap_period: cap_cb as f64 / f64::from(r_in),
        out_cb,
        out_period: out_cb as f64 / f64::from(r_out),
        out_phase: 0.5 * out_cb as f64 / f64::from(r_out),
        cap_next: 0,
        out_next: 0,
        cap_buf: vec![0.0; cap_cb * ch_in],
        cap_map: SegMap::default(),
        cap_ring_total: 0,
        backlog_front: 0,
        block_ring_start: Vec::with_capacity((duration / (block_ms / 1000.0)) as usize + 16),
        out_map: SegMap::default(),
        out_ring_in: 0,
        out_ring_out: 0,
        emitted_front: 0,
        emitted_total: 0,
        delay_out: p.probe.frames,
        warmup_s: p.run.warmup_s,
        latencies: Vec::new(),
        played_any: false,
        underrun_frames: 0,
        trimmed_frames: 0,
        audio: p
            .record_audio
            .then(|| Vec::with_capacity((duration * f64::from(r_out)) as usize * ch_out + 4096)),
        tick_start: Vec::with_capacity((duration / (block_ms / 1000.0)) as usize + 16),
        tick_end: Vec::with_capacity((duration / (block_ms / 1000.0)) as usize + 16),
        emit_end: Vec::with_capacity((duration / (block_ms / 1000.0)) as usize + 16),
        calls_end: Vec::with_capacity((duration / (block_ms / 1000.0)) as usize + 16),
        t1_samples: Vec::with_capacity((duration * f64::from(r_out) / out_cb.max(1) as f64) as usize + 16),
    };
    let mut joins = JoinStats::with_capacity(sim.tick_end.capacity());

    let mut stalls = p.run.stalls.clone();
    stalls.sort_by(|a, b| a.at_s.total_cmp(&b.at_s));
    let mut next_stall = 0usize;
    let mut stall_records: Vec<StallRecord> = Vec::new();
    let mut pending_recovery: Option<usize> = None;

    let mut timeline: Vec<TickRecord> = Vec::with_capacity(sim.block_ring_start.capacity());
    let mut work: Vec<f32> = Vec::with_capacity(block * ch_in);
    let (mut dropped_frames, mut overflow_frames, mut drop_events) = (0u64, 0u64, 0u64);
    let mut waiting_max = 0usize;
    let mut tick_errors = 0u64;
    let mut first_error: Option<String> = None;
    let mut vram = VramPeak::default();
    let mut vram_at = Instant::now();
    let mut progress_at = Instant::now();
    let mut cancelled = false;

    let cycles0 = (process_cycles(), thread_cycles());
    let cpu0 = process_cpu_time();
    let load0 = load_cpu.ms();
    let wcpu0 = thread_cpu_time();
    let mut clock = if virtual_clock {
        Clock::Virtual(0.0)
    } else {
        Clock::Wall(Instant::now())
    };

    loop {
        if p.cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break;
        }
        let now = clock.now();
        if now >= duration {
            break;
        }
        sim.deliver_capture(now, &mut input, p.input);
        let waiting_before = input.waiting_frames();
        let dropped = input.wake(block);
        let waiting = input.waiting_frames();
        sim.backlog_front += dropped.frames as u64;
        if dropped.frames > 0 || dropped.overflow_frames > 0 {
            drop_events += 1;
            dropped_frames += dropped.frames as u64 + dropped.overflow_frames;
            overflow_frames += dropped.overflow_frames;
        }
        waiting_max = waiting_max.max(waiting);
        if let Some(i) = pending_recovery {
            let rec = &mut stall_records[i];
            if rec.waiting_before_ms.is_none() {
                rec.waiting_before_ms = Some(waiting_before as f64 * 1000.0 / f64::from(r_in));
                rec.waiting_after_ms = Some(waiting as f64 * 1000.0 / f64::from(r_in));
            }
            if waiting <= budget_frames {
                rec.recovery_blocks = Some(timeline.len() as u64 - rec.tick - 1);
                pending_recovery = None;
            }
        }

        if !input.take_block(&mut work, block) {
            clock.wait_until(sim.next_capture_at().min(duration));
            continue;
        }
        let first_ring = sim.backlog_front;
        sim.block_ring_start.push(first_ring);
        sim.backlog_front += block as u64;
        let t_start = clock.now();
        let age_ms = sim
            .cap_map
            .get((first_ring + block as u64 - 1) as f64)
            .map_or(0.0, |last| (t_start - sim.capture_time(last + 1.0)) * 1000.0);

        let mut stall_ms = 0.0;
        if let Some(st) = stalls.get(next_stall).filter(|st| st.at_s <= t_start) {
            stall_ms = st.ms;
            clock.stall(Duration::from_secs_f64(st.ms / 1000.0));
            stall_records.push(StallRecord {
                at_s: t_start,
                ms: st.ms,
                tick: timeline.len() as u64,
                recovery_blocks: None,
                waiting_before_ms: None,
                waiting_after_ms: None,
                latency_recovery_ms: None,
                latency_before_p50_ms: None,
                latency_end_p50_ms: None,
            });
            pending_recovery = Some(stall_records.len() - 1);
            next_stall += 1;
        }

        let model_before = times.lock().map(|t| t.convert_ms.len()).unwrap_or(0);
        let c0 = Instant::now();
        let out = tick.process(&mut work);
        let measured = c0.elapsed();
        let declared = p.engine.take_simulated_cost();
        // What this block costs on the run's clock (wall: the real time it took, including a
        // sleeping stub's wait; virtual: real compute + the stub's declared cost; simulated: the
        // declared cost only).
        let compute = match clock_kind {
            ClockKind::Wall | ClockKind::Virtual => measured + declared,
            ClockKind::Simulated => declared,
        };
        clock.charge(compute);
        let (converted_ms, calls) = times
            .lock()
            .map(|t| {
                let new = &t.convert_ms[model_before.min(t.convert_ms.len())..];
                (new.iter().sum::<f64>(), new.len() as u32)
            })
            .unwrap_or((0.0, 0));
        let model_ms = match clock_kind {
            ClockKind::Simulated => declared.as_secs_f64() * 1000.0,
            _ => converted_ms + declared.as_secs_f64() * 1000.0,
        };
        let failed = out.error.is_some();
        if let Some(e) = out.error {
            tick_errors += 1;
            first_error.get_or_insert(e);
        }

        let t_end = clock.now();
        if calls > 0 {
            joins.record(tick.harness.last_join());
        }
        sim.run_output(t_end, &mut output);
        let pushed = output.push(&out.audio);
        sim.note_push(pushed, out.owed_frames);
        sim.tick_start.push(t_start);
        sim.tick_end.push(t_end);
        sim.emit_end.push(sim.emitted_total);
        sim.calls_end.push(sim.calls_end.last().copied().unwrap_or(0) + u64::from(calls));
        timeline.push(TickRecord {
            tick: timeline.len() as u64,
            t_ms: t_start * 1000.0,
            age_ms,
            compute_ms: compute.as_secs_f64() * 1000.0,
            model_ms,
            calls,
            stall_ms,
            waiting_ms: waiting as f64 * 1000.0 / f64::from(r_in),
            dropped_ms: (dropped.frames as u64 + dropped.overflow_frames) as f64 * 1000.0 / f64::from(r_in),
            queued_out_ms: output.queued_frames() as f64 * 1000.0 / f64::from(r_out),
            error: failed,
        });

        // Bookkeeping outside the timed region.
        if vram_at.elapsed() >= Duration::from_millis(500) {
            vram_at = Instant::now();
            vram.sample();
        }
        if progress_at.elapsed() >= Duration::from_millis(250) {
            progress_at = Instant::now();
            (p.progress)((t_end / duration).min(1.0), "running on the simulated device clock");
        }
    }
    let clock_s = clock.now().max(1e-9);
    // The device keeps playing to the end of the run.
    sim.run_output(clock_s.max(duration), &mut output);
    // The voice changer's CPU: the process's over the run minus what the load threads used in
    // the same window.
    let process_cpu_ms = match (cpu0, process_cpu_time()) {
        (Some(a), Some(b)) => Some((b.saturating_sub(a).as_secs_f64() * 1000.0 - (load_cpu.ms() - load0)).max(0.0)),
        _ => None,
    };
    let worker_cpu = match (wcpu0, thread_cpu_time()) {
        (Some(a), Some(b)) => Some(b.saturating_sub(a)),
        _ => None,
    };
    let cycles1 = (process_cycles(), thread_cycles());
    if cancelled {
        return Err("cancelled".into());
    }

    let stall_latency_recovery = latency_recovery(&sim.latencies, &mut stall_records, clock_s, 10f64.max(block_ms));

    let mut m = Metrics::new();
    let compute: Vec<f64> = timeline.iter().map(|t| t.compute_ms).collect();
    // A fixed-block model converts one model block every few device blocks: its real-time budget
    // is the longer of the two (the harness holds output back to cover the bursts).
    let budget_ms = p
        .engine
        .fixed_block()
        .map_or(block_ms, |b| block_ms.max(b as f64 / 16.0));
    let basis = super::metrics::budget_basis(&timeline);
    let rtf: Vec<f64> = basis.iter().map(|c| c / budget_ms).collect();
    insert_summary(&mut m, "block_ms", &compute);
    insert_summary(&mut m, "rtf", &rtf);
    insert_summary(&mut m, "call_ms", &super::metrics::call_compute(&timeline));
    // Per model call, like offline runs (the ticks where a fixed-block engine only accumulates
    // would otherwise count as 0 ms).
    let model: Vec<f64> = timeline.iter().filter(|t| t.calls > 0).map(|t| t.model_ms / f64::from(t.calls)).collect();
    insert_summary(&mut m, "model_ms", &model);
    let ages: Vec<f64> = timeline.iter().map(|t| t.age_ms).collect();
    insert_summary(&mut m, "block_age_ms", &ages);
    let lat: Vec<f64> = sim.latencies.iter().map(|x| x.1).collect();
    insert_summary(&mut m, "latency_ms", &lat);
    let ms_in = |frames: f64| frames * 1000.0 / f64::from(r_in);
    let ms_out = |frames: f64| frames * 1000.0 / f64::from(r_out);
    m.insert("blocks".into(), timeline.len() as f64);
    m.insert(
        "over_budget_ratio".into(),
        basis.iter().filter(|&&c| c > budget_ms).count() as f64 / basis.len().max(1) as f64,
    );
    let (warmup_ms, model_errors, skipped, attribution, p95) = times
        .lock()
        .map(|t| {
            insert_components(&mut m, &t.components);
            let a = insert_call_metrics(&mut m, &timeline, &t.calls, p.engine.info.gpu());
            (t.warmup_ms, t.errors, t.skipped, a, p95_shares(&t.calls))
        })
        .unwrap_or_default();
    let join_details = joins.insert(&mut m, p.engine.info.output_rate);
    let latency_p50 = m.get("latency_ms_p50").copied();
    // The T1 decomposition of every latency sample, with a fixed-block engine's calls mapped to
    // its blocks.
    let engine_block = p.engine.fixed_block().map(|b| b as f64 * f64::from(r_in) / 16_000.0);
    let engine_map = engine_block.map(|b| (b, sim.engine_block_offset(b)));
    let t1_engine = engine_map.and_then(|(b, o)| o.map(|o| (b, o)));
    let t1: Vec<T1Parts> = sim.t1_samples.iter().filter_map(|&(t, e)| sim.t1_parts(t, e, t1_engine)).collect();
    let mut t1_details = insert_t1(&mut m, &t1, latency_p50);
    if let (Some(obj), Some((b, o))) = (t1_details.as_object_mut(), engine_map) {
        obj.insert("engine_block_frames".into(), b.into());
        obj.insert("engine_block_offset_frames".into(), o.into());
        if o.is_none() {
            obj.insert("note".into(), "the model calls do not fit one block offset: engine_wait / model are not split from hold".into());
        }
    }
    m.insert("model_errors".into(), model_errors as f64);
    m.insert("algorithmic_latency_ms".into(), p.probe.ms);
    m.insert("engine_load_ms".into(), p.engine.info.load_ms);
    m.insert("harness_init_ms".into(), init_ms);
    m.insert("underruns".into(), output.underruns() as f64);
    m.insert("underrun_ms".into(), ms_out(sim.underrun_frames as f64));
    m.insert("dropped_ms".into(), ms_in(dropped_frames as f64));
    m.insert("drop_events".into(), drop_events as f64);
    m.insert("overflow_ms".into(), ms_in(overflow_frames as f64));
    m.insert("backlog_max_ms".into(), ms_in(waiting_max as f64));
    m.insert(
        "backlog_over_budget_ms".into(),
        ms_in(waiting_max.saturating_sub(budget_frames + block) as f64),
    );
    m.insert("output_trimmed_ms".into(), ms_out(sim.trimmed_frames as f64));
    if !stall_records.is_empty() {
        // An unrecovered stall counts every block processed after it.
        let worst = stall_records
            .iter()
            .map(|r| r.recovery_blocks.unwrap_or(timeline.len() as u64 - r.tick))
            .max()
            .unwrap_or(0);
        m.insert("stall_recovery_blocks".into(), worst as f64);
    }
    // Absent without a pre-stall baseline: a threshold on it then fails as "not measured".
    if let Some(ms) = stall_latency_recovery {
        m.insert("stall_latency_recovery_ms".into(), ms);
    }
    // CPU per second of the run's clock; meaningless when real compute is not on the clock.
    if clock_kind != ClockKind::Simulated {
        if let Some(ms) = process_cpu_ms {
            m.insert("cpu_cores".into(), cores(Duration::from_secs_f64(ms / 1000.0), Duration::from_secs_f64(clock_s)));
            // P3B-A: core-seconds per audio-second of the blocks processed.
            let audio_s = timeline.len() as f64 * block_ms / 1000.0;
            if audio_s > 0.0 {
                m.insert("core_s_per_audio_s".into(), ms / 1000.0 / audio_s);
            }
        }
        if let Some(w) = worker_cpu {
            m.insert("worker_cpu_cores".into(), cores(w, Duration::from_secs_f64(clock_s)));
        }
    }
    let details = serde_json::json!({
        "clock": p.run.clock.as_str(),
        "seconds": duration,
        "clock_seconds": clock_s,
        "block_frames": block,
        "block_ms": block_ms,
        "budget_ms": budget_ms,
        "capture_callback_frames": cap_cb,
        "playback_callback_frames": out_cb,
        "latency_budget_frames": budget_frames,
        "latency_samples": lat.len(),
        "emitted_frames": sim.emitted_total,
        "played_frames": sim.out_ring_out,
        "tick_errors": tick_errors,
        "first_error": first_error,
        "warmup_ms": warmup_ms,
        "algorithmic_latency": p.probe,
        "stalls": stall_records,
        "vram_peak_process_mb": vram.peak_mb,
        "worker_priority": priority.as_str(),
        "silent_ticks": skipped,
        "worker_thread": worker_state,
        "cycles": super::metrics::cycle_details(cycles0, cycles1, attribution.calls),
        "attribution": attribution,
        "p95_shares": p95,
        "joins": join_details,
        "t1": t1_details,
    });
    Ok(RunOutcome {
        metrics: m,
        details,
        timeline,
        audio: sim.audio.map(|a| (a, r_out, ch_out)),
        extra_audio: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::engine::{build_engine, probe_algorithmic_latency};
    use crate::services::bench::signals::{render_input, synth};
    use crate::services::bench::spec::{BenchSpec, RunSpec, SignalKind};

    fn spec(v: serde_json::Value) -> BenchSpec {
        BenchSpec::from_value(v).unwrap()
    }

    /// Output prefill for tests that check accounting rather than timing. Debug builds run the
    /// harness unoptimized and `cargo test` runs tests in parallel, so a block can take tens of
    /// ms; a 35 ms cushion would then underrun, which these tests must not depend on.
    const CUSHION_MS: f64 = 80.0;

    fn paced(spec: &BenchSpec, input: &[f32]) -> RunOutcome {
        paced_with_floor(spec, input, None)
    }

    /// [`paced`] with the engine's output-buffer minimum set (what a factory build reports for a CPU
    /// template voice on long blocks).
    fn paced_with_floor(spec: &BenchSpec, input: &[f32], floor: Option<[f64; 2]>) -> RunOutcome {
        let mut engine = build_engine(&spec.engine, &spec.stream, WorkerPriority::Auto, spec.seed).unwrap();
        engine.info.min_output_buffer_ms = floor;
        let probe = probe_algorithmic_latency(
            &spec.stream,
            engine.info.output_rate,
            engine.info.min_context_samples,
            engine.fixed_block(),
            engine.info.model_delay_ms,
        )
        .unwrap();
        let RunSpec::Paced(run) = &spec.runs[0] else { panic!() };
        let cancel = AtomicBool::new(false);
        run_paced(&PacedInput {
            stream: &spec.stream,
            run,
            engine: &engine,
            input,
            probe,
            record_audio: true,
            cancel: &cancel,
            progress: &|_, _| {},
        })
        .unwrap()
    }

    fn stall_record(at_s: f64, ms: f64) -> StallRecord {
        StallRecord {
            at_s,
            ms,
            tick: 0,
            recovery_blocks: None,
            waiting_before_ms: None,
            waiting_after_ms: None,
            latency_recovery_ms: None,
            latency_before_p50_ms: None,
            latency_end_p50_ms: None,
        }
    }

    /// Recovery is judged against the pre-stall median; without one (no latency measured before
    /// the stall) it is absent, never "the rest of the run".
    #[test]
    fn latency_recovery_needs_a_pre_stall_baseline() {
        // 10 ms callbacks from 0.5 s to 6 s; 60 ms latency, 400 ms from 2.0 s until 2.5 s.
        let lat: Vec<(f64, f64)> = (50..600)
            .map(|i| {
                let t = f64::from(i) / 100.0;
                (t, if (2.0..2.5).contains(&t) { 400.0 } else { 60.0 })
            })
            .collect();
        let mut recs = vec![stall_record(2.0, 200.0)];
        let worst = latency_recovery(&lat, &mut recs, 6.0, 10.7).expect("baseline exists");
        // Back at 60 ms from 2.5 s: 300 ms after the stall ended at 2.2 s.
        assert!((worst - 300.0).abs() < 1e-6, "{worst}");
        assert_eq!(recs[0].latency_recovery_ms, Some(worst));
        assert_eq!(recs[0].latency_before_p50_ms, Some(60.0));

        // Never back: the rest of the run counts (a lower bound).
        let stuck: Vec<(f64, f64)> = lat.iter().map(|&(t, l)| (t, if t >= 2.0 { 400.0 } else { l })).collect();
        let mut recs = vec![stall_record(2.0, 200.0)];
        let worst = latency_recovery(&stuck, &mut recs, 6.0, 10.7).unwrap();
        assert!((worst - 3_800.0).abs() < 1e-6, "{worst}");
        assert_eq!(recs[0].latency_recovery_ms, None);

        // Stall before the first measured latency (0.5 s): no baseline, no metric (it used to
        // report 5600 ms "unrecovered").
        let mut recs = vec![stall_record(0.2, 200.0)];
        assert_eq!(latency_recovery(&lat, &mut recs, 6.0, 10.7), None);
        assert_eq!(recs[0].latency_before_p50_ms, None);
        assert_eq!(recs[0].latency_recovery_ms, None);
        assert_eq!(recs[0].latency_end_p50_ms, Some(60.0));
        assert_eq!(latency_recovery(&lat, &mut [], 6.0, 10.7), None, "no stalls");
    }

    #[test]
    fn segmap_merges_and_looks_up() {
        let mut m = SegMap::default();
        m.push(0, 100, 10);
        m.push(10, 110, 5); // continuation: merged
        m.push(15, 200, 5); // a jump in value (a hole / trim)
        m.push(20, 300, 0); // empty: ignored
        assert_eq!(m.segs.len(), 2);
        assert_eq!(m.get(0.0), Some(100.0));
        assert_eq!(m.get(14.5), Some(114.5));
        assert_eq!(m.get(15.0), Some(200.0));
        assert_eq!(m.get(19.9), Some(204.9));
        assert_eq!(m.get(20.0), None);
        assert_eq!(m.get(-1.0), None);
    }

    /// Steady state on a fast engine: no underruns, no drops, and every played sample is traced
    /// to its capture with a constant latency made of the known parts.
    #[test]
    fn steady_run_has_constant_explained_latency() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 2.0, "compute_jitter_ms": 0.5},
            "stream": {"output_channels": 1, "output_prefill_ms": CUSHION_MS, "output_max_queue_ms": 150},
            "runs": [{"mode": "paced", "name": "p", "seconds": 6, "clock": "simulated"}]
        }));
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        let out = paced(&s, &input);
        let m = &out.metrics;
        assert_eq!(m["underruns"], 0.0, "{m:?}");
        assert_eq!(m["dropped_ms"], 0.0);
        assert_eq!(m["model_errors"], 0.0);
        assert!((m["blocks"] - 6.0 * 48_000.0 / 512.0).abs() <= 2.0, "{}", m["blocks"]);
        // Simulated clock: block time is exactly the stub's declared 2 +- 0.5 ms.
        assert!(m["block_ms_min"] >= 1.5 && m["block_ms_max"] <= 2.5, "{m:?}");
        assert!(!m.contains_key("cpu_cores"), "no CPU figure off the real clock");
        let alg = m["algorithmic_latency_ms"];
        let (p50, p99, min) = (m["latency_ms_p50"], m["latency_ms_p99"], m["latency_ms_min"]);
        // Latency = algorithmic delay + capture wait + compute + output cushion: above the
        // algorithmic delay plus the prefill, below that plus a block, a callback and slack.
        assert!(min >= alg + CUSHION_MS, "min {min} < algorithmic {alg} + prefill");
        assert!(p99 < alg + CUSHION_MS + 10.7 + 10.0 + 15.0, "p99 {p99} (alg {alg})");
        // Constant: with no underrun, drop or trim, every callback plays the frames right after
        // the previous callback's, 10 ms later, so the traced latency does not move at all.
        assert!(m["latency_ms_max"] - min < 0.01, "latency spread {min}..{}", m["latency_ms_max"]);
        assert!(p50 > alg + 5.0, "queueing is visible ({p50} vs {alg})");
        assert_eq!(out.timeline.len() as f64, m["blocks"]);
        let (audio, rate, ch) = out.audio.unwrap();
        assert_eq!((rate, ch), (48_000, 1));
        assert!(audio.len() >= 6 * 48_000 - 480, "{}", audio.len());
    }

    /// P3B-A: the T1 decomposition of a fixed-block engine (40 ms engine blocks on 512-frame device
    /// blocks): the model calls map to the engine blocks with one offset, so the wait for the rest
    /// of the engine block and the model call are split from the hold-back; every sample's parts add
    /// up to its latency, and the median decomposition explains the p50.
    #[test]
    fn fixed_block_latency_splits_into_engine_wait_model_and_hold() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 3.0, "fixed_block_ms": 40},
            "stream": {"output_channels": 1, "output_prefill_ms": CUSHION_MS, "output_max_queue_ms": 200},
            "runs": [{"mode": "paced", "name": "p", "seconds": 8, "clock": "simulated", "warmup_s": 1}]
        }));
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        let out = paced(&s, &input);
        let m = &out.metrics;
        let t1 = &out.details["t1"];
        assert_eq!(t1["engine_block_frames"], 1920.0, "{t1}");
        assert!(t1["engine_block_offset_frames"].is_number(), "the calls fit one block offset: {t1}");
        // 40 ms blocks over 10.67 ms device blocks: most samples wait for the rest of their engine
        // block, and the call that converts them is a 3 ms model tick.
        assert!(m["t1_engine_wait_ms_p50"] > 0.0 && m["t1_engine_wait_ms_max"] <= 45.0, "{m:?}");
        assert!(m["t1_model_ms_max"] <= 3.5, "the model tick is the stub's 3 ms: {m:?}");
        assert!(m["t1_residual_p50_ms"].abs() <= 1.0, "{m:?}");
        // The parts' means add up to the mean latency exactly.
        assert!((t1["sum_of_means_ms"].as_f64().unwrap() - t1["latency_mean_ms"].as_f64().unwrap()).abs() < 1e-6, "{t1}");
        assert!(m["t1_hold_ms_min"] >= -1e-9, "no sample is emitted before its model call: {m:?}");
    }

    /// A CPU template voice on 120 ms blocks (one model call of about 90 ms per block, 512-frame device
    /// blocks): at the shipped 35 / 70 ms output buffer the device runs dry during every call, the
    /// recovery trims each catch-up burst back to the ring target, and well under 70 % of the converted
    /// audio is played (the P7-B finding). With the floor `voice_changer::block_buffer` gives such a
    /// voice (110 / 136 ms, carried as `min_output_buffer_ms`) the same run plays it all. This pins the
    /// failure and the fix end to end through the paced run's `OutputSide` (`OutputPlayback` + the ring).
    #[test]
    fn a_burst_engine_loses_audio_at_the_default_buffer_and_plays_it_with_the_floor() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 90.0, "fixed_block_ms": 120},
            "stream": {"output_channels": 1},
            "runs": [{"mode": "paced", "name": "p", "seconds": 60, "clock": "simulated", "warmup_s": 1}]
        }));
        assert_eq!((s.stream.output_prefill_ms, s.stream.output_max_queue_ms), (35.0, 70.0), "the shipped buffer");
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        let ratio = |o: &RunOutcome| o.details["played_frames"].as_f64().unwrap() / o.details["emitted_frames"].as_f64().unwrap();

        let without = paced_with_floor(&s, &input, None);
        let (m0, r0) = (&without.metrics, ratio(&without));
        assert_eq!(m0["model_errors"], 0.0);
        assert!(r0 < 0.7, "the default buffer plays {r0:.3} of the converted audio");
        assert!(m0["output_trimmed_ms"] > 20_000.0, "the recovery trims the catch-up: {} ms", m0["output_trimmed_ms"]);

        let with = paced_with_floor(&s, &input, Some([110.0, 136.0]));
        let (m1, r1) = (&with.metrics, ratio(&with));
        eprintln!(
            "burst engine, 60 s simulated: 35/70 played {:.1} %, trimmed {:.1} s, dry {:.1} s | 110/136 played {:.2} %, trimmed {:.2} s, dry {:.0} ms, latency p50 {:.0} / p99 {:.0} ms (35/70: {:.0} / {:.0})",
            100.0 * r0,
            m0["output_trimmed_ms"] / 1000.0,
            m0["underrun_ms"] / 1000.0,
            100.0 * r1,
            m1["output_trimmed_ms"] / 1000.0,
            m1["underrun_ms"],
            m1["latency_ms_p50"],
            m1["latency_ms_p99"],
            m0["latency_ms_p50"],
            m0["latency_ms_p99"],
        );
        assert_eq!(m1["model_errors"], 0.0);
        assert!(r1 >= 0.98, "the floor plays {r1:.3} of the converted audio ({:?})", m1);
        assert!(m1["underrun_ms"] < 600.0, "dry output: {} ms ({} underruns)", m1["underrun_ms"], m1["underruns"]);
        assert!(m1["output_trimmed_ms"] < 1_000.0, "{} ms trimmed", m1["output_trimmed_ms"]);
        assert_eq!(m1["dropped_ms"], 0.0);
    }

    /// The traced latency is the real one: with an identity engine, the played audio is the
    /// input delayed by exactly the traced latency (cross-correlation agrees within a sample).
    #[test]
    fn traced_latency_matches_the_played_audio() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 1.0},
            "input": {"kind": "synthetic", "signal": "chirp_train", "seconds": 8},
            "stream": {"output_channels": 1, "output_prefill_ms": CUSHION_MS, "output_max_queue_ms": 150},
            "runs": [{"mode": "paced", "name": "p", "seconds": 6, "clock": "simulated"}]
        }));
        let input = render_input(&s.input, 48_000, 1, 3).unwrap();
        let out = paced(&s, &input);
        let traced = out.metrics["latency_ms_p50"];
        assert!(out.metrics["latency_ms_max"] - out.metrics["latency_ms_min"] < 0.01);
        let (played, _, _) = out.audio.unwrap();
        // Played sample j left the device at out_phase + j / 48k (no underruns); the capture it
        // carries was sampled at (j - lag) / 48k.
        let reference = &input[..48_000 * 3];
        let e = crate::services::bench::e2e_loopback::estimate_delay(reference, &played, 24_000).unwrap();
        let phase_ms = 0.5 * 480.0 / 48.0;
        let measured = e.lag / 48.0 + phase_ms;
        assert!(e.correlation > 0.8, "{e:?}");
        // Two independent measurements of the same thing: the frame-accurate trace and the
        // cross-correlation of what the device actually played.
        assert!((measured - traced).abs() < 0.05, "audio lag {measured} ms vs traced p50 {traced} ms");
    }

    fn stall_run(stream: serde_json::Value) -> RunOutcome {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 3.0},
            "stream": stream,
            "runs": [{"mode": "paced", "name": "p", "seconds": 10, "clock": "simulated",
                      "stalls": [{"at_s": 3.0, "ms": 3000}]}]
        }));
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        paced(&s, &input)
    }

    /// Plan T7: after an injected 3 s stall the backlog is back within the budget at once and
    /// never exceeds budget + one block after enforcement.
    #[test]
    fn a_3s_stall_recovers_within_the_budget() {
        let out = stall_run(serde_json::json!({}));
        let m = &out.metrics;
        let stall = &out.details["stalls"][0];
        assert_eq!(m["stall_recovery_blocks"], 0.0, "{stall}");
        assert_eq!(m["backlog_over_budget_ms"], 0.0, "backlog max {}", m["backlog_max_ms"]);
        assert!(m["backlog_max_ms"] <= 300.0 + 10.7, "{}", m["backlog_max_ms"]);
        // ~3 s of stale input dropped in one step, down to the low-water mark (budget / 2).
        assert!(m["dropped_ms"] > 2_800.0 && m["dropped_ms"] < 3_000.0, "{}", m["dropped_ms"]);
        assert_eq!(m["drop_events"], 1.0);
        assert!(stall["waiting_before_ms"].as_f64().unwrap() > 2_990.0, "{stall}");
        assert_eq!(stall["waiting_after_ms"].as_f64().unwrap(), 150.0, "{stall}");
        // The device starved for the whole stall (one underrun per 10 ms callback).
        assert!(m["underruns"] >= 290.0 && m["underrun_ms"] > 2_900.0, "{m:?}");
        // The stalled block's compute excludes the injected stall.
        let stalled = out.timeline.iter().find(|t| t.stall_ms > 0.0).unwrap();
        assert!(stalled.compute_ms < 100.0 && stalled.stall_ms == 3000.0);
        // Latency recovery is reported either way; unrecovered = the rest of the run.
        let before = stall["latency_before_p50_ms"].as_f64().unwrap();
        let end = stall["latency_end_p50_ms"].as_f64().unwrap();
        match stall["latency_recovery_ms"].as_f64() {
            Some(ms) => assert_eq!(m["stall_latency_recovery_ms"], ms),
            None => {
                assert_eq!(m["stall_latency_recovery_ms"], 4_000.0);
                assert!(end > before + 10.0, "{before} -> {end}");
            }
        }
    }

    /// With the output queue capped near the prefill, the latency comes back to its pre-stall
    /// level within a fraction of a second, and the recovery detector sees it.
    #[test]
    fn stall_latency_recovery_is_detected() {
        let out = stall_run(serde_json::json!({"output_prefill_ms": 35, "output_max_queue_ms": 40}));
        let m = &out.metrics;
        let stall = &out.details["stalls"][0];
        let rec = stall["latency_recovery_ms"].as_f64().unwrap_or_else(|| panic!("not recovered: {stall}"));
        assert!(rec < 500.0, "{stall}");
        assert_eq!(m["stall_latency_recovery_ms"], rec);
        let (before, end) = (stall["latency_before_p50_ms"].as_f64().unwrap(), stall["latency_end_p50_ms"].as_f64().unwrap());
        assert!((end - before).abs() < 10.7, "{before} -> {end}");
    }

    /// A PC that is persistently too slow: the budget drops audio in steps, backlog stays bounded.
    #[test]
    fn an_overloaded_engine_is_bounded_by_the_budget() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 13.0},
            "stream": {"latency_budget_ms": 200},
            "runs": [{"mode": "paced", "name": "p", "seconds": 8, "clock": "simulated"}]
        }));
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        let out = paced(&s, &input);
        let m = &out.metrics;
        assert!(m["over_budget_ratio"] > 0.9, "{}", m["over_budget_ratio"]);
        assert!(m["rtf_p50"] > 1.0);
        assert!(m["dropped_ms"] > 500.0 && m["drop_events"] >= 3.0, "{m:?}");
        assert_eq!(m["backlog_over_budget_ms"], 0.0, "{}", m["backlog_max_ms"]);
        assert!(m["latency_ms_p99"] < 200.0 + 70.0 + 60.0 + 40.0, "{}", m["latency_ms_p99"]);
    }

    #[test]
    fn model_failures_are_concealed_and_counted() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "fail_every": 25},
            "stream": {"output_prefill_ms": CUSHION_MS, "output_max_queue_ms": 150},
            "runs": [{"mode": "paced", "name": "p", "seconds": 3, "clock": "simulated"}]
        }));
        let input = synth(SignalKind::Tones, 48_000, 3.0, 0.3, 1);
        let out = paced(&s, &input);
        let m = &out.metrics;
        assert!(m["model_errors"] >= 10.0, "{}", m["model_errors"]);
        assert_eq!(m["underruns"], 0.0, "a failed block still plays its full length");
        assert!(out.timeline.iter().filter(|t| t.error).count() as f64 == m["model_errors"]);
    }

    #[test]
    fn wall_clock_run_keeps_real_time() {
        let s = spec(serde_json::json!({
            "name": "t",
            "engine": {"kind": "stub", "compute_ms": 1.0},
            "stream": {"output_prefill_ms": CUSHION_MS, "output_max_queue_ms": 150},
            "runs": [{"mode": "paced", "name": "p", "seconds": 1.5, "warmup_s": 0.3}]
        }));
        let input = render_input(&s.input, 48_000, 1, 1).unwrap();
        let t0 = Instant::now();
        let out = paced(&s, &input);
        let wall = t0.elapsed().as_secs_f64();
        assert!((1.5..10.0).contains(&wall), "{wall}");
        let m = &out.metrics;
        // Real time on whatever the PC is doing (tests run in parallel, beside concurrent
        // builds): check the loop's structure, not its timing (1.5 s is 140 blocks on an idle PC;
        // a starved worker converts fewer and the budget drops the rest).
        assert!(m["blocks"] >= 10.0, "{}", m["blocks"]);
        assert_eq!(m["backlog_over_budget_ms"], 0.0);
        assert!(m.contains_key("latency_ms_p99") && m.contains_key("cpu_cores"), "{m:?}");
    }
}
