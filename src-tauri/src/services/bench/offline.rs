//! Offline driver: the whole input through the production harness as fast as the engine goes.
//!
//! Every block goes through the same processing as a live voice-changer block ([`VcTick`]:
//! gain, harness conversion, concealment, formant tilt), fed in the worker's block size, so the
//! per-block compute distribution, RTF and throughput describe exactly the work a live run does,
//! without device timing. Latency comes from the identity probe (the harness's algorithmic delay).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::services::neural_audio::VcStreamHarness;
use crate::services::rt_thread::{self, WorkerPriority};

use super::ceiling::StreamRecorder;
use super::engine::{AlgorithmicLatency, Engine, ModelTimes};
use super::metrics::{cores, insert_components, insert_summary, process_cpu_time, thread_cpu_time, Metrics};
use super::report::{RunOutcome, TickRecord};
use super::spec::{OfflineRun, StreamSpec};
use super::vc_rt::VcTick;

pub(super) struct OfflineInput<'a> {
    pub stream: &'a StreamSpec,
    pub run: &'a OfflineRun,
    pub engine: &'a Engine,
    /// Interleaved at the stream's input rate / channels.
    pub input: &'a [f32],
    pub probe: AlgorithmicLatency,
    pub record_audio: bool,
    pub cancel: &'a AtomicBool,
    pub progress: &'a (dyn Fn(f64, &str) + Sync),
    /// Record the model's 16 kHz input stream (the quality run's ceiling converts it).
    pub recorder: Option<Arc<Mutex<StreamRecorder>>>,
}

/// Run on a dedicated thread (the worker priority policy applies to it alone).
pub(super) fn run_offline(p: &OfflineInput<'_>) -> Result<RunOutcome, String> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("sigil-bench-offline".into())
            .stack_size(super::BENCH_STACK_BYTES)
            .spawn_scoped(scope, || offline_worker(p))
            .map_err(|e| format!("cannot start the benchmark thread: {e}"))?
            .join()
            .map_err(|_| "the offline benchmark panicked".to_string())?
    })
}

fn offline_worker(p: &OfflineInput<'_>) -> Result<RunOutcome, String> {
    let s = p.stream;
    let ch_in = usize::from(s.input_channels);
    let ch_out = usize::from(s.output_channels);
    let block = s.block_frames as usize;
    let block_ms = s.block_ms();
    let priority = WorkerPriority::from_str_lossy(&s.worker_priority);
    rt_thread::set_worker_priority(priority, p.engine.info.gpu());
    p.engine.set_simulated(false);

    let times = Arc::new(Mutex::new(ModelTimes::default()));
    let init0 = Instant::now();
    let harness = VcStreamHarness::new(
        p.engine.handle_recording(times.clone(), p.recorder.clone()),
        s.input_rate,
        s.output_rate,
        ch_in,
        ch_out,
        s.extra_samples_16k(),
        s.crossfade_samples(p.engine.info.output_rate),
        block,
    )?;
    let init_ms = init0.elapsed().as_secs_f64() * 1000.0;
    let mut tick = VcTick::new(harness, s);

    let blocks_per_pass = p.input.len() / ch_in / block;
    if blocks_per_pass == 0 {
        return Err(format!("the input is shorter than one block ({block} frames)"));
    }
    let total = blocks_per_pass * p.run.repeat as usize;
    let mut timeline = Vec::with_capacity(total);
    let mut audio = p.record_audio.then(|| Vec::with_capacity(p.input.len() / ch_in * ch_out * 2));
    let mut work = Vec::with_capacity(block * ch_in);
    let mut errors = 0u64;
    let mut first_error: Option<String> = None;
    let mut compute_total = 0.0f64;
    let cpu0 = process_cpu_time();
    let wcpu0 = thread_cpu_time();
    let wall0 = Instant::now();
    let mut last_progress = Instant::now();
    let mut cancelled = false;
    'passes: for pass in 0..p.run.repeat as usize {
        for b in 0..blocks_per_pass {
            if p.cancel.load(Ordering::Relaxed) {
                cancelled = true;
                break 'passes;
            }
            let start = b * block * ch_in;
            work.clear();
            work.extend_from_slice(&p.input[start..start + block * ch_in]);
            let model_before = times.lock().map(|t| t.convert_ms.len()).unwrap_or(0);
            let c0 = Instant::now();
            let out = tick.process(&mut work);
            let compute_ms = c0.elapsed().as_secs_f64() * 1000.0;
            compute_total += compute_ms;
            let (model_ms, calls) = times
                .lock()
                .map(|t| {
                    let new = &t.convert_ms[model_before.min(t.convert_ms.len())..];
                    (new.iter().sum(), new.len() as u32)
                })
                .unwrap_or((0.0, 0));
            debug_assert_eq!(out.audio.len(), out.owed_frames * ch_out);
            let failed = out.error.is_some();
            if let Some(e) = out.error {
                errors += 1;
                first_error.get_or_insert(e);
            }
            timeline.push(TickRecord {
                tick: (pass * blocks_per_pass + b) as u64,
                t_ms: c0.duration_since(wall0).as_secs_f64() * 1000.0,
                compute_ms,
                model_ms,
                calls,
                error: failed,
                ..TickRecord::default()
            });
            if let Some(a) = audio.as_mut() {
                a.extend_from_slice(&out.audio);
            }
            if last_progress.elapsed().as_millis() >= 250 {
                last_progress = Instant::now();
                (p.progress)(timeline.len() as f64 / total as f64, "converting");
            }
        }
    }
    let wall = wall0.elapsed();
    if cancelled {
        return Err("cancelled".into());
    }

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
    let (model_ms, warmup_ms, model_errors, skipped) = {
        let t = times.lock().map_err(|_| "model timings poisoned")?;
        insert_components(&mut m, &t.components);
        (t.convert_ms.clone(), t.warmup_ms, t.errors, t.skipped)
    };
    insert_summary(&mut m, "model_ms", &model_ms);
    m.insert("blocks".into(), timeline.len() as f64);
    m.insert(
        "over_budget_ratio".into(),
        basis.iter().filter(|&&c| c > budget_ms).count() as f64 / basis.len().max(1) as f64,
    );
    m.insert("model_errors".into(), model_errors as f64);
    m.insert("algorithmic_latency_ms".into(), p.probe.ms);
    m.insert("engine_load_ms".into(), p.engine.info.load_ms);
    m.insert("harness_init_ms".into(), init_ms);
    let audio_s = timeline.len() as f64 * block_ms / 1000.0;
    if compute_total > 0.0 {
        m.insert("throughput_x".into(), audio_s / (compute_total / 1000.0));
    }
    if let (Some(a), Some(b)) = (cpu0, process_cpu_time()) {
        m.insert("cpu_cores".into(), cores(b.saturating_sub(a), wall));
    }
    if let (Some(a), Some(b)) = (wcpu0, thread_cpu_time()) {
        m.insert("worker_cpu_cores".into(), cores(b.saturating_sub(a), wall));
    }
    let details = serde_json::json!({
        "blocks_per_pass": blocks_per_pass,
        "passes": p.run.repeat,
        "block_ms": block_ms,
        "budget_ms": budget_ms,
        "audio_seconds": audio_s,
        "wall_ms": wall.as_secs_f64() * 1000.0,
        "warmup_ms": warmup_ms,
        "tick_errors": errors,
        "first_error": first_error,
        "algorithmic_latency": p.probe,
        "worker_priority": priority.as_str(),
        "silent_ticks": skipped,
    });
    Ok(RunOutcome {
        metrics: m,
        details,
        timeline,
        audio: audio.map(|a| (a, s.output_rate, ch_out)),
        extra_audio: Vec::new(),
    })
}
