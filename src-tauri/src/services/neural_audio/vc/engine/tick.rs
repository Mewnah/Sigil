//! One tick of a template voice (split out of `engine/mod.rs` by P3B-S0): content features,
//! retrieval, F0, `enc`, the flowdec window (anchored noise, NSF phase, speaker vector) and the post
//! stage.
//!
//! # Owner (P3B)
//! P3B-D (content plan, F0 overlap, running normalisation, frame mappings). Keep P3B-S0's call
//! sites on every tick path, the overlapped one included: the telemetry stage spans
//! (`neural_audio::telemetry::stage_span`), the probe hooks (`probe.override_content`,
//! `probe.override_f0`, `probe.tap_fed_f0` / `tap_unvoiced`, `probe.record_stages`) and
//! `f0_cache.advance_unvoiced` on the silence-skip path.
//!
//! # P3B-D paths (each with its switch back to the P3 path)
//! - **Content plan** (`content::ContentEncoder::encode_fixed`): on a GPU rung the encoder runs a
//!   fixed-shape session plan at the content window's length (whole-graph DirectML compile,
//!   IOBinding and a CUDA graph per the `io_binding` / `cuda_graphs` levers). Kill switch
//!   `SIGIL_VC_CONTENT_PLAN=0`; `SIGIL_VC_CONTENT_GRAPH=0` keeps the plan but captures no CUDA
//!   graph (the graph's speed against the VRAM it holds).
//! - **F0 overlap** (`overlap::F0Helper`): on a GPU rung the F0 cache's update runs on a helper
//!   thread while this thread runs content and retrieval; the tick joins it before the coarse pitch
//!   and protect. Kill switch `SIGIL_VC_F0_OVERLAP=0`.
//! - **Content window** (`content_stats::ContentWindow`, `plan::ShapePlan::content_window_16k`):
//!   with a content context longer than the stream's context, the encoder sees real left context;
//!   its features are mapped back onto the `enc` window with an offset of whole 50 Hz frames in
//!   the three places that read them (the 2x upsampling onto the 10 ms grid, protect's copy of the
//!   unblended features, retrieval's start frame). Off (the default) the window is the context and
//!   every mapping is the P3 one.
//!
//! Paths are decided at the voice's first tick (`warm_once`, load time), where the plan is built
//! and the helper thread started; ticks after that allocate nothing on these paths.

use std::time::Instant;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::telemetry::{self, Stage};

use super::super::rvc::coarse_pitch;
use super::super::traits::{ConversionCtx, StageRecord};
use super::content::{self, PlanReport};
use super::content_stats::{NormState, PlanSwitch};
use super::noise::{AnchoredNoise, NoiseStream};
use super::overlap;
use super::plan::{CONTENT_HOP_16K, HOP_16K};
use super::{Lever, SigilRvcEngine, TickStats, SILENCE_HANGOVER_MS};

/// Why a tick cannot run after a failed re-plan.
pub(super) const GENERATOR_MISSING: &str = "the voice's generator could not be rebuilt for the stream's shape; restart the voice changer";

/// Benchmark override of the content context (ms) of the benchmark's stream instance: a voice whose
/// request leaves the context automatic (`SigilRvcRequest::content_context_16k` = `None`), that was
/// planned with a stream's geometry (`SigilRvcRequest::stream`, not the offline ceiling) AND whose
/// sessions record under the benchmark's `bench` component prefix. A production voice never reads
/// it (the voice changer's factory also passes a stream geometry, but no `bench` prefix), so a
/// variable left set in a shell cannot change what the app plays; production sets the context
/// through the `content_context_ms` param. Read at the voice's first tick.
pub const CONTENT_CONTEXT_ENV: &str = "SIGIL_VC_CONTENT_CONTEXT_MS";

/// Whether sessions recorded under `prefix` belong to a benchmark instance (`bench`, `bench.*`).
fn is_bench_prefix(prefix: Option<&str>) -> bool {
    prefix.is_some_and(|p| p == "bench" || p.starts_with("bench."))
}

/// The content context (16 kHz samples) the benchmark override `env` asks for: only for the
/// benchmark's stream instance (`bench_stream`), and only a positive, finite number of milliseconds.
fn content_context_override(bench_stream: bool, env: Option<&str>) -> Option<usize> {
    if !bench_stream {
        return None;
    }
    env.and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|ms| ms.is_finite() && *ms > 0.0)
        .map(|ms| (ms * 16.0).round() as usize)
}

fn db_to_rms(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// The content encoder's outcome and notes after its plan was built, joined or refused (load time).
fn note_content_plan(outcomes: &mut [(String, EpOutcome)], warnings: &mut Vec<String>, report: PlanReport) {
    if let Some(o) = outcomes.iter_mut().find(|(c, _)| c == "vc.content") {
        o.1 = report.outcome;
    }
    for n in report.notes {
        if !warnings.contains(&n) {
            warnings.push(n);
        }
    }
}

impl SigilRvcEngine {
    /// Whether the voice's content encoder runs a fixed-shape session plan (P3B-D; after the first
    /// tick, which the load runs). `false` on the CPU, with the kill switch, or when the encoder
    /// file cannot run plans (fp16 I/O) or its plan could not be built (see the warnings).
    pub fn content_runs_fixed_shape(&self) -> bool {
        self.content_stats.runs_plan()
    }

    /// Whether ticks run F0 on the overlap helper (P3B-D; after the first tick).
    pub fn f0_overlapped(&self) -> bool {
        self.f0_helper.active()
    }

    /// Whether the content encoder normalises with running statistics (the `content_norm` lever
    /// in effect; after the first tick).
    pub fn content_norm_active(&self) -> bool {
        self.content_stats.active()
    }

    /// Switch the voice to its encoder's `stream` variant with running statistics (the
    /// `content_norm` lever; load time). The stock encoder's dynamic session is released when
    /// nothing else runs it, so the voice holds one copy of the encoder.
    pub(super) fn enable_content_norm(&mut self, sv: content::StreamVariant) -> Result<(), String> {
        let running = super::content_stats::RunningNorm::new(&sv.conv0)?;
        let resolved = content::ResolvedEncoder {
            path: sv.path,
            variant: sv.variant,
            cpu_fallback: sv.cpu_fallback,
        };
        let encoder = content::ContentEncoderPool::global().acquire_resolved(self.embedder, &resolved, &self.ep, &self.session)?;
        {
            let enc = encoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if !enc.takes_norm() {
                return Err(format!("{} does not take normalisation statistics", resolved.path.display()));
            }
            if enc.channels() != self.emb {
                return Err(format!("the stream variant produces {}-channel features, the voice uses {}", enc.channels(), self.emb));
            }
            if let Some(o) = self.outcomes.iter_mut().find(|(c, _)| c == "vc.content") {
                o.1 = enc.outcome().clone();
            }
        }
        if let Some(stock) = self.encoder.as_ref() {
            stock.lock().unwrap_or_else(std::sync::PoisonError::into_inner).park();
        }
        self.content_stats.norm = Some(super::content_stats::NormState {
            encoder,
            lease: None,
            running,
        });
        Ok(())
    }

    /// Speaker vector of this tick into `self.g`.
    fn speaker_vector(&mut self, speaker_id: i64) -> Result<(), String> {
        self.g.resize(self.speakers.dim(), 0.0);
        match &self.blend {
            Some(w) => self.speakers.blend(w, &mut self.g),
            None => {
                self.g.copy_from_slice(self.speakers.vector(speaker_id));
                Ok(())
            }
        }
    }

    /// Decide the voice's content and F0 paths (its first tick: load time). The content encoder's
    /// rung decides both: on the CPU (the Streamer) content keeps its dynamic session and F0 runs
    /// after it on this thread.
    fn resolve_tick_paths(&mut self) -> Result<(), String> {
        if self.content_stats.plan != PlanSwitch::Unresolved {
            return Ok(());
        }
        let active = self
            .encoder
            .as_ref()
            .ok_or("the content encoder was released")?
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .outcome()
            .active
            .clone();
        let gpu = content::plan_rung(&active);
        self.content_stats.plan = if gpu && content::content_plan_switch() { PlanSwitch::On } else { PlanSwitch::Off };
        self.content_stats.no_graph = !content::content_graph_switch();
        if self.content_context_16k.is_none() {
            // Benchmark override of the content context (the `content_context_ms` param reaches the
            // engine through the factory; the benchmark's spec has no field for it yet). Only the
            // benchmark's stream instance takes it: its offline ceiling keeps its own shapes, and a
            // production voice (no `bench` prefix) does not even read the variable.
            let bench_stream = self.stream_geometry.is_some() && is_bench_prefix(self.session.component_prefix);
            let env = if bench_stream { std::env::var(CONTENT_CONTEXT_ENV).ok() } else { None };
            if let Some(ctx) = content_context_override(bench_stream, env.as_deref()) {
                self.content_context_16k = Some(ctx);
            }
        }
        if self.levers.levers.get(Lever::ContentNorm) && self.content_stats.norm.is_none() {
            let enabled = content::resolve_stream_variant(self.embedder, active != "cpu", self.levers.precision).and_then(|sv| self.enable_content_norm(sv));
            if let Err(why) = enabled {
                self.levers.levers.set(Lever::ContentNorm, false);
                self.levers
                    .notes
                    .push(format!("content_norm: {why}; each window normalises itself"));
            }
        }
        let (on, why) = if !self.has_f0 {
            (false, Some("the voice has no pitch guidance"))
        } else if !gpu {
            (false, Some("the content encoder runs on the CPU (F0 and content would share its cores)"))
        } else if !overlap::f0_overlap_switch() {
            (false, Some("switched off (SIGIL_VC_F0_OVERLAP=0)"))
        } else {
            (true, None)
        };
        self.f0_helper.configure(on, why);
        Ok(())
    }

    pub(super) fn convert_inner(&mut self, ctx: &ConversionCtx<'_>, allow_skip: bool) -> Result<Vec<f32>, String> {
        let want = ctx.new_frames.min(ctx.mono_16k.len());
        if want == 0 {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let out_len = ((want as u64 * u64::from(self.output_sample_rate) + 8_000) / 16_000) as usize;
        let c = self.plan.context_16k;
        // The fixed shape; any other call (one-off conversions) is cut / zero-padded to it.
        let direct = ctx.mono_16k.len() == c && ctx.stream_pos_16k.is_multiple_of(HOP_16K as u64);
        let mut buf = std::mem::take(&mut self.ctx_buf);
        let (mono, pos): (&[f32], u64) = if direct {
            (ctx.mono_16k, ctx.stream_pos_16k)
        } else {
            let len = ctx.mono_16k.len();
            buf.clear();
            if len >= c {
                buf.extend_from_slice(&ctx.mono_16k[len - c..]);
            } else {
                buf.resize(c - len, 0.0);
                buf.extend_from_slice(ctx.mono_16k);
            }
            let end = ctx.stream_pos_16k + len as u64;
            let pos = end.saturating_sub(c as u64);
            (&buf, pos - pos % HOP_16K as u64)
        };
        let result = self.convert_tick(ctx, mono, pos, want, out_len, allow_skip);
        self.ctx_buf = buf;
        let mut out = result?;
        self.stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
        if !self.stats.skipped {
            let _span = telemetry::stage_span(Stage::Post);
            // The input span the returned audio was converted from (earlier by the lookahead).
            let len = ctx.mono_16k.len();
            let end = len.saturating_sub(self.plan.lookahead_frames * HOP_16K);
            let source = &ctx.mono_16k[end.saturating_sub(want)..end];
            self.post.process(source, &mut out, &self.controls)?;
        }
        if self.probe.active() {
            let s = self.stats;
            self.probe.record_stages(&StageRecord {
                content_ms: s.content_ms,
                retrieval_ms: 0.0,
                f0_ms: s.f0_ms,
                enc_ms: s.enc_ms,
                flowdec_ms: s.flowdec_ms,
                post_ms: 0.0,
                total_ms: s.total_ms,
                f0_frames: s.f0_frames,
                skipped: s.skipped,
                stream_pos_16k: ctx.stream_pos_16k,
            });
        }
        Ok(out)
    }

    fn convert_tick(
        &mut self,
        ctx: &ConversionCtx<'_>,
        mono: &[f32],
        pos: u64,
        want: usize,
        out_len: usize,
        allow_skip: bool,
    ) -> Result<Vec<f32>, String> {
        self.stats = TickStats::default();
        // The voice's content and F0 paths (decided once, at its first tick: load time).
        self.resolve_tick_paths()?;
        let p = self.plan;
        // The content encoder's window: the context, or more with a content context (P3B-D).
        let cw = p.content_window_16k(self.content_context_16k);
        // Silence skip: no inference while the returned span is quiet (after a short hangover).
        if allow_skip && self.silence_skip {
            let tail = &mono[mono.len() - want.min(mono.len())..];
            let rms = (tail
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / tail.len().max(1) as f64)
                .sqrt() as f32;
            if rms < db_to_rms(self.controls.silence_db) {
                self.quiet_ticks = self.quiet_ticks.saturating_add(1);
                let block_ms = self.plan.block_16k as f64 / 16.0;
                let hangover = (SILENCE_HANGOVER_MS / block_ms).ceil() as u32;
                if self.quiet_ticks > hangover {
                    self.stats.skipped = true;
                    let first = pos / HOP_16K as u64;
                    if self.has_f0 {
                        // The block's frames, unvoiced, at their absolute positions (P3B-F).
                        let block = p.block_16k / HOP_16K;
                        self.f0_cache
                            .advance_unvoiced(first + p.enc_frames.saturating_sub(block) as u64, block);
                    }
                    // The content window's history and the running statistics stay continuous
                    // through skipped ticks (a short pause belongs to the statistics' slice).
                    self.content_stats.ring.advance(mono, pos, cw);
                    if let Some(n) = self.content_stats.norm.as_mut() {
                        n.running.advance(mono, pos);
                    }
                    if self.probe.active() {
                        let end_f = p.enc_frames.saturating_sub(p.lookahead_frames);
                        let k = out_len.div_ceil(self.upp.max(1)).min(end_f);
                        self.probe.tap_unvoiced(first + (end_f - k) as u64, k);
                    }
                    return Ok(vec![0.0; out_len]);
                }
            } else {
                self.quiet_ticks = 0;
            }
        }
        let (t, emb) = (p.enc_frames, self.emb);
        // 50 Hz frames the content window adds before the context (0 without a content context).
        let off = p.content_offset_frames50(cw);

        // 0. F0 on the helper thread while this thread runs the content encoder (GPU rungs).
        let whole_window = !self.levers.levers.get(Lever::F0Cache);
        let job = if self.has_f0 {
            self.f0_helper
                .post(&mut self.pitch, &mut self.f0_cache, mono, pos, whole_window, t)
        } else {
            None
        };
        let overlapped = job.is_some();

        // 1. Content features over the content window.
        let t0 = Instant::now();
        let frames50 = {
            let _span = telemetry::stage_span(Stage::Content);
            let window = self.content_stats.ring.window(mono, pos, cw);
            let plan_on = self.content_stats.plan == PlanSwitch::On;
            let mut levers = content::plan_levers(&self.levers);
            if self.content_stats.no_graph {
                levers.cuda_graphs = false;
            }
            // `content_norm`: the encoder's stream variant with the running statistics; else the
            // stock encoder (each window normalises itself).
            let (shared, lease, stats) = match self.content_stats.norm.as_mut() {
                Some(NormState { encoder, lease, running }) => (&*encoder, lease, Some(running.update(mono, pos, cw / CONTENT_HOP_16K))),
                None => (
                    self.encoder
                        .as_ref()
                        .ok_or("the content encoder was released")?,
                    &mut self.content_stats.lease,
                    None,
                ),
            };
            let mut enc = shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((mean, inv_std)) = stats {
                enc.set_norm(mean, inv_std)?;
            }
            if plan_on {
                let (n, report) = enc.encode_fixed(window, levers, lease, &mut self.feats)?;
                if let Some(r) = report {
                    note_content_plan(&mut self.outcomes, &mut self.warnings, r);
                }
                n
            } else {
                enc.encode(window, &mut self.feats)?
            }
        };
        if frames50 <= off {
            return Err(format!("the content encoder returned {frames50} frames for a window of {off} frames of left context and the context"));
        }
        if self.probe.active() {
            // The context's frames (50 Hz frame `k` covers `pos + 320 k`), as before P3B-D.
            self.probe
                .override_content(pos, &mut self.feats[off * emb..frames50 * emb], frames50 - off, emb);
        }
        self.stats.content_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let protect_on = self.has_f0 && ctx.protect < 0.5;
        if protect_on {
            self.feats0.clear();
            self.feats0
                .extend_from_slice(&self.feats[off * emb..frames50 * emb]);
        }
        // 2. Retrieval on the frames that reach the flowdec window.
        if let Some(r) = self.retrieval.as_mut() {
            if self.retrieval_on && ctx.index_rate > 0.0 {
                let _span = telemetry::stage_span(Stage::Retrieval);
                let skip = (off + p.window_start / 2).min(frames50 - 1);
                r.blend(&mut self.feats[skip * emb..frames50 * emb], frames50 - skip, ctx.index_rate)?;
            }
        }
        // 3. 2x upsample onto the 10 ms grid (RVC realtime: duplicate the last 50 Hz frame).
        upsample_into(&self.feats[off * emb..frames50 * emb], emb, t, &mut self.phone);
        let ctx_frames50 = frames50 - off;

        // 4. F0 (cached) → shifted Hz and coarse bins; protect.
        let overlapped_f0 = job.map(|j| j.join(&mut self.f0_ctx));
        if self.has_f0 {
            let t1 = Instant::now();
            match overlapped_f0 {
                Some(r) => {
                    let st = r?;
                    self.stats.f0_frames = st.extracted;
                    self.stats.f0_ms = st.ms;
                }
                None => {
                    {
                        let _span = telemetry::stage_span(Stage::F0);
                        let f0 = if whole_window {
                            self.f0_cache.extract_window(self.pitch.as_mut(), mono)?
                        } else {
                            self.f0_cache.update(self.pitch.as_mut(), mono, pos)?
                        };
                        if f0.len() < t {
                            return Err(format!("the F0 contour has {} frames, the context {t}", f0.len()));
                        }
                        self.f0_ctx.copy_from_slice(&f0[..t]);
                    }
                    self.stats.f0_frames = self.f0_cache.last_extracted_frames();
                }
            }
            if self.probe.active() {
                self.probe
                    .override_f0(pos / HOP_16K as u64, &mut self.f0_ctx);
            }
            if !overlapped {
                self.stats.f0_ms = t1.elapsed().as_secs_f64() * 1000.0;
            }
            let shift = 2f32.powf(ctx.pitch_shift_semitones / 12.0);
            for ((pf, cb), &hz) in self
                .pitchf
                .iter_mut()
                .zip(self.coarse.iter_mut())
                .zip(self.f0_ctx.iter())
            {
                let v = hz * shift;
                *pf = if v.is_finite() && v > 0.0 { v } else { 0.0 };
                *cb = coarse_pitch(*pf);
            }
            if protect_on {
                let protect = ctx.protect.max(0.0);
                let last = ctx_frames50 - 1;
                for (j, &pf) in self.pitchf.iter().enumerate() {
                    if pf < 1.0 {
                        let src = (j / 2).min(last) * emb;
                        for (x, &x0) in self.phone[j * emb..(j + 1) * emb]
                            .iter_mut()
                            .zip(&self.feats0[src..src + emb])
                        {
                            *x = *x * protect + x0 * (1.0 - protect);
                        }
                    }
                }
            }
        }

        // 5. enc over the context.
        let t2 = Instant::now();
        {
            let _span = telemetry::stage_span(Stage::Enc);
            self.generator
                .as_mut()
                .ok_or(GENERATOR_MISSING)?
                .encode(&self.phone, &self.coarse)?;
        }
        self.stats.enc_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 6. The flowdec window: anchored noise, NSF phase, speaker vector.
        let (ws, f) = (p.window_start, p.flow_frames);
        let abs_ws = pos / HOP_16K as u64 + ws as u64;
        let noise_scale = if self.controls.noise_scale.is_finite() {
            self.controls.noise_scale.max(0.0)
        } else {
            0.0
        };
        if self.levers.levers.get(Lever::AnchoredNoise) {
            self.z_frames
                .copy_window(abs_ws, f, noise_scale, true, &mut self.z);
            self.src_frames
                .copy_window(abs_ws, f, 1.0, false, &mut self.src);
        } else {
            // Legacy: fresh noise every tick.
            let fresh = AnchoredNoise::new(
                self.noise_seed
                    ^ self
                        .tick
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(1),
            );
            fresh.fill(NoiseStream::Flow, 0, noise_scale, &mut self.z);
            fresh.fill(NoiseStream::Source, 0, 1.0, &mut self.src);
        }
        let phase0 = if self.has_f0 && self.levers.levers.get(Lever::NsfPhase0) {
            match self.phase_frame {
                Some(pf) if pf == abs_ws => {}
                Some(pf) if pf < abs_ws && pf >= self.prev_f0_start && abs_ws <= self.prev_f0_start + self.prev_f0.len() as u64 => {
                    let from = (pf - self.prev_f0_start) as usize;
                    let to = (abs_ws - self.prev_f0_start) as usize;
                    self.phase.advance(&self.prev_f0[from..to]);
                }
                _ => self.phase.reset(),
            }
            self.phase_frame = Some(abs_ws);
            self.phase.phase_at()
        } else {
            0.0
        };
        self.speaker_vector(ctx.speaker_id)?;

        // 7. flowdec over the window (partial decode) or the whole context (legacy).
        let t3 = Instant::now();
        {
            let _span = telemetry::stage_span(Stage::Flowdec);
            self.generator
                .as_mut()
                .ok_or(GENERATOR_MISSING)?
                .decode_window(ws, &self.z, &self.g, &self.pitchf[ws..ws + f], phase0, &self.src, &mut self.audio)?;
        }
        self.stats.flowdec_ms = t3.elapsed().as_secs_f64() * 1000.0;
        if self.has_f0 {
            self.prev_f0.clear();
            self.prev_f0.extend_from_slice(&self.pitchf[ws..ws + f]);
            self.prev_f0_start = abs_ws;
        }
        self.tick = self.tick.wrapping_add(1);
        // Lookahead: the newest decoded frames are context only.
        let end = self
            .audio
            .len()
            .saturating_sub(p.lookahead_frames * self.upp);
        let n = out_len.min(end);
        if self.has_f0 && self.probe.active() {
            // The F0 the returned frames were decoded with, on the absolute grid.
            let end_f = t.saturating_sub(p.lookahead_frames);
            let k = n.div_ceil(self.upp.max(1)).min(end_f);
            self.probe
                .tap_fed_f0(pos / HOP_16K as u64 + (end_f - k) as u64, &self.pitchf[end_f - k..end_f]);
        }
        Ok(self.audio[end - n..end].to_vec())
    }
}

/// Nearest 2x upsampling of 50 Hz features onto `t` 10 ms frames into `out` (`t × emb`): frame `j`
/// takes 50 Hz frame `min(j / 2, last)` (RVC realtime's duplicated last frame + interpolate ×2).
/// `feats` starts at the context's first 50 Hz frame (a content window with left context passes
/// its features from the offset on).
pub(super) fn upsample_into(feats: &[f32], emb: usize, t: usize, out: &mut Vec<f32>) {
    out.resize(t * emb, 0.0);
    let frames = feats.len() / emb.max(1);
    if frames == 0 {
        out.fill(0.0);
        return;
    }
    for j in 0..t {
        let src = (j / 2).min(frames - 1) * emb;
        out[j * emb..(j + 1) * emb].copy_from_slice(&feats[src..src + emb]);
    }
}

// The mappings above pair 10 ms frame `j` with content frame `j / 2`: a content frame covers two
// 10 ms frames.
const _: () = assert!(CONTENT_HOP_16K == 2 * HOP_16K, "a content frame covers two 10 ms frames");

#[cfg(test)]
mod tests {
    use super::*;

    /// The content-context environment override is for the benchmark's stream instance only: a
    /// production voice (no `bench` component prefix) neither reads nor honours it, and the offline
    /// ceiling (no stream geometry) is not a bench stream.
    #[test]
    fn content_context_env_override_is_for_the_bench_stream_only() {
        assert!(is_bench_prefix(Some("bench")) && is_bench_prefix(Some("bench.ceiling")));
        assert!(!is_bench_prefix(None) && !is_bench_prefix(Some("")) && !is_bench_prefix(Some("governor")));
        assert!(!is_bench_prefix(Some("benchmark")), "a prefix that merely starts with the letters");
        // A bench stream takes a positive, finite number of milliseconds (16 samples each).
        assert_eq!(content_context_override(true, Some("1000")), Some(16_000));
        assert_eq!(content_context_override(true, Some(" 62.5 ")), Some(1_000));
        for bad in ["0", "-5", "abc", "", "inf", "NaN"] {
            assert_eq!(content_context_override(true, Some(bad)), None, "{bad:?}");
        }
        assert_eq!(content_context_override(true, None), None);
        // Anything else (a production voice with the variable set in its environment) ignores it.
        assert_eq!(content_context_override(false, Some("1000")), None);
        assert_eq!(content_context_override(false, None), None);
    }
}
