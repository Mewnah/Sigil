//! CUDA graphs captured per fixed shape (P3-C2; the `cuda_graphs` lever, needs IOBinding): one
//! graph launch per tick instead of hundreds of small kernel launches (the HiFiGAN decoder alone
//! launches several hundred) and no ONNX Runtime call at all per tick.
//!
//! # Sigil's own capture, not ONNX Runtime's
//! ONNX Runtime's `enable_cuda_graph` begins its capture in `cudaStreamCaptureModeGlobal`: while it
//! records, every *other* thread of the process is refused "unsafe" CUDA calls (measured on the dev
//! rig: the benchmark's GPU load session failed with `CUDA failure 900: operation not permitted when
//! stream is capturing`, and the plan's capture was invalidated). In Sigil those threads can be a
//! CUDA speech engine (ggml aborts the process on a CUDA error) or another voice-engine session. So
//! plans record the graph themselves ([`CapturedGraph::capture`]):
//!
//! 1. the session runs on the plan's own stream (`user_compute_stream`) with device-resident,
//!    bound I/O at fixed addresses;
//! 2. two regular runs ([`WARMUP_RUNS`]) settle the arena and pick the cuDNN algorithms;
//! 3. one run with device synchronisation disabled is recorded in
//!    `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` (only the capturing thread is restricted) and
//!    instantiated;
//! 4. the check: regular runs on two non-zero pseudo-random input patterns give the references,
//!    then each pattern is replayed with the device outputs first set to a NaN sentinel, and every
//!    replay must reproduce its reference ([`outputs_match`]; every output must have a finite
//!    reference value). A graph that recorded nothing (ORT's kernels went to another stream) leaves
//!    the sentinel; one that recorded part of the run replays the first pattern on intermediates the
//!    second regular run left; one that baked in host-side values (a node that ran on the CPU)
//!    differs on the second pattern. A failed check leaves the plan without a graph, on bound host
//!    tensors (with a note); afterwards the inputs are zeros again.
//!
//! Each tick then uploads the written inputs, launches the graph and downloads the read outputs on
//! the same stream, with one synchronisation. The session's arena belongs to that session alone
//! (graph plans never use the shared environment arena), so the intermediate buffers the graph
//! captured are never handed to anyone else. Shapes are fixed per plan: a preset change builds new
//! plans, whose build captures new graphs ("recaptured when the preset changes"), and a stale graph
//! can never replay with the wrong shapes.
//!
//! # Frozen by P3-S0
//! - [`available`]`() -> bool` (cheap; `voice_changer::capabilities` calls it).

use std::sync::Arc;

use super::cuda_driver::{CuGraphExec, CuStream, PrimaryContext};

/// Regular runs before the capture run.
pub const WARMUP_RUNS: usize = 2;

/// Largest difference between a replay and the reference run, relative to the output's peak (the
/// same kernels on the same inputs; cuDNN may pick another algorithm between runs only if its
/// search is not settled).
pub const REPLAY_TOLERANCE: f32 = 1e-3;

/// Whether session plans can capture CUDA graphs in this build: an NVIDIA adapter is present (the
/// capture and device copies need the CUDA driver; checked without loading it). Whether a given
/// graph can be captured is only known when its plan is built.
pub fn available() -> bool {
    super::cuda_stream::nvidia_adapter_present()
}

/// Whether a replay's `output` equals the regular run's `reference` within [`REPLAY_TOLERANCE`].
pub fn outputs_match(reference: &[f32], output: &[f32]) -> bool {
    if reference.len() != output.len() {
        return false;
    }
    let peak = reference.iter().fold(1f32, |m, x| m.max(x.abs()));
    reference
        .iter()
        .zip(output)
        .all(|(a, b)| a.is_finite() == b.is_finite() && (!a.is_finite() || (a - b).abs() <= REPLAY_TOLERANCE * peak))
}

/// An instantiated graph of one plan's session (destroyed on drop).
pub struct CapturedGraph {
    exec: CuGraphExec,
    ctx: Arc<PrimaryContext>,
}

// SAFETY: an executable graph handle of a context usable from any thread (the owning plan
// serialises launches).
unsafe impl Send for CapturedGraph {}

impl CapturedGraph {
    /// Record everything `record` enqueues on `stream` (thread-local capture mode) and instantiate
    /// it. `record` must enqueue work on `stream` only and must not synchronise it.
    pub fn capture(
        ctx: &Arc<PrimaryContext>,
        stream: CuStream,
        record: impl FnOnce() -> Result<(), String>,
    ) -> Result<CapturedGraph, String> {
        let scope = ctx.enter().map_err(|r| format!("cuCtxPushCurrent: {}", ctx.driver().describe(r)))?;
        scope.begin_capture(stream)?;
        // The run happens with this context pushed (it is the primary context ORT uses as well).
        let recorded = record();
        let ended = scope.end_capture(stream);
        if scope.is_capturing(stream) {
            // Never leave the stream capturing (an ended capture never is; defensive).
            let _ = scope.end_capture(stream);
        }
        let graph = match (recorded, ended) {
            (Ok(()), Ok(g)) => g,
            (Err(e), Ok(g)) => {
                scope.destroy_graph(g);
                return Err(format!("the recorded run failed: {e}"));
            }
            (Err(e), Err(end)) => return Err(format!("the recorded run failed: {e} ({end})")),
            (Ok(()), Err(end)) => return Err(end),
        };
        let exec = scope.instantiate(graph)?;
        Ok(CapturedGraph { exec, ctx: Arc::clone(ctx) })
    }

    /// Enqueue one replay on `stream`. Never allocates.
    #[inline]
    pub fn launch(&self, scope: &super::cuda_driver::ContextScope<'_>, stream: CuStream) -> Result<(), i32> {
        scope.launch(self.exec, stream)
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        if let Ok(scope) = self.ctx.enter() {
            scope.destroy_exec(self.exec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_are_checked_against_the_reference_run() {
        assert!(outputs_match(&[1.0, -2.0, 0.5], &[1.0, -2.0, 0.5]));
        assert!(outputs_match(&[1.0, -2.0, 0.5], &[1.0005, -2.001, 0.5]), "within 1e-3 of the peak");
        assert!(!outputs_match(&[1.0, -2.0, 0.5], &[1.0, -2.0, 0.6]));
        assert!(!outputs_match(&[1.0, 2.0], &[1.0]));
        assert!(!outputs_match(&[1.0], &[f32::NAN]));
        assert!(outputs_match(&[f32::NAN], &[f32::NAN]), "a NaN reference only needs a NaN replay");
        assert!(outputs_match(&[0.0; 4], &[0.0005; 4]), "silence: an absolute floor of 1e-3");
        assert_eq!(WARMUP_RUNS, 2);
    }
}
