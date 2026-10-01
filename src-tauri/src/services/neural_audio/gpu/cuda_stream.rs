//! Highest-priority CUDA stream (P3-C2; the `gpu_priority` lever on the CUDA rung): created on the
//! device's primary context through the CUDA driver API (`cuda_driver`, `nvcuda.dll` loaded with
//! libloading from System32) at the device's greatest priority (`cuCtxGetStreamPriorityRange` +
//! `cuStreamCreateWithPriority`), and handed to ONNX Runtime as the session's
//! `user_compute_stream`, so the GPU scheduler runs conversion kernels ahead of VRChat's and OBS's
//! normal-priority CUDA work. No admin rights, nothing load-time linked.
//!
//! The stream is non-blocking (no implicit sync with the legacy default stream). A session plan owns
//! its stream (it must outlive the session) and also enqueues its host ↔ device copies on it, so a
//! tick is: uploads → ORT's kernels → downloads → one `cuStreamSynchronize`.
//!
//! The plan says "cudart via libloading"; the driver API is used instead because it ships with every
//! NVIDIA driver (no dependency on the NVIDIA runtime pack or on which cudart another engine
//! loaded) and exposes the primary context explicitly. Streams are the same driver objects either
//! way.
//!
//! # Frozen by P3-S0
//! - [`available`]`() -> bool`, [`PriorityStream::highest`]`(device) -> Result<PriorityStream,
//!   String>`, [`PriorityStream::raw`] (the `cudaStream_t` for the provider option).

use std::sync::{Arc, OnceLock};

use super::cuda_driver::{CuStream, PrimaryContext};
use crate::services::diagnostics::gpu_mem;

/// Whether a priority CUDA stream can be created in this build: an NVIDIA adapter is present
/// (checked through DXGI, without loading the CUDA driver; cached). Creating the stream can still
/// fail (driver too old, device missing); session plans then run on ORT's default stream with a
/// note.
pub fn available() -> bool {
    nvidia_adapter_present()
}

/// Whether DXGI lists a hardware NVIDIA adapter (cached for the process).
pub(crate) fn nvidia_adapter_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(|| cfg!(windows) && !gpu_mem::nvidia_adapter_indices(&gpu_mem::adapters()).is_empty())
}

/// A CUDA stream at the device's highest priority (destroyed on drop).
pub struct PriorityStream {
    raw: usize,
    priority: i32,
    ctx: Arc<PrimaryContext>,
}

// SAFETY: a CUstream is a driver handle usable from any thread (the owning plan serialises use).
unsafe impl Send for PriorityStream {}
unsafe impl Sync for PriorityStream {}

impl PriorityStream {
    /// A new non-blocking stream at the greatest priority of CUDA device `device`.
    pub fn highest(device: i32) -> Result<PriorityStream, String> {
        let ctx = Arc::new(PrimaryContext::retain(device)?);
        let (_least, greatest) = ctx.stream_priority_range()?;
        Self::on(ctx, greatest)
    }

    /// A new non-blocking stream at the device's lowest (default) priority: the copy stream of a
    /// plan whose sessions run on ORT's own stream.
    pub fn normal(device: i32) -> Result<PriorityStream, String> {
        let ctx = Arc::new(PrimaryContext::retain(device)?);
        let (least, _greatest) = ctx.stream_priority_range()?;
        Self::on(ctx, least)
    }

    fn on(ctx: Arc<PrimaryContext>, priority: i32) -> Result<PriorityStream, String> {
        let stream = ctx.create_stream(priority)?;
        let priority = ctx.stream_priority(stream).unwrap_or(priority);
        Ok(PriorityStream { raw: stream as usize, priority, ctx })
    }

    /// The `cudaStream_t` handle.
    pub fn raw(&self) -> usize {
        self.raw
    }

    pub(crate) fn handle(&self) -> CuStream {
        self.raw as CuStream
    }

    /// The priority the driver gave the stream (lower is more urgent; `0` is the default).
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// The primary context the stream belongs to.
    pub(crate) fn context(&self) -> &Arc<PrimaryContext> {
        &self.ctx
    }

    /// CUDA device ordinal.
    pub fn device(&self) -> i32 {
        self.ctx.ordinal()
    }
}

impl Drop for PriorityStream {
    fn drop(&mut self) {
        self.ctx.destroy_stream(self.handle());
    }
}

impl std::fmt::Debug for PriorityStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PriorityStream")
            .field("raw", &format_args!("{:#x}", self.raw))
            .field("priority", &self.priority)
            .field("device", &self.device())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::gpu::cuda_driver::CudaDriver;

    /// With an NVIDIA driver: the highest-priority stream gets the device's greatest priority (more
    /// urgent than a normal one); without one, a clean error.
    #[test]
    fn highest_priority_stream_or_a_clean_error() {
        let has_device = CudaDriver::get().map(|d| d.device_count() > 0).unwrap_or(false);
        match PriorityStream::highest(0) {
            Ok(s) => {
                let normal = PriorityStream::normal(0).unwrap();
                eprintln!("{s:?} vs {normal:?}");
                assert!(s.raw() != 0 && s.raw() != normal.raw());
                assert!(s.priority() <= normal.priority());
                assert_eq!(s.device(), 0);
                let (least, greatest) = s.context().stream_priority_range().unwrap();
                assert_eq!((s.priority(), normal.priority()), (greatest, least));
                let scope = s.context().enter().unwrap();
                scope.sync(s.handle()).unwrap();
            }
            Err(why) => {
                assert!(!has_device, "a CUDA device exists but the stream failed: {why}");
                eprintln!("no priority stream: {why}");
            }
        }
        assert!(PriorityStream::highest(4096).is_err());
    }
}
