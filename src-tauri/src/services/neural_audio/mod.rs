pub mod chunk_processor;
/// Input denoisers on the voice-changer path only (P3-E; scaffolded by P3-S0).
pub mod denoise;
/// GPU layer of the Sigil Voice Engine: session plans, IOBinding, CUDA graphs, priority stream /
/// queue, TensorRT-RTX, VRAM ledger (P3-C2; scaffolded by P3-S0).
pub mod gpu;
pub mod onnx_voice;
pub mod ort_session;
pub mod sidecar_bridge;
/// Hand-encoded stand-in ONNX models for engine tests (no downloads, no Python `onnx`).
#[cfg(test)]
pub(crate) mod test_onnx;
pub mod vc;

pub use chunk_processor::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, VoiceChunkPipeline, DEFAULT_CHUNK_FRAMES,
};
pub use onnx_voice::OnnxVoiceBridge;
pub use sidecar_bridge::SidecarVoiceBridge;
pub use vc::{preflight_rvc, RvcModel, RvcPresetProfile, VcStreamHarness, VoiceConversionModel};
