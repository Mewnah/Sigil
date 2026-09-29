pub mod chunk_processor;
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
pub use vc::{
    preflight_rvc, DioPitchExtractor, FcpePitchExtractor, PitchExtractor, RmvpePitchExtractor,
    RvcModel, RvcPresetProfile, VcStreamHarness, VoiceConversionModel,
};
