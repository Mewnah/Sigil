pub mod chunk_processor;
pub mod onnx_voice;
pub mod sidecar_bridge;

pub use chunk_processor::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, VoiceChunkPipeline, DEFAULT_CHUNK_FRAMES,
};
pub use onnx_voice::OnnxVoiceBridge;
pub use sidecar_bridge::SidecarVoiceBridge;
