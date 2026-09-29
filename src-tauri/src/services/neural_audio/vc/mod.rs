//! Neural voice-conversion engines (RVC) and stream harness.
//! Lives inside the voice-changer duplex — does not open its own cpal streams.

pub mod bases;
pub mod faiss;
pub mod fcpe;
pub mod f0_post;
/// RVC .pth/.zip importer: allow-list unpickler, weight_norm fold, template weights writer and the
/// isolated `--headless vc-import-worker` (P1-G3 spike, P3-B production; scaffolded by P1-S0b).
pub mod import;
pub mod int8_gate;
pub mod manifest;
mod pitch_onnx;
pub mod rmvpe;
pub mod rvc;
pub mod stream;
/// Fake RVC slot (content encoder + generator) for execution-provider smoke tests.
#[cfg(test)]
pub(crate) mod test_models;
pub mod traits;
pub mod world;

pub use bases::preflight_rvc;
pub use fcpe::FcpePitchExtractor;
pub use int8_gate::supports_int8_content_encoder;
pub use rmvpe::RmvpePitchExtractor;
pub use rvc::{RvcModel, RvcPresetProfile};
pub use stream::VcStreamHarness;
pub use traits::{PitchExtractor, VoiceConversionModel};
pub use world::DioPitchExtractor;
