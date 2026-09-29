//! Neural voice-conversion engines (RVC) and stream harness.
//! Lives inside the voice-changer duplex — does not open its own cpal streams.

pub mod bases;
pub mod faiss;
pub mod fcpe;
pub mod f0_post;
pub mod int8_gate;
pub mod manifest;
mod pitch_onnx;
pub mod rmvpe;
pub mod rvc;
pub mod stream;
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
