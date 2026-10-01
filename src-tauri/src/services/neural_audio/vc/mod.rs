//! Neural voice-conversion engines (RVC) and stream harness.
//! Lives inside the voice-changer duplex — does not open its own cpal streams.
//!
//! P3 layout (P3-S0 scaffold; owners in each module's docs): [`engine`] is the Sigil Voice Engine
//! (template voices, P3-C1 / P3-D / P3-E); [`assets`] freezes the model-store ids of the base
//! assets; the pitch bank adds [`swiftf0`], [`crepe`], [`pesto`], [`f0_fusion`], [`autokey`] and
//! [`formant`] (P3-D); [`import`] is the importer and slot store (P3-B).
//!
//! P5 (P5-S0 scaffold): [`families`] holds the voice-conversion families beyond RVC (DDSP-SVC,
//! Sigil-ZS, MeanVC2, so-vits-svc, sidecar v2) and the shared zero-shot reference types; the engine
//! factory (`voice_changer::factory`) dispatches to them by `manifest.format`.

pub mod assets;
pub mod autokey;
pub mod bases;
pub mod crepe;
/// The Sigil Voice Engine: the `SigilRvc` family over split templates (P3).
pub mod engine;
pub mod f0_fusion;
pub mod faiss;
pub mod fcpe;
pub mod f0_post;
/// Voice-conversion families beyond RVC (P5; registry and reference types by P5-S0).
pub mod families;
pub mod formant;
/// RVC .pth/.zip importer: allow-list unpickler, weight_norm fold, template weights writer and the
/// isolated `--headless vc-import-worker` (P1-G3 spike, P3-B production; scaffolded by P1-S0b).
pub mod import;
pub mod int8_gate;
pub mod manifest;
pub mod pesto;
mod pitch_onnx;
pub mod rmvpe;
pub mod rvc;
pub mod stream;
pub mod swiftf0;
/// Fake RVC slot (content encoder + generator) for execution-provider smoke tests.
#[cfg(test)]
pub(crate) mod test_models;
pub mod traits;
pub mod world;

pub use bases::preflight_rvc;
pub use int8_gate::supports_int8_content_encoder;
pub use rvc::{RvcModel, RvcPresetProfile};
pub use stream::VcStreamHarness;
pub use traits::{VoiceControls, VoiceConversionModel};
