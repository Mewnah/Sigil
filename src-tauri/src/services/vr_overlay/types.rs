//! Wire types of the `vr-overlay` plugin (frozen by P6-S0; structs may gain fields). Kept free of
//! Tauri so the rasterizer and the worker build and test on their own; `mod.rs` re-exports them.

use serde::{Deserialize, Serialize};

/// Fastest texture update rate.
pub const MAX_UPDATE_HZ: u32 = 10;

/// Where the overlay floats (TS `RustVrOverlayAttach`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VrOverlayAttach {
    /// Follows the head, below the line of sight.
    #[default]
    Head,
    LeftHand,
    RightHand,
    /// Fixed in the play space where it was placed.
    World,
}

/// Overlay look and placement (TS `RustVrOverlayConfig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VrOverlayConfig {
    pub attach: VrOverlayAttach,
    /// Overlay width in metres.
    pub width_m: f32,
    /// Distance from the head (head attach) in metres.
    pub distance_m: f32,
    /// Vertical offset in metres (negative = below).
    pub offset_y_m: f32,
    /// Text size in texture pixels (the texture is 1024 px wide).
    pub font_size_px: u32,
    /// Overlay alpha 0..1.
    pub opacity: f32,
    /// Most caption rows shown (wrapped rows count; the newest are shown).
    pub max_lines: u32,
    /// CSS colour of the text (`rgba(...)` or `#rrggbb[aa]`).
    pub text_color: String,
    /// CSS colour of the background box.
    pub background_color: String,
}

impl Default for VrOverlayConfig {
    fn default() -> Self {
        Self {
            attach: VrOverlayAttach::Head,
            width_m: 0.6,
            distance_m: 1.2,
            offset_y_m: -0.25,
            font_size_px: 36,
            opacity: 0.9,
            max_lines: 3,
            text_color: "rgba(255,255,255,1)".into(),
            background_color: "rgba(0,0,0,0.6)".into(),
        }
    }
}

/// Final or still revising (TS `RustVrOverlayLineKind`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VrOverlayLineKind {
    #[default]
    Final,
    Interim,
}

/// One caption line (TS `RustVrOverlayLine`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VrOverlayLine {
    pub text: String,
    pub kind: VrOverlayLineKind,
    /// Pubsub topic the line came from (`text.stt`, `others.translation`, ...), for styling.
    pub source: String,
    /// Lower-case BCP-47 base language when known (font fallback, RTL).
    pub lang: Option<String>,
}

/// What the overlay shows now (TS `RustVrOverlayCaption`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VrOverlayCaption {
    pub lines: Vec<VrOverlayLine>,
}

/// Overlay state (TS `RustVrOverlayState`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VrOverlayState {
    #[default]
    Off,
    /// Started; SteamVR is not running yet (it is never launched by Sigil).
    Waiting,
    Running,
    /// The OpenVR runtime pack is not installed.
    Unavailable,
    Error,
}

/// Why the overlay waits or failed (P6-B addition to the status; TS localizes it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VrOverlayReason {
    /// `runtime/openvr` is not installed.
    RuntimeMissing,
    /// SteamVR is not running (Sigil waits for it and never starts it).
    SteamvrNotRunning,
    /// SteamVR is not installed on this PC.
    SteamvrNotInstalled,
    /// SteamVR is too old for the interfaces Sigil uses (update SteamVR).
    SteamvrTooOld,
    /// SteamVR quit; Sigil reconnects when it runs again.
    SteamvrQuit,
    /// Connecting failed (see `error`); retried.
    ConnectFailed,
    /// An overlay call failed while running (see `error`).
    OverlayError,
}

/// Result of the commands and payload of [`STATE_EVENT`] (TS `RustVrOverlayStatus`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VrOverlayStatus {
    pub state: VrOverlayState,
    /// `runtime/openvr` is installed.
    pub runtime_installed: bool,
    pub error: Option<String>,
    /// Why it waits or failed (P6-B; absent while off or running fine).
    #[serde(default)]
    pub reason: Option<VrOverlayReason>,
}
