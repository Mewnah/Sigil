//! [`VrOverlayConfig`] as the worker uses it: every number clamped to a sane range (a hand-edited
//! settings file or a mistyped field never produces a 0-pixel texture, an invisible overlay or one
//! inside the user's head) and the colours parsed.

use super::color::{css_color_or, Rgba};
use super::{VrOverlayAttach, VrOverlayConfig};

/// Width of the overlay texture in pixels at scale 1. `fontSizePx` is relative to it: 36 px is
/// 36/1024 of the overlay's width.
pub(crate) const TEXTURE_WIDTH: u32 = 1024;
/// Largest texture area uploaded with `SetOverlayRaw` (pixels; 2 MiB of RGBA). Bigger layouts (many
/// lines of large text) are rendered at a lower scale with the same aspect ratio, so the overlay keeps
/// its size in metres and only loses sharpness.
pub(crate) const MAX_TEXTURE_PIXELS: u32 = TEXTURE_WIDTH * 512;
/// Smallest upload cap the worker falls back to when SteamVR refuses uploads (OpenVR documents an
/// upper bound for `SetOverlayRaw` without naming it): 512 KiB of RGBA.
pub(crate) const MIN_TEXTURE_PIXELS: u32 = TEXTURE_WIDTH * 128;

pub(crate) const FONT_PX: (f32, f32) = (12.0, 96.0);
pub(crate) const MAX_LINES: (u32, u32) = (1, 8);
pub(crate) const WIDTH_M: (f32, f32) = (0.1, 3.0);
pub(crate) const DISTANCE_M: (f32, f32) = (0.3, 5.0);
pub(crate) const OFFSET_Y_M: (f32, f32) = (-2.0, 2.0);
pub(crate) const OPACITY: (f32, f32) = (0.05, 1.0);

const DEFAULT_TEXT: Rgba = [255, 255, 255, 255];
const DEFAULT_BACKGROUND: Rgba = [0, 0, 0, 153];

/// What the rasterizer needs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Look {
    pub font_px: f32,
    /// Visual rows shown (the newest; older wrapped rows scroll out at the top).
    pub max_lines: u32,
    pub text: Rgba,
    pub background: Rgba,
    /// Largest texture area (pixels): [`MAX_TEXTURE_PIXELS`], lowered by the worker after refused
    /// uploads.
    pub max_pixels: u32,
}

/// The whole normalized configuration.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Normalized {
    pub look: Look,
    pub attach: VrOverlayAttach,
    pub width_m: f32,
    pub distance_m: f32,
    pub offset_y_m: f32,
    pub opacity: f32,
}

fn clamp_f(v: f32, (lo, hi): (f32, f32), default: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        default
    }
}

impl Normalized {
    pub(crate) fn from_config(c: &VrOverlayConfig) -> Self {
        let d = VrOverlayConfig::default();
        Self {
            look: Look {
                font_px: clamp_f(c.font_size_px as f32, FONT_PX, d.font_size_px as f32),
                max_lines: c.max_lines.clamp(MAX_LINES.0, MAX_LINES.1),
                text: css_color_or(&c.text_color, DEFAULT_TEXT),
                background: css_color_or(&c.background_color, DEFAULT_BACKGROUND),
                max_pixels: MAX_TEXTURE_PIXELS,
            },
            attach: c.attach,
            width_m: clamp_f(c.width_m, WIDTH_M, d.width_m),
            distance_m: clamp_f(c.distance_m, DISTANCE_M, d.distance_m),
            offset_y_m: clamp_f(c.offset_y_m, OFFSET_Y_M, d.offset_y_m),
            opacity: clamp_f(c.opacity, OPACITY, d.opacity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_pass_through() {
        let n = Normalized::from_config(&VrOverlayConfig::default());
        assert_eq!(n.look.font_px, 36.0);
        assert_eq!(n.look.max_lines, 3);
        assert_eq!(n.look.text, [255, 255, 255, 255]);
        assert_eq!(n.look.background, [0, 0, 0, 153]);
        assert_eq!(n.attach, VrOverlayAttach::Head);
        assert_eq!((n.width_m, n.distance_m, n.offset_y_m, n.opacity), (0.6, 1.2, -0.25, 0.9));
    }

    #[test]
    fn clamps_and_repairs() {
        let c = VrOverlayConfig {
            width_m: 0.0,
            distance_m: f32::NAN,
            offset_y_m: 99.0,
            font_size_px: 4000,
            opacity: -1.0,
            max_lines: 0,
            text_color: "not a colour".into(),
            background_color: "#00ff0080".into(),
            ..VrOverlayConfig::default()
        };
        let n = Normalized::from_config(&c);
        assert_eq!(n.width_m, WIDTH_M.0);
        assert_eq!(n.distance_m, 1.2);
        assert_eq!(n.offset_y_m, OFFSET_Y_M.1);
        assert_eq!(n.look.font_px, FONT_PX.1);
        assert_eq!(n.opacity, OPACITY.0);
        assert_eq!(n.look.max_lines, 1);
        assert_eq!(n.look.text, DEFAULT_TEXT);
        assert_eq!(n.look.background, [0, 255, 0, 128]);
    }
}
