//! Caption rasterizer for the SteamVR overlay: cosmic-text shapes the lines (system fonts, script
//! fallback, bidi for RTL scripts) and tiny-skia draws the background box; glyphs are blended by
//! hand so an interim line can be dimmed. The texture has a fixed size per [`Look`] (1024 px wide,
//! `max_lines` rows high), so the overlay never changes size or jumps while text changes: the text
//! is centred and anchored at the bottom, and older wrapped rows scroll out at the top, like live
//! captions.
//!
//! CJK lines get a CJK font as their primary family, chosen by the line's language (or, without
//! one, by its kana / hangul), so Japanese is never drawn with Chinese glyph shapes (Han
//! unification) whatever the system locale is.
//!
//! [`with_renderer`] holds one process-wide renderer: loading the system font list takes up to a
//! second (release) and is done once, on the overlay worker thread, the first time something is
//! shown.

use std::sync::{Mutex, OnceLock, PoisonError};

use cosmic_text::{fontdb, Align, Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent, Wrap};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

use super::color::Rgba;
use super::settings::{Look, MIN_TEXTURE_PIXELS, TEXTURE_WIDTH};
use super::{VrOverlayCaption, VrOverlayLineKind};

/// Longest text kept per line (characters; the newest part of a long interim is kept).
const MAX_LINE_CHARS: usize = 1000;
/// Most caption lines laid out (the newest).
const MAX_CAPTION_LINES: usize = 16;
/// Interim lines are drawn at this share of the text colour's alpha.
const INTERIM_ALPHA: f32 = 0.72;

/// An RGBA texture (straight alpha, row-major, `width * height * 4` bytes) ready for
/// `SetOverlayRaw`.
#[derive(Clone, Debug)]
pub(crate) struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub stats: RenderStats,
}

impl Texture {
    /// Nothing to show (the overlay is hidden instead of showing an empty box).
    pub(crate) fn is_blank(&self) -> bool {
        self.stats.rows_shown == 0
    }
}

/// What a render did (tests, logs).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RenderStats {
    /// Wrapped rows of all lines.
    pub rows_total: usize,
    /// Rows drawn (the newest `max_lines`).
    pub rows_shown: usize,
    /// Glyphs drawn.
    pub glyphs: usize,
    /// Drawn glyphs no font had (`.notdef`).
    pub missing_glyphs: usize,
    /// Render scale (< 1 when the layout was larger than [`Look::max_pixels`]).
    pub scale: f32,
}

/// Font group of a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Script {
    /// Latin, Cyrillic, Greek, Arabic, Hebrew, Indic, Thai ... (Segoe UI plus script fallback).
    Base,
    Japanese,
    ChineseSimplified,
    ChineseTraditional,
    /// Cantonese / Hong Kong.
    ChineseHongKong,
    Korean,
}

fn is_kana(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}')
}

fn is_hangul(c: char) -> bool {
    matches!(c, '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' | '\u{AC00}'..='\u{D7AF}')
}

fn is_han(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}' | '\u{20000}'..='\u{2FA1F}')
}

/// The font group of a line from its language (`TextEvent.lang`: lower-case BCP-47) or, for other
/// languages and unknown ones, from the characters it contains.
pub(crate) fn script_for(lang: Option<&str>, text: &str) -> Script {
    let lang = lang
        .map(|l| l.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let primary = lang.split(['-', '_']).next().unwrap_or("");
    match primary {
        "ja" => return Script::Japanese,
        "ko" => return Script::Korean,
        "yue" => return Script::ChineseHongKong,
        "zh" | "cmn" | "wuu" | "hak" | "nan" => {
            let rest = &lang[primary.len()..];
            return if rest.contains("hk") || rest.contains("mo") {
                Script::ChineseHongKong
            } else if rest.contains("hant") || rest.contains("tw") {
                Script::ChineseTraditional
            } else {
                Script::ChineseSimplified
            };
        }
        _ => {}
    }
    let (mut han, mut kana, mut hangul) = (false, false, false);
    for c in text.chars() {
        kana |= is_kana(c);
        hangul |= is_hangul(c);
        han |= is_han(c);
    }
    if kana {
        Script::Japanese
    } else if hangul {
        Script::Korean
    } else if han {
        Script::ChineseSimplified
    } else {
        Script::Base
    }
}

/// Font families found on this PC, per group (first available of each candidate list).
#[derive(Clone, Debug, Default)]
struct Families {
    base: Option<String>,
    japanese: Option<String>,
    chinese_simplified: Option<String>,
    chinese_traditional: Option<String>,
    chinese_hong_kong: Option<String>,
    korean: Option<String>,
}

const BASE_FAMILIES: &[&str] = &["Segoe UI", "Arial", "Noto Sans", "DejaVu Sans", "Helvetica"];
const JAPANESE_FAMILIES: &[&str] = &[
    "Yu Gothic UI",
    "Yu Gothic",
    "Meiryo UI",
    "Meiryo",
    "MS UI Gothic",
    "MS Gothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
];
const CHINESE_SIMPLIFIED_FAMILIES: &[&str] = &[
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "DengXian",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Noto Sans SC",
];
const CHINESE_TRADITIONAL_FAMILIES: &[&str] = &[
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "PMingLiU",
    "MingLiU",
    "Noto Sans CJK TC",
    "Noto Sans TC",
];
const CHINESE_HONG_KONG_FAMILIES: &[&str] = &[
    "MingLiU_HKSCS",
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "Noto Sans CJK HK",
    "Noto Sans HK",
];
const KOREAN_FAMILIES: &[&str] = &["Malgun Gothic", "Gulim", "Dotum", "Noto Sans CJK KR", "Noto Sans KR"];

impl Families {
    fn resolve(db: &fontdb::Database) -> Self {
        let names: std::collections::HashSet<String> = db
            .faces()
            .flat_map(|f| f.families.iter().map(|(name, _)| name.to_ascii_lowercase()))
            .collect();
        let first = |list: &[&str]| {
            list.iter()
                .find(|n| names.contains(&n.to_ascii_lowercase()))
                .map(|n| n.to_string())
        };
        Self {
            base: first(BASE_FAMILIES),
            japanese: first(JAPANESE_FAMILIES),
            chinese_simplified: first(CHINESE_SIMPLIFIED_FAMILIES),
            chinese_traditional: first(CHINESE_TRADITIONAL_FAMILIES),
            chinese_hong_kong: first(CHINESE_HONG_KONG_FAMILIES),
            korean: first(KOREAN_FAMILIES),
        }
    }

    fn family(&self, script: Script) -> Family<'_> {
        let name = match script {
            Script::Base => &self.base,
            Script::Japanese => &self.japanese,
            Script::ChineseSimplified => &self.chinese_simplified,
            Script::ChineseTraditional => self
                .chinese_traditional
                .as_ref()
                .map_or(&self.chinese_simplified, |_| &self.chinese_traditional),
            Script::ChineseHongKong => self
                .chinese_hong_kong
                .as_ref()
                .map_or(&self.chinese_traditional, |_| &self.chinese_hong_kong),
            Script::Korean => &self.korean,
        };
        match name.as_deref().or(self.base.as_deref()) {
            Some(n) => Family::Name(n),
            None => Family::SansSerif,
        }
    }
}

/// One line ready for layout.
struct PreparedLine {
    text: String,
    script: Script,
    alpha: f32,
}

/// Control characters (line breaks included: one caption line is one paragraph) become spaces;
/// only the newest [`MAX_LINE_CHARS`] characters are kept.
fn sanitize(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    let count = trimmed.chars().count();
    if count <= MAX_LINE_CHARS {
        return trimmed.to_string();
    }
    trimmed
        .chars()
        .skip(count - MAX_LINE_CHARS)
        .collect::<String>()
        .trim_start()
        .to_string()
}

fn prepare(caption: &VrOverlayCaption) -> Vec<PreparedLine> {
    let lines = &caption.lines;
    let skip = lines.len().saturating_sub(MAX_CAPTION_LINES);
    lines
        .iter()
        .skip(skip)
        .filter_map(|l| {
            let text = sanitize(&l.text);
            (!text.is_empty()).then(|| PreparedLine {
                script: script_for(l.lang.as_deref(), &text),
                alpha: if l.kind == VrOverlayLineKind::Interim { INTERIM_ALPHA } else { 1.0 },
                text,
            })
        })
        .collect()
}

/// Texture size and metrics of a look (fixed per look, whatever the text).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Geometry {
    width: u32,
    height: u32,
    scale: f32,
    font: f32,
    line_height: f32,
    pad_x: f32,
    pad_y: f32,
}

impl Geometry {
    fn of(look: &Look) -> Self {
        let font = look.font_px;
        let line_height = (font * 1.3).round();
        let pad_x = (font * 0.6).round();
        let pad_y = (font * 0.35).round();
        let w = TEXTURE_WIDTH as f32;
        let h = (2.0 * pad_y + line_height * look.max_lines as f32).ceil();
        let cap = look.max_pixels.max(MIN_TEXTURE_PIXELS) as f32;
        let scale = if w * h > cap { (cap / (w * h)).sqrt() } else { 1.0 };
        Self {
            width: ((w * scale).floor() as u32).max(16),
            height: ((h * scale).floor() as u32).max(8),
            scale,
            font: font * scale,
            line_height: line_height * scale,
            pad_x: pad_x * scale,
            pad_y: pad_y * scale,
        }
    }
}

/// Source-over of a straight-alpha colour onto a premultiplied RGBA pixel.
fn blend(data: &mut [u8], width: u32, height: u32, x: i32, y: i32, rgb: [u8; 3], alpha: u32) {
    if alpha == 0 || x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
        return;
    }
    let i = ((y as u32 * width + x as u32) * 4) as usize;
    let a = alpha.min(255);
    let inv = 255 - a;
    for c in 0..3 {
        let src = (u32::from(rgb[c]) * a + 127) / 255;
        let dst = (u32::from(data[i + c]) * inv + 127) / 255;
        data[i + c] = (src + dst).min(255) as u8;
    }
    let dst_a = (u32::from(data[i + 3]) * inv + 127) / 255;
    data[i + 3] = (a + dst_a).min(255) as u8;
}

fn rounded_rect(left: f32, top: f32, right: f32, bottom: f32, radius: f32) -> Option<tiny_skia::Path> {
    let r = radius
        .min((right - left) / 2.0)
        .min((bottom - top) / 2.0)
        .max(0.0);
    let mut pb = PathBuilder::new();
    pb.move_to(left + r, top);
    pb.line_to(right - r, top);
    pb.quad_to(right, top, right, top + r);
    pb.line_to(right, bottom - r);
    pb.quad_to(right, bottom, right - r, bottom);
    pb.line_to(left + r, bottom);
    pb.quad_to(left, bottom, left, bottom - r);
    pb.line_to(left, top + r);
    pb.quad_to(left, top, left + r, top);
    pb.close();
    pb.finish()
}

/// The overlay's text renderer (see the module docs).
pub(crate) struct TextRenderer {
    fonts: FontSystem,
    cache: SwashCache,
    families: Families,
}

impl TextRenderer {
    /// Loads the system fonts (slow: call once, off the UI and IPC threads).
    pub(crate) fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        let families = Families::resolve(&db);
        if let Some(base) = &families.base {
            db.set_sans_serif_family(base.clone());
        }
        tracing::debug!(faces = db.len(), ?families, "vr_overlay: font database loaded");
        Self {
            fonts: FontSystem::new_with_locale_and_db("en-US".to_string(), db),
            cache: SwashCache::new(),
            families,
        }
    }

    /// Render `caption` with `look`. Always returns a texture of the look's fixed size.
    pub(crate) fn render(&mut self, caption: &VrOverlayCaption, look: &Look) -> Texture {
        let g = Geometry::of(look);
        let mut stats = RenderStats {
            scale: g.scale,
            ..RenderStats::default()
        };
        let Some(mut pixmap) = Pixmap::new(g.width, g.height) else {
            return Texture {
                width: g.width,
                height: g.height,
                rgba: vec![0; (g.width * g.height * 4) as usize],
                stats,
            };
        };
        let lines = prepare(caption);
        if lines.is_empty() {
            return Texture {
                width: g.width,
                height: g.height,
                rgba: pixmap.take_demultiplied(),
                stats,
            };
        }

        let Self { fonts, cache, families } = self;
        let mut buffer = Buffer::new(fonts, Metrics::new(g.font, g.line_height));
        buffer.set_wrap(Wrap::WordOrGlyph);
        buffer.set_size(Some((g.width as f32 - 2.0 * g.pad_x).max(g.font)), None);
        let text = look.text;
        let attrs_of = |l: &PreparedLine| {
            let a = (f32::from(text[3]) * l.alpha).round().clamp(0.0, 255.0) as u8;
            Attrs::new()
                .family(families.family(l.script))
                .color(Color::rgba(text[0], text[1], text[2], a))
        };
        let mut spans: Vec<(&str, Attrs)> = Vec::with_capacity(lines.len() * 2);
        for (i, l) in lines.iter().enumerate() {
            if i > 0 {
                spans.push(("\n", attrs_of(l)));
            }
            spans.push((l.text.as_str(), attrs_of(l)));
        }
        let default_attrs = Attrs::new().family(families.family(Script::Base));
        buffer.set_rich_text(spans, &default_attrs, Shaping::Advanced, Some(Align::Center));
        buffer.shape_until_scroll(fonts, false);

        let runs: Vec<_> = buffer.layout_runs().collect();
        stats.rows_total = runs.len();
        let first_shown = runs.len().saturating_sub(look.max_lines as usize);
        let shown = &runs[first_shown..];
        let (Some(top_run), Some(bottom_run)) = (shown.first(), shown.last()) else {
            return Texture {
                width: g.width,
                height: g.height,
                rgba: pixmap.take_demultiplied(),
                stats,
            };
        };
        let block_top = top_run.line_top;
        let block_bottom = bottom_run.line_top + bottom_run.line_height;
        let shift_y = (g.height as f32 - g.pad_y) - block_bottom;

        let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
        for run in shown {
            for glyph in run.glyphs {
                min_x = min_x.min(glyph.x);
                max_x = max_x.max(glyph.x + glyph.w);
            }
        }
        if min_x > max_x {
            return Texture {
                width: g.width,
                height: g.height,
                rgba: pixmap.take_demultiplied(),
                stats,
            };
        }

        // Background box around the shown rows.
        let bg: Rgba = look.background;
        if bg[3] > 0 {
            let margin_x = g.pad_x * 0.6;
            let margin_y = g.pad_y * 0.8;
            let left = (g.pad_x + min_x - margin_x).max(0.0);
            let right = (g.pad_x + max_x + margin_x).min(g.width as f32);
            let top = (block_top + shift_y - margin_y).max(0.0);
            let bottom = (block_bottom + shift_y + margin_y).min(g.height as f32);
            if let Some(path) = rounded_rect(left, top, right, bottom, g.font * 0.35) {
                let mut paint = Paint::default();
                paint.set_color_rgba8(bg[0], bg[1], bg[2], bg[3]);
                paint.anti_alias = true;
                pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
            }
        }

        // Glyphs.
        let (width, height) = (g.width, g.height);
        let data = pixmap.data_mut();
        for run in shown {
            stats.rows_shown += 1;
            for glyph in run.glyphs {
                stats.glyphs += 1;
                if glyph.glyph_id == 0 {
                    stats.missing_glyphs += 1;
                }
                let color = glyph
                    .color_opt
                    .unwrap_or(Color::rgba(text[0], text[1], text[2], text[3]));
                let physical = glyph.physical((g.pad_x, run.line_y + shift_y), 1.0);
                let Some(image) = cache.get_image(fonts, physical.cache_key) else {
                    continue;
                };
                let x0 = physical.x + image.placement.left;
                let y0 = physical.y - image.placement.top;
                let (w, h) = (image.placement.width as i32, image.placement.height as i32);
                match image.content {
                    SwashContent::Mask => {
                        for oy in 0..h {
                            for ox in 0..w {
                                let coverage = u32::from(image.data[(oy * w + ox) as usize]);
                                let alpha = (coverage * u32::from(color.a()) + 127) / 255;
                                blend(data, width, height, x0 + ox, y0 + oy, [color.r(), color.g(), color.b()], alpha);
                            }
                        }
                    }
                    SwashContent::Color => {
                        // Colour glyphs (emoji) keep their colours, dimmed with the line.
                        for oy in 0..h {
                            for ox in 0..w {
                                let i = ((oy * w + ox) * 4) as usize;
                                let px = &image.data[i..i + 4];
                                let alpha = (u32::from(px[3]) * u32::from(color.a()) + 127) / 255;
                                blend(data, width, height, x0 + ox, y0 + oy, [px[0], px[1], px[2]], alpha);
                            }
                        }
                    }
                    SwashContent::SubpixelMask => {}
                }
            }
        }
        Texture {
            width,
            height,
            rgba: pixmap.take_demultiplied(),
            stats,
        }
    }
}

/// Run `f` with the process-wide renderer, creating it on first use.
pub(crate) fn with_renderer<T>(f: impl FnOnce(&mut TextRenderer) -> T) -> T {
    static RENDERER: OnceLock<Mutex<Option<TextRenderer>>> = OnceLock::new();
    let mut guard = RENDERER
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    f(guard.get_or_insert_with(TextRenderer::new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::vr_overlay::settings::MAX_TEXTURE_PIXELS;
    use crate::services::vr_overlay::{VrOverlayLine, VrOverlayLineKind};

    fn look() -> Look {
        Look {
            font_px: 36.0,
            max_lines: 3,
            text: [255, 255, 255, 255],
            background: [0, 0, 0, 153],
            max_pixels: MAX_TEXTURE_PIXELS,
        }
    }

    fn caption(lines: &[(&str, VrOverlayLineKind, Option<&str>)]) -> VrOverlayCaption {
        VrOverlayCaption {
            lines: lines
                .iter()
                .map(|(t, k, lang)| VrOverlayLine {
                    text: t.to_string(),
                    kind: *k,
                    source: "text.stt".into(),
                    lang: lang.map(str::to_string),
                })
                .collect(),
        }
    }

    fn final_line(text: &str) -> VrOverlayCaption {
        caption(&[(text, VrOverlayLineKind::Final, None)])
    }

    /// (first, last) row with any non-transparent pixel.
    fn opaque_rows(t: &Texture) -> Option<(u32, u32)> {
        let rows: Vec<u32> = (0..t.height)
            .filter(|y| (0..t.width).any(|x| t.rgba[((y * t.width + x) * 4 + 3) as usize] > 0))
            .collect();
        Some((*rows.first()?, *rows.last()?))
    }

    #[test]
    fn script_choice() {
        assert_eq!(script_for(Some("ja"), "hello"), Script::Japanese);
        assert_eq!(script_for(Some("zh"), "你好"), Script::ChineseSimplified);
        assert_eq!(script_for(Some("zh-hant"), "你好"), Script::ChineseTraditional);
        assert_eq!(script_for(Some("zh-TW"), "你好"), Script::ChineseTraditional);
        assert_eq!(script_for(Some("yue"), "你好"), Script::ChineseHongKong);
        assert_eq!(script_for(Some("zh-hk"), "你好"), Script::ChineseHongKong);
        assert_eq!(script_for(Some("ko"), "abc"), Script::Korean);
        // Unknown or non-CJK language: the characters decide (a mistagged line still gets a CJK font).
        assert_eq!(script_for(None, "今日はいい天気"), Script::Japanese);
        assert_eq!(script_for(Some("en"), "안녕하세요"), Script::Korean);
        assert_eq!(script_for(None, "今天天气很好"), Script::ChineseSimplified);
        assert_eq!(script_for(None, "مرحبا"), Script::Base);
        assert_eq!(script_for(Some("ar"), "hello"), Script::Base);
    }

    #[test]
    fn sanitize_keeps_one_paragraph_and_the_newest_text() {
        assert_eq!(sanitize("  a\nb\tc\r\n "), "a b c");
        let long: String = std::iter::repeat_n('x', MAX_LINE_CHARS + 50)
            .chain("end".chars())
            .collect();
        let s = sanitize(&long);
        assert_eq!(s.chars().count(), MAX_LINE_CHARS);
        assert!(s.ends_with("end"));
    }

    #[test]
    fn geometry_is_fixed_and_capped() {
        let g = Geometry::of(&look());
        assert_eq!(g.width, TEXTURE_WIDTH);
        assert_eq!(g.scale, 1.0);
        // 2 * 13 px padding + 3 rows of 47 px.
        assert_eq!(g.height, 167);
        let big = Look {
            font_px: 96.0,
            max_lines: 8,
            ..look()
        };
        let gb = Geometry::of(&big);
        assert!(gb.scale < 1.0);
        assert!(gb.width * gb.height <= MAX_TEXTURE_PIXELS);
        // Same aspect ratio as the unscaled layout (within rounding).
        let unscaled_h = (2.0 * (96.0f32 * 0.35).round() + (96.0f32 * 1.3).round() * 8.0).ceil();
        let aspect = gb.width as f32 / gb.height as f32;
        assert!((aspect - TEXTURE_WIDTH as f32 / unscaled_h).abs() < 0.02, "{aspect}");
    }

    #[test]
    fn blank_caption_is_transparent() {
        let t = with_renderer(|r| r.render(&caption(&[("   ", VrOverlayLineKind::Final, None)]), &look()));
        assert!(t.is_blank());
        assert_eq!((t.width, t.height), (1024, 167));
        assert_eq!(t.rgba.len(), (t.width * t.height * 4) as usize);
        assert!(t.rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn text_is_drawn_bottom_anchored_with_a_box() {
        let t = with_renderer(|r| r.render(&final_line("Hello world"), &look()));
        assert!(!t.is_blank());
        assert_eq!((t.width, t.height), (1024, 167));
        assert_eq!(t.stats.rows_shown, 1);
        assert!(t.stats.glyphs >= 10);
        let (top, bottom) = opaque_rows(&t).expect("pixels drawn");
        // One row of three, anchored at the bottom: nothing in the top third.
        assert!(top > t.height / 3, "top {top}");
        assert!(bottom >= t.height - 13, "bottom {bottom}");
        // White text over a translucent black box: some pixels are bright and opaque.
        let bright = t
            .rgba
            .chunks_exact(4)
            .filter(|p| p[0] > 200 && p[3] > 200)
            .count();
        assert!(bright > 50, "{bright}");
        // The box is narrower than the texture for a short line (centred).
        let row = (top + bottom) / 2;
        let alpha_at = |x: u32| t.rgba[((row * t.width + x) * 4 + 3) as usize];
        assert_eq!(alpha_at(0), 0);
        assert_eq!(alpha_at(t.width - 1), 0);
        assert!(alpha_at(t.width / 2) > 0);
    }

    #[test]
    fn size_does_not_depend_on_text() {
        let a = with_renderer(|r| r.render(&final_line("a"), &look()));
        let b = with_renderer(|r| r.render(&final_line(&"long words ".repeat(40)), &look()));
        assert_eq!((a.width, a.height), (b.width, b.height));
    }

    #[test]
    fn long_text_scrolls_to_the_newest_rows() {
        let t = with_renderer(|r| r.render(&final_line(&"the quick brown fox jumps over the lazy dog ".repeat(12)), &look()));
        assert!(t.stats.rows_total > 3, "{:?}", t.stats);
        assert_eq!(t.stats.rows_shown, 3);
        // Several lines: only the newest rows are shown.
        let many = caption(&[
            ("first line", VrOverlayLineKind::Final, None),
            ("second line", VrOverlayLineKind::Final, None),
            ("third line", VrOverlayLineKind::Final, None),
            ("fourth line", VrOverlayLineKind::Interim, None),
        ]);
        let t = with_renderer(|r| r.render(&many, &Look { max_lines: 2, ..look() }));
        assert_eq!((t.stats.rows_total, t.stats.rows_shown), (4, 2));
    }

    #[test]
    fn interim_lines_are_dimmer() {
        let clear = Look {
            background: [0, 0, 0, 0],
            ..look()
        };
        let max_alpha = |kind| {
            let t = with_renderer(|r| r.render(&caption(&[("Interim test", kind, None)]), &clear));
            t.rgba.chunks_exact(4).map(|p| p[3]).max().unwrap_or(0)
        };
        let fin = max_alpha(VrOverlayLineKind::Final);
        let interim = max_alpha(VrOverlayLineKind::Interim);
        assert!(fin >= 250, "{fin}");
        assert!(interim < fin && interim > 100, "{interim} vs {fin}");
    }

    #[test]
    fn huge_look_stays_within_the_upload_cap() {
        let big = Look {
            font_px: 96.0,
            max_lines: 8,
            ..look()
        };
        let t = with_renderer(|r| r.render(&final_line("Big text"), &big));
        assert!(t.width * t.height <= MAX_TEXTURE_PIXELS);
        assert!(!t.is_blank());
        assert!(t.stats.scale < 1.0);
    }

    #[test]
    fn a_lower_pixel_cap_shrinks_the_texture_not_the_layout() {
        let full = with_renderer(|r| r.render(&final_line("Capped text"), &look()));
        let small = Look {
            max_pixels: MIN_TEXTURE_PIXELS,
            ..look()
        };
        let capped = with_renderer(|r| r.render(&final_line("Capped text"), &small));
        assert!(capped.width * capped.height <= MIN_TEXTURE_PIXELS);
        assert!(capped.width < full.width && capped.stats.scale < 1.0);
        let aspect = |t: &Texture| t.width as f32 / t.height as f32;
        assert!((aspect(&capped) - aspect(&full)).abs() < 0.05);
        assert_eq!(capped.stats.rows_shown, 1);
        // A cap below the floor is raised to it.
        let tiny = with_renderer(|r| r.render(&final_line("x"), &Look { max_pixels: 1, ..look() }));
        assert_eq!((tiny.width, tiny.height), (capped.width, capped.height));
    }

    /// CJK and RTL text finds a font on Windows (the platform this overlay ships on).
    #[cfg(windows)]
    #[test]
    fn cjk_and_rtl_have_glyphs() {
        for (text, lang) in [
            ("こんにちは、世界", Some("ja")),
            ("今日は晴れです", None),
            ("你好，世界", Some("zh")),
            ("你好，世界", Some("yue")),
            ("안녕하세요 여러분", Some("ko")),
            ("مرحبا بالعالم", Some("ar")),
            ("שלום עולם", None),
            ("Привет, мир", Some("ru")),
            ("สวัสดีครับ", Some("th")),
        ] {
            let t = with_renderer(|r| r.render(&caption(&[(text, VrOverlayLineKind::Final, lang)]), &look()));
            assert_eq!(t.stats.rows_shown, 1, "{text}");
            assert_eq!(t.stats.missing_glyphs, 0, "{text}: {:?}", t.stats);
            assert!(t.stats.glyphs > 0, "{text}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn cjk_lines_use_a_cjk_primary_family() {
        with_renderer(|r| {
            // Only checked where the Windows CJK fonts are installed (they are by default).
            if r.families.japanese.is_some() {
                assert!(matches!(r.families.family(Script::Japanese), Family::Name(n) if JAPANESE_FAMILIES.contains(&n)));
            }
            if r.families.korean.is_some() {
                assert!(matches!(r.families.family(Script::Korean), Family::Name(n) if KOREAN_FAMILIES.contains(&n)));
            }
            // A missing group falls back to the base family, never to nothing.
            let empty = Families {
                base: Some("Segoe UI".into()),
                ..Families::default()
            };
            assert!(matches!(empty.family(Script::Japanese), Family::Name("Segoe UI")));
        });
    }
}
