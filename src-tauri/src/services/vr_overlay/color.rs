//! CSS colour strings of the overlay settings (`rgba(...)`, `rgb(...)`, `#rgb[a]`, `#rrggbb[aa]` and a
//! few names) to straight (not premultiplied) RGBA bytes. The settings UI writes `rgba(r,g,b,a)`; the
//! other forms are accepted because users can type colours by hand.

/// Straight-alpha RGBA.
pub(crate) type Rgba = [u8; 4];

/// Parse a CSS colour, or `None` when it is not one of the supported forms.
pub(crate) fn parse_css_color(input: &str) -> Option<Rgba> {
    let s = input.trim().to_ascii_lowercase();
    if s.is_empty() || s.len() > 64 {
        return None;
    }
    if let Some(hex) = s.strip_prefix('#') {
        return parse_hex(hex);
    }
    if let Some(args) = function_args(&s, "rgba").or_else(|| function_args(&s, "rgb")) {
        return parse_rgb_args(args);
    }
    named(&s)
}

/// `input` parsed, or `fallback` when it is not a colour.
pub(crate) fn css_color_or(input: &str, fallback: Rgba) -> Rgba {
    parse_css_color(input).unwrap_or(fallback)
}

fn function_args<'a>(s: &'a str, name: &str) -> Option<&'a str> {
    let rest = s.strip_prefix(name)?.trim_start();
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    Some(inner)
}

fn parse_hex(hex: &str) -> Option<Rgba> {
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let nibble = |i: usize| u8::from_str_radix(&hex[i..i + 1], 16).ok().map(|v| v * 17);
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    match hex.len() {
        3 => Some([nibble(0)?, nibble(1)?, nibble(2)?, 255]),
        4 => Some([nibble(0)?, nibble(1)?, nibble(2)?, nibble(3)?]),
        6 => Some([byte(0)?, byte(2)?, byte(4)?, 255]),
        8 => Some([byte(0)?, byte(2)?, byte(4)?, byte(6)?]),
        _ => None,
    }
}

/// `r, g, b[, a]` or the space syntax `r g b[ / a]`; channels as 0–255 or percentages, alpha as
/// 0–1 or a percentage. Out-of-range values are clamped like browsers do.
fn parse_rgb_args(args: &str) -> Option<Rgba> {
    let (channels, alpha) = match args.split_once('/') {
        Some((c, a)) => (c, Some(a.trim())),
        None => (args, None),
    };
    let mut parts: Vec<&str> = if channels.contains(',') {
        channels.split(',').map(str::trim).collect()
    } else {
        channels.split_whitespace().collect()
    };
    let alpha = match (alpha, parts.len()) {
        (Some(a), 3) => Some(a),
        (None, 4) => parts.pop(),
        (None, 3) => None,
        _ => return None,
    };
    let channel = |p: &str| -> Option<u8> {
        let v = if let Some(pct) = p.strip_suffix('%') {
            pct.trim().parse::<f32>().ok()? * 2.55
        } else {
            p.parse::<f32>().ok()?
        };
        v.is_finite().then(|| v.round().clamp(0.0, 255.0) as u8)
    };
    let a = match alpha {
        None => 255,
        Some(a) => {
            let v = if let Some(pct) = a.strip_suffix('%') {
                pct.trim().parse::<f32>().ok()? / 100.0
            } else {
                a.parse::<f32>().ok()?
            };
            if !v.is_finite() {
                return None;
            }
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        }
    };
    Some([channel(parts[0])?, channel(parts[1])?, channel(parts[2])?, a])
}

fn named(s: &str) -> Option<Rgba> {
    Some(match s {
        "transparent" => [0, 0, 0, 0],
        "black" => [0, 0, 0, 255],
        "white" => [255, 255, 255, 255],
        "red" => [255, 0, 0, 255],
        "green" => [0, 128, 0, 255],
        "lime" => [0, 255, 0, 255],
        "blue" => [0, 0, 255, 255],
        "yellow" => [255, 255, 0, 255],
        "cyan" | "aqua" => [0, 255, 255, 255],
        "magenta" | "fuchsia" => [255, 0, 255, 255],
        "orange" => [255, 165, 0, 255],
        "gray" | "grey" => [128, 128, 128, 255],
        "silver" => [192, 192, 192, 255],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_forms() {
        assert_eq!(parse_css_color("rgba(255,255,255,1)"), Some([255, 255, 255, 255]));
        assert_eq!(parse_css_color("rgba(0,0,0,0.6)"), Some([0, 0, 0, 153]));
        assert_eq!(parse_css_color(" RGBA( 10 , 20 , 30 , 0.5 ) "), Some([10, 20, 30, 128]));
        assert_eq!(parse_css_color("rgb(1,2,3)"), Some([1, 2, 3, 255]));
        assert_eq!(parse_css_color("rgb(1 2 3 / 50%)"), Some([1, 2, 3, 128]));
        assert_eq!(parse_css_color("rgb(100%, 0%, 50%)"), Some([255, 0, 128, 255]));
        // Clamped like browsers.
        assert_eq!(parse_css_color("rgba(300,-5,20,2)"), Some([255, 0, 20, 255]));
        // The settings colour picker can write a 0 alpha as "0".
        assert_eq!(parse_css_color("rgba(5,6,7,0)"), Some([5, 6, 7, 0]));
    }

    #[test]
    fn hex_and_names() {
        assert_eq!(parse_css_color("#fff"), Some([255, 255, 255, 255]));
        assert_eq!(parse_css_color("#0008"), Some([0, 0, 0, 136]));
        assert_eq!(parse_css_color("#102030"), Some([16, 32, 48, 255]));
        assert_eq!(parse_css_color("#10203040"), Some([16, 32, 48, 64]));
        assert_eq!(parse_css_color("white"), Some([255, 255, 255, 255]));
        assert_eq!(parse_css_color("Transparent"), Some([0, 0, 0, 0]));
    }

    #[test]
    fn rejects_garbage() {
        for bad in [
            "",
            "#12",
            "#12345",
            "#ggg",
            "rgba(1,2)",
            "rgba(a,b,c)",
            "rgb(1,2,3,4,5)",
            "url(x)",
            "hsl(1,2%,3%)",
        ] {
            assert_eq!(parse_css_color(bad), None, "{bad:?}");
        }
        assert_eq!(css_color_or("nope", [1, 2, 3, 4]), [1, 2, 3, 4]);
    }
}
