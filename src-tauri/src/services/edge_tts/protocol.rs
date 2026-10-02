//! The Edge "Read aloud" protocol as pure functions (P6-H): URLs, the browser headers, the
//! `Sec-MS-GEC` token, the two request messages and parsing of the service's messages.
//!
//! Follows the protocol the Edge browser speaks (as documented by the `edge-tts` project, whose
//! constants these mirror): a WebSocket per request to
//! `wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1` with the trusted
//! client token, a connection id, `Sec-MS-GEC` (SHA-256 of the 5-minute Windows-epoch tick and the
//! token, upper-case hex) and `Sec-MS-GEC-Version`; then a `speech.config` text message, an `ssml`
//! text message, and the answers: `turn.start`, `audio.metadata`, binary `audio` messages (2-byte
//! big-endian header length, the header lines, MP3 data) and `turn.end`. The service is unofficial:
//! when Microsoft changes it, these constants are the place to update.

use sha2::{Digest, Sha256};

use super::cloud::{civil_from_days, MONTHS, WEEKDAYS};

/// The Read-aloud trusted client token (public: every Edge browser sends it).
pub(crate) const TRUSTED_CLIENT_TOKEN: &str = "6A5AA1D4EAFF4E9FB37E23D68491D6F4";
/// Synthesis WebSocket (query parameters added per request).
pub(crate) const WSS_URL: &str = "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1";
/// Voice list.
pub(crate) const VOICES_URL: &str = "https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list";
/// The Edge version the service sees (`Sec-MS-GEC-Version`, user agent).
pub(crate) const CHROMIUM_FULL_VERSION: &str = "143.0.3650.75";
pub(crate) const CHROMIUM_MAJOR_VERSION: &str = "143";
/// `Origin` of the Read-aloud extension.
pub(crate) const ORIGIN: &str = "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold";
/// The only output format every client of the service uses (24 kHz mono MP3, 48 kbit/s).
pub(crate) const OUTPUT_FORMAT: &str = "audio-24khz-48kbitrate-mono-mp3";
/// Sample rate of [`OUTPUT_FORMAT`].
pub(crate) const SAMPLE_RATE: u32 = 24_000;
/// Seconds from 1601-01-01 (Windows file time) to 1970-01-01.
const WIN_EPOCH_SECS: i64 = 11_644_473_600;
/// Largest escaped text per request (the service's limit is 4096 bytes of SSML text).
pub(crate) const MAX_TEXT_BYTES: usize = 4096;

/// `Sec-MS-GEC` for the clock time `unix_secs` (already corrected for clock skew): the time in
/// Windows file-time ticks (100 ns), rounded down to 5 minutes, followed by the client token,
/// hashed with SHA-256, upper-case hex.
pub(crate) fn sec_ms_gec(unix_secs: i64) -> String {
    let mut secs = unix_secs + WIN_EPOCH_SECS;
    secs -= secs.rem_euclid(300);
    let ticks = i128::from(secs) * 10_000_000;
    let digest = Sha256::digest(format!("{ticks}{TRUSTED_CLIENT_TOKEN}").as_bytes());
    hex::encode_upper(digest)
}

/// `Sec-MS-GEC-Version`.
pub(crate) fn sec_ms_gec_version() -> String {
    format!("1-{CHROMIUM_FULL_VERSION}")
}

/// The Edge user agent.
pub(crate) fn user_agent() -> String {
    format!(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 Safari/537.36 Edg/{v}.0.0.0",
        v = CHROMIUM_MAJOR_VERSION
    )
}

/// `Sec-CH-UA` of the voice-list request.
pub(crate) fn sec_ch_ua() -> String {
    format!("\" Not;A Brand\";v=\"99\", \"Microsoft Edge\";v=\"{v}\", \"Chromium\";v=\"{v}\"", v = CHROMIUM_MAJOR_VERSION)
}

/// The synthesis URL for one connection (`base` = [`WSS_URL`], or a test server).
pub(crate) fn wss_url(base: &str, connection_id: &str, gec: &str) -> String {
    format!(
        "{base}?TrustedClientToken={TRUSTED_CLIENT_TOKEN}&ConnectionId={connection_id}&Sec-MS-GEC={gec}&Sec-MS-GEC-Version={}",
        sec_ms_gec_version()
    )
}

/// The voice-list URL (`base` = [`VOICES_URL`], or a test server).
pub(crate) fn voices_url(base: &str, gec: &str) -> String {
    format!("{base}?trustedclienttoken={TRUSTED_CLIENT_TOKEN}&Sec-MS-GEC={gec}&Sec-MS-GEC-Version={}", sec_ms_gec_version())
}

/// `X-Timestamp` of the messages: `Thu Oct 01 2026 12:34:56 GMT+0000 (Coordinated Universal Time)`.
pub(crate) fn js_date(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{} {} {:02} {} {:02}:{:02}:{:02} GMT+0000 (Coordinated Universal Time)",
        WEEKDAYS[days.rem_euclid(7) as usize],
        MONTHS[(m - 1) as usize],
        d,
        y,
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

/// The `speech.config` message (sent once per connection, before the SSML).
pub(crate) fn speech_config_message(timestamp: &str) -> String {
    format!(
        "X-Timestamp:{timestamp}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n\
{{\"context\":{{\"synthesis\":{{\"audio\":{{\"metadataoptions\":{{\"sentenceBoundaryEnabled\":\"false\",\"wordBoundaryEnabled\":\"true\"}},\"outputFormat\":\"{OUTPUT_FORMAT}\"}}}}}}}}\r\n"
    )
}

/// The `ssml` message. `escaped_text` must already be XML-escaped ([`escape_text`]). The Edge
/// browser appends a `Z` to this timestamp; the service expects it.
pub(crate) fn ssml_message(request_id: &str, timestamp: &str, voice: &str, rate: &str, pitch: &str, escaped_text: &str) -> String {
    format!(
        "X-RequestId:{request_id}\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:{timestamp}Z\r\nPath:ssml\r\n\r\n\
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='{voice}'>\
<prosody pitch='{pitch}' rate='{rate}' volume='+0%'>{escaped_text}</prosody></voice></speak>"
    )
}

/// Whether `c` may appear in a voice name inside the SSML `name='…'` attribute.
fn voice_char_ok(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_' | '(' | ')' | ',' | '.')
}

/// The service's voice name for `voice`: a short name becomes the long form the SSML names, e.g.
/// `en-US-AvaMultilingualNeural` → `Microsoft Server Speech Text to Speech Voice (en-US,
/// AvaMultilingualNeural)`; the long form passes through. A short name is `<locale>-<Name>Neural`
/// whose locale is `<lang>[-<Script>]-<REGION>[-<subregion>…]`: the service lists voices with a
/// script subtag (`iu-Cans-CA-SiqiniqNeural`, `iu-Latn-CA-…`) and with a sub-region
/// (`zh-CN-liaoning-XiaobeiNeural`). Anything else (characters that could break out of the SSML
/// attribute) is refused. Mirrored by `isEdgeVoiceName` in `src/core/services/tts/services/edge.ts`.
pub(crate) fn voice_name(voice: &str) -> Result<String, String> {
    let v = voice.trim();
    if v.is_empty() || v.len() > 160 || !v.chars().all(voice_char_ok) {
        return Err(format!("\"{}\" is not an Edge voice name.", v.chars().take(80).collect::<String>()));
    }
    if v.starts_with("Microsoft Server Speech Text to Speech Voice (") && v.ends_with(')') {
        return Ok(v.to_string());
    }
    let refused = || format!("\"{v}\" is not an Edge voice name (expected one like en-US-AvaMultilingualNeural).");
    let segments: Vec<&str> = v.split('-').collect();
    let Some((name, locale)) = segments.split_last() else {
        return Err(refused());
    };
    let name_ok = name.len() > "Neural".len() && name.ends_with("Neural") && name.chars().all(|c| c.is_ascii_alphanumeric());
    if !name_ok || !short_locale_ok(locale) {
        return Err(refused());
    }
    Ok(format!("Microsoft Server Speech Text to Speech Voice ({}, {name})", locale.join("-")))
}

/// Whether `segments` form a voice locale: `<lang>[-<Script>]-<REGION>[-<subregion>…]` (`en-US`,
/// `iu-Cans-CA`, `zh-CN-liaoning`).
fn short_locale_ok(segments: &[&str]) -> bool {
    let all = |s: &str, f: fn(&char) -> bool| s.chars().all(|c| f(&c));
    let Some((lang, rest)) = segments.split_first() else {
        return false;
    };
    if !(2..=8).contains(&lang.len()) || !all(lang, char::is_ascii_lowercase) {
        return false;
    }
    // An optional ISO 15924 script subtag: `Cans`, `Latn`.
    let script = |s: &str| s.len() == 4 && s.starts_with(|c: char| c.is_ascii_uppercase()) && all(&s[1..], char::is_ascii_lowercase);
    let rest = match rest.first() {
        Some(s) if script(s) => &rest[1..],
        _ => rest,
    };
    let Some((region, subregions)) = rest.split_first() else {
        return false;
    };
    let region_ok = (region.len() >= 2 && all(region, char::is_ascii_uppercase)) || (region.len() == 3 && all(region, char::is_ascii_digit));
    region_ok && subregions.iter().all(|s| !s.is_empty() && all(s, char::is_ascii_lowercase))
}

/// SSML `rate` for a speed factor (1 = normal): `+0%`, `-25%`, `+50%` (clamped to 0.5–2).
pub(crate) fn rate_attr(rate: f32) -> String {
    let r = if rate.is_finite() && rate > 0.0 { rate.clamp(0.5, 2.0) } else { 1.0 };
    let pct = ((r - 1.0) * 100.0).round() as i32;
    format!("{pct:+}%")
}

/// SSML `pitch` for an offset in Hz: `+0Hz`, `-20Hz` (clamped to ±100 Hz).
pub(crate) fn pitch_attr(hz: i32) -> String {
    format!("{:+}Hz", hz.clamp(-100, 100))
}

/// `text` made safe for the SSML body: control characters the service refuses become spaces, and
/// `&`, `<`, `>` are escaped (quotes may stay in element text).
pub(crate) fn escape_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\u{0}'..='\u{8}' | '\u{b}' | '\u{c}' | '\u{e}'..='\u{1f}' => out.push(' '),
            // Unpaired surrogates cannot occur in a Rust string; other characters pass.
            c => out.push(c),
        }
    }
    out
}

/// Split escaped text into pieces of at most `max_bytes`, preferring a break after whitespace or
/// punctuation, never inside a character or an `&…;` entity.
pub(crate) fn split_escaped(text: &str, max_bytes: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    let max_bytes = max_bytes.max(16);
    while rest.len() > max_bytes {
        // The last char boundary within the limit.
        let mut cut = max_bytes;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        // Do not cut inside an entity.
        if let Some(amp) = rest[..cut].rfind('&') {
            if !rest[amp..cut].contains(';') {
                cut = amp;
            }
        }
        // Prefer a natural break in the last quarter of the piece.
        let head = &rest[..cut];
        let natural = head
            .char_indices()
            .rev()
            .take_while(|(i, _)| *i >= cut * 3 / 4)
            .find(|(_, c)| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '。' | '、' | '！' | '？'))
            .map(|(i, c)| i + c.len_utf8());
        let mut cut = natural.filter(|&n| n > 0).unwrap_or(cut);
        if cut == 0 {
            // Cannot happen for escaped text (entities are short); still always make progress.
            cut = rest.chars().next().map_or(rest.len(), char::len_utf8);
        }
        let piece = rest[..cut].trim();
        if !piece.is_empty() {
            out.push(piece.to_string());
        }
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// Header lines (`Name:value`) of a service message, and its body.
pub(crate) fn split_text_message(msg: &str) -> (Vec<(&str, &str)>, &str) {
    let (head, body) = msg.split_once("\r\n\r\n").unwrap_or((msg, ""));
    let headers = head
        .split("\r\n")
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();
    (headers, body)
}

/// A header value (case-insensitive name).
pub(crate) fn header<'a>(headers: &[(&'a str, &'a str)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| *v)
}

/// A binary message of the service.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BinaryMessage<'a> {
    pub path: Option<String>,
    pub content_type: Option<String>,
    pub data: &'a [u8],
}

/// Parse a binary message: a 2-byte big-endian header length, the header lines, then the data.
pub(crate) fn parse_binary_message(msg: &[u8]) -> Result<BinaryMessage<'_>, String> {
    if msg.len() < 2 {
        return Err("the service sent a binary message without a header length.".into());
    }
    let len = usize::from(u16::from_be_bytes([msg[0], msg[1]]));
    if 2 + len > msg.len() {
        return Err("the service sent a binary message whose header is longer than the message.".into());
    }
    let head = String::from_utf8_lossy(&msg[2..2 + len]);
    let (headers, _) = split_text_message(&head);
    let path = header(&headers, "Path").map(str::to_string);
    let content_type = header(&headers, "Content-Type").map(str::to_string);
    Ok(BinaryMessage { path, content_type, data: &msg[2 + len..] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::edge_tts::cloud::days_from_civil;

    #[test]
    fn sec_ms_gec_hashes_the_rounded_windows_tick() {
        // 2026-10-01 12:34:56 UTC.
        let t = days_from_civil(2026, 10, 1) * 86_400 + 12 * 3600 + 34 * 60 + 56;
        let token = sec_ms_gec(t);
        assert_eq!(token.len(), 64);
        assert!(token.chars().all(|c| c.is_ascii_digit() || c.is_ascii_uppercase()));
        // Constant within a 5-minute window, different in the next one.
        let window_start = t - (t + WIN_EPOCH_SECS).rem_euclid(300);
        assert_eq!(sec_ms_gec(window_start), token);
        assert_eq!(sec_ms_gec(window_start + 299), token);
        assert_ne!(sec_ms_gec(window_start + 300), token);
        // The hashed text: Windows ticks (100 ns) of the window start, then the token.
        let ticks = i128::from(window_start + WIN_EPOCH_SECS) * 10_000_000;
        let expected = hex::encode_upper(Sha256::digest(format!("{ticks}{TRUSTED_CLIENT_TOKEN}")));
        assert_eq!(token, expected);
        assert!(format!("{ticks}").ends_with("0000000"));
    }

    #[test]
    fn urls_and_headers_match_the_browser() {
        let url = wss_url(WSS_URL, "abc", "GEC");
        assert_eq!(
            url,
            format!(
                "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=6A5AA1D4EAFF4E9FB37E23D68491D6F4&ConnectionId=abc&Sec-MS-GEC=GEC&Sec-MS-GEC-Version=1-{CHROMIUM_FULL_VERSION}"
            )
        );
        assert!(voices_url(VOICES_URL, "G").contains("trustedclienttoken=6A5AA1D4EAFF4E9FB37E23D68491D6F4&Sec-MS-GEC=G&"));
        assert!(user_agent().contains("Edg/143.0.0.0"));
        assert!(sec_ch_ua().contains("\"Microsoft Edge\";v=\"143\""));
    }

    #[test]
    fn js_dates_look_like_the_browser_s() {
        let t = days_from_civil(2026, 10, 1) * 86_400 + 12 * 3600 + 4 * 60 + 5;
        assert_eq!(js_date(t), "Thu Oct 01 2026 12:04:05 GMT+0000 (Coordinated Universal Time)");
        assert_eq!(js_date(0), "Thu Jan 01 1970 00:00:00 GMT+0000 (Coordinated Universal Time)");
    }

    #[test]
    fn messages_have_the_expected_shape() {
        let cfg = speech_config_message("T");
        assert!(cfg.starts_with("X-Timestamp:T\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n"));
        let json = cfg.split("\r\n\r\n").nth(1).unwrap().trim_end();
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["context"]["synthesis"]["audio"]["outputFormat"], OUTPUT_FORMAT);
        let ssml = ssml_message("RID", "T", "V", "+0%", "+0Hz", "a &amp; b");
        assert!(ssml.starts_with("X-RequestId:RID\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:TZ\r\nPath:ssml\r\n\r\n<speak "));
        assert!(ssml.contains("<voice name='V'><prosody pitch='+0Hz' rate='+0%' volume='+0%'>a &amp; b</prosody></voice></speak>"));
    }

    #[test]
    fn voice_names_convert_and_refuse_injection() {
        assert_eq!(
            voice_name("en-US-AvaMultilingualNeural").unwrap(),
            "Microsoft Server Speech Text to Speech Voice (en-US, AvaMultilingualNeural)"
        );
        assert_eq!(
            voice_name(" zh-CN-liaoning-XiaobeiNeural ").unwrap(),
            "Microsoft Server Speech Text to Speech Voice (zh-CN-liaoning, XiaobeiNeural)"
        );
        let long = "Microsoft Server Speech Text to Speech Voice (ja-JP, NanamiNeural)";
        assert_eq!(voice_name(long).unwrap(), long);
        assert!(voice_name("").is_err());
        assert!(voice_name("en-US-Ava'><x>Neural").is_err());
        assert!(voice_name("Ava").is_err());
        assert!(voice_name("EN-us-AvaNeural").is_err());
        assert!(voice_name("en-US-Ava").is_err());
        assert!(voice_name("fil-PH-BlessicaNeural").is_ok());
        // Voices the service lists with a script subtag.
        assert_eq!(
            voice_name("iu-Cans-CA-SiqiniqNeural").unwrap(),
            "Microsoft Server Speech Text to Speech Voice (iu-Cans-CA, SiqiniqNeural)"
        );
        assert_eq!(
            voice_name("iu-Latn-CA-TaqqiqNeural").unwrap(),
            "Microsoft Server Speech Text to Speech Voice (iu-Latn-CA, TaqqiqNeural)"
        );
        assert_eq!(voice_name("sr-Latn-RS-NicholasNeural").unwrap(), "Microsoft Server Speech Text to Speech Voice (sr-Latn-RS, NicholasNeural)");
        assert!(voice_name("es-419-ValentinaNeural").is_ok());
        for bad in ["en-US-Neural", "iu-cans-CA-SiqiniqNeural", "iu-Cans-SiqiniqNeural", "en-US--AvaNeural", "en-US-Ava.Neural", "-US-AvaNeural", "en-Us-AvaNeural"] {
            assert!(voice_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prosody_attributes_are_clamped() {
        assert_eq!(rate_attr(1.0), "+0%");
        assert_eq!(rate_attr(1.25), "+25%");
        assert_eq!(rate_attr(0.1), "-50%");
        assert_eq!(rate_attr(9.0), "+100%");
        assert_eq!(rate_attr(f32::NAN), "+0%");
        assert_eq!(pitch_attr(0), "+0Hz");
        assert_eq!(pitch_attr(-20), "-20Hz");
        assert_eq!(pitch_attr(500), "+100Hz");
    }

    #[test]
    fn text_is_escaped_and_split_safely() {
        assert_eq!(escape_text("a<b>&c\u{1}d\ne"), "a&lt;b&gt;&amp;c d\ne");
        let text = "word ".repeat(2000);
        let pieces = split_escaped(&escape_text(&text), MAX_TEXT_BYTES);
        assert!(pieces.len() >= 3);
        assert!(pieces.iter().all(|p| p.len() <= MAX_TEXT_BYTES && !p.is_empty()));
        assert_eq!(pieces.join(" ").split_whitespace().count(), 2000);
        // Never inside a character or an entity.
        let jp = "日本語&amp;".repeat(1000);
        for p in split_escaped(&jp, 100) {
            assert!(p.len() <= 100);
            assert!(!p.ends_with("&am") && !p.ends_with('&'));
            let opens = p.matches('&').count();
            let closes = p.matches(';').count();
            assert_eq!(opens, closes, "{p}");
        }
        assert_eq!(split_escaped("  short  ", 4096), vec!["short".to_string()]);
        assert!(split_escaped("   ", 4096).is_empty());
    }

    #[test]
    fn service_messages_parse() {
        let (h, body) = split_text_message("X-RequestId:1\r\nPath:turn.end\r\n\r\n{}");
        assert_eq!(header(&h, "path"), Some("turn.end"));
        assert_eq!(body, "{}");
        let head = b"X-RequestId:1\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n";
        let mut msg = (head.len() as u16).to_be_bytes().to_vec();
        msg.extend_from_slice(head);
        msg.extend_from_slice(&[1, 2, 3]);
        let b = parse_binary_message(&msg).unwrap();
        assert_eq!(b.path.as_deref(), Some("audio"));
        assert_eq!(b.content_type.as_deref(), Some("audio/mpeg"));
        assert_eq!(b.data, &[1, 2, 3]);
        assert!(parse_binary_message(&[0]).is_err());
        assert!(parse_binary_message(&[0, 9, b'a']).is_err());
    }
}
