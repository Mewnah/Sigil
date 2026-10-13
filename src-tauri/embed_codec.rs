//! Compile-time secret embedding, shared by `build.rs` (encode side) and the crate
//! (`services/embedded_secrets.rs`, decode side). `build.rs` includes this file with
//! `#[path = "embed_codec.rs"] mod embed_codec;`, the crate with a `#[path]` of its own, so both
//! agree on the format and the unit tests run in the crate's test binary (a build script has none).
//!
//! # What this is, and is not
//! The OAuth client secret of Kick and the Google Fonts API key must ship inside the app (Kick has no
//! public-client flow, see `docs/RELEASE_AND_ENGINEERING.md`), but they must not sit in the JS bundle
//! that the embedded web server hands to the LAN. `build.rs` reads them from the process environment
//! and the untracked `.env`, XORs each value with a random per-build key and passes the hex through
//! `cargo:rustc-env`; the crate decodes them at run time. That keeps the values out of a `strings`
//! search of `Sigil.exe` and out of every text file. **It is not protection**: the key is in the same
//! binary, and anyone who can run the app can read the plaintext from memory or from the request it
//! sends. The real defences are that the secret can be rotated in the provider's console and that the
//! redirect URIs it works with are localhost-only.
//!
//! # Names
//! A secret `NAME` travels as `SIGIL_EMBED_V_<NAME>` (hex of the XORed bytes); the key as
//! `SIGIL_EMBED_KEY` (hex). No variable is emitted for a secret that is not set, and `option_env!`
//! then gives `None`: a build without secrets compiles, and the feature says so at run time.
//!
//! Secret values are never printed, logged or put in an error message by this file.

/// Prefix of the per-secret variable `build.rs` emits (`SIGIL_EMBED_V_<NAME>`).
pub const VALUE_VAR_PREFIX: &str = "SIGIL_EMBED_V_";
/// The variable holding the hex of the per-build XOR key.
pub const KEY_VAR: &str = "SIGIL_EMBED_KEY";
/// Bytes of the per-build key.
pub const KEY_LEN: usize = 32;
/// Longest value that is embedded (real secrets are well under 100 bytes).
pub const MAX_SECRET_LEN: usize = 512;

/// Lower-case hex of `bytes`.
pub fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

/// Bytes of a hex string (either case); `None` for odd length or a non-hex character.
pub fn hex_decode(hex: &str) -> Option<Vec<u8>> {
    let hex = hex.as_bytes();
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    hex.chunks(2)
        .map(|pair| Some(nibble(pair[0])? << 4 | nibble(pair[1])?))
        .collect()
}

/// Whether `value` may be embedded: non-empty, at most [`MAX_SECRET_LEN`] bytes, one line, no
/// control characters (a value that failed to parse as one token is skipped, not truncated).
pub fn is_embeddable(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_SECRET_LEN && !value.chars().any(char::is_control)
}

/// XOR `data` with `key` (repeated) and return the hex. An empty key leaves the bytes as they are.
pub fn encode(data: &str, key: &[u8]) -> String {
    let mixed: Vec<u8> = data
        .bytes()
        .enumerate()
        .map(|(i, b)| if key.is_empty() { b } else { b ^ key[i % key.len()] })
        .collect();
    hex_encode(&mixed)
}

/// Inverse of [`encode`]: `cipher_hex` and `key_hex` as emitted by `build.rs`. `None` for bad hex, an
/// empty key or bytes that are not UTF-8.
pub fn decode(cipher_hex: &str, key_hex: &str) -> Option<String> {
    let key = hex_decode(key_hex).filter(|k| !k.is_empty())?;
    let mixed = hex_decode(cipher_hex)?;
    let plain: Vec<u8> = mixed.iter().enumerate().map(|(i, b)| b ^ key[i % key.len()]).collect();
    String::from_utf8(plain).ok().filter(|s| !s.is_empty())
}

/// The `KEY=VALUE` pairs of a dotenv file, in order (a later duplicate wins when looked up from the
/// back). Same shapes Vite's dotenv accepts for single-line values: an optional leading `export`, `#`
/// comment lines, a UTF-8 BOM, single, double or backtick quotes, and an unquoted value that ends at
/// ` #`. Multi-line quoted values and `${VAR}` expansion are not supported (no secret needs them).
pub fn parse_dotenv(text: &str) -> Vec<(String, String)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").map_or(line, str::trim_start);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let valid_key = key.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid_key {
            continue;
        }
        out.push((key.to_string(), parse_dotenv_value(value.trim())));
    }
    out
}

fn parse_dotenv_value(value: &str) -> String {
    for quote in ['"', '\'', '`'] {
        if let Some(rest) = value.strip_prefix(quote) {
            if let Some(end) = rest.find(quote) {
                return rest[..end].to_string();
            }
        }
    }
    match value.find(" #") {
        Some(i) => value[..i].trim_end().to_string(),
        None => value.to_string(),
    }
}

/// The last value of `name` in `pairs` (a later line overrides an earlier one).
pub fn lookup<'a>(pairs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    pairs.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fake values only. Real secrets never appear in tests, fixtures or docs.
    const FAKE: &str = "fake-secret-0123456789-abcdef";
    const KEY: [u8; KEY_LEN] = [
        7, 200, 13, 91, 3, 250, 66, 18, 129, 4, 77, 160, 31, 9, 55, 211, 1, 98, 240, 42, 17, 130, 8, 64, 177, 29, 5,
        99, 150, 12, 201, 35,
    ];

    #[test]
    fn hex_round_trips_and_rejects_garbage() {
        assert_eq!(hex_encode(&[0, 1, 15, 16, 255]), "00010f10ff");
        assert_eq!(hex_decode("00010F10ff").unwrap(), [0, 1, 15, 16, 255]);
        assert_eq!(hex_decode("").unwrap(), Vec::<u8>::new());
        assert!(hex_decode("abc").is_none(), "odd length");
        assert!(hex_decode("zz").is_none(), "not hex");
    }

    #[test]
    fn encode_decode_round_trip_and_hides_the_plain_text() {
        let key_hex = hex_encode(&KEY);
        let cipher = encode(FAKE, &KEY);
        assert_eq!(decode(&cipher, &key_hex).as_deref(), Some(FAKE));
        // The value is not a plain strings hit, in text or as the hex of its bytes.
        assert!(!cipher.contains(FAKE));
        assert!(!cipher.contains(&hex_encode(FAKE.as_bytes())));
        // Values longer than the key wrap around it.
        let long = "x".repeat(KEY_LEN * 3 + 5);
        assert_eq!(decode(&encode(&long, &KEY), &key_hex).as_deref(), Some(long.as_str()));
        // Non-ASCII survives (UTF-8 bytes are XORed, not chars).
        assert_eq!(decode(&encode("kľúč-鍵", &KEY), &key_hex).as_deref(), Some("kľúč-鍵"));
    }

    #[test]
    fn decode_fails_closed() {
        let key_hex = hex_encode(&KEY);
        let cipher = encode(FAKE, &KEY);
        assert!(decode(&cipher, "").is_none(), "no key");
        assert!(decode(&cipher, "zz").is_none(), "bad key");
        assert!(decode("zz", &key_hex).is_none(), "bad cipher");
        assert!(decode("", &key_hex).is_none(), "empty value");
        // A different key gives garbage, not the secret.
        let other = hex_encode(&[1u8; KEY_LEN]);
        assert_ne!(decode(&cipher, &other).as_deref(), Some(FAKE));
        // Bytes that are not UTF-8 after XOR are refused.
        assert!(decode("ff", "00").is_none());
    }

    #[test]
    fn embeddable_values_are_one_short_line() {
        assert!(is_embeddable(FAKE));
        assert!(!is_embeddable(""));
        assert!(!is_embeddable("two\nlines"));
        assert!(!is_embeddable("tab\there"));
        assert!(!is_embeddable(&"a".repeat(MAX_SECRET_LEN + 1)));
        assert!(is_embeddable(&"a".repeat(MAX_SECRET_LEN)));
    }

    #[test]
    fn dotenv_parsing_matches_the_shapes_vite_accepts() {
        let text = "\u{feff}# comment\r\n\
                    SIGIL_A=plain\r\n\
                    export SIGIL_B = spaced value  \n\
                    SIGIL_C=\"quoted # not a comment\"\n\
                    SIGIL_D='single'\n\
                    SIGIL_E=value # trailing comment\n\
                    SIGIL_F=\n\
                    1BAD=skipped\n\
                    no equals sign\n\
                    SIGIL_A=override\n\
                    SIGIL_G=a=b=c\n";
        let pairs = parse_dotenv(text);
        assert_eq!(lookup(&pairs, "SIGIL_A"), Some("override"), "a later line wins");
        assert_eq!(lookup(&pairs, "SIGIL_B"), Some("spaced value"));
        assert_eq!(lookup(&pairs, "SIGIL_C"), Some("quoted # not a comment"));
        assert_eq!(lookup(&pairs, "SIGIL_D"), Some("single"));
        assert_eq!(lookup(&pairs, "SIGIL_E"), Some("value"));
        assert_eq!(lookup(&pairs, "SIGIL_F"), Some(""));
        assert_eq!(lookup(&pairs, "SIGIL_G"), Some("a=b=c"));
        assert_eq!(lookup(&pairs, "1BAD"), None);
        assert_eq!(lookup(&pairs, "MISSING"), None);
    }
}
