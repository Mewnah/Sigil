//! Secrets compiled into `Sigil.exe` at build time (never the JS bundle).
//!
//! `build.rs` reads the names listed under `secret` in `scripts/release/env-classification.json` from
//! the process environment and the untracked `.env`, XORs each value with a per-build key and hands the
//! hex to this crate through `cargo:rustc-env` (`embed_codec.rs` holds both halves of the format).
//! This module decodes them on demand. The secrets are the Kick OAuth client secret (Kick has no
//! public-client flow), the optional Twitch client secret of a confidential app, and the Google Fonts
//! API key. They are used by [`crate::services::oauth_token`] and [`crate::services::google_fonts`]
//! and by nothing else; the frontend only learns whether one is configured, never its value.
//!
//! This is obfuscation, not protection: the key is in the same binary, and the plaintext is in memory
//! (and on the wire to the provider) while the app runs. It keeps the values out of a search of the
//! binary's text and out of any served file. Rotate a leaked secret in the provider's console.
//!
//! A build without a secret compiles; its getter returns `None` and the feature reports it.
//! Nothing here logs or formats a secret. Messages built from provider answers go through [`scrub`].

#[path = "../../embed_codec.rs"]
#[allow(dead_code)] // `build.rs` uses the encode side, the crate the decode side
pub mod codec;

/// One compiled-in value: the variable name it came from and its XORed hex, if the build had it.
type Slot = (&'static str, Option<&'static str>);

macro_rules! slot {
    ($name:literal) => {
        ($name, option_env!(concat!("SIGIL_EMBED_V_", $name)))
    };
}

/// Every secret `build.rs` can embed. A unit test keeps this list equal to the `secret` names of
/// `scripts/release/env-classification.json`.
const SLOTS: &[Slot] = &[
    slot!("SIGIL_KICK_CLIENT_SECRET"),
    slot!("CURSES_KICK_CLIENT_SECRET"),
    slot!("SIGIL_TWITCH_CLIENT_SECRET"),
    slot!("CURSES_TWITCH_CLIENT_SECRET"),
    slot!("SIGIL_GOOGLE_FONTS_API_KEY"),
    slot!("VITE_GOOGLE_FONTS_API_KEY"),
];

/// The per-build XOR key (hex); `None` in a build that embedded nothing.
const KEY: Option<&str> = option_env!("SIGIL_EMBED_KEY");

/// The first of `names` that `slots` holds, decoded with `key`. The order is the caller's precedence
/// (`SIGIL_*` before the legacy `CURSES_*`, the same order the frontend used for its env fallbacks).
fn first_of(slots: &[Slot], key: Option<&str>, names: &[&str]) -> Option<String> {
    let key = key?;
    names.iter().find_map(|name| {
        let cipher = slots.iter().find(|(n, _)| n == name).and_then(|(_, c)| *c)?;
        codec::decode(cipher, key)
    })
}

fn get(names: &[&str]) -> Option<String> {
    first_of(SLOTS, KEY, names)
}

/// Kick OAuth client secret (`SIGIL_KICK_CLIENT_SECRET`, legacy `CURSES_KICK_CLIENT_SECRET`).
pub fn kick_client_secret() -> Option<String> {
    get(&["SIGIL_KICK_CLIENT_SECRET", "CURSES_KICK_CLIENT_SECRET"])
}

/// Twitch OAuth client secret of a confidential app (`SIGIL_TWITCH_CLIENT_SECRET`, legacy
/// `CURSES_TWITCH_CLIENT_SECRET`). `None` means a Public app: the implicit grant.
pub fn twitch_client_secret() -> Option<String> {
    get(&["SIGIL_TWITCH_CLIENT_SECRET", "CURSES_TWITCH_CLIENT_SECRET"])
}

/// Google Fonts Developer API key (`SIGIL_GOOGLE_FONTS_API_KEY`, legacy `VITE_GOOGLE_FONTS_API_KEY`).
pub fn google_fonts_api_key() -> Option<String> {
    get(&["SIGIL_GOOGLE_FONTS_API_KEY", "VITE_GOOGLE_FONTS_API_KEY"])
}

/// `message` with every occurrence of each non-empty entry of `secrets` replaced by `<redacted>`.
/// Applied to text built from a provider's answer or a request error before it reaches a log line, a
/// toast or the frontend, in case a server echoes what it was sent.
pub fn scrub(message: &str, secrets: &[&str]) -> String {
    let mut out = message.to_string();
    for secret in secrets.iter().filter(|s| !s.is_empty()) {
        out = out.replace(secret, "<redacted>");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fake values only; the real ones never appear in tests.
    const KEY_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

    fn cipher(plain: &str) -> &'static str {
        let key = codec::hex_decode(KEY_HEX).unwrap();
        Box::leak(codec::encode(plain, &key).into_boxed_str())
    }

    #[test]
    fn first_of_follows_the_callers_precedence() {
        let slots: Vec<Slot> = vec![
            ("SIGIL_X", Some(cipher("fake-new"))),
            ("CURSES_X", Some(cipher("fake-legacy"))),
            ("SIGIL_Y", None),
            ("CURSES_Y", Some(cipher("fake-only-legacy"))),
        ];
        let k = Some(KEY_HEX);
        assert_eq!(first_of(&slots, k, &["SIGIL_X", "CURSES_X"]).as_deref(), Some("fake-new"));
        assert_eq!(first_of(&slots, k, &["CURSES_X", "SIGIL_X"]).as_deref(), Some("fake-legacy"));
        // The legacy name still works when only it is set.
        assert_eq!(first_of(&slots, k, &["SIGIL_Y", "CURSES_Y"]).as_deref(), Some("fake-only-legacy"));
        assert_eq!(first_of(&slots, k, &["SIGIL_Z"]), None, "an unknown name");
    }

    #[test]
    fn nothing_embedded_gives_none() {
        let slots: Vec<Slot> = vec![("SIGIL_X", None)];
        assert_eq!(first_of(&slots, Some(KEY_HEX), &["SIGIL_X"]), None);
        // A value without a key cannot be decoded (a build that lost its key variable).
        let slots: Vec<Slot> = vec![("SIGIL_X", Some(cipher("fake")))];
        assert_eq!(first_of(&slots, None, &["SIGIL_X"]), None);
        assert_eq!(first_of(&slots, Some("zz"), &["SIGIL_X"]), None);
    }

    #[test]
    fn slots_are_exactly_the_secrets_classified_in_the_env_file() {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../../../scripts/release/env-classification.json")).unwrap();
        let mut classified: Vec<&str> =
            doc["secret"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let mut slots: Vec<&str> = SLOTS.iter().map(|(n, _)| *n).collect();
        classified.sort_unstable();
        slots.sort_unstable();
        assert_eq!(slots, classified, "src/services/embedded_secrets.rs SLOTS and env-classification.json drifted");
        // And nothing is both public (Vite exposes it) and secret.
        for public in doc["public"].as_array().unwrap() {
            assert!(!classified.contains(&public.as_str().unwrap()), "{public} is classified both ways");
        }
    }

    #[test]
    fn scrub_removes_every_occurrence_and_ignores_empty_secrets() {
        let msg = "Kick token request: HTTP 400: bad fake-secret and again fake-secret / other-fake";
        let out = scrub(msg, &["fake-secret", "", "other-fake"]);
        assert!(!out.contains("fake-secret") && !out.contains("other-fake"), "{out}");
        assert_eq!(out.matches("<redacted>").count(), 3);
        assert_eq!(scrub("clean text", &["fake-secret"]), "clean text");
        assert_eq!(scrub("clean text", &[""]), "clean text");
    }
}
