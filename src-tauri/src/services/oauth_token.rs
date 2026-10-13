//! `plugin:oauth-token|<cmd>`: the OAuth token calls of Kick and Twitch, run in Rust so the client
//! secret never reaches the frontend.
//!
//! Kick's token endpoint requires `client_secret` for `authorization_code` (with the PKCE
//! `code_verifier`) and for `refresh_token`; there is no public-client flow. A desktop app therefore has
//! to ship the secret, but it must not sit in the JS bundle (which the embedded web server hands to the
//! LAN). It is compiled into `Sigil.exe` by `build.rs` ([`crate::services::embedded_secrets`], lightly
//! obfuscated: not protection) and added here, to a request the frontend describes without it.
//!
//! - IPC (`plugin-commands/oauth-token.txt`; wire camelCase; TS wrapper `src/utils/oauthTokens.ts`):
//!   [`oauth_token_exchange`] (code + PKCE verifier), [`oauth_token_refresh`] and
//!   [`oauth_client_secret_configured`] (a boolean: whether this build holds the provider's secret).
//! - The frontend sends the code, the verifier, the redirect URI (which must be one of Sigil's two
//!   localhost callbacks per provider: the desktop loopback and the Vite dev server) or the refresh
//!   token, plus the public client id; it gets back `{accessToken, refreshToken?, expiresIn?}` and
//!   nothing else of the provider's answer.
//! - Requests are Kick's documented `application/x-www-form-urlencoded` POST to
//!   `https://id.kick.com/oauth/token` (Twitch: `https://id.twitch.tv/oauth2/token`) on the shared
//!   [`http_client`] with a per-request [`TOKEN_TIMEOUT`].
//! - Twitch normally runs as a Public app (implicit grant, no secret); a Confidential app compiles its
//!   secret in and gets the same two commands. A build without the Kick secret still compiles: the
//!   commands answer with a message that names the variable to set.
//! - No code, verifier, token or secret is logged, echoed in an error or kept: provider messages go
//!   through [`scrub`], and a body that is not a token is never quoted.

use std::time::Duration;

use reqwest::header::ACCEPT;
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::embedded_secrets::{self, scrub};
use crate::services::http_client::{
    describe_api_error, describe_request_error, http_client, read_body_capped, HEALTH_TIMEOUT,
};
use crate::services::oauth_loopback::{OAUTH_LOOPBACK_PORT, OAUTH_LOOPBACK_REDIRECT_HOST};

const PLUGIN_NAME: &str = "oauth-token";

/// Per-request timeout of a token call (the frontend's old 30 s).
pub const TOKEN_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest token answer read (a real one is under 2 KiB).
const MAX_TOKEN_BODY: usize = 64 * 1024;
/// Port of the Vite dev server (`pnpm dev`), the second allowed redirect host.
const DEV_SERVER_PORT: u16 = 1420;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Provider {
    Kick,
    Twitch,
}

impl Provider {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "kick" => Ok(Self::Kick),
            "twitch" => Ok(Self::Twitch),
            _ => Err("invalid OAuth provider".to_string()),
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Kick => "kick",
            Self::Twitch => "twitch",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Kick => "Kick",
            Self::Twitch => "Twitch",
        }
    }

    fn token_url(self) -> &'static str {
        match self {
            Self::Kick => "https://id.kick.com/oauth/token",
            Self::Twitch => "https://id.twitch.tv/oauth2/token",
        }
    }

    /// Kick refuses every token request without the secret; a Twitch Public app sends none.
    fn secret_required(self) -> bool {
        self == Self::Kick
    }

    /// The compiled-in secret, if this build has one.
    fn client_secret(self) -> Option<String> {
        match self {
            Self::Kick => embedded_secrets::kick_client_secret(),
            Self::Twitch => embedded_secrets::twitch_client_secret(),
        }
    }

    /// The only redirect URIs Sigil registers: the desktop loopback server of
    /// [`crate::services::oauth_loopback`] and the Vite dev server. Compared exactly.
    fn allowed_redirects(self) -> [String; 2] {
        let path = format!("/oauth/{}/callback", self.slug());
        [
            format!("http://{OAUTH_LOOPBACK_REDIRECT_HOST}:{OAUTH_LOOPBACK_PORT}{path}"),
            format!("http://localhost:{DEV_SERVER_PORT}{path}"),
        ]
    }

    fn missing_secret_message(self) -> String {
        let (name, legacy) = match self {
            Self::Kick => ("SIGIL_KICK_CLIENT_SECRET", "CURSES_KICK_CLIENT_SECRET"),
            Self::Twitch => ("SIGIL_TWITCH_CLIENT_SECRET", "CURSES_TWITCH_CLIENT_SECRET"),
        };
        format!(
            "{} sign-in is not available in this build: no client secret was compiled in. Set {name} (or {legacy}) \
             in .env or the build environment and rebuild Sigil.",
            self.label()
        )
    }
}

/// What the frontend gets back: the token fields it uses, none of the provider's other fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u64>,
}

// ---------------------------------------------------------------------------------------------
// Input validation (the frontend is trusted, but Rust holds the secret: check what it is sent)
// ---------------------------------------------------------------------------------------------

fn check_len(what: &str, value: &str, min: usize, max: usize) -> Result<(), String> {
    let n = value.chars().count();
    if n < min || n > max {
        // Only the length is mentioned: the value may be a code or a token.
        return Err(format!("invalid {what}: expected {min} to {max} characters."));
    }
    Ok(())
}

fn validate_client_id(client_id: &str) -> Result<(), String> {
    check_len("client id", client_id, 1, 128)?;
    if !client_id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) {
        return Err("invalid client id: only letters, digits, '.', '_' and '-' are allowed.".into());
    }
    Ok(())
}

/// A code or token: printable ASCII without spaces, so it cannot smuggle another form field in.
fn validate_opaque(what: &str, value: &str, max: usize) -> Result<(), String> {
    check_len(what, value, 1, max)?;
    if !value.chars().all(|c| c.is_ascii_graphic()) {
        return Err(format!("invalid {what}: printable characters without spaces only."));
    }
    Ok(())
}

/// RFC 7636: 43 to 128 characters of `[A-Za-z0-9-._~]`.
fn validate_verifier(verifier: &str) -> Result<(), String> {
    check_len("code verifier", verifier, 43, 128)?;
    if !verifier.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~')) {
        return Err("invalid code verifier: only the RFC 7636 characters are allowed.".into());
    }
    Ok(())
}

fn validate_redirect(provider: Provider, redirect_uri: &str) -> Result<(), String> {
    let allowed = provider.allowed_redirects();
    if allowed.iter().any(|a| a == redirect_uri) {
        return Ok(());
    }
    Err(format!(
        "invalid redirect_uri: {} sign-in only uses {} or {}.",
        provider.label(),
        allowed[0],
        allowed[1]
    ))
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

type Form = Vec<(&'static str, String)>;

struct ExchangeArgs<'a> {
    client_id: &'a str,
    code: &'a str,
    code_verifier: &'a str,
    redirect_uri: &'a str,
}

/// The `authorization_code` form Kick documents (Twitch takes the same fields). `client_secret` is
/// added only when the build has one.
fn exchange_form(provider: Provider, args: &ExchangeArgs<'_>, secret: Option<&str>) -> Result<Form, String> {
    validate_client_id(args.client_id)?;
    validate_opaque("authorization code", args.code, 2048)?;
    validate_verifier(args.code_verifier)?;
    validate_redirect(provider, args.redirect_uri)?;
    let mut form: Form = vec![
        ("grant_type", "authorization_code".into()),
        ("client_id", args.client_id.into()),
        ("code", args.code.into()),
        ("redirect_uri", args.redirect_uri.into()),
        ("code_verifier", args.code_verifier.into()),
    ];
    if let Some(secret) = secret {
        form.push(("client_secret", secret.into()));
    }
    Ok(form)
}

/// The `refresh_token` form.
fn refresh_form(client_id: &str, refresh_token: &str, secret: Option<&str>) -> Result<Form, String> {
    validate_client_id(client_id)?;
    validate_opaque("refresh token", refresh_token, 4096)?;
    let mut form: Form = vec![
        ("grant_type", "refresh_token".into()),
        ("client_id", client_id.into()),
        ("refresh_token", refresh_token.into()),
    ];
    if let Some(secret) = secret {
        form.push(("client_secret", secret.into()));
    }
    Ok(form)
}

/// Pick the three fields the frontend uses out of a provider's token answer. The body is never quoted
/// in an error: it may hold tokens.
fn parse_token_body(label: &str, body: &[u8]) -> Result<TokenResponse, String> {
    let json: Value = serde_json::from_slice(body)
        .map_err(|_| format!("{label} token request: the answer was not JSON."))?;
    let non_empty = |key: &str| {
        json.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let access_token = non_empty("access_token")
        .ok_or_else(|| format!("{label} token request: the answer has no access_token."))?;
    Ok(TokenResponse {
        access_token,
        refresh_token: non_empty("refresh_token"),
        expires_in: json.get("expires_in").and_then(Value::as_u64),
    })
}

/// POST `form` to `url` and return the token fields. Errors name the provider and the HTTP status or
/// the provider's own message, with `secrets` (the client secret, the code, the verifier, the refresh
/// token) scrubbed out.
async fn post_token(
    client: &Client,
    label: &str,
    url: &str,
    form: &Form,
    secrets: &[&str],
) -> Result<TokenResponse, String> {
    let what = format!("{label} token request");
    let sent = client
        .post(url)
        .header(ACCEPT, "application/json")
        .timeout(TOKEN_TIMEOUT)
        .form(form)
        .send()
        .await;
    let resp = match sent {
        Ok(resp) => resp,
        Err(e) => return Err(scrub(&describe_request_error(&what, &e, TOKEN_TIMEOUT), secrets)),
    };
    if !resp.status().is_success() {
        return Err(scrub(&describe_api_error(&what, resp).await, secrets));
    }
    let body = read_body_capped(&what, resp, MAX_TOKEN_BODY, HEALTH_TIMEOUT).await?;
    parse_token_body(label, &body)
}

/// Exchange an authorization code. `secret` is the provider's client secret (`None` for a Twitch
/// Public app); Kick without one is refused before any request is made.
async fn exchange(
    client: &Client,
    provider: Provider,
    url: &str,
    secret: Option<&str>,
    args: &ExchangeArgs<'_>,
) -> Result<TokenResponse, String> {
    if provider.secret_required() && secret.is_none() {
        return Err(provider.missing_secret_message());
    }
    let form = exchange_form(provider, args, secret)?;
    let secrets = [secret.unwrap_or(""), args.code, args.code_verifier];
    post_token(client, provider.label(), url, &form, &secrets).await
}

/// Redeem a refresh token (same secret rules as [`exchange`]).
async fn refresh(
    client: &Client,
    provider: Provider,
    url: &str,
    secret: Option<&str>,
    client_id: &str,
    refresh_token: &str,
) -> Result<TokenResponse, String> {
    if provider.secret_required() && secret.is_none() {
        return Err(provider.missing_secret_message());
    }
    let form = refresh_form(client_id, refresh_token, secret)?;
    let secrets = [secret.unwrap_or(""), refresh_token];
    post_token(client, provider.label(), url, &form, &secrets).await
}

fn log_outcome(provider: Provider, what: &str, outcome: &Result<TokenResponse, String>) {
    match outcome {
        Ok(_) => tracing::info!(target: "sigil::oauth", "{} {what}: ok", provider.label()),
        Err(e) => tracing::warn!(target: "sigil::oauth", "{} {what} failed: {e}", provider.label()),
    }
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

/// `authorization_code` grant: swap the code from the redirect for tokens, with the PKCE verifier
/// and (when the build has it) the client secret.
#[command]
pub async fn oauth_token_exchange(
    provider: String,
    client_id: String,
    code: String,
    code_verifier: String,
    redirect_uri: String,
) -> Result<TokenResponse, String> {
    let provider = Provider::parse(&provider)?;
    let secret = provider.client_secret();
    let args = ExchangeArgs {
        client_id: &client_id,
        code: &code,
        code_verifier: &code_verifier,
        redirect_uri: &redirect_uri,
    };
    let outcome = exchange(http_client(), provider, provider.token_url(), secret.as_deref(), &args).await;
    log_outcome(provider, "token exchange", &outcome);
    outcome
}

/// `refresh_token` grant: a new access token (and usually a new refresh token).
#[command]
pub async fn oauth_token_refresh(
    provider: String,
    client_id: String,
    refresh_token: String,
) -> Result<TokenResponse, String> {
    let provider = Provider::parse(&provider)?;
    let secret = provider.client_secret();
    let outcome = refresh(
        http_client(),
        provider,
        provider.token_url(),
        secret.as_deref(),
        &client_id,
        &refresh_token,
    )
    .await;
    log_outcome(provider, "token refresh", &outcome);
    outcome
}

/// Whether this build holds the provider's client secret. The value itself never leaves Rust.
#[command]
pub fn oauth_client_secret_configured(provider: String) -> Result<bool, String> {
    Ok(Provider::parse(&provider)?.client_secret().is_some())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            oauth_token_exchange,
            oauth_token_refresh,
            oauth_client_secret_configured
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::http_client::STALL_TIMEOUT;
    use std::collections::HashMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    // Fake values only. The real secret never appears in a test.
    const FAKE_SECRET: &str = "fake-client-secret-0001";
    const FAKE_CODE: &str = "fakeAuthCode0002";
    const FAKE_VERIFIER: &str = "fakeVerifierfakeVerifierfakeVerifierfakeVerifier0003";
    const FAKE_REFRESH: &str = "fakeRefreshToken0004";
    const CLIENT_ID: &str = "01FAKECLIENTID0005";
    const KICK_REDIRECT: &str = "http://localhost:17890/oauth/kick/callback";

    fn test_client() -> Client {
        Client::builder().no_proxy().build().unwrap()
    }

    /// One-shot HTTP server on 127.0.0.1 (never a wildcard address): reads the whole request, answers
    /// with `response` and hands the raw request text back.
    async fn serve_capture(response: String) -> (String, oneshot::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = sock.read(&mut chunk).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let length = text[..head_end]
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                        .unwrap_or(0);
                    if buf.len() >= head_end + 4 + length {
                        break;
                    }
                }
            }
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
            let _ = tx.send(String::from_utf8_lossy(&buf).to_string());
        });
        (format!("http://{addr}/oauth/token"), rx)
    }

    fn json_response(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn form_of(request: &str) -> HashMap<String, String> {
        let body = request.split("\r\n\r\n").nth(1).unwrap_or("");
        body.split('&')
            .filter_map(|pair| pair.split_once('='))
            .map(|(k, v)| (k.to_string(), v.replace("%3A", ":").replace("%2F", "/")))
            .collect()
    }

    fn args() -> ExchangeArgs<'static> {
        ExchangeArgs {
            client_id: CLIENT_ID,
            code: FAKE_CODE,
            code_verifier: FAKE_VERIFIER,
            redirect_uri: KICK_REDIRECT,
        }
    }

    #[test]
    fn providers_parse_and_point_at_the_documented_endpoints() {
        assert_eq!(Provider::parse("kick"), Ok(Provider::Kick));
        assert_eq!(Provider::parse("twitch"), Ok(Provider::Twitch));
        for bad in ["", "Kick", "youtube", "kick ", "../kick"] {
            assert!(Provider::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(Provider::Kick.token_url(), "https://id.kick.com/oauth/token");
        assert_eq!(Provider::Twitch.token_url(), "https://id.twitch.tv/oauth2/token");
        assert!(Provider::Kick.secret_required());
        assert!(!Provider::Twitch.secret_required(), "a Twitch Public app has no secret");
    }

    #[test]
    fn redirects_are_the_two_localhost_callbacks_and_nothing_else() {
        assert_eq!(
            Provider::Kick.allowed_redirects(),
            [
                "http://localhost:17890/oauth/kick/callback".to_string(),
                "http://localhost:1420/oauth/kick/callback".to_string()
            ]
        );
        for ok in Provider::Kick.allowed_redirects() {
            assert!(validate_redirect(Provider::Kick, &ok).is_ok());
        }
        for ok in Provider::Twitch.allowed_redirects() {
            assert!(validate_redirect(Provider::Twitch, &ok).is_ok());
        }
        for bad in [
            "https://tauri.localhost/oauth/kick/callback",
            "http://127.0.0.1:17890/oauth/kick/callback",
            "http://localhost:17890/oauth/kick/callback/",
            "http://localhost:17890/oauth/kick/callback?x=1",
            "http://localhost:17891/oauth/kick/callback",
            "http://localhost:17890/oauth/twitch/callback",
            "http://localhost:17890/oauth/kick/callback/../x",
            "http://evil.example/oauth/kick/callback",
            "http://localhost:17890@evil.example/oauth/kick/callback",
            "",
        ] {
            assert!(validate_redirect(Provider::Kick, bad).is_err(), "{bad:?} must be refused");
        }
        // The other provider's callback is not accepted either.
        assert!(validate_redirect(Provider::Twitch, KICK_REDIRECT).is_err());
    }

    #[test]
    fn exchange_form_has_the_documented_fields() {
        let form = exchange_form(Provider::Kick, &args(), Some(FAKE_SECRET)).unwrap();
        let map: HashMap<_, _> = form.iter().map(|(k, v)| (*k, v.as_str())).collect();
        assert_eq!(map["grant_type"], "authorization_code");
        assert_eq!(map["client_id"], CLIENT_ID);
        assert_eq!(map["client_secret"], FAKE_SECRET);
        assert_eq!(map["code"], FAKE_CODE);
        assert_eq!(map["redirect_uri"], KICK_REDIRECT);
        assert_eq!(map["code_verifier"], FAKE_VERIFIER);
        assert_eq!(form.len(), 6, "no field beyond Kick's documented set");

        let public = exchange_form(Provider::Twitch, &ExchangeArgs { redirect_uri: "http://localhost:17890/oauth/twitch/callback", ..args() }, None).unwrap();
        assert!(public.iter().all(|(k, _)| *k != "client_secret"), "a Public app sends no secret");
    }

    #[test]
    fn refresh_form_has_the_documented_fields() {
        let form = refresh_form(CLIENT_ID, FAKE_REFRESH, Some(FAKE_SECRET)).unwrap();
        let map: HashMap<_, _> = form.iter().map(|(k, v)| (*k, v.as_str())).collect();
        assert_eq!(map["grant_type"], "refresh_token");
        assert_eq!(map["client_id"], CLIENT_ID);
        assert_eq!(map["client_secret"], FAKE_SECRET);
        assert_eq!(map["refresh_token"], FAKE_REFRESH);
        assert_eq!(form.len(), 4);
    }

    #[test]
    fn inputs_are_validated_without_echoing_them() {
        assert!(validate_client_id("").is_err());
        assert!(validate_client_id("id with space").is_err());
        assert!(validate_client_id("id&client_secret=x").is_err());
        assert!(validate_client_id(&"a".repeat(129)).is_err());
        assert!(validate_client_id(CLIENT_ID).is_ok());

        assert!(validate_opaque("authorization code", "", 2048).is_err());
        assert!(validate_opaque("authorization code", "a b", 2048).is_err());
        assert!(validate_opaque("authorization code", "a&b=c\n", 2048).is_err());
        assert!(validate_opaque("authorization code", &"x".repeat(2049), 2048).is_err());
        assert!(validate_opaque("authorization code", FAKE_CODE, 2048).is_ok());

        assert!(validate_verifier("short").is_err());
        assert!(validate_verifier(&"a".repeat(129)).is_err());
        assert!(validate_verifier(&format!("{}!", "a".repeat(50))).is_err());
        assert!(validate_verifier(FAKE_VERIFIER).is_ok());
        assert!(validate_verifier(&"aB3-._~".repeat(8)).is_ok());

        let e = validate_opaque("refresh token", "bad token\nwith-fake-value", 4096).unwrap_err();
        assert!(!e.contains("with-fake-value"), "{e}");
        let e = validate_verifier("sh").unwrap_err();
        assert!(e.contains("43") && e.contains("128"), "{e}");
    }

    #[tokio::test]
    async fn exchange_posts_the_documented_form_and_returns_only_the_token_fields() {
        let body = r#"{"access_token":"fake-access","token_type":"Bearer","refresh_token":"fake-refresh","expires_in":7775999,"scope":"user:read channel:read"}"#;
        let (url, request) = serve_capture(json_response("200 OK", body)).await;
        let got = exchange(&test_client(), Provider::Kick, &url, Some(FAKE_SECRET), &args()).await.unwrap();
        assert_eq!(
            got,
            TokenResponse {
                access_token: "fake-access".into(),
                refresh_token: Some("fake-refresh".into()),
                expires_in: Some(7_775_999)
            }
        );
        let wire = serde_json::to_value(&got).unwrap();
        let mut keys: Vec<_> = wire.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["accessToken", "expiresIn", "refreshToken"], "scope and token_type stay in Rust");

        let request = request.await.unwrap();
        let head = request.lines().next().unwrap();
        assert!(head.starts_with("POST /oauth/token "), "{head}");
        let lower = request.to_ascii_lowercase();
        assert!(lower.contains("content-type: application/x-www-form-urlencoded"), "{request}");
        assert!(lower.contains("accept: application/json"), "{request}");
        let form = form_of(&request);
        assert_eq!(form["grant_type"], "authorization_code");
        assert_eq!(form["client_id"], CLIENT_ID);
        assert_eq!(form["client_secret"], FAKE_SECRET);
        assert_eq!(form["code"], FAKE_CODE);
        assert_eq!(form["code_verifier"], FAKE_VERIFIER);
        assert_eq!(form["redirect_uri"], KICK_REDIRECT);
        // Nothing sensitive in the URL or the headers.
        assert!(!head.contains(FAKE_SECRET) && !head.contains(FAKE_CODE));
    }

    #[tokio::test]
    async fn refresh_posts_the_refresh_grant() {
        let body = r#"{"access_token":"fake-access-2","refresh_token":"fake-refresh-2","expires_in":3600}"#;
        let (url, request) = serve_capture(json_response("200 OK", body)).await;
        let got = refresh(&test_client(), Provider::Kick, &url, Some(FAKE_SECRET), CLIENT_ID, FAKE_REFRESH)
            .await
            .unwrap();
        assert_eq!(got.access_token, "fake-access-2");
        assert_eq!(got.refresh_token.as_deref(), Some("fake-refresh-2"));
        assert_eq!(got.expires_in, Some(3600));
        let form = form_of(&request.await.unwrap());
        assert_eq!(form["grant_type"], "refresh_token");
        assert_eq!(form["refresh_token"], FAKE_REFRESH);
        assert_eq!(form["client_secret"], FAKE_SECRET);
        assert_eq!(form["client_id"], CLIENT_ID);
    }

    #[tokio::test]
    async fn twitch_scope_array_and_a_missing_refresh_token_are_fine() {
        let body = r#"{"access_token":"fake-access","expires_in":14000,"scope":["chat:read","chat:edit"],"token_type":"bearer"}"#;
        let (url, request) = serve_capture(json_response("200 OK", body)).await;
        let twitch_args = ExchangeArgs { redirect_uri: "http://localhost:1420/oauth/twitch/callback", ..args() };
        let got = exchange(&test_client(), Provider::Twitch, &url, None, &twitch_args).await.unwrap();
        assert_eq!(got.refresh_token, None);
        assert_eq!(got.expires_in, Some(14000));
        assert!(!form_of(&request.await.unwrap()).contains_key("client_secret"));
    }

    #[tokio::test]
    async fn provider_errors_name_the_status_and_never_echo_what_was_sent() {
        // A server that echoes the secret, the code and the verifier back in its message.
        let body = format!(
            r#"{{"message":"invalid_grant: bad {FAKE_SECRET} {FAKE_CODE} {FAKE_VERIFIER}"}}"#
        );
        let (url, _request) = serve_capture(json_response("400 Bad Request", &body)).await;
        let err = exchange(&test_client(), Provider::Kick, &url, Some(FAKE_SECRET), &args()).await.unwrap_err();
        assert!(err.starts_with("Kick token request: HTTP 400"), "{err}");
        assert!(err.contains("invalid_grant"), "the provider's reason is kept: {err}");
        for secret in [FAKE_SECRET, FAKE_CODE, FAKE_VERIFIER] {
            assert!(!err.contains(secret), "{err}");
        }
        assert!(err.contains("<redacted>"), "the echo was scrubbed, not dropped by luck: {err}");

        // Twitch words its message differently; the frontend's "invalid client" hint keys on it.
        let body = format!(r#"{{"status":400,"message":"invalid client secret {FAKE_SECRET}"}}"#);
        let (url, _request) = serve_capture(json_response("400 Bad Request", &body)).await;
        let twitch_args = ExchangeArgs { redirect_uri: "http://localhost:17890/oauth/twitch/callback", ..args() };
        let err = exchange(&test_client(), Provider::Twitch, &url, Some(FAKE_SECRET), &twitch_args).await.unwrap_err();
        assert!(err.to_ascii_lowercase().contains("invalid client"), "{err}");
        assert!(!err.contains(FAKE_SECRET), "{err}");
    }

    #[tokio::test]
    async fn an_answer_that_is_not_a_token_is_reported_without_quoting_it() {
        for body in ["<html>not json fake-leak</html>", r#"{"error":"nope","fake-leak":1}"#, r#"{"access_token":""}"#] {
            let (url, _request) = serve_capture(json_response("200 OK", body)).await;
            let err = exchange(&test_client(), Provider::Kick, &url, Some(FAKE_SECRET), &args()).await.unwrap_err();
            assert!(err.starts_with("Kick token request:"), "{err}");
            assert!(!err.contains("fake-leak"), "a body that is not a token is never quoted: {err}");
        }
    }

    #[tokio::test]
    async fn a_dead_endpoint_fails_with_a_connect_message_and_no_secret() {
        // Bind and drop: the port is closed when the request arrives.
        let port = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}/oauth/token");
        let err = refresh(&test_client(), Provider::Kick, &url, Some(FAKE_SECRET), CLIENT_ID, FAKE_REFRESH)
            .await
            .unwrap_err();
        assert!(err.starts_with("Kick token request"), "{err}");
        assert!(!err.contains(FAKE_SECRET) && !err.contains(FAKE_REFRESH), "{err}");
    }

    #[tokio::test]
    async fn kick_without_a_compiled_in_secret_is_refused_before_any_request() {
        // The URL is unreachable on purpose: no request may be made.
        let url = "http://127.0.0.1:9/oauth/token";
        let err = exchange(&test_client(), Provider::Kick, url, None, &args()).await.unwrap_err();
        assert!(err.contains("SIGIL_KICK_CLIENT_SECRET") && err.contains("CURSES_KICK_CLIENT_SECRET"), "{err}");
        assert!(err.contains("rebuild"), "{err}");
        let err = refresh(&test_client(), Provider::Kick, url, None, CLIENT_ID, FAKE_REFRESH).await.unwrap_err();
        assert!(err.contains("SIGIL_KICK_CLIENT_SECRET"), "{err}");
    }

    #[tokio::test]
    async fn invalid_input_never_reaches_the_network() {
        let url = "http://127.0.0.1:9/oauth/token";
        let bad_redirect = ExchangeArgs { redirect_uri: "https://evil.example/cb", ..args() };
        let err = exchange(&test_client(), Provider::Kick, url, Some(FAKE_SECRET), &bad_redirect).await.unwrap_err();
        assert!(err.starts_with("invalid redirect_uri"), "{err}");
        assert!(!err.contains(FAKE_SECRET));
        let bad_code = ExchangeArgs { code: "a b", ..args() };
        assert!(exchange(&test_client(), Provider::Kick, url, Some(FAKE_SECRET), &bad_code).await.unwrap_err().starts_with("invalid authorization code"));
    }

    #[test]
    fn the_token_timeout_fits_the_shared_clients_read_timeout() {
        assert!(TOKEN_TIMEOUT <= STALL_TIMEOUT);
        assert_eq!(TOKEN_TIMEOUT, Duration::from_secs(30));
    }

    #[test]
    fn the_secret_check_command_reports_a_boolean_for_known_providers_only() {
        assert!(oauth_client_secret_configured("kick".into()).is_ok());
        assert!(oauth_client_secret_configured("twitch".into()).is_ok());
        assert!(oauth_client_secret_configured("discord".into()).is_err());
    }
}
