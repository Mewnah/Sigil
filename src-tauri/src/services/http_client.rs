use reqwest::Client;
use std::sync::OnceLock;
use std::time::Duration;

/// Shared client for outbound HTTP from plugins (local TTS/STT, translate, model downloads).
/// Prevents indefinitely hung tasks when a backend stops responding.
pub fn http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest Client builder")
    })
}
