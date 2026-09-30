//! HTTP sidecar TTS for heavy / custom neural models (`POST {endpoint}/speech` → WAV,
//! `GET {endpoint}/health`; reference server: `scripts/neural-sidecar/tts_sidecar.py`).
//!
//! Synthesis streams into the TTS sink (`neural_sidecar_tts_speak_to_sink`, P2-I): the request asks
//! for a streamed answer (`"stream": true`; servers that do not know the member send the whole WAV,
//! which works too) and the shared HTTP speech core (`openai_speech::speak_http`) decodes the WAV as
//! it arrives, so the first audio plays before the server finished the sentence. The output device
//! and the top-level route (device / voice changer / both) replace the old per-backend
//! "send into the voice changer" command.
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_neural_sidecar_tts_endpoint`). Timeouts: health 10 s; synthesis on this PC or the LAN 60 s +
//! 1 s per character to the first audio (max 10 min) and 60 s between chunks, else 60 s / 30 s (see
//! `openai_speech`).

use crate::services::http_client::{api_url, http_client, pick_endpoint, probe_ok};
use crate::services::openai_speech::{speak_http, HttpSpeech, SpeakReport, SpeechFormat};
use serde::Serialize;
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8766";

#[derive(Clone)]
pub struct NeuralSidecarTtsState {
    endpoint: Arc<Mutex<String>>,
}

impl NeuralSidecarTtsState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }

    fn endpoint_for(&self, call: Option<&str>) -> String {
        let stored = self.endpoint.lock().unwrap_or_else(PoisonError::into_inner);
        pick_endpoint(call, &stored)
    }
}

#[derive(Serialize, Debug)]
struct SpeechRequest<'a> {
    text: &'a str,
    voice: &'a str,
    /// Ask for a streamed WAV (the reference sidecar sends a header with placeholder sizes, then
    /// the audio as it is generated).
    stream: bool,
}

#[command]
async fn set_neural_sidecar_tts_endpoint<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    endpoint: String,
) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) =
        endpoint.trim().trim_end_matches('/').to_string();
    Ok(())
}

/// The streaming request for one sentence: `POST {base}/speech {text, voice, stream: true}`.
fn sidecar_spec(base: &str, text: &str, voice: &str) -> Result<HttpSpeech, String> {
    let url = api_url(base, "/speech")?;
    let voice = voice.trim();
    let body = serde_json::to_value(SpeechRequest {
        text,
        voice: if voice.is_empty() { "default" } else { voice },
        stream: true,
    })
    .map_err(|e| format!("Neural sidecar TTS request: {e}"))?;
    Ok(HttpSpeech::new("Neural sidecar TTS", url, body, text, SpeechFormat::Wav))
}

/// Speak `text` into the begun TTS sink utterance `utteranceId` (streamed WAV). Resolves once the
/// last audio was queued; `Err("cancelled")` after stop / skip / `openai_speech_cancel`.
#[command]
async fn neural_sidecar_tts_speak_to_sink<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    text: String,
    voice: String,
    endpoint: Option<String>,
    utterance_id: String,
) -> Result<SpeakReport, String> {
    let base = state.endpoint_for(endpoint.as_deref());
    let spec = sidecar_spec(&base, &text, &voice)?;
    speak_http(spec, &utterance_id).await
}

#[command]
async fn check_neural_sidecar_tts<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    endpoint: Option<String>,
) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint_for(endpoint.as_deref()), "/health") else {
        return Ok(false);
    };
    Ok(probe_ok("Neural sidecar TTS availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("neural-sidecar-tts")
        .setup(|app, _api| {
            app.manage(NeuralSidecarTtsState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_neural_sidecar_tts_endpoint,
            neural_sidecar_tts_speak_to_sink,
            check_neural_sidecar_tts,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::openai_speech::testing::*;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn sidecar_endpoint_override_does_not_touch_stored_value() {
        let s = NeuralSidecarTtsState::new();
        assert_eq!(s.endpoint_for(Some("http://10.0.0.2:9000")), "http://10.0.0.2:9000");
        assert_eq!(s.endpoint_for(None), DEFAULT_ENDPOINT);
    }

    #[test]
    fn sidecar_request_asks_for_a_stream() {
        let spec = sidecar_spec("http://127.0.0.1:8766", "Hi.", "  ").unwrap();
        assert_eq!(spec.url().as_str(), "http://127.0.0.1:8766/speech");
        assert_eq!(spec.body(), &serde_json::json!({"text": "Hi.", "voice": "default", "stream": true}));
        let spec = sidecar_spec("http://127.0.0.1:8766", "Hi.", " alice ").unwrap();
        assert_eq!(spec.body()["voice"], "alice");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sidecar_streamed_wav_plays_while_it_arrives() {
        // What tts_sidecar.py sends in streaming mode: a header with placeholder sizes, then PCM.
        let values: Vec<i16> = (0..6000).map(|i| (i % 500) as i16 - 250).collect();
        let mut body = chunked_head("audio/wav");
        body.extend_from_slice(&chunk(&wav_header(22_050, 1, 1, 16, u32::MAX)));
        for part in pcm16(&values).chunks(2000) {
            body.extend_from_slice(&chunk(part));
        }
        body.extend_from_slice(LAST_CHUNK);
        let (l, base) = listener().await;
        let server = tokio::spawn(async move {
            let (mut sock, _) = l.accept().await.unwrap();
            let seen = read_request(&mut sock).await;
            sock.write_all(&body).await.unwrap();
            let _ = sock.shutdown().await;
            seen
        });
        let (sink, pushes) = Recorder::new();
        let report = speak(sidecar_spec(&base, "Hello there.", "default").unwrap(), "sidecar-test", Box::new(sink))
            .await
            .unwrap();
        let seen = server.await.unwrap();
        assert!(seen.head.starts_with("POST /speech "), "{}", seen.head);
        assert_eq!(seen.json()["stream"], true);
        assert_eq!(report.frames, values.len() as u64);
        assert_eq!(report.sample_rate, 22_050);
        assert_eq!(pushes.samples(), f(&values));
    }

    #[tokio::test]
    async fn empty_text_makes_no_request() {
        // An unroutable endpoint proves no request is attempted.
        let spec = sidecar_spec("http://invalid.invalid", "  ", "v").unwrap();
        assert_eq!(speak_http(spec, "sidecar-empty").await.unwrap().frames, 0);
    }
}
