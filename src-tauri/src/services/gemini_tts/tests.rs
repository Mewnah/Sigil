//! Gemini TTS client tests against a local mock of the API (no network).

use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

use super::*;
use crate::services::edge_tts::cloud::{FailureKind, SpeakError};
use crate::services::openai_speech::testing::{chunk, chunked_head, client, full_response, listener, read_request, Recorder, Seen, LAST_CHUNK};

const KEY: &str = "AIza-test-key-123";

fn pcm_b64(samples: &[i16]) -> String {
    base64::engine::general_purpose::STANDARD.encode(samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())
}

fn sse(event: Option<&str>, data: &Value) -> Vec<u8> {
    let mut s = String::new();
    if let Some(e) = event {
        s.push_str(&format!("event: {e}\n"));
    }
    s.push_str(&format!("data: {data}\n\n"));
    s.into_bytes()
}

fn delta(samples: &[i16]) -> Value {
    json!({"event_type": "content.delta", "index": 0, "delta": {"type": "audio", "data": pcm_b64(samples), "mime_type": "audio/l16"}})
}

/// One scripted HTTP answer.
enum Answer {
    /// A complete response.
    Full(Vec<u8>),
    /// A chunked SSE response: pieces sent with a pause between them, then held open (or ended).
    Sse { pieces: Vec<Vec<u8>>, pause: Duration, hold: Option<Duration> },
}

/// Serve one connection per answer, in order; returns the base URL and the requests seen.
async fn serve(answers: Vec<Answer>) -> (String, tokio::task::JoinHandle<Vec<Seen>>) {
    let (l, base) = listener().await;
    let task = tokio::spawn(async move {
        let mut seen = Vec::new();
        for answer in answers {
            let Ok((mut sock, _)) = l.accept().await else { break };
            seen.push(read_request(&mut sock).await);
            match answer {
                Answer::Full(bytes) => {
                    let _ = sock.write_all(&bytes).await;
                }
                Answer::Sse { pieces, pause, hold } => {
                    let _ = sock.write_all(&chunked_head("text/event-stream")).await;
                    for p in pieces {
                        let _ = sock.write_all(&chunk(&p)).await;
                        let _ = sock.flush().await;
                        if !pause.is_zero() {
                            tokio::time::sleep(pause).await;
                        }
                    }
                    match hold {
                        Some(h) => tokio::time::sleep(h).await,
                        None => {
                            let _ = sock.write_all(LAST_CHUNK).await;
                        }
                    }
                }
            }
            let _ = sock.shutdown().await;
        }
        seen
    });
    (base, task)
}

fn request(base: &str, timeout_ms: u32) -> GeminiTtsRequest {
    GeminiTtsRequest {
        api_key: KEY.into(),
        base_url: base.into(),
        model: "gemini-3.8-flash-tts".into(),
        text: "Have a wonderful day!".into(),
        voice: "Puck".into(),
        style: "cheerful and friendly".into(),
        language: Some("en".into()),
        timeout_ms,
    }
}

fn failure(r: Result<SpeakReport, SpeakError>) -> crate::services::edge_tts::cloud::Failure {
    match r {
        Err(SpeakError::Failed(f)) => f,
        other => panic!("expected a failure, got {other:?}"),
    }
}

fn error_body(code: u16, status: &str, message: &str, details: Value) -> Vec<u8> {
    json!({"error": {"code": code, "status": status, "message": message, "details": details}}).to_string().into_bytes()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_streams_audio_deltas_into_the_sink_with_the_key_in_a_header() {
    let a: Vec<i16> = (0..2400).map(|i| (i % 100) as i16 * 100).collect();
    let b: Vec<i16> = vec![-500; 1200];
    let all: Vec<i16> = a.iter().chain(&b).copied().collect();
    let pieces = vec![
        sse(Some("interaction.start"), &json!({"event_type": "interaction.start", "interaction": {"input": [{"type": "user_input", "content": [{"type": "text", "text": "Have a wonderful day!"}]}]}})),
        sse(Some("content.start"), &json!({"event_type": "content.start", "index": 0, "content": {"type": "audio"}})),
        sse(Some("content.delta"), &delta(&a)),
        sse(Some("content.delta"), &delta(&b)),
        sse(Some("content.stop"), &json!({"event_type": "content.stop", "index": 0})),
        // The summary repeats the audio: it must not play twice.
        sse(Some("interaction.complete"), &json!({"event_type": "interaction.complete", "interaction": {"steps": [{"type": "model_output", "content": [{"type": "audio", "data": pcm_b64(&all)}]}]}})),
    ];
    let (base, task) = serve(vec![Answer::Sse { pieces, pause: Duration::ZERO, hold: None }]).await;
    let (rec, pushes) = Recorder::new();
    let report = speak_with(&request(&base, 3000), "gem-ok", Box::new(rec), &client()).await.unwrap();
    assert_eq!(report.sample_rate, 24_000);
    assert_eq!(report.frames, all.len() as u64);
    let got = pushes.samples();
    assert_eq!(got.len(), all.len());
    assert!((got[3] - 300.0 / 32_768.0).abs() < 1e-6);
    assert!((got[all.len() - 1] + 500.0 / 32_768.0).abs() < 1e-6);

    let seen = &task.await.unwrap()[0];
    assert_eq!(seen.path(), "/v1beta/interactions");
    assert_eq!(seen.header("x-goog-api-key").as_deref(), Some(KEY));
    assert!(!seen.head.split("\r\n").next().unwrap().contains(KEY), "the key is never in the URL");
    assert!(seen.header("accept").unwrap().contains("text/event-stream"));
    let body = seen.json();
    assert_eq!(body["model"], "gemini-3.8-flash-tts");
    assert_eq!(body["stream"], true);
    assert_eq!(body["response_format"]["type"], "audio");
    assert_eq!(body["generation_config"]["speech_config"][0]["voice"], "Puck");
    assert_eq!(body["input"][0]["type"], "user_input");
    assert_eq!(body["input"][0]["content"][0]["text"], "Have a wonderful day!");
    assert_eq!(body["input"][0]["content"][0]["annotations"][0]["style"], "cheerful and friendly");
}

/// The stream shape of Google's speech-generation docs (2026-10): `step.delta` events whose `delta` is
/// `{type: "audio", data}` (headerless 24 kHz L16), named only by their JSON `event_type`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_streams_documented_step_deltas() {
    let a: Vec<i16> = vec![300; 1200];
    let b: Vec<i16> = vec![-300; 2400];
    let step_delta = |samples: &[i16]| json!({"event_type": "step.delta", "index": 0, "delta": {"type": "audio", "data": pcm_b64(samples)}});
    let all: Vec<i16> = a.iter().chain(&b).copied().collect();
    let pieces = vec![
        sse(None, &json!({"event_type": "interaction.start", "interaction": {"id": "i-1", "status": "in_progress"}})),
        sse(None, &json!({"event_type": "step.start", "index": 0, "step": {"type": "model_output"}})),
        sse(None, &step_delta(&a)),
        sse(None, &step_delta(&b)),
        // A completed step that repeats its audio must not play twice.
        sse(None, &json!({"event_type": "step.complete", "index": 0, "step": {"type": "model_output", "content": [{"type": "audio", "data": pcm_b64(&all)}]}})),
        sse(None, &json!({"event_type": "interaction.complete", "interaction": {"id": "i-1", "status": "completed"}})),
    ];
    let (base, _) = serve(vec![Answer::Sse { pieces, pause: Duration::from_millis(20), hold: None }]).await;
    let (rec, pushes) = Recorder::new();
    let report = speak_with(&request(&base, 3000), "gem-steps", Box::new(rec), &client()).await.unwrap();
    assert_eq!(report.sample_rate, 24_000);
    assert_eq!(report.frames, all.len() as u64);
    let got = pushes.samples();
    assert_eq!(got.len(), all.len());
    assert!((got[0] - 300.0 / 32_768.0).abs() < 1e-6);
    assert!((got[all.len() - 1] + 300.0 / 32_768.0).abs() < 1e-6);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_uses_summary_audio_when_nothing_streamed() {
    let audio: Vec<i16> = vec![1000; 4800];
    let pieces = vec![sse(None, &json!({"event_type": "interaction.complete", "interaction": {"steps": [{"type": "model_output", "content": [{"type": "audio", "data": pcm_b64(&audio)}]}]}}))];
    let (base, _) = serve(vec![Answer::Sse { pieces, pause: Duration::ZERO, hold: None }]).await;
    let (rec, pushes) = Recorder::new();
    let report = speak_with(&request(&base, 3000), "gem-summary", Box::new(rec), &client()).await.unwrap();
    assert_eq!(report.frames, 4800);
    assert_eq!(pushes.samples().len(), 4800);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_a_whole_json_answer_with_a_wav_plays() {
    let mut wav = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec { channels: 1, sample_rate: 24_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::new(&mut wav, spec).unwrap();
        for _ in 0..1000 {
            w.write_sample(2000_i16).unwrap();
        }
        w.finalize().unwrap();
    }
    let body = json!({"steps": [{"type": "model_output", "content": [{"type": "audio", "data": base64::engine::general_purpose::STANDARD.encode(wav.get_ref()), "mime_type": "audio/wav"}]}]});
    let (base, _) = serve(vec![Answer::Full(full_response("200 OK", "application/json", body.to_string().as_bytes()))]).await;
    let (rec, _) = Recorder::new();
    let report = speak_with(&request(&base, 3000), "gem-json", Box::new(rec), &client()).await.unwrap();
    assert_eq!(report.frames, 1000);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_falls_back_to_generate_content_where_interactions_is_missing() {
    let chunk_of = |samples: &[i16]| json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": pcm_b64(samples)}}]}}]});
    let gc_answer = || Answer::Sse { pieces: vec![sse(None, &chunk_of(&[100; 600])), sse(None, &chunk_of(&[200; 600]))], pause: Duration::ZERO, hold: None };
    let (base, task) = serve(vec![
        Answer::Full(full_response("404 Not Found", "text/html", b"<html><title>Error 404 (Not Found)!!1</title></html>")),
        gc_answer(),
        gc_answer(),
    ])
    .await;
    let (rec, _) = Recorder::new();
    let report = speak_with(&request(&base, 3000), "gem-gc1", Box::new(rec), &client()).await.unwrap();
    assert_eq!(report.frames, 1200);
    // Remembered: the next sentence goes straight to generateContent.
    let (rec, _) = Recorder::new();
    speak_with(&request(&base, 3000), "gem-gc2", Box::new(rec), &client()).await.unwrap();
    let seen = task.await.unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].path(), "/v1beta/interactions");
    assert_eq!(seen[1].path(), "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent?alt=sse");
    assert_eq!(seen[2].path(), seen[1].path());
    let body = seen[1].json();
    assert_eq!(body["generationConfig"]["responseModalities"][0], "AUDIO");
    assert_eq!(body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"], "Puck");
    // Gemini 3.8 reads its prompt verbatim: no style words in front of the text on this path.
    assert_eq!(body["contents"][0]["parts"][0]["text"], "Have a wonderful day!");
    assert_eq!(seen[1].header("x-goog-api-key").as_deref(), Some(KEY));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_errors_are_classified_for_the_fallback() {
    let quota = error_body(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.", json!([{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "21s"}]));
    let bad_key = error_body(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", json!([{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]));
    let busy = error_body(503, "UNAVAILABLE", "The model is overloaded.", json!([]));
    let no_model = error_body(404, "NOT_FOUND", "models/gemini-9-tts is not found for API version v1beta.", json!([]));
    // A refusal of one sentence's text: that sentence only (`no_audio`, no breaker), never the
    // 10-minute `rejected` pause that a refused voice or model gets.
    let chat_prompt = error_body(400, "INVALID_ARGUMENT", "Model tried to generate text, but it should only be used for TTS.", json!([]));
    let too_long = error_body(400, "INVALID_ARGUMENT", "The input text is too long for this model.", json!([]));
    let bare = error_body(400, "INVALID_ARGUMENT", "Request contains an invalid argument.", json!([]));
    let bad_voice = error_body(400, "INVALID_ARGUMENT", "Voice name Xyzzy is not supported.", json!([]));
    let bad_shape = error_body(400, "INVALID_ARGUMENT", "Invalid JSON payload received. Unknown name \"speech_config\": Cannot find field.", json!([]));
    let location = error_body(400, "FAILED_PRECONDITION", "User location is not supported for the API use.", json!([]));
    for (status, body, kind, retry) in [
        ("429 Too Many Requests", quota, FailureKind::RateLimited, Some(Duration::from_secs(21))),
        ("400 Bad Request", bad_key, FailureKind::Auth, None),
        ("503 Service Unavailable", busy, FailureKind::Unavailable, None),
        ("404 Not Found", no_model, FailureKind::Rejected, None),
        ("400 Bad Request", chat_prompt, FailureKind::NoAudio, None),
        ("400 Bad Request", too_long, FailureKind::NoAudio, None),
        ("400 Bad Request", bare, FailureKind::NoAudio, None),
        ("400 Bad Request", bad_voice, FailureKind::Rejected, None),
        ("400 Bad Request", bad_shape, FailureKind::Rejected, None),
        ("400 Bad Request", location, FailureKind::Rejected, None),
    ] {
        let (base, task) = serve(vec![Answer::Full(full_response(status, "application/json", &body))]).await;
        let (rec, _) = Recorder::new();
        let t = Instant::now();
        let f = failure(speak_with(&request(&base, 3000), &format!("gem-err-{status}"), Box::new(rec), &client()).await);
        assert_eq!(f.kind, kind, "{status}: {f:?}");
        assert_eq!(f.retry_after, retry, "{status}");
        assert!(!f.message.contains(KEY));
        assert!(t.elapsed() < Duration::from_secs(1), "an outage is reported at once");
        assert_eq!(task.await.unwrap().len(), 1, "{status}: no second request");
    }

    // The settings check has no sentence: every refusal there is about the settings.
    let chat_prompt = error_body(400, "INVALID_ARGUMENT", "Model tried to generate text, but it should only be used for TTS.", json!([]));
    assert_eq!(api_failure(400, &chat_prompt, None, Asked::Check).kind, FailureKind::Rejected);
    assert_eq!(api_failure(400, &chat_prompt, None, Asked::Speech).kind, FailureKind::NoAudio);
    // An error event inside the stream is about the sentence too.
    let event = json!({"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Model tried to generate text, but it should only be used for TTS."}});
    assert_eq!(stream_failure(&event).kind, FailureKind::NoAudio);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_sink_failures_are_this_pc_s_not_the_service_s() {
    // The sink refuses a push: untagged, with the sink's own text (never a Gemini failure).
    let pieces = vec![sse(Some("content.delta"), &delta(&[3; 2400])), sse(Some("content.delta"), &delta(&[3; 2400]))];
    let (base, _) = serve(vec![Answer::Sse { pieces, pause: Duration::ZERO, hold: None }]).await;
    let (mut rec, _) = Recorder::new();
    rec.fail_with = Some(crate::services::tts_sink::SinkError::Unavailable("timed out waiting for queued speech to play".into()));
    let r = speak_with(&request(&base, 3000), "gem-sinkfail", Box::new(rec), &client()).await;
    assert_eq!(r, Err(SpeakError::Local("TTS sink unavailable: timed out waiting for queued speech to play".into())));
    assert!(!r.unwrap_err().into_wire().starts_with(crate::services::edge_tts::cloud::TAG_PREFIX));

    // A second request for an utterance already being spoken: untagged too.
    let held = REGISTRY.register("gem-dup").unwrap();
    let (rec, _) = Recorder::new();
    let r = speak_with(&request("http://127.0.0.1:9", 3000), "gem-dup", Box::new(rec), &client()).await;
    assert!(matches!(&r, Err(SpeakError::Local(m)) if m.contains("already being spoken")), "{r:?}");
    drop(held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_without_a_key_fails_at_once() {
    let mut r = request("http://127.0.0.1:9", 3000);
    r.api_key = "  ".into();
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&r, "gem-nokey", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::Auth);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_stalls_are_reported_within_the_timeout() {
    // Headers, then nothing.
    let (base, _) = serve(vec![Answer::Sse { pieces: vec![], pause: Duration::ZERO, hold: Some(Duration::from_secs(5)) }]).await;
    let (rec, _) = Recorder::new();
    let t = Instant::now();
    let f = failure(speak_with(&request(&base, 600), "gem-stall1", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::Stalled);
    assert!(!f.partial);
    assert!(t.elapsed() >= Duration::from_millis(550) && t.elapsed() < Duration::from_millis(1500), "{:?}", t.elapsed());

    // Some audio, then nothing: partial.
    let (base, _) = serve(vec![Answer::Sse { pieces: vec![sse(Some("content.delta"), &delta(&[5; 2400]))], pause: Duration::ZERO, hold: Some(Duration::from_secs(5)) }]).await;
    let (rec, pushes) = Recorder::new();
    let f = failure(speak_with(&request(&base, 600), "gem-stall2", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::Stalled);
    assert!(f.partial);
    assert!(pushes.count() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_answers_without_audio_are_no_audio() {
    let text_only = sse(None, &json!({"candidates": [{"content": {"parts": [{"text": "Sure! Here you go."}]}, "finishReason": "STOP"}]}));
    let (base, _) = serve(vec![Answer::Sse { pieces: vec![text_only], pause: Duration::ZERO, hold: None }]).await;
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request(&base, 3000), "gem-text", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::NoAudio);
    assert!(f.message.contains("text instead of audio"), "{}", f.message);

    let safety = sse(None, &json!({"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}));
    let (base, _) = serve(vec![Answer::Sse { pieces: vec![safety], pause: Duration::ZERO, hold: None }]).await;
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request(&base, 3000), "gem-safety", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::NoAudio);
    assert!(f.message.contains("PROHIBITED_CONTENT"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_an_error_event_mid_stream_is_partial() {
    let pieces = vec![
        sse(Some("content.delta"), &delta(&[7; 2400])),
        sse(Some("error"), &json!({"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}})),
    ];
    let (base, _) = serve(vec![Answer::Sse { pieces, pause: Duration::ZERO, hold: None }]).await;
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request(&base, 3000), "gem-midfail", Box::new(rec), &client()).await);
    assert_eq!(f.kind, FailureKind::Unavailable);
    assert!(f.partial);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_cancel_ends_a_request_at_once() {
    let (base, _) = serve(vec![Answer::Sse { pieces: vec![], pause: Duration::ZERO, hold: Some(Duration::from_secs(5)) }]).await;
    let (rec, _) = Recorder::new();
    let task = tokio::spawn(async move { speak_with(&request(&base, 10_000), "gem-cancel", Box::new(rec), &client()).await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t = Instant::now();
    cancel_command(&REGISTRY, Some("gem-cancel"));
    assert_eq!(task.await.unwrap(), Err(SpeakError::Cancelled));
    assert!(t.elapsed() < Duration::from_millis(300));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_check_reads_the_model_without_speaking() {
    let model = json!({"name": "models/gemini-3.8-flash-tts"}).to_string();
    let (base, task) = serve(vec![Answer::Full(full_response("200 OK", "application/json", model.as_bytes()))]).await;
    assert!(check_with(&client(), KEY, "models/gemini-3.8-flash-tts", &base).await.unwrap());
    let seen = &task.await.unwrap()[0];
    assert!(seen.head.starts_with("GET /v1beta/models/gemini-3.8-flash-tts "));
    assert_eq!(seen.header("x-goog-api-key").as_deref(), Some(KEY));

    let (base, _) = serve(vec![Answer::Full(full_response("404 Not Found", "application/json", &error_body(404, "NOT_FOUND", "models/x is not found", json!([]))))]).await;
    let e = check_with(&client(), KEY, "x", &base).await.unwrap_err();
    assert!(e.contains("does not know the model"), "{e}");

    let bad_key = error_body(400, "INVALID_ARGUMENT", "API key not valid.", json!([{"reason": "API_KEY_INVALID"}]));
    let (base, _) = serve(vec![Answer::Full(full_response("400 Bad Request", "application/json", &bad_key))]).await;
    let e = check_with(&client(), KEY, "m", &base).await.unwrap_err();
    assert!(e.contains("API key was refused"), "{e}");
    assert!(!e.contains(KEY));

    let port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
    assert!(!check_with(&client(), KEY, "m", &format!("http://127.0.0.1:{port}")).await.unwrap());
    assert!(check_with(&client(), "", "m", "").await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gemini_tts_an_unreachable_api_is_a_network_failure_within_the_connect_bound() {
    // Nothing listens: Windows retries a refused connect for about 2 s, the 1 s connect bound of a
    // 3 s request ends it first (as for the Edge voices), so the local voice takes over after ~1 s.
    let port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
    let (rec, _) = Recorder::new();
    let t = Instant::now();
    let f = failure(speak_with(&request(&format!("http://127.0.0.1:{port}"), 3000), "gem-down", Box::new(rec), &client_for(3000)).await);
    assert_eq!(f.kind, FailureKind::Network, "{f:?}");
    assert!(f.message.contains("cannot reach 127.0.0.1"), "{}", f.message);
    assert!(!f.message.contains(KEY));
    assert!(t.elapsed() < Duration::from_millis(1500), "{:?}", t.elapsed());
}

#[test]
fn gemini_tts_connect_bounds_scale_with_the_timeout() {
    // A third of the limit, at least 1 s (never more than the limit), rounded up to 250 ms.
    assert_eq!(connect_bound(Duration::from_secs(3)), Duration::from_secs(1));
    assert_eq!(connect_bound(Duration::from_millis(500)), Duration::from_millis(500));
    assert_eq!(connect_bound(Duration::from_millis(1200)), Duration::from_millis(1000));
    assert_eq!(connect_bound(Duration::from_millis(3100)), Duration::from_millis(1250));
    assert_eq!(connect_bound(Duration::from_secs(10)), Duration::from_millis(3500));
    assert_eq!(connect_bound(MAX_TIMEOUT), Duration::from_secs(40));
    // One client per bound, reused.
    let a = client_for(3000);
    let b = client_for(3000);
    drop((a, b));
    assert!(CLIENTS.lock().unwrap().contains_key(&1000));
    let _ = client_for(0);
    assert!(!CLIENTS.lock().unwrap().contains_key(&0));
}

#[test]
fn gemini_tts_api_roots_models_and_timeouts() {
    assert_eq!(api_root("").unwrap(), "https://generativelanguage.googleapis.com/v1beta");
    assert_eq!(api_root("generativelanguage.googleapis.com/").unwrap(), "https://generativelanguage.googleapis.com/v1beta");
    assert_eq!(api_root("https://proxy.example/gemini/v1alpha").unwrap(), "https://proxy.example/gemini/v1alpha");
    assert_eq!(api_root("http://127.0.0.1:8080").unwrap(), "http://127.0.0.1:8080/v1beta");
    assert!(api_root("https://x.example/?key=abc").is_err());
    assert_eq!(model_id(" models/gemini-3.8-flash-lite-tts ").unwrap(), "gemini-3.8-flash-lite-tts");
    assert_eq!(model_id("").unwrap(), DEFAULT_MODEL);
    assert!(model_id("../evil").is_err());
    assert!(model_id("a b").is_err());
    assert_eq!(timing(0), (AUTO_FIRST_AUDIO, AUTO_STALL));
    assert_eq!(timing(5000), (Duration::from_secs(5), Duration::from_secs(5)));
    assert_eq!(timing(10), (MIN_TIMEOUT, MIN_TIMEOUT));
    assert!(key_header(" ").is_err());
    assert!(key_header("k").unwrap().is_sensitive());
}

#[test]
fn gemini_tts_request_bodies_follow_the_api_shapes() {
    let mut r = request("", 0);
    r.style = String::new();
    r.voice = String::new();
    let b = request_body(Api::Interactions, &r, "m", "Hi");
    assert!(b["input"][0]["content"][0].get("annotations").is_none());
    assert_eq!(b["generation_config"]["speech_config"][0]["voice"], DEFAULT_VOICE);
    let g = request_body(Api::GenerateContent, &r, "m", "Hi");
    assert_eq!(g["contents"][0]["parts"][0]["text"], "Hi");
    // The style: a field on /interactions; on generateContent words in the prompt, but only for
    // models that do not read the prompt verbatim.
    r.style = "calm".into();
    let b = request_body(Api::Interactions, &r, "gemini-3.8-flash-tts", "Hi");
    assert_eq!(b["input"][0]["content"][0]["annotations"], json!([{"type": "speech_metadata", "style": "calm"}]));
    assert_eq!(b["input"][0]["content"][0]["text"], "Hi");
    let old = request_body(Api::GenerateContent, &r, "gemini-2.5-flash-preview-tts", "Hi");
    assert_eq!(old["contents"][0]["parts"][0]["text"], "Say in a calm tone: Hi");
    let new = request_body(Api::GenerateContent, &r, "gemini-3.8-flash-lite-tts", "Hi");
    assert_eq!(new["contents"][0]["parts"][0]["text"], "Hi");
    for (model, verbatim) in [
        ("gemini-3.8-flash-tts", true),
        ("gemini-3.8-flash-lite-tts", true),
        ("gemini-3.10-flash-tts", true),
        ("gemini-4-flash-tts", true),
        ("gemini-3.1-flash-tts-preview", false),
        ("gemini-2.5-pro-preview-tts", false),
        ("my-proxy-model", false),
    ] {
        assert_eq!(reads_prompt_verbatim(model), verbatim, "{model}");
    }
    let wire: GeminiTtsRequest = serde_json::from_value(json!({"apiKey": "k", "model": "m", "text": "t", "voice": "v", "timeoutMs": 5})).unwrap();
    assert_eq!(wire.api_key, "k");
    assert_eq!(wire.timeout_ms, 5);
    assert_eq!(wire.base_url, "");
}
