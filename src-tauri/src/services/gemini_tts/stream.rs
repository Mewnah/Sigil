//! Decoding Gemini TTS answers (P6-H): server-sent events, the audio parts inside their JSON
//! (Interactions API deltas and `generateContent` `inlineData`), errors and finish reasons, and the
//! base64 PCM / WAV audio itself. Pure code, tested without a network.

use base64::Engine;
use serde_json::Value;

use crate::services::edge_tts::cloud::Block;

/// Sample rate Gemini TTS answers with unless the mime type says otherwise.
pub(crate) const DEFAULT_RATE: u32 = 24_000;

// ---------------------------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------------------------

/// One server-sent event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental `text/event-stream` parser (lines end with LF or CRLF; a blank line ends an event).
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Feed body bytes; returns the events they completed.
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.line(&String::from_utf8_lossy(&line), &mut out);
        }
        out
    }

    /// The body ended: a last event without its blank line still counts.
    pub(crate) fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            let line = String::from_utf8_lossy(&rest).trim_end_matches('\r').to_string();
            self.line(&line, &mut out);
        }
        self.dispatch(&mut out);
        out
    }

    fn line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "data" => self.data.push(value.to_string()),
            "event" => self.event = Some(value.to_string()),
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if self.data.is_empty() {
            self.event = None;
            return;
        }
        out.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
        self.data.clear();
    }
}

// ---------------------------------------------------------------------------------------------
// JSON answers
// ---------------------------------------------------------------------------------------------

/// Base64 audio found in an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AudioPart {
    pub data: String,
    pub mime: Option<String>,
    pub rate: Option<u32>,
}

/// What one JSON answer (or event) holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Extracted {
    pub audio: Vec<AudioPart>,
    /// The model wrote text (it should only speak).
    pub text: bool,
    /// An `{"error": {…}}` object.
    pub error: Option<Value>,
    /// A finish reason other than a normal stop.
    pub finish: Option<String>,
}

fn str_field<'a>(o: &'a serde_json::Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| o.get(*k).and_then(Value::as_str))
}

fn rate_field(o: &serde_json::Map<String, Value>) -> Option<u32> {
    ["sample_rate", "sampleRate", "sample_rate_hertz", "sampleRateHertz"]
        .iter()
        .find_map(|k| o.get(*k).and_then(Value::as_u64))
        .and_then(|r| u32::try_from(r).ok())
}

/// Collect audio, text, errors and finish reasons from `v` (any answer shape: Interactions events
/// and outputs, `generateContent` candidates). Echoes of the request (`input`, `contents`,
/// `user_input` items) are skipped.
pub(crate) fn extract(v: &Value, out: &mut Extracted) {
    extract_in(v, out, false);
    // One answer never carries the same chunk twice; a whole answer may list its audio in two places.
    let mut seen = std::collections::HashSet::new();
    out.audio.retain(|a| seen.insert(a.data.clone()));
}

/// [`extract`] of `v`; `audio_ctx`: `v` is the value of an `audio` member (its `data` is audio).
fn extract_in(v: &Value, out: &mut Extracted, audio_ctx: bool) {
    match v {
        Value::Array(items) => items.iter().for_each(|i| extract_in(i, out, audio_ctx)),
        Value::Object(o) => {
            if let Some(err) = o.get("error").filter(|e| e.is_object()) {
                if err.get("message").is_some() || err.get("code").is_some() {
                    out.error.get_or_insert_with(|| err.clone());
                }
            }
            let kind = o.get("type").and_then(Value::as_str);
            if kind == Some("user_input") {
                return;
            }
            // generateContent: {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "…"}}
            for key in ["inlineData", "inline_data"] {
                if let Some(Value::Object(inline)) = o.get(key) {
                    let mime = str_field(inline, &["mimeType", "mime_type"]);
                    if let (Some(data), true) = (inline.get("data").and_then(Value::as_str), mime.is_none_or(|m| m.starts_with("audio/"))) {
                        out.audio.push(AudioPart { data: data.to_string(), mime: mime.map(str::to_string), rate: rate_field(inline) });
                    }
                }
            }
            // Interactions: {"type": "audio", "data": "…", "mime_type": "audio/l16"}
            let mime = str_field(o, &["mime_type", "mimeType"]);
            let is_audio = kind == Some("audio") || mime.is_some_and(|m| m.starts_with("audio/")) || (audio_ctx && kind.is_none());
            if is_audio {
                if let Some(data) = o.get("data").and_then(Value::as_str) {
                    out.audio.push(AudioPart { data: data.to_string(), mime: mime.map(str::to_string), rate: rate_field(o) });
                }
            }
            let has_text = o.get("text").and_then(Value::as_str).is_some_and(|t| !t.trim().is_empty());
            if has_text && kind != Some("audio") {
                out.text = true;
            }
            if let Some(reason) = str_field(o, &["finishReason", "finish_reason"]) {
                if !matches!(reason, "STOP" | "FINISH_REASON_UNSPECIFIED" | "MAX_TOKENS" | "stop") {
                    out.finish.get_or_insert_with(|| reason.to_string());
                }
            }
            for (k, child) in o {
                if matches!(k.as_str(), "input" | "contents" | "inlineData" | "inline_data" | "error" | "data") {
                    continue;
                }
                if child.is_object() || child.is_array() {
                    extract_in(child, out, matches!(k.as_str(), "audio" | "output_audio" | "outputAudio"));
                }
            }
        }
        _ => {}
    }
}

/// How an event's audio counts: as it streams, or only when nothing streamed (a whole-answer
/// summary such as `interaction.complete`, which repeats the audio of the deltas).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EventRole {
    Incremental,
    Summary,
}

/// The role of an event named `name` (the SSE `event:` field or the JSON `event_type`): whole-answer
/// events (`interaction.*`, `*.complete`, `*.stop`, `*.done`, `*.end`) summarize; everything else
/// (`content.delta`, `content.start`, unnamed `generateContent` chunks) streams.
pub(crate) fn event_role(name: Option<&str>) -> EventRole {
    let Some(n) = name.map(str::to_ascii_lowercase) else {
        return EventRole::Incremental;
    };
    let summary = n.starts_with("interaction")
        || n.contains("complete")
        || ["stop", "done", "end", "finished"].iter().any(|s| n.ends_with(s));
    if summary && !n.contains("delta") {
        EventRole::Summary
    } else {
        EventRole::Incremental
    }
}

/// The event name of `v` (its `event_type`), when it has one.
pub(crate) fn json_event_name(v: &Value) -> Option<String> {
    v.get("event_type").or_else(|| v.get("eventType")).and_then(Value::as_str).map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------------------------

/// The `rate=` parameter of a mime type (`audio/L16;codec=pcm;rate=24000`).
pub(crate) fn mime_rate(mime: &str) -> Option<u32> {
    mime.split(';').skip(1).find_map(|p| {
        let (k, v) = p.split_once('=')?;
        matches!(k.trim().to_ascii_lowercase().as_str(), "rate" | "samplerate" | "sample_rate")
            .then(|| v.trim().trim_matches('"').parse::<u32>().ok())
            .flatten()
            .filter(|r| (4_000..=192_000).contains(r))
    })
}

/// Decodes the audio parts of one answer into blocks, keeping an odd PCM byte between parts.
#[derive(Debug, Default)]
pub(crate) struct AudioDecoder {
    carry: Option<u8>,
}

impl AudioDecoder {
    /// Decode one base64 part: raw little-endian 16-bit PCM (`audio/l16`, `audio/pcm`, or no mime
    /// type), a WAV file, or (should Google change the default) any format rodio decodes.
    pub(crate) fn decode(&mut self, part: &AudioPart) -> Result<Option<Block>, String> {
        let cleaned: String = part.data.chars().filter(|c| !c.is_whitespace()).collect();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&cleaned)
            .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(&cleaned))
            .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&cleaned))
            .map_err(|e| format!("the audio data is not valid base64 ({e})."))?;
        if bytes.is_empty() {
            return Ok(None);
        }
        let mime = part.mime.as_deref().unwrap_or("").to_ascii_lowercase();
        if bytes.starts_with(b"RIFF") {
            return wav_block(&bytes).map(Some);
        }
        let essence = mime.split(';').next().unwrap_or("").trim();
        let raw = essence.is_empty()
            || matches!(essence, "audio/l16" | "audio/pcm" | "audio/raw" | "audio/x-raw" | "audio/s16le" | "application/octet-stream");
        if !raw {
            if matches!(essence, "audio/mulaw" | "audio/basic" | "audio/alaw" | "audio/pcmu" | "audio/pcma") {
                return Err(format!("the audio format {essence} is not supported."));
            }
            return compressed_block(bytes).map(Some);
        }
        let rate = mime_rate(&mime).or(part.rate).unwrap_or(DEFAULT_RATE);
        let mut data = bytes;
        if let Some(b) = self.carry.take() {
            data.insert(0, b);
        }
        if data.len() % 2 == 1 {
            self.carry = data.pop();
        }
        let samples: Vec<f32> = data.chunks_exact(2).map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0).collect();
        Ok((!samples.is_empty()).then_some(Block { samples, rate, channels: 1 }))
    }
}

/// A complete WAV (16-bit PCM or 32-bit float).
fn wav_block(bytes: &[u8]) -> Result<Block, String> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| format!("the WAV audio cannot be read ({e})."))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader.samples::<i16>().filter_map(Result::ok).map(|s| f32::from(s) / 32_768.0).collect(),
        (hound::SampleFormat::Int, bits @ (24 | 32)) => {
            let scale = (1_i64 << (bits - 1)) as f32;
            reader.samples::<i32>().filter_map(Result::ok).map(|s| s as f32 / scale).collect()
        }
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().filter_map(Result::ok).collect(),
        (format, bits) => return Err(format!("WAV audio with {bits}-bit {format:?} samples is not supported.")),
    };
    Ok(Block { samples, rate: spec.sample_rate, channels: spec.channels.max(1) })
}

/// A complete compressed clip (MP3 / Ogg / FLAC).
fn compressed_block(bytes: Vec<u8>) -> Result<Block, String> {
    use rodio::Source;
    let decoder = rodio::Decoder::new(std::io::Cursor::new(bytes)).map_err(|e| format!("the audio cannot be decoded ({e})."))?;
    let rate = decoder.sample_rate();
    let channels = decoder.channels().max(1);
    Ok(Block { samples: decoder.map(|s| f32::from(s) / 32_768.0).collect(), rate, channels })
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// What a Google API error says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GoogleError {
    pub code: Option<u16>,
    pub status: Option<String>,
    pub message: Option<String>,
    /// `ErrorInfo.reason`, e.g. `API_KEY_INVALID`.
    pub reason: Option<String>,
    /// `RetryInfo.retryDelay` in milliseconds.
    pub retry_ms: Option<u64>,
}

/// A protobuf duration string (`"37s"`, `"1.5s"`) in milliseconds.
pub(crate) fn duration_ms(s: &str) -> Option<u64> {
    let secs: f64 = s.trim().strip_suffix('s')?.parse().ok()?;
    (secs.is_finite() && secs >= 0.0).then(|| (secs * 1000.0).round() as u64)
}

/// Parse `{"error": {"code", "status", "message", "details": [...]}}` (or the inner object).
pub(crate) fn google_error(v: &Value) -> GoogleError {
    let e = v.get("error").filter(|e| e.is_object()).unwrap_or(v);
    let mut out = GoogleError {
        code: e.get("code").and_then(Value::as_u64).and_then(|c| u16::try_from(c).ok()),
        status: e.get("status").and_then(Value::as_str).map(str::to_string),
        message: e.get("message").and_then(Value::as_str).map(|m| m.trim().to_string()).filter(|m| !m.is_empty()),
        ..GoogleError::default()
    };
    if let Some(details) = e.get("details").and_then(Value::as_array) {
        for d in details {
            if let Some(reason) = d.get("reason").and_then(Value::as_str) {
                out.reason.get_or_insert_with(|| reason.to_string());
            }
            if let Some(delay) = d.get("retryDelay").and_then(Value::as_str).and_then(duration_ms) {
                out.retry_ms.get_or_insert(delay);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn sse_events_split_across_chunks_and_line_endings() {
        let mut p = SseParser::default();
        let mut events = p.feed(b"event: content.delta\r\ndata: {\"a\":");
        assert!(events.is_empty());
        events.extend(p.feed(b"1}\r\n\r\n: comment\n\ndata: x\ndata: y\n"));
        events.extend(p.feed(b"\n"));
        events.extend(p.feed(b"data: last"));
        events.extend(p.finish());
        assert_eq!(
            events,
            vec![
                SseEvent { event: Some("content.delta".into()), data: "{\"a\":1}".into() },
                SseEvent { event: None, data: "x\ny".into() },
                SseEvent { event: None, data: "last".into() },
            ]
        );
    }

    #[test]
    fn audio_is_found_in_every_answer_shape() {
        let mut x = Extracted::default();
        // Interactions delta.
        extract(&json!({"event_type": "content.delta", "index": 0, "delta": {"type": "audio", "data": "AAA=", "mime_type": "audio/l16"}}), &mut x);
        // generateContent chunk.
        extract(
            &json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "BBB="}}]}, "finishReason": "STOP"}]}),
            &mut x,
        );
        // Unary interaction (steps).
        extract(&json!({"steps": [{"type": "model_output", "content": [{"type": "audio", "data": "CCC=", "mime_type": "audio/wav"}]}]}), &mut x);
        // An `audio` member without a type, and a whole answer listing the same audio twice.
        extract(&json!({"output_audio": {"data": "DDD="}, "outputs": [{"type": "audio", "data": "DDD="}]}), &mut x);
        let data: Vec<_> = x.audio.iter().map(|a| a.data.as_str()).collect();
        assert_eq!(data, ["AAA=", "BBB=", "CCC=", "DDD="]);
        assert_eq!(x.audio[1].mime.as_deref(), Some("audio/L16;codec=pcm;rate=24000"));
        assert!(!x.text && x.error.is_none() && x.finish.is_none());
    }

    #[test]
    fn request_echoes_text_errors_and_finish_reasons() {
        let mut x = Extracted::default();
        extract(
            &json!({"interaction": {"input": [{"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}], "response_format": {"type": "audio"}}}),
            &mut x,
        );
        assert_eq!(x, Extracted::default(), "the echoed request is not an answer");
        extract(&json!({"candidates": [{"content": {"parts": [{"text": "I can't do that"}]}, "finishReason": "SAFETY"}]}), &mut x);
        assert!(x.text);
        assert_eq!(x.finish.as_deref(), Some("SAFETY"));
        extract(&json!({"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}), &mut x);
        assert_eq!(x.error.as_ref().unwrap()["code"], 429);
    }

    #[test]
    fn event_roles() {
        assert_eq!(event_role(None), EventRole::Incremental);
        assert_eq!(event_role(Some("content.delta")), EventRole::Incremental);
        assert_eq!(event_role(Some("content.start")), EventRole::Incremental);
        // The documented 2026-10 stream.
        assert_eq!(event_role(Some("step.delta")), EventRole::Incremental);
        assert_eq!(event_role(Some("step.start")), EventRole::Incremental);
        assert_eq!(event_role(Some("step.complete")), EventRole::Summary);
        assert_eq!(event_role(Some("step.stop")), EventRole::Summary);
        assert_eq!(event_role(Some("interaction.complete")), EventRole::Summary);
        assert_eq!(event_role(Some("content.stop")), EventRole::Summary);
        assert_eq!(event_role(Some("audio.chunk")), EventRole::Incremental);
        assert_eq!(event_role(Some("response.output_audio.delta")), EventRole::Incremental);
        assert_eq!(event_role(Some("response.done")), EventRole::Summary);
        assert_eq!(json_event_name(&json!({"event_type": "interaction.start"})).as_deref(), Some("interaction.start"));
    }

    #[test]
    fn pcm_parts_decode_with_an_odd_byte_carried() {
        let mut d = AudioDecoder::default();
        let pcm: Vec<u8> = [1000_i16, -1000, 32_767].iter().flat_map(|s| s.to_le_bytes()).collect();
        let first = d.decode(&AudioPart { data: b64(&pcm[..3]), mime: Some("audio/l16;rate=16000".into()), rate: None }).unwrap().unwrap();
        assert_eq!(first.rate, 16_000);
        assert_eq!(first.samples, vec![1000.0 / 32_768.0]);
        let second = d.decode(&AudioPart { data: b64(&pcm[3..]), mime: None, rate: None }).unwrap().unwrap();
        assert_eq!(second.rate, DEFAULT_RATE);
        assert_eq!(second.samples, vec![-1000.0 / 32_768.0, 32_767.0 / 32_768.0]);
        assert!(d.decode(&AudioPart { data: "!!".into(), mime: None, rate: None }).is_err());
        assert!(d.decode(&AudioPart { data: b64(&[1, 2]), mime: Some("audio/mulaw".into()), rate: None }).is_err());
        assert_eq!(d.decode(&AudioPart { data: String::new(), mime: None, rate: None }).unwrap(), None);
    }

    #[test]
    fn wav_parts_decode() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: 24_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
            for s in [0_i16, 16_384, -16_384] {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
        }
        let block = AudioDecoder::default()
            .decode(&AudioPart { data: b64(buf.get_ref()), mime: Some("audio/wav".into()), rate: None })
            .unwrap()
            .unwrap();
        assert_eq!(block.rate, 24_000);
        assert_eq!(block.samples, vec![0.0, 0.5, -0.5]);
    }

    #[test]
    fn google_errors_parse_with_retry_and_reason() {
        let e = google_error(&json!({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": " Quota exceeded ", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37.5s"}
        ]}}));
        assert_eq!(e.code, Some(429));
        assert_eq!(e.status.as_deref(), Some("RESOURCE_EXHAUSTED"));
        assert_eq!(e.message.as_deref(), Some("Quota exceeded"));
        assert_eq!(e.retry_ms, Some(37_500));
        let k = google_error(&json!({"error": {"code": 400, "message": "API key not valid.", "details": [{"reason": "API_KEY_INVALID"}]}}));
        assert_eq!(k.reason.as_deref(), Some("API_KEY_INVALID"));
        assert_eq!(duration_ms("2s"), Some(2000));
        assert_eq!(duration_ms("soon"), None);
        assert_eq!(mime_rate("audio/L16;codec=pcm;rate=24000"), Some(24_000));
        assert_eq!(mime_rate("audio/l16"), None);
    }
}
