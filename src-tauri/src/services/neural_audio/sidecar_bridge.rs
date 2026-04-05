//! WebSocket client for neural voice conversion sidecar (synchronous PCM exchange per chunk).
//!
//! **Note:** [`SidecarVoiceBridge::exchange`] blocks the calling thread until the server responds.
//! It runs on the voice-changer audio thread, so a slow sidecar increases end-to-end latency.

use serde::Serialize;
use tungstenite::{connect, stream::MaybeTlsStream, Message, WebSocket};

#[derive(Serialize)]
struct Handshake {
    r#type: &'static str,
    protocol_version: u32,
    sample_rate_in: u32,
    channels_in: usize,
    chunk_frames: usize,
}

fn encode_pcm_chunk(samples: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + samples.len() * 4);
    v.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

fn decode_pcm_chunk(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.len() < 4 {
        return Err("sidecar: packet too short".into());
    }
    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let need = 4usize.saturating_add(n.saturating_mul(4));
    if data.len() < need {
        return Err(format!(
            "sidecar: need {} bytes for {} floats, have {}",
            need,
            n,
            data.len()
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = 4 + i * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    Ok(out)
}

pub struct SidecarVoiceBridge {
    socket: WebSocket<MaybeTlsStream<std::net::TcpStream>>,
}

impl SidecarVoiceBridge {
    pub fn connect(
        ws_url: &str,
        sample_rate_in: u32,
        channels_in: usize,
        chunk_frames: usize,
    ) -> Result<Self, String> {
        let (socket, _) = connect(ws_url).map_err(|e| format!("sidecar WebSocket connect: {}", e))?;
        let mut socket = socket;
        let handshake = Handshake {
            r#type: "sigil_vc_handshake",
            protocol_version: 1,
            sample_rate_in,
            channels_in,
            chunk_frames,
        };
        let text = serde_json::to_string(&handshake).map_err(|e| e.to_string())?;
        socket
            .send(Message::Text(text))
            .map_err(|e| format!("sidecar handshake: {}", e))?;
        let _ = socket.read();
        Ok(Self { socket })
    }

    /// Send one PCM chunk and wait for the converted chunk (same layout).
    pub fn exchange(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let payload = encode_pcm_chunk(samples);
        self
            .socket
            .send(Message::Binary(payload))
            .map_err(|e| format!("sidecar send: {}", e))?;
        match self.socket.read().map_err(|e| format!("sidecar read: {}", e))? {
            Message::Binary(b) => decode_pcm_chunk(&b),
            Message::Close(_) => Err("sidecar closed".into()),
            _ => Err("sidecar: unexpected message type".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_chunk_roundtrip() {
        let v = vec![0.25f32, -0.5, 1.0];
        let enc = encode_pcm_chunk(&v);
        assert_eq!(decode_pcm_chunk(&enc).unwrap(), v);
    }
}
