use std::{collections::HashMap, sync::Arc};

use futures::StreamExt;
use serde::Deserialize;
use tauri::{async_runtime::RwLock, AppHandle, Manager, Runtime};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use warp::{
    filters::BoxedFilter,
    ws::{Message, WebSocket, Ws},
    Filter, Reply,
};

use crate::services::whisper::{process_audio_chunk, WhisperState};

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
}

pub type Peers = Arc<RwLock<HashMap<String, mpsc::UnboundedSender<Result<Message, warp::Error>>>>>;

pub fn path<R: Runtime>(mut input: mpsc::Receiver<String>, output: mpsc::Sender<String>, app: AppHandle<R>) -> BoxedFilter<(impl Reply,)> {
    let peers = Peers::default();

    let input_peers = peers.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            if let Some(input) = input.recv().await {
                let p = input_peers.read().await;
                let str = input.as_str();
                for peer in p.values() {
                    peer.send(Ok(Message::text(str))).ok();
                }
            }
        }
    });

    let peers = warp::any().map(move || peers.clone());
    let output = warp::any().map(move || output.clone());
    let app = warp::any().map(move || app.clone());
    let t = warp::path("pubsub")
        .and(warp::ws())
        .and(peers)
        .and(output)
        .and(app)
        .and(warp::query::<PeerQueryData>())
        .map(|ws: Ws, peers, output, app, q| ws.on_upgrade(move |socket| peer_handler(socket, peers, output, app, q)))
        .boxed();
    t
}

pub async fn peer_handler<R: Runtime>(ws: WebSocket, peers: Peers, output: mpsc::Sender<String>, app: AppHandle<R>, query: PeerQueryData) {
    eprintln!("[PubSub] New peer connection request: {}", query.id);
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    let rx = UnboundedReceiverStream::new(rx);
    tauri::async_runtime::spawn(rx.forward(peer_tx));

    if peers.read().await.contains_key(&query.id) {
        println!("already registered");
        return;
    }

    peers.write().await.insert(query.id.clone(), tx);

    while let Some(result) = peer_rx.next().await {
        let Ok(msg) = result else {
            break;
        };

        if msg.is_binary() {
            let bytes = msg.as_bytes();
            eprintln!("[PubSub] Received binary message: {} bytes", bytes.len());

            // Convert bytes to f32 (assuming Little Endian Float32)
            if bytes.len() % 4 == 0 {
                let chunks: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                // eprintln!("[PubSub] Converted to {} samples", chunks.len());

                // Feed to Whisper
                if let Some(state) = app.try_state::<WhisperState>() {
                    process_audio_chunk(&state, &app, chunks);
                }
            }
            continue;
        }

        let Ok(msg_str) = msg.to_str() else { break };
        output.send(msg_str.to_string()).await.ok();
        let p = peers.read().await;
        for (id, peer) in p.iter() {
            if !query.id.eq(id) {
                // do not send to self
                peer.send(Ok(Message::text(msg_str))).ok();
            }
        }
    }
    peers.write().await.remove(&query.id);
}
