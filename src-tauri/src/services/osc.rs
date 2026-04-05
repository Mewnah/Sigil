use rosc::{encoder, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Manager, Runtime, State,
};

pub struct OscPlugin {
    socket: Option<UdpSocket>,
    bind_error: Option<String>,
}

impl Default for OscPlugin {
    fn default() -> Self {
        match UdpSocket::bind("127.0.0.1:0") {
            Ok(socket) => Self {
                socket: Some(socket),
                bind_error: None,
            },
            Err(e) => Self {
                socket: None,
                bind_error: Some(format!("UDP bind 127.0.0.1:0 failed: {e}")),
            },
        }
    }
}

impl OscPlugin {
    fn send(&self, rpc: RpcOscMessage) -> Result<(), String> {
        let Some(socket) = &self.socket else {
            return Err(
                self.bind_error
                    .clone()
                    .unwrap_or_else(|| "OSC socket unavailable".into()),
            );
        };

        let addr = resolve_osc_dest(&rpc.host, rpc.port)?;
        let args: Vec<OscType> = rpc
            .args
            .iter()
            .map(|arg| match arg {
                OscValue::Bool(v) => OscType::Bool(*v),
                OscValue::Float(v) => OscType::Float(*v as f32),
                OscValue::Int(v) => OscType::Int(*v as i32),
                OscValue::String(v) => OscType::String(v.clone()),
            })
            .collect();

        let msg_buf = encoder::encode(&OscPacket::Message(OscMessage {
            addr: rpc.path.clone(),
            args,
        }))
        .map_err(|e| format!("OSC encode failed: {e}"))?;

        socket
            .send_to(&msg_buf, addr)
            .map_err(|e| format!("OSC send_to {addr} failed: {e}"))?;
        Ok(())
    }
}

fn resolve_osc_dest(host: &str, port: u16) -> Result<SocketAddr, String> {
    if port == 0 {
        return Err("OSC destination port must not be 0".into());
    }
    let host = host.trim();
    if host.is_empty() {
        return Err("OSC host is empty".into());
    }
    let mut iter = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("Invalid OSC host/port {host}:{port}: {e}"))?;
    iter
        .next()
        .ok_or_else(|| format!("Could not resolve OSC address {host}:{port}"))
}

#[derive(Serialize, Deserialize, Debug)]
enum OscValue {
    Bool(bool),
    Float(f64),
    Int(i64),
    String(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcOscMessage {
    path: String,
    args: Vec<OscValue>,
    #[serde(default = "default_osc_host")]
    host: String,
    #[serde(default = "default_osc_port")]
    port: u16,
}

fn default_osc_host() -> String {
    "127.0.0.1".into()
}

fn default_osc_port() -> u16 {
    9000
}

#[derive(Serialize)]
pub struct OscBindStatus {
    pub ok: bool,
    pub error: Option<String>,
}

#[command]
fn send(rpc: RpcOscMessage, state: State<OscPlugin>) -> Result<(), String> {
    state.send(rpc)
}

#[command]
fn bind_status(state: State<OscPlugin>) -> OscBindStatus {
    match &state.bind_error {
        Some(err) => OscBindStatus {
            ok: false,
            error: Some(err.clone()),
        },
        None => OscBindStatus {
            ok: true,
            error: None,
        },
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("osc")
        .invoke_handler(tauri::generate_handler![send, bind_status])
        .setup(|app, _api| {
            app.manage(OscPlugin::default());
            Ok(())
        })
        .build()
}
