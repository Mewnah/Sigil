//! Test helpers for the web routes: fake machines, guards, and a raw WebSocket client that can read
//! close frames (warp's test client swallows them).

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use tungstenite::client::IntoClientRequest;
use tungstenite::http::HeaderValue;
use tungstenite::Message as RawMessage;
use warp::{filters::BoxedFilter, reply::Response};

use super::auth::{AccessGuard, BindMode, LanAccessSettings, LanPolicy};
use super::origin::{LocalMachine, OriginGuard, OriginRules};

/// A valid LAN token.
pub const TOKEN: &str = "0123456789abcdefghjkmnpqrs";
/// Another valid LAN token.
pub const OTHER_TOKEN: &str = "zyxwvtsrqpnmkjhgfedcba9876";

/// No interface addresses: only loopback peers are on this machine.
pub struct NoInterfaces;
impl LocalMachine for NoInterfaces {
    fn is_own_ip(&self, _: IpAddr) -> bool {
        false
    }
    fn is_own_hostname(&self, _: &str) -> bool {
        false
    }
}

/// This PC's LAN address in the route tests (pages a LAN client loaded have this origin host).
pub const OWN_LAN_IP: [u8; 4] = [192, 168, 1, 50];

/// This machine is 192.168.1.50, and every peer counts as another machine, so warp's loopback
/// test sockets act as LAN clients.
pub struct LanOnly;
impl LocalMachine for LanOnly {
    fn is_own_ip(&self, ip: IpAddr) -> bool {
        ip == IpAddr::from(OWN_LAN_IP)
    }
    fn is_own_hostname(&self, _: &str) -> bool {
        false
    }
    fn is_this_machine(&self, _: IpAddr) -> bool {
        false
    }
}

pub fn policy(mode: BindMode, token: &str, legacy: bool) -> LanPolicy {
    LanPolicy::from_settings(&LanAccessSettings {
        mode,
        token: token.to_string(),
        legacy_no_token: legacy,
    })
}

/// The LAN mode with [`TOKEN`].
pub fn lan_policy() -> LanPolicy {
    policy(BindMode::Lan, TOKEN, false)
}

pub fn guard(port: u16, origin_check: bool, machine: Box<dyn LocalMachine>, policy: LanPolicy) -> Arc<AccessGuard> {
    let rules = OriginRules { port, dev_port: None };
    Arc::new(AccessGuard::new(OriginGuard::new(rules, origin_check, None, machine), policy))
}

/// What a raw client saw after connecting.
#[derive(Debug, PartialEq, Eq)]
pub enum RawOutcome {
    /// The handshake failed with this HTTP status.
    Refused(u16),
    /// Upgraded; these text messages arrived, then a close frame with this code and reason.
    Closed { texts: Vec<String>, code: u16, reason: String },
    /// Upgraded and still open after the wait (texts received so far).
    Open { texts: Vec<String> },
}

/// Serve `route` on 127.0.0.1 and connect a blocking tungstenite client to `path_and_query` with
/// `origin`, reading frames for up to `wait`.
pub async fn raw_connect(route: BoxedFilter<(Response,)>, path_and_query: &str, origin: Option<&str>, wait: std::time::Duration) -> RawOutcome {
    let (addr, server) = warp::serve(route).bind_ephemeral(SocketAddr::from(([127, 0, 0, 1], 0)));
    let server = tokio::spawn(server);
    let url = format!("ws://{addr}{path_and_query}");
    let origin = origin.map(str::to_string);
    let outcome = tokio::task::spawn_blocking(move || {
        let mut request = url.into_client_request().expect("request");
        if let Some(origin) = origin {
            request
                .headers_mut()
                .insert("origin", HeaderValue::from_str(&origin).unwrap());
        }
        let (mut socket, _) = match tungstenite::connect(request) {
            Ok(ok) => ok,
            Err(tungstenite::Error::Http(response)) => return RawOutcome::Refused(response.status().as_u16()),
            Err(e) => panic!("connect failed: {e}"),
        };
        if let tungstenite::stream::MaybeTlsStream::Plain(stream) = socket.get_ref() {
            stream.set_read_timeout(Some(wait)).unwrap();
        }
        let mut texts = Vec::new();
        loop {
            match socket.read() {
                Ok(RawMessage::Text(text)) => texts.push(text.to_string()),
                Ok(RawMessage::Close(frame)) => {
                    let (code, reason) = frame.map_or((1005, String::new()), |f| (u16::from(f.code), f.reason.to_string()));
                    return RawOutcome::Closed { texts, code, reason };
                }
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                    return RawOutcome::Open { texts };
                }
                Err(e) => panic!("read failed: {e}"),
            }
        }
    })
    .await
    .unwrap();
    server.abort();
    outcome
}
