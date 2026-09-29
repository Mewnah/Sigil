//! Loopback ports for helpers: pick a free `127.0.0.1` port, check one, and audit that a helper
//! listens on loopback only (a helper reachable from the LAN is stopped; Sigil's own web server is
//! the only LAN-facing listener, see CLAUDE.md).
//!
//! The audit reads the OS table of listening TCP sockets with their owning process ids
//! (`GetExtendedTcpTable`, iphlpapi, resolved at run time) and flags every listener of the helper's
//! processes whose address is not loopback, on any port. If the table cannot be read, it falls back
//! to connecting to the helper's port on each non-loopback address of this PC.

use std::ffi::c_void;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
use std::sync::OnceLock;
use std::time::Duration;
use windows::core::{s, w};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};

/// How long one connect probe of the fallback audit may take. A listening socket on this PC accepts
/// within microseconds; a closed port is refused (or, on Windows, retried for ~2 s, hence the cap).
pub const AUDIT_CONNECT_TIMEOUT: Duration = Duration::from_millis(300);

/// A free port on `127.0.0.1` right now (the OS picks it from the ephemeral range, which also skips
/// Hyper-V / WSL excluded ranges). The helper binds it a moment later; a collision in between is
/// detected and retried by the caller.
pub fn pick_free_port() -> Result<u16, String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| format!("cannot reserve a loopback port: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("cannot read the reserved port: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

/// Whether `127.0.0.1:port` can be bound right now.
pub fn loopback_port_free(port: u16) -> bool {
    port != 0 && TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// Whether a listener on `ip` is reachable from other machines (anything but loopback, including
/// the wildcard addresses and IPv4-mapped loopback's opposite).
pub fn is_exposed(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => !v4.is_loopback(),
        IpAddr::V6(v6) => !(v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|m| m.is_loopback())),
    }
}

// ---------------------------------------------------------------------------------------------
// OS listener table
// ---------------------------------------------------------------------------------------------

/// One listening TCP socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Listener {
    pub addr: SocketAddr,
    pub pid: u32,
}

/// `GetExtendedTcpTable(PVOID, PDWORD, BOOL, ULONG af, TCP_TABLE_CLASS, ULONG) -> DWORD`.
type GetExtendedTcpTableFn = unsafe extern "system" fn(*mut c_void, *mut u32, i32, u32, u32, u32) -> u32;

const AF_INET: u32 = 2;
const AF_INET6: u32 = 23;
const TCP_TABLE_OWNER_PID_LISTENER: u32 = 3;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
/// `MIB_TCPROW_OWNER_PID`: state, local addr, local port, remote addr, remote port, pid.
const ROW4_WORDS: usize = 6;
/// `MIB_TCP6ROW_OWNER_PID`: local addr (4), scope, local port, remote addr (4), scope, remote port,
/// state, pid.
const ROW6_WORDS: usize = 14;

fn get_extended_tcp_table() -> Option<GetExtendedTcpTableFn> {
    static CELL: OnceLock<Option<GetExtendedTcpTableFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe {
        let module = LoadLibraryExW(w!("iphlpapi.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
        let proc = GetProcAddress(module, s!("GetExtendedTcpTable"))?;
        // SAFETY: documented signature (iphlpapi.h); the module is never freed.
        Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, GetExtendedTcpTableFn>(proc))
    })
}

/// The raw listener table of one address family as DWORDs (`dwNumEntries`, then the rows).
fn raw_table(af: u32) -> Result<Vec<u32>, String> {
    let f = get_extended_tcp_table().ok_or("GetExtendedTcpTable is not available")?;
    let mut bytes: u32 = 16 * 1024;
    for _ in 0..5 {
        let mut buf = vec![0u32; (bytes as usize).div_ceil(4)];
        let mut len = (buf.len() * 4) as u32;
        // SAFETY: `buf` is writable and 4-byte aligned (the rows hold only DWORDs and bytes); `len`
        // is its size in bytes.
        let rc = unsafe { f(buf.as_mut_ptr().cast(), &mut len, 0, af, TCP_TABLE_OWNER_PID_LISTENER, 0) };
        match rc {
            0 => {
                buf.truncate((len as usize).div_ceil(4));
                return Ok(buf);
            }
            ERROR_INSUFFICIENT_BUFFER => bytes = len.saturating_add(4096),
            e => return Err(format!("GetExtendedTcpTable failed (error {e})")),
        }
    }
    Err("GetExtendedTcpTable: the table kept growing".into())
}

/// Port field of a row: network byte order in the low 16 bits.
fn row_port(word: u32) -> u16 {
    u16::from_be(word as u16)
}

fn parse_v4(table: &[u32]) -> Vec<Listener> {
    let n = table.first().copied().unwrap_or(0) as usize;
    table
        .get(1..)
        .unwrap_or_default()
        .chunks_exact(ROW4_WORDS)
        .take(n)
        .map(|r| Listener {
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::from(r[1].to_ne_bytes())), row_port(r[2])),
            pid: r[5],
        })
        .collect()
}

fn parse_v6(table: &[u32]) -> Vec<Listener> {
    let n = table.first().copied().unwrap_or(0) as usize;
    table
        .get(1..)
        .unwrap_or_default()
        .chunks_exact(ROW6_WORDS)
        .take(n)
        .map(|r| {
            let mut octets = [0u8; 16];
            for (i, w) in r[..4].iter().enumerate() {
                octets[i * 4..i * 4 + 4].copy_from_slice(&w.to_ne_bytes());
            }
            Listener {
                addr: SocketAddr::new(IpAddr::V6(Ipv6Addr::from(octets)), row_port(r[5])),
                pid: r[13],
            }
        })
        .collect()
}

/// Every listening TCP socket on this PC with its owning process (IPv4, plus IPv6 when installed).
pub fn tcp_listeners() -> Result<Vec<Listener>, String> {
    let mut out = parse_v4(&raw_table(AF_INET)?);
    // IPv6 can be disabled; its absence is not an error.
    if let Ok(t) = raw_table(AF_INET6) {
        out.extend(parse_v6(&t));
    }
    Ok(out)
}

/// The owner of the listener that receives TCP connections to `127.0.0.1:port` among `listeners`:
/// an exact `127.0.0.1` bind (or its IPv4-mapped IPv6 form) wins over the IPv4 wildcard, which wins
/// over a dual-stack `[::]` listener. `None` when nothing listens there.
pub fn loopback_receiver(listeners: &[Listener], port: u16) -> Option<u32> {
    let candidates = [
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(Ipv4Addr::LOCALHOST.to_ipv6_mapped()),
        IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    ];
    candidates.iter().find_map(|ip| {
        listeners
            .iter()
            .find(|l| l.addr.port() == port && l.addr.ip() == *ip)
            .map(|l| l.pid)
    })
}

/// The process that receives TCP connections to `127.0.0.1:port` right now, from the OS listener
/// table (`Ok(None)`: nothing listens there).
pub fn loopback_listener_pid(port: u16) -> Result<Option<u32>, String> {
    Ok(loopback_receiver(&tcp_listeners()?, port))
}

/// Addresses beyond loopback that processes in `pids` listen on.
pub fn exposed_listeners(pids: &[u32]) -> Result<Vec<SocketAddr>, String> {
    if pids.is_empty() {
        return Ok(Vec::new());
    }
    let mut out: Vec<SocketAddr> = tcp_listeners()?
        .into_iter()
        .filter(|l| pids.contains(&l.pid) && is_exposed(l.addr.ip()))
        .map(|l| l.addr)
        .collect();
    out.sort();
    out.dedup();
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Fallback audit: connect probes
// ---------------------------------------------------------------------------------------------

/// This PC's addresses other than loopback. IPv6 link-local addresses are skipped (they need a
/// scope id to connect).
pub fn non_loopback_addresses() -> Vec<IpAddr> {
    let mut out: Vec<IpAddr> = local_ip_address::list_afinet_netifas()
        .map(|list| list.into_iter().map(|(_, ip)| ip).collect())
        .unwrap_or_default();
    out.retain(|ip| {
        !ip.is_loopback()
            && !ip.is_unspecified()
            && !ip.is_multicast()
            && match ip {
                IpAddr::V4(_) => true,
                IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) != 0xfe80,
            }
    });
    out.sort();
    out.dedup();
    out
}

/// The addresses among `addrs` where something accepts TCP connections on `port`.
pub async fn listening_on(addrs: &[IpAddr], port: u16, timeout: Duration) -> Vec<SocketAddr> {
    let probes = addrs.iter().map(|ip| {
        let sa = SocketAddr::new(*ip, port);
        async move {
            match tokio::time::timeout(timeout, tokio::net::TcpStream::connect(sa)).await {
                Ok(Ok(_stream)) => Some(sa),
                _ => None,
            }
        }
    });
    futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// Non-loopback addresses of this PC where `port` accepts connections. Empty = loopback only.
pub async fn exposure(port: u16) -> Vec<SocketAddr> {
    listening_on(&non_loopback_addresses(), port, AUDIT_CONNECT_TIMEOUT).await
}

// ---------------------------------------------------------------------------------------------
// Test hook: force the next auto-picked ports of one helper (port-collision tests)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub mod forced {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Mutex, OnceLock};

    fn table() -> &'static Mutex<HashMap<String, VecDeque<u16>>> {
        static T: OnceLock<Mutex<HashMap<String, VecDeque<u16>>>> = OnceLock::new();
        T.get_or_init(Default::default)
    }

    /// Make the next auto picks for helper `name` return `ports`, in order.
    pub fn set(name: &str, ports: &[u16]) {
        table()
            .lock()
            .unwrap()
            .insert(name.to_string(), ports.iter().copied().collect());
    }

    pub fn next(name: &str) -> Option<u16> {
        table()
            .lock()
            .unwrap()
            .get_mut(name)
            .and_then(VecDeque::pop_front)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picked_port_is_bindable_and_detected_when_taken() {
        let port = pick_free_port().unwrap();
        assert!(port > 0);
        assert!(loopback_port_free(port));
        let held = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).unwrap();
        assert!(!loopback_port_free(port));
        drop(held);
        assert!(!loopback_port_free(0));
    }

    #[test]
    fn exposure_classification() {
        assert!(!is_exposed("127.0.0.1".parse().unwrap()));
        assert!(!is_exposed("127.0.0.2".parse().unwrap()));
        assert!(!is_exposed("::1".parse().unwrap()));
        assert!(!is_exposed("::ffff:127.0.0.1".parse().unwrap()));
        assert!(is_exposed("0.0.0.0".parse().unwrap()));
        assert!(is_exposed("::".parse().unwrap()));
        assert!(is_exposed("192.168.1.20".parse().unwrap()));
        assert!(is_exposed("fe80::1".parse().unwrap()));
        assert!(is_exposed("::ffff:192.168.1.20".parse().unwrap()));
    }

    #[test]
    fn table_rows_parse() {
        // One IPv4 row: 127.0.0.1:8080 owned by pid 42 (addresses and ports in network order).
        let v4 = [1, 2, u32::from_ne_bytes([127, 0, 0, 1]), u32::from(8080u16.to_be()), 0, 0, 42];
        assert_eq!(
            parse_v4(&v4),
            [Listener {
                addr: "127.0.0.1:8080".parse().unwrap(),
                pid: 42
            }]
        );
        // A truncated table never reads past its end.
        assert!(parse_v4(&[3, 1, 2]).is_empty());
        let mut v6 = vec![1u32];
        v6.extend_from_slice(&[0, 0, 0, u32::from_ne_bytes([0, 0, 0, 1])]); // ::1
        v6.extend_from_slice(&[0, u32::from(443u16.to_be()), 0, 0, 0, 0, 0, 0, 2, 7]);
        assert_eq!(
            parse_v6(&v6),
            [Listener {
                addr: "[::1]:443".parse().unwrap(),
                pid: 7
            }]
        );
    }

    #[test]
    fn listener_table_finds_our_own_loopback_listener() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let me = std::process::id();
        let all = tcp_listeners().expect("listener table");
        assert!(all.contains(&Listener { addr, pid: me }), "{addr} / pid {me} missing");
        // Loopback only: nothing exposed for this process from this listener.
        let exposed = exposed_listeners(&[me]).unwrap();
        assert!(!exposed.contains(&addr));
        assert!(exposed_listeners(&[]).unwrap().is_empty());
        assert_eq!(loopback_listener_pid(addr.port()).unwrap(), Some(me));
        drop(listener);
    }

    #[test]
    fn loopback_receiver_prefers_the_most_specific_bind() {
        let l = |addr: &str, pid: u32| Listener {
            addr: addr.parse().unwrap(),
            pid,
        };
        let table = [
            l("[::]:5000", 1),
            l("0.0.0.0:5000", 2),
            l("127.0.0.1:5000", 3),
            l("127.0.0.1:6000", 4),
            l("192.168.1.2:7000", 5),
            l("[::]:8000", 6),
            l("[::1]:9000", 7),
        ];
        assert_eq!(loopback_receiver(&table, 5000), Some(3));
        assert_eq!(loopback_receiver(&table[..2], 5000), Some(2));
        assert_eq!(loopback_receiver(&table[..1], 5000), Some(1));
        assert_eq!(loopback_receiver(&table, 6000), Some(4));
        // A LAN-only bind and an IPv6-loopback bind never receive 127.0.0.1 connections.
        assert_eq!(loopback_receiver(&table, 7000), None);
        assert_eq!(loopback_receiver(&table, 9000), None);
        assert_eq!(loopback_receiver(&table, 8000), Some(6));
        assert_eq!(loopback_receiver(&[l("[::ffff:127.0.0.1]:5000", 8), l("0.0.0.0:5000", 2)], 5000), Some(8));
        assert_eq!(loopback_receiver(&table, 1), None);
    }

    #[test]
    fn non_loopback_addresses_exclude_loopback() {
        for ip in non_loopback_addresses() {
            assert!(!ip.is_loopback(), "{ip}");
            assert!(!ip.is_unspecified(), "{ip}");
        }
    }

    #[tokio::test]
    async fn listening_on_finds_only_listeners() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let lo = [IpAddr::V4(Ipv4Addr::LOCALHOST)];
        assert_eq!(listening_on(&lo, port, AUDIT_CONNECT_TIMEOUT).await, [SocketAddr::new(lo[0], port)]);
        drop(listener);
        let free = pick_free_port().unwrap();
        assert!(listening_on(&lo, free, AUDIT_CONNECT_TIMEOUT)
            .await
            .is_empty());
        // A loopback-only listener is invisible on the LAN addresses (no firewall prompt: nothing
        // here listens beyond 127.0.0.1).
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(exposure(port).await.is_empty());
    }
}
