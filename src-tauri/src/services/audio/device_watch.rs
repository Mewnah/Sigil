//! Audio device watcher: emits `audio:devices_changed` and nudges the capture hub when devices are
//! plugged, unplugged, enabled, disabled or when a default device changes.
//!
//! - Windows: an `IMMNotificationClient` registered with the MMDevice enumerator on the watcher's
//!   own MTA thread. The COM object is a small hand-written vtable (the `#[implement]` macro needs
//!   `windows-core` as a direct dependency, which Sigil does not have). Its callbacks only set a
//!   bit and unpark the watcher (MMDevice forbids blocking in them). A slow safety poll catches
//!   anything the notifications miss (e.g. an audio service restart).
//! - Fallback (registration failed, or not Windows): poll the device list every second.
//!
//! Notifications are debounced (one unplug raises several), then the device list is enumerated
//! with cpal and compared with the previous snapshot. The event's `reason` comes from that diff:
//! `removed` > `added` > `default_changed` > `state_changed`. A notification that changes nothing
//! visible still nudges the hub (a failed stream may be reopenable).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;

use super::AudioDevice;

/// Tauri event carrying [`DevicesChanged`] (TS `RustAudioDevicesChanged`).
pub const DEVICES_CHANGED_EVENT: &str = "audio:devices_changed";

/// Device lists plus the current defaults (names are cpal names, id == name).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DeviceSnapshot {
    pub inputs: Vec<AudioDevice>,
    pub outputs: Vec<AudioDevice>,
    pub default_input: Option<String>,
    pub default_output: Option<String>,
}

/// Payload of `audio:devices_changed` (and of `plugin:audio|get_device_snapshot`, reason `snapshot`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DevicesChanged {
    pub reason: String,
    #[serde(flatten)]
    pub snapshot: DeviceSnapshot,
}

/// Enumerate the cpal default host's devices (tens of ms; COM calls: not on the main thread).
pub fn snapshot_cpal() -> Result<DeviceSnapshot, String> {
    fn list<I: Iterator<Item = cpal::Device>>(devices: Result<I, cpal::DevicesError>) -> Result<Vec<AudioDevice>, String> {
        Ok(devices
            .map_err(|e| e.to_string())?
            .filter_map(|d| d.name().ok())
            .map(|name| AudioDevice::named(name.trim()))
            .collect())
    }
    let host = cpal::default_host();
    Ok(DeviceSnapshot {
        inputs: list(host.input_devices())?,
        outputs: list(host.output_devices())?,
        default_input: host.default_input_device().and_then(|d| d.name().ok()).map(|n| n.trim().to_string()),
        default_output: host.default_output_device().and_then(|d| d.name().ok()).map(|n| n.trim().to_string()),
    })
}

fn name_counts(devices: &[AudioDevice]) -> BTreeMap<&str, usize> {
    let mut counts = BTreeMap::new();
    for d in devices {
        *counts.entry(d.name.as_str()).or_insert(0) += 1;
    }
    counts
}

/// Whether any device of `prev` is missing from `next` (identical names counted separately).
fn lost_any(prev: &[AudioDevice], next: &[AudioDevice]) -> bool {
    let next = name_counts(next);
    name_counts(prev)
        .into_iter()
        .any(|(name, n)| next.get(name).copied().unwrap_or(0) < n)
}

/// Why `next` differs from `prev`, or `None` when nothing visible changed.
pub fn change_reason(prev: &DeviceSnapshot, next: &DeviceSnapshot) -> Option<&'static str> {
    if lost_any(&prev.inputs, &next.inputs) || lost_any(&prev.outputs, &next.outputs) {
        Some("removed")
    } else if lost_any(&next.inputs, &prev.inputs) || lost_any(&next.outputs, &prev.outputs) {
        Some("added")
    } else if prev.default_input != next.default_input || prev.default_output != next.default_output {
        Some("default_changed")
    } else if prev != next {
        Some("state_changed")
    } else {
        None
    }
}

/// What the watcher reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WatchNotice {
    /// The device lists or defaults changed.
    Changed(DevicesChanged),
    /// A device notification arrived but nothing visible changed.
    Nudge,
}

/// Watcher timing.
#[derive(Clone, Debug)]
pub struct WatchConfig {
    /// Poll period when notifications are unavailable.
    pub poll: Duration,
    /// Safety re-check period while notifications work.
    pub safety_poll: Duration,
    /// Quiet time collecting a burst of notifications before enumerating.
    pub debounce: Duration,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            poll: Duration::from_secs(1),
            safety_poll: Duration::from_secs(10),
            debounce: Duration::from_millis(150),
        }
    }
}

/// Bits raised by the notification callbacks.
pub(crate) const NOTE_STATE: u32 = 1;
pub(crate) const NOTE_ADDED: u32 = 2;
pub(crate) const NOTE_REMOVED: u32 = 4;
pub(crate) const NOTE_DEFAULT: u32 = 8;
pub(crate) const NOTE_PROPERTY: u32 = 16;

/// Shared between the COM callbacks (any MMDevice thread) and the watcher thread.
pub(crate) struct WatchSignal {
    pending: AtomicU32,
    waiter: Thread,
}

impl WatchSignal {
    pub(crate) fn new(waiter: Thread) -> Self {
        Self {
            pending: AtomicU32::new(0),
            waiter,
        }
    }

    /// Non-blocking: safe inside MMDevice notification callbacks.
    pub(crate) fn raise(&self, bits: u32) {
        self.pending.fetch_or(bits, Ordering::AcqRel);
        self.waiter.unpark();
    }

    pub(crate) fn take(&self) -> u32 {
        self.pending.swap(0, Ordering::AcqRel)
    }
}

/// A running watcher; dropping it stops the thread.
pub struct DeviceWatcher {
    stop: Arc<AtomicBool>,
    thread: Thread,
    handle: Option<JoinHandle<()>>,
}

impl DeviceWatcher {
    #[allow(dead_code)] // The app keeps its watcher for its lifetime; tests stop theirs.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.thread.unpark();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DeviceWatcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Start watching. `list` enumerates devices; `on_notice` runs on the watcher thread.
pub fn start<L, N>(list: L, on_notice: N, config: WatchConfig) -> Result<DeviceWatcher, String>
where
    L: Fn() -> Result<DeviceSnapshot, String> + Send + 'static,
    N: Fn(WatchNotice) + Send + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let handle = {
        let stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("sigil-audio-devices".into())
            .spawn(move || run(list, on_notice, config, &stop))
            .map_err(|e| format!("Failed to start the audio device watcher: {e}"))?
    };
    Ok(DeviceWatcher {
        stop,
        thread: handle.thread().clone(),
        handle: Some(handle),
    })
}

fn run<L, N>(list: L, on_notice: N, config: WatchConfig, stop: &AtomicBool)
where
    L: Fn() -> Result<DeviceSnapshot, String>,
    N: Fn(WatchNotice),
{
    let signal = Arc::new(WatchSignal::new(thread::current()));
    let registration = register(&signal);
    let notifications = match &registration {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(target: "sigil::audio", "device notifications unavailable ({e}); polling every {:?}", config.poll);
            false
        }
    };
    let mut last = list().ok();
    let mut last_error_log: Option<Instant> = None;
    while !stop.load(Ordering::Acquire) {
        thread::park_timeout(if notifications { config.safety_poll } else { config.poll });
        if stop.load(Ordering::Acquire) {
            break;
        }
        let mut bits = signal.take();
        if bits != 0 {
            // Collect the rest of the burst (an unplug raises state, default and property notes).
            let until = Instant::now() + config.debounce;
            while let Some(left) = until.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
                thread::park_timeout(left);
                if stop.load(Ordering::Acquire) {
                    return;
                }
            }
            bits |= signal.take();
        }
        let next = match list() {
            Ok(next) => next,
            Err(e) => {
                if last_error_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(60)) {
                    tracing::warn!(target: "sigil::audio", "audio device enumeration failed: {e}");
                    last_error_log = Some(Instant::now());
                }
                continue;
            }
        };
        let reason = match &last {
            Some(prev) => change_reason(prev, &next),
            None => Some("added"),
        };
        match reason {
            Some(reason) => {
                tracing::info!(
                    target: "sigil::audio",
                    "audio devices {reason}: {} input(s), {} output(s), default in {:?}, out {:?}",
                    next.inputs.len(),
                    next.outputs.len(),
                    next.default_input,
                    next.default_output
                );
                on_notice(WatchNotice::Changed(DevicesChanged {
                    reason: reason.to_string(),
                    snapshot: next.clone(),
                }));
            }
            None if bits != 0 => on_notice(WatchNotice::Nudge),
            None => {}
        }
        last = Some(next);
    }
    drop(registration);
}

#[cfg(windows)]
fn register(signal: &Arc<WatchSignal>) -> Result<com::Registration, String> {
    com::Registration::register(Arc::clone(signal))
}

#[cfg(not(windows))]
fn register(_signal: &Arc<WatchSignal>) -> Result<(), String> {
    Err("device notifications are Windows-only".to_string())
}

#[cfg(windows)]
pub(crate) mod com {
    //! Hand-written COM object implementing `IMMNotificationClient`.

    use std::ffi::c_void;
    use std::sync::atomic::{fence, AtomicU32, Ordering};
    use std::sync::Arc;

    use windows::core::{IUnknown, IUnknown_Vtbl, Interface, GUID, HRESULT, PCWSTR};
    use windows::Win32::Foundation::{E_NOINTERFACE, E_POINTER, PROPERTYKEY, S_OK};
    use windows::Win32::Media::Audio::{
        EDataFlow, ERole, IMMDeviceEnumerator, IMMNotificationClient, IMMNotificationClient_Vtbl, MMDeviceEnumerator,
        DEVICE_STATE,
    };
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED};

    use super::{WatchSignal, NOTE_ADDED, NOTE_DEFAULT, NOTE_PROPERTY, NOTE_REMOVED, NOTE_STATE};

    /// The COM object: the vtable pointer must come first.
    #[repr(C)]
    struct Notifier {
        vtbl: *const IMMNotificationClient_Vtbl,
        refs: AtomicU32,
        signal: Arc<WatchSignal>,
    }

    static VTBL: IMMNotificationClient_Vtbl = IMMNotificationClient_Vtbl {
        base__: IUnknown_Vtbl {
            QueryInterface: query_interface,
            AddRef: add_ref,
            Release: release,
        },
        OnDeviceStateChanged: on_device_state_changed,
        OnDeviceAdded: on_device_added,
        OnDeviceRemoved: on_device_removed,
        OnDefaultDeviceChanged: on_default_device_changed,
        OnPropertyValueChanged: on_property_value_changed,
    };

    unsafe extern "system" fn query_interface(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        if out.is_null() {
            return E_POINTER;
        }
        // SAFETY: COM passes valid pointers; `this` is a live `Notifier`.
        unsafe {
            if !iid.is_null() && (*iid == IUnknown::IID || *iid == IMMNotificationClient::IID) {
                add_ref(this);
                *out = this;
                S_OK
            } else {
                *out = std::ptr::null_mut();
                E_NOINTERFACE
            }
        }
    }

    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        // SAFETY: `this` is a live `Notifier` (COM holds a reference while calling).
        let notifier = unsafe { &*(this as *const Notifier) };
        notifier.refs.fetch_add(1, Ordering::Relaxed) + 1
    }

    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        // SAFETY: as above; the last release frees the box created in `new_client`.
        let left = unsafe { &*(this as *const Notifier) }.refs.fetch_sub(1, Ordering::Release) - 1;
        if left == 0 {
            fence(Ordering::Acquire);
            drop(unsafe { Box::from_raw(this as *mut Notifier) });
        }
        left
    }

    fn raise(this: *mut c_void, bits: u32) -> HRESULT {
        // SAFETY: `this` is a live `Notifier` for the duration of the callback.
        unsafe { &*(this as *const Notifier) }.signal.raise(bits);
        S_OK
    }

    unsafe extern "system" fn on_device_state_changed(this: *mut c_void, _id: PCWSTR, _state: DEVICE_STATE) -> HRESULT {
        raise(this, NOTE_STATE)
    }

    unsafe extern "system" fn on_device_added(this: *mut c_void, _id: PCWSTR) -> HRESULT {
        raise(this, NOTE_ADDED)
    }

    unsafe extern "system" fn on_device_removed(this: *mut c_void, _id: PCWSTR) -> HRESULT {
        raise(this, NOTE_REMOVED)
    }

    unsafe extern "system" fn on_default_device_changed(
        this: *mut c_void,
        _flow: EDataFlow,
        _role: ERole,
        _id: PCWSTR,
    ) -> HRESULT {
        raise(this, NOTE_DEFAULT)
    }

    unsafe extern "system" fn on_property_value_changed(this: *mut c_void, _id: PCWSTR, _key: PROPERTYKEY) -> HRESULT {
        raise(this, NOTE_PROPERTY)
    }

    /// A new notification client holding `signal` (one reference, owned by the returned interface).
    pub(crate) fn new_client(signal: Arc<WatchSignal>) -> IMMNotificationClient {
        let raw = Box::into_raw(Box::new(Notifier {
            vtbl: &VTBL,
            refs: AtomicU32::new(1),
            signal,
        }));
        // SAFETY: `raw` is a COM object whose first field is a valid IMMNotificationClient vtable;
        // the interface takes over the initial reference.
        unsafe { IMMNotificationClient::from_raw(raw.cast()) }
    }

    /// The registered client; unregisters (and balances COM initialization) on drop.
    pub(crate) struct Registration {
        objects: Option<(IMMDeviceEnumerator, IMMNotificationClient)>,
        com_initialized: bool,
    }

    impl Registration {
        pub(crate) fn register(signal: Arc<WatchSignal>) -> Result<Self, String> {
            // SAFETY: plain COM initialization of this (watcher) thread.
            let com_initialized = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok();
            let mut registration = Self {
                objects: None,
                com_initialized,
            };
            // SAFETY: COM is initialized on this thread (or was already, in a compatible mode).
            let enumerator: IMMDeviceEnumerator = unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
                .map_err(|e| format!("MMDeviceEnumerator: {e}"))?;
            let client = new_client(signal);
            // SAFETY: `client` stays alive in `objects` until it is unregistered in `drop`.
            unsafe { enumerator.RegisterEndpointNotificationCallback(&client) }
                .map_err(|e| format!("RegisterEndpointNotificationCallback: {e}"))?;
            registration.objects = Some((enumerator, client));
            Ok(registration)
        }
    }

    impl Drop for Registration {
        fn drop(&mut self) {
            if let Some((enumerator, client)) = self.objects.take() {
                // SAFETY: unregisters the client registered above.
                let _ = unsafe { enumerator.UnregisterEndpointNotificationCallback(&client) };
                drop(client);
                drop(enumerator);
            }
            if self.com_initialized {
                // SAFETY: balances the successful CoInitializeEx on this thread.
                unsafe { CoUninitialize() };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(inputs: &[&str], outputs: &[&str], di: Option<&str>, dout: Option<&str>) -> DeviceSnapshot {
        DeviceSnapshot {
            inputs: inputs.iter().map(|n| AudioDevice::named(n)).collect(),
            outputs: outputs.iter().map(|n| AudioDevice::named(n)).collect(),
            default_input: di.map(str::to_string),
            default_output: dout.map(str::to_string),
        }
    }

    #[test]
    fn change_reasons_come_from_the_diff() {
        let base = snap(&["Mic", "USB Mic"], &["Speakers"], Some("USB Mic"), Some("Speakers"));
        assert_eq!(change_reason(&base, &base.clone()), None);
        // Unplug: the device goes and the default moves: "removed" wins.
        let unplugged = snap(&["Mic"], &["Speakers"], Some("Mic"), Some("Speakers"));
        assert_eq!(change_reason(&base, &unplugged), Some("removed"));
        assert_eq!(change_reason(&unplugged, &base), Some("added"));
        let default_moved = snap(&["Mic", "USB Mic"], &["Speakers"], Some("Mic"), Some("Speakers"));
        assert_eq!(change_reason(&base, &default_moved), Some("default_changed"));
        let reordered = snap(&["USB Mic", "Mic"], &["Speakers"], Some("USB Mic"), Some("Speakers"));
        assert_eq!(change_reason(&base, &reordered), Some("state_changed"));
        // Two identical devices: unplugging one of them is a removal.
        let twins = snap(&["Mic", "Mic"], &[], None, None);
        let one = snap(&["Mic"], &[], None, None);
        assert_eq!(change_reason(&twins, &one), Some("removed"));
        assert_eq!(change_reason(&one, &twins), Some("added"));
        // Output-only changes count too.
        let headset = snap(&["Mic", "USB Mic"], &["Speakers", "Headset"], Some("USB Mic"), Some("Speakers"));
        assert_eq!(change_reason(&base, &headset), Some("added"));
    }

    #[test]
    fn devices_changed_wire_format_matches_the_frozen_ts_type() {
        let ev = DevicesChanged {
            reason: "added".into(),
            snapshot: snap(&["Mic"], &["Speakers"], Some("Mic"), None),
        };
        assert_eq!(
            serde_json::to_value(&ev).unwrap(),
            serde_json::json!({
                "reason": "added",
                "inputs": [{"id": "Mic", "name": "Mic"}],
                "outputs": [{"id": "Speakers", "name": "Speakers"}],
                "default_input": "Mic",
                "default_output": null
            })
        );
    }

    #[test]
    fn polling_watcher_reports_changes_and_stops() {
        use std::sync::mpsc;
        use std::sync::Mutex;
        use std::sync::atomic::AtomicUsize;
        let world = Arc::new(Mutex::new(snap(&["Mic"], &["Speakers"], Some("Mic"), Some("Speakers"))));
        let listed = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel();
        let list = {
            let world = Arc::clone(&world);
            let listed = Arc::clone(&listed);
            move || {
                listed.fetch_add(1, Ordering::SeqCst);
                Ok(world.lock().unwrap().clone())
            }
        };
        let config = WatchConfig {
            poll: Duration::from_millis(20),
            safety_poll: Duration::from_millis(20),
            debounce: Duration::from_millis(5),
        };
        let watcher = start(list, move |n| { let _ = tx.send(n); }, config).unwrap();
        // Change the world only after the watcher took its baseline snapshot.
        let deadline = Instant::now() + Duration::from_secs(5);
        while listed.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        *world.lock().unwrap() = snap(&["Mic", "USB Mic"], &["Speakers"], Some("USB Mic"), Some("Speakers"));
        let notice = rx.recv_timeout(Duration::from_secs(5)).expect("change reported");
        match notice {
            WatchNotice::Changed(ev) => {
                assert_eq!(ev.reason, "added");
                assert_eq!(ev.snapshot.default_input.as_deref(), Some("USB Mic"));
            }
            other => panic!("unexpected {other:?}"),
        }
        watcher.stop();
    }

    /// The hand-written COM object: reference counting, QueryInterface, callbacks raising bits,
    /// and a real registration with the MMDevice enumerator (no audio device needed).
    #[cfg(windows)]
    #[test]
    fn com_notification_client_counts_references_and_raises_bits() {
        use windows::core::{IUnknown, Interface, PCWSTR};
        use windows::Win32::Foundation::PROPERTYKEY;
        use windows::Win32::Media::Audio::{eCapture, eConsole, DEVICE_STATE_ACTIVE};

        std::thread::spawn(|| {
            let signal = Arc::new(WatchSignal::new(std::thread::current()));
            let client = com::new_client(Arc::clone(&signal));
            assert_eq!(Arc::strong_count(&signal), 2);
            {
                let unknown: IUnknown = client.cast().expect("QueryInterface(IUnknown)");
                let again: IMMNotificationClientAlias = unknown.cast().expect("QueryInterface(IMMNotificationClient)");
                drop(again);
            }
            // SAFETY: calling our own vtable entries with a live object.
            unsafe {
                let vtbl = Interface::vtable(&client);
                assert!((vtbl.OnDeviceAdded)(client.as_raw(), PCWSTR::null()).is_ok());
                assert!((vtbl.OnDefaultDeviceChanged)(client.as_raw(), eCapture, eConsole, PCWSTR::null()).is_ok());
                assert!((vtbl.OnDeviceStateChanged)(client.as_raw(), PCWSTR::null(), DEVICE_STATE_ACTIVE).is_ok());
                assert!((vtbl.OnPropertyValueChanged)(client.as_raw(), PCWSTR::null(), PROPERTYKEY::default()).is_ok());
            }
            assert_eq!(signal.take(), NOTE_ADDED | NOTE_DEFAULT | NOTE_STATE | NOTE_PROPERTY);
            assert_eq!(signal.take(), 0);
            drop(client);
            assert_eq!(Arc::strong_count(&signal), 1, "last Release frees the object");

            // A real registration round trip.
            let registration = com::Registration::register(Arc::clone(&signal)).expect("register");
            drop(registration);
            assert_eq!(Arc::strong_count(&signal), 1, "unregistering releases the client");
        })
        .join()
        .unwrap();
    }

    #[cfg(windows)]
    type IMMNotificationClientAlias = windows::Win32::Media::Audio::IMMNotificationClient;
}
