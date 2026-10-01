//! DirectML on a high-priority D3D12 command queue (P3-C2; the `dml_hp` accelerator and the
//! `gpu_priority` lever on the DirectML rung): Sigil creates the session's D3D12 device, a DIRECT
//! command queue at `D3D12_COMMAND_QUEUE_PRIORITY_HIGH` (no admin rights; `GLOBAL_REALTIME` is never
//! requested) and the DirectML device, and passes them to ONNX Runtime through
//! `OrtDmlApi::SessionOptionsAppendExecutionProvider_DML1` (exposed by ort rc.13, ADR 0001). The
//! Windows GPU scheduler then runs the voice engine's command lists ahead of normal-priority work
//! such as VRChat's rendering and OBS's encoder queue.
//!
//! The queue mirrors ONNX Runtime's own DirectML setup (DXGI adapter `EnumAdapters1(device_id)`,
//! feature level 11_0, a DIRECT queue with `DISABLE_GPU_TIMEOUT`); only the priority differs. One
//! queue per adapter is created lazily and shared by every voice-engine session on it (D3D12 devices
//! are per-adapter singletons anyway) **while a session uses it**: the cache holds weak references,
//! so the objects are released with the last plan. A device the driver removed (a TDR or driver
//! reset, `GetDeviceRemovedReason`) is never handed out again: while Sigil held a removed device,
//! `D3D12CreateDevice` on that adapter would fail for every DirectML session in the process (ONNX
//! Runtime's own default-queue EP and the canary included) until a restart.
//!
//! `windows` 0.61 is built without the Direct3D 12 / DirectML features (a Cargo.toml hotspot), so the
//! calls needed are made through their stable COM ABI: `D3D12CreateDevice` (`d3d12.dll`, System32),
//! `ID3D12Device::CreateCommandQueue` (vtable slot 8), `ID3D12Device::GetDeviceRemovedReason` (slot
//! 37) and `DMLCreateDevice` (`DirectML.dll`, the copy next to `Sigil.exe` that ONNX Runtime uses).
//!
//! # Frozen by P3-S0
//! - [`available`]`() -> bool` (cheap; `voice_changer::capabilities` calls it).

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, Weak};

/// The live entry of `key` in a cache of weak references, or a new one from `create` (stored
/// weakly). An entry whose value is gone or `usable` refuses is replaced. `create` errors are not
/// cached (a later call retries).
pub(crate) fn shared_or_create<K: Ord + Copy, T>(
    cache: &mut BTreeMap<K, Weak<T>>,
    key: K,
    usable: impl Fn(&T) -> Result<(), String>,
    create: impl FnOnce() -> Result<T, String>,
) -> Result<Arc<T>, String> {
    if let Some(live) = cache.get(&key).and_then(Weak::upgrade) {
        match usable(&live) {
            Ok(()) => return Ok(live),
            Err(why) => tracing::warn!(target: "sigil::gpu", "{why}; the high-priority DirectML queue is created again"),
        }
    }
    // Dead or unusable: forget it (plans that still hold it keep it alive until they are dropped).
    cache.remove(&key);
    let fresh = Arc::new(create()?);
    cache.insert(key, Arc::downgrade(&fresh));
    Ok(fresh)
}

/// Whether DirectML sessions can run on a high-priority queue in this build: Windows, the linked
/// ONNX Runtime exposes `OrtDmlApi`, and `d3d12.dll` and `DirectML.dll` export the entry points.
/// Cheap after the first call (nothing is created until a session asks for a queue).
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(imp::entry_points_present)
}

pub use imp::DmlQueue;

#[cfg(windows)]
mod imp {
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

    use ort::session::builder::SessionBuilder;
    use ort::AsPointer;
    use windows::core::{Interface, GUID};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

    use crate::services::neural_audio::ort_session::provider_dir;

    /// `D3D_FEATURE_LEVEL_11_0` (ONNX Runtime's minimum for DirectML devices).
    const FEATURE_LEVEL_11_0: i32 = 0xb000;
    /// `D3D12_COMMAND_LIST_TYPE_DIRECT` (what ONNX Runtime's own DirectML queue uses).
    const COMMAND_LIST_TYPE_DIRECT: i32 = 0;
    /// `D3D12_COMMAND_QUEUE_PRIORITY_HIGH` (NORMAL = 0; GLOBAL_REALTIME = 10000 needs privileges).
    pub const QUEUE_PRIORITY_HIGH: i32 = 100;
    /// `D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT` (as ONNX Runtime sets it).
    const QUEUE_FLAG_DISABLE_GPU_TIMEOUT: i32 = 1;
    /// `DML_CREATE_DEVICE_FLAG_NONE`.
    const DML_FLAG_NONE: i32 = 0;

    const IID_ID3D12_DEVICE: GUID = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);
    const IID_ID3D12_COMMAND_QUEUE: GUID = GUID::from_u128(0x0ec870a6_5d7e_4c22_8cfc_5baae07616ed);
    const IID_IDML_DEVICE: GUID = GUID::from_u128(0x6dbd6437_96fd_423f_a98c_ae5e7c2a573f);

    /// `D3D12_COMMAND_QUEUE_DESC`.
    #[repr(C)]
    struct QueueDesc {
        ty: i32,
        priority: i32,
        flags: i32,
        node_mask: u32,
    }

    type D3D12CreateDeviceFn = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;
    type DmlCreateDeviceFn = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;
    type CreateCommandQueueFn = unsafe extern "system" fn(*mut c_void, *const QueueDesc, *const GUID, *mut *mut c_void) -> i32;
    type GetDeviceRemovedReasonFn = unsafe extern "system" fn(*mut c_void) -> i32;
    type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;

    /// `ID3D12Device::CreateCommandQueue`: IUnknown (3) + ID3D12Object (4) + GetNodeCount (1).
    const VTBL_CREATE_COMMAND_QUEUE: usize = 8;
    /// `ID3D12Device::GetDeviceRemovedReason` (d3d12.h order: ... `Evict` 35, `CreateFence` 36).
    const VTBL_GET_DEVICE_REMOVED_REASON: usize = 37;
    /// `IUnknown::Release`.
    const VTBL_RELEASE: usize = 2;

    struct Entry {
        d3d12_create_device: D3D12CreateDeviceFn,
        dml_create_device: DmlCreateDeviceFn,
        _d3d12: libloading::Library,
        _dml: libloading::Library,
    }

    fn entry() -> Result<&'static Entry, String> {
        static ENTRY: OnceLock<Result<Entry, String>> = OnceLock::new();
        ENTRY
            .get_or_init(|| {
                use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32};
                // SAFETY: the system's Direct3D 12 runtime from System32.
                let d3d12: libloading::Library = unsafe { Library::load_with_flags("d3d12.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }
                    .map(Into::into)
                    .map_err(|e| format!("d3d12.dll could not be loaded ({e})"))?;
                // The DirectML.dll ONNX Runtime uses sits next to Sigil.exe (bundled; loading the
                // same file returns the already loaded module).
                let dml_path = provider_dir()
                    .map(|d| d.join("DirectML.dll"))
                    .filter(|p| p.is_file())
                    .ok_or_else(|| "DirectML.dll is not installed next to Sigil.exe".to_string())?;
                // SAFETY: Sigil's bundled DirectML runtime, by full path.
                let dml: libloading::Library =
                    unsafe { Library::load_with_flags(&dml_path, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
                        .map(Into::into)
                        .map_err(|e| format!("{} could not be loaded ({e})", dml_path.display()))?;
                // SAFETY: the documented signatures of these exports (d3d12.h, DirectML.h).
                unsafe {
                    let d3d12_create_device = *d3d12
                        .get::<D3D12CreateDeviceFn>(b"D3D12CreateDevice\0")
                        .map_err(|e| format!("d3d12.dll has no D3D12CreateDevice ({e})"))?;
                    let dml_create_device = *dml
                        .get::<DmlCreateDeviceFn>(b"DMLCreateDevice\0")
                        .map_err(|e| format!("DirectML.dll has no DMLCreateDevice ({e})"))?;
                    Ok(Entry { d3d12_create_device, dml_create_device, _d3d12: d3d12, _dml: dml })
                }
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn entry_points_present() -> bool {
        ort::ep::directml::api().is_some() && entry().is_ok()
    }

    /// `IUnknown::Release` through the vtable.
    ///
    /// # Safety
    /// `p` must be a live COM object the caller owns one reference of.
    unsafe fn release(p: *mut c_void) {
        if p.is_null() {
            return;
        }
        // SAFETY: COM objects start with a vtable pointer; slot 2 is Release.
        unsafe {
            let vtbl = *(p as *const *const usize);
            let f: ReleaseFn = std::mem::transmute::<usize, ReleaseFn>(*vtbl.add(VTBL_RELEASE));
            f(p);
        }
    }

    fn hr(what: &str, code: i32) -> String {
        let text = windows::core::HRESULT(code).message();
        format!("{what} failed ({:#010X}: {})", code as u32, text.trim())
    }

    /// A D3D12 device, a high-priority DIRECT command queue and the DirectML device on one
    /// adapter (each COM reference released on drop).
    pub struct DmlQueue {
        adapter: u32,
        d3d12: *mut c_void,
        queue: *mut c_void,
        dml: *mut c_void,
    }

    // SAFETY: D3D12 devices and command queues and DirectML devices are free-threaded COM objects.
    unsafe impl Send for DmlQueue {}
    unsafe impl Sync for DmlQueue {}

    impl std::fmt::Debug for DmlQueue {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DmlQueue").field("adapter", &self.adapter).field("priority", &QUEUE_PRIORITY_HIGH).finish()
        }
    }

    impl DmlQueue {
        /// Create the objects on DXGI adapter `adapter` (DirectML's `device_id`).
        fn create(adapter: u32) -> Result<DmlQueue, String> {
            let entry = entry()?;
            // SAFETY: plain factory creation (DXGI needs no COM apartment).
            let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.map_err(|e| format!("CreateDXGIFactory1: {e}"))?;
            // SAFETY: valid factory; an out-of-range index is an error, not UB.
            let dxgi_adapter = unsafe { factory.EnumAdapters1(adapter) }
                .map_err(|e| format!("the selected GPU does not exist on this PC (DXGI adapter {adapter}: {e})"))?;
            let mut out = DmlQueue { adapter, d3d12: std::ptr::null_mut(), queue: std::ptr::null_mut(), dml: std::ptr::null_mut() };
            // SAFETY: the adapter pointer is live for the call; out pointers are valid; on failure
            // the partially built `out` releases what it holds.
            unsafe {
                let code = (entry.d3d12_create_device)(dxgi_adapter.as_raw(), FEATURE_LEVEL_11_0, &IID_ID3D12_DEVICE, &mut out.d3d12);
                if code < 0 || out.d3d12.is_null() {
                    return Err(hr("D3D12CreateDevice", code));
                }
                let desc = QueueDesc {
                    ty: COMMAND_LIST_TYPE_DIRECT,
                    priority: QUEUE_PRIORITY_HIGH,
                    flags: QUEUE_FLAG_DISABLE_GPU_TIMEOUT,
                    node_mask: 0,
                };
                let vtbl = *(out.d3d12 as *const *const usize);
                let create_queue: CreateCommandQueueFn = std::mem::transmute::<usize, CreateCommandQueueFn>(*vtbl.add(VTBL_CREATE_COMMAND_QUEUE));
                let code = create_queue(out.d3d12, &desc, &IID_ID3D12_COMMAND_QUEUE, &mut out.queue);
                if code < 0 || out.queue.is_null() {
                    return Err(hr("ID3D12Device::CreateCommandQueue (high priority)", code));
                }
                let code = (entry.dml_create_device)(out.d3d12, DML_FLAG_NONE, &IID_IDML_DEVICE, &mut out.dml);
                if code < 0 || out.dml.is_null() {
                    return Err(hr("DMLCreateDevice", code));
                }
            }
            // D3D12 returns the adapter's existing device while anything still holds it, even a
            // removed one: never build on that.
            out.check_not_removed()?;
            tracing::info!(target: "sigil::gpu", "DirectML high-priority queue created on adapter {adapter}");
            Ok(out)
        }

        fn queues() -> &'static Mutex<BTreeMap<u32, Weak<DmlQueue>>> {
            static QUEUES: Mutex<BTreeMap<u32, Weak<DmlQueue>>> = Mutex::new(BTreeMap::new());
            &QUEUES
        }

        /// The shared high-priority queue of DXGI adapter `adapter`: the one live plans already use,
        /// else a new one (released with the last plan that holds it). A device the driver removed
        /// is replaced; a failure is not cached, so a later session retries.
        pub fn shared(adapter: u32) -> Result<Arc<DmlQueue>, String> {
            let mut queues = Self::queues().lock().unwrap_or_else(PoisonError::into_inner);
            super::shared_or_create(&mut queues, adapter, DmlQueue::check_not_removed, || DmlQueue::create(adapter))
        }

        /// Whether a live queue of `adapter` is cached (tests).
        #[cfg(test)]
        pub(crate) fn cached(adapter: u32) -> bool {
            Self::queues()
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&adapter)
                .is_some_and(|w| w.strong_count() > 0)
        }

        /// `Err` with the reason when the driver removed the D3D12 device (TDR, driver reset or
        /// update, GPU unplugged): every queue and DirectML device made from it is dead.
        pub fn check_not_removed(&self) -> Result<(), String> {
            // SAFETY: a live ID3D12Device this object holds a reference of; slot 37 is
            // GetDeviceRemovedReason (no arguments, returns an HRESULT; free-threaded).
            let code = unsafe {
                let vtbl = *(self.d3d12 as *const *const usize);
                let f: GetDeviceRemovedReasonFn = std::mem::transmute::<usize, GetDeviceRemovedReasonFn>(*vtbl.add(VTBL_GET_DEVICE_REMOVED_REASON));
                f(self.d3d12)
            };
            if code < 0 {
                return Err(hr(&format!("the GPU (DXGI adapter {}) was reset: the D3D12 device was removed; GetDeviceRemovedReason", self.adapter), code));
            }
            Ok(())
        }

        /// DXGI adapter index.
        pub fn adapter(&self) -> u32 {
            self.adapter
        }

        /// The queue priority (`D3D12_COMMAND_QUEUE_PRIORITY_HIGH`).
        pub fn priority(&self) -> i32 {
            QUEUE_PRIORITY_HIGH
        }

        /// Register DirectML on this device and queue for `builder` (DML1; ONNX Runtime takes its
        /// own references). The builder must not have another DirectML EP appended.
        pub fn append_to(&self, builder: &mut SessionBuilder) -> Result<(), String> {
            let api = ort::ep::directml::api().ok_or_else(|| "DirectML is not supported by the linked ONNX Runtime".to_string())?;
            // SAFETY: live options, device and queue pointers (this object holds references for
            // the life of the process).
            let status = unsafe { (api.SessionOptionsAppendExecutionProvider_DML1)(builder.ptr_mut(), self.dml, self.queue) };
            // SAFETY: a status just returned by ORT (released by `result_from_status`).
            unsafe { ort::Error::result_from_status(status) }.map_err(|e| e.to_string())
        }
    }

    impl Drop for DmlQueue {
        fn drop(&mut self) {
            // SAFETY: each pointer holds one reference owned by this object.
            unsafe {
                release(self.dml);
                release(self.queue);
                release(self.d3d12);
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::sync::Arc;

    use ort::session::builder::SessionBuilder;

    pub(super) fn entry_points_present() -> bool {
        false
    }

    /// DirectML exists only on Windows.
    #[derive(Debug)]
    pub struct DmlQueue;

    impl DmlQueue {
        pub fn shared(_adapter: u32) -> Result<Arc<DmlQueue>, String> {
            Err("DirectML is only available on Windows".into())
        }

        pub fn check_not_removed(&self) -> Result<(), String> {
            Ok(())
        }

        pub fn adapter(&self) -> u32 {
            0
        }

        pub fn priority(&self) -> i32 {
            0
        }

        pub fn append_to(&self, _builder: &mut SessionBuilder) -> Result<(), String> {
            Err("DirectML is only available on Windows".into())
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::cell::Cell;

    struct Fake {
        id: u32,
        removed: Cell<bool>,
    }

    /// The cache hands out the live value, creates a new one once the last holder dropped it or
    /// when the live one is unusable (a removed device), and never caches a failure.
    #[test]
    fn weak_cache_shares_live_values_and_replaces_dead_or_removed_ones() {
        let mut cache: BTreeMap<u32, Weak<Fake>> = BTreeMap::new();
        let made = Cell::new(0u32);
        let usable = |f: &Fake| if f.removed.get() { Err(format!("device {} removed", f.id)) } else { Ok(()) };
        let create = || {
            made.set(made.get() + 1);
            Ok(Fake { id: made.get(), removed: Cell::new(false) })
        };
        let a = shared_or_create(&mut cache, 0, usable, create).unwrap();
        let b = shared_or_create(&mut cache, 0, usable, create).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "one live value per key");
        assert_eq!(made.get(), 1);
        let other = shared_or_create(&mut cache, 1, usable, create).unwrap();
        assert_eq!((other.id, made.get()), (2, 2));
        // The last holders go: nothing is kept alive, the next call creates a new value.
        drop((a, b));
        assert!(cache[&0].upgrade().is_none(), "the cache holds no strong reference");
        let c = shared_or_create(&mut cache, 0, usable, create).unwrap();
        assert_eq!((c.id, made.get()), (3, 3));
        // A removed device is replaced even while a plan still holds it.
        c.removed.set(true);
        let d = shared_or_create(&mut cache, 0, usable, create).unwrap();
        assert_eq!(d.id, 4);
        assert!(!Arc::ptr_eq(&c, &d));
        // Failures are not cached.
        drop(d);
        let err = shared_or_create(&mut cache, 0, usable, || Err::<Fake, _>("no GPU".to_string())).err().unwrap();
        assert_eq!(err, "no GPU");
        assert!(!cache.contains_key(&0));
        assert_eq!(shared_or_create(&mut cache, 0, usable, create).unwrap().id, 5);
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem;

    /// On a PC with a hardware adapter, the queue is created and shared per adapter while it is in
    /// use, reports its device as not removed, and is released with its last holder; a missing
    /// adapter is a clean error. (Session runs on it are covered by `plan`'s tests.)
    #[test]
    fn high_priority_queue_is_created_and_shared_per_adapter() {
        let Some(adapter) = gpu_mem::adapters().into_iter().find(|a| !a.software) else {
            eprintln!("no hardware GPU; skipped");
            return;
        };
        if !available() {
            eprintln!("DirectML entry points not present next to the test binary; skipped");
            return;
        }
        let q = DmlQueue::shared(adapter.index).expect("high-priority queue");
        assert_eq!(q.adapter(), adapter.index);
        assert_eq!(q.priority(), 100);
        q.check_not_removed().expect("a fresh device is not removed");
        let again = DmlQueue::shared(adapter.index).unwrap();
        assert!(std::sync::Arc::ptr_eq(&q, &again), "one queue per adapter");
        assert!(DmlQueue::cached(adapter.index));
        let err = DmlQueue::shared(63).unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        // Other tests may hold the same adapter's queue concurrently (DirectML plans): only when
        // these were the last holders is the cache empty afterwards.
        let last = std::sync::Arc::strong_count(&q) == 2;
        drop((q, again));
        if last {
            assert!(!DmlQueue::cached(adapter.index), "the queue is released with its last holder");
        }
    }
}
