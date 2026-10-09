//! The CUDA driver API subset the GPU layer needs (P3-C2), loaded lazily with `libloading` from
//! the NVIDIA display driver's `nvcuda.dll` (System32 only; nothing is load-time linked).
//!
//! Why the driver API and not `cudart64_13.dll`: `nvcuda.dll` ships with every NVIDIA driver (the
//! CUDA EP cannot run without it), so the priority stream and device copies do not depend on the
//! NVIDIA runtime pack being installed or on which cudart copy another engine loaded. ONNX
//! Runtime's CUDA provider links cudart statically and runs on each device's **primary context**;
//! everything here retains and uses that same context ([`PrimaryContext`]), so streams and device
//! pointers are shared with ORT sessions on that device.
//!
//! Every call returns `Err` with the driver's error name (`CUDA_ERROR_OUT_OF_MEMORY`, ...); nothing
//! here panics. The hot-path calls ([`ContextScope::copy_htod`], [`ContextScope::copy_dtoh`],
//! [`ContextScope::copy_2d_dtod`], [`ContextScope::copy_2d_dtoh`], [`ContextScope::sync`],
//! [`PrimaryContext::enter`]) never allocate on success.

use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

/// `CUresult`.
type CuResult = i32;
/// `CUdevice`.
pub type CuDevice = i32;
/// `CUcontext`.
pub type CuContext = *mut c_void;
/// `CUstream`.
pub type CuStream = *mut c_void;
/// `CUdeviceptr`.
pub type CuDevicePtr = u64;
/// `CUgraph`.
pub type CuGraph = *mut c_void;
/// `CUgraphExec`.
pub type CuGraphExec = *mut c_void;
/// `CUevent`.
pub type CuEvent = *mut c_void;

/// `CU_EVENT_DEFAULT` (timing on, the waiting thread spins or yields per the context's flags).
pub const EVENT_DEFAULT: u32 = 0;
/// `CU_EVENT_BLOCKING_SYNC`: `cuEventSynchronize` blocks the calling thread (an OS wait woken by
/// the driver) instead of spinning, whatever the context's scheduling flags (P3B-G).
pub const EVENT_BLOCKING_SYNC: u32 = 1;
/// `CU_EVENT_DISABLE_TIMING`: no timestamps (cheaper record and wait).
pub const EVENT_DISABLE_TIMING: u32 = 2;

/// `CU_CTX_SCHED_*` (the low 3 bits of a context's flags): how a host thread waits for the GPU in
/// `cuStreamSynchronize` / `cuCtxSynchronize` and in ONNX Runtime's own syncs. `AUTO` (the driver
/// default) spins while the process has fewer active contexts than logical CPUs.
pub const CTX_SCHED_AUTO: u32 = 0;
pub const CTX_SCHED_SPIN: u32 = 1;
pub const CTX_SCHED_YIELD: u32 = 2;
pub const CTX_SCHED_BLOCKING_SYNC: u32 = 4;
pub const CTX_SCHED_MASK: u32 = 7;

/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: only this thread's unsafe calls are refused during the
/// capture; other threads (STT / TTS engines, a game overlay) are not affected, unlike the global
/// mode ONNX Runtime's own capture uses.
pub const CAPTURE_MODE_THREAD_LOCAL: u32 = 1;

const CUDA_SUCCESS: CuResult = 0;
/// `CUDA_ERROR_NOT_READY` (`cuEventQuery` / `cuStreamQuery`: the work has not completed yet).
pub const CUDA_ERROR_NOT_READY: CuResult = 600;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` / `_MINOR`.
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;
/// `CU_STREAM_NON_BLOCKING`: no implicit synchronisation with the legacy default stream.
pub const STREAM_NON_BLOCKING: u32 = 1;
/// `CUmemorytype`.
const MEMORYTYPE_HOST: u32 = 1;
const MEMORYTYPE_DEVICE: u32 = 2;

/// `CUDA_MEMCPY2D` (the `_v2` layout; `CUmemorytype` is a 32-bit enum, padded by `repr(C)`).
#[repr(C)]
struct Memcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    src_host: *const c_void,
    src_device: CuDevicePtr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: CuDevicePtr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

type FnInit = unsafe extern "system" fn(u32) -> CuResult;
type FnDriverGetVersion = unsafe extern "system" fn(*mut i32) -> CuResult;
type FnDeviceGetCount = unsafe extern "system" fn(*mut i32) -> CuResult;
type FnDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type FnDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, CuDevice) -> CuResult;
type FnDeviceGetName = unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult;
type FnPrimaryCtxRetain = unsafe extern "system" fn(*mut CuContext, CuDevice) -> CuResult;
type FnPrimaryCtxRelease = unsafe extern "system" fn(CuDevice) -> CuResult;
type FnCtxPush = unsafe extern "system" fn(CuContext) -> CuResult;
type FnCtxPop = unsafe extern "system" fn(*mut CuContext) -> CuResult;
type FnCtxStreamPriorityRange = unsafe extern "system" fn(*mut i32, *mut i32) -> CuResult;
type FnStreamCreateWithPriority = unsafe extern "system" fn(*mut CuStream, u32, i32) -> CuResult;
type FnStreamGetPriority = unsafe extern "system" fn(CuStream, *mut i32) -> CuResult;
type FnStreamDestroy = unsafe extern "system" fn(CuStream) -> CuResult;
type FnStreamSynchronize = unsafe extern "system" fn(CuStream) -> CuResult;
type FnMemcpyHtoDAsync = unsafe extern "system" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult;
type FnMemcpyDtoHAsync = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult;
type FnMemcpy2DAsync = unsafe extern "system" fn(*const Memcpy2D, CuStream) -> CuResult;
type FnMemsetD32Async = unsafe extern "system" fn(CuDevicePtr, u32, usize, CuStream) -> CuResult;
type FnMemHostAlloc = unsafe extern "system" fn(*mut *mut c_void, usize, u32) -> CuResult;
type FnMemFreeHost = unsafe extern "system" fn(*mut c_void) -> CuResult;
type FnMemGetInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult;
type FnGetErrorName = unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult;
type FnStreamBeginCapture = unsafe extern "system" fn(CuStream, u32) -> CuResult;
type FnStreamEndCapture = unsafe extern "system" fn(CuStream, *mut CuGraph) -> CuResult;
type FnStreamIsCapturing = unsafe extern "system" fn(CuStream, *mut u32) -> CuResult;
type FnGraphInstantiate = unsafe extern "system" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult;
type FnGraphLaunch = unsafe extern "system" fn(CuGraphExec, CuStream) -> CuResult;
type FnGraphExecDestroy = unsafe extern "system" fn(CuGraphExec) -> CuResult;
type FnGraphDestroy = unsafe extern "system" fn(CuGraph) -> CuResult;
type FnEventCreate = unsafe extern "system" fn(*mut CuEvent, u32) -> CuResult;
type FnEventRecord = unsafe extern "system" fn(CuEvent, CuStream) -> CuResult;
type FnEventQuery = unsafe extern "system" fn(CuEvent) -> CuResult;
type FnEventSynchronize = unsafe extern "system" fn(CuEvent) -> CuResult;
type FnEventElapsedTime = unsafe extern "system" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type FnEventDestroy = unsafe extern "system" fn(CuEvent) -> CuResult;
type FnStreamWaitEvent = unsafe extern "system" fn(CuStream, CuEvent, u32) -> CuResult;
type FnCtxSetFlags = unsafe extern "system" fn(u32) -> CuResult;
type FnCtxGetFlags = unsafe extern "system" fn(*mut u32) -> CuResult;
type FnPrimaryCtxSetFlags = unsafe extern "system" fn(CuDevice, u32) -> CuResult;
type FnPrimaryCtxGetState = unsafe extern "system" fn(CuDevice, *mut u32, *mut i32) -> CuResult;

/// The loaded driver entry points (process lifetime; the library is never unloaded).
pub struct CudaDriver {
    _lib: libloading::Library,
    driver_version: i32,
    device_count: i32,
    device_get: FnDeviceGet,
    device_get_attribute: FnDeviceGetAttribute,
    device_get_name: FnDeviceGetName,
    primary_ctx_retain: FnPrimaryCtxRetain,
    primary_ctx_release: FnPrimaryCtxRelease,
    ctx_push: FnCtxPush,
    ctx_pop: FnCtxPop,
    ctx_stream_priority_range: FnCtxStreamPriorityRange,
    stream_create_with_priority: FnStreamCreateWithPriority,
    stream_get_priority: FnStreamGetPriority,
    stream_destroy: FnStreamDestroy,
    stream_synchronize: FnStreamSynchronize,
    memcpy_htod_async: FnMemcpyHtoDAsync,
    memcpy_dtoh_async: FnMemcpyDtoHAsync,
    memcpy_2d_async: FnMemcpy2DAsync,
    memset_d32_async: FnMemsetD32Async,
    mem_host_alloc: FnMemHostAlloc,
    mem_free_host: FnMemFreeHost,
    mem_get_info: FnMemGetInfo,
    get_error_name: FnGetErrorName,
    stream_begin_capture: FnStreamBeginCapture,
    stream_end_capture: FnStreamEndCapture,
    stream_is_capturing: FnStreamIsCapturing,
    graph_instantiate: FnGraphInstantiate,
    graph_launch: FnGraphLaunch,
    graph_exec_destroy: FnGraphExecDestroy,
    graph_destroy: FnGraphDestroy,
    // P3B-G: events (GPU time per plan run, blocking waits, cross-stream ordering).
    event_create: FnEventCreate,
    event_record: FnEventRecord,
    event_query: FnEventQuery,
    event_synchronize: FnEventSynchronize,
    event_elapsed_time: FnEventElapsedTime,
    event_destroy: FnEventDestroy,
    stream_wait_event: FnStreamWaitEvent,
    // P3B-G: the context's scheduling flags (optional exports: `cuCtxSetFlags` is CUDA 12.1+).
    ctx_set_flags: Option<FnCtxSetFlags>,
    ctx_get_flags: Option<FnCtxGetFlags>,
    primary_ctx_set_flags: Option<FnPrimaryCtxSetFlags>,
    primary_ctx_get_state: Option<FnPrimaryCtxGetState>,
}

// SAFETY: the struct holds only function pointers into a library that stays loaded for the life of
// the process, plus plain integers; the CUDA driver API is thread-safe.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

#[cfg(windows)]
const DRIVER_DLL: &str = "nvcuda.dll";
#[cfg(not(windows))]
const DRIVER_DLL: &str = "libcuda.so.1";

fn load_library() -> Result<libloading::Library, String> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
        // SAFETY: the NVIDIA driver's user-mode library from System32 only (never PATH or the
        // current directory); its initialisation routines are the driver's own.
        unsafe { Library::load_with_flags(DRIVER_DLL, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .map(libloading::Library::from)
            .map_err(|e| format!("the NVIDIA driver's {DRIVER_DLL} could not be loaded ({e})"))
    }
    #[cfg(not(windows))]
    {
        // SAFETY: the system CUDA driver library.
        unsafe { libloading::Library::new(DRIVER_DLL) }.map_err(|e| format!("{DRIVER_DLL} could not be loaded ({e})"))
    }
}

/// Resolve one export as `T` (a function pointer type of exactly that export's signature).
///
/// # Safety
/// `T` must be the export's real signature.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T, String> {
    let mut z = Vec::with_capacity(name.len() + 1);
    z.extend_from_slice(name.as_bytes());
    z.push(0);
    // SAFETY: forwarded to the caller (the signature matches cuda.h for this export).
    unsafe { lib.get::<T>(z.as_slice()) }
        .map(|s| *s)
        .map_err(|e| format!("{DRIVER_DLL} has no {name} ({e}); the NVIDIA driver is too old"))
}

impl CudaDriver {
    fn load() -> Result<CudaDriver, String> {
        let lib = load_library()?;
        // SAFETY: every signature below is the one cuda.h declares for that export (the `_v2`
        // names are what the CUDA headers map the unsuffixed names to since CUDA 3.2 / 11).
        unsafe {
            let init: FnInit = sym(&lib, "cuInit")?;
            let driver_get_version: FnDriverGetVersion = sym(&lib, "cuDriverGetVersion")?;
            let device_get_count: FnDeviceGetCount = sym(&lib, "cuDeviceGetCount")?;
            let get_error_name: FnGetErrorName = sym(&lib, "cuGetErrorName")?;
            let named = |r: CuResult| -> String { error_name(get_error_name, r) };
            let r = init(0);
            if r != CUDA_SUCCESS {
                return Err(format!("the CUDA driver failed to initialise ({})", named(r)));
            }
            let mut driver_version = 0;
            let _ = driver_get_version(&mut driver_version);
            let mut device_count = 0;
            let r = device_get_count(&mut device_count);
            if r != CUDA_SUCCESS {
                return Err(format!("the CUDA driver lists no devices ({})", named(r)));
            }
            Ok(CudaDriver {
                driver_version,
                device_count,
                device_get: sym(&lib, "cuDeviceGet")?,
                device_get_attribute: sym(&lib, "cuDeviceGetAttribute")?,
                device_get_name: sym(&lib, "cuDeviceGetName")?,
                primary_ctx_retain: sym(&lib, "cuDevicePrimaryCtxRetain")?,
                primary_ctx_release: sym(&lib, "cuDevicePrimaryCtxRelease_v2")?,
                ctx_push: sym(&lib, "cuCtxPushCurrent_v2")?,
                ctx_pop: sym(&lib, "cuCtxPopCurrent_v2")?,
                ctx_stream_priority_range: sym(&lib, "cuCtxGetStreamPriorityRange")?,
                stream_create_with_priority: sym(&lib, "cuStreamCreateWithPriority")?,
                stream_get_priority: sym(&lib, "cuStreamGetPriority")?,
                stream_destroy: sym(&lib, "cuStreamDestroy_v2")?,
                stream_synchronize: sym(&lib, "cuStreamSynchronize")?,
                memcpy_htod_async: sym(&lib, "cuMemcpyHtoDAsync_v2")?,
                memcpy_dtoh_async: sym(&lib, "cuMemcpyDtoHAsync_v2")?,
                memcpy_2d_async: sym(&lib, "cuMemcpy2DAsync_v2")?,
                memset_d32_async: sym(&lib, "cuMemsetD32Async")?,
                mem_host_alloc: sym(&lib, "cuMemHostAlloc")?,
                mem_free_host: sym(&lib, "cuMemFreeHost")?,
                mem_get_info: sym(&lib, "cuMemGetInfo_v2")?,
                get_error_name,
                stream_begin_capture: sym(&lib, "cuStreamBeginCapture_v2")?,
                stream_end_capture: sym(&lib, "cuStreamEndCapture")?,
                stream_is_capturing: sym(&lib, "cuStreamIsCapturing")?,
                graph_instantiate: sym(&lib, "cuGraphInstantiateWithFlags")?,
                graph_launch: sym(&lib, "cuGraphLaunch")?,
                graph_exec_destroy: sym(&lib, "cuGraphExecDestroy")?,
                graph_destroy: sym(&lib, "cuGraphDestroy")?,
                event_create: sym(&lib, "cuEventCreate")?,
                event_record: sym(&lib, "cuEventRecord")?,
                event_query: sym(&lib, "cuEventQuery")?,
                event_synchronize: sym(&lib, "cuEventSynchronize")?,
                // CUDA 12.8+ headers map cuEventElapsedTime to the `_v2` export (same signature).
                event_elapsed_time: sym(&lib, "cuEventElapsedTime_v2").or_else(|_| sym(&lib, "cuEventElapsedTime"))?,
                event_destroy: sym(&lib, "cuEventDestroy_v2")?,
                stream_wait_event: sym(&lib, "cuStreamWaitEvent")?,
                ctx_set_flags: sym(&lib, "cuCtxSetFlags").ok(),
                ctx_get_flags: sym(&lib, "cuCtxGetFlags").ok(),
                primary_ctx_set_flags: sym(&lib, "cuDevicePrimaryCtxSetFlags_v2").ok(),
                primary_ctx_get_state: sym(&lib, "cuDevicePrimaryCtxGetState").ok(),
                _lib: lib,
            })
        }
    }

    /// The driver, loaded and initialised on first use (the first result is kept for the process).
    pub fn get() -> Result<&'static CudaDriver, String> {
        static DRIVER: OnceLock<Result<CudaDriver, String>> = OnceLock::new();
        DRIVER.get_or_init(CudaDriver::load).as_ref().map_err(Clone::clone)
    }

    /// `cuDriverGetVersion` (e.g. `13040` for CUDA 13.4).
    pub fn driver_version(&self) -> i32 {
        self.driver_version
    }

    pub fn device_count(&self) -> i32 {
        self.device_count
    }

    fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: {}", error_name(self.get_error_name, r)))
        }
    }

    /// Like [`check`](Self::check), without formatting on success or failure (hot path: the error
    /// text is built only by the caller's cold path).
    #[inline]
    fn ok(r: CuResult) -> Result<(), CuResult> {
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(r)
        }
    }

    /// The error name of a raw result (cold path).
    pub fn describe(&self, r: i32) -> String {
        error_name(self.get_error_name, r)
    }

    fn device(&self, ordinal: i32) -> Result<CuDevice, String> {
        if ordinal < 0 || ordinal >= self.device_count {
            return Err(format!("CUDA device {ordinal} does not exist ({} CUDA device(s) found)", self.device_count));
        }
        let mut dev = 0;
        // SAFETY: valid out pointer; ordinal range checked.
        self.check(unsafe { (self.device_get)(&mut dev, ordinal) }, "cuDeviceGet")?;
        Ok(dev)
    }

    /// Compute capability `(major, minor)` of CUDA device `ordinal`.
    pub fn compute_capability(&self, ordinal: i32) -> Result<(i32, i32), String> {
        let dev = self.device(ordinal)?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid out pointers and device.
        unsafe {
            self.check((self.device_get_attribute)(&mut major, ATTR_CC_MAJOR, dev), "cuDeviceGetAttribute")?;
            self.check((self.device_get_attribute)(&mut minor, ATTR_CC_MINOR, dev), "cuDeviceGetAttribute")?;
        }
        Ok((major, minor))
    }

    /// Marketing name of CUDA device `ordinal` (`NVIDIA GeForce RTX 4090`).
    pub fn device_name(&self, ordinal: i32) -> Result<String, String> {
        let dev = self.device(ordinal)?;
        let mut buf = [0 as c_char; 256];
        // SAFETY: the buffer length is passed; the driver NUL-terminates within it.
        self.check(unsafe { (self.device_get_name)(buf.as_mut_ptr(), buf.len() as i32, dev) }, "cuDeviceGetName")?;
        // SAFETY: NUL-terminated by the driver (and the last byte of `buf` is never written past).
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }
}

fn error_name(get_error_name: FnGetErrorName, r: CuResult) -> String {
    let mut p: *const c_char = std::ptr::null();
    // SAFETY: the driver returns a static string (or leaves the pointer null for unknown codes).
    let known = unsafe { get_error_name(r, &mut p) } == CUDA_SUCCESS && !p.is_null();
    if known {
        // SAFETY: static NUL-terminated string owned by the driver.
        format!("{} ({r})", unsafe { CStr::from_ptr(p) }.to_string_lossy())
    } else {
        format!("CUDA error {r}")
    }
}

/// A retained primary context of one device (released on drop). ONNX Runtime's CUDA EP runs on the
/// same context, so streams and pointers made under it are valid in its sessions.
pub struct PrimaryContext {
    driver: &'static CudaDriver,
    device: CuDevice,
    ordinal: i32,
    ctx: CuContext,
}

// SAFETY: a CUDA context handle may be used from any thread (it is made current per call through
// `ContextScope`); retain / release are thread-safe.
unsafe impl Send for PrimaryContext {}
unsafe impl Sync for PrimaryContext {}

impl PrimaryContext {
    /// Retain the primary context of CUDA device `ordinal` (the one ORT's CUDA EP uses).
    pub fn retain(ordinal: i32) -> Result<PrimaryContext, String> {
        let driver = CudaDriver::get()?;
        let device = driver.device(ordinal)?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: valid out pointer and device.
        driver.check(unsafe { (driver.primary_ctx_retain)(&mut ctx, device) }, "cuDevicePrimaryCtxRetain")?;
        if ctx.is_null() {
            return Err("cuDevicePrimaryCtxRetain returned no context".into());
        }
        Ok(PrimaryContext { driver, device, ordinal, ctx })
    }

    pub fn driver(&self) -> &'static CudaDriver {
        self.driver
    }

    /// CUDA device ordinal.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Make this context current on the calling thread until the scope drops (then the previous
    /// one is restored). Never allocates.
    #[inline]
    pub fn enter(&self) -> Result<ContextScope<'_>, i32> {
        // SAFETY: a retained, valid context.
        CudaDriver::ok(unsafe { (self.driver.ctx_push)(self.ctx) })?;
        Ok(ContextScope { ctx: self })
    }

    /// `(least, greatest)` stream priorities of this device (greatest is numerically lowest, e.g.
    /// `(0, -5)`).
    pub fn stream_priority_range(&self) -> Result<(i32, i32), String> {
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        let (mut least, mut greatest) = (0, 0);
        // SAFETY: valid out pointers; a context is current.
        self.driver.check(
            unsafe { (self.driver.ctx_stream_priority_range)(&mut least, &mut greatest) },
            "cuCtxGetStreamPriorityRange",
        )?;
        Ok((least, greatest))
    }

    /// Create a non-blocking stream at `priority` (clamped by the driver to the device's range).
    pub fn create_stream(&self, priority: i32) -> Result<CuStream, String> {
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        let mut stream: CuStream = std::ptr::null_mut();
        // SAFETY: valid out pointer; a context is current.
        self.driver.check(
            unsafe { (self.driver.stream_create_with_priority)(&mut stream, STREAM_NON_BLOCKING, priority) },
            "cuStreamCreateWithPriority",
        )?;
        if stream.is_null() {
            return Err("cuStreamCreateWithPriority returned no stream".into());
        }
        Ok(stream)
    }

    /// The priority a stream really got.
    pub fn stream_priority(&self, stream: CuStream) -> Result<i32, String> {
        let mut p = 0;
        // SAFETY: a stream created under this context.
        self.driver.check(unsafe { (self.driver.stream_get_priority)(stream, &mut p) }, "cuStreamGetPriority")?;
        Ok(p)
    }

    /// Destroy a stream created by [`create_stream`](Self::create_stream) (errors are logged).
    pub fn destroy_stream(&self, stream: CuStream) {
        if stream.is_null() {
            return;
        }
        let Ok(_scope) = self.enter() else {
            return;
        };
        // SAFETY: a stream created under this context and not used afterwards.
        let r = unsafe { (self.driver.stream_destroy)(stream) };
        if r != CUDA_SUCCESS {
            tracing::warn!(target: "sigil::gpu", "cuStreamDestroy: {}", self.driver.describe(r));
        }
    }

    /// `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize), String> {
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out pointers; a context is current.
        self.driver.check(unsafe { (self.driver.mem_get_info)(&mut free, &mut total) }, "cuMemGetInfo")?;
        Ok((free, total))
    }

    /// Page-locked host memory of `bytes` (for fast asynchronous copies), freed by
    /// [`free_host`](Self::free_host).
    pub fn alloc_host(&self, bytes: usize) -> Result<*mut c_void, String> {
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out pointer; flags 0 = portable-less page-locked memory of this context.
        self.driver.check(unsafe { (self.driver.mem_host_alloc)(&mut p, bytes.max(1), 0) }, "cuMemHostAlloc")?;
        if p.is_null() {
            return Err("cuMemHostAlloc returned no memory".into());
        }
        Ok(p)
    }

    /// A new event of this context (`EVENT_*` flags), destroyed by
    /// [`destroy_event`](Self::destroy_event) (or owned by a [`CudaEvent`]).
    pub fn create_event(&self, flags: u32) -> Result<CuEvent, String> {
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        let mut ev: CuEvent = std::ptr::null_mut();
        // SAFETY: valid out pointer; a context is current.
        self.driver.check(unsafe { (self.driver.event_create)(&mut ev, flags) }, "cuEventCreate")?;
        if ev.is_null() {
            return Err("cuEventCreate returned no event".into());
        }
        Ok(ev)
    }

    /// Destroy an event from [`create_event`](Self::create_event) (errors are logged). The driver
    /// defers the release until the event's recorded work completes.
    pub fn destroy_event(&self, ev: CuEvent) {
        if ev.is_null() {
            return;
        }
        let Ok(_scope) = self.enter() else {
            return;
        };
        // SAFETY: an event created under this context, destroyed once.
        let r = unsafe { (self.driver.event_destroy)(ev) };
        if r != CUDA_SUCCESS {
            tracing::warn!(target: "sigil::gpu", "cuEventDestroy: {}", self.driver.describe(r));
        }
    }

    /// The scheduling flags of this context (`CTX_SCHED_*`, masked), when the driver can tell
    /// (`cuCtxGetFlags`, else the primary context's state).
    pub fn sched_flags(&self) -> Option<u32> {
        let _scope = self.enter().ok()?;
        if let Some(get) = self.driver.ctx_get_flags {
            let mut flags = 0u32;
            // SAFETY: valid out pointer; a context is current.
            if unsafe { get(&mut flags) } == CUDA_SUCCESS {
                return Some(flags & CTX_SCHED_MASK);
            }
        }
        let get_state = self.driver.primary_ctx_get_state?;
        let (mut flags, mut active) = (0u32, 0i32);
        // SAFETY: valid out pointers and device.
        (unsafe { get_state(self.device, &mut flags, &mut active) } == CUDA_SUCCESS).then_some(flags & CTX_SCHED_MASK)
    }

    /// Set this (primary) context's scheduling flags (`CTX_SCHED_*`; process-wide: every CUDA user
    /// of the device in this process, ONNX Runtime's own syncs included, waits that way from now on).
    /// `cuCtxSetFlags` (CUDA 12.1+) on the current context, else `cuDevicePrimaryCtxSetFlags`.
    pub fn set_sched_flags(&self, sched: u32) -> Result<(), String> {
        let sched = sched & CTX_SCHED_MASK;
        let _scope = self.enter().map_err(|r| format!("cuCtxPushCurrent: {}", self.driver.describe(r)))?;
        if let (Some(set), Some(get)) = (self.driver.ctx_set_flags, self.driver.ctx_get_flags) {
            let mut flags = 0u32;
            // SAFETY: valid out pointer; a context is current.
            self.driver.check(unsafe { get(&mut flags) }, "cuCtxGetFlags")?;
            // SAFETY: the current context's flags with only the scheduling bits replaced.
            return self.driver.check(unsafe { set((flags & !CTX_SCHED_MASK) | sched) }, "cuCtxSetFlags");
        }
        let set = self
            .driver
            .primary_ctx_set_flags
            .ok_or_else(|| "this NVIDIA driver cannot change a context's scheduling flags".to_string())?;
        // SAFETY: a valid device; only the scheduling bits are set.
        self.driver.check(unsafe { set(self.device, sched) }, "cuDevicePrimaryCtxSetFlags")
    }

    /// Free memory from [`alloc_host`](Self::alloc_host).
    pub fn free_host(&self, p: *mut c_void) {
        if p.is_null() {
            return;
        }
        let Ok(_scope) = self.enter() else {
            return;
        };
        // SAFETY: memory from cuMemHostAlloc under this context, freed once.
        let r = unsafe { (self.driver.mem_free_host)(p) };
        if r != CUDA_SUCCESS {
            tracing::warn!(target: "sigil::gpu", "cuMemFreeHost: {}", self.driver.describe(r));
        }
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `retain`.
        let r = unsafe { (self.driver.primary_ctx_release)(self.device) };
        if r != CUDA_SUCCESS {
            tracing::warn!(target: "sigil::gpu", "cuDevicePrimaryCtxRelease: {}", self.driver.describe(r));
        }
    }
}

/// The context of a [`PrimaryContext::enter`] is current on this thread until drop. The copy and
/// sync calls take one so they cannot run without a current context.
pub struct ContextScope<'a> {
    ctx: &'a PrimaryContext,
}

impl ContextScope<'_> {
    /// Enqueue a host → device copy of `bytes` on `stream`.
    ///
    /// # Safety
    /// `dst` must be device memory of this context with room for `bytes`; `src` must stay valid
    /// (and unmodified) until the stream is synchronised.
    #[inline]
    pub unsafe fn copy_htod(&self, dst: CuDevicePtr, src: *const c_void, bytes: usize, stream: CuStream) -> Result<(), i32> {
        // SAFETY: forwarded to the caller.
        CudaDriver::ok(unsafe { (self.ctx.driver.memcpy_htod_async)(dst, src, bytes, stream) })
    }

    /// Enqueue a device → host copy of `bytes` on `stream`.
    ///
    /// # Safety
    /// `src` must be device memory of this context holding `bytes`; `dst` must stay valid until the
    /// stream is synchronised.
    #[inline]
    pub unsafe fn copy_dtoh(&self, dst: *mut c_void, src: CuDevicePtr, bytes: usize, stream: CuStream) -> Result<(), i32> {
        // SAFETY: forwarded to the caller.
        CudaDriver::ok(unsafe { (self.ctx.driver.memcpy_dtoh_async)(dst, src, bytes, stream) })
    }

    /// Enqueue a strided device → device copy of `height` rows of `width_bytes` on `stream`:
    /// row `r` of `dst + r * dst_pitch` gets `src + r * src_pitch`.
    ///
    /// # Safety
    /// Both ranges must lie inside device allocations of this context.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_2d_dtod(
        &self,
        dst: CuDevicePtr,
        dst_pitch: usize,
        src: CuDevicePtr,
        src_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: CuStream,
    ) -> Result<(), i32> {
        let desc = Memcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: MEMORYTYPE_DEVICE,
            src_host: std::ptr::null(),
            src_device: src,
            src_array: std::ptr::null_mut(),
            src_pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: MEMORYTYPE_DEVICE,
            dst_host: std::ptr::null_mut(),
            dst_device: dst,
            dst_array: std::ptr::null_mut(),
            dst_pitch,
            width_in_bytes: width_bytes,
            height,
        };
        // SAFETY: `desc` lives across the call (the driver copies the descriptor); ranges per the
        // caller.
        CudaDriver::ok(unsafe { (self.ctx.driver.memcpy_2d_async)(&desc, stream) })
    }

    /// Enqueue a strided device → host copy of `height` rows of `width_bytes` on `stream`: row `r`
    /// of `dst + r * dst_pitch` (host memory) gets `src + r * src_pitch` (device memory).
    ///
    /// # Safety
    /// `src` must lie inside a device allocation of this context; `dst` must have room for every
    /// row and stay valid (and unread) until the stream is synchronised.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_2d_dtoh(
        &self,
        dst: *mut c_void,
        dst_pitch: usize,
        src: CuDevicePtr,
        src_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: CuStream,
    ) -> Result<(), i32> {
        let desc = Memcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: MEMORYTYPE_DEVICE,
            src_host: std::ptr::null(),
            src_device: src,
            src_array: std::ptr::null_mut(),
            src_pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: MEMORYTYPE_HOST,
            dst_host: dst,
            dst_device: 0,
            dst_array: std::ptr::null_mut(),
            dst_pitch,
            width_in_bytes: width_bytes,
            height,
        };
        // SAFETY: as in `copy_2d_dtod`; ranges per the caller.
        CudaDriver::ok(unsafe { (self.ctx.driver.memcpy_2d_async)(&desc, stream) })
    }

    /// Enqueue filling `count` 32-bit words at `dst` with `value` on `stream` (e.g. a NaN sentinel).
    ///
    /// # Safety
    /// `dst` must be 4-byte aligned device memory of this context with room for `count` words.
    #[inline]
    pub unsafe fn memset_d32(&self, dst: CuDevicePtr, value: u32, count: usize, stream: CuStream) -> Result<(), i32> {
        // SAFETY: forwarded to the caller.
        CudaDriver::ok(unsafe { (self.ctx.driver.memset_d32_async)(dst, value, count, stream) })
    }

    /// Wait for everything enqueued on `stream`.
    #[inline]
    pub fn sync(&self, stream: CuStream) -> Result<(), i32> {
        // SAFETY: a stream of this context (or null = the legacy default stream).
        CudaDriver::ok(unsafe { (self.ctx.driver.stream_synchronize)(stream) })
    }

    /// Enqueue a record of `ev` on `stream` (it completes when everything enqueued before it on
    /// `stream` has). Never allocates.
    #[inline]
    pub fn record_event(&self, ev: CuEvent, stream: CuStream) -> Result<(), i32> {
        // SAFETY: an event and a stream of this context.
        CudaDriver::ok(unsafe { (self.ctx.driver.event_record)(ev, stream) })
    }

    /// Whether `ev`'s recorded work has completed (an event never recorded counts as complete).
    /// Never allocates or blocks.
    #[inline]
    pub fn query_event(&self, ev: CuEvent) -> Result<bool, i32> {
        // SAFETY: an event of this context.
        match unsafe { (self.ctx.driver.event_query)(ev) } {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            r => Err(r),
        }
    }

    /// Wait for `ev`'s recorded work: blocks the thread for an `EVENT_BLOCKING_SYNC` event, else
    /// waits per the context's scheduling flags. Never allocates.
    #[inline]
    pub fn sync_event(&self, ev: CuEvent) -> Result<(), i32> {
        // SAFETY: an event of this context.
        CudaDriver::ok(unsafe { (self.ctx.driver.event_synchronize)(ev) })
    }

    /// Make `stream`'s later work wait for `ev`'s recorded work (device-side; the host does not
    /// wait). Never allocates.
    #[inline]
    pub fn stream_wait_event(&self, stream: CuStream, ev: CuEvent) -> Result<(), i32> {
        // SAFETY: a stream and an event of this context; flags 0 (CU_EVENT_WAIT_DEFAULT).
        CudaDriver::ok(unsafe { (self.ctx.driver.stream_wait_event)(stream, ev, 0) })
    }

    /// Milliseconds between two completed timing events (`EVENT_DEFAULT`). Never allocates.
    #[inline]
    pub fn elapsed_ms(&self, start: CuEvent, end: CuEvent) -> Result<f32, i32> {
        let mut ms = 0f32;
        // SAFETY: two timing events of this context, both recorded and completed.
        CudaDriver::ok(unsafe { (self.ctx.driver.event_elapsed_time)(&mut ms, start, end) })?;
        Ok(ms)
    }

    /// Start recording `stream` into a graph ([`CAPTURE_MODE_THREAD_LOCAL`]).
    pub fn begin_capture(&self, stream: CuStream) -> Result<(), String> {
        // SAFETY: a stream of this context that is not capturing.
        let r = unsafe { (self.ctx.driver.stream_begin_capture)(stream, CAPTURE_MODE_THREAD_LOCAL) };
        self.ctx.driver.check(r, "cuStreamBeginCapture")
    }

    /// Stop recording `stream`; the recorded graph, or why the capture was invalidated (the stream is
    /// back to normal either way).
    pub fn end_capture(&self, stream: CuStream) -> Result<CuGraph, String> {
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: a stream this thread is capturing (or whose capture was invalidated).
        let r = unsafe { (self.ctx.driver.stream_end_capture)(stream, &mut graph) };
        if r != CUDA_SUCCESS || graph.is_null() {
            if !graph.is_null() {
                // SAFETY: a graph returned by the driver, destroyed once.
                unsafe { (self.ctx.driver.graph_destroy)(graph) };
            }
            return Err(format!("cuStreamEndCapture: {}", self.ctx.driver.describe(r)));
        }
        Ok(graph)
    }

    /// Whether `stream` is capturing (a capture left open by a failed run).
    pub fn is_capturing(&self, stream: CuStream) -> bool {
        let mut status = 0u32;
        // SAFETY: a stream of this context; valid out pointer.
        let r = unsafe { (self.ctx.driver.stream_is_capturing)(stream, &mut status) };
        r == CUDA_SUCCESS && status != 0
    }

    /// Instantiate a recorded graph (the graph itself is destroyed; the executable stays).
    pub fn instantiate(&self, graph: CuGraph) -> Result<CuGraphExec, String> {
        let mut exec: CuGraphExec = std::ptr::null_mut();
        // SAFETY: a graph from `end_capture`, consumed here.
        let r = unsafe { (self.ctx.driver.graph_instantiate)(&mut exec, graph, 0) };
        // SAFETY: the executable holds its own copy; the recorded graph is not used again.
        unsafe { (self.ctx.driver.graph_destroy)(graph) };
        if r != CUDA_SUCCESS || exec.is_null() {
            return Err(format!("cuGraphInstantiate: {}", self.ctx.driver.describe(r)));
        }
        Ok(exec)
    }

    /// Enqueue one replay of `exec` on `stream`. Never allocates.
    #[inline]
    pub fn launch(&self, exec: CuGraphExec, stream: CuStream) -> Result<(), i32> {
        // SAFETY: an executable graph of this context.
        CudaDriver::ok(unsafe { (self.ctx.driver.graph_launch)(exec, stream) })
    }

    /// Destroy a recorded graph that is not instantiated.
    pub fn destroy_graph(&self, graph: CuGraph) {
        if !graph.is_null() {
            // SAFETY: a graph from `end_capture`, destroyed once.
            let _ = unsafe { (self.ctx.driver.graph_destroy)(graph) };
        }
    }

    /// Destroy an executable graph.
    pub fn destroy_exec(&self, exec: CuGraphExec) {
        if !exec.is_null() {
            // SAFETY: an executable from `instantiate`, destroyed once.
            let _ = unsafe { (self.ctx.driver.graph_exec_destroy)(exec) };
        }
    }
}

/// An event of a retained primary context (destroyed on drop).
pub struct CudaEvent {
    ev: CuEvent,
    ctx: std::sync::Arc<PrimaryContext>,
}

// SAFETY: an event handle of a context usable from any thread (the owner serialises use).
unsafe impl Send for CudaEvent {}

impl CudaEvent {
    pub fn new(ctx: &std::sync::Arc<PrimaryContext>, flags: u32) -> Result<CudaEvent, String> {
        Ok(CudaEvent { ev: ctx.create_event(flags)?, ctx: std::sync::Arc::clone(ctx) })
    }

    /// The raw handle (valid while `self` lives).
    #[inline]
    pub fn raw(&self) -> CuEvent {
        self.ev
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx.destroy_event(self.ev);
    }
}

impl Drop for ContextScope<'_> {
    fn drop(&mut self) {
        let mut popped: CuContext = std::ptr::null_mut();
        // SAFETY: balances the push in `PrimaryContext::enter` on this thread.
        let _ = unsafe { (self.ctx.driver.ctx_pop)(&mut popped) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memcpy2d_matches_the_cuda_header_layout() {
        // `CUDA_MEMCPY2D_v2` on 64-bit: 16 eight-byte slots (the two enums are padded).
        assert_eq!(std::mem::size_of::<Memcpy2D>(), 128);
        assert_eq!(std::mem::offset_of!(Memcpy2D, src_host), 24);
        assert_eq!(std::mem::offset_of!(Memcpy2D, dst_memory_type), 72);
        assert_eq!(std::mem::offset_of!(Memcpy2D, width_in_bytes), 112);
    }

    /// Runs on any machine: without an NVIDIA driver the loader reports why; with one, the driver
    /// lists at least one device and device ordinals out of range are clean errors.
    #[test]
    fn driver_loads_or_explains_why_not() {
        match CudaDriver::get() {
            Err(why) => {
                eprintln!("no CUDA driver: {why}");
                assert!(!why.is_empty());
            }
            Ok(d) => {
                eprintln!("CUDA driver {} with {} device(s)", d.driver_version(), d.device_count());
                assert!(d.driver_version() > 0);
                assert!(d.compute_capability(d.device_count()).unwrap_err().contains("does not exist"));
                assert!(PrimaryContext::retain(-1).is_err());
                if d.device_count() > 0 {
                    let (major, _) = d.compute_capability(0).unwrap();
                    assert!(major >= 3, "{major}");
                    assert!(!d.device_name(0).unwrap().is_empty());
                    let ctx = PrimaryContext::retain(0).unwrap();
                    let (least, greatest) = ctx.stream_priority_range().unwrap();
                    assert!(greatest <= least, "{least} {greatest}");
                    let (free, total) = ctx.mem_info().unwrap();
                    assert!(free <= total && total > 0);
                    assert!(d.describe(2).contains("OUT_OF_MEMORY"), "{}", d.describe(2));
                }
            }
        }
    }
}
