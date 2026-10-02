//! Hand-written minimal OpenVR FFI over a lazily loaded `openvr_api.dll` (the `runtime/openvr` pack,
//! OpenVR SDK v2.15.6). Nothing is load-time linked, so Sigil starts on PCs without SteamVR.
//!
//! - Exports used (cdecl): `VR_InitInternal2`, `VR_ShutdownInternal`, `VR_GetGenericInterface`,
//!   `VR_IsInterfaceVersionValid`, `VR_IsRuntimeInstalled`, `VR_GetVRInitErrorAsEnglishDescription`.
//! - Interfaces are taken as C function tables (`"FnTable:IVRSystem_026"`,
//!   `"FnTable:IVROverlay_028"`); the entry indexes below are those of `headers/openvr_capi.h` at
//!   SDK v2.15.6 (commit 0924064316de3effbcd1acf1e309182a2deb1c05). An older SteamVR that lacks
//!   these interface versions is reported as too old, never called.
//! - Sigil is a `VRApplication_Background` app: OpenVR refuses to start for it while SteamVR is not
//!   running (`VRInitError_Init_NoServerForBackgroundApp`) instead of launching SteamVR, and Sigil
//!   only even tries while SteamVR's `vrserver.exe` is running ([`process_running`]). The overlay
//!   disconnects on `VREvent_Quit` so SteamVR can exit (and never has to kill Sigil).
//! - The library stays loaded once loaded (unloading it while SteamVR's client DLL may still hold
//!   threads is not safe); a session always ends with `VR_ShutdownInternal`.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::placement::Mat34;
use super::render::Texture;
use super::worker::{ConnectError, PollOutcome, VrBackend, VrSession};

/// File of the runtime pack.
pub(crate) const DLL_NAME: &str = "openvr_api.dll";
/// Interface versions the tables below describe.
pub(crate) const SYSTEM_VERSION: &str = "IVRSystem_026";
pub(crate) const OVERLAY_VERSION: &str = "IVROverlay_028";
const FN_TABLE_PREFIX: &str = "FnTable:";
/// Process SteamVR's clients connect to.
pub(crate) const STEAMVR_SERVER_EXE: &str = "vrserver.exe";
/// Overlay key (unique per SteamVR session; another Sigil instance gets a pid-suffixed key).
const OVERLAY_KEY: &str = "sigil.captions";
const OVERLAY_NAME: &str = "Sigil captions";

/// `EVRApplicationType::VRApplication_Background`.
const APPLICATION_BACKGROUND: i32 = 3;
/// `ETrackingUniverseOrigin::TrackingUniverseStanding`.
const UNIVERSE_STANDING: i32 = 1;
/// `k_unTrackedDeviceIndex_Hmd`.
pub(crate) const HMD_DEVICE: u32 = 0;
/// `k_unTrackedDeviceIndexInvalid`.
const INVALID_DEVICE: u32 = u32::MAX;
/// `ETrackedControllerRole`.
const ROLE_LEFT_HAND: i32 = 1;
const ROLE_RIGHT_HAND: i32 = 2;
/// `EVROverlayError::VROverlayError_KeyInUse`.
const OVERLAY_ERROR_KEY_IN_USE: i32 = 17;

/// `EVRInitError` values Sigil tells apart.
pub(crate) mod init_error {
    pub const NONE: i32 = 0;
    pub const INSTALLATION_NOT_FOUND: i32 = 100;
    pub const INSTALLATION_CORRUPT: i32 = 101;
    pub const VR_CLIENT_DLL_NOT_FOUND: i32 = 102;
    pub const FILE_NOT_FOUND: i32 = 103;
    pub const INTERFACE_NOT_FOUND: i32 = 105;
    pub const INVALID_INTERFACE: i32 = 106;
    pub const NOT_INITIALIZED: i32 = 109;
    pub const PATH_REGISTRY_NOT_FOUND: i32 = 110;
    pub const RETRY: i32 = 115;
    pub const ANOTHER_APP_LAUNCHING: i32 = 117;
    pub const SHUTTING_DOWN: i32 = 119;
    pub const NO_SERVER_FOR_BACKGROUND_APP: i32 = 121;
}

/// `EVREventType` values Sigil reacts to.
pub(crate) mod event {
    pub const TRACKED_DEVICE_ACTIVATED: u32 = 100;
    pub const TRACKED_DEVICE_DEACTIVATED: u32 = 101;
    pub const TRACKED_DEVICE_ROLE_CHANGED: u32 = 108;
    pub const QUIT: u32 = 700;
    pub const DRIVER_REQUESTED_QUIT: u32 = 704;
    pub const RESTART_REQUESTED: u32 = 705;
}

/// Entry indexes of `VR_IVRSystem_FnTable` (IVRSystem_026).
mod system_index {
    pub const GET_DEVICE_TO_ABSOLUTE_TRACKING_POSE: usize = 12;
    pub const GET_TRACKED_DEVICE_INDEX_FOR_CONTROLLER_ROLE: usize = 18;
    pub const POLL_NEXT_EVENT: usize = 30;
}

/// Entry indexes of `VR_IVROverlay_FnTable` (IVROverlay_028).
mod overlay_index {
    pub const CREATE_OVERLAY: usize = 1;
    pub const DESTROY_OVERLAY: usize = 3;
    pub const GET_OVERLAY_ERROR_NAME_FROM_ENUM: usize = 8;
    pub const SET_OVERLAY_ALPHA: usize = 16;
    pub const SET_OVERLAY_WIDTH_IN_METERS: usize = 22;
    pub const SET_OVERLAY_TRANSFORM_ABSOLUTE: usize = 33;
    pub const SET_OVERLAY_TRANSFORM_TRACKED_DEVICE_RELATIVE: usize = 35;
    pub const SHOW_OVERLAY: usize = 43;
    pub const HIDE_OVERLAY: usize = 44;
    pub const POLL_NEXT_OVERLAY_EVENT: usize = 48;
    pub const SET_OVERLAY_RAW: usize = 62;
}

/// `VREvent_t` (Windows packing: three 32-bit fields, 4 bytes padding, a 48-byte union aligned to 8).
/// Fields Sigil does not read are written by OpenVR and keep the layout.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct VrEvent {
    pub event_type: u32,
    pub tracked_device_index: u32,
    pub event_age_seconds: f32,
    pub data: [u64; 6],
}
const _: () = assert!(std::mem::size_of::<VrEvent>() == 64);

/// `TrackedDevicePose_t` (fields Sigil does not read keep the layout).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct TrackedDevicePose {
    pub device_to_absolute_tracking: Mat34,
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub tracking_result: i32,
    pub pose_is_valid: bool,
    pub device_is_connected: bool,
}
const _: () = assert!(std::mem::size_of::<TrackedDevicePose>() == 80);

type InitInternal2Fn = unsafe extern "C" fn(*mut i32, i32, *const c_char) -> u32;
type ShutdownInternalFn = unsafe extern "C" fn();
type GetGenericInterfaceFn = unsafe extern "C" fn(*const c_char, *mut i32) -> isize;
type IsInterfaceVersionValidFn = unsafe extern "C" fn(*const c_char) -> bool;
type IsRuntimeInstalledFn = unsafe extern "C" fn() -> bool;
type GetRuntimePathFn = unsafe extern "C" fn(*mut c_char, u32, *mut u32) -> bool;
type InitErrorDescriptionFn = unsafe extern "C" fn(i32) -> *const c_char;

type PollNextEventFn = unsafe extern "system" fn(*mut VrEvent, u32) -> bool;
type GetDeviceToAbsoluteTrackingPoseFn = unsafe extern "system" fn(i32, f32, *mut TrackedDevicePose, u32);
type GetTrackedDeviceIndexForControllerRoleFn = unsafe extern "system" fn(i32) -> u32;

type CreateOverlayFn = unsafe extern "system" fn(*const c_char, *const c_char, *mut u64) -> i32;
type HandleFn = unsafe extern "system" fn(u64) -> i32;
type ErrorNameFn = unsafe extern "system" fn(i32) -> *const c_char;
type SetFloatFn = unsafe extern "system" fn(u64, f32) -> i32;
type SetTransformAbsoluteFn = unsafe extern "system" fn(u64, i32, *const Mat34) -> i32;
type SetTransformRelativeFn = unsafe extern "system" fn(u64, u32, *const Mat34) -> i32;
type PollNextOverlayEventFn = unsafe extern "system" fn(u64, *mut VrEvent, u32) -> bool;
type SetOverlayRawFn = unsafe extern "system" fn(u64, *mut c_void, u32, u32, u32) -> i32;

struct Exports {
    init: InitInternal2Fn,
    shutdown: ShutdownInternalFn,
    get_interface: GetGenericInterfaceFn,
    interface_valid: IsInterfaceVersionValidFn,
    runtime_installed: IsRuntimeInstalledFn,
    /// `VR_GetRuntimePath` (OpenVR 1.6+; optional).
    runtime_path: Option<GetRuntimePathFn>,
    init_error_description: InitErrorDescriptionFn,
}

/// The loaded `openvr_api.dll` (kept for the process lifetime).
pub(crate) struct OpenVrLibrary {
    _lib: libloading::Library,
    f: Exports,
    path: PathBuf,
}

fn symbol<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller names an export of openvr_api.dll with the matching signature `T`.
    unsafe { lib.get::<T>(name) }.map(|s| *s).map_err(|e| {
        format!(
            "{DLL_NAME} has no {}: {e}",
            String::from_utf8_lossy(name.strip_suffix(b"\0").unwrap_or(name))
        )
    })
}

/// Copy a C string owned by OpenVR (static error names), or "" for null.
fn c_text(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: OpenVR returns NUL-terminated static strings for these enums.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl OpenVrLibrary {
    fn load(dir: &Path) -> Result<Self, String> {
        let lib = crate::services::native_runtime::load_library_in_dir(dir, DLL_NAME)?;
        let f = Exports {
            init: symbol(&lib, b"VR_InitInternal2\0")?,
            shutdown: symbol(&lib, b"VR_ShutdownInternal\0")?,
            get_interface: symbol(&lib, b"VR_GetGenericInterface\0")?,
            interface_valid: symbol(&lib, b"VR_IsInterfaceVersionValid\0")?,
            runtime_installed: symbol(&lib, b"VR_IsRuntimeInstalled\0")?,
            runtime_path: symbol(&lib, b"VR_GetRuntimePath\0").ok(),
            init_error_description: symbol(&lib, b"VR_GetVRInitErrorAsEnglishDescription\0")?,
        };
        Ok(Self {
            _lib: lib,
            f,
            path: dir.join(DLL_NAME),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// SteamVR is installed: `VR_IsRuntimeInstalled` (its path registry, `openvrpaths.vrpath`) and,
    /// where the runtime path can be read, SteamVR's client DLL is in it (an uninstalled SteamVR
    /// can leave the registry and a few folders behind). Reads files only; starts nothing.
    pub(crate) fn steamvr_installed(&self) -> bool {
        // SAFETY: no arguments; reads OpenVR's path registry.
        if !unsafe { (self.f.runtime_installed)() } {
            return false;
        }
        match self.runtime_path() {
            Some(root) => ["bin/vrclient_x64.dll", "bin/win64/vrclient_x64.dll", "bin/vrclient.dll"]
                .iter()
                .any(|rel| root.join(rel).is_file()),
            None => true,
        }
    }

    /// `VR_GetRuntimePath`: the SteamVR folder the path registry names.
    fn runtime_path(&self) -> Option<PathBuf> {
        let f = self.f.runtime_path?;
        let mut size = 1024u32;
        for _ in 0..2 {
            let mut buf = vec![0u8; size as usize];
            let mut required = 0u32;
            // SAFETY: a writable buffer of `size` bytes and a valid out pointer.
            let ok = unsafe { f(buf.as_mut_ptr().cast::<c_char>(), size, &mut required) };
            if ok {
                let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                let text = String::from_utf8_lossy(&buf[..end]).into_owned();
                return (!text.is_empty()).then(|| PathBuf::from(text));
            }
            if required <= size || required > 32 * 1024 {
                return None;
            }
            size = required;
        }
        None
    }

    fn describe(&self, error: i32) -> String {
        // SAFETY: returns a static string for every EVRInitError value.
        let text = c_text(unsafe { (self.f.init_error_description)(error) });
        if text.is_empty() {
            format!("OpenVR error {error}")
        } else {
            format!("{text} (OpenVR error {error})")
        }
    }

    fn interface_valid(&self, version: &str) -> bool {
        let Ok(v) = CString::new(version) else { return false };
        // SAFETY: NUL-terminated version string.
        unsafe { (self.f.interface_valid)(v.as_ptr()) }
    }

    /// The function table of `version`, or null.
    fn fn_table(&self, version: &str) -> Result<*const usize, i32> {
        let v = CString::new(format!("{FN_TABLE_PREFIX}{version}")).map_err(|_| init_error::INVALID_INTERFACE)?;
        let mut error = 0i32;
        // SAFETY: NUL-terminated name and a valid out pointer.
        let table = unsafe { (self.f.get_interface)(v.as_ptr(), &mut error) };
        if error != init_error::NONE || table == 0 {
            return Err(if error == init_error::NONE {
                init_error::INTERFACE_NOT_FOUND
            } else {
                error
            });
        }
        Ok(table as *const usize)
    }

    fn shutdown(&self) {
        // SAFETY: balances a successful VR_InitInternal2 (idempotent inside OpenVR).
        unsafe { (self.f.shutdown)() }
    }
}

/// The process-wide library, loading it from `dir` on first use.
fn library(dir: &Path) -> Result<Arc<OpenVrLibrary>, String> {
    static LIBRARY: Mutex<Option<Arc<OpenVrLibrary>>> = Mutex::new(None);
    let mut guard = LIBRARY.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(lib) = guard.as_ref() {
        return Ok(lib.clone());
    }
    let lib = Arc::new(OpenVrLibrary::load(dir)?);
    tracing::info!(path = %lib.path().display(), "vr_overlay: loaded the OpenVR runtime");
    *guard = Some(lib.clone());
    Ok(lib)
}

/// Read entry `index` of an OpenVR function table as a function pointer of type `F`.
///
/// # Safety
/// `table` must be the table of the interface version whose entry `index` has signature `F`.
unsafe fn entry<F: Copy>(table: *const usize, index: usize, name: &str) -> Result<F, String> {
    debug_assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    // SAFETY: per the contract, `table` has more than `index` pointer-sized entries.
    let raw = unsafe { *table.add(index) };
    if raw == 0 {
        return Err(format!("SteamVR has no {name}"));
    }
    // SAFETY: a non-null function pointer of signature `F` (same size as usize).
    Ok(unsafe { std::mem::transmute_copy::<usize, F>(&raw) })
}

struct SystemFns {
    poll_next_event: PollNextEventFn,
    device_pose: GetDeviceToAbsoluteTrackingPoseFn,
    controller_index: GetTrackedDeviceIndexForControllerRoleFn,
}

struct OverlayFns {
    create: CreateOverlayFn,
    destroy: HandleFn,
    error_name: ErrorNameFn,
    set_alpha: SetFloatFn,
    set_width: SetFloatFn,
    set_transform_absolute: SetTransformAbsoluteFn,
    set_transform_relative: SetTransformRelativeFn,
    show: HandleFn,
    hide: HandleFn,
    poll_overlay_event: PollNextOverlayEventFn,
    set_raw: SetOverlayRawFn,
}

impl SystemFns {
    /// # Safety
    /// `t` is the IVRSystem_026 function table.
    unsafe fn read(t: *const usize) -> Result<Self, String> {
        use system_index as i;
        // SAFETY: per the contract.
        unsafe {
            Ok(Self {
                poll_next_event: entry(t, i::POLL_NEXT_EVENT, "PollNextEvent")?,
                device_pose: entry(t, i::GET_DEVICE_TO_ABSOLUTE_TRACKING_POSE, "GetDeviceToAbsoluteTrackingPose")?,
                controller_index: entry(
                    t,
                    i::GET_TRACKED_DEVICE_INDEX_FOR_CONTROLLER_ROLE,
                    "GetTrackedDeviceIndexForControllerRole",
                )?,
            })
        }
    }
}

impl OverlayFns {
    /// # Safety
    /// `t` is the IVROverlay_028 function table.
    unsafe fn read(t: *const usize) -> Result<Self, String> {
        use overlay_index as i;
        // SAFETY: per the contract.
        unsafe {
            Ok(Self {
                create: entry(t, i::CREATE_OVERLAY, "CreateOverlay")?,
                destroy: entry(t, i::DESTROY_OVERLAY, "DestroyOverlay")?,
                error_name: entry(t, i::GET_OVERLAY_ERROR_NAME_FROM_ENUM, "GetOverlayErrorNameFromEnum")?,
                set_alpha: entry(t, i::SET_OVERLAY_ALPHA, "SetOverlayAlpha")?,
                set_width: entry(t, i::SET_OVERLAY_WIDTH_IN_METERS, "SetOverlayWidthInMeters")?,
                set_transform_absolute: entry(t, i::SET_OVERLAY_TRANSFORM_ABSOLUTE, "SetOverlayTransformAbsolute")?,
                set_transform_relative: entry(
                    t,
                    i::SET_OVERLAY_TRANSFORM_TRACKED_DEVICE_RELATIVE,
                    "SetOverlayTransformTrackedDeviceRelative",
                )?,
                show: entry(t, i::SHOW_OVERLAY, "ShowOverlay")?,
                hide: entry(t, i::HIDE_OVERLAY, "HideOverlay")?,
                poll_overlay_event: entry(t, i::POLL_NEXT_OVERLAY_EVENT, "PollNextOverlayEvent")?,
                set_raw: entry(t, i::SET_OVERLAY_RAW, "SetOverlayRaw")?,
            })
        }
    }
}

/// Only one OpenVR session per process (a stopped worker that has not finished closing blocks the
/// next connect until it has).
static SESSION_OPEN: AtomicBool = AtomicBool::new(false);

/// Calls `VR_ShutdownInternal` when dropped unless disarmed (connect failures after a successful
/// init).
struct InitGuard {
    lib: Option<Arc<OpenVrLibrary>>,
}

impl InitGuard {
    fn disarm(mut self) {
        self.lib = None;
    }
}

impl Drop for InitGuard {
    fn drop(&mut self) {
        if let Some(lib) = self.lib.take() {
            lib.shutdown();
            SESSION_OPEN.store(false, Ordering::SeqCst);
        }
    }
}

/// How an init error is reported.
pub(crate) fn classify_init_error(error: i32, description: String) -> ConnectError {
    use init_error as e;
    match error {
        e::NO_SERVER_FOR_BACKGROUND_APP | e::SHUTTING_DOWN | e::NOT_INITIALIZED | e::RETRY | e::ANOTHER_APP_LAUNCHING => ConnectError::NotRunning,
        e::INSTALLATION_NOT_FOUND | e::INSTALLATION_CORRUPT | e::VR_CLIENT_DLL_NOT_FOUND | e::FILE_NOT_FOUND | e::PATH_REGISTRY_NOT_FOUND => {
            ConnectError::NotInstalled
        }
        e::INTERFACE_NOT_FOUND | e::INVALID_INTERFACE => ConnectError::TooOld(description),
        _ => ConnectError::Failed(description),
    }
}

/// How long a "SteamVR installed?" answer is reused while waiting.
const INSTALLED_CACHE: std::time::Duration = std::time::Duration::from_secs(60);

/// The real backend: `runtime/openvr` (or `SIGIL_OPENVR_DIR` for development).
pub(crate) struct OpenVrBackend {
    locate: Box<dyn Fn() -> Option<PathBuf> + Send>,
    installed: Option<(std::time::Instant, bool)>,
}

impl OpenVrBackend {
    pub(crate) fn new(locate: Box<dyn Fn() -> Option<PathBuf> + Send>) -> Self {
        Self { locate, installed: None }
    }

    fn dir(&self) -> Option<PathBuf> {
        (self.locate)().filter(|d| d.join(DLL_NAME).is_file())
    }
}

impl VrBackend for OpenVrBackend {
    fn runtime_installed(&mut self) -> bool {
        self.dir().is_some()
    }

    fn server_running(&mut self) -> Option<bool> {
        process_running(STEAMVR_SERVER_EXE)
    }

    fn steamvr_installed(&mut self) -> Option<bool> {
        if let Some((at, installed)) = self.installed {
            if at.elapsed() < INSTALLED_CACHE {
                return Some(installed);
            }
        }
        let dir = self.dir()?;
        let installed = library(&dir).ok()?.steamvr_installed();
        self.installed = Some((std::time::Instant::now(), installed));
        Some(installed)
    }

    fn connect(&mut self) -> Result<Box<dyn VrSession>, ConnectError> {
        let dir = self
            .dir()
            .ok_or_else(|| ConnectError::Failed("the OpenVR runtime is not installed".into()))?;
        let lib = library(&dir).map_err(ConnectError::Failed)?;
        if SESSION_OPEN.swap(true, Ordering::SeqCst) {
            return Err(ConnectError::Failed("the previous SteamVR session is still closing".into()));
        }
        let mut error = init_error::NONE;
        // SAFETY: valid out pointer; Background never launches SteamVR; null startup info.
        unsafe { (lib.f.init)(&mut error, APPLICATION_BACKGROUND, std::ptr::null()) };
        if error != init_error::NONE {
            // OpenVR already unloaded its client on failure: nothing to shut down.
            SESSION_OPEN.store(false, Ordering::SeqCst);
            return Err(classify_init_error(error, lib.describe(error)));
        }
        let guard = InitGuard { lib: Some(lib.clone()) };

        for version in [SYSTEM_VERSION, OVERLAY_VERSION] {
            if !lib.interface_valid(version) {
                return Err(ConnectError::TooOld(format!("SteamVR does not offer {version}")));
            }
        }
        let tables = lib
            .fn_table(SYSTEM_VERSION)
            .and_then(|s| lib.fn_table(OVERLAY_VERSION).map(|o| (s, o)));
        let (sys_table, ovl_table) = tables.map_err(|e| classify_init_error(e, lib.describe(e)))?;
        // SAFETY: the tables of exactly these interface versions.
        let sys = unsafe { SystemFns::read(sys_table) }.map_err(ConnectError::TooOld)?;
        // SAFETY: as above.
        let ovl = unsafe { OverlayFns::read(ovl_table) }.map_err(ConnectError::TooOld)?;

        let name = CString::new(OVERLAY_NAME).expect("static name");
        let mut handle = 0u64;
        let mut result = OVERLAY_ERROR_KEY_IN_USE;
        for key in [OVERLAY_KEY.to_string(), format!("{OVERLAY_KEY}.{}", std::process::id())] {
            let key = CString::new(key).expect("ASCII key");
            // SAFETY: NUL-terminated key and name, valid out pointer.
            result = unsafe { (ovl.create)(key.as_ptr(), name.as_ptr(), &mut handle) };
            if result != OVERLAY_ERROR_KEY_IN_USE {
                break;
            }
        }
        if result != 0 || handle == 0 {
            // SAFETY: static name for an EVROverlayError value.
            let what = c_text(unsafe { (ovl.error_name)(result) });
            return Err(ConnectError::Failed(format!("SteamVR refused the caption overlay: {what} ({result})")));
        }
        guard.disarm();
        tracing::info!("vr_overlay: connected to SteamVR");
        Ok(Box::new(OpenVrSession {
            lib,
            sys,
            ovl,
            handle,
            closed: false,
        }))
    }
}

/// A live connection with Sigil's overlay created.
struct OpenVrSession {
    lib: Arc<OpenVrLibrary>,
    sys: SystemFns,
    ovl: OverlayFns,
    handle: u64,
    closed: bool,
}

impl OpenVrSession {
    fn check(&self, what: &str, result: i32) -> Result<(), String> {
        if result == 0 {
            return Ok(());
        }
        // SAFETY: static name for an EVROverlayError value.
        let name = c_text(unsafe { (self.ovl.error_name)(result) });
        Err(format!("{what} failed: {name} ({result})"))
    }
}

impl VrSession for OpenVrSession {
    fn poll(&mut self) -> PollOutcome {
        let mut out = PollOutcome::default();
        let mut ev = VrEvent {
            event_type: 0,
            tracked_device_index: 0,
            event_age_seconds: 0.0,
            data: [0; 6],
        };
        let size = std::mem::size_of::<VrEvent>() as u32;
        // Bounded per tick; the rest waits for the next one.
        for _ in 0..64 {
            // SAFETY: valid event buffer of the size passed.
            if !unsafe { (self.sys.poll_next_event)(&mut ev, size) } {
                break;
            }
            match ev.event_type {
                event::QUIT | event::DRIVER_REQUESTED_QUIT | event::RESTART_REQUESTED => out.quit = true,
                event::TRACKED_DEVICE_ACTIVATED | event::TRACKED_DEVICE_DEACTIVATED | event::TRACKED_DEVICE_ROLE_CHANGED => {
                    out.devices_changed = true
                }
                _ => {}
            }
        }
        // The overlay takes no input; its queue is drained so it never fills up.
        for _ in 0..64 {
            // SAFETY: as above, for this overlay's handle.
            if !unsafe { (self.ovl.poll_overlay_event)(self.handle, &mut ev, size) } {
                break;
            }
        }
        out
    }

    fn set_width(&mut self, metres: f32) -> Result<(), String> {
        // SAFETY: valid handle.
        self.check("SetOverlayWidthInMeters", unsafe { (self.ovl.set_width)(self.handle, metres) })
    }

    fn set_alpha(&mut self, alpha: f32) -> Result<(), String> {
        // SAFETY: valid handle.
        self.check("SetOverlayAlpha", unsafe { (self.ovl.set_alpha)(self.handle, alpha) })
    }

    fn set_texture(&mut self, texture: &Texture) -> Result<(), String> {
        let expected = texture.width as usize * texture.height as usize * 4;
        if texture.rgba.len() != expected {
            return Err(format!("texture is {} bytes, expected {expected}", texture.rgba.len()));
        }
        // SAFETY: `width * height * 4` bytes of RGBA that OpenVR copies and does not write.
        let result = unsafe { (self.ovl.set_raw)(self.handle, texture.rgba.as_ptr() as *mut c_void, texture.width, texture.height, 4) };
        self.check("SetOverlayRaw", result)
    }

    fn set_visible(&mut self, visible: bool) -> Result<(), String> {
        // SAFETY: valid handle.
        if visible {
            self.check("ShowOverlay", unsafe { (self.ovl.show)(self.handle) })
        } else {
            self.check("HideOverlay", unsafe { (self.ovl.hide)(self.handle) })
        }
    }

    fn place_relative(&mut self, device: u32, transform: &Mat34) -> Result<(), String> {
        // SAFETY: valid handle and matrix.
        let r = unsafe { (self.ovl.set_transform_relative)(self.handle, device, transform) };
        self.check("SetOverlayTransformTrackedDeviceRelative", r)
    }

    fn place_absolute(&mut self, transform: &Mat34) -> Result<(), String> {
        // SAFETY: valid handle and matrix.
        let r = unsafe { (self.ovl.set_transform_absolute)(self.handle, UNIVERSE_STANDING, transform) };
        self.check("SetOverlayTransformAbsolute", r)
    }

    fn hmd_pose(&mut self) -> Option<Mat34> {
        let mut pose = TrackedDevicePose {
            device_to_absolute_tracking: [[0.0; 4]; 3],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            tracking_result: 0,
            pose_is_valid: false,
            device_is_connected: false,
        };
        // SAFETY: one pose for the HMD (device index 0).
        unsafe { (self.sys.device_pose)(UNIVERSE_STANDING, 0.0, &mut pose, 1) };
        let m = pose.device_to_absolute_tracking;
        (pose.pose_is_valid && m.iter().flatten().all(|v| v.is_finite())).then_some(m)
    }

    fn controller(&mut self, left: bool) -> Option<u32> {
        let role = if left { ROLE_LEFT_HAND } else { ROLE_RIGHT_HAND };
        // SAFETY: plain value call.
        let index = unsafe { (self.sys.controller_index)(role) };
        (index != INVALID_DEVICE && index != HMD_DEVICE).then_some(index)
    }

    fn close(mut self: Box<Self>) {
        self.close_now();
    }
}

impl OpenVrSession {
    fn close_now(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        // SAFETY: our overlay; then the matching shutdown for the session's init.
        unsafe { (self.ovl.destroy)(self.handle) };
        self.lib.shutdown();
        SESSION_OPEN.store(false, Ordering::SeqCst);
        tracing::info!("vr_overlay: disconnected from SteamVR");
    }
}

impl Drop for OpenVrSession {
    fn drop(&mut self) {
        self.close_now();
    }
}

/// Whether a process with this executable name runs (`None` when processes cannot be listed).
#[cfg(windows)]
pub(crate) fn process_running(exe: &str) -> Option<bool> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    // SAFETY: snapshot handle closed below; the entry carries its own size.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).ok()?;
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut found = false;
        let mut more = Process32FirstW(snapshot, &mut entry).is_ok();
        while more {
            let len = entry
                .szExeFile
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(entry.szExeFile.len());
            if String::from_utf16_lossy(&entry.szExeFile[..len]).eq_ignore_ascii_case(exe) {
                found = true;
                break;
            }
            more = Process32NextW(snapshot, &mut entry).is_ok();
        }
        let _ = CloseHandle(snapshot);
        Some(found)
    }
}

#[cfg(not(windows))]
pub(crate) fn process_running(_exe: &str) -> Option<bool> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_windows_sdk() {
        assert_eq!(std::mem::size_of::<VrEvent>(), 64);
        assert_eq!(std::mem::offset_of!(VrEvent, data), 16);
        assert_eq!(std::mem::size_of::<TrackedDevicePose>(), 80);
        assert_eq!(std::mem::offset_of!(TrackedDevicePose, tracking_result), 72);
        assert_eq!(std::mem::offset_of!(TrackedDevicePose, pose_is_valid), 76);
    }

    #[test]
    fn init_errors_are_classified() {
        assert_eq!(classify_init_error(121, String::new()), ConnectError::NotRunning);
        assert_eq!(classify_init_error(119, String::new()), ConnectError::NotRunning);
        assert_eq!(classify_init_error(100, String::new()), ConnectError::NotInstalled);
        assert_eq!(classify_init_error(110, String::new()), ConnectError::NotInstalled);
        assert_eq!(classify_init_error(105, "x".into()), ConnectError::TooOld("x".into()));
        assert_eq!(classify_init_error(108, "no HMD".into()), ConnectError::Failed("no HMD".into()));
    }

    #[test]
    fn fn_table_entries_are_read_by_index() {
        extern "system" fn a(_: u64) -> i32 {
            7
        }
        let table: [usize; 3] = [0, a as *const () as usize, 0];
        // SAFETY: entry 1 is `a`, of type HandleFn.
        let f: HandleFn = unsafe { entry(table.as_ptr(), 1, "a") }.unwrap();
        // SAFETY: calls `a`.
        assert_eq!(unsafe { f(1) }, 7);
        // SAFETY: entry 0 is null and reported as missing.
        let missing: Result<HandleFn, String> = unsafe { entry(table.as_ptr(), 0, "zero") };
        assert!(missing.unwrap_err().contains("zero"));
    }

    #[cfg(windows)]
    #[test]
    fn process_listing_finds_this_test() {
        let me = std::env::current_exe().unwrap();
        let name = me.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(process_running(&name), Some(true));
        assert_eq!(process_running("no-such-process-sigil.exe"), Some(false));
    }

    /// Loads a real `openvr_api.dll` from `SIGIL_OPENVR_DIR` (for example Steam's own copy in
    /// `C:\Program Files (x86)\Steam`) and checks the exports and the never-launch rule: while
    /// `vrserver.exe` is not running, a Background init must fail with "not running" (or "not
    /// installed") and SteamVR must still not be running afterwards.
    #[test]
    #[ignore = "needs SIGIL_OPENVR_DIR pointing at a folder with openvr_api.dll"]
    fn real_runtime_never_launches_steamvr() {
        let Some(dir) = std::env::var_os("SIGIL_OPENVR_DIR").map(PathBuf::from) else {
            return;
        };
        let mut backend = OpenVrBackend::new(Box::new(move || Some(dir.clone())));
        assert!(backend.runtime_installed());
        let lib_installed = backend.steamvr_installed();
        eprintln!("SteamVR installed: {lib_installed:?}");
        if backend.server_running() == Some(true) {
            eprintln!("SteamVR is running: connecting for real");
            let session = backend.connect().expect("connect while SteamVR runs");
            session.close();
            return;
        }
        match backend.connect() {
            Err(e @ (ConnectError::NotRunning | ConnectError::NotInstalled)) => eprintln!("Background init without SteamVR: {e:?}"),
            Err(other) => panic!("unexpected init result {other:?}"),
            Ok(s) => {
                s.close();
                panic!("connected although vrserver.exe was not running");
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert_eq!(backend.server_running(), Some(false), "SteamVR was launched");
    }
}
