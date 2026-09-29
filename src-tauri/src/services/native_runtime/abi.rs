//! Rust mirror of `native/common/sigil_engine.h` (ABI v1). Frozen: the header and this module
//! change together, and an incompatible change bumps [`SIGIL_ENGINE_ABI_VERSION`].

// A mirror of the C header: every constant is kept even while Sigil does not use it yet.
#![allow(dead_code)]

use std::ffi::{c_char, c_void};

/// `SIGIL_ENGINE_ABI_VERSION`.
pub const SIGIL_ENGINE_ABI_VERSION: u32 = 1;

/// `sigil_status`: 0 = success, negative = failure.
pub type SigilStatus = i32;
pub const SIGIL_OK: SigilStatus = 0;
pub const SIGIL_E_INVALID_ARGUMENT: SigilStatus = -1;
pub const SIGIL_E_UNSUPPORTED: SigilStatus = -2;
pub const SIGIL_E_LOAD_FAILED: SigilStatus = -3;
pub const SIGIL_E_RUNTIME: SigilStatus = -4;
pub const SIGIL_E_OUT_OF_MEMORY: SigilStatus = -5;
pub const SIGIL_E_CANCELLED: SigilStatus = -6;
pub const SIGIL_E_BACKEND_UNAVAILABLE: SigilStatus = -7;

pub const SIGIL_CAP_PROCESS: u64 = 1 << 0;
pub const SIGIL_CAP_STREAM: u64 = 1 << 1;
pub const SIGIL_CAP_RENDER: u64 = 1 << 2;
pub const SIGIL_CAP_CANCEL: u64 = 1 << 3;
pub const SIGIL_CAP_GPU: u64 = 1 << 4;

pub const SIGIL_LOG_TRACE: i32 = 0;
pub const SIGIL_LOG_DEBUG: i32 = 1;
pub const SIGIL_LOG_INFO: i32 = 2;
pub const SIGIL_LOG_WARN: i32 = 3;
pub const SIGIL_LOG_ERROR: i32 = 4;

/// `sigil_engine_info`.
#[repr(C)]
#[derive(Debug)]
pub struct SigilEngineInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub name: *const c_char,
    pub version: *const c_char,
    pub variant: *const c_char,
    pub kind: *const c_char,
    pub build_info: *const c_char,
    pub capabilities: u64,
}

/// `sizeof(sigil_engine_info)` in ABI v1 (x86_64); later versions only append fields.
pub const SIGIL_ENGINE_INFO_V1_SIZE: u32 = 56;

/// `sigil_audio_view` (borrowed, interleaved f32).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SigilAudioView {
    pub samples: *const f32,
    pub frames: u64,
    pub sample_rate: u32,
    pub channels: u32,
}

/// `sigil_audio_buffer` (engine-owned; released with `sigil_free_audio`).
#[repr(C)]
#[derive(Debug)]
pub struct SigilAudioBuffer {
    pub samples: *mut f32,
    pub frames: u64,
    pub sample_rate: u32,
    pub channels: u32,
}

impl Default for SigilAudioBuffer {
    fn default() -> Self {
        Self {
            samples: std::ptr::null_mut(),
            frames: 0,
            sample_rate: 0,
            channels: 0,
        }
    }
}

/// Opaque `sigil_engine`.
#[repr(C)]
pub struct SigilEngine {
    _private: [u8; 0],
}

/// Opaque `sigil_stream`.
#[repr(C)]
pub struct SigilStream {
    _private: [u8; 0],
}

/// `sigil_log_fn` (nullable).
pub type SigilLogFn = Option<unsafe extern "C" fn(user_data: *mut c_void, level: i32, message: *const c_char)>;

pub type AbiVersionFn = unsafe extern "C" fn() -> u32;
pub type GetInfoFn = unsafe extern "C" fn() -> *const SigilEngineInfo;
pub type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
pub type SetLogCallbackFn = unsafe extern "C" fn(SigilLogFn, *mut c_void);
pub type CreateFn = unsafe extern "C" fn(*const c_char, *mut *mut SigilEngine) -> SigilStatus;
pub type DestroyFn = unsafe extern "C" fn(*mut SigilEngine);
pub type CancelFn = unsafe extern "C" fn(*mut SigilEngine);
pub type QueryFn = unsafe extern "C" fn(*mut SigilEngine, *const c_char, *mut *mut c_char) -> SigilStatus;
pub type ProcessFn =
    unsafe extern "C" fn(*mut SigilEngine, *const SigilAudioView, *const c_char, *mut *mut c_char) -> SigilStatus;
pub type RenderFn =
    unsafe extern "C" fn(*mut SigilEngine, *const SigilAudioView, *const c_char, *mut SigilAudioBuffer) -> SigilStatus;
pub type StreamOpenFn = unsafe extern "C" fn(*mut SigilEngine, *const c_char, *mut *mut SigilStream) -> SigilStatus;
pub type StreamPushFn = unsafe extern "C" fn(*mut SigilStream, *const SigilAudioView, *mut *mut c_char) -> SigilStatus;
pub type StreamFinishFn = unsafe extern "C" fn(*mut SigilStream, *mut *mut c_char) -> SigilStatus;
pub type StreamCloseFn = unsafe extern "C" fn(*mut SigilStream);
pub type FreeStringFn = unsafe extern "C" fn(*mut c_char);
pub type FreeAudioFn = unsafe extern "C" fn(*mut SigilAudioBuffer);

/// Every export of an ABI v1 engine; a DLL missing any of them is refused.
pub const SYMBOLS: [&str; 16] = [
    "sigil_engine_abi_version",
    "sigil_engine_get_info",
    "sigil_engine_last_error",
    "sigil_engine_set_log_callback",
    "sigil_engine_create",
    "sigil_engine_destroy",
    "sigil_engine_cancel",
    "sigil_engine_query",
    "sigil_engine_process",
    "sigil_engine_render",
    "sigil_stream_open",
    "sigil_stream_push",
    "sigil_stream_finish",
    "sigil_stream_close",
    "sigil_free_string",
    "sigil_free_audio",
];

/// Symbolic name of a status code (for error messages).
pub fn status_name(status: SigilStatus) -> &'static str {
    match status {
        SIGIL_OK => "SIGIL_OK",
        SIGIL_E_INVALID_ARGUMENT => "SIGIL_E_INVALID_ARGUMENT",
        SIGIL_E_UNSUPPORTED => "SIGIL_E_UNSUPPORTED",
        SIGIL_E_LOAD_FAILED => "SIGIL_E_LOAD_FAILED",
        SIGIL_E_RUNTIME => "SIGIL_E_RUNTIME",
        SIGIL_E_OUT_OF_MEMORY => "SIGIL_E_OUT_OF_MEMORY",
        SIGIL_E_CANCELLED => "SIGIL_E_CANCELLED",
        SIGIL_E_BACKEND_UNAVAILABLE => "SIGIL_E_BACKEND_UNAVAILABLE",
        _ => "SIGIL_E_UNKNOWN",
    }
}
