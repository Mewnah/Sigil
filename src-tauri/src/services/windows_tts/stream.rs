//! A write-only `IStream` COM object that SAPI renders into (the base stream of an `SpStream`).
//!
//! Hand-written vtable, like `audio::device_watch` (the `#[implement]` macro needs `windows-core` as
//! a direct dependency, which Sigil does not have). SAPI calls only `QueryInterface`,
//! `Seek(0, STREAM_SEEK_CUR)` and `Write` on a base stream in raw-PCM mode (measured with the
//! Windows 11 desktop voices); `Write` comes from SAPI's own worker thread, so the object is
//! thread-safe (atomic reference count and position, the sink behind `Arc<dyn StreamSink>`).
//!
//! - `Write` hands the bytes to the [`StreamSink`]; when it refuses (the utterance was cancelled,
//!   or the sink failed), the write fails with `E_ABORT` and SAPI stops the speech at once.
//! - `Seek` only reports the position (a move to anywhere else fails): raw PCM is never patched.
//! - Reading, cloning, locking and copying are not supported.
//! - A panic in the sink never unwinds into SAPI: it becomes `E_FAIL`.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use windows::core::{IUnknown, IUnknown_Vtbl, Interface, GUID, HRESULT};
use windows::Win32::Foundation::{E_ABORT, E_FAIL, E_NOINTERFACE, E_NOTIMPL, E_POINTER, S_OK, STG_E_INVALIDFUNCTION};
use windows::Win32::System::Com::{
    ISequentialStream, ISequentialStream_Vtbl, IStream, IStream_Vtbl, STATSTG, STGM_WRITE, STGTY_STREAM, STREAM_SEEK,
    STREAM_SEEK_CUR, STREAM_SEEK_END, STREAM_SEEK_SET,
};

/// Receives what SAPI writes.
pub(crate) trait StreamSink: Send + Sync {
    /// Bytes SAPI wrote (called on SAPI's thread, in order). `false` stops the speech.
    fn write(&self, bytes: &[u8]) -> bool;
}

/// The COM object: the vtable pointer must come first.
#[repr(C)]
struct PcmStream {
    vtbl: *const IStream_Vtbl,
    refs: AtomicU32,
    /// Bytes written so far (the stream position).
    pos: AtomicU64,
    sink: Arc<dyn StreamSink>,
}

static VTBL: IStream_Vtbl = IStream_Vtbl {
    base__: ISequentialStream_Vtbl {
        base__: IUnknown_Vtbl {
            QueryInterface: query_interface,
            AddRef: add_ref,
            Release: release,
        },
        Read: read,
        Write: write,
    },
    Seek: seek,
    SetSize: set_size,
    CopyTo: copy_to,
    Commit: commit,
    Revert: revert,
    LockRegion: lock_region,
    UnlockRegion: unlock_region,
    Stat: stat,
    Clone: clone,
};

/// A new stream feeding `sink` (one reference, owned by the returned interface).
pub(crate) fn new_stream(sink: Arc<dyn StreamSink>) -> IStream {
    let raw = Box::into_raw(Box::new(PcmStream {
        vtbl: &VTBL,
        refs: AtomicU32::new(1),
        pos: AtomicU64::new(0),
        sink,
    }));
    // SAFETY: `raw` is a COM object whose first field is a valid IStream vtable; the interface
    // takes over the initial reference.
    unsafe { IStream::from_raw(raw.cast()) }
}

/// # Safety
/// `this` must be a live [`PcmStream`] (COM holds a reference while it calls a method).
unsafe fn object<'a>(this: *mut c_void) -> &'a PcmStream {
    // SAFETY: guaranteed by the caller.
    unsafe { &*(this as *const PcmStream) }
}

unsafe extern "system" fn query_interface(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM passes valid pointers; `this` is a live `PcmStream`.
    unsafe {
        if !iid.is_null() && (*iid == IUnknown::IID || *iid == ISequentialStream::IID || *iid == IStream::IID) {
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
    // SAFETY: `this` is live while COM calls it.
    unsafe { object(this) }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last release frees the box created in `new_stream`.
    let left = unsafe { object(this) }.refs.fetch_sub(1, Ordering::Release) - 1;
    if left == 0 {
        fence(Ordering::Acquire);
        drop(unsafe { Box::from_raw(this as *mut PcmStream) });
    }
    left
}

unsafe extern "system" fn read(_this: *mut c_void, _pv: *mut c_void, _cb: u32, read: *mut u32) -> HRESULT {
    if !read.is_null() {
        // SAFETY: a valid out pointer from the caller.
        unsafe { *read = 0 };
    }
    E_NOTIMPL
}

unsafe extern "system" fn write(this: *mut c_void, pv: *const c_void, cb: u32, written: *mut u32) -> HRESULT {
    if !written.is_null() {
        // SAFETY: a valid out pointer from the caller.
        unsafe { *written = 0 };
    }
    if cb == 0 {
        return S_OK;
    }
    if pv.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; SAPI hands `cb` readable bytes at `pv` for the duration of the call.
    let (stream, bytes) = unsafe { (object(this), std::slice::from_raw_parts(pv as *const u8, cb as usize)) };
    let accepted = catch_unwind(AssertUnwindSafe(|| stream.sink.write(bytes)));
    match accepted {
        Ok(true) => {
            stream.pos.fetch_add(u64::from(cb), Ordering::AcqRel);
            if !written.is_null() {
                // SAFETY: as above.
                unsafe { *written = cb };
            }
            S_OK
        }
        Ok(false) => E_ABORT,
        Err(_) => E_FAIL,
    }
}

/// The position a seek lands on, if the stream allows it (only the current position).
pub(crate) fn seek_target(pos: u64, offset: i64, origin: STREAM_SEEK) -> Option<u64> {
    let base = match origin {
        STREAM_SEEK_SET => 0i128,
        // Write-only: the end is the position.
        STREAM_SEEK_CUR | STREAM_SEEK_END => i128::from(pos),
        _ => return None,
    };
    let target = base + i128::from(offset);
    (target == i128::from(pos)).then_some(pos)
}

unsafe extern "system" fn seek(this: *mut c_void, offset: i64, origin: STREAM_SEEK, new_pos: *mut u64) -> HRESULT {
    // SAFETY: `this` is live while COM calls it.
    let pos = unsafe { object(this) }.pos.load(Ordering::Acquire);
    match seek_target(pos, offset, origin) {
        Some(p) => {
            if !new_pos.is_null() {
                // SAFETY: a valid out pointer from the caller.
                unsafe { *new_pos = p };
            }
            S_OK
        }
        None => STG_E_INVALIDFUNCTION,
    }
}

unsafe extern "system" fn set_size(this: *mut c_void, size: u64) -> HRESULT {
    // Growing is harmless (nothing is preallocated); shrinking below what was written is refused.
    // SAFETY: `this` is live while COM calls it.
    if size >= unsafe { object(this) }.pos.load(Ordering::Acquire) {
        S_OK
    } else {
        STG_E_INVALIDFUNCTION
    }
}

unsafe extern "system" fn copy_to(_this: *mut c_void, _to: *mut c_void, _cb: u64, read: *mut u64, written: *mut u64) -> HRESULT {
    // SAFETY: valid (or null) out pointers from the caller.
    unsafe {
        if !read.is_null() {
            *read = 0;
        }
        if !written.is_null() {
            *written = 0;
        }
    }
    E_NOTIMPL
}

unsafe extern "system" fn commit(_this: *mut c_void, _flags: u32) -> HRESULT {
    S_OK // nothing is buffered
}

unsafe extern "system" fn revert(_this: *mut c_void) -> HRESULT {
    E_NOTIMPL
}

unsafe extern "system" fn lock_region(_this: *mut c_void, _offset: u64, _cb: u64, _kind: u32) -> HRESULT {
    STG_E_INVALIDFUNCTION
}

unsafe extern "system" fn unlock_region(_this: *mut c_void, _offset: u64, _cb: u64, _kind: u32) -> HRESULT {
    STG_E_INVALIDFUNCTION
}

unsafe extern "system" fn stat(this: *mut c_void, out: *mut STATSTG, _flags: u32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `out` is a valid STATSTG from the caller. No name is returned (the
    // caller frees `pwcsName` only when it is set).
    unsafe {
        let size = object(this).pos.load(Ordering::Acquire);
        *out = STATSTG {
            r#type: STGTY_STREAM.0 as u32,
            cbSize: size,
            grfMode: STGM_WRITE,
            ..STATSTG::default()
        };
    }
    S_OK
}

unsafe extern "system" fn clone(_this: *mut c_void, out: *mut *mut c_void) -> HRESULT {
    if !out.is_null() {
        // SAFETY: a valid out pointer from the caller.
        unsafe { *out = std::ptr::null_mut() };
    }
    E_NOTIMPL
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use windows::Win32::System::Com::STATFLAG_NONAME;

    #[derive(Default)]
    struct Recorder {
        bytes: Mutex<Vec<u8>>,
        refuse_after: Option<usize>,
        panic: bool,
    }

    impl StreamSink for Recorder {
        fn write(&self, bytes: &[u8]) -> bool {
            assert!(!self.panic, "sink panicked");
            let mut b = self.bytes.lock().unwrap();
            if self.refuse_after.is_some_and(|n| b.len() >= n) {
                return false;
            }
            b.extend_from_slice(bytes);
            true
        }
    }

    #[test]
    fn writes_reach_the_sink_and_move_the_position() {
        let rec = Arc::new(Recorder::default());
        let stream = new_stream(rec.clone());
        let seq: ISequentialStream = stream.cast().unwrap();
        let mut written = 0u32;
        // SAFETY: valid buffers for the duration of the calls.
        unsafe {
            assert_eq!(seq.Write([1u8, 2, 3].as_ptr().cast(), 3, Some(&mut written)), S_OK);
            assert_eq!(written, 3);
            assert_eq!(seq.Write([4u8].as_ptr().cast(), 1, None), S_OK);
            assert_eq!(seq.Write(std::ptr::null(), 0, Some(&mut written)), S_OK, "an empty write is fine");
            assert_eq!(written, 0);
            assert_eq!(seq.Write(std::ptr::null(), 4, None), E_POINTER);
        }
        assert_eq!(*rec.bytes.lock().unwrap(), vec![1, 2, 3, 4]);
        // Position: Seek(0, CUR) as SAPI does; anything else is refused.
        let mut pos = 0u64;
        // SAFETY: valid out pointers.
        unsafe {
            stream.Seek(0, STREAM_SEEK_CUR, Some(&mut pos)).unwrap();
            assert_eq!(pos, 4);
            stream.Seek(4, STREAM_SEEK_SET, Some(&mut pos)).unwrap();
            stream.Seek(0, STREAM_SEEK_END, None).unwrap();
            assert!(stream.Seek(0, STREAM_SEEK_SET, None).is_err(), "no rewind");
            assert!(stream.Seek(1, STREAM_SEEK_CUR, None).is_err());
            let mut st = STATSTG::default();
            stream.Stat(&mut st, STATFLAG_NONAME).unwrap();
            assert_eq!((st.cbSize, st.r#type), (4, STGTY_STREAM.0 as u32));
            assert!(st.pwcsName.is_null());
            stream.SetSize(10).unwrap();
            assert!(stream.SetSize(2).is_err());
            stream.Commit(windows::Win32::System::Com::STGC_DEFAULT).unwrap();
            assert!(stream.Clone().is_err());
            let mut buf = [0u8; 4];
            let mut n = 7u32;
            assert_eq!(seq.Read(buf.as_mut_ptr().cast(), 4, Some(&mut n)), E_NOTIMPL);
            assert_eq!(n, 0);
        }
    }

    #[test]
    fn a_refusing_sink_aborts_the_write() {
        let rec = Arc::new(Recorder {
            refuse_after: Some(2),
            ..Recorder::default()
        });
        let stream = new_stream(rec.clone());
        let mut written = 9u32;
        // SAFETY: valid buffers.
        unsafe {
            assert_eq!(stream.Write([1u8, 2].as_ptr().cast(), 2, None), S_OK);
            assert_eq!(stream.Write([3u8].as_ptr().cast(), 1, Some(&mut written)), E_ABORT);
        }
        assert_eq!(written, 0);
        let mut pos = 0u64;
        // SAFETY: valid out pointer.
        unsafe { stream.Seek(0, STREAM_SEEK_CUR, Some(&mut pos)).unwrap() };
        assert_eq!(pos, 2, "a refused write does not move the position");
    }

    #[test]
    fn a_panicking_sink_fails_the_write_without_unwinding() {
        let rec = Arc::new(Recorder {
            panic: true,
            ..Recorder::default()
        });
        let stream = new_stream(rec);
        // SAFETY: a valid buffer. (The panic message on stderr is expected.)
        let hr = unsafe { stream.Write([1u8].as_ptr().cast(), 1, None) };
        assert_eq!(hr, E_FAIL);
    }

    #[test]
    fn interfaces_and_reference_counting() {
        let rec = Arc::new(Recorder::default());
        let stream = new_stream(rec.clone());
        assert_eq!(Arc::strong_count(&rec), 2);
        let unknown: IUnknown = stream.cast().unwrap();
        let seq: ISequentialStream = unknown.cast().unwrap();
        assert!(unknown.cast::<windows::Win32::System::Com::IPersistStream>().is_err());
        drop(stream);
        drop(unknown);
        assert_eq!(Arc::strong_count(&rec), 2, "still referenced");
        drop(seq);
        assert_eq!(Arc::strong_count(&rec), 1, "the last release frees the object");
    }

    #[test]
    fn seek_targets() {
        assert_eq!(seek_target(10, 0, STREAM_SEEK_CUR), Some(10));
        assert_eq!(seek_target(10, 10, STREAM_SEEK_SET), Some(10));
        assert_eq!(seek_target(10, 0, STREAM_SEEK_END), Some(10));
        assert_eq!(seek_target(10, -1, STREAM_SEEK_CUR), None);
        assert_eq!(seek_target(10, 0, STREAM_SEEK_SET), None);
        assert_eq!(seek_target(0, i64::MIN, STREAM_SEEK_CUR), None);
        assert_eq!(seek_target(10, 0, STREAM_SEEK(7)), None);
    }
}
