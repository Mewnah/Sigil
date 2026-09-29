//! Lazy DLL loading by full path.
//!
//! Nothing Sigil loads from a runtime pack is load-time linked or found through `PATH`:
//! - [`load_library_in_dir`] loads `dir\file` with `LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
//!   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`: the DLL's imports resolve from its own folder, then the
//!   application folder, System32 and the folders added with [`add_dll_directory`]. `PATH` and the
//!   current directory are never searched, so a stray `ggml.dll` or `cudart64_13.dll` elsewhere on
//!   the PC cannot be picked up.
//! - [`add_dll_directory`] registers a pack folder (`AddDllDirectory`) for the process lifetime, so
//!   an engine's dependencies in another pack (the NVIDIA runtime's `cudart64_13.dll`) resolve.
//!   `SetDefaultDllDirectories` is deliberately never called: it would drop `PATH` for every other
//!   `LoadLibrary` in the process (ADR 0003).
//! - [`preload_ordered`] loads a set of DLLs by full path in dependency order and keeps them loaded,
//!   so later imports by module name bind to them (the NVIDIA runtime pack).
//!
//! Every failure is an `Err` with the path and the OS reason; nothing here panics.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};

/// Load `dir/file_name` by full path so that the DLL's own folder resolves its dependencies
/// (`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` on Windows).
pub fn load_library_in_dir(dir: &Path, file_name: &str) -> Result<libloading::Library, String> {
    if file_name.is_empty() || file_name.contains(['/', '\\']) || file_name.contains("..") {
        return Err(format!("invalid DLL name {file_name:?}"));
    }
    let path = dir.join(file_name);
    if !path.is_file() {
        return Err(format!("{} does not exist", path.display()));
    }
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs a fully qualified path.
    let path = absolute(&path);
    #[cfg(windows)]
    {
        use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
        // SAFETY: loading a DLL runs its initialisation routines; callers only load Sigil runtime
        // packs whose files were verified (model store install + pack manifest hashes).
        unsafe { Library::load_with_flags(&path, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
            .map(libloading::Library::from)
            .map_err(|e| format!("cannot load {}: {}", path.display(), describe_load_error(&error_chain(&e))))
    }
    #[cfg(not(windows))]
    {
        // SAFETY: as above.
        unsafe { libloading::Library::new(&path) }.map_err(|e| format!("cannot load {}: {}", path.display(), error_chain(&e)))
    }
}

/// `error: source: source ...` (libloading's own message is only "LoadLibraryExW failed"; the OS
/// error with its code is the source).
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut text = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        let s_text = s.to_string();
        if !text.contains(&s_text) {
            text.push_str(": ");
            text.push_str(&s_text);
        }
        source = s.source();
    }
    text
}

/// `path` made absolute against the current directory (no symlink resolution, no `\\?\` prefix).
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Add a plain-English hint to the Win32 errors a DLL load typically fails with.
fn describe_load_error(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    let hint = if lower.contains("(os error 126)") || lower.contains("module could not be found") {
        Some("a DLL it depends on is missing")
    } else if lower.contains("(os error 193)") || lower.contains("not a valid win32 application") {
        Some("the file is not a 64-bit Windows DLL (damaged download?)")
    } else if lower.contains("(os error 127)") || lower.contains("procedure could not be found") {
        Some("a DLL it depends on is a different version")
    } else if lower.contains("(os error 1114)") {
        Some("the DLL failed to initialize")
    } else if lower.contains("(os error 5)") {
        Some("access was denied (blocked by antivirus?)")
    } else {
        None
    };
    match hint {
        Some(h) => format!("{h} ({raw})"),
        None => raw.to_string(),
    }
}

fn added_dirs() -> &'static Mutex<HashSet<PathBuf>> {
    static DIRS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    DIRS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Register `dir` with `AddDllDirectory` for the rest of the process (idempotent). Later
/// `LoadLibraryExW` calls with `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` / `USER_DIRS` search it.
pub fn add_dll_directory(dir: &Path) -> Result<(), String> {
    let dir = absolute(dir);
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    let mut added = added_dirs().lock().unwrap_or_else(PoisonError::into_inner);
    if added.contains(&dir) {
        return Ok(());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows::core::PCWSTR;
        use windows::Win32::System::LibraryLoader::AddDllDirectory;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: NUL-terminated absolute path; the cookie is intentionally never removed (the
        // folder stays a search location while DLLs loaded from it are in use).
        let cookie = unsafe { AddDllDirectory(PCWSTR(wide.as_ptr())) };
        if cookie.is_null() {
            return Err(format!(
                "AddDllDirectory({}) failed: {}",
                dir.display(),
                std::io::Error::last_os_error()
            ));
        }
    }
    added.insert(dir);
    Ok(())
}

/// Every `*.dll` below `root` (at most `max_depth` folder levels deep), keyed by lower-case file
/// name. When a name occurs twice, the shallower path wins (then the lexically smaller one), so the
/// result does not depend on directory enumeration order.
pub fn find_dlls(root: &Path, max_depth: usize) -> BTreeMap<String, PathBuf> {
    fn walk(dir: &Path, depth: usize, max_depth: usize, out: &mut BTreeMap<String, (usize, PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut subdirs = Vec::new();
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            let path = entry.path();
            if ft.is_dir() {
                subdirs.push(path);
            } else if ft.is_file() {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".dll") {
                    let replace = match out.get(&name) {
                        None => true,
                        Some((d, p)) => depth < *d || (depth == *d && path < *p),
                    };
                    if replace {
                        out.insert(name, (depth, path));
                    }
                }
            }
        }
        if depth < max_depth {
            for d in subdirs {
                walk(&d, depth + 1, max_depth, out);
            }
        }
    }
    let mut found = BTreeMap::new();
    walk(root, 0, max_depth, &mut found);
    found.into_iter().map(|(k, (_, p))| (k, p)).collect()
}

/// Load `order` (file name, required) by full path from `index` (as returned by [`find_dlls`]) and
/// keep every module loaded for the life of the process. Registers each folder that holds one of
/// them with [`add_dll_directory`] first, so DLLs they load by name at run time (cuDNN's
/// sub-libraries) resolve too. Returns the names that were loaded; a missing or failing required
/// DLL is an `Err` (DLLs loaded before it stay loaded: they are valid and harmless).
pub fn preload_ordered(order: &[(&str, bool)], index: &BTreeMap<String, PathBuf>) -> Result<Vec<String>, String> {
    let mut dirs: Vec<PathBuf> = order
        .iter()
        .filter_map(|(name, _)| index.get(&name.to_ascii_lowercase()))
        .filter_map(|p| p.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        if let Err(e) = add_dll_directory(d) {
            tracing::debug!(target: "sigil::runtime", "{e}");
        }
    }
    let mut loaded = Vec::new();
    for (name, required) in order {
        let Some(path) = index.get(&name.to_ascii_lowercase()) else {
            if *required {
                return Err(format!("{name} is missing from the runtime pack"));
            }
            continue;
        };
        let (dir, file) = match (path.parent(), path.file_name().and_then(|f| f.to_str())) {
            (Some(d), Some(f)) => (d, f),
            _ => return Err(format!("invalid DLL path {}", path.display())),
        };
        match load_library_in_dir(dir, file) {
            Ok(lib) => {
                // Deliberately leaked (never FreeLibrary'd): later imports bind to it by name.
                std::mem::forget(lib);
                loaded.push((*name).to_string());
            }
            Err(e) if *required => return Err(e),
            Err(e) => tracing::info!(target: "sigil::runtime", "optional {name} not loaded: {e}"),
        }
    }
    Ok(loaded)
}
