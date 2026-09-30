//! The installed `runtime/bergamot` pack: `sigil-pack.json` (written by `scripts/runtime-packs/bergamot/build.ps1`)
//! says which protocol the helper speaks, which CPU features it was compiled for, and the sha256 of every file. The
//! client checks all three before starting the helper, so an old pack, a pack built for a newer CPU or a swapped
//! executable fails with a clear message instead of a crash loop.

use super::super::MtError;
use super::protocol::PROTOCOL;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

/// Manifest file name inside the pack folder.
pub const MANIFEST: &str = "sigil-pack.json";

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct PackFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct PackManifest {
    pub name: String,
    pub version: String,
    pub protocol: u32,
    /// `avx2` (default build), `avx512` or `sse2` (the compatibility build).
    pub arch: String,
    pub files: Vec<PackFile>,
}

impl PackManifest {
    pub fn parse(text: &str) -> Result<PackManifest, String> {
        serde_json::from_str(text).map_err(|e| format!("invalid {MANIFEST}: {e}"))
    }
}

/// CPU features the running processor has, for [`arch_supported`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512: bool,
}

impl CpuFeatures {
    pub fn detect() -> CpuFeatures {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                avx512: std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512dq")
                    && std::arch::is_x86_feature_detected!("avx512vl"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            CpuFeatures::default()
        }
    }
}

/// Whether a helper built for `arch` runs on this CPU. MSVC's `/arch:AVX2` also emits FMA3.
pub fn arch_supported(arch: &str, cpu: CpuFeatures) -> Result<(), String> {
    match arch.trim().to_ascii_lowercase().as_str() {
        "" | "sse2" | "sse4" | "x86-64" | "x86-64-v2" | "baseline" => Ok(()),
        "avx2" | "x86-64-v3" => {
            if cpu.avx2 && cpu.fma {
                Ok(())
            } else {
                Err(
                    "this Bergamot pack needs a processor with AVX2 (most PCs from 2015 on); build the compatibility pack with build.ps1 -Arch sse2"
                        .into(),
                )
            }
        }
        "avx512" | "x86-64-v4" => {
            if cpu.avx512 {
                Ok(())
            } else {
                Err("this Bergamot pack needs a processor with AVX-512; use the default AVX2 pack".into())
            }
        }
        other => Err(format!("this Bergamot pack was built for an unknown processor type {other:?}")),
    }
}

fn sha256_file(path: &Path) -> std::io::Result<(u64, String)> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut size = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        size += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((size, hex::encode(hasher.finalize())))
}

/// Size and modification time of a verified executable.
type Stamp = (u64, Option<SystemTime>);

/// Verified executables, so a helper restart does not hash the file again.
fn verified() -> &'static Mutex<HashMap<PathBuf, Stamp>> {
    static CELL: OnceLock<Mutex<HashMap<PathBuf, Stamp>>> = OnceLock::new();
    CELL.get_or_init(Default::default)
}

/// Check the pack in `dir` before starting `exe` from it: protocol, CPU features, and the executable's size and
/// sha256 against the manifest. A pack without a manifest (a hand-assembled import) is accepted with a warning; the
/// helper's `hello` still checks the protocol.
pub fn check_pack(dir: &Path, exe: &Path, cpu: CpuFeatures) -> Result<Option<PackManifest>, MtError> {
    let manifest_path = dir.join(MANIFEST);
    let text = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(target: super::LOG_TARGET, "the Bergamot pack has no {MANIFEST}; starting it unchecked");
            return Ok(None);
        }
        Err(e) => return Err(MtError::runtime(format!("cannot read the Bergamot pack manifest: {e}"))),
    };
    let manifest = PackManifest::parse(&text).map_err(MtError::runtime)?;
    if manifest.protocol != PROTOCOL {
        return Err(MtError::runtime(format!(
            "the installed Bergamot pack ({}) speaks protocol {}, this Sigil needs protocol {PROTOCOL}: rebuild it with scripts/runtime-packs/bergamot/build.ps1 and import it again",
            manifest.version, manifest.protocol
        )));
    }
    arch_supported(&manifest.arch, cpu).map_err(MtError::runtime)?;
    let exe_name = exe
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let Some(expected) = manifest
        .files
        .iter()
        .find(|f| f.path.eq_ignore_ascii_case(&exe_name))
    else {
        return Err(MtError::runtime(format!("the Bergamot pack manifest does not list {exe_name}")));
    };
    let meta = std::fs::metadata(exe).map_err(|e| MtError::runtime(format!("cannot read {exe_name}: {e}")))?;
    let stamp = (meta.len(), meta.modified().ok());
    {
        let cache = verified().lock().unwrap_or_else(PoisonError::into_inner);
        if cache.get(exe) == Some(&stamp) {
            return Ok(Some(manifest));
        }
    }
    let (size, sha) = sha256_file(exe).map_err(|e| MtError::runtime(format!("cannot read {exe_name}: {e}")))?;
    if size != expected.size || !sha.eq_ignore_ascii_case(expected.sha256.trim()) {
        return Err(MtError::runtime(format!(
            "{exe_name} does not match the Bergamot pack manifest (changed after it was built?): import the pack again"
        )));
    }
    verified()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(exe.to_path_buf(), stamp);
    Ok(Some(manifest))
}
