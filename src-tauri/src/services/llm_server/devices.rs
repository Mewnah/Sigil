//! GPU device selection and footprint measurement for llama-server.
//!
//! - **Device pinning.** llama.cpp spreads layers over every GPU it can see (a PC with an RTX card
//!   and a Radeon iGPU lists both as Vulkan devices), which puts part of the model on the wrong card
//!   and makes the VRAM claim wrong. When the PC has more than one hardware GPU, Sigil asks the
//!   runtime pack for its device list once (`llama-server --list-devices`, run as an isolated worker)
//!   and passes `--device <name>` for the device whose name matches the primary GPU; without a match
//!   it keeps the model on one device (`-sm none`).
//! - **Measured footprint.** Once the server is ready, the device buffer sizes it logged
//!   (`Vulkan0 model buffer size = … MiB`, KV and compute buffers) replace the plan's estimate in the
//!   VRAM claim.

use crate::services::process_host::logs::LogStream;
use crate::services::process_host::{run_worker, HelperLogLine, WorkerSpec};
use crate::services::sysinfo::SystemInfo;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

/// Runtime overhead besides the logged buffers (backend context, driver allocations), MiB.
pub const VULKAN_BASE_MB: u64 = 100;
/// The CUDA context of the cuda13 pack (about 0.4 GB on the dev rig for other ggml CUDA engines).
pub const CUDA_BASE_MB: u64 = 400;

/// One device from `--list-devices`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// llama.cpp device name (`Vulkan0`, `CUDA1`).
    pub name: String,
    /// Adapter name (`NVIDIA GeForce RTX 4090`).
    pub description: String,
    pub total_mib: Option<u64>,
}

/// Parse `--list-devices` output: lines `  Vulkan0: NVIDIA GeForce RTX 4090 (24138 MiB, 23370 MiB free)`.
pub fn parse_list_devices(output: &str) -> Vec<Device> {
    let mut out = Vec::new();
    for line in output.lines() {
        let t = line.trim();
        let Some((name, rest)) = t.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let valid_name = !name.is_empty()
            && name.len() <= 32
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name.chars().last().is_some_and(|c| c.is_ascii_digit())
            && name.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
        if !valid_name {
            continue;
        }
        let rest = rest.trim();
        let (description, total_mib) = match rest.rfind(" (") {
            Some(i) if rest.ends_with(')') => {
                let inner = &rest[i + 2..rest.len() - 1];
                let total = inner
                    .split(',')
                    .next()
                    .and_then(|s| s.trim().strip_suffix("MiB"))
                    .and_then(|s| s.trim().parse::<u64>().ok());
                (rest[..i].trim().to_string(), total)
            }
            _ => (rest.to_string(), None),
        };
        if description.is_empty() {
            continue;
        }
        out.push(Device {
            name: name.to_string(),
            description,
            total_mib,
        });
    }
    out
}

fn normalize(name: &str) -> String {
    let lower = name
        .to_ascii_lowercase()
        .replace("(tm)", "")
        .replace("(r)", "");
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// What to pass so the model stays on the primary GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pin {
    /// One device (or none listed): no flag needed.
    None,
    /// `--device <name>`.
    Device(String),
    /// No device matched the primary GPU: keep the model on one device (`-sm none`).
    SingleDevice,
}

/// Choose the device matching `primary_name` among `devices`.
pub fn pick_device(devices: &[Device], primary_name: &str) -> Pin {
    if devices.len() <= 1 {
        return Pin::None;
    }
    let want = normalize(primary_name);
    if want.is_empty() {
        return Pin::SingleDevice;
    }
    let exact: Vec<&Device> = devices
        .iter()
        .filter(|d| normalize(&d.description) == want)
        .collect();
    let chosen = match exact.as_slice() {
        [one] => Some(*one),
        [] => {
            let partial: Vec<&Device> = devices
                .iter()
                .filter(|d| {
                    let n = normalize(&d.description);
                    !n.is_empty() && (n.contains(&want) || want.contains(&n))
                })
                .collect();
            (partial.len() == 1).then(|| partial[0])
        }
        // Two identical cards: the first one (DXGI and Vulkan both list the primary first).
        [first, ..] => Some(*first),
    };
    match chosen {
        Some(d) => Pin::Device(d.name.clone()),
        None => Pin::SingleDevice,
    }
}

/// Hardware (non-software) GPUs of the PC.
pub fn hardware_gpu_count(sys: &SystemInfo) -> usize {
    sys.gpus.iter().filter(|g| !g.adapter.software).count()
}

type DeviceCache = HashMap<(PathBuf, Option<SystemTime>), Vec<Device>>;

fn cache() -> &'static Mutex<DeviceCache> {
    static C: OnceLock<Mutex<DeviceCache>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// The device list of the llama-server at `exe` (cached per binary; blocking, runs the binary once).
pub fn list_devices(exe: &Path) -> Result<Vec<Device>, String> {
    let key = (exe.to_path_buf(), std::fs::metadata(exe).and_then(|m| m.modified()).ok());
    if let Some(d) = cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(d.clone());
    }
    let out = run_worker(&WorkerSpec {
        name: "llama-devices".into(),
        program: exe.to_path_buf(),
        args: vec!["--list-devices".into()],
        env_remove: vec!["LLAMA_ARG_*".into(), "GGML_VK_VISIBLE_DEVICES".into()],
        cwd: exe.parent().map(Path::to_path_buf),
        timeout_ms: 30_000,
        max_output_bytes: 64 * 1024,
        ..Default::default()
    })?;
    if out.timed_out {
        return Err(format!("listing the GPU devices {}", out.describe()));
    }
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let devices = parse_list_devices(&text);
    cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, devices.clone());
    Ok(devices)
}

/// process_host's note when it launches an instance of a helper (every attempt: the first start,
/// port retries, supervised restarts).
const INSTANCE_START_NOTE: &str = "started pid ";
/// llama.cpp's first line of a model load.
const MODEL_LOAD_MARKER: &str = "llama_model_loader: loaded meta data";

/// A line that starts a new load: what came before it belongs to an earlier attempt.
fn starts_a_load(l: &HelperLogLine) -> bool {
    (l.stream == LogStream::Host && l.text.starts_with(INSTANCE_START_NOTE)) || l.text.contains(MODEL_LOAD_MARKER)
}

/// Device buffers (MiB) llama-server logged while loading, summed over GPU devices: model weights,
/// KV cache and compute buffers. Only the latest instance counts (the helper's log keeps earlier
/// attempts: a start retried after a crash, a supervised restart). `None` until both the model and
/// a KV or compute buffer of that instance were seen.
pub fn measured_buffers_mb(lines: &[HelperLogLine]) -> Option<u64> {
    let mut total_mib = 0f64;
    let mut model = false;
    let mut other = false;
    for l in lines {
        if starts_a_load(l) {
            total_mib = 0.0;
            model = false;
            other = false;
            continue;
        }
        let Some(i) = l.text.find(" buffer size") else {
            continue;
        };
        let before = &l.text[..i];
        let mut words = before.split_whitespace().rev();
        let (Some(kind), Some(device)) = (words.next(), words.next()) else {
            continue;
        };
        if !is_gpu_device(device) {
            continue;
        }
        let Some(value) = l.text[i..]
            .split_once('=')
            .and_then(|(_, v)| v.split_whitespace().next())
            .and_then(|v| v.parse::<f64>().ok())
        else {
            continue;
        };
        if !(0.0..1.0e7).contains(&value) {
            continue;
        }
        match kind {
            "model" => model = true,
            "KV" | "compute" | "RS" => other = true,
            _ => {}
        }
        total_mib += value;
    }
    (model && other).then(|| total_mib.ceil() as u64)
}

/// `Vulkan0`, `CUDA1`, `ROCm0` (not `CPU`, `CPU_Mapped`, `Vulkan_Host`, `CUDA_Host`).
fn is_gpu_device(name: &str) -> bool {
    let digits = name
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .to_string();
    let prefix = &name[..name.len() - digits.len()];
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) && matches!(prefix, "Vulkan" | "CUDA" | "ROCm" | "SYCL" | "Metal" | "MUSA")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The b11146 Vulkan pack's output on the dev rig (RTX 4090 + Radeon iGPU).
    const DEV_RIG: &str = "0.00.001.191 I srv  llama_server: initializing ...\r\nAvailable devices:\r\n  Vulkan0: NVIDIA GeForce RTX 4090 (24138 MiB, 23370 MiB free)\r\n  Vulkan1: AMD Radeon(TM) Graphics (16171 MiB, 15362 MiB free)\r\n";

    #[test]
    fn parses_the_device_list() {
        let d = parse_list_devices(DEV_RIG);
        assert_eq!(d.len(), 2, "{d:?}");
        assert_eq!(d[0].name, "Vulkan0");
        assert_eq!(d[0].description, "NVIDIA GeForce RTX 4090");
        assert_eq!(d[0].total_mib, Some(24_138));
        assert_eq!(d[1].description, "AMD Radeon(TM) Graphics");
        // Log lines with colons are not devices.
        assert!(parse_list_devices("0.00.001 I srv  llama_server: initializing ...\nerror: no devices").is_empty());
        let cuda =
            parse_list_devices("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3070, compute capability 8.6, VMM: yes (8191 MiB, 7000 MiB free)\n");
        assert_eq!(cuda[0].name, "CUDA0");
        assert_eq!(cuda[0].total_mib, Some(8191));
        assert!(cuda[0].description.starts_with("NVIDIA GeForce RTX 3070"));
    }

    #[test]
    fn pins_the_primary_gpu() {
        let d = parse_list_devices(DEV_RIG);
        assert_eq!(pick_device(&d, "NVIDIA GeForce RTX 4090"), Pin::Device("Vulkan0".into()));
        assert_eq!(pick_device(&d, "AMD Radeon(TM) Graphics"), Pin::Device("Vulkan1".into()));
        assert_eq!(pick_device(&d[..1], "NVIDIA GeForce RTX 4090"), Pin::None, "one device: nothing to pin");
        assert_eq!(pick_device(&d, "Intel(R) Arc(TM) A770"), Pin::SingleDevice);
        assert_eq!(pick_device(&d, ""), Pin::SingleDevice);
        // The CUDA list carries extra text after the name: a unique partial match.
        let cuda = parse_list_devices("  CUDA0: NVIDIA GeForce RTX 3070, compute capability 8.6 (8191 MiB, 7000 MiB free)\n  CUDA1: NVIDIA GeForce GTX 1080 Ti, compute capability 6.1 (11264 MiB, 11000 MiB free)\n");
        assert_eq!(pick_device(&cuda, "NVIDIA GeForce RTX 3070"), Pin::Device("CUDA0".into()));
        assert_eq!(pick_device(&cuda, "NVIDIA GeForce GTX 1080 Ti"), Pin::Device("CUDA1".into()));
        // Two identical cards: the first.
        let twins = parse_list_devices(
            "  Vulkan0: NVIDIA GeForce RTX 3060 (12288 MiB, 1 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 3060 (12288 MiB, 1 MiB free)\n",
        );
        assert_eq!(pick_device(&twins, "NVIDIA GeForce RTX 3060"), Pin::Device("Vulkan0".into()));
    }

    fn line(text: &str) -> HelperLogLine {
        HelperLogLine {
            seq: 0,
            at_unix_ms: 0,
            stream: LogStream::Stderr,
            text: text.into(),
        }
    }

    #[test]
    fn measures_device_buffers_from_the_load_log() {
        let lines: Vec<HelperLogLine> = [
            "load_tensors: offloading 32 repeating layers to GPU",
            "load_tensors:      Vulkan0 model buffer size =  1024.50 MiB",
            "load_tensors:   CPU_Mapped model buffer size =   265.50 MiB",
            "llama_kv_cache:    Vulkan0 KV buffer size =   256.00 MiB",
            "llama_context: Vulkan_Host  output buffer size =     0.46 MiB",
            "llama_context:    Vulkan0 compute buffer size =   300.25 MiB",
            "llama_context: Vulkan_Host compute buffer size =    16.01 MiB",
        ]
        .iter()
        .map(|t| line(t))
        .collect();
        assert_eq!(measured_buffers_mb(&lines), Some(1581));
        // Model buffer alone (still loading): not yet.
        assert_eq!(measured_buffers_mb(&lines[..2]), None);
        assert_eq!(measured_buffers_mb(&[]), None);
        // CPU-only runs log no GPU buffers.
        let cpu = [
            line("load_tensors:   CPU_Mapped model buffer size =  1100.00 MiB"),
            line("llama_kv_cache:        CPU KV buffer size =   256.00 MiB"),
        ];
        assert_eq!(measured_buffers_mb(&cpu), None);
        // CUDA names, garbage values.
        let cuda = [
            line("CUDA0 model buffer size = 4000 MiB"),
            line("CUDA0 compute buffer size = nan MiB"),
            line("CUDA0 KV buffer size = 512.0 MiB"),
        ];
        assert_eq!(measured_buffers_mb(&cuda), Some(4512));
        assert!(is_gpu_device("CUDA12") && !is_gpu_device("CUDA_Host") && !is_gpu_device("CPU") && !is_gpu_device("Vulkan"));
    }

    fn host(text: &str) -> HelperLogLine {
        HelperLogLine {
            stream: LogStream::Host,
            ..line(text)
        }
    }

    #[test]
    fn only_the_latest_instance_is_measured() {
        let attempt = |model: &str| {
            vec![
                line(&format!("load_tensors:      Vulkan0 model buffer size =  {model} MiB")),
                line("llama_kv_cache:    Vulkan0 KV buffer size =   256.00 MiB"),
                line("llama_context:    Vulkan0 compute buffer size =   300.00 MiB"),
            ]
        };
        // First attempt crashed after allocating; process_host retried and the second became ready.
        let mut lines = vec![host("starting C:\\rt\\llama-server.exe"), host("started pid 100 on 127.0.0.1:5000")];
        lines.extend(attempt("4000.00"));
        lines.push(host("process 100 ended: exit code 1"));
        lines.push(host("started pid 200 on 127.0.0.1:5001"));
        assert_eq!(measured_buffers_mb(&lines), None, "the new instance has logged nothing yet");
        lines.extend(attempt("4000.00"));
        assert_eq!(measured_buffers_mb(&lines), Some(4556), "one instance, not both");
        // A supervised restart that reloads a different size is measured afresh.
        lines.push(host("restarting in 1000 ms"));
        lines.push(host("started pid 300 on 127.0.0.1:5002"));
        lines.extend(attempt("3000.00"));
        assert_eq!(measured_buffers_mb(&lines), Some(3556));
        // llama.cpp's own load marker also starts a new count (a model reloaded in one process).
        lines.push(line("llama_model_loader: loaded meta data with 35 key-value pairs and 291 tensors from m.gguf"));
        assert_eq!(measured_buffers_mb(&lines), None);
        // Output lines that merely mention "started pid" are not host notes.
        let mut lines = attempt("1000.00");
        lines.push(line("started pid 5 in some other tool"));
        assert_eq!(measured_buffers_mb(&lines), Some(1556));
    }
}
