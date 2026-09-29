//! CPU and system memory: brand string (CPUID), physical / logical / performance core counts
//! (`GetLogicalProcessorInformationEx`), AVX2 / AVX-512 / VNNI support (std's detection, which also
//! checks OS support for the wider registers) and total RAM (`GlobalMemoryStatusEx`).
//!
//! The feature bits match `native_runtime::VariantSupport::detect` (AVX2 means AVX2 + FMA, AVX-512
//! means AVX-512F), so the UI and the runtime-pack variant choice never disagree.

use super::CpuInfo;

/// Probe the CPU. Never fails: unknown values are empty / zero / `false`.
pub fn probe() -> CpuInfo {
    let (physical, logical, performance) = imp::core_counts();
    let fallback_logical = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);
    let logical_cores = if logical > 0 { logical } else { fallback_logical };
    let physical_cores = if physical > 0 { physical } else { logical_cores };
    let (avx2, avx512, vnni) = features();
    CpuInfo {
        brand: brand(),
        physical_cores,
        logical_cores,
        performance_cores: if performance > 0 { performance } else { physical_cores },
        avx2,
        avx512,
        vnni,
    }
}

/// Total physical memory in MiB (0 when unknown).
pub fn total_ram_mb() -> u64 {
    imp::total_ram_bytes() / (1024 * 1024)
}

/// `(avx2+fma, avx512f, avx512-vnni or avx-vnni)`.
fn features() -> (bool, bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let avx2 = has!("avx2") && has!("fma");
        let avx512 = has!("avx512f");
        let avx512_vnni = has!("avx512vnni");
        let avx_vnni = has!("avxvnni");
        (avx2, avx512, avx512_vnni || avx_vnni)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false, false)
    }
}

/// CPUID brand string, e.g. `AMD Ryzen 9 7950X 16-Core Processor` (whitespace collapsed).
fn brand() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return String::new();
        }
        let mut bytes = Vec::with_capacity(48);
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            let r = __cpuid(leaf);
            for reg in [r.eax, r.ebx, r.ecx, r.edx] {
                bytes.extend_from_slice(&reg.to_le_bytes());
            }
        }
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        collapse_whitespace(&String::from_utf8_lossy(&bytes[..end]))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        String::new()
    }
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Count cores in a `GetLogicalProcessorInformationEx(RelationProcessorCore)` buffer:
/// `(physical, logical, performance)` where performance cores are those in the highest
/// efficiency class (all cores on non-hybrid CPUs). Pure, so it is unit-tested with a synthetic
/// buffer. Record layout (x64): `u32 Relationship`, `u32 Size`, then `PROCESSOR_RELATIONSHIP`:
/// `u8 Flags`, `u8 EfficiencyClass`, `[u8; 20] Reserved`, `u16 GroupCount` (offset 30) and
/// `GROUP_AFFINITY[GroupCount]` from offset 32 (16 bytes each, `u64 Mask` first).
pub(crate) fn parse_core_records(buf: &[u8]) -> (u32, u32, u32) {
    const RELATION_PROCESSOR_CORE: u32 = 0;
    let read_u32 = |o: usize| {
        buf.get(o..o + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let read_u16 = |o: usize| buf.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let read_u64 = |o: usize| {
        buf.get(o..o + 8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    };
    let mut cores: Vec<(u8, u32)> = Vec::new(); // (efficiency class, logical processors)
    let mut o = 0usize;
    while let (Some(rel), Some(size)) = (read_u32(o), read_u32(o + 4)) {
        let size = size as usize;
        if size < 32 || o + size > buf.len() {
            break;
        }
        if rel == RELATION_PROCESSOR_CORE {
            let class = buf[o + 9];
            let groups = read_u16(o + 30).unwrap_or(0) as usize;
            let mut threads = 0u32;
            for g in 0..groups {
                let at = o + 32 + g * 16;
                if at + 16 > o + size {
                    break;
                }
                threads += read_u64(at).map(u64::count_ones).unwrap_or(0);
            }
            cores.push((class, threads.max(1)));
        }
        o += size;
    }
    let physical = cores.len() as u32;
    let logical = cores.iter().map(|&(_, t)| t).sum();
    let top = cores.iter().map(|&(c, _)| c).max().unwrap_or(0);
    let performance = cores.iter().filter(|&&(c, _)| c == top).count() as u32;
    (physical, logical, performance)
}

#[cfg(windows)]
mod imp {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    // kernel32 is already imported by every Windows binary: no new load-time dependency.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: i32, buffer: *mut u8, length: *mut u32) -> i32;
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn core_counts() -> (u32, u32, u32) {
        const RELATION_PROCESSOR_CORE: i32 = 0;
        let mut len = 0u32;
        // SAFETY: size query with a null buffer (fails with ERROR_INSUFFICIENT_BUFFER, sets len).
        unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len) };
        if len == 0 || len > 16 * 1024 * 1024 {
            return (0, 0, 0);
        }
        // u64 storage keeps the records 8-byte aligned, as the API expects.
        let mut storage = vec![0u64; (len as usize).div_ceil(8)];
        let mut got = (storage.len() * 8) as u32;
        // SAFETY: `storage` holds at least `got` bytes; the API writes at most `got` bytes.
        let ok = unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, storage.as_mut_ptr().cast(), &mut got) };
        if ok == 0 {
            return (0, 0, 0);
        }
        // SAFETY: reinterpret the initialized u64 buffer as bytes (no alignment requirement).
        let bytes = unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), (got as usize).min(storage.len() * 8)) };
        super::parse_core_records(bytes)
    }

    pub fn total_ram_bytes() -> u64 {
        let mut s = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `length` is set as the API requires; valid out pointer.
        if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
            s.total_phys
        } else {
            0
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn core_counts() -> (u32, u32, u32) {
        (0, 0, 0)
    }
    pub fn total_ram_bytes() -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `RelationProcessorCore` record (x64 layout) with `threads` bits set in group 0.
    fn core_record(class: u8, threads: u32) -> Vec<u8> {
        let mut r = vec![0u8; 48];
        r[0..4].copy_from_slice(&0u32.to_le_bytes()); // RelationProcessorCore
        r[4..8].copy_from_slice(&48u32.to_le_bytes()); // Size
        r[9] = class;
        r[30..32].copy_from_slice(&1u16.to_le_bytes()); // GroupCount
        let mask: u64 = (1u64 << threads) - 1;
        r[32..40].copy_from_slice(&mask.to_le_bytes());
        r
    }

    #[test]
    fn parses_hybrid_and_smt_core_records() {
        // 8 P-cores with SMT (class 1) + 16 E-cores (class 0), like a Core i9-13900K.
        let mut buf = Vec::new();
        for _ in 0..8 {
            buf.extend(core_record(1, 2));
        }
        for _ in 0..16 {
            buf.extend(core_record(0, 1));
        }
        assert_eq!(parse_core_records(&buf), (24, 32, 8));

        // Non-hybrid 16-core SMT CPU: all cores are performance cores.
        let buf: Vec<u8> = (0..16).flat_map(|_| core_record(0, 2)).collect();
        assert_eq!(parse_core_records(&buf), (16, 32, 16));

        // Truncated / garbage buffers never panic.
        assert_eq!(parse_core_records(&[]), (0, 0, 0));
        assert_eq!(parse_core_records(&buf[..20]), (0, 0, 0));
        let mut bad = core_record(0, 2);
        bad[4..8].copy_from_slice(&4u32.to_le_bytes()); // Size too small: stop
        assert_eq!(parse_core_records(&bad), (0, 0, 0));
    }

    #[test]
    fn probes_this_machine() {
        let cpu = probe();
        assert!(cpu.logical_cores >= 1);
        assert!(cpu.physical_cores >= 1 && cpu.physical_cores <= cpu.logical_cores);
        assert!(cpu.performance_cores >= 1 && cpu.performance_cores <= cpu.physical_cores);
        assert_eq!(cpu.brand, cpu.brand.trim());
        #[cfg(windows)]
        assert!(total_ram_mb() > 512);
    }
}
