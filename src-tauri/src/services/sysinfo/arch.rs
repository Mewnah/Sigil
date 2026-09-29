//! GPU generations: NVIDIA compute capability → architecture, plus best-effort PCI device-id
//! tables (used for NVIDIA only when NVML is unavailable, and for AMD / Intel as information).
//!
//! Pure functions; unit-tested below. The eligibility rules in [`super::eps`] compare NVIDIA
//! generations through [`NvidiaGen`] (ordered oldest → newest).

use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_INTEL, VENDOR_NVIDIA};

/// NVIDIA GPU generations, oldest first (the derive order is the comparison order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NvidiaGen {
    Kepler,
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    Ada,
    Hopper,
    Blackwell,
}

impl NvidiaGen {
    pub fn as_str(self) -> &'static str {
        match self {
            NvidiaGen::Kepler => "kepler",
            NvidiaGen::Maxwell => "maxwell",
            NvidiaGen::Pascal => "pascal",
            NvidiaGen::Volta => "volta",
            NvidiaGen::Turing => "turing",
            NvidiaGen::Ampere => "ampere",
            NvidiaGen::Ada => "ada",
            NvidiaGen::Hopper => "hopper",
            NvidiaGen::Blackwell => "blackwell",
        }
    }

    pub fn parse(s: &str) -> Option<NvidiaGen> {
        Some(match s {
            "kepler" => NvidiaGen::Kepler,
            "maxwell" => NvidiaGen::Maxwell,
            "pascal" => NvidiaGen::Pascal,
            "volta" => NvidiaGen::Volta,
            "turing" => NvidiaGen::Turing,
            "ampere" => NvidiaGen::Ampere,
            "ada" => NvidiaGen::Ada,
            "hopper" => NvidiaGen::Hopper,
            "blackwell" => NvidiaGen::Blackwell,
            _ => return None,
        })
    }

    /// The lowest compute capability of the generation (what a floor check compares against when
    /// only the generation is known).
    pub fn min_compute_capability(self) -> (u32, u32) {
        match self {
            NvidiaGen::Kepler => (3, 0),
            NvidiaGen::Maxwell => (5, 0),
            NvidiaGen::Pascal => (6, 0),
            NvidiaGen::Volta => (7, 0),
            NvidiaGen::Turing => (7, 5),
            NvidiaGen::Ampere => (8, 0),
            NvidiaGen::Ada => (8, 9),
            NvidiaGen::Hopper => (9, 0),
            NvidiaGen::Blackwell => (10, 0),
        }
    }
}

/// Architecture of an NVIDIA compute capability (`None` for a capability newer than this table,
/// which callers treat as "newer than Blackwell").
pub fn nvidia_gen_from_cc(major: u32, minor: u32) -> Option<NvidiaGen> {
    Some(match (major, minor) {
        (3, _) => NvidiaGen::Kepler,
        (5, _) => NvidiaGen::Maxwell,
        (6, _) => NvidiaGen::Pascal,
        (7, 0..=4) => NvidiaGen::Volta,
        (7, _) => NvidiaGen::Turing,
        (8, 9) => NvidiaGen::Ada,
        (8, _) => NvidiaGen::Ampere,
        (9, _) => NvidiaGen::Hopper,
        (10..=12, _) => NvidiaGen::Blackwell,
        _ => return None,
    })
}

/// `"8.9"` → `(8, 9)`.
pub fn parse_compute_capability(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// NVIDIA generation from the PCI device id (upper byte ranges of NVIDIA's id allocation). Only a
/// fallback for when NVML cannot report the compute capability; `None` when the id is unknown.
pub fn nvidia_gen_from_device_id(device_id: u32) -> Option<NvidiaGen> {
    Some(match device_id {
        0x0FC0..=0x12FF => NvidiaGen::Kepler,
        0x15F0..=0x15FF => NvidiaGen::Pascal, // GP100 (Tesla P100)
        0x1300..=0x17FF => NvidiaGen::Maxwell,
        0x1B00..=0x1D7F => NvidiaGen::Pascal,
        0x1D80..=0x1DFF => NvidiaGen::Volta, // GV100 (Titan V, V100)
        0x1E00..=0x1FFF => NvidiaGen::Turing,
        0x2000..=0x20FF => NvidiaGen::Ampere, // GA100
        0x2100..=0x21FF => NvidiaGen::Turing, // TU116 / TU117 variants
        0x2200..=0x22FF => NvidiaGen::Ampere, // GA102
        0x2300..=0x23FF => NvidiaGen::Hopper, // GH100
        0x2400..=0x25FF => NvidiaGen::Ampere, // GA103 / GA104 / GA106 / GA107
        0x2600..=0x28FF => NvidiaGen::Ada,    // AD102 … AD107
        0x2900..=0x2FFF => NvidiaGen::Blackwell,
        _ => return None,
    })
}

/// Best-effort architecture name for AMD / Intel adapters (information only; eligibility never
/// depends on it). `None` when unknown.
pub fn other_arch_from_device_id(vendor_id: u32, device_id: u32) -> Option<&'static str> {
    match vendor_id {
        VENDOR_AMD => Some(match device_id {
            0x7310..=0x731F | 0x7340..=0x734F | 0x7360..=0x736F => "rdna1",
            0x73A0..=0x73FF | 0x7420..=0x743F => "rdna2",
            // Rembrandt, Raphael, Granite Ridge, Van Gogh, Mendocino APUs
            0x1681 | 0x164E | 0x13C0 | 0x163F | 0x1506 => "rdna2",
            0x7440..=0x74BF => "rdna3",
            0x15BF | 0x15C8 | 0x1900 | 0x1901 => "rdna3", // Phoenix / Phoenix 2 / Hawk Point APUs
            0x150E | 0x1586 | 0x1114 => "rdna3.5",        // Strix Point / Strix Halo / Krackan APUs
            0x7550..=0x75BF => "rdna4",
            // Vega iGPUs: Raven Ridge, Picasso, Renoir, Cezanne, Lucienne, Barcelo APUs
            0x15DD | 0x15D8 | 0x1636 | 0x1638 | 0x164C | 0x15E7 => "gcn5",
            _ => return None,
        }),
        VENDOR_INTEL => Some(match device_id {
            0x5690..=0x56CF => "xe-hpg",                   // Arc A-series (Alchemist), Flex
            0xE200..=0xE2FF => "xe2",                      // Arc B-series (Battlemage)
            0x6420 | 0x64A0 | 0x64B0 => "xe2",             // Lunar Lake
            0x7D40..=0x7D6F | 0x7DD0..=0x7DDF => "xe-lpg", // Meteor / Arrow Lake
            0x9A40..=0x9AFF | 0x4680..=0x46FF | 0xA780..=0xA7FF | 0x4C80..=0x4C9F => "xe-lp",
            _ => return None,
        }),
        _ => None,
    }
}

/// Whether an AMD PCI device id belongs to an APU's integrated GPU. AMD gives every APU iGPU since
/// Kaveri (2014) an id in `0x1000..=0x1FFF` (Kaveri 0x13xx, Raven / Picasso 0x15DD / 0x15D8,
/// Renoir 0x1636, Cezanne 0x1638, Lucienne 0x164C, Barcelo 0x15E7, Van Gogh 0x163F, Rembrandt
/// 0x1681, Raphael 0x164E, Mendocino 0x1506, Phoenix 0x15BF / 0x15C8, Hawk Point 0x1900,
/// Strix 0x150E / 0x1586, Krackan 0x1114, Granite Ridge 0x13C0), while its discrete Radeons use
/// 0x66xx–0x7Fxx; the older APU families (Llano, Kabini / Mullins, Carrizo / Stoney, Trinity /
/// Richland) are listed explicitly. A range rather than a list, so a new APU is not mistaken for
/// a discrete card.
pub fn is_amd_apu(device_id: u32) -> bool {
    matches!(
        device_id,
        0x1000..=0x1FFF | 0x9640..=0x964F | 0x9830..=0x985F | 0x9870..=0x9877 | 0x98E4 | 0x9900..=0x99FF
    )
}

/// Integrated (shared-memory) GPU heuristic: tiny dedicated VRAM, any Intel GPU outside the Arc
/// discrete ranges, or an AMD APU ([`is_amd_apu`]). An APU with a large BIOS carve-out (a 4 GB+
/// UMA frame buffer shows up as dedicated VRAM) still counts as integrated, so the resource policy
/// never treats system RAM as spare VRAM.
pub fn is_integrated(vendor_id: u32, device_id: u32, dedicated_vram_mb: u64) -> bool {
    if dedicated_vram_mb < 1024 {
        return true;
    }
    match vendor_id {
        VENDOR_INTEL => !matches!(device_id, 0x5690..=0x56CF | 0xE200..=0xE2FF),
        VENDOR_AMD => is_amd_apu(device_id),
        VENDOR_NVIDIA => false,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_capability_maps_to_generation() {
        assert_eq!(nvidia_gen_from_cc(8, 9), Some(NvidiaGen::Ada)); // RTX 4090
        assert_eq!(nvidia_gen_from_cc(8, 6), Some(NvidiaGen::Ampere)); // RTX 3060
        assert_eq!(nvidia_gen_from_cc(8, 0), Some(NvidiaGen::Ampere)); // A100
        assert_eq!(nvidia_gen_from_cc(7, 5), Some(NvidiaGen::Turing)); // RTX 2060 / GTX 1660
        assert_eq!(nvidia_gen_from_cc(7, 0), Some(NvidiaGen::Volta));
        assert_eq!(nvidia_gen_from_cc(6, 1), Some(NvidiaGen::Pascal)); // GTX 1080
        assert_eq!(nvidia_gen_from_cc(12, 0), Some(NvidiaGen::Blackwell)); // RTX 5090
        assert_eq!(nvidia_gen_from_cc(9, 0), Some(NvidiaGen::Hopper));
        assert_eq!(nvidia_gen_from_cc(15, 0), None);
        assert!(NvidiaGen::Turing < NvidiaGen::Ampere && NvidiaGen::Pascal < NvidiaGen::Turing);
        for g in [NvidiaGen::Kepler, NvidiaGen::Turing, NvidiaGen::Blackwell] {
            assert_eq!(NvidiaGen::parse(g.as_str()), Some(g));
            let (a, b) = g.min_compute_capability();
            assert_eq!(nvidia_gen_from_cc(a, b), Some(g));
        }
    }

    #[test]
    fn parses_compute_capability_strings() {
        assert_eq!(parse_compute_capability("8.9"), Some((8, 9)));
        assert_eq!(parse_compute_capability(" 12.0 "), Some((12, 0)));
        assert_eq!(parse_compute_capability("8"), None);
        assert_eq!(parse_compute_capability("x.y"), None);
    }

    #[test]
    fn nvidia_device_ids_map_to_generations() {
        assert_eq!(nvidia_gen_from_device_id(0x2684), Some(NvidiaGen::Ada)); // RTX 4090
        assert_eq!(nvidia_gen_from_device_id(0x2882), Some(NvidiaGen::Ada)); // RTX 4060
        assert_eq!(nvidia_gen_from_device_id(0x2503), Some(NvidiaGen::Ampere)); // RTX 3060
        assert_eq!(nvidia_gen_from_device_id(0x2204), Some(NvidiaGen::Ampere)); // RTX 3090
        assert_eq!(nvidia_gen_from_device_id(0x1F08), Some(NvidiaGen::Turing)); // RTX 2060
        assert_eq!(nvidia_gen_from_device_id(0x2184), Some(NvidiaGen::Turing)); // GTX 1660
        assert_eq!(nvidia_gen_from_device_id(0x1B80), Some(NvidiaGen::Pascal)); // GTX 1080
        assert_eq!(nvidia_gen_from_device_id(0x1C82), Some(NvidiaGen::Pascal)); // GTX 1050 Ti
        assert_eq!(nvidia_gen_from_device_id(0x13C2), Some(NvidiaGen::Maxwell)); // GTX 970
        assert_eq!(nvidia_gen_from_device_id(0x2B85), Some(NvidiaGen::Blackwell)); // RTX 5090
        assert_eq!(nvidia_gen_from_device_id(0x2D04), Some(NvidiaGen::Blackwell)); // RTX 5060 Ti
        assert_eq!(nvidia_gen_from_device_id(0x0100), None);
    }

    #[test]
    fn other_vendors_and_integrated_heuristic() {
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x744C), Some("rdna3")); // RX 7900 XTX
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x73BF), Some("rdna2")); // RX 6900 XT
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x7550), Some("rdna4")); // RX 9070
        assert_eq!(other_arch_from_device_id(VENDOR_INTEL, 0x56A0), Some("xe-hpg")); // Arc A770
        assert_eq!(other_arch_from_device_id(VENDOR_INTEL, 0xE20B), Some("xe2")); // Arc B580
        assert_eq!(other_arch_from_device_id(VENDOR_NVIDIA, 0x2684), None);
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x0001), None);

        assert!(is_integrated(VENDOR_INTEL, 0xA780, 128)); // UHD 770
        assert!(is_integrated(VENDOR_INTEL, 0x7D55, 2048)); // Meteor Lake Arc iGPU
        assert!(!is_integrated(VENDOR_INTEL, 0x56A0, 16_000)); // Arc A770 16 GB
        assert!(is_integrated(VENDOR_AMD, 0x15BF, 4096)); // 780M with a 4 GB carve-out
        assert!(!is_integrated(VENDOR_AMD, 0x744C, 24_000));
        // Common APUs with a 4 GB+ UMA frame buffer are integrated, not 4 GB discrete cards.
        for (id, name) in [
            (0x1638, "Cezanne (Ryzen 7 5700G)"),
            (0x1636, "Renoir (Ryzen 7 4750G)"),
            (0x164C, "Lucienne"),
            (0x15E7, "Barcelo"),
            (0x15DD, "Raven Ridge"),
            (0x15D8, "Picasso"),
            (0x1506, "Mendocino"),
            (0x1114, "Krackan Point"),
            (0x1901, "Hawk Point"),
            (0x1318, "Kaveri"),
            (0x9874, "Carrizo"),
            (0x98E4, "Stoney Ridge"),
        ] {
            assert!(is_integrated(VENDOR_AMD, id, 4096), "{name}");
            assert!(is_integrated(VENDOR_AMD, id, 8192), "{name}");
        }
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x1638), Some("gcn5"));
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x1506), Some("rdna2"));
        assert_eq!(other_arch_from_device_id(VENDOR_AMD, 0x1114), Some("rdna3.5"));
        // Discrete Radeons of every generation stay discrete.
        for id in [0x67DF, 0x687F, 0x66AF, 0x7300, 0x731F, 0x73BF, 0x73FF, 0x744C, 0x7480, 0x7550, 0x7590] {
            assert!(!is_integrated(VENDOR_AMD, id, 8192), "{id:#06x}");
        }
        assert!(!is_integrated(VENDOR_NVIDIA, 0x2684, 24_000));
        assert!(is_integrated(VENDOR_NVIDIA, 0x1D01, 512)); // tiny VRAM counts as integrated-class
    }
}
