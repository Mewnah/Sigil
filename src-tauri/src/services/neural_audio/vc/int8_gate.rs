//! CPU feature detection for int8 content-encoder quantization (Streamer profile).
//!
//! The Sigil Voice Engine's content-encoder pool (`engine::content`) prefers an entry's
//! `model.int8.onnx` on the CPU EP when [`int8_content_encoder_enabled`] says the CPU runs int8
//! well (VNNI); `SIGIL_VC_INT8=0` / `1` overrides the detection for benchmarks and A/B tests.

/// Environment override of the int8 content-encoder choice (`0` = never, `1` = always).
pub const INT8_ENV: &str = "SIGIL_VC_INT8";

/// True when dynamic int8 on the content encoder is expected to help (~AVX512-VNNI or Zen4+).
///
/// Requires AVX2, then accepts AVX512-VNNI, AVX-VNNI (CPUID leaf 7 sub-leaf 1, EAX bit 4) or, as
/// a Zen4 heuristic, AVX512F.
#[cfg(target_arch = "x86_64")]
pub fn supports_int8_content_encoder() -> bool {
    if !is_x86_feature_detected!("avx2") {
        return false;
    }
    // Separate checks: `a || b` over these macros trips clippy::nonminimal_bool (macro expansion).
    if is_x86_feature_detected!("avx512vnni") {
        return true;
    }
    if is_x86_feature_detected!("avx512f") {
        return true;
    }
    avx_vnni_supported()
}

#[cfg(target_arch = "x86")]
pub fn supports_int8_content_encoder() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn supports_int8_content_encoder() -> bool {
    false
}

/// Parse an [`INT8_ENV`] value: `Some(true)` / `Some(false)` for a recognised switch, `None` to
/// keep the CPU detection.
fn parse_override(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Whether the content-encoder pool picks int8 variants on the CPU EP: [`INT8_ENV`] when set,
/// else [`supports_int8_content_encoder`].
pub fn int8_content_encoder_enabled() -> bool {
    std::env::var(INT8_ENV)
        .ok()
        .and_then(|v| parse_override(&v))
        .unwrap_or_else(supports_int8_content_encoder)
}

/// AVX-VNNI (VEX-encoded VNNI, Alder Lake+ / Zen5): CPUID leaf 7 sub-leaf 1, EAX bit 4.
// `__cpuid*` became safe fns in newer toolchains; the block is still required on the 1.70 MSRV.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
fn avx_vnni_supported() -> bool {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    const AVX_VNNI: u32 = 1 << 4;
    // SAFETY: the CPUID instruction exists on every x86_64 CPU; leaf 7 is only queried when the
    // maximum basic leaf reports it.
    unsafe {
        if __cpuid(0).eax < 7 {
            return false;
        }
        __cpuid_count(7, 0).eax >= 1 && (__cpuid_count(7, 1).eax & AVX_VNNI) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_parse_and_default_to_detection() {
        assert_eq!(parse_override("0"), Some(false));
        assert_eq!(parse_override(" ON "), Some(true));
        assert_eq!(parse_override("auto"), None);
        assert_eq!(parse_override(""), None);
        // Without the variable the detection decides (whatever this CPU is).
        if std::env::var_os(INT8_ENV).is_none() {
            assert_eq!(int8_content_encoder_enabled(), supports_int8_content_encoder());
        }
    }
}
