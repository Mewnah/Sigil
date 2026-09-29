//! CPU feature detection for int8 content-encoder quantization (Streamer profile).

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
