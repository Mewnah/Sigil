/**
 * Pure helpers for the Rust `sysinfo` payloads: typed views of the fields P1-D2 added, GPU and
 * architecture labels, VRAM sizes and execution-provider rows. No Tauri, DOM or valtio here
 * (unit-tested under Node).
 */
import type { RustEligibility, RustGpuInfo, RustSystemInfo } from "@/types";

/** Machine-readable eligibility reason (Rust `sysinfo::eps` codes). */
export type EligibilityCode =
  | "ok"
  | "unverified"
  | "no_gpu"
  | "software_only"
  | "no_nvidia_gpu"
  | "gpu_too_old"
  | "driver_too_old"
  | "no_d3d12"
  | "unsupported_os"
  | "gpu_unknown";

/**
 * Rust `sysinfo::GpuProbe`: whether the GPU facts are current, from an earlier check (the driver
 * did not answer the last one), or missing (the driver never answered: `gpus` is empty although the
 * PC may have a GPU).
 */
export type GpuProbeState = "complete" | "previous" | "missing";

/** Rust `sysinfo::Eligibility` with `code` narrowed to the known codes. */
export interface SysEligibility extends RustEligibility {
  code?: EligibilityCode | string;
}

/** Rust `sysinfo::GpuInfo` including the P1-D2 fields. */
export interface SysGpuInfo extends RustGpuInfo {
  /** Shared-memory GPU (Intel iGPU, AMD APU, tiny VRAM). */
  integrated?: boolean;
  /** Direct3D 12 at feature level 11_0+ (DirectML); null when not tested. */
  d3d12?: boolean | null;
  /** Highest Direct3D 12 feature level, e.g. "12_2". */
  d3d12_feature_level?: string | null;
  nvml_index?: number | null;
  pci_bus_id?: string | null;
}

export type EpKey = "cuda" | "tensorrt" | "tensorrt_rtx" | "directml";

/** Rust `sysinfo::SystemInfo` including the P1-D2 fields. */
export interface SysSystemInfo extends RustSystemInfo {
  gpus: SysGpuInfo[];
  eps: Record<EpKey, SysEligibility>;
  cpu: RustSystemInfo["cpu"] & {
    /** P-cores on hybrid CPUs; all cores otherwise. */
    performance_cores?: number;
  };
  nvml_version?: string | null;
  /** Plain-English probe findings (English; for the log). */
  notes?: string[];
  /** Missing from payloads of older builds (= "complete"). */
  gpu_probe?: GpuProbeState | string;
}

/** EP rows in display order. */
export const EP_KEYS: readonly EpKey[] = ["cuda", "tensorrt", "tensorrt_rtx", "directml"];

/** Product names (never translated). */
export const EP_LABELS: Record<EpKey, string> = {
  cuda: "CUDA",
  tensorrt: "TensorRT",
  tensorrt_rtx: "TensorRT-RTX",
  directml: "DirectML",
};

/** Oldest GPU family each NVIDIA EP supports (product names, for the "too old" hint). */
export const EP_MIN_GPU: Partial<Record<EpKey, string>> = {
  cuda: "GeForce RTX 20 / GTX 16",
  tensorrt: "GeForce RTX 20 / GTX 16",
  tensorrt_rtx: "GeForce RTX 30",
};

const ARCH_LABELS: Record<string, string> = {
  kepler: "Kepler",
  maxwell: "Maxwell",
  pascal: "Pascal",
  volta: "Volta",
  turing: "Turing",
  ampere: "Ampere",
  ada: "Ada Lovelace",
  hopper: "Hopper",
  blackwell: "Blackwell",
  gcn5: "Vega",
  rdna1: "RDNA",
  rdna2: "RDNA 2",
  rdna3: "RDNA 3",
  "rdna3.5": "RDNA 3.5",
  rdna4: "RDNA 4",
  "xe-lp": "Xe-LP",
  "xe-lpg": "Xe-LPG",
  "xe-hpg": "Xe-HPG",
  xe2: "Xe2",
};

/** "ada" → "Ada Lovelace"; unknown ids are shown as-is; null for none. */
export function archLabel(arch: string | null | undefined): string | null {
  if (!arch) return null;
  return ARCH_LABELS[arch] ?? arch;
}

/** How complete the GPU facts are ("complete" for payloads without the field or unknown values). */
export function gpuProbeState(info: SysSystemInfo | null | undefined): GpuProbeState {
  const s = info?.gpu_probe;
  return s === "previous" || s === "missing" ? s : "complete";
}

/** The GPU Sigil uses by default. */
export function primaryGpu(info: SysSystemInfo | null | undefined): SysGpuInfo | null {
  if (!info || info.primary_gpu == null) return null;
  return info.gpus.find((g) => g.index === info.primary_gpu) ?? null;
}

/** Hardware GPUs (no software renderer), the primary first, then in DXGI order. */
export function hardwareGpus(info: SysSystemInfo | null | undefined): SysGpuInfo[] {
  if (!info) return [];
  const primary = info.primary_gpu;
  return info.gpus
    .filter((g) => !g.software)
    .sort((a, b) => Number(b.index === primary) - Number(a.index === primary) || a.index - b.index);
}

/**
 * The size a card or a PC's memory is sold as: Windows reports a little less than the nominal size
 * (reserved ranges, an iGPU carve-out), so round up, with a small tolerance for sizes just above a
 * whole GB. 24,138 MiB → "24 GB", 31,927 MiB of RAM → "32 GB", 485 MiB → "485 MB".
 */
export function formatVramNominal(mb: number): string {
  if (!Number.isFinite(mb) || mb < 0) return "?";
  if (mb < 1024) return `${Math.round(mb)} MB`;
  return `${Math.max(1, Math.ceil(mb / 1024 - 0.1))} GB`;
}

/** A measured amount: 23,370 MiB → "22.8 GB", 900 MiB → "900 MB". */
export function formatMb(mb: number): string {
  if (!Number.isFinite(mb) || mb < 0) return "?";
  if (mb < 1024) return `${Math.round(mb)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

export type EpStatus = "ok" | "unverified" | "no";

export function epStatus(e: SysEligibility | null | undefined): EpStatus {
  if (!e || !e.eligible) return "no";
  return e.code === "unverified" ? "unverified" : "ok";
}

/** The NVIDIA GPU an eligibility message is about: the primary if NVIDIA, else the first one. */
export function mainNvidiaGpu(info: SysSystemInfo | null | undefined): SysGpuInfo | null {
  const nvidia = hardwareGpus(info).filter((g) => g.vendor === "nvidia");
  return nvidia[0] ?? null;
}

/** Interpolation values for `models.gpu.ep_code_<code>` (all pre-formatted strings). */
export function eligibilityParams(key: EpKey, info: SysSystemInfo | null | undefined): Record<string, string> {
  const nvidia = mainNvidiaGpu(info);
  const hardware = hardwareGpus(info);
  const gpu = key === "directml" ? hardware.map((g) => g.name).join(", ") : (nvidia?.name ?? "");
  return {
    gpu,
    arch: archLabel(nvidia?.arch) ?? "",
    driver: info?.nvidia_driver ?? "?",
    cuda: info?.cuda_driver_version ?? "?",
    min: EP_MIN_GPU[key] ?? "",
  };
}

/** One line of GPU facts: "Ada Lovelace · compute 8.9 · D3D12 12_2". */
export function gpuFacts(g: SysGpuInfo): string[] {
  const facts: string[] = [];
  const arch = archLabel(g.arch);
  if (arch) facts.push(arch);
  if (g.compute_capability) facts.push(`compute ${g.compute_capability}`);
  if (g.d3d12_feature_level) facts.push(`D3D12 ${g.d3d12_feature_level.replace("_", ".")}`);
  return facts;
}

/** CPU instruction-set badges ("AVX2", "AVX-512", "VNNI"). */
export function cpuFeatureBadges(cpu: SysSystemInfo["cpu"] | null | undefined): string[] {
  if (!cpu) return [];
  const out: string[] = [];
  if (cpu.avx2) out.push("AVX2");
  if (cpu.avx512) out.push("AVX-512");
  if (cpu.vnni) out.push("VNNI");
  return out;
}
