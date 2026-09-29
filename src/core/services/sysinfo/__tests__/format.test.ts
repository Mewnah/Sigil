import { describe, expect, it } from "vitest";
import {
  archLabel,
  cpuFeatureBadges,
  eligibilityParams,
  epStatus,
  formatMb,
  formatVramNominal,
  gpuFacts,
  gpuProbeState,
  hardwareGpus,
  mainNvidiaGpu,
  primaryGpu,
  type SysGpuInfo,
  type SysSystemInfo,
} from "../format";

function gpu(index: number, vendor: string, name: string, vram: number, extra: Partial<SysGpuInfo> = {}): SysGpuInfo {
  return {
    index,
    name,
    vendor,
    vendor_id: vendor === "nvidia" ? 0x10de : vendor === "amd" ? 0x1002 : 0x1414,
    device_id: 0,
    luid: `00000000:${index}`,
    dedicated_vram_mb: vram,
    shared_memory_mb: 16_000,
    software: vendor === "microsoft",
    driver_version: null,
    nvidia_driver: vendor === "nvidia" ? "616.56" : null,
    vram: null,
    arch: null,
    compute_capability: null,
    ...extra,
  };
}

const ok = { eligible: true, reason: null, code: "ok" };

/** The dev rig as the Rust probe reports it: iGPU first in DXGI order, the 4090 primary. */
const devRig: SysSystemInfo = {
  gpus: [
    gpu(0, "amd", "AMD Radeon(TM) Graphics", 485, { arch: "rdna2", integrated: true, d3d12_feature_level: "12_1" }),
    gpu(1, "nvidia", "NVIDIA GeForce RTX 4090", 24_138, { arch: "ada", compute_capability: "8.9", d3d12_feature_level: "12_2" }),
    gpu(2, "microsoft", "Microsoft Basic Render Driver", 0),
  ],
  primary_gpu: 1,
  nvidia_driver: "616.56",
  cuda_driver_version: "13.4",
  eps: { cuda: ok, tensorrt: ok, tensorrt_rtx: ok, directml: ok },
  cpu: { brand: "AMD Ryzen 7 7800X3D 8-Core Processor", physical_cores: 8, logical_cores: 16, avx2: true, avx512: true, vnni: true },
  total_ram_mb: 65_536,
  probed_at_unix_ms: 1,
};

describe("sysinfo format helpers", () => {
  it("finds the primary GPU and lists hardware GPUs primary-first", () => {
    expect(primaryGpu(devRig)?.name).toBe("NVIDIA GeForce RTX 4090");
    expect(hardwareGpus(devRig).map((g) => g.index)).toEqual([1, 0]);
    expect(mainNvidiaGpu(devRig)?.index).toBe(1);
    expect(primaryGpu(null)).toBeNull();
    expect(primaryGpu({ ...devRig, primary_gpu: null })).toBeNull();
    expect(hardwareGpus(undefined)).toEqual([]);
  });

  it("labels architectures and GPU facts", () => {
    expect(archLabel("ada")).toBe("Ada Lovelace");
    expect(archLabel("rdna3")).toBe("RDNA 3");
    expect(archLabel("gcn5")).toBe("Vega"); // Ryzen 4000G / 5000G APUs
    expect(archLabel("future-arch")).toBe("future-arch");
    expect(archLabel(null)).toBeNull();
    expect(gpuFacts(devRig.gpus[1])).toEqual(["Ada Lovelace", "compute 8.9", "D3D12 12.2"]);
    expect(gpuFacts(devRig.gpus[2])).toEqual([]);
  });

  it("formats VRAM as cards are sold and as measured", () => {
    expect(formatVramNominal(24_138)).toBe("24 GB");
    expect(formatVramNominal(8188)).toBe("8 GB");
    expect(formatVramNominal(12_282)).toBe("12 GB");
    expect(formatVramNominal(485)).toBe("485 MB");
    expect(formatVramNominal(31_927)).toBe("32 GB"); // 32 GB of RAM as Windows reports it
    expect(formatVramNominal(8200)).toBe("8 GB"); // just above a whole GB
    expect(formatVramNominal(6144)).toBe("6 GB");
    expect(formatVramNominal(1024)).toBe("1 GB");
    expect(formatMb(23_370)).toBe("22.8 GB");
    expect(formatMb(900)).toBe("900 MB");
    expect(formatMb(Number.NaN)).toBe("?");
    expect(formatMb(-1)).toBe("?");
  });

  it("maps eligibility to a status and interpolation values", () => {
    expect(epStatus(ok)).toBe("ok");
    expect(epStatus({ eligible: true, reason: "x", code: "unverified" })).toBe("unverified");
    expect(epStatus({ eligible: false, reason: "x", code: "gpu_too_old" })).toBe("no");
    expect(epStatus(undefined)).toBe("no");
    // Older backends without `code` still show eligible EPs as supported.
    expect(epStatus({ eligible: true, reason: null })).toBe("ok");

    expect(eligibilityParams("cuda", devRig)).toEqual({
      gpu: "NVIDIA GeForce RTX 4090",
      arch: "Ada Lovelace",
      driver: "616.56",
      cuda: "13.4",
      min: "GeForce RTX 20 / GTX 16",
    });
    expect(eligibilityParams("tensorrt_rtx", devRig).min).toBe("GeForce RTX 30");
    expect(eligibilityParams("directml", devRig).gpu).toBe("NVIDIA GeForce RTX 4090, AMD Radeon(TM) Graphics");
    expect(eligibilityParams("cuda", null)).toMatchObject({ gpu: "", driver: "?", cuda: "?" });
  });

  it("tells a driver that did not answer apart from a PC without a GPU", () => {
    expect(gpuProbeState(devRig)).toBe("complete"); // payloads from before the field existed
    expect(gpuProbeState({ ...devRig, gpu_probe: "complete" })).toBe("complete");
    expect(gpuProbeState({ ...devRig, gpu_probe: "previous" })).toBe("previous");
    expect(gpuProbeState({ ...devRig, gpus: [], primary_gpu: null, gpu_probe: "missing" })).toBe("missing");
    expect(gpuProbeState({ ...devRig, gpu_probe: "from-the-future" })).toBe("complete");
    expect(gpuProbeState(null)).toBe("complete");
  });

  it("lists CPU instruction-set badges", () => {
    expect(cpuFeatureBadges(devRig.cpu)).toEqual(["AVX2", "AVX-512", "VNNI"]);
    expect(cpuFeatureBadges({ ...devRig.cpu, avx512: false, vnni: false })).toEqual(["AVX2"]);
    expect(cpuFeatureBadges(null)).toEqual([]);
  });
});
