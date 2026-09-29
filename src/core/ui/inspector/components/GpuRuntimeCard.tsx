/**
 * GPU runtime card (Models & Runtimes tab): graphics cards and drivers, the primary GPU's video
 * memory (what Sigil holds against the Windows budget, what every app uses, Sigil's claims), which
 * GPU execution providers this PC can run and why not, the CPU, and the model sizes the resource
 * governor recommends for this PC right now.
 *
 * Owned by P1-D2. Reads `window.ApiServer.sysinfo` / `window.ApiServer.resources` (and the
 * diagnostics NVIDIA-pack status, read-only). Desktop only: in the web build it shows a short note.
 */
import { isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, memo, ReactNode, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { RiCpuFill, RiRefreshLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { diagnosticsState, refreshPackStatus } from "@/core/services/diagnostics";
import {
  claimsBySubsystem,
  deviceUsage,
  modelLabel,
  RECOMMENDATION_TOPICS,
  sigilUsage,
  type ResRecommendation,
  type ResSnapshot,
} from "@/core/services/resources";
import {
  cpuFeatureBadges,
  eligibilityParams,
  EP_KEYS,
  EP_LABELS,
  epStatus,
  formatMb,
  formatVramNominal,
  gpuFacts,
  gpuProbeState,
  hardwareGpus,
  type EpKey,
  type SysGpuInfo,
  type SysSystemInfo,
} from "@/core/services/sysinfo";

const Section: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <div className="flex flex-col gap-1.5">
    <span className="text-xs font-semibold opacity-80">{title}</span>
    {children}
  </div>
);

const Meter: FC<{ fraction: number; label: string; tone?: "primary" | "neutral" }> = ({ fraction, label, tone = "primary" }) => (
  <div className="flex flex-col gap-0.5">
    <progress
      className={classNames("progress w-full h-1.5", tone === "primary" ? "progress-primary" : "progress-neutral")}
      value={Math.round(fraction * 1000)}
      max={1000}
      aria-label={label}
    />
    <span className="text-xs opacity-70 leading-snug">{label}</span>
  </div>
);

const GpuRow: FC<{ gpu: SysGpuInfo; primary: boolean; info: SysSystemInfo }> = ({ gpu, primary, info }) => {
  const { t } = useTranslation();
  const facts = gpuFacts(gpu);
  const driver =
    gpu.vendor === "nvidia"
      ? info.cuda_driver_version && gpu.nvidia_driver
        ? t("models.gpu.driver_cuda", { driver: gpu.nvidia_driver, cuda: info.cuda_driver_version })
        : gpu.nvidia_driver
          ? t("models.gpu.driver", { driver: gpu.nvidia_driver })
          : null
      : gpu.driver_version
        ? t("models.gpu.driver", { driver: gpu.driver_version })
        : null;
  return (
    <div className="flex flex-col text-xs leading-snug">
      <div className="flex items-center gap-1.5 min-w-0">
        <span className={classNames("truncate", primary ? "font-semibold" : "opacity-80")} title={gpu.name}>
          {gpu.name}
        </span>
        {primary && <span className="badge badge-xs badge-primary shrink-0">{t("models.gpu.primary")}</span>}
        {gpu.integrated && <span className="badge badge-xs badge-ghost shrink-0">{t("models.gpu.integrated")}</span>}
        <span className="ms-auto shrink-0 font-mono opacity-70">{formatVramNominal(gpu.dedicated_vram_mb)}</span>
      </div>
      {facts.length > 0 && <span className="opacity-60">{facts.join(" · ")}</span>}
      {driver && <span className="opacity-60">{driver}</span>}
    </div>
  );
};

const MemorySection: FC<{ snapshot: ResSnapshot | null }> = ({ snapshot }) => {
  const { t } = useTranslation();
  if (!snapshot || snapshot.budget.adapter_index == null) return null;
  const sigil = sigilUsage(snapshot.budget);
  const device = deviceUsage(snapshot.budget);
  const claims = claimsBySubsystem(snapshot);
  return (
    <Section title={t("models.gpu.vram_title")}>
      {sigil.budgetMb > 0 && (
        <Meter
          fraction={sigil.fraction}
          label={t("models.gpu.vram_sigil", { used: formatMb(sigil.usedMb), budget: formatMb(sigil.budgetMb) })}
        />
      )}
      {device && (
        <Meter
          tone="neutral"
          fraction={device.fraction}
          label={t("models.gpu.vram_device", { used: formatMb(device.usedMb), total: formatMb(device.totalMb) })}
        />
      )}
      {claims.length > 0 && (
        <div className="flex flex-col gap-0.5">
          <span className="text-xs opacity-60">{t("models.gpu.claims_title")}</span>
          {claims.map((c) => (
            <div key={c.subsystem} className="flex items-center justify-between gap-2 text-xs">
              <span className="opacity-80 truncate">{t(`models.gpu.subsystem_${c.subsystem}`)}</span>
              <span className="font-mono opacity-70 shrink-0">{formatMb(c.mb)}</span>
            </div>
          ))}
        </div>
      )}
      {snapshot.vc_running && (
        <span className="badge badge-sm badge-info badge-outline self-start">{t("models.gpu.vc_running")}</span>
      )}
    </Section>
  );
};

const EP_BADGE: Record<ReturnType<typeof epStatus>, string> = {
  ok: "badge-success",
  unverified: "badge-warning",
  no: "badge-ghost",
};

const EpRow: FC<{ ep: EpKey; info: SysSystemInfo; cudaRuntimeMissing: boolean }> = ({ ep, info, cudaRuntimeMissing }) => {
  const { t, i18n } = useTranslation();
  const e = info.eps[ep];
  const status = epStatus(e);
  const codeKey = e?.code ? `models.gpu.ep_code_${e.code}` : null;
  // Known codes are translated; an unknown code (newer backend) falls back to Rust's English reason.
  const detail =
    status === "ok"
      ? null
      : codeKey && i18n.exists(codeKey)
        ? t(codeKey, eligibilityParams(ep, info))
        : (e?.reason ?? null);
  return (
    <div className="flex flex-col text-xs leading-snug">
      <div className="flex items-center justify-between gap-2">
        <span className="opacity-80">{EP_LABELS[ep]}</span>
        <span className={classNames("badge badge-sm shrink-0", EP_BADGE[status])}>{t(`models.gpu.ep_${status}`)}</span>
      </div>
      {detail && <span className="opacity-60 break-words">{detail}</span>}
      {ep === "cuda" && status !== "no" && cudaRuntimeMissing && (
        <span className="text-info break-words">{t("models.gpu.cuda_runtime_missing")}</span>
      )}
    </div>
  );
};

const RecommendationRow: FC<{ rec: ResRecommendation }> = ({ rec }) => {
  const { t, i18n } = useTranslation();
  const reasonKey = rec.reason_code ? `models.gpu.rec_reason_${rec.reason_code}` : null;
  const reason =
    reasonKey && i18n.exists(reasonKey)
      ? t(reasonKey, {
          free: formatMb(rec.available_mb ?? 0),
          need: formatMb(rec.vram_mb ?? 0),
        })
      : rec.reason;
  const onGpu = rec.device !== "cpu";
  return (
    <div className="flex flex-col text-xs leading-snug">
      <span className="opacity-60">{t(`models.gpu.rec_topic_${rec.topic}`)}</span>
      <div className="flex items-center gap-1.5 min-w-0">
        <span className="font-semibold truncate" title={rec.choice}>
          {modelLabel(rec.choice)}
        </span>
        <span className={classNames("badge badge-xs shrink-0", onGpu ? "badge-primary" : "badge-neutral")}>
          {t(onGpu ? "models.gpu.rec_device_gpu" : "models.gpu.rec_device_cpu")}
        </span>
        {onGpu && !!rec.vram_mb && <span className="ms-auto shrink-0 font-mono opacity-60">{formatMb(rec.vram_mb)}</span>}
      </div>
      <span className="opacity-60 break-words">{reason}</span>
    </div>
  );
};

export const GpuRuntimeCard: FC = memo(() => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const sys = useSnapshot(window.ApiServer.sysinfo.state);
  const res = useSnapshot(window.ApiServer.resources.state);
  const pack = useSnapshot(diagnosticsState).pack;

  useEffect(() => {
    if (!tauri) return;
    void window.ApiServer.sysinfo.load();
    const stop = window.ApiServer.resources.watch();
    if (!diagnosticsState.pack) void refreshPackStatus();
    return stop;
  }, [tauri]);

  const info = sys.info as SysSystemInfo | null;
  const snapshot = res.snapshot as ResSnapshot | null;
  const gpus = hardwareGpus(info);
  // A driver that did not answer is not "no graphics card" (the English probe notes go in the tooltip).
  const probe = gpuProbeState(info);
  const probeNotes = info?.notes?.length ? info.notes.join("\n") : undefined;
  const cudaRuntimeMissing = !!pack && !pack.installed && !pack.pending_removal;
  const recs = RECOMMENDATION_TOPICS.map((topic) => res.recommendations[topic] as ResRecommendation | undefined).filter(
    (r): r is ResRecommendation => !!r,
  );

  return (
    <div className="rounded-box border border-base-content/10 bg-base-200/40 p-3 flex flex-col gap-3">
      <div className="flex items-center gap-2 font-semibold text-sm">
        <RiCpuFill className="text-lg text-primary shrink-0" />
        <span className="truncate">{t("models.gpu_runtime_title")}</span>
        {tauri && (
          <button
            className="btn btn-ghost btn-xs ms-auto gap-1"
            disabled={sys.loading}
            onClick={() => void window.ApiServer.sysinfo.refresh()}
            title={t("models.gpu.refresh")}
            aria-label={t("models.gpu.refresh")}
          >
            <RiRefreshLine className={classNames(sys.loading && "animate-spin")} />
          </button>
        )}
      </div>

      {!tauri ? (
        <p className="text-xs text-base-content/60 leading-relaxed">{t("models.gpu.desktop_only")}</p>
      ) : !info ? (
        <p className="text-xs text-base-content/60 leading-relaxed">
          {sys.error ? t("models.gpu.error", { error: sys.error }) : t("models.gpu.checking")}
        </p>
      ) : (
        <>
          {sys.error && <div className="alert alert-warning alert-soft text-xs leading-snug break-words">{t("models.gpu.error", { error: sys.error })}</div>}

          <Section title={t("models.gpu.gpus_title")}>
            {probe === "missing" ? (
              <div className="alert alert-warning alert-soft text-xs leading-snug break-words" title={probeNotes}>
                {t("models.gpu.gpu_probe_missing")}
              </div>
            ) : gpus.length === 0 ? (
              <span className="text-xs opacity-60">{t("models.gpu.no_gpu")}</span>
            ) : (
              gpus.map((g) => <GpuRow key={g.luid || g.index} gpu={g} primary={g.index === info.primary_gpu} info={info} />)
            )}
            {probe === "previous" && (
              <span className="text-xs text-warning leading-snug break-words" title={probeNotes}>
                {t("models.gpu.gpu_probe_previous")}
              </span>
            )}
          </Section>

          <MemorySection snapshot={snapshot} />

          <Section title={t("models.gpu.ep_title")}>
            {EP_KEYS.map((ep) => (
              <EpRow key={ep} ep={ep} info={info} cudaRuntimeMissing={cudaRuntimeMissing} />
            ))}
          </Section>

          <Section title={t("models.gpu.cpu_title")}>
            <div className="flex flex-col text-xs leading-snug">
              <span className="opacity-80 break-words">{info.cpu.brand || "?"}</span>
              <span className="opacity-60">
                {info.cpu.performance_cores && info.cpu.performance_cores < info.cpu.physical_cores
                  ? t("models.gpu.cpu_cores_hybrid", {
                      physical: info.cpu.physical_cores,
                      performance: info.cpu.performance_cores,
                      logical: info.cpu.logical_cores,
                    })
                  : t("models.gpu.cpu_cores", { physical: info.cpu.physical_cores, logical: info.cpu.logical_cores })}
                {info.total_ram_mb > 0 && ` · ${t("models.gpu.ram", { size: formatVramNominal(info.total_ram_mb) })}`}
              </span>
              {cpuFeatureBadges(info.cpu).length > 0 && (
                <div className="flex flex-wrap gap-1 pt-0.5">
                  {cpuFeatureBadges(info.cpu).map((b) => (
                    <span key={b} className="badge badge-xs badge-ghost font-mono">
                      {b}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </Section>

          <Section title={t("models.gpu.rec_title")}>
            {recs.length === 0 ? (
              <span className="text-xs opacity-60">{t("models.gpu.rec_loading")}</span>
            ) : (
              recs.map((r) => <RecommendationRow key={r.topic} rec={r} />)
            )}
            <span className="text-xs opacity-50 leading-snug">{t("models.gpu.rec_hint")}</span>
          </Section>
        </>
      )}
    </div>
  );
});
GpuRuntimeCard.displayName = "GpuRuntimeCard";

export default GpuRuntimeCard;
