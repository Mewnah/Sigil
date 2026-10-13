// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Live metrics: the execution provider in use (with the reason of a fallback), the latency panel
// (processing p50 / p95 / p99, mic-to-speaker, skipped audio, gaps, VRAM, the governor's quality
// level), the engine levers really in use, and TensorRT engine builds.
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import { VC_LEVERS } from "@/core/services/voice_changer/contract";
import {
    assessMetrics,
    componentShortName,
    effectiveLatencyBudgetMs,
    epDisplayName,
    epStatus,
    formatChunkMs,
    formatMs,
    GOVERNOR_LOWEST_LEVEL,
    leversInUse,
} from "@/core/services/voice_changer/metrics";
import classNames from "classnames";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";

/** Model execution provider in use; a fallback is a warning with Rust's reason. */
export const VcEpStatus: FC<{ metrics: VoiceChangerMetrics }> = ({ metrics }) => {
    const { t } = useTranslation();
    const status = epStatus(metrics.ep);
    if (!status) return null;
    const accelerator = metrics.levers?.accelerator ?? "none";
    if (status.kind === "granted") {
        return (
            <div
                className={classNames(
                    "text-xs",
                    status.active === "cpu" ? "text-base-content/60" : "text-success",
                )}
            >
                {t("voice_changer.ep_running_on", {
                    ep: accelerator !== "none" ? epDisplayName(accelerator) : epDisplayName(status.active),
                })}
            </div>
        );
    }
    return (
        <div role="status" className="alert alert-warning alert-soft flex flex-col items-start gap-1 text-xs leading-snug">
            <span className="font-medium">
                {t("voice_changer.ep_fallback", {
                    requested: epDisplayName(status.requested),
                    active: epDisplayName(status.active),
                })}
            </span>
            <span className="break-words opacity-80">
                {t("voice_changer.ep_fallback_reason", {
                    reason: status.reason ?? t("voice_changer.ep_reason_unknown"),
                })}
            </span>
        </div>
    );
};

export const VcStat: FC<{ label: string; value: string; tone?: "warn" | "error" }> = ({ label, value, tone }) => (
    <div className="flex min-w-0 items-baseline justify-between gap-2">
        <span className="min-w-0 truncate font-sans text-base-content/55">{label}</span>
        <span
            className={classNames(
                "shrink-0 tabular-nums",
                tone === "error" ? "text-error" : tone === "warn" ? "text-warning" : "text-base-content/80",
            )}
        >
            {value}
        </span>
    </div>
);

/** Live latency metrics of the running engine, with plain-language warnings. */
export const VcMetricsPanel: FC<{
    metrics: VoiceChangerMetrics;
    prev: VoiceChangerMetrics | null;
    budgetMs: number;
}> = ({ metrics: m, prev, budgetMs }) => {
    const { t } = useTranslation();
    const ms = t("voice_changer.unit_ms");
    const a = assessMetrics(m, budgetMs, prev);
    // The budget the engine applies: a voice that converts a long chunk per call needs more than a small stored one.
    const budgetInForce = effectiveLatencyBudgetMs(budgetMs, m);
    const callTone = a.realtime === "slow" ? "error" : a.realtime === "spikes" ? "warn" : undefined;
    if (a.starting) {
        return <div className="text-xs text-base-content/60">{t("voice_changer.metrics_starting")}</div>;
    }
    const processValue =
        m.process_p99_ms != null
            ? `${formatMs(m.process_ms)} / ${formatMs(m.process_p99_ms)} ${ms}`
            : `${formatMs(m.process_ms)} ${ms}`;
    const governed = m.governor_level != null && m.governor_level > 0;
    const warnings: { key: string; tone: "warn" | "error" }[] = [];
    if (a.realtime === "slow") warnings.push({ key: "voice_changer.metrics_slow_warn", tone: "error" });
    else if (a.realtime === "spikes") warnings.push({ key: "voice_changer.metrics_spikes_warn", tone: "warn" });
    if (a.droppingNow || a.backlogHigh) warnings.push({ key: "voice_changer.metrics_dropping_warn", tone: "warn" });
    if (a.underrunNow) warnings.push({ key: "voice_changer.metrics_underrun_warn", tone: "warn" });
    if (governed) warnings.push({ key: "voice_changer.metrics_governor_warn", tone: "warn" });
    return (
        <div className="flex flex-col gap-1.5">
            <div className="grid grid-cols-1 gap-y-0.5 rounded-lg border border-base-content/10 px-2.5 py-2 font-mono text-xs">
                {m.e2e_ms != null ? (
                    <VcStat label={t("voice_changer.metrics_e2e_label")} value={`${formatMs(m.e2e_ms)} ${ms}`} />
                ) : null}
                <VcStat
                    label={m.process_p99_ms != null ? t("voice_changer.metrics_process_label") : t("voice_changer.metrics_process_avg_label")}
                    value={processValue}
                    // A voice judged by its model calls shows that verdict on the calls row, not on the per-tick numbers.
                    tone={a.byCalls ? undefined : callTone}
                />
                {m.call_p50_ms != null && m.call_p99_ms != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_call_label")}
                        value={`${formatMs(m.call_p50_ms)} / ${formatMs(m.call_p99_ms)} ${ms}`}
                        tone={a.byCalls ? callTone : undefined}
                    />
                ) : null}
                {m.process_p50_ms != null && m.process_p95_ms != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_process_pct_label")}
                        value={`${formatMs(m.process_p50_ms)} / ${formatMs(m.process_p95_ms)} ${ms}`}
                    />
                ) : null}
                {m.engine_block_ms != null && m.engine_block_ms > 0 ? (
                    <VcStat label={t("voice_changer.metrics_chunk_label")} value={`${formatChunkMs(m.engine_block_ms)} ${ms}`} />
                ) : null}
                <VcStat label={t("voice_changer.metrics_device_buffer_label")} value={`${formatMs(m.chunk_ms)} ${ms}`} />
                {m.backlog_ms != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_backlog_label")}
                        value={`${formatMs(m.backlog_ms)} / ${formatMs(budgetInForce)} ${ms}`}
                        tone={a.backlogHigh ? "warn" : undefined}
                    />
                ) : null}
                <VcStat label={t("voice_changer.metrics_buffer_label")} value={`${formatMs(m.buffer_ms)} ${ms}`} />
                {m.dropped_ms != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_dropped_label")}
                        value={`${formatMs(m.dropped_ms)} ${ms}`}
                        tone={a.droppingNow ? "warn" : undefined}
                    />
                ) : null}
                {m.underruns != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_underruns_label")}
                        value={String(m.underruns)}
                        tone={a.underrunNow ? "warn" : undefined}
                    />
                ) : null}
                {m.vram_mb != null && m.vram_mb > 0 ? (
                    <VcStat label={t("voice_changer.metrics_vram_label")} value={t("voice_changer.unit_mb_value", { value: Math.round(m.vram_mb) })} />
                ) : null}
                {m.governor_level != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_governor_label")}
                        value={
                            governed
                                ? t("voice_changer.governor_reduced", { level: m.governor_level, max: GOVERNOR_LOWEST_LEVEL })
                                : t("voice_changer.governor_full")
                        }
                        tone={governed ? "warn" : undefined}
                    />
                ) : null}
            </div>
            {warnings.map((w) => (
                <p
                    key={w.key}
                    className={classNames("text-xs leading-snug", w.tone === "error" ? "text-error" : "text-warning")}
                >
                    {t(w.key)}
                </p>
            ))}
            <VcLeversInUse metrics={m} />
        </div>
    );
};

/** The engine levers, accelerator and precision the running engine really uses, and why a requested one is not. */
export const VcLeversInUse: FC<{ metrics: VoiceChangerMetrics }> = ({ metrics }) => {
    const { t } = useTranslation();
    const levers = metrics.levers;
    if (!levers) return null;
    const used = leversInUse(levers);
    const notes = levers.notes;
    return (
        <details className="rounded-lg border border-base-content/10 text-xs [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden">
            <summary className="flex items-center gap-2 px-2.5 py-1.5 text-base-content/70 hover:bg-base-content/5">
                <span className="min-w-0 flex-1 truncate">
                    {t("voice_changer.levers_in_use_summary", { count: used.length, total: VC_LEVERS.length })}
                </span>
                {notes.length > 0 ? <span className="badge badge-warning badge-xs">{notes.length}</span> : null}
            </summary>
            <div className="flex flex-col gap-1.5 border-t border-base-content/10 px-2.5 py-2">
                <div className="flex flex-wrap gap-1">
                    {VC_LEVERS.map((l) => (
                        <span
                            key={l}
                            className={classNames("badge badge-sm", levers[l] ? "badge-success badge-outline" : "badge-ghost opacity-60")}
                        >
                            {t(`voice_changer.lever_name_${l}`)}
                        </span>
                    ))}
                </div>
                <span className="text-base-content/60">
                    {t("voice_changer.levers_precision_accelerator", {
                        precision: levers.precision,
                        accelerator: levers.accelerator === "none" ? t("voice_changer.accelerator_none") : epDisplayName(levers.accelerator),
                    })}
                </span>
                {notes.length > 0 ? (
                    <ul className="list-disc space-y-0.5 ps-4 text-warning">
                        {notes.map((n) => (
                            <li key={n} className="break-words">
                                {n}
                            </li>
                        ))}
                    </ul>
                ) : null}
            </div>
        </details>
    );
};

/** TensorRT engine builds of the starting engine (first start builds, later starts load the cache). */
export const VcTrtProgressPanel: FC = () => {
    const { t } = useTranslation();
    const trt = useSnapshot(window.ApiServer.voiceChanger.state).trt;
    if (trt.length === 0) return null;
    const building = trt.some((r) => r.stage === "building");
    return (
        <div role="status" className="flex flex-col gap-1.5 rounded-lg border border-info/30 bg-info/5 p-2.5 text-xs leading-snug">
            {building ? <span className="text-base-content/70">{t("voice_changer.trt_building_hint")}</span> : null}
            {trt.map((r) => {
                const name = componentShortName(r.component);
                const seconds = Math.round(r.elapsed_ms / 1000);
                return (
                    <div key={r.component} className="flex flex-col gap-0.5">
                        <span className={classNames("break-words", r.stage === "failed" ? "text-error" : "text-base-content/80")}>
                            {r.stage === "building"
                                ? t("voice_changer.trt_stage_building", { component: name, percent: Math.round(r.fraction * 100), seconds })
                                : r.stage === "cache_hit"
                                  ? t("voice_changer.trt_stage_cache_hit", { component: name })
                                  : r.stage === "done"
                                    ? t("voice_changer.trt_stage_done", { component: name, seconds })
                                    : t("voice_changer.trt_stage_failed", { component: name, message: r.message || t("voice_changer.ep_reason_unknown") })}
                        </span>
                        {r.stage === "building" ? (
                            <progress
                                className="progress progress-info h-1.5 w-full"
                                value={r.fraction > 0 ? Math.round(r.fraction * 1000) : undefined}
                                max={1000}
                                aria-label={name}
                            />
                        ) : null}
                    </div>
                );
            })}
        </div>
    );
};
