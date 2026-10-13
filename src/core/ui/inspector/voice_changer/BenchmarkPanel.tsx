// Voice-changer inspector (owned by P3-G): benchmark and tuning of RVC mode.
// - Autotune: sweeps execution provider x block x F0 method x precision x context on a bundled
//   clip, picks the fastest configuration that stays real time with full fidelity, stores it for this
//   GPU and driver; "Apply" puts it into the settings.
// - Benchmark this PC: a paced run of the selected voice with the current settings (compute
//   p50 / p95 / p99, real-time factor, latency, gaps, CPU, VRAM, the EP it ran on).
// - Live recording: the per-block timeline of a real session (VRChat, OBS), saved as a report
//   folder with no audio and no paths, to compare PCs.
import { autotuneParamRows, type BenchSummary } from "@/core/services/voice_changer/tuning";
import { LIVE_RECORD_MAX_SECONDS } from "@/core/services/voice_changer/bench";
import { epDisplayName, formatMs } from "@/core/services/voice_changer/metrics";
import type { VcAutotuneResult } from "@/core/services/voice_changer/ipc";
import classNames from "classnames";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { VcDetails, VcNote } from "./common";
import { VcStat } from "./Metrics";

function fmt(n: number | null | undefined, digits = 1): string {
    return n == null || !Number.isFinite(n) ? "–" : n.toFixed(digits);
}

const Progress: FC<{ fraction: number | null | undefined; label: string; message?: string | null }> = ({ fraction, label, message }) => (
    <div className="flex flex-col gap-0.5">
        <progress
            className="progress progress-primary h-1.5 w-full"
            value={fraction != null && fraction > 0 ? Math.round(fraction * 1000) : undefined}
            max={1000}
            aria-label={label}
        />
        {message ? <span className="break-words text-xs text-base-content/60">{message}</span> : null}
    </div>
);

const AutotuneResultCard: FC<{ result: VcAutotuneResult; applied: boolean; retuneDisabled: boolean }> = ({ result, applied, retuneDisabled }) => {
    const { t } = useTranslation();
    const bench = window.ApiServer.voiceChanger.bench;
    const ms = t("voice_changer.unit_ms");
    const rows = autotuneParamRows(result.params);
    const when = result.created_unix_ms > 0 ? new Date(result.created_unix_ms).toLocaleString() : "";
    const stale = result.stale ?? [];
    return (
        <div className="flex flex-col gap-1.5 rounded-lg border border-base-content/10 p-2 text-xs">
            <span className="text-base-content/70">
                {t("voice_changer.autotune_result_for", { gpu: result.gpu ?? t("voice_changer.autotune_cpu_only"), when })}
            </span>
            {stale.length > 0 ? (
                <div className="flex flex-col gap-1">
                    <VcNote tone="warn">{t("voice_changer.autotune_stale", { reasons: stale.join("; ") })}</VcNote>
                    <button type="button" className="btn btn-xs self-start" disabled={retuneDisabled} onClick={() => void bench.runAutotune(true)}>
                        {t("voice_changer.autotune_retune")}
                    </button>
                </div>
            ) : null}
            {result.met_limit === false ? <VcNote tone="warn">{t("voice_changer.autotune_not_realtime")}</VcNote> : null}
            <div className="grid grid-cols-1 gap-y-0.5 font-mono">
                <VcStat label={t("voice_changer.bench_rtf_p99")} value={fmt(result.rtf_p99, 2)} />
                <VcStat label={t("voice_changer.bench_block_p95")} value={`${formatMs(result.block_ms_p95)} ${ms}`} />
                {result.latency_ms_p50 != null ? (
                    <VcStat
                        label={t("voice_changer.bench_latency")}
                        value={`${formatMs(result.latency_ms_p50)}${result.latency_ms_p99 != null ? ` / ${formatMs(result.latency_ms_p99)}` : ""} ${ms}`}
                    />
                ) : null}
                {result.fidelity != null ? <VcStat label={t("voice_changer.autotune_fidelity")} value={fmt(result.fidelity, 3)} /> : null}
                {result.execution_provider_active ? (
                    <VcStat label={t("voice_changer.bench_ep")} value={epDisplayName(result.execution_provider_active)} />
                ) : null}
            </div>
            {rows.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                    {rows.map((r) => (
                        <span key={r.key} className="badge badge-ghost badge-xs whitespace-nowrap font-mono">
                            {`${r.key}=${r.value}`}
                        </span>
                    ))}
                </div>
            ) : null}
            {result.notes.map((n) => (
                <VcNote key={n}>{n}</VcNote>
            ))}
            {result.report_dir ? (
                <span className="select-all break-all text-base-content/50" title={t("voice_changer.bench_report_folder")}>
                    {result.report_dir}
                </span>
            ) : null}
            <button type="button" className="btn btn-primary btn-xs self-start" disabled={applied} onClick={() => void bench.applyAutotune()}>
                {applied ? t("voice_changer.autotune_applied") : t("voice_changer.autotune_apply")}
            </button>
        </div>
    );
};

const BenchSummaryCard: FC<{ summary: BenchSummary }> = ({ summary: s }) => {
    const { t } = useTranslation();
    const ms = t("voice_changer.unit_ms");
    const ok = s.status === "passed" || s.status === "failed";
    return (
        <div className="flex flex-col gap-1.5 rounded-lg border border-base-content/10 p-2 text-xs">
            {ok ? (
                <div className="grid grid-cols-1 gap-y-0.5 font-mono">
                    <VcStat
                        label={t("voice_changer.bench_block_pct")}
                        value={`${fmt(s.blockP50)} / ${fmt(s.blockP95)} / ${fmt(s.blockP99)} ${ms}`}
                    />
                    <VcStat label={t("voice_changer.bench_rtf_p99")} value={fmt(s.rtfP99, 2)} tone={s.rtfP99 != null && s.rtfP99 > 1 ? "error" : s.rtfP99 != null && s.rtfP99 > 0.5 ? "warn" : undefined} />
                    <VcStat label={t("voice_changer.bench_latency")} value={`${fmt(s.latencyP50)} / ${fmt(s.latencyP99)} ${ms}`} />
                    <VcStat label={t("voice_changer.metrics_underruns_label")} value={fmt(s.underruns, 0)} tone={s.underruns ? "warn" : undefined} />
                    <VcStat label={t("voice_changer.metrics_dropped_label")} value={`${fmt(s.droppedMs, 0)} ${ms}`} tone={s.droppedMs ? "warn" : undefined} />
                    {s.cpuCores != null ? <VcStat label={t("voice_changer.bench_cpu")} value={t("voice_changer.bench_cpu_value", { cores: fmt(s.cpuCores, 2) })} /> : null}
                    {s.vramMb != null && s.vramMb > 0 ? (
                        <VcStat label={t("voice_changer.metrics_vram_label")} value={t("voice_changer.unit_mb_value", { value: Math.round(s.vramMb) })} />
                    ) : null}
                    {s.epActive ? <VcStat label={t("voice_changer.bench_ep")} value={epDisplayName(s.epActive)} /> : null}
                </div>
            ) : null}
            {s.epActive && s.epRequested && s.epActive !== s.epRequested ? (
                <VcNote tone="warn">
                    {t("voice_changer.ep_fallback", { requested: epDisplayName(s.epRequested), active: epDisplayName(s.epActive) })}{" "}
                    {s.epReason ? t("voice_changer.ep_fallback_reason", { reason: s.epReason }) : null}
                </VcNote>
            ) : null}
            {ok && s.rtfP99 != null ? (
                <VcNote tone={s.rtfP99 > 1 ? "error" : s.rtfP99 > 0.5 ? "warn" : "muted"}>
                    {s.rtfP99 > 1
                        ? t("voice_changer.bench_verdict_slow")
                        : s.rtfP99 > 0.5
                          ? t("voice_changer.bench_verdict_tight")
                          : t("voice_changer.bench_verdict_ok")}
                </VcNote>
            ) : null}
            {s.warnings.map((w) => (
                <VcNote key={w} tone="warn">
                    {w}
                </VcNote>
            ))}
            {s.error ? <VcNote tone="error">{s.error}</VcNote> : null}
        </div>
    );
};

/** Seconds since `startedAt`, ticking once a second while `active`. */
function useElapsed(startedAt: number | null, active: boolean): number {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        if (!active) return;
        const id = window.setInterval(() => setNow(Date.now()), 1000);
        return () => window.clearInterval(id);
    }, [active]);
    return startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0;
}

function mmss(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

export const BenchmarkPanel: FC = () => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const snap = useSnapshot(vcApi.state);
    const b = useSnapshot(vcApi.bench.state);
    const caps = snap.capabilities;
    const running = snap.isRunning;
    const hasVoice = snap.voiceChanger.rvcSlotDir.trim() !== "";
    const busy = b.autotune.running || b.benchmark.running;
    const elapsed = useElapsed(b.liveRecord.startedAt, b.liveRecord.recording);
    const blockedReason = !hasVoice
        ? t("voice_changer.bench_needs_voice")
        : running
          ? t("voice_changer.bench_stop_first")
          : null;

    return (
        <VcDetails titleKey="voice_changer.section_benchmark" infoKey="voice_changer.section_benchmark_info">
            {caps.autotune ? (
                <div className="flex flex-col gap-1.5">
                    <span className="text-xs font-semibold uppercase tracking-wide text-base-content/55">{t("voice_changer.autotune_title")}</span>
                    <VcNote>{t("voice_changer.autotune_body_v2")}</VcNote>
                    {b.autotune.running ? (
                        <>
                            <Progress
                                fraction={b.autotune.progress?.fraction}
                                label={t("voice_changer.autotune_title")}
                                message={b.autotune.progress?.candidate ?? b.autotune.progress?.message ?? t("voice_changer.autotune_preparing")}
                            />
                            <button type="button" className="btn btn-xs self-start" onClick={() => void vcApi.bench.cancelAutotune()}>
                                {t("voice_changer.bench_cancel")}
                            </button>
                        </>
                    ) : (
                        <div className="flex flex-wrap gap-1">
                            <button
                                type="button"
                                className="btn btn-primary btn-xs"
                                disabled={busy || !!blockedReason}
                                onClick={() => void vcApi.bench.runAutotune(true)}
                            >
                                {t("voice_changer.autotune_quick")}
                            </button>
                            <button type="button" className="btn btn-xs" disabled={busy || !!blockedReason} onClick={() => void vcApi.bench.runAutotune(false)}>
                                {t("voice_changer.autotune_full")}
                            </button>
                        </div>
                    )}
                    {blockedReason && !b.autotune.running ? <VcNote>{blockedReason}</VcNote> : null}
                    {b.autotune.error ? <VcNote tone="error">{t("voice_changer.autotune_failed", { message: b.autotune.error })}</VcNote> : null}
                    {b.autotune.result && !b.autotune.running ? (
                        <AutotuneResultCard
                            result={b.autotune.result as VcAutotuneResult}
                            applied={b.autotune.applied}
                            retuneDisabled={busy || !!blockedReason}
                        />
                    ) : null}
                </div>
            ) : null}

            <div className="flex flex-col gap-1.5">
                <span className="text-xs font-semibold uppercase tracking-wide text-base-content/55">{t("voice_changer.bench_title")}</span>
                <VcNote>{t("voice_changer.bench_body")}</VcNote>
                {b.benchmark.running ? (
                    <>
                        <Progress fraction={b.benchmark.progress?.fraction} label={t("voice_changer.bench_title")} message={b.benchmark.progress?.message} />
                        <button type="button" className="btn btn-xs self-start" onClick={() => void vcApi.bench.cancelBenchmark()}>
                            {t("voice_changer.bench_cancel")}
                        </button>
                    </>
                ) : (
                    <button
                        type="button"
                        className="btn btn-xs self-start"
                        disabled={busy || !!blockedReason}
                        onClick={() => void vcApi.bench.runBenchmark()}
                    >
                        {t("voice_changer.bench_run")}
                    </button>
                )}
                {blockedReason && !b.benchmark.running && !caps.autotune ? <VcNote>{blockedReason}</VcNote> : null}
                {b.benchmark.error && !b.benchmark.summary?.error ? (
                    <VcNote tone="error">{t("voice_changer.bench_failed", { message: b.benchmark.error })}</VcNote>
                ) : null}
                {b.benchmark.summary && !b.benchmark.running ? <BenchSummaryCard summary={b.benchmark.summary as BenchSummary} /> : null}
            </div>

            {caps.live_record ? (
                <div className="flex flex-col gap-1.5">
                    <span className="text-xs font-semibold uppercase tracking-wide text-base-content/55">{t("voice_changer.live_record_title")}</span>
                    <VcNote>{t("voice_changer.live_record_body")}</VcNote>
                    {b.liveRecord.recording ? (
                        <div className="flex items-center gap-2">
                            <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-error" aria-hidden />
                            <span className="flex-1 text-xs tabular-nums">
                                {t("voice_changer.live_record_elapsed", { elapsed: mmss(elapsed), max: mmss(b.liveRecord.maxSeconds) })}
                            </span>
                            <button type="button" className="btn btn-xs" onClick={() => void vcApi.bench.stopLiveRecord()}>
                                {t("voice_changer.live_record_stop")}
                            </button>
                        </div>
                    ) : (
                        <button
                            type="button"
                            className="btn btn-xs self-start"
                            disabled={!running}
                            onClick={() => void vcApi.bench.startLiveRecord(LIVE_RECORD_MAX_SECONDS)}
                        >
                            {t("voice_changer.live_record_start")}
                        </button>
                    )}
                    {!running && !b.liveRecord.recording ? <VcNote>{t("voice_changer.live_record_needs_running")}</VcNote> : null}
                    {b.liveRecord.error ? <VcNote tone="error">{t("voice_changer.live_record_failed", { message: b.liveRecord.error })}</VcNote> : null}
                    {b.liveRecord.summary ? (
                        <div className={classNames("flex flex-col gap-1 rounded-lg border border-base-content/10 p-2 text-xs")}>
                            <div className="grid grid-cols-1 gap-y-0.5 font-mono">
                                <VcStat label={t("voice_changer.live_record_duration")} value={mmss(Math.round(b.liveRecord.summary.seconds))} />
                                {"latency_ms_p50" in b.liveRecord.summary.metrics ? (
                                    <VcStat
                                        label={t("voice_changer.bench_latency")}
                                        value={`${fmt(b.liveRecord.summary.metrics.latency_ms_p50)} / ${fmt(b.liveRecord.summary.metrics.latency_ms_p99)} ${t("voice_changer.unit_ms")}`}
                                    />
                                ) : null}
                                {"block_ms_p99" in b.liveRecord.summary.metrics ? (
                                    <VcStat
                                        label={t("voice_changer.bench_block_p99")}
                                        value={`${fmt(b.liveRecord.summary.metrics.block_ms_p99)} ${t("voice_changer.unit_ms")}`}
                                    />
                                ) : null}
                                {"underruns" in b.liveRecord.summary.metrics ? (
                                    <VcStat label={t("voice_changer.metrics_underruns_label")} value={fmt(b.liveRecord.summary.metrics.underruns, 0)} />
                                ) : null}
                                {"dropped_ms" in b.liveRecord.summary.metrics ? (
                                    <VcStat
                                        label={t("voice_changer.metrics_dropped_label")}
                                        value={`${fmt(b.liveRecord.summary.metrics.dropped_ms, 0)} ${t("voice_changer.unit_ms")}`}
                                    />
                                ) : null}
                            </div>
                            {(b.liveRecord.summary.notes ?? []).map((n) => (
                                <VcNote key={n}>{n}</VcNote>
                            ))}
                            <span className="text-base-content/60">{t("voice_changer.live_record_saved")}</span>
                            <span className="select-all break-all text-base-content/50">{b.liveRecord.summary.report_dir}</span>
                        </div>
                    ) : null}
                </div>
            ) : null}
        </VcDetails>
    );
};
