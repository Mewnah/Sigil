import { VoiceChangerState } from "@/core/services/voice_changer/schema";
import type { VoiceChangerMetrics, VoiceChangerModelSlot } from "@/core/services/voice_changer";
import {
    chunkFramesToMsAt48k,
    crossfadeMaxMs as crossfadeMaxMsFor,
    isVoiceChangerChunkFrames,
    VOICE_CHANGER_CHUNK_FRAMES,
} from "@/core/services/voice_changer/chunkFrames";
import {
    AUX_MIC_GATES,
    type AuxMicGate,
    clampLatencyBudgetMs,
    effectivePitchExtractor,
    GPU_ONLY_PITCH_EXTRACTORS,
    PITCH_EXTRACTORS,
    VC_NUMBER_RANGES,
    WORKER_PRIORITIES,
    type WorkerPriority,
} from "@/core/services/voice_changer/contract";
import { assessMetrics, epDisplayName, epStatus, formatMs } from "@/core/services/voice_changer/metrics";
import { findSlot } from "@/core/services/voice_changer/slots";
import classNames from "classnames";
import { FC, ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { appDataDir, join } from "@tauri-apps/api/path";
import { open } from "@tauri-apps/plugin-dialog";
import Inspector from "./components";
import { InputSelect, InputContainer, InputText } from "./components/input";
import { RiInformationLine, RiMagicFill, RiPlayFill, RiRefreshLine, RiStopFill } from "react-icons/ri";
import Tooltip from "../dropdown/Tooltip";
import { toast } from "react-toastify";

interface AudioDevice {
    id: string;
    name: string;
}

const LATENCY_BUDGET = VC_NUMBER_RANGES.latency_budget_ms;

/** Scroll/clipping region for the service inspector panel (SimpleBar + inspector content width). */
function getInspectorPopoverClip(el: HTMLElement): DOMRect {
    const scroll = el.closest(".simplebar-content-wrapper");
    if (scroll instanceof HTMLElement) return scroll.getBoundingClientRect();
    let p: HTMLElement | null = el.parentElement;
    while (p) {
        const cn = typeof p.className === "string" ? p.className : "";
        if (cn.includes("22rem") && cn.includes("flex-col")) {
            return p.getBoundingClientRect();
        }
        p = p.parentElement;
    }
    const bar = el.closest(".simplebar-wrapper") ?? el.closest(".overflow-y-auto");
    if (bar instanceof HTMLElement) return bar.getBoundingClientRect();
    return new DOMRect(0, 0, window.innerWidth, window.innerHeight);
}

const InspectorInfoButton: FC<{ text: string }> = ({ text }) => {
    const { t } = useTranslation();
    const [show, setShow] = useState(false);
    const wrapRef = useRef<HTMLDivElement>(null);
    const [box, setBox] = useState<{ top: number; left: number; width: number; maxHeight: number } | null>(null);

    useLayoutEffect(() => {
        if (!show) {
            setBox(null);
            return;
        }
        const el = wrapRef.current;
        if (!el) return;
        const place = () => {
            const br = el.getBoundingClientRect();
            const clip = getInspectorPopoverClip(el);
            const margin = 8;
            const width = Math.max(160, Math.min(280, clip.width - margin * 2));
            let left = br.right - width;
            left = Math.max(clip.left + margin, Math.min(left, clip.right - margin - width));

            const capH = Math.min(320, Math.floor(window.innerHeight * 0.55));
            let top = br.bottom + margin;
            let maxHeight = Math.min(capH, clip.bottom - margin - top);
            if (maxHeight < 100) {
                maxHeight = Math.min(capH, br.top - clip.top - margin * 2);
                top = Math.max(clip.top + margin, br.top - margin - maxHeight);
                maxHeight = Math.min(maxHeight, clip.bottom - margin - top);
            }
            maxHeight = Math.max(100, maxHeight);
            setBox({ top, left, width, maxHeight });
        };
        place();
        window.addEventListener("resize", place);
        window.addEventListener("scroll", place, true);
        return () => {
            window.removeEventListener("resize", place);
            window.removeEventListener("scroll", place, true);
        };
    }, [show]);

    return (
        <div ref={wrapRef} className="relative shrink-0">
            <button
                type="button"
                className="btn btn-ghost btn-xs btn-circle min-h-7 w-7 p-0"
                aria-label={t("voice_changer.info_learn_more")}
                aria-expanded={show}
                onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setShow((s) => !s);
                }}
            >
                <RiInformationLine className="text-lg text-base-content/70" />
            </button>
            {show && box
                ? createPortal(
                      <>
                          <button
                              type="button"
                              className="fixed inset-0 z-[10000] cursor-default bg-transparent"
                              aria-label={t("voice_changer.info_close_overlay")}
                              onClick={() => setShow(false)}
                          />
                          <div
                              className="fixed z-[10001] overflow-y-auto whitespace-pre-line rounded-lg border border-base-content/15 bg-base-100 p-3 text-xs font-normal text-base-content/80 shadow-lg"
                              style={{
                                  top: box.top,
                                  left: box.left,
                                  width: box.width,
                                  maxHeight: box.maxHeight,
                              }}
                          >
                              {text}
                          </div>
                      </>,
                      document.body,
                  )
                : null}
        </div>
    );
};

const VC_RANGE =
    "range range-sm range-primary h-6 w-full min-w-0 shrink-0 cursor-pointer [width:100%]";

/** Label on its own row; uniform full-width slider and end caps (matches inspector tone controls). */
const VcSliderField: FC<{
    labelKey: string;
    /** Optional (i) next to the label. */
    infoKey?: string;
    inactive?: boolean;
    controlsRow: ReactNode;
    rangeMin: number;
    rangeMax: number;
    rangeStep: number;
    rangeValue: number;
    onRangeChange: (v: number) => void;
    /** Fired when a drag ends or the slider loses focus (for values that need an engine restart). */
    onRangeCommit?: () => void;
    lowLabelKey: string;
    highLabelKey: string;
}> = ({
    labelKey,
    infoKey,
    inactive,
    controlsRow,
    rangeMin,
    rangeMax,
    rangeStep,
    rangeValue,
    onRangeChange,
    onRangeCommit,
    lowLabelKey,
    highLabelKey,
}) => {
    const { t } = useTranslation();
    return (
        <div className="flex w-full min-w-0 flex-col gap-2">
            {infoKey ? (
                <div className="flex min-w-0 items-center gap-1">
                    <div className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</div>
                    <InspectorInfoButton text={t(infoKey)} />
                </div>
            ) : (
                <div className="text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</div>
            )}
            <div
                className={classNames(
                    "flex w-full min-w-0 flex-col gap-2",
                    inactive && "pointer-events-none opacity-50",
                )}
            >
                <div className="flex min-h-8 w-full min-w-0 flex-wrap items-center gap-2">{controlsRow}</div>
                <input
                    type="range"
                    min={rangeMin}
                    max={rangeMax}
                    step={rangeStep}
                    value={rangeValue}
                    onChange={(e) => onRangeChange(parseFloat(e.target.value))}
                    onPointerUp={onRangeCommit}
                    onBlur={onRangeCommit}
                    className={VC_RANGE}
                />
                <div className="flex w-full min-w-0 justify-between gap-2 text-xs text-base-content/40">
                    <span className="min-w-0 truncate">{t(lowLabelKey)}</span>
                    <span className="min-w-0 truncate text-end">{t(highLabelKey)}</span>
                </div>
            </div>
        </div>
    );
};

/** Section title matching other service inspectors (`Inspector.SubHeader` + optional (i)). */
const VcSubHeader: FC<{ titleKey: string; infoKey?: string }> = ({ titleKey, infoKey }) => {
    const { t } = useTranslation();
    return (
        <Inspector.SubHeader>
            <span className="min-w-0 flex-1 truncate">{t(titleKey)}</span>
            {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
        </Inspector.SubHeader>
    );
};

/** Collapsible group (same pattern as OBS inspector) so dense RVC/advanced options stay scannable. */
const VcDetails: FC<{
    titleKey: string;
    infoKey?: string;
    defaultOpen?: boolean;
    children: ReactNode;
}> = ({ titleKey, infoKey, defaultOpen = false, children }) => {
    const { t } = useTranslation();
    const [open, setOpen] = useState(defaultOpen);
    return (
        <details
            className="rounded-lg border border-base-content/10 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden"
            open={open}
            onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
        >
            <summary className="flex min-h-9 items-center gap-2 px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wide text-base-content/55 hover:bg-base-content/5">
                <span className="min-w-0 flex-1 truncate">{t(titleKey)}</span>
                {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
            </summary>
            <div className="flex flex-col gap-3 border-t border-base-content/10 px-2.5 py-2.5">{children}</div>
        </details>
    );
};

/** Device list of one direction; `null` until the first reply. */
function useAudioDevices(kind: "input" | "output"): AudioDevice[] | null {
    const [devices, setDevices] = useState<AudioDevice[] | null>(null);
    useEffect(() => {
        let cancelled = false;
        invoke<AudioDevice[]>(kind === "input" ? "plugin:audio|list_input_devices" : "plugin:audio|list_output_devices")
            .then((d) => {
                if (!cancelled) setDevices(Array.isArray(d) ? d : []);
            })
            .catch((e) => {
                console.error(e);
                if (!cancelled) setDevices([]);
            });
        return () => {
            cancelled = true;
        };
    }, [kind]);
    return devices;
}

/**
 * Options for the listed devices, plus the saved one when it is not connected right now (so a restored
 * setting is shown by name instead of a blank select, and is kept until the user picks another).
 */
function useDeviceOptions(devices: AudioDevice[] | null, saved: string, builtIn: readonly string[]) {
    const { t } = useTranslation();
    return useMemo(() => {
        const listed = (devices ?? []).map((d) => ({ label: d.name, value: d.id }));
        const missing = devices !== null && saved !== "" && !builtIn.includes(saved) && !listed.some((o) => o.value === saved);
        return missing ? [...listed, { label: t("voice_changer.device_not_connected", { name: saved }), value: saved }] : listed;
    }, [devices, saved, builtIn, t]);
}

/** Select value of "system default" (Radix Select shows its placeholder for an empty value). */
const INPUT_DEVICE_DEFAULT = "__vc_input_default__";
const NO_BUILT_IN: readonly string[] = [];

const VoiceInputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
    labelKey: string;
    defaultLabelKey: string;
}> = ({ value, onChange, labelKey, defaultLabelKey }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("input");
    const options = useDeviceOptions(devices, value, NO_BUILT_IN);

    return (
        <InputSelect
            label={labelKey}
            value={value || INPUT_DEVICE_DEFAULT}
            onValueChange={(v) => onChange(v === INPUT_DEVICE_DEFAULT ? "" : v)}
            options={[{ label: t(defaultLabelKey), value: INPUT_DEVICE_DEFAULT }, ...options]}
        />
    );
};

const MONITOR_OUTPUT_OFF = "__vc_monitor_off__";
const OUTPUT_BUILT_IN: readonly string[] = ["default"];

const VoiceMonitorOutputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
}> = ({ value, onChange }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("output");
    const options = useDeviceOptions(devices, value, OUTPUT_BUILT_IN);

    const selectValue = value === "" ? MONITOR_OUTPUT_OFF : value || "default";

    return (
        <InputSelect
            label="voice_changer.monitor_device"
            value={selectValue}
            onValueChange={(v) => {
                if (v === MONITOR_OUTPUT_OFF) onChange("");
                else onChange(v);
            }}
            options={[
                { label: t("voice_changer.monitor_off"), value: MONITOR_OUTPUT_OFF },
                { label: t("voice_changer.device_system_default"), value: "default" },
                ...options,
            ]}
        />
    );
};

const VoiceOutputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
    labelKey: string;
    defaultLabelKey: string;
}> = ({ value, onChange, labelKey, defaultLabelKey }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("output");
    const options = useDeviceOptions(devices, value || "default", OUTPUT_BUILT_IN);

    return (
        <InputSelect
            label={labelKey}
            value={value || "default"}
            onValueChange={onChange}
            options={[{ label: t(defaultLabelKey), value: "default" }, ...options]}
        />
    );
};

/**
 * Number input that keeps what the user types until blur / Enter and only then commits the clamped value
 * (a controlled input that clamps on every keystroke makes e.g. "250" impossible to type when min is 100).
 */
const VcDraftNumberInput: FC<{
    value: number;
    min: number;
    max: number;
    step: number;
    ariaLabel: string;
    onCommit: (v: number) => void;
}> = ({ value, min, max, step, ariaLabel, onCommit }) => {
    const [draft, setDraft] = useState<string | null>(null);
    const commit = () => {
        if (draft === null) return;
        const x = parseFloat(draft);
        setDraft(null);
        if (!Number.isNaN(x)) onCommit(Math.min(max, Math.max(min, x)));
    };
    return (
        <input
            type="number"
            className="input input-bordered input-sm w-24 shrink-0 font-mono"
            min={min}
            max={max}
            step={step}
            aria-label={ariaLabel}
            value={draft ?? String(value)}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
                if (e.key === "Enter") commit();
                if (e.key === "Escape") setDraft(null);
            }}
        />
    );
};

/** Model execution provider in use; a fallback is a warning with Rust's reason. */
const VcEpStatus: FC<{ metrics: VoiceChangerMetrics }> = ({ metrics }) => {
    const { t } = useTranslation();
    const status = epStatus(metrics.ep);
    if (!status) return null;
    if (status.kind === "granted") {
        return (
            <div
                className={classNames(
                    "text-xs",
                    status.active === "cpu" ? "text-base-content/60" : "text-success",
                )}
            >
                {t("voice_changer.ep_running_on", { ep: epDisplayName(status.active) })}
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

const VcStat: FC<{ label: string; value: string; tone?: "warn" | "error" }> = ({ label, value, tone }) => (
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
const VcMetricsPanel: FC<{
    metrics: VoiceChangerMetrics;
    prev: VoiceChangerMetrics | null;
    budgetMs: number;
}> = ({ metrics: m, prev, budgetMs }) => {
    const { t } = useTranslation();
    const ms = t("voice_changer.unit_ms");
    const a = assessMetrics(m, budgetMs, prev);
    if (a.starting) {
        return <div className="text-xs text-base-content/60">{t("voice_changer.metrics_starting")}</div>;
    }
    const processValue =
        m.process_p99_ms != null
            ? `${formatMs(m.process_ms)} / ${formatMs(m.process_p99_ms)} ${ms}`
            : `${formatMs(m.process_ms)} ${ms}`;
    const warnings: { key: string; tone: "warn" | "error" }[] = [];
    if (a.realtime === "slow") warnings.push({ key: "voice_changer.metrics_slow_warn", tone: "error" });
    else if (a.realtime === "spikes") warnings.push({ key: "voice_changer.metrics_spikes_warn", tone: "warn" });
    if (a.droppingNow || a.backlogHigh) warnings.push({ key: "voice_changer.metrics_dropping_warn", tone: "warn" });
    if (a.underrunNow) warnings.push({ key: "voice_changer.metrics_underrun_warn", tone: "warn" });
    return (
        <div className="flex flex-col gap-1.5">
            <div className="grid grid-cols-1 gap-y-0.5 rounded-lg border border-base-content/10 px-2.5 py-2 font-mono text-xs">
                <VcStat
                    label={m.process_p99_ms != null ? t("voice_changer.metrics_process_label") : t("voice_changer.metrics_process_avg_label")}
                    value={processValue}
                    tone={a.realtime === "slow" ? "error" : a.realtime === "spikes" ? "warn" : undefined}
                />
                <VcStat label={t("voice_changer.metrics_block_label")} value={`${formatMs(m.chunk_ms)} ${ms}`} />
                {m.backlog_ms != null ? (
                    <VcStat
                        label={t("voice_changer.metrics_backlog_label")}
                        value={`${formatMs(m.backlog_ms)} / ${formatMs(budgetMs)} ${ms}`}
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
            </div>
            {warnings.map((w) => (
                <p
                    key={w.key}
                    className={classNames("text-xs leading-snug", w.tone === "error" ? "text-error" : "text-warning")}
                >
                    {t(w.key)}
                </p>
            ))}
        </div>
    );
};

const SLOT_CUSTOM = "__vc_slot_custom__";
const SLOT_NONE = "__vc_slot_none__";

/**
 * RVC voice picker over the folders in `app_data/voice-changer/models/` (Rust `list_voice_changer_model_slots`),
 * with "Other folder…" for a voice anywhere else. Picking a voice applies its saved `slot.json`.
 */
const VcVoicePicker: FC<{ slotDir: string; onSlotDirChange: (path: string) => void; onCommit: () => void }> = ({
    slotDir,
    onSlotDirChange,
    onCommit,
}) => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const [slots, setSlots] = useState<VoiceChangerModelSlot[] | null>(null);
    const [scanning, setScanning] = useState(false);
    const [modelsDir, setModelsDir] = useState("");
    const [custom, setCustom] = useState(false);

    const rescan = useCallback(async () => {
        setScanning(true);
        try {
            setSlots(await vcApi.listModelSlots());
        } finally {
            setScanning(false);
        }
    }, [vcApi]);

    useEffect(() => {
        if (!isTauri()) return;
        void rescan();
        let cancelled = false;
        appDataDir()
            .then((d) => join(d, "voice-changer", "models"))
            .then((p) => {
                if (!cancelled) setModelsDir(p);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [rescan]);

    const matched = slots ? findSlot(slots, slotDir) : undefined;
    const hasDir = slotDir.trim() !== "";
    const hasSlots = !!slots && slots.length > 0;
    const showPathField = !hasSlots || custom || (hasDir && !matched);
    // "Other folder…" stays selected while its field is open, even if the typed folder is a listed voice.
    const pickerValue = custom ? SLOT_CUSTOM : matched ? matched.path : hasDir ? SLOT_CUSTOM : SLOT_NONE;

    const pathField = (
        <VcPathField
            labelKey={hasSlots ? "voice_changer.rvc_voice_folder" : "voice_changer.rvc_slot_dir"}
            value={slotDir}
            directory
            onChange={onSlotDirChange}
            onCommit={onCommit}
        />
    );

    return (
        <div className="flex flex-col gap-2">
            {hasSlots ? (
                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                    <InputSelect
                        label="voice_changer.rvc_voice"
                        value={pickerValue}
                        onValueChange={(v) => {
                            if (v === SLOT_NONE) return;
                            if (v === SLOT_CUSTOM) {
                                setCustom(true);
                                return;
                            }
                            setCustom(false);
                            void vcApi.selectModelSlot(v);
                        }}
                        options={[
                            ...(pickerValue === SLOT_NONE ? [{ label: t("voice_changer.rvc_voice_choose"), value: SLOT_NONE }] : []),
                            ...slots!.map((s) => ({ label: s.id, value: s.path })),
                            { label: t("voice_changer.rvc_voice_custom"), value: SLOT_CUSTOM },
                        ]}
                    />
                    <Tooltip content={t("voice_changer.rvc_voices_rescan")}>
                        <button
                            type="button"
                            className="btn btn-ghost btn-sm btn-square mb-0.5"
                            aria-label={t("voice_changer.rvc_voices_rescan")}
                            disabled={scanning}
                            onClick={() => void rescan()}
                        >
                            <RiRefreshLine className={classNames("text-base", scanning && "animate-spin")} />
                        </button>
                    </Tooltip>
                </div>
            ) : null}
            {showPathField ? pathField : null}
            {slots !== null && !hasSlots ? (
                <div className="flex items-start gap-2">
                    <p className="min-w-0 flex-1 break-words text-xs leading-snug text-base-content/55">
                        {t("voice_changer.rvc_voices_empty", {
                            path: modelsDir || "%AppData%/com.sigil.app/voice-changer/models",
                        })}
                    </p>
                    <button
                        type="button"
                        className="btn btn-ghost btn-xs shrink-0"
                        disabled={scanning}
                        onClick={() => void rescan()}
                    >
                        {t("voice_changer.rvc_voices_rescan")}
                    </button>
                </div>
            ) : null}
        </div>
    );
};

/** Path row with optional file/folder browse (Tauri dialog) and optional (i) next to the label. */
const VcPathField: FC<{
    labelKey: string;
    infoKey?: string;
    value: string;
    onChange: (path: string) => void;
    onCommit: () => void;
    directory?: boolean;
    extensions?: string[];
    filterName?: string;
}> = ({ labelKey, infoKey, value, onChange, onCommit, directory, extensions, filterName }) => {
    const { t } = useTranslation();
    const label = (
        <label className="label py-0">
            <span className="label-text">{t(labelKey)}</span>
        </label>
    );
    return (
        <div className="flex flex-col gap-1">
            {infoKey ? (
                <div className="flex items-center gap-1">
                    {label}
                    <InspectorInfoButton text={t(infoKey)} />
                </div>
            ) : (
                label
            )}
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    className="input input-bordered input-sm min-w-0 flex-1 font-mono text-xs"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onBlur={() => onCommit()}
                />
                {isTauri() ? (
                    <button
                        type="button"
                        className="btn btn-sm h-8 min-h-8 shrink-0 border border-base-content/20 bg-base-100 font-semibold normal-case text-base-content hover:bg-base-200"
                        onClick={async () => {
                            const path = await open(
                                directory
                                    ? { directory: true, multiple: false }
                                    : {
                                          filters: [{ name: filterName || "ONNX", extensions: extensions || ["onnx"] }],
                                          multiple: false,
                                      },
                            );
                            if (path === null || Array.isArray(path)) return;
                            onChange(path);
                            onCommit();
                        }}
                    >
                        {t("voice_changer.onnx_browse")}
                    </button>
                ) : (
                    <span className="max-w-[10rem] shrink-0 text-xs text-base-content/50">
                        {t("voice_changer.onnx_browse_unavailable")}
                    </span>
                )}
            </div>
        </div>
    );
};

export const Inspector_VoiceChanger: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const voiceChanger = snap.voiceChanger;
    const isRunning = snap.isRunning;
    const isSidecar = voiceChanger.mode === "sidecar";
    const isOnnx = voiceChanger.mode === "onnx";
    const isRvc = voiceChanger.mode === "rvc";
    const isClassic = voiceChanger.mode === "classic";
    const vcApi = window.ApiServer.voiceChanger;
    /** GPU-only extractors (rmvpe) are greyed out when RVC runs on the CPU, where they run as fcpe. */
    const rvcEpIsCpu = voiceChanger.rvcExecutionProvider === "cpu";
    const pitchFallsBack = effectivePitchExtractor(voiceChanger.pitchExtractor, voiceChanger.rvcExecutionProvider).fellBack;
    const vramCapApplies = voiceChanger.rvcExecutionProvider === "cuda";
    /** Latest metrics sample and the one before it (to tell growing drop / underrun counters). */
    const [samples, setSamples] = useState<{ cur: VoiceChangerMetrics | null; prev: VoiceChangerMetrics | null }>({
        cur: null,
        prev: null,
    });
    const metrics = samples.cur;
    /**
     * The harness clamps crossfade to one processing chunk, so cap the controls there: a longer value would
     * only restart the engine without changing anything. The stored value is kept (it applies again after a
     * larger chunk size is picked); the controls show what is in effect. While running, the chunk length uses
     * the real capture rate (`metrics.chunk_ms`); when stopped the rate is unknown and 48 kHz is assumed.
     */
    const crossfadeMaxMs = crossfadeMaxMsFor(voiceChanger.chunkFrames, isRunning ? metrics?.chunk_ms : null);
    const crossfadeShownMs = Math.min(voiceChanger.crossfadeMs, crossfadeMaxMs);
    /**
     * The service refuses to start RVC until the voice-permission notice is acknowledged (Start, or a
     * restart into RVC such as a mode switch while running) and flags it; the notice shows while flagged.
     */
    const consentOpen = snap.rvcConsentPending;
    const [consentAck, setConsentAck] = useState(false);
    useEffect(() => {
        if (consentOpen) setConsentAck(false);
    }, [consentOpen]);
    /**
     * Extra context and crossfade are read at harness construction: restart once per committed change (drag
     * release / blur), not per drag step. The service compares against what the running engine started with.
     */
    const commitHarnessParams = () => void vcApi.commitHarnessParams();

    const chunkOptions = useMemo(
        () =>
            VOICE_CHANGER_CHUNK_FRAMES.map(frames => ({
                value: String(frames),
                label: t("voice_changer.chunk_frames_row", {
                    frames,
                    ms: chunkFramesToMsAt48k(frames),
                }),
            })),
        [t],
    );

    useEffect(() => {
        setSamples({ cur: null, prev: null });
        if (!isTauri() || !isRunning) return;
        let cancelled = false;
        const tick = async () => {
            if (cancelled || document.visibilityState === "hidden") return;
            const m = await vcApi.getMetrics();
            if (!cancelled && m) setSamples((s) => ({ cur: m, prev: s.cur }));
        };
        void tick();
        const id = window.setInterval(tick, 600);
        const onVis = () => {
            if (document.visibilityState === "visible") void tick();
        };
        document.addEventListener("visibilitychange", onVis);
        return () => {
            cancelled = true;
            window.clearInterval(id);
            document.removeEventListener("visibilitychange", onVis);
        };
    }, [isRunning, vcApi]);

    const patchVcStructural = async <K extends keyof VoiceChangerState>(key: K, v: VoiceChangerState[K]) => {
        window.ApiServer.voiceChanger.state.voiceChanger[key] = v;
        await window.ApiServer.voiceChanger.syncParamsAndRestartIfRunning();
    };

    /** Start / stop; a refused start is already reported by the service (toast), so only swallow it here. */
    const toggleEngine = () => {
        vcApi.toggle().catch(() => {});
    };

    /** RVC without the acknowledgement does not start; the service flags it and the notice opens. */
    const handleToggle = toggleEngine;

    const confirmConsentAndStart = async () => {
        if (!consentAck) return;
        // Stored in Rust's settings (clears the flag); if that fails it still holds for this session.
        await vcApi.acknowledgeRvcConsent();
        vcApi.start().catch(() => {});
    };

    const handleVocoderOversampleChange = (value: string) => {
        const n = parseInt(value, 10) as VoiceChangerState["vocoderOversample"];
        if (n === 1 || n === 2 || n === 4 || n === 8 || n === 16 || n === 32) {
            void window.ApiServer.voiceChanger.setVocoderOversample(n);
        }
    };

    const onChunkFramesChange = async (v: string) => {
        const n = parseInt(v, 10);
        if (!isVoiceChangerChunkFrames(n)) return;
        vcApi.state.voiceChanger.chunkFrames = n;
        await vcApi.syncParamsAndRestartIfRunning();
    };

    return (
        <Inspector.Body>
            <Inspector.Header>
                <RiMagicFill /> {t("voice_changer.title")}
            </Inspector.Header>
            <Inspector.Content>
                <div className="alert alert-warning alert-soft text-xs leading-snug">
                    {t("voice_changer.wip_banner_body")}
                </div>
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-base-content/10 pb-3">
                    <div className="flex min-w-0 flex-1 items-start gap-2">
                        <div
                            className={classNames(
                                "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                                isRunning ? "animate-pulse bg-success" : "bg-base-content/30",
                            )}
                        />
                        <div className="min-w-0">
                            <div className="text-sm font-medium">
                                {isRunning ? t("voice_changer.status_running") : t("voice_changer.status_stopped")}
                            </div>
                            {isTauri() && isRunning && metrics && epStatus(metrics.ep)?.kind === "granted" ? (
                                <VcEpStatus metrics={metrics} />
                            ) : null}
                        </div>
                    </div>
                    <Tooltip content={isRunning ? t("voice_changer.tooltip_stop") : t("voice_changer.tooltip_start")}>
                        <button
                            type="button"
                            onClick={handleToggle}
                            className={`btn btn-sm shrink-0 ${isRunning ? "btn-error" : "btn-primary"}`}
                        >
                            {isRunning ? (
                                <>
                                    <RiStopFill className="me-1" /> {t("voice_changer.btn_stop")}
                                </>
                            ) : (
                                <>
                                    <RiPlayFill className="me-1" /> {t("voice_changer.btn_start")}
                                </>
                            )}
                        </button>
                    </Tooltip>
                </div>
                {isTauri() && isRunning && metrics ? (
                    <>
                        {epStatus(metrics.ep)?.kind === "fallback" ? <VcEpStatus metrics={metrics} /> : null}
                        <VcMetricsPanel metrics={metrics} prev={samples.prev} budgetMs={voiceChanger.latencyBudgetMs} />
                    </>
                ) : null}

                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                    <InputSelect
                        label="voice_changer.mode"
                        value={voiceChanger.mode}
                        onValueChange={async (v) => {
                            await patchVcStructural("mode", v as VoiceChangerState["mode"]);
                        }}
                        options={[
                            { label: t("voice_changer.mode_classic"), value: "classic" },
                            { label: t("voice_changer.mode_sidecar"), value: "sidecar" },
                            { label: t("voice_changer.mode_onnx"), value: "onnx" },
                            { label: t("voice_changer.mode_rvc"), value: "rvc" },
                        ]}
                    />
                    <div className="pb-0.5">
                        <InspectorInfoButton text={t("voice_changer.section_how_it_works_info")} />
                    </div>
                </div>

                {isRvc && (
                    <>
                        <VcSubHeader
                            titleKey="voice_changer.section_rvc_model"
                            infoKey="voice_changer.rvc_slot_help"
                        />
                        {voiceChanger.rvcProfile === "quality" && (
                            <div className="alert alert-info alert-soft text-xs leading-snug">
                                {t("voice_changer.rvc_quality_banner")}
                            </div>
                        )}
                        <InputSelect
                            label="voice_changer.rvc_profile"
                            value={voiceChanger.rvcProfile}
                            onValueChange={(v) => {
                                void vcApi.applyRvcProfile(v as VoiceChangerState["rvcProfile"]);
                            }}
                            options={[
                                { label: t("voice_changer.rvc_profile_streamer"), value: "streamer" },
                                { label: t("voice_changer.rvc_profile_low_latency"), value: "low_latency" },
                                { label: t("voice_changer.rvc_profile_quality"), value: "quality" },
                            ]}
                        />
                        <VcVoicePicker
                            slotDir={voiceChanger.rvcSlotDir}
                            onSlotDirChange={(path) => {
                                vcApi.state.voiceChanger.rvcSlotDir = path;
                            }}
                            onCommit={() => void vcApi.syncParamsAndRestartIfRunning()}
                        />
                        <InputContainer label="voice_changer.pitch_extractor" vertical>
                            <select
                                className="select select-bordered select-sm w-full"
                                value={voiceChanger.pitchExtractor}
                                onChange={(e) => {
                                    const v = e.target.value as VoiceChangerState["pitchExtractor"];
                                    void patchVcStructural("pitchExtractor", v);
                                }}
                            >
                                {PITCH_EXTRACTORS.map((p) => {
                                    const gpuOnly = GPU_ONLY_PITCH_EXTRACTORS.includes(p);
                                    const unavailable = gpuOnly && rvcEpIsCpu;
                                    return (
                                        <option
                                            key={p}
                                            value={p}
                                            disabled={unavailable}
                                            className={unavailable ? "opacity-40" : undefined}
                                        >
                                            {unavailable ? t("voice_changer.pitch_extractor_gpu_only", { name: p }) : p}
                                        </option>
                                    );
                                })}
                            </select>
                        </InputContainer>
                        {pitchFallsBack ? (
                            <p className="-mt-1 text-xs leading-snug text-warning">
                                {t("voice_changer.pitch_extractor_cpu_note", { name: voiceChanger.pitchExtractor })}
                            </p>
                        ) : null}

                        <VcDetails
                            titleKey="voice_changer.section_rvc_bases"
                            infoKey="voice_changer.section_rvc_bases_info"
                            defaultOpen={
                                !!(
                                    voiceChanger.rvcHubertPath.trim() ||
                                    voiceChanger.rvcFcpePath.trim() ||
                                    voiceChanger.rvcRmvpePath.trim()
                                )
                            }
                        >
                            <VcPathField
                                labelKey="voice_changer.rvc_hubert_path"
                                value={voiceChanger.rvcHubertPath}
                                onChange={(path) => {
                                    vcApi.state.voiceChanger.rvcHubertPath = path;
                                }}
                                onCommit={() => void vcApi.syncParamsAndRestartIfRunning()}
                            />
                            <VcPathField
                                labelKey="voice_changer.rvc_fcpe_path"
                                value={voiceChanger.rvcFcpePath}
                                onChange={(path) => {
                                    vcApi.state.voiceChanger.rvcFcpePath = path;
                                }}
                                onCommit={() => void vcApi.syncParamsAndRestartIfRunning()}
                            />
                            <VcPathField
                                labelKey="voice_changer.rvc_rmvpe_path"
                                value={voiceChanger.rvcRmvpePath}
                                onChange={(path) => {
                                    vcApi.state.voiceChanger.rvcRmvpePath = path;
                                }}
                                onCommit={() => void vcApi.syncParamsAndRestartIfRunning()}
                            />
                        </VcDetails>

                        <VcDetails
                            titleKey="voice_changer.section_rvc_tuning"
                            infoKey="voice_changer.section_rvc_tuning_info"
                            defaultOpen={voiceChanger.rvcProfile === "quality"}
                        >
                            <VcSliderField
                                labelKey="voice_changer.extra_context_ms"
                                controlsRow={
                                    <>
                                        <input
                                            type="number"
                                            className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                            min={0}
                                            max={2000}
                                            step={1}
                                            value={voiceChanger.extraContextMs}
                                            onChange={(e) => {
                                                const x = parseFloat(e.target.value);
                                                if (!Number.isNaN(x)) {
                                                    vcApi.state.voiceChanger.extraContextMs = Math.min(
                                                        2000,
                                                        Math.max(0, x),
                                                    );
                                                }
                                            }}
                                            onBlur={commitHarnessParams}
                                        />
                                        <span className="shrink-0 text-xs text-base-content/60">
                                            {t("voice_changer.unit_ms")}
                                        </span>
                                    </>
                                }
                                rangeMin={0}
                                rangeMax={500}
                                rangeStep={1}
                                rangeValue={voiceChanger.extraContextMs}
                                onRangeChange={(v) => {
                                    vcApi.state.voiceChanger.extraContextMs = Math.round(v);
                                }}
                                onRangeCommit={commitHarnessParams}
                                lowLabelKey="voice_changer.label_lower_latency"
                                highLabelKey="voice_changer.label_smoother"
                            />
                            <VcSliderField
                                labelKey="voice_changer.crossfade_ms"
                                controlsRow={
                                    <>
                                        <input
                                            type="number"
                                            className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                            min={0}
                                            max={crossfadeMaxMs}
                                            step={1}
                                            value={crossfadeShownMs}
                                            onChange={(e) => {
                                                const x = parseFloat(e.target.value);
                                                if (!Number.isNaN(x)) {
                                                    vcApi.state.voiceChanger.crossfadeMs = Math.min(
                                                        crossfadeMaxMs,
                                                        Math.max(0, x),
                                                    );
                                                }
                                            }}
                                            onBlur={commitHarnessParams}
                                        />
                                        <span className="shrink-0 text-xs text-base-content/60">
                                            {t("voice_changer.unit_ms")}
                                        </span>
                                    </>
                                }
                                rangeMin={0}
                                rangeMax={crossfadeMaxMs}
                                rangeStep={1}
                                rangeValue={crossfadeShownMs}
                                onRangeChange={(v) => {
                                    vcApi.state.voiceChanger.crossfadeMs = Math.min(crossfadeMaxMs, Math.round(v));
                                }}
                                onRangeCommit={commitHarnessParams}
                                lowLabelKey="voice_changer.label_lower_latency"
                                highLabelKey="voice_changer.label_smoother"
                            />
                            <VcSliderField
                                labelKey="voice_changer.index_rate"
                                controlsRow={
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        value={voiceChanger.indexRate}
                                        onChange={(e) => {
                                            const x = parseFloat(e.target.value);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.indexRate = Math.min(1, Math.max(0, x));
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParams()}
                                    />
                                }
                                rangeMin={0}
                                rangeMax={1}
                                rangeStep={0.01}
                                rangeValue={voiceChanger.indexRate}
                                onRangeChange={(v) => {
                                    vcApi.state.voiceChanger.indexRate = v;
                                    void vcApi.syncParams();
                                }}
                                lowLabelKey="voice_changer.label_deeper"
                                highLabelKey="voice_changer.label_higher"
                            />
                            <VcSliderField
                                labelKey="voice_changer.protect"
                                controlsRow={
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                        min={0}
                                        max={0.5}
                                        step={0.01}
                                        value={voiceChanger.protect}
                                        onChange={(e) => {
                                            const x = parseFloat(e.target.value);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.protect = Math.min(0.5, Math.max(0, x));
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParams()}
                                    />
                                }
                                rangeMin={0}
                                rangeMax={0.5}
                                rangeStep={0.01}
                                rangeValue={voiceChanger.protect}
                                onRangeChange={(v) => {
                                    vcApi.state.voiceChanger.protect = v;
                                    void vcApi.syncParams();
                                }}
                                lowLabelKey="voice_changer.label_deeper"
                                highLabelKey="voice_changer.label_higher"
                            />
                        </VcDetails>

                        <VcDetails
                            titleKey="voice_changer.section_rvc_expert"
                            infoKey="voice_changer.section_rvc_expert_info"
                        >
                            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                                <InputSelect
                                    label="voice_changer.onnx_execution_provider"
                                    value={voiceChanger.rvcExecutionProvider}
                                    onValueChange={async (v) => {
                                        await patchVcStructural(
                                            "rvcExecutionProvider",
                                            v as VoiceChangerState["rvcExecutionProvider"],
                                        );
                                    }}
                                    options={[
                                        { label: t("tts.ep_cpu_plain"), value: "cpu" },
                                        { label: t("tts.ep_cuda_plain"), value: "cuda" },
                                        { label: t("tts.ep_dml_plain"), value: "dml" },
                                    ]}
                                />
                            </div>
                            <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                                <InputContainer label="voice_changer.speaker_id">
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-full"
                                        min={0}
                                        step={1}
                                        value={voiceChanger.speakerId}
                                        onChange={(e) => {
                                            const x = parseInt(e.target.value, 10);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.speakerId = Math.max(0, x);
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParams()}
                                    />
                                </InputContainer>
                                <InputContainer label="voice_changer.gpu_device_index">
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-full"
                                        min={0}
                                        step={1}
                                        value={voiceChanger.gpuDeviceIndex}
                                        onChange={(e) => {
                                            const x = parseInt(e.target.value, 10);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.gpuDeviceIndex = Math.max(0, x);
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParamsAndRestartIfRunning()}
                                    />
                                </InputContainer>
                            </div>
                            <div className="flex flex-col gap-1">
                                <InputContainer
                                    label="voice_changer.vram_cap_mb"
                                    className={vramCapApplies ? undefined : "opacity-50"}
                                >
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-full"
                                        min={256}
                                        max={8192}
                                        step={64}
                                        disabled={!vramCapApplies}
                                        value={voiceChanger.vramCapMb}
                                        onChange={(e) => {
                                            const x = parseInt(e.target.value, 10);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.vramCapMb = Math.min(
                                                    8192,
                                                    Math.max(256, x),
                                                );
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParamsAndRestartIfRunning()}
                                    />
                                </InputContainer>
                                {/* Always shown: on CUDA the per-session note matters most (total VRAM can exceed the cap). */}
                                <p className="text-xs leading-snug text-base-content/55">
                                    {t("voice_changer.vram_cap_cuda_only")}
                                </p>
                            </div>
                            <label className="label cursor-pointer justify-start gap-2 py-0">
                                <input
                                    type="checkbox"
                                    className="checkbox checkbox-sm checkbox-primary"
                                    checked={voiceChanger.silenceSkip}
                                    onChange={async (e) => {
                                        vcApi.state.voiceChanger.silenceSkip = e.target.checked;
                                        await vcApi.syncParamsAndRestartIfRunning();
                                    }}
                                />
                                <span className="label-text text-sm">{t("voice_changer.silence_skip")}</span>
                            </label>
                            <button
                                type="button"
                                className="btn btn-ghost btn-xs self-start"
                                onClick={async () => {
                                    const snapDbg = await vcApi.captureDebug();
                                    if (snapDbg) {
                                        console.info("[VoiceChanger] debug", snapDbg);
                                        toast.info(t("voice_changer.debug_captured"));
                                    }
                                }}
                            >
                                {t("voice_changer.debug_capture")}
                            </button>
                        </VcDetails>
                    </>
                )}

                {isSidecar && (
                    <>
                        <InputText
                            label="voice_changer.sidecar_ws_url"
                            value={voiceChanger.sidecarWsUrl}
                            onChange={(e) => {
                                window.ApiServer.voiceChanger.state.voiceChanger.sidecarWsUrl = e.target.value;
                            }}
                            onBlur={() => void window.ApiServer.voiceChanger.syncParamsAndRestartIfRunning()}
                        />
                        <div className="flex justify-end -mt-2">
                            <InspectorInfoButton text={t("voice_changer.sidecar_ws_info")} />
                        </div>
                    </>
                )}

                {isOnnx && (
                    <>
                        <VcPathField
                            labelKey="voice_changer.onnx_model_path"
                            infoKey="voice_changer.onnx_files_info"
                            value={voiceChanger.onnxModelPath}
                            onChange={(path) => {
                                vcApi.state.voiceChanger.onnxModelPath = path;
                            }}
                            onCommit={() => void vcApi.syncParamsAndRestartIfRunning()}
                        />
                        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                            <InputSelect
                                label="voice_changer.onnx_execution_provider"
                                value={voiceChanger.onnxExecutionProvider}
                                onValueChange={async (v) => {
                                    await patchVcStructural("onnxExecutionProvider", v as VoiceChangerState["onnxExecutionProvider"]);
                                }}
                                options={[
                                    { label: t("tts.ep_cpu_plain"), value: "cpu" },
                                    { label: t("tts.ep_cuda_plain"), value: "cuda" },
                                    { label: t("tts.ep_dml_plain"), value: "dml" },
                                ]}
                            />
                            <div className="pb-0.5">
                                <InspectorInfoButton text={t("voice_changer.onnx_ep_help")} />
                            </div>
                        </div>
                    </>
                )}

                <VcSubHeader titleKey="voice_changer.section_devices" infoKey="voice_changer.section_devices_info" />
                <VoiceInputDeviceSelect
                    value={voiceChanger.inputDevice}
                    onChange={(device) => void vcApi.setInputDevice(device)}
                    labelKey="voice_changer.device_input"
                    defaultLabelKey="voice_changer.device_system_default"
                />
                <VoiceOutputDeviceSelect
                    value={voiceChanger.outputDevice}
                    onChange={async (device) => {
                        window.ApiServer.voiceChanger.state.voiceChanger.outputDevice = device;
                        await window.ApiServer.voiceChanger.syncParamsAndRestartIfRunning();
                    }}
                    labelKey="voice_changer.device_output"
                    defaultLabelKey="voice_changer.device_system_default"
                />
                <VcDetails titleKey="voice_changer.monitor_section">
                    <VoiceMonitorOutputDeviceSelect
                        value={voiceChanger.monitorOutputDevice}
                        onChange={async (device) => {
                            vcApi.state.voiceChanger.monitorOutputDevice = device;
                            await vcApi.syncParamsAndRestartIfRunning();
                        }}
                    />
                    <InputContainer label="voice_changer.monitor_gain">
                        <input
                            type="number"
                            className="input input-bordered input-sm w-full"
                            min={0}
                            max={4}
                            step={0.1}
                            value={voiceChanger.monitorGain}
                            onChange={(e) => {
                                const x = parseFloat(e.target.value);
                                if (!Number.isNaN(x)) {
                                    vcApi.state.voiceChanger.monitorGain = Math.min(4, Math.max(0, x));
                                }
                            }}
                            onBlur={() => void vcApi.syncParams()}
                        />
                    </InputContainer>
                </VcDetails>

                <VcSubHeader titleKey="voice_changer.section_sound" infoKey="voice_changer.section_sound_info" />
                <VcSliderField
                    labelKey="voice_changer.pitch"
                    controlsRow={
                        <>
                            <input
                                type="number"
                                className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                min={-24}
                                max={24}
                                step={0.1}
                                value={voiceChanger.pitch}
                                onChange={(e) => {
                                    const x = parseFloat(e.target.value);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.pitch = Math.min(24, Math.max(-24, x));
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                            <span className="shrink-0 text-xs text-base-content/60">
                                {t("voice_changer.pitch_unit_semitones")}
                            </span>
                            <button
                                type="button"
                                className="btn btn-ghost btn-xs shrink-0"
                                onClick={() => void vcApi.setPitch(0)}
                            >
                                {t("voice_changer.pitch_reset")}
                            </button>
                        </>
                    }
                    rangeMin={-24}
                    rangeMax={24}
                    rangeStep={0.1}
                    rangeValue={voiceChanger.pitch}
                    onRangeChange={(v) => void vcApi.setPitch(v)}
                    lowLabelKey="voice_changer.label_deeper"
                    highLabelKey="voice_changer.label_higher"
                />
                <VcSliderField
                    labelKey="voice_changer.formant_shift"
                    controlsRow={
                        <input
                            type="number"
                            className="input input-bordered input-sm w-24 shrink-0 font-mono"
                            min={-1}
                            max={1}
                            step={0.01}
                            value={voiceChanger.formant}
                            onChange={(e) => {
                                const x = parseFloat(e.target.value);
                                if (!Number.isNaN(x)) {
                                    vcApi.state.voiceChanger.formant = Math.min(1, Math.max(-1, x));
                                }
                            }}
                            onBlur={() => void vcApi.syncParams()}
                        />
                    }
                    rangeMin={-1}
                    rangeMax={1}
                    rangeStep={0.05}
                    rangeValue={voiceChanger.formant}
                    onRangeChange={(v) => void vcApi.setFormant(v)}
                    lowLabelKey="voice_changer.label_darker"
                    highLabelKey="voice_changer.label_bright"
                />
                {isClassic && (
                    <VcDetails
                        titleKey="voice_changer.section_tone_vocoder"
                        infoKey="voice_changer.section_tone_vocoder_info"
                        defaultOpen
                    >
                        <VcSliderField
                            labelKey="voice_changer.analysis_window"
                            controlsRow={
                                <>
                                    <input
                                        type="number"
                                        className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                        min={30}
                                        max={60}
                                        value={voiceChanger.vocoderWindowMs}
                                        aria-label={t("voice_changer.analysis_window")}
                                        onChange={(e) => {
                                            const x = parseInt(e.target.value, 10);
                                            if (!Number.isNaN(x)) {
                                                vcApi.state.voiceChanger.vocoderWindowMs = Math.min(
                                                    60,
                                                    Math.max(30, x),
                                                );
                                            }
                                        }}
                                        onBlur={() => void vcApi.syncParams()}
                                    />
                                    <span className="shrink-0 text-xs text-base-content/60">
                                        {t("voice_changer.unit_ms")}
                                    </span>
                                </>
                            }
                            rangeMin={30}
                            rangeMax={60}
                            rangeStep={1}
                            rangeValue={voiceChanger.vocoderWindowMs}
                            onRangeChange={(v) => void vcApi.setVocoderWindowMs(Math.round(v))}
                            lowLabelKey="voice_changer.label_lower_latency"
                            highLabelKey="voice_changer.label_smoother"
                        />
                        <div className="grid w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                            <InputSelect
                                label="voice_changer.oversampling"
                                value={String(voiceChanger.vocoderOversample)}
                                onValueChange={handleVocoderOversampleChange}
                                options={[
                                    { label: t("voice_changer.oversample_1"), value: "1" },
                                    { label: t("voice_changer.oversample_2"), value: "2" },
                                    { label: t("voice_changer.oversample_4"), value: "4" },
                                    { label: t("voice_changer.oversample_8"), value: "8" },
                                    { label: t("voice_changer.oversample_16"), value: "16" },
                                    { label: t("voice_changer.oversample_32"), value: "32" },
                                ]}
                            />
                            <div className="pb-0.5">
                                <InspectorInfoButton text={t("voice_changer.oversampling_hint")} />
                            </div>
                        </div>
                    </VcDetails>
                )}

                <VcSubHeader titleKey="voice_changer.section_levels" />
                <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                    <InputContainer label="voice_changer.input_gain">
                        <input
                            type="number"
                            className="input input-bordered input-sm w-full"
                            min={0}
                            max={4}
                            step={0.1}
                            value={voiceChanger.inputGain}
                            onChange={(e) => {
                                const x = parseFloat(e.target.value);
                                if (!Number.isNaN(x)) {
                                    vcApi.state.voiceChanger.inputGain = Math.min(4, Math.max(0, x));
                                }
                            }}
                            onBlur={() => void vcApi.syncParams()}
                        />
                    </InputContainer>
                    <InputContainer label="voice_changer.output_gain">
                        <input
                            type="number"
                            className="input input-bordered input-sm w-full"
                            min={0}
                            max={4}
                            step={0.1}
                            value={voiceChanger.outputGain}
                            onChange={(e) => {
                                const x = parseFloat(e.target.value);
                                if (!Number.isNaN(x)) {
                                    vcApi.state.voiceChanger.outputGain = Math.min(4, Math.max(0, x));
                                }
                            }}
                            onBlur={() => void vcApi.syncParams()}
                        />
                    </InputContainer>
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                    <InputSelect
                        label="voice_changer.aux_mic_gate"
                        value={voiceChanger.auxMicGate}
                        onValueChange={(v) => void vcApi.setAuxMicGate(v as AuxMicGate)}
                        options={AUX_MIC_GATES.map((g) => ({ label: t(`voice_changer.aux_mic_gate_${g}`), value: g }))}
                    />
                    <div className="pb-0.5">
                        <InspectorInfoButton text={t("voice_changer.aux_mic_gate_info")} />
                    </div>
                </div>

                <VcSubHeader
                    titleKey="voice_changer.section_timing_buffer"
                    infoKey="voice_changer.section_timing_buffer_info"
                />
                <InputSelect
                    label="voice_changer.latency_preset"
                    value={voiceChanger.latencyPreset}
                    onValueChange={async (v) => {
                        if (v === "low_latency") {
                            await vcApi.applyLatencyPreset("low_latency");
                        } else if (v === "stable") {
                            await vcApi.applyLatencyPreset("stable");
                        }
                    }}
                    options={[
                        { label: t("voice_changer.latency_low"), value: "low_latency" },
                        { label: t("voice_changer.latency_stable"), value: "stable" },
                        ...(voiceChanger.latencyPreset === "custom"
                            ? [{ label: t("voice_changer.latency_custom"), value: "custom" }]
                            : []),
                    ]}
                />
                <VcSliderField
                    labelKey="voice_changer.latency_budget"
                    infoKey="voice_changer.latency_budget_info"
                    controlsRow={
                        <>
                            <VcDraftNumberInput
                                value={voiceChanger.latencyBudgetMs}
                                min={LATENCY_BUDGET.min}
                                max={LATENCY_BUDGET.max}
                                step={10}
                                ariaLabel={t("voice_changer.latency_budget")}
                                onCommit={(v) => void vcApi.setLatencyBudgetMs(v)}
                            />
                            <span className="shrink-0 text-xs text-base-content/60">
                                {t("voice_changer.unit_ms")}
                            </span>
                            {voiceChanger.latencyBudgetMs !== LATENCY_BUDGET.default ? (
                                <button
                                    type="button"
                                    className="btn btn-ghost btn-xs shrink-0"
                                    onClick={() => void vcApi.setLatencyBudgetMs(LATENCY_BUDGET.default)}
                                >
                                    {t("voice_changer.pitch_reset")}
                                </button>
                            ) : null}
                        </>
                    }
                    rangeMin={LATENCY_BUDGET.min}
                    rangeMax={LATENCY_BUDGET.max}
                    rangeStep={10}
                    rangeValue={voiceChanger.latencyBudgetMs}
                    onRangeChange={(v) => {
                        vcApi.state.voiceChanger.latencyBudgetMs = clampLatencyBudgetMs(v);
                    }}
                    // Hot param: synced once per drag (release / blur); the engine applies it on its next block.
                    onRangeCommit={() => void vcApi.setLatencyBudgetMs(vcApi.state.voiceChanger.latencyBudgetMs)}
                    lowLabelKey="voice_changer.label_lower_latency"
                    highLabelKey="voice_changer.label_fewer_skips"
                />
                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                    <InputSelect
                        label="voice_changer.worker_priority"
                        value={voiceChanger.workerPriority}
                        onValueChange={(v) => void vcApi.setWorkerPriority(v as WorkerPriority)}
                        options={WORKER_PRIORITIES.map((p) => ({ label: t(`voice_changer.worker_priority_${p}`), value: p }))}
                    />
                    <div className="pb-0.5">
                        <InspectorInfoButton text={t("voice_changer.worker_priority_info")} />
                    </div>
                </div>

                <VcDetails titleKey="voice_changer.section_advanced" infoKey="voice_changer.section_advanced_info">
                    <InputSelect
                        label="voice_changer.chunk_frames"
                        value={String(voiceChanger.chunkFrames)}
                        onValueChange={(v) => void onChunkFramesChange(v)}
                        options={chunkOptions}
                    />
                    <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                        <InputContainer label="voice_changer.output_prefill_ms">
                            <input
                                type="number"
                                className="input input-bordered input-sm w-full"
                                min={5}
                                max={300}
                                step={1}
                                value={voiceChanger.outputPrefillMs}
                                onChange={(e) => {
                                    const x = parseFloat(e.target.value);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.outputPrefillMs = x;
                                        vcApi.markBufferCustom();
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                        </InputContainer>
                        <InputContainer label="voice_changer.output_max_queue_ms">
                            <input
                                type="number"
                                className="input input-bordered input-sm w-full"
                                min={10}
                                max={600}
                                step={1}
                                value={voiceChanger.outputMaxQueueMs}
                                onChange={(e) => {
                                    const x = parseFloat(e.target.value);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.outputMaxQueueMs = x;
                                        vcApi.markBufferCustom();
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                        </InputContainer>
                    </div>
                    <label className="label cursor-pointer justify-start gap-2 py-0">
                        <input
                            type="checkbox"
                            className="checkbox checkbox-sm checkbox-primary"
                            checked={voiceChanger.bufferTruncateEnabled}
                            onChange={async (e) => {
                                vcApi.state.voiceChanger.bufferTruncateEnabled = e.target.checked;
                                await vcApi.syncParams();
                            }}
                        />
                        <span className="label-text text-sm">{t("voice_changer.buffer_truncate_enable")}</span>
                    </label>
                    <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                        <InputContainer label="voice_changer.buffer_truncate_rms">
                            <input
                                type="number"
                                className="input input-bordered input-sm w-full font-mono text-xs"
                                min={1e-8}
                                max={1}
                                step={0.0001}
                                value={voiceChanger.bufferTruncateRms}
                                onChange={(e) => {
                                    const x = parseFloat(e.target.value);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.bufferTruncateRms = x;
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                        </InputContainer>
                        <InputContainer label="voice_changer.buffer_truncate_chunks">
                            <input
                                type="number"
                                className="input input-bordered input-sm w-full"
                                min={1}
                                max={500}
                                step={1}
                                value={voiceChanger.bufferTruncateChunks}
                                onChange={(e) => {
                                    const x = parseInt(e.target.value, 10);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.bufferTruncateChunks = Math.min(
                                            500,
                                            Math.max(1, x),
                                        );
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                        </InputContainer>
                    </div>
                    {(isOnnx || isRvc) && (
                        <p className="text-xs leading-snug text-base-content/55">{t("voice_changer.vram_note")}</p>
                    )}
                    <p className="text-xs leading-snug text-base-content/55 whitespace-pre-line">
                        {t("voice_changer.tips_body")}
                    </p>
                </VcDetails>
            </Inspector.Content>

            {consentOpen
                ? createPortal(
                      <div className="fixed inset-0 z-[10050] flex items-center justify-center bg-black/50 p-4">
                          <div
                              role="dialog"
                              aria-modal="true"
                              aria-labelledby="vc-rvc-consent-title"
                              className="w-full max-w-md rounded-lg border border-base-content/15 bg-base-100 p-4 shadow-xl"
                          >
                              <h3 id="vc-rvc-consent-title" className="text-base font-semibold">
                                  {t("voice_changer.rvc_voice_consent_title")}
                              </h3>
                              <p className="mt-2 text-sm text-base-content/80 whitespace-pre-line">
                                  {t("voice_changer.rvc_voice_consent_body")}
                              </p>
                              <label className="label mt-3 cursor-pointer justify-start gap-2 py-0">
                                  <input
                                      type="checkbox"
                                      className="checkbox checkbox-sm checkbox-primary"
                                      checked={consentAck}
                                      onChange={(e) => setConsentAck(e.target.checked)}
                                  />
                                  <span className="label-text text-sm">{t("voice_changer.rvc_voice_consent_ack")}</span>
                              </label>
                              <div className="mt-4 flex justify-end gap-2">
                                  <button
                                      type="button"
                                      className="btn btn-ghost btn-sm"
                                      onClick={() => vcApi.dismissRvcConsent()}
                                  >
                                      {t("voice_changer.info_close_overlay")}
                                  </button>
                                  <button
                                      type="button"
                                      className="btn btn-primary btn-sm"
                                      disabled={!consentAck}
                                      onClick={() => void confirmConsentAndStart()}
                                  >
                                      {t("voice_changer.btn_start")}
                                  </button>
                              </div>
                          </div>
                      </div>,
                      document.body,
                  )
                : null}
        </Inspector.Body>
    );
};
