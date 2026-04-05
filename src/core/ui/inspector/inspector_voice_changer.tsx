import { VoiceChangerState } from "@/core/services/voice_changer/schema";
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import {
    chunkFramesToMsAt48k,
    isVoiceChangerChunkFrames,
    VOICE_CHANGER_CHUNK_FRAMES,
} from "@/core/services/voice_changer/chunkFrames";
import classNames from "classnames";
import { FC, ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import Inspector from "./components";
import { InputSelect, InputContainer, InputText } from "./components/input";
import { RiInformationLine, RiMagicFill, RiPlayFill, RiStopFill } from "react-icons/ri";
import Tooltip from "../dropdown/Tooltip";

interface AudioDevice {
    id: string;
    name: string;
}

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
            {show && box ? (
                <>
                    <button
                        type="button"
                        className="fixed inset-0 z-[199] cursor-default bg-transparent"
                        aria-label={t("voice_changer.info_close_overlay")}
                        onClick={() => setShow(false)}
                    />
                    <div
                        className="fixed z-[200] overflow-y-auto whitespace-pre-line rounded-lg border border-base-content/15 bg-base-100 p-3 text-xs font-normal text-base-content/80 shadow-lg"
                        style={{
                            top: box.top,
                            left: box.left,
                            width: box.width,
                            maxHeight: box.maxHeight,
                        }}
                    >
                        {text}
                    </div>
                </>
            ) : null}
        </div>
    );
};

const VC_RANGE =
    "range range-sm range-primary h-6 w-full min-w-0 shrink-0 cursor-pointer [width:100%]";

/** Label on its own row; uniform full-width slider and end caps (matches inspector tone controls). */
const VcSliderField: FC<{
    labelKey: string;
    inactive?: boolean;
    controlsRow: ReactNode;
    rangeMin: number;
    rangeMax: number;
    rangeStep: number;
    rangeValue: number;
    onRangeChange: (v: number) => void;
    lowLabelKey: string;
    highLabelKey: string;
}> = ({
    labelKey,
    inactive,
    controlsRow,
    rangeMin,
    rangeMax,
    rangeStep,
    rangeValue,
    onRangeChange,
    lowLabelKey,
    highLabelKey,
}) => {
    const { t } = useTranslation();
    return (
        <div className="flex w-full min-w-0 flex-col gap-2">
            <div className="text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</div>
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

const VoiceInputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
    labelKey: string;
    defaultLabelKey: string;
}> = ({ value, onChange, labelKey, defaultLabelKey }) => {
    const { t } = useTranslation();
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        invoke<AudioDevice[]>("plugin:audio|list_input_devices")
            .then(setDevices)
            .catch(console.error);
    }, []);

    return (
        <InputSelect
            label={labelKey}
            value={value || ""}
            onValueChange={onChange}
            options={[
                { label: t(defaultLabelKey), value: "" },
                ...devices.map(d => ({ label: d.name, value: d.id }))
            ]}
        />
    );
};

const MONITOR_OUTPUT_OFF = "__vc_monitor_off__";

const VoiceMonitorOutputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
}> = ({ value, onChange }) => {
    const { t } = useTranslation();
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        invoke<AudioDevice[]>("plugin:audio|list_output_devices")
            .then(setDevices)
            .catch(console.error);
    }, []);

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
                ...devices.map(d => ({ label: d.name, value: d.id }))
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
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        invoke<AudioDevice[]>("plugin:audio|list_output_devices")
            .then(setDevices)
            .catch(console.error);
    }, []);

    return (
        <InputSelect
            label={labelKey}
            value={value || "default"}
            onValueChange={onChange}
            options={[
                { label: t(defaultLabelKey), value: "default" },
                ...devices.map(d => ({ label: d.name, value: d.id }))
            ]}
        />
    );
};

export const Inspector_VoiceChanger: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const voiceChanger = snap.voiceChanger;
    const isRunning = snap.isRunning;
    const isSidecar = voiceChanger.mode === "sidecar";
    const isOnnx = voiceChanger.mode === "onnx";
    const vcApi = window.ApiServer.voiceChanger;

    const [metrics, setMetrics] = useState<VoiceChangerMetrics | null>(null);

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
        if (!isTauri() || !isRunning) {
            setMetrics(null);
            return;
        }
        let cancelled = false;
        const tick = async () => {
            if (cancelled || document.visibilityState === "hidden") return;
            const m = await vcApi.getMetrics();
            if (!cancelled && m) setMetrics(m);
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

    const patchVc = async <K extends keyof VoiceChangerState>(key: K, v: VoiceChangerState[K]) => {
        window.ApiServer.voiceChanger.state.voiceChanger[key] = v;
        await window.ApiServer.voiceChanger.syncParams();
    };

    const handleToggle = () => {
        window.ApiServer.voiceChanger.toggle();
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
        await vcApi.syncParams();
        await vcApi.restartIfRunning();
    };

    return (
        <Inspector.Body>
            <Inspector.Header>
                <RiMagicFill /> {t("voice_changer.title")}
            </Inspector.Header>
            <Inspector.Content>
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
                            {isTauri() && isRunning && metrics && (
                                <div className="mt-0.5 font-mono text-xs text-base-content/60">
                                    {t("voice_changer.metrics_buf", { value: metrics.buffer_ms.toFixed(1) })}{" "}
                                    {t("voice_changer.metrics_proc", { value: metrics.process_ms.toFixed(2) })}
                                </div>
                            )}
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

                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                    <InputSelect
                        label="voice_changer.mode"
                        value={voiceChanger.mode}
                        onValueChange={async (v) => {
                            await patchVc("mode", v as VoiceChangerState["mode"]);
                        }}
                        options={[
                            { label: t("voice_changer.mode_classic"), value: "classic" },
                            { label: t("voice_changer.mode_sidecar"), value: "sidecar" },
                            { label: t("voice_changer.mode_onnx"), value: "onnx" },
                        ]}
                    />
                    <div className="pb-0.5">
                        <InspectorInfoButton text={t("voice_changer.section_how_it_works_info")} />
                    </div>
                </div>

                {isSidecar && (
                    <>
                        <InputText
                            label="voice_changer.sidecar_ws_url"
                            value={voiceChanger.sidecarWsUrl}
                            onChange={(e) => {
                                window.ApiServer.voiceChanger.state.voiceChanger.sidecarWsUrl = e.target.value;
                            }}
                            onBlur={() => void window.ApiServer.voiceChanger.syncParams()}
                        />
                        <div className="flex justify-end -mt-2">
                            <InspectorInfoButton text={t("voice_changer.sidecar_ws_info")} />
                        </div>
                    </>
                )}

                {isOnnx && (
                    <>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1">
                                <label className="label py-0">
                                    <span className="label-text">{t("voice_changer.onnx_model_path")}</span>
                                </label>
                                <InspectorInfoButton text={t("voice_changer.onnx_files_info")} />
                            </div>
                            <div className="flex items-center gap-2">
                                <input
                                    type="text"
                                    className="input input-bordered input-sm min-w-0 flex-1 font-mono text-xs"
                                    value={voiceChanger.onnxModelPath}
                                    onChange={(e) => {
                                        window.ApiServer.voiceChanger.state.voiceChanger.onnxModelPath = e.target.value;
                                    }}
                                    onBlur={() => void window.ApiServer.voiceChanger.syncParams()}
                                />
                                {isTauri() ? (
                                    <button
                                        type="button"
                                        className="btn btn-sm h-8 min-h-8 shrink-0 border border-base-content/20 bg-base-100 font-semibold normal-case text-base-content hover:bg-base-200"
                                        onClick={async () => {
                                            const path = await open({
                                                filters: [{ name: "ONNX", extensions: ["onnx"] }],
                                                multiple: false,
                                            });
                                            if (path === null || Array.isArray(path)) return;
                                            window.ApiServer.voiceChanger.state.voiceChanger.onnxModelPath = path;
                                            await window.ApiServer.voiceChanger.syncParams();
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
                        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
                            <InputSelect
                                label="voice_changer.onnx_execution_provider"
                                value={voiceChanger.onnxExecutionProvider}
                                onValueChange={async (v) => {
                                    await patchVc("onnxExecutionProvider", v as VoiceChangerState["onnxExecutionProvider"]);
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

                <VcSubHeader titleKey="voice_changer.section_timing_buffer" infoKey="voice_changer.section_timing_buffer_info" />
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
                            onBlur={async () => {
                                await vcApi.syncParams();
                                await vcApi.restartIfRunning();
                            }}
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
                            onBlur={async () => {
                                await vcApi.syncParams();
                                await vcApi.restartIfRunning();
                            }}
                        />
                    </InputContainer>
                </div>

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
                            <span className="shrink-0 text-xs text-base-content/60">{t("voice_changer.pitch_unit_semitones")}</span>
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

                {(isSidecar || isOnnx) && (
                    <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-base-content/45">
                        {t("voice_changer.sidecar_postprocess_divider")}
                    </div>
                )}

                <VcSliderField
                    labelKey="voice_changer.formant_shift"
                    controlsRow={
                        <>
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
                        </>
                    }
                    rangeMin={-1}
                    rangeMax={1}
                    rangeStep={0.05}
                    rangeValue={voiceChanger.formant}
                    onRangeChange={(v) => void vcApi.setFormant(v)}
                    lowLabelKey="voice_changer.label_darker"
                    highLabelKey="voice_changer.label_bright"
                />

                <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-base-content/45">
                    {t("voice_changer.phase_vocoder_divider")}
                </div>

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
                                        vcApi.state.voiceChanger.vocoderWindowMs = Math.min(60, Math.max(30, x));
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                            <span className="shrink-0 text-xs text-base-content/60">{t("voice_changer.unit_ms")}</span>
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

                <VcSubHeader titleKey="voice_changer.section_devices" infoKey="voice_changer.section_devices_info" />
                <VoiceInputDeviceSelect
                    value={voiceChanger.inputDevice}
                    onChange={(device) => {
                        window.ApiServer.voiceChanger.state.voiceChanger.inputDevice = device;
                    }}
                    labelKey="voice_changer.device_input"
                    defaultLabelKey="voice_changer.device_system_default"
                />

                <VoiceOutputDeviceSelect
                    value={voiceChanger.outputDevice}
                    onChange={async (device) => {
                        window.ApiServer.voiceChanger.state.voiceChanger.outputDevice = device;
                        await window.ApiServer.voiceChanger.syncParams();
                        await window.ApiServer.voiceChanger.restartIfRunning();
                    }}
                    labelKey="voice_changer.device_output"
                    defaultLabelKey="voice_changer.device_system_default"
                />

                <div className="pt-1 text-xs font-semibold uppercase tracking-wide text-base-content/45">
                    {t("voice_changer.monitor_section")}
                </div>
                <VoiceMonitorOutputDeviceSelect
                    value={voiceChanger.monitorOutputDevice}
                    onChange={async (device) => {
                        vcApi.state.voiceChanger.monitorOutputDevice = device;
                        await vcApi.syncParams();
                        await vcApi.restartIfRunning();
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

                <VcSubHeader titleKey="voice_changer.section_advanced" infoKey="voice_changer.section_advanced_info" />
                <div className="flex flex-col gap-2">
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
                    <div className="flex justify-end -mt-1">
                        <InspectorInfoButton text={t("voice_changer.buffer_truncate_hint")} />
                    </div>
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
                                        vcApi.state.voiceChanger.bufferTruncateChunks = Math.min(500, Math.max(1, x));
                                    }
                                }}
                                onBlur={() => void vcApi.syncParams()}
                            />
                        </InputContainer>
                    </div>
                </div>

                <Inspector.Description>{t("voice_changer.vram_note")}</Inspector.Description>

                <Inspector.SubHeader>
                    <span className="min-w-0 flex-1 truncate">{t("voice_changer.tips_title")}</span>
                    <InspectorInfoButton text={t("voice_changer.tips_body")} />
                </Inspector.SubHeader>
            </Inspector.Content>
        </Inspector.Body>
    );
};
