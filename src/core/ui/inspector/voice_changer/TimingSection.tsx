// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Timing and buffer (latency preset, budget, worker priority) and the advanced block / buffer settings;
// in RVC mode also the pacing controls (P3B-H: output cushion, device block, SOLA search, content context).
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import {
    chunkFramesToMsAt48k,
    isVoiceChangerChunkFrames,
    VOICE_CHANGER_CHUNK_FRAMES,
} from "@/core/services/voice_changer/chunkFrames";
import {
    clampLatencyBudgetMs,
    VC_NUMBER_RANGES,
    WORKER_PRIORITIES,
    type WorkerPriority,
} from "@/core/services/voice_changer/contract";
import { FC, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputContainer, InputSelect } from "../components/input";
import { formatMs, raisedOutputBuffer } from "@/core/services/voice_changer/metrics";
import { InspectorInfoButton, VcDetails, VcDraftNumberInput, VcNote, VcSliderField, VcSubHeader } from "./common";
import { PacingControls } from "./PacingControls";

const LATENCY_BUDGET = VC_NUMBER_RANGES.latency_budget_ms;

/** `metrics` is the running engine's latest sample, if any (the live output cushion). */
export const TimingSection: FC<{ metrics?: VoiceChangerMetrics | null }> = ({ metrics = null }) => {
    const { t } = useTranslation();
    const vcSnap = useSnapshot(window.ApiServer.voiceChanger.state);
    const voiceChanger = vcSnap.voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const isOnnx = voiceChanger.mode === "onnx";
    const isRvc = voiceChanger.mode === "rvc";
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
    const onChunkFramesChange = async (v: string) => {
        const n = parseInt(v, 10);
        if (!isVoiceChangerChunkFrames(n)) return;
        vcApi.state.voiceChanger.chunkFrames = n;
        await vcApi.syncParamsAndRestartIfRunning();
    };
    // The buffer the engine plays with when the playing voice needs more than the stored settings.
    const raised = vcSnap.isRunning ? raisedOutputBuffer(metrics, voiceChanger.outputPrefillMs, voiceChanger.outputMaxQueueMs) : null;
    return (
        <>
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
            {raised ? (
                <VcNote>
                    {t("voice_changer.output_buffer_raised", {
                        prefill: formatMs(raised.prefillMs),
                        queue: formatMs(raised.maxQueueMs),
                        stored_prefill: formatMs(voiceChanger.outputPrefillMs),
                        stored_queue: formatMs(voiceChanger.outputMaxQueueMs),
                    })}
                </VcNote>
            ) : null}
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
                {isRvc ? <PacingControls metrics={metrics} /> : null}
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
        </>
    );
};
