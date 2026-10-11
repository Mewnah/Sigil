// Voice-changer inspector (P3B-H): the pacing controls of RVC mode, under Advanced. The output
// cushion (fixed prefill or the adaptive controller, with the cushion the queue holds right now),
// the device block (as set, or a divisor of the engine block for imported voices), the SOLA search
// window and the content encoder's left context (each automatic or a number of ms). They are read
// when the engine starts, so a change restarts a running engine (`restart.ts`).
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import {
    AUTO_MS,
    DEVICE_BLOCKS,
    OUTPUT_CUSHIONS,
    VC_NUMBER_RANGES,
    type DeviceBlock,
    type OutputCushion,
} from "@/core/services/voice_changer/contract";
import { deviceBlockNote } from "@/core/services/voice_changer/chunkFrames";
import { runsOnTemplateEngine } from "@/core/services/voice_changer/families";
import { formatMs, raisedOutputBuffer } from "@/core/services/voice_changer/metrics";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InspectorInfoButton, VcChoiceSelect, VcDraftNumberInput, VcNote } from "./common";
import { useSelectedVoice, useVoiceFamily } from "./hooks";

const R = VC_NUMBER_RANGES;

/** A number of ms that is either automatic (`AUTO_MS`) or set: a checkbox and the number beside it. */
const AutoMsField: FC<{
    labelKey: string;
    infoKey: string;
    value: number;
    min: number;
    max: number;
    step: number;
    /** The number a click on "Automatic" (off) starts from. */
    customStart: number;
    onCommit: (v: number) => void;
    note?: string;
}> = ({ labelKey, infoKey, value, min, max, step, customStart, onCommit, note }) => {
    const { t } = useTranslation();
    const auto = value < 0;
    return (
        <div className="flex w-full min-w-0 flex-col gap-1">
            <div className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</span>
                <InspectorInfoButton text={t(infoKey)} />
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
                <label className="label cursor-pointer justify-start gap-2 py-0">
                    <input
                        type="checkbox"
                        className="checkbox checkbox-sm checkbox-primary"
                        checked={auto}
                        onChange={(e) => onCommit(e.target.checked ? AUTO_MS : customStart)}
                    />
                    <span className="label-text text-sm">{t("voice_changer.auto_ms_automatic")}</span>
                </label>
                {auto ? null : (
                    <>
                        <VcDraftNumberInput value={value} min={min} max={max} step={step} ariaLabel={t(labelKey)} onCommit={(v) => onCommit(Math.round(v * 10) / 10)} />
                        <span className="shrink-0 text-xs text-base-content/60">{t("voice_changer.unit_ms")}</span>
                    </>
                )}
            </div>
            {note ? <VcNote>{note}</VcNote> : null}
        </div>
    );
};

/** The RVC pacing controls (render them in RVC mode). `metrics` is the running engine's latest sample. */
export const PacingControls: FC<{ metrics: VoiceChangerMetrics | null }> = ({ metrics }) => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const voice = useSelectedVoice();
    const { kind } = useVoiceFamily(voice);
    const running = snap.isRunning && vc.mode === "rvc";
    // No listed voice counts as a template voice (the default setup), like the engine section.
    const template = runsOnTemplateEngine(kind);
    const cushionNow = running && metrics?.cushion_ms != null ? metrics.cushion_ms : null;
    const adaptive = vc.outputCushion === "adaptive";
    // The prefill the cushion is capped at is the one in force: a voice that needs a larger buffer raises
    // it above the stored number.
    const inForce = running ? raisedOutputBuffer(metrics, vc.outputPrefillMs, vc.outputMaxQueueMs) : null;
    const capMs = inForce ? inForce.prefillMs : vc.outputPrefillMs;
    let cushionNote: string | undefined;
    if (adaptive) {
        cushionNote =
            cushionNow != null
                ? t("voice_changer.output_cushion_adaptive_live", { ms: formatMs(cushionNow), max: formatMs(capMs) })
                : t("voice_changer.output_cushion_adaptive_note", { max: formatMs(capMs) });
    } else if (cushionNow != null) {
        cushionNote = t("voice_changer.output_cushion_live", { ms: formatMs(cushionNow) });
    }
    const modelDivisor = vc.deviceBlock === "model_divisor";
    const blockState = deviceBlockNote(modelDivisor, template, running && metrics ? metrics.chunk_ms : null);
    let blockNote: string | undefined;
    if (blockState === "template_only") blockNote = t("voice_changer.device_block_template_only");
    else if (blockState === "live" && metrics) blockNote = t("voice_changer.device_block_live", { ms: formatMs(metrics.chunk_ms) });
    else if (blockState === "not_applied" && metrics) blockNote = t("voice_changer.device_block_not_applied", { ms: formatMs(metrics.chunk_ms) });
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.output_cushion"
                infoKey="voice_changer.output_cushion_info"
                value={vc.outputCushion}
                options={OUTPUT_CUSHIONS.map((c) => ({ value: c, label: t(`voice_changer.output_cushion_${c}`) }))}
                onChange={(v) => void vcApi.update({ outputCushion: v as OutputCushion })}
                note={cushionNote}
            />
            <VcChoiceSelect
                labelKey="voice_changer.device_block"
                infoKey="voice_changer.device_block_info"
                value={vc.deviceBlock}
                options={DEVICE_BLOCKS.map((b) => ({ value: b, label: t(`voice_changer.device_block_${b}`) }))}
                onChange={(v) => void vcApi.update({ deviceBlock: v as DeviceBlock })}
                note={blockNote}
                noteTone={blockState === "template_only" || blockState === "not_applied" ? "warn" : "muted"}
            />
            <AutoMsField
                labelKey="voice_changer.sola_search_ms"
                infoKey="voice_changer.sola_search_ms_info"
                value={vc.solaSearchMs}
                min={R.sola_search_ms.min}
                max={R.sola_search_ms.max}
                step={0.5}
                customStart={4}
                onCommit={(v) => void vcApi.update({ solaSearchMs: v })}
            />
            <AutoMsField
                labelKey="voice_changer.content_context_ms"
                infoKey="voice_changer.content_context_ms_info"
                value={vc.contentContextMs}
                min={R.content_context_ms.min}
                max={R.content_context_ms.max}
                step={10}
                customStart={1000}
                onCommit={(v) => void vcApi.update({ contentContextMs: v })}
                note={template ? undefined : t("voice_changer.template_only_note")}
            />
        </>
    );
};
