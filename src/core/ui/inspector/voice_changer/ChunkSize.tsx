// Voice-changer inspector: the Chunk size control (the `block_ms` setting), in the main RVC settings under the preset.
// Auto first ("Auto (N ms on this PC)": what the engine runs on Auto, read from the running engine, else "about N ms"
// estimated for the selected voice, preset and PC), then every size the engine plans. A voice the setting does not reach
// (a bring-your-own ONNX voice, a family that sets its own block) shows the control disabled with the reason. The rules and
// their sources are in `chunkSize.ts`.
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import { chunkSizeScope, chunkSizeView, type ChunkSizeScope, type ChunkSizeView } from "@/core/services/voice_changer/chunkSize";
import { familyOf } from "@/core/services/voice_changer/families";
import type { VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { formatChunkMs, formatMs } from "@/core/services/voice_changer/metrics";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { VcChoiceSelect, VcNote } from "./common";
import { useVoiceFamily } from "./hooks";

/**
 * What the Chunk size shows for the selected voice on this PC: the rules (`chunkSize.ts`) applied to the settings, the
 * system info's physical cores and the running engine's sample (`metrics`: its device buffer and the chunk it converts).
 * `runningMs`: the chunk the running engine reports converting per call (`engine_block_ms`), else `null`; `deviceMs`:
 * the Audio device buffer in ms (the running engine's, else the setting at 48 kHz).
 */
export function useChunkSize(
    voice: VcSlotInfo | undefined,
    metrics: VoiceChangerMetrics | null,
): { scope: ChunkSizeScope; view: ChunkSizeView; runningMs: number | null; deviceMs: number } {
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const sysinfo = useSnapshot(window.ApiServer.sysinfo.state);
    const vc = snap.voiceChanger;
    const { kind } = useVoiceFamily(voice);
    const running = snap.isRunning && vc.mode === "rvc";
    const scope = chunkSizeScope(kind, voice ? familyOf(voice) : null, voice?.family_options);
    const runningMs = running && metrics?.engine_block_ms != null && metrics.engine_block_ms > 0 ? metrics.engine_block_ms : null;
    const view = chunkSizeView({
        scope,
        profile: vc.rvcProfile,
        ep: vc.rvcExecutionProvider,
        physicalCores: sysinfo.info?.cpu.physical_cores,
        chunkFrames: vc.chunkFrames,
        runningDeviceMs: running ? metrics?.chunk_ms : null,
        runningBlockMs: runningMs,
        storedMs: vc.blockMs,
    });
    const deviceMs = running && metrics != null && metrics.chunk_ms > 0 ? metrics.chunk_ms : (vc.chunkFrames / 48_000) * 1000;
    return { scope, view, runningMs, deviceMs };
}

/** `metrics` is the running engine's latest sample, if any (the device buffer and the chunk it runs). */
export const ChunkSizeSelect: FC<{ voice: VcSlotInfo | undefined; metrics: VoiceChangerMetrics | null }> = ({ voice, metrics }) => {
    const { t } = useTranslation();
    const vc = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const { scope, view, runningMs, deviceMs } = useChunkSize(voice, metrics);
    const runningNote = runningMs !== null ? <VcNote>{t("voice_changer.chunk_size_running", { ms: formatChunkMs(runningMs) })}</VcNote> : null;

    if (!view.applies) {
        // A bring-your-own ONNX voice converts one Audio device buffer per call: say which, rather than "set by this voice".
        const fixedLabel =
            view.reason === "legacy"
                ? t("voice_changer.chunk_size_fixed_legacy", { ms: formatMs(deviceMs) })
                : t("voice_changer.chunk_size_fixed");
        return (
            <div className="flex w-full min-w-0 flex-col gap-1">
                <VcChoiceSelect
                    labelKey="voice_changer.chunk_size"
                    infoKey="voice_changer.chunk_size_info"
                    value="fixed"
                    options={[{ value: "fixed", label: fixedLabel }]}
                    onChange={() => {}}
                    disabled
                    note={t(`voice_changer.chunk_size_reason_${view.reason ?? "family"}`)}
                />
                {runningNote}
            </div>
        );
    }

    const label = (ms: number) =>
        view.runsAsMs !== null && ms === vc.blockMs
            ? t("voice_changer.chunk_size_value_runs_as", { set: ms, ms: view.runsAsMs })
            : t("voice_changer.chunk_size_value", { ms });
    return (
        <div className="flex w-full min-w-0 flex-col gap-1">
            <VcChoiceSelect
                labelKey="voice_changer.chunk_size"
                infoKey="voice_changer.chunk_size_info"
                value={String(vc.blockMs)}
                options={[
                    {
                        value: "0",
                        label:
                            view.autoMs === null
                                ? t("voice_changer.chunk_size_auto")
                                : t(view.autoExact ? "voice_changer.chunk_size_auto_ms" : "voice_changer.chunk_size_auto_about_ms", { ms: view.autoMs }),
                    },
                    ...view.choices.map((ms) => ({ value: String(ms), label: label(ms) })),
                ]}
                onChange={(v) => void vcApi.update({ blockMs: Number(v) })}
                note={scope.kind === "ddsp" ? t("voice_changer.chunk_size_ddsp_note") : undefined}
            />
            {view.shortChunk ? (
                <div className="alert alert-warning alert-soft text-xs leading-snug">
                    {view.shortChunk.fewCores
                        ? t("voice_changer.chunk_size_few_cores_note", {
                              cores: view.shortChunk.cores,
                              block: view.shortChunk.blockMs,
                              current: view.shortChunk.currentMs,
                          })
                        : t("voice_changer.chunk_size_short_note", {
                              block: view.shortChunk.blockMs,
                              current: view.shortChunk.currentMs,
                          })}
                </div>
            ) : null}
            {runningNote}
        </div>
    );
};
