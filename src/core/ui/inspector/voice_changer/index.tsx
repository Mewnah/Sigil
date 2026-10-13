// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
//
// Layout: index (status, TensorRT builds, metrics, mode) -> RvcSection (voice library, the selected voice's family
// section (P5-G), preset, voice settings, timing, engine, base models, benchmark / tuning) / SidecarSection / OnnxSection (per mode) -> DevicesSection ->
// SoundSection -> LevelsSection (with the input denoiser) -> TimingSection, plus the RVC ConsentDialog. Shared
// building blocks are in common.tsx. What this build runs comes from `voice_changer_get_capabilities`
// (src/core/services/voice_changer/ipc.ts), refreshed when the panel opens and when voice-changer models install.
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import { crossfadeEngineBlockMs, crossfadeMaxMsForEngine } from "@/core/services/voice_changer/chunkFrames";
import { epStatus } from "@/core/services/voice_changer/metrics";
import type { VoiceChangerState } from "@/core/services/voice_changer/schema";
import classNames from "classnames";
import { FC, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { isTauri } from "@tauri-apps/api/core";
import { RiMagicFill, RiPlayFill, RiStopFill } from "react-icons/ri";
import Tooltip from "../../dropdown/Tooltip";
import Inspector from "../components";
import { InputSelect } from "../components/input";
import { InspectorInfoButton, patchVcStructural } from "./common";
import { useChunkSize } from "./ChunkSize";
import { ConsentDialog } from "./ConsentDialog";
import { DevicesSection } from "./Devices";
import { useSelectedVoice, useVoiceFamily } from "./hooks";
import { LevelsSection } from "./LevelsSection";
import { VcEpStatus, VcMetricsPanel, VcTrtProgressPanel } from "./Metrics";
import { OnnxSection, SidecarSection } from "./ModeSections";
import { RvcSection } from "./RvcSection";
import { SoundSection } from "./SoundSection";
import { TimingSection } from "./TimingSection";

/** A key that changes when a voice-changer model (`vc/...`) is installed or removed. */
function useInstalledVcModelsKey(): string {
    const states = useSnapshot(window.ApiServer.models.state).states;
    return useMemo(
        () =>
            Object.entries(states)
                .filter(([id, st]) => id.startsWith("vc/") && st?.state === "installed")
                .map(([id]) => id)
                .sort()
                .join(","),
        [states],
    );
}

export const Inspector_VoiceChanger: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const voiceChanger = snap.voiceChanger;
    const isRunning = snap.isRunning;
    const isSidecar = voiceChanger.mode === "sidecar";
    const isOnnx = voiceChanger.mode === "onnx";
    const isRvc = voiceChanger.mode === "rvc";
    const vcApi = window.ApiServer.voiceChanger;
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
    // A template voice's overlap is clamped to its engine block (not the device chunk): offer up to that. A family
    // voice's is clamped to the family's own block, so it keeps the chunk cap.
    const selectedVoice = useSelectedVoice();
    const { kind: voiceKind } = useVoiceFamily(selectedVoice);
    // On Auto the engine still converts a fixed chunk (the running engine's own, else what Auto runs on this PC): the
    // crossfade is clamped at it, so the control offers up to it rather than up to the 10 ms device buffer.
    const chunk = useChunkSize(selectedVoice, metrics);
    const engineBlock = crossfadeEngineBlockMs(voiceKind, isRvc, voiceChanger.blockMs, chunk.runningMs ?? chunk.view.autoMs ?? 0);
    const crossfadeMaxMs = crossfadeMaxMsForEngine(voiceChanger.chunkFrames, isRunning ? metrics?.chunk_ms : null, engineBlock);
    const crossfadeShownMs = Math.min(voiceChanger.crossfadeMs, crossfadeMaxMs);
    const installedVcModels = useInstalledVcModelsKey();

    // What this build runs depends on installed models too (e.g. PESTO): refresh on open and on installs.
    useEffect(() => {
        if (!isTauri()) return;
        void vcApi.refreshCapabilities();
    }, [vcApi, installedVcModels]);

    // Opening the panel while stopped reads the metrics once: when the opt-in prewarm is on
    // (SIGIL_VC_PREWARM=1, off by default) the backend takes that as the cue to warm the GPU path the saved
    // settings point at (P3B-H; once per process, idle priority); otherwise it changes nothing.
    useEffect(() => {
        if (!isTauri() || window.ApiServer.voiceChanger.state.isRunning) return;
        void vcApi.getMetrics();
    }, [vcApi]);

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

    /** Start / stop; a refused start is already reported by the service (toast), so only swallow it here. */
    const toggleEngine = () => {
        vcApi.toggle().catch(() => {});
    };

    /** RVC without the acknowledgement does not start; the service flags it and the notice opens. */
    const handleToggle = toggleEngine;

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
                <VcTrtProgressPanel />
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
                            { label: t("voice_changer.mode_rvc_engine"), value: "rvc" },
                        ]}
                    />
                    <div className="pb-0.5">
                        <InspectorInfoButton text={t("voice_changer.section_how_it_works_info")} />
                    </div>
                </div>

                {isRvc && <RvcSection crossfadeMaxMs={crossfadeMaxMs} crossfadeShownMs={crossfadeShownMs} metrics={metrics} />}
                {isSidecar && <SidecarSection />}
                {isOnnx && <OnnxSection />}
                <DevicesSection />
                <SoundSection />
                <LevelsSection />
                <TimingSection metrics={metrics} />
            </Inspector.Content>

            <ConsentDialog />
        </Inspector.Body>
    );
};

export default Inspector_VoiceChanger;
