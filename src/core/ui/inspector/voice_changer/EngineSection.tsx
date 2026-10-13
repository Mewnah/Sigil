// Voice-changer inspector (owned by P3-G): the Sigil Voice Engine's expert settings for RVC mode.
// Execution provider (with the TensorRT-RTX and DirectML high-priority accelerators), precision,
// every engine lever (auto / on / off, with what this build and EP run and what the running engine
// really uses), silence gate, hot voices and VRAM budget, GPU index, the legacy VRAM cap, and the
// debug snapshot. Only values this build runs are offered; a stored value it cannot run stays
// visible with what runs instead (`options.ts`). P5-G: a P5 family voice is not a bring-your-own ONNX voice
// (no legacy notes); its family reports the levers it really uses while it runs, and the silence threshold shows
// when the family answers it.
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import {
    LEVER_MODES,
    rvcBaseEp,
    VC_LEVERS,
    VC_NUMBER_RANGES,
    type LeverMode,
    type Precision,
    type RvcExecutionProviderId,
} from "@/core/services/voice_changer/contract";
import type { VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { epDisplayName } from "@/core/services/voice_changer/metrics";
import {
    executionProviderOptions,
    generalLeverNotes,
    LEGACY_PATH_LEVERS,
    leverViews,
    nvidiaAdapterPresent,
    precisionOptions,
} from "@/core/services/voice_changer/options";
import { runsOnTemplateEngine, showsControl } from "@/core/services/voice_changer/families";
import { leverStateKey } from "@/core/services/voice_changer/tuning";
import type { VoiceChangerState } from "@/core/services/voice_changer/schema";
import classNames from "classnames";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import { InputContainer } from "../components/input";
import { InspectorInfoButton, VcChoiceSelect, VcDetails, VcDraftNumberInput, VcNote, VcNumberSlider } from "./common";
import { useVoiceFamily } from "./hooks";

const R = VC_NUMBER_RANGES;

/** SVE blocks offered (ms); 0 = from the chunk size. */
const BLOCK_MS_CHOICES = [0, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200] as const;

/** `template`: the voice runs on the engine (not a bring-your-own ONNX voice); `family`: a P5 family voice (its preset may round the block). */
export const BlockMsSelect: FC<{ template: boolean; family?: boolean }> = ({ template, family }) => {
    const { t } = useTranslation();
    const vc = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const values: number[] = [...BLOCK_MS_CHOICES];
    if (!values.includes(vc.blockMs)) values.push(vc.blockMs);
    values.sort((a, b) => a - b);
    return (
        <VcChoiceSelect
            labelKey="voice_changer.block_ms"
            infoKey="voice_changer.block_ms_info"
            value={String(vc.blockMs)}
            options={values.map((ms) => ({
                value: String(ms),
                label: ms === 0 ? t("voice_changer.block_ms_auto") : t("voice_changer.block_ms_value", { ms }),
            }))}
            onChange={(v) => void vcApi.update({ blockMs: Number(v) })}
            note={family ? t("voice_changer.family_block_note") : template ? undefined : t("voice_changer.template_only_note")}
        />
    );
};

/** Lookahead choices (ms); 0 = off. The engine rounds up to its 10 ms frames and allows up to 80. */
const LOOKAHEAD_MS_CHOICES = [0, 10, 20, 40, 60, 80] as const;

/** SVE lookahead (P3-I0): future context decoded into every returned sample, at that much delay. */
export const LookaheadMsSelect: FC<{ template: boolean; family?: boolean }> = ({ template, family }) => {
    const { t } = useTranslation();
    const vc = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const values: number[] = [...LOOKAHEAD_MS_CHOICES];
    if (!values.includes(vc.lookaheadMs)) values.push(vc.lookaheadMs);
    values.sort((a, b) => a - b);
    return (
        <VcChoiceSelect
            labelKey="voice_changer.lookahead_ms"
            infoKey="voice_changer.lookahead_ms_info_v2"
            value={String(vc.lookaheadMs)}
            options={values.map((ms) => ({
                value: String(ms),
                label: ms === 0 ? t("voice_changer.lookahead_ms_off") : t("voice_changer.lookahead_ms_value", { ms }),
            }))}
            onChange={(v) => void vcApi.update({ lookaheadMs: Number(v) })}
            note={family ? t("voice_changer.family_block_note") : template ? undefined : t("voice_changer.template_only_note")}
        />
    );
};

const LeverRow: FC<{
    lever: (typeof VC_LEVERS)[number];
    mode: LeverMode;
    available: boolean;
    applies: boolean;
    inUse: boolean | null;
    note: string | null;
    ep: RvcExecutionProviderId;
    legacyVoice: boolean;
}> = ({ lever, mode, available, applies, inUse, note, ep, legacyVoice }) => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    let status: { text: string; tone: "muted" | "ok" | "warn" } | null = null;
    if (inUse !== null) {
        status = inUse
            ? { text: t("voice_changer.lever_in_use"), tone: "ok" }
            : { text: note ?? t("voice_changer.lever_not_used"), tone: note && mode === "on" ? "warn" : "muted" };
    } else if (!available) {
        status = { text: t("voice_changer.lever_unavailable"), tone: mode === "on" ? "warn" : "muted" };
    } else if (!applies) {
        status = { text: t("voice_changer.lever_not_for_ep", { ep: epDisplayName(rvcBaseEp(ep)) }), tone: mode === "on" ? "warn" : "muted" };
    } else if (legacyVoice && !LEGACY_PATH_LEVERS.includes(lever)) {
        status = { text: t("voice_changer.lever_imported_only"), tone: mode === "on" ? "warn" : "muted" };
    }
    return (
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2 gap-y-0.5">
            <div className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 truncate text-xs font-medium text-base-content/80">{t(`voice_changer.lever_name_${lever}`)}</span>
                <InspectorInfoButton text={t(`voice_changer.lever_info_${lever}`)} />
            </div>
            <select
                className="select select-bordered select-xs w-24"
                aria-label={t(`voice_changer.lever_name_${lever}`)}
                value={mode}
                onChange={(e) => void vcApi.update({ [leverStateKey(lever)]: e.target.value as LeverMode } as Partial<VoiceChangerState>)}
            >
                {LEVER_MODES.map((m) => (
                    <option key={m} value={m} disabled={m === "on" && !available && mode !== "on"}>
                        {t(`voice_changer.lever_mode_${m}`)}
                    </option>
                ))}
            </select>
            {status ? (
                <span
                    className={classNames(
                        "col-span-2 text-xs leading-snug",
                        status.tone === "ok" ? "text-success" : status.tone === "warn" ? "text-warning" : "text-base-content/50",
                    )}
                >
                    {status.text}
                </span>
            ) : null}
        </div>
    );
};

/** RVC engine settings (expert). `metrics` is the running engine's latest sample, if any. */
export const EngineSection: FC<{ voice: VcSlotInfo | undefined; metrics: VoiceChangerMetrics | null }> = ({ voice, metrics }) => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    // Whether GPU priority applies on the CUDA rung depends on an NVIDIA adapter (system info).
    const sysinfo = useSnapshot(window.ApiServer.sysinfo.state);
    const vc = snap.voiceChanger;
    const caps = snap.capabilities as typeof window.ApiServer.voiceChanger.state.capabilities;
    const vcApi = window.ApiServer.voiceChanger;
    const running = snap.isRunning && vc.mode === "rvc";
    const levers = running ? metrics?.levers ?? null : null;
    const { kind, controls } = useVoiceFamily(voice);
    // No listed voice counts as a template voice (the default setup), like before P5.
    const template = runsOnTemplateEngine(kind);
    const legacyVoice = kind === "legacy";
    const familyVoice = kind === "family";
    const base = rvcBaseEp(vc.rvcExecutionProvider);

    const epOptions = executionProviderOptions(caps, vc.rvcExecutionProvider);
    const epChoice = epOptions.find((o) => o.value === vc.rvcExecutionProvider);
    const precOptions = precisionOptions(caps, vc.precision);
    const precChoice = precOptions.find((o) => o.value === vc.precision);
    const views = leverViews(VC_LEVERS, caps, vc.rvcExecutionProvider, levers, nvidiaAdapterPresent(sysinfo.info));
    const otherNotes = generalLeverNotes(levers, VC_LEVERS);
    const leverModes = VC_LEVERS.map((l) => vc[leverStateKey(l)] as LeverMode);
    const anyNotAuto = leverModes.some((m) => m !== "auto");
    const accelerator = epChoice && epChoice.available && (vc.rvcExecutionProvider === "tensorrt_rtx" || vc.rvcExecutionProvider === "dml_hp");

    return (
        <VcDetails titleKey="voice_changer.section_engine" infoKey="voice_changer.section_engine_info">
            <VcChoiceSelect
                labelKey="voice_changer.onnx_execution_provider"
                infoKey="voice_changer.rvc_ep_info"
                value={vc.rvcExecutionProvider}
                options={epOptions.map((o) => ({
                    value: o.value,
                    label: o.available
                        ? t(`voice_changer.ep_name_${o.value}`)
                        : t("voice_changer.option_unavailable", { name: t(`voice_changer.ep_name_${o.value}`) }),
                    disabled: !o.available && o.value !== vc.rvcExecutionProvider,
                }))}
                onChange={(v) => void vcApi.update({ rvcExecutionProvider: v as RvcExecutionProviderId })}
                note={
                    epChoice?.runsAs
                        ? t("voice_changer.ep_runs_as", {
                              name: t(`voice_changer.ep_name_${epChoice.value}`),
                              base: epDisplayName(epChoice.runsAs),
                          })
                        : vc.rvcExecutionProvider === "tensorrt_rtx" && accelerator
                          ? t("voice_changer.trt_first_start_note")
                          : undefined
                }
                noteTone={epChoice?.runsAs ? "warn" : "muted"}
            />
            <VcChoiceSelect
                labelKey="voice_changer.precision"
                infoKey="voice_changer.precision_info"
                value={vc.precision}
                options={precOptions.map((o) => ({
                    value: o.value,
                    label: o.available
                        ? t(`voice_changer.precision_${o.value}`)
                        : t("voice_changer.option_unavailable", { name: t(`voice_changer.precision_${o.value}`) }),
                    disabled: !o.available && o.value !== vc.precision,
                }))}
                onChange={(v) => void vcApi.update({ precision: v as Precision })}
                note={
                    precChoice?.runsAs
                        ? t("voice_changer.precision_runs_as", { name: precChoice.value, run: precChoice.runsAs })
                        : legacyVoice
                          ? t("voice_changer.template_only_note")
                          : undefined
                }
                noteTone={precChoice?.runsAs ? "warn" : "muted"}
            />

            <div className="flex flex-col gap-2 rounded-lg border border-base-content/10 p-2">
                <div className="flex items-center gap-1">
                    <span className="min-w-0 flex-1 text-xs font-semibold uppercase tracking-wide text-base-content/55">
                        {t("voice_changer.levers_title")}
                    </span>
                    <InspectorInfoButton text={t("voice_changer.levers_info")} />
                </div>
                {legacyVoice ? <VcNote>{t("voice_changer.levers_legacy_note")}</VcNote> : null}
                {familyVoice ? <VcNote>{t("voice_changer.family_levers_note")}</VcNote> : null}
                {views.map((v, i) => (
                    <LeverRow
                        key={v.lever}
                        lever={v.lever}
                        mode={leverModes[i]}
                        available={v.available}
                        applies={v.applies}
                        inUse={v.inUse}
                        note={v.note}
                        ep={vc.rvcExecutionProvider}
                        legacyVoice={legacyVoice}
                    />
                ))}
                {otherNotes.map((n) => (
                    <VcNote key={n} tone="warn">
                        {n}
                    </VcNote>
                ))}
                {running ? <VcNote>{t("voice_changer.levers_restart_note")}</VcNote> : null}
                {anyNotAuto ? (
                    <button
                        type="button"
                        className="btn btn-ghost btn-xs self-start"
                        onClick={() => {
                            const patch: Partial<VoiceChangerState> = {};
                            for (const l of VC_LEVERS) (patch as Record<string, unknown>)[leverStateKey(l)] = "auto";
                            void vcApi.update(patch);
                        }}
                    >
                        {t("voice_changer.levers_reset")}
                    </button>
                ) : null}
            </div>

            <label className="label cursor-pointer justify-start gap-2 py-0">
                <input
                    type="checkbox"
                    className="checkbox checkbox-sm checkbox-primary"
                    checked={vc.silenceSkip}
                    onChange={(e) => void vcApi.update({ silenceSkip: e.target.checked })}
                />
                <span className="label-text text-sm">{t("voice_changer.silence_skip")}</span>
            </label>
            {vc.silenceSkip && (template || (familyVoice && showsControl(controls, "silence_db"))) ? (
                <VcNumberSlider
                    labelKey="voice_changer.silence_db"
                    infoKey="voice_changer.silence_db_info"
                    value={vc.silenceDb}
                    min={R.silence_db.min}
                    max={R.silence_db.max}
                    step={1}
                    digits={0}
                    unitKey="voice_changer.unit_dbfs"
                    lowLabelKey="voice_changer.label_converts_quieter"
                    highLabelKey="voice_changer.label_skips_more"
                    resetTo={R.silence_db.default}
                    onChange={(v) => void vcApi.update({ silenceDb: v })}
                />
            ) : null}

            <div className="grid w-full min-w-0 grid-cols-1 gap-2 ui-sm:grid-cols-2">
                <VcChoiceSelect
                    labelKey="voice_changer.hot_slots"
                    infoKey="voice_changer.hot_slots_info"
                    value={String(vc.hotSlots)}
                    options={Array.from({ length: Math.max(caps.max_hot_slots, vc.hotSlots) }, (_, i) => ({
                        value: String(i + 1),
                        label: String(i + 1),
                        disabled: i + 1 > caps.max_hot_slots,
                    }))}
                    onChange={(v) => void vcApi.update({ hotSlots: Number(v) })}
                />
                <InputContainer label="voice_changer.vram_budget_mb" vertical>
                    <div className="flex items-center gap-1">
                        <VcDraftNumberInput
                            value={vc.vramBudgetMb}
                            min={R.vram_budget_mb.min}
                            max={R.vram_budget_mb.max}
                            step={128}
                            ariaLabel={t("voice_changer.vram_budget_mb")}
                            onCommit={(v) => void vcApi.update({ vramBudgetMb: Math.round(v) })}
                        />
                        <InspectorInfoButton text={t("voice_changer.vram_budget_info")} />
                    </div>
                </InputContainer>
            </div>
            {vc.hotSlots > 1 && base === "cpu" ? <VcNote>{t("voice_changer.hot_slots_cpu_note")}</VcNote> : null}

            <div className="grid w-full min-w-0 grid-cols-1 gap-2 ui-sm:grid-cols-2">
                <InputContainer label="voice_changer.speaker_id" vertical>
                    <input
                        type="number"
                        className="input input-bordered input-sm w-full"
                        min={0}
                        step={1}
                        value={vc.speakerId}
                        onChange={(e) => {
                            const x = parseInt(e.target.value, 10);
                            if (!Number.isNaN(x)) vcApi.state.voiceChanger.speakerId = Math.max(0, x);
                        }}
                        onBlur={() => void vcApi.update({})}
                    />
                </InputContainer>
                <InputContainer label="voice_changer.gpu_device_index" vertical>
                    <input
                        type="number"
                        className="input input-bordered input-sm w-full"
                        min={0}
                        step={1}
                        value={vc.gpuDeviceIndex}
                        onChange={(e) => {
                            const x = parseInt(e.target.value, 10);
                            if (!Number.isNaN(x)) vcApi.state.voiceChanger.gpuDeviceIndex = Math.max(0, x);
                        }}
                        onBlur={() => void vcApi.update({})}
                    />
                </InputContainer>
            </div>
            {legacyVoice || !voice ? (
                <div className="flex flex-col gap-1">
                    <InputContainer label="voice_changer.vram_cap_mb" className={base === "cuda" ? undefined : "opacity-50"}>
                        <input
                            type="number"
                            className="input input-bordered input-sm w-full"
                            min={R.vram_cap_mb.min}
                            max={R.vram_cap_mb.max}
                            step={64}
                            disabled={base !== "cuda"}
                            value={vc.vramCapMb}
                            onChange={(e) => {
                                const x = parseInt(e.target.value, 10);
                                if (!Number.isNaN(x)) vcApi.state.voiceChanger.vramCapMb = Math.min(R.vram_cap_mb.max, Math.max(R.vram_cap_mb.min, x));
                            }}
                            onBlur={() => void vcApi.update({})}
                        />
                    </InputContainer>
                    <VcNote>{t("voice_changer.vram_cap_cuda_only")}</VcNote>
                </div>
            ) : null}
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
    );
};
