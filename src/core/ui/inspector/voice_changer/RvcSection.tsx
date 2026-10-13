// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// RVC mode (the Sigil Voice Engine): the voice library, the models the setup uses, the preset,
// the per-voice settings, timing, the engine's expert settings, base model paths, and benchmark /
// tuning. Only values this build runs are offered (`voice_changer_get_capabilities`); a stored value
// it cannot run stays visible with what runs instead. P5-G: a P5 family voice (zero-shot, singing,
// sidecar, ...) gets its family section under the library and only the controls its family answers.
import type { VoiceChangerMetrics } from "@/core/services/voice_changer";
import {
    F0_RANGES,
    VC_NUMBER_RANGES,
    type F0Range,
    type PitchExtractorId,
    type RvcProfileId,
} from "@/core/services/voice_changer/contract";
import { showsControl } from "@/core/services/voice_changer/families";
import type { VcFamilyControl, VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { pitchExtractorOptions, profileOptions } from "@/core/services/voice_changer/options";
import { voiceModelIds } from "@/core/services/voice_changer/requirements";
import { isTemplateSlot } from "@/core/services/voice_changer/slots";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { ModelRequirement } from "../components/ModelRequirement";
import { BenchmarkPanel } from "./BenchmarkPanel";
import { VcChoiceSelect, VcDetails, VcNote, VcNumberSlider, VcPathField, VcSliderField, VcSubHeader } from "./common";
import { ChunkSizeSelect } from "./ChunkSize";
import { EngineSection, LookaheadMsSelect } from "./EngineSection";
import { FamilySection } from "./FamilySection";
import { useSelectedVoice, useVoiceFamily } from "./hooks";
import { VoiceLibrary } from "./VoiceLibrary";

const R = VC_NUMBER_RANGES;

/** Preset select (Streamer / Low latency / Quality / Studio-TRT). */
const PresetSelect: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const options = profileOptions(snap.capabilities as typeof window.ApiServer.voiceChanger.state.capabilities, vc.rvcProfile);
    const current = options.find((o) => o.value === vc.rvcProfile);
    // The few-cores warning about a Chunk size set shorter than the Streamer keeps up with on this PC is part of the
    // Chunk size control (`ChunkSize.tsx`), right under the size it is about.
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.rvc_profile"
                infoKey="voice_changer.rvc_profile_info"
                value={vc.rvcProfile}
                options={options.map((o) => ({
                    value: o.value,
                    label: o.available
                        ? t(`voice_changer.rvc_profile_${o.value}`)
                        : t("voice_changer.option_unavailable", { name: t(`voice_changer.rvc_profile_${o.value}`) }),
                    disabled: !o.available && o.value !== vc.rvcProfile,
                }))}
                onChange={(v) => void window.ApiServer.voiceChanger.applyRvcProfile(v as RvcProfileId)}
                note={current && !current.available ? t("voice_changer.rvc_profile_studio_trt_unavailable") : undefined}
                noteTone="warn"
            />
            {vc.rvcProfile === "quality" || vc.rvcProfile === "studio_trt" ? (
                <div className="alert alert-info alert-soft text-xs leading-snug">{t("voice_changer.rvc_quality_banner")}</div>
            ) : null}
        </>
    );
};

/** F0 method: every method this build runs, the stored one always visible with what runs instead. */
const PitchSelect: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const caps = snap.capabilities as typeof window.ApiServer.voiceChanger.state.capabilities;
    const options = pitchExtractorOptions(caps, vc.pitchExtractor, vc.rvcExecutionProvider);
    const current = options.find((o) => o.value === vc.pitchExtractor);
    const cpu = vc.rvcExecutionProvider === "cpu";
    const label = (p: PitchExtractorId) => t(`voice_changer.f0_name_${p}`);
    let note: string | undefined;
    if (current?.runsAs) {
        note = !current.available
            ? t("voice_changer.pitch_extractor_runs_as", { name: label(current.value), run: label(current.runsAs as PitchExtractorId) })
            : t("voice_changer.pitch_extractor_cpu_note", { name: current.value });
    }
    return (
        <VcChoiceSelect
            labelKey="voice_changer.pitch_extractor"
            infoKey="voice_changer.pitch_extractor_info"
            value={vc.pitchExtractor}
            options={options.map((o) => {
                const gpuOnlyOnCpu = o.gpuOnly && cpu;
                const name = label(o.value);
                return {
                    value: o.value,
                    label: !o.available
                        ? t("voice_changer.option_unavailable", { name })
                        : gpuOnlyOnCpu
                          ? t("voice_changer.pitch_extractor_gpu_only", { name })
                          : name,
                    disabled: o.value !== vc.pitchExtractor && (!o.available || gpuOnlyOnCpu),
                };
            })}
            // A switch while RVC runs (the voice reloads its F0 extractor in the background).
            onChange={(v) => void window.ApiServer.voiceChanger.update({ pitchExtractor: v as PitchExtractorId })}
            note={note}
            noteTone="warn"
        />
    );
};

/** The models the selected voice and F0 method use (download with consent, or import). */
const VoiceRequirements: FC<{ voice: VcSlotInfo | undefined }> = ({ voice }) => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const ids = voiceModelIds({
        mode: vc.mode,
        pitchExtractor: vc.pitchExtractor,
        rvcExecutionProvider: vc.rvcExecutionProvider,
        fcpePath: vc.rvcFcpePath,
        rmvpePath: vc.rvcRmvpePath,
        voice: voice ?? null,
        capabilities: snap.capabilities as typeof window.ApiServer.voiceChanger.state.capabilities,
    });
    if (ids.length === 0) return null;
    return <ModelRequirement ids={ids} reason={t("voice_changer.voice_models_reason")} hideWhenReady />;
};

/**
 * Settings saved with the selected voice (`slot.json`): F0, range, speaker, index, protect, envelope, variation.
 * A family voice shows only what its family answers (`VcFamilyCapability.controls`, F0 only when it tracks pitch).
 */
const VoiceSettings: FC<{ voice: VcSlotInfo | undefined }> = ({ voice }) => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const lib = useSnapshot(window.ApiServer.voiceChanger.library.state);
    const vc = snap.voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const { kind, controls } = useVoiceFamily(voice);
    const family = kind === "family";
    const template = voice ? isTemplateSlot(voice) : false;
    const noIndex = !!voice && lib.slotStore && !voice.has_index;
    const speakers = voice?.speaker_count ?? null;
    const shows = (c: VcFamilyControl) => showsControl(controls, c);
    // RVC voices track pitch unless trained without it; family voices when their family does.
    const tracksPitch = family ? !!controls?.needsPitch : voice?.f0 !== false;
    if (family) {
        const any =
            tracksPitch ||
            (["index", "protect", "rms_mix", "noise_scale"] as const).some(shows) ||
            (shows("speaker") && speakers != null && speakers > 1);
        if (!any) return <VcNote>{t("voice_changer.family_voice_settings_none")}</VcNote>;
    }
    return (
        <>
            {tracksPitch ? <PitchSelect /> : family ? null : <VcNote>{t("voice_changer.voice_no_pitch_note")}</VcNote>}
            {tracksPitch ? (
                <VcChoiceSelect
                    labelKey="voice_changer.f0_range"
                    infoKey="voice_changer.f0_range_info"
                    value={vc.f0Range}
                    options={F0_RANGES.map((r) => ({ value: r, label: t(`voice_changer.f0_range_${r}`) }))}
                    onChange={(v) => void vcApi.update({ f0Range: v as F0Range })}
                />
            ) : null}
            {shows("speaker") && speakers != null && speakers > 1 ? (
                <VcChoiceSelect
                    labelKey="voice_changer.speaker"
                    value={String(vc.speakerId)}
                    options={Array.from({ length: Math.max(speakers, vc.speakerId + 1) }, (_, i) => ({
                        value: String(i),
                        label: t("voice_changer.speaker_n", { n: i }),
                        disabled: i >= speakers,
                    }))}
                    onChange={(v) => void vcApi.update({ speakerId: Number(v) })}
                />
            ) : null}
            {shows("index") ? (
                <>
                    <VcNumberSlider
                        labelKey="voice_changer.index_rate"
                        infoKey="voice_changer.index_rate_info"
                        value={vc.indexRate}
                        min={R.index_rate.min}
                        max={R.index_rate.max}
                        step={0.01}
                        inactive={noIndex}
                        lowLabelKey="voice_changer.label_own_timbre"
                        highLabelKey="voice_changer.label_voice_timbre"
                        onChange={(v) => void vcApi.update({ indexRate: v })}
                    />
                    {noIndex ? <VcNote>{t("voice_changer.index_rate_no_index")}</VcNote> : null}
                </>
            ) : null}
            {shows("protect") ? (
                <VcNumberSlider
                    labelKey="voice_changer.protect"
                    infoKey="voice_changer.protect_info"
                    value={vc.protect}
                    min={R.protect.min}
                    max={R.protect.max}
                    step={0.01}
                    lowLabelKey="voice_changer.label_protect_more"
                    highLabelKey="voice_changer.label_protect_off"
                    onChange={(v) => void vcApi.update({ protect: v })}
                />
            ) : null}
            {shows("rms_mix") ? (
                <VcNumberSlider
                    labelKey="voice_changer.rms_mix"
                    infoKey="voice_changer.rms_mix_info"
                    value={vc.rmsMix}
                    min={R.rms_mix.min}
                    max={R.rms_mix.max}
                    step={0.01}
                    lowLabelKey="voice_changer.label_follow_your_voice"
                    highLabelKey="voice_changer.label_model_envelope"
                    resetTo={R.rms_mix.default}
                    onChange={(v) => void vcApi.update({ rmsMix: v })}
                />
            ) : null}
            {(family ? shows("noise_scale") : template || !voice) ? (
                <VcNumberSlider
                    labelKey="voice_changer.noise_scale"
                    infoKey="voice_changer.noise_scale_info"
                    value={vc.noiseScale}
                    min={R.noise_scale.min}
                    max={R.noise_scale.max}
                    step={0.01}
                    inactive={!template && !family}
                    lowLabelKey="voice_changer.label_steadier"
                    highLabelKey="voice_changer.label_more_variation"
                    resetTo={R.noise_scale.default}
                    onChange={(v) => void vcApi.update({ noiseScale: v })}
                />
            ) : null}
        </>
    );
};

/**
 * The RVC settings. `crossfadeMaxMs` / `crossfadeShownMs` come from the inspector (they depend on the running
 * engine's block length); `metrics` is the running engine's latest sample.
 */
export const RvcSection: FC<{ crossfadeMaxMs: number; crossfadeShownMs: number; metrics: VoiceChangerMetrics | null }> = ({
    crossfadeMaxMs,
    crossfadeShownMs,
    metrics,
}) => {
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const voice = useSelectedVoice();
    const { kind } = useVoiceFamily(voice);
    const template = voice ? isTemplateSlot(voice) : false;
    /** Extra context and crossfade are read at harness construction: restart once per committed change. */
    const commitHarnessParams = () => void vcApi.commitHarnessParams();
    return (
        <>
            <VcSubHeader titleKey="voice_changer.section_voices" infoKey="voice_changer.section_voices_info" />
            <VoiceLibrary />
            <VoiceRequirements voice={voice} />
            <FamilySection voice={voice} />

            <VcSubHeader titleKey="voice_changer.section_preset" infoKey="voice_changer.section_preset_info" />
            <PresetSelect />
            <ChunkSizeSelect voice={voice} metrics={metrics} />

            <VcSubHeader titleKey="voice_changer.section_voice_settings" infoKey="voice_changer.section_voice_settings_info" />
            <VoiceSettings voice={voice} />

            <VcDetails
                titleKey="voice_changer.section_rvc_tuning"
                infoKey="voice_changer.section_rvc_tuning_info_v2"
                defaultOpen={voiceChanger.rvcProfile === "quality"}
            >
                <LookaheadMsSelect template={kind !== "legacy"} family={kind === "family"} />
                <VcSliderField
                    labelKey="voice_changer.extra_context_ms"
                    controlsRow={
                        <>
                            <input
                                type="number"
                                className="input input-bordered input-sm w-24 shrink-0 font-mono"
                                min={R.extra_context_ms.min}
                                max={R.extra_context_ms.max}
                                step={1}
                                value={voiceChanger.extraContextMs}
                                onChange={(e) => {
                                    const x = parseFloat(e.target.value);
                                    if (!Number.isNaN(x)) {
                                        vcApi.state.voiceChanger.extraContextMs = Math.min(R.extra_context_ms.max, Math.max(R.extra_context_ms.min, x));
                                    }
                                }}
                                onBlur={commitHarnessParams}
                            />
                            <UnitMs />
                        </>
                    }
                    rangeMin={0}
                    rangeMax={template ? 1000 : 500}
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
                                        vcApi.state.voiceChanger.crossfadeMs = Math.min(crossfadeMaxMs, Math.max(0, x));
                                    }
                                }}
                                onBlur={commitHarnessParams}
                            />
                            <UnitMs />
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
            </VcDetails>

            <EngineSection voice={voice} metrics={metrics} />

            <VcDetails
                titleKey="voice_changer.section_rvc_bases"
                infoKey="voice_changer.section_rvc_bases_info"
                defaultOpen={!!(voiceChanger.rvcHubertPath.trim() || voiceChanger.rvcFcpePath.trim() || voiceChanger.rvcRmvpePath.trim())}
            >
                <VcPathField
                    labelKey="voice_changer.rvc_hubert_path"
                    value={voiceChanger.rvcHubertPath}
                    onChange={(path) => {
                        vcApi.state.voiceChanger.rvcHubertPath = path;
                    }}
                    onCommit={() => void vcApi.update({})}
                />
                <VcPathField
                    labelKey="voice_changer.rvc_fcpe_path"
                    value={voiceChanger.rvcFcpePath}
                    onChange={(path) => {
                        vcApi.state.voiceChanger.rvcFcpePath = path;
                    }}
                    onCommit={() => void vcApi.update({})}
                />
                <VcPathField
                    labelKey="voice_changer.rvc_rmvpe_path"
                    value={voiceChanger.rvcRmvpePath}
                    onChange={(path) => {
                        vcApi.state.voiceChanger.rvcRmvpePath = path;
                    }}
                    onCommit={() => void vcApi.update({})}
                />
            </VcDetails>

            <BenchmarkPanel />
        </>
    );
};

const UnitMs: FC = () => {
    const { t } = useTranslation();
    return <span className="shrink-0 text-xs text-base-content/60">{t("voice_changer.unit_ms")}</span>;
};
