// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Pitch, formant and the classic phase vocoder.
import { VC_NUMBER_RANGES } from "@/core/services/voice_changer/contract";
import type { VoiceChangerState } from "@/core/services/voice_changer/schema";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputSelect } from "../components/input";
import { InspectorInfoButton, VcDetails, VcNote, VcNumberSlider, VcSliderField, VcSubHeader } from "./common";

const FORMANT = VC_NUMBER_RANGES.formant_semitones;

export const SoundSection: FC = () => {
    const { t } = useTranslation();
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const isClassic = voiceChanger.mode === "classic";
    const isRvc = voiceChanger.mode === "rvc";
    /** RVC mode: the classic tilt applies to converted audio only while the true formant shift is off. */
    const tiltInactive = isRvc && voiceChanger.formantSemitones !== 0;
    const handleVocoderOversampleChange = (value: string) => {
        const n = parseInt(value, 10) as VoiceChangerState["vocoderOversample"];
        if (n === 1 || n === 2 || n === 4 || n === 8 || n === 16 || n === 32) {
            void window.ApiServer.voiceChanger.setVocoderOversample(n);
        }
    };
    return (
        <>
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
            {isRvc ? (
                <VcNumberSlider
                    labelKey="voice_changer.formant_semitones"
                    infoKey="voice_changer.formant_semitones_info"
                    value={voiceChanger.formantSemitones}
                    min={FORMANT.min}
                    max={FORMANT.max}
                    step={0.1}
                    digits={1}
                    unitKey="voice_changer.pitch_unit_semitones"
                    lowLabelKey="voice_changer.label_deeper_formant"
                    highLabelKey="voice_changer.label_brighter_formant"
                    resetTo={FORMANT.default}
                    onChange={(v) => void vcApi.update({ formantSemitones: v })}
                />
            ) : null}
            <VcSliderField
                labelKey="voice_changer.formant_shift"
                inactive={tiltInactive}
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
            {tiltInactive ? <VcNote>{t("voice_changer.formant_tilt_rvc_note")}</VcNote> : null}
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
        </>
    );
};
