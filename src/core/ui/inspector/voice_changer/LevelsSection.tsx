// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Input / output gain and the mic gate while TTS plays through the voice changer.
import { AUX_MIC_GATES, type AuxMicGate, type Denoiser } from "@/core/services/voice_changer/contract";
import { denoiserOptions } from "@/core/services/voice_changer/options";
import { denoiserModelIds } from "@/core/services/voice_changer/requirements";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputContainer, InputSelect } from "../components/input";
import { ModelRequirement } from "../components/ModelRequirement";
import { InspectorInfoButton, VcChoiceSelect, VcSubHeader } from "./common";

/** Input denoiser of the voice-changer path (every mode; speech-to-text keeps the raw microphone). */
const DenoiserSelect: FC = () => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const caps = snap.capabilities as typeof window.ApiServer.voiceChanger.state.capabilities;
    const options = denoiserOptions(caps, vc.denoiser);
    const current = options.find((o) => o.value === vc.denoiser);
    const name = (d: Denoiser) => t(`voice_changer.denoiser_${d}`);
    const ids = denoiserModelIds(vc.denoiser, caps);
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.denoiser"
                infoKey="voice_changer.denoiser_info"
                value={vc.denoiser}
                options={options.map((o) => ({
                    value: o.value,
                    label: o.available ? name(o.value) : t("voice_changer.option_unavailable", { name: name(o.value) }),
                    disabled: !o.available && o.value !== vc.denoiser,
                }))}
                onChange={(v) => void window.ApiServer.voiceChanger.update({ denoiser: v as Denoiser })}
                note={
                    current?.runsAs
                        ? current.runsAs === "off"
                            ? t("voice_changer.denoiser_runs_off", { name: name(current.value) })
                            : t("voice_changer.denoiser_runs_as", { name: name(current.value), run: name(current.runsAs as Denoiser) })
                        : undefined
                }
                noteTone="warn"
            />
            {ids.length > 0 ? <ModelRequirement ids={ids} reason={t("voice_changer.denoiser_models_reason")} hideWhenReady /> : null}
        </>
    );
};

export const LevelsSection: FC = () => {
    const { t } = useTranslation();
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    return (
        <>
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
            <DenoiserSelect />
        </>
    );
};
