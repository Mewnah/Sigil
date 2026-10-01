// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// WebSocket (sidecar) and ONNX chunk mode settings.
import type { VoiceChangerState } from "@/core/services/voice_changer/schema";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputSelect, InputText } from "../components/input";
import { InspectorInfoButton, patchVcStructural, VcPathField } from "./common";

/** WebSocket converter URL. */
export const SidecarSection: FC = () => {
    const { t } = useTranslation();
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    return (
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
    );
};

/** ONNX chunk model and its execution provider. */
export const OnnxSection: FC = () => {
    const { t } = useTranslation();
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    return (
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
    );
};
