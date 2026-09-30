import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { neuralOnnxSettings } from "@/core/services/tts/services/neuralOnnx";
import {
  LocalTtsModelInfo,
  hubCatalogId,
  localTtsState,
  normalizeHubRepo,
  voicesFor,
} from "@/core/services/tts/services/localNeural";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FC, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { RiFolderOpenLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { InputRange, InputSelect, InputText } from "../components/input";
import { useLocalModels } from "./LocalNeural";
import { TtsOutputDevice } from "./Routing";

type NeuralOnnxState = TTS_State["neuralOnnx"];

/**
 * Panel of the legacy `neuralOnnx` backend ("ONNX (local)"): Kitten from a Hugging Face repo, or a local speech
 * package folder. Since P2 it speaks through the in-process `local-tts` engine and the TTS sink (so the output device
 * and the top-level route apply); KittenML 0.8 repos download as Sigil's verified catalog packages. The persisted keys
 * are unchanged; the execution-provider hint and the per-backend voice-changer flag are no longer shown (the local
 * engines run on the processor; the route is the Routing section below). Owned by P2-H.
 */
export const NeuralOnnx: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.neuralOnnx);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const loaded = useSnapshot(localTtsState).loaded as LocalTtsModelInfo | null;
  const tauri = isTauri();
  const { models } = useLocalModels();
  const up = <K extends keyof NeuralOnnxState>(key: K, v: NeuralOnnxState[K]) =>
    window.ApiServer.patchService("tts", (s) => (s.data.neuralOnnx[key] = v));

  const hub = data.modelSource !== "local";
  const repoValid = !hub || normalizeHubRepo(data.hubId) !== null;
  const catalogId = hub ? hubCatalogId(data.hubId) : null;

  // Voices: the mapped catalog package's (KittenML's own voice names), or what the running backend loaded.
  const voices = useMemo(() => {
    if (catalogId) return voicesFor(catalogId, models, null);
    const settings = neuralOnnxSettings(window.ApiServer.state.services.tts.data.neuralOnnx);
    return typeof settings === "string" ? null : voicesFor(settings.model, [], loaded);
  }, [catalogId, models, loaded, data.hubId, data.localModelDir, data.modelSource]);

  const voiceOptions = useMemo(() => {
    if (!voices) return null;
    const options = [
      { label: t("tts.local_neural_voice_default", { name: voices.defaultVoice || "—" }), value: "" },
      ...voices.voices.map((v) => ({ label: v.name, value: v.id })),
    ];
    const current = data.voice.trim();
    if (current && !voices.voices.some((v) => v.id === current)) options.push({ label: current, value: current });
    return options;
  }, [voices, data.voice, t]);

  const browse = useCallback(async () => {
    try {
      const path = await open({ directory: true, multiple: false });
      if (typeof path === "string" && path) up("localModelDir", path);
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }, []);

  return (
    <>
      <Inspector.SubHeader>{t("tts.neural_onnx_title")}</Inspector.SubHeader>
      <div className="text-xs text-base-content/70">{t("tts.neural_onnx_desc_local_engine")}</div>
      <div className="text-xs text-base-content/60">
        {t("tts.neural_onnx_prefer_local_neural")}{" "}
        <button
          type="button"
          className="link link-accent"
          onClick={() => window.ApiServer.patchService("tts", (s) => (s.data.backend = TTS_Backends.localNeural))}
          disabled={state.status !== ServiceNetworkState.disconnected}
        >
          {t("tts.neural_onnx_switch_to_local_neural")}
        </button>
      </div>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect
          label="tts.neural_onnx_model_source"
          value={data.modelSource}
          onValueChange={(v) => up("modelSource", v as NeuralOnnxState["modelSource"])}
          options={[
            { label: t("tts.neural_onnx_source_hub"), value: "hub" },
            { label: t("tts.neural_onnx_source_local"), value: "local" },
          ]}
        />
        {hub && (
          <>
            <InputText label="tts.neural_onnx_hub" value={data.hubId} onChange={(e) => up("hubId", e.target.value)} />
            {!repoValid && <div className="text-xs text-error">{t("tts.neural_onnx_hub_invalid", { repo: data.hubId.trim() })}</div>}
            {catalogId ? (
              <ModelRequirement ids={[catalogId]} hideWhenReady reason={t("tts.neural_onnx_hub_catalog_reason")} />
            ) : (
              repoValid && <div className="text-xs text-base-content/60">{t("tts.neural_onnx_hub_download_note")}</div>
            )}
          </>
        )}
        {!hub && (
          <div className="flex flex-col gap-1">
            <label className="font-medium text-base-content/80 text-xs" htmlFor="neural-onnx-folder">
              {t("tts.neural_onnx_local_dir")}
            </label>
            <div className="flex gap-2 items-center">
              <input
                id="neural-onnx-folder"
                type="text"
                className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
                value={data.localModelDir}
                onChange={(e) => up("localModelDir", e.target.value)}
              />
              {tauri ? (
                <button type="button" className="btn btn-sm btn-outline shrink-0 gap-1" onClick={browse}>
                  <RiFolderOpenLine /> {t("tts.neural_onnx_browse_folder")}
                </button>
              ) : (
                <span className="text-xs text-base-content/50 shrink-0 max-w-[10rem]">{t("tts.neural_onnx_browse_unavailable")}</span>
              )}
            </div>
            <div className="text-xs text-base-content/60">{t("tts.neural_onnx_local_hint_v2")}</div>
          </div>
        )}
      </Inspector.Deactivatable>
      {voiceOptions ? (
        <InputSelect label="tts.neural_onnx_voice" value={data.voice.trim()} options={voiceOptions} onValueChange={(v) => up("voice", v)} />
      ) : (
        <InputText label="tts.neural_onnx_voice" value={data.voice} onChange={(e) => up("voice", e.target.value)} />
      )}
      <InputRange value={data.speed} onChange={(e) => up("speed", e.target.value)} label="tts.neural_onnx_speed" step="0.05" min="0.5" max="2" />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
    </>
  );
};
