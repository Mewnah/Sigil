import type { STT_State } from "@/core/services/stt/schema";
import { VOSK_MODELS_FALLBACK, type VoskCatalogModel } from "@/core/services/stt/services/vosk";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputRustAudioInput, InputSelect, InputText } from "../components/input";

export const Vosk: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.vosk);
  const up = <K extends keyof STT_State["vosk"]>(key: K, v: STT_State["vosk"][K]) => window.ApiServer.state.services.stt.data.vosk[key] = v;

  const [voskCatalog, setVoskCatalog] = useState<VoskCatalogModel[]>(VOSK_MODELS_FALLBACK);

  useEffect(() => {
    invoke<VoskCatalogModel[]>("plugin:vosk-stt|get_vosk_models")
      .then(setVoskCatalog)
      .catch(() => { /* keep fallback */ });
  }, []);

  const modelOptions = voskCatalog.map(m => ({
    label: `${m.name} (${m.size})`,
    value: m.id
  }));

  return <>
    <Inspector.SubHeader>{t("stt.vosk_title")}</Inspector.SubHeader>
    <Inspector.Description>{t("stt.vosk_desc")}</Inspector.Description>

    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />

    <InputSelect
      label="stt.vosk_model"
      options={modelOptions}
      value={pr.model}
      onValueChange={e => up("model", e)}
    />
    <Inspector.Description>{t("stt.vosk_model_desc")}</Inspector.Description>

    <InputText
      label="stt.vosk_model_url"
      placeholder="https://example.com/model.zip"
      value={pr.modelUrl}
      onChange={e => up("modelUrl", e.target.value)}
    />
    <Inspector.Description>{t("stt.vosk_model_url_desc")}</Inspector.Description>

    <InputCheckbox
      label={t("stt.vosk_use_local")}
      onChange={(e) => up("useLocalZip", e)}
      value={pr.useLocalZip}
    />
    {pr.useLocalZip && (
      <div className="flex flex-col gap-2 mb-2">
        <div className="flex gap-2 items-center">
          <input
            type="text"
            className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
            value={pr.localZipPath}
            onChange={(e) => up("localZipPath", e.target.value)}
            placeholder={t("stt.vosk_local_path")}
            aria-label={t("stt.vosk_local_path")}
          />
          {isTauri() && (
            <button
              type="button"
              className="btn btn-sm btn-outline shrink-0"
              onClick={async () => {
                const path = await open({
                  filters: [{ name: "Vosk model", extensions: ["zip"] }],
                  multiple: false,
                });
                if (path === null || Array.isArray(path)) return;
                up("localZipPath", path);
              }}
            >
              {t("stt.vosk_browse_zip")}
            </button>
          )}
        </div>
        {isTauri() && (
          <button
            type="button"
            className="btn btn-sm btn-neutral w-fit"
            onClick={async () => {
              const path = await open({
                filters: [{ name: "Vosk model", extensions: ["zip"] }],
                multiple: false,
              });
              if (path === null || Array.isArray(path)) return;
              try {
                await invoke("plugin:vosk-stt|import_vosk_model_zip", {
                  modelId: window.ApiServer.state.services.stt.data.vosk.model,
                  sourcePath: path as string,
                });
                toast.success(t("stt.vosk_import_ok"));
              } catch (e) {
                toast.error(String(e));
              }
            }}
          >
            {t("stt.vosk_import_cached")}
          </button>
        )}
        <Inspector.Description>{t("stt.vosk_local_hint")}</Inspector.Description>
      </div>
    )}
  </>
}
