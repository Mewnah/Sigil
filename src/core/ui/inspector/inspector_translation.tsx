import { Translation_Backends, Translation_State } from "@/core/services/translation/schema";
import { Translation_LibreTranslateService } from "@/core/services/translation/services/libretranslate";
import { ServiceNetworkState } from "@/types";
import { FC, memo, useEffect, useState } from "react";
import { RiTranslate2 } from "react-icons/ri";
import { useSnapshot } from "valtio";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputSelect, InputSelectOption, InputText } from "./components/input";
import { useTranslation } from "react-i18next";

const Azure: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.azure);
  const { azure: azureLanguages } = useSnapshot(window.ApiServer.translation.lib);
  const up = <K extends keyof Translation_State["azure"]>(key: K, v: Translation_State["azure"][K]) => window.ApiServer.state.services.translation.data.azure[key] = v;
  // Stored codes stay selectable while the language list is loading or offline. A blank source
  // language makes Azure auto-detect it (azureTranslateUrl leaves `from` out).
  const languages = azureLanguages as InputSelectOption[];
  const fromOpts: InputSelectOption[] = [
    { label: t("transl.azure_from_auto"), value: "" },
    ...mergeOption(languages, pr.languageFrom.trim()),
  ];
  const toOpts = mergeOption(languages, pr.language);

  return <>
    <Inspector.SubHeader>{t('transl.azure_title')}</Inspector.SubHeader>
    <InputText label="transl.azure_key" type="password" value={pr.key} onChange={e => up("key", e.target.value)} />
    <InputText label="transl.azure_location_optional" value={pr.location} placeholder="westeurope" onChange={e => up("location", e.target.value)} />
    <div className="text-xs text-base-content/70 mb-2">{t("transl.azure_location_hint")}</div>
    <InputSelect
      label="transl.azure_from_language"
      options={fromOpts}
      value={pr.languageFrom.trim()}
      onValueChange={e => up("languageFrom", e)}
    />
    <InputSelect
      label="transl.azure_to_language"
      options={toOpts}
      value={pr.language}
      onValueChange={e => up("language", e)}
    />
    <InputSelect
      label="transl.azure_profanity"
      options={[
        { label: t('transl.azure_profanity_masked'), value: 'Marked' },
        { label: t('transl.azure_profanity_removed'), value: 'Deleted' },
        { label: t('transl.azure_profanity_raw'), value: 'NoAction' },
      ]}
      value={pr.profanity}
      onValueChange={e => up("profanity", e as any)}
    />
    <InputCheckbox label="common.field_interim_results" onChange={e => up("interim", e)} value={pr.interim} />
  </>
})

const LANGUAGE_PROBE_DEBOUNCE_MS = 400;

function mergeOption(opts: InputSelectOption[], current: string): InputSelectOption[] {
  if (!current || opts.some((o) => o.value === current)) return opts;
  return [{ label: current, value: current }, ...opts];
}

const LibreTranslatePanel: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.libretranslate);
  const [langOpts, setLangOpts] = useState<InputSelectOption[]>([]);
  const up = <K extends keyof Translation_State["libretranslate"]>(key: K, v: Translation_State["libretranslate"][K]) =>
    window.ApiServer.state.services.translation.data.libretranslate[key] = v;

  useEffect(() => {
    let cancel = false;
    // Debounced: the effect re-runs on every keystroke in the URL field. The endpoint is passed per
    // call, so probing here never changes what a running translation talks to.
    const timer = setTimeout(() => {
      void Translation_LibreTranslateService.getLanguages(pr.endpoint, pr.apiKey ?? "").then((list) => {
        if (cancel) return;
        setLangOpts(
          list.map((l) => ({
            label: `${l.name} (${l.code})`,
            value: l.code,
          }))
        );
      });
    }, LANGUAGE_PROBE_DEBOUNCE_MS);
    return () => {
      cancel = true;
      clearTimeout(timer);
    };
  }, [pr.endpoint, pr.apiKey]);

  const fromOpts = mergeOption(langOpts, pr.languageFrom);
  const toOpts = mergeOption(langOpts, pr.language);

  return <>
    <Inspector.SubHeader>{t("transl.libretranslate_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("transl.libretranslate_desc")}</div>
    <InputText
      label="transl.libretranslate_endpoint"
      value={pr.endpoint}
      onChange={(e) => up("endpoint", e.target.value)}
    />
    <InputText
      label="transl.libretranslate_api_key"
      type="password"
      value={pr.apiKey}
      onChange={(e) => up("apiKey", e.target.value)}
    />
    {fromOpts.length > 0 ? (
      <InputSelect
        label="transl.libretranslate_from_language"
        options={fromOpts}
        value={pr.languageFrom}
        onValueChange={(e) => up("languageFrom", e as string)}
      />
    ) : (
      <InputText
        label="transl.libretranslate_from_language"
        value={pr.languageFrom}
        onChange={(e) => up("languageFrom", e.target.value)}
      />
    )}
    {toOpts.length > 0 ? (
      <InputSelect
        label="transl.libretranslate_to_language"
        options={toOpts}
        value={pr.language}
        onValueChange={(e) => up("language", e as string)}
      />
    ) : (
      <InputText
        label="transl.libretranslate_to_language"
        value={pr.language}
        onChange={(e) => up("language", e.target.value)}
      />
    )}
    <InputCheckbox
      label="transl.libretranslate_auto_detect"
      value={pr.autoDetect}
      onChange={(v) => up("autoDetect", v)}
    />
    <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
  </>;
});

const OnDevice: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.onDevice);
  const up = <K extends keyof Translation_State["onDevice"]>(key: K, v: Translation_State["onDevice"][K]) =>
    window.ApiServer.state.services.translation.data.onDevice[key] = v;

  return <>
    <Inspector.SubHeader>{t("transl.ondevice_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("transl.ondevice_desc")}</div>
    <InputText
      label="transl.ondevice_from"
      value={pr.languageFrom}
      onChange={(e) => up("languageFrom", e.target.value)}
    />
    <InputText
      label="transl.ondevice_to"
      value={pr.language}
      onChange={(e) => up("language", e.target.value)}
    />
    <InputCheckbox
      label="transl.ondevice_auto_detect"
      value={pr.autoDetect}
      onChange={(v) => up("autoDetect", v)}
    />
    <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
  </>;
});

const Inspector_Translation: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation);
  const state = useSnapshot(window.ApiServer.translation.serviceState);

  const handleStart = (v: boolean) => window.ApiServer.state.services.translation.showActionButton = v;
  const up = <K extends keyof Translation_State>(key: K, v: Translation_State[K]) => window.ApiServer.patchService("translation", s => s.data[key] = v);

  return <Inspector.Body>
    <Inspector.Header><RiTranslate2 /> {t('transl.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>{t("transl.provider_subheader")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect options={[
          { label: t("transl.backend_azure"), value: Translation_Backends.azure },
          { label: t("transl.backend_libretranslate"), value: Translation_Backends.libretranslate },
          { label: t("transl.backend_ondevice"), value: Translation_Backends.onDevice },
        ]} label="common.field_service" value={data.data.backend} onValueChange={e => up("backend", e as Translation_Backends)} />

        {data.data.backend === Translation_Backends.azure && <Azure />}
        {data.data.backend === Translation_Backends.libretranslate && <LibreTranslatePanel />}
        {data.data.backend === Translation_Backends.onDevice && <OnDevice />}
      </Inspector.Deactivatable>

      <ServiceButton status={state.status} onStart={() => window.ApiServer.translation.start()} onStop={() => window.ApiServer.translation.stop()} />

      <div className="contents">
        <div className="h-2" />
        <InputCheckbox label="common.field_action_bar" onChange={handleStart} value={data.showActionButton} />
        <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
      </div>
    </Inspector.Content>
  </Inspector.Body>
};

export default Inspector_Translation;
