import { Translation_Backends, Translation_State } from "@/core/services/translation/schema";
import { ServiceNetworkState } from "@/types";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { RiTranslate2 } from "react-icons/ri";
import { useSnapshot } from "valtio";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputSelect } from "./components/input";
import { TranslationLiveSettings } from "./translation/LiveSettings";
import { TRANSLATION_PANELS, TRANSLATION_PROVIDER_ORDER } from "./translation/registry";
import { TranslationStatusLine } from "./translation/StatusLine";

// Translation inspector shell (P4-S0 split): the per-backend panels, provider order, live settings and status line
// live in ./translation/ (P4-F; LlmServer.tsx is P4-B's). Settings inside Deactivatable need a restart; the live
// settings below the service button apply to the next line.

const Inspector_Translation: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation);
  const state = useSnapshot(window.ApiServer.translation.serviceState);

  const handleStart = (v: boolean) => window.ApiServer.state.services.translation.showActionButton = v;
  const up = <K extends keyof Translation_State>(key: K, v: Translation_State[K]) => window.ApiServer.patchService("translation", s => s.data[key] = v);

  const Panel = TRANSLATION_PANELS[data.data.backend]?.Panel;

  return <Inspector.Body>
    <Inspector.Header><RiTranslate2 /> {t('transl.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>{t("transl.provider_subheader")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect
          options={TRANSLATION_PROVIDER_ORDER.map((b) => ({ label: t(TRANSLATION_PANELS[b].label), value: b }))}
          label="common.field_service"
          value={data.data.backend}
          onValueChange={e => up("backend", e as Translation_Backends)}
        />

        {Panel && <Panel />}
      </Inspector.Deactivatable>

      <div className="pt-4 flex flex-col gap-2">
        <ServiceButton status={state.status} onStart={() => window.ApiServer.translation.start()} onStop={() => window.ApiServer.translation.stop()} />
        <TranslationStatusLine />
      </div>

      <TranslationLiveSettings />

      <div className="contents">
        <div className="h-2" />
        <InputCheckbox label="common.field_action_bar" onChange={handleStart} value={data.showActionButton} />
        <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
      </div>
    </Inspector.Content>
  </Inspector.Body>
};

export default Inspector_Translation;
