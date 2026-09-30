import { TTS_BACKENDS } from "@/core/services/tts";
import { TTS_Routes } from "@/core/services/tts/schema";
import { ServiceNetworkState } from "@/types";
import { Services } from "@/services-registry";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputNativeAudioOutput, InputSelect } from "../components/input";

/**
 * How the selected backend plays: through the TTS sink (sentence pipeline, device and voice-changer routing), or
 * by itself (Web Speech, SAPI, and backends not moved to the sink yet).
 */
function playsThroughSink(backend: string): boolean {
  const Backend = (TTS_BACKENDS as Record<string, { prototype: object } | undefined>)[backend];
  if (!Backend) return false;
  const proto = Backend.prototype as { synthesizePcm?: unknown; synthesizeToSink?: unknown };
  return typeof proto.synthesizePcm === "function" || typeof proto.synthesizeToSink === "function";
}

/**
 * Output device picker for TTS backends that play through the sink: the shared cpal output picker
 * ({@link InputNativeAudioOutput}: "" = system default, old SAPI names migrated, a missing saved device marked).
 */
export const TtsOutputDevice = InputNativeAudioOutput;

/**
 * Top-level TTS output routing (device / voice changer / both; `tts/schema/route.ts`, Rust `tts_sink`). The route
 * applies to every backend that plays through the TTS sink.
 */
export const Routing: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data);
  const vc = useSnapshot(window.ApiServer.voiceChanger.state);
  const tts = useSnapshot(window.ApiServer.tts.serviceState);
  const viaSink = playsThroughSink(data.backend);
  const toVc = data.route !== TTS_Routes.device;

  const setRoute = (route: string) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.route = route as TTS_Routes;
    });

  return <>
    <Inspector.SubHeader>{t("tts.routing_subheader")}</Inspector.SubHeader>
    <Inspector.Deactivatable active={tts.status === ServiceNetworkState.disconnected || viaSink}>
      <InputSelect
        label="tts.route"
        value={data.route}
        onValueChange={setRoute}
        options={[
          { label: t("tts.route_device"), value: TTS_Routes.device },
          { label: t("tts.route_voice_changer"), value: TTS_Routes.voiceChanger },
          { label: t("tts.route_both"), value: TTS_Routes.both },
        ]}
      />
    </Inspector.Deactivatable>
    {!viaSink && <Inspector.Description>
      <span className="text-base-content/60 text-xs">{t("tts.route_self_playing")}</span>
    </Inspector.Description>}
    {viaSink && toVc && <Inspector.Description>
      <span className="text-base-content/60 text-xs">
        {t("tts.route_vc_hint")}{" "}
        {!vc.isRunning && <span className="text-warning">{t("tts.route_vc_not_running")} </span>}
        <button
          type="button"
          className="link link-primary link-hover"
          onClick={() => window.ApiServer.changeTab({ tab: Services.voice_changer })}
        >
          {t("tts.route_open_voice_changer")}
        </button>
      </span>
    </Inspector.Description>}
  </>;
};
