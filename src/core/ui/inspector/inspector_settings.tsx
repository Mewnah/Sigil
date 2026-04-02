import { useGetState } from "@/client";
import { ServiceNetworkState } from "@/types";
import { getVersion } from '@tauri-apps/api/app';
import { FC, memo, useEffect, useMemo, useState } from "react";
import { RiFileCopyLine, RiSettings2Fill } from "react-icons/ri";
import { SiDiscord, SiGithub, SiPatreon, SiTwitch, SiTwitter } from "react-icons/si";
import { useSnapshot } from "valtio";
import Dropdown from "../dropdown/Dropdown";
import Tooltip from "../dropdown/Tooltip";
import Logo from "../logo";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputChips, InputNativeAudioOutput, InputNetworkStatus, InputSelect, InputShortcut, InputText, InputWebAudioInput } from "./components/input";
import { useTranslation } from "react-i18next";
import { i18nLanguages, loadLanguageFile } from "@/i18n";
// 1. Core & Tech
const themesCore = [
  'sigil-dark',
  'sigil-light',
  'neon-city',
  'matrix',
  'synthwave',
  'terminal',
  'steampunk',
]

// 2. Nature & Elements
const themesNature = [
  'overgrowth',
  'abyss',
  'ember',
  'zephyr',
  'terracotta',
]

// 3. Aesthetic & Vibes
const themesVibes = [
  'vaporwave',
  'lofi',
  'dracula',
  'nord',
  'coffee',
]

// 4. Luxury & Precious
const themesLuxury = [
  'midas',
  'rose-gold',
  'royal',
]

// 5. Space & Sci-Fi
const themesSpace = [
  'starlight',
  'mars',
  'nebula',
  'monochrome',
]

const UI_SCALE_MIN = 0.8;
const UI_SCALE_MAX = 1.5;

const languageOptions = i18nLanguages.map(({ code, name }) => ({ label: name, value: code }));

const AddrInput = () => {
  const [v, setV] = useState(window.ApiServer.state.linkAddress);
  const upd = (v: string) => {
    setV(v);
    window.ApiServer.state.linkAddress = v;
  }
  return <InputText value={v} onChange={e => upd(e.target.value)} label="settings.field_ip_address" placeholder="192.168..smth" />
}

const ExportMenu: FC = () => {
  const { t } = useTranslation();
  const [name, setName] = useState("");
  return <div className="menu bg-base-100 p-4 w-72 rounded-box flex flex-col space-y-2">
    <span className="menu-title"><span>{t("settings_export.menu_title")}</span></span>
    <InputText label="settings_export.author" value={name} onChange={e => setName(e.target.value)} />
    <button className="btn btn-sm btn-primary" onClick={() => name && window.ApiClient.document.exportDocument(name)}>{t("settings_export.btn_export")}</button>
  </div>;
}

const Inspector_Settings: FC = memo(() => {
  const { t } = useTranslation();
  const themeSelectOptions = useMemo(
    () => [
      { label: t("settings.theme_group_core_tech"), options: themesCore.map((theme) => ({ value: theme, label: theme })) },
      { label: t("settings.theme_group_nature"), options: themesNature.map((theme) => ({ value: theme, label: theme })) },
      { label: t("settings.theme_group_vibes"), options: themesVibes.map((theme) => ({ value: theme, label: theme })) },
      { label: t("settings.theme_group_luxury"), options: themesLuxury.map((theme) => ({ value: theme, label: theme })) },
      { label: t("settings.theme_group_space"), options: themesSpace.map((theme) => ({ value: theme, label: theme })) },
    ],
    [t],
  );
  const { clientTheme, uiScale, uiLanguage, backgroundInputTimer, audioInputDevice, audioOutputDevice } = useSnapshot(window.ApiServer.state);
  const { state: linkStatus } = useSnapshot(window.ApiShared.pubsub.serviceState);
  const author = useGetState(state => state.author);

  const [version, setVersion] = useState("")
  useEffect(() => {
    getVersion().then(setVersion);
  }, [])

  const handleChangeTheme = (v: string) => window.ApiServer.changeTheme(v);
  const handleChangeLanguage = (v: string) => window.ApiServer.changeLanguage(v);

  const handleChangeScale = (v: string | number) => {
    const _v = typeof v === "string" ? parseFloat(v) : v;
    window.ApiServer.changeScale(Math.max(UI_SCALE_MIN, Math.min(UI_SCALE_MAX, _v)));
  }

  return <Inspector.Body>
    <Inspector.Header><RiSettings2Fill /> {t('settings.title')}</Inspector.Header>
    <Inspector.Content>
      <div className="flex flex-col items-center space-y-1">
        <span className="text-4xl leading-none font-header font-black"><Logo /></span>
        <div className="self-center text-xs italic opacity-60 mt-1">{t("settings.hero_tagline")}</div>
        <div className="self-center text-sm opacity-70 mt-2">
          {t("settings.credits_line_before")}
          <span className="text-primary font-semibold">{t("settings.credits_author_name")}</span>
        </div>
        <p className="self-center text-xs opacity-55 text-center max-w-[22rem] leading-relaxed px-2">
          {t("settings.credits_mmpneo")}
        </p>
        <div className="self-center text-xs opacity-40 mt-1">{version ? `v${version}` : ""}</div>
        <div className="flex space-x-1 self-center mt-3">
          <Tooltip
            content={t("settings.discord_tooltip_title")}
            body={
              <span>
                {t("settings.discord_tooltip_body")
                  .split("\n")
                  .map((line, i) => (
                    <span key={i}>
                      {i > 0 ? <br /> : null}
                      {line}
                    </span>
                  ))}
              </span>
            }
          >
            <a target="_blank" rel="noopener noreferrer" href="https://discord.gg/Sw6pw8fGYS" aria-label="Discord" className="btn text-primary btn-ghost btn-circle text-2xl hover:scale-110 transition-transform"><SiDiscord /></a>
          </Tooltip>
          <Tooltip content={t("settings.tooltip_curses_github_title")} body={<span>{t("settings.tooltip_curses_github_body")}</span>}>
            <a target="_blank" rel="noopener noreferrer" href="https://github.com/mmpneo/curses" aria-label="GitHub" className="btn text-primary btn-ghost btn-circle text-2xl hover:scale-110 transition-transform"><SiGithub /></a>
          </Tooltip>
        </div>
        <div className="self-center text-[10px] opacity-30 mt-2">{t("settings.footer_vrchat")}</div>
      </div>
      <div className="divider"></div>
      <Inspector.SubHeader>{t("settings.section_application")}</Inspector.SubHeader>
      <InputSelect label="settings.field_app_theme" options={themeSelectOptions} value={clientTheme} onValueChange={handleChangeTheme} />
      <InputChips label="settings.field_ui_scale" value={uiScale} onChange={e => handleChangeScale(e)} options={[
        { label: t("settings.ui_scale_s"), value: .8 },
        { label: t("settings.ui_scale_m"), value: 1 },
        { label: t("settings.ui_scale_l"), value: 1.2 },
        { label: t("settings.ui_scale_x"), value: 1.4 },
      ]} />
      <InputSelect label="settings.field_language" options={languageOptions} value={uiLanguage} onValueChange={handleChangeLanguage} />
      <Inspector.Description><span className="text-primary font-semibold mt-2" onClick={loadLanguageFile}>{t('settings.btn_import_translation')}</span></Inspector.Description>

      <Inspector.SubHeader>{t("settings.section_audio")}</Inspector.SubHeader>
      <InputWebAudioInput
        label="settings.field_default_input_device"
        value={audioInputDevice}
        onChange={e => window.ApiServer.state.audioInputDevice = e}
      />
      <InputNativeAudioOutput
        label="settings.field_default_output_device"
        value={audioOutputDevice}
        onChange={e => window.ApiServer.state.audioOutputDevice = e}
      />
      <Inspector.Description>
        <span className="text-base-content/60 text-xs">
          {t("settings.audio_devices_hint")}
        </span>
      </Inspector.Description>


      <Inspector.SubHeader>{t('settings.section_template')}</Inspector.SubHeader>
      {author && <span className="text-sm text-secondary font-semibold">{t("settings.created_by", { author })}</span>}
      <div className="flex items-center space-x-2">
        <button onClick={() => window.ApiClient.document.importDocument()} className="flex-grow btn btn-sm gap-2"><RiFileCopyLine /> {t('settings.btn_import_template')}</button>
        <Dropdown className="flex-grow btn btn-sm gap-2" targetOffset={24} placement="right" content={<ExportMenu />}>
          <RiFileCopyLine /> {t('settings.btn_export_template')}
        </Dropdown>
      </div>

      {window.Config.features.background_input && <>
        <Inspector.SubHeader>{t("settings.section_background_input")}</Inspector.SubHeader>
        <InputShortcut label="settings.field_shortcut" shortcut="bgInput" />
        <InputText label="settings.field_bg_timer" value={backgroundInputTimer} onChange={e => window.ApiServer.state.backgroundInputTimer = e.target.value} type="number" />
        <div className="text-xs opacity-70">{t("settings.background_input_hint")}</div>
      </>}


      <Inspector.SubHeader>{t('settings.section_link_apps')}</Inspector.SubHeader>
      <Inspector.Description>{t('settings.section_link_apps_desc')}</Inspector.Description>
      <Inspector.Deactivatable active={linkStatus === ServiceNetworkState.disconnected}>
        <AddrInput />
        <InputNetworkStatus value={linkStatus} label="common.field_connection_status" />
      </Inspector.Deactivatable>
      <ServiceButton startLabel="common.btn_connect" stopLabel="common.btn_disconnect" status={linkStatus} onStart={() => window.ApiShared.pubsub.linkConnect()} onStop={() => window.ApiShared.pubsub.linkDisconnect()} />
      <button className="btn btn-sm btn-ghost" onClick={() => window.ApiShared.pubsub.copyLinkAddress()}>{t('settings.btn_copy_address')}</button>

    </Inspector.Content>
  </Inspector.Body>
})
export default Inspector_Settings;
