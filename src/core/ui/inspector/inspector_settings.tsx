import { useGetState } from "@/client";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { getVersion } from '@tauri-apps/api/app';
import { confirm as dialogConfirm } from "@tauri-apps/plugin-dialog";
import { FC, memo, useEffect, useMemo, useState } from "react";
import { RiFileCopyLine, RiSettings2Fill } from "react-icons/ri";
import { toast } from "react-toastify";
import { ObsBrowserSourceService } from "@/core/services/obs/browserSource";
import {
  formatLinkAddress,
  groupLanToken,
  isValidLanToken,
  maskLanToken,
  normalizeLanToken,
  splitLinkAddress,
  type LanAccessMode,
} from "@/shared/services/peer/lanAccess";
import { SiDiscord, SiGithub, SiPatreon, SiTwitch, SiTwitter } from "react-icons/si";
import { useSnapshot } from "valtio";
import Dropdown from "../dropdown/Dropdown";
import Tooltip from "../dropdown/Tooltip";
import Logo from "../logo";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { webServerStatusLine } from "./webServerStatus";
import { LanOffHint } from "./lanAccessHints";
import { InputCheckbox, InputChips, InputNativeAudioOutput, InputNetworkStatus, InputRustAudioInput, InputSelect, InputShortcut, InputText, InputWebAudioInput } from "./components/input";
import DiagnosticsPanel from "@/core/services/diagnostics/DiagnosticsPanel";
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

/**
 * Link apps target: the other Sigil's address and its LAN token (P6-D). Both live in `linkAddress`
 * as `<ip>:<port>?token=<token>`; pasting a whole "Copy my address" value into the address field
 * fills both.
 */
const LinkAddressInputs: FC = () => {
  const { t } = useTranslation();
  const [initial] = useState(() => splitLinkAddress(window.ApiServer.state.linkAddress));
  const [address, setAddress] = useState(initial.address);
  const [token, setToken] = useState(initial.token);
  const save = (nextAddress: string, nextToken: string) => {
    window.ApiServer.state.linkAddress = formatLinkAddress(nextAddress, nextToken);
  };
  const onAddress = (value: string) => {
    if (/[?#]/.test(value)) {
      const split = splitLinkAddress(value);
      const nextToken = split.token || token;
      setAddress(split.address);
      setToken(nextToken);
      save(split.address, nextToken);
      return;
    }
    setAddress(value);
    save(value, token);
  };
  const onToken = (value: string) => {
    setToken(value);
    save(address, value);
  };
  const tokenInvalid = token.trim() !== "" && !isValidLanToken(token);
  return <>
    <InputText value={address} onChange={e => onAddress(e.target.value)} label="settings.field_ip_address" placeholder="192.168.1.50:3030" />
    <InputText value={token} onChange={e => onToken(e.target.value)} label="settings.field_link_remote_token" placeholder="abcde-fghij-…" autoComplete="off" spellCheck={false} dir="ltr" />
    {tokenInvalid && <span className="text-xs text-error">{t("settings.link_remote_token_invalid")}</span>}
    <Inspector.Description>
      <span className="text-base-content/60 text-xs">{t("settings.link_remote_token_hint")}</span>
    </Inspector.Description>
  </>;
};

const obsLinks = new ObsBrowserSourceService();

/**
 * LAN access (LAN security stage 2, P6-D; desktop host only): who may connect from other devices,
 * the persistent LAN token (show, copy, regenerate), links for other devices, and the legacy
 * no-token switch with its caution.
 */
export const LanAccessSection: FC = () => {
  const { t } = useTranslation();
  const { lanAccess } = useSnapshot(window.ApiServer.state);
  const { status } = useSnapshot(window.ApiShared.pubsub.webServer);
  const [revealed, setRevealed] = useState(false);
  const [busy, setBusy] = useState(false);
  const network = window.Config.serverNetwork;
  const addresses = network.lanAddresses ?? [];
  const [ip, setIp] = useState(network.ip);
  const token = normalizeLanToken(lanAccess.token) ?? "";
  const lan = status?.lan;
  const modeOptions = [
    { label: t("settings.lan_mode_lan"), value: "lan" },
    { label: t("settings.lan_mode_localhost"), value: "localhost" },
  ];
  const addressOptions = addresses.map((a) => ({ label: a.interface ? `${a.ip} (${a.interface})` : a.ip, value: a.ip }));

  const copy = (text: string) => {
    void navigator.clipboard.writeText(text);
    toast.success(t("toasts.copied"));
  };

  const regenerate = async () => {
    const message = t("settings.lan_token_regenerate_confirm");
    const ok = isTauri()
      ? await dialogConfirm(message, { title: t("settings.section_lan_access"), kind: "warning" })
      : window.confirm(message);
    if (!ok)
      return;
    setBusy(true);
    try {
      if (await window.ApiShared.pubsub.regenerateLanToken())
        toast.success(t("toasts.lan_token_regenerated"));
    } finally {
      setBusy(false);
    }
  };

  const switchFailed = !!lan && status?.state === "listening" && lan.boundMode !== null && lan.boundMode !== lan.mode;

  return <>
    <Inspector.SubHeader>{t("settings.section_lan_access")}</Inspector.SubHeader>
    <InputSelect
      label="settings.field_lan_mode"
      options={modeOptions}
      value={lanAccess.mode}
      onValueChange={(v) => {
        if (v === "lan" || v === "localhost")
          window.ApiServer.state.lanAccess.mode = v as LanAccessMode;
      }}
    />
    <Inspector.Description>
      <span className="text-base-content/60 text-xs">{t(lanAccess.mode === "localhost" ? "settings.lan_mode_localhost_hint" : "settings.lan_mode_lan_hint")}</span>
    </Inspector.Description>
    {lanAccess.mode === "localhost" && <>
      <Inspector.Description>
        <span className="text-base-content/60 text-xs">{t("settings.lan_mode_localhost_turn_on", { option: t("settings.lan_mode_lan") })}</span>
      </Inspector.Description>
      <Inspector.Description>
        <span className="text-base-content/60 text-xs">{t("settings.lan_mode_firewall_note")}</span>
      </Inspector.Description>
    </>}
    {lan?.boundAddress && !switchFailed && <span className="text-xs text-base-content/70" dir="ltr">{t("settings.lan_bound_address", { address: lan.boundAddress })}</span>}
    {switchFailed && <span className="text-xs text-warning">{t("settings.lan_mode_switch_failed", { address: lan?.boundAddress ?? "", error: status?.error ?? "" })}</span>}

    {lanAccess.mode === "lan" && <>
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2 min-h-8">
          <span className="font-medium text-base-content/80 text-xs">{t("settings.field_lan_token")}</span>
          <code className="font-mono text-xs break-all text-end select-all" dir="ltr">
            {token ? (revealed ? groupLanToken(token) : maskLanToken(token)) : t("settings.lan_token_missing")}
          </code>
        </div>
        <div className="flex flex-wrap justify-end gap-1">
          <button className="btn btn-xs btn-ghost" disabled={!token} onClick={() => setRevealed((v) => !v)}>
            {t(revealed ? "settings.btn_lan_token_hide" : "settings.btn_lan_token_show")}
          </button>
          <button className="btn btn-xs btn-ghost" disabled={!token} onClick={() => copy(token)}>{t("settings.btn_lan_token_copy")}</button>
          <button className="btn btn-xs btn-ghost text-warning" disabled={busy} onClick={() => void regenerate()}>{t("settings.btn_lan_token_regenerate")}</button>
        </div>
      </div>
      <Inspector.Description>
        <span className="text-base-content/60 text-xs">{t("settings.lan_token_hint")}</span>
      </Inspector.Description>

      {addressOptions.length > 1 && <InputSelect label="settings.field_lan_link_address" options={addressOptions} value={ip} onValueChange={setIp} />}
      <div className="flex flex-wrap gap-1">
        <button className="btn btn-sm btn-ghost gap-2" disabled={!token} onClick={() => copy(window.ApiShared.peer.getLanClientLink("/client", ip))}>
          <RiFileCopyLine /> {t("settings.btn_copy_lan_client_link")}
        </button>
        <button className="btn btn-sm btn-ghost gap-2" disabled={!token} onClick={() => copy(obsLinks.getCaptionsLink(undefined, { lan: true, ip }))}>
          <RiFileCopyLine /> {t("settings.btn_copy_lan_captions_link")}
        </button>
      </div>
      <Inspector.Description>
        <span className="text-base-content/60 text-xs">{t("settings.lan_links_hint")}</span>
      </Inspector.Description>

      <InputCheckbox label="settings.field_lan_legacy_no_token" value={lanAccess.legacyNoToken} onChange={(v) => window.ApiServer.state.lanAccess.legacyNoToken = v} />
      <Inspector.Description>
        <span className={lanAccess.legacyNoToken ? "text-warning text-xs" : "text-base-content/60 text-xs"}>{t("settings.lan_legacy_no_token_caution")}</span>
      </Inspector.Description>
    </>}
  </>;
};

const ExportMenu: FC = () => {
  const { t } = useTranslation();
  const [name, setName] = useState("");
  return <div className="menu bg-base-100 p-4 w-72 rounded-box flex flex-col space-y-2">
    <span className="menu-title"><span>{t("settings_export.menu_title")}</span></span>
    <InputText label="settings_export.author" value={name} onChange={e => setName(e.target.value)} />
    <button className="btn btn-sm btn-primary" onClick={() => name && window.ApiClient.document.exportDocument(name)}>{t("settings_export.btn_export")}</button>
  </div>;
}

/** Local web server status and the LAN Origin-check kill switch (desktop host only). */
const WebServerSection: FC = () => {
  const { t } = useTranslation();
  const { lanOriginCheck } = useSnapshot(window.ApiServer.state);
  const { status } = useSnapshot(window.ApiShared.pubsub.webServer);
  const statusLine = webServerStatusLine(status);
  return <>
    <Inspector.SubHeader>{t("settings.section_web_server")}</Inspector.SubHeader>
    {statusLine && <span className={statusLine.error ? "text-xs text-error" : "text-xs text-base-content/70"}>{t(statusLine.key, statusLine.values)}</span>}
    <InputCheckbox label="settings.field_lan_origin_check" value={lanOriginCheck} onChange={v => window.ApiServer.state.lanOriginCheck = v} />
    <Inspector.Description>
      <span className="text-base-content/60 text-xs">{t("settings.lan_origin_check_hint")}</span>
    </Inspector.Description>
    {status?.originCheckEnv != null && <span className="text-xs text-warning">{t("settings.lan_origin_check_env_forced")}</span>}
  </>;
};

/**
 * Link apps: connect this Sigil to another one (its address and LAN token) and copy this one's address. The desktop host in
 * "This PC only" (the default of a new install) also gets the hint that says where other devices are let in.
 */
export const LinkAppsSection: FC = () => {
  const { t } = useTranslation();
  const { state: linkStatus } = useSnapshot(window.ApiShared.pubsub.serviceState);
  return <>
    <Inspector.SubHeader>{t('settings.section_link_apps')}</Inspector.SubHeader>
    <Inspector.Description>{t('settings.section_link_apps_desc')}</Inspector.Description>
    <Inspector.Deactivatable active={linkStatus === ServiceNetworkState.disconnected}>
      <LinkAddressInputs />
      <InputNetworkStatus value={linkStatus} label="common.field_connection_status" />
    </Inspector.Deactivatable>
    <ServiceButton startLabel="common.btn_connect" stopLabel="common.btn_disconnect" status={linkStatus} onStart={() => window.ApiShared.pubsub.linkConnect()} onStop={() => window.ApiShared.pubsub.linkDisconnect()} />
    <button className="btn btn-sm btn-ghost" onClick={() => window.ApiShared.pubsub.copyLinkAddress()}>{t('settings.btn_copy_address')}</button>
    {isTauri() && <LanOffHint />}
  </>;
};

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
  const { clientTheme, uiScale, uiLanguage, backgroundInputTimer, audioInputDevice, audioInputDeviceNative, audioOutputDevice } = useSnapshot(window.ApiServer.state);
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
      {isTauri() ? (
        <InputRustAudioInput
          label="settings.field_default_input_device"
          value={audioInputDeviceNative}
          onChange={(v) => {
            const s = window.ApiServer.state;
            s.audioInputDeviceNative = v;
            // A deliberate pick (including "system default") ends the one-time legacy device migration.
            s.audioInputDeviceNativeMigrated = true;
          }}
        />
      ) : (
        <InputWebAudioInput
          label="settings.field_default_input_device"
          value={audioInputDevice}
          onChange={e => window.ApiServer.state.audioInputDevice = e}
        />
      )}
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
      <DiagnosticsPanel />

      {isTauri() && <>
        <Inspector.SubHeader>{t("settings.section_shortcuts")}</Inspector.SubHeader>
        <InputShortcut label="settings.field_shortcut_mute_sound" shortcut="muteSound" />
        <Inspector.Description>
          <span className="text-base-content/60 text-xs">{t("settings.shortcuts_hint")}</span>
        </Inspector.Description>
      </>}


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


      <LinkAppsSection />

      {isTauri() && <LanAccessSection />}
      {isTauri() && <WebServerSection />}

    </Inspector.Content>
  </Inspector.Body>
})
export default Inspector_Settings;
