import { lazy, type ComponentType, type FC, type LazyExoticComponent } from "react";
import {
  RiChatVoiceFill,
  RiDiscordFill,
  RiGamepadFill,
  RiMagicFill,
  RiMessage2Fill,
  RiMicFill,
  RiRecordCircleFill,
  RiSparklingFill,
  RiTranslate2,
  RiTwitchFill,
  RiUserVoiceFill,
  RiVolumeUpFill,
} from "react-icons/ri";
import { SiDiscord, SiObsstudio, SiTwitch } from "react-icons/si";
import { Services } from "@/services-registry";
import { KickIcon } from "./icons/KickIcon";

const Inspector_STT = lazy(() => import("./inspector/inspector_stt"));
const Inspector_Translation = lazy(() => import("./inspector/inspector_translation"));
const Inspector_TTS = lazy(() => import("./inspector/inspector_tts"));
const Inspector_VRC = lazy(() => import("./inspector/inspector_vrc"));
const Inspector_Transform = lazy(() => import("./inspector/inspector_transform"));
const Inspector_Twitch = lazy(() => import("./inspector/inspector_twitch"));
const Inspector_Discord = lazy(() => import("./inspector/inspector_discord"));
const Inspector_Kick = lazy(() => import("./inspector/inspector_kick"));
const Inspector_OBS = lazy(() => import("./inspector/inspector_obs"));
const Inspector_VoiceChanger = lazy(() =>
  import("./inspector/inspector_voice_changer").then((m) => ({ default: m.Inspector_VoiceChanger }))
);

export type ServiceNavGroup = "live" | "integrations";

/** Compact sidebar: first row under "Services", second under "Integrations". */
export type SidebarSection = "services" | "integrations";

export type HostServiceIcon = ComponentType<{ className?: string }>;

export interface HostServicePanelDefinition {
  tab: Services;
  Inspector: LazyExoticComponent<FC>;
  /** Header title and nav label (English; nav does not use i18n today). */
  title: string;
  navGroup: ServiceNavGroup;
  navOrder: number;
  NavIcon: HostServiceIcon;
  /** Compact sidebar icon (defaults to NavIcon if omitted at use site). */
  SidebarIcon: HostServiceIcon;
  /** When set, sidebar uses `t(key)` for the tooltip. */
  sidebarTooltipI18nKey?: string;
  sidebarTooltipFallback: string;
  sidebarOrder: number;
  sidebarSection: SidebarSection;
}

const ALL_PANELS: HostServicePanelDefinition[] = [
  {
    tab: Services.stt,
    Inspector: Inspector_STT,
    title: "Speech to Text",
    navGroup: "live",
    navOrder: 0,
    NavIcon: RiMicFill,
    SidebarIcon: RiUserVoiceFill,
    sidebarTooltipI18nKey: "stt.title",
    sidebarTooltipFallback: "Speech to Text",
    sidebarOrder: 0,
    sidebarSection: "services",
  },
  {
    tab: Services.tts,
    Inspector: Inspector_TTS,
    title: "Text to Speech",
    navGroup: "live",
    navOrder: 1,
    NavIcon: RiVolumeUpFill,
    SidebarIcon: RiChatVoiceFill,
    sidebarTooltipI18nKey: "tts.title",
    sidebarTooltipFallback: "Text to Speech",
    sidebarOrder: 2,
    sidebarSection: "services",
  },
  {
    tab: Services.translation,
    Inspector: Inspector_Translation,
    title: "Translation",
    navGroup: "live",
    navOrder: 2,
    NavIcon: RiTranslate2,
    SidebarIcon: RiTranslate2,
    sidebarTooltipI18nKey: "transl.title",
    sidebarTooltipFallback: "Translation",
    sidebarOrder: 3,
    sidebarSection: "services",
  },
  {
    tab: Services.transform,
    Inspector: Inspector_Transform,
    title: "AI Transform",
    navGroup: "live",
    navOrder: 3,
    NavIcon: RiSparklingFill,
    SidebarIcon: RiSparklingFill,
    sidebarTooltipFallback: "AI Transform",
    sidebarOrder: 1,
    sidebarSection: "services",
  },
  {
    tab: Services.voice_changer,
    Inspector: Inspector_VoiceChanger,
    title: "Voice Changer",
    navGroup: "live",
    navOrder: 4,
    NavIcon: RiMagicFill,
    SidebarIcon: RiMagicFill,
    sidebarTooltipFallback: "Voice Changer",
    sidebarOrder: 9,
    sidebarSection: "integrations",
  },
  {
    tab: Services.twitch,
    Inspector: Inspector_Twitch,
    title: "Twitch",
    navGroup: "integrations",
    navOrder: 0,
    NavIcon: RiTwitchFill,
    SidebarIcon: SiTwitch,
    sidebarTooltipI18nKey: "twitch.title",
    sidebarTooltipFallback: "Twitch",
    sidebarOrder: 5,
    sidebarSection: "integrations",
  },
  {
    tab: Services.kick,
    Inspector: Inspector_Kick,
    title: "Kick",
    navGroup: "integrations",
    navOrder: 1,
    NavIcon: KickIcon,
    SidebarIcon: KickIcon,
    sidebarTooltipFallback: "Kick",
    sidebarOrder: 6,
    sidebarSection: "integrations",
  },
  {
    tab: Services.discord,
    Inspector: Inspector_Discord,
    title: "Discord",
    navGroup: "integrations",
    navOrder: 2,
    NavIcon: RiDiscordFill,
    SidebarIcon: SiDiscord,
    sidebarTooltipI18nKey: "discord.title",
    sidebarTooltipFallback: "Discord",
    sidebarOrder: 7,
    sidebarSection: "integrations",
  },
  {
    tab: Services.obs,
    Inspector: Inspector_OBS,
    title: "OBS Studio",
    navGroup: "integrations",
    navOrder: 3,
    NavIcon: RiRecordCircleFill,
    SidebarIcon: SiObsstudio,
    sidebarTooltipI18nKey: "obs.title",
    sidebarTooltipFallback: "OBS Studio",
    sidebarOrder: 4,
    sidebarSection: "integrations",
  },
  {
    tab: Services.vrc,
    Inspector: Inspector_VRC,
    title: "VRChat",
    navGroup: "integrations",
    navOrder: 4,
    NavIcon: RiGamepadFill,
    SidebarIcon: RiMessage2Fill,
    sidebarTooltipI18nKey: "vrc.title",
    sidebarTooltipFallback: "VRChat",
    sidebarOrder: 8,
    sidebarSection: "integrations",
  },
];

function sortByNavOrder(panels: HostServicePanelDefinition[]) {
  return [...panels].sort((a, b) => a.navOrder - b.navOrder);
}

export const HOST_SERVICE_PANELS_LIVE = sortByNavOrder(
  ALL_PANELS.filter((p) => p.navGroup === "live")
);
export const HOST_SERVICE_PANELS_INTEGRATIONS = sortByNavOrder(
  ALL_PANELS.filter((p) => p.navGroup === "integrations")
);

export const HOST_SERVICE_PANELS_SIDEBAR = [...ALL_PANELS].sort(
  (a, b) => a.sidebarOrder - b.sidebarOrder
);

export const HOST_SERVICE_PANELS_SIDEBAR_PRIMARY = HOST_SERVICE_PANELS_SIDEBAR.filter(
  (p) => p.sidebarSection === "services"
);
export const HOST_SERVICE_PANELS_SIDEBAR_INTEGRATIONS = HOST_SERVICE_PANELS_SIDEBAR.filter(
  (p) => p.sidebarSection === "integrations"
);

export const HOST_SERVICE_PANEL_BY_TAB = ALL_PANELS.reduce(
  (acc, p) => {
    acc[p.tab] = p;
    return acc;
  },
  {} as Record<Services, HostServicePanelDefinition>
);

/** Non-`Services` inspector tabs — titles for the window header. */
export const EXTRA_INSPECTOR_TAB_TITLES: Record<string, string> = {
  scenes: "Canvas & Elements",
  project: "Project",
  elements: "Elements",
  files: "Files",
  settings: "Settings",
  text: "Text Element",
  image: "Image Element",
};

export function isHostServiceInspectorTab(tab: string | undefined): tab is Services {
  return tab !== undefined && tab in HOST_SERVICE_PANEL_BY_TAB;
}

export function getInspectorHeaderTitle(tab: string | undefined): string {
  if (!tab) return "Dashboard";
  if (isHostServiceInspectorTab(tab)) return HOST_SERVICE_PANEL_BY_TAB[tab].title;
  return EXTRA_INSPECTOR_TAB_TITLES[tab] ?? tab;
}

export const HostServiceInspector: FC<{ tab: Services }> = ({ tab }) => {
  const def = HOST_SERVICE_PANEL_BY_TAB[tab];
  const Cmp = def.Inspector;
  return <Cmp key={tab} />;
};
