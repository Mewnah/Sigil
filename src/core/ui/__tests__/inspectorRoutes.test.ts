/**
 * Rewrite (the `transform` service) has no left-menu entry or inspector of its own: it is the second tab of Speech
 * to Text. Requests for the old tab (the bottom-bar card, saved links) resolve there, in the store's `changeTab`.
 */
import { afterEach, describe, expect, it } from "vitest";
import { Services } from "@/services-registry";
import { STT_REWRITE_SECTION, resolveInspectorTab } from "../inspectorRoutes";
import {
  HOST_SERVICE_PANELS_INTEGRATIONS,
  HOST_SERVICE_PANELS_LIVE,
  HOST_SERVICE_PANEL_BY_TAB,
  getInspectorHeaderTitle,
  isHostServiceInspectorTab,
} from "../servicePanels";
import { selectSttView } from "../inspector/stt/SttTabs";
import { useAppUIStore } from "../store";

const closed = { tab: undefined, show: false, expand: false, selections: [] as string[] };

afterEach(() => {
  useAppUIStore.setState({ sidebar: { ...closed } });
});

describe("resolveInspectorTab", () => {
  it("sends the Rewrite tab to the Speech to Text inspector's Rewrite section", () => {
    expect(STT_REWRITE_SECTION).toBe("rewrite");
    expect(resolveInspectorTab({ tab: Services.transform })).toEqual({ tab: Services.stt, value: "rewrite" });
    // Whatever value an old link carried, the section is what matters.
    expect(resolveInspectorTab({ tab: Services.transform, value: "x" })).toEqual({ tab: Services.stt, value: "rewrite" });
  });

  it("passes every other path through untouched", () => {
    const stt = { tab: Services.stt };
    expect(resolveInspectorTab(stt)).toBe(stt);
    const models = { tab: "models" as const, value: "stt/some-model" };
    expect(resolveInspectorTab(models)).toBe(models);
    for (const tab of Object.values(Services).filter((s) => s !== Services.transform))
      expect(resolveInspectorTab({ tab })).toEqual({ tab });
    expect(resolveInspectorTab(undefined)).toBeUndefined();
  });
});

describe("changeTab with the Rewrite alias", () => {
  const open = () => useAppUIStore.getState().sidebar;
  const change = (v?: Parameters<ReturnType<typeof useAppUIStore.getState>["changeTab"]>[0]) =>
    useAppUIStore.getState().changeTab(v);

  it("opens Speech to Text on the Rewrite tab, and toggles it shut on the same request", () => {
    change({ tab: Services.transform });
    expect(open()).toMatchObject({ show: true, tab: { tab: Services.stt, value: "rewrite" } });
    change({ tab: Services.transform });
    expect(open()).toMatchObject({ show: false, tab: undefined });
  });

  it("treats a direct request for the Rewrite section like the alias", () => {
    change({ tab: Services.stt, value: STT_REWRITE_SECTION });
    change({ tab: Services.transform });
    expect(open().show).toBe(false);
  });

  it("opens the Speech tab (no section) when Speech to Text is chosen from the Rewrite tab", () => {
    change({ tab: Services.transform });
    change({ tab: Services.stt });
    expect(open()).toMatchObject({ show: true, tab: { tab: Services.stt } });
    expect(open().tab?.value).toBeUndefined();
  });

  it("leaves other tabs as they were", () => {
    change({ tab: Services.tts });
    expect(open()).toMatchObject({ show: true, tab: { tab: Services.tts } });
    change({ tab: "settings" });
    expect(open().tab?.tab).toBe("settings");
    change({ tab: "settings" });
    expect(open().show).toBe(false);
  });
});

describe("the tabs inside the Speech to Text panel keep the store in step", () => {
  const sidebar = () => useAppUIStore.getState().sidebar;
  const change = (v: Parameters<ReturnType<typeof useAppUIStore.getState>["changeTab"]>[0]) => useAppUIStore.getState().changeTab(v);

  it("setSidebarTab moves the open panel without toggling or showing it", () => {
    const { setSidebarTab } = useAppUIStore.getState();
    // Closed stays closed (nothing here opens a panel).
    setSidebarTab({ tab: Services.stt, value: STT_REWRITE_SECTION });
    expect(sidebar()).toMatchObject({ show: false, tab: { tab: Services.stt, value: "rewrite" } });
    // Open stays open, on the same request twice.
    change({ tab: Services.stt });
    setSidebarTab({ tab: Services.stt, value: STT_REWRITE_SECTION });
    setSidebarTab({ tab: Services.stt, value: STT_REWRITE_SECTION });
    expect(sidebar()).toMatchObject({ show: true, tab: { tab: Services.stt, value: "rewrite" } });
    // It resolves the Rewrite alias like changeTab does.
    setSidebarTab({ tab: Services.transform });
    expect(sidebar().tab).toEqual({ tab: Services.stt, value: "rewrite" });
  });

  it("a click on a tab names that tab in the store", () => {
    change({ tab: Services.stt });
    selectSttView("rewrite");
    expect(sidebar()).toMatchObject({ show: true, tab: { tab: Services.stt, value: "rewrite" } });
    selectSttView("speech");
    expect(sidebar().show).toBe(true);
    expect(sidebar().tab).toEqual({ tab: Services.stt, value: undefined });
  });

  it("bottom-bar Rewrite card, then the Speech tab, then the card again shows Rewrite (it does not close the panel)", () => {
    change({ tab: Services.transform });
    selectSttView("speech");
    change({ tab: Services.transform });
    expect(sidebar()).toMatchObject({ show: true, tab: { tab: Services.stt, value: "rewrite" } });
    // And the card now closes it, because the panel is showing Rewrite.
    change({ tab: Services.transform });
    expect(sidebar().show).toBe(false);
  });

  it("Speech to Text from the left menu, then the Rewrite tab, then the menu item shows Speech (it does not close the panel)", () => {
    change({ tab: Services.stt });
    selectSttView("rewrite");
    change({ tab: Services.stt });
    expect(sidebar()).toMatchObject({ show: true, tab: { tab: Services.stt } });
    expect(sidebar().tab?.value).toBeUndefined();
    // On the Speech tab the same item closes it.
    change({ tab: Services.stt });
    expect(sidebar().show).toBe(false);
  });
});

describe("service panels", () => {
  it("has no panel of its own for Rewrite", () => {
    expect(Services.transform in HOST_SERVICE_PANEL_BY_TAB).toBe(false);
    expect(isHostServiceInspectorTab(Services.transform)).toBe(false);
    expect(isHostServiceInspectorTab(Services.stt)).toBe(true);
    expect(getInspectorHeaderTitle(Services.stt)).toBe("Speech to Text");
    const all = [...HOST_SERVICE_PANELS_LIVE, ...HOST_SERVICE_PANELS_INTEGRATIONS].map((p) => p.tab);
    expect(all).not.toContain(Services.transform);
  });

  it("orders the Speech group by the data flow: speak, translate, speak aloud", () => {
    expect(HOST_SERVICE_PANELS_LIVE.map((p) => p.tab)).toEqual([
      Services.stt,
      Services.translation,
      Services.tts,
      Services.voice_changer,
      Services.others,
    ]);
  });

  it("orders the integrations VRChat, OBS Studio, Twitch, Kick, Discord", () => {
    expect(HOST_SERVICE_PANELS_INTEGRATIONS.map((p) => p.tab)).toEqual([
      Services.vrc,
      Services.obs,
      Services.twitch,
      Services.kick,
      Services.discord,
    ]);
  });
});
