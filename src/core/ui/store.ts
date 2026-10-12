import { InspectorTabPath } from "@/types";
import { resolveInspectorTab } from "./inspectorRoutes";
import { create } from "zustand";
import { devtools } from "zustand/middleware";

export type CanvasInspectorSubTab = "elements" | "scenes";

const CANVAS_INSPECTOR_SUBTAB_KEY = "sigil.canvasInspectorSubTab";

function readStoredCanvasInspectorSubTab(): CanvasInspectorSubTab {
  try {
    const v = localStorage.getItem(CANVAS_INSPECTOR_SUBTAB_KEY);
    if (v === "elements" || v === "scenes") return v;
  } catch {
    //
  }
  return "elements";
}

export type SidebarSlice = {
  tab: InspectorTabPath | undefined;
  show: boolean;
  expand: boolean;
  selections: string[];
};

interface AppUIState {
  statsPanelCollapsed: boolean;
  toggleStatsPanel: () => void;
  setStatsPanelCollapsed: (collapsed: boolean) => void;

  /** Last-selected Elements vs Scenes in the Canvas & Elements inspector; persisted in localStorage. */
  canvasInspectorSubTab: CanvasInspectorSubTab;
  setCanvasInspectorSubTab: (subTab: CanvasInspectorSubTab) => void;

  sidebar: SidebarSlice;
  changeTab: (v?: InspectorTabPath) => void;
  /**
   * Moves the open panel to another tab or section without toggling or showing it. The tabs inside an inspector (the
   * Speech | Rewrite tabs of Speech to Text) use it, so the store always names what the panel shows: the status-bar
   * cards, the nav and the toasts read it. `changeTab` is for opening and closing.
   */
  setSidebarTab: (v: InspectorTabPath) => void;
  closeSidebar: () => void;
  toggleSidebarExpand: () => void;
  setSidebarShow: (show: boolean) => void;
  setSidebarSelections: (selections: string[]) => void;
}

const initialSidebar: SidebarSlice = {
  tab: undefined,
  show: false,
  expand: false,
  selections: [],
};

export const useAppUIStore = create<AppUIState>()(
  devtools(
    (set) => ({
      statsPanelCollapsed: false,
      toggleStatsPanel: () =>
        set((s) => ({ statsPanelCollapsed: !s.statsPanelCollapsed })),
      setStatsPanelCollapsed: (collapsed: boolean) =>
        set({ statsPanelCollapsed: collapsed }),

      canvasInspectorSubTab: readStoredCanvasInspectorSubTab(),
      setCanvasInspectorSubTab: (subTab) => {
        try {
          localStorage.setItem(CANVAS_INSPECTOR_SUBTAB_KEY, subTab);
        } catch {
          //
        }
        set({ canvasInspectorSubTab: subTab });
      },

      sidebar: { ...initialSidebar },

      changeTab: (requested?: InspectorTabPath) =>
        set((s) => {
          // Aliases (Rewrite lives in the STT inspector) resolve first, so the toggle compares the real target.
          const v = resolveInspectorTab(requested);
          const { tab, show } = s.sidebar;
          if (tab?.tab === v?.tab && tab?.value === v?.value && show) {
            return {
              sidebar: { ...s.sidebar, show: false, tab: undefined },
            };
          }
          return { sidebar: { ...s.sidebar, tab: v, show: true } };
        }),

      setSidebarTab: (requested: InspectorTabPath) =>
        set((s) => ({ sidebar: { ...s.sidebar, tab: resolveInspectorTab(requested) } })),

      closeSidebar: () =>
        set((s) => ({
          sidebar: { ...s.sidebar, tab: undefined, show: false },
        })),

      toggleSidebarExpand: () =>
        set((s) => ({
          sidebar: { ...s.sidebar, expand: !s.sidebar.expand },
        })),

      setSidebarShow: (show: boolean) =>
        set((s) => ({ sidebar: { ...s.sidebar, show } })),

      setSidebarSelections: (selections: string[]) =>
        set((s) => ({ sidebar: { ...s.sidebar, selections } })),
    }),
    { name: "SigilAppUI" }
  )
);

/** @deprecated Prefer useAppUIStore */
export const useUIStore = useAppUIStore;

export const useStatsPanelCollapsed = () =>
  useAppUIStore((s) => s.statsPanelCollapsed);
export const useToggleStatsPanel = () =>
  useAppUIStore((s) => s.toggleStatsPanel);
