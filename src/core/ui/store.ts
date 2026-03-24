import { InspectorTabPath } from "@/types";
import { create } from "zustand";
import { devtools } from "zustand/middleware";

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

  sidebar: SidebarSlice;
  changeTab: (v?: InspectorTabPath) => void;
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

      sidebar: { ...initialSidebar },

      changeTab: (v?: InspectorTabPath) =>
        set((s) => {
          const { tab, show } = s.sidebar;
          if (tab?.tab === v?.tab && tab?.value === v?.value && show) {
            return {
              sidebar: { ...s.sidebar, show: false, tab: undefined },
            };
          }
          return { sidebar: { ...s.sidebar, tab: v, show: true } };
        }),

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
