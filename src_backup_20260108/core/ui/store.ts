import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

/**
 * UI State Store - Zustand
 * Replaces Valtio for better performance with React 18
 */

interface UIState {
    // Stats Panel
    statsPanelCollapsed: boolean;
    toggleStatsPanel: () => void;
    setStatsPanelCollapsed: (collapsed: boolean) => void;
}

export const useUIStore = create<UIState>()(
    devtools(
        (set) => ({
            // Stats Panel State
            statsPanelCollapsed: false,

            toggleStatsPanel: () =>
                set((state: UIState) => ({ statsPanelCollapsed: !state.statsPanelCollapsed })),

            setStatsPanelCollapsed: (collapsed: boolean) =>
                set({ statsPanelCollapsed: collapsed }),
        }),
        { name: 'SigilUIStore' }
    )
);

// Selector hooks for optimal performance
export const useStatsPanelCollapsed = () => useUIStore((state: UIState) => state.statsPanelCollapsed);
export const useToggleStatsPanel = () => useUIStore((state: UIState) => state.toggleStatsPanel);
