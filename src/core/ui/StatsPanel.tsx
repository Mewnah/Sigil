import { FC, memo, lazy, Suspense, useEffect } from "react";
import { motion } from "framer-motion";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import { RiArrowRightSLine, RiArrowLeftSLine } from "react-icons/ri";
import { ServiceNetworkState } from "@/types";
import { useAppUIStore, useStatsPanelCollapsed, useToggleStatsPanel } from "./store";
import Inspector from "./inspector";

interface StatCardProps {
    label: string;
    value: string | number;
    unit?: string;
    color?: string;
}

const StatCard: FC<StatCardProps> = memo(({ label, value, unit, color = "text-base-content" }) => (
    <div className="px-3 py-2 bg-base-200 rounded-lg">
        <div className="text-xs text-base-content/60 mb-1">{label}</div>
        <div className={classNames("text-lg font-bold", color)}>
            {value}
            {unit && <span className="text-sm font-normal ms-1">{unit}</span>}
        </div>
    </div>
));

// Main Stats Panel
const StatsPanel: FC = memo(() => {
    // Zustand: Selective subscription (only re-renders when collapsed changes)
    const collapsed = useStatsPanelCollapsed();
    const toggleStatsPanel = useToggleStatsPanel();

    // Valtio: Service states (keep for now, managed by backend)
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);

    // Get active tab/inspector path
    const tab = useAppUIStore((s) => s.sidebar.tab);

    // Listen for keyboard shortcut (Ctrl+I)
    useEffect(() => {
        window.addEventListener("sigil:toggle-stats", toggleStatsPanel);
        return () => window.removeEventListener("sigil:toggle-stats", toggleStatsPanel);
    }, [toggleStatsPanel]);

    if (collapsed) {
        return (
            <motion.aside
                initial={{ width: 40 }}
                animate={{ width: 40 }}
                className="flex-none h-full bg-transparent border-l border-base-content/10 flex flex-col items-center justify-center bg-base-200/50"
            >
                <button
                    onClick={toggleStatsPanel}
                    className="btn btn-ghost btn-circle btn-sm"
                    aria-label="Expand stats panel"
                >
                    <RiArrowLeftSLine className="text-xl" />
                </button>
            </motion.aside>
        );
    }

    return (
        <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 380, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="flex-none h-full bg-transparent border-l border-base-content/10 flex flex-col overflow-hidden"
            role="complementary"
            aria-label="Live statistics and settings"
        >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-base-content/10 bg-base-200/30">
                <h2 className="text-sm font-bold opacity-80 uppercase tracking-wide">Stats & Inspector</h2>
                <button
                    onClick={toggleStatsPanel}
                    className="btn btn-ghost btn-sm btn-circle"
                    aria-label="Collapse stats panel"
                >
                    <RiArrowRightSLine className="text-xl" />
                </button>
            </div>

            {/* Live Stats Section (Fixed Height ~30-40%) */}
            <div className="px-4 py-4 border-b border-base-content/10 space-y-3 bg-transparent shrink-0">

                <div className="grid grid-cols-2 gap-2">
                    <StatCard
                        label="STT Service"
                        value={sttState.status === ServiceNetworkState.connected ? "Online" : "Offline"}
                        color={sttState.status === ServiceNetworkState.connected ? "text-success" : "text-base-content/50"}
                    />
                    <StatCard
                        label="TTS Service"
                        value={ttsState.status === ServiceNetworkState.connected ? "Online" : "Offline"}
                        color={ttsState.status === ServiceNetworkState.connected ? "text-success" : "text-base-content/50"}
                    />
                    <StatCard
                        label="AI Service"
                        value={transformState.status === ServiceNetworkState.connected ? "Online" : "Offline"}
                        color={transformState.status === ServiceNetworkState.connected ? "text-success" : "text-base-content/50"}
                    />

                </div>
            </div>

            {/* Quick Settings Inspector (Takes remaining space) */}
            <div className="flex-1 overflow-hidden flex flex-col bg-base-200/30 relative">
                <Inspector path={tab} />
            </div>
        </motion.aside>
    );
});

export default StatsPanel;
