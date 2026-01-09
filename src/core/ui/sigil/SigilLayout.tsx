import { FC, memo, useState } from "react";
import { useSnapshot } from "valtio";
import { ErrorBoundary } from "react-error-boundary";
import { SigilHeader } from "./SigilHeader";
import QuickActionDock from "../QuickActionDock";
import StatsPanel from "../StatsPanel";
import { TTSInputBar } from "./TTSInputBar";
import { EditorViewport } from "../editor-view";
import HistoryPanel from "../HistoryPanel";
import CanvasToolbar, { canvasToolbarState } from "../CanvasToolbar";
import { SigilNavigation } from "./SigilNavigation";
import { SigilDashboard } from "./SigilDashboard";
import Inspector, { PropertyInspector } from "../inspector";
import { BottomPanel } from "../BottomPanel";
import { AnimatePresence, motion } from "framer-motion";

const PanelErrorFallback = ({ error }: { error: Error }) => (
    <div className="p-4 bg-error/10 text-error border border-error rounded-lg h-full overflow-auto">
        <h3 className="font-bold text-sm">Panel Error</h3>
        <pre className="text-[10px] mt-2 font-mono whitespace-pre-wrap">{error.message}</pre>
        <pre className="text-[10px] opacity-50">{error.stack}</pre>
    </div>
);

export const SigilLayout: FC = memo(() => {
    const { tab, selections } = useSnapshot(window.ApiServer.ui.sidebarState);
    const toolbarState = useSnapshot(canvasToolbarState);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    // Determines active element for right panel (single select for now)
    const activeSelectionId = selections && selections.length > 0 ? selections[0] : null;

    return (
        <div className="flex flex-col h-screen w-screen bg-base-300 text-base-content overflow-hidden font-sans selection:bg-primary/30 relative">

            {/* HEADER */}
            {!toolbarState.presentationMode && (
                <div className="flex-none z-50 relative shadow-xl">
                    <SigilHeader
                        sidebarCollapsed={sidebarCollapsed}
                        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                    />
                </div>
            )}

            {/* MAIN CONTENT ROW */}
            <div className="flex flex-1 overflow-hidden relative">

                {/* 1. SIDEBAR NAVIGATION (Leftmost) */}
                {!toolbarState.presentationMode && (
                    <div className="flex-none h-full z-50 relative shadow-lg">
                        <SigilNavigation collapsed={sidebarCollapsed} />
                    </div>
                )}

                {/* 2. LEFT INSPECTOR PANEL (Navigation Driven) */}
                <AnimatePresence>
                    {tab?.tab && !toolbarState.presentationMode && (
                        <motion.div
                            initial={{ width: 0, opacity: 0 }}
                            animate={{ width: "22rem", opacity: 1 }}
                            exit={{ width: 0, opacity: 0 }}
                            transition={{ duration: 0.2, ease: "easeInOut" }}
                            className="flex-none h-full z-40 relative shadow-lg border-r border-base-content/5 bg-base-100 overflow-hidden"
                        >
                            <div className="w-[22rem] h-full">
                                <ErrorBoundary FallbackComponent={PanelErrorFallback}>
                                    <Inspector path={tab} />
                                </ErrorBoundary>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* 3. CANVAS AREA (Middle, flexible) */}
                <div className="flex flex-col flex-1 relative min-w-0">

                    {/* CANVAS TOOLBAR (Zoom, Grid, Align) */}
                    {!toolbarState.presentationMode && (
                        <div className="flex-none z-30 relative">
                            <CanvasToolbar />
                        </div>
                    )}

                    {/* VIEWPORT & FLOATING OVERLAYS */}
                    <div className="relative flex-1 overflow-hidden w-full h-full group/layout">

                        {/* 0. CANVAS (Background) */}
                        <div className="absolute inset-0 z-0">
                            <EditorViewport />
                            {/* History Panel */}
                            {toolbarState.showHistory && !toolbarState.presentationMode && (
                                <div className="absolute left-4 bottom-4 z-20">
                                    <HistoryPanel />
                                </div>
                            )}
                        </div>

                        {/* 1. FLOATING OVERLAYS (TTS) */}
                        {!toolbarState.presentationMode && (
                            <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-end p-4">
                                {/* TTS Input (Bottom-Center - Floating above Bottom Panel) */}
                                <div className="flex-none flex justify-center mt-4 pointer-events-auto mb-4">
                                    <div className="w-[600px] max-w-full bg-base-100/80 backdrop-blur-xl rounded-full shadow-2xl border border-white/10 p-1">
                                        <TTSInputBar />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* BOTTOM PANEL (Service Controls) */}
                    {!toolbarState.presentationMode && (
                        <BottomPanel />
                    )}

                </div>

                {/* 4. RIGHT PANEL (Logs OR Property Inspector) */}
                {!toolbarState.presentationMode && (
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "22rem" }}
                        className="flex-none h-full z-40 relative shadow-lg border-l border-base-content/5 bg-base-100 hidden lg:block overflow-hidden"
                    >
                        <div className="w-[22rem] h-full flex flex-col">
                            <ErrorBoundary FallbackComponent={PanelErrorFallback}>
                                {activeSelectionId ? (
                                    // A. Property Inspector (When Element Selected)
                                    <PropertyInspector selectionId={activeSelectionId} />
                                ) : (
                                    // B. System Logs (Default)
                                    <div className="flex flex-col h-full p-4 bg-base-100">
                                        <div className="font-bold border-b border-base-content/10 pb-2 mb-2 flex justify-between items-center text-sm opacity-70">
                                            <span>System Logs</span>
                                        </div>
                                        <div className="flex-1 overflow-auto text-xs font-mono opacity-50 space-y-1 select-text">
                                            <div className="text-success">[System] Ready.</div>
                                            <div>[Info] Layout initialized.</div>
                                            <div className="text-base-content/30 italic">Select an element to view properties.</div>
                                        </div>
                                    </div>
                                )}
                            </ErrorBoundary>
                        </div>
                    </motion.div>
                )}

            </div>

        </div>
    );
});
