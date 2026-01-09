import { FC, memo, useState } from "react";
import { useSnapshot } from "valtio";
import { SigilHeader } from "./SigilHeader";
import QuickActionDock from "../QuickActionDock";
import StatsPanel from "../StatsPanel";
import { TTSInputBar } from "./TTSInputBar";
import { EditorViewport } from "../editor-view";
import HistoryPanel from "../HistoryPanel";
import { Services } from "@/core";
import CanvasToolbar, { canvasToolbarState } from "../CanvasToolbar";
import { SigilNavigation } from "./SigilNavigation";
import { SigilDashboard } from "./SigilDashboard";
import Inspector from "../inspector";

export const SigilLayout: FC = memo(() => {
    const { tab } = useSnapshot(window.ApiServer.ui.sidebarState);
    const toolbarState = useSnapshot(canvasToolbarState);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    // Modern "Floating" Layout with Sidebar
    // 1. Header (Top)
    // 2. Body (Flex Row)
    //    a. Navigation (Left)
    //    b. Canvas Area (Right, flex-1)
    //       i. Toolbar (Top of Canvas)
    //       ii. Viewport (Fill)
    //       iii. Floating Overlays (Dock, Stats, TTS)

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

                {/* SIDEBAR NAVIGATION */}
                {!toolbarState.presentationMode && (
                    <div className="flex-none h-full z-40 relative shadow-lg">
                        <SigilNavigation collapsed={sidebarCollapsed} />
                    </div>
                )}

                {/* CANVAS AREA CONTAINER */}
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

                        {/* 1. FLOATING OVERLAYS */}
                        {!toolbarState.presentationMode && (
                            <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-between p-4">

                                <div className="flex flex-1 min-h-0 relative">
                                    {/* Quick Actions (Left-Middle) */}
                                    <div className="pointer-events-auto h-full flex flex-col justify-center mr-auto pl-4">
                                        <div className="rounded-2xl overflow-hidden shadow-2xl border border-white/5 bg-base-100/60 backdrop-blur-md">
                                            <QuickActionDock />
                                        </div>
                                    </div>

                                    {/* Stats (Right-Middle) */}
                                    <div className="pointer-events-auto h-full flex flex-col justify-center ml-auto pr-4">
                                        <div className="rounded-2xl overflow-hidden shadow-2xl border border-white/5 bg-base-100/60 backdrop-blur-md">
                                            <StatsPanel />
                                        </div>
                                    </div>
                                </div>

                                {/* TTS Input (Bottom-Center) */}
                                <div className="flex-none flex justify-center mt-4 pointer-events-auto mb-8">
                                    <div className="w-[600px] max-w-full bg-base-100/80 backdrop-blur-xl rounded-full shadow-2xl border border-white/10 p-1">
                                        <TTSInputBar />
                                    </div>
                                </div>

                            </div>
                        )}
                    </div>
                </div>

                {/* INSPECTOR SIDEBAR (Right Side) - Shows when a tab is ACTIVE */}
                {tab?.tab && !toolbarState.presentationMode && (
                    <div className="flex-none h-full z-40 relative shadow-lg border-l border-base-content/5 bg-base-100">
                        <Inspector path={tab} />
                    </div>
                )}
            </div>

            {/* Dashboard Overlay */}
            {!tab?.tab && !toolbarState.presentationMode && (
                <div className="absolute inset-0 z-[60] bg-base-300 top-[48px] left-[64px]">
                    <SigilDashboard />
                </div>
            )}

        </div>
    );
});
