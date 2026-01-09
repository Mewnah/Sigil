import { FC, memo, useState } from "react";
import { useSnapshot } from "valtio";
import { SigilHeader } from "./SigilHeader";
import QuickActionDock from "../QuickActionDock";
import StatsPanel from "../StatsPanel";
import { TTSInputBar } from "./TTSInputBar";
import { EditorViewport } from "../editor-view";
import HistoryPanel from "../HistoryPanel";
import { Services } from "@/core";
import { canvasToolbarState } from "../CanvasToolbar";

export const SigilLayout: FC = memo(() => {
    const { tab } = useSnapshot(window.ApiServer.ui.sidebarState);
    const toolbarState = useSnapshot(canvasToolbarState);

    // Modern "Floating" Layout
    // 1. EditorViewport is the full-screen background (Z-0)
    // 2. Header, Dock, Stats, and TTS are floating layers (Z-10+)

    return (
        <div className="flex flex-col h-screen w-screen bg-base-300 text-base-content overflow-hidden font-sans selection:bg-primary/30 relative">

            {/* BACKGROUND LAYER: Canvas/Viewport */}
            {/* Positions absolutely to fill screen below header area (roughly) or full screen if header is floating */}
            {/* Let's keep Header static at top for stability, Canvas fills rest */}

            {!toolbarState.presentationMode && (
                <div className="flex-none z-50 relative shadow-xl">
                    <SigilHeader sidebarCollapsed={false} onToggleSidebar={() => { }} />
                </div>
            )}

            <div className="relative flex-1 overflow-hidden w-full h-full group/layout">
                {/* CANVAS - Fills container */}
                <div className="absolute inset-0 z-0">
                    <EditorViewport />
                    {/* History Panel is usually internal or floating. We place it here. */}
                    {toolbarState.showHistory && !toolbarState.presentationMode && (
                        <div className="absolute left-20 bottom-20 z-20">
                            {/* Adjusted position to avoid Dock/TTS overlapping */}
                            <HistoryPanel />
                        </div>
                    )}
                </div>

                {/* FLOATING UI LAYER */}
                {!toolbarState.presentationMode && (
                    <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-between p-4">

                        <div className="flex flex-1 min-h-0 relative">
                            {/* LEFT: Quick Action Dock */}
                            <div className="pointer-events-auto h-full flex flex-col justify-center mr-auto">
                                <div className="rounded-2xl overflow-hidden shadow-2xl border border-white/5 bg-base-100/60 backdrop-blur-md">
                                    <QuickActionDock />
                                </div>
                            </div>

                            {/* RIGHT: Stats Panel */}
                            <div className="pointer-events-auto h-full flex flex-col justify-center ml-auto">
                                {/* StatsPanel manages its own sizing/collapse */}
                                <div className="rounded-2xl overflow-hidden shadow-2xl border border-white/5 bg-base-100/60 backdrop-blur-md">
                                    <StatsPanel />
                                </div>
                            </div>
                        </div>

                        {/* BOTTOM: TTS Input */}
                        {/* Centered floating bar */}
                        <div className="flex-none flex justify-center mt-4 pointer-events-auto">
                            <div className="w-[600px] max-w-full bg-base-100/80 backdrop-blur-xl rounded-full shadow-2xl border border-white/10 p-1">
                                <TTSInputBar />
                            </div>
                        </div>

                    </div>
                )}
            </div>

            {/* Presentation Mode: Full Canvas (Header/Dock hidden above) */}
            {/* Note: EditorViewport is already in z-0. If presentationMode is true, the overlays are hidden via conditionals above. */}

        </div>
    );
});
