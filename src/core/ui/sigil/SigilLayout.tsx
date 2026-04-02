import { FC, memo, useState } from "react";
import { RiCloseLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { useShallow } from "zustand/react/shallow";
import { useAppUIStore } from "../store";
import { ErrorBoundary, FallbackProps } from "react-error-boundary";
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
import { StatusFooter } from "./StatusFooter";
import { StudioRightDefaultPanel } from "./StudioRightDefaultPanel";
import { AnimatePresence, motion } from "framer-motion";
import Tooltip from "../dropdown/Tooltip";
import { toast } from "react-toastify";
import { useTranslation } from "react-i18next";

const PanelErrorFallback = ({ error, resetErrorBoundary }: FallbackProps) => (
    <div className="p-4 bg-error/10 text-error border border-error rounded-lg h-full overflow-auto flex flex-col gap-3">
        <h3 className="font-bold text-sm">Panel error</h3>
        <pre className="text-[10px] font-mono whitespace-pre-wrap opacity-90 flex-1 min-h-0">{error.message}</pre>
        <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-sm btn-primary" onClick={resetErrorBoundary}>
                Try again
            </button>
            <button
                type="button"
                className="btn btn-sm btn-ghost"
                onClick={() => {
                    useAppUIStore.getState().setSidebarSelections([]);
                    window.ApiServer.closeSidebar();
                    resetErrorBoundary();
                }}
            >
                Dismiss
            </button>
        </div>
    </div>
);

export const SigilLayout: FC = memo(() => {
    const { t } = useTranslation();
    const { tab, selections } = useAppUIStore(
        useShallow((s) => ({ tab: s.sidebar.tab, selections: s.sidebar.selections }))
    );
    const toolbarState = useSnapshot(canvasToolbarState);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    const handleDuplicateSelection = () => {
        if (!selections || selections.length === 0) return;
        const created = window.ApiClient.elements.duplicateElements(selections);
        if (created.length === 0) return;
        useAppUIStore.getState().setSidebarSelections(created);
        toast.success(t("elements.toast_duplicated_n", { count: created.length }));
    };

    const handleAlignElements = (alignment: 'left' | 'center' | 'right' | 'top' | 'middle' | 'bottom') => {
        if (!selections || selections.length === 0) return;
        const canvas = window.ApiClient.document.fileBinder.get().canvas;
        const activeScene = window.ApiClient.scenes.state.activeScene;

        window.ApiClient.document.patch(state => {
            selections.forEach(id => {
                const el = state.elements[id];
                if (!el || !el.scenes[activeScene] || !el.scenes[activeScene].rect) return;
                const rect = el.scenes[activeScene].rect;

                switch (alignment) {
                    case 'left': rect.x = 0; break;
                    case 'center': rect.x = Math.round(canvas.w / 2 - rect.w / 2); break;
                    case 'right': rect.x = Math.round(canvas.w - rect.w); break;
                    case 'top': rect.y = 0; break;
                    case 'middle': rect.y = Math.round(canvas.h / 2 - rect.h / 2); break;
                    case 'bottom': rect.y = Math.round(canvas.h - rect.h); break;
                }
            });
        });
    };

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
                            <div className="w-88 h-full">
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
                            <CanvasToolbar
                                onAlignElements={handleAlignElements}
                                onDuplicateSelection={handleDuplicateSelection}
                                duplicateDisabled={!selections || selections.length === 0}
                            />
                        </div>
                    )}

                    {/* VIEWPORT & FLOATING OVERLAYS */}
                    <div className="relative flex-1 overflow-hidden w-full h-full group/layout">

                        {/* 0. CANVAS (Background) */}
                        <div className="absolute inset-0 z-0">
                            <EditorViewport />
                        </div>

                        {/* 1. FLOATING OVERLAYS (TTS) */}
                        {!toolbarState.presentationMode && (
                            <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-end p-3">
                                {/* TTS Input (Bottom-Center - Floating above Bottom Panel) */}
                                <div className="flex-none flex justify-center mt-2 pointer-events-auto mb-2">
                                    <div className="w-[600px] max-w-full bg-base-100/80 backdrop-blur-xl rounded-lg shadow-2xl border border-white/10 p-1">
                                        <TTSInputBar />
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* 2. History — sibling above TTS layer (z-10) so panel stays visible & clickable */}
                        {toolbarState.showHistory && !toolbarState.presentationMode && (
                            <div className="pointer-events-auto absolute bottom-4 left-4 z-20">
                                <HistoryPanel />
                            </div>
                        )}
                    </div>

                    {/* BOTTOM PANEL (Service Controls) */}
                    {!toolbarState.presentationMode && (
                        <StatusFooter />
                    )}

                </div>

                {/* 4. RIGHT PANEL (Logs OR Property Inspector) */}
                {!toolbarState.presentationMode && (
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "22rem" }}
                        className="flex-none h-full z-40 relative shadow-lg border-l border-base-content/5 bg-base-100 hidden lg:block overflow-hidden"
                    >
                        <div className="w-88 h-full flex flex-col">
                            <ErrorBoundary
                                FallbackComponent={PanelErrorFallback}
                                resetKeys={[activeSelectionId ?? ""]}
                            >
                                {activeSelectionId ? (
                                    // A. Property Inspector (When Element Selected)
                                    <PropertyInspector selectionId={activeSelectionId} />
                                ) : (
                                    <StudioRightDefaultPanel />
                                )}
                            </ErrorBoundary>
                        </div>
                    </motion.div>
                )}

            </div>

            {toolbarState.presentationMode && (
                <div className="pointer-events-auto fixed right-3 top-3 z-[100]">
                    <Tooltip content="Exit presentation (Esc or F11)" placement="left">
                        <button
                            type="button"
                            className="btn btn-sm gap-1 border border-base-content/10 bg-base-300/95 text-base-content shadow-lg backdrop-blur-sm hover:bg-base-300"
                            onClick={() => {
                                canvasToolbarState.presentationMode = false;
                            }}
                            aria-label="Exit presentation"
                        >
                            <RiCloseLine className="text-base" aria-hidden />
                            Exit presentation
                        </button>
                    </Tooltip>
                </div>
            )}

        </div>
    );
});
