import { FC, memo } from "react";
import { RiAddLine, RiSubtractLine, RiGridLine, RiAlignLeft, RiAlignCenter, RiAlignRight, RiAlignTop, RiAlignVertically, RiAlignBottom, RiEyeLine, RiFullscreenLine, RiHistoryLine, RiArrowGoBackLine, RiArrowGoForwardLine, RiFileCopyLine } from "react-icons/ri";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { proxy } from "valtio";
import classNames from "classnames";
import { documentUndoState } from "@/client/services/document";
import Tooltip from "./dropdown/Tooltip";

// Canvas toolbar state
export const canvasToolbarState = proxy({
    zoom: 100,
    showGrid: false,
    gridSize: 20,
    showHistory: false,
    presentationMode: false,
});

interface CanvasToolbarProps {
    onZoomChange?: (zoom: number) => void;
    onAlignElements?: (alignment: 'left' | 'center' | 'right' | 'top' | 'middle' | 'bottom') => void;
    onDuplicateSelection?: () => void;
    duplicateDisabled?: boolean;
    onPresentationMode?: () => void;
}

const CanvasToolbar: FC<CanvasToolbarProps> = memo(({ onZoomChange, onAlignElements, onDuplicateSelection, duplicateDisabled, onPresentationMode }) => {
    const { t } = useTranslation();
    const state = useSnapshot(canvasToolbarState);
    const undo = useSnapshot(documentUndoState);

    const handleZoomIn = () => {
        const newZoom = Math.min(200, state.zoom + 10);
        canvasToolbarState.zoom = newZoom;
        onZoomChange?.(newZoom);
    };

    const handleZoomOut = () => {
        const newZoom = Math.max(25, state.zoom - 10);
        canvasToolbarState.zoom = newZoom;
        onZoomChange?.(newZoom);
    };

    const handleZoomReset = () => {
        canvasToolbarState.zoom = 100;
        onZoomChange?.(100);
    };

    const toggleGrid = () => {
        canvasToolbarState.showGrid = !canvasToolbarState.showGrid;
    };

    const toggleHistory = () => {
        canvasToolbarState.showHistory = !canvasToolbarState.showHistory;
    };

    const togglePresentationMode = () => {
        canvasToolbarState.presentationMode = !canvasToolbarState.presentationMode;
        onPresentationMode?.();
    };

    return (
        <div className="flex items-center gap-1 h-8 px-2 bg-base-100 border-b border-base-content/5">
            {/* Zoom Controls */}
            <div className="flex items-center gap-1 border-r border-base-content/10 pr-2 mr-1">
                <Tooltip content="Zoom out" placement="bottom">
                    <button
                        onClick={handleZoomOut}
                        className="w-6 h-6 flex items-center justify-center rounded hover:bg-base-content/10 text-base-content/60 hover:text-base-content transition-colors"
                    >
                        <RiSubtractLine size={14} />
                    </button>
                </Tooltip>
                <button
                    onClick={handleZoomReset}
                    className="px-2 h-6 text-xs font-medium text-base-content/60 hover:text-base-content hover:bg-base-content/10 rounded transition-colors min-w-[48px]"
                >
                    {state.zoom}%
                </button>
                <Tooltip content="Zoom in" placement="bottom">
                    <button
                        onClick={handleZoomIn}
                        className="w-6 h-6 flex items-center justify-center rounded hover:bg-base-content/10 text-base-content/60 hover:text-base-content transition-colors"
                    >
                        <RiAddLine size={14} />
                    </button>
                </Tooltip>
            </div>

            {/* Undo / Redo */}
            <div className="flex items-center gap-0.5 border-r border-base-content/10 pr-2 mr-1">
                <Tooltip content="Undo (Ctrl+Z)" placement="bottom">
                    <button
                        type="button"
                        disabled={!undo.canUndo}
                        onClick={() => window.ApiClient.document.undo()}
                        className="w-6 h-6 flex items-center justify-center rounded hover:bg-base-content/10 text-base-content/60 hover:text-base-content transition-colors disabled:opacity-25 disabled:pointer-events-none"
                    >
                        <RiArrowGoBackLine size={14} />
                    </button>
                </Tooltip>
                <Tooltip content="Redo (Ctrl+Y / Ctrl+Shift+Z)" placement="bottom">
                    <button
                        type="button"
                        disabled={!undo.canRedo}
                        onClick={() => window.ApiClient.document.redo()}
                        className="w-6 h-6 flex items-center justify-center rounded hover:bg-base-content/10 text-base-content/60 hover:text-base-content transition-colors disabled:opacity-25 disabled:pointer-events-none"
                    >
                        <RiArrowGoForwardLine size={14} />
                    </button>
                </Tooltip>
            </div>

            {/* Grid Toggle */}
            <Tooltip content={state.showGrid ? "Hide grid" : "Show grid"} placement="bottom">
                <button
                    onClick={toggleGrid}
                    className={classNames(
                        "w-6 h-6 flex items-center justify-center rounded transition-colors",
                        state.showGrid
                            ? "bg-primary/20 text-primary"
                            : "text-base-content/60 hover:text-base-content hover:bg-base-content/10"
                    )}
                >
                    <RiGridLine size={14} />
                </button>
            </Tooltip>

            {/* Divider */}
            <div className="w-px h-4 bg-base-content/10 mx-1" />

            <Tooltip content={t("elements.duplicate_tooltip")} placement="bottom">
                <button
                    type="button"
                    disabled={duplicateDisabled}
                    onClick={() => onDuplicateSelection?.()}
                    className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors disabled:opacity-25 disabled:pointer-events-none"
                >
                    <RiFileCopyLine size={14} />
                </button>
            </Tooltip>

            <div className="w-px h-4 bg-base-content/10 mx-1" />

            {/* Alignment Tools */}
            <div className="flex items-center gap-0.5">
                <Tooltip content="Align left" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('left')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignLeft size={14} />
                    </button>
                </Tooltip>
                <Tooltip content="Align center" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('center')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignCenter size={14} />
                    </button>
                </Tooltip>
                <Tooltip content="Align right" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('right')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignRight size={14} />
                    </button>
                </Tooltip>
                <div className="w-px h-4 bg-base-content/10 mx-0.5" />
                <Tooltip content="Align top" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('top')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignTop size={14} />
                    </button>
                </Tooltip>
                <Tooltip content="Align middle" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('middle')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignVertically size={14} />
                    </button>
                </Tooltip>
                <Tooltip content="Align bottom" placement="bottom">
                    <button
                        onClick={() => onAlignElements?.('bottom')}
                        className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                    >
                        <RiAlignBottom size={14} />
                    </button>
                </Tooltip>
            </div>

            {/* Spacer */}
            <div className="flex-1" />

            {/* History Toggle */}
            <Tooltip content={state.showHistory ? "Hide edit history panel" : "Show edit history (undo / redo)"} placement="bottom">
                <button
                    type="button"
                    onClick={toggleHistory}
                    className={classNames(
                        "w-6 h-6 flex items-center justify-center rounded transition-colors mr-2",
                        state.showHistory
                            ? "bg-primary/20 text-primary"
                            : "text-base-content/60 hover:text-base-content hover:bg-base-content/10"
                    )}
                >
                    <RiHistoryLine size={14} />
                </button>
            </Tooltip>

            {/* Presentation Mode */}
            <Tooltip content="Presentation mode (F11)" placement="bottom">
                <button
                    type="button"
                    onClick={togglePresentationMode}
                    className="w-6 h-6 flex items-center justify-center rounded text-base-content/60 hover:text-base-content hover:bg-base-content/10 transition-colors"
                >
                    <RiFullscreenLine size={14} />
                </button>
            </Tooltip>
        </div>
    );
});

export default CanvasToolbar;
