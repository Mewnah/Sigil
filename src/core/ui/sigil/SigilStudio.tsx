import { FC, memo, Suspense, lazy, useState, useEffect, useMemo } from "react";
import { useGetState, useUpdateState } from "@/client";
import { ElementType } from "@/client/elements/schema";
import { useSnapshot } from "valtio";
import { createPortal } from "react-dom";
import { RiAddCircleFill, RiImageFill, RiTextWrap, RiDeleteBin5Fill, RiCloseLine, RiLayoutMasonryFill, RiSearchLine, RiDragMoveFill, RiFileCopyLine, RiLockLine, RiLockUnlockLine } from "react-icons/ri";
import Tooltip from "../dropdown/Tooltip";
import classNames from "classnames";
import { ConfirmModal } from "../components/ConfirmModal";

// Lazy load element inspectors - use vertical versions for right side panel
const Inspector_ElementText = lazy(() => import("../inspector/inspector_text"));
const Inspector_ElementImage = lazy(() => import("../inspector/inspector_image"));

interface ElementRowProps {
    id: string;
    name: string;
    type: ElementType;
    isActive: boolean;
    index: number;
    onSelect: () => void;
    onDelete: () => void;
    onCopy: () => void;
    onDragStart: (index: number) => void;
    onDragOver: (index: number) => void;
    onDragEnd: () => void;
}

const ElementRow: FC<ElementRowProps> = memo(({ id, name, type, isActive, index, onSelect, onDelete, onCopy, onDragStart, onDragOver, onDragEnd }) => {
    const Icon = type === ElementType.text ? RiTextWrap : RiImageFill;

    return (
        <div
            draggable
            onDragStart={() => onDragStart(index)}
            onDragOver={(e) => { e.preventDefault(); onDragOver(index); }}
            onDragEnd={onDragEnd}
            className={classNames(
                "flex items-center gap-2 px-2 py-1.5 rounded transition-colors cursor-pointer group",
                isActive ? "bg-primary/20 text-primary" : "hover:bg-base-content/5"
            )}
            onClick={onSelect}
        >
            <RiDragMoveFill className="text-sm text-base-content/30 cursor-grab active:cursor-grabbing flex-shrink-0" />
            <Icon className="text-base flex-shrink-0" />
            <span className="flex-1 truncate font-medium text-xs">{name}</span>
            <span className="hidden sm:block text-[10px] text-base-content/30 uppercase flex-shrink-0">{type}</span>
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 flex-shrink-0">
                <Tooltip content="Duplicate" placement="top">
                    <button
                        onClick={(e) => { e.stopPropagation(); onCopy(); }}
                        title="Duplicate element"
                        className="p-0.5 rounded hover:bg-base-content/10 transition-all"
                    >
                        <RiFileCopyLine className="text-sm" />
                    </button>
                </Tooltip>
                <Tooltip content="Delete" placement="top">
                    <button
                        onClick={(e) => { e.stopPropagation(); onDelete(); }}
                        title="Delete element"
                        className="p-0.5 rounded hover:bg-error/20 hover:text-error transition-all"
                    >
                        <RiDeleteBin5Fill className="text-sm" />
                    </button>
                </Tooltip>
            </div>
        </div>
    );
});

interface SigilStudioProps {
    slots: {
        left: HTMLDivElement | null;
        right: HTMLDivElement | null;
        bottom: HTMLDivElement | null;
    };
}

const SigilStudio: FC<SigilStudioProps> = memo(({ slots }) => {
    const elementsIds = useGetState(state => state.elementsIds);
    const elements = useGetState(state => state.elements);
    const { tab } = useSnapshot(window.ApiServer.ui.sidebarState);

    // Determine if an element is selected
    const selectedElementId = tab?.value;
    const selectedElementType = tab?.tab as ElementType | undefined;
    const isElementSelected = selectedElementId && (selectedElementType === ElementType.text || selectedElementType === ElementType.image);

    // Resizable bottom panel state
    const [panelHeight, setPanelHeight] = useState(280);
    const [isResizing, setIsResizing] = useState(false);

    // Search filter
    const [searchQuery, setSearchQuery] = useState('');

    // Delete confirmation modal state
    const [deleteModalOpen, setDeleteModalOpen] = useState(false);
    const [elementToDelete, setElementToDelete] = useState<string | null>(null);
    const [skipDeleteConfirm, setSkipDeleteConfirm] = useState(false);

    // Filter elements based on search
    const filteredElementIds = useMemo(() => {
        if (!elementsIds || !searchQuery.trim()) return elementsIds;
        return elementsIds.filter(id => {
            const element = elements?.[id];
            return element?.name?.toLowerCase().includes(searchQuery.toLowerCase());
        });
    }, [elementsIds, elements, searchQuery]);

    // Handlers
    const handleSelect = (id: string, type: ElementType) => {
        window.ApiServer.changeTab({ tab: type as any, value: id });
    };

    const handleDeleteClick = (id: string) => {
        if (skipDeleteConfirm) {
            window.ApiClient.elements.removeElement(id);
        } else {
            setElementToDelete(id);
            setDeleteModalOpen(true);
        }
    };

    const confirmDelete = (dontAskAgain: boolean) => {
        if (elementToDelete) {
            window.ApiClient.elements.removeElement(elementToDelete);

            // If the deleted element was selected, deselect it
            if (elementToDelete === selectedElementId) {
                window.ApiServer.changeTab({ tab: 'scenes' as any });
            }

            if (dontAskAgain) {
                setSkipDeleteConfirm(true);
            }
        }
        setDeleteModalOpen(false);
        setElementToDelete(null);
    };

    const handleCopyElement = (id: string) => {
        window.ApiClient.elements.duplicateElement(id, "main");
    };

    const handleAddText = () => {
        window.ApiClient.elements.addElement(ElementType.text, "main");
    };

    const handleAddImage = () => {
        window.ApiClient.elements.addElement(ElementType.image, "main");
    };

    const handleCloseProperties = () => {
        window.ApiServer.changeTab({ tab: 'scenes' as any });
    };

    // Resize handlers
    const handleResizeStart = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    };

    useEffect(() => {
        if (!isResizing) return;

        const handleMouseMove = (e: MouseEvent) => {
            const newHeight = window.innerHeight - e.clientY;
            setPanelHeight(Math.max(100, Math.min(600, newHeight)));
        };

        const handleMouseUp = () => setIsResizing(false);

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = 'ns-resize';

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
        };
    }, [isResizing]);

    // Drag and Drop handlers
    const handleDragStart = (e: React.DragEvent, index: number) => {
        e.dataTransfer.setData("text/plain", index.toString());
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    const handleDrop = (e: React.DragEvent, targetIndex: number) => {
        e.preventDefault();
        const msg = e.dataTransfer.getData("text/plain");
        const sourceIndex = parseInt(msg, 10);
        if (isNaN(sourceIndex) || sourceIndex === targetIndex) return;

        // Logic to reorder elements would go here, updating state via API
        // For now, this is visual only
        console.log(`Reorder: ${sourceIndex} -> ${targetIndex}`);
    };

    // --------------------------------------------------------------------------------
    // RENDER LOGIC
    // --------------------------------------------------------------------------------

    // --------------------------------------------------------------------------------
    // RENDER LOGIC
    // --------------------------------------------------------------------------------

    // 1. Elements Panel (Now Left Sidebar)
    const [leftTab, setLeftTab] = useState<'elements' | 'scene'>('elements');

    // ported from inspector_scenes.tsx
    const scenes = useGetState(state => state.scenes);
    const canvas = useGetState(state => state.canvas);
    const sceneState = useSnapshot(window.ApiClient.scenes.state);
    const updateState = useUpdateState();
    const snapToGrid = useGetState(state => state.snapToGrid);

    const renderScenePanel = () => (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Canvas Settings */}
            <div className="p-3 border-b border-base-content/5 space-y-3">
                <div className="text-[10px] font-bold uppercase tracking-wide text-base-content/40">Canvas</div>
                <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                        <label className="text-xs text-base-content/60">Width</label>
                        <input
                            type="number"
                            aria-label="Canvas Width"
                            className="w-full bg-base-content/5 rounded px-2 py-1 text-xs border border-transparent focus:border-primary/50 outline-none"
                            value={canvas?.w}
                            onChange={e => updateState(state => { state.canvas.w = parseFloat(e.target.value) })}
                        />
                    </div>
                    <div className="space-y-1">
                        <label className="text-xs text-base-content/60">Height</label>
                        <input
                            type="number"
                            aria-label="Canvas Height"
                            className="w-full bg-base-content/5 rounded px-2 py-1 text-xs border border-transparent focus:border-primary/50 outline-none"
                            value={canvas?.h}
                            onChange={e => updateState(state => { state.canvas.h = parseFloat(e.target.value) })}
                        />
                    </div>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                    <input
                        type="checkbox"
                        aria-label="Snap to Grid"
                        className="checkbox checkbox-xs checkbox-primary rounded-sm"
                        checked={snapToGrid}
                        onChange={e => updateState(state => { state.snapToGrid = e.target.checked })}
                    />
                    <span className="text-xs text-base-content/80">Snap to Grid</span>
                </label>
            </div>

            {/* Scenes List */}
            <div className="flex-1 flex flex-col min-h-0">
                <div className="px-3 py-2 border-b border-base-content/5 flex items-center justify-between">
                    <div className="text-[10px] font-bold uppercase tracking-wide text-base-content/40">Scenes</div>
                    <button
                        onClick={() => window.ApiClient.scenes.addScene()}
                        title="Add Scene"
                        className="text-base-content/40 hover:text-primary transition-colors"
                    >
                        <RiAddCircleFill className="text-lg" />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {scenes && Object.keys(scenes).map(sceneId => {
                        const isActive = sceneId === sceneState.activeScene;
                        return (
                            <div key={sceneId} className={classNames(
                                "flex items-center gap-2 px-2 py-1.5 rounded group transition-colors",
                                isActive ? "bg-primary/20" : "hover:bg-base-content/5"
                            )}>
                                <input
                                    type="radio"
                                    name="active-scene"
                                    aria-label={`Activate scene ${scenes[sceneId].name}`}
                                    className="radio radio-xs radio-primary flex-shrink-0"
                                    checked={isActive}
                                    onChange={() => window.ApiClient.scenes.setActive(sceneId)}
                                />
                                <input
                                    type="text"
                                    aria-label={`Rename scene ${scenes[sceneId].name}`}
                                    className="flex-1 bg-transparent text-xs font-medium outline-none min-w-0"
                                    value={scenes[sceneId].name}
                                    onChange={e => updateState(state => { state.scenes[sceneId].name = e.target.value })}
                                    disabled={sceneId === 'main'} // Main scene might be locked name-wise?
                                />
                                <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={() => window.ApiClient.scenes.duplicateScene(sceneId)}
                                        title="Duplicate"
                                        className="p-1 hover:text-primary transition-colors"
                                    >
                                        <RiFileCopyLine />
                                    </button>
                                    {sceneId !== 'main' && (
                                        <button
                                            onClick={() => window.ApiClient.scenes.deleteScene(sceneId)}
                                            title="Delete"
                                            className="p-1 hover:text-error transition-colors"
                                        >
                                            <RiDeleteBin5Fill />
                                        </button>
                                    )}
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    );

    const renderElementsList = () => (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Add Element Section */}
            <div className="px-3 py-3 border-b border-base-content/5 flex-shrink-0 space-y-2">
                <div className="text-[10px] font-bold uppercase tracking-wide text-base-content/40">Add Element</div>
                <div className="flex gap-2">
                    <button
                        onClick={handleAddText}
                        className="flex-1 flex items-center justify-center gap-2 h-9 bg-primary/10 hover:bg-primary/20 text-primary rounded-lg text-xs font-semibold transition-colors border border-primary/20"
                    >
                        <RiAddCircleFill /> Text
                    </button>
                    <button
                        onClick={handleAddImage}
                        className="flex-1 flex items-center justify-center gap-2 h-9 bg-primary/10 hover:bg-primary/20 text-primary rounded-lg text-xs font-semibold transition-colors border border-primary/20"
                    >
                        <RiAddCircleFill /> Image
                    </button>
                </div>
            </div>

            {/* Search Filter */}
            {elementsIds && elementsIds.length > 0 && (
                <div className="px-3 py-2 border-b border-base-content/5 flex-shrink-0">
                    <div className="relative">
                        <RiSearchLine className="absolute left-2 top-1/2 -translate-y-1/2 text-base-content/30" />
                        <input
                            type="text"
                            placeholder="Search elements..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full h-7 pl-7 pr-2 text-xs bg-base-content/5 rounded border-none outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-base-content/30"
                        />
                    </div>
                </div>
            )}

            {/* Elements List */}
            <div className="flex-1 overflow-y-auto p-2 scrollbar-hide">
                <div className="flex flex-col gap-1">
                    {filteredElementIds?.map((id, index) => {
                        const element = elements?.[id];
                        if (!element) return null;
                        return (
                            <ElementRow
                                key={id}
                                id={id}
                                name={element.name}
                                type={element.type}
                                isActive={id === selectedElementId}
                                index={index}
                                onSelect={() => handleSelect(id, element.type)}
                                onDelete={() => handleDeleteClick(id)}
                                onCopy={() => handleCopyElement(id)}
                                onDragStart={(i) => { }}
                                onDragOver={(i) => { }}
                                onDragEnd={() => { }}
                            />
                        );
                    })}
                </div>
            </div>
        </div>
    );

    const renderElementsPanel = () => (
        <div className="flex flex-col h-full bg-base-200 overflow-hidden pointer-events-auto">
            {/* Header - Match inspector header style */}
            <div className="flex font-header items-center px-4 py-4 text-lg font-bold uppercase tracking-wide text-base-content border-b border-base-content/5 bg-base-100 flex-shrink-0">
                <RiLayoutMasonryFill className="mr-2" />
                Canvas & Elements
            </div>

            {/* Tabs */}
            <div className="flex border-b border-base-content/5 bg-base-100 flex-shrink-0">
                <button
                    onClick={() => setLeftTab('elements')}
                    className={classNames(
                        "flex-1 py-3 text-sm font-semibold transition-colors relative",
                        leftTab === 'elements' ? "text-primary" : "text-base-content/40 hover:text-base-content/60"
                    )}
                >
                    Elements
                    {leftTab === 'elements' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-primary" />}
                </button>
                <button
                    onClick={() => setLeftTab('scene')}
                    className={classNames(
                        "flex-1 py-3 text-sm font-semibold transition-colors relative",
                        leftTab === 'scene' ? "text-primary" : "text-base-content/40 hover:text-base-content/60"
                    )}
                >
                    Scene
                    {leftTab === 'scene' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-primary" />}
                </button>
            </div>

            {/* Content */}
            {leftTab === 'elements' ? renderElementsList() : renderScenePanel()}

            {/* Delete Modal (Rendered here within portal context) */}
            <ConfirmModal
                isOpen={deleteModalOpen}
                title="Delete Element"
                message="Are you sure you want to delete this element? This action cannot be undone."
                confirmText="Delete"
                variant="danger"
                onConfirm={confirmDelete}
                onCancel={() => setDeleteModalOpen(false)}
                showDontAskAgain={true}
            />
        </div>
    );

    // 2. Properties Panel (Right Slot - Vertical Layout)
    const renderPropertiesPanel = () => {
        if (!isElementSelected) return null;

        return (
            <div className="w-[400px] h-full bg-base-200 flex flex-col overflow-hidden pointer-events-auto border-l border-base-content/10">
                {/* Properties Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-base-content/5 bg-base-200 flex-shrink-0">
                    <div className="flex items-center gap-2">
                        {selectedElementType === ElementType.text ? <RiTextWrap className="text-base" /> : <RiImageFill className="text-base" />}
                        <span className="font-bold text-sm">
                            {elements?.[selectedElementId]?.name || "Element"}
                        </span>
                    </div>
                    <button
                        onClick={handleCloseProperties}
                        title="Close properties"
                        className="p-1 rounded hover:bg-base-content/10 transition-colors"
                    >
                        <RiCloseLine className="text-lg" />
                    </button>
                </div>

                {/* Properties Content */}
                <div className="flex-1 overflow-y-auto">
                    <Suspense fallback={<div className="flex items-center justify-center h-full text-sm text-base-content/30">Loading properties...</div>}>
                        {selectedElementType === ElementType.text ? (
                            <Inspector_ElementText id={selectedElementId} />
                        ) : (
                            <Inspector_ElementImage id={selectedElementId} />
                        )}
                    </Suspense>
                </div>
            </div>
        );
    };

    return (
        <>
            {slots.left && createPortal(renderElementsPanel(), slots.left)}
            {slots.right && createPortal(renderPropertiesPanel(), slots.right)}
        </>
    );
});

export default SigilStudio;
