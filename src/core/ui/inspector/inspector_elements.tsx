import { useGetState } from "@/client";
import { ElementType } from "@/client/elements/schema";
import { FC, memo, useState, MouseEvent } from "react";
import { createPortal } from "react-dom";
import { RiAddCircleFill, RiDeleteBin5Fill, RiFileCopyLine, RiLayoutMasonryFill, RiSave3Fill, RiTextWrap, RiImageFill } from "react-icons/ri";
import { ElementTypeIcon } from "../ElementTypeIcon";
import { useSnapshot } from "valtio";
import { useShallow } from "zustand/react/shallow";
import { useAppUIStore } from "../store";
import Inspector from "./components";
import Tooltip from "../dropdown/Tooltip";
import { useTranslation } from "react-i18next";
import classNames from "classnames";
import { toast } from "react-toastify";
import { ConfirmModal } from "../components/ConfirmModal";

interface ElementRowProps {
    id: string;
    name: string;
    type: ElementType;
    isActive: boolean;
    onSelect: (e: MouseEvent) => void;
    onDuplicate: () => void;
    onDelete: () => void;
}

const ElementRow: FC<ElementRowProps> = memo(({ id, name, type, isActive, onSelect, onDuplicate, onDelete }) => {
    const { t } = useTranslation();
    return (
        <div
            className={classNames(
                "flex items-center gap-3 px-3 py-2 rounded transition-colors cursor-pointer group",
                isActive ? "bg-primary/20 text-primary" : "hover:bg-base-content/5"
            )}
            onClick={onSelect}
        >
            <ElementTypeIcon type={type} className="text-lg flex-shrink-0" />
            <span className="flex-1 truncate font-medium text-sm">{name}</span>
            <span className="text-xs text-base-content/30 uppercase">{type}</span>
            <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <Tooltip content={t("elements.duplicate_tooltip")} placement="top">
                    <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onDuplicate(); }}
                        title={t("elements.duplicate_tooltip")}
                        className="p-1 rounded hover:bg-primary/15 hover:text-primary transition-all"
                        aria-label={t("elements.duplicate_tooltip")}
                    >
                        <RiFileCopyLine />
                    </button>
                </Tooltip>
                <Tooltip content="Delete" placement="top">
                    <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onDelete(); }}
                        title="Delete element"
                        className="p-1 rounded hover:bg-error/20 hover:text-error transition-all"
                        aria-label="Delete element"
                    >
                        <RiDeleteBin5Fill />
                    </button>
                </Tooltip>
            </div>
        </div>
    );
});

const TemplateRow: FC<{ template: any, onInstantiate: () => void, onDelete: () => void }> = memo(({ template, onInstantiate, onDelete }) => {
    return (
        <div className="flex items-center gap-3 px-3 py-2 rounded transition-colors group hover:bg-base-content/5">
            <RiLayoutMasonryFill className="text-lg flex-shrink-0 opacity-50" />
            <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">{template.name}</div>
                <div className="text-xs text-base-content/30 uppercase">{template.type}</div>
            </div>
            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity gap-1">
                <Tooltip content="Add to Canvas" placement="top">
                    <button onClick={onInstantiate} aria-label="Add to Canvas" className="p-1 rounded hover:bg-primary/20 hover:text-primary transition-all">
                        <RiAddCircleFill />
                    </button>
                </Tooltip>
                <Tooltip content="Delete" placement="top">
                    <button onClick={onDelete} aria-label="Delete Template" className="p-1 rounded hover:bg-error/20 hover:text-error transition-all">
                        <RiDeleteBin5Fill />
                    </button>
                </Tooltip>
            </div>
        </div>
    );
});

const Inspector_Elements: FC = () => {
    const { t } = useTranslation();
    const elementsIds = useGetState(state => state.elementsIds);
    const elements = useGetState(state => state.elements);
    const { elementTemplates } = useSnapshot(window.ApiServer.state);
    const { tab, selections } = useAppUIStore(
        useShallow((s) => ({ tab: s.sidebar.tab, selections: s.sidebar.selections }))
    );
    const [view, setView] = useState<'active' | 'templates'>('active');
    const [deleteModalOpen, setDeleteModalOpen] = useState(false);
    const [elementIdsPendingDelete, setElementIdsPendingDelete] = useState<string[]>([]);

    const requestDeleteElements = (ids: string[]) => {
        if (ids.length === 0) return;
        setElementIdsPendingDelete(ids);
        setTimeout(() => setDeleteModalOpen(true), 0);
    };

    const handleDeleteModalCancel = () => {
        setDeleteModalOpen(false);
        setElementIdsPendingDelete([]);
    };

    const confirmDeleteElements = () => {
        const ids = elementIdsPendingDelete;
        if (ids.length === 0) return;

        if (ids.length === 1) {
            const id = ids[0];
            window.ApiClient.elements.removeElement(id);
            const currentSelections = useAppUIStore.getState().sidebar.selections ?? [];
            if (currentSelections.includes(id)) {
                useAppUIStore.getState().setSidebarSelections(currentSelections.filter((pid) => pid !== id));
            }
        } else {
            ids.forEach((id) => window.ApiClient.elements.removeElement(id));
            useAppUIStore.getState().setSidebarSelections([]);
            window.ApiServer.changeTab({ tab: undefined as any });
            toast.success("Deleted selected elements");
        }

        setDeleteModalOpen(false);
        setElementIdsPendingDelete([]);
    };

    const handleAddText = () => {
        window.ApiClient.elements.addElement(ElementType.text, "main");
    };

    const handleAddImage = () => {
        window.ApiClient.elements.addElement(ElementType.image, "main");
    };

    const handleSelectElement = (id: string, type: ElementType, e: MouseEvent) => {
        if (e.shiftKey) {
            // Multi-selection logic
            const currentSelections = [...(useAppUIStore.getState().sidebar.selections ?? [])];
            const idx = currentSelections.indexOf(id);

            if (idx === -1) {
                currentSelections.push(id);
            } else {
                currentSelections.splice(idx, 1);
            }

            useAppUIStore.getState().setSidebarSelections(currentSelections);
            // Do NOT change tab, as this would replace the canvas inspector.
            // Selection now drives the Right Panel (PropertyInspector).
        } else {
            // Single selection
            useAppUIStore.getState().setSidebarSelections([id]);
            // window.ApiServer.changeTab({ tab: type, value: id });
        }
    };

    const handleDeleteElement = (id: string) => {
        requestDeleteElements([id]);
    };

    const handleDuplicateElement = (id: string) => {
        const newId = window.ApiClient.elements.duplicateElement(id);
        if (!newId) return;
        useAppUIStore.getState().setSidebarSelections([newId]);
        toast.success(t("elements.toast_duplicated_one"));
    };

    const handleDuplicateSelected = () => {
        if (!selections || selections.length === 0) return;
        const created = window.ApiClient.elements.duplicateElements(selections);
        if (created.length === 0) return;
        useAppUIStore.getState().setSidebarSelections(created);
        toast.success(
            created.length === 1
                ? t("elements.toast_duplicated_one")
                : t("elements.toast_duplicated_n", { count: created.length })
        );
    };

    const handleDeleteSelected = () => {
        if (!selections || selections.length === 0) return;
        requestDeleteElements([...selections]);
    };

    const handleSaveTemplate = () => {
        const selectedId = tab?.value;
        if (!selectedId || !elements[selectedId]) return;

        const element = elements[selectedId];
        const templateId = Math.random().toString(36).substr(2, 9);

        window.ApiServer.state.elementTemplates.push({
            id: templateId,
            name: element.name,
            type: element.type,
            data: JSON.parse(JSON.stringify(element))
        });
        toast.success("Template saved!");
    };

    const handleInstantiateTemplate = (template: any) => {
        const newId = Math.random().toString(36).substr(2, 9);
        const newElement = JSON.parse(JSON.stringify(template.data));
        newElement.name = `${template.name} Copy`;

        // Manual insertion
        window.ApiClient.document.patch(state => {
            state.elements[newId] = newElement;
            state.elementsIds.push(newId);
        });

        toast.success("Added from template");
    };

    const handleDeleteTemplate = (id: string) => {
        if (confirm("Delete this template?")) {
            const index = window.ApiServer.state.elementTemplates.findIndex((t: any) => t.id === id);
            if (index !== -1) {
                window.ApiServer.state.elementTemplates.splice(index, 1);
            }
        }
    };

    const deleteModalMessage =
        elementIdsPendingDelete.length <= 1
            ? "Are you sure you want to delete this element? This action cannot be undone."
            : `Are you sure you want to delete ${elementIdsPendingDelete.length} selected elements? This action cannot be undone.`;

    return (
        <>
        <Inspector.Body>
            <Inspector.Header>
                <div className="flex gap-2 p-1 bg-base-300 rounded-lg w-full">
                    <button
                        className={classNames("flex-1 text-xs py-1 rounded font-bold transition-colors", view === 'active' ? "bg-base-100 shadow text-base-content" : "text-base-content/50 hover:text-base-content")}
                        onClick={() => setView('active')}
                    >
                        Active
                    </button>
                    <button
                        className={classNames("flex-1 text-xs py-1 rounded font-bold transition-colors", view === 'templates' ? "bg-base-100 shadow text-base-content" : "text-base-content/50 hover:text-base-content")}
                        onClick={() => setView('templates')}
                    >
                        Templates
                    </button>
                </div>
            </Inspector.Header>
            <Inspector.Content>
                {view === 'active' ? (
                    <>
                        {/* Add Buttons */}
                        <Inspector.SubHeader>Add New Element</Inspector.SubHeader>
                        <div className="flex flex-wrap gap-2">
                            <button
                                onClick={handleAddText}
                                className="flex-1 min-w-[5rem] flex items-center justify-center gap-2 h-10 bg-white/5 hover:bg-white/10 rounded text-sm font-medium transition-colors"
                            >
                                <RiTextWrap /> Text
                            </button>
                            <button
                                onClick={handleAddImage}
                                className="flex-1 min-w-[5rem] flex items-center justify-center gap-2 h-10 bg-white/5 hover:bg-white/10 rounded text-sm font-medium transition-colors"
                            >
                                <RiImageFill /> Image
                            </button>
                        </div>

                        {/* Elements List */}
                        <Inspector.SubHeader>
                            <div className="flex justify-between items-center w-full">
                                <span>Active Elements ({elementsIds?.length || 0})</span>
                                {(selections && selections.length > 0) && (
                                    <div className="flex items-center gap-1">
                                        <button
                                            type="button"
                                            onClick={handleDuplicateSelected}
                                            className="text-[10px] bg-primary/10 hover:bg-primary/20 text-primary px-2 py-0.5 rounded uppercase font-bold tracking-wider transition-colors"
                                        >
                                            {t("elements.duplicate_n", { count: selections.length })}
                                        </button>
                                        <button
                                            type="button"
                                            onClick={handleDeleteSelected}
                                            className="text-[10px] bg-error/10 hover:bg-error/20 text-error px-2 py-0.5 rounded uppercase font-bold tracking-wider transition-colors"
                                        >
                                            Delete {selections.length}
                                        </button>
                                    </div>
                                )}
                            </div>
                        </Inspector.SubHeader>
                        <div className="flex flex-col gap-1 -mx-3">
                            {elementsIds?.map(id => {
                                const element = elements?.[id];
                                if (!element) return null;

                                const isActive = tab?.value === id || (selections && selections.includes(id));

                                return (
                                    <ElementRow
                                        key={id}
                                        id={id}
                                        name={element.name}
                                        type={element.type}
                                        isActive={isActive}
                                        onSelect={(e) => handleSelectElement(id, element.type, e)}
                                        onDuplicate={() => handleDuplicateElement(id)}
                                        onDelete={() => handleDeleteElement(id)}
                                    />
                                );
                            })}

                            {(!elementsIds || elementsIds.length === 0) && (
                                <div className="text-center text-base-content/30 text-sm py-8">
                                    No elements yet.
                                </div>
                            )}
                        </div>

                        {/* Save as Template Button */}
                        {tab?.value && elements[tab.value] && (
                            <div className="mt-4 pt-4 border-t border-base-content/10">
                                <button
                                    onClick={handleSaveTemplate}
                                    className="w-full flex items-center justify-center gap-2 h-8 bg-primary/10 hover:bg-primary/20 text-primary rounded text-xs font-bold transition-colors"
                                >
                                    <RiSave3Fill /> Save Selection as Template
                                </button>
                            </div>
                        )}
                    </>
                ) : (
                    <>
                        {/* Templates List */}
                        <Inspector.SubHeader>Saved Templates ({elementTemplates?.length || 0})</Inspector.SubHeader>
                        <div className="flex flex-col gap-1 -mx-3">
                            {elementTemplates?.map((template: any) => (
                                <TemplateRow
                                    key={template.id}
                                    template={template}
                                    onInstantiate={() => handleInstantiateTemplate(template)}
                                    onDelete={() => handleDeleteTemplate(template.id)}
                                />
                            ))}

                            {(!elementTemplates || elementTemplates.length === 0) && (
                                <div className="text-center text-base-content/30 text-sm py-8 space-y-2">
                                    <RiLayoutMasonryFill className="mx-auto text-2xl opacity-50" />
                                    <div>No templates saved.</div>
                                    <div className="text-xs opacity-50">Select an active element and click "Save as Template" to create one.</div>
                                </div>
                            )}
                        </div>
                    </>
                )}
            </Inspector.Content>
        </Inspector.Body>
        {createPortal(
            <ConfirmModal
                isOpen={deleteModalOpen}
                title={elementIdsPendingDelete.length <= 1 ? "Delete Element" : "Delete Elements"}
                message={deleteModalMessage}
                confirmText="Delete"
                variant="danger"
                onConfirm={() => confirmDeleteElements()}
                onCancel={handleDeleteModalCancel}
                showDontAskAgain={false}
            />,
            document.body
        )}
        </>
    );
};

export default Inspector_Elements;
