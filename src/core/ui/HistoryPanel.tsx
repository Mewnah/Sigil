import { FC, memo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { RiArrowGoBackLine, RiArrowGoForwardLine } from "react-icons/ri";
import { documentUndoState } from "@/client/services/document";

const HistoryPanel: FC = memo(() => {
    const { t } = useTranslation();
    const docUndo = useSnapshot(documentUndoState);

    return (
        <div className="w-64 max-w-[calc(100vw-2rem)] bg-base-200 shadow-xl rounded-lg border border-base-content/10 flex flex-col max-h-96 z-50 overflow-hidden font-sans">
            <div className="flex items-center justify-between p-2 border-b border-base-content/5 bg-base-300">
                <div className="text-xs font-bold uppercase tracking-wider opacity-50 px-2">
                    {t("history_panel.title")}
                </div>
                <div className="flex gap-1">
                    <button
                        type="button"
                        disabled={!docUndo.canUndo}
                        onClick={() => window.ApiClient.document.undo()}
                        className="p-1 rounded hover:bg-base-content/10 disabled:opacity-20 transition-colors"
                        title="Undo"
                    >
                        <RiArrowGoBackLine />
                    </button>
                    <button
                        type="button"
                        disabled={!docUndo.canRedo}
                        onClick={() => window.ApiClient.document.redo()}
                        className="p-1 rounded hover:bg-base-content/10 disabled:opacity-20 transition-colors"
                        title="Redo"
                    >
                        <RiArrowGoForwardLine />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto w-full custom-scrollbar p-4">
                <p className="text-xs leading-relaxed text-base-content/60 text-left">
                    {t("history_panel.hint")}
                </p>
            </div>
        </div>
    );
});

export default HistoryPanel;
