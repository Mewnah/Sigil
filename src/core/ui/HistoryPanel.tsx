import { FC, memo } from "react";
import { useSnapshot } from "valtio";
import { RiArrowGoBackLine, RiArrowGoForwardLine } from "react-icons/ri";
import classNames from "classnames";

const HistoryPanel: FC = memo(() => {
    const history = useSnapshot(window.ApiServer.history.store);

    // Sort past: newest at top (reversed).
    const reversedPast = [...history.past].reverse();

    return (
        <div className="absolute left-4 bottom-4 w-64 bg-base-200 shadow-xl rounded-lg border border-base-content/10 flex flex-col max-h-96 z-50 overflow-hidden font-sans">
            <div className="flex items-center justify-between p-2 border-b border-base-content/5 bg-base-300">
                <div className="text-xs font-bold uppercase tracking-wider opacity-50 px-2">History</div>
                <div className="flex gap-1">
                    <button
                        disabled={!history.canUndo}
                        onClick={() => window.ApiServer.history.undo()}
                        className="p-1 rounded hover:bg-base-content/10 disabled:opacity-20 transition-colors"
                        title="Undo"
                    >
                        <RiArrowGoBackLine />
                    </button>
                    <button
                        disabled={!history.canRedo}
                        onClick={() => window.ApiServer.history.redo()}
                        className="p-1 rounded hover:bg-base-content/10 disabled:opacity-20 transition-colors"
                        title="Redo"
                    >
                        <RiArrowGoForwardLine />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto w-full custom-scrollbar">
                {reversedPast.length === 0 ? (
                    <div className="p-4 text-center text-xs opacity-50">No history yet</div>
                ) : (
                    <div className="flex flex-col text-xs">
                        {reversedPast.map((snap, i) => (
                            <button
                                key={`${snap.timestamp}-${i}`}
                                onClick={() => window.ApiServer.history.restore(snap.timestamp)}
                                className={classNames(
                                    "px-4 py-2 text-left hover:bg-base-content/5 flex justify-between group transition-colors border-b border-base-content/5 last:border-0",
                                    i === 0 ? "font-bold text-primary bg-primary/5" : "text-base-content/70"
                                )}
                            >
                                <span className="truncate pr-2">{snap.description}</span>
                                <span className="opacity-0 group-hover:opacity-50 text-[10px] font-mono whitespace-nowrap">
                                    {new Date(snap.timestamp).toLocaleTimeString()}
                                </span>
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
});

export default HistoryPanel;
