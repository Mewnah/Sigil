import { FC, memo, useRef, useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import { RiTerminalBoxFill, RiArrowLeftSLine, RiArrowRightSLine } from "react-icons/ri";
import { motion, AnimatePresence } from "framer-motion";

// Store session start time globally so it persists
const SESSION_START_TIME = Date.now();

// Logs Panel Component
const LogsPanel: FC = () => {
    const scrollContainer = useRef<HTMLDivElement>(null);
    const { lastId, list } = useSnapshot(window.ApiShared.pubsub.textHistory);

    useEffect(() => {
        setTimeout(() => scrollContainer.current?.scrollTo({ top: scrollContainer.current.scrollHeight, behavior: "smooth" }));
    }, [lastId]);

    return (
        <div className="flex flex-col h-full">
            <div ref={scrollContainer} className="flex-1 overflow-y-auto scrollbar-hide">
                <div className="flex flex-col gap-1 p-2">
                    {list.length === 0 ? (
                        <div className="text-center text-base-content/30 text-xs py-8">No logs yet...</div>
                    ) : (
                        list.map(event => (
                            <div key={event.id} className="flex flex-col rounded-md bg-base-content/5 hover:bg-base-content/10 transition-colors px-3 py-2">
                                <div className="text-[10px] text-base-content/40 uppercase tracking-wide">{event.event}</div>
                                <div className="text-sm text-base-content/90 break-words">{event.value}</div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

// Quick Stats Panel
const StatsPanel: FC = () => {
    const { list } = useSnapshot(window.ApiShared.pubsub.textHistory);

    // Calculate word count from all transcriptions
    const wordCount = list.reduce((acc, event) => {
        return acc + (event.value?.split(/\s+/).filter(Boolean).length || 0);
    }, 0);

    return (
        <div className="px-3 py-2 border-b border-base-content/5 bg-base-300/50">
            <div className="grid grid-cols-2 gap-2 text-center">
                <div>
                    <div className="text-lg font-bold text-base-content">{list.length}</div>
                    <div className="text-[10px] text-base-content/40 uppercase">Entries</div>
                </div>
                <div>
                    <div className="text-lg font-bold text-base-content">{wordCount}</div>
                    <div className="text-[10px] text-base-content/40 uppercase">Words</div>
                </div>
            </div>
        </div>
    );
};

interface LogsSidebarProps {
    isOpen: boolean;
    onToggle: () => void;
}

export const LogsSidebar: FC<LogsSidebarProps> = memo(({ isOpen, onToggle }) => {
    return (
        <>
            {/* Logs Sidebar - Right */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: 280, opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="flex-shrink-0 bg-base-200 border-l border-base-content/5 flex flex-col overflow-hidden"
                    >
                        {/* Logs Header */}
                        <div className="flex items-center gap-2 px-3 py-2 border-b border-base-content/5 flex-shrink-0">
                            <RiTerminalBoxFill className="text-primary" />
                            <span className="font-bold text-xs uppercase tracking-wide text-base-content/70">Live Logs</span>
                        </div>
                        <StatsPanel />
                        <LogsPanel />
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Toggle Button */}
            <button
                onClick={onToggle}
                title={isOpen ? "Hide logs" : "Show logs"}
                className="absolute top-1/2 -translate-y-1/2 bg-base-300 border border-base-content/10 rounded-l-md p-1 hover:bg-primary/20 transition-colors z-20"
                style={{ right: isOpen ? 280 : 0 }}
            >
                {isOpen ? <RiArrowRightSLine className="text-lg" /> : <RiArrowLeftSLine className="text-lg" />}
            </button>
        </>
    );
});
