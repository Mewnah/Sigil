import { FC, memo, useLayoutEffect, useRef } from "react";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import { systemLogState } from "@/core/services/systemLog";

type SystemLogsPanelProps = {
  /** When true, omit title/footer (used inside tabbed right panel). */
  embedded?: boolean;
};

function formatTime(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return "";
  }
}

function levelClass(level: string): string {
  switch (level) {
    case "error":
      return "text-error";
    case "success":
      return "text-success";
    case "warning":
      return "text-warning";
    default:
      return "text-base-content/75";
  }
}

export const SystemLogsPanel: FC<SystemLogsPanelProps> = memo(({ embedded }) => {
  const { entries } = useSnapshot(systemLogState);
  const bottomRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length]);

  return (
    <div className={classNames("flex flex-col h-full min-h-0 bg-base-100", embedded ? "px-3 pb-2 pt-1" : "p-4")}>
      {!embedded && (
        <div className="font-bold border-b border-base-content/10 pb-2 mb-2 flex justify-between items-center text-sm opacity-70 shrink-0">
          <span>System Logs</span>
        </div>
      )}
      <div className="flex-1 min-h-0 overflow-auto text-xs font-mono space-y-1.5 select-text pr-1 custom-scrollbar">
        {entries.length === 0 ? (
          <div className="text-base-content/40 leading-relaxed">
            No service events yet. After startup, STT, TTS, Translation, and AI Transform status changes appear here. Use the dock or footer to start
            services.
          </div>
        ) : (
          entries.map((e) => (
            <div key={e.id} className={classNames("leading-snug break-words", levelClass(e.level))}>
              <span className="text-base-content/35">{formatTime(e.at)}</span>{" "}
              <span className="font-semibold text-base-content/60">[{e.tag}]</span> {e.message}
            </div>
          ))
        )}
        <div ref={bottomRef} aria-hidden />
      </div>
      {!embedded && (
        <p className="text-base-content/30 italic text-[10px] pt-2 mt-2 border-t border-base-content/5 shrink-0">
          Select an element on the canvas to view its properties.
        </p>
      )}
    </div>
  );
});
