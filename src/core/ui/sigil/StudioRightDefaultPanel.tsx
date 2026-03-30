import { FC, memo, useState } from "react";
import classNames from "classnames";
import { SystemLogsPanel } from "./SystemLogsPanel";
import { TextTranscriptPanel } from "./TextTranscriptPanel";

type TabId = "transcript" | "services";

export const StudioRightDefaultPanel: FC = memo(() => {
  const [tab, setTab] = useState<TabId>("transcript");

  return (
    <div className="flex flex-col h-full min-h-0 bg-base-100">
      <div className="flex shrink-0 gap-0.5 border-b border-base-content/10 px-2 pt-2">
        <button
          type="button"
          onClick={() => setTab("transcript")}
          className={classNames(
            "px-3 py-1.5 text-xs font-medium rounded-t-md transition-colors",
            tab === "transcript"
              ? "bg-base-200 text-base-content border border-b-0 border-base-content/10 -mb-px"
              : "text-base-content/50 hover:text-base-content hover:bg-base-200/50"
          )}
        >
          Transcript
        </button>
        <button
          type="button"
          onClick={() => setTab("services")}
          className={classNames(
            "px-3 py-1.5 text-xs font-medium rounded-t-md transition-colors",
            tab === "services"
              ? "bg-base-200 text-base-content border border-b-0 border-base-content/10 -mb-px"
              : "text-base-content/50 hover:text-base-content hover:bg-base-200/50"
          )}
        >
          Service log
        </button>
      </div>
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden bg-base-200/30">
        {tab === "transcript" ? <TextTranscriptPanel /> : <SystemLogsPanel embedded />}
      </div>
      <p className="text-base-content/30 italic text-[10px] px-3 py-2 border-t border-base-content/5 shrink-0 bg-base-100">
        Select an element on the canvas to view its properties.
      </p>
    </div>
  );
});
