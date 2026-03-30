import { FC, memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useSnapshot } from "valtio";
import classNames from "classnames";

/** `event` field from pubsub textHistory (topic suffix, e.g. stt, translation). */
function sourceLabel(eventKey: string): string {
  const map: Record<string, string> = {
    stt: "Speech-to-text",
    translation: "Translation",
    transform: "AI (output)",
    transform_source: "AI (source)",
    transform_raw: "AI (thinking)",
    textfield: "Typed / shortcut",
  };
  return map[eventKey] || eventKey || "Text";
}

export const TextTranscriptPanel: FC = memo(() => {
  const { lastId, list } = useSnapshot(window.ApiShared.pubsub.textHistory);
  const [sttOnly, setSttOnly] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (!sttOnly) return list;
    return list.filter((e) => e.event === "stt");
  }, [list, sttOnly]);

  useLayoutEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lastId, filtered.length]);

  return (
    <div className="flex flex-col h-full min-h-0 px-3 pb-2 pt-2">
      <p className="text-[11px] leading-snug text-base-content/50 mb-2 shrink-0">
        Final text only (not live partials). Everything that flows through the text pipeline is listed — speech-to-text uses the{" "}
        <span className="text-base-content/70">Speech-to-text</span> label.
      </p>
      <label className="flex items-center gap-2 text-[11px] text-base-content/60 mb-2 cursor-pointer select-none shrink-0">
        <input
          type="checkbox"
          className="checkbox checkbox-xs checkbox-primary"
          checked={sttOnly}
          onChange={(e) => setSttOnly(e.target.checked)}
        />
        Show only speech-to-text
      </label>
      <div className="flex-1 min-h-0 overflow-y-auto text-xs space-y-2 pr-1 custom-scrollbar select-text">
        {filtered.length === 0 ? (
          <div className="text-base-content/40 leading-relaxed text-[11px]">
            {sttOnly
              ? "No STT finals yet. Start speech-to-text from the dock or footer, then speak."
              : "No transcript lines yet. Start STT, type in the TTS bar, or use other text sources — finals appear here."}
          </div>
        ) : (
          [...filtered].reverse().map((event, i) => (
            <div
              key={event.id}
              className={classNames(
                "rounded-md border border-base-content/5 bg-base-content/[0.04] px-2.5 py-2",
                event.event === "stt" && "border-primary/20 bg-primary/5"
              )}
            >
              <div className="text-[10px] uppercase tracking-wide text-base-content/45 mb-1">
                {sourceLabel(event.event)}{" "}
                <span className="text-base-content/25 font-mono normal-case">#{filtered.length - i}</span>
              </div>
              <div className="text-sm text-base-content/90 break-words leading-snug font-mono">{event.value}</div>
            </div>
          ))
        )}
        <div ref={bottomRef} aria-hidden />
      </div>
    </div>
  );
});
