import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { CSSProperties, FC, MouseEvent, useEffect, useState } from "react";
import { CAPTION_FRAME_EVENT, CaptionFrame, CaptionFrameLine, readCaptionFrame } from "./frame";
import "./captions.css";

/** Line height as a multiple of the font size (also sizes the box's row limit). */
const LINE_HEIGHT = 1.3;

/**
 * The caption window's page (P6-B): renders the latest {@link CaptionFrame} the main window sent. Text is rendered as
 * React text (never HTML); `dir="auto"` lays out right-to-left lines correctly and `lang` picks CJK glyph shapes.
 */
export const CaptionWindowView: FC = () => {
  const [frame, setFrame] = useState<CaptionFrame | null>(null);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    let lastSeq = -1;
    listen<unknown>(CAPTION_FRAME_EVENT, (event) => {
      const next = readCaptionFrame(event.payload);
      // The main window re-sends the latest frame while the window is open: skip repeats.
      if (!next || next.seq === lastSeq) return;
      lastSeq = next.seq;
      setFrame(next);
    })
      .then((fn) => {
        if (disposed) fn();
        else unlisten = fn;
      })
      .catch((error) => console.error("[captions] cannot listen for captions", error));
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  if (!frame) return null;
  const { style, unlocked } = frame;

  const startDrag = (e: MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    getCurrentWindow().startDragging().catch(() => { });
  };

  const box: CSSProperties = {
    fontFamily: style.fontFamily,
    fontSize: `${style.fontSize}px`,
    lineHeight: LINE_HEIGHT,
    color: style.textColor,
    backgroundColor: style.backgroundColor,
    maxHeight: `calc(${style.maxLines * LINE_HEIGHT}em + 0.4em)`,
  };

  return (
    <div className={unlocked ? "sigil-captions unlocked" : "sigil-captions"} onMouseDown={unlocked ? startDrag : undefined}>
      {unlocked && frame.hint && <div className="sigil-captions-hint">{frame.hint}</div>}
      {frame.lines.length > 0 && (
        <div className="sigil-captions-box" style={box}>
          {frame.lines.map((line) => (
            <p key={line.key} className={line.interim ? "interim" : undefined} lang={line.lang ?? undefined} dir="auto">
              <LineText line={line} />
            </p>
          ))}
        </div>
      )}
    </div>
  );
};

const LineText: FC<{ line: CaptionFrameLine }> = ({ line }) => {
  if (!line.interim || line.stable === null || line.stable >= line.text.length) return <>{line.text}</>;
  return (
    <>
      {line.text.slice(0, line.stable)}
      <span className="tentative">{line.text.slice(line.stable)}</span>
    </>
  );
};

export default CaptionWindowView;
