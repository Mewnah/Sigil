import { FC, memo } from "react";
import Inspector_Elements from "./inspector_elements";
import Inspector_Scenes from "./inspector_scenes";
import classNames from "classnames";
import { RiStackFill, RiLayoutMasonryFill } from "react-icons/ri";
import { useAppUIStore, type CanvasInspectorSubTab } from "../store";
import { useShallow } from "zustand/react/shallow";

export type InspectorCanvasSubTab = CanvasInspectorSubTab;

const Inspector_Canvas: FC = memo(() => {
  const { subTab, setCanvasInspectorSubTab } = useAppUIStore(
    useShallow((s) => ({
      subTab: s.canvasInspectorSubTab,
      setCanvasInspectorSubTab: s.setCanvasInspectorSubTab,
    }))
  );

  return (
    <div className="flex flex-col h-full w-full bg-base-100">
      <div className="flex-none p-4 border-b border-base-content/10">
        <h2 className="text-xl font-bold mb-4">Canvas &amp; Elements</h2>
        <div className="flex bg-base-200 rounded-lg p-1 gap-1">
          <button
            type="button"
            onClick={() => setCanvasInspectorSubTab("elements")}
            className={classNames(
              "flex-1 flex items-center justify-center gap-2 py-2 rounded text-sm font-medium transition-all",
              subTab === "elements"
                ? "bg-base-100 shadow text-primary"
                : "text-base-content/50 hover:text-base-content"
            )}
          >
            <RiStackFill /> Elements
          </button>
          <button
            type="button"
            onClick={() => setCanvasInspectorSubTab("scenes")}
            className={classNames(
              "flex-1 flex items-center justify-center gap-2 py-2 rounded text-sm font-medium transition-all",
              subTab === "scenes"
                ? "bg-base-100 shadow text-primary"
                : "text-base-content/50 hover:text-base-content"
            )}
          >
            <RiLayoutMasonryFill /> Scenes
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden relative">
        {subTab === "elements" && <Inspector_Elements />}
        {subTab === "scenes" && <Inspector_Scenes />}
      </div>
    </div>
  );
});

export default Inspector_Canvas;
