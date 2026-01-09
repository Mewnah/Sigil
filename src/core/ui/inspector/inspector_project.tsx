import { FC, memo, useState } from "react";
import Inspector_Elements from "./inspector_elements";
import Inspector_Scenes from "./inspector_scenes";
import classNames from "classnames";
import { RiStackFill, RiLayoutMasonryFill } from "react-icons/ri";

const Inspector_Project: FC = memo(() => {
    const [subTab, setSubTab] = useState<'elements' | 'scenes'>('elements');

    return (
        <div className="flex flex-col h-full w-full bg-base-100">
            {/* Project Header / Tabs */}
            <div className="flex-none p-4 border-b border-base-content/10">
                <h2 className="text-xl font-bold mb-4">Project</h2>
                <div className="flex bg-base-200 rounded-lg p-1 gap-1">
                    <button
                        onClick={() => setSubTab('elements')}
                        className={classNames(
                            "flex-1 flex items-center justify-center gap-2 py-2 rounded text-sm font-medium transition-all",
                            subTab === 'elements' ? "bg-base-100 shadow text-primary" : "text-base-content/50 hover:text-base-content"
                        )}
                    >
                        <RiStackFill /> Elements
                    </button>
                    <button
                        onClick={() => setSubTab('scenes')}
                        className={classNames(
                            "flex-1 flex items-center justify-center gap-2 py-2 rounded text-sm font-medium transition-all",
                            subTab === 'scenes' ? "bg-base-100 shadow text-primary" : "text-base-content/50 hover:text-base-content"
                        )}
                    >
                        <RiLayoutMasonryFill /> Scenes
                    </button>
                </div>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-hidden relative">
                {subTab === 'elements' && <Inspector_Elements />}
                {subTab === 'scenes' && <Inspector_Scenes />}
            </div>
        </div>
    );
});

export default Inspector_Project;
