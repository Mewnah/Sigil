import { FC, memo } from "react";
import { useSnapshot } from "valtio";
import { RiFileAddLine, RiFolderOpenLine, RiTimeLine } from "react-icons/ri";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";

const DashboardCard: FC<{
  title: string;
  icon: any;
  onClick?: () => void;
  color?: string;
}> = ({ title, icon: Icon, onClick, color = "primary" }) => (
  <button
    onClick={onClick}
    className={`btn btn-outline h-auto py-8 flex flex-col gap-4 hover:bg-${color} hover:text-white border-2`}
  >
    <Icon size={40} />
    <span className="text-lg font-bold">{title}</span>
  </button>
);

export const SigilDashboard: FC<any> = memo(() => {
  const { recentSnapshots } = useSnapshot(window.ApiServer.state);

  const handleNewProject = () => {
    if (confirm("Create new project? Unsaved changes will be lost.")) {
      // Reset elements
      window.ApiClient.document.patch((state) => {
        state.elements = {};
        state.elementsIds = [];
      });

      // Navigate to editor
      window.ApiServer.changeTab({ tab: "scenes" });
    }
  };

  const handleOpenProject = async () => {
    try {
      const selectedPath = await open({
        multiple: false,
        filters: [
          {
            name: "Sigil Project",
            extensions: ["json", "sigil"],
          },
        ],
      });

      if (selectedPath && typeof selectedPath === "string") {
        const contents = await readTextFile(selectedPath);
        const snapshot = JSON.parse(contents);

        // Load parsed snapshot
        window.ApiClient.document.patch((state) => {
          Object.assign(state, snapshot);
        });

        window.ApiServer.changeTab({ tab: "scenes" });
      }
    } catch (error) {
      console.error("Failed to open project:", error);
      alert("Failed to open project file.");
    }
  };

  const loadSnapshot = async (snap: any) => {
    try {
      const contents = await readTextFile(snap.path);
      const snapshot = JSON.parse(contents);

      window.ApiClient.document.patch((state) => {
        Object.assign(state, snapshot);
      });

      window.ApiServer.changeTab({ tab: "scenes" });
    } catch (error) {
      console.error("Failed to load snapshot:", error);
      alert(
        "Failed to load snapshot. The file may have been moved or deleted.",
      );
    }
  };

  return (
    <div className="flex flex-col h-full bg-base-300 p-8 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full flex flex-col gap-12 mt-12">
        <div className="text-center space-y-2">
          <h1 className="text-5xl font-black tracking-tight text-white">
            Sigil<span className="text-primary">Studio</span>
          </h1>
          <p className="text-xl text-base-content/50 font-medium">
            Stream overlay composer
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Start Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-bold border-b border-base-content/10 pb-4">
              Start
            </h2>
            <div className="grid grid-cols-2 gap-6">
              <DashboardCard
                title="New Project"
                icon={RiFileAddLine}
                onClick={handleNewProject}
                color="primary"
              />
              <DashboardCard
                title="Open Project"
                icon={RiFolderOpenLine}
                onClick={handleOpenProject}
                color="secondary"
              />
            </div>
          </div>

          {/* Recent Section */}
          <div className="space-y-6">
            <div className="flex justify-between items-center border-b border-base-content/10 pb-4">
              <h2 className="text-2xl font-bold">Recent</h2>
              {recentSnapshots.length > 0 && (
                <button
                  className="btn btn-ghost btn-xs text-error"
                  onClick={() => {
                    window.ApiServer.state.recentSnapshots = [];
                  }}
                >
                  Clear
                </button>
              )}
            </div>

            {recentSnapshots.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-12 bg-base-200 rounded-2xl border border-dashed border-base-content/10 text-base-content/30 gap-4">
                <RiTimeLine size={48} />
                <p className="font-medium">No recent projects</p>
              </div>
            ) : (
              <div className="space-y-2">
                {recentSnapshots.map((snap) => (
                  <button
                    key={snap.id}
                    onClick={() => loadSnapshot(snap)}
                    className="w-full p-4 bg-base-200 hover:bg-base-100 border border-base-content/5 rounded-xl flex items-center justify-between group transition-all text-left"
                  >
                    <div>
                      <div className="font-bold group-hover:text-primary transition-colors">
                        {snap.name}
                      </div>
                      <div className="text-xs opacity-50 font-mono mt-1">
                        {snap.path}
                      </div>
                    </div>
                    <div className="text-xs opacity-50 bg-base-300 px-2 py-1 rounded">
                      {new Date(snap.date).toLocaleDateString()}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});
