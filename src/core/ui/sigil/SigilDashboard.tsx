import { type FC, memo } from "react";
import { RiFileAddLine, RiFolderOpenLine } from "react-icons/ri";
import { resetTemplate } from "../projectFileActions";

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
  return (
    <div className="flex flex-col h-full bg-base-300 p-8 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full flex flex-col gap-12 mt-12">
        <div className="text-center space-y-2">
          <h1 className="font-header text-5xl font-black tracking-tight text-white">
            Sigil<span className="text-primary">Studio</span>
          </h1>
          <p className="text-xl text-base-content/50 font-medium">
            Stream overlay composer
          </p>
        </div>

        <div className="max-w-xl mx-auto w-full space-y-6">
          <h2 className="text-2xl font-bold border-b border-base-content/10 pb-4 text-center">
            Start
          </h2>
          <p className="text-sm text-base-content/60 text-center">
            Your layout is one working template (auto-saved on the desktop app). Import a{" "}
            <code className="text-xs">.sigiltmp</code> or Curses <code className="text-xs">.cursestmp</code> backup, a legacy{" "}
            <code className="text-xs">.json</code> export — or reset to a fresh template.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <DashboardCard
              title="Reset template"
              icon={RiFileAddLine}
              onClick={() => void resetTemplate()}
              color="primary"
            />
            <DashboardCard
              title="Import template"
              icon={RiFolderOpenLine}
              onClick={() => void window.ApiClient.document.importDocument()}
              color="secondary"
            />
          </div>
        </div>
      </div>
    </div>
  );
});
