import { FC, memo } from "react";
import classNames from "classnames";
import {
    RiDashboardFill,
    RiFolderMusicFill,
    RiHardDrive2Fill,
    RiSettings3Fill,
    RiLayoutMasonryFill,
    RiStackFill,
} from "react-icons/ri";
import { useAppUIStore } from "../store";
import Tooltip from "../dropdown/Tooltip";
import {
    HOST_SERVICE_PANELS_INTEGRATIONS,
    HOST_SERVICE_PANELS_LIVE,
} from "../servicePanels";

interface NavItemProps {
    label: string;
    icon: React.ReactNode;
    active?: boolean;
    collapsed?: boolean;
    onClick: () => void;
}

const NavItem: FC<NavItemProps> = ({ label, icon, active, collapsed, onClick }) => {
    const button = (
        <button
            onClick={onClick}
            title={collapsed ? label : undefined}
            className={classNames(
                "w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all text-sm font-medium",
                collapsed ? "justify-center" : "",
                active
                    ? "bg-primary/20 text-primary"
                    : "text-base-content/60 hover:bg-base-content/5 hover:text-base-content"
            )}
        >
            <span className="text-lg flex-shrink-0">{icon}</span>
            {!collapsed && <span className="flex-1 truncate text-left">{label}</span>}
        </button>
    );

    if (collapsed) {
        return <Tooltip content={label} placement="right">{button}</Tooltip>;
    }
    return button;
};

interface NavGroupProps {
    title: string;
    collapsed?: boolean;
    children: React.ReactNode;
}

const NavGroup: FC<NavGroupProps> = ({ title, collapsed, children }) => (
    <div className="mb-4">
        {!collapsed && (
            <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-widest text-base-content/30 select-none">
                {title}
            </div>
        )}
        <div className="space-y-0.5">
            {children}
        </div>
    </div>
);

interface SigilNavigationProps {
    collapsed?: boolean;
}

export const SigilNavigation: FC<SigilNavigationProps> = memo(({ collapsed = false }) => {
    const tab = useAppUIStore((s) => s.sidebar.tab);

    const navigate = (t: string) => window.ApiServer.changeTab({ tab: t as any });
    const isActive = (t: string) => tab?.tab === t;

    return (
        <nav className={classNames(
            "w-full bg-base-200 border-r border-base-content/5 flex flex-col h-full overflow-y-auto transition-all",
            collapsed ? "py-4 px-1" : "py-6 px-3"
        )}>
            {/* Main Section */}
            <NavGroup title="Main" collapsed={collapsed}>
                <NavItem
                    label="Dashboard"
                    icon={<RiDashboardFill />}
                    active={!tab || tab.tab === undefined}
                    collapsed={collapsed}
                    onClick={() => navigate(undefined as any)}
                />
                <NavItem
                    label="Project"
                    icon={<RiStackFill />}
                    active={tab?.tab === 'project'}
                    collapsed={collapsed}
                    onClick={() => navigate('project')}
                />
            </NavGroup>

            {/* Live Services */}
            <NavGroup title="Live Services" collapsed={collapsed}>
                {HOST_SERVICE_PANELS_LIVE.map((p) => {
                    const Icon = p.NavIcon;
                    return (
                        <NavItem
                            key={p.tab}
                            label={p.title}
                            icon={<Icon />}
                            active={isActive(p.tab)}
                            collapsed={collapsed}
                            onClick={() => navigate(p.tab)}
                        />
                    );
                })}
            </NavGroup>

            {/* Studio */}
            <NavGroup title="Studio" collapsed={collapsed}>
                <NavItem
                    label="Canvas & Elements"
                    icon={<RiLayoutMasonryFill />}
                    active={isActive('scenes') || isActive('elements')}
                    collapsed={collapsed}
                    onClick={() => navigate('scenes')}
                />
                <NavItem
                    label="Files"
                    icon={<RiFolderMusicFill />}
                    active={isActive('files')}
                    collapsed={collapsed}
                    onClick={() => navigate('files')}
                />
            </NavGroup>

            {/* Integrations */}
            <NavGroup title="Integrations" collapsed={collapsed}>
                {HOST_SERVICE_PANELS_INTEGRATIONS.map((p) => {
                    const Icon = p.NavIcon;
                    return (
                        <NavItem
                            key={p.tab}
                            label={p.title}
                            icon={<Icon />}
                            active={isActive(p.tab)}
                            collapsed={collapsed}
                            onClick={() => navigate(p.tab)}
                        />
                    );
                })}
            </NavGroup>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Models & Runtimes and Settings at bottom */}
            <div className="pt-4 border-t border-base-content/5 space-y-0.5">
                <NavItem
                    label="Models & Runtimes"
                    icon={<RiHardDrive2Fill />}
                    active={isActive('models')}
                    collapsed={collapsed}
                    onClick={() => navigate('models')}
                />
                <NavItem
                    label="Settings"
                    icon={<RiSettings3Fill />}
                    active={isActive('settings')}
                    collapsed={collapsed}
                    onClick={() => navigate('settings')}
                />
            </div>
        </nav>
    );
});
