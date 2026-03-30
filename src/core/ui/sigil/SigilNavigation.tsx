import { FC, memo } from "react";
import classNames from "classnames";
import {
    RiDashboardFill,
    RiMicFill,
    RiVolumeUpFill,
    RiTranslate2,
    RiSparklingFill,
    RiFolderMusicFill,
    RiSettings3Fill,
    RiLayoutMasonryFill,
    RiTwitchFill,
    RiDiscordFill,
    RiGamepadFill,
    RiRecordCircleFill,
    RiStackFill,
    RiMagicFill
} from "react-icons/ri";
import { KickIcon } from "../icons/KickIcon";
import { Services } from "@/services-registry";
import { useAppUIStore } from "../store";
import Tooltip from "../dropdown/Tooltip";

interface NavItemProps {
    label: string;
    icon: React.ReactNode;
    active?: boolean;
    collapsed?: boolean;
    shortcut?: string;
    onClick: () => void;
}

const NavItem: FC<NavItemProps> = ({ label, icon, active, collapsed, shortcut, onClick }) => {
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
            {!collapsed && (
                <>
                    <span className="flex-1 truncate text-left">{label}</span>
                    {shortcut && <span className="text-[10px] text-base-content/30 font-mono">{shortcut}</span>}
                </>
            )}
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
                    shortcut="Ctrl+0"
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
                <NavItem
                    label="Speech to Text"
                    icon={<RiMicFill />}
                    active={isActive(Services.stt)}
                    collapsed={collapsed}
                    shortcut="Ctrl+1"
                    onClick={() => navigate(Services.stt)}
                />
                <NavItem
                    label="Text to Speech"
                    icon={<RiVolumeUpFill />}
                    active={isActive(Services.tts)}
                    collapsed={collapsed}
                    shortcut="Ctrl+2"
                    onClick={() => navigate(Services.tts)}
                />
                <NavItem
                    label="Translation"
                    icon={<RiTranslate2 />}
                    active={isActive(Services.translation)}
                    collapsed={collapsed}
                    shortcut="Ctrl+3"
                    onClick={() => navigate(Services.translation)}
                />
                <NavItem
                    label="AI Transform"
                    icon={<RiSparklingFill />}
                    active={isActive(Services.transform)}
                    collapsed={collapsed}
                    shortcut="Ctrl+4"
                    onClick={() => navigate(Services.transform)}
                />
                <NavItem
                    label="Voice Changer"
                    icon={<RiMagicFill />}
                    active={isActive(Services.voice_changer)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.voice_changer)}
                />
            </NavGroup>

            {/* Studio */}
            <NavGroup title="Studio" collapsed={collapsed}>
                <NavItem
                    label="Canvas & Elements"
                    icon={<RiLayoutMasonryFill />}
                    active={isActive('scenes') || isActive('elements')}
                    collapsed={collapsed}
                    shortcut="Ctrl+5"
                    onClick={() => navigate('scenes')}
                />
                <NavItem
                    label="Files"
                    icon={<RiFolderMusicFill />}
                    active={isActive('files')}
                    collapsed={collapsed}
                    shortcut="Ctrl+6"
                    onClick={() => navigate('files')}
                />
            </NavGroup>

            {/* Integrations */}
            <NavGroup title="Integrations" collapsed={collapsed}>
                <NavItem
                    label="Twitch"
                    icon={<RiTwitchFill />}
                    active={isActive(Services.twitch)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.twitch)}
                />
                <NavItem
                    label="Kick"
                    icon={<KickIcon />}
                    active={isActive(Services.kick)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.kick)}
                />
                <NavItem
                    label="Discord"
                    icon={<RiDiscordFill />}
                    active={isActive(Services.discord)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.discord)}
                />
                <NavItem
                    label="OBS Studio"
                    icon={<RiRecordCircleFill />}
                    active={isActive(Services.obs)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.obs)}
                />
                <NavItem
                    label="VRChat"
                    icon={<RiGamepadFill />}
                    active={isActive(Services.vrc)}
                    collapsed={collapsed}
                    onClick={() => navigate(Services.vrc)}
                />
            </NavGroup>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Settings at bottom */}
            <div className="pt-4 border-t border-base-content/5">
                <NavItem
                    label="Settings"
                    icon={<RiSettings3Fill />}
                    active={isActive('settings')}
                    collapsed={collapsed}
                    shortcut="Ctrl+7"
                    onClick={() => navigate('settings')}
                />
            </div>
        </nav>
    );
});
