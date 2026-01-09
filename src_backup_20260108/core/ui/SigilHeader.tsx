import { FC, memo } from "react";
import { motion } from "framer-motion";
import { appWindow } from "@tauri-apps/api/window";
import { exit } from "@tauri-apps/api/process";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import { RiSettings2Fill } from "react-icons/ri";
import { VscChromeClose, VscChromeMaximize, VscChromeMinimize } from "react-icons/vsc";
import { ServiceNetworkState } from "@/types";
import Logo from "./logo";
import Tooltip from "./dropdown/Tooltip";

// Status dot for services
const StatusDot: FC<{
    label: string;
    status: ServiceNetworkState;
}> = ({ label, status }) => {
    const colors = {
        [ServiceNetworkState.connected]: "bg-success",
        [ServiceNetworkState.connecting]: "bg-warning animate-pulse",
        [ServiceNetworkState.disconnected]: "bg-base-content/30",
        [ServiceNetworkState.error]: "bg-error",
    };

    const statusText = {
        [ServiceNetworkState.connected]: "Connected",
        [ServiceNetworkState.connecting]: "Connecting...",
        [ServiceNetworkState.disconnected]: "Off",
        [ServiceNetworkState.error]: "Error",
    };

    return (
        <Tooltip content={`${label}: ${statusText[status]}`}>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg hover:bg-base-content/5 transition-colors cursor-default">
                <div className={classNames("w-2 h-2 rounded-full", colors[status])} aria-hidden="true" />
                <span className="text-xs font-medium text-base-content/70">{label}</span>
            </div>
        </Tooltip>
    );
};

// Window controls (minimize, maximize, close)
const WindowControls: FC = memo(() => {
    const handleMinimize = () => appWindow.minimize();
    const handleMaximize = () => appWindow.toggleMaximize();
    const handleClose = () => exit(0);

    return (
        <div className="flex items-center" role="group" aria-label="Window controls">
            <button
                onClick={handleMinimize}
                className="w-10 h-8 flex items-center justify-center text-base-content/50 hover:bg-base-content/10 hover:text-base-content transition-colors"
                aria-label="Minimize window"
            >
                <VscChromeMinimize />
            </button>
            <button
                onClick={handleMaximize}
                className="w-10 h-8 flex items-center justify-center text-base-content/50 hover:bg-base-content/10 hover:text-base-content transition-colors"
                aria-label="Maximize window"
            >
                <VscChromeMaximize />
            </button>
            <button
                onClick={handleClose}
                className="w-10 h-8 flex items-center justify-center text-base-content/50 hover:bg-error hover:text-error-content transition-colors"
                aria-label="Close window"
            >
                <VscChromeClose />
            </button>
        </div>
    );
});

// Main compact header
const SigilHeader: FC = memo(() => {
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);

    const handleOpenSettings = () => {
        window.ApiServer.changeTab({ tab: "settings" });
    };

    return (
        <header
            data-tauri-drag-region
            className="flex-none h-10 flex items-center justify-between bg-base-200 border-b border-base-content/10"
            role="banner"
        >
            {/* Left: Logo + Name */}
            <div data-tauri-drag-region className="flex items-center gap-2 px-3">
                <Logo />
                <span className="text-sm font-bold font-header text-base-content/80">Sigil</span>
            </div>

            {/* Center: Service Status */}
            <div className="flex items-center gap-1" role="status" aria-label="Service status">
                <StatusDot label="STT" status={sttState.status} />
                <StatusDot label="AI" status={transformState.status} />
                <StatusDot label="TTS" status={ttsState.status} />
            </div>

            {/* Right: Settings + Window controls */}
            <div className="flex items-center">
                <Tooltip content="Settings">
                    <button
                        onClick={handleOpenSettings}
                        className="w-10 h-8 flex items-center justify-center text-base-content/50 hover:bg-base-content/10 hover:text-base-content transition-colors"
                        aria-label="Open settings"
                    >
                        <RiSettings2Fill />
                    </button>
                </Tooltip>
                <div className="w-px h-4 bg-base-content/10 mx-1" aria-hidden="true" />
                <WindowControls />
            </div>
        </header>
    );
});

export default SigilHeader;
