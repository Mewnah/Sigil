import { FC, memo } from "react";
import { motion } from "framer-motion";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import {
    RiMicFill,
    RiVolumeUpFill,
    RiSparklingFill,
    RiTwitchFill,
    RiDiscordFill,
    RiSettings3Fill,
} from "react-icons/ri";
import { SiObsstudio } from "react-icons/si";
import { ServiceNetworkState } from "@/types";
import Tooltip from "./dropdown/Tooltip";

interface DockIconProps {
    icon: React.ReactNode;
    label: string;
    active?: boolean;
    status?: ServiceNetworkState;
    onClick: () => void;
}

const DockIcon: FC<DockIconProps> = memo(({ icon, label, active, status, onClick }) => {
    const statusColor = status === ServiceNetworkState.connected
        ? "bg-success"
        : status === ServiceNetworkState.connecting
            ? "bg-warning"
            : status === ServiceNetworkState.error
                ? "bg-error"
                : "bg-base-content/20";

    return (
        <Tooltip content={label} placement="right">
            <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={onClick}
                className={classNames(
                    "relative w-14 h-14 flex items-center justify-center rounded-xl transition-all",
                    active
                        ? "bg-primary text-primary-content shadow-lg"
                        : "bg-base-200 text-base-content/70 hover:bg-base-300 hover:text-base-content"
                )}
                aria-label={label}
            >
                <div className="text-2xl">{icon}</div>
                {status !== undefined && (
                    <div
                        className={classNames(
                            "absolute bottom-1 right-1 w-3 h-3 rounded-full border-2 border-base-200",
                            statusColor
                        )}
                        aria-hidden="true"
                    />
                )}
            </motion.button>
        </Tooltip>
    );
});

const DockDivider: FC = () => (
    <div className="w-10 h-px bg-base-content/10 mx-2" aria-hidden="true" />
);

// Main Quick Action Dock
const QuickActionDock: FC = memo(() => {
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);

    const handleToggleSTT = () => {
        if (sttState.status === ServiceNetworkState.connected) {
            window.ApiServer.stt.stop();
        } else {
            window.ApiServer.stt.start();
        }
    };

    const handleToggleTTS = () => {
        if (ttsState.status === ServiceNetworkState.connected) {
            window.ApiServer.tts.stop();
        } else {
            window.ApiServer.tts.start();
        }
    };

    const handleToggleAI = () => {
        if (transformState.status === ServiceNetworkState.connected) {
            window.ApiServer.transform.stop();
        } else {
            window.ApiServer.transform.start();
        }
    };

    const handleOpenSettings = () => {
        window.ApiServer.changeTab({ tab: "settings" });
    };

    return (
        <motion.aside
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="flex-none w-20 h-full bg-transparent flex flex-col items-center py-4 gap-3"
            role="toolbar"
            aria-label="Quick actions"
        >
            {/* Services */}
            <DockIcon
                icon={<RiMicFill />}
                label="Speech-to-Text"
                active={sttState.status === ServiceNetworkState.connected}
                status={sttState.status}
                onClick={handleToggleSTT}
            />
            <DockIcon
                icon={<RiSparklingFill />}
                label="AI Transform"
                active={transformState.status === ServiceNetworkState.connected}
                status={transformState.status}
                onClick={handleToggleAI}
            />
            <DockIcon
                icon={<RiVolumeUpFill />}
                label="Text-to-Speech"
                active={ttsState.status === ServiceNetworkState.connected}
                status={ttsState.status}
                onClick={handleToggleTTS}
            />

            <DockDivider />

            {/* Integrations (placeholder - can expand later) */}
            <DockIcon
                icon={<RiTwitchFill />}
                label="Twitch (Coming Soon)"
                onClick={() => { }}
            />
            <DockIcon
                icon={<RiDiscordFill />}
                label="Discord (Coming Soon)"
                onClick={() => { }}
            />
            <DockIcon
                icon={<SiObsstudio />}
                label="OBS (Coming Soon)"
                onClick={() => { }}
            />

            {/* Spacer */}
            <div className="flex-1" />

            {/* Settings at bottom */}
            <DockIcon
                icon={<RiSettings3Fill />}
                label="Settings"
                onClick={handleOpenSettings}
            />
        </motion.aside>
    );
});

export default QuickActionDock;
