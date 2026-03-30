import { FC, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import {
    RiMicFill,
    RiMicOffFill,
    RiVolumeUpFill,
    RiVolumeMuteFill,
    RiSparklingFill,
    RiPlayFill,
    RiStopFill,
    RiSettings3Fill,
} from "react-icons/ri";
import { ServiceNetworkState } from "@/types";
import { Services } from "@/services-registry";
import Tooltip from "./dropdown/Tooltip";
import { SttMuteState } from "../services/stt/types";

interface ServiceCardProps {
    label: string;
    icon: React.ReactNode;
    activeIcon?: React.ReactNode;
    status: ServiceNetworkState;
    muted?: boolean;
    onToggle: () => void;
    onSettings?: () => void;
}

const ServiceCard: FC<ServiceCardProps> = memo(({
    label,
    icon,
    activeIcon,
    status,
    muted,
    onToggle,
    onSettings,
}) => {
    const isConnected = status === ServiceNetworkState.connected;
    const isConnecting = status === ServiceNetworkState.connecting;
    const isError = status === ServiceNetworkState.error;

    const statusColor = isError
        ? "border-error/50 bg-error/10"
        : isConnected
            ? "border-success/50 bg-success/10"
            : isConnecting
                ? "border-warning/50 bg-warning/10"
                : "border-base-content/10 bg-base-100";

    return (
        <div
            className={classNames(
                "flex items-center gap-2 px-3 py-2 rounded-xl border-2 transition-all",
                statusColor
            )}
            role="group"
            aria-label={`${label} controls`}
        >
            {/* Icon */}
            <div className={classNames(
                "text-xl",
                isConnected ? "text-success" : isError ? "text-error" : "text-base-content/50"
            )}>
                {muted ? activeIcon || icon : icon}
            </div>

            {/* Label */}
            <span className="text-sm font-medium min-w-[50px]">{label}</span>

            {/* Toggle Button */}
            <Tooltip content={isConnected ? `Stop ${label}` : `Start ${label}`}>
                <button
                    onClick={onToggle}
                    className={classNames(
                        "btn btn-sm btn-circle",
                        isConnected ? "btn-error" : "btn-success",
                        isConnecting && "loading"
                    )}
                    aria-label={isConnected ? `Stop ${label}` : `Start ${label}`}
                    disabled={isConnecting}
                >
                    {!isConnecting && (isConnected ? <RiStopFill /> : <RiPlayFill />)}
                </button>
            </Tooltip>

            {/* Settings Button */}
            {onSettings && (
                <Tooltip content={`${label} Settings`}>
                    <button
                        onClick={onSettings}
                        className="btn btn-sm btn-ghost btn-circle text-base-content/50 hover:text-base-content"
                        aria-label={`${label} settings`}
                    >
                        <RiSettings3Fill />
                    </button>
                </Tooltip>
            )}
        </div>
    );
});

// Main Control Bar
const ControlBar: FC = memo(() => {
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

    const handleToggleTransform = () => {
        if (transformState.status === ServiceNetworkState.connected) {
            window.ApiServer.transform.stop();
        } else {
            window.ApiServer.transform.start();
        }
    };

    const openSettings = (tab: string) => {
        window.ApiServer.changeTab({ tab: tab as any });
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex-none px-4 py-3 bg-base-200 border-t border-base-content/10"
            role="toolbar"
            aria-label="Service controls"
        >
            <div className="flex items-center justify-center gap-3 flex-wrap">
                <ServiceCard
                    label="STT"
                    icon={<RiMicFill />}
                    activeIcon={<RiMicOffFill />}
                    status={sttState.status}
                    muted={sttState.muted === SttMuteState.muted}
                    onToggle={handleToggleSTT}
                    onSettings={() => openSettings(Services.stt)}
                />

                <ServiceCard
                    label="AI"
                    icon={<RiSparklingFill />}
                    status={transformState.status}
                    onToggle={handleToggleTransform}
                    onSettings={() => openSettings(Services.transform)}
                />

                <ServiceCard
                    label="TTS"
                    icon={<RiVolumeUpFill />}
                    activeIcon={<RiVolumeMuteFill />}
                    status={ttsState.status}
                    onToggle={handleToggleTTS}
                    onSettings={() => openSettings(Services.tts)}
                />
            </div>
        </motion.div>
    );
});

export default ControlBar;
