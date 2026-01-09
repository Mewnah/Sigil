import { FC, memo } from "react";
import { motion } from "framer-motion";
import { useSnapshot } from "valtio";
import { ServiceNetworkState, InspectorTabPath } from "@/types";
import { Services } from "@/core";
import {
    RiUserVoiceFill,
    RiChatVoiceFill,
    RiTranslate2,
    RiSparklingFill,
    RiPlayCircleFill,
    RiStopCircleFill,
    RiSettings2Fill,
    RiMessage2Fill,
} from "react-icons/ri";
import { SiDiscord, SiTwitch } from "react-icons/si";
import classNames from "classnames";

// Service status card component
interface ServiceCardProps {
    title: string;
    icon: React.ReactNode;
    status: ServiceNetworkState;
    onToggle?: () => void;
    onSettings?: () => void;
}

const StatusDot: FC<{ status: ServiceNetworkState }> = ({ status }) => {
    const colors = {
        [ServiceNetworkState.connected]: "bg-success",
        [ServiceNetworkState.connecting]: "bg-warning animate-pulse",
        [ServiceNetworkState.disconnected]: "bg-base-content/30",
        [ServiceNetworkState.error]: "bg-error",
    };
    return <div className={classNames("w-2 h-2 rounded-full", colors[status])} />;
};

const ServiceCard: FC<ServiceCardProps> = memo(({ title, icon, status, onToggle, onSettings }) => {
    const isConnected = status === ServiceNetworkState.connected;
    const isConnecting = status === ServiceNetworkState.connecting;

    return (
        <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="relative bg-base-200 rounded-xl p-4 flex flex-col gap-3 border border-base-content/5 hover:border-primary/30 transition-colors"
        >
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="text-2xl text-primary">{icon}</div>
                <div className="flex-1">
                    <div className="font-semibold text-sm">{title}</div>
                    <div className="flex items-center gap-1.5">
                        <StatusDot status={status} />
                        <span className="text-xs text-base-content/60">
                            {isConnected ? "Active" : isConnecting ? "Connecting..." : "Inactive"}
                        </span>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
                <button
                    onClick={onToggle}
                    className={classNames(
                        "btn btn-sm flex-1 gap-1",
                        isConnected ? "btn-error btn-outline" : "btn-primary"
                    )}
                >
                    {isConnected ? (
                        <>
                            <RiStopCircleFill /> Stop
                        </>
                    ) : (
                        <>
                            <RiPlayCircleFill /> Start
                        </>
                    )}
                </button>
                <button onClick={onSettings} className="btn btn-sm btn-ghost btn-square" title="Settings">
                    <RiSettings2Fill />
                </button>
            </div>
        </motion.div>
    );
});

// Quick action button component
interface QuickActionProps {
    label: string;
    icon: React.ReactNode;
    onClick: () => void;
    variant?: "primary" | "secondary" | "ghost";
}

const QuickAction: FC<QuickActionProps> = ({ label, icon, onClick, variant = "ghost" }) => (
    <button
        onClick={onClick}
        className={classNames("btn gap-2", {
            "btn-primary": variant === "primary",
            "btn-secondary": variant === "secondary",
            "btn-ghost": variant === "ghost",
        })}
    >
        {icon}
        {label}
    </button>
);

// Kick logo
const KickIcon: FC = () => (
    <svg width={20} height={20} viewBox="0 0 512 512" fill="currentColor">
        <path d="M115.2 0h76.8v204.8L345.6 0H448L268.8 230.4 460.8 512H345.6L192 281.6V512h-76.8V0z" />
    </svg>
);

// Main Dashboard component
const Dashboard: FC = memo(() => {
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const translationState = useSnapshot(window.ApiServer.translation.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);

    const navigateTo = (tab: InspectorTabPath) => {
        window.ApiServer.changeTab(tab);
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="w-full h-full overflow-auto p-4"
        >
            <div className="max-w-4xl mx-auto space-y-8">
                {/* Header */}
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold font-header">
                        Welcome to <span className="text-primary">Sigil</span>
                    </h1>
                    <p className="text-base-content/60">
                        VRChat streaming toolkit for Twitch & Kick
                    </p>
                </div>

                {/* Service Cards Grid */}
                <section className="space-y-3">
                    <h2 className="text-lg font-semibold text-base-content/80">Services</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        <ServiceCard
                            title="Speech to Text"
                            icon={<RiUserVoiceFill />}
                            status={sttState.status}
                            onToggle={() => {
                                if (sttState.status === ServiceNetworkState.connected) window.ApiServer.stt.stop();
                                else if (sttState.status === ServiceNetworkState.disconnected) window.ApiServer.stt.start();
                            }}
                            onSettings={() => navigateTo({ tab: Services.stt })}
                        />
                        <ServiceCard
                            title="AI Transform"
                            icon={<RiSparklingFill />}
                            status={transformState.status}
                            onToggle={() => {
                                if (transformState.status === ServiceNetworkState.connected) window.ApiServer.transform.stop();
                                else if (transformState.status === ServiceNetworkState.disconnected) window.ApiServer.transform.start();
                            }}
                            onSettings={() => navigateTo({ tab: Services.transform })}
                        />
                        <ServiceCard
                            title="Text to Speech"
                            icon={<RiChatVoiceFill />}
                            status={ttsState.status}
                            onToggle={() => {
                                if (ttsState.status === ServiceNetworkState.connected) window.ApiServer.tts.stop();
                                else if (ttsState.status === ServiceNetworkState.disconnected) window.ApiServer.tts.start();
                            }}
                            onSettings={() => navigateTo({ tab: Services.tts })}
                        />
                        <ServiceCard
                            title="Translation"
                            icon={<RiTranslate2 />}
                            status={translationState.status}
                            onToggle={() => {
                                if (translationState.status === ServiceNetworkState.connected) window.ApiServer.translation.stop();
                                else if (translationState.status === ServiceNetworkState.disconnected) window.ApiServer.translation.start();
                            }}
                            onSettings={() => navigateTo({ tab: Services.translation })}
                        />
                    </div>
                </section>

                {/* Integrations */}
                <section className="space-y-3">
                    <h2 className="text-lg font-semibold text-base-content/80">Integrations</h2>
                    <div className="flex flex-wrap gap-3">
                        <QuickAction
                            label="Twitch"
                            icon={<SiTwitch />}
                            onClick={() => navigateTo({ tab: Services.twitch })}
                        />
                        <QuickAction
                            label="Kick"
                            icon={<KickIcon />}
                            onClick={() => navigateTo({ tab: Services.kick })}
                        />
                        <QuickAction
                            label="Discord"
                            icon={<SiDiscord />}
                            onClick={() => navigateTo({ tab: Services.discord })}
                        />
                        <QuickAction
                            label="VRChat"
                            icon={<RiMessage2Fill />}
                            onClick={() => navigateTo({ tab: Services.vrc })}
                        />
                    </div>
                </section>

                {/* Quick Actions */}
                <section className="space-y-3">
                    <h2 className="text-lg font-semibold text-base-content/80">Quick Actions</h2>
                    <div className="flex flex-wrap gap-3">
                        <QuickAction
                            label="Settings"
                            icon={<RiSettings2Fill />}
                            onClick={() => navigateTo({ tab: "settings" })}
                        />
                    </div>
                </section>
            </div>
        </motion.div>
    );
});

export default Dashboard;
