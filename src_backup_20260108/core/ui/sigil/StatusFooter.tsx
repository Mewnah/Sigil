import { FC, memo } from "react";
import { useSnapshot } from "valtio";
import { ServiceNetworkState } from "@/types";
import { RiMicFill, RiVolumeUpFill, RiSparklingFill, RiTranslate2, RiSettings3Line, RiTwitchFill, RiDiscordFill, RiGamepadFill, RiRecordCircleFill, RiPlayFill, RiStopFill } from "react-icons/ri";
import { KickIcon } from "../icons/KickIcon";
import { Services } from "@/core";

interface StatusCardProps {
    label: string;
    icon: React.ReactNode;
    status: ServiceNetworkState;
    serviceId: string;
    onToggle?: () => void;
    canToggle?: boolean;
}

const StatusCard: FC<StatusCardProps> = ({ label, icon, status, serviceId, onToggle, canToggle = true }) => {
    const isConn = status === ServiceNetworkState.connected;
    const isConnecting = status === ServiceNetworkState.connecting;

    const handleOpenSettings = (e: React.MouseEvent) => {
        e.stopPropagation();
        window.ApiServer.changeTab({ tab: serviceId as any });
    };

    const handleToggle = (e: React.MouseEvent) => {
        e.stopPropagation();
        onToggle?.();
    };

    return (
        <div className="bg-base-300 border border-base-content/10 p-2 rounded-lg flex items-center gap-2 transition-all group h-14 flex-shrink-0 min-w-[200px]">
            {/* Icon + Info - clickable to open settings */}
            <button
                onClick={handleOpenSettings}
                className="flex items-center gap-3 flex-1 min-w-0 text-left hover:opacity-80 transition-opacity"
            >
                <div className={`p-1.5 rounded-md transition-colors flex-shrink-0 ${isConn ? 'bg-success/10 text-success' : 'bg-base-content/5 text-base-content/40'}`}>
                    {icon}
                </div>
                <div className="flex-1 min-w-0 overflow-hidden">
                    <div className="font-medium text-xs text-base-content/90 truncate">{label}</div>
                    <div className="text-xs text-base-content/40 font-mono uppercase tracking-wide">
                        {isConnecting ? 'CONNECTING...' : isConn ? 'ONLINE' : 'OFFLINE'}
                    </div>
                </div>
            </button>

            {/* Action buttons */}
            <div className="flex items-center gap-1 flex-shrink-0">
                {/* Toggle button */}
                {canToggle && (
                    <button
                        onClick={handleToggle}
                        disabled={isConnecting}
                        className={`p-2 rounded-md transition-all ${isConnecting
                            ? 'bg-primary/20 text-primary animate-pulse cursor-wait'
                            : isConn
                                ? 'bg-error/10 text-error hover:bg-error/20'
                                : 'bg-success/10 text-success hover:bg-success/20'
                            }`}
                        title={isConn ? 'Stop service' : 'Start service'}
                    >
                        {isConnecting ? <RiPlayFill className="animate-spin" /> : isConn ? <RiStopFill /> : <RiPlayFill />}
                    </button>
                )}

                {/* Settings button */}
                <button
                    onClick={handleOpenSettings}
                    className="p-2 rounded-md bg-base-content/5 text-base-content/40 hover:bg-primary/10 hover:text-primary transition-all"
                    title="Open settings"
                >
                    <RiSettings3Line />
                </button>
            </div>
        </div>
    );
}

export const StatusFooter: FC = memo(() => {
    const services = useSnapshot(window.ApiServer.state.services);

    // Use reactive snapshots for each service's status
    // Core services with serviceState
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);
    const translationState = useSnapshot(window.ApiServer.translation.serviceState);

    // Integration services with different state structures
    const twitchState = useSnapshot(window.ApiServer.twitch.state);
    const kickState = useSnapshot(window.ApiServer.kick.state);
    const obsState = useSnapshot(window.ApiServer.obs.wsState);
    // VRC and Discord don't have connection status - always show as "ready"

    // Toggle handlers for core services
    const toggleService = (service: string) => {
        const srv = (window.ApiServer as any)[service];
        if (!srv) return;

        if (srv.serviceState?.status === ServiceNetworkState.connected) {
            srv.stop?.();
        } else if (srv.serviceState?.status === ServiceNetworkState.disconnected) {
            srv.start?.();
        }
    };

    const CARDS = [
        { id: Services.stt, label: "Speech to Text", icon: <RiMicFill />, show: services.stt.showActionButton, status: sttState.status, onToggle: () => toggleService('stt'), canToggle: true },
        { id: Services.tts, label: "Text to Speech", icon: <RiVolumeUpFill />, show: services.tts.showActionButton, status: ttsState.status, onToggle: () => toggleService('tts'), canToggle: true },
        { id: Services.translation, label: "Translation", icon: <RiTranslate2 />, show: services.translation.showActionButton, status: translationState.status, onToggle: () => toggleService('translation'), canToggle: true },
        { id: Services.transform, label: "AI Transform", icon: <RiSparklingFill />, show: services.transform.showActionButton, status: transformState.status, onToggle: () => toggleService('transform'), canToggle: true },
        { id: Services.twitch, label: "Twitch", icon: <RiTwitchFill />, show: services.twitch.showActionButton, status: twitchState.user ? ServiceNetworkState.connected : ServiceNetworkState.disconnected, canToggle: false },
        { id: Services.kick, label: "Kick", icon: <KickIcon />, show: services.kick.showActionButton, status: kickState.user ? ServiceNetworkState.connected : ServiceNetworkState.disconnected, canToggle: false },
        { id: Services.discord, label: "Discord", icon: <RiDiscordFill />, show: services.discord.showActionButton, status: services.discord.data.channelHook ? ServiceNetworkState.connected : ServiceNetworkState.disconnected, canToggle: false },
        { id: Services.vrc, label: "VRChat", icon: <RiGamepadFill />, show: services.vrc.showActionButton, status: services.vrc.data.enable ? ServiceNetworkState.connected : ServiceNetworkState.disconnected, canToggle: false },
        {
            id: 'obs', label: "OBS Studio", icon: <RiRecordCircleFill />, show: services.obs.showActionButton, status: obsState.status, onToggle: () => {
                if (obsState.status === ServiceNetworkState.connected) {
                    window.ApiServer.obs.wsDisconnect();
                } else {
                    window.ApiServer.obs.wsConnect();
                }
            }, canToggle: true
        },
    ];

    const activeCards = CARDS.filter(c => c.show);

    if (activeCards.length === 0) return null;

    return (
        <div className="p-2 bg-base-200 border-t border-base-content/5 overflow-x-auto">
            <div className="flex gap-2 justify-center" style={{ minWidth: 'max-content' }}>
                {activeCards.map(card => (
                    <StatusCard
                        key={card.id}
                        label={card.label}
                        icon={card.icon}
                        status={card.status}
                        serviceId={card.id}
                        onToggle={card.onToggle}
                        canToggle={card.canToggle}
                    />
                ))}
            </div>
        </div>
    );
});
