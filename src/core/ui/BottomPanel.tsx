import { FC, memo } from "react";
import { useSnapshot } from "valtio";
import classNames from "classnames";
import {
    RiMicFill,
    RiVolumeUpFill,
    RiSparklingFill,
    RiSettings3Fill,
    RiPlayFill,
    RiStopFill
} from "react-icons/ri";
import { ServiceNetworkState } from "@/types";
import { Services } from "@/core";

const ServiceCard: FC<{
    label: string;
    icon: React.ReactNode;
    status: ServiceNetworkState;
    onToggle: () => void;
    onSettings: () => void;
}> = memo(({ label, icon, status, onToggle, onSettings }) => {
    const isConnected = status === ServiceNetworkState.connected;
    const isConnecting = status === ServiceNetworkState.connecting;
    const isError = status === ServiceNetworkState.error;

    const statusColor = isConnected
        ? "text-success"
        : isConnecting
            ? "text-warning"
            : isError
                ? "text-error"
                : "text-base-content/30";

    return (
        <div className="flex items-center gap-3 bg-base-100/50 hover:bg-base-100 border border-base-content/5 hover:border-base-content/10 transition-all rounded-lg p-2 pr-4 min-w-[200px]">
            {/* Icon Box */}
            <div className={classNames("w-10 h-10 rounded flex items-center justify-center text-xl bg-base-200", statusColor)}>
                {icon}
            </div>

            {/* Controls */}
            <div className="flex flex-col flex-1 gap-1">
                <div className="text-[10px] uppercase font-bold tracking-wider opacity-50 leading-none">{label}</div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={onToggle}
                        className={classNames(
                            "btn btn-xs flex-1 gap-1",
                            isConnected ? "btn-error btn-outline" : "btn-primary"
                        )}
                    >
                        {isConnected ? <><RiStopFill /> Stop</> : <><RiPlayFill /> Start</>}
                    </button>
                    <button
                        onClick={onSettings}
                        className="btn btn-xs btn-square btn-ghost"
                        title="Settings"
                    >
                        <RiSettings3Fill />
                    </button>
                </div>
            </div>
        </div>
    );
});

export const BottomPanel: FC = memo(() => {
    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);

    const toggleSTT = () => sttState.status === ServiceNetworkState.connected ? window.ApiServer.stt.stop() : window.ApiServer.stt.start();
    const toggleTTS = () => ttsState.status === ServiceNetworkState.connected ? window.ApiServer.tts.stop() : window.ApiServer.tts.start();
    const toggleTransform = () => transformState.status === ServiceNetworkState.connected ? window.ApiServer.transform.stop() : window.ApiServer.transform.start();

    const openSettings = (tab: any) => window.ApiServer.changeTab({ tab });

    return (
        <div className="h-16 flex-none bg-base-200 border-t border-base-content/5 flex items-center px-4 gap-4 overflow-x-auto">
            <ServiceCard
                label="Speech to Text"
                icon={<RiMicFill />}
                status={sttState.status}
                onToggle={toggleSTT}
                onSettings={() => openSettings(Services.stt)}
            />
            <ServiceCard
                label="AI Transform"
                icon={<RiSparklingFill />}
                status={transformState.status}
                onToggle={toggleTransform}
                onSettings={() => openSettings(Services.transform)}
            />
            <ServiceCard
                label="Text to Speech"
                icon={<RiVolumeUpFill />}
                status={ttsState.status}
                onToggle={toggleTTS}
                onSettings={() => openSettings(Services.tts)}
            />
        </div>
    );
});
