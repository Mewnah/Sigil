import { FC, memo, useEffect, useRef, useState } from "react";
import { useSnapshot } from "valtio";
import { toast } from "react-toastify";
import { useTranslation } from "react-i18next";
import { ServiceNetworkState } from "@/types";
import {
    RiMicFill,
    RiVolumeUpFill,
    RiSparklingFill,
    RiTranslate2,
    RiSettings3Line,
    RiTwitchFill,
    RiDiscordFill,
    RiGamepadFill,
    RiRecordCircleFill,
    RiPlayFill,
    RiStopFill,
} from "react-icons/ri";
import { KickIcon } from "../icons/KickIcon";
import { Services } from "@/services-registry";
import { DEFAULT_ACTION_BAR_SERVICE_ORDER } from "@/core/schema";
import { TTS_Backends } from "@/core/services/tts/schema";

const ALL_ACTION_BAR_IDS = new Set<string>(DEFAULT_ACTION_BAR_SERVICE_ORDER);

function normalizeActionBarOrder(saved: readonly string[]): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const id of saved) {
        if (ALL_ACTION_BAR_IDS.has(id) && !seen.has(id)) {
            out.push(id);
            seen.add(id);
        }
    }
    for (const id of DEFAULT_ACTION_BAR_SERVICE_ORDER) {
        if (!seen.has(id)) {
            out.push(id);
            seen.add(id);
        }
    }
    return out;
}

function reorderBefore(fullOrder: string[], draggedId: string, targetId: string): string[] {
    if (draggedId === targetId) return fullOrder;
    const next = fullOrder.filter((id) => id !== draggedId);
    const ti = next.indexOf(targetId);
    if (ti < 0) return fullOrder;
    next.splice(ti, 0, draggedId);
    return next;
}

function reorderAfter(fullOrder: string[], draggedId: string, afterId: string): string[] {
    if (draggedId === afterId) return fullOrder;
    const next = fullOrder.filter((id) => id !== draggedId);
    const ai = next.indexOf(afterId);
    if (ai < 0) return fullOrder;
    next.splice(ai + 1, 0, draggedId);
    return next;
}

/** Apply drop at `insertAt` within the visible id list (0 = before first visible, n = after last). */
function applyFooterReorder(
    fullOrder: string[],
    draggedId: string,
    insertAt: number,
    visibleIds: string[]
): string[] {
    const others = visibleIds.filter((id) => id !== draggedId);
    if (others.length === 0) return normalizeActionBarOrder(fullOrder);
    const normalized = normalizeActionBarOrder(fullOrder);
    if (insertAt <= 0) return reorderBefore(normalized, draggedId, others[0]);
    if (insertAt >= others.length) return reorderAfter(normalized, draggedId, others[others.length - 1]);
    return reorderBefore(normalized, draggedId, others[insertAt]);
}

function pointerToInsert(clientX: number, rowEl: HTMLElement, activeIds: string[]): { insertIndex: number; lineLeftPx: number } {
    const rowRect = rowEl.getBoundingClientRect();
    let insertIndex = activeIds.length;
    for (let i = 0; i < activeIds.length; i++) {
        const el = rowEl.querySelector(`[data-action-bar-card="${activeIds[i]}"]`) as HTMLElement | null;
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (clientX < r.left + r.width / 2) {
            insertIndex = i;
            break;
        }
    }

    let lineLeftPx: number;
    if (activeIds.length === 0) {
        lineLeftPx = 8;
    } else if (insertIndex === 0) {
        const first = rowEl.querySelector(`[data-action-bar-card="${activeIds[0]}"]`) as HTMLElement;
        const r = first.getBoundingClientRect();
        lineLeftPx = r.left - rowRect.left - 6;
    } else if (insertIndex >= activeIds.length) {
        const last = rowEl.querySelector(`[data-action-bar-card="${activeIds[activeIds.length - 1]}"]`) as HTMLElement;
        const r = last.getBoundingClientRect();
        lineLeftPx = r.right - rowRect.left + 6;
    } else {
        const prev = rowEl.querySelector(`[data-action-bar-card="${activeIds[insertIndex - 1]}"]`) as HTMLElement;
        const next = rowEl.querySelector(`[data-action-bar-card="${activeIds[insertIndex]}"]`) as HTMLElement;
        const pr = prev.getBoundingClientRect();
        const nr = next.getBoundingClientRect();
        lineLeftPx = (pr.right + nr.left) / 2 - rowRect.left;
    }

    return { insertIndex, lineLeftPx: Math.max(0, lineLeftPx) };
}

/** Minimal 2×3 dot grip — avoids loud directional drag icons. */
const ActionBarGrip: FC = () => (
    <span className="flex flex-col justify-center gap-[3px] py-0.5 px-1" aria-hidden>
        {[0, 1, 2].map((row) => (
            <span key={row} className="flex gap-[3px] justify-center">
                <span className="w-[3px] h-[3px] rounded-full bg-base-content/25" />
                <span className="w-[3px] h-[3px] rounded-full bg-base-content/25" />
            </span>
        ))}
    </span>
);

const ActionBarDragGhost: FC<{ label: string; icon: React.ReactNode; x: number; y: number; hint: string }> = ({
    label,
    icon,
    x,
    y,
    hint,
}) => (
    <div
        className="fixed z-[100] pointer-events-none flex flex-col gap-0.5 px-3 py-2 w-[min(240px,calc(100vw-32px))] rounded-lg bg-base-300/95 backdrop-blur-sm border-2 border-primary shadow-2xl shadow-primary/15"
        style={{ left: x + 14, top: y + 14, transform: "rotate(1.5deg)" }}
        role="status"
        aria-live="polite"
    >
        <div className="flex items-center gap-2 min-w-0">
            <div className="p-1.5 rounded-md bg-primary/15 text-primary flex-shrink-0">{icon}</div>
            <div className="min-w-0 flex-1">
                <div className="font-semibold text-xs text-base-content truncate">{label}</div>
                <div className="text-[10px] text-base-content/50 leading-tight">{hint}</div>
            </div>
        </div>
    </div>
);

const InsertionMarker: FC<{ leftPx: number }> = ({ leftPx }) => (
    <div
        className="absolute top-1/2 -translate-y-1/2 w-1 rounded-full bg-primary pointer-events-none z-20 h-11 shadow-lg shadow-primary/50 ring-2 ring-primary/20 transition-[left] duration-75 ease-out"
        style={{ left: leftPx, marginLeft: -2 }}
        aria-hidden
    />
);

type ActionBarCardDef = {
    id: string;
    label: string;
    icon: React.ReactNode;
    show: boolean;
    status: ServiceNetworkState;
    canToggle: boolean;
    onToggle?: () => void;
};

interface StatusCardProps {
    label: string;
    icon: React.ReactNode;
    status: ServiceNetworkState;
    serviceId: string;
    onToggle?: () => void;
    canToggle?: boolean;
    reorderEnabled?: boolean;
    onGripPointerDown?: (serviceId: string, e: React.PointerEvent) => void;
    isReorderSource?: boolean;
}

const StatusCard: FC<StatusCardProps> = ({
    label,
    icon,
    status,
    serviceId,
    onToggle,
    canToggle = true,
    reorderEnabled,
    onGripPointerDown,
    isReorderSource,
}) => {
    const { t } = useTranslation();
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

    const handleGripDown = (e: React.PointerEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.button !== 0) return;
        onGripPointerDown?.(serviceId, e);
    };

    return (
        <div
            data-action-bar-card={serviceId}
            className={`bg-base-300 border p-2 rounded-lg flex items-center gap-2 transition-[opacity,transform,box-shadow] duration-150 group h-14 flex-shrink-0 min-w-[200px] border-base-content/10 ${
                isReorderSource
                    ? "opacity-40 scale-[0.97] shadow-inner ring-2 ring-dashed ring-primary/35 border-primary/20"
                    : ""
            }`}
        >
            {reorderEnabled && (
                <button
                    type="button"
                    onPointerDown={handleGripDown}
                    className="cursor-grab active:cursor-grabbing flex-shrink-0 touch-none rounded hover:bg-base-content/5 outline-none focus-visible:ring-1 focus-visible:ring-primary/50 [touch-action:none]"
                    title={t("common.drag_reorder_action_bar")}
                    aria-label={t("common.drag_reorder_action_bar")}
                >
                    <ActionBarGrip />
                </button>
            )}
            <button
                onClick={handleOpenSettings}
                className="flex items-center gap-3 flex-1 min-w-0 text-left hover:opacity-80 transition-opacity"
            >
                <div
                    className={`p-1.5 rounded-md transition-colors flex-shrink-0 ${isConn ? "bg-success/10 text-success" : "bg-base-content/5 text-base-content/40"}`}
                >
                    {icon}
                </div>
                <div className="flex-1 min-w-0 overflow-hidden">
                    <div className="font-medium text-xs text-base-content/90 truncate">{label}</div>
                    <div className="text-xs text-base-content/40 font-mono uppercase tracking-wide">
                        {isConnecting ? "CONNECTING..." : isConn ? "ONLINE" : "OFFLINE"}
                    </div>
                </div>
            </button>

            <div className="flex items-center gap-1 flex-shrink-0">
                {canToggle && (
                    <button
                        onClick={handleToggle}
                        disabled={isConnecting}
                        className={`p-2 rounded-md transition-all ${
                            isConnecting
                                ? "bg-primary/20 text-primary animate-pulse cursor-wait"
                                : isConn
                                  ? "bg-error/10 text-error hover:bg-error/20"
                                  : "bg-success/10 text-success hover:bg-success/20"
                        }`}
                        title={isConn ? "Stop service" : "Start service"}
                    >
                        {isConnecting ? <RiPlayFill className="animate-spin" /> : isConn ? <RiStopFill /> : <RiPlayFill />}
                    </button>
                )}

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
};

export const StatusFooter: FC = memo(() => {
    const { t } = useTranslation();
    const services = useSnapshot(window.ApiServer.state.services);
    const actionBarOrderSnap = useSnapshot(window.ApiServer.state.actionBarServiceOrder);

    const sttState = useSnapshot(window.ApiServer.stt.serviceState);
    const ttsState = useSnapshot(window.ApiServer.tts.serviceState);
    const transformState = useSnapshot(window.ApiServer.transform.serviceState);
    const translationState = useSnapshot(window.ApiServer.translation.serviceState);

    const twitchState = useSnapshot(window.ApiServer.twitch.state);
    const kickState = useSnapshot(window.ApiServer.kick.state);
    const obsState = useSnapshot(window.ApiServer.obs.wsState);

    const fullOrder = normalizeActionBarOrder(actionBarOrderSnap as string[]);

    const toggleService = (service: string) => {
        const srv = (window.ApiServer as any)[service];
        if (!srv) return;

        if (service === "tts") {
            if (srv.serviceState?.status === ServiceNetworkState.connected) {
                srv.stop?.();
                return;
            }
            if (srv.serviceState?.status !== ServiceNetworkState.disconnected) return;
            const tts = window.ApiServer.state.services.tts.data;
            if (tts.backend === TTS_Backends.native && !tts.native.voice?.trim()) {
                toast.info(t("toasts.tts_choose_voice"), { autoClose: 6000 });
                window.ApiServer.changeTab({ tab: Services.tts });
                return;
            }
            if (tts.backend === TTS_Backends.windows && (!tts.windows.voice?.trim() || !tts.windows.device?.trim())) {
                toast.info(t("toasts.tts_choose_output"), { autoClose: 6000 });
                window.ApiServer.changeTab({ tab: Services.tts });
                return;
            }
            srv.start?.();
            return;
        }

        if (srv.serviceState?.status === ServiceNetworkState.connected) {
            srv.stop?.();
        } else if (srv.serviceState?.status === ServiceNetworkState.disconnected) {
            srv.start?.();
        }
    };

    const CARDS: ActionBarCardDef[] = [
        {
            id: Services.stt,
            label: "Speech to Text",
            icon: <RiMicFill />,
            show: services.stt.showActionButton,
            status: sttState.status,
            onToggle: () => toggleService("stt"),
            canToggle: true,
        },
        {
            id: Services.tts,
            label: "Text to Speech",
            icon: <RiVolumeUpFill />,
            show: services.tts.showActionButton,
            status: ttsState.status,
            onToggle: () => toggleService("tts"),
            canToggle: true,
        },
        {
            id: Services.translation,
            label: "Translation",
            icon: <RiTranslate2 />,
            show: services.translation.showActionButton,
            status: translationState.status,
            onToggle: () => toggleService("translation"),
            canToggle: true,
        },
        {
            id: Services.transform,
            label: "AI Transform",
            icon: <RiSparklingFill />,
            show: services.transform.showActionButton,
            status: transformState.status,
            onToggle: () => toggleService("transform"),
            canToggle: true,
        },
        {
            id: Services.twitch,
            label: "Twitch",
            icon: <RiTwitchFill />,
            show: services.twitch.showActionButton,
            status: twitchState.user ? ServiceNetworkState.connected : ServiceNetworkState.disconnected,
            canToggle: false,
        },
        {
            id: Services.kick,
            label: "Kick",
            icon: <KickIcon />,
            show: services.kick.showActionButton,
            status: kickState.user ? ServiceNetworkState.connected : ServiceNetworkState.disconnected,
            canToggle: false,
        },
        {
            id: Services.discord,
            label: "Discord",
            icon: <RiDiscordFill />,
            show: services.discord.showActionButton,
            status: services.discord.data.channelHook ? ServiceNetworkState.connected : ServiceNetworkState.disconnected,
            canToggle: false,
        },
        {
            id: Services.vrc,
            label: "VRChat",
            icon: <RiGamepadFill />,
            show: services.vrc.showActionButton,
            status: services.vrc.data.enable ? ServiceNetworkState.connected : ServiceNetworkState.disconnected,
            canToggle: false,
        },
        {
            id: Services.obs,
            label: "OBS Studio",
            icon: <RiRecordCircleFill />,
            show: services.obs.showActionButton,
            status: obsState.status,
            onToggle: () => {
                if (obsState.status === ServiceNetworkState.connected) {
                    window.ApiServer.obs.wsDisconnect();
                } else {
                    window.ApiServer.obs.wsConnect();
                }
            },
            canToggle: true,
        },
    ];

    const cardById = Object.fromEntries(CARDS.map((c) => [c.id, c])) as Record<string, ActionBarCardDef>;

    const activeSorted: ActionBarCardDef[] = [];
    for (const id of fullOrder) {
        const c = cardById[id];
        if (c?.show) activeSorted.push(c);
    }
    for (const c of CARDS) {
        if (c.show && !activeSorted.some((x) => x.id === c.id)) activeSorted.push(c);
    }

    const rowRef = useRef<HTMLDivElement>(null);
    const activeSortedRef = useRef(activeSorted);
    activeSortedRef.current = activeSorted;
    const insertIndexRef = useRef(0);

    const [reorderDraggingId, setReorderDraggingId] = useState<string | null>(null);
    const [insertUi, setInsertUi] = useState<{ insertIndex: number; lineLeftPx: number }>({ insertIndex: 0, lineLeftPx: 0 });
    const [ghostPos, setGhostPos] = useState<{ x: number; y: number } | null>(null);
    const draggingRef = useRef<string | null>(null);

    useEffect(() => {
        if (!reorderDraggingId) return;
        draggingRef.current = reorderDraggingId;

        const onMove = (e: PointerEvent) => {
            const row = rowRef.current;
            if (!row) return;
            const ids = activeSortedRef.current.map((c) => c.id);
            const { insertIndex, lineLeftPx } = pointerToInsert(e.clientX, row, ids);
            insertIndexRef.current = insertIndex;
            setInsertUi({ insertIndex, lineLeftPx });
            setGhostPos({ x: e.clientX, y: e.clientY });
        };

        const onEnd = () => {
            const dragged = draggingRef.current;
            if (dragged) {
                const visibleIds = activeSortedRef.current.map((c) => c.id);
                const insertAt = insertIndexRef.current;
                const prev = normalizeActionBarOrder(window.ApiServer.state.actionBarServiceOrder as string[]);
                const next = applyFooterReorder(prev, dragged, insertAt, visibleIds);
                if (JSON.stringify(prev) !== JSON.stringify(next)) {
                    window.ApiServer.state.actionBarServiceOrder = next;
                }
            }
            draggingRef.current = null;
            setReorderDraggingId(null);
            setGhostPos(null);
        };

        document.body.style.userSelect = "none";
        document.body.style.cursor = "grabbing";
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onEnd, true);
        window.addEventListener("pointercancel", onEnd, true);

        return () => {
            document.body.style.userSelect = "";
            document.body.style.cursor = "";
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onEnd, true);
            window.removeEventListener("pointercancel", onEnd, true);
        };
    }, [reorderDraggingId]);

    const onGripPointerDown = (id: string, e: React.PointerEvent) => {
        const row = rowRef.current;
        if (row) {
            const ids = activeSortedRef.current.map((c) => c.id);
            const { insertIndex, lineLeftPx } = pointerToInsert(e.clientX, row, ids);
            insertIndexRef.current = insertIndex;
            setInsertUi({ insertIndex, lineLeftPx });
        }
        setGhostPos({ x: e.clientX, y: e.clientY });
        setReorderDraggingId(id);
    };

    if (activeSorted.length === 0) return null;

    const reorderEnabled = activeSorted.length > 1;

    const ghostCard = reorderDraggingId ? cardById[reorderDraggingId] : null;

    return (
        <div className="flex-none min-h-16 bg-base-200 border-t border-base-content/5 flex items-center px-3 py-2 overflow-x-auto">
            <div
                ref={rowRef}
                className="relative flex gap-2 justify-center w-full"
                style={{ minWidth: "max-content" }}
            >
                {activeSorted.map((card) => (
                    <StatusCard
                        key={card.id}
                        label={card.label}
                        icon={card.icon}
                        status={card.status}
                        serviceId={card.id}
                        onToggle={card.onToggle}
                        canToggle={card.canToggle}
                        reorderEnabled={reorderEnabled}
                        onGripPointerDown={onGripPointerDown}
                        isReorderSource={reorderDraggingId === card.id}
                    />
                ))}
                {reorderDraggingId && activeSorted.length > 0 && (
                    <InsertionMarker leftPx={insertUi.lineLeftPx} />
                )}
            </div>
            {ghostCard && ghostPos && (
                <ActionBarDragGhost
                    label={ghostCard.label}
                    icon={ghostCard.icon}
                    x={ghostPos.x}
                    y={ghostPos.y}
                    hint={t("common.action_bar_drag_ghost_hint")}
                />
            )}
        </div>
    );
});
