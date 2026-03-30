import hotkeys from "hotkeys-js";
import { ServiceNetworkState } from "@/types";
import { canvasToolbarState } from "./CanvasToolbar";

function isEditableEventTarget(e: KeyboardEvent): boolean {
    const t = e.target as HTMLElement | null;
    if (!t) return false;
    if (t.isContentEditable) return true;
    const tag = t.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/**
 * Global keyboard shortcuts for Sigil UI
 * Separate from recording shortcuts in keyboard service
 */
export const initializeUIShortcuts = () => {
    // Ctrl/Cmd + I: Toggle Stats Panel
    hotkeys("ctrl+i,cmd+i", (e) => {
        e.preventDefault();
        const event = new CustomEvent("sigil:toggle-stats");
        window.dispatchEvent(event);
    });

    // Ctrl/Cmd + ,: Open Settings
    hotkeys("ctrl+comma,cmd+comma", (e) => {
        e.preventDefault();
        window.ApiServer.changeTab({ tab: "settings" });
    });

    // Ctrl/Cmd + 1/2/3: Toggle STT/AI/TTS
    hotkeys("ctrl+1,cmd+1", (e) => {
        e.preventDefault();
        const stt = window.ApiServer.stt;
        if (stt.serviceState.status === ServiceNetworkState.connected) {
            stt.stop();
        } else {
            stt.start();
        }
    });

    hotkeys("ctrl+2,cmd+2", (e) => {
        e.preventDefault();
        const ai = window.ApiServer.transform;
        if (ai.serviceState.status === ServiceNetworkState.connected) {
            ai.stop();
        } else {
            ai.start();
        }
    });

    hotkeys("ctrl+3,cmd+3", (e) => {
        e.preventDefault();
        const tts = window.ApiServer.tts;
        if (tts.serviceState.status === ServiceNetworkState.connected) {
            tts.stop();
        } else {
            tts.start();
        }
    });

    hotkeys("ctrl+shift+z,cmd+shift+z", (e) => {
        if (isEditableEventTarget(e)) return;
        e.preventDefault();
        window.ApiClient.document.redo();
    });

    hotkeys("ctrl+y,cmd+y", (e) => {
        if (isEditableEventTarget(e)) return;
        e.preventDefault();
        window.ApiClient.document.redo();
    });

    hotkeys("ctrl+z,cmd+z", (e) => {
        if (e.shiftKey || isEditableEventTarget(e)) return;
        e.preventDefault();
        window.ApiClient.document.undo();
    });

    hotkeys("f11", (e) => {
        if (isEditableEventTarget(e)) return;
        e.preventDefault();
        canvasToolbarState.presentationMode = !canvasToolbarState.presentationMode;
    });

    hotkeys("escape", (e) => {
        if (!canvasToolbarState.presentationMode) return;
        if (isEditableEventTarget(e)) return;
        e.preventDefault();
        canvasToolbarState.presentationMode = false;
    });
};

export const cleanupUIShortcuts = () => {
    hotkeys.unbind("ctrl+i,cmd+i");
    hotkeys.unbind("ctrl+comma,cmd+comma");
    hotkeys.unbind("ctrl+1,cmd+1");
    hotkeys.unbind("ctrl+2,cmd+2");
    hotkeys.unbind("ctrl+3,cmd+3");
    hotkeys.unbind("ctrl+shift+z,cmd+shift+z");
    hotkeys.unbind("ctrl+y,cmd+y");
    hotkeys.unbind("ctrl+z,cmd+z");
    hotkeys.unbind("f11");
    hotkeys.unbind("escape");
};
