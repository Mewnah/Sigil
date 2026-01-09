import hotkeys from "hotkeys-js";
import { ServiceNetworkState } from "@/types";

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
};

export const cleanupUIShortcuts = () => {
    hotkeys.unbind("ctrl+i,cmd+i");
    hotkeys.unbind("ctrl+comma,cmd+comma");
    hotkeys.unbind("ctrl+1,cmd+1");
    hotkeys.unbind("ctrl+2,cmd+2");
    hotkeys.unbind("ctrl+3,cmd+3");
};
