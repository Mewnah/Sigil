import { FC, memo, useState } from "react";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { exit } from "@tauri-apps/plugin-process";
import { VscChromeClose, VscChromeMaximize, VscChromeMinimize } from "react-icons/vsc";
import { RiSideBarLine, RiMoonFill, RiSunFill, RiKeyboardBoxLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { Services } from "@/services-registry";
import { useAppUIStore } from "../store";
import Tooltip from "../dropdown/Tooltip";
const appWindow = getCurrentWebviewWindow()

// Map tab IDs to human-readable names
const TAB_NAMES: Record<string, string> = {
    [Services.stt]: "Speech to Text",
    [Services.tts]: "Text to Speech",
    [Services.translation]: "Translation",
    [Services.transform]: "AI Transform",
    [Services.twitch]: "Twitch",
    [Services.kick]: "Kick",
    [Services.discord]: "Discord",
    [Services.vrc]: "VRChat",
    [Services.obs]: "OBS Studio",
    [Services.voice_changer]: "Voice Changer",
    "scenes": "Canvas & Elements",
    "project": "Project",
    "elements": "Elements",
    "files": "Files",
    "settings": "Settings",
    "text": "Text Element",
    "image": "Image Element",
};

const WindowControls: FC = memo(() => {
    const handleMinimize = async () => {
        try {
            await appWindow.minimize();
        } catch (e) {
            console.error("Failed to minimize:", e);
        }
    };

    const handleMaximize = async () => {
        try {
            await appWindow.toggleMaximize();
        } catch (e) {
            console.error("Failed to toggle maximize:", e);
        }
    };

    const handleClose = async () => {
        try {
            await appWindow.close();
        } catch (e) {
            console.error("Failed to close:", e);
            try {
                await exit(0);
            } catch (e2) {
                console.error("Failed to exit:", e2);
            }
        }
    };

    return (
        <div className="flex items-center h-full">
            <button
                onClick={handleMinimize}
                title="Minimize"
                className="h-full w-12 flex items-center justify-center hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                tabIndex={-1}
            >
                <VscChromeMinimize />
            </button>
            <button
                onClick={handleMaximize}
                title="Maximize"
                className="h-full w-12 flex items-center justify-center hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                tabIndex={-1}
            >
                <VscChromeMaximize />
            </button>
            <button
                onClick={handleClose}
                title="Close"
                className="h-full w-12 flex items-center justify-center hover:bg-error transition-colors text-base-content/50 hover:text-white"
                tabIndex={-1}
            >
                <VscChromeClose />
            </button>
        </div>
    );
});

interface SigilHeaderProps {
    onToggleSidebar?: () => void;
    sidebarCollapsed?: boolean;
}

export const SigilHeader: FC<SigilHeaderProps> = memo(({ onToggleSidebar, sidebarCollapsed }) => {
    const tab = useAppUIStore((s) => s.sidebar.tab);
    const { clientTheme } = useSnapshot(window.ApiServer.state);
    const [showShortcuts, setShowShortcuts] = useState(false);

    // Get current page name
    const currentPage = tab?.tab ? (TAB_NAMES[tab.tab] || tab.tab) : "Dashboard";

    // Determine if current theme is dark
    const isDarkTheme = clientTheme === 'sigil-dark' || clientTheme === 'curses' || clientTheme === 'streamer';

    // Simple theme toggle
    const toggleTheme = () => {
        const newTheme = isDarkTheme ? 'sigil-light' : 'sigil-dark';
        window.ApiServer.changeTheme(newTheme);
    };

    return (
        <>
            <header className="h-10 bg-base-300 flex select-none border-b border-base-content/5">
                {/* Left Section */}
                <div className="flex items-center">
                    {/* Sidebar Toggle */}
                    <Tooltip content={sidebarCollapsed ? "Expand sidebar (Ctrl+B)" : "Collapse sidebar (Ctrl+B)"} placement="bottom">
                        <button
                            onClick={onToggleSidebar}
                            title="Toggle sidebar"
                            className="h-10 w-10 flex items-center justify-center hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                        >
                            <RiSideBarLine className={sidebarCollapsed ? "rotate-180" : ""} />
                        </button>
                    </Tooltip>
                </div>

                {/* Draggable Area */}
                <div data-tauri-drag-region className="flex-1 flex items-center px-4 gap-3">
                    <div className="font-header font-black text-sm tracking-wider text-primary">SIGIL <span className="text-base-content/30 font-normal">STUDIO</span></div>
                    <div className="h-4 w-px bg-base-content/10"></div>
                    <div className="text-sm text-base-content/70 font-medium">{currentPage}</div>
                </div>

                {/* Right Section - Quick Actions */}
                <div className="flex items-center gap-1 px-2">
                    {/* Keyboard Shortcuts Help */}
                    <Tooltip content="Keyboard shortcuts" placement="bottom">
                        <button
                            onClick={() => setShowShortcuts(true)}
                            title="Keyboard shortcuts"
                            className="h-8 w-8 flex items-center justify-center rounded hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                        >
                            <RiKeyboardBoxLine />
                        </button>
                    </Tooltip>

                    {/* Theme Toggle */}
                    <Tooltip content={`Current: ${clientTheme === 'sigil-light' ? 'Light Mode' : 'Dark Mode'}. Click to toggle.`} placement="bottom">
                        <button
                            onClick={() => {
                                if (clientTheme === 'sigil-dark') window.ApiServer.changeTheme('sigil-light');
                                else window.ApiServer.changeTheme('sigil-dark');
                            }}
                            title="Toggle Theme"
                            className="h-8 w-8 flex items-center justify-center rounded hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                        >
                            {clientTheme === 'sigil-light' ? <RiSunFill /> : <RiMoonFill />}
                        </button>
                    </Tooltip>
                </div>

                {/* Window Controls */}
                <div className="flex-none">
                    <WindowControls />
                </div>
            </header>

            {/* Keyboard Shortcuts Modal */}
            {showShortcuts && (
                <>
                    <div className="fixed inset-0 bg-black/50 z-50" onClick={() => setShowShortcuts(false)} />
                    <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-200 rounded-xl shadow-2xl p-6 z-50 w-96">
                        <h2 className="text-lg font-bold mb-4">Keyboard Shortcuts</h2>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between"><span>Dashboard</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+0</kbd></div>
                            <div className="flex justify-between"><span>Speech to Text</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+1</kbd></div>
                            <div className="flex justify-between"><span>Text to Speech</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+2</kbd></div>
                            <div className="flex justify-between"><span>Translation</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+3</kbd></div>
                            <div className="flex justify-between"><span>AI Transform</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+4</kbd></div>
                            <div className="flex justify-between"><span>Canvas & Elements</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+5</kbd></div>
                            <div className="flex justify-between"><span>Files</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+6</kbd></div>
                            <div className="flex justify-between"><span>Settings</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+7</kbd></div>
                            <div className="border-t border-base-content/10 my-2" />
                            <div className="flex justify-between"><span>Toggle Sidebar</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Ctrl+B</kbd></div>
                            <div className="flex justify-between"><span>Go Back/Close</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs">Esc</kbd></div>
                        </div>
                        <button onClick={() => setShowShortcuts(false)} className="mt-4 w-full py-2 bg-primary text-black rounded font-medium">Close</button>
                    </div>
                </>
            )}
        </>
    );
});
