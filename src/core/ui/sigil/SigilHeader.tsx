import { FC, memo, useState } from "react";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { exit } from "@tauri-apps/plugin-process";
import { VscChromeClose, VscChromeMaximize, VscChromeMinimize } from "react-icons/vsc";
import { RiSideBarLine, RiMoonFill, RiSunFill, RiKeyboardBoxLine, RiBookReadLine, RiVolumeMuteFill, RiVolumeUpFill } from "react-icons/ri";
import { HelpManualModal } from "../HelpManualModal";
import { useSnapshot } from "valtio";
import { useAppUIStore } from "../store";
import Tooltip from "../dropdown/Tooltip";
import { getInspectorHeaderTitle } from "../servicePanels";
import { useTranslation } from "react-i18next";
import classNames from "classnames";

/** Lazy singleton – only call Tauri API when actually rendered inside the desktop window. */
let _appWindow: ReturnType<typeof getCurrentWebviewWindow> | null = null;
function getAppWindow() {
  if (!_appWindow) _appWindow = getCurrentWebviewWindow();
  return _appWindow;
}

const WindowControls: FC = memo(() => {
    const handleMinimize = async () => {
        try {
            await getAppWindow().minimize();
        } catch (e) {
            console.error("Failed to minimize:", e);
        }
    };

    const handleMaximize = async () => {
        try {
            await getAppWindow().toggleMaximize();
        } catch (e) {
            console.error("Failed to toggle maximize:", e);
        }
    };

    const handleClose = async () => {
        try {
            await getAppWindow().close();
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
    const { t } = useTranslation();
    const tab = useAppUIStore((s) => s.sidebar.tab);
    const { clientTheme, muteSoundEffects } = useSnapshot(window.ApiServer.state);
    const [showShortcuts, setShowShortcuts] = useState(false);
    const [showManual, setShowManual] = useState(false);

    const handleToggleMuteSoundEffects = () => {
        window.ApiServer.state.muteSoundEffects = !window.ApiServer.state.muteSoundEffects;
    };

    // Get current page name
    const currentPage = getInspectorHeaderTitle(tab?.tab);

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
                    <Tooltip
                        content={t('main.btn_mute_effects')}
                        body={t('main.btn_mute_effects_desc')}
                        placement="bottom"
                    >
                        <button
                            onClick={handleToggleMuteSoundEffects}
                            title={t('main.btn_mute_effects')}
                            className={classNames(
                                "h-8 w-8 flex items-center justify-center rounded transition-colors",
                                muteSoundEffects
                                    ? "bg-error/20 text-error hover:bg-error/30"
                                    : "hover:bg-base-content/5 text-base-content/50 hover:text-base-content"
                            )}
                        >
                            {muteSoundEffects ? <RiVolumeMuteFill /> : <RiVolumeUpFill />}
                        </button>
                    </Tooltip>
                    <Tooltip content={t('manual.title')} placement="bottom">
                        <button
                            onClick={() => setShowManual(true)}
                            title={t('manual.title')}
                            className="h-8 w-8 flex items-center justify-center rounded hover:bg-base-content/5 transition-colors text-base-content/50 hover:text-base-content"
                        >
                            <RiBookReadLine />
                        </button>
                    </Tooltip>
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

            <HelpManualModal open={showManual} onClose={() => setShowManual(false)} />

            {/* Keyboard Shortcuts Modal */}
            {showShortcuts && (
                <>
                    <div className="fixed inset-0 bg-black/50 z-50" onClick={() => setShowShortcuts(false)} />
                    <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-200 rounded-xl shadow-2xl p-6 z-50 w-[min(28rem,calc(100vw-2rem))] max-h-[min(32rem,80vh)] overflow-y-auto">
                        <h2 className="text-lg font-bold mb-2">Keyboard shortcuts</h2>
                        <p className="text-xs text-base-content/60 mb-3">Use Cmd on macOS where Ctrl is shown. Open inspectors from the sidebar.</p>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between gap-2"><span>Toggle STT start/stop</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+1</kbd></div>
                            <div className="flex justify-between gap-2"><span>Toggle Rewrite start/stop</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+2</kbd></div>
                            <div className="flex justify-between gap-2"><span>Toggle TTS start/stop</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+3</kbd></div>
                            <div className="border-t border-base-content/10 my-2" />
                            <div className="flex justify-between gap-2"><span>Open Settings</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+,</kbd></div>
                            <div className="flex justify-between gap-2"><span>Toggle stats panel</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+I</kbd></div>
                            <div className="flex justify-between gap-2"><span>Toggle sidebar nav collapse</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+B</kbd></div>
                            <div className="flex justify-between gap-2"><span>Undo / Redo (canvas)</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Ctrl+Z / Y</kbd></div>
                            <div className="flex justify-between gap-2"><span>Presentation mode</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">F11</kbd></div>
                            <div className="flex justify-between gap-2"><span>Exit presentation</span><kbd className="bg-base-300 px-2 py-0.5 rounded text-xs shrink-0">Esc</kbd></div>
                        </div>
                        <button type="button" onClick={() => setShowShortcuts(false)} className="mt-4 w-full py-2 bg-primary text-black rounded font-medium">Close</button>
                    </div>
                </>
            )}
        </>
    );
});
