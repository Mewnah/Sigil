import { FC, useEffect } from "react";
import NiceModal from "@ebay/nice-modal-react";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.min.css';
import { useSnapshot } from "valtio";
import { AnimatePresence } from "framer-motion";

import { initializeUIShortcuts, cleanupUIShortcuts } from "../shortcuts";
import OverlayInput from "../overlay-input";
import BackgroundInput from "../background-input";
import { SigilLayout } from "./SigilLayout";
import "../file-modal";

// Shortcut Recorder (Inline for now or import if extracted)
// We'll reuse the logic from EditorView if possible, or just re-implement simple version
const ShortcutRecorderStub: FC = () => null;

export const SigilRoot: FC = () => {
    const { showOverlay } = useSnapshot(window.ApiServer.state);

    // Initialize keyboard shortcuts
    useEffect(() => {
        initializeUIShortcuts();
        return () => cleanupUIShortcuts();
    }, []);

    return (
        <NiceModal.Provider>
            <SigilLayout />

            {/* Global Overlays */}
            <AnimatePresence>
                {showOverlay && <OverlayInput onClose={() => window.ApiServer.state.showOverlay = false} />}
            </AnimatePresence>

            <BackgroundInput />
            <ToastContainer className="toasts" draggable={false} closeOnClick limit={3} hideProgressBar theme="colored" />
        </NiceModal.Provider>
    );
}

export default SigilRoot;
