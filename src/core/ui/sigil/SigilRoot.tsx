import { FC, Suspense, lazy, useEffect } from "react";
import NiceModal from "@ebay/nice-modal-react";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.min.css';
import { useSnapshot } from "valtio";
import { AnimatePresence } from "framer-motion";

import { initializeUIShortcuts, cleanupUIShortcuts } from "../shortcuts";
import OverlayInput from "../overlay-input";
import BackgroundInput from "../background-input";
import { SigilLayout } from "./SigilLayout";
import { LanguageOfferToastHost } from "../inspector/stt/LanguageOffer";
import "../file-modal";

// The first-run wizard is loaded only when a new install needs it.
const Onboarding = lazy(() => import("../Onboarding"));

// Shortcut Recorder (Inline for now or import if extracted)
// We'll reuse the logic from EditorView if possible, or just re-implement simple version
const ShortcutRecorderStub: FC = () => null;

export const SigilRoot: FC = () => {
    const { showOverlay, onboardingComplete } = useSnapshot(window.ApiServer.state);
    const showWizard = window.Config.isApp() && !onboardingComplete;

    // Initialize keyboard shortcuts
    useEffect(() => {
        initializeUIShortcuts();
        return () => cleanupUIShortcuts();
    }, []);

    return (
        <NiceModal.Provider>
            <SigilLayout blocked={showWizard} />

            {/* First run: only a new install (no settings file, see state/firstRun.ts) leaves onboardingComplete false. */}
            <AnimatePresence>
                {showWizard && (
                    <Suspense key="onboarding" fallback={null}>
                        <Onboarding key="onboarding" />
                    </Suspense>
                )}
            </AnimatePresence>

            {/* Global Overlays */}
            <AnimatePresence>
                {showOverlay && <OverlayInput onClose={() => window.ApiServer.state.showOverlay = false} />}
            </AnimatePresence>

            <BackgroundInput />
            <LanguageOfferToastHost />
            <ToastContainer className="toasts" draggable={false} closeOnClick limit={3} hideProgressBar theme="colored" />
        </NiceModal.Provider>
    );
}

export default SigilRoot;
