// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// RVC voice-permission notice (shown while the service flags a refused RVC start).
import { FC, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";

export const ConsentDialog: FC = () => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    /**
     * The service refuses to start RVC until the voice-permission notice is acknowledged (Start, or a
     * restart into RVC such as a mode switch while running) and flags it; the notice shows while flagged.
     */
    const consentOpen = useSnapshot(window.ApiServer.voiceChanger.state).rvcConsentPending;
    const [consentAck, setConsentAck] = useState(false);
    useEffect(() => {
        if (consentOpen) setConsentAck(false);
    }, [consentOpen]);
    const confirmConsentAndStart = async () => {
        if (!consentAck) return;
        // Stored in Rust's settings (clears the flag); if that fails it still holds for this session.
        await vcApi.acknowledgeRvcConsent();
        vcApi.start().catch(() => {});
    };
    if (!consentOpen) return null;
    return createPortal(
        <div className="fixed inset-0 z-[10050] flex items-center justify-center bg-black/50 p-4">
            <div
                role="dialog"
                aria-modal="true"
                aria-labelledby="vc-rvc-consent-title"
                className="w-full max-w-md rounded-lg border border-base-content/15 bg-base-100 p-4 shadow-xl"
            >
                <h3 id="vc-rvc-consent-title" className="text-base font-semibold">
                    {t("voice_changer.rvc_voice_consent_title")}
                </h3>
                <p className="mt-2 text-sm text-base-content/80 whitespace-pre-line">
                    {t("voice_changer.rvc_voice_consent_body")}
                </p>
                <label className="label mt-3 cursor-pointer justify-start gap-2 py-0">
                    <input
                        type="checkbox"
                        className="checkbox checkbox-sm checkbox-primary"
                        checked={consentAck}
                        onChange={(e) => setConsentAck(e.target.checked)}
                    />
                    <span className="label-text text-sm">{t("voice_changer.rvc_voice_consent_ack")}</span>
                </label>
                <div className="mt-4 flex justify-end gap-2">
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => vcApi.dismissRvcConsent()}
                    >
                        {t("voice_changer.info_close_overlay")}
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={!consentAck}
                        onClick={() => void confirmConsentAndStart()}
                    >
                        {t("voice_changer.btn_start")}
                    </button>
                </div>
            </div>
        </div>,
        document.body,
    );
};
