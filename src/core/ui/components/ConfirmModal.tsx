import { FC, memo, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RiAlertFill, RiCloseLine } from "react-icons/ri";

interface ConfirmModalProps {
    isOpen: boolean;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    variant?: "danger" | "warning" | "info";
    showDontAskAgain?: boolean;
    onConfirm: (dontAskAgain: boolean) => void;
    onCancel: () => void;
}

export const ConfirmModal: FC<ConfirmModalProps> = memo(({
    isOpen,
    title,
    message,
    confirmText = "Confirm",
    cancelText = "Cancel",
    variant = "danger",
    showDontAskAgain = false,
    onConfirm,
    onCancel
}) => {
    const [dontAskAgain, setDontAskAgain] = useState(false);
    const [confirmReady, setConfirmReady] = useState(false);

    useEffect(() => {
        if (!isOpen) {
            setConfirmReady(false);
            return;
        }
        let raf1 = 0;
        let raf2 = 0;
        raf1 = requestAnimationFrame(() => {
            raf2 = requestAnimationFrame(() => setConfirmReady(true));
        });
        return () => {
            cancelAnimationFrame(raf1);
            cancelAnimationFrame(raf2);
        };
    }, [isOpen]);

    const variantStyles = {
        danger: {
            icon: "text-error",
            button: "bg-error hover:bg-error/80 text-white"
        },
        warning: {
            icon: "text-warning",
            button: "bg-warning hover:bg-warning/80 text-black"
        },
        info: {
            icon: "text-info",
            button: "bg-info hover:bg-info/80 text-white"
        }
    };

    const styles = variantStyles[variant];

    const handleConfirm = () => {
        if (!confirmReady) return;
        onConfirm(dontAskAgain);
        setDontAskAgain(false);
    };

    const handleCancel = () => {
        setDontAskAgain(false);
        onCancel();
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]"
                        onClick={handleCancel}
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 10 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 10 }}
                        transition={{ duration: 0.15, ease: "easeOut" }}
                        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[100] w-full max-w-sm"
                    >
                        <div className="bg-base-200 border border-base-content/10 rounded-xl shadow-2xl overflow-hidden">
                            {/* Header */}
                            <div className="flex items-center justify-between px-4 py-3 border-b border-base-content/5">
                                <div className="flex items-center gap-2">
                                    <RiAlertFill className={`text-lg ${styles.icon}`} />
                                    <h3 className="font-bold text-base-content">{title}</h3>
                                </div>
                                <button
                                    type="button"
                                    onClick={handleCancel}
                                    title="Close"
                                    className="p-1 rounded hover:bg-base-content/10 transition-colors cursor-pointer"
                                >
                                    <RiCloseLine className="text-lg" />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="px-4 py-4">
                                <p className="text-sm text-base-content/70">{message}</p>

                                {showDontAskAgain && (
                                    <label className="flex items-center gap-2 mt-4 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={dontAskAgain}
                                            onChange={(e) => setDontAskAgain(e.target.checked)}
                                            className="checkbox checkbox-sm checkbox-primary"
                                        />
                                        <span className="text-xs text-base-content/50">Don't ask again</span>
                                    </label>
                                )}
                            </div>

                            {/* Actions */}
                            <div className="flex justify-end gap-2 px-4 py-3 bg-base-300/50">
                                <button
                                    type="button"
                                    onClick={handleCancel}
                                    className="px-4 py-2 text-sm font-medium rounded-lg bg-base-content/10 hover:bg-base-content/20 transition-colors cursor-pointer"
                                >
                                    {cancelText}
                                </button>
                                <button
                                    type="button"
                                    disabled={!confirmReady}
                                    onClick={handleConfirm}
                                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${confirmReady ? "cursor-pointer" : "cursor-not-allowed opacity-50 pointer-events-none"} ${styles.button}`}
                                >
                                    {confirmText}
                                </button>
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
});
