import { FC, memo, PropsWithChildren, ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import classNames from "classnames";
import Tooltip from "./dropdown/Tooltip";

interface DockItemProps {
    icon: ReactNode;
    label: string;
    active?: boolean;
    status?: "connected" | "connecting" | "disconnected" | "error";
    onClick?: () => void;
}

const StatusIndicator: FC<{ status?: DockItemProps["status"] }> = ({ status }) => {
    if (!status || status === "disconnected") return null;

    const colors = {
        connected: "bg-success",
        connecting: "bg-warning animate-pulse",
        error: "bg-error",
    };

    return (
        <div className={classNames(
            "absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-base-200",
            colors[status]
        )} />
    );
};

export const DockItem: FC<DockItemProps> = memo(({ icon, label, active, status, onClick }) => (
    <Tooltip content={label} placement="top">
        <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={onClick}
            className={classNames(
                "relative w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-colors",
                active
                    ? "bg-primary text-primary-content shadow-lg shadow-primary/25"
                    : "bg-base-100/50 text-base-content/70 hover:bg-base-100 hover:text-base-content"
            )}
        >
            {icon}
            <StatusIndicator status={status} />
        </motion.button>
    </Tooltip>
));

interface DockProps {
    position?: "bottom" | "left";
    children: ReactNode;
}

export const Dock: FC<DockProps> = memo(({ position = "bottom", children }) => {
    const isBottom = position === "bottom";

    return (
        <motion.div
            initial={{ opacity: 0, y: isBottom ? 20 : 0, x: isBottom ? 0 : -20 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            className={classNames(
                "fixed z-40 p-2 bg-base-200/80 backdrop-blur-lg rounded-2xl shadow-2xl border border-base-content/5",
                isBottom
                    ? "bottom-4 left-1/2 -translate-x-1/2 flex-row"
                    : "left-4 top-1/2 -translate-y-1/2 flex-col"
            )}
        >
            <div className={classNames(
                "flex gap-2",
                isBottom ? "flex-row" : "flex-col"
            )}>
                {children}
            </div>
        </motion.div>
    );
});

// Dock divider
export const DockDivider: FC<{ vertical?: boolean }> = ({ vertical }) => (
    <div className={classNames(
        "bg-base-content/10 rounded-full",
        vertical ? "w-px h-8 mx-1" : "h-px w-8 my-1"
    )} />
);

export default { Dock, DockItem, DockDivider };
