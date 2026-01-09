import { FC, memo, PropsWithChildren, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence, useDragControls, PanInfo } from "framer-motion";
import classNames from "classnames";
import { RiCloseLine } from "react-icons/ri";
import SimpleBar from "simplebar-react";

interface DrawerProps {
    open: boolean;
    onClose: () => void;
    title?: string;
    position?: "right" | "left";
    width?: number;
    minWidth?: number;
    maxWidth?: number;
    children: React.ReactNode;
}

export const Drawer: FC<DrawerProps> = memo(({
    open,
    onClose,
    title,
    position = "right",
    width: initialWidth = 350,
    minWidth = 280,
    maxWidth = 600,
    children
}) => {
    const [width, setWidth] = useState(initialWidth);
    const [isResizing, setIsResizing] = useState(false);
    const drawerRef = useRef<HTMLDivElement>(null);

    const isRight = position === "right";

    // Handle resize drag
    useEffect(() => {
        if (!isResizing) return;

        const handleMouseMove = (e: MouseEvent) => {
            const newWidth = isRight
                ? window.innerWidth - e.clientX
                : e.clientX;
            setWidth(Math.min(maxWidth, Math.max(minWidth, newWidth)));
        };

        const handleMouseUp = () => setIsResizing(false);

        window.addEventListener("mousemove", handleMouseMove);
        window.addEventListener("mouseup", handleMouseUp);
        return () => {
            window.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
        };
    }, [isResizing, isRight, minWidth, maxWidth]);

    return (
        <AnimatePresence>
            {open && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
                    />

                    {/* Drawer */}
                    <motion.div
                        ref={drawerRef}
                        initial={{ x: isRight ? "100%" : "-100%" }}
                        animate={{ x: 0 }}
                        exit={{ x: isRight ? "100%" : "-100%" }}
                        transition={{ type: "spring", damping: 30, stiffness: 300 }}
                        style={{ width }}
                        className={classNames(
                            "fixed top-0 bottom-0 z-50 bg-base-100 shadow-2xl flex flex-col",
                            isRight ? "right-0 rounded-l-2xl" : "left-0 rounded-r-2xl"
                        )}
                    >
                        {/* Resize handle */}
                        <div
                            onMouseDown={() => setIsResizing(true)}
                            className={classNames(
                                "absolute top-0 bottom-0 w-1 cursor-ew-resize hover:bg-primary/50 transition-colors",
                                isRight ? "left-0" : "right-0"
                            )}
                        />

                        {/* Header */}
                        {title && (
                            <div className="flex-none px-4 py-3 border-b border-base-content/10 flex items-center justify-between">
                                <span className="font-bold font-header text-lg">{title}</span>
                                <button
                                    onClick={onClose}
                                    className="btn btn-ghost btn-sm btn-circle"
                                    aria-label="Close"
                                >
                                    <RiCloseLine className="text-xl" />
                                </button>
                            </div>
                        )}

                        {/* Content */}
                        <SimpleBar className="flex-1 overflow-hidden">
                            <div className="p-4">
                                {children}
                            </div>
                        </SimpleBar>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
});

// Quick drawer for inspector panels
interface InspectorDrawerProps {
    open: boolean;
    onClose: () => void;
    title: string;
    children: React.ReactNode;
}

export const InspectorDrawer: FC<InspectorDrawerProps> = memo(({
    open,
    onClose,
    title,
    children
}) => (
    <Drawer
        open={open}
        onClose={onClose}
        title={title}
        position="right"
        width={380}
        minWidth={320}
        maxWidth={500}
    >
        {children}
    </Drawer>
));

export default Drawer;
