import { FC, memo, useState, KeyboardEvent } from "react";
import { RiSendPlaneFill } from "react-icons/ri";
import { TextEventSource, TextEventType } from "@/types";

export const TTSInputBar: FC = memo(() => {
    const [inputValue, setInputValue] = useState('');

    const handleSubmit = () => {
        if (!inputValue.trim()) return;

        // Publish to the text pipeline for TTS
        window.ApiShared.pubsub.publishText(TextEventSource.textfield, {
            type: TextEventType.final,
            value: inputValue.trim()
        });

        setInputValue('');
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="flex-shrink-0 bg-transparent px-1 py-1">
            <div className="flex items-center gap-1.5">
                <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type to speak..."
                    className="flex-1 h-9 px-2.5 text-sm bg-base-300 border border-base-content/10 rounded-lg outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 placeholder:text-base-content/30"
                />
                <button
                    onClick={handleSubmit}
                    disabled={!inputValue.trim()}
                    title="Send to TTS"
                    className="h-9 w-9 flex items-center justify-center bg-primary text-black rounded-lg hover:bg-primary/80 disabled:opacity-30 disabled:hover:bg-primary transition-colors"
                >
                    <RiSendPlaneFill />
                </button>
            </div>
        </div>
    );
});
