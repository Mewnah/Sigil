import { FC, useEffect, useRef } from "react";
import { useGetState } from "@/client";
import { useSnapshot } from "valtio";
import { Element_AudioVizState } from "./schema";

const Element_AudioViz: FC<{ id: string }> = ({ id }) => {
    const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
    // Subscribe to element data
    const data = useGetState(state => state.elements[id].scenes[activeScene].data as Element_AudioVizState);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const rafRef = useRef<number>();

    useEffect(() => {
        // Initialize audio input if not already
        window.ApiServer.sound.startInput().catch(console.error);

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const render = () => {
            const analyser = window.ApiServer.sound.analyser;
            if (!analyser) {
                rafRef.current = requestAnimationFrame(render);
                return;
            }

            // Resize canvas to container
            if (containerRef.current) {
                const { clientWidth, clientHeight } = containerRef.current;
                if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
                    canvas.width = clientWidth;
                    canvas.height = clientHeight;
                }
            }

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const { barColor, barCount, gap, sensitivity, radius, mirror } = data;

            // Calculate bar width based on available space
            // Mirror mode effectively doubles the bars visualized (center out or duplicate?)
            // We'll duplicate: rendering left and right? Or just spread spectrum?
            // Standard: spread spectrum across width.

            const effectiveCount = mirror ? barCount / 2 : barCount;
            const totalGap = (barCount - 1) * gap;
            const availableWidth = canvas.width - totalGap;
            const barWidth = availableWidth / barCount;

            ctx.fillStyle = barColor;

            // Draw bars
            for (let i = 0; i < barCount; i++) {
                // Map bar index to frequency index (logarithmic or linear?)
                // Linear for simplicity
                const freqIndex = Math.floor((i / barCount) * (bufferLength / 2)); // Use lower half of spectrum which has most energy usually

                let value = dataArray[freqIndex] || 0;
                value = value * sensitivity;

                const percent = Math.min(1, value / 255);
                const barHeight = percent * canvas.height;

                const x = i * (barWidth + gap);
                // Draw from bottom
                const y = canvas.height - barHeight;

                // Radius if possible (simple rect doesn't support radius easily in old context, use roundRect if avail or path)
                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(x, y, barWidth, barHeight, radius);
                    ctx.fill();
                } else {
                    ctx.fillRect(x, y, barWidth, barHeight);
                }
            }

            rafRef.current = requestAnimationFrame(render);
        };

        render();

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        }
    }, [data.barColor, data.barCount, data.gap, data.sensitivity, data.radius, data.mirror]); // Re-bind if configs change significantly? 
    // Actually refs in render loop point to CURRENT data if we use refs or just read from snapshot if it's stable?
    // `data` from `useGetState` returns a snapshot-like object but inside effect we use `data` closure.
    // We added deps so it re-starts render loop with new data closure. Good.

    return (
        <div ref={containerRef} className="w-full h-full">
            <canvas ref={canvasRef} className="block" />
        </div>
    );
};

export default Element_AudioViz;
