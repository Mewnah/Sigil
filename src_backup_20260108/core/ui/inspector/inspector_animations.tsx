import { useGetState, useUpdateState } from "@/client";
import { FC } from "react";
import { useSnapshot } from "valtio";
import Inspector from "./components";
import { InputText, InputSelect } from "./components/input";
import { AnimationStateSchema, AnimationConfigSchema } from "@/client/elements/schema";

const ANIMATION_TYPES = [
    { label: "None", value: "none" },
    { label: "Fade", value: "fade" },
    { label: "Slide Up", value: "slide-up" },
    { label: "Slide Down", value: "slide-down" },
    { label: "Slide Left", value: "slide-left" },
    { label: "Slide Right", value: "slide-right" },
    { label: "Scale", value: "scale" },
];

const EASING_TYPES = [
    { label: "Linear", value: "linear" },
    { label: "Ease In", value: "easeIn" },
    { label: "Ease Out", value: "easeOut" },
    { label: "Ease In Out", value: "easeInOut" },
    { label: "Anticipate", value: "anticipate" },
    { label: "Back In", value: "backIn" },
    { label: "Back Out", value: "backOut" },
];

const AnimationConfig: FC<{
    label: string,
    path: "enter" | "exit",
    id: string,
    sceneId: string
}> = ({ label, path, id, sceneId }) => {
    const update = useUpdateState();
    const anim = useGetState(state => state.elements[id]?.scenes[sceneId]?.animation?.[path]);

    if (!anim) return null; // Should be defaulted by schema

    const updateAnim = (key: keyof typeof anim, value: any) => {
        update(state => {
            const target = state.elements[id].scenes[sceneId].animation[path];
            if (target) {
                // @ts-ignore
                target[key] = value;
            } else {
                // init if missing? Schema defaults should handle this, but safe guard.
                state.elements[id].scenes[sceneId].animation[path] = { ...AnimationConfigSchema.parse({}), [key]: value };
            }
        });

    };

    return (
        <div className="flex flex-col space-y-2 p-2 bg-base-200/50 rounded-lg">
            <div className="text-xs font-bold opacity-70 uppercase">{label}</div>

            <InputSelect
                label="Type"
                value={anim.type}
                options={ANIMATION_TYPES}
                onValueChange={v => updateAnim("type", v)}
            />

            {anim.type !== 'none' && (
                <>
                    <div className="grid grid-cols-2 gap-2">
                        <InputText
                            label="Duration (ms)"
                            type="number"
                            value={anim.duration}
                            onChange={e => updateAnim("duration", Number(e.target.value))}
                        />
                        <InputText
                            label="Delay (ms)"
                            type="number"
                            value={anim.delay}
                            onChange={e => updateAnim("delay", Number(e.target.value))}
                        />
                    </div>
                    <InputSelect
                        label="Ease"
                        value={anim.ease}
                        options={EASING_TYPES}
                        onValueChange={v => updateAnim("ease", v)}
                    />
                </>
            )}
        </div>
    );
};

export const Inspector_Animations: FC<{ id: string }> = ({ id }) => {
    const { activeScene } = useSnapshot(window.ApiClient.scenes.state);

    // Ensure animation object exists in state if it's missing (legacy data)
    const animState = useGetState(state => state.elements[id]?.scenes[activeScene]?.animation);

    if (!animState) {
        // Proactively patch if missing? Or just rely on schema defaults during generic usage?
        // If data exists but animation key is missing, we might render nothing.
        return (
            <Inspector.Content>
                <Inspector.SubHeader>Animations</Inspector.SubHeader>
                <div className="text-xs opacity-50 p-2">No animation data available.</div>
            </Inspector.Content>
        )
    }

    return (
        <Inspector.Content>
            <Inspector.SubHeader>Animations</Inspector.SubHeader>
            <div className="flex flex-col space-y-4">
                <AnimationConfig label="Enter" path="enter" id={id} sceneId={activeScene} />
                <AnimationConfig label="Exit" path="exit" id={id} sceneId={activeScene} />
            </div>
        </Inspector.Content>
    );
};

export default Inspector_Animations;
