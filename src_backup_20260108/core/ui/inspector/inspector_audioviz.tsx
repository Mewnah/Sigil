import { useGetState } from "@/client";
import { useUpdateElement } from "@/utils";
import { FC } from "react";
import { useSnapshot } from "valtio";
import Inspector from "./components";
import { InputCheckbox, InputColor, InputRange } from "./components/input";
import TransformInput from "./components/transform-input";
import Inspector_Animations from "./inspector_animations";
import { Element_AudioVizState } from "@/client/elements/audio-viz/schema";
import { ElementSceneState } from "@/client/elements/schema";
import { useTranslation } from "react-i18next";

const Inspector_ElementAudioViz: FC<{ id: string }> = ({ id }) => {
    const { t } = useTranslation();
    const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
    // Type assertion or safe check?
    const data = useGetState(state => state.elements[id]?.scenes[activeScene]?.data as Element_AudioVizState);
    const up = useUpdateElement<Element_AudioVizState>(id);

    if (!data) return null;

    return <>
        <Inspector.Body>
            <Inspector.Header>Audio Visualizer</Inspector.Header>
            <Inspector.Content>
                <Inspector.SubHeader>Settings</Inspector.SubHeader>
                <TransformInput id={id} />

                <InputColor label="Bar Color" value={data.barColor} onChange={(v: string) => up("barColor", v)} />
                <InputRange label="Bar Count" min={4} max={128} step={1} value={data.barCount} onChange={(e: any) => up("barCount", Number(e.target.value))} />
                <InputRange label="Gap" min={0} max={10} step={1} value={data.gap} onChange={(e: any) => up("gap", Number(e.target.value))} />
                <InputRange label="Sensitivity" min={0.1} max={5} step={0.1} value={data.sensitivity} onChange={(e: any) => up("sensitivity", Number(e.target.value))} />
                <InputRange label="Corner Radius" min={0} max={20} step={1} value={data.radius} onChange={(e: any) => up("radius", Number(e.target.value))} />
                <InputCheckbox label="Mirror Mode" value={data.mirror} onChange={(v: boolean) => up("mirror", v)} />

                <div className="h-4" />
                <Inspector_Animations id={id} />
            </Inspector.Content>
        </Inspector.Body>
    </>
}

export default Inspector_ElementAudioViz;
