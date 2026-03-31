import { FC, memo }        from "react";
import { useTranslation } from "react-i18next";
import { useGetState }     from "../index";
import { ElementInstance } from "./element-instance";
import { useSnapshot } from "valtio";

export const ElementSimpleTransform: FC<{ id: string }> = memo(({ id }) => {
  const {activeScene} = useSnapshot(window.ApiClient.scenes.state);
  const rect = useGetState(state => state.elements[id].scenes[activeScene]?.rect);
  return <div
    className="absolute transition-all duration-100"
    style={{
      width: rect?.w || 0,
      height: rect?.h || 0,
      left: rect?.x || 0,
      top: rect?.y || 0,
    }}
  >
    <ElementInstance id={id} />
  </div>
});

const View: FC = () => {
  const { t } = useTranslation();
  const canvas = useGetState(state => state.canvas);
  const ids = useGetState(state => state.elementsIds);
  const mirror = useSnapshot(window.ApiShared.peer.clientMirrorState);
  return (
    <div
      className="overflow-auto w-screen h-screen flex items-center justify-center"
      style={{
        minHeight: "100vh",
        backgroundColor: "transparent",
        boxSizing: "border-box",
      }}
    >
    {(mirror.phase === "reconnecting" || mirror.phase === "failed") && (
      <div
        role="status"
        className="fixed top-0 left-0 right-0 z-[100] px-3 py-2 text-sm leading-snug shadow-md"
        style={{
          backgroundColor: "rgba(250, 204, 21, 0.92)",
          color: "#1c1917",
        }}
      >
        {mirror.phase === "failed" ? t("client.peer_failed") : t("client.peer_reconnecting")}
        {mirror.lastError ? ` ${t("client.peer_error_detail", { detail: mirror.lastError })}` : null}
      </div>
    )}
    <div style={{ width: canvas?.w, height: canvas?.h }} className="relative shrink-0">
      {ids?.map((elementId) => <ElementSimpleTransform id={elementId} key={elementId} />)}
    </div>
  </div>
  );
};

export default View;
