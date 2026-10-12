import classNames from "classnames";
import { FC, memo, useCallback, useEffect, useRef, useState } from "react";
import { useEffectOnce } from "react-use";
import { useGetState } from "../..";
import { StyleToEventController } from "../utils";
import { Element_ImageState } from "./schema";
import { useSnapshot } from "valtio";

const Element_Image: FC<{ id: string }> = memo(({ id }) => {
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  useSnapshot(window.ApiClient.files.fileUrls);
  // The element can be gone, or lack the active scene, for a render (a scene switch or delete);
  // every hook below still runs, whatever the document holds (see ElementInstance).
  const state = useGetState(state => state.elements[id]?.scenes?.[activeScene]?.data as Element_ImageState | undefined);
  const fileUrl = state ? window.ApiClient.files.getFileUrl(state.fileId) : undefined;
  const activeFileUrl = state ? window.ApiClient.files.getFileUrl(state.activeFileId) : undefined;
  const attributeController = useRef<StyleToEventController>();

  const [attributes, setAttributes] = useState<Record<string, string>>({})

  useEffectOnce(() => {
    attributeController.current = new StyleToEventController(state?.styleCss ?? "", attrs => {
      setAttributes(Object.fromEntries(attrs.map(a => [a.toLowerCase(), ""])));
    });
  });

  const [active, setActive] = useState(false);

  const activeDurationRef = useRef(state?.activeDuration ?? 100);
  activeDurationRef.current = state?.activeDuration ?? 100;

  const timeoutN = useRef<number>(-1);
  const debounce = useCallback(() => {
    setActive(true);
    clearTimeout(timeoutN.current);
    if (document.visibilityState === "visible")
      timeoutN.current = setTimeout(() => setActive(false), activeDurationRef.current) as unknown as number;
  }, [])

  const activeEvent = state?.activeEvent;
  useEffect(() => {
    if (activeEvent) {
      const sub = window.ApiShared.pubsub.subscribe(activeEvent, _ => debounce());
      return () => window.ApiShared.pubsub.unsubscribe(sub);
    }
  }, [activeEvent, debounce]);

  const styleCss = state?.styleCss;
  useEffect(() => {
    if (styleCss !== undefined) attributeController.current?.OnStyleChange(styleCss)
  }, [styleCss]);

  if (!state) return null;

  return <>
    <style>
      {`
      .image {
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
        outline: 1px solid transparent;
        margin: auto;
        object-fit: contain;
        opacity: ${state.styleOpacity};
        background-image: ${fileUrl ? `url("${fileUrl}")` : "none"};
        transition: opacity ${state.activeTransitionDuration}ms ease-in-out, background-image ${state.activeTransitionDuration}ms ease-in-out;
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
      }
      .image.active{
        opacity: ${state.activeStyleOpacity};
        background-image: ${activeFileUrl ? `url("${activeFileUrl}")` : "none"};
      }
      `}
    </style>
    <style>{state.styleCss}</style>
    <div {...attributes} className={classNames("image", { active })} />
  </>
});

export default Element_Image;
