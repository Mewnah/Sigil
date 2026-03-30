import { FC, memo, useMemo } from "react";
import root from 'react-shadow';
import { useGetState } from "../index";
import Element_Image from "../elements/image";
import Element_Text from "../elements/text";
import { ElementType, AnimationConfig } from "../elements/schema";
import { useSnapshot } from "valtio";
import { motion, Variants } from "framer-motion";

// Animation variants map
const getVariants = (config: AnimationConfig): Variants => {
  const { type, duration, delay, ease } = config;

  const baseTransition = {
    duration: duration / 1000,
    delay: delay / 1000,
    ease: ease === 'linear' ? 'linear' :
      ease === 'easeIn' ? 'easeIn' :
        ease === 'easeOut' ? 'easeOut' :
          ease === 'easeInOut' ? 'easeInOut' :
            ease === 'anticipate' ? 'anticipate' :
              ease === 'backIn' ? 'backIn' :
                ease === 'backOut' ? 'backOut' : 'easeOut'
  };

  const variantsMap: Record<string, { hidden: object, visible: object }> = {
    'none': { hidden: {}, visible: {} },
    'fade': { hidden: { opacity: 0 }, visible: { opacity: 1 } },
    'slide-up': { hidden: { y: 20, opacity: 0 }, visible: { y: 0, opacity: 1 } },
    'slide-down': { hidden: { y: -20, opacity: 0 }, visible: { y: 0, opacity: 1 } },
    'slide-left': { hidden: { x: 20, opacity: 0 }, visible: { x: 0, opacity: 1 } },
    'slide-right': { hidden: { x: -20, opacity: 0 }, visible: { x: 0, opacity: 1 } },
    'scale': { hidden: { scale: 0.8, opacity: 0 }, visible: { scale: 1, opacity: 1 } },
  };

  const selected = variantsMap[type] || variantsMap['none'];

  return {
    hidden: { ...selected.hidden },
    visible: { ...selected.visible, transition: baseTransition },
    exit: { ...selected.hidden, transition: baseTransition },
  };
};

export const ElementInstance: FC<{ id: string; }> = memo(({ id }) => {
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const { type, scenes } = useGetState(state => state.elements[id]);

  const animation = useMemo(() => {
    return scenes[activeScene]?.animation;
  }, [scenes, activeScene]);

  if (!(activeScene in scenes))
    return <></>;

  const variants = useMemo(() => {
    if (!animation?.enter || animation.enter.type === 'none') return undefined;
    return getVariants(animation.enter);
  }, [animation?.enter]);

  function render() {
    switch (type) {
      case ElementType.text: return <Element_Text id={id} />;
      case ElementType.image: return <Element_Image id={id} />;
      default: return <>unknown element</>;
    }
  }

  const content = render();

  // Wrap in motion.div if animations are configured
  if (variants) {
    return (
      <root.div className="absolute inset-0 min-h-0 min-w-0 overflow-hidden">
        <motion.div
          initial="hidden"
          animate="visible"
          exit="exit"
          variants={variants}
          className="h-full min-h-0 w-full min-w-0 overflow-hidden"
        >
          {content}
        </motion.div>
      </root.div>
    );
  }

  return (
    <root.div className="absolute inset-0 min-h-0 min-w-0 overflow-hidden">{content}</root.div>
  );
});
