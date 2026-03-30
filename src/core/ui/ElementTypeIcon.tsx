import { ElementType } from "@/client/elements/schema";
import { FC } from "react";
import { RiImageFill, RiTextWrap } from "react-icons/ri";

type Props = { type: ElementType; className?: string };

/** Shared icon for element rows (sidebar, inspector lists, studio). */
export const ElementTypeIcon: FC<Props> = ({ type, className }) => {
  switch (type) {
    case ElementType.text:
      return <RiTextWrap className={className} />;
    case ElementType.image:
      return <RiImageFill className={className} />;
  }
};
