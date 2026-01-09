import { FC } from "react";

interface KickIconProps {
    className?: string;
}

export const KickIcon: FC<KickIconProps> = ({ className }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        height="1em"
        width="1em"
        className={className}
    >
        <path
            fill="currentColor"
            fillRule="evenodd"
            d="M5 1C2.79086 1 1 2.79086 1 5v14c0 2.2091 1.79086 4 4 4h14c2.2091 0 4 -1.7909 4 -4V5c0 -2.20914 -1.7909 -4 -4 -4H5Zm5.3696 3.5H5.47827v15h4.89133v-3.2609H12v1.6305h1.6304V19.5h4.8913v-4.8913h-1.6304v-1.6304h-1.6304v-1.9566h1.6304V9.3913h1.6304V4.5h-4.8913v1.63043H12v1.63044h-1.6304V4.5Z"
            clipRule="evenodd"
        />
    </svg>
);
