// Voice-changer inspector (P5-G): how a voice family is named and labelled (shared by the voice list, the
// "new voice" panel and the selected voice's family section).
import type { VcFamilyCapability, VcFamilyId } from "@/core/services/voice_changer/ipc";
import { epDisplayName } from "@/core/services/voice_changer/metrics";
import classNames from "classnames";
import type { TFunction } from "i18next";
import { FC } from "react";
import { useTranslation } from "react-i18next";

/** Translated family name (`voice_changer.family_name_<id>`). */
export function familyName(t: TFunction, family: VcFamilyId): string {
    return t(`voice_changer.family_name_${family}`);
}

/** Short family label for badges (`voice_changer.family_short_<id>`). */
export function familyShortName(t: TFunction, family: VcFamilyId): string {
    return t(`voice_changer.family_short_${family}`);
}

/** The EP rungs a family runs on, as display names (`CUDA / DirectML`). */
export function familyEpList(fam: Pick<VcFamilyCapability, "execution_providers">): string {
    return fam.execution_providers.map((e) => epDisplayName(e)).join(" / ");
}

/** Experimental / zero-shot / singing badges of a family. */
export const FamilyBadges: FC<{ fam: Pick<VcFamilyCapability, "experimental" | "zero_shot" | "singing">; className?: string }> = ({ fam, className }) => {
    const { t } = useTranslation();
    return (
        <span className={classNames("flex flex-wrap gap-1", className)}>
            {fam.experimental ? (
                <span className="badge badge-warning badge-outline badge-xs whitespace-nowrap">{t("voice_changer.family_badge_experimental")}</span>
            ) : null}
            {fam.zero_shot ? <span className="badge badge-info badge-outline badge-xs whitespace-nowrap">{t("voice_changer.family_badge_zero_shot")}</span> : null}
            {fam.singing ? <span className="badge badge-secondary badge-outline badge-xs whitespace-nowrap">{t("voice_changer.family_badge_singing")}</span> : null}
        </span>
    );
};
