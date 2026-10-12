import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";

/**
 * Where a profile in "This PC only" (`lanAccess.mode = "localhost"`, the default of a new install) learns how to use a
 * phone or a second PC: next to the things that only work from another device (the OBS inspector's second-PC and phone
 * link, Link apps). A short muted note, worded as the next step rather than as a fault: OBS on this PC needs nothing.
 * Renders nothing once LAN access is on. Names the setting and its value with the labels of the switch itself, so the hint
 * and the switch cannot drift apart in any language.
 */
export const LanOffHint: FC = () => {
  const { t } = useTranslation();
  const { mode } = useSnapshot(window.ApiServer.state.lanAccess);
  if (mode !== "localhost") return null;
  return (
    <p className="text-xs text-base-content/60 leading-snug">
      {t("settings.lan_off_hint", {
        field: t("settings.field_lan_mode"),
        option: t("settings.lan_mode_lan"),
        section: t("settings.section_lan_access"),
      })}
    </p>
  );
};
