import Logo from "@/core/ui/logo";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import style from "./loading.module.css";

/** Failed connection attempts before the loading view says what may be wrong (the first ones are routine: Sigil still starting). */
export const LOADING_HINT_AFTER_FAILURES = 3;

export default function ClientLoadingView() {
  const { t } = useTranslation();
  // The mirror retries inside the page for as long as it takes, so this view can stay up for hours:
  // after a few failed attempts it says why, instead of pulsing "Connecting" with no clue.
  const mirror = useSnapshot(window.ApiShared.peer.clientMirrorState);
  const network = window.Config?.clientNetwork;
  const stuck = mirror.failures >= LOADING_HINT_AFTER_FAILURES;
  return (
    <div
      data-theme="sigil"
      className={style.screen}
      style={{
        backgroundColor: "transparent",
        color: "#c4b5fd",
        minHeight: "100vh",
      }}
    >
      <div
        className="p-6 py-4 animate-pulse leading-none rounded-box text-2xl flex flex-col gap-2"
        style={{
          backgroundColor: "rgba(0,0,0,0.45)",
          borderRadius: "0.75rem",
          color: "#c4b5fd",
        }}
      >
        <Logo />{" "}
        <span className="flex items-center text-lg gap-2">
          {t("client.loading_connecting")}
        </span>
        {stuck && (
          <span role="status" className="text-sm leading-snug" style={{ maxWidth: "28rem" }}>
            {t("client.loading_still_trying", { address: network ? `${network.host}:${network.port}` : "" })}
            {mirror.lastError ? ` ${t("client.peer_error_detail", { detail: mirror.lastError })}` : null}
          </span>
        )}
      </div>
    </div>
  );
}
