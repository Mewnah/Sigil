import { localPageUrl } from "@/utils/localPages";
import { invoke } from "@tauri-apps/api/core";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { SiGooglechrome, SiMicrosoftedge } from "react-icons/si";
import { toast } from "react-toastify";
import Inspector from "../components";

/**
 * mic.html opens in an external Chrome/Edge on 127.0.0.1 (a secure context for getUserMedia). Same-PC
 * pages never need the LAN token (P6-D), so the URL (which lands in the browser's history and the
 * process command line) carries none; the page forwards a `?token=` only if one is added by hand.
 */
const micPageUrl = () => localPageUrl("mic.html");

/** Chrome / Edge: Web Speech in an external browser window (mic.html). */
export const ExternalBrowser: FC<{ browser: "chrome" | "edge" }> = ({ browser }) => {
  const { t } = useTranslation();
  const handleOpen = () => {
    void invoke("plugin:web|open_browser", {
      data: {
        browser,
        url: micPageUrl()
      }
    }).catch((e) => {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("open_browser:", e);
      toast.error(`${t("integrations.oauth_open_browser_failed")} ${msg}`);
    });
  };

  return <>
    <Inspector.SubHeader>{t('stt.browser_title')}</Inspector.SubHeader>
    <button className="btn btn-sm btn-neutral gap-2" onClick={handleOpen}>
      {browser === "chrome" ? <><SiGooglechrome /> {t("stt.browser_open_chrome")}</> : <><SiMicrosoftedge /> {t("stt.browser_open_edge")}</>}
    </button>
    <Inspector.Description>{t("stt.browser_keep_open", { browser: browser === "chrome" ? "Chrome" : "Edge" })}</Inspector.Description>
  </>
}
