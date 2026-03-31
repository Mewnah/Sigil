import Logo from "@/core/ui/logo";
import { useTranslation } from "react-i18next";
import style from "./loading.module.css";

export default function ClientLoadingView() {
  const { t } = useTranslation();
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
      </div>
    </div>
  );
}
