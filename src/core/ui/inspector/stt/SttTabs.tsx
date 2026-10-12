import classNames from "classnames";
import { FC, KeyboardEvent, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { Services } from "@/services-registry";
import { ServiceNetworkState } from "@/types";
import Inspector from "../components";
import { STT_REWRITE_SECTION } from "../../inspectorRoutes";
import { useAppUIStore } from "../../store";

/** The two tabs of the Speech to Text inspector: the speech engine, and Rewrite (an embedded panel, see `inspectorRoutes.ts`). */
export type SttView = "speech" | "rewrite";

const VIEWS: readonly SttView[] = ["speech", "rewrite"];

/** The tab an inspector path opens: `{ tab: "stt", value: "rewrite" }` (the status-bar card) opens Rewrite. */
export function sttViewForSection(section: string | undefined): SttView {
  return section === STT_REWRITE_SECTION ? "rewrite" : "speech";
}

/**
 * A click on one of the tabs. The panel stays open and the store follows it (`setSidebarTab` never toggles), so the
 * status-bar card, the left menu and the toasts act on the tab the user is looking at.
 */
export function selectSttView(view: SttView): void {
  useAppUIStore.getState().setSidebarTab({ tab: Services.stt, value: view === "rewrite" ? STT_REWRITE_SECTION : undefined });
}

/**
 * The tab an arrow key, Home or End moves to; null for any other key. With two tabs both arrows go to the other one,
 * which is also what a right-to-left layout needs, so no direction is read.
 */
export function nextSttView(view: SttView, key: string): SttView | null {
  if (key === "ArrowRight" || key === "ArrowLeft") return VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length];
  if (key === "Home") return VIEWS[0];
  if (key === "End") return VIEWS[VIEWS.length - 1];
  return null;
}

const tabId = (view: SttView) => `stt-tab-${view}`;
const PANEL_ID = "stt-tabpanel";

/** Dot on the Rewrite tab while the Rewrite service is starting (warning) or running (success). */
export const RewriteDot: FC = () => {
  const { t } = useTranslation();
  const { status } = useSnapshot(window.ApiServer.transform.serviceState);
  if (status !== ServiceNetworkState.connected && status !== ServiceNetworkState.connecting) return null;
  const connected = status === ServiceNetworkState.connected;
  return (
    <>
      <span aria-hidden="true" data-testid="rewrite-dot" className={classNames("w-1.5 h-1.5 rounded-full", connected ? "bg-success" : "bg-warning")} />
      <span className="sr-only">{connected ? t("common.status_connected") : t("common.status_connecting")}</span>
    </>
  );
};

/**
 * Speech | Rewrite. A click only changes what the panel shows (`onChange`): it never goes through `changeTab`, which
 * would toggle the panel shut. Roving focus as the tabs pattern asks: Tab reaches the open tab, the arrow keys (and
 * Home / End) move between them, and the panel below is labelled by the open tab.
 */
export const SttTabs: FC<{ view: SttView; onChange: (view: SttView) => void }> = ({ view, onChange }) => {
  const { t } = useTranslation();

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    const next = nextSttView(view, e.key);
    if (!next) return;
    e.preventDefault();
    onChange(next);
    document.getElementById(tabId(next))?.focus();
  };

  const tab = (v: SttView, label: string, description: string, className: string, extra?: ReactNode) => (
    <Inspector.Tab
      id={tabId(v)}
      role="tab"
      aria-selected={view === v}
      aria-controls={PANEL_ID}
      tabIndex={view === v ? 0 : -1}
      className={className}
      tooltip={label}
      tooltipBody={description}
      active={view === v}
      onClick={() => onChange(v)}
      onKeyDown={onKeyDown}
    >
      {label}
      {extra}
    </Inspector.Tab>
  );

  return (
    <Inspector.Tabs role="tablist" aria-label={t("stt.title")}>
      {tab("speech", t("stt.tab_speech"), t("stt.tab_speech_desc"), "!text-sm px-4")}
      {tab("rewrite", t("stt.tab_rewrite"), t("stt.tab_rewrite_desc"), "!text-sm px-4 items-center gap-2", <RewriteDot />)}
    </Inspector.Tabs>
  );
};

/** The panel under the tabs: one element for both, named by the open tab. */
export const SttTabPanel: FC<{ view: SttView; children: ReactNode }> = ({ view, children }) => (
  <div role="tabpanel" id={PANEL_ID} aria-labelledby={tabId(view)}>
    {children}
  </div>
);
