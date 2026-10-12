import type { ReactNode } from "react";
import { MirrorRefusedError, MirrorStartError } from "@/shared/services/peer/errors";

/**
 * What a page shows when start-up fails (`src/index.tsx`). The host window always gets the error
 * panel. A `/client` mirror is usually an OBS browser source with nobody to refresh it, where a
 * panel on stream is worse than nothing, but it is also opened in ordinary browsers (a phone, a
 * second PC) where a blank page that reloads itself hides the reason:
 *
 * - the host refused the link (LAN token), or the page can never work (no WebRTC, a bad host or
 *   port): a reload cannot change it, so the panel stays, in OBS too;
 * - anything else may be transient (Sigil restarting while the page loaded): the page reloads
 *   itself with a growing delay. In OBS it stays transparent meanwhile; in a browser the panel
 *   stays up and says when the next try is.
 */

/** OBS's browser source (CEF) injects `window.obsstudio`; a normal browser tab has none. */
export function isObsBrowserSource(win: object = window): boolean {
  return typeof (win as { obsstudio?: unknown }).obsstudio !== "undefined";
}

export type StartupFailureKind = "host" | "refused" | "final" | "maybe-transient";

export function classifyStartupFailure(error: unknown, isClient: boolean): StartupFailureKind {
  if (!isClient) return "host";
  if (error instanceof MirrorRefusedError) return "refused";
  if (error instanceof MirrorStartError) return "final";
  return "maybe-transient";
}

export interface StartupFailureEnv {
  isClient: boolean;
  /** Where the mirror connects (the panel names it). */
  clientNetwork?: { host: string; port: string };
  /** The page runs in OBS's browser source. */
  obs: boolean;
  renderView(view: ReactNode): void;
  /** Reload the page later; the delay in ms, or null when a reload is already pending. */
  scheduleReload(reason: string, error: unknown): number | null;
}

/** Show (or withhold) the panel and queue the reload that fits the failure. Returns what it did. */
export function handleStartupFailure(error: unknown, env: StartupFailureEnv): "panel" | "blank" {
  const kind = classifyStartupFailure(error, env.isClient);
  const client = env.isClient && env.clientNetwork ? { host: env.clientNetwork.host, port: env.clientNetwork.port } : undefined;
  if (kind === "maybe-transient") {
    const delay = env.scheduleReload("the mirror failed to start", error);
    if (env.obs) {
      env.renderView(null);
      return "blank";
    }
    env.renderView(<StartupErrorView error={error} client={client} retryInSeconds={delay === null ? undefined : Math.round(delay / 1000)} />);
    return "panel";
  }
  env.renderView(<StartupErrorView error={error} client={client} />);
  return "panel";
}

/** Shown when initialization fails; every dynamic value is rendered as text. */
export function StartupErrorView({ error, client, retryInSeconds }: { error: any; client?: { host: string; port: string }; retryInSeconds?: number }) {
  const address = client ? `${client.host}:${client.port}` : "";
  return (
    <div style={{ padding: 24, color: "#f87171", background: "#1a1a1a", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ color: "#fff", marginTop: 0 }}>Application failed to start</h1>
      {client && (
        <>
          <p style={{ color: "#ccc", margin: "12px 0" }}>
            The OBS browser source could not connect to Sigil (PeerJS). Keep the desktop app running on this PC, use{" "}
            <code>{`http://${address}/client`}</code> (same pattern as Curses), and check that nothing blocks WebSockets
            to that host and port.
          </p>
          <p style={{ color: "#888", fontSize: 13 }}>
            Trying Peer server: <code>{address}</code>
          </p>
        </>
      )}
      {retryInSeconds !== undefined && (
        <p style={{ color: "#ccc", margin: "12px 0" }}>This page reloads itself and tries again in {retryInSeconds} s.</p>
      )}
      <pre style={{ color: "#fca5a5", whiteSpace: "pre-wrap", fontSize: 12 }}>
        {[String(error?.message || error), String(error?.stack || "")].join("\n")}
      </pre>
    </div>
  );
}
