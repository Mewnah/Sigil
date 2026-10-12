import type { WebServerStatus } from "@/shared/services/pubsub";

/** One line of the Settings "Local web server" section: the string key, its values, and whether it is an error. */
export type WebServerStatusLine = { key: string; values: Record<string, string | number>; error: boolean };

/**
 * The Settings line for the Rust supervisor's last report. `failed` (no bind worked yet, Sigil keeps trying) and
 * `stopped` (a running server ended, Sigil starts it again) are told apart, because the first needs the user's help
 * if it lasts and the second does not. Nothing while `starting`, or before the first report.
 */
export function webServerStatusLine(status: Pick<WebServerStatus, "state" | "port" | "error"> | null | undefined): WebServerStatusLine | null {
  if (!status)
    return null;
  switch (status.state) {
    case "listening":
      return { key: "settings.web_server_status_listening", values: { port: status.port }, error: false };
    case "failed":
      return { key: "settings.web_server_status_failed", values: { error: status.error ?? status.state }, error: true };
    case "stopped":
      return { key: "settings.web_server_status_stopped", values: { error: status.error ?? status.state }, error: true };
    default:
      return null;
  }
}
