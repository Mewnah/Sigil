import { invoke } from "@tauri-apps/api/core";
import throttle from "lodash/throttle";
import { pushSystemLog } from "@/core/services/systemLog";

export type OscArg =
  | { Bool: boolean }
  | { Float: number }
  | { Int: number }
  | { String: string };

function parseOscPort(raw: unknown): number {
  const n = parseInt(String(raw ?? "9000"), 10);
  if (!Number.isFinite(n) || n < 1 || n > 65535) return 9000;
  return n;
}

function oscDestination(): { host: string; port: number } {
  const d = window.ApiServer.state.services.vrc.data;
  const host = String(d.oscHost ?? "127.0.0.1").trim() || "127.0.0.1";
  return { host, port: parseOscPort(d.oscPort) };
}

function oscInvokeErrorMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  if (e && typeof e === "object" && "message" in e) {
    const m = (e as { message?: unknown }).message;
    if (typeof m === "string") return m;
  }
  return String(e);
}

const logOscSendFailure = throttle(
  (msg: string) => {
    pushSystemLog("VRChat OSC", msg, "warning");
  },
  4000,
  { trailing: false },
);

export function sendOsc(path: string, args: OscArg[]) {
  const { host, port } = oscDestination();
  void invoke("plugin:osc|send", {
    rpc: {
      path,
      args,
      host,
      port,
    },
  }).catch((e: unknown) => {
    logOscSendFailure(`Send failed (${path}): ${oscInvokeErrorMessage(e)}`);
  });
}
