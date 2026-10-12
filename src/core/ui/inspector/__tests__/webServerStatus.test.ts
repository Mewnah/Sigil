import { describe, expect, it } from "vitest";
import { webServerStatusLine } from "../webServerStatus";

const status = (state: "starting" | "listening" | "failed" | "stopped", error: string | null = null) => ({ state, port: 3030, error });

describe("webServerStatusLine", () => {
  it("shows the port while the server listens", () => {
    expect(webServerStatusLine(status("listening"))).toEqual({ key: "settings.web_server_status_listening", values: { port: 3030 }, error: false });
    // A LAN switch that did not happen still listens (on the previous mode): not an error line.
    expect(webServerStatusLine(status("listening", "could not listen on 127.0.0.1:3030"))?.error).toBe(false);
  });

  it("tells a server that never started (failed) from one that ended and restarts (stopped)", () => {
    const failed = webServerStatusLine(status("failed", "could not listen on 0.0.0.0:3030: busy"));
    expect(failed).toEqual({ key: "settings.web_server_status_failed", values: { error: "could not listen on 0.0.0.0:3030: busy" }, error: true });
    const stopped = webServerStatusLine(status("stopped", "the local web server stopped unexpectedly"));
    expect(stopped).toEqual({ key: "settings.web_server_status_stopped", values: { error: "the local web server stopped unexpectedly" }, error: true });
    expect(stopped?.key).not.toBe(failed?.key);
  });

  it("falls back to the state name when there is no error text", () => {
    expect(webServerStatusLine(status("failed"))?.values).toEqual({ error: "failed" });
    expect(webServerStatusLine(status("stopped"))?.values).toEqual({ error: "stopped" });
  });

  it("shows nothing while starting or before the first report", () => {
    expect(webServerStatusLine(status("starting"))).toBeNull();
    expect(webServerStatusLine(undefined)).toBeNull();
    expect(webServerStatusLine(null)).toBeNull();
  });
});
