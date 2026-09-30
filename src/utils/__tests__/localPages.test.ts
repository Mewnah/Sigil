import { afterEach, describe, expect, it } from "vitest";
import { clientQueryWithoutForeignServer, clientServerAddress, isLoopbackHost, localPageUrl } from "../localPages";

const g = globalThis as { window?: unknown };

afterEach(() => {
  delete g.window;
});

function withServerNetwork(host: string, port: string) {
  g.window = { Config: { serverNetwork: { ip: "192.168.1.50", host, port } } };
}

describe("localPageUrl", () => {
  it("matches the hand-built URLs it replaces", () => {
    withServerNetwork("localhost", "3030");
    // inspector/stt/ExternalMic.tsx micPageUrl()
    expect(localPageUrl("mic.html")).toBe("http://127.0.0.1:3030/mic.html?port=3030");
    // ObsBrowserSourceService.getCaptionsLink()
    expect(localPageUrl("obs-captions.html", { params: { mode: "plain", size: "48", lines: undefined } })).toBe(
      "http://localhost:3030/obs-captions.html?port=3030&mode=plain&size=48"
    );
    expect(localPageUrl("inputfield.html")).toBe("http://localhost:3030/inputfield.html?port=3030");
  });

  it("honours explicit host and port and never lets params override the port", () => {
    withServerNetwork("localhost", "3030");
    expect(localPageUrl("obs-captions.html", { host: "192.168.1.50", port: 4040, params: { port: "1" } })).toBe(
      "http://192.168.1.50:4040/obs-captions.html?port=4040"
    );
  });

  it("brackets IPv6 hosts and encodes params", () => {
    expect(localPageUrl("mic.html", { host: "::1", port: "3030", params: { lang: "en US&x=1" } })).toBe(
      "http://[::1]:3030/mic.html?port=3030&lang=en+US%26x%3D1"
    );
    expect(localPageUrl("mic.html", { host: "[::1]", port: 3030 })).toBe("http://[::1]:3030/mic.html?port=3030");
  });

  it("works without window (node) given a port", () => {
    expect(localPageUrl("inputfield.html", { port: 3030 })).toBe("http://localhost:3030/inputfield.html?port=3030");
  });
});

describe("isLoopbackHost", () => {
  it("accepts loopback names and addresses only", () => {
    for (const h of ["localhost", "LOCALHOST", "localhost.", "sigil.localhost", "a.b.localhost", "127.0.0.1", "127.5.6.7", "::1", "[::1]", "[0:0:0:0:0:0:0:1]"])
      expect(isLoopbackHost(h), h).toBe(true);
    for (const h of [
      "evil.example",
      "localhost.evil.example",
      "evillocalhost",
      "-x.localhost",
      "localhost@evil.example",
      "127.0.0.1.evil.example",
      "127.0.0.256",
      "128.0.0.1",
      "192.168.1.50",
      "0.0.0.0",
      "::2",
      "localtest.me",
      "",
    ])
      expect(isLoopbackHost(h), h).toBe(false);
  });
});

describe("clientServerAddress (/client connects only to its own server)", () => {
  const obs = { hostname: "localhost", port: "3030" };
  const lan = { hostname: "192.168.1.50", port: "3030" };
  const dev = { hostname: "localhost", port: "1420" };

  it("defaults to the server that served the page", () => {
    expect(clientServerAddress("", obs)).toEqual({ host: "localhost", port: "3030", ignored: [] });
    expect(clientServerAddress("?id=abc", lan)).toEqual({ host: "192.168.1.50", port: "3030", ignored: [] });
  });

  it("keeps old Curses links and loopback targets", () => {
    expect(clientServerAddress("?host=localhost&port=3030&id=x", obs)).toEqual({ host: "localhost", port: "3030", ignored: [] });
    expect(clientServerAddress("?host=127.0.0.1&port=3030", dev)).toEqual({ host: "127.0.0.1", port: "3030", ignored: [] });
    expect(clientServerAddress("?host=192.168.1.50&port=3031", lan)).toEqual({ host: "192.168.1.50", port: "3031", ignored: [] });
    expect(clientServerAddress("?port=3031", obs)).toEqual({ host: "localhost", port: "3031", ignored: [] });
  });

  it("ignores a foreign host together with its port", () => {
    expect(clientServerAddress("?host=evil.example&port=80", obs)).toEqual({ host: "localhost", port: "3030", ignored: ["host", "port"] });
    expect(clientServerAddress("?host=192.168.1.77&port=3030", lan)).toEqual({ host: "192.168.1.50", port: "3030", ignored: ["host", "port"] });
    expect(clientServerAddress("?host=localhost.evil.example", obs).ignored).toEqual(["host"]);
  });

  it("ignores ports that are not port numbers (URL authority injection)", () => {
    for (const port of ["3030@evil.example", "80/x", "0", "65536", "", "30 30"])
      expect(clientServerAddress(`?port=${encodeURIComponent(port)}`, obs), port).toEqual({ host: "localhost", port: "3030", ignored: ["port"] });
  });

  it("removes only the ignored parameters from the query", () => {
    expect(clientQueryWithoutForeignServer("?host=evil.example&port=80&id=abc", obs)).toBe("?id=abc");
    expect(clientQueryWithoutForeignServer("?host=evil.example", obs)).toBe("");
    expect(clientQueryWithoutForeignServer("?host=localhost&port=3030", obs)).toBeNull();
    expect(clientQueryWithoutForeignServer("", obs)).toBeNull();
    expect(clientQueryWithoutForeignServer("?port=x&id=1", obs)).toBe("?id=1");
  });
});
