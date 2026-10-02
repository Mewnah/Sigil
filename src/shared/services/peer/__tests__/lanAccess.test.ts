import { afterEach, describe, expect, it } from "vitest";
import {
  formatLinkAddress,
  groupLanToken,
  isValidLanToken,
  LAN_TOKEN_LENGTH,
  lanClientUrl,
  lanLinkAddress,
  lanRefusalFromCloseCode,
  lanTokenFromQuery,
  linkSocketUrl,
  maskLanToken,
  normalizeLanToken,
  parseLinkAddress,
  splitLinkAddress,
  WS_CLOSE_LAN_DISABLED,
  WS_CLOSE_LAN_UNAUTHORIZED,
} from "../lanAccess";
import { localPageUrl } from "@/utils/localPages";

const TOKEN = "0123456789abcdefghjkmnpqrs";

describe("LAN token format (same vectors as Rust web::auth normalizes_typed_tokens)", () => {
  it("normalizes typed tokens like the server", () => {
    const cases: [string, string | null][] = [
      ["0123456789abcdefghjkmnpqrs", "0123456789abcdefghjkmnpqrs"],
      ["0123456789ABCDEFGHJKMNPQRS", "0123456789abcdefghjkmnpqrs"],
      ["01234-56789-abcde-fghjk-mnpqr-s", "0123456789abcdefghjkmnpqrs"],
      [" 0123456789abcdefghjkmnpqrs\t", "0123456789abcdefghjkmnpqrs"],
      ["O1234567890abcdefghjkmnpqr", "01234567890abcdefghjkmnpqr"],
      ["Il23456789abcdefghjkmnpqrs", "1123456789abcdefghjkmnpqrs"],
      ["0123456789abcdefghjkmnpqr", null],
      ["0123456789abcdefghjkmnpqrst", null],
      ["0123456789abcdefghjkmnpqru", null],
      ["0123456789abcdefghjkmnpq+s", null],
      ["0123456789abcdefghjkmnpqré", null],
      ["", null],
    ];
    for (const [raw, expected] of cases) expect(normalizeLanToken(raw), raw).toBe(expected);
    expect(normalizeLanToken("0".repeat(129))).toBeNull();
    expect(normalizeLanToken(`${TOKEN}${"-".repeat(128)}`)).toBeNull();
    expect(normalizeLanToken(null)).toBeNull();
    expect(normalizeLanToken(undefined)).toBeNull();
    expect(normalizeLanToken(`${TOKEN.slice(0, 25)}😀`)).toBeNull();
    expect(isValidLanToken(TOKEN)).toBe(true);
    expect(isValidLanToken("short")).toBe(false);
    expect(TOKEN).toHaveLength(LAN_TOKEN_LENGTH);
  });

  it("groups and masks for display", () => {
    expect(groupLanToken(TOKEN)).toBe("01234-56789-abcde-fghjk-mnpqr-s");
    expect(normalizeLanToken(groupLanToken(TOKEN))).toBe(TOKEN);
    const masked = maskLanToken(TOKEN);
    expect(masked.startsWith("0123")).toBe(true);
    expect(masked.endsWith("pqrs")).toBe(true);
    expect(masked).not.toContain("abcdefgh");
    expect(maskLanToken("abc")).toBe("•••");
  });

  it("reads the token from a page query", () => {
    expect(lanTokenFromQuery(`?host=localhost&token=${TOKEN.toUpperCase()}`)).toBe(TOKEN);
    expect(lanTokenFromQuery(new URLSearchParams({ token: groupLanToken(TOKEN) }))).toBe(TOKEN);
    expect(lanTokenFromQuery("?token=nope")).toBe("");
    expect(lanTokenFromQuery("")).toBe("");
  });

  it("maps the server's close codes", () => {
    expect(WS_CLOSE_LAN_UNAUTHORIZED).toBe(4401);
    expect(WS_CLOSE_LAN_DISABLED).toBe(4403);
    expect(lanRefusalFromCloseCode(4401)).toBe("token");
    expect(lanRefusalFromCloseCode(4403)).toBe("lan_disabled");
    expect(lanRefusalFromCloseCode(1006)).toBeNull();
    expect(lanRefusalFromCloseCode(undefined)).toBeNull();
  });
});

describe("LAN links", () => {
  it("builds /client links for other devices with the token", () => {
    expect(lanClientUrl("192.168.1.50", "3030", TOKEN)).toBe(`http://192.168.1.50:3030/client?token=${TOKEN}`);
    expect(lanClientUrl("192.168.1.50", 3030, TOKEN.toUpperCase(), "client")).toBe(`http://192.168.1.50:3030/client?token=${TOKEN}`);
    // No (valid) token yet: the link still works on this PC or in legacy mode.
    expect(lanClientUrl("192.168.1.50", 3030, "")).toBe("http://192.168.1.50:3030/client");
    expect(lanClientUrl("fe80::50", 3030, TOKEN)).toBe(`http://[fe80::50]:3030/client?token=${TOKEN}`);
  });

  it("appends the token to local pages only when given", () => {
    const g = globalThis as { window?: unknown };
    g.window = { Config: { serverNetwork: { ip: "192.168.1.50", host: "localhost", port: "3030" } } };
    try {
      // This PC: no token (unchanged URLs).
      expect(localPageUrl("obs-captions.html", { params: { mode: "plain" } })).toBe(
        "http://localhost:3030/obs-captions.html?port=3030&mode=plain"
      );
      expect(localPageUrl("mic.html", { token: "" })).toBe("http://127.0.0.1:3030/mic.html?port=3030");
      // Another device: its LAN address plus the token, last; a `token` in params never wins.
      expect(localPageUrl("obs-captions.html", { host: "192.168.1.50", token: TOKEN, params: { size: "48", token: "x" } })).toBe(
        `http://192.168.1.50:3030/obs-captions.html?port=3030&size=48&token=${TOKEN}`
      );
    } finally {
      delete g.window;
    }
  });
});

describe("Link apps address", () => {
  it("parses addresses with and without the remote token", () => {
    expect(parseLinkAddress("192.168.1.50:3030")).toEqual({ host: "192.168.1.50", port: "3030", token: "", tokenInvalid: false });
    expect(parseLinkAddress(` 192.168.1.50:3030?token=${TOKEN.toUpperCase()} `)).toEqual({
      host: "192.168.1.50",
      port: "3030",
      token: TOKEN,
      tokenInvalid: false,
    });
    // Pasted LAN links and URLs work too.
    expect(parseLinkAddress(`http://192.168.1.50:3030/client?token=${TOKEN}`)?.token).toBe(TOKEN);
    expect(parseLinkAddress(`ws://10.0.0.2:4040/pubsub?id=x&token=${TOKEN}`)).toMatchObject({ host: "10.0.0.2", port: "4040", token: TOKEN });
    expect(parseLinkAddress(`10.0.0.2:4040#token=${groupLanToken(TOKEN)}`)?.token).toBe(TOKEN);
    expect(parseLinkAddress("10.0.0.2:4040?token=garbage")).toEqual({ host: "10.0.0.2", port: "4040", token: "", tokenInvalid: true });
    expect(parseLinkAddress("10.0.0.2:4040?token=")?.tokenInvalid).toBe(false);
  });

  it("refuses what the old validator refused", () => {
    for (const bad of ["", "192.168.1.50", "192.168.1.256:3030", "192.168.1:3030", "host.local:3030", "192.168.1.50:0", "192.168.1.50:70000", "192.168.1.50:3030x", "ftp://192.168.1.50:3030", "[::1]:3030"])
      expect(parseLinkAddress(bad), bad).toBeNull();
  });

  it("stores the token in the address and splits it back for the two fields", () => {
    const stored = formatLinkAddress("192.168.1.50:3030", groupLanToken(TOKEN));
    expect(stored).toBe(`192.168.1.50:3030?token=${TOKEN}`);
    expect(splitLinkAddress(stored)).toEqual({ address: "192.168.1.50:3030", token: TOKEN });
    expect(formatLinkAddress("192.168.1.50:3030", "")).toBe("192.168.1.50:3030");
    // An address that already had a token gets the new one, not two.
    expect(formatLinkAddress(stored, "")).toBe("192.168.1.50:3030");
    // A half-typed token is kept as typed (the field shows it) and flagged by the parser.
    const typed = formatLinkAddress("192.168.1.50:3030", "abc");
    expect(splitLinkAddress(typed)).toEqual({ address: "192.168.1.50:3030", token: "abc" });
    expect(parseLinkAddress(typed)?.tokenInvalid).toBe(true);
    expect(splitLinkAddress("")).toEqual({ address: "", token: "" });
    expect(lanLinkAddress("192.168.1.50", 3030, TOKEN)).toBe(`192.168.1.50:3030?token=${TOKEN}`);
    expect(lanLinkAddress("192.168.1.50", "3030", "")).toBe("192.168.1.50:3030");
  });

  it("puts the token on the pubsub socket URL", () => {
    expect(linkSocketUrl({ host: "10.0.0.2", port: "3030", token: TOKEN }, "abc-1")).toBe(`ws://10.0.0.2:3030/pubsub?id=abc-1&token=${TOKEN}`);
    expect(linkSocketUrl({ host: "10.0.0.2", port: "3030", token: "" }, "abc-1")).toBe("ws://10.0.0.2:3030/pubsub?id=abc-1");
  });
});

afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});
