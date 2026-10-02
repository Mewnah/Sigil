import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { describe, expect, it, vi } from "vitest";

/**
 * The helper pages' socket loop against a LAN refusal (P6-D). Sigil refuses a page after the
 * WebSocket upgrade (close 4401 / 4403 at once), so the browser fires `onopen` for a refused socket
 * too: the pages must not announce a connection, or re-announce the refusal, on every 30 s retry.
 * Runs each page's inline script in a VM with a fake DOM, WebSocket and clock.
 */

const LAN_UNAUTHORIZED = 4401;
const LAN_DISABLED = 4403;

class El {
  textContent = "";
  className = "";
  value = "";
  selectedIndex = 0;
  options: unknown[] = [];
  style: Record<string, unknown> = { setProperty() {} };
  children: El[] = [];
  onclick: unknown = null;
  oninput: unknown = null;
  onchange: unknown = null;
  append(...nodes: El[]) {
    this.children.push(...nodes);
  }
  appendChild(node: El) {
    this.children.push(node);
    return node;
  }
  get childElementCount() {
    return this.children.length;
  }
  get firstChild() {
    return this.children.length ? { remove: () => this.children.shift() } : null;
  }
  addEventListener() {}
  remove() {}
  focus() {}
}

class FakeSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  readyState = FakeSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((msg: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  constructor(readonly url: string) {}
  send(data: string) {
    this.sent.push(data);
  }
  close() {}
  /** The upgrade succeeded. */
  accept() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }
  /** Closed by the server with `code`. */
  closeWith(code: number) {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.({ code });
  }
  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

function loadPage(file: string, location: { search: string; hostname: string; port: string; protocol?: string }) {
  const html = readFileSync(fileURLToPath(new URL(`../../../../../public/${file}`, import.meta.url)), "utf8");
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]!);
  const elements = new Map<string, El>();
  const el = (id: string) => {
    if (!elements.has(id)) elements.set(id, new El());
    return elements.get(id)!;
  };
  const sockets: FakeSocket[] = [];
  let now = 0;
  let nextTimer = 1;
  let timers: { id: number; at: number; fn: () => void }[] = [];
  const warn = vi.fn();
  const sandbox: Record<string, unknown> = {
    location: { protocol: "http:", ...location },
    document: { getElementById: el, createElement: () => new El(), documentElement: new El() },
    WebSocket: class extends FakeSocket {
      constructor(url: string) {
        super(url);
        sockets.push(this);
      }
    },
    URLSearchParams,
    setTimeout: (fn: () => void, ms = 0) => {
      const id = nextTimer++;
      timers.push({ id, at: now + ms, fn });
      return id;
    },
    clearTimeout: (id: number | null) => {
      timers = timers.filter((t) => t.id !== id);
    },
    console: { log() {}, error() {}, warn },
    localStorage: { getItem: () => null, setItem() {} },
    addEventListener() {},
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  // The page's main script is its last one (mic.html's first only lists languages).
  vm.runInContext(scripts.at(-1)!, context);
  return {
    el,
    sockets,
    warn,
    socket: () => sockets.at(-1)!,
    advance(ms: number) {
      const end = now + ms;
      for (;;) {
        timers.sort((a, b) => a.at - b.at);
        const next = timers[0];
        if (!next || next.at > end) break;
        timers.shift();
        now = next.at;
        next.fn();
      }
      now = end;
    },
  };
}

const LAN_PAGE = { search: "?port=3030&token=stale0token", hostname: "192.168.1.50", port: "3030" };

describe("mic.html", () => {
  const logs = (page: ReturnType<typeof loadPage>) => page.el("list").children.map((row) => row.children[1]!.textContent);
  const refusals = (page: ReturnType<typeof loadPage>) => logs(page).filter((l) => l.startsWith("Sigil refused"));
  const connects = (page: ReturnType<typeof loadPage>) => logs(page).filter((l) => l === "Connected to Sigil");

  it("reports a refusal once and never 'Connected' for a socket closed at once", () => {
    const page = loadPage("mic.html", LAN_PAGE);
    for (let attempt = 0; attempt < 4; attempt++) {
      const ws = page.socket();
      expect(ws.url).toContain("&token=stale0token");
      ws.accept();
      ws.closeWith(LAN_UNAUTHORIZED);
      page.advance(30_000);
    }
    expect(page.sockets).toHaveLength(5);
    expect(refusals(page)).toHaveLength(1);
    expect(connects(page)).toHaveLength(0);
    expect(page.sockets.flatMap((s) => s.sent)).toEqual([]);
  });

  it("announces a connection once it proved usable, and a later revocation once", () => {
    const page = loadPage("mic.html", LAN_PAGE);
    const ok = page.socket();
    ok.accept();
    expect(connects(page)).toHaveLength(0);
    page.advance(1_500);
    expect(connects(page)).toHaveLength(1);

    // New token in Sigil: this page is told once, and the retries stay quiet.
    ok.closeWith(LAN_UNAUTHORIZED);
    expect(refusals(page)).toHaveLength(1);
    for (let attempt = 0; attempt < 3; attempt++) {
      page.advance(30_000);
      page.socket().accept();
      page.socket().closeWith(LAN_UNAUTHORIZED);
    }
    expect(refusals(page)).toHaveLength(1);
    expect(connects(page)).toHaveLength(1);

    // A message proves the socket at once (no need to wait).
    page.advance(30_000);
    const again = page.socket();
    again.accept();
    again.receive({ topic: "other", data: {} });
    expect(connects(page)).toHaveLength(2);
  });

  it("names This PC only", () => {
    const page = loadPage("mic.html", LAN_PAGE);
    page.socket().accept();
    page.socket().closeWith(LAN_DISABLED);
    expect(logs(page)).toEqual(["Sigil only accepts connections from its own PC (Settings → LAN access)."]);
  });
});

describe("obs-captions.html", () => {
  it("keeps showing the refusal while retrying and says connected only for a usable socket", () => {
    const page = loadPage("obs-captions.html", LAN_PAGE);
    const status = () => page.el("status").textContent;
    expect(status()).toBe("connecting to 192.168.1.50:3030");
    page.socket().accept();
    expect(status()).toBe("connecting to 192.168.1.50:3030");
    page.socket().closeWith(LAN_UNAUTHORIZED);
    const refusal = status();
    expect(refusal).toMatch(/LAN token missing or out of date/);
    for (let attempt = 0; attempt < 3; attempt++) {
      page.advance(30_000);
      expect(status()).toBe(refusal);
      page.socket().accept();
      expect(status()).toBe(refusal);
      page.socket().closeWith(LAN_UNAUTHORIZED);
      expect(status()).toBe(refusal);
    }
    page.advance(30_000);
    page.socket().accept();
    page.advance(1_500);
    expect(status()).toBe("connected 192.168.1.50:3030");
  });
});

describe("inputfield.html", () => {
  it("warns about a refusal once until a connection worked again", () => {
    const page = loadPage("inputfield.html", LAN_PAGE);
    for (let attempt = 0; attempt < 3; attempt++) {
      page.socket().accept();
      page.socket().closeWith(LAN_UNAUTHORIZED);
      page.advance(30_000);
    }
    expect(page.warn).toHaveBeenCalledTimes(1);
    page.socket().accept();
    page.advance(1_500);
    page.socket().closeWith(LAN_DISABLED);
    expect(page.warn).toHaveBeenCalledTimes(2);
    expect(page.warn.mock.calls[1]![0]).toMatch(/own PC/);
  });
});
