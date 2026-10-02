import { describe, expect, it } from "vitest";
import {
  encodeWire,
  INBOUND_TEXT_TOPICS,
  MAX_VIA_HOPS,
  neutralizeMarkup,
  newMessageId,
  nextVia,
  RecentMessageIds,
  routeInboundMessage,
  sanitizeClientPubSubFrame,
  sanitizeEmotes,
  type InboundRoute,
} from "../inbound";

const SELF = "host-self";
const OTHER = "host-other";

const msg = (topic: unknown, data: unknown, via?: unknown, mid?: unknown) =>
  JSON.stringify({ topic, data, ...(via !== undefined && { via }), ...(mid !== undefined && { mid }) });

const final = (value: string) => ({ type: 0, value });
const caption = { v: 1, source: "text.stt", type: "final", value: "hello", at: 1 };

function kind(route: InboundRoute) {
  return route.kind === "ignore" ? `ignore:${route.reason}` : route.kind;
}

describe("routeInboundMessage", () => {
  it("routes mic.html speech into the host STT pipeline", () => {
    const route = routeInboundMessage(msg("text.stt", { type: 1, value: "hel", emotes: {} }), "local", SELF);
    expect(route).toEqual({ kind: "external_stt", event: { type: 1, value: "hel" } });
  });

  it("accepts the mic lifecycle only from local pages", () => {
    expect(routeInboundMessage(msg("stt.external.state", { active: true }), "local", SELF)).toEqual({
      kind: "mic_state",
      active: true,
    });
    expect(kind(routeInboundMessage(msg("stt.external.state", { active: true }), "link", SELF))).toBe(
      "ignore:topic_not_allowed"
    );
    expect(kind(routeInboundMessage(msg("stt.external.state", { active: true }, [OTHER]), "local", SELF))).toBe(
      "ignore:topic_not_allowed"
    );
    expect(kind(routeInboundMessage(msg("stt.external.state", { active: "yes" }), "local", SELF))).toBe(
      "ignore:invalid_data"
    );
  });

  it("relays inputfield text and other allow-listed text topics", () => {
    for (const topic of INBOUND_TEXT_TOPICS) {
      if (topic === "text.stt") continue;
      const route = routeInboundMessage(msg(topic, final("hi")), "local", SELF);
      expect(route).toEqual({ kind: "text", topic, data: { type: 0, value: "hi" }, via: [] });
    }
  });

  it("validates obs.caption envelopes", () => {
    expect(routeInboundMessage(msg("obs.caption", caption), "link", SELF)).toEqual({
      kind: "caption",
      topic: "obs.caption",
      data: caption,
      via: [],
    });
    expect(kind(routeInboundMessage(msg("obs.caption", { ...caption, v: 2 }), "local", SELF))).toBe(
      "ignore:invalid_data"
    );
  });

  it("drops topics outside the allow-list", () => {
    for (const topic of [
      "text",
      "stt.external.command",
      "scenes:change",
      "peers:init_data",
      "text.stt.extra",
      "others.stt",
      "",
      42,
      null,
    ]) {
      expect(kind(routeInboundMessage(msg(topic, final("x")), "local", SELF)), String(topic)).toBe(
        "ignore:topic_not_allowed"
      );
    }
  });

  it("drops malformed messages", () => {
    expect(kind(routeInboundMessage("not json", "local", SELF))).toBe("ignore:not_json");
    expect(kind(routeInboundMessage("[1,2]", "local", SELF))).toBe("ignore:not_json");
    expect(kind(routeInboundMessage("null", "local", SELF))).toBe("ignore:not_json");
    expect(kind(routeInboundMessage(msg("text.textfield", "hi"), "local", SELF))).toBe("ignore:invalid_data");
    expect(kind(routeInboundMessage(msg("text.textfield", [final("x")]), "local", SELF))).toBe(
      "ignore:invalid_data"
    );
    expect(kind(routeInboundMessage(msg("text.textfield", { type: 7, value: "x" }), "local", SELF))).toBe(
      "ignore:invalid_data"
    );
    expect(kind(routeInboundMessage(msg("__sigil_keepalive", {}), "link", SELF))).toBe("ignore:keepalive");
  });

  it("drops its own echoes and validates via", () => {
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), [OTHER, SELF]), "link", SELF))).toBe(
      "ignore:own_echo"
    );
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), "nope"), "local", SELF))).toBe(
      "ignore:bad_via"
    );
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), [1]), "local", SELF))).toBe("ignore:bad_via");
    const tooLong = Array.from({ length: MAX_VIA_HOPS + 1 }, (_, i) => `h${i}`);
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), tooLong), "local", SELF))).toBe(
      "ignore:bad_via"
    );
  });

  it("never feeds another Sigil host's STT into ours (Link apps loop)", () => {
    // Another host linked to us and publishing its own STT.
    const viaLocal = routeInboundMessage(msg("text.stt", final("hi"), [OTHER]), "local", SELF);
    expect(viaLocal).toMatchObject({ kind: "text", topic: "text.stt", via: [OTHER] });
    // The host we linked to, broadcasting its STT (old versions send no via).
    const fromLink = routeInboundMessage(msg("text.stt", final("hi")), "link", SELF);
    expect(fromLink).toMatchObject({ kind: "text", topic: "text.stt", via: [] });
  });
});

describe("via chain", () => {
  it("appends this host and bounds the chain", () => {
    expect(nextVia([], SELF)).toEqual([SELF]);
    expect(nextVia([OTHER], SELF)).toEqual([OTHER, SELF]);
    expect(nextVia([SELF, OTHER], SELF)).toEqual([OTHER, SELF]);
    const long = Array.from({ length: MAX_VIA_HOPS }, (_, i) => `h${i}`);
    const next = nextVia(long, SELF);
    expect(next).toHaveLength(MAX_VIA_HOPS);
    expect(next.at(-1)).toBe(SELF);
  });

  it("encodes the wire format pages already parse", () => {
    const wire = encodeWire("text.stt", JSON.stringify(final("a\"b")), [SELF]);
    expect(JSON.parse(wire)).toEqual({ topic: "text.stt", data: { type: 0, value: 'a"b' }, via: [SELF] });
  });

  it("stops a two-host Link loop after one hop", () => {
    // A publishes; B relays with via [A, B]; the relay reaching A again is its own echo.
    const fromA = encodeWire("text.translation", JSON.stringify(final("hola")), ["A"]);
    const atB = routeInboundMessage(fromA, "local", "B");
    expect(atB.kind).toBe("text");
    const relayed = encodeWire("text.translation", JSON.stringify(final("hola")), nextVia(atB.kind === "text" ? atB.via : [], "B"));
    expect(kind(routeInboundMessage(relayed, "link", "A"))).toBe("ignore:own_echo");
  });
});

describe("peer text sanitization (host webview has IPC)", () => {
  const ZWSP = "\u200B";

  it("drops peer emote maps so a quote/onerror URL never reaches the text element", () => {
    const attack = msg("text.textfield", { type: 0, value: "hi", emotes: { hi: 'x" onerror="alert(1)', 0: "javascript:alert(1)" } });
    const route = routeInboundMessage(attack, "local", SELF);
    expect(route.kind).toBe("text");
    if (route.kind !== "text") return;
    expect(route.data).toEqual({ type: 0, value: "hi" });
    expect("emotes" in route.data).toBe(false);
    // Same for mic.html speech and for another Sigil host's relayed text.
    const stt = routeInboundMessage(msg("text.stt", { type: 0, value: "hi", emotes: { hi: "x" } }), "local", SELF);
    expect(stt).toEqual({ kind: "external_stt", event: { type: 0, value: "hi" } });
    const linked = routeInboundMessage(msg("text.translation", { type: 0, value: "hi", emotes: { hi: "x" } }, [OTHER]), "link", SELF);
    expect(linked).toMatchObject({ kind: "text", data: { type: 0, value: "hi" } });
    expect(linked.kind === "text" && "emotes" in linked.data).toBe(false);
  });

  it("neutralizes tags in peer text (word-animation path writes it as HTML)", () => {
    const payload = '<img src=x onerror="alert(1)"> and <svg/onload=alert(1)>';
    const route = routeInboundMessage(msg("text.textfield", final(payload)), "local", SELF);
    expect(route.kind).toBe("text");
    if (route.kind !== "text") return;
    expect(route.data.value).toBe(`<${ZWSP}img src=x onerror="alert(1)"> and <${ZWSP}svg/onload=alert(1)>`);
    expect(route.data.value).not.toMatch(/<[A-Za-z/!?]/);
    const stt = routeInboundMessage(msg("text.stt", { type: 1, value: "<b>hi</b>" }), "local", SELF);
    expect(stt).toEqual({ kind: "external_stt", event: { type: 1, value: `<${ZWSP}b>hi<${ZWSP}/b>` } });
  });

  it("neutralizeMarkup leaves ordinary text alone and is idempotent", () => {
    for (const plain of ["hello", "<3", "a < b", "x<1", "<", "<<", "5 <= 6", "fish & chips > chips"])
      expect(neutralizeMarkup(plain), plain).toBe(plain);
    for (const markup of ["<a href=x>", "</script>", "<!-- x -->", "<?xml ?>", "<IMG SRC=x>"]) {
      const once = neutralizeMarkup(markup);
      expect(once, markup).not.toMatch(/<[A-Za-z/!?]/);
      expect(neutralizeMarkup(once)).toBe(once);
      expect(once.replaceAll(ZWSP, "")).toBe(markup);
    }
  });
});

describe("sanitizeEmotes (client side, host-scanned emotes)", () => {
  it("keeps emote CDN URLs", () => {
    const emotes = {
      0: "https://static-cdn.jtvnw.net/emoticons/v2/emotesv2_abc123/default/dark/1.0",
      Kappa: "https://static-cdn.jtvnw.net/emoticons/v2/25/default/dark/1.0",
      catJAM: "https://cdn.betterttv.net/emote/5f1b0186cf6d2144653d2970/1x",
      "7": "//cdn.7tv.app/emote/60ae958e229664e8667aea38/2x.webp",
      OMEGALUL: "https://cdn.frankerfacez.com/emote/128054/1",
      kick: "https://files.kick.com/emotes/37226/fullsize",
    };
    expect(sanitizeEmotes(emotes)).toEqual(emotes);
  });

  it("drops URLs that could add attributes, tags or scripts", () => {
    for (const url of [
      'x" onerror="alert(1)',
      'https://cdn.example/x" onerror="alert(1)',
      "https://cdn.example/x' onerror='alert(1)",
      "https://cdn.example/x onerror=alert(1)",
      "https://cdn.example/x/onerror=alert(1)",
      "https://cdn.example/x?a=b",
      "https://cdn.example/x#y",
      "https://cdn.example/<svg>",
      "https://cdn.example/`x`",
      "https://cdn.example/a\\b",
      "javascript:alert(1)",
      "data:image/svg+xml,<svg onload=alert(1)>",
      "http://cdn.example/x.png",
      "https://user@cdn.example/x",
      "//",
      "",
      `https://cdn.example/${"a".repeat(600)}`,
    ]) {
      expect(sanitizeEmotes({ k: url }), url).toEqual({});
    }
    expect(sanitizeEmotes({ k: 42 })).toEqual({});
    expect(sanitizeEmotes(null)).toEqual({});
    expect(sanitizeEmotes("x")).toEqual({});
    expect(sanitizeEmotes([["k", "https://a.b/c"]])).toEqual({});
    expect(sanitizeEmotes({ ["k".repeat(101)]: "https://a.b/c" })).toEqual({});
  });

  it("bounds the number of entries", () => {
    const many = Object.fromEntries(Array.from({ length: 400 }, (_, i) => [`e${i}`, `https://cdn.example/${i}`]));
    expect(Object.keys(sanitizeEmotes(many))).toHaveLength(256);
  });
});

describe("sanitizeClientPubSubFrame (/client PeerJS bridge)", () => {
  it("sanitizes text frames and passes other topics through", () => {
    const frame = {
      topic: "text.stt",
      data: { type: 0, value: "<img src=x onerror=alert(1)> Kappa", emotes: { Kappa: "https://cdn.example/25/1.0", bad: 'x" onerror="y' } },
    };
    expect(sanitizeClientPubSubFrame(frame)).toEqual({
      topic: "text.stt",
      data: { type: 0, value: "<\u200Bimg src=x onerror=alert(1)> Kappa", emotes: { Kappa: "https://cdn.example/25/1.0" } },
    });
    const scene = { topic: "scenes:change", data: "main" };
    expect(sanitizeClientPubSubFrame(scene)).toEqual(scene);
    const caption = { topic: "obs.caption", data: { v: 1 } };
    expect(sanitizeClientPubSubFrame(caption)).toEqual(caption);
  });

  it("rejects malformed frames", () => {
    expect(sanitizeClientPubSubFrame(null)).toBeNull();
    expect(sanitizeClientPubSubFrame("text.stt")).toBeNull();
    expect(sanitizeClientPubSubFrame({ topic: 1, data: {} })).toBeNull();
    expect(sanitizeClientPubSubFrame({ topic: "text.stt", data: "hi" })).toBeNull();
    expect(sanitizeClientPubSubFrame({ topic: "text.textfield", data: { type: 9, value: "x" } })).toBeNull();
  });

  // P6-S0: other players' captions and readings reach /client canvases too, validated like text.
  it("sanitizes other players' captions like text", () => {
    const frame = { topic: "others.stt", data: { type: 1, value: "<b>hi</b>", emotes: { x: "javascript:alert(1)" }, tags: ["emotion:happy"] } };
    expect(sanitizeClientPubSubFrame(frame)).toEqual({
      topic: "others.stt",
      data: { type: 1, value: "<​b>hi<​/b>", emotes: {}, tags: ["emotion:happy"] },
    });
    expect(sanitizeClientPubSubFrame({ topic: "others.translation", data: { type: 0 } })).toEqual({
      topic: "others.translation",
      data: { type: 0, value: "", emotes: {} },
    });
    expect(sanitizeClientPubSubFrame({ topic: "others.stt", data: "hi" })).toBeNull();
  });

  it("validates readings and neutralizes every string", () => {
    const data = {
      utteranceId: "mic-abc:1:7",
      source: "others.stt",
      lang: "ja",
      value: "<i>漢字</i>",
      hiragana: "<s>かんじ",
      segments: [{ text: "漢字", reading: "<u>かんじ" }],
    };
    const out = sanitizeClientPubSubFrame({ topic: "readings.stt", data });
    expect(out?.topic).toBe("readings.stt");
    expect(out?.data).toMatchObject({
      utteranceId: "mic-abc:1:7",
      source: "others.stt",
      value: "<​i>漢字<​/i>",
      hiragana: "<​s>かんじ",
      segments: [{ text: "漢字", reading: "<​u>かんじ" }],
    });
    expect(sanitizeClientPubSubFrame({ topic: "readings.stt", data: { ...data, utteranceId: "bad id" } })).toBeNull();
    expect(sanitizeClientPubSubFrame({ topic: "readings.stt", data: { ...data, source: "text" + ".nope" } })).toBeNull();
  });

  it("never accepts other players' captions or readings from WebSocket peers", () => {
    for (const topic of ["others.stt", "others.translation", "others", "readings.stt"]) {
      expect(kind(routeInboundMessage(msg(topic, final("x")), "local", SELF)), topic).toBe("ignore:topic_not_allowed");
      expect(kind(routeInboundMessage(msg(topic, final("x"), [OTHER]), "link", SELF)), topic).toBe("ignore:topic_not_allowed");
    }
  });
});

describe("message ids (duplicate suppression)", () => {
  it("validates mid and carries it through the route", () => {
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), [OTHER], 5), "link", SELF))).toBe("ignore:bad_mid");
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), [OTHER], "has space"), "link", SELF))).toBe(
      "ignore:bad_mid"
    );
    expect(kind(routeInboundMessage(msg("text.translation", final("x"), [OTHER], "m".repeat(65)), "link", SELF))).toBe(
      "ignore:bad_mid"
    );
    expect(routeInboundMessage(msg("text.translation", final("x"), [OTHER], "m1"), "link", SELF)).toMatchObject({
      kind: "text",
      mid: "m1",
    });
    expect(routeInboundMessage(msg("obs.caption", caption, [OTHER], "m2"), "link", SELF)).toMatchObject({
      kind: "caption",
      mid: "m2",
    });
    // Pages and older Sigil versions send no mid.
    const noMid = routeInboundMessage(msg("text.translation", final("x"), [OTHER]), "link", SELF);
    expect(noMid.kind).toBe("text");
    expect(noMid.kind === "text" ? noMid.mid : "unexpected").toBeUndefined();
  });

  it("drops a second copy of the same message", () => {
    const seen = new RecentMessageIds();
    const wire = msg("text.stt", final("hello"), [OTHER], "abc");
    expect(routeInboundMessage(wire, "local", SELF, seen).kind).toBe("text");
    expect(seen.has("abc")).toBe(true);
    expect(kind(routeInboundMessage(wire, "link", SELF, seen))).toBe("ignore:duplicate");
    // A different message with the same text is not a duplicate.
    expect(routeInboundMessage(msg("text.stt", final("hello"), [OTHER], "abd"), "link", SELF, seen).kind).toBe("text");
  });

  it("does not remember ids of messages it refused", () => {
    const seen = new RecentMessageIds();
    expect(kind(routeInboundMessage(msg("text.stt", { type: 7 }, [OTHER], "bad"), "local", SELF, seen))).toBe(
      "ignore:invalid_data"
    );
    expect(kind(routeInboundMessage(msg("scenes:change", final("x"), [OTHER], "nope"), "local", SELF, seen))).toBe(
      "ignore:topic_not_allowed"
    );
    expect(seen.size).toBe(0);
  });

  it("forgets the oldest ids first", () => {
    const seen = new RecentMessageIds(3);
    for (const id of ["a", "b", "c"]) expect(seen.add(id)).toBe(true);
    expect(seen.add("b")).toBe(false);
    seen.add("d");
    expect(seen.size).toBe(3);
    expect(seen.has("a")).toBe(false);
    expect(seen.has("d")).toBe(true);
  });

  it("encodes mid on the wire and generates valid ids", () => {
    const id = newMessageId();
    expect(id).toMatch(/^[A-Za-z0-9_-]{16}$/);
    expect(newMessageId()).not.toBe(id);
    const wire = encodeWire("text.stt", JSON.stringify(final("a")), [SELF], id);
    expect(JSON.parse(wire)).toEqual({ topic: "text.stt", data: { type: 0, value: "a" }, via: [SELF], mid: id });
    const routed = routeInboundMessage(wire, "local", OTHER);
    expect(routed).toMatchObject({ kind: "text", mid: id, via: [SELF] });
  });

  it("two hosts linked to each other deliver each event once (pure topology model)", () => {
    // Each host is a WS peer on the other's server (its Link socket) and serves the other's Link
    // socket. A publish reaches the other host twice: through its own Link socket (source local on
    // the other side) and through its server broadcast (the other host's Link socket, source link).
    type Host = { id: string; seen: RecentMessageIds; delivered: string[] };
    const A: Host = { id: "A", seen: new RecentMessageIds(), delivered: [] };
    const B: Host = { id: "B", seen: new RecentMessageIds(), delivered: [] };
    const other = (h: Host) => (h === A ? B : A);
    const queue: { to: Host; wire: string; source: "local" | "link" }[] = [];
    const fanOut = (from: Host, wire: string, toLink: boolean) => {
      queue.push({ to: other(from), wire, source: "link" }); // server broadcast → other's Link socket
      if (toLink) queue.push({ to: other(from), wire, source: "local" }); // own Link socket → other's server
    };
    const publish = (from: Host, value: string) => {
      const mid = newMessageId();
      from.seen.add(mid);
      from.delivered.push(value);
      fanOut(from, encodeWire("text.translation", JSON.stringify(final(value)), [from.id], mid), true);
    };
    const drain = () => {
      for (let guard = 0; queue.length; guard++) {
        expect(guard).toBeLessThan(100);
        const { to, wire, source } = queue.shift()!;
        const route = routeInboundMessage(wire, source, to.id, to.seen);
        if (route.kind !== "text") continue;
        to.delivered.push(route.data.value);
        const mid = route.mid ?? newMessageId();
        fanOut(to, encodeWire(route.topic, JSON.stringify(route.data), nextVia(route.via, to.id), mid), source === "local");
      }
    };

    publish(A, "hello");
    drain();
    expect(A.delivered).toEqual(["hello"]);
    expect(B.delivered).toEqual(["hello"]);
    publish(B, "hola");
    drain();
    expect(A.delivered).toEqual(["hello", "hola"]);
    expect(B.delivered).toEqual(["hello", "hola"]);
  });
});
