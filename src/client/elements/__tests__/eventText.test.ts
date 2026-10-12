import fs from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import template from "lodash/template";
import {
  compileEventText,
  escapeEventHtml,
  parseEventText,
  renderEventText,
} from "../eventText";
import { templateEventsSubscriber } from "../utils";

/** The sentinel every attack string tries to set. If it ever becomes defined, code ran. */
const SENTINEL = "__x";
const g = globalThis as unknown as Record<string, unknown>;

beforeEach(() => {
  delete g[SENTINEL];
  // `window.__x = 1` in an attack string would set the sentinel on globalThis if it ran.
  vi.stubGlobal("window", globalThis);
});
afterEach(() => {
  delete g[SENTINEL];
  vi.unstubAllGlobals();
});

describe("supported forms", () => {
  const event = { name: "Ada", n: 3, a: { b: "deep" }, list: ["p", "q", { k: "v" }], "key with space": "ks" };

  it.each([
    ["${event.name}", "Ada"],
    ["${event.a.b}", "deep"],
    ["<%= event.name %>", "Ada"],
    ["<%= event.a.b %>", "deep"],
    ["<%=event.name%>", "Ada"],
    ["<%-event.name%>", "Ada"],
    ["${ event.name }", "Ada"],
    ["${\n  event.name\n}", "Ada"],
    ["<%=\n event.a.b \n%>", "deep"],
    ["Hello ${event.name}, you have <%= event.n %> new <%- event.a.b %>!", "Hello Ada, you have 3 new deep!"],
    ["${event.name}${event.n}", "Ada3"],
    ["${event.name} and ${event.name}", "Ada and Ada"],
    ["${event.list[0]}", "p"],
    ["${event.list[ 1 ]}", "q"],
    ["${event.list[2].k}", "v"],
    ['${event["name"]}', "Ada"],
    ["${event['name']}", "Ada"],
    ['${event["key with space"]}', "ks"],
    ["${event.list.length}", "3"],
    ["${event?.name}", "Ada"],
    ["${event?.a?.b}", "deep"],
    ["${event?.[\"name\"]}", "Ada"],
    ["${event?.list?.[0]}", "p"],
    ["no placeholders at all", "no placeholders at all"],
    ["", ""],
  ])("%j renders %j", (text, expected) => {
    expect(renderEventText(text, event)).toBe(expected);
  });

  it("renders the event value itself", () => {
    expect(renderEventText("${event}", "hello")).toBe("hello");
    expect(renderEventText("<%= event %>", "hello")).toBe("hello");
    expect(renderEventText("<%- event %>", "hello")).toBe("hello");
    expect(renderEventText("[${event}]", 42)).toBe("[42]");
  });

  it("allows non-ASCII property names", () => {
    expect(renderEventText("${event.名前}", { 名前: "えいだ" })).toBe("えいだ");
    expect(renderEventText("${event.größe}", { größe: 7 })).toBe("7");
  });

  it("compiles once and renders many times", () => {
    const render = compileEventText("hi ${event.who}");
    expect(render({ who: "a" })).toBe("hi a");
    expect(render({ who: "b" })).toBe("hi b");
    expect(render({})).toBe("hi ");
  });

  it("a text without placeholders is returned as is, for any value", () => {
    const render = compileEventText("plain");
    expect(render(null)).toBe("plain");
    expect(render({ anything: 1 })).toBe("plain");
  });
});

describe("values", () => {
  it.each([
    [undefined, ""],
    [null, ""],
    ["", ""],
    [0, "0"],
    [-0, "0"],
    [42, "42"],
    [1.5, "1.5"],
    [NaN, "NaN"],
    [false, "false"],
    [true, "true"],
    [10n, "10"],
    [["a", "b"], "a,b"],
    [[1, [2, 3]], "1,2,3"],
    [[null, 1, undefined], ",1,"],
    [[], ""],
    [{}, "[object Object]"],
    [{ a: 1 }, "[object Object]"],
    [Symbol("s"), ""],
    [() => "no", ""],
  ])("${event} of %o gives %j", (value, expected) => {
    expect(renderEventText("${event}", value)).toBe(expected);
  });

  it("missing paths render as nothing, like undefined did in lodash", () => {
    expect(renderEventText("[${event.nope}]", { a: 1 })).toBe("[]");
    expect(renderEventText("[${event.nope.deeper.still}]", { a: 1 })).toBe("[]");
    expect(renderEventText("[${event.a.b}]", { a: "str" })).toBe("[]");
    expect(renderEventText("[${event.a.b}]", { a: 5 })).toBe("[]");
    expect(renderEventText("[${event.list[9]}]", { list: [1] })).toBe("[]");
    expect(renderEventText("[<%= event.nope %>]", {})).toBe("[]");
    expect(renderEventText("[${event.a}]", null)).toBe("[]");
    expect(renderEventText("[${event.a}]", undefined)).toBe("[]");
    expect(renderEventText("[${event.a}]", "a string")).toBe("[]");
  });

  it("does not call anything the value carries", () => {
    const calls: string[] = [];
    const evil = {
      toString: () => (calls.push("toString"), "x"),
      valueOf: () => (calls.push("valueOf"), 1),
      [Symbol.toPrimitive]: () => (calls.push("toPrimitive"), "y"),
      toJSON: () => (calls.push("toJSON"), "z"),
    };
    expect(renderEventText("${event}", evil)).toBe("[object Object]");
    expect(renderEventText("<%- event %>", evil)).toBe("[object Object]");
    expect(renderEventText("${event.toString}", evil)).toBe("");
    expect(calls).toEqual([]);
  });

  it("an array nested without limit (or cyclic) stops instead of recursing forever", () => {
    const cyclic: unknown[] = [1];
    cyclic.push(cyclic);
    expect(() => renderEventText("${event}", cyclic)).not.toThrow();
    expect(renderEventText("${event}", cyclic)).toMatch(/^1,1,1,/);
  });
});

describe("escaping", () => {
  const raw = `<b class="x">Tom & 'Jerry'</b>`;
  const escaped = "&lt;b class=&quot;x&quot;&gt;Tom &amp; &#39;Jerry&#39;&lt;/b&gt;";

  it("<%- %> escapes & < > \" ' in the value", () => {
    expect(renderEventText("<%- event %>", raw)).toBe(escaped);
    expect(renderEventText("<%- event.t %>", { t: raw })).toBe(escaped);
    expect(renderEventText("<%-event.t%>", { t: raw })).toBe(escaped);
  });

  it("${} and <%= %> insert as is, the way lodash did", () => {
    expect(renderEventText("${event}", raw)).toBe(raw);
    expect(renderEventText("<%= event %>", raw)).toBe(raw);
  });

  it("escapes once, and only values: the template's own text is not touched", () => {
    expect(renderEventText("<i>&</i> <%- event %>", "&amp;")).toBe("<i>&</i> &amp;amp;");
  });

  it("escapes the text of arrays and numbers too", () => {
    expect(renderEventText("<%- event %>", ["<a>", "&"])).toBe("&lt;a&gt;,&amp;");
    expect(renderEventText("<%- event %>", 7)).toBe("7");
    expect(renderEventText("<%- event.x %>", {})).toBe("");
  });

  it("escapeEventHtml handles each of the five characters", () => {
    expect(escapeEventHtml(`&<>"'`)).toBe("&amp;&lt;&gt;&quot;&#39;");
    expect(escapeEventHtml("plain `text`")).toBe("plain `text`");
  });
});

describe("attack strings never run code", () => {
  /** Render and assert nothing executed and the prototype chain is untouched. */
  function render(text: string, event: unknown = { a: "A" }) {
    const out = renderEventText(text, event);
    expect(g[SENTINEL]).toBeUndefined();
    expect(Object.keys(Object.prototype)).toEqual([]);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    return out;
  }

  it("the sentinel really detects execution (lodash runs the very same strings)", () => {
    // Guards the assertions below against passing for the wrong reason.
    template("<% window.__x = 1 %>")({});
    expect(g[SENTINEL]).toBe(1);
    delete g[SENTINEL];
    template('${constructor.constructor("globalThis.__x = 2")()}')({});
    expect(g[SENTINEL]).toBe(2);
    delete g[SENTINEL];
    template('<%= event.constructor.constructor("globalThis.__x = 3")() %>')({ event: {} });
    expect(g[SENTINEL]).toBe(3);
  });

  it.each([
    "<% window.__x=1 %>",
    "<% window.__x = 1; %>",
    "<%window.__x=1%>",
    "<% globalThis.__x = 1 %>",
    "<%\nwindow.__x = 1\n%>",
    "<% if (event) { window.__x = 1 } %>shown<% } %>",
    "<% for (;;) {} %>",
    "<% throw new Error('x') %>",
    "<% print('x') %>",
    "<% __p += 'x' %>",
  ])("a <% code %> block is left as typed: %j", (text) => {
    expect(render(text)).toBe(text);
  });

  it("a <% code %> block between real values stays, the values still render", () => {
    expect(render("a <% window.__x=1 %> b ${event.a} c")).toBe("a <% window.__x=1 %> b A c");
  });

  it("stripping would have changed what a conditional shows, so code blocks are not stripped", () => {
    const text = "<% if (event.show) { %>SECRET<% } %>";
    const out = render(text, { show: false });
    expect(out).toBe(text);
  });

  it.each([
    '${constructor.constructor("window.__x=1")()}',
    '${constructor.constructor("globalThis.__x=1")()}',
    "${this.constructor}",
    "<%= this.constructor %>",
    '<%= this.constructor.constructor("window.__x=1")() %>',
    '<%- constructor.constructor("window.__x=1")() %>',
    '${event.constructor.constructor("window.__x=1")()}',
    '<%= event.constructor.constructor("window.__x=1")() %>',
    "${window.__x=1}",
    "${globalThis.__x=1}",
    "${(window.__x=1)}",
    "${window}",
    "${globalThis}",
    "${self}",
    "${document.cookie}",
    "${Function}",
    "${eval('window.__x=1')}",
    "${new Function('window.__x=1')()}",
    "${import('x')}",
    "<%= window.__x=1 %>",
    "<%- window.__x=1 %>",
    "<%= _.template('x') %>",
    "<%= print %>",
    "${__e('x')}",
    "${obj}",
    "${arguments}",
    "${this}",
    "${event.a(1)}",
    "${event.toString()}",
    "${event.a.toUpperCase()}",
    "${event.a.constructor()}",
    "${event[event.a]}",
    "${event[0+0]}",
    '${event["a"+"b"]}',
    "${event[`a`]}",
    "${event.a`x`}",
    "${event.a + event.a}",
    "${event.a ? 1 : 2}",
    "${event.a || 'x'}",
    "${event.a ?? 'x'}",
    "${event.a, window.__x=1}",
    "${event.a; window.__x=1}",
    "${event.a\n;window.__x=1}",
    "${event = window}",
    "${event.a = 1}",
    "${-event.a}",
    "${!event.a}",
    "${typeof event}",
    "${delete event.a}",
    "${await event}",
    "${...event}",
    "${event.a.b.c.d(}",
    "${event..a}",
    "${event.}",
    "${event.1}",
    "${event.a-b}",
    "${event.a b}",
    "${event .a}",
    "${event. a}",
    "${event [0]}",
    "${event[]}",
    "${event[0}",
    "${event[-1]}",
    "${event[1.5]}",
    "${event[0x1]}",
    "${event[1e3]}",
    '${event["a]}',
    '${event["a\\u0062"]}',
    '${event["\\x5f_proto__"]}',
    "${events.a}",
    "${Event.a}",
    "${eventX}",
    "${xevent}",
    "${$event}",
    "${event$}",
    "${event#a}",
    "${}",
    "${ }",
    "<%= %>",
    "<%- %>",
    "<%=%>",
  ])("a refused expression is left as typed: %j", (text) => {
    expect(render(text)).toBe(text);
  });

  it.each([
    "${event.__proto__}",
    "${event.__proto__.polluted}",
    "${event.__proto__.__proto__}",
    "${event.constructor}",
    "${event.constructor.name}",
    "${event.constructor.prototype}",
    "${event.prototype}",
    "${event.a.__proto__}",
    "${event.a.constructor.constructor}",
    "<%= event.__proto__ %>",
    "<%- event.constructor %>",
    "<%= event.constructor.prototype.polluted %>",
    '${event["__proto__"]}',
    "${event['constructor']}",
    '${event["prototype"]}',
    '<%= event["constructor"]["constructor"] %>',
    '${event?.["__proto__"]}',
    "${event?.constructor}",
    "${event?.__proto__?.x}",
    '${event.list[0]["constructor"]}',
  ])("prototype paths are refused: %j", (text) => {
    expect(render(text, { a: {}, list: [{}] })).toBe(text);
  });

  it("refuses even when the data itself owns such a key (JSON can define them)", () => {
    const parsed = JSON.parse('{"__proto__": {"x": "own"}, "constructor": "c", "prototype": "p", "ok": "fine"}');
    expect(Object.prototype.hasOwnProperty.call(parsed, "__proto__")).toBe(true);
    expect(render("${event.__proto__.x}", parsed)).toBe("${event.__proto__.x}");
    expect(render("${event.constructor}", parsed)).toBe("${event.constructor}");
    expect(render("${event.prototype}", parsed)).toBe("${event.prototype}");
    expect(render("${event.ok}", parsed)).toBe("fine");
  });

  it("never reaches inherited properties", () => {
    const event = Object.create({ inherited: "no" }) as Record<string, unknown>;
    event.own = "yes";
    expect(render("${event.own}|${event.inherited}", event)).toBe("yes|");
    for (const name of ["toString", "valueOf", "hasOwnProperty", "isPrototypeOf", "__defineGetter__", "__lookupGetter__", "toLocaleString", "propertyIsEnumerable"]) {
      expect(render("[${event." + name + "}]", {})).toBe("[]");
      expect(render("[${event." + name + "}]", [])).toBe("[]");
    }
    expect(render("[${event.map}][${event.concat}][${event.push}]", [1])).toBe("[][][]");
  });

  it("never runs a getter", () => {
    const event = {
      get boom() {
        g[SENTINEL] = "getter ran";
        return "x";
      },
      safe: "s",
    };
    expect(renderEventText("[${event.boom}][${event.safe}]", event)).toBe("[][s]");
    expect(g[SENTINEL]).toBeUndefined();
  });

  it("does not read through functions or primitives", () => {
    const event = { fn: () => 1, str: "text", num: 5 };
    expect(render("[${event.fn.name}][${event.fn.length}][${event.str.length}][${event.num.toFixed}]", event)).toBe("[][][][]");
  });

  it("an attack inside a bracket key is just a missing key", () => {
    expect(render('[${event["window.__x=1"]}]', {})).toBe("[]");
    expect(render("[${event['a b c']}]", { "a b c": "ok" })).toBe("[ok]");
    expect(render('[${event["]"]}]', { "]": "bracket" })).toBe("[bracket]");
  });

  it("untrusted event DATA is inserted as text, never read as a template", () => {
    expect(render("${event.a}", { a: "<% window.__x=1 %> ${constructor.constructor('x')()}" })).toBe(
      "<% window.__x=1 %> ${constructor.constructor('x')()}"
    );
    expect(render("<%= event %>", "${event.a}")).toBe("${event.a}");
  });

  it("pairs are one unit: delimiters inside a refused block are not read again", () => {
    // lodash paired them the same way; the inner ${event.a} is part of the code block, not a value.
    expect(render('<% x = "${event.a}" %>')).toBe('<% x = "${event.a}" %>');
    expect(render("${ ${event.a} }")).toBe("${ ${event.a} }");
  });
});

describe("text that merely looks like a template", () => {
  it.each([
    "$",
    "${",
    "$ {event.a}",
    "{event.a}",
    "{{event.a}}",
    "$(event.a)",
    "#{event.a}",
    "%{event.a}",
    "<",
    "<%",
    "<% not closed",
    "<%= not closed",
    "<%- not closed",
    "not opened %>",
    "100% sure",
    "50%> 30",
    "<%%>",
    "<% %>",
    "${ not closed",
    "}",
    "cost: $5 {approx}",
    "price ${",
  ])("is kept as typed: %j", (text) => {
    expect(renderEventText(text, { a: "A" })).toBe(text);
  });

  it("an unterminated opener does not stop a later complete one", () => {
    expect(renderEventText("<%= open ${event.a}", { a: "A" })).toBe("<%= open A");
    expect(renderEventText("${ open <%= event.a %>", { a: "A" })).toBe("${ open A");
    expect(renderEventText("<% open <%= event.a %> and ${event.a}", { a: "A" })).toBe("<% open <%= event.a %> and A");
  });

  it("newlines and unicode in the plain text survive", () => {
    expect(renderEventText("line 1\nline 2 ${event.a}\r\n end 🎉", { a: "A" })).toBe("line 1\nline 2 A\r\n end 🎉");
  });

  it("backslashes, quotes and backticks in the plain text are literal", () => {
    expect(renderEventText("a\\b 'c' \"d\" `e` \\${event.a}", { a: "A" })).toBe("a\\b 'c' \"d\" `e` \\A");
  });
});

describe("parseEventText", () => {
  it("splits plain text and values", () => {
    expect(parseEventText("a ${event.x} b <%- event.y[0] %> c")).toEqual([
      "a ",
      { path: ["x"], escape: false },
      " b ",
      { path: ["y", "0"], escape: true },
      " c",
    ]);
  });

  it("a text with no values is a single string (or nothing)", () => {
    expect(parseEventText("just text")).toEqual(["just text"]);
    expect(parseEventText("")).toEqual([]);
    expect(parseEventText("<% code %>")).toEqual(["<% code %>"]);
  });

  it("a bare event is an empty path", () => {
    expect(parseEventText("${event}")).toEqual([{ path: [], escape: false }]);
  });

  it("`?.` reads like `.`", () => {
    expect(parseEventText("${event?.a?.[0]}")).toEqual([{ path: ["a", "0"], escape: false }]);
  });
});

describe("agrees with lodash on the forms that are kept", () => {
  const events: unknown[] = [
    "hi",
    "",
    42,
    0,
    false,
    true,
    null,
    undefined,
    [1, 2],
    [],
    {},
    `<b>&'"</b>`,
    { name: "N", n: 5, a: { b: "x<y>" }, list: ["p", "q"], k: "kv", zero: 0, no: false, nul: null, arr: [1, [2, 3]] },
  ];
  const texts = [
    "${event}",
    "<%= event %>",
    "<%- event %>",
    "${ event }",
    "${event.name}",
    "${event.a.b}",
    "<%= event.name %>",
    "<%= event.a.b %>",
    "<%- event.a.b %>",
    "<%-event.name%>",
    "Hello ${event.name}!",
    "x<%=event.n%>y<%-event.a.b%>z",
    "${event.list[1]}",
    '${event["k"]}',
    "${event['k']}",
    "${event.zero}|${event.no}|${event.nul}|${event.missing}",
    "${event.arr}",
    "<%- event.arr %>",
    "plain text",
    "$ { event }",
    "100% ${event}%",
    "{event} {{event}} #{event}",
    "${event}${event}",
    "a\nb ${event}\nc",
  ];

  for (const text of texts) {
    it(`${JSON.stringify(text)}`, () => {
      for (const event of events) {
        let expected: string | undefined;
        try {
          expected = template(text)({ event });
        } catch {
          // lodash threw (a path through null/undefined): Sigil renders nothing for that part instead, below.
          expect(() => renderEventText(text, event)).not.toThrow();
          continue;
        }
        expect(renderEventText(text, event)).toBe(expected);
      }
    });
  }

  it("differs on purpose where lodash threw: a path through a missing value is empty, not an error", () => {
    expect(() => template("${event.a.b}")({ event: null })).toThrow();
    expect(renderEventText("[${event.a.b}]", null)).toBe("[]");
    expect(renderEventText("[${event.a.b}]", {})).toBe("[]");
  });
});

describe("performance on long text", () => {
  /** Generous absolute limits: each case would take many seconds or minutes if the reader were quadratic. */
  function time<T>(fn: () => T): { ms: number; value: T } {
    const t0 = performance.now();
    const value = fn();
    return { ms: performance.now() - t0, value };
  }

  it("1 MB of plain text with a few values", () => {
    const text = "lorem ipsum dolor sit amet ".repeat(40_000) + "${event.a}";
    expect(text.length).toBeGreaterThan(1_000_000);
    const { ms, value } = time(() => renderEventText(text, { a: "END" }));
    expect(value.endsWith("amet END")).toBe(true);
    expect(ms).toBeLessThan(500);
  });

  it("100,000 values", () => {
    const text = "${event.a}-".repeat(100_000);
    const { ms, value } = time(() => renderEventText(text, { a: "x" }));
    expect(value.length).toBe(200_000);
    expect(ms).toBeLessThan(1500);
  });

  it("render of a compiled text is cheap per event", () => {
    const render = compileEventText("[${event.a.b}] <%- event.c %> ${event.list[1]}");
    const event = { a: { b: "x" }, c: "<>", list: [1, 2] };
    const { ms } = time(() => {
      for (let i = 0; i < 100_000; i++) render(event);
    });
    expect(render(event)).toBe("[x] &lt;&gt; 2");
    expect(ms).toBeLessThan(1500);
  });

  it("200,000 unterminated <%= openers (the quadratic case of a naive reader)", () => {
    const text = "<%= ".repeat(200_000) + "${event.a}";
    const { ms, value } = time(() => renderEventText(text, { a: "A" }));
    expect(value.endsWith("<%= A")).toBe(true);
    expect(ms).toBeLessThan(1500);
  });

  it("200,000 unterminated ${ openers", () => {
    const text = "${ ".repeat(200_000) + "<%= event.a %>";
    const { ms, value } = time(() => renderEventText(text, { a: "A" }));
    expect(value.endsWith("${ A")).toBe(true);
    expect(ms).toBeLessThan(1500);
  });

  it("200,000 unterminated <% code openers", () => {
    const text = "<% ".repeat(200_000);
    const { ms, value } = time(() => renderEventText(text, {}));
    expect(value).toBe(text);
    expect(ms).toBeLessThan(1500);
  });

  it("100,000 code blocks", () => {
    const text = "<% window.__x=1 %>".repeat(100_000);
    const { ms, value } = time(() => renderEventText(text, {}));
    expect(value).toBe(text);
    expect(ms).toBeLessThan(1500);
    expect(g[SENTINEL]).toBeUndefined();
  });

  it("openers nested inside one long pair", () => {
    const text = "${" + "${ ".repeat(100_000) + "}" + "<%=".repeat(100_000) + "%>";
    const { ms } = time(() => renderEventText(text, { a: 1 }));
    expect(ms).toBeLessThan(1500);
  });

  it("alternating refused and good expressions", () => {
    const text = "${event.a()}${event.a}".repeat(50_000);
    const { ms, value } = time(() => renderEventText(text, { a: "x" }));
    expect(value.length).toBe(50_000 * ("${event.a()}".length + 1));
    expect(ms).toBeLessThan(1500);
  });

  it("a very long path of missing keys", () => {
    const text = "${event" + ".a".repeat(50_000) + "}";
    const { ms, value } = time(() => renderEventText(text, { a: 1 }));
    expect(value).toBe("");
    expect(ms).toBeLessThan(1000);
  });

  it("megabytes of whitespace inside one pair", () => {
    const text = "${" + " ".repeat(2_000_000) + "event.a" + " ".repeat(2_000_000) + "}";
    const { ms, value } = time(() => renderEventText(text, { a: "A" }));
    expect(value).toBe("A");
    expect(ms).toBeLessThan(1500);
  });

  it("a long key in brackets, and a long unterminated key", () => {
    const long = "k".repeat(500_000);
    expect(renderEventText(`\${event["${long}"]}`, { [long]: "found" })).toBe("found");
    const text = `\${event["${long}}`.repeat(20);
    const { ms } = time(() => renderEventText(text, {}));
    expect(ms).toBeLessThan(1500);
  });

  it("a large array value", () => {
    const big = Array.from({ length: 200_000 }, (_, i) => i);
    const { ms, value } = time(() => renderEventText("${event}", big));
    expect(value.startsWith("0,1,2,")).toBe(true);
    expect(ms).toBeLessThan(1500);
  });
});

describe("templateEventsSubscriber", () => {
  type Handler = (value: unknown) => void;
  function stubPubsub() {
    const handlers = new Map<string, Handler[]>();
    const pubsub = {
      subscribe: vi.fn((name: string, fn: Handler) => {
        handlers.set(name, [...(handlers.get(name) ?? []), fn]);
        return `token-${name}-${handlers.get(name)!.length}`;
      }),
      publish: (name: string, value: unknown) => handlers.get(name)?.forEach((fn) => fn(value)),
    };
    vi.stubGlobal("window", { ApiShared: { pubsub } });
    return pubsub;
  }

  it("renders the text of an event with the published value", () => {
    const pubsub = stubPubsub();
    const seen: Array<[string, string]> = [];
    const tokens = templateEventsSubscriber(
      [
        { event: "text.stt", text: "Said: ${event.value}" },
        { event: "twitch.follow", text: "<%- event.user %> followed" },
      ],
      (event, value) => seen.push([event, value])
    );
    expect(tokens).toEqual(["token-text.stt-1", "token-twitch.follow-1"]);
    pubsub.publish("text.stt", { value: "hello" });
    pubsub.publish("twitch.follow", { user: "<b>Bo</b>" });
    expect(seen).toEqual([
      ["text.stt", "Said: hello"],
      ["twitch.follow", "&lt;b&gt;Bo&lt;/b&gt; followed"],
    ]);
  });

  it("a hostile imported text is shown as text and runs nothing", () => {
    const pubsub = stubPubsub();
    const seen: string[] = [];
    templateEventsSubscriber(
      [
        { event: "e1", text: "<% window.__x=1 %>" },
        { event: "e2", text: '${constructor.constructor("window.__x=1")()}' },
        { event: "e3", text: "<%= this.constructor %>" },
      ],
      (_event, value) => seen.push(value)
    );
    pubsub.publish("e1", {});
    pubsub.publish("e2", {});
    pubsub.publish("e3", {});
    expect(seen).toEqual(["<% window.__x=1 %>", '${constructor.constructor("window.__x=1")()}', "<%= this.constructor %>"]);
    expect(g[SENTINEL]).toBeUndefined();
  });

  it("skips entries that have no event or no text, and ignores a bad list", () => {
    const pubsub = stubPubsub();
    const bad = [
      { event: "", text: "x" },
      { event: "a", text: "" },
      { event: "b", text: 5 as unknown as string },
      { event: 7 as unknown as string, text: "x" },
      null as unknown as { event: string; text: string },
    ];
    expect(templateEventsSubscriber(bad, () => {})).toEqual([]);
    expect(pubsub.subscribe).not.toHaveBeenCalled();
    expect(templateEventsSubscriber(undefined as unknown as [], () => {})).toEqual([]);
    expect(templateEventsSubscriber({} as unknown as [], () => {})).toEqual([]);
  });

  it("a value that throws when read gives _invalid_value_ instead of breaking the subscriber", () => {
    const pubsub = stubPubsub();
    const seen: string[] = [];
    templateEventsSubscriber([{ event: "p", text: "${event.a}" }], (_e, v) => seen.push(v));
    const hostile = new Proxy({}, {
      getOwnPropertyDescriptor() {
        throw new Error("boom");
      },
    });
    expect(() => pubsub.publish("p", hostile)).not.toThrow();
    expect(seen).toEqual(["_invalid_value_"]);
  });

  it("an error in the consumer is not swallowed or answered with a second call", () => {
    const pubsub = stubPubsub();
    const fn = vi.fn(() => {
      throw new Error("consumer bug");
    });
    templateEventsSubscriber([{ event: "p", text: "${event}" }], fn);
    expect(() => pubsub.publish("p", "v")).toThrow("consumer bug");
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

describe("no code-evaluating template engine in src", () => {
  const srcDir = path.resolve(__dirname, "../../..");
  const BANNED: Array<[RegExp, string]> = [
    [/lodash\/template/, "lodash/template"],
    [/lodash\/(?:fp\/)?templateSettings/, "lodash templateSettings"],
    [/import\s*\{[^}]*\btemplate\b[^}]*\}\s*from\s*["']lodash(?:-es)?["']/, "template imported from lodash"],
    [/\b_\.template\s*\(/, "_.template("],
    [/\bnew\s+Function\b/, "new Function"],
    [/\beval\s*\(/, "eval("],
    [/\bfrom\s+["'](?:handlebars|mustache|ejs|nunjucks|underscore|lodash\.template|lodash-es\/template)["']/, "template engine import"],
  ];

  function sourceFiles(dir: string): string[] {
    const out: string[] = [];
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ent.name === "__tests__" || ent.name === "node_modules") continue;
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) out.push(...sourceFiles(p));
      else if (/\.(?:ts|tsx|js|jsx|mjs)$/.test(ent.name) && !/\.test\./.test(ent.name)) out.push(p);
    }
    return out;
  }

  it("nothing under src/ compiles text to code", () => {
    const found: string[] = [];
    for (const file of sourceFiles(srcDir)) {
      const text = fs.readFileSync(file, "utf8");
      for (const [re, label] of BANNED) {
        if (re.test(text)) found.push(`${path.relative(srcDir, file)}: ${label}`);
      }
    }
    expect(found).toEqual([]);
  });
});
