import { describe, expect, it } from "vitest";
import { isDefaultChain, lineNeeds, TIER_ROUTES, tierRouteFor } from "../routingTable";
import { Service_Translation_Schema, TRANSLATION_DEFAULT_CHAIN, Translation_Backends as B } from "../schema";

function auto(patch: (a: ReturnType<typeof Service_Translation_Schema.parse>["auto"]) => void = () => {}) {
  const a = Service_Translation_Schema.parse({}).auto;
  patch(a);
  return a;
}

describe("the shipped tier routes", () => {
  it("are CPU-tier routes that reorder the default chain and add no engine", () => {
    expect(TIER_ROUTES.length).toBeGreaterThan(0);
    for (const route of TIER_ROUTES) {
      expect(route.tier).toBe("cpu");
      expect(route.chain[0]).toBe(B.nllb);
      expect([...route.chain].sort()).toEqual([...TRANSLATION_DEFAULT_CHAIN].sort());
      expect(route.nllbBeam).toBe(4);
    }
  });

  it("never take a direction of the audience's Asian languages, which Hy-MT2 translates better on every tier", () => {
    for (const route of TIER_ROUTES) {
      for (const d of route.directions) expect(d).not.toMatch(/(^|>)(ja|ko|zh|yue)(>|$)/);
    }
  });

  it("cover English with fr, pt, nl and es in both directions, and nothing else (German and Italian tied)", () => {
    const all = TIER_ROUTES.flatMap((r) => [...r.directions]).sort();
    const expected = ["fr", "pt", "nl", "es"].flatMap((l) => [`en>${l}`, `${l}>en`]).sort();
    expect(all).toEqual(expected);
  });
});

describe("tierRouteFor", () => {
  it("routes the European pairs to NLLB beam 4 first on the cpu tier", () => {
    for (const [from, to] of [["en", "fr"], ["en", "es"], ["en", "pt"], ["en", "nl"], ["fr", "en"], ["es", "en"], ["pt", "en"], ["nl", "en"]]) {
      const route = tierRouteFor(auto(), from, to, "cpu");
      expect(route, `${from} → ${to}`).not.toBeNull();
      expect(route?.chain).toEqual([B.nllb, B.llm, B.bergamot]);
      expect(route?.nllbBeam).toBe(4);
      expect(route?.nllbBudgetFactor).toBe(1.5);
    }
  });

  it("matches regional and legacy spellings of the languages", () => {
    expect(tierRouteFor(auto(), "en-US", "pt-BR", "cpu")).not.toBeNull();
    expect(tierRouteFor(auto(), "EN_us", "fr-CA", "cpu")).not.toBeNull();
    expect(tierRouteFor(auto(), "es-419", "en", "cpu")).not.toBeNull();
    expect(tierRouteFor(auto(), "en", "pt-PT", "cpu")).not.toBeNull();
  });

  it("applies to the cpu tier only", () => {
    expect(tierRouteFor(auto(), "en", "fr", "default")).toBeNull();
  });

  it("leaves every other pair on the chain: the Asian targets, Russian, the German and Italian ties, a source left on detect, pairs without English", () => {
    for (const [from, to] of [["en", "ja"], ["en", "ko"], ["en", "zh"], ["en", "zh-TW"], ["ja", "en"], ["ko", "en"], ["zh", "en"], ["en", "ru"], ["ru", "en"], ["en", "de"], ["de", "en"], ["en", "it"], ["it", "en"], ["fr", "es"], ["es", "fr"], ["", "fr"], ["", "en"], ["en", ""], ["en", "en"]]) {
      expect(tierRouteFor(auto(), from, to, "cpu"), `${from || "(detect)"} → ${to}`).toBeNull();
    }
  });

  it("keeps the AI model first for a line that carries glossary terms or the background text, which only it uses", () => {
    expect(tierRouteFor(auto(), "en", "fr", "cpu", { glossary: true })).toBeNull();
    expect(tierRouteFor(auto(), "en", "fr", "cpu", { background: true })).toBeNull();
    expect(tierRouteFor(auto(), "en", "fr", "cpu", { glossary: false, background: false })).not.toBeNull();
    expect(tierRouteFor(auto(), "en", "fr", "cpu", {})).not.toBeNull();
  });

  it("reads what a request carries: glossary terms and non-blank background, not the rolling context", () => {
    expect(lineNeeds({})).toEqual({ glossary: false, background: false });
    expect(lineNeeds({ glossary: [] })).toEqual({ glossary: false, background: false });
    expect(lineNeeds({ glossary: [{ source: "a", target: "b" }] })).toEqual({ glossary: true, background: false });
    expect(lineNeeds({ background: "  " })).toEqual({ glossary: false, background: false });
    expect(lineNeeds({ background: "A stream about horror games" })).toEqual({ glossary: false, background: true });
  });

  it("applies only while the chain is the shipped default", () => {
    expect(isDefaultChain([...TRANSLATION_DEFAULT_CHAIN])).toBe(true);
    expect(isDefaultChain([B.nllb, B.llm, B.bergamot])).toBe(false);
    expect(isDefaultChain([B.llm, B.nllb])).toBe(false);
    expect(isDefaultChain([B.llm, B.bergamot, B.nllb, B.deepl])).toBe(false);
    expect(tierRouteFor(auto((a) => (a.chain = [B.llm, B.nllb])), "en", "fr", "cpu")).toBeNull();
    expect(tierRouteFor(auto((a) => (a.chain = [B.deepl, B.llm, B.bergamot, B.nllb])), "en", "fr", "cpu")).toBeNull();
    expect(tierRouteFor(auto((a) => (a.chain = [B.nllb, B.llm, B.bergamot])), "en", "fr", "cpu")).toBeNull();
  });
});
