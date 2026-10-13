/**
 * Interface size scales the root font-size, so every rem size follows it. Text written in px (`text-[10px]`) does not, and
 * on a 4K monitor at 100% it is the tiny text Auto is there to fix. `src/style.css` therefore gives each px text size the
 * shells use a rem twin that wins over the utility. This keeps that list complete: a new `text-[Npx]` anywhere in `src/`
 * fails here until it has its line in style.css (or is written in rem).
 */
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const SRC = path.resolve(__dirname, "../../..");

function sourceFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "__tests__" || entry.name === "node_modules" ? [] : sourceFiles(full);
    return /\.(tsx|ts)$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [full] : [];
  });
}

/** The px sizes of every `text-[Npx]` class in the sources. */
function pxTextSizes(): number[] {
  const sizes = new Set<number>();
  for (const file of sourceFiles(SRC)) {
    for (const match of fs.readFileSync(file, "utf8").matchAll(/text-\[(\d+(?:\.\d+)?)px\]/g)) sizes.add(Number(match[1]));
  }
  return [...sizes].sort((a, b) => a - b);
}

/** style.css without comments, so a rule inside a comment does not count. */
const css = fs.readFileSync(path.join(SRC, "style.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");

/** The declarations of the top-level (unlayered, not in a media query) rule for `selector`, or undefined. */
function topLevelRule(selector: string): string | undefined {
  let depth = 0;
  for (let i = 0; i < css.length; i++) {
    if (css[i] === "}") depth--;
    else if (css[i] === "{") {
      if (depth === 0) {
        const start = Math.max(css.lastIndexOf("}", i), css.lastIndexOf(";", i)) + 1;
        const prelude = css.slice(start, i).trim();
        if (prelude === selector) return css.slice(i + 1, css.indexOf("}", i));
      }
      depth++;
    }
  }
  return undefined;
}

describe("px text sizes follow the Interface size", () => {
  const needed = [...new Set([10, 11, ...pxTextSizes()])].sort((a, b) => a - b);

  it("finds the px text sizes the shells use", () => {
    expect(pxTextSizes()).toEqual(expect.arrayContaining([10, 11]));
  });

  for (const px of needed) {
    it(`text-[${px}px] has a top-level rem rule in style.css (${px / 16}rem, the same ${px}px at size 1)`, () => {
      const rule = topLevelRule(`.text-\\[${px}px\\]`);
      expect(rule, `style.css has no top-level .text-\\[${px}px\\] rule`).toBeDefined();
      const rem = /font-size:\s*([\d.]+)rem/.exec(rule ?? "");
      expect(rem, "the rule sets a font-size in rem").not.toBeNull();
      expect(Number(rem?.[1])).toBeCloseTo(px / 16, 5);
    });
  }
});
