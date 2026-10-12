import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ts from "typescript";
import { describe, expect, it } from "vitest";

/**
 * Rules of hooks for the /client components that run on an OBS source: a hook after an early
 * `return` changes the hook count between two renders of the same component, React throws, and
 * with no error boundary the whole page unmounted (a blank source until someone refreshed it).
 * `ElementInstance` returned before a `useMemo` whenever the active scene lacked the element, which
 * a scene switch on a multi-scene template does. No DOM is available to render it here, so the
 * source is checked: no hook call may follow a `return` in the same component. Every file under
 * src/client/ui and src/client/elements is checked, so a new early return in any /client component
 * is caught, not just the two that broke.
 */

const ROOTS = ["..", "../../elements"];

/** The .ts / .tsx files below `dir` (relative to this test), tests excluded. */
function sourceFiles(dir: string): string[] {
  const root = new URL(`${dir}/`, import.meta.url);
  const found: string[] = [];
  for (const entry of readdirSync(fileURLToPath(root), { withFileTypes: true })) {
    if (entry.name === "__tests__") continue;
    if (entry.isDirectory()) found.push(...sourceFiles(`${dir}/${entry.name}`));
    else if (/\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)) found.push(`${dir}/${entry.name}`);
  }
  return found;
}

/** `use…` calls that come after a `return` in the same function (nested functions are their own). */
function hooksAfterReturn(source: string): string[] {
  const file = ts.createSourceFile("component.tsx", source, ts.ScriptTarget.ESNext, true, ts.ScriptKind.TSX);
  const offenders: string[] = [];
  const isFunction = (node: ts.Node): node is ts.FunctionLikeDeclaration =>
    ts.isArrowFunction(node) || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node);

  const check = (fn: ts.FunctionLikeDeclaration) => {
    let returned = false;
    const visit = (node: ts.Node) => {
      if (node !== fn && isFunction(node)) {
        check(node);
        return;
      }
      if (ts.isReturnStatement(node)) returned = true;
      if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && /^use[A-Z]/.test(node.expression.text) && returned) {
        const { line } = file.getLineAndCharacterOfPosition(node.getStart());
        offenders.push(`${node.expression.text} at line ${line + 1}`);
      }
      ts.forEachChild(node, visit);
    };
    if (fn.body) ts.forEachChild(fn.body, visit);
  };
  const top = (node: ts.Node) => {
    if (isFunction(node)) check(node);
    else ts.forEachChild(node, top);
  };
  top(file);
  return offenders;
}

describe("hooks never follow a return in the /client components", () => {
  it("detects the bug it guards against", () => {
    const buggy = `
      export const A = memo(({ id }) => {
        const x = useThing(id);
        if (!x) return <></>;
        const y = useMemo(() => x, [x]);
        return <b>{y}</b>;
      });`;
    expect(hooksAfterReturn(buggy)).toEqual(["useMemo at line 5"]);
    const fine = `
      export const A = memo(({ id }) => {
        const x = useThing(id);
        const y = useMemo(() => x, [x]);
        if (!x) return <></>;
        return <b>{y}</b>;
      });`;
    expect(hooksAfterReturn(fine)).toEqual([]);
  });

  const files = ROOTS.flatMap(sourceFiles);

  it("looks at the components that exist", () => {
    const names = files.map((file) => file.split("/").pop());
    expect(names).toEqual(expect.arrayContaining(["element-instance.tsx", "view.tsx", "view_loading.tsx", "clientRecovery.tsx"]));
    expect(files.some((file) => file.includes("elements/image/"))).toBe(true);
    expect(files.some((file) => file.includes("elements/text/"))).toBe(true);
  });

  for (const file of files) {
    it(file.replace(/^\.\.\/\.\.\//, "src/client/").replace(/^\.\.\//, "src/client/ui/"), () => {
      const source = readFileSync(fileURLToPath(new URL(file, import.meta.url)), "utf8");
      expect(hooksAfterReturn(source)).toEqual([]);
    });
  }
});

describe("selectors on the /client document tolerate a missing element or scene", () => {
  // A selector that throws on a document without the element (a scene switch, a deleted scene, a
  // host that has not sent it yet) throws in the store subscription as well as in render. Only the
  // components are looked at; the document services write to elements they know exist.
  it("does not index an element or one of its scenes without `?.`", () => {
    const offenders: string[] = [];
    for (const file of ROOTS.flatMap(sourceFiles).filter((name) => name.endsWith(".tsx"))) {
      const source = readFileSync(fileURLToPath(new URL(file, import.meta.url)), "utf8");
      source.split(/\r?\n/).forEach((line, index) => {
        if (/\.elements\[[^\]]+\]\.(scenes|type)\b|\.scenes\[[^\]]+\]\.data\b/.test(line)) offenders.push(`${file}:${index + 1}: ${line.trim()}`);
      });
    }
    expect(offenders).toEqual([]);
  });
});
