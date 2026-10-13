/**
 * Which build-time environment variables may reach the frontend bundle (names only, never values).
 *
 * `env-classification.json` lists the `public` names (OAuth client ids, redirect overrides, the Tauri
 * debug flag: the allow-list `vite.config.ts` gives Vite as `envPrefix`) and the `secret` names (what
 * `src-tauri/build.rs` compiles into Sigil.exe). This module loads that file and refuses a combination
 * that could put a secret into `import.meta.env`. `vite.config.ts` and `check-no-secrets.mjs` use it;
 * `scripts/release/__tests__/env-classification.test.mjs` and `src/utils/__tests__/noFrontendSecrets.test.ts`
 * guard it.
 */
import fs from "node:fs";
import path from "node:path";

/** Variable names that look like credentials (also how `check-no-secrets.mjs` finds values to look for). */
export const SECRET_NAME_PATTERN = /SECRET|API_KEY|TOKEN|PASSWORD/i;

const NAME_SHAPE = /^[A-Z][A-Z0-9_]*$/;

/** Path of the classification file under `root` (the repository root). */
export function classificationPath(root) {
  return path.join(root, "scripts", "release", "env-classification.json");
}

/** Parse and shape-check the classification file: `{ public: string[], secret: string[] }`. */
export function loadEnvClassification(root) {
  const file = classificationPath(root);
  const doc = JSON.parse(fs.readFileSync(file, "utf8"));
  for (const key of ["public", "secret"]) {
    if (!Array.isArray(doc[key]) || !doc[key].every((n) => typeof n === "string")) {
      throw new Error(`${file}: "${key}" must be an array of variable names`);
    }
  }
  return { public: [...doc.public], secret: [...doc.secret] };
}

/**
 * Everything wrong with a classification, as messages (names only). Empty when it is safe:
 * - every name is an upper-case variable name;
 * - no `public` name looks like a credential (`SECRET`, `API_KEY`, `TOKEN`, `PASSWORD`);
 * - no name is in both lists;
 * - no `secret` name begins with a `public` name, because Vite matches `envPrefix` entries as
 *   prefixes: `SIGIL_KICK_CLIENT_ID` would also expose a `SIGIL_KICK_CLIENT_ID_SECRET`.
 */
export function classificationProblems(classification) {
  const problems = [];
  const { public: pub, secret } = classification;
  for (const name of [...pub, ...secret]) {
    if (!NAME_SHAPE.test(name)) problems.push(`${JSON.stringify(name)} is not an upper-case variable name`);
  }
  for (const name of pub) {
    if (SECRET_NAME_PATTERN.test(name)) problems.push(`public name ${name} looks like a credential`);
    if (secret.includes(name)) problems.push(`${name} is listed as both public and secret`);
  }
  for (const s of secret) {
    for (const p of pub) {
      if (s !== p && s.startsWith(p)) problems.push(`secret ${s} starts with public ${p}: Vite's prefix match would expose it`);
    }
  }
  return problems;
}

/**
 * The value for Vite's `envPrefix`: the public names (each a complete name, so the prefix match is an
 * exact match). Throws, so the build stops, when the classification is unsafe.
 */
export function viteEnvPrefix(classification) {
  const problems = classificationProblems(classification);
  if (problems.length > 0) {
    throw new Error(`scripts/release/env-classification.json is unsafe:\n  ${problems.join("\n  ")}`);
  }
  return [...classification.public];
}
