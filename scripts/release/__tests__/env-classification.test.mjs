import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { loadEnv } from "vite";
import {
  SECRET_NAME_PATTERN,
  classificationPath,
  classificationProblems,
  loadEnvClassification,
  viteEnvPrefix,
} from "../env-classification.mjs";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

const tmpRoots = [];
function tmp() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-envclass-"));
  tmpRoots.push(d);
  return d;
}
afterEach(() => {
  while (tmpRoots.length) fs.rmSync(tmpRoots.pop(), { recursive: true, force: true });
});

describe("the repository's env-classification.json", () => {
  const classification = loadEnvClassification(ROOT);

  it("is safe: nothing public looks like a credential, nothing is both, no prefix can expose a secret", () => {
    expect(classificationProblems(classification)).toEqual([]);
    expect(() => viteEnvPrefix(classification)).not.toThrow();
  });

  it("classifies the OAuth client secrets and the Google Fonts key as secret and the client ids as public", () => {
    for (const name of [
      "SIGIL_KICK_CLIENT_SECRET",
      "CURSES_KICK_CLIENT_SECRET",
      "SIGIL_TWITCH_CLIENT_SECRET",
      "CURSES_TWITCH_CLIENT_SECRET",
      "SIGIL_GOOGLE_FONTS_API_KEY",
      "VITE_GOOGLE_FONTS_API_KEY",
    ]) {
      expect(classification.secret, name).toContain(name);
      expect(classification.public, name).not.toContain(name);
    }
    for (const name of ["SIGIL_KICK_CLIENT_ID", "SIGIL_TWITCH_CLIENT_ID", "CURSES_KICK_CLIENT_ID", "CURSES_TWITCH_CLIENT_ID", "TAURI_DEBUG"]) {
      expect(classification.public, name).toContain(name);
    }
  });

  it("every secret name is one that check-no-secrets.mjs would pick up by name alone", () => {
    for (const name of classification.secret) expect(SECRET_NAME_PATTERN.test(name), name).toBe(true);
  });

  it("is a file of names only", () => {
    const text = fs.readFileSync(classificationPath(ROOT), "utf8");
    expect(JSON.parse(text)).toHaveProperty("secret");
    // No value-like content: every string in the lists is an upper-case identifier.
    for (const name of [...classification.public, ...classification.secret]) expect(name).toMatch(/^[A-Z][A-Z0-9_]*$/);
  });
});

describe("classificationProblems", () => {
  const ok = { public: ["SIGIL_KICK_CLIENT_ID"], secret: ["SIGIL_KICK_CLIENT_SECRET"] };

  it("accepts a sound classification", () => {
    expect(classificationProblems(ok)).toEqual([]);
    expect(viteEnvPrefix(ok)).toEqual(["SIGIL_KICK_CLIENT_ID"]);
  });

  it("flags a public name that looks like a credential", () => {
    for (const name of ["SIGIL_KICK_CLIENT_SECRET", "VITE_SOME_API_KEY", "SIGIL_AUTH_TOKEN", "SIGIL_DB_PASSWORD", "sigil_api_key"]) {
      expect(classificationProblems({ public: [name], secret: [] }).join(), name).toMatch(/credential|upper-case/);
    }
  });

  it("flags a name in both lists", () => {
    expect(classificationProblems({ public: ["SIGIL_X"], secret: ["SIGIL_X"] }).join()).toContain("both public and secret");
  });

  it("flags a secret that a public name is a prefix of (Vite matches envPrefix by prefix)", () => {
    const problems = classificationProblems({ public: ["SIGIL_KICK_CLIENT_ID"], secret: ["SIGIL_KICK_CLIENT_ID_SECRET"] });
    expect(problems.join()).toContain("prefix match would expose");
  });

  it("flags names that are not upper-case identifiers (including a bare prefix like VITE_ in lower case)", () => {
    expect(classificationProblems({ public: ["vite_"], secret: [] }).length).toBeGreaterThan(0);
    expect(classificationProblems({ public: ["WITH SPACE"], secret: [] }).length).toBeGreaterThan(0);
  });

  it("viteEnvPrefix throws on an unsafe classification and returns a copy otherwise", () => {
    expect(() => viteEnvPrefix({ public: ["SIGIL_API_KEY"], secret: [] })).toThrow(/unsafe/);
    const copy = viteEnvPrefix(ok);
    copy.push("X");
    expect(ok.public).toEqual(["SIGIL_KICK_CLIENT_ID"]);
  });

  it("loadEnvClassification rejects a malformed file", () => {
    const root = tmp();
    fs.mkdirSync(path.join(root, "scripts", "release"), { recursive: true });
    fs.writeFileSync(classificationPath(root), JSON.stringify({ public: "nope", secret: [] }));
    expect(() => loadEnvClassification(root)).toThrow(/array of variable names/);
  });
});

describe("what Vite lets into import.meta.env with the allow-list", () => {
  // Fake values; a scratch env dir so the repository's real .env is never read.
  const FAKE = {
    SIGIL_KICK_CLIENT_ID: "fake-public-kick-id",
    SIGIL_KICK_CLIENT_SECRET: "fake-kick-secret-value",
    CURSES_KICK_CLIENT_SECRET: "fake-legacy-kick-secret",
    SIGIL_TWITCH_CLIENT_ID: "fake-public-twitch-id",
    SIGIL_TWITCH_CLIENT_SECRET: "fake-twitch-secret-value",
    VITE_GOOGLE_FONTS_API_KEY: "fake-fonts-key-value",
    SIGIL_GOOGLE_FONTS_API_KEY: "fake-fonts-key-value-2",
    SIGIL_GPU_LOCK: "C:\\some\\private\\path",
    VITE_ANYTHING_ELSE: "fake-other",
    TAURI_SIGNING_PRIVATE_KEY: "fake-signing-key",
  };

  function exposed(prefixes) {
    const dir = tmp();
    fs.writeFileSync(path.join(dir, ".env"), Object.entries(FAKE).map(([k, v]) => `${k}=${v}`).join("\n"));
    return loadEnv("production", dir, prefixes);
  }

  it("exposes only the public names, none of the secrets", () => {
    const classification = loadEnvClassification(ROOT);
    const env = exposed(viteEnvPrefix(classification));
    for (const name of classification.secret) expect(env, name).not.toHaveProperty(name);
    for (const name of ["VITE_ANYTHING_ELSE", "SIGIL_GPU_LOCK", "TAURI_SIGNING_PRIVATE_KEY"]) expect(env, name).not.toHaveProperty(name);
    expect(Object.keys(env).filter((k) => !classification.public.includes(k))).toEqual([]);
    // Booleans only: a failing assertion must never print an env value (a shell may export a real one).
    expect(typeof env.SIGIL_KICK_CLIENT_ID === "string" && env.SIGIL_KICK_CLIENT_ID.length > 0).toBe(true);
    expect(typeof env.SIGIL_TWITCH_CLIENT_ID === "string" && env.SIGIL_TWITCH_CLIENT_ID.length > 0).toBe(true);
    expect(Object.values(env).includes(FAKE.SIGIL_KICK_CLIENT_SECRET)).toBe(false);
  });

  it("(why this exists) the old prefixes exposed the secrets", () => {
    const env = exposed(["VITE_", "TAURI_", "SIGIL_", "CURSES_"]);
    expect(Object.keys(env)).toEqual(
      expect.arrayContaining(["SIGIL_KICK_CLIENT_SECRET", "VITE_GOOGLE_FONTS_API_KEY", "TAURI_SIGNING_PRIVATE_KEY"]),
    );
  });
});
