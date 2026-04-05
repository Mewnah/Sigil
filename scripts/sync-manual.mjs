/**
 * Copies docs/manual → public/manual so the in-app Help viewer and static server
 * can fetch markdown and assets at /manual/...
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const src = path.join(root, "docs", "manual");
const dest = path.join(root, "public", "manual");

function copyRecursive(from, to) {
  if (!fs.existsSync(from)) {
    console.warn("[sync-manual] Source missing:", from);
    return;
  }
  fs.mkdirSync(to, { recursive: true });
  for (const ent of fs.readdirSync(from, { withFileTypes: true })) {
    const sp = path.join(from, ent.name);
    const dp = path.join(to, ent.name);
    if (ent.isDirectory()) {
      copyRecursive(sp, dp);
    } else {
      fs.copyFileSync(sp, dp);
    }
  }
}

copyRecursive(src, dest);

const backendsSrc = path.join(root, "docs", "BACKENDS.md");
const backendsPublic = path.join(root, "public", "BACKENDS.md");
if (fs.existsSync(backendsSrc)) {
  fs.copyFileSync(backendsSrc, backendsPublic);
  console.log("[sync-manual] Copied BACKENDS.md → public/BACKENDS.md");
}

const archSrc = path.join(root, "ARCHITECTURE.md");
if (fs.existsSync(archSrc)) {
  fs.copyFileSync(archSrc, path.join(root, "public", "ARCHITECTURE.md"));
  console.log("[sync-manual] Copied ARCHITECTURE.md → public/ARCHITECTURE.md");
}

const readmeSrc = path.join(root, "README.md");
if (fs.existsSync(readmeSrc)) {
  fs.copyFileSync(readmeSrc, path.join(root, "public", "repo-README.md"));
  console.log("[sync-manual] Copied README.md → public/repo-README.md");
}

console.log("[sync-manual] Synced", src, "→", dest);
