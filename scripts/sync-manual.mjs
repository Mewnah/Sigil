/**
 * Copies docs/manual → public/manual so the in-app Help viewer and static server
 * can fetch markdown and assets at /manual/...
 * Also copies consolidated reference docs to public/docs/ and root stubs to public/.
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

if (fs.existsSync(dest)) {
  fs.rmSync(dest, { recursive: true });
}
copyRecursive(src, dest);

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

const publicDocsDir = path.join(root, "public", "docs");
fs.mkdirSync(publicDocsDir, { recursive: true });

const consolidatedDocs = [
  "OPERATOR_REFERENCE.md",
  "DEVELOPER_GUIDE.md",
  "RELEASE_AND_ENGINEERING.md",
  "DOCUMENTATION_GAPS_AND_RECOMMENDATIONS.md",
  "README.md",
];

for (const name of consolidatedDocs) {
  const from = path.join(root, "docs", name);
  if (fs.existsSync(from)) {
    fs.copyFileSync(from, path.join(publicDocsDir, name));
    console.log("[sync-manual] Copied docs/" + name + " → public/docs/" + name);
  }
}

const obsoletePublic = [
  path.join(root, "public", "BACKENDS.md"),
  path.join(root, "public", "STT_RELIABILITY.md"),
  path.join(root, "public", "docs", "NETWORK_THREAT_MODEL.md"),
];
for (const fp of obsoletePublic) {
  if (fs.existsSync(fp)) {
    fs.unlinkSync(fp);
    console.log("[sync-manual] Removed obsolete", path.relative(root, fp));
  }
}

console.log("[sync-manual] Synced", src, "→", dest);
