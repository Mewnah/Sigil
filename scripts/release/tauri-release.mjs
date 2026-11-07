// `pnpm tauri:release`: `tauri build` with the builder's home folder stripped from what the compilers embed in
// Sigil.exe (Rust panic locations, C/C++ __FILE__ in whisper.cpp/ggml, the PDB path), so a published installer
// does not carry the build machine's user name. verify-bundle.ps1 checks the result.
import { spawnSync } from "node:child_process";
import os from "node:os";

const home = os.homedir();
const env = { ...process.env };

// CARGO_ENCODED_RUSTFLAGS separates flags with \x1f, so a home folder with spaces stays one flag.
const previous = env.CARGO_ENCODED_RUSTFLAGS || (env.RUSTFLAGS || "").trim().split(/\s+/).filter(Boolean).join("\x1f");
env.CARGO_ENCODED_RUSTFLAGS = [previous, `--remap-path-prefix=${home}=~`, "-Clink-arg=/PDBALTPATH:%_PDB%"]
  .filter(Boolean)
  .join("\x1f");
delete env.RUSTFLAGS;

if (process.platform === "win32") {
  // shortcut: CFLAGS/CXXFLAGS are split on spaces, so a home folder with a space keeps its C/C++ paths; pass the flag through cmake's toolchain if that ever matters.
  if (/\s/.test(home)) console.warn(`tauri-release: home folder "${home}" has a space; C/C++ __FILE__ paths are not trimmed`);
  else for (const key of ["CFLAGS", "CXXFLAGS"]) env[key] = [env[key], `/d1trimfile:${home}\\`].filter(Boolean).join(" ");
}

const r = spawnSync("pnpm", ["exec", "tauri", "build", ...process.argv.slice(2)], { stdio: "inherit", env, shell: process.platform === "win32" });
process.exit(r.status ?? 1);
