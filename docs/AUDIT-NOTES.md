# Dependency audit notes

Snapshot taken during integration work. Re-run before releases.

## npm (`npm audit`)

Last run reported **12 vulnerabilities** (6 moderate, 6 high), mostly in transitive deps (vite/esbuild, rollup, workbox, lodash, ajv, nanoid, minimatch). Many fixes require **breaking** upgrades (`npm audit fix --force`, e.g. Vite 8, PWA plugin).

**Recommendation:** Plan a dedicated toolchain upgrade (Vite 6+/8, `@vitejs/plugin-react`, `vite-plugin-pwa`) on a separate PR; do not blindly `--force` on the integration branch without full UI and Tauri dev testing.

## Rust (`cargo audit`)

`cargo audit` requires [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit): `cargo install cargo-audit` then `cd src-tauri && cargo audit`.

Track `whisper-rs` / `whisper-rs-sys` alignment in `Cargo.toml` when updating whisper.cpp bindings.
