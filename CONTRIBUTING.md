# Contributing to Sigil

## Setup

- [Node.js](https://nodejs.org/) (LTS) and [pnpm](https://pnpm.io/)
- [Rust](https://rustup.rs/) (stable) for Tauri

```bash
pnpm install
pnpm typecheck
pnpm tauri dev
```

## Checks before a pull request

```bash
pnpm typecheck
pnpm i18n:check
pnpm test:e2e
```

## Internationalization (i18n)

- Strings in `public/i18n/*/translation.json` sometimes include **repository paths** (for example `docs/OPERATOR_REFERENCE.md#…`). **Do not translate** the path or filename segments; translate the surrounding sentence only so links stay valid.
- English (`en`) is the source of truth for keys; run `pnpm i18n:check` after edits.

## Releases (maintainers)

Before tagging a release:

- Refresh **dependency audits** and the log in [docs/RELEASE_AND_ENGINEERING.md](docs/RELEASE_AND_ENGINEERING.md#dependency-audit-notes) (`pnpm audit`, `cargo audit` in `src-tauri`).
- Run the **local packaged smoke** steps in [docs/RELEASE_AND_ENGINEERING.md](docs/RELEASE_AND_ENGINEERING.md#local-packaged-smoke-manual) on a real `pnpm tauri build` artifact (Playwright in CI does not replace this).

## Documentation

- **User manual:** `docs/manual/` — synced to `public/manual/` via `scripts/sync-manual.mjs` on `pnpm dev` / `pnpm build`.
- **Architecture and systems:** [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md), [docs/OPERATOR_REFERENCE.md](docs/OPERATOR_REFERENCE.md), [docs/RELEASE_AND_ENGINEERING.md](docs/RELEASE_AND_ENGINEERING.md).
- **Remaining doc backlog:** [docs/DOCUMENTATION_GAPS_AND_RECOMMENDATIONS.md](docs/DOCUMENTATION_GAPS_AND_RECOMMENDATIONS.md).

## Code layout (short)

- `src/client/` — document/canvas code safe for `/client` (no Tauri `invoke`).
- `src/core/` — host services and UI shell.
- `src/shared/` — pubsub, peer bridge.

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for full layering and connectivity.

## License

Contributions are accepted under the same license as the project — see [LICENSE.md](LICENSE.md).
