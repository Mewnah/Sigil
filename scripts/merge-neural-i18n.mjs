/**
 * RETIRED (P0-S0): this script overwrote whole locale sections with English placeholders and
 * destroyed existing translations. Stage new English strings in scripts/i18n-staging/<WS-ID>.json
 * and merge them with:
 *   node scripts/merge-i18n-staging.mjs && pnpm i18n:sync && pnpm i18n:check
 * See scripts/i18n-staging/README.md.
 */
console.error(
  "scripts/merge-neural-i18n.mjs is retired (it overwrote translations). Use scripts/merge-i18n-staging.mjs, then pnpm i18n:sync. See scripts/i18n-staging/README.md.",
);
process.exit(1);
