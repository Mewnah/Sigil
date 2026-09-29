# i18n staging

Workstreams never edit `public/i18n/*/translation.json` directly: 21 locale files with CI-enforced key
parity are a merge magnet. Instead, each workstream puts its **new English UI strings** in one file
here, named after the workstream id:

```
scripts/i18n-staging/<WS-ID>.json      e.g. P0-B.json, P0-D.json
```

## Format

The same nested shape as `public/i18n/en/translation.json`, English only:

```json
{
  "voice_changer": {
    "latency_budget": "Latency budget",
    "latency_budget_hint": "Older audio is dropped instead of queued once the backlog exceeds this."
  }
}
```

- Only add keys. Reuse existing keys where the wording already exists.
- To change existing English wording, stage the new value; the merge refuses it unless the
  integrator passes `--allow-en-changes` (other locales keep their translation).
- To delete keys from every locale, list them under a top-level `"$remove": ["tts.tiktok_title"]`.
- Keys starting with `$` are reserved; keys may not contain dots.
- Use `t("section.key")` in code with the staged key; until the merge the UI shows the key name,
  which is expected during a phase.

## Merge (INTEGRATE step only)

```
node scripts/merge-i18n-staging.mjs --dry-run    # report, write nothing
node scripts/merge-i18n-staging.mjs              # add staged keys to en/translation.json
pnpm i18n:sync && pnpm i18n:check                # copy new English keys to every locale, verify parity
```

The merge is non-destructive: it adds keys to English only, fails (writing nothing) on conflicting
values, and touches other locales only for explicit `$remove` entries. After a successful merge the
integrator deletes the merged staging files. The retired `merge-neural-i18n.mjs` and
`merge-http-tts-i18n.mjs` overwrote whole locale sections with English and must not be used.
