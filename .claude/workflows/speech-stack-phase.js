export const meta = {
  name: 'speech-stack-phase',
  description: 'Run one phase of the speech-stack implementation plan in place: scaffold, implement (3 at a time, disjoint files), integrate, adversarial review, fix, final gate',
  phases: [
    { title: 'Scaffold', detail: 'checkpoint + hotspot scaffolding and stubs (one agent)' },
    { title: 'Implement', detail: 'workstream implementers, max 3 at once, disjoint owned files' },
    { title: 'Integrate', detail: 'apply hotspot requests, i18n staging, docs; full gates' },
    { title: 'Review', detail: 'adversarial review per workstream' },
    { title: 'Fix', detail: 'fix confirmed findings per workstream' },
    { title: 'Gate', detail: 'final full gates and repair' },
  ],
}

const A = args
const CONC = A.concurrency || 3

async function pool(items, n, fn) {
  const out = new Array(items.length)
  let next = 0
  async function worker() {
    while (true) {
      const k = next++
      if (k >= items.length) return
      try { out[k] = await fn(items[k], k) } catch (e) { out[k] = null }
    }
  }
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, () => worker()))
  return out
}

const COMMON = [
  'You are implementing part of phase ' + A.phase + ' of the Sigil speech-stack plan, working IN PLACE in the repo C:/Users/chris/Desktop/Sigil (Windows 11; use the Bash tool with POSIX syntax or PowerShell as convenient).',
  'BEFORE STARTING read: (1) ' + A.planMd + ' lines 1-481 (principles, decisions, VC best-engine design, phases, GLOBAL RULES, deferred items) and follow the global rules exactly; (2) your workstream entry by id in ' + A.planJson + ' (JSON array under key "workstreams"); (3) the relevant subsystem map in ' + A.mapsJson + '; (4) docs/SPEECH_STACK_RESEARCH_2026-09.md sections relevant to your workstream; (5) CLAUDE.md conventions.',
  'GIT: the repo is on the local branch speech-stack; HEAD is the commit of the previous phase, so `git diff HEAD` / `git status` show exactly the changes of this phase. The orchestrator commits at the end of each phase. Agents must NOT commit, push, branch, stash, reset, checkout, clean, restore or create worktrees (read-only git only), and nothing is ever pushed to GitHub. HARD RULES: never revert or reformat code you do not own; other agents are editing other files concurrently. Edit ONLY your workstream owned_paths/new_paths; for any change needed in a hotspot file (Cargo.toml/lock, build.rs, main.rs, services/mod.rs, capabilities, tauri.conf.json, src/core/schema.ts, services-registry.ts, src/core/index.ts, servicePanels.tsx, SigilNavigation.tsx, inspector/index.tsx, src/types.ts, public/i18n/*, package.json, lockfiles, ci.yml, .gitignore, neural_audio/mod.rs, vc/mod.rs) put it in hotspot_requests instead (unless you ARE the scaffold/integrate agent). UI strings go to scripts/i18n-staging/<WS-ID>.json (English, nested like public/i18n/en/translation.json).',
  'NETWORK: the user\'s router blocks the PC on bursts of connections. Do at most a few sequential web lookups (WebSearch/WebFetch) only when essential; never parallel downloads; never clone repos; download only small files explicitly listed in your workstream external_assets and never anything over 50 MB in this phase (implement the code path and document the manual test instead). pnpm install must use --network-concurrency=2.',
  'BUILD: keep your own files compiling. Useful checks: `pnpm typecheck`; `cargo check --manifest-path src-tauri/Cargo.toml` (shared target dir; waiting on the cargo lock is normal); targeted `cargo test --manifest-path src-tauri/Cargo.toml <filter>`; `pnpm test:unit` once vitest exists. If a failure is in a file you do not own and not caused by you, ignore it (another agent is mid-edit) and re-check later; never edit it.',
  'ALREADY DONE by the orchestrator (verify, do not redo): ' + A.doneNotes,
  A.extraNotes || '',
].join('\n\n')

const REPORT = {
  type: 'object',
  required: ['ws_id', 'status', 'summary', 'files_changed', 'hotspot_requests', 'i18n_staging_file', 'tests', 'verification', 'known_issues', 'doc_deltas'],
  properties: {
    ws_id: { type: 'string' },
    status: { type: 'string', enum: ['done', 'partial', 'blocked'] },
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    hotspot_requests: { type: 'array', items: { type: 'object', required: ['file', 'change'], properties: { file: { type: 'string' }, change: { type: 'string' } } } },
    i18n_staging_file: { type: 'string' },
    tests: { type: 'array', items: { type: 'string' } },
    verification: { type: 'array', items: { type: 'object', required: ['command', 'result'], properties: { command: { type: 'string' }, result: { type: 'string' } } } },
    known_issues: { type: 'array', items: { type: 'string' } },
    doc_deltas: { type: 'array', items: { type: 'string' } },
    external_assets_fetched: { type: 'array', items: { type: 'string' } },
  },
}
const REVIEW = {
  type: 'object',
  required: ['ws_id', 'verdict', 'findings'],
  properties: {
    ws_id: { type: 'string' },
    verdict: { type: 'string', enum: ['clean', 'issues'] },
    findings: { type: 'array', items: { type: 'object', required: ['severity', 'file', 'issue', 'failure_scenario', 'fix'], properties: { severity: { type: 'string', enum: ['high', 'medium', 'low'] }, file: { type: 'string' }, line: { type: 'number' }, issue: { type: 'string' }, failure_scenario: { type: 'string' }, fix: { type: 'string' } } } },
  },
}
const GATE = {
  type: 'object',
  required: ['status', 'gates', 'fixes_applied', 'remaining_issues'],
  properties: {
    status: { type: 'string', enum: ['green', 'red'] },
    gates: { type: 'array', items: { type: 'object', required: ['command', 'result'], properties: { command: { type: 'string' }, result: { type: 'string' } } } },
    fixes_applied: { type: 'array', items: { type: 'string' } },
    remaining_issues: { type: 'array', items: { type: 'string' } },
  },
}

const GATES = 'pnpm typecheck; pnpm i18n:check; pnpm test:unit (if the script exists); pnpm build; cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets (no NEW warnings in touched files); cargo test --manifest-path src-tauri/Cargo.toml (dist/ must exist)'

// ── Scaffold ──
phase('Scaffold')
const scaffoldReports = []
for (const id of (A.scaffoldIds || [])) {
  const r = await agent(COMMON + '\n\nYOU ARE THE SCAFFOLD AGENT for workstream ' + id + ' (you may edit hotspot files). First take the phase checkpoint outside the repo (belt and braces; HEAD is the previous phase commit): `git diff --binary HEAD > ' + A.checkpointDir + '/' + id + '-tracked.diff` and a zip of untracked files (`git ls-files --others --exclude-standard`, excluding node_modules, target, dist) at ' + A.checkpointDir + '/' + id + '-untracked.zip. Then implement the workstream exactly (stubs with frozen signatures for every implementer of this phase, conventions, splits, tooling). Keep behaviour unchanged unless the workstream says otherwise. Finish with pnpm typecheck and cargo check green, and run the existing Rust tests touched by any split. Report precisely which stubs/signatures you froze so implementers can rely on them (put them in summary).',
    { label: 'scaffold:' + id, phase: 'Scaffold', schema: REPORT })
  scaffoldReports.push(r)
  log('Scaffold ' + id + ': ' + (r ? r.status : 'FAILED'))
}
// A.scaffoldNotes / A.priorReports / A.stopAfter let a relaunch reuse work finished outside this run.
const scaffoldText = A.scaffoldNotes || JSON.stringify(scaffoldReports.filter(Boolean).map(r => ({ ws_id: r.ws_id, status: r.status, summary: r.summary, known_issues: r.known_issues })))

// ── Implement ──
phase('Implement')
const reports = await pool(A.implementIds || [], CONC, id => agent(
  COMMON + '\n\nSCAFFOLD RESULTS (stubs and frozen signatures you can rely on):\n' + scaffoldText + '\n\nYOUR WORKSTREAM: ' + id + '. Implement it completely and to a high standard (this is production code for real users; quality first). Add focused tests. Meet its acceptance criteria. If something is genuinely infeasible here, implement the documented fallback and explain in known_issues. Report every hotspot change you need.',
  { label: 'impl:' + id, phase: 'Implement', schema: REPORT }))
const okReports = (A.priorReports || []).concat(reports.filter(Boolean))
log('Implemented: ' + okReports.map(r => r.ws_id + '=' + r.status).join(', '))
if (A.stopAfter === 'Implement') return { scaffoldReports, reports: reports.filter(Boolean) }

// ── Integrate ──
phase('Integrate')
const integrateReports = []
for (const id of (A.integrateIds || [])) {
  const r = await agent(COMMON + '\n\nYOU ARE THE INTEGRATE AGENT for workstream ' + id + ' (you may edit hotspot files and docs). Implementer reports (JSON):\n' + JSON.stringify(okReports) + '\n\nScaffold reports:\n' + scaffoldText + '\n\nApply every reasonable hotspot request (Cargo deps, registrations, capabilities, schema keys, bundle resources, main.rs hooks), merge i18n staging (node scripts/merge-i18n-staging.mjs then pnpm i18n:sync; if the merger does not exist yet, merge the staging JSON into public/i18n/en/translation.json by hand and run pnpm i18n:sync), apply doc deltas to the docs/ sources, then run the full gates: ' + GATES + '. Fix integration breakages with minimal edits (you may touch any file for integration fixes, but do not rewrite implementers\' logic). Report what you applied and the gate results.',
    { label: 'integrate:' + id, phase: 'Integrate', schema: REPORT })
  integrateReports.push(r)
}

// ── Review ──
phase('Review')
const reviews = await pool(okReports, CONC, r => agent(
  COMMON + '\n\nYOU ARE AN ADVERSARIAL REVIEWER for workstream ' + r.ws_id + '. Its report:\n' + JSON.stringify(r) + '\n\nRead the workstream entry and review the actual changes (git diff HEAD -- <files> for tracked files; read new files fully). Hunt for real defects: logic bugs, broken edge cases, real-time-safety violations (allocation/locking/logging in audio callbacks), epoch/race bugs, resource leaks, security issues (untrusted input, key leakage, origin checks), missed acceptance criteria, regressions to existing behaviour, persisted-enum renames, missing tests. Only report issues with a concrete failure scenario; no style nits. Do NOT edit files.',
  { label: 'review:' + r.ws_id, phase: 'Review', schema: REVIEW }))

// ── Fix ──
phase('Fix')
const toFix = reviews.filter(Boolean).filter(v => (v.findings || []).some(f => f.severity !== 'low'))
const fixes = await pool(toFix, CONC, v => agent(
  COMMON + '\n\nYOU ARE THE FIXER for workstream ' + v.ws_id + '. Review findings (JSON):\n' + JSON.stringify(v.findings) + '\n\nVerify each finding against the code; fix every real one (high and medium first, low ones too when cheap), staying inside the workstream owned/new paths (hotspot needs go in hotspot_requests). Add or update tests that would have caught them. Re-run the relevant checks.',
  { label: 'fix:' + v.ws_id, phase: 'Fix', schema: REPORT }))

// ── Gate ──
phase('Gate')
const lateHotspots = fixes.filter(Boolean).flatMap(f => f.hotspot_requests || [])
const gate = await agent(COMMON + '\n\nYOU ARE THE FINAL GATE AGENT for phase ' + A.phase + ' (you may edit any file, minimally). Apply these late hotspot requests from fixers if still needed: ' + JSON.stringify(lateHotspots) + '. Merge any new i18n staging files. Then run every gate: ' + GATES + '. Fix all failures with minimal, correct edits (never by deleting tests or weakening checks). Then check the phase exit criteria from the plan and report which are met, which need a manual/hardware check, and which are not met.',
  { label: 'gate:' + A.phase, phase: 'Gate', schema: GATE })

return { scaffoldReports, reports: okReports, integrateReports, reviews: reviews.filter(Boolean), fixes: fixes.filter(Boolean), gate }
