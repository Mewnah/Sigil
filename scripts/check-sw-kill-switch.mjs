#!/usr/bin/env node
/**
 * Opt-in check of public/sw.js (the kill switch for the caching service worker that Curses and early
 * Sigil builds registered at /sw.js). It is NOT part of `pnpm test:unit` or CI: it needs the
 * Playwright Chromium (`pnpm test:e2e:install`) and takes about 30 s.
 *
 *   pnpm check:sw-kill
 *   node scripts/check-sw-kill-switch.mjs --scenario "my worker=path/to/sw.js,3,real" --json out.json
 *
 * What it does, per scenario, in headless Chromium against a throw-away server on 127.0.0.1 (random
 * port, nothing listens on another address, a temporary profile):
 *   1. The "old server" serves the Curses page and a Workbox-like worker at /sw.js: it precaches the
 *      page, calls skipWaiting() at the top, never calls clients.claim() (vite-plugin-pwa with only
 *      `workbox.skipWaiting: true`, as in Curses 177f5bc), and answers navigations from its cache.
 *   2. The old page is the REAL Curses page code: register('/sw.js', { scope: '/' }) and, on
 *      `updatefound`, `await registration.update(); location.reload()`. `--scenario ...,register-only`
 *      drops that handler.
 *   3. Windows: /client (controlled by the old worker after a reload), /mic.html (opened afterwards,
 *      so the old worker's navigation fallback shows the Curses page in it) and, with 3 windows,
 *      /obs-captions.html opened BEFORE the worker existed (never controlled).
 *   4. The server switches to "Sigil" (current pages, the worker under test at /sw.js) and the /client
 *      window is reloaded once, which is all an OBS source does.
 *
 * Passing means: /client shows the Sigil page and has no registration within the limit, every cache
 * is gone, /mic.html shows its own page, the never-controlled window was not navigated, and nothing
 * keeps reloading afterwards. Every state change is printed with its time, so a stall is visible.
 */
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_WORKER = path.join(ROOT, "public", "sw.js");
const RECOVERY_LIMIT_MS = Number(process.env.SW_KILL_LIMIT_MS ?? 10_000);
const STABLE_FOR_MS = 4_000;
const NUDGE_AFTER_MS = 12_000;
const POLL_MS = 250;

const args = process.argv.slice(2);
const argValues = (flag) => args.flatMap((a, i) => (a === flag && args[i + 1] ? [args[i + 1]] : []));

/** `label=worker.js,windows,page` with windows 1|3 and page real|register-only. */
function parseScenario(text) {
  const eq = text.indexOf("=");
  const label = eq > 0 ? text.slice(0, eq) : text;
  const [worker, windows = "3", page = "real"] = (eq > 0 ? text.slice(eq + 1) : text).split(",");
  return { label, worker: path.resolve(worker), windows: Number(windows), page };
}

const scenarios = argValues("--scenario").length
  ? argValues("--scenario").map(parseScenario)
  : [
      { label: "public/sw.js, 3 windows, real Curses handler", worker: DEFAULT_WORKER, windows: 3, page: "real" },
      { label: "public/sw.js, 1 window, real Curses handler", worker: DEFAULT_WORKER, windows: 1, page: "real" },
      { label: "public/sw.js, 1 window, register only", worker: DEFAULT_WORKER, windows: 1, page: "register-only" },
    ];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const withTimeout = (promise, ms, fallback) =>
  Promise.race([promise, new Promise((resolve) => setTimeout(() => resolve(fallback), ms))]);

const OLD_PAGE = (variant) => `<!doctype html><meta charset="utf-8"><title>old</title>
<h1 id="h">OLD CURSES</h1>
<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js', { scope: '/', type: 'classic' })${
    variant === "real"
      ? `.then((sw) => {
    sw.addEventListener("updatefound", async _ => { console.log("found update"); await sw.update(); location.reload(); });
  })`
      : ""
  };
}
</script>`;
const PAGE = (name) => `<!doctype html><meta charset="utf-8"><title>${name}</title><h1 id="h">${name}</h1>`;

/** A Workbox-6-like worker as vite-plugin-pwa generates it with `workbox.skipWaiting: true`. */
const OLD_WORKER = (oldPage) => `
const CACHE = 'workbox-precache-v2-' + self.registration.scope;
self.skipWaiting();
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.put('/index.html', new Response(${JSON.stringify(oldPage)}, { headers: { 'content-type': 'text/html' } }))));
});
self.addEventListener('fetch', (e) => {
  if (e.request.mode !== 'navigate') return;
  e.respondWith(caches.open(CACHE).then((c) => c.match('/index.html')).then((hit) => hit || fetch(e.request)));
});
`;

async function runScenario(scenario) {
  const t0 = Date.now();
  const log = [];
  const note = (message) => log.push(`${String(Date.now() - t0).padStart(6)} ms  ${message}`);
  const workerSource = fs.readFileSync(scenario.worker);
  const oldPage = OLD_PAGE(scenario.page);
  let mode = "old";
  let triggeredAt = null;
  const swHits = [];

  const server = http.createServer((req, res) => {
    const p = new URL(req.url, "http://x").pathname;
    if (p === "/sw.js") {
      if (triggeredAt !== null) swHits.push(Date.now() - triggeredAt);
      note(`server[${mode}] GET /sw.js`);
      if (mode === "old") {
        res.writeHead(200, { "content-type": "text/javascript" });
        return res.end(OLD_WORKER(oldPage));
      }
      res.writeHead(200, { "content-type": "text/javascript; charset=utf-8", "cache-control": "no-cache" });
      return res.end(workerSource);
    }
    if (p === "/favicon.ico") {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "content-type": "text/html" });
    if (p.startsWith("/obs-captions")) return res.end(PAGE("CAPTIONS"));
    if (mode === "old") return res.end(oldPage);
    return res.end(PAGE(p.startsWith("/mic") ? "MIC" : "SIGIL"));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "sw-kill-"));
  const context = await chromium.launchPersistentContext(profile, { headless: true });
  const navigations = {};
  const open = async (name) => {
    const page = await context.newPage();
    navigations[name] = 0;
    page.on("framenavigated", (frame) => {
      if (frame !== page.mainFrame()) return;
      navigations[name] += 1;
      note(`[${name}] navigated ${frame.url().replace(base, "")}`);
    });
    page.on("console", (m) => note(`[${name}] console: ${m.text()}`));
    return page;
  };
  const snap = (page) =>
    withTimeout(
      page
        .evaluate(async () => {
          const reg = await navigator.serviceWorker.getRegistration();
          return {
            h: document.getElementById("h")?.textContent ?? null,
            ctl: navigator.serviceWorker.controller?.state ?? null,
            regs: (await navigator.serviceWorker.getRegistrations()).length,
            caches: (await caches.keys()).length,
            installing: reg?.installing?.state ?? null,
            waiting: reg?.waiting?.state ?? null,
            active: reg?.active?.state ?? null,
          };
        })
        .catch((e) => ({ err: String(e.message).split("\n")[0] })),
      1500,
      { err: "snapshot timed out" },
    );

  const result = { label: scenario.label, ok: false, problems: [], recoveredMs: null, nudge: null };
  try {
    const captions = scenario.windows === 3 ? await open("captions") : null;
    if (captions) await captions.goto(`${base}/obs-captions.html`, { timeout: 8000 });

    const client = await open("client");
    await client.goto(`${base}/client?host=127.0.0.1&port=3030`, { timeout: 8000 });
    await withTimeout(client.evaluate(() => navigator.serviceWorker.ready.then(() => true)), 5000, null);
    await client.reload({ timeout: 8000 });
    note(`old state: ${JSON.stringify(await snap(client))}`);

    let mic = null;
    if (scenario.windows === 3) {
      mic = await open("mic");
      await mic.goto(`${base}/mic.html`, { timeout: 8000 });
      note(`old mic window (hijacked by the old fallback): ${JSON.stringify(await snap(mic))}`);
    }

    mode = "sigil";
    triggeredAt = Date.now();
    for (const key of Object.keys(navigations)) navigations[key] = 0;
    note("== server now serves Sigil + the worker under test; reloading /client once ==");
    await client.reload({ timeout: 8000 });

    // Poll until /client shows Sigil and has no registration, log every state change.
    let last = "";
    const waitRecovered = async (limit) => {
      const start = Date.now();
      while (Date.now() - start < limit) {
        const s = await snap(client);
        const line = JSON.stringify(s);
        if (line !== last) {
          note(`client ${line}`);
          last = line;
        }
        if (s.h === "SIGIL" && s.regs === 0) return Date.now() - triggeredAt;
        await sleep(POLL_MS);
      }
      return null;
    };
    result.recoveredMs = await waitRecovered(RECOVERY_LIMIT_MS);
    if (result.recoveredMs === null) {
      note(`not recovered after ${RECOVERY_LIMIT_MS} ms; waiting up to ${NUDGE_AFTER_MS} ms, then one more reload`);
      result.recoveredMs = await waitRecovered(NUDGE_AFTER_MS - RECOVERY_LIMIT_MS);
      if (result.recoveredMs === null) {
        const nudgedAt = Date.now();
        await client.reload({ timeout: 8000 }).catch((e) => note(`nudge reload: ${String(e.message).split("\n")[0]}`));
        const after = await waitRecovered(8000);
        result.nudge = { recoveredAfterMs: after === null ? null : Date.now() - nudgedAt };
      }
    }

    // Stability: nothing may keep reloading or re-fetching the worker once the page is right.
    const navsAtRecovery = JSON.stringify(navigations);
    const hitsAtRecovery = swHits.length;
    await sleep(STABLE_FOR_MS);
    const finalClient = await snap(client);
    const finalMic = mic ? await snap(mic) : null;
    const finalCaptions = captions ? await snap(captions) : null;
    result.final = { client: finalClient, mic: finalMic, captions: finalCaptions };
    result.navigations = { ...navigations };
    result.swHitsAfterTrigger = [...swHits];

    const p = result.problems;
    if (result.recoveredMs === null || result.recoveredMs > RECOVERY_LIMIT_MS) p.push(`/client did not show Sigil without a registration within ${RECOVERY_LIMIT_MS} ms (${result.recoveredMs ?? "never"})`);
    if (finalClient.caches !== 0) p.push(`caches left: ${finalClient.caches}`);
    if (finalClient.h !== "SIGIL" || finalClient.regs !== 0) p.push(`client ends as ${JSON.stringify(finalClient)}`);
    if (mic && finalMic?.h !== "MIC") p.push(`mic.html ends as ${finalMic?.h}`);
    if (captions && navigations.captions !== 0) p.push(`the never-controlled window was navigated ${navigations.captions} time(s)`);
    if (JSON.stringify(navigations) !== navsAtRecovery) p.push(`navigations kept going after recovery: ${navsAtRecovery} -> ${JSON.stringify(navigations)}`);
    if (swHits.length !== hitsAtRecovery) p.push(`/sw.js was fetched again after recovery (${swHits.length - hitsAtRecovery}x)`);
    result.ok = p.length === 0;
  } catch (error) {
    result.problems.push(`harness error: ${String(error.message).split("\n")[0]}`);
  } finally {
    await withTimeout(context.close().catch(() => {}), 8000, null);
    server.close();
    server.closeAllConnections?.();
    try {
      fs.rmSync(profile, { recursive: true, force: true, maxRetries: 3 });
    } catch {
      /* a temp profile left behind is harmless */
    }
  }
  result.log = log;
  return result;
}

// Hard stop: a hung browser must never leave this process running.
setTimeout(() => {
  console.error("check-sw-kill-switch: hard timeout");
  process.exit(2);
}, 150_000).unref();

const results = await Promise.all(scenarios.map(runScenario));
for (const r of results) {
  console.log(`\n=== ${r.ok ? "PASS" : "FAIL"}  ${r.label}`);
  for (const line of r.log) console.log("  " + line);
  console.log(`  recovered: ${r.recoveredMs === null ? "no" : `${r.recoveredMs} ms`}${r.nudge ? `  after a nudge reload: ${JSON.stringify(r.nudge)}` : ""}`);
  console.log(`  navigations: ${JSON.stringify(r.navigations)}  /sw.js fetches after the trigger (ms): ${JSON.stringify(r.swHitsAfterTrigger)}`);
  console.log(`  final: ${JSON.stringify(r.final)}`);
  for (const problem of r.problems) console.log(`  problem: ${problem}`);
}
const jsonOut = argValues("--json")[0];
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(results, null, 2));
const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} scenarios passed`);
process.exit(failed ? 1 : 0);
