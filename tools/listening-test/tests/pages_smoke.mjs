// SPDX-License-Identifier: AGPL-3.0-or-later
// Headless smoke of the two listening-test pages from file:// (no server, no network): answer every
// trial like a listener would and save the exported answers. Driven by
// scripts/speech-bench/tests/test_listening.py (PageTests); runnable by hand:
//
//   node tools/listening-test/tests/pages_smoke.mjs <mt test/index.html> <tts test/index.html> <out dir>
//
// Exit codes: 0 ok, 1 a check failed, 3 Chromium is not installed (the Python test skips).
import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [mtPage, ttsPage, outDir] = process.argv.slice(2);
if (!mtPage || !ttsPage || !outDir) {
  console.error("usage: pages_smoke.mjs <mt index.html> <tts index.html> <out dir>");
  process.exit(2);
}

let browser;
try {
  browser = await chromium.launch({ headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
} catch (e) {
  console.error("cannot launch Chromium: " + e.message);
  process.exit(3);
}

const total = async (page) => {
  const text = await page.getByTestId("progress").textContent();
  const m = /of (\d+)/.exec(text || "");
  if (!m) throw new Error("no progress text: " + text);
  return Number(m[1]);
};

async function saveDownload(page, file) {
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("export").click()]);
  await download.saveAs(path.join(outDir, file));
}

try {
  const context = await browser.newContext({ acceptDownloads: true });
  const errors = [];

  // --- MT A/B ---------------------------------------------------------------------------------
  const mt = await context.newPage();
  mt.on("pageerror", (e) => errors.push("mt: " + e.message));
  await mt.goto(pathToFileURL(path.resolve(mtPage)).href);
  await mt.getByTestId("trial").waitFor({ state: "visible" });
  await mt.getByTestId("rater").fill("smoke");
  const n = await total(mt);
  for (let i = 0; i < n; i++) {
    const a = await mt.getByTestId("text-A").textContent();
    const b = await mt.getByTestId("text-B").textContent();
    if (!a || !b || a === b) throw new Error(`trial ${i}: empty or identical sides`);
    await mt.getByTestId(i % 3 === 2 ? "choose-tie" : "choose-A").click();
    await mt.waitForTimeout(260);
  }
  const done = await mt.getByTestId("progress").textContent();
  if (!done.startsWith(`${n} of ${n}`)) throw new Error("MT progress after answering: " + done);
  await saveDownload(mt, "mt.json");

  // --- TTS MUSHRA-lite --------------------------------------------------------------------------
  const tts = await context.newPage();
  tts.on("pageerror", (e) => errors.push("tts: " + e.message));
  await tts.goto(pathToFileURL(path.resolve(ttsPage)).href);
  await tts.getByTestId("trial").waitFor({ state: "visible" });
  await tts.getByTestId("rater").fill("smoke");
  const m = await total(tts);
  for (let i = 0; i < m; i++) {
    for (const sid of ["s1", "s2"]) {
      const slider = tts.getByTestId("rate-" + sid);
      if (!(await slider.isDisabled())) throw new Error("a slider is enabled before its sample was played");
      await tts.getByTestId("play-" + sid).click();
      await slider.waitFor({ state: "visible" });
      await tts.waitForFunction((id) => !document.querySelector(`[data-testid="rate-${id}"]`).disabled, sid, { timeout: 10000 });
      await tts.getByTestId("rate-" + sid).fill(sid === "s1" ? "70" : "40");
    }
    if (i < m - 1) await tts.getByTestId("next").click();
  }
  const ttsDone = await tts.getByTestId("progress").textContent();
  if (!ttsDone.startsWith(`${m} of ${m}`)) throw new Error("TTS progress after rating: " + ttsDone);
  await saveDownload(tts, "tts.json");

  if (errors.length) throw new Error("page errors: " + errors.join("; "));
  console.log(`ok: ${n} MT trials, ${m} TTS trials`);
  await browser.close();
  process.exit(0);
} catch (e) {
  console.error(e.stack || String(e));
  await browser.close();
  process.exit(1);
}
