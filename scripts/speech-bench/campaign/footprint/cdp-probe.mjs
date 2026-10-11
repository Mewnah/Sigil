// Connects to the Sigil webview over CDP, prints what the host services say, optionally starts STT, takes a screenshot.
import { createRequire } from 'node:module';
const require = createRequire(new URL('../../../../package.json', import.meta.url));
const { chromium } = require('@playwright/test');
const [, , portArg = '9333', outPng = '', startStt = '0'] = process.argv;
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${portArg}`);
const ctx = browser.contexts()[0];
const pages = ctx.pages();
console.log('pages', pages.map((p) => p.url()));
const page = pages.find((p) => !p.url().startsWith('devtools')) ?? pages[0];
page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) console.log('console', m.type(), m.text().slice(0, 300)); });
await page.waitForTimeout(1500);
const info = await page.evaluate(() => {
  const s = window.ApiServer;
  const st = s?.state?.services?.stt;
  const tr = s?.state?.services?.translation;
  return {
    hasApi: !!s, title: document.title, bodyText: document.body.innerText.slice(0, 400),
    stt: st ? { backend: st.data?.backend, autoStart: st.data?.autoStart, local: st.data?.local } : null,
    sttStatus: s?.stt?.serviceState ?? null,
    transl: tr ? { backend: tr.data?.backend, autoStart: tr.data?.autoStart } : null,
  };
});
console.log(JSON.stringify(info, null, 1));
if (startStt === '1') {
  const r = await page.evaluate(async () => { try { await window.ApiServer.stt.start({ automatic: true }); return 'started'; } catch (e) { return 'error ' + e; } });
  console.log('start', r);
  await page.waitForTimeout(8000);
  console.log('status after start', JSON.stringify(await page.evaluate(() => window.ApiServer.stt.serviceState)));
}
if (outPng) await page.screenshot({ path: outPng });
await browser.close();
