// P7-A footprint driver: opens the OBS captions page in headless Chromium (the browser-source stand-in) and feeds the running Sigil
// with caption traffic over its pubsub WebSocket (text.stt interims at ~6 Hz while "speaking", a final after each sentence).
// Usage: node driver.mjs <port> <seconds> [speakingShare=0.4] [page=1]
import { createRequire } from 'node:module';
const require = createRequire(new URL('../../../../package.json', import.meta.url));
const { chromium } = require('@playwright/test');
const [, , portArg, secondsArg, shareArg = '0.4', pageArg = '1'] = process.argv;
const port = Number(portArg), seconds = Number(secondsArg), share = Number(shareArg), withPage = pageArg !== '0';
const sentences = [
  'Hey everyone thanks for joining the stream tonight we are going to explore a few new worlds',
  'Can you hear me okay or is my microphone too quiet right now',
  'I will be right back I need to grab some water and check the chat',
  'That world has the best lighting I have seen in a long time you should all visit it',
  'Does anyone want to join my instance we have plenty of room for more people',
  'Wait I think the avatar loaded wrong let me reset it and try again',
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let browser = null, page = null, shown = '';
if (withPage) {
  browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1920, height: 400 } });
  await page.goto(`http://127.0.0.1:${port}/obs-captions.html?port=${port}`);
  await sleep(2500);
}
const ws = new WebSocket(`ws://127.0.0.1:${port}/pubsub?id=p7a-injector-${Date.now()}`);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('ws error')); setTimeout(() => rej(new Error('ws timeout')), 8000); });
// Everything the host broadcasts comes back on this socket: counting the topics shows that captions reach the OBS feed and,
// in the translation scenarios, that translations are produced (`text.translation`).
const topics = {};
ws.onmessage = (e) => { try { const m = JSON.parse(String(e.data)); if (m && typeof m.topic === 'string') topics[m.topic] = (topics[m.topic] ?? 0) + 1; } catch { /* not JSON */ } };
const send = (type, value) => ws.send(JSON.stringify({ topic: 'text.stt', data: { type, value } }));
// The OBS page is the proof that the caption path works: count how often its caption line changed to something new.
const seen = new Set(); let pageChanges = 0, last = '';
const watch = page ? setInterval(async () => {
  const t = await page.locator('#caption').innerText().catch(() => '');
  if (t && t !== last) { pageChanges++; last = t; seen.add(t); }
}, 250) : null;
const t0 = Date.now(); let sent = 0, finals = 0, i = 0;
while ((Date.now() - t0) / 1000 < seconds) {
  const words = sentences[i++ % sentences.length].split(' ');
  const speakStart = Date.now();
  for (let w = 1; w <= words.length && (Date.now() - t0) / 1000 < seconds; w++) {
    send(1, words.slice(0, w).join(' ')); sent++;
    await sleep(170); // ~6 partials / s, about 2.5 words / s of speech
  }
  send(0, words.join(' ') + '.'); sent++; finals++;
  const spoke = Date.now() - speakStart;
  const gap = Math.max(500, spoke * (1 - share) / share);
  await sleep(Math.min(gap, Math.max(0, seconds * 1000 - (Date.now() - t0))));
}
if (watch) clearInterval(watch);
if (page) shown = await page.locator('#caption').innerText().catch(() => '');
console.log(JSON.stringify({ sent, finals, seconds, pageChanges, distinctCaptions: seen.size, topics, pageShowed: shown.slice(0, 80) }));
ws.close();
if (browser) await browser.close();
