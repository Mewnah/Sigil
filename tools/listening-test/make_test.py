#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a blind A/B listening test (plan T6: preference >= 60 %, 5+ listeners x 20 pairs).

Input: a manifest of trials, each two renditions of the same clip by two systems (Sigil and a
competitor, or two Sigil settings):

    {
      "title": "Voice changer A/B",
      "question": "Which one sounds more natural and closer to a real voice?",
      "trials": [
        {"clip": "speech-01", "a": "out/sigil/speech-01.wav", "system_a": "sigil",
                              "b": "out/wokada/speech-01.flac", "system_b": "wokada"},
        ...
      ]
    }

Output folder (give ``test/`` to the listeners; keep ``key.json`` yourself):

    test/index.html      the page: every trial in a shuffled order, the two renditions in a random
                         A/B order, "A" / "B" / "no preference", a comment box; answers stay in the
                         browser (localStorage) and "Download my answers" saves a JSON file
    test/audio/<id>.wav  the renditions under random names, every one converted to the same WAV
                         format (rate, channels, 16-bit PCM, no metadata): neither the name nor the
                         format nor the container tells which system made it
    key.json             which system each trial's A and B is, the shuffle seed and the format
                         (never give it to listeners)

Blinding: the test id and the file names come from the OS's secure random source, independent of
``--seed``, so nothing on the page leads back to the shuffle; the seed (given or drawn) is kept in
``key.json`` only, to rebuild the same order.

Conversion: every rendition is decoded (WAV with the standard library; FLAC / MP3 / OGG and others
with ``soundfile`` when installed, else ``ffmpeg`` on the PATH), mixed to ``--channels`` (default 1:
voice-changer output is mono content), resampled to ``--rate`` (default: the highest rate among the
renditions, so none loses bandwidth; needs ``scipy``, else ``ffmpeg``) and written as 16-bit PCM
with TPDF dither. Levels are not changed: level-match the renditions first (see README). Needs
``numpy``.

Works offline: open ``index.html`` from the folder. Scoring: ``score.py``.

    python tools/listening-test/make_test.py manifest.json --out listening-test-1 [--seed 7] [--rate 48000]
"""

import argparse
import html
import json
import math
import os
import random
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import wave


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg: #fafafa; --fg: #1b1b1f; --card: #ffffff; --line: #d8d8de; --accent: #4b5bdc; }
  @media (prefers-color-scheme: dark) { :root { --bg: #16161a; --fg: #e9e9ee; --card: #202027; --line: #36363f; --accent: #8f9bff; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .trial { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
  .trial h2 { font-size: 1rem; margin: 0 0 8px; }
  .players { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .players div { display: flex; flex-direction: column; gap: 4px; }
  audio { width: 100%; }
  fieldset { border: 0; padding: 0; margin: 12px 0 0; display: flex; gap: 16px; flex-wrap: wrap; }
  textarea { width: 100%; box-sizing: border-box; margin-top: 8px; min-height: 2.4em; background: transparent; color: inherit; border: 1px solid var(--line); border-radius: 6px; }
  .bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line); padding: 12px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  button { background: var(--accent); color: white; border: 0; border-radius: 6px; padding: 8px 14px; font: inherit; cursor: pointer; }
  input[type=text] { font: inherit; padding: 6px; border: 1px solid var(--line); border-radius: 6px; background: transparent; color: inherit; }
  @media (max-width: 520px) { .players { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p>__QUESTION__ Listen to both versions of each clip (use headphones), then pick one. The order is random; nothing tells which system made which.</p>
<p><label>Your name or initials: <input type="text" id="listener" autocomplete="off"></label></p>
<div id="trials"></div>
<div class="bar"><button id="save">Download my answers</button><span id="progress"></span></div>
</main>
<script>
const TEST = __TEST_JSON__;
const KEY = "sigil-ab-" + TEST.test_id;
const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } })();
function store() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch {} }
const root = document.getElementById("trials");
const listener = document.getElementById("listener");
listener.value = saved.listener || "";
listener.addEventListener("input", () => { saved.listener = listener.value; store(); });
saved.answers = saved.answers || {};
function progress() {
  const n = Object.values(saved.answers).filter(a => a.choice).length;
  document.getElementById("progress").textContent = n + " of " + TEST.trials.length + " answered";
}
TEST.trials.forEach((t, i) => {
  const box = document.createElement("section");
  box.className = "trial";
  const ans = saved.answers[t.id] || {};
  box.innerHTML = `<h2>Trial ${i + 1}</h2>
    <div class="players"><div>A<audio controls preload="none" src="audio/${t.a}"></audio></div>
    <div>B<audio controls preload="none" src="audio/${t.b}"></audio></div></div>
    <fieldset><label><input type="radio" name="c${i}" value="A"> A is better</label>
    <label><input type="radio" name="c${i}" value="B"> B is better</label>
    <label><input type="radio" name="c${i}" value="none"> No preference</label></fieldset>
    <textarea placeholder="Comment (optional)"></textarea>`;
  box.querySelectorAll("input[type=radio]").forEach(r => {
    if (r.value === ans.choice) r.checked = true;
    r.addEventListener("change", () => { saved.answers[t.id] = Object.assign(saved.answers[t.id] || {}, { choice: r.value }); store(); progress(); });
  });
  const c = box.querySelector("textarea");
  c.value = ans.comment || "";
  c.addEventListener("input", () => { saved.answers[t.id] = Object.assign(saved.answers[t.id] || {}, { comment: c.value }); store(); });
  root.appendChild(box);
});
progress();
document.getElementById("save").addEventListener("click", () => {
  const out = { schema: "sigil-ab-answers/1", test_id: TEST.test_id, listener: saved.listener || "", answers: saved.answers };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "answers-" + (saved.listener || "anonymous").replace(/[^A-Za-z0-9_-]+/g, "_") + ".json";
  a.click();
});
</script>
</body>
</html>
"""


def _np():
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        sys.exit("make_test.py needs numpy to convert the renditions to one format: pip install numpy")
    return np


def read_wav(path):
    """A RIFF/WAVE file (PCM 8/16/24/32-bit, IEEE float 32/64, WAVE_FORMAT_EXTENSIBLE) as
    ``(float32 array [frames, channels], rate)``."""
    np = _np()
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None or len(fmt) < 16:
        raise ValueError("no fmt / data chunk")
    tag, channels, rate = struct.unpack("<HHI", fmt[:8])
    bits = struct.unpack("<H", fmt[14:16])[0]
    if tag == 0xFFFE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    width = bits // 8
    if channels < 1 or width < 1:
        raise ValueError("bad fmt chunk")
    pcm = pcm[: len(pcm) // (width * channels) * width * channels]
    if tag == 3 and bits in (32, 64):
        x = np.frombuffer(pcm, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits} bit)")
    return x.reshape(-1, channels), rate


def _ffmpeg(src, extra):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "converted.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *extra, "-c:a", "pcm_f32le", out], check=True)
        return read_wav(out)


def decode(path):
    """Any rendition as ``(float32 array [frames, channels], rate)``."""
    np = _np()
    errors = []
    try:
        import soundfile  # noqa: PLC0415
    except ImportError:
        soundfile = None
    if soundfile is not None:
        try:
            x, rate = soundfile.read(path, dtype="float32", always_2d=True)
            return np.ascontiguousarray(x), int(rate)
        except Exception as e:  # noqa: BLE001 - the other decoders may still read it
            errors.append(f"soundfile: {e}")
    try:
        return read_wav(path)
    except (ValueError, struct.error) as e:
        errors.append(f"wav: {e}")
    if shutil.which("ffmpeg"):
        return _ffmpeg(path, [])
    sys.exit(f"cannot decode {path} ({'; '.join(errors)}): install soundfile (pip install soundfile) or ffmpeg")


def resample(x, rate, target, path):
    """``x`` at ``target`` Hz (polyphase with scipy, else ffmpeg's resampler on the source)."""
    if rate == target:
        return x
    np = _np()
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415
    except ImportError:
        resample_poly = None
    if resample_poly is not None:
        g = math.gcd(int(rate), int(target))
        return resample_poly(x, target // g, rate // g, axis=0).astype(np.float32)
    if shutil.which("ffmpeg"):
        y, r = _ffmpeg(path, ["-ar", str(target)])
        if r != target:
            sys.exit(f"ffmpeg did not resample {path} to {target} Hz")
        return y
    sys.exit(f"{path} is at {rate} Hz and the test at {target} Hz: install scipy (pip install scipy) or ffmpeg to resample")


def to_channels(x, channels):
    """``x`` with ``channels`` channels (a mono downmix, repeated for stereo)."""
    np = _np()
    if x.shape[1] == channels:
        return x
    mono = x.mean(axis=1, keepdims=True)
    return np.repeat(mono, channels, axis=1).astype(np.float32)


def write_pcm16(path, x, rate):
    """16-bit PCM WAV with TPDF dither and no metadata chunks; returns the clipped sample count."""
    np = _np()
    rng = np.random.default_rng(secrets.randbits(64))
    y = x.astype(np.float64) * 32768.0 + (rng.random(x.shape) - rng.random(x.shape))
    clipped = int(np.count_nonzero((y > 32767.0) | (y < -32768.0)))
    q = np.clip(np.round(y), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(q.tobytes())
    return clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed (default: drawn at random; kept in key.json only)")
    ap.add_argument("--rate", type=int, default=None, help="sample rate of every rendition (default: the highest among them)")
    ap.add_argument("--channels", type=int, default=1, choices=(1, 2), help="channels of every rendition (default 1)")
    args = ap.parse_args()
    with open(args.manifest, encoding="utf-8") as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(args.manifest))
    trials = m.get("trials") or []
    if not trials:
        sys.exit("the manifest has no trials")
    # The shuffle is reproducible from the seed; nothing the listeners see derives from it.
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    rng = random.Random(seed)
    test_id = secrets.token_hex(8)
    test_dir = os.path.join(args.out, "test")
    audio_dir = os.path.join(test_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    def source(path):
        src = path if os.path.isabs(path) else os.path.join(base, path)
        if not os.path.isfile(src):
            sys.exit(f"missing audio: {src}")
        return src

    decoded = {}
    for t in trials:
        for side in ("a", "b"):
            src = source(t[side])
            if src not in decoded:
                decoded[src] = decode(src)
    rate = args.rate or max(r for _, r in decoded.values())
    used = set()

    def blind_copy(path):
        src = source(path)
        x, r = decoded[src]
        y = to_channels(resample(x, r, rate, src), args.channels)
        while True:
            name = secrets.token_hex(8) + ".wav"
            if name not in used:
                used.add(name)
                break
        clipped = write_pcm16(os.path.join(audio_dir, name), y, rate)
        if clipped:
            print(f"warning: {path}: {clipped} samples clipped at full scale (lower its level)", file=sys.stderr)
        return name

    page_trials, key = [], []
    order = list(range(len(trials)))
    rng.shuffle(order)
    for n, i in enumerate(order):
        t = trials[i]
        flip = rng.random() < 0.5
        first, second = (("b", "system_b"), ("a", "system_a")) if flip else (("a", "system_a"), ("b", "system_b"))
        trial_id = "t%03d" % n
        fa, fb = blind_copy(t[first[0]]), blind_copy(t[second[0]])
        page_trials.append({"id": trial_id, "a": fa, "b": fb})
        key.append({"id": trial_id, "clip": t.get("clip", ""), "A": t[first[1]], "B": t[second[1]], "files": {"A": t[first[0]], "B": t[second[0]]}})
    title = m.get("title", "Voice changer listening test")
    question = m.get("question", "Which version sounds better?")
    test = {"test_id": test_id, "trials": page_trials}
    page = (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace("__QUESTION__", html.escape(question))
        .replace("__TEST_JSON__", json.dumps(test))
    )
    with open(os.path.join(test_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    key_doc = {
        "schema": "sigil-ab-key/1",
        "test_id": test_id,
        "title": title,
        "seed": seed,
        "format": {"rate": rate, "channels": args.channels, "subtype": "PCM_16"},
        "trials": key,
    }
    with open(os.path.join(args.out, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key_doc, f, indent=2)
    print(
        f"{len(page_trials)} trials -> {test_dir} (give this folder to listeners; {rate} Hz, {args.channels} ch, 16-bit); "
        f"key -> {os.path.join(args.out, 'key.json')}"
    )


if __name__ == "__main__":
    main()
