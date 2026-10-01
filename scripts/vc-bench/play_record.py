#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Play a clip into one audio device while recording another: a live recording of any voice
changer (w-okada VCClient, the Deiteris fork, RVC WebUI realtime, Applio realtime, ...) on the
same input Sigil converts, for Sigil's ``compare`` mode.

Route the app's input to a virtual cable (e.g. VB-CABLE "CABLE Input" as this script's output,
"CABLE Output" as the app's microphone) and its output to a second cable, which this script
records. The recording starts before playback and ends ``--tail`` seconds after it, so the app's
latency is inside the file; Sigil aligns candidates to the input on their energy envelopes.

    python scripts/vc-bench/play_record.py --list
    python scripts/vc-bench/play_record.py --clip speech.wav --play "CABLE Input" --record "CABLE-B Output" --out wokada.wav

Needs ``sounddevice`` and ``soundfile`` (``pip install sounddevice soundfile``) in any Python.
The clip plays at full level on the named device only: never point --play at your speakers while a
voice chat uses the cable.
"""

import argparse
import sys
import threading


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the audio devices and exit")
    ap.add_argument("--clip")
    ap.add_argument("--play", help="output device (name or index)")
    ap.add_argument("--record", help="input device (name or index)")
    ap.add_argument("--out")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--lead", type=float, default=1.0, help="recording before playback starts (s)")
    ap.add_argument("--tail", type=float, default=2.0, help="recording after playback ends (s)")
    args = ap.parse_args()
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    if args.list:
        print(sd.query_devices())
        return
    for k in ("clip", "play", "record", "out"):
        if not getattr(args, k):
            sys.exit(f"--{k} is required (see --help)")
    dev = lambda d: int(d) if d.isdigit() else d  # noqa: E731
    clip, sr = sf.read(args.clip, dtype="float32", always_2d=True)
    clip = clip.mean(axis=1)
    if sr != args.rate:
        import librosa

        clip = librosa.resample(clip, orig_sr=sr, target_sr=args.rate)
    total = int((args.lead + len(clip) / args.rate + args.tail) * args.rate)
    recorded = np.zeros(total, dtype="float32")
    pos = {"rec": 0, "play": 0}
    done = threading.Event()

    def rec_cb(indata, frames, _time, _status):
        n = min(frames, total - pos["rec"])
        recorded[pos["rec"] : pos["rec"] + n] = indata[:n, 0]
        pos["rec"] += n
        if pos["rec"] >= total:
            done.set()

    lead = int(args.lead * args.rate)

    def play_cb(outdata, frames, _time, _status):
        outdata.fill(0)
        start = pos["rec"] - lead  # playback follows the recording clock
        if start >= 0:
            i = pos["play"]
            n = min(frames, max(0, len(clip) - i))
            outdata[:n, 0] = clip[i : i + n]
            pos["play"] += frames

    with sd.InputStream(device=dev(args.record), channels=1, samplerate=args.rate, callback=rec_cb):
        with sd.OutputStream(device=dev(args.play), channels=1, samplerate=args.rate, callback=play_cb):
            done.wait(timeout=total / args.rate + 10)
    sf.write(args.out, recorded[: pos["rec"]], args.rate, subtype="FLOAT")
    print(f"recorded {pos['rec'] / args.rate:.1f} s from {args.record} -> {args.out}")


if __name__ == "__main__":
    main()
