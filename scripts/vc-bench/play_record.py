#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Play a clip into one audio device while recording another: a live recording of any voice
changer (w-okada VCClient, the Deiteris fork, RVC WebUI realtime, Applio realtime, ...) for Sigil's
``compare`` mode, for PCs without Sigil.

Prefer Sigil's own meter where Sigil runs: a compare corpus ``captures`` entry plays every clip
through the system under test with the play-and-record mode of ``bench/e2e_loopback.rs`` (both
streams on one time base from the drivers' timestamps, calibrated, with glitch and dropout
detection), and Sigil's live voice changer is captured the same way. This script opens two
independent PortAudio streams: they do NOT share a clock (playback follows the recording's sample
count, so their relative start is only as good as the two drivers' start-up), and it measures no
latency. Sigil aligns the recording to the input on energy envelopes and reports the offset, labelled
``uncalibrated`` (the two streams' start-up skew and buffers are inside it): compare mode never sets
it against a calibrated ``e2e_loopback`` latency. To use the latency anyway, calibrate the bare loop
yourself and state ``latency_ms`` with ``latency_kind: "e2e_loopback"`` in the corpus point.

What it does record: every callback's status flags (input overflow, output underflow, priming
output), with when they happened, into a JSON sidecar next to the WAV (``<wav>.json``,
``sigil-vc-bench-sidecar/1``, ``capture.status``). An overflow or underflow means the recording or
the playback has a gap: re-record that clip.

Route the app's input to a virtual cable (e.g. VB-CABLE "CABLE Input" as this script's output,
"CABLE Output" as the app's microphone) and its output to a second cable, which this script
records. The recording starts before playback and ends ``--tail`` seconds after it, so the app's
latency is inside the file.

    python scripts/vc-bench/play_record.py --list
    python scripts/vc-bench/play_record.py --clip speech.wav --play "CABLE Input" --record "CABLE-B Output" --out wokada.wav --system w_okada
    python scripts/vc-bench/play_record.py --corpus <corpus>/corpus.json --play "CABLE Input" --record "CABLE-B Output" --out-dir <corpus>/out/wokada-live --system w_okada

Needs ``sounddevice`` and ``soundfile`` (``pip install sounddevice soundfile``) in any Python.
The clip plays at full level on the named device only: never point --play at your speakers while a
voice chat uses the cable.
"""

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcbench_common as vb  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the audio devices and exit")
    ap.add_argument("--clip")
    ap.add_argument("--corpus", help="a sigil-compare-corpus/1 manifest: every clip, one after the other")
    ap.add_argument("--play", help="output device (name or index)")
    ap.add_argument("--record", help="input device (name or index)")
    ap.add_argument("--out", help="output WAV (--clip)")
    ap.add_argument("--out-dir", help="output folder (--corpus): <out-dir>/<clip>.wav")
    ap.add_argument("--system", default="unknown", help="the app under test (w_okada, deiteris, rvc_webui, applio, sigil, ...)")
    ap.add_argument("--app-version", default="")
    ap.add_argument("--voice", default="", help="the voice file the app loaded (its sha256 goes into the sidecar)")
    ap.add_argument("--block-ms", type=float, default=0.0, help="the app's block (chunk) setting, for the seam check")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--lead", type=float, default=1.0, help="recording before playback starts (s)")
    ap.add_argument("--tail", type=float, default=2.0, help="recording after playback ends (s)")
    return ap.parse_args(argv)


class StatusLog:
    """Callback status flags with their stream time (thread-safe enough: one writer per stream)."""

    def __init__(self):
        self.events = []

    def note(self, stream: str, pos: int, rate: int, status) -> None:
        if not status:
            return
        names = [n for n in ("input_overflow", "input_underflow", "output_overflow", "output_underflow", "priming_output") if getattr(status, n, False)]
        if names:
            self.events.append({"stream": stream, "at_s": round(pos / rate, 4), "flags": names})

    def summary(self) -> dict:
        count = {}
        for e in self.events:
            for n in e["flags"]:
                count[n] = count.get(n, 0) + 1
        # Priming output at start-up is expected; anything else is a gap in the audio.
        gaps = sum(v for k, v in count.items() if k != "priming_output")
        return {"counts": count, "gaps": gaps, "events": self.events[:200]}


def record_one(clip_path: str, args, out_path: str) -> dict:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    dev = lambda d: int(d) if str(d).isdigit() else d  # noqa: E731
    clip, sr = sf.read(clip_path, dtype="float32", always_2d=True)
    clip = clip.mean(axis=1)
    if sr != args.rate:
        import librosa

        clip = librosa.resample(clip, orig_sr=sr, target_sr=args.rate)
    total = int((args.lead + len(clip) / args.rate + args.tail) * args.rate)
    recorded = np.zeros(total, dtype="float32")
    pos = {"rec": 0, "play": 0}
    done = threading.Event()
    log = StatusLog()
    lead = int(args.lead * args.rate)

    def rec_cb(indata, frames, _time, status):
        log.note("input", pos["rec"], args.rate, status)
        n = min(frames, total - pos["rec"])
        recorded[pos["rec"] : pos["rec"] + n] = indata[:n, 0]
        pos["rec"] += n
        if pos["rec"] >= total:
            done.set()

    def play_cb(outdata, frames, _time, status):
        log.note("output", pos["play"], args.rate, status)
        outdata.fill(0)
        if pos["rec"] - lead >= 0:  # playback follows the recording's sample count
            i = pos["play"]
            n = min(frames, max(0, len(clip) - i))
            outdata[:n, 0] = clip[i : i + n]
            pos["play"] += frames

    with sd.InputStream(device=dev(args.record), channels=1, samplerate=args.rate, callback=rec_cb):
        with sd.OutputStream(device=dev(args.play), channels=1, samplerate=args.rate, callback=play_cb):
            done.wait(timeout=total / args.rate + 10)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, recorded[: pos["rec"]], args.rate, subtype="FLOAT")
    status = log.summary()
    config = {"block_ms": args.block_ms or None, "samplerate": args.rate}
    side = vb.make_sidecar(
        system=args.system,
        app_version=args.app_version,
        input_path=clip_path,
        voice_path=args.voice,
        config=config,
        latency_kind="uncalibrated",
        latency_note="not measured here (two independent streams, no bare-loop calibration); Sigil reads the offset from the recording and labels it uncalibrated: it is never compared with a calibrated e2e_loopback latency (Sigil's corpus captures and burst meter)",
        output_rate=args.rate,
        notes=["live recording by scripts/vc-bench/play_record.py (two PortAudio streams, no shared clock)"],
        extra={
            "capture": {"play": str(args.play), "record": str(args.record), "lead_s": args.lead, "tail_s": args.tail, "status": status},
            "glitches": status["gaps"],
        },
    )
    vb.write_sidecar(out_path, side)
    flag = f", {status['gaps']} status gap(s): re-record" if status["gaps"] else ""
    print(f"recorded {pos['rec'] / args.rate:.1f} s from {args.record} -> {out_path}{flag}")
    return status


def main():
    args = parse_args()
    if args.list:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not (args.play and args.record):
        sys.exit("--play and --record are required (see --help)")
    if args.corpus:
        if not args.out_dir:
            sys.exit("--corpus needs --out-dir")
        for clip_id, path in vb.corpus_clips(args.corpus):
            record_one(path, args, os.path.join(args.out_dir, f"{clip_id}.wav"))
    else:
        if not (args.clip and args.out):
            sys.exit("--clip and --out are required (or --corpus and --out-dir)")
        record_one(args.clip, args, args.out)


if __name__ == "__main__":
    main()
