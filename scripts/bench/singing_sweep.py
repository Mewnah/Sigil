"""Write the sung E2 -> D#6 sweep the DDSP-SVC singing benchmark converts (P5-A; T10).

The same signal as ``ddsp_svc/tests.rs`` ``singing_sweep``: E2 (82.41 Hz) to D#6 (1244.5 Hz) over
8 s with +-25 cent vibrato at 5.5 Hz, eight harmonics through a vowel-like envelope, a 0.5 s lead,
0.5 s held at each end and a 30 ms fade. At +12 semitones the voice sings E3 -> D#7.

    python scripts/bench/singing_sweep.py <out.wav>

Score a converted sweep against the intended notes with ``score_singing.py``. Needs numpy and
soundfile.
"""

import sys

import numpy as np
import soundfile as sf

RATE = 48_000
LEAD_S = 0.5
SECONDS = 8.0
LO_HZ = 82.41
HI_HZ = 1244.5


def sweep(rate: int = RATE, seconds: float = SECONDS, lead: float = LEAD_S, lo: float = LO_HZ, hi: float = HI_HZ) -> np.ndarray:
    sung = 0.5 + seconds + 0.5
    n = int((lead + sung + 0.03 + 0.3) * rate)
    t = np.arange(n) / rate - lead
    base = np.where(t < 0.5, lo, lo * (hi / lo) ** np.clip((t - 0.5) / seconds, 0, 1))
    f = base * 2 ** (0.25 * np.sin(2 * np.pi * 5.5 * t) / 12)
    f = np.where((t < 0) | (t >= sung + 0.03), 0, f)
    phase = np.cumsum(f / rate)
    s = np.zeros(n)
    for h in range(1, 9):
        fh = f * h
        formant = np.exp(-(((fh - 700) / 900) ** 2)) + 0.5 * np.exp(-(((fh - 2500) / 1200) ** 2)) + 0.05
        s += np.where((fh <= 7000) & (t >= 0) & (t < sung + 0.03), formant / h * np.sin(2 * np.pi * h * phase), 0)
    fade = np.minimum(np.clip(t / 0.02, 0, 1), np.clip((sung + 0.03 - t) / 0.03, 0, 1))
    return (0.25 * fade * s).astype(np.float32)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sf.write(sys.argv[1], sweep(), RATE, subtype="FLOAT")
    print("wrote", sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
