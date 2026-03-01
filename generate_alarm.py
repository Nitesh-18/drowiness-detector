"""
generate_alarm.py
=================
Run this once to create sounds/alarm.wav if you don't have one.
  python generate_alarm.py
"""

import struct
import math
import wave
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "sounds", "alarm.wav")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

SAMPLE_RATE  = 44100
DURATION_SEC = 0.8          # length of one alarm burst
FREQS        = [880, 1100]  # alternating tones (Hz)


def generate_tone(freq: float, duration: float, sample_rate: int = 44100) -> list:
    """Generate a sine-wave tone as 16-bit PCM samples."""
    n_samples = int(sample_rate * duration)
    samples   = []
    fade      = max(1, int(sample_rate * 0.01))   # 10 ms fade

    for i in range(n_samples):
        val = math.sin(2 * math.pi * freq * i / sample_rate)
        # Apply simple linear fade in/out
        if i < fade:
            val *= i / fade
        elif i >= n_samples - fade:
            val *= (n_samples - i) / fade
        samples.append(int(val * 32000))   # scale to 16-bit

    return samples


def write_wav(filepath: str, samples: list, sample_rate: int = 44100):
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        for s in samples:
            wf.writeframes(struct.pack("<h", max(-32768, min(32767, s))))


if __name__ == "__main__":
    all_samples = []
    for freq in FREQS:
        all_samples.extend(generate_tone(freq, DURATION_SEC / len(FREQS)))

    write_wav(OUTPUT, all_samples)
    print(f"✅  Generated alarm tone → {OUTPUT}")
