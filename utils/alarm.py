"""
utils/alarm.py — Cross-platform alarm sound implementation
===========================================================
Priority order for playback:
  1. pygame  (plays WAV/MP3 in a background thread — best quality)
  2. playsound (simple fallback)
  3. subprocess  (Linux: aplay / paplay)
  4. Generated beep via numpy + sounddevice (no external file needed)
  5. Terminal bell (last resort — silent on many systems)

The alarm loops until .stop() is called.
"""

import os
import sys
import threading
import logging

logger = logging.getLogger(__name__)


class Alarm:
    """Manages a looping alarm sound that plays until explicitly stopped."""

    def __init__(self, sound_path: str | None = None):
        self.sound_path = sound_path
        self._playing   = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Determine which backend is available at construction time
        self._backend = self._detect_backend()
        logger.info(f"🔊 Alarm backend: {self._backend}")

    # ── Backend detection ──────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """Return name of the first usable audio backend."""
        # 1. pygame
        try:
            import pygame
            pygame.mixer.init()
            return "pygame"
        except Exception:
            pass

        # 2. playsound
        try:
            import playsound  # noqa: F401
            return "playsound"
        except Exception:
            pass

        # 3. Linux subprocess (aplay / paplay)
        if sys.platform.startswith("linux"):
            import shutil
            for cmd in ("paplay", "aplay", "mpg123", "ffplay"):
                if shutil.which(cmd):
                    return f"subprocess:{cmd}"

        # 4. sounddevice + numpy (generates a beep, no file needed)
        try:
            import sounddevice  # noqa: F401
            import numpy        # noqa: F401
            return "sounddevice"
        except Exception:
            pass

        # 5. Terminal bell
        return "bell"

    # ── Public API ─────────────────────────────────────────────────────────

    def play(self):
        """Start the alarm (non-blocking). Safe to call multiple times."""
        if self._playing:
            return
        self._playing = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.warning("🚨 ALARM STARTED")

    def stop(self):
        """Stop the alarm."""
        if not self._playing:
            return
        self._stop_event.set()
        self._playing = False
        # Stop pygame mixer immediately if active
        if self._backend == "pygame":
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass
        logger.info("🔕 Alarm stopped")

    # ── Internal loop ──────────────────────────────────────────────────────

    def _loop(self):
        """Loop alarm until stop_event is set."""
        while not self._stop_event.is_set():
            self._play_once()

    def _play_once(self):
        """Play one instance of the alarm sound (or a beep)."""
        # Check if we should stop before playing
        if self._stop_event.is_set():
            return

        # ── pygame ────────────────────────────────────────────────────────
        if self._backend == "pygame":
            try:
                import pygame
                if self.sound_path and os.path.exists(self.sound_path):
                    pygame.mixer.music.load(self.sound_path)
                    pygame.mixer.music.play()
                    # Wait for it to finish (or stop signal)
                    while pygame.mixer.music.get_busy():
                        if self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            return
                        self._stop_event.wait(0.05)
                else:
                    self._beep_sounddevice()
                return
            except Exception as e:
                logger.debug(f"pygame error: {e}")

        # ── playsound ─────────────────────────────────────────────────────
        if self._backend == "playsound":
            try:
                from playsound import playsound
                if self.sound_path and os.path.exists(self.sound_path):
                    playsound(self.sound_path, block=True)
                    return
            except Exception as e:
                logger.debug(f"playsound error: {e}")

        # ── subprocess ────────────────────────────────────────────────────
        if self._backend.startswith("subprocess:"):
            cmd = self._backend.split(":")[1]
            if self.sound_path and os.path.exists(self.sound_path):
                try:
                    import subprocess
                    proc = subprocess.Popen(
                        [cmd, self.sound_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    while proc.poll() is None:
                        if self._stop_event.is_set():
                            proc.terminate()
                            return
                        self._stop_event.wait(0.05)
                    return
                except Exception as e:
                    logger.debug(f"subprocess error: {e}")

        # ── sounddevice beep ──────────────────────────────────────────────
        if self._backend == "sounddevice":
            self._beep_sounddevice()
            return

        # ── Terminal bell (last resort) ───────────────────────────────────
        print("\a", end="", flush=True)
        self._stop_event.wait(0.5)

    def _beep_sounddevice(self, frequency: float = 880.0, duration: float = 0.6):
        """Generate a simple sine-wave beep via sounddevice."""
        try:
            import sounddevice as sd
            import numpy as np
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Sine wave with fade-in / fade-out to avoid clicks
            wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            fade = int(sample_rate * 0.01)
            wave[:fade]  *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
            sd.play(wave, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.debug(f"sounddevice beep error: {e}")
            print("\a", end="", flush=True)
