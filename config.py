"""
⚙️  config.py — Centralised configurable parameters
====================================================
Tweak these values to match your environment / preferences.
"""

import os


class Config:
    # ── Camera ──────────────────────────────────────────────────────────────
    CAMERA_INDEX  : int = 0      # 0 = default webcam; try 1/2 for external cams
    FRAME_WIDTH   : int = 640    # Resolution width  (lower = faster)
    FRAME_HEIGHT  : int = 480    # Resolution height

    # ── Eye Aspect Ratio threshold ───────────────────────────────────────────
    # EAR < threshold  →  eye considered CLOSED
    # Typical open-eye EAR ≈ 0.25-0.35; closed ≈ 0.15-0.20
    # Increase if you get false positives; decrease if closures aren't detected
    EAR_THRESHOLD : float = 0.22

    # ── Alarm timing ────────────────────────────────────────────────────────
    # How many consecutive seconds eyes must be closed before alarm fires
    ALARM_THRESHOLD_SEC : float = 2.0

    # ── Sound ───────────────────────────────────────────────────────────────
    # Path to .wav / .mp3 alarm file.  If None, a beep tone is generated.
    ALARM_SOUND_PATH: str | None = os.path.join(
        os.path.dirname(__file__), "sounds", "alarm.wav"
    )
