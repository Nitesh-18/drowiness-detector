r"""
utils/ear.py — Eye Aspect Ratio (EAR) computation
==================================================
Formula from:
  Soukupová & Čech (2016) "Real-Time Eye Blink Detection using Facial Landmarks"
  https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

Eye landmark layout (6 points):

        p2 ─── p3
       /           \
      p1             p4
       \           /
        p6 ─── p5

EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)

A high EAR → eye open.   A low EAR → eye closed / blinking.
"""

import numpy as np
from scipy.spatial.distance import euclidean


def compute_ear(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio for a single eye.

    Parameters
    ----------
    eye : np.ndarray of shape (6, 2)
        Six (x, y) coordinates ordered as:
        [p1, p2, p3, p4, p5, p6]
        i.e.  left-corner, top-left, top-right,
              right-corner, bottom-right, bottom-left

    Returns
    -------
    float
        EAR value in range ~[0, 0.5].
        Typical open-eye value ≈ 0.25-0.35
        Typical closed-eye value ≈ 0.10-0.20
    """
    # Vertical distances (top-to-bottom)
    v1 = euclidean(eye[1], eye[5])   # p2 ↔ p6
    v2 = euclidean(eye[2], eye[4])   # p3 ↔ p5

    # Horizontal distance (left-to-right corner)
    h  = euclidean(eye[0], eye[3])   # p1 ↔ p4

    # Guard against division by zero (shouldn't happen, but just in case)
    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)
