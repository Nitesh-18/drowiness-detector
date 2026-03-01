"""
👁️ Real-Time Drowsiness / Eye Closure Detection
================================================
Detects eye closure using MediaPipe Face Mesh + Eye Aspect Ratio (EAR).
Triggers an alarm if eyes remain closed beyond a configurable threshold.

Author   : You
Requires : See requirements.txt
Usage    : python main.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from urllib.request import urlretrieve

from utils.ear import compute_ear
from utils.alarm import Alarm
from config import Config

# ──────────────────────────────────────────────
#  MediaPipe setup
# ──────────────────────────────────────────────
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")

# MediaPipe Face Mesh landmark indices for left and right eyes
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]   # 6 points: p1..p6
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]   # 6 points: p1..p6


def ensure_face_landmarker_model(model_path: Path) -> Path:
    """Download the FaceLandmarker model if it does not exist locally."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading FaceLandmarker model to {model_path} ...")
    try:
        urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "❌ Could not download face_landmarker.task. "
            "Please check internet connectivity and try again."
        ) from exc

    if model_path.stat().st_size == 0:
        raise RuntimeError("❌ Downloaded model file is empty.")

    return model_path


def create_face_detector():
    """
    Create a face landmark detector.

    Returns
    -------
    tuple[str, object]
        (backend, detector)
        backend is one of: "solutions", "tasks"
    """
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh
        detector = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return "solutions", detector

    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision

    model_path = ensure_face_landmarker_model(FACE_LANDMARKER_MODEL_PATH)
    options = mp_tasks_vision.FaceLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_tasks_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_tasks_vision.FaceLandmarker.create_from_options(options)
    return "tasks", detector


def get_eye_coords(landmarks, indices, img_w, img_h):
    """
    Extract (x, y) pixel coordinates for a set of landmark indices.
    MediaPipe returns normalised [0,1] coords, so we scale by image size.
    """
    return np.array(
        [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices],
        dtype=np.float64
    )


def draw_eye_contour(frame, coords, color=(0, 255, 0)):
    """Draw a polygon around the given eye coordinates."""
    pts = coords.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def main():
    cfg   = Config()
    alarm = Alarm(cfg.ALARM_SOUND_PATH)
    backend, detector = create_face_detector()

    # ── Camera initialisation ──────────────────
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open camera index {cfg.CAMERA_INDEX}. "
            "Try changing CAMERA_INDEX in config.py"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    # ── State variables ────────────────────────
    eyes_closed_since: float | None = None   # timestamp when closure started
    alarm_triggered   = False

    print(f"✅ Starting drowsiness detector ({backend} backend). Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame grab failed. Retrying…")
                continue

            # Flip so it acts like a mirror (more natural UX)
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            # Convert BGR→RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if backend == "solutions":
                rgb.flags.writeable = False        # slight perf gain
                results = detector.process(rgb)
                rgb.flags.writeable = True
                landmarks_list = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)
                landmarks_list = results.face_landmarks[0] if results.face_landmarks else None

            ear_avg  = 0.0
            eyes_open = True

            if landmarks_list:
                lm = landmarks_list

                # ── Compute EAR for both eyes ──────────────
                left_coords  = get_eye_coords(lm, LEFT_EYE_IDX,  img_w, img_h)
                right_coords = get_eye_coords(lm, RIGHT_EYE_IDX, img_w, img_h)

                left_ear  = compute_ear(left_coords)
                right_ear = compute_ear(right_coords)
                ear_avg   = (left_ear + right_ear) / 2.0

                # ── Draw eye landmarks ─────────────────────
                eye_color = (0, 255, 0) if ear_avg >= cfg.EAR_THRESHOLD else (0, 0, 255)
                draw_eye_contour(frame, left_coords,  eye_color)
                draw_eye_contour(frame, right_coords, eye_color)

                # ── Eye open/closed decision ───────────────
                eyes_open = ear_avg >= cfg.EAR_THRESHOLD

            # ── Closure duration tracking ──────────────────
            now = time.monotonic()

            if not eyes_open:
                if eyes_closed_since is None:
                    eyes_closed_since = now          # start the timer
                closed_duration = now - eyes_closed_since
            else:
                eyes_closed_since = None             # reset timer
                closed_duration   = 0.0
                if alarm_triggered:
                    alarm.stop()
                    alarm_triggered = False

            # ── Trigger alarm after threshold ─────────────
            if closed_duration >= cfg.ALARM_THRESHOLD_SEC and not alarm_triggered:
                alarm.play()
                alarm_triggered = True

            # ──────────────────────────────────────────────
            #  HUD overlay
            # ──────────────────────────────────────────────
            hud_color = (0, 255, 0)   # green = OK

            # EAR readout
            cv2.putText(
                frame, f"EAR: {ear_avg:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # Threshold reminder
            cv2.putText(
                frame, f"Threshold: {cfg.EAR_THRESHOLD:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1
            )

            # Closure duration
            if closed_duration > 0:
                cv2.putText(
                    frame, f"Closed: {closed_duration:.1f}s",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
                hud_color = (0, 165, 255)   # orange

            # Big warning banner
            if alarm_triggered:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, img_h // 3), (img_w, 2 * img_h // 3),
                              (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(
                    frame, "⚠  WAKE UP! ⚠",
                    (img_w // 2 - 160, img_h // 2 + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 3
                )
                hud_color = (0, 0, 255)

            # Status dot (top-right)
            status_label = "OPEN" if eyes_open else "CLOSED"
            cv2.circle(frame, (img_w - 20, 20), 12, hud_color, -1)
            cv2.putText(
                frame, status_label,
                (img_w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2
            )

            # Quit hint
            cv2.putText(
                frame, "Q - quit",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
            )

            cv2.imshow("Drowsiness Detector", frame)

            # Q or ESC → quit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        detector.close()

    # ── Cleanup ────────────────────────────────
    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detector stopped.")


if __name__ == "__main__":
    main()
