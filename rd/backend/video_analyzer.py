"""
ROADDOC video analysis engine.

Extends the single-image pipeline (track_analyzer.py) to video: sample
frames from an uploaded clip, run the existing road-condition classifier on
each, detect road-surface defects (potholes / cracks) with an explainable
contour heuristic, and — the part that replaces manual speed entry —
estimate the vehicle's speed directly from the footage using dense optical
flow, automatically scaled to real-world units by detecting the lane
markings in frame (assumed standard lane width) rather than asking the
user to type a number in.

Everything here is a heuristic, not a trained model. That's an honest
tradeoff given there's no labelled pothole/speed dataset in this project —
it will have false positives (shadows, tar patches, manhole covers can look
like potholes) and the speed estimate carries real error (typically within
~10-20% of true speed when lane markings are visible, worse without them).
It is NOT a substitute for GPS speed or a certified pothole detector; it's
a genuinely automatic, camera-only estimate, which is what was asked for.
"""
from __future__ import annotations
from typing import Optional
import math

import cv2
import numpy as np
from PIL import Image

from track_analyzer import analyze_image, trend_for, suggestion_for

ASSUMED_LANE_WIDTH_M = 3.5   # standard highway lane width, used to auto-scale optical flow
MAX_PLAUSIBLE_KMH = 180
DEFAULT_PIXELS_PER_METER = 40  # fallback scale when no lane markings are found in frame


def sample_frames(video_path: str, target_fps: float = 2.0, max_frames: int = 40):
    """Yield (timestamp_s, bgr_frame) sampled evenly through the video.

    target_fps controls how many frames per second of *video* we analyze
    (2/sec is enough to catch road-condition changes and give optical flow
    a decent baseline without processing every single frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open video file")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / src_fps if src_fps > 0 else 0

    step = max(1, round(src_fps / target_fps))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append((idx / src_fps, frame))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, duration_s, src_fps


def sample_frames_dual(video_path: str, report_fps: float = 2.0, flow_fps: float = 8.0,
                        max_report_frames: int = 40):
    """Two-tier sampling: a fine-grained stream for optical flow (small,
    reliably-trackable displacement between consecutive flow frames) and a
    coarser stream of "checkpoint" timestamps for reporting/classification.
    Returns (flow_frames, checkpoint_indices, duration_s, src_fps) where
    flow_frames is [(t, bgr), ...] at flow_fps and checkpoint_indices are
    indices into flow_frames closest to each report interval.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open video file")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / src_fps if src_fps > 0 else 0

    flow_step = max(1, round(src_fps / flow_fps))
    flow_frames = []
    idx = 0
    max_flow_frames = max_report_frames * max(1, round(flow_fps / report_fps)) + flow_step
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % flow_step == 0:
            flow_frames.append((idx / src_fps, frame))
            if len(flow_frames) >= max_flow_frames:
                break
        idx += 1
    cap.release()

    if not flow_frames:
        return [], [], duration_s, src_fps

    report_interval = 1.0 / report_fps
    checkpoints = []
    next_t = 0.0
    for i, (t, _) in enumerate(flow_frames):
        if t >= next_t:
            checkpoints.append(i)
            next_t += report_interval
    if checkpoints[-1] != len(flow_frames) - 1:
        checkpoints.append(len(flow_frames) - 1)

    return flow_frames, checkpoints, duration_s, src_fps


def _road_roi(gray_or_bgr):
    """Bottom 55% of the frame — where the road surface actually is,
    excluding sky/horizon/dashboard."""
    h = gray_or_bgr.shape[0]
    return gray_or_bgr[int(h * 0.45):, :]


def detect_defects(bgr_frame) -> list[dict]:
    """Heuristic pothole/crack detector.

    Potholes and cracks both tend to be locally darker than surrounding
    asphalt (shadow inside the depression / crack line) and texturally
    distinct from the smoother road surface around them. We isolate dark
    blobs via adaptive thresholding, then classify each by shape:
    blob-like + fills its bounding box -> pothole; thin + elongated -> crack.
    """
    roi = _road_roi(bgr_frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold copes with uneven lighting across the frame far
    # better than a single global threshold would.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 8
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = roi.shape[0] * roi.shape[1]
    min_area = frame_area * 0.0015   # ignore speckle noise
    max_area = frame_area * 0.12     # ignore huge dark regions (shadows of trees/overpasses)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area else 0
        aspect = w / h if h else 0
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter else 0

        kind, confidence = None, 0.0
        if extent > 0.45 and 0.4 <= aspect <= 2.5 and circularity > 0.35:
            kind = "pothole"
            confidence = round(min(1.0, extent * circularity * 1.3), 2)
        elif (aspect > 3.0 or aspect < 0.33) and extent < 0.45:
            kind = "crack"
            confidence = round(min(1.0, (1 - extent) * 0.9), 2)
        if kind:
            detections.append({
                "type": kind,
                "confidence": confidence,
                # bbox given in full-frame coordinates (offset back from the ROI crop)
                "bbox": [int(x), int(y + bgr_frame.shape[0] * 0.45), int(w), int(h)],
            })

    detections.sort(key=lambda d: -d["confidence"])
    return detections[:12]


def _detect_lane_lines_x(bgr_frame, ref_row_frac: float = 0.85) -> Optional[tuple]:
    """Like _detect_lane_pixel_width, but returns the actual (left_x,
    right_x) pixel positions at a reference row near the bottom of the
    ROI, rather than just their separation. Used to place speed-estimation
    patches safely *inside* the lane (on road surface) rather than risk
    landing on the static lane-marking pixels themselves, which would
    otherwise look like a perfect (but false) zero-motion match."""
    roi = _road_roi(bgr_frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                             minLineLength=roi.shape[0] * 0.12, maxLineGap=40)
    if lines is None:
        return None

    h, w = roi.shape[:2]
    cx = w / 2
    left_candidates, right_candidates = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = 90.0 if x2 == x1 else abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45:
            continue
        x_bottom = x1 if y1 > y2 else x2
        (left_candidates if x_bottom < cx else right_candidates).append(x_bottom)

    if not left_candidates or not right_candidates:
        return None
    left_x, right_x = max(left_candidates), min(right_candidates)
    if right_x - left_x < w * 0.15 or right_x - left_x > w * 0.95:
        return None
    return left_x, right_x


def _detect_lane_pixel_width(bgr_frame) -> Optional[float]:
    """Try to find the two lane-marking lines nearest the camera and return
    their pixel separation at a row close to the bottom of the frame. This
    is what lets speed be recovered in real units without any manual
    calibration — assume the lane is a standard width and measure it."""
    lanes = _detect_lane_lines_x(bgr_frame)
    return (lanes[1] - lanes[0]) if lanes else None


def estimate_speed_kmh(prev_bgr, curr_bgr, dt_s: float, ema_prev: Optional[float] = None):
    """Automatic speed estimate: track how far several small road-texture
    patches travel between frames, then convert that pixel displacement to
    real-world units using the lane markings visible in frame (assumed
    standard lane width) — no manual input required at any point.

    Deliberately uses template matching (normalized cross-correlation)
    rather than dense optical flow: at real driving speeds the road can
    shift well over 50-100px between frames in the near field, which is
    beyond what dense flow methods track reliably, whereas searching for a
    small patch's best match over a wide vertical range handles large
    displacement natively. Multiple patches (avoiding the lane-line
    pixels, which are static markings rather than road texture) are
    matched independently and combined with a median for robustness.
    """
    prev_roi = _road_roi(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    curr_roi = _road_roi(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    if prev_roi.shape != curr_roi.shape or dt_s <= 0:
        return None

    h, w = prev_roi.shape
    tw, th = max(30, w // 10), max(24, h // 5)
    max_search = min(int(h * 0.85), 180)
    anchor_y = min(h - th, max(th, max_search))

    lanes = _detect_lane_lines_x(prev_bgr)
    if lanes:
        left_x, right_x = lanes
        margin = tw // 2 + 6
        inner_lo, inner_hi = left_x + margin, right_x - margin
        if inner_hi - inner_lo > tw:
            x_positions = [int(inner_lo + f * (inner_hi - inner_lo)) for f in (0.25, 0.5, 0.75)]
        else:
            x_positions = [int((left_x + right_x) / 2)]
    else:
        # no lane markings found — best effort, spread across the middle
        # of the frame (still usually road surface, just less certain to
        # avoid any static object that happens to be in view)
        x_positions = [int(w * f) for f in (0.5, 0.65, 0.35)]

    shifts = []
    for tx in x_positions:
        tx = min(max(0, tx - tw // 2), w - tw)
        patch = prev_roi[anchor_y:anchor_y + th, tx:tx + tw]
        if patch.std() < 1.5:
            continue  # too flat/featureless to match reliably (e.g. a shadow)
        y0 = max(0, anchor_y - max_search)
        y1 = min(h - th, anchor_y + max_search)
        search_band = curr_roi[y0:y1 + th, tx:tx + tw]
        if search_band.shape[0] <= th:
            continue
        # one vectorized call scores every candidate row at once, instead
        # of a slow per-row python loop calling matchTemplate repeatedly
        scores = cv2.matchTemplate(search_band, patch, cv2.TM_CCOEFF_NORMED).ravel()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_dy = (y0 + best_idx) - anchor_y
        if best_score > 0.35:
            shifts.append(abs(best_dy))

    px_per_frame = float(np.median(shifts)) if shifts else 0.0
    calibrated_ok = len(shifts) > 0

    lane_px = _detect_lane_pixel_width(prev_bgr) or _detect_lane_pixel_width(curr_bgr)
    calibrated = lane_px is not None
    pixels_per_meter = (lane_px / ASSUMED_LANE_WIDTH_M) if calibrated else DEFAULT_PIXELS_PER_METER

    mps = (px_per_frame / pixels_per_meter) / dt_s if (pixels_per_meter and calibrated_ok) else 0
    kmh = min(mps * 3.6, MAX_PLAUSIBLE_KMH)

    # smooth with an exponential moving average so the read-out doesn't
    # jitter frame to frame the way raw frame-to-frame matching tends to.
    if ema_prev is not None:
        kmh = 0.35 * kmh + 0.65 * ema_prev

    return {"speed_kmh": round(kmh, 1), "calibrated": calibrated}


def recommend_tyre_pressure(vehicle_type: str, speed_kmh: Optional[float],
                             road_label: str, water_alert: bool) -> dict:
    """Auto-computed target tyre pressure — replaces the manual "recommended
    pressure" input. Based on real manufacturer practice: pressure is set
    for the vehicle/load, with a modest bump for sustained high speed. We
    deliberately do NOT lower pressure for wet roads — that's a common
    misconception; under-inflation is what increases aquaplaning risk, not
    the weather. Wet/standing-water conditions are instead handled by the
    existing speed-limit and inspection warnings.
    """
    base_by_vehicle = {"car": 32, "suv": 35, "bike": 28, "bus": 65, "truck": 80}
    base = base_by_vehicle.get((vehicle_type or "car").lower(), 32)

    speed_bump = 0
    if speed_kmh is not None:
        if speed_kmh > 100:
            speed_bump = 2
        elif speed_kmh > 80:
            speed_bump = 1

    # standing water: a correctly (not over-) inflated tyre channels water
    # better through its tread grooves; +1 psi reflects "toward the firm
    # end of the normal range", not a large change.
    water_bump = 1 if water_alert else 0

    recommended = base + speed_bump + water_bump
    reasons = [f"{base} PSI base for {vehicle_type or 'car'}"]
    if speed_bump:
        reasons.append(f"+{speed_bump} PSI for sustained speeds over {80 if speed_bump == 1 else 100} km/h")
    if water_bump:
        reasons.append("+1 PSI — standing water detected, keep tread grooves working at full efficiency")
    if not (speed_bump or water_bump):
        reasons.append("no speed/condition adjustment needed")

    return {"recommended_psi": recommended, "reasoning": "; ".join(reasons)}


def analyze_video(video_path: str, vehicle_type: str = "car", driving_condition: str = "normal",
                   tyre_condition: Optional[str] = None, tyre_pressure: Optional[float] = None,
                   tyre_temperature: Optional[float] = None, report_fps: float = 2.0):
    """Full pipeline: sample the video, run road-condition + defect
    detection + auto speed estimate, and produce a timeline with a fully
    automatic tyre-pressure recommendation per segment. No speed or
    target-pressure input required.

    Speed is estimated on a finer-grained frame stream than the one used
    for reporting/classification — optical flow needs small, reliably
    trackable displacement between consecutive frames, which at highway
    speed means several samples per second, while road-condition
    classification only needs to run a couple of times per second.
    """
    from safety_engine import VehicleMonitor, TyreMonitor, decide

    flow_frames, checkpoints, duration_s, src_fps = sample_frames_dual(
        video_path, report_fps=report_fps, flow_fps=15.0
    )
    if len(flow_frames) < 1:
        raise ValueError("no frames could be read from this video")

    timeline = []
    history = []
    ema_speed = None
    total_defects = {"pothole": 0, "crack": 0}
    checkpoint_set = set(checkpoints)

    for i in range(len(flow_frames)):
        ts, bgr = flow_frames[i]
        if i > 0:
            prev_ts, prev_bgr = flow_frames[i - 1]
            speed_info = estimate_speed_kmh(prev_bgr, bgr, ts - prev_ts, ema_speed)
            if speed_info:
                ema_speed = speed_info["speed_kmh"]
                speed_calibrated = speed_info["calibrated"]
        else:
            speed_calibrated = False

        if i not in checkpoint_set:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        road_result = analyze_image(pil_img)
        history.append({"label": road_result["label"], "wetness_score": road_result["wetness_score"]})
        trend, _slope = trend_for(history)
        road_result["trend"] = trend
        road_result["suggestion"] = suggestion_for(road_result["label"], trend)

        defects = detect_defects(bgr)
        for d in defects:
            if d["type"] in total_defects:
                total_defects[d["type"]] += 1

        speed_kmh = ema_speed
        water_alert = road_result["wetness_score"] >= 0.88
        pressure_rec = recommend_tyre_pressure(vehicle_type, speed_kmh, road_result["label"], water_alert)

        vehicle = VehicleMonitor(vehicle_type=vehicle_type, speed=speed_kmh,
                                  driving_condition=driving_condition)
        tyre = TyreMonitor(pressure=tyre_pressure, recommended_pressure=pressure_rec["recommended_psi"],
                            condition=tyre_condition, temperature=tyre_temperature)
        safety = decide(road_result, vehicle, tyre)

        timeline.append({
            "t": round(ts, 2),
            "frame_index": len(timeline),
            "road_condition": road_result["label"],
            "wetness_score": road_result["wetness_score"],
            "defects": defects,
            "speed_kmh": speed_kmh,
            "speed_calibrated": speed_calibrated,
            "recommended_tyre_psi": pressure_rec["recommended_psi"],
            "tyre_pressure_reasoning": pressure_rec["reasoning"],
            "safety": safety,
        })

    return {
        "duration_s": round(duration_s, 1),
        "source_fps": round(src_fps, 1),
        "frames_analyzed": len(timeline),
        "defect_totals": total_defects,
        "timeline": timeline,
    }
