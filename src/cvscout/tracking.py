from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cvscout.schemas import BBox, FrameTracks, Track

_MODEL: Any | None = None


def iou(a: BBox, b: BBox) -> float:
    ax2 = a.x + a.width
    ay2 = a.y + a.height
    bx2 = b.x + b.width
    by2 = b.y + b.height

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = a.width * a.height + b.width * b.height - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO  # lazy import

        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def detect_frame(frame_bgr: np.ndarray, confidence_threshold: float = 0.25) -> list[Track]:
    model = _get_model()
    result = model.predict(source=frame_bgr, verbose=False, conf=confidence_threshold, imgsz=640)[0]
    tracks: list[Track] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        tracks.append(
            Track(
                track_id=-1,
                bbox=BBox(x=max(0.0, x1), y=max(0.0, y1), width=width, height=height),
                confidence=conf,
            )
        )
    return tracks


@dataclass
class _LiveTrack:
    track_id: int
    bbox: BBox
    age: int = 0


def track_sequence(
    detections_by_frame: list[list[Track]],
    iou_threshold: float = 0.3,
    max_age: int = 5,
) -> list[FrameTracks]:
    live_tracks: list[_LiveTrack] = []
    next_track_id = 0
    output: list[FrameTracks] = []

    for frame_idx, detections in enumerate(detections_by_frame):
        unmatched_detection_idxs = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        pair_scores: list[tuple[float, int, int]] = []
        for live_idx, live in enumerate(live_tracks):
            for det_idx, detection in enumerate(detections):
                score = iou(live.bbox, detection.bbox)
                if score >= iou_threshold:
                    pair_scores.append((score, live_idx, det_idx))

        pair_scores.sort(reverse=True, key=lambda item: item[0])
        matched_live_idxs: set[int] = set()
        for _, live_idx, det_idx in pair_scores:
            if live_idx in matched_live_idxs or det_idx not in unmatched_detection_idxs:
                continue
            matched_live_idxs.add(live_idx)
            unmatched_detection_idxs.remove(det_idx)
            matches.append((live_idx, det_idx))

        frame_tracks: list[Track] = []

        for live_idx, det_idx in matches:
            detection = detections[det_idx]
            live = live_tracks[live_idx]
            live.bbox = detection.bbox
            live.age = 0
            frame_tracks.append(
                Track(track_id=live.track_id, bbox=detection.bbox, confidence=detection.confidence)
            )

        for live_idx, live in enumerate(live_tracks):
            if live_idx not in matched_live_idxs:
                live.age += 1

        for det_idx in sorted(unmatched_detection_idxs):
            detection = detections[det_idx]
            live_tracks.append(_LiveTrack(track_id=next_track_id, bbox=detection.bbox, age=0))
            frame_tracks.append(
                Track(track_id=next_track_id, bbox=detection.bbox, confidence=detection.confidence)
            )
            next_track_id += 1

        live_tracks = [track for track in live_tracks if track.age <= max_age]
        frame_tracks.sort(key=lambda t: t.track_id)
        output.append(FrameTracks(frame_idx=frame_idx, tracks=frame_tracks))

    return output
