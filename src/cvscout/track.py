from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvscout.schemas import FrameTracks
from cvscout.tracking import detect_frame, track_sequence
from cvscout.video import VideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple tracking on a video and output JSONL.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video")
    parser.add_argument("--max-frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with VideoReader(args.video) as reader:
        total = reader.get_frame_count()
        fps = reader.get_fps() if reader.get_fps() > 0 else 1.0
        frame_limit = max(0, min(args.max_frames, total))

        detections_by_frame = []
        for frame_idx in range(frame_limit):
            frame = reader.read_frame(frame_idx)
            detections_by_frame.append(detect_frame(frame))

    tracked = track_sequence(detections_by_frame, iou_threshold=args.iou_threshold, max_age=args.max_age)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for frame_idx, frame_tracks in enumerate(tracked):
            payload = FrameTracks(
                frame_idx=frame_idx,
                timestamp_s=frame_idx / fps,
                tracks=frame_tracks.tracks,
            )
            handle.write(json.dumps(payload.model_dump(by_alias=True)) + "\n")


if __name__ == "__main__":
    main()
