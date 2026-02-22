from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Simple OpenCV-backed random-access video reader."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise ValueError(f"Unable to open video: {self.video_path}")

        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def read_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self._frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {max(self._frame_count - 1, 0)}]"
            )

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")

        return frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
