from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cvscout.video import VideoReader


def _write_test_video(path: Path, *, frames: int = 3, width: int = 64, height: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for test video")

    for idx in range(frames):
        image = np.full((height, width, 3), idx * 20, dtype=np.uint8)
        writer.write(image)

    writer.release()


def test_video_reader_rejects_out_of_range_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path, frames=2)

    with VideoReader(video_path) as reader:
        assert reader.get_frame_count() >= 2

        with pytest.raises(IndexError):
            reader.read_frame(-1)

        with pytest.raises(IndexError):
            reader.read_frame(reader.get_frame_count())
